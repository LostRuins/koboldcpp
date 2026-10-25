#include "engine/framework/assets/tensor_source.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace engine::assets {
namespace {

bool pattern_matches(std::string_view name, std::string_view pattern) {
    if (pattern.empty()) {
        return false;
    }
    if (pattern.back() == '*') {
        pattern.remove_suffix(1);
        return name.substr(0, pattern.size()) == pattern;
    }
    return name == pattern;
}

bool has_suffix(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::vector<std::byte> f32_bytes(const std::vector<float> & values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

void validate_expected_shape(
    std::string_view name,
    const std::vector<int64_t> & actual,
    const std::optional<std::vector<int64_t>> & expected) {
    if (expected.has_value() && actual != *expected) {
        throw std::runtime_error("tensor shape mismatch for " + std::string(name));
    }
}

int64_t checked_element_count(std::string_view name, const std::vector<int64_t> & shape) {
    int64_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("tensor shape contains a non-positive dimension: " + std::string(name));
        }
        if (count > std::numeric_limits<int64_t>::max() / dim) {
            throw std::runtime_error("tensor element count overflow: " + std::string(name));
        }
        count *= dim;
    }
    return count;
}

bool is_weight_v_tensor(std::string_view name) {
    return has_suffix(name, ".weight_v") || has_suffix(name, ".parametrizations.weight.original1");
}

std::string weight_norm_base_name(std::string_view name) {
    constexpr std::string_view kWeightV = ".weight_v";
    constexpr std::string_view kParamWeightV = ".parametrizations.weight.original1";
    if (has_suffix(name, kWeightV)) {
        return std::string(name.substr(0, name.size() - kWeightV.size()));
    }
    if (has_suffix(name, kParamWeightV)) {
        return std::string(name.substr(0, name.size() - kParamWeightV.size()));
    }
    throw std::runtime_error("not a weight-norm value tensor: " + std::string(name));
}

std::string weight_g_name_for_v(std::string_view name) {
    constexpr std::string_view kWeightV = ".weight_v";
    constexpr std::string_view kParamWeightV = ".parametrizations.weight.original1";
    if (has_suffix(name, kWeightV)) {
        return std::string(name.substr(0, name.size() - kWeightV.size())) + ".weight_g";
    }
    if (has_suffix(name, kParamWeightV)) {
        return std::string(name.substr(0, name.size() - kParamWeightV.size())) +
               ".parametrizations.weight.original0";
    }
    throw std::runtime_error("not a weight-norm value tensor: " + std::string(name));
}

class WeightNormFoldedTensorSource final : public TensorSource {
public:
    WeightNormFoldedTensorSource(std::shared_ptr<const TensorSource> source, std::vector<std::string> patterns)
        : source_(std::move(source)), patterns_(std::move(patterns)) {
        std::unordered_set<std::string> helper_names;
        const auto source_tensors = source_->tensors();
        std::unordered_map<std::string, TensorMetadata> metadata_by_name;
        metadata_by_name.reserve(source_tensors.size());
        for (const auto & tensor : source_tensors) {
            metadata_by_name.emplace(tensor.name, tensor);
        }

        for (const auto & tensor : source_tensors) {
            if (!is_weight_v_tensor(tensor.name)) {
                continue;
            }
            const auto base = weight_norm_base_name(tensor.name);
            const std::string weight_name = base + ".weight";
            if (!matches_fold_pattern(weight_name) || metadata_by_name.find(weight_name) != metadata_by_name.end()) {
                continue;
            }
            const auto g_name = weight_g_name_for_v(tensor.name);
            if (metadata_by_name.find(g_name) == metadata_by_name.end()) {
                throw std::runtime_error("weight-norm fold is missing scale tensor: " + g_name);
            }
            folded_.emplace(
                weight_name,
                FoldedWeight{tensor.name, g_name, TensorMetadata{weight_name, "F32", tensor.shape}});
            helper_names.insert(tensor.name);
            helper_names.insert(g_name);
        }

        for (const auto & tensor : source_tensors) {
            if (helper_names.find(tensor.name) == helper_names.end()) {
                routes_.emplace(tensor.name, tensor.name);
            }
        }
    }

    const std::filesystem::path & source_path() const noexcept override { return source_->source_path(); }

    bool has_tensor(std::string_view name) const noexcept override {
        const std::string key(name);
        return folded_.find(key) != folded_.end() || routes_.find(key) != routes_.end();
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        const std::string key(name);
        if (const auto folded = folded_.find(key); folded != folded_.end()) {
            return folded->second.metadata;
        }
        return source_->require_metadata(require_name(name));
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(routes_.size() + folded_.size());
        for (const auto & [name, unused] : routes_) {
            (void)unused;
            out.push_back(source_->require_metadata(name));
        }
        for (const auto & [unused, folded] : folded_) {
            (void)unused;
            out.push_back(folded.metadata);
        }
        std::sort(out.begin(), out.end(), [](const TensorMetadata & lhs, const TensorMetadata & rhs) {
            return lhs.name < rhs.name;
        });
        return out;
    }

    void release_storage() const override { source_->release_storage(); }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const std::string key(name);
        if (const auto folded = folded_.find(key); folded != folded_.end()) {
            RawTensorData data;
            data.metadata = folded->second.metadata;
            data.bytes = f32_bytes(fold_weight_norm(folded->second));
            return data;
        }
        return source_->require_tensor_data(require_name(name));
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const std::string key(name);
        if (const auto folded = folded_.find(key); folded != folded_.end()) {
            validate_expected_shape(name, folded->second.metadata.shape, expected_shape);
            return fold_weight_norm(folded->second);
        }
        return source_->require_f32(require_name(name), expected_shape);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return source_->require_i64_scalar(require_name(name));
    }

private:
    struct FoldedWeight {
        std::string weight_v_name;
        std::string weight_g_name;
        TensorMetadata metadata;
    };

    bool matches_fold_pattern(std::string_view name) const {
        for (const auto & pattern : patterns_) {
            if (pattern_matches(name, pattern)) {
                return true;
            }
        }
        return false;
    }

    std::vector<float> fold_weight_norm(const FoldedWeight & folded) const {
        const auto & shape = folded.metadata.shape;
        if (shape.empty()) {
            throw std::runtime_error("weight-norm tensor shape is empty: " + folded.metadata.name);
        }
        const int64_t leading = shape.front();
        const int64_t elements = checked_element_count(folded.metadata.name, shape);
        if (leading <= 0 || elements % leading != 0) {
            throw std::runtime_error("weight-norm tensor shape mismatch: " + folded.metadata.name);
        }
        const int64_t inner_size = elements / leading;
        const auto weight_v = source_->require_f32(folded.weight_v_name, shape);
        const auto weight_g_metadata = source_->require_metadata(folded.weight_g_name);
        if (checked_element_count(folded.weight_g_name, weight_g_metadata.shape) != leading) {
            throw std::runtime_error("weight-norm scale tensor shape mismatch: " + folded.weight_g_name);
        }
        const auto weight_g = source_->require_f32(folded.weight_g_name, weight_g_metadata.shape);

        std::vector<float> out(weight_v.size(), 0.0F);
        for (int64_t row = 0; row < leading; ++row) {
            const size_t base = static_cast<size_t>(row * inner_size);
            double norm_sq = 0.0;
            for (int64_t index = 0; index < inner_size; ++index) {
                const float value = weight_v[base + static_cast<size_t>(index)];
                norm_sq += static_cast<double>(value) * static_cast<double>(value);
            }
            if (norm_sq <= 0.0) {
                throw std::runtime_error("weight-norm value tensor has zero norm: " + folded.weight_v_name);
            }
            const float scale = weight_g[static_cast<size_t>(row)] / static_cast<float>(std::sqrt(norm_sq));
            for (int64_t index = 0; index < inner_size; ++index) {
                out[base + static_cast<size_t>(index)] = weight_v[base + static_cast<size_t>(index)] * scale;
            }
        }
        return out;
    }

    const std::string & require_name(std::string_view name) const {
        const auto it = routes_.find(std::string(name));
        if (it == routes_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        return it->second;
    }

    std::shared_ptr<const TensorSource> source_;
    std::vector<std::string> patterns_;
    std::unordered_map<std::string, std::string> routes_;
    std::unordered_map<std::string, FoldedWeight> folded_;
};

}  // namespace

std::shared_ptr<const TensorSource> make_weight_norm_folded_tensor_source(
    std::shared_ptr<const TensorSource> source,
    std::vector<std::string> patterns) {
    return std::make_shared<WeightNormFoldedTensorSource>(std::move(source), std::move(patterns));
}

}  // namespace engine::assets
