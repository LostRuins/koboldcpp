#include "engine/framework/assets/weight_adapter.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cmath>
#include <regex>
#include <stdexcept>

namespace engine::assets {
namespace {
namespace json = engine::io::json;

bool ends_with(const std::string & name, std::string_view suffix) {
    return name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string target_name(std::string name) {
    for (const auto prefix : {"base_model.model.", "text_encoders.", "lora_unet_", "lora_transformer_", "lycoris_"}) {
        if (name.rfind(prefix, 0) == 0) {
            name.erase(0, std::char_traits<char>::length(prefix));
            break;
        }
    }
    return name;
}

enum class FactorFormat { Native, Peft, Comfy, LoKr };

struct FactorNames {
    std::map<std::string, std::string> tensors;
    std::optional<FactorFormat> format;
};

float scalar(const TensorSource & source, const std::string & name) {
    const auto values = source.require_f32(name);
    if (values.size() != 1 || !std::isfinite(values.front())) {
        throw std::runtime_error("adapter scalar must have one finite value: " + name);
    }
    return values.front();
}

float pattern_value(const json::Value & config, const std::string & field,
                    const std::string & target, float fallback) {
    const auto * pattern = config.find(field);
    if (pattern == nullptr) return fallback;
    std::optional<float> matched;
    for (const auto & [key, value] : pattern->as_object()) {
        if (!std::regex_match(target, std::regex("(.*\\.)?(" + key + ")"))) continue;
        if (matched && *matched != value.as_f32()) {
            throw std::runtime_error("ambiguous adapter " + field + " for " + target);
        }
        matched = value.as_f32();
    }
    return matched.value_or(fallback);
}

struct Matrix {
    std::vector<float> values;
    int64_t rows;
    int64_t cols;
};

Matrix read_matrix(const TensorSource & source, const std::string & name) {
    const auto shape = source.require_metadata(name).shape;
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
        throw std::runtime_error("adapter factor must be a non-empty matrix: " + name);
    }
    return {source.require_f32(name, shape), shape[0], shape[1]};
}

Matrix kronecker_factor(const TensorSource & source, const FactorNames & names,
                        const std::string & key, int64_t & rank) {
    const auto full = names.tensors.find(key);
    const auto a = names.tensors.find(key + "_a");
    const auto b = names.tensors.find(key + "_b");
    if (full != names.tensors.end()) {
        if (a != names.tensors.end() || b != names.tensors.end()) {
            throw std::runtime_error("adapter supplies full and decomposed " + key);
        }
        return read_matrix(source, full->second);
    }
    if (a == names.tensors.end() || b == names.tensors.end()) {
        throw std::runtime_error("adapter is missing LoKr factor " + key);
    }
    const auto left = read_matrix(source, a->second);
    const auto right = read_matrix(source, b->second);
    if (left.cols != right.rows || (rank != 0 && rank != left.cols)) {
        throw std::runtime_error("adapter LoKr factor rank mismatch: " + key);
    }
    rank = left.cols;
    Matrix result{std::vector<float>(static_cast<size_t>(left.rows * right.cols), 0.0F), left.rows, right.cols};
    for (int64_t o = 0; o < result.rows; ++o) {
        for (int64_t k = 0; k < left.cols; ++k) {
            for (int64_t i = 0; i < result.cols; ++i) {
                result.values[o * result.cols + i] += left.values[o * left.cols + k] * right.values[k * right.cols + i];
            }
        }
    }
    return result;
}

float update_scale(const TensorSource & source, const FactorNames & names,
                   const json::Value & config, const std::string & target,
                   int64_t rank, const WeightAdapterReadOptions & options) {
    if (options.scale_override) return options.strength * *options.scale_override;
    // Full-matrix LoKr stores the complete Kronecker factors; no rank divisor.
    if (rank == 0) return options.strength;
    const float configured_rank = pattern_value(config, "rank_pattern", target,
        json::optional_f32(config, "r", static_cast<float>(rank)));
    if (configured_rank != static_cast<float>(rank)) {
        throw std::runtime_error("adapter configured rank differs from factors: " + target);
    }
    float alpha = pattern_value(config, "alpha_pattern", target,
        json::optional_f32(config, "lora_alpha", static_cast<float>(rank)));
    if (const auto it = names.tensors.find("alpha"); it != names.tensors.end()) alpha = scalar(source, it->second);
    const bool rs = json::optional_bool(config, "use_rslora", false);
    const float coefficient = alpha / (rs ? std::sqrt(static_cast<float>(rank)) : static_cast<float>(rank));
    const float scale = options.strength * coefficient;
    if (!std::isfinite(scale)) throw std::runtime_error("adapter scale is not finite: " + target);
    return scale;
}
}  // namespace

WeightAdapter read_weight_adapter(const std::filesystem::path & path, const WeightAdapterReadOptions & options) {
    if (!std::isfinite(options.strength) || (options.scale_override && !std::isfinite(*options.scale_override))) {
        throw std::runtime_error("adapter strength and scale must be finite");
    }
    auto file = path;
    if (io::is_existing_directory(path)) {
        const auto resolved = find_tensor_file(path, "adapter_model");
        if (!resolved) throw std::runtime_error("missing adapter_model checkpoint in " + path.string());
        file = *resolved;
    }
    const auto config_path = file.parent_path() / "adapter_config.json";
    const bool has_config = io::is_existing_file(config_path);
    const auto source = open_tensor_source(file);
    std::map<std::string, FactorNames> groups;
    WeightAdapter result;
    struct Suffix {
        std::string_view suffix;
        std::string_view part;
        std::optional<FactorFormat> format;
    };
    const Suffix suffixes[] = {
        {".lora_A.weight", "a", FactorFormat::Peft}, {".lora_B.weight", "b", FactorFormat::Peft},
        {".lora_A.default.weight", "a", FactorFormat::Peft}, {".lora_B.default.weight", "b", FactorFormat::Peft},
        {".lora_A", "a", FactorFormat::Native}, {".lora_B", "b", FactorFormat::Native},
        {".lora_down.weight", "a", FactorFormat::Comfy}, {".lora_up.weight", "b", FactorFormat::Comfy},
        {".lokr_w1", "w1", FactorFormat::LoKr}, {".lokr_w2", "w2", FactorFormat::LoKr},
        {".lokr_w1_a", "w1_a", FactorFormat::LoKr}, {".lokr_w1_b", "w1_b", FactorFormat::LoKr},
        {".lokr_w2_a", "w2_a", FactorFormat::LoKr}, {".lokr_w2_b", "w2_b", FactorFormat::LoKr},
        {".alpha", "alpha", std::nullopt},
    };
    for (const auto & tensor : source->tensors()) {
        bool factor = false;
        for (const auto & [suffix, part, format] : suffixes) {
            if (!ends_with(tensor.name, suffix)) continue;
            const auto target = target_name(tensor.name.substr(0, tensor.name.size() - suffix.size()));
            auto & group = groups[target];
            if (format && group.format && format != group.format) {
                throw std::runtime_error("mixed adapter formats for " + target);
            }
            if (format) group.format = format;
            if (!group.tensors.emplace(std::string(part), tensor.name).second) {
                throw std::runtime_error("duplicate adapter factor: " + tensor.name);
            }
            factor = true;
            break;
        }
        if (factor) continue;
        if (!ends_with(tensor.name, ".weight") && !ends_with(tensor.name, ".bias")) {
            throw std::runtime_error("unsupported adapter tensor: " + tensor.name);
        }
        const auto name = target_name(tensor.name);
        if (!result.replacements.emplace(name, WeightReplacement{tensor.shape, source->require_f32(tensor.name)}).second) {
            throw std::runtime_error("duplicate adapter replacement: " + name);
        }
    }
    const bool needs_config = std::any_of(groups.begin(), groups.end(), [](const auto & entry) {
        return entry.second.format == FactorFormat::Peft;
    });
    const auto empty_config = json::Value::make_object({});
    const auto peft_config = needs_config && has_config ? json::parse_file(config_path) : empty_config;
    if (json::optional_bool(peft_config, "use_dora", false)) {
        throw std::runtime_error("DoRA adapters are not supported");
    }
    if (json::optional_bool(peft_config, "fan_in_fan_out", false) ||
        json::optional_bool(peft_config, "use_qalora", false) ||
        json::optional_bool(peft_config, "lora_bias", false)) {
        throw std::runtime_error("adapter requires unsupported transposed, QALoRA, or bias updates");
    }
    for (const auto & [target, names] : groups) {
        const bool uses_config = names.format == FactorFormat::Peft && has_config;
        const auto & config = uses_config ? peft_config : empty_config;
        const bool full_kronecker = names.tensors.count("w1") != 0 && names.tensors.count("w2") != 0;
        if (options.require_scaling_metadata && !uses_config && !options.scale_override &&
            names.tensors.count("alpha") == 0 && !full_kronecker) {
            throw std::runtime_error("adapter has no scaling metadata for " + target + "; specify its effective scale explicitly");
        }
        WeightUpdate update;
        const auto a = names.tensors.find("a");
        const auto b = names.tensors.find("b");
        if (a != names.tensors.end() || b != names.tensors.end()) {
            if (a == names.tensors.end() || b == names.tensors.end()) {
                throw std::runtime_error("adapter is missing an A/B pair: " + target);
            }
            if (names.tensors.size() != 2 + names.tensors.count("alpha")) {
                throw std::runtime_error("mixed adapter algorithms for " + target);
            }
            auto down = read_matrix(*source, a->second);
            auto up = read_matrix(*source, b->second);
            if (up.cols != down.rows) throw std::runtime_error("adapter A/B rank mismatch: " + target);
            update.scale = update_scale(*source, names, config, target, down.rows, options);
            update.factors = LowRankWeightUpdate{std::move(down.values), std::move(up.values), down.rows, down.cols, up.rows};
        } else {
            int64_t rank = 0;
            auto left = kronecker_factor(*source, names, "w1", rank);
            auto right = kronecker_factor(*source, names, "w2", rank);
            update.scale = update_scale(*source, names, config, target, rank, options);
            update.factors = KroneckerWeightUpdate{std::move(left.values), std::move(right.values),
                left.rows, left.cols, right.rows, right.cols};
        }
        result.updates.emplace(target, std::move(update));
    }
    if (result.updates.empty() && result.replacements.empty()) throw std::runtime_error("adapter has no supported tensors");
    return result;
}

void WeightAdapterBindings::add(std::string name, std::vector<WeightAdapterTarget> targets) {
    if (ends_with(name, ".weight")) name.resize(name.size() - 7);
    const auto insert = [&](const std::string & key) {
        const auto it = targets_.find(key);
        if (it != targets_.end()) {
            const auto & old = it->second;
            bool same = old.size() == targets.size();
            for (size_t i = 0; same && i < old.size(); ++i) {
                same = old[i].tensor == targets[i].tensor && old[i].first_row == targets[i].first_row && old[i].rows == targets[i].rows;
            }
            if (!same) throw std::runtime_error("ambiguous adapter target binding: " + key);
        } else {
            targets_.emplace(key, targets);
        }
    };
    insert(name);
    std::replace(name.begin(), name.end(), '.', '_');
    insert(name);
}

void WeightAdapterBindings::add(std::string name, std::string tensor) {
    add(std::move(name), std::vector<WeightAdapterTarget>{{std::move(tensor), 0, 0}});
}

const std::vector<WeightAdapterTarget> & WeightAdapterBindings::resolve(const std::string & name) const {
    auto key = name;
    if (ends_with(key, ".weight")) key.resize(key.size() - 7);
    const auto it = targets_.find(key);
    if (it == targets_.end()) throw std::runtime_error("adapter target has no model binding: " + name);
    return it->second;
}

WeightAdapter bind_weight_adapter(const TensorSource & base, WeightAdapter adapter,
                                 const WeightAdapterBindings & bindings, WeightMergeMode merge_mode) {
    WeightAdapter result;
    for (auto & [name, update] : adapter.updates) {
        const auto & targets = bindings.resolve(name);
        if (const auto * low = std::get_if<LowRankWeightUpdate>(&update.factors)) {
            if (targets.size() == 1 && targets[0].first_row == 0 && targets[0].rows == 0) {
                require_tensor_shape(base, targets[0].tensor, {low->out, low->in});
                update.merge_mode = merge_mode;
                if (!result.updates.emplace(targets[0].tensor, std::move(update)).second) {
                    throw std::runtime_error("multiple adapter updates target " + targets[0].tensor);
                }
                continue;
            }
            std::vector<std::pair<int64_t, int64_t>> ranges;
            for (const auto & target : targets) {
                const int64_t rows = target.rows == 0 ? low->out : target.rows;
                if (target.first_row < 0 || rows <= 0 || target.first_row > low->out - rows) {
                    throw std::runtime_error("adapter output slice is invalid: " + name);
                }
                require_tensor_shape(base, target.tensor, {rows, low->in});
                LowRankWeightUpdate sliced{low->a,
                    std::vector<float>(low->b.begin() + target.first_row * low->r,
                                       low->b.begin() + (target.first_row + rows) * low->r),
                    low->r, low->in, rows};
                WeightUpdate bound{std::move(sliced), update.scale, merge_mode};
                if (!result.updates.emplace(target.tensor, std::move(bound)).second) {
                    throw std::runtime_error("multiple adapter updates target " + target.tensor);
                }
                ranges.emplace_back(target.first_row, target.first_row + rows);
            }
            std::sort(ranges.begin(), ranges.end());
            int64_t next = 0;
            for (const auto & range : ranges) {
                if (range.first != next) throw std::runtime_error("adapter output slices overlap or leave a gap: " + name);
                next = range.second;
            }
            if (next != low->out) throw std::runtime_error("adapter output rows were not fully bound: " + name);
        } else {
            const auto & kron = std::get<KroneckerWeightUpdate>(update.factors);
            if (targets.size() != 1 || targets[0].first_row != 0 || targets[0].rows != 0) {
                throw std::runtime_error("sliced LoKr targets are not supported: " + name);
            }
            require_tensor_shape(base, targets[0].tensor,
                {kron.left_rows * kron.right_rows, kron.left_cols * kron.right_cols});
            update.merge_mode = merge_mode;
            if (!result.updates.emplace(targets[0].tensor, std::move(update)).second) {
                throw std::runtime_error("multiple adapter updates target " + targets[0].tensor);
            }
        }
    }
    for (auto & [name, replacement] : adapter.replacements) {
        const auto & targets = bindings.resolve(name);
        if (targets.size() != 1 || targets[0].first_row != 0 || targets[0].rows != 0) {
            throw std::runtime_error("sliced full replacements are not supported: " + name);
        }
        require_tensor_shape(base, targets[0].tensor, replacement.shape);
        if (merge_mode == WeightMergeMode::RoundedBF16Delta) {
            for (auto & value : replacement.values) value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
        }
        if (!result.replacements.emplace(targets[0].tensor, std::move(replacement)).second) {
            throw std::runtime_error("multiple adapter replacements target " + targets[0].tensor);
        }
    }
    return result;
}

}  // namespace engine::assets
