#include "engine/models/supertonic/runtime.h"

#include "engine/framework/audio/output.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/deferred_tensor_writer.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::supertonic {
namespace {

namespace json = engine::io::json;
namespace core = engine::core;
namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct SupertonicTensorOutput {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

struct SupertonicLatentTensor {
    std::vector<int64_t> shape;
    std::vector<float> f32;
};

struct SupertonicChunkOutput {
    runtime::AudioBuffer audio;
    float duration_seconds = 0.0F;
};

constexpr size_t kSupertonicDurationArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kSupertonicTextArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kSupertonicVectorArenaBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kSupertonicVocoderArenaBytes = 2048ull * 1024ull * 1024ull;
constexpr size_t kSupertonicIoArenaBytes = 64ull * 1024ull * 1024ull;
constexpr int kSupertonicVectorStepsPerGraph = 1;
constexpr size_t kSupertonicSmallGraphCacheSlots = 4;
constexpr size_t kSupertonicLargeGraphCacheSlots = 2;

std::vector<int64_t> shape_to_vector(const core::TensorShape & shape) {
    std::vector<int64_t> out;
    out.reserve(shape.rank);
    for (size_t i = 0; i < shape.rank; ++i) {
        out.push_back(shape.dims[i]);
    }
    return out;
}

core::TensorShape shape_from_vector(const std::vector<int64_t> & dims) {
    if (dims.size() == 1) {
        return core::TensorShape::from_dims({dims[0]});
    }
    if (dims.size() == 2) {
        return core::TensorShape::from_dims({dims[0], dims[1]});
    }
    if (dims.size() == 3) {
        return core::TensorShape::from_dims({dims[0], dims[1], dims[2]});
    }
    if (dims.size() == 4) {
        return core::TensorShape::from_dims({dims[0], dims[1], dims[2], dims[3]});
    }
    throw std::runtime_error("Supertonic tensor rank is outside GGML supported range");
}

core::TensorShape binary_broadcast_shape(const core::TensorShape & lhs, const core::TensorShape & rhs) {
    const size_t rank = std::max(lhs.rank, rhs.rank);
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t ld = i < rank - lhs.rank ? 1 : lhs.dims[i - (rank - lhs.rank)];
        const int64_t rd = i < rank - rhs.rank ? 1 : rhs.dims[i - (rank - rhs.rank)];
        if (ld != rd && ld != 1 && rd != 1) {
            throw std::runtime_error("Supertonic binary broadcast mismatch: lhs=" + lhs.to_string() + " rhs=" + rhs.to_string());
        }
        out[i] = std::max(ld, rd);
    }
    return shape_from_vector(out);
}

bool shape_fits_capacity(const std::vector<int64_t> & capacity, const std::vector<int64_t> & shape) {
    if (capacity.size() != shape.size() || capacity.empty() || capacity.size() > core::kMaxTensorRank) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0 || capacity[i] < shape[i]) {
            return false;
        }
    }
    return true;
}

void append_cache_key_shape(std::vector<int64_t> & key, const std::vector<int64_t> & shape) {
    key.push_back(static_cast<int64_t>(shape.size()));
    key.insert(key.end(), shape.begin(), shape.end());
}

size_t vector_num_elements(const std::vector<int64_t> & shape) {
    return static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
}

size_t logical_offset(const std::vector<int64_t> & shape, const std::array<int64_t, 4> & index) {
    size_t offset = 0;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        offset *= static_cast<size_t>(shape[axis]);
        offset += static_cast<size_t>(index[axis]);
    }
    return offset;
}

template <typename T>
void copy_padded_values(
    T * dst,
    const std::vector<int64_t> & dst_shape,
    const T * src,
    const std::vector<int64_t> & src_shape) {
    if (!shape_fits_capacity(dst_shape, src_shape)) {
        throw std::runtime_error("Supertonic padded copy shape mismatch");
    }
    const size_t dst_count = vector_num_elements(dst_shape);
    const int64_t dst_count_i64 = static_cast<int64_t>(dst_count);
#ifdef _OPENMP
#pragma omp parallel for if(dst_count_i64 >= 4096)
#endif
    for (int64_t i = 0; i < dst_count_i64; ++i) {
        dst[static_cast<size_t>(i)] = T{};
    }
    if (dst_shape == src_shape) {
        std::memcpy(dst, src, dst_count * sizeof(T));
        return;
    }
    const size_t src_count = vector_num_elements(src_shape);
    const int64_t src_count_i64 = static_cast<int64_t>(src_count);
#ifdef _OPENMP
#pragma omp parallel for if(src_count_i64 >= 4096)
#endif
    for (int64_t src_index = 0; src_index < src_count_i64; ++src_index) {
        const size_t src_offset = static_cast<size_t>(src_index);
        size_t remainder = src_offset;
        std::array<int64_t, 4> index = {0, 0, 0, 0};
        for (size_t axis = src_shape.size(); axis > 0; --axis) {
            const size_t dim_axis = axis - 1;
            index[dim_axis] = static_cast<int64_t>(remainder % static_cast<size_t>(src_shape[dim_axis]));
            remainder /= static_cast<size_t>(src_shape[dim_axis]);
        }
        dst[logical_offset(dst_shape, index)] = src[src_offset];
    }
}

template <typename T>
std::vector<T> slice_padded_values(
    const std::vector<T> & src,
    const std::vector<int64_t> & src_shape,
    const std::vector<int64_t> & dst_shape) {
    if (!shape_fits_capacity(src_shape, dst_shape)) {
        throw std::runtime_error("Supertonic padded slice shape mismatch");
    }
    if (src_shape == dst_shape) {
        return src;
    }
    const size_t dst_count = vector_num_elements(dst_shape);
    std::vector<T> dst(dst_count, T{});
    const int64_t dst_count_i64 = static_cast<int64_t>(dst_count);
#ifdef _OPENMP
#pragma omp parallel for if(dst_count_i64 >= 4096)
#endif
    for (int64_t dst_index = 0; dst_index < dst_count_i64; ++dst_index) {
        const size_t dst_offset = static_cast<size_t>(dst_index);
        size_t remainder = dst_offset;
        std::array<int64_t, 4> index = {0, 0, 0, 0};
        for (size_t axis = dst_shape.size(); axis > 0; --axis) {
            const size_t dim_axis = axis - 1;
            index[dim_axis] = static_cast<int64_t>(remainder % static_cast<size_t>(dst_shape[dim_axis]));
            remainder /= static_cast<size_t>(dst_shape[dim_axis]);
        }
        dst[dst_offset] = src[logical_offset(src_shape, index)];
    }
    return dst;
}

void upload_f32_tensor(
    ggml_tensor * tensor,
    const std::vector<int64_t> & capacity_shape,
    const std::vector<int64_t> & logical_shape,
    const std::vector<float> & values,
    const std::string & label) {
    if (values.size() != vector_num_elements(logical_shape)) {
        throw std::runtime_error("Supertonic f32 input count mismatch: " + label);
    }
    std::vector<float> padded;
    const float * src = values.data();
    size_t count = values.size();
    if (capacity_shape != logical_shape) {
        padded.resize(vector_num_elements(capacity_shape));
        copy_padded_values(padded.data(), capacity_shape, values.data(), logical_shape);
        src = padded.data();
        count = padded.size();
    }
    core::write_tensor_f32(
        core::wrap_tensor(tensor, shape_from_vector(capacity_shape), GGML_TYPE_F32),
        src,
        count);
}

void upload_i32_tensor(
    ggml_tensor * tensor,
    const std::vector<int64_t> & capacity_shape,
    const std::vector<int64_t> & logical_shape,
    const std::vector<int32_t> & values,
    const std::string & label) {
    if (values.size() != vector_num_elements(logical_shape)) {
        throw std::runtime_error("Supertonic i32 input count mismatch: " + label);
    }
    std::vector<int32_t> padded;
    const int32_t * src = values.data();
    size_t count = values.size();
    if (capacity_shape != logical_shape) {
        padded.resize(vector_num_elements(capacity_shape));
        copy_padded_values(padded.data(), capacity_shape, values.data(), logical_shape);
        src = padded.data();
        count = padded.size();
    }
    core::write_tensor_i32(
        core::wrap_tensor(tensor, shape_from_vector(capacity_shape), GGML_TYPE_I32),
        src,
        count);
}

struct SupertonicBackendWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, core::TensorValue> tensors;
};

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

bool is_pointwise_linear_weight(std::string_view name) {
    const bool pointwise_weight = ends_with(name, ".pwconv1.weight") || ends_with(name, ".pwconv2.weight");
    if (!pointwise_weight) {
        return false;
    }
    if (starts_with(name, "duration_predictor.tts.dp.sentence_encoder.convnext.") ||
        starts_with(name, "text_encoder.tts.ttl.text_encoder.convnext.")) {
        return true;
    }
    static constexpr std::array<int, 4> kPointwiseLinearBlocks = {0, 6, 12, 18};
    for (int block : kPointwiseLinearBlocks) {
        const std::string prefix =
            "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks." + std::to_string(block) + ".convnext.";
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

bool is_reshaped_linear_weight(std::string_view name) {
    return is_pointwise_linear_weight(name) ||
        name == "vector_estimator.vector_estimator.tts.ttl.vector_field.proj_in.net.weight" ||
        name == "vector_estimator.vector_estimator.tts.ttl.vector_field.proj_out.net.weight";
}

assets::TensorStorageType storage_type_for_weight(
    std::string_view name,
    assets::TensorStorageType requested_storage_type) {
    if (requested_storage_type != assets::TensorStorageType::Q8_0) {
        return requested_storage_type;
    }
    if (name == "duration_predictor.tts.dp.predictor.layers.0.weight" ||
        name == "duration_predictor.tts.dp.predictor.layers.1.weight" ||
        name == "vector_estimator.vector_estimator.tts.ttl.vector_field.time_encoder.mlp.0.linear.weight" ||
        name == "vector_estimator.vector_estimator.tts.ttl.vector_field.time_encoder.mlp.2.linear.weight" ||
        is_reshaped_linear_weight(name)) {
        return requested_storage_type;
    }
    return assets::TensorStorageType::F32;
}

const core::TensorValue & require_weight_tensor(
    const SupertonicBackendWeights & weights,
    const std::string & name) {
    const auto found = weights.tensors.find(name);
    if (found == weights.tensors.end()) {
        throw std::runtime_error("Supertonic missing tensor weight: " + name);
    }
    return found->second;
}

std::shared_ptr<const SupertonicBackendWeights> load_backend_weights(
    const std::shared_ptr<const SupertonicAssets> & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType weight_storage_type) {
    auto weights = std::make_shared<SupertonicBackendWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "supertonic.weights",
        64ull * 1024ull * 1024ull);
    const auto tensors = assets->weights->tensors();
    weights->tensors.reserve(tensors.size());
    for (const auto & metadata : tensors) {
        const auto & name = metadata.name;
        const ggml_type source_type = assets::ggml_type_for_tensor_dtype(metadata.dtype);
        if (source_type != GGML_TYPE_I64) {
            const auto tensor_storage_type = storage_type_for_weight(name, weight_storage_type);
            core::TensorValue tensor;
            if (tensor_storage_type == assets::TensorStorageType::Q8_0 && is_reshaped_linear_weight(name)) {
                if (metadata.shape.size() != 3 || metadata.shape[2] != 1) {
                    throw std::runtime_error("Supertonic q8 reshaped linear weight must have source shape [out, in, 1]: " + name);
                }
                tensor = weights->store->load_tensor_as_shape(
                    *assets->weights,
                    name,
                    tensor_storage_type,
                    metadata.shape,
                    core::TensorShape::from_dims({metadata.shape[0], metadata.shape[1]}));
            } else {
                tensor = weights->store->load_tensor(*assets->weights, name, tensor_storage_type, metadata.shape);
            }
            weights->tensors.emplace(name, tensor);
            continue;
        }
    }
    weights->store->upload();
    assets->weights->release_storage();
    return weights;
}

bool same_shape(const core::TensorShape & lhs, const core::TensorShape & rhs) {
    if (lhs.rank != rhs.rank) {
        return false;
    }
    for (size_t i = 0; i < lhs.rank; ++i) {
        if (lhs.dims[i] != rhs.dims[i]) {
            return false;
        }
    }
    return true;
}

core::TensorValue broadcast_to(core::ModuleBuildContext & ctx, core::TensorValue input, const core::TensorShape & target) {
    if (same_shape(input.shape, target)) {
        return input;
    }
    if (input.shape.num_elements() == 1) {
        if (input.shape.rank == 1) {
            input = core::wrap_tensor(ggml_cont_1d(ctx.ggml, input.tensor, input.tensor->ne[0]), input.shape, input.type);
        } else if (input.shape.rank == 2) {
            input = core::wrap_tensor(ggml_cont_2d(ctx.ggml, input.tensor, input.tensor->ne[0], input.tensor->ne[1]), input.shape, input.type);
        } else if (input.shape.rank == 3) {
            input = core::wrap_tensor(ggml_cont_3d(ctx.ggml, input.tensor, input.tensor->ne[0], input.tensor->ne[1], input.tensor->ne[2]), input.shape, input.type);
        } else {
            input = core::wrap_tensor(ggml_cont_4d(ctx.ggml, input.tensor, input.tensor->ne[0], input.tensor->ne[1], input.tensor->ne[2], input.tensor->ne[3]), input.shape, input.type);
        }
        std::vector<int64_t> dims(target.rank, 1);
        auto reshaped = core::reshape_tensor(ctx, input, shape_from_vector(dims));
        return modules::RepeatModule({target}).build(ctx, reshaped);
    }
    if (input.shape.rank < target.rank) {
        std::vector<int64_t> dims(target.rank - input.shape.rank, 1);
        for (size_t i = 0; i < input.shape.rank; ++i) {
            dims.push_back(input.shape.dims[i]);
        }
        input = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, input), shape_from_vector(dims));
    }
    bool compatible = input.shape.rank == target.rank;
    for (size_t i = 0; compatible && i < target.rank; ++i) {
        compatible = input.shape.dims[i] == target.dims[i] || input.shape.dims[i] == 1;
    }
    if (!compatible) {
        throw std::runtime_error("Supertonic broadcast shape mismatch");
    }
    if (input.shape.rank == 1) {
        input = core::wrap_tensor(ggml_cont_1d(ctx.ggml, input.tensor, input.tensor->ne[0]), input.shape, input.type);
    } else if (input.shape.rank == 2) {
        input = core::wrap_tensor(ggml_cont_2d(ctx.ggml, input.tensor, input.tensor->ne[0], input.tensor->ne[1]), input.shape, input.type);
    } else if (input.shape.rank == 3) {
        input = core::wrap_tensor(ggml_cont_3d(ctx.ggml, input.tensor, input.tensor->ne[0], input.tensor->ne[1], input.tensor->ne[2]), input.shape, input.type);
    } else {
        input = core::wrap_tensor(ggml_cont_4d(ctx.ggml, input.tensor, input.tensor->ne[0], input.tensor->ne[1], input.tensor->ne[2], input.tensor->ne[3]), input.shape, input.type);
    }
    return modules::RepeatModule({target}).build(ctx, input);
}

struct SupertonicVocoderBlockWeights {
    core::TensorValue gamma;
    core::TensorValue dw_weight;
    core::TensorValue dw_bias;
    core::TensorValue norm_weight;
    core::TensorValue norm_bias;
    core::TensorValue pw1_weight;
    core::TensorValue pw1_bias;
    core::TensorValue pw2_weight;
    core::TensorValue pw2_bias;
};

struct SupertonicVocoderWeights {
    core::TensorValue ttl_scale;
    core::TensorValue latent_mean;
    core::TensorValue latent_std;
    core::TensorValue embed_weight;
    core::TensorValue embed_bias;
    std::array<SupertonicVocoderBlockWeights, 10> blocks;
    core::TensorValue final_norm_weight;
    core::TensorValue final_norm_bias;
    core::TensorValue final_norm_running_mean;
    core::TensorValue final_norm_running_var;
    core::TensorValue head1_weight;
    core::TensorValue head1_bias;
    core::TensorValue head_prelu_slope;
    core::TensorValue head2_weight;
};

SupertonicVocoderWeights load_vocoder_weights(const SupertonicBackendWeights & weights) {
    SupertonicVocoderWeights out;
    out.ttl_scale = require_weight_tensor(weights, "vocoder.tts.ttl.normalizer.scale");
    out.latent_mean = require_weight_tensor(weights, "vocoder.tts.ae.latent_mean");
    out.latent_std = require_weight_tensor(weights, "vocoder.tts.ae.latent_std");
    out.embed_weight = require_weight_tensor(weights, "vocoder.tts.ae.decoder.embed.net.weight");
    out.embed_bias = require_weight_tensor(weights, "vocoder.tts.ae.decoder.embed.net.bias");
    for (size_t i = 0; i < out.blocks.size(); ++i) {
        const std::string prefix = "vocoder.tts.ae.decoder.convnext." + std::to_string(i);
        auto & block = out.blocks[i];
        block.gamma = require_weight_tensor(weights, prefix + ".gamma");
        block.dw_weight = require_weight_tensor(weights, prefix + ".dwconv.net.weight");
        block.dw_bias = require_weight_tensor(weights, prefix + ".dwconv.net.bias");
        block.norm_weight = require_weight_tensor(weights, prefix + ".norm.norm.weight");
        block.norm_bias = require_weight_tensor(weights, prefix + ".norm.norm.bias");
        block.pw1_weight = require_weight_tensor(weights, prefix + ".pwconv1.weight");
        block.pw1_bias = require_weight_tensor(weights, prefix + ".pwconv1.bias");
        block.pw2_weight = require_weight_tensor(weights, prefix + ".pwconv2.weight");
        block.pw2_bias = require_weight_tensor(weights, prefix + ".pwconv2.bias");
    }
    out.final_norm_weight = require_weight_tensor(weights, "vocoder.tts.ae.decoder.final_norm.norm.weight");
    out.final_norm_bias = require_weight_tensor(weights, "vocoder.tts.ae.decoder.final_norm.norm.bias");
    out.final_norm_running_mean = require_weight_tensor(weights, "vocoder.tts.ae.decoder.final_norm.norm.running_mean");
    out.final_norm_running_var = require_weight_tensor(weights, "vocoder.tts.ae.decoder.final_norm.norm.running_var");
    out.head1_weight = require_weight_tensor(weights, "vocoder.tts.ae.decoder.head.layer1.net.weight");
    out.head1_bias = require_weight_tensor(weights, "vocoder.tts.ae.decoder.head.layer1.net.bias");
    out.head_prelu_slope = require_weight_tensor(weights, "vocoder.tts.ae.decoder.head.act.weight");
    out.head2_weight = require_weight_tensor(weights, "vocoder.tts.ae.decoder.head.layer2.weight");
    return out;
}

core::TensorValue edge_pad_time(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t left,
    int64_t right) {
    auto out = input;
    if (left > 0) {
        auto edge = modules::SliceModule({2, 0, 1}).build(ctx, input);
        edge = broadcast_to(ctx, edge, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], left}));
        out = modules::ConcatModule({2}).build(ctx, edge, out);
    }
    if (right > 0) {
        auto edge = modules::SliceModule({2, input.shape.dims[2] - 1, 1}).build(ctx, input);
        edge = broadcast_to(ctx, edge, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], right}));
        out = modules::ConcatModule({2}).build(ctx, out, edge);
    }
    return out;
}

core::TensorValue batch_norm_eval(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SupertonicVocoderWeights & weights) {
    const auto var_eps = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, weights.final_norm_running_var.tensor, 1.0F, 1.0e-5F),
        weights.final_norm_running_var.shape,
        GGML_TYPE_F32);
    const auto inv_std = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, var_eps.tensor),
        weights.final_norm_running_var.shape,
        GGML_TYPE_F32);
    const auto scale = core::wrap_tensor(
        ggml_div(ctx.ggml, weights.final_norm_weight.tensor, inv_std.tensor),
        weights.final_norm_weight.shape,
        GGML_TYPE_F32);
    const auto mean_scale = core::wrap_tensor(
        ggml_mul(ctx.ggml, weights.final_norm_running_mean.tensor, scale.tensor),
        weights.final_norm_running_mean.shape,
        GGML_TYPE_F32);
    const auto bias = core::wrap_tensor(
        ggml_sub(ctx.ggml, weights.final_norm_bias.tensor, mean_scale.tensor),
        weights.final_norm_bias.shape,
        GGML_TYPE_F32);
    return modules::BatchNorm1dEvalModule({input.shape.dims[1]}).build(ctx, input, {scale, bias});
}

core::TensorValue vocoder_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SupertonicVocoderBlockWeights & weights,
    int dilation) {
    const int64_t left_pad = dilation * 6;
    auto x = edge_pad_time(ctx, input, left_pad, 0);
    x = modules::DepthwiseConv1dModule({
        x.shape.dims[1],
        weights.dw_weight.shape.dims[2],
        1,
        0,
        dilation,
        true,
    }).build(ctx, x, {weights.dw_weight, weights.dw_bias});
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::LayerNormModule({x.shape.last_dim(), 1.0e-6F, true, true}).build(ctx, x, {weights.norm_weight, weights.norm_bias});
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::PointwiseConv1dModule({
        x.shape.dims[1],
        weights.pw1_weight.shape.dims[0],
        true,
    }).build(ctx, x, {weights.pw1_weight, weights.pw1_bias});
    x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
    x = modules::PointwiseConv1dModule({
        x.shape.dims[1],
        weights.pw2_weight.shape.dims[0],
        true,
    }).build(ctx, x, {weights.pw2_weight, weights.pw2_bias});
    const auto gamma = broadcast_to(ctx, weights.gamma, x.shape);
    x = core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, gamma.tensor), x.shape, GGML_TYPE_F32);
    return modules::ResidualAddModule{}.build(ctx, input, x);
}

class SupertonicVocoderRuntime {
public:
    SupertonicVocoderRuntime(
        std::shared_ptr<const SupertonicBackendWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        const SupertonicConfig & config,
        size_t arena_bytes,
        int threads)
        : weights_(load_vocoder_weights(*weights)),
          backend_(backend),
          backend_type_(backend_type),
          config_(config),
          arena_bytes_(arena_bytes),
          threads_(std::max(1, threads)) {
        if (backend_ == nullptr) {
            throw std::runtime_error("Supertonic vocoder requires backend");
        }
    }

    SupertonicTensorOutput run(const SupertonicLatentTensor & latent) {
        const auto total_start = std::chrono::steady_clock::now();
        CachedGraph & graph = graph_for(latent.shape);
        auto timing_start = std::chrono::steady_clock::now();
        upload(graph, latent);
        engine::debug::timing_log_scalar(
            "supertonic.vocoder.input_upload_ms",
            engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, graph.graph, nullptr, "supertonic.vocoder");
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Supertonic vocoder GGML model graph compute failed");
        }
        engine::debug::timing_log_scalar(
            "supertonic.vocoder.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        SupertonicTensorOutput result;
        result.shape = {latent.shape.at(0), latent.shape.at(2) * config_.chunk_compress_factor * config_.base_chunk_size};
        const auto full_output = core::read_tensor_f32(graph.output);
        result.values = slice_padded_values(full_output, shape_to_vector(graph.output_shape), result.shape);
        engine::debug::timing_log_scalar(
            "supertonic.vocoder.output_read_ms",
            engine::debug::elapsed_ms(timing_start));
        engine::debug::timing_log_scalar(
            "supertonic.vocoder.total_ms",
            engine::debug::elapsed_ms(total_start));
        return result;
    }

private:
    struct CachedGraph {
        ~CachedGraph() {
            core::release_backend_graph_resources(backend, graph);
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
            }
            if (io_buffer != nullptr) {
                ggml_backend_buffer_free(io_buffer);
            }
        }

        std::unique_ptr<ggml_context, GgmlContextDeleter> io_ggml;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ggml;
        ggml_backend_buffer_t io_buffer = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_t backend = nullptr;
        std::vector<int64_t> input_shape;
        core::TensorShape output_shape = {};
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
    };

    CachedGraph & graph_for(const std::vector<int64_t> & input_shape) {
        if (auto * found = graphs_.find(input_shape)) {
            return **found;
        }
        return prepare(input_shape);
    }

    CachedGraph & prepare(const std::vector<int64_t> & input_shape) {
        const auto build_start = std::chrono::steady_clock::now();
        auto prepared = std::make_unique<CachedGraph>();
        prepared->input_shape = input_shape;
        prepared->backend = backend_;
        ggml_init_params params{arena_bytes_, nullptr, true};
        ggml_init_params io_params{kSupertonicIoArenaBytes, nullptr, true};
        prepared->io_ggml.reset(ggml_init(io_params));
        prepared->ggml.reset(ggml_init(params));
        if (prepared->io_ggml == nullptr || prepared->ggml == nullptr) {
            throw std::runtime_error("failed to initialize Supertonic vocoder GGML context");
        }

        core::ModuleBuildContext io_ctx{prepared->io_ggml.get(), "supertonic.vocoder.io", backend_type_};
        core::ModuleBuildContext ctx{prepared->ggml.get(), "supertonic.vocoder", backend_type_};
        auto latent = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(input_shape));
        ggml_set_input(latent.tensor);
        prepared->input = latent.tensor;
        auto output = build_graph(ctx, latent);
        output = core::wrap_tensor(ggml_cont(ctx.ggml, output.tensor), output.shape, output.type);
        prepared->output = output.tensor;
        prepared->output_shape = output.shape;
        ggml_set_output(prepared->output);
        prepared->graph = ggml_new_graph_custom(ctx.ggml, 65536, false);
        ggml_build_forward_expand(prepared->graph, prepared->output);
        prepared->io_buffer = ggml_backend_alloc_ctx_tensors(prepared->io_ggml.get(), backend_);
        if (prepared->io_buffer == nullptr) {
            throw std::runtime_error("failed to allocate Supertonic vocoder IO buffer");
        }
        prepared->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (prepared->gallocr == nullptr ||
            !ggml_gallocr_reserve(prepared->gallocr, prepared->graph) ||
            !ggml_gallocr_alloc_graph(prepared->gallocr, prepared->graph)) {
            throw std::runtime_error("failed to allocate Supertonic vocoder backend graph buffer");
        }
        graphs_.put(input_shape, std::move(prepared));
        engine::debug::timing_log_scalar(
            "supertonic.vocoder.prepare_ms",
            engine::debug::elapsed_ms(build_start));
        auto * cached = graphs_.find(input_shape);
        if (cached == nullptr) {
            throw std::runtime_error("Supertonic vocoder graph cache insert failed");
        }
        return **cached;
    }

    core::TensorValue build_graph(core::ModuleBuildContext & ctx, const core::TensorValue & latent) const {
        const int64_t batch = latent.shape.dims[0];
        const int64_t latent_length = latent.shape.dims[2];
        const int64_t frames = latent_length * config_.chunk_compress_factor;
        auto scale = broadcast_to(ctx, weights_.ttl_scale, latent.shape);
        auto x = core::wrap_tensor(ggml_div(ctx.ggml, latent.tensor, scale.tensor), latent.shape, GGML_TYPE_F32);
        x = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::TensorShape::from_dims({batch, config_.latent_dim, config_.chunk_compress_factor, latent_length}));
        x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
        x = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::TensorShape::from_dims({batch, config_.latent_dim, frames}));
        auto latent_std = broadcast_to(ctx, weights_.latent_std, x.shape);
        auto latent_mean = broadcast_to(ctx, weights_.latent_mean, x.shape);
        x = core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, latent_std.tensor), x.shape, GGML_TYPE_F32);
        x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, latent_mean.tensor), x.shape, GGML_TYPE_F32);

        x = edge_pad_time(ctx, x, 6, 0);
        x = modules::Conv1dModule({
            x.shape.dims[1],
            weights_.embed_weight.shape.dims[0],
            weights_.embed_weight.shape.dims[2],
            1,
            0,
            1,
            true,
        }).build(ctx, x, {weights_.embed_weight, weights_.embed_bias});
        constexpr std::array<int, 10> kDilations = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
        for (size_t i = 0; i < weights_.blocks.size(); ++i) {
            x = vocoder_block(ctx, x, weights_.blocks[i], kDilations[i]);
        }
        x = batch_norm_eval(ctx, x, weights_);
        x = edge_pad_time(ctx, x, 2, 0);
        x = modules::Conv1dModule({
            x.shape.dims[1],
            weights_.head1_weight.shape.dims[0],
            weights_.head1_weight.shape.dims[2],
            1,
            0,
            1,
            true,
        }).build(ctx, x, {weights_.head1_weight, weights_.head1_bias});
        x = modules::PReluModule().build(ctx, x, weights_.head_prelu_slope);
        x = modules::Conv1dModule({
            x.shape.dims[1],
            weights_.head2_weight.shape.dims[0],
            1,
            1,
            0,
            1,
            false,
        }).build(ctx, x, {weights_.head2_weight, std::nullopt});
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        return core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::TensorShape::from_dims({batch, frames * config_.base_chunk_size}));
    }

    void upload(CachedGraph & graph, const SupertonicLatentTensor & latent) {
        upload_f32_tensor(graph.input, graph.input_shape, latent.shape, latent.f32, "vocoder.latent");
    }

    SupertonicVocoderWeights weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    SupertonicConfig config_;
    size_t arena_bytes_ = 0;
    int threads_ = 1;
    runtime::CacheSlots<std::vector<int64_t>, std::unique_ptr<CachedGraph>> graphs_{kSupertonicLargeGraphCacheSlots};
};

class SupertonicNetwork {
public:
    SupertonicNetwork(
        core::ModuleBuildContext & ctx,
        core::ModuleBuildContext & tensor_ctx,
        const SupertonicBackendWeights & weights,
        core::DeferredTensorWriter & tensor_writer)
        : ctx_(ctx),
          tensor_ctx_(tensor_ctx),
          weights_(weights),
          tensor_writer_(tensor_writer),
          ggml_(ctx.ggml) {}

    core::TensorValue duration_predictor(
        const core::TensorValue & text_ids,
        const core::TensorValue & text_mask,
        const core::TensorValue & style_dp) {
        auto x_btc = modules::EmbeddingModule({8322, 64}).build(ctx_, text_ids, weight("duration_predictor.tts.dp.sentence_encoder.text_embedder.char_embedder.weight"));
        const auto text_mask_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, text_mask);
        x_btc = mul(x_btc, text_mask_btc);
        auto sentence_mask = modules::SliceModule({2, 0, 1}).build(ctx_, text_mask);
        auto sentence_token = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, weight("duration_predictor.tts.dp.sentence_encoder.sentence_token"));
        sentence_token = core::ensure_backend_addressable_layout(ctx_, sentence_token);
        x_btc = modules::ConcatModule({1}).build(ctx_, sentence_token, core::ensure_backend_addressable_layout(ctx_, x_btc));
        auto encoder_mask = modules::ConcatModule({2}).build(ctx_, sentence_mask, text_mask);
        auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x_btc);
        x = core::ensure_backend_addressable_layout(ctx_, x);
        x = convnext(x, "duration_predictor.tts.dp.sentence_encoder.convnext", {1, 1, 1, 1, 1, 1});
        auto convnext_out = x;
        x = self_encoder(x, encoder_mask, "duration_predictor.tts.dp.sentence_encoder.attn_encoder", 2, 2);
        x = add(x, convnext_out);
        x = modules::SliceModule({2, 0, 1}).build(ctx_, x);
        x = modules::Conv1dModule({
            x.shape.dims[1],
            weight("duration_predictor.tts.dp.sentence_encoder.proj_out.net.weight").shape.dims[0],
            1,
            1,
            0,
            1,
            false,
        }).build(ctx_, x, {weight("duration_predictor.tts.dp.sentence_encoder.proj_out.net.weight"), std::nullopt});
        x = mul(x, sentence_mask);
        auto pooled = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, x), core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1]}));
        auto style_flat = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, style_dp), core::TensorShape::from_dims({style_dp.shape.dims[0], style_dp.shape.dims[1] * style_dp.shape.dims[2]}));
        auto joined = modules::ConcatModule({1}).build(ctx_, pooled, style_flat);
        auto h = modules::LinearModule({
            joined.shape.last_dim(),
            weight("duration_predictor.tts.dp.predictor.layers.0.weight").shape.dims[0],
            true,
        }).build(
            ctx_,
            joined,
            {weight("duration_predictor.tts.dp.predictor.layers.0.weight"), weight("duration_predictor.tts.dp.predictor.layers.0.bias")});
        h = modules::PReluModule().build(ctx_, h,
            weight("duration_predictor.tts.dp.predictor.activation.weight"));
        auto duration = modules::LinearModule({
            h.shape.last_dim(),
            weight("duration_predictor.tts.dp.predictor.layers.1.weight").shape.dims[0],
            true,
        }).build(
            ctx_,
            h,
            {weight("duration_predictor.tts.dp.predictor.layers.1.weight"), weight("duration_predictor.tts.dp.predictor.layers.1.bias")});
        duration = core::wrap_tensor(ggml_exp(ggml_, duration.tensor), duration.shape, GGML_TYPE_F32);
        return core::reshape_tensor(ctx_, duration, core::TensorShape::from_dims({duration.shape.dims[0]}));
    }

    core::TensorValue text_encoder(
        const core::TensorValue & text_ids,
        const core::TensorValue & text_mask,
        const core::TensorValue & style_ttl) {
        auto x = modules::EmbeddingModule({8322, 256}).build(ctx_, text_ids, weight("text_encoder.tts.ttl.text_encoder.text_embedder.char_embedder.weight"));
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
        x = core::ensure_backend_addressable_layout(ctx_, x);
        x = convnext(x, "text_encoder.tts.ttl.text_encoder.convnext", {1, 1, 2, 2, 4, 4});
        auto convnext_out = x;
        x = self_encoder(x, text_mask, "text_encoder.tts.ttl.text_encoder.attn_encoder", 4, 4);
        x = mul(add(x, convnext_out), text_mask);
        auto style_bct = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, style_ttl);
        style_bct = core::ensure_backend_addressable_layout(ctx_, style_bct);
        auto style_key_bct = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, weight("text_encoder.tts.ttl.style_encoder.style_token_layer.style_key"));
        style_key_bct = core::ensure_backend_addressable_layout(ctx_, style_key_bct);
        auto prompted_input = x;
        x = add(prompted_input, text_cross_attention(prompted_input, style_key_bct, "text_encoder.tts.ttl.speech_prompted_text_encoder.attention1", 2, text_mask, true, false, nullptr, &style_bct));
        x = add(prompted_input, text_cross_attention(x, style_key_bct, "text_encoder.tts.ttl.speech_prompted_text_encoder.attention2", 2, text_mask, true, false, nullptr, &style_bct));
        auto x_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
        x_btc = modules::LayerNormModule({x_btc.shape.last_dim(), 1.0e-6F, true, true}).build(
            ctx_,
            x_btc,
            {
                weight("text_encoder.tts.ttl.speech_prompted_text_encoder.norm.norm.weight"),
                weight("text_encoder.tts.ttl.speech_prompted_text_encoder.norm.norm.bias"),
            });
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x_btc);
        x = core::ensure_backend_addressable_layout(ctx_, x);
        return mul(x, text_mask);
    }

    core::TensorValue vector_estimator_steps(
        const core::TensorValue & noisy_latent,
        const core::TensorValue & text_embedding,
        const core::TensorValue & text_mask,
        const core::TensorValue & style_ttl,
        const core::TensorValue & latent_mask,
        const core::TensorValue & start_step,
        const core::TensorValue & total_step,
        int step_count) {
        auto state = vector_conditioning(text_embedding, text_mask, style_ttl, latent_mask);
        auto x = noisy_latent;
        for (int offset = 0; offset < step_count; ++offset) {
            auto current_step = start_step;
            if (offset != 0) {
                current_step = add(start_step, constant_scalar(static_cast<float>(offset), "vector_estimator.step_offset"));
            }
            x = vector_estimator_step(x, state, current_step, total_step);
        }
        return x;
    }

private:
    struct VectorConditioning {
        core::TensorValue latent_mask;
        core::TensorValue mask_pair;
        core::TensorValue text_pair;
        core::TensorValue text_mask_pair;
        core::TensorValue style_key_pair;
        core::TensorValue style_value_pair;
    };

    VectorConditioning vector_conditioning(
        const core::TensorValue & text_embedding,
        const core::TensorValue & text_mask,
        const core::TensorValue & style_ttl,
        const core::TensorValue & latent_mask) {
        auto uncond_text = broadcast_to(
            ctx_,
            weight("vector_estimator.vector_estimator.tts.ttl.uncond_masker.text_special_token"),
            text_embedding.shape);
        VectorConditioning state;
        state.latent_mask = latent_mask;
        state.mask_pair = modules::ConcatModule({0}).build(ctx_, latent_mask, latent_mask);
        state.text_pair = modules::ConcatModule({0}).build(ctx_, text_embedding, uncond_text);
        state.text_mask_pair = modules::ConcatModule({0}).build(ctx_, text_mask, text_mask);
        auto style_key = broadcast_to(ctx_, weight("vector_estimator.vector_estimator.Expand_output_0"), style_ttl.shape);
        state.style_key_pair = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(
            ctx_,
            modules::ConcatModule({0}).build(
                ctx_,
                style_key,
                weight("vector_estimator.vector_estimator.tts.ttl.uncond_masker.style_key_special_token")));
        state.style_value_pair = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(
            ctx_,
            modules::ConcatModule({0}).build(
                ctx_,
                style_ttl,
                weight("vector_estimator.vector_estimator.tts.ttl.uncond_masker.style_value_special_token")));
        return state;
    }

    core::TensorValue vector_estimator_step(
        const core::TensorValue & noisy_latent,
        const VectorConditioning & state,
        const core::TensorValue & current_step,
        const core::TensorValue & total_step) {
        auto time = time_embedding(current_step, total_step);
        auto latent_pair = modules::ConcatModule({0}).build(ctx_, noisy_latent, noisy_latent);
        const auto & proj_in_weight = weight("vector_estimator.vector_estimator.tts.ttl.vector_field.proj_in.net.weight");
        auto latent_pair_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, latent_pair);
        latent_pair_btc = core::ensure_backend_addressable_layout(ctx_, latent_pair_btc);
        const auto proj_in_weight_2d = core::reshape_tensor(
            ctx_,
            proj_in_weight,
            core::TensorShape::from_dims({proj_in_weight.shape.dims[0], proj_in_weight.shape.dims[1]}));
        auto x = modules::LinearModule({latent_pair.shape.dims[1], proj_in_weight.shape.dims[0], false}).build(
            ctx_,
            latent_pair_btc,
            {proj_in_weight_2d, std::nullopt});
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
        x = mul(x, state.mask_pair);
        for (int group = 0; group < 4; ++group) {
            const int base = group * 6;
            x = vector_estimator_convnext_btc_pointwise(x, "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks." + std::to_string(base), {1, 2, 4, 8}, state.mask_pair);
            x = add_time_condition(x, time, base + 1);
            x = vector_estimator_convnext(x, "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks." + std::to_string(base + 2), {1}, state.mask_pair);
            x = text_memory_block(x, state.text_pair, state.text_mask_pair, state.mask_pair, base + 3);
            x = vector_estimator_convnext(x, "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks." + std::to_string(base + 4), {1}, state.mask_pair);
            x = style_memory_block(x, state.style_key_pair, state.style_value_pair, state.mask_pair, base + 5);
        }
        x = vector_estimator_convnext(x, "vector_estimator.vector_estimator.tts.ttl.vector_field.last_convnext", {1, 1, 1, 1}, state.mask_pair);
        const auto & proj_out_weight = weight("vector_estimator.vector_estimator.tts.ttl.vector_field.proj_out.net.weight");
        auto x_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
        x_btc = core::ensure_backend_addressable_layout(ctx_, x_btc);
        const auto proj_out_weight_2d = core::reshape_tensor(
            ctx_,
            proj_out_weight,
            core::TensorShape::from_dims({proj_out_weight.shape.dims[0], proj_out_weight.shape.dims[1]}));
        x = modules::LinearModule({x.shape.dims[1], proj_out_weight.shape.dims[0], false}).build(
            ctx_,
            x_btc,
            {proj_out_weight_2d, std::nullopt});
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
        x = mul(x, state.mask_pair);
        auto conditional = modules::SliceModule({0, 0, 1}).build(ctx_, x);
        auto unconditional = modules::SliceModule({0, 1, 1}).build(ctx_, x);
        auto guided = core::wrap_tensor(
            ggml_sub(
                ggml_,
                ggml_scale(ggml_, core::ensure_backend_addressable_layout(ctx_, conditional).tensor, 4.0F),
                ggml_scale(ggml_, core::ensure_backend_addressable_layout(ctx_, unconditional).tensor, 3.0F)),
            conditional.shape,
            GGML_TYPE_F32);
        x = add(noisy_latent, div(guided, total_step));
        return mul(x, state.latent_mask);
    }

    const core::TensorValue & weight(const std::string & name) const {
        return require_weight_tensor(weights_, name);
    }

    core::TensorValue constant_scalar(float value, const std::string & label) {
        (void)label;
        return tensor_writer_.make_f32_tensor(tensor_ctx_, core::TensorShape::from_dims({1}), std::vector<float>{value});
    }

    core::TensorValue add(const core::TensorValue & lhs, const core::TensorValue & rhs) {
        const auto shape = binary_broadcast_shape(lhs.shape, rhs.shape);
        auto a = broadcast_to(ctx_, lhs, shape);
        auto b = broadcast_to(ctx_, rhs, shape);
        return modules::AddModule().build(ctx_, a, b);
    }

    core::TensorValue mul(const core::TensorValue & lhs, const core::TensorValue & rhs) {
        const auto shape = binary_broadcast_shape(lhs.shape, rhs.shape);
        auto a = broadcast_to(ctx_, lhs, shape);
        auto b = broadcast_to(ctx_, rhs, shape);
        return modules::MulModule().build(ctx_, a, b);
    }

    core::TensorValue div(const core::TensorValue & lhs, const core::TensorValue & rhs) {
        const auto shape = binary_broadcast_shape(lhs.shape, rhs.shape);
        auto a = broadcast_to(ctx_, lhs, shape);
        auto b = broadcast_to(ctx_, rhs, shape);
        return core::wrap_tensor(ggml_div(ggml_, a.tensor, b.tensor), shape, GGML_TYPE_F32);
    }

    core::TensorValue sub(const core::TensorValue & lhs, const core::TensorValue & rhs) {
        const auto shape = binary_broadcast_shape(lhs.shape, rhs.shape);
        auto a = broadcast_to(ctx_, lhs, shape);
        auto b = broadcast_to(ctx_, rhs, shape);
        return core::wrap_tensor(ggml_sub(ggml_, a.tensor, b.tensor), shape, GGML_TYPE_F32);
    }

    core::TensorValue gelu(const core::TensorValue & value) {
        return modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx_, value);
    }

    core::TensorValue text_cross_attention(
        const core::TensorValue & query_bct,
        const core::TensorValue & text_bct,
        const std::string & prefix,
        int64_t heads,
        const std::optional<core::TensorValue> & output_mask_bct = std::nullopt,
        bool tanh_key = true,
        bool rotary = false,
        const core::TensorValue * memory_mask_bct = nullptr,
        const core::TensorValue * value_bct = nullptr) {
        auto query = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, query_bct);
        auto memory = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, text_bct);
        auto value_memory = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, value_bct == nullptr ? text_bct : *value_bct);
        const auto & q_weight = weight(prefix + ".W_query.linear.weight");
        auto q_weight_t = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, q_weight);
        q_weight_t = core::ensure_backend_addressable_layout(ctx_, q_weight_t);
        auto q = modules::LinearModule({query.shape.last_dim(), q_weight.shape.dims[1], true}).build(
            ctx_,
            query,
            {q_weight_t, weight(prefix + ".W_query.linear.bias")});
        const auto & k_weight = weight(prefix + ".W_key.linear.weight");
        auto k_weight_t = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, k_weight);
        k_weight_t = core::ensure_backend_addressable_layout(ctx_, k_weight_t);
        auto k = modules::LinearModule({memory.shape.last_dim(), k_weight.shape.dims[1], true}).build(
            ctx_,
            memory,
            {k_weight_t, weight(prefix + ".W_key.linear.bias")});
        const auto & v_weight = weight(prefix + ".W_value.linear.weight");
        auto v_weight_t = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, v_weight);
        v_weight_t = core::ensure_backend_addressable_layout(ctx_, v_weight_t);
        auto v = modules::LinearModule({value_memory.shape.last_dim(), v_weight.shape.dims[1], true}).build(
            ctx_,
            value_memory,
            {v_weight_t, weight(prefix + ".W_value.linear.bias")});
        const int64_t projection_dim = q.shape.last_dim();
        const int64_t head_dim = projection_dim / heads;
        q = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, q), core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], heads, head_dim}));
        k = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, k), core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], heads, head_dim}));
        v = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, v), core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], heads, head_dim}));
        q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, q);
        k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, k);
        v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, v);
        q = modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx_, q);
        k = modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx_, k);
        v = modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx_, v);
        if (rotary) {
            if (!output_mask_bct.has_value() || memory_mask_bct == nullptr) {
                throw std::runtime_error("Supertonic rotary attention requires query and memory masks");
            }
            q = apply_rotary64(q, *output_mask_bct);
            k = apply_rotary64(k, *memory_mask_bct);
        }
        if (tanh_key) {
            k = core::wrap_tensor(ggml_tanh(ggml_, core::ensure_backend_addressable_layout(ctx_, k).tensor), k.shape, GGML_TYPE_F32);
        }
        const auto k_t = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx_, k);
        auto scores = modules::MatMulModule().build(ctx_, q, k_t);
        scores = core::wrap_tensor(
            ggml_scale(ggml_, scores.tensor, 1.0F / 16.0F),
            scores.shape,
            GGML_TYPE_F32);
        std::optional<core::TensorValue> key_mask;
        if (memory_mask_bct != nullptr) {
            key_mask = core::reshape_tensor(
                ctx_,
                core::ensure_backend_addressable_layout(ctx_, *memory_mask_bct),
                core::TensorShape::from_dims({1, memory_mask_bct->shape.dims[0], 1, memory_mask_bct->shape.dims[2]}));
            auto score_mask = core::wrap_tensor(
                ggml_scale_bias(ggml_, key_mask->tensor, 1.0e30F, -1.0e30F),
                key_mask->shape,
                GGML_TYPE_F32);
            scores = add(scores, broadcast_to(ctx_, score_mask, scores.shape));
        }
        auto attn = core::wrap_tensor(
            ggml_soft_max(ggml_, core::ensure_backend_addressable_layout(ctx_, scores).tensor),
            scores.shape,
            GGML_TYPE_F32);
        if (key_mask.has_value()) {
            attn = mul(attn, broadcast_to(ctx_, *key_mask, attn.shape));
        }
        auto context = modules::MatMulModule().build(ctx_, attn, v);
        context = modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx_, context);
        context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, context);
        context = core::reshape_tensor(
            ctx_,
            core::ensure_backend_addressable_layout(ctx_, context),
            core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], heads * head_dim}));
        const auto & out_weight = weight(prefix + ".out_fc.linear.weight");
        auto out_weight_t = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, out_weight);
        out_weight_t = core::ensure_backend_addressable_layout(ctx_, out_weight_t);
        context = modules::LinearModule({context.shape.last_dim(), out_weight.shape.dims[1], true}).build(
            ctx_,
            context,
            {out_weight_t, weight(prefix + ".out_fc.linear.bias")});
        auto out = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, context);
        out = core::ensure_backend_addressable_layout(ctx_, out);
        if (output_mask_bct.has_value()) {
            out = mul(out, *output_mask_bct);
        }
        return out;
    }

    core::TensorValue apply_rotary64(const core::TensorValue & value, const core::TensorValue & mask_bct) {
        const int64_t frames = value.shape.dims[2];
        const auto & positions = rotary_positions(frames);
        auto active_frames = modules::ReduceSumModule({2}).build(ctx_, mask_bct);
        active_frames = core::reshape_tensor(
            ctx_,
            core::ensure_backend_addressable_layout(ctx_, active_frames),
            core::TensorShape::from_dims({1, mask_bct.shape.dims[0], 1, 1}));
        auto theta = core::reshape_tensor(
            ctx_,
            core::ensure_backend_addressable_layout(
                ctx_,
                weight("vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.theta")),
            core::TensorShape::from_dims({1, 1, 1, 32}));
        auto angles = mul(div(positions, active_frames), theta);
        auto cos_part = core::wrap_tensor(ggml_cos(ggml_, core::ensure_backend_addressable_layout(ctx_, angles).tensor), angles.shape, GGML_TYPE_F32);
        auto sin_part = core::wrap_tensor(ggml_sin(ggml_, core::ensure_backend_addressable_layout(ctx_, angles).tensor), angles.shape, GGML_TYPE_F32);
        auto left = modules::SliceModule({3, 0, 32}).build(ctx_, value);
        auto right = modules::SliceModule({3, 32, 32}).build(ctx_, value);
        auto rotated_left = sub(mul(left, cos_part), mul(right, sin_part));
        auto rotated_right = add(mul(left, sin_part), mul(right, cos_part));
        return modules::ConcatModule({3}).build(ctx_, rotated_left, rotated_right);
    }

    const core::TensorValue & rotary_positions(int64_t frames) {
        const auto found = rotary_positions_.find(frames);
        if (found != rotary_positions_.end()) {
            return found->second;
        }
        std::vector<float> position_values(static_cast<size_t>(frames));
#ifdef _OPENMP
#pragma omp parallel for if(frames >= 4096)
#endif
        for (int64_t i = 0; i < frames; ++i) {
            position_values[static_cast<size_t>(i)] = static_cast<float>(i);
        }
        auto positions = tensor_writer_.make_f32_tensor(
            tensor_ctx_,
            core::TensorShape::from_dims({1, 1, frames, 1}),
            position_values);
        return rotary_positions_.emplace(frames, positions).first->second;
    }

    const core::TensorValue & relative_position_mask(int64_t frames, int64_t offset) {
        const int64_t key = frames * 32 + (offset + 16);
        const auto found = relative_position_masks_.find(key);
        if (found != relative_position_masks_.end()) {
            return found->second;
        }
        std::vector<float> values(static_cast<size_t>(frames * frames), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frames * frames >= 4096)
#endif
        for (int64_t query = 0; query < frames; ++query) {
            for (int64_t key = 0; key < frames; ++key) {
                const int64_t distance = key - query;
                if (distance == offset) {
                    values[static_cast<size_t>(query * frames + key)] = 1.0F;
                }
            }
        }
        auto mask = tensor_writer_.make_f32_tensor(
            tensor_ctx_,
            core::TensorShape::from_dims({1, 1, frames, frames}),
            values);
        return relative_position_masks_.emplace(key, mask).first->second;
    }

    core::TensorValue relative_key_scores(
        const core::TensorValue & q_scaled,
        const std::string & prefix,
        const core::TensorShape & score_shape) {
        const int64_t frames = score_shape.dims[2];
        std::optional<core::TensorValue> total;
        for (int64_t offset = -4; offset <= 4; ++offset) {
            auto rel = modules::SliceModule({1, offset + 4, 1}).build(ctx_, weight(prefix + ".emb_rel_k"));
            rel = core::reshape_tensor(
                ctx_,
                core::ensure_backend_addressable_layout(ctx_, rel),
                core::TensorShape::from_dims({1, 1, 1, rel.shape.dims[2]}));
            auto logits = modules::ReduceSumModule({3}).build(ctx_, mul(q_scaled, rel));
            logits = broadcast_to(ctx_, logits, score_shape);
            logits = mul(logits, relative_position_mask(frames, offset));
            total = total.has_value() ? add(*total, logits) : logits;
        }
        return *total;
    }

    core::TensorValue relative_value_context(
        const core::TensorValue & attn,
        const std::string & prefix,
        int64_t head_dim) {
        const int64_t frames = attn.shape.dims[2];
        const auto context_shape = core::TensorShape::from_dims({attn.shape.dims[0], attn.shape.dims[1], frames, head_dim});
        std::optional<core::TensorValue> total;
        for (int64_t offset = -4; offset <= 4; ++offset) {
            auto diagonal = mul(attn, relative_position_mask(frames, offset));
            auto weight_sum = modules::ReduceSumModule({3}).build(ctx_, diagonal);
            auto rel = modules::SliceModule({1, offset + 4, 1}).build(ctx_, weight(prefix + ".emb_rel_v"));
            rel = core::reshape_tensor(
                ctx_,
                core::ensure_backend_addressable_layout(ctx_, rel),
                core::TensorShape::from_dims({1, 1, 1, rel.shape.dims[2]}));
            auto contribution = broadcast_to(ctx_, mul(weight_sum, rel), context_shape);
            total = total.has_value() ? add(*total, contribution) : contribution;
        }
        return *total;
    }

    core::TensorValue conv_self_attention(
        const core::TensorValue & x_bct,
        const std::string & prefix,
        int64_t heads) {
        const auto & q_weight = weight(prefix + ".conv_q.weight");
        auto q = modules::Conv1dModule({
            x_bct.shape.dims[1],
            q_weight.shape.dims[0],
            1,
            1,
            0,
            1,
            true,
        }).build(ctx_, x_bct, {q_weight, weight(prefix + ".conv_q.bias")});
        const auto & k_weight = weight(prefix + ".conv_k.weight");
        auto k = modules::Conv1dModule({
            x_bct.shape.dims[1],
            k_weight.shape.dims[0],
            1,
            1,
            0,
            1,
            true,
        }).build(ctx_, x_bct, {k_weight, weight(prefix + ".conv_k.bias")});
        const auto & v_weight = weight(prefix + ".conv_v.weight");
        auto v = modules::Conv1dModule({
            x_bct.shape.dims[1],
            v_weight.shape.dims[0],
            1,
            1,
            0,
            1,
            true,
        }).build(ctx_, x_bct, {v_weight, weight(prefix + ".conv_v.bias")});
        q = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, q);
        k = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, k);
        v = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, v);
        const int64_t head_dim = q.shape.last_dim() / heads;
        q = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, q), core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], heads, head_dim}));
        k = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, k), core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], heads, head_dim}));
        v = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, v), core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], heads, head_dim}));
        q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, q);
        k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, k);
        v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, v);
        auto q_scaled = core::wrap_tensor(
            ggml_scale(ggml_, core::ensure_backend_addressable_layout(ctx_, q).tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
            q.shape,
            GGML_TYPE_F32);
        const auto k_t = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx_, k);
        auto scores = modules::MatMulModule().build(ctx_, q_scaled, k_t);
        scores = add(scores, relative_key_scores(q_scaled, prefix, scores.shape));
        auto attn = core::wrap_tensor(
            ggml_soft_max(ggml_, core::ensure_backend_addressable_layout(ctx_, scores).tensor),
            scores.shape,
            GGML_TYPE_F32);
        auto context = modules::MatMulModule().build(ctx_, attn, v);
        context = add(context, relative_value_context(attn, prefix, head_dim));
        context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, context);
        context = core::reshape_tensor(
            ctx_,
            core::ensure_backend_addressable_layout(ctx_, context),
            core::TensorShape::from_dims({x_bct.shape.dims[0], x_bct.shape.dims[2], x_bct.shape.dims[1]}));
        context = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, context);
        const auto & out_weight = weight(prefix + ".conv_o.weight");
        return modules::Conv1dModule({
            context.shape.dims[1],
            out_weight.shape.dims[0],
            1,
            1,
            0,
            1,
            true,
        }).build(ctx_, context, {out_weight, weight(prefix + ".conv_o.bias")});
    }

    core::TensorValue convnext(core::TensorValue x, const std::string & prefix, const std::vector<int> & dilations) {
        for (size_t i = 0; i < dilations.size(); ++i) {
            const std::string p = prefix + ".convnext." + std::to_string(i);
            auto y = edge_pad_time(ctx_, x, static_cast<int64_t>(dilations[i]) * 2, static_cast<int64_t>(dilations[i]) * 2);
            const auto & dw_weight = weight(p + ".dwconv.weight");
            y = modules::DepthwiseConv1dModule({
                y.shape.dims[1],
                dw_weight.shape.dims[2],
                1,
                0,
                dilations[i],
                true,
            }).build(ctx_, y, {dw_weight, weight(p + ".dwconv.bias")});
            y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            y = modules::LayerNormModule({x.shape.dims[1], 1.0e-6F, true, true}).build(ctx_, y, {weight(p + ".norm.norm.weight"), weight(p + ".norm.norm.bias")});
            y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            const auto & pw1_weight = weight(p + ".pwconv1.weight");
            auto y_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            y_btc = core::ensure_backend_addressable_layout(ctx_, y_btc);
            const auto pw1_weight_2d = core::reshape_tensor(ctx_, pw1_weight, core::TensorShape::from_dims({pw1_weight.shape.dims[0], pw1_weight.shape.dims[1]}));
            y_btc = modules::LinearModule({y.shape.dims[1], pw1_weight.shape.dims[0], true}).build(
                ctx_,
                y_btc,
                {pw1_weight_2d, weight(p + ".pwconv1.bias")});
            y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y_btc);
            y = gelu(y);
            const auto & pw2_weight = weight(p + ".pwconv2.weight");
            y_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            y_btc = core::ensure_backend_addressable_layout(ctx_, y_btc);
            const auto pw2_weight_2d = core::reshape_tensor(ctx_, pw2_weight, core::TensorShape::from_dims({pw2_weight.shape.dims[0], pw2_weight.shape.dims[1]}));
            y_btc = modules::LinearModule({y.shape.dims[1], pw2_weight.shape.dims[0], true}).build(
                ctx_,
                y_btc,
                {pw2_weight_2d, weight(p + ".pwconv2.bias")});
            y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y_btc);
            y = mul(y, weight(p + ".gamma"));
            x = add(x, y);
        }
        return x;
    }

    core::TensorValue vector_estimator_convnext(
        core::TensorValue x,
        const std::string & prefix,
        const std::vector<int> & dilations,
        const core::TensorValue & mask) {
        for (size_t i = 0; i < dilations.size(); ++i) {
            const std::string p = prefix + ".convnext." + std::to_string(i);
            auto residual = mul(x, mask);
            auto y = edge_pad_time(ctx_, residual, static_cast<int64_t>(dilations[i]) * 2, static_cast<int64_t>(dilations[i]) * 2);
            const auto & dw_weight = weight(p + ".dwconv.weight");
            y = modules::DepthwiseConv1dModule({
                y.shape.dims[1],
                dw_weight.shape.dims[2],
                1,
                0,
                dilations[i],
                true,
            }).build(ctx_, y, {dw_weight, weight(p + ".dwconv.bias")});
            y = mul(y, mask);
            y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            y = modules::LayerNormModule({x.shape.dims[1], 1.0e-6F, true, true}).build(ctx_, y, {weight(p + ".norm.norm.weight"), weight(p + ".norm.norm.bias")});
            y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            const auto & pw1_weight = weight(p + ".pwconv1.weight");
            y = modules::Conv1dModule({
                y.shape.dims[1],
                pw1_weight.shape.dims[0],
                1,
                1,
                0,
                1,
                true,
            }).build(ctx_, y, {pw1_weight, weight(p + ".pwconv1.bias")});
            y = gelu(y);
            const auto & pw2_weight = weight(p + ".pwconv2.weight");
            y = modules::Conv1dModule({
                y.shape.dims[1],
                pw2_weight.shape.dims[0],
                1,
                1,
                0,
                1,
                true,
            }).build(ctx_, y, {pw2_weight, weight(p + ".pwconv2.bias")});
            y = mul(y, weight(p + ".gamma"));
            x = add(residual, y);
        }
        return x;
    }

    core::TensorValue vector_estimator_convnext_btc_pointwise(
        core::TensorValue x,
        const std::string & prefix,
        const std::vector<int> & dilations,
        const core::TensorValue & mask) {
        for (size_t i = 0; i < dilations.size(); ++i) {
            const std::string p = prefix + ".convnext." + std::to_string(i);
            auto residual = mul(x, mask);
            auto y = edge_pad_time(ctx_, residual, static_cast<int64_t>(dilations[i]) * 2, static_cast<int64_t>(dilations[i]) * 2);
            const auto & dw_weight = weight(p + ".dwconv.weight");
            y = modules::DepthwiseConv1dModule({
                y.shape.dims[1],
                dw_weight.shape.dims[2],
                1,
                0,
                dilations[i],
                true,
            }).build(ctx_, y, {dw_weight, weight(p + ".dwconv.bias")});
            y = mul(y, mask);
            auto y_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            y_btc = modules::LayerNormModule({x.shape.dims[1], 1.0e-6F, true, true}).build(
                ctx_,
                y_btc,
                {weight(p + ".norm.norm.weight"), weight(p + ".norm.norm.bias")});
            const auto & pw1_weight = weight(p + ".pwconv1.weight");
            const auto pw1_weight_2d = core::reshape_tensor(ctx_, pw1_weight, core::TensorShape::from_dims({pw1_weight.shape.dims[0], pw1_weight.shape.dims[1]}));
            y_btc = modules::LinearModule({y_btc.shape.last_dim(), pw1_weight.shape.dims[0], true}).build(
                ctx_,
                core::ensure_backend_addressable_layout(ctx_, y_btc),
                {pw1_weight_2d, weight(p + ".pwconv1.bias")});
            y_btc = gelu(y_btc);
            const auto & pw2_weight = weight(p + ".pwconv2.weight");
            const auto pw2_weight_2d = core::reshape_tensor(ctx_, pw2_weight, core::TensorShape::from_dims({pw2_weight.shape.dims[0], pw2_weight.shape.dims[1]}));
            y_btc = modules::LinearModule({y_btc.shape.last_dim(), pw2_weight.shape.dims[0], true}).build(
                ctx_,
                core::ensure_backend_addressable_layout(ctx_, y_btc),
                {pw2_weight_2d, weight(p + ".pwconv2.bias")});
            y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y_btc);
            y = mul(y, weight(p + ".gamma"));
            x = add(residual, y);
        }
        return x;
    }

    core::TensorValue self_encoder(
        core::TensorValue x,
        const core::TensorValue & mask,
        const std::string & prefix,
        int64_t layers,
        int64_t heads) {
        for (int64_t i = 0; i < layers; ++i) {
            x = mul(x, mask);
            x = add(x, conv_self_attention(x, prefix + ".attn_layers." + std::to_string(i), heads));
            auto normed = modules::LayerNormModule({x.shape.dims[1], 1.0e-6F, true, true}).build(
                ctx_,
                modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x),
                {weight(prefix + ".norm_layers_1." + std::to_string(i) + ".norm.weight"), weight(prefix + ".norm_layers_1." + std::to_string(i) + ".norm.bias")});
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, normed);
            x = core::ensure_backend_addressable_layout(ctx_, x);
            auto y = mul(x, mask);
            const auto & ffn1_weight = weight(prefix + ".ffn_layers." + std::to_string(i) + ".conv_1.weight");
            y = modules::Conv1dModule({
                y.shape.dims[1],
                ffn1_weight.shape.dims[0],
                1,
                1,
                0,
                1,
                true,
            }).build(ctx_, y, {ffn1_weight, weight(prefix + ".ffn_layers." + std::to_string(i) + ".conv_1.bias")});
            y = modules::ReluModule().build(ctx_, y);
            y = mul(y, mask);
            const auto & ffn2_weight = weight(prefix + ".ffn_layers." + std::to_string(i) + ".conv_2.weight");
            y = modules::Conv1dModule({
                y.shape.dims[1],
                ffn2_weight.shape.dims[0],
                1,
                1,
                0,
                1,
                true,
            }).build(ctx_, y, {ffn2_weight, weight(prefix + ".ffn_layers." + std::to_string(i) + ".conv_2.bias")});
            y = mul(y, mask);
            x = add(x, y);
            x = modules::LayerNormModule({x.shape.dims[1], 1.0e-6F, true, true}).build(
                ctx_,
                modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x),
                {weight(prefix + ".norm_layers_2." + std::to_string(i) + ".norm.weight"), weight(prefix + ".norm_layers_2." + std::to_string(i) + ".norm.bias")});
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
            x = core::ensure_backend_addressable_layout(ctx_, x);
        }
        return x;
    }

    core::TensorValue time_embedding(const core::TensorValue & current_step, const core::TensorValue & total_step) {
        static constexpr std::array<float, 32> kFrequencies = {
            1.0F,
            0.7429639F,
            0.55199546F,
            0.4101127F,
            0.30469894F,
            0.22638035F,
            0.16819243F,
            0.12496091F,
            0.092841454F,
            0.068977855F,
            0.051248062F,
            0.038075458F,
            0.028288694F,
            0.021017481F,
            0.015615228F,
            0.011601552F,
            0.008619536F,
            0.006404005F,
            0.004757944F,
            0.003534982F,
            0.002626364F,
            0.001951293F,
            0.001449740F,
            0.001077105F,
            0.0008002502F,
            0.0005945571F,
            0.0004417345F,
            0.00032819266F,
            0.00024383534F,
            0.00018116087F,
            0.000134596F,
            0.0001F,
        };
        const auto & frequency = time_frequency(kFrequencies);
        auto ratio = div(current_step, total_step);
        ratio = core::wrap_tensor(
            ggml_scale(ggml_, core::ensure_backend_addressable_layout(ctx_, ratio).tensor, 1000.0F),
            ratio.shape,
            GGML_TYPE_F32);
        auto args = mul(broadcast_to(ctx_, ratio, core::TensorShape::from_dims({1, 32})), frequency);
        auto sin_part = core::wrap_tensor(ggml_sin(ggml_, args.tensor), args.shape, GGML_TYPE_F32);
        auto cos_part = core::wrap_tensor(ggml_cos(ggml_, args.tensor), args.shape, GGML_TYPE_F32);
        auto encoded = modules::ConcatModule({1}).build(ctx_, sin_part, cos_part);
        const std::string prefix = "vector_estimator.vector_estimator.tts.ttl.vector_field.time_encoder.mlp.";
        auto h = modules::LinearModule({
            encoded.shape.last_dim(),
            weight(prefix + "0.linear.weight").shape.dims[0],
            true,
        }).build(ctx_, encoded, {weight(prefix + "0.linear.weight"), weight(prefix + "0.linear.bias")});
        auto softplus = core::wrap_tensor(ggml_softplus(ggml_, h.tensor), h.shape, GGML_TYPE_F32);
        auto mish_gate = core::wrap_tensor(ggml_tanh(ggml_, softplus.tensor), h.shape, GGML_TYPE_F32);
        h = mul(h, mish_gate);
        return modules::LinearModule({
            h.shape.last_dim(),
            weight(prefix + "2.linear.weight").shape.dims[0],
            true,
        }).build(ctx_, h, {weight(prefix + "2.linear.weight"), weight(prefix + "2.linear.bias")});
    }

    const core::TensorValue & time_frequency(const std::array<float, 32> & values) {
        if (time_frequency_.has_value()) {
            return *time_frequency_;
        }
        auto frequency = tensor_writer_.make_f32_tensor(
            tensor_ctx_,
            core::TensorShape::from_dims({1, 32}),
            std::vector<float>(values.begin(), values.end()));
        time_frequency_ = frequency;
        return *time_frequency_;
    }

    core::TensorValue add_time_condition(core::TensorValue x, const core::TensorValue & time, int block_index) {
        const std::string block = "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks." + std::to_string(block_index);
        const auto & linear_weight = weight(block + ".linear.linear.weight");
        auto linear_weight_t = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, linear_weight);
        linear_weight_t = core::ensure_backend_addressable_layout(ctx_, linear_weight_t);
        auto cond = modules::LinearModule({time.shape.last_dim(), linear_weight.shape.dims[1], true}).build(
            ctx_,
            time,
            {linear_weight_t, weight(block + ".linear.linear.bias")});
        cond = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, cond), core::TensorShape::from_dims({1, cond.shape.dims[1], 1}));
        return add(x, cond);
    }

    core::TensorValue text_memory_block(
        core::TensorValue x,
        const core::TensorValue & memory,
        const core::TensorValue & memory_mask,
        const core::TensorValue & output_mask,
        int block_index) {
        const std::string block = "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks." + std::to_string(block_index);
        x = mul(x, output_mask);
        auto y = text_cross_attention(x, memory, block + ".attn", 8, output_mask, false, true, &memory_mask);
        x = add(x, y);
        return normalize_vector_block(x, output_mask, block);
    }

    core::TensorValue style_memory_block(
        core::TensorValue x,
        const core::TensorValue & style_key,
        const core::TensorValue & style_value,
        const core::TensorValue & output_mask,
        int block_index) {
        const std::string block = "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks." + std::to_string(block_index);
        x = mul(x, output_mask);
        auto y = text_cross_attention(x, style_key, block + ".attention", 2, output_mask, true, false, nullptr, &style_value);
        x = add(x, y);
        return normalize_vector_block(x, output_mask, block);
    }

    core::TensorValue normalize_vector_block(
        core::TensorValue x,
        const core::TensorValue & output_mask,
        const std::string & block) {
        auto normed = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
        normed = modules::LayerNormModule({x.shape.dims[1], 1.0e-6F, true, true}).build(
            ctx_,
            normed,
            {weight(block + ".norm.norm.weight"), weight(block + ".norm.norm.bias")});
        normed = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, normed);
        normed = core::ensure_backend_addressable_layout(ctx_, normed);
        return mul(normed, output_mask);
    }

    core::ModuleBuildContext & ctx_;
    core::ModuleBuildContext & tensor_ctx_;
    const SupertonicBackendWeights & weights_;
    core::DeferredTensorWriter & tensor_writer_;
    ggml_context * ggml_ = nullptr;
    std::unordered_map<int64_t, core::TensorValue> rotary_positions_;
    std::unordered_map<int64_t, core::TensorValue> relative_position_masks_;
    std::optional<core::TensorValue> time_frequency_;
};

class SupertonicDurationPredictorRuntime {
public:
    SupertonicDurationPredictorRuntime(
        std::shared_ptr<const SupertonicBackendWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t arena_bytes,
        int threads)
        : weights_(std::move(weights)),
          backend_(backend),
          backend_type_(backend_type),
          arena_bytes_(arena_bytes),
          threads_(std::max(1, threads)) {
        if (weights_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("Supertonic duration predictor requires weights and backend");
        }
    }

    SupertonicTensorOutput run(
        const std::vector<int32_t> & text_ids,
        const std::vector<float> & text_mask,
        int64_t text_length,
        const std::vector<float> & style_dp,
        const std::vector<int64_t> & style_dp_shape) {
        const std::vector<int64_t> text_shape{1, text_length};
        const std::vector<int64_t> mask_shape{1, 1, text_length};
        const auto total_start = std::chrono::steady_clock::now();
        CachedGraph & graph = graph_for(text_shape, mask_shape, style_dp_shape);
        auto timing_start = std::chrono::steady_clock::now();
        upload_i32_tensor(graph.text_ids, graph.text_shape, text_shape, text_ids, "duration_predictor.text_ids");
        upload_f32_tensor(graph.text_mask, graph.text_mask_shape, mask_shape, text_mask, "duration_predictor.text_mask");
        upload_f32_tensor(graph.style_dp, graph.style_shape, style_dp_shape, style_dp, "duration_predictor.style_dp");
        engine::debug::timing_log_scalar("supertonic.model.duration_predictor.input_upload_ms", engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, graph.graph, nullptr, "supertonic.duration_predictor");
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Supertonic duration predictor compute failed");
        }
        engine::debug::timing_log_scalar("supertonic.model.duration_predictor.graph.compute_ms", engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        SupertonicTensorOutput out;
        out.shape = shape_to_vector(graph.output_shape);
        out.values = core::read_tensor_f32(graph.output);
        engine::debug::timing_log_scalar("supertonic.model.duration_predictor.output_read_ms", engine::debug::elapsed_ms(timing_start));
        engine::debug::timing_log_scalar("supertonic.model.duration_predictor.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    struct CachedGraph {
        ~CachedGraph() {
            core::release_backend_graph_resources(backend, graph);
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
            }
            if (io_buffer != nullptr) {
                ggml_backend_buffer_free(io_buffer);
            }
        }

        std::unique_ptr<ggml_context, GgmlContextDeleter> io_ggml;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ggml;
        ggml_backend_buffer_t io_buffer = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        std::vector<int64_t> text_shape;
        std::vector<int64_t> text_mask_shape;
        std::vector<int64_t> style_shape;
        core::TensorShape output_shape = {};
        ggml_tensor * text_ids = nullptr;
        ggml_tensor * text_mask = nullptr;
        ggml_tensor * style_dp = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_t backend = nullptr;
        int threads = 1;
    };

    std::vector<int64_t> cache_key(
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & mask_shape,
        const std::vector<int64_t> & style_shape) const {
        std::vector<int64_t> key;
        append_cache_key_shape(key, text_shape);
        append_cache_key_shape(key, mask_shape);
        append_cache_key_shape(key, style_shape);
        return key;
    }

    CachedGraph & graph_for(
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & mask_shape,
        const std::vector<int64_t> & style_shape) {
        const auto key = cache_key(text_shape, mask_shape, style_shape);
        if (auto * found = graphs_.find(key)) {
            return **found;
        }
        return prepare(key, text_shape, mask_shape, style_shape);
    }

    CachedGraph & prepare(
        const std::vector<int64_t> & key,
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & mask_shape,
        const std::vector<int64_t> & style_shape) {
        const auto build_start = std::chrono::steady_clock::now();
        core::DeferredTensorWriter tensor_writer;
        auto graph = std::make_unique<CachedGraph>();
        graph->text_shape = text_shape;
        graph->text_mask_shape = mask_shape;
        graph->style_shape = style_shape;
        graph->backend = backend_;
        graph->threads = threads_;
        ggml_init_params params{arena_bytes_, nullptr, true};
        ggml_init_params io_params{kSupertonicIoArenaBytes, nullptr, true};
        graph->io_ggml.reset(ggml_init(io_params));
        graph->ggml.reset(ggml_init(params));
        if (graph->io_ggml == nullptr || graph->ggml == nullptr) {
            throw std::runtime_error("failed to initialize Supertonic duration predictor context");
        }
        core::ModuleBuildContext io_ctx{graph->io_ggml.get(), "supertonic.duration_predictor.io", backend_type_};
        core::ModuleBuildContext ctx{graph->ggml.get(), "supertonic.duration_predictor", backend_type_};
        auto ids_tensor = core::make_tensor(io_ctx, GGML_TYPE_I32, shape_from_vector(text_shape));
        auto mask_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(mask_shape));
        auto style_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(style_shape));
        ggml_set_input(ids_tensor.tensor);
        ggml_set_input(mask_tensor.tensor);
        ggml_set_input(style_tensor.tensor);
        SupertonicNetwork network(ctx, io_ctx, *weights_, tensor_writer);
        auto output = network.duration_predictor(ids_tensor, mask_tensor, style_tensor);
        output = core::wrap_tensor(ggml_cont(ctx.ggml, output.tensor), output.shape, output.type);
        graph->text_ids = ids_tensor.tensor;
        graph->text_mask = mask_tensor.tensor;
        graph->style_dp = style_tensor.tensor;
        graph->output = output.tensor;
        graph->output_shape = output.shape;
        ggml_set_output(graph->output);
        graph->graph = ggml_new_graph_custom(ctx.ggml, 65536, false);
        ggml_build_forward_expand(graph->graph, graph->output);
        graph->io_buffer = ggml_backend_alloc_ctx_tensors(graph->io_ggml.get(), backend_);
        if (graph->io_buffer == nullptr) {
            throw std::runtime_error("failed to allocate Supertonic duration predictor IO buffer");
        }
        graph->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (graph->gallocr == nullptr ||
            !ggml_gallocr_reserve(graph->gallocr, graph->graph) ||
            !ggml_gallocr_alloc_graph(graph->gallocr, graph->graph)) {
            throw std::runtime_error("failed to allocate Supertonic duration predictor backend graph buffer");
        }
        tensor_writer.flush();
        graphs_.put(key, std::move(graph));
        engine::debug::timing_log_scalar("supertonic.model.duration_predictor.prepare_ms", engine::debug::elapsed_ms(build_start));
        auto * cached = graphs_.find(key);
        if (cached == nullptr) {
            throw std::runtime_error("Supertonic duration predictor graph cache insert failed");
        }
        return **cached;
    }

    std::shared_ptr<const SupertonicBackendWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    size_t arena_bytes_ = 0;
    int threads_ = 1;
    runtime::CacheSlots<std::vector<int64_t>, std::unique_ptr<CachedGraph>> graphs_{kSupertonicSmallGraphCacheSlots};
};

class SupertonicTextEncoderRuntime {
public:
    SupertonicTextEncoderRuntime(
        std::shared_ptr<const SupertonicBackendWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t arena_bytes,
        int threads)
        : weights_(std::move(weights)),
          backend_(backend),
          backend_type_(backend_type),
          arena_bytes_(arena_bytes),
          threads_(std::max(1, threads)) {
        if (weights_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("Supertonic text encoder requires weights and backend");
        }
    }

    SupertonicTensorOutput run(
        const std::vector<int32_t> & text_ids,
        const std::vector<float> & text_mask,
        int64_t text_length,
        const std::vector<float> & style_ttl,
        const std::vector<int64_t> & style_ttl_shape) {
        const std::vector<int64_t> text_shape{1, text_length};
        const std::vector<int64_t> mask_shape{1, 1, text_length};
        const auto total_start = std::chrono::steady_clock::now();
        CachedGraph & graph = graph_for(text_shape, mask_shape, style_ttl_shape);
        auto timing_start = std::chrono::steady_clock::now();
        upload_i32_tensor(graph.text_ids, graph.text_shape, text_shape, text_ids, "text_encoder.text_ids");
        upload_f32_tensor(graph.text_mask, graph.text_mask_shape, mask_shape, text_mask, "text_encoder.text_mask");
        upload_f32_tensor(graph.style_ttl, graph.style_shape, style_ttl_shape, style_ttl, "text_encoder.style_ttl");
        engine::debug::timing_log_scalar("supertonic.model.text_encoder.input_upload_ms", engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, graph.graph, nullptr, "supertonic.text_encoder");
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Supertonic text encoder compute failed");
        }
        engine::debug::timing_log_scalar("supertonic.model.text_encoder.graph.compute_ms", engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        const auto full_output = core::read_tensor_f32(graph.output);
        SupertonicTensorOutput out;
        out.shape = {graph.output_shape.dims[0], graph.output_shape.dims[1], text_length};
        out.values = slice_padded_values(full_output, shape_to_vector(graph.output_shape), out.shape);
        engine::debug::timing_log_scalar("supertonic.model.text_encoder.output_read_ms", engine::debug::elapsed_ms(timing_start));
        engine::debug::timing_log_scalar("supertonic.model.text_encoder.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    struct CachedGraph {
        ~CachedGraph() {
            core::release_backend_graph_resources(backend, graph);
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
            }
            if (io_buffer != nullptr) {
                ggml_backend_buffer_free(io_buffer);
            }
        }

        std::unique_ptr<ggml_context, GgmlContextDeleter> io_ggml;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ggml;
        ggml_backend_buffer_t io_buffer = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        std::vector<int64_t> text_shape;
        std::vector<int64_t> text_mask_shape;
        std::vector<int64_t> style_shape;
        core::TensorShape output_shape = {};
        ggml_tensor * text_ids = nullptr;
        ggml_tensor * text_mask = nullptr;
        ggml_tensor * style_ttl = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_t backend = nullptr;
        int threads = 1;
    };

    std::vector<int64_t> cache_key(
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & mask_shape,
        const std::vector<int64_t> & style_shape) const {
        std::vector<int64_t> key;
        append_cache_key_shape(key, text_shape);
        append_cache_key_shape(key, mask_shape);
        append_cache_key_shape(key, style_shape);
        return key;
    }

    CachedGraph & graph_for(
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & mask_shape,
        const std::vector<int64_t> & style_shape) {
        const auto key = cache_key(text_shape, mask_shape, style_shape);
        if (auto * found = graphs_.find(key)) {
            return **found;
        }
        return prepare(key, text_shape, mask_shape, style_shape);
    }

    CachedGraph & prepare(
        const std::vector<int64_t> & key,
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & mask_shape,
        const std::vector<int64_t> & style_shape) {
        const auto build_start = std::chrono::steady_clock::now();
        core::DeferredTensorWriter tensor_writer;
        auto graph = std::make_unique<CachedGraph>();
        graph->text_shape = text_shape;
        graph->text_mask_shape = mask_shape;
        graph->style_shape = style_shape;
        graph->backend = backend_;
        graph->threads = threads_;
        ggml_init_params params{arena_bytes_, nullptr, true};
        ggml_init_params io_params{kSupertonicIoArenaBytes, nullptr, true};
        graph->io_ggml.reset(ggml_init(io_params));
        graph->ggml.reset(ggml_init(params));
        if (graph->io_ggml == nullptr || graph->ggml == nullptr) {
            throw std::runtime_error("failed to initialize Supertonic text encoder context");
        }
        core::ModuleBuildContext io_ctx{graph->io_ggml.get(), "supertonic.text_encoder.io", backend_type_};
        core::ModuleBuildContext ctx{graph->ggml.get(), "supertonic.text_encoder", backend_type_};
        auto ids_tensor = core::make_tensor(io_ctx, GGML_TYPE_I32, shape_from_vector(text_shape));
        auto mask_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(mask_shape));
        auto style_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(style_shape));
        ggml_set_input(ids_tensor.tensor);
        ggml_set_input(mask_tensor.tensor);
        ggml_set_input(style_tensor.tensor);
        SupertonicNetwork network(ctx, io_ctx, *weights_, tensor_writer);
        auto output = network.text_encoder(ids_tensor, mask_tensor, style_tensor);
        output = core::wrap_tensor(ggml_cont(ctx.ggml, output.tensor), output.shape, output.type);
        graph->text_ids = ids_tensor.tensor;
        graph->text_mask = mask_tensor.tensor;
        graph->style_ttl = style_tensor.tensor;
        graph->output = output.tensor;
        graph->output_shape = output.shape;
        ggml_set_output(graph->output);
        graph->graph = ggml_new_graph_custom(ctx.ggml, 65536, false);
        ggml_build_forward_expand(graph->graph, graph->output);
        graph->io_buffer = ggml_backend_alloc_ctx_tensors(graph->io_ggml.get(), backend_);
        if (graph->io_buffer == nullptr) {
            throw std::runtime_error("failed to allocate Supertonic text encoder IO buffer");
        }
        graph->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (graph->gallocr == nullptr ||
            !ggml_gallocr_reserve(graph->gallocr, graph->graph) ||
            !ggml_gallocr_alloc_graph(graph->gallocr, graph->graph)) {
            throw std::runtime_error("failed to allocate Supertonic text encoder backend graph buffer");
        }
        tensor_writer.flush();
        graphs_.put(key, std::move(graph));
        engine::debug::timing_log_scalar("supertonic.model.text_encoder.prepare_ms", engine::debug::elapsed_ms(build_start));
        auto * cached = graphs_.find(key);
        if (cached == nullptr) {
            throw std::runtime_error("Supertonic text encoder graph cache insert failed");
        }
        return **cached;
    }

    std::shared_ptr<const SupertonicBackendWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    size_t arena_bytes_ = 0;
    int threads_ = 1;
    runtime::CacheSlots<std::vector<int64_t>, std::unique_ptr<CachedGraph>> graphs_{kSupertonicSmallGraphCacheSlots};
};

class SupertonicVectorEstimatorRuntime {
public:
    SupertonicVectorEstimatorRuntime(
        std::shared_ptr<const SupertonicBackendWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t arena_bytes,
        int threads)
        : weights_(std::move(weights)),
          backend_(backend),
          backend_type_(backend_type),
          arena_bytes_(arena_bytes),
          threads_(std::max(1, threads)) {
        if (weights_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("Supertonic vector estimator requires weights and backend");
        }
    }

    SupertonicTensorOutput run_denoising(
        const std::vector<float> & noisy_latent,
        const std::vector<int64_t> & latent_shape,
        const std::vector<float> & text_embedding,
        const std::vector<int64_t> & text_shape,
        const std::vector<float> & text_mask,
        const std::vector<int64_t> & text_mask_shape,
        const std::vector<float> & style_ttl,
        const std::vector<int64_t> & style_shape,
        const std::vector<float> & latent_mask,
        int total_steps) {
        if (total_steps <= 0) {
            throw std::runtime_error("Supertonic vector estimator requires positive denoise steps");
        }
        const std::vector<int64_t> mask_shape{latent_shape.at(0), 1, latent_shape.at(2)};
        const auto total_start = std::chrono::steady_clock::now();
        CachedGraph & graph = graph_for(
            latent_shape,
            text_shape,
            text_mask_shape,
            style_shape,
            mask_shape,
            kSupertonicVectorStepsPerGraph);
        auto timing_start = std::chrono::steady_clock::now();
        upload_f32_tensor(graph.noisy_latent, graph.latent_shape, latent_shape, noisy_latent, "vector_estimator.noisy_latent");
        upload_f32_tensor(graph.text_embedding, graph.text_shape, text_shape, text_embedding, "vector_estimator.text_embedding");
        upload_f32_tensor(graph.text_mask, graph.text_mask_shape, text_mask_shape, text_mask, "vector_estimator.text_mask");
        upload_f32_tensor(graph.style_ttl, graph.style_shape, style_shape, style_ttl, "vector_estimator.style_ttl");
        upload_f32_tensor(graph.latent_mask, graph.mask_shape, mask_shape, latent_mask, "vector_estimator.latent_mask");
        const float total_step_value = static_cast<float>(total_steps);
        core::write_tensor_f32(
            core::wrap_tensor(graph.total_step, core::TensorShape::from_dims({1}), GGML_TYPE_F32),
            &total_step_value,
            1);
        engine::debug::timing_log_scalar("supertonic.model.vector_estimator.input_upload_ms", engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        core::set_backend_threads(backend_, threads_);
        for (int step = 0; step < total_steps; ++step) {
            const float start_step_value = static_cast<float>(step);
            core::write_tensor_f32(
                core::wrap_tensor(graph.start_step, core::TensorShape::from_dims({1}), GGML_TYPE_F32),
                &start_step_value,
                1);
            const ggml_status status = core::compute_backend_graph(backend_, graph.graph, nullptr, "supertonic.vector_estimator");
            ggml_backend_synchronize(backend_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Supertonic vector estimator compute failed");
            }
            if (step + 1 < total_steps) {
                ggml_backend_tensor_copy(graph.output, graph.noisy_latent);
            }
        }
        engine::debug::timing_log_scalar("supertonic.model.vector_estimator.graph.compute_ms", engine::debug::elapsed_ms(timing_start));

        timing_start = std::chrono::steady_clock::now();
        const auto full_output = core::read_tensor_f32(graph.output);
        SupertonicTensorOutput out;
        out.shape = latent_shape;
        out.values = slice_padded_values(full_output, shape_to_vector(graph.output_shape), out.shape);
        engine::debug::timing_log_scalar("supertonic.model.vector_estimator.output_read_ms", engine::debug::elapsed_ms(timing_start));
        engine::debug::timing_log_scalar("supertonic.model.vector_estimator.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    struct CachedGraph {
        ~CachedGraph() {
            core::release_backend_graph_resources(backend, graph);
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
            }
            if (io_buffer != nullptr) {
                ggml_backend_buffer_free(io_buffer);
            }
        }

        std::unique_ptr<ggml_context, GgmlContextDeleter> io_ggml;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ggml;
        ggml_backend_buffer_t io_buffer = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        std::vector<int64_t> latent_shape;
        std::vector<int64_t> text_shape;
        std::vector<int64_t> text_mask_shape;
        std::vector<int64_t> style_shape;
        std::vector<int64_t> mask_shape;
        int step_count = 0;
        core::TensorShape output_shape = {};
        ggml_tensor * noisy_latent = nullptr;
        ggml_tensor * text_embedding = nullptr;
        ggml_tensor * text_mask = nullptr;
        ggml_tensor * style_ttl = nullptr;
        ggml_tensor * latent_mask = nullptr;
        ggml_tensor * start_step = nullptr;
        ggml_tensor * total_step = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_t backend = nullptr;
        int threads = 1;
    };

    std::vector<int64_t> cache_key(
        const std::vector<int64_t> & latent_shape,
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & text_mask_shape,
        const std::vector<int64_t> & style_shape,
        const std::vector<int64_t> & mask_shape,
        int step_count) const {
        std::vector<int64_t> key;
        append_cache_key_shape(key, latent_shape);
        append_cache_key_shape(key, text_shape);
        append_cache_key_shape(key, text_mask_shape);
        append_cache_key_shape(key, style_shape);
        append_cache_key_shape(key, mask_shape);
        key.push_back(static_cast<int64_t>(step_count));
        return key;
    }

    CachedGraph & graph_for(
        const std::vector<int64_t> & latent_shape,
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & text_mask_shape,
        const std::vector<int64_t> & style_shape,
        const std::vector<int64_t> & mask_shape,
        int step_count) {
        const auto key = cache_key(latent_shape, text_shape, text_mask_shape, style_shape, mask_shape, step_count);
        if (auto * found = graphs_.find(key)) {
            return **found;
        }
        return prepare(
            key,
            latent_shape,
            text_shape,
            text_mask_shape,
            style_shape,
            mask_shape,
            step_count);
    }

    CachedGraph & prepare(
        const std::vector<int64_t> & key,
        const std::vector<int64_t> & latent_shape,
        const std::vector<int64_t> & text_shape,
        const std::vector<int64_t> & text_mask_shape,
        const std::vector<int64_t> & style_shape,
        const std::vector<int64_t> & mask_shape,
        int step_count) {
        const auto build_start = std::chrono::steady_clock::now();
        core::DeferredTensorWriter tensor_writer;
        auto graph = std::make_unique<CachedGraph>();
        graph->latent_shape = latent_shape;
        graph->text_shape = text_shape;
        graph->text_mask_shape = text_mask_shape;
        graph->style_shape = style_shape;
        graph->mask_shape = mask_shape;
        graph->step_count = step_count;
        graph->backend = backend_;
        graph->threads = threads_;
        ggml_init_params params{arena_bytes_, nullptr, true};
        ggml_init_params io_params{kSupertonicIoArenaBytes, nullptr, true};
        graph->io_ggml.reset(ggml_init(io_params));
        graph->ggml.reset(ggml_init(params));
        if (graph->io_ggml == nullptr || graph->ggml == nullptr) {
            throw std::runtime_error("failed to initialize Supertonic vector estimator context");
        }
        core::ModuleBuildContext io_ctx{graph->io_ggml.get(), "supertonic.vector_estimator.io", backend_type_};
        core::ModuleBuildContext ctx{graph->ggml.get(), "supertonic.vector_estimator", backend_type_};
        auto latent_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(latent_shape));
        auto text_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(text_shape));
        auto text_mask_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(text_mask_shape));
        auto style_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(style_shape));
        auto mask_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, shape_from_vector(mask_shape));
        auto start_step_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        auto total_step_tensor = core::make_tensor(io_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        ggml_set_input(latent_tensor.tensor);
        ggml_set_input(text_tensor.tensor);
        ggml_set_input(text_mask_tensor.tensor);
        ggml_set_input(style_tensor.tensor);
        ggml_set_input(mask_tensor.tensor);
        ggml_set_input(start_step_tensor.tensor);
        ggml_set_input(total_step_tensor.tensor);
        SupertonicNetwork network(ctx, io_ctx, *weights_, tensor_writer);
        auto output = network.vector_estimator_steps(
            latent_tensor,
            text_tensor,
            text_mask_tensor,
            style_tensor,
            mask_tensor,
            start_step_tensor,
            total_step_tensor,
            step_count);
        output = core::wrap_tensor(ggml_cont(ctx.ggml, output.tensor), output.shape, output.type);
        graph->noisy_latent = latent_tensor.tensor;
        graph->text_embedding = text_tensor.tensor;
        graph->text_mask = text_mask_tensor.tensor;
        graph->style_ttl = style_tensor.tensor;
        graph->latent_mask = mask_tensor.tensor;
        graph->start_step = start_step_tensor.tensor;
        graph->total_step = total_step_tensor.tensor;
        graph->output = output.tensor;
        graph->output_shape = output.shape;
        ggml_set_output(graph->output);
        graph->graph = ggml_new_graph_custom(ctx.ggml, 262144, false);
        ggml_build_forward_expand(graph->graph, graph->output);
        graph->io_buffer = ggml_backend_alloc_ctx_tensors(graph->io_ggml.get(), backend_);
        if (graph->io_buffer == nullptr) {
            throw std::runtime_error("failed to allocate Supertonic vector estimator IO buffer");
        }
        graph->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (graph->gallocr == nullptr ||
            !ggml_gallocr_reserve(graph->gallocr, graph->graph) ||
            !ggml_gallocr_alloc_graph(graph->gallocr, graph->graph)) {
            throw std::runtime_error("failed to allocate Supertonic vector estimator backend graph buffer");
        }
        tensor_writer.flush();
        graphs_.put(key, std::move(graph));
        engine::debug::timing_log_scalar("supertonic.model.vector_estimator.prepare_ms", engine::debug::elapsed_ms(build_start));
        auto * cached = graphs_.find(key);
        if (cached == nullptr) {
            throw std::runtime_error("Supertonic vector estimator graph cache insert failed");
        }
        return **cached;
    }

    std::shared_ptr<const SupertonicBackendWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    size_t arena_bytes_ = 0;
    int threads_ = 1;
    runtime::CacheSlots<std::vector<int64_t>, std::unique_ptr<CachedGraph>> graphs_{kSupertonicLargeGraphCacheSlots};
};


SupertonicLatentTensor make_latent_tensor(const std::vector<int64_t> & shape, std::vector<float> data) {
    SupertonicLatentTensor input;
    input.shape = shape;
    input.f32 = std::move(data);
    return input;
}

std::vector<float> load_style(const std::filesystem::path & path, const std::string & key, std::vector<int64_t> & shape_out) {
    const auto root = json::parse_file(path);
    const auto & value = root.require(key);
    shape_out = json::number_array_as<int64_t>(value.require("dims"));
    std::vector<float> out;
    const auto flatten = [&](const auto & self, const json::Value & item) -> void {
        if (item.is_number()) {
            out.push_back(item.as_f32());
            return;
        }
        for (const auto & child : item.as_array()) {
            self(self, child);
        }
    };
    flatten(flatten, value.require("data"));
    return out;
}

std::vector<float> length_to_mask(int64_t length) {
    return std::vector<float>(static_cast<size_t>(length), 1.0F);
}

class NumpyMt19937Normal {
public:
    explicit NumpyMt19937Normal(uint32_t seed)
        : rng_(seed) {}

    float next() {
        if (has_cached_) {
            has_cached_ = false;
            return static_cast<float>(cached_);
        }
        double x = 0.0;
        double y = 0.0;
        double radius = 0.0;
        do {
            x = 2.0 * uniform53() - 1.0;
            y = 2.0 * uniform53() - 1.0;
            radius = x * x + y * y;
        } while (radius >= 1.0 || radius == 0.0);
        const double scale = std::sqrt(-2.0 * std::log(radius) / radius);
        cached_ = x * scale;
        has_cached_ = true;
        return static_cast<float>(y * scale);
    }

private:
    double uniform53() {
        const uint32_t a = rng_() >> 5U;
        const uint32_t b = rng_() >> 6U;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) / 9007199254740992.0;
    }

    std::mt19937 rng_;
    bool has_cached_ = false;
    double cached_ = 0.0;
};

std::vector<float> random_normal(size_t count, NumpyMt19937Normal & rng) {
    std::vector<float> values(count);
    for (float & value : values) {
        value = rng.next();
    }
    return values;
}

SupertonicChunkOutput synthesize_supertonic_chunk(
    const std::shared_ptr<const SupertonicAssets> & assets,
    SupertonicDurationPredictorRuntime & duration_model,
    SupertonicTextEncoderRuntime & text_model,
    SupertonicVectorEstimatorRuntime & vector_model,
    SupertonicVocoderRuntime & vocoder_model,
    const SupertonicTextTokenizer & tokenizer,
    const std::string & text,
    const SupertonicGenerationOptions & options,
    const std::vector<float> & style_ttl,
    const std::vector<int64_t> & style_ttl_shape,
    const std::vector<float> & style_dp,
    const std::vector<int64_t> & style_dp_shape,
    NumpyMt19937Normal & rng) {
    const auto chunk_start = std::chrono::steady_clock::now();
    auto timing_start = std::chrono::steady_clock::now();
    const auto text_inputs = tokenizer.encode(text, options.language);
    engine::debug::trace_log_scalar("supertonic.chunk.text_chars", static_cast<int64_t>(text.size()));
    engine::debug::trace_log_scalar("supertonic.chunk.text_tokens", text_inputs.length);
    engine::debug::timing_log_scalar("supertonic.chunk.encode_ms", engine::debug::elapsed_ms(timing_start));

    std::vector<int32_t> ids;
    ids.reserve(text_inputs.ids.size());
    for (const int64_t id : text_inputs.ids) {
        ids.push_back(static_cast<int32_t>(id));
    }

    timing_start = std::chrono::steady_clock::now();
    auto duration = duration_model.run(ids, text_inputs.mask, text_inputs.length, style_dp, style_dp_shape);
    engine::debug::timing_log_scalar("supertonic.chunk.duration_predictor_ms", engine::debug::elapsed_ms(timing_start));
    if (duration.values.empty() || !(duration.values[0] > 0.0F)) {
        throw std::runtime_error("Supertonic duration predictor returned invalid duration");
    }
    engine::debug::trace_log_scalar("supertonic.chunk.duration_raw_seconds", duration.values[0]);
    duration.values[0] /= options.speaking_rate;
    engine::debug::trace_log_scalar("supertonic.chunk.duration_seconds", duration.values[0]);

    timing_start = std::chrono::steady_clock::now();
    auto text_emb = text_model.run(ids, text_inputs.mask, text_inputs.length, style_ttl, style_ttl_shape);
    engine::debug::timing_log_scalar("supertonic.chunk.text_encoder_ms", engine::debug::elapsed_ms(timing_start));

    const auto & config = assets->config;
    const int64_t chunk_size = config.base_chunk_size * config.chunk_compress_factor;
    const int64_t wav_length = static_cast<int64_t>(duration.values[0] * static_cast<float>(config.sample_rate));
    const int64_t latent_length = (wav_length + chunk_size - 1) / chunk_size;
    const int64_t latent_channels = config.latent_dim * config.chunk_compress_factor;
    auto latent = random_normal(static_cast<size_t>(latent_channels * latent_length), rng);
    auto latent_mask = length_to_mask(latent_length);

    timing_start = std::chrono::steady_clock::now();
    auto denoised = vector_model.run_denoising(
        latent,
        {1, latent_channels, latent_length},
        text_emb.values,
        text_emb.shape,
        text_inputs.mask,
        {1, 1, text_inputs.length},
        style_ttl,
        style_ttl_shape,
        latent_mask,
        options.num_inference_steps);
    latent = std::move(denoised.values);
    engine::debug::timing_log_scalar("supertonic.chunk.vector_total_ms", engine::debug::elapsed_ms(timing_start));

    timing_start = std::chrono::steady_clock::now();
    auto wav = vocoder_model.run(make_latent_tensor({1, latent_channels, latent_length}, latent));
    engine::debug::timing_log_scalar("supertonic.chunk.vocoder_ms", engine::debug::elapsed_ms(timing_start));
    SupertonicChunkOutput out;
    out.audio.sample_rate = config.sample_rate;
    out.audio.channels = 1;
    out.audio.samples = std::move(wav.values);
    out.duration_seconds = duration.values[0];
    engine::debug::trace_log_scalar("supertonic.chunk.latent_length", latent_length);
    engine::debug::trace_log_scalar("supertonic.chunk.wav_length", wav_length);
    engine::debug::trace_log_scalar("supertonic.chunk.output_samples", static_cast<int64_t>(out.audio.samples.size()));
    engine::debug::timing_log_scalar("supertonic.chunk.total_ms", engine::debug::elapsed_ms(chunk_start));
    return out;
}

}  // namespace

struct SupertonicRuntime::State {
    struct BackendOwner {
        ggml_backend_t backend = nullptr;

        ~BackendOwner() {
            if (backend != nullptr) {
                ggml_backend_free(backend);
            }
        }

        BackendOwner() = default;
        BackendOwner(const BackendOwner &) = delete;
        BackendOwner & operator=(const BackendOwner &) = delete;
    };

    struct Style {
        std::vector<float> ttl;
        std::vector<int64_t> ttl_shape;
        std::vector<float> dp;
        std::vector<int64_t> dp_shape;
    };

    State(
        std::shared_ptr<const SupertonicAssets> assets_in,
        core::BackendConfig backend_config,
        assets::TensorStorageType weight_storage_type,
        std::size_t style_cache_slots)
        : assets(std::move(assets_in)),
          threads(std::max(1, backend_config.threads)),
          styles(style_cache_slots) {
        if (assets == nullptr) {
            throw std::runtime_error("Supertonic native runtime requires assets");
        }
        backend_config.threads = threads;
        backend.backend = core::init_backend(backend_config);
        backend_type = core::backend_type(backend.backend);
        weights = load_backend_weights(assets, backend.backend, backend_type, weight_storage_type);
        duration_model = std::make_unique<SupertonicDurationPredictorRuntime>(
            weights,
            backend.backend,
            backend_type,
            kSupertonicDurationArenaBytes,
            threads);
        text_model = std::make_unique<SupertonicTextEncoderRuntime>(
            weights,
            backend.backend,
            backend_type,
            kSupertonicTextArenaBytes,
            threads);
        vector_model = std::make_unique<SupertonicVectorEstimatorRuntime>(
            weights,
            backend.backend,
            backend_type,
            kSupertonicVectorArenaBytes,
            threads);
        vocoder_model = std::make_unique<SupertonicVocoderRuntime>(
            weights,
            backend.backend,
            backend_type,
            assets->config,
            kSupertonicVocoderArenaBytes,
            threads);
    }

    const Style & style(const std::string & voice) {
        if (const auto * cached = styles.find(voice)) {
            engine::debug::trace_log_scalar("supertonic.style_cache.hit", 1);
            engine::debug::trace_log_scalar("supertonic.style_cache.slots", static_cast<int64_t>(styles.capacity()));
            engine::debug::trace_log_scalar("supertonic.style_cache.entries", static_cast<int64_t>(styles.size()));
            engine::debug::trace_log_scalar("supertonic.style_cache.evicted", 0);
            return *cached;
        }
        const bool will_evict = styles.capacity() > 0 && styles.size() >= styles.capacity();
        const auto style_path = assets->resources.require_file("voice_style_" + voice);
        Style loaded;
        loaded.ttl = load_style(style_path, "style_ttl", loaded.ttl_shape);
        loaded.dp = load_style(style_path, "style_dp", loaded.dp_shape);
        if (styles.capacity() == 0) {
            uncached_style = std::move(loaded);
            engine::debug::trace_log_scalar("supertonic.style_cache.hit", 0);
            engine::debug::trace_log_scalar("supertonic.style_cache.slots", 0);
            engine::debug::trace_log_scalar("supertonic.style_cache.entries", 0);
            engine::debug::trace_log_scalar("supertonic.style_cache.evicted", 0);
            return uncached_style;
        }
        styles.put(voice, std::move(loaded));
        engine::debug::trace_log_scalar("supertonic.style_cache.hit", 0);
        engine::debug::trace_log_scalar("supertonic.style_cache.slots", static_cast<int64_t>(styles.capacity()));
        engine::debug::trace_log_scalar("supertonic.style_cache.entries", static_cast<int64_t>(styles.size()));
        engine::debug::trace_log_scalar("supertonic.style_cache.evicted", will_evict ? 1 : 0);
        const auto * cached = styles.find(voice);
        if (cached == nullptr) {
            throw std::runtime_error("Supertonic style cache insert failed");
        }
        return *cached;
    }

    std::shared_ptr<const SupertonicAssets> assets;
    int threads = 1;
    core::BackendType backend_type = core::BackendType::Cpu;
    BackendOwner backend;
    std::shared_ptr<const SupertonicBackendWeights> weights;
    std::unique_ptr<SupertonicDurationPredictorRuntime> duration_model;
    std::unique_ptr<SupertonicTextEncoderRuntime> text_model;
    std::unique_ptr<SupertonicVectorEstimatorRuntime> vector_model;
    std::unique_ptr<SupertonicVocoderRuntime> vocoder_model;
    runtime::CacheSlots<std::string, Style> styles;
    Style uncached_style;
};

SupertonicRuntime::SupertonicRuntime(
    std::shared_ptr<const SupertonicAssets> assets,
    core::BackendConfig backend_config,
    assets::TensorStorageType weight_storage_type,
    std::size_t style_cache_slots)
    : state_(std::make_unique<State>(std::move(assets), backend_config, weight_storage_type, style_cache_slots)) {}

SupertonicRuntime::~SupertonicRuntime() = default;

runtime::AudioBuffer SupertonicRuntime::synthesize(
    const std::string & text,
    const SupertonicGenerationOptions & options,
    const SupertonicTextTokenizer & tokenizer) {
    const auto total_start = std::chrono::steady_clock::now();
    auto timing_start = std::chrono::steady_clock::now();
    const auto & style = state_->style(options.voice);
    engine::debug::timing_log_scalar("supertonic.style_ms", engine::debug::elapsed_ms(timing_start));

    engine::debug::trace_log_scalar("supertonic.text_chars", static_cast<int64_t>(text.size()));
    engine::debug::trace_log_scalar("supertonic.threads", state_->threads);
    NumpyMt19937Normal rng(options.seed);
    auto chunk_audio = synthesize_supertonic_chunk(
        state_->assets,
        *state_->duration_model,
        *state_->text_model,
        *state_->vector_model,
        *state_->vocoder_model,
        tokenizer,
        text,
        options,
        style.ttl,
        style.ttl_shape,
        style.dp,
        style.dp_shape,
        rng);
    runtime::AudioBuffer out = std::move(chunk_audio.audio);
    const size_t trim = std::min(
        out.samples.size(),
        static_cast<size_t>(chunk_audio.duration_seconds * static_cast<float>(out.sample_rate)));
    out.samples.resize(trim);
    engine::debug::trace_log_scalar("supertonic.output_samples", static_cast<int64_t>(out.samples.size()));
    const double total_ms = engine::debug::elapsed_ms(total_start);
    engine::debug::timing_log_scalar("session.wall_ms", total_ms);
    return out;
}

}  // namespace engine::models::supertonic
