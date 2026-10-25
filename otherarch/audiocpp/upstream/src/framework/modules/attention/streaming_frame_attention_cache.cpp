#include "engine/framework/modules/attention/streaming_frame_attention_cache.h"

#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>

namespace engine::modules {
StreamingFrameAttentionCacheModule::StreamingFrameAttentionCacheModule(StreamingFrameAttentionCacheConfig config)
    : config_(config) {
    if (config_.cache_frames <= 0) {
        throw std::runtime_error("StreamingFrameAttentionCacheModule requires positive cache_frames");
    }
}

StreamingFrameAttentionCacheOutputs StreamingFrameAttentionCacheModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & key_cache,
    const core::TensorValue & value_cache,
    const core::TensorValue & current_key,
    const core::TensorValue & current_value) const {
    if (key_cache.shape.rank != 4 || value_cache.shape.rank != 4 ||
        current_key.shape.rank != 4 || current_value.shape.rank != 4) {
        throw std::runtime_error(
            "StreamingFrameAttentionCacheModule expects rank-4 [batch, heads, frames, dim] tensors");
    }
    if (key_cache.shape.dims[2] <= 0 || value_cache.shape.dims[2] <= 0 ||
        current_key.shape.dims[2] <= 0 || current_value.shape.dims[2] <= 0) {
        throw std::runtime_error("StreamingFrameAttentionCacheModule requires positive frame counts");
    }
    if (key_cache.type != GGML_TYPE_F32 || value_cache.type != GGML_TYPE_F32 ||
        current_key.type != GGML_TYPE_F32 || current_value.type != GGML_TYPE_F32) {
        throw std::runtime_error("StreamingFrameAttentionCacheModule expects f32 tensors");
    }
    if (key_cache.shape.dims[2] != config_.cache_frames || value_cache.shape.dims[2] != config_.cache_frames) {
        throw std::runtime_error("StreamingFrameAttentionCacheModule cache frame count does not match config");
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (axis == 2) {
            continue;
        }
        if (key_cache.shape.dims[axis] != current_key.shape.dims[axis] ||
            value_cache.shape.dims[axis] != current_value.shape.dims[axis] ||
            key_cache.shape.dims[axis] != value_cache.shape.dims[axis]) {
            throw std::runtime_error("StreamingFrameAttentionCacheModule cache/current shapes are incompatible");
        }
    }

    auto key_context = ConcatModule({2}).build(ctx, key_cache, current_key);
    auto value_context = ConcatModule({2}).build(ctx, value_cache, current_value);
    auto next_key_context = ConcatModule({2}).build(ctx, key_cache, current_key);
    auto next_value_context = ConcatModule({2}).build(ctx, value_cache, current_value);
    auto next_key = SliceModule({2, next_key_context.shape.dims[2] - config_.cache_frames, config_.cache_frames})
                        .build(ctx, next_key_context);
    auto next_value = SliceModule({2, next_value_context.shape.dims[2] - config_.cache_frames, config_.cache_frames})
                          .build(ctx, next_value_context);
    return {
        key_context,
        value_context,
        core::ensure_backend_addressable_layout(ctx, next_key),
        core::ensure_backend_addressable_layout(ctx, next_value),
    };
}

void StreamingFrameAttentionCache::add_layer(
    ggml_tensor * key_cache,
    ggml_tensor * value_cache,
    ggml_tensor * next_key_cache,
    ggml_tensor * next_value_cache) {
    if (key_cache == nullptr || value_cache == nullptr || next_key_cache == nullptr || next_value_cache == nullptr) {
        throw std::runtime_error("StreamingFrameAttentionCache missing layer tensor");
    }
    if (ggml_nbytes(key_cache) != ggml_nbytes(value_cache) ||
        ggml_nbytes(key_cache) != ggml_nbytes(next_key_cache) ||
        ggml_nbytes(value_cache) != ggml_nbytes(next_value_cache)) {
        throw std::runtime_error("StreamingFrameAttentionCache layer tensor sizes do not match");
    }
    layers_.push_back({key_cache, value_cache, next_key_cache, next_value_cache});
}

void StreamingFrameAttentionCache::zero_inputs(ggml_backend_t backend) {
    for (const auto & layer : layers_) {
        const size_t key_bytes = ggml_nbytes(layer.key_cache);
        zero_scratch_.assign(key_bytes, 0);
        ggml_backend_tensor_set(layer.key_cache, zero_scratch_.data(), 0, key_bytes);
        const size_t value_bytes = ggml_nbytes(layer.value_cache);
        zero_scratch_.assign(value_bytes, 0);
        ggml_backend_tensor_set(layer.value_cache, zero_scratch_.data(), 0, value_bytes);
    }
    ggml_backend_synchronize(backend);
}

void StreamingFrameAttentionCache::copy_from_outputs(
    ggml_backend_t backend,
    const StreamingFrameAttentionCache & source) {
    if (source.layers_.size() != layers_.size()) {
        throw std::runtime_error("StreamingFrameAttentionCache layer count mismatch");
    }
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        if (ggml_nbytes(source.layers_[layer].next_key_cache) != ggml_nbytes(layers_[layer].key_cache) ||
            ggml_nbytes(source.layers_[layer].next_value_cache) != ggml_nbytes(layers_[layer].value_cache)) {
            throw std::runtime_error("StreamingFrameAttentionCache import tensor sizes do not match");
        }
        ggml_backend_tensor_copy_async(backend, backend, source.layers_[layer].next_key_cache, layers_[layer].key_cache);
        ggml_backend_tensor_copy_async(backend, backend, source.layers_[layer].next_value_cache, layers_[layer].value_cache);
    }
}

void StreamingFrameAttentionCache::import_from(
    ggml_backend_t backend,
    const StreamingFrameAttentionCache & source) {
    copy_from_outputs(backend, source);
    ggml_backend_synchronize(backend);
}

void StreamingFrameAttentionCache::commit_outputs(ggml_backend_t backend) {
    copy_from_outputs(backend, *this);
}

void StreamingFrameAttentionCache::set_outputs() const {
    for (const auto & layer : layers_) {
        ggml_set_output(layer.next_key_cache);
        ggml_set_output(layer.next_value_cache);
    }
}

void StreamingFrameAttentionCache::build_forward_expand(ggml_cgraph * graph) const {
    if (graph == nullptr) {
        throw std::runtime_error("StreamingFrameAttentionCache requires graph before build_forward_expand");
    }
    for (const auto & layer : layers_) {
        ggml_build_forward_expand(graph, layer.next_key_cache);
        ggml_build_forward_expand(graph, layer.next_value_cache);
    }
}

size_t StreamingFrameAttentionCache::layer_count() const noexcept {
    return layers_.size();
}

}  // namespace engine::modules
