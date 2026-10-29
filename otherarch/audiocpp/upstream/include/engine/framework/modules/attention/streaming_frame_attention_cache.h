#pragma once

#include "engine/framework/core/module.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::modules {

struct StreamingFrameAttentionCacheConfig {
    int64_t cache_frames = 0;
};

struct StreamingFrameAttentionCacheOutputs {
    core::TensorValue key_context;
    core::TensorValue value_context;
    core::TensorValue next_key_cache;
    core::TensorValue next_value_cache;
};

class StreamingFrameAttentionCacheModule {
public:
    explicit StreamingFrameAttentionCacheModule(StreamingFrameAttentionCacheConfig config);

    StreamingFrameAttentionCacheOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & key_cache,
        const core::TensorValue & value_cache,
        const core::TensorValue & current_key,
        const core::TensorValue & current_value) const;

private:
    StreamingFrameAttentionCacheConfig config_;
};

class StreamingFrameAttentionCache {
public:
    void add_layer(
        ggml_tensor * key_cache,
        ggml_tensor * value_cache,
        ggml_tensor * next_key_cache,
        ggml_tensor * next_value_cache);

    void zero_inputs(ggml_backend_t backend);
    void import_from(ggml_backend_t backend, const StreamingFrameAttentionCache & source);
    void commit_outputs(ggml_backend_t backend);
    void set_outputs() const;
    void build_forward_expand(ggml_cgraph * graph) const;

    size_t layer_count() const noexcept;

private:
    struct Layer {
        ggml_tensor * key_cache = nullptr;
        ggml_tensor * value_cache = nullptr;
        ggml_tensor * next_key_cache = nullptr;
        ggml_tensor * next_value_cache = nullptr;
    };

    void copy_from_outputs(ggml_backend_t backend, const StreamingFrameAttentionCache & source);

    std::vector<Layer> layers_;
    std::vector<unsigned char> zero_scratch_;
};

}  // namespace engine::modules
