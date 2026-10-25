#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::codecs {

struct MelLatentVae44kConfig {
    int64_t latent_channels = 40;
    int64_t mel_bins = 128;
    int64_t hidden_dim = 512;
    int64_t levels = 3;
    int64_t resblocks_per_level = 3;
    int64_t attention_heads = 1;
    int64_t mp_conv_kernel = 3;
    int64_t upsample_level = 1;
    float mp_norm_eps = 1.0e-4F;
    float mp_silu_scale = 1.0F / 0.596F;
    float mp_residual_t = 0.3F;
    float clip_act = 256.0F;
    bool folded_mpconv_weights = false;
    std::string tensor_prefix;
};

struct MelLatentVae44kRuntimeOptions {
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
};

struct MelLatentVae44kMel {
    std::vector<float> values;
    int64_t bins = 0;
    int64_t frames = 0;
};

class MelLatentVae44kRuntime {
public:
    MelLatentVae44kRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        MelLatentVae44kConfig config = {},
        MelLatentVae44kRuntimeOptions options = {});
    ~MelLatentVae44kRuntime();

    MelLatentVae44kRuntime(const MelLatentVae44kRuntime &) = delete;
    MelLatentVae44kRuntime & operator=(const MelLatentVae44kRuntime &) = delete;
    MelLatentVae44kRuntime(MelLatentVae44kRuntime &&) noexcept;
    MelLatentVae44kRuntime & operator=(MelLatentVae44kRuntime &&) noexcept;

    MelLatentVae44kMel decode(const std::vector<float> & latent, int64_t frames);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::codecs
