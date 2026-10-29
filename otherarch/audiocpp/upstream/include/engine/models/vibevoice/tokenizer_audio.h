#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/vibevoice/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models::vibevoice {

class VibeVoiceTokenizerEncoderGraph;
class VibeVoiceTokenizerDecoderGraph;
class VibeVoiceTokenizerStreamingGraph;

struct VibeVoiceTokenizerLatents {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dim = 0;
};

struct VibeVoiceAcousticLatentSample {
    std::vector<VibeVoiceTokenizerLatents> latents;
    std::vector<float> std_values;
    uint64_t next_rng_index = 0;
};

struct VibeVoiceTokenizerStreamingState {
    std::vector<std::vector<float>> caches;
    void reset();
    void set_to_zero();
};

struct VibeVoiceTokenizerBlockWeights {
    assets::TensorDataF32 norm;
    modules::DepthwiseConv1dWeights mixer;
    assets::TensorDataF32 gamma;
    assets::TensorDataF32 ffn_norm;
    modules::LinearWeights ffn_linear1;
    modules::LinearWeights ffn_linear2;
    assets::TensorDataF32 ffn_gamma;
    int64_t channels = 0;
};

struct VibeVoiceTokenizerEncoderWeights {
    std::vector<modules::Conv1dWeights> downsample_layers;
    std::vector<std::vector<VibeVoiceTokenizerBlockWeights>> stages;
    std::optional<assets::TensorDataF32> norm;
    modules::Conv1dWeights head;
};

struct VibeVoiceTokenizerDecoderWeights {
    modules::Conv1dWeights stem;
    std::vector<modules::ConvTranspose1dWeights> upsample_layers;
    std::vector<std::vector<VibeVoiceTokenizerBlockWeights>> stages;
    std::optional<assets::TensorDataF32> norm;
    modules::Conv1dWeights head;
};

struct VibeVoiceAudioTokenizerWeights {
    VibeVoiceTokenizerEncoderWeights encoder;
    VibeVoiceTokenizerDecoderWeights decoder;
};

struct VibeVoiceSemanticTokenizerWeights {
    VibeVoiceTokenizerEncoderWeights encoder;
};

struct VibeVoiceTokenizerWeightsBundle {
    std::shared_ptr<core::BackendWeightStore> store;
    VibeVoiceAudioTokenizerWeights acoustic;
    VibeVoiceSemanticTokenizerWeights semantic;
};

class VibeVoiceTokenizerWeightsRuntime final {
public:
    VibeVoiceTokenizerWeightsRuntime(
        std::shared_ptr<const VibeVoiceAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t weight_context_bytes = 256ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 128ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);

    ~VibeVoiceTokenizerWeightsRuntime();

    VibeVoiceTokenizerWeightsRuntime(const VibeVoiceTokenizerWeightsRuntime &) = delete;
    VibeVoiceTokenizerWeightsRuntime & operator=(const VibeVoiceTokenizerWeightsRuntime &) = delete;

    const VibeVoiceAssets & assets() const noexcept;
    const VibeVoiceTokenizerWeightsBundle & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::ConstantTensorCache & constants() const noexcept;
    int threads() const noexcept;

    VibeVoiceTokenizerLatents encode_acoustic(const runtime::AudioBuffer & audio) const;
    std::vector<VibeVoiceTokenizerLatents> encode_acoustic_batch(
        const std::vector<runtime::AudioBuffer> & audio) const;
    VibeVoiceTokenizerLatents encode_semantic(const runtime::AudioBuffer & audio) const;
    runtime::AudioBuffer decode_acoustic(const VibeVoiceTokenizerLatents & latents) const;
    VibeVoiceTokenizerLatents encode_semantic_streaming(
        const runtime::AudioBuffer & audio,
        VibeVoiceTokenizerStreamingState & state) const;
    std::vector<VibeVoiceTokenizerLatents> encode_semantic_streaming_batch(
        const std::vector<runtime::AudioBuffer> & audio,
        std::vector<VibeVoiceTokenizerStreamingState *> states) const;
    runtime::AudioBuffer decode_acoustic_streaming(
        const VibeVoiceTokenizerLatents & latents,
        VibeVoiceTokenizerStreamingState & state) const;
    std::vector<runtime::AudioBuffer> decode_acoustic_streaming_batch(
        const std::vector<VibeVoiceTokenizerLatents> & latents,
        std::vector<VibeVoiceTokenizerStreamingState *> states) const;
    void release_prompt_graphs() const;

private:
    std::shared_ptr<const VibeVoiceAssets> assets_;
    std::shared_ptr<const VibeVoiceTokenizerWeightsBundle> weights_;
    std::unique_ptr<core::ConstantTensorCache> acoustic_encoder_constants_;
    std::unique_ptr<core::ConstantTensorCache> semantic_encoder_constants_;
    std::unique_ptr<core::ConstantTensorCache> acoustic_decoder_constants_;
    std::unique_ptr<core::ConstantTensorCache> semantic_streaming_constants_;
    std::unique_ptr<core::ConstantTensorCache> acoustic_streaming_constants_;
    mutable std::unique_ptr<VibeVoiceTokenizerEncoderGraph> acoustic_encoder_graph_;
    mutable std::unique_ptr<VibeVoiceTokenizerEncoderGraph> semantic_encoder_graph_;
    mutable std::unique_ptr<VibeVoiceTokenizerDecoderGraph> acoustic_decoder_graph_;
    mutable std::unique_ptr<VibeVoiceTokenizerStreamingGraph> semantic_streaming_graph_;
    mutable std::unique_ptr<VibeVoiceTokenizerStreamingGraph> acoustic_streaming_graph_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
};

VibeVoiceTokenizerWeightsBundle load_vibevoice_tokenizer_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

VibeVoiceAcousticLatentSample sample_vibevoice_acoustic_latents_gaussian(
    const std::vector<VibeVoiceTokenizerLatents> & means,
    float fixed_std,
    uint64_t seed,
    uint64_t start_rng_index,
    sampling::TorchRandnPrecision precision,
    const std::vector<float> * prompt_noise_values = nullptr);

VibeVoiceTokenizerLatents scale_vibevoice_acoustic_latents_for_connector(
    const VibeVoiceTokenizerLatents & latents,
    float speech_scaling_factor,
    float speech_bias_factor);

VibeVoiceTokenizerLatents unscale_vibevoice_acoustic_latents_for_decoder(
    const VibeVoiceTokenizerLatents & latents,
    float speech_scaling_factor,
    float speech_bias_factor);

}  // namespace engine::models::vibevoice
