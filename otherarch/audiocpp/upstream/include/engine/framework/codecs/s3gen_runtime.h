#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::codecs::s3gen {

struct S3GenConditioning {
    std::vector<int32_t> prompt_tokens;
    int64_t prompt_token_count = 0;
    std::vector<float> prompt_feat;
    int64_t prompt_feat_frames = 0;
    int64_t prompt_feat_dims = 0;
    std::vector<float> embedding;
    int64_t embedding_size = 0;
};

struct S3Token2MelOutputs {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct S3FlowEncoderOutputs {
    std::vector<float> hidden;
    int64_t frames = 0;
    int64_t storage_frames = 0;
    int64_t hidden_size = 0;
};

struct S3GenInferenceOutputs {
    std::vector<float> waveform;
    int64_t samples = 0;
    std::vector<float> mel;
    int64_t mel_channels = 0;
    int64_t mel_frames = 0;
};

struct S3FlowDecoderRunTiming {
    int64_t calls = 0;
    double conditioning_write_ms = 0.0;
    double time_embedding_ms = 0.0;
    double input_write_ms = 0.0;
    double time_write_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
};

struct S3FlowCFMTimingBreakdown {
    int64_t steps = 0;
    int64_t decoder_calls = 0;
    double initial_state_ms = 0.0;
    double schedule_ms = 0.0;
    double zero_conditioning_ms = 0.0;
    double runner_setup_ms = 0.0;
    double host_update_ms = 0.0;
    S3FlowDecoderRunTiming conditioned;
    S3FlowDecoderRunTiming unconditioned;
};

struct S3GenTimingBreakdown {
    double token2mel_ms = 0.0;
    double token2mel_embed_ms = 0.0;
    double token2mel_encoder_ms = 0.0;
    double token2mel_mu_ms = 0.0;
    double token2mel_cfm_ms = 0.0;
    S3FlowCFMTimingBreakdown token2mel_cfm;
    double vocoder_ms = 0.0;
};

enum class S3FlowVariant { Conditional, MeanFlow };

struct S3FlowEncoderConfig {
    int64_t vocabulary_size = 6561;
    int64_t hidden_size = 512;
    int64_t heads = 8;
    int64_t feed_forward_size = 2048;
    int64_t layers = 6;
    int64_t upsample_layers = 4;
    int upsample_factor = 2;
    int64_t prelook_kernel_size = 4;
    int64_t prelook_output_kernel_size = 3;
    int64_t upsample_kernel_size = 5;
    int64_t speaker_embedding_size = 192;
};

struct S3FlowDecoderConfig {
    // Single-scale estimator: one entry block, repeated middle blocks, one exit block.
    S3FlowVariant variant = S3FlowVariant::Conditional;
    int64_t hidden_size = 256;
    int64_t heads = 8;
    int64_t head_dim = 64;
    int64_t feed_forward_size = 1024;
    int64_t time_embedding_size = 320;
    int64_t time_hidden_size = 1024;
    int64_t middle_blocks = 12;
    int64_t attention_layers = 4;
};

struct S3GenConfig {
    S3GenConfig();
    int64_t mel_channels = 80;
    S3FlowEncoderConfig encoder;
    S3FlowDecoderConfig decoder;
    modules::HiftVocoderConfig vocoder;
};

struct S3GenWeightBinding {
    std::string flow_prefix = "flow.";
    std::string vocoder_prefix = "mel2wav.";
    modules::HiftVocoderWeightLayout vocoder_layout =
        modules::HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
};

struct S3GenRuntimeOptions {
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    // Reuse the last exact-shape encoder graph set between synthesis calls.
    bool retain_encoder_graphs = false;
};

class S3GenRuntime {
public:
    S3GenRuntime(std::shared_ptr<const assets::TensorSource> source,
                 const core::ExecutionContext & execution, S3GenConfig config,
                 S3GenRuntimeOptions options = {}, S3GenWeightBinding binding = {});
    ~S3GenRuntime();
    S3GenRuntime(S3GenRuntime &&) noexcept;
    S3GenRuntime & operator=(S3GenRuntime &&) noexcept;
    S3GenRuntime(const S3GenRuntime &) = delete;
    S3GenRuntime & operator=(const S3GenRuntime &) = delete;

    bool is_meanflow() const;
    S3FlowEncoderOutputs encode(
        const std::vector<float> & embeddings, int64_t frames,
        int64_t capacity_frames, int64_t hidden_size) const;
    S3Token2MelOutputs token_to_mel(
        const S3GenConditioning & reference,
        const std::vector<int32_t> & speech_tokens, int64_t speech_token_count,
        int64_t num_steps = 10, float cfg_rate = 0.7f, bool cosine_schedule = true,
        const std::vector<float> & full_noise = {}, uint64_t flow_seed = 0,
        S3GenTimingBreakdown * timing = nullptr) const;
    modules::HiftVocoderOutput decode_waveform(
        const std::vector<float> & mel, int64_t frames,
        uint64_t seed = 0, uint64_t prior_noise_values = 0) const;
    S3GenInferenceOutputs synthesize(
        const S3GenConditioning & reference,
        const std::vector<int32_t> & speech_tokens, int64_t speech_token_count,
        int64_t num_steps = 10, float cfg_rate = 0.7f, bool cosine_schedule = true,
        const std::vector<float> & full_noise = {}, uint64_t flow_seed = 0,
        uint64_t vocoder_seed = 0, S3GenTimingBreakdown * timing = nullptr) const;

    const modules::HiftVocoderComponent & vocoder() const;
    void release_flow_graphs() const;
    void release_vocoder_graphs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::codecs::s3gen
