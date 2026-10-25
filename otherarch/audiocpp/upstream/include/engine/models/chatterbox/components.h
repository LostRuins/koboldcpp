#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/chatterbox/assets.h"
#include "engine/framework/codecs/s3gen_runtime.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::modules {
struct HiftVocoderWeights;
}  // namespace engine::modules

namespace engine::models::chatterbox {

struct EmbedReferenceOutputs : codecs::s3gen::S3GenConditioning {
    double prompt_mel_ms = 0.0;
    double speaker_ms = 0.0;
    double tokenizer_ms = 0.0;
};

void apply_s3_trim_fade(std::vector<float> & waveform, int sample_rate);

namespace components {
struct S3TokenizerV2Weights;
}  // namespace components

struct SpeakerEncoderOutputs {
    std::vector<float> embedding;
    int64_t embedding_size = 0;
};

struct TokenizerOutputs {
    std::vector<int32_t> tokens;
    int64_t token_count = 0;
};


struct VoiceEncoderConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 512;
    int64_t win_size = 400;
    int64_t hop_size = 160;
    int64_t num_mels = 40;
    int64_t partial_frames = 160;
    int64_t partial_rate = 1;
    float min_coverage = 0.75F;
    int64_t hidden_size = 256;
    int64_t speaker_embed_size = 256;
    bool final_relu = true;
};

std::vector<float> compute_voice_encoder_mel(
    const std::vector<float> & waveform,
    const VoiceEncoderConfig & config);

struct VoiceEncoderLayerWeights {
    std::vector<float> weight_ih;
    std::vector<float> weight_hh;
    std::vector<float> bias_ih;
    std::vector<float> bias_hh;
};

struct VoiceEncoderWeights {
    VoiceEncoderConfig config;
    std::vector<VoiceEncoderLayerWeights> lstm_layers;
    std::vector<float> proj_weight;
    std::vector<float> proj_bias;
    std::vector<float> similarity_weight;
    std::vector<float> similarity_bias;
};

struct S3TokenizerComponentWeights {
    std::shared_ptr<const components::S3TokenizerV2Weights> runtime_weights;
};


std::shared_ptr<const VoiceEncoderWeights> load_voice_encoder_weights(
    const engine::assets::TensorSource & source);

class VoiceEncoderComponent {
public:
    static VoiceEncoderComponent load_from_source(
        const engine::assets::TensorSource & source,
        engine::core::BackendConfig backend);

    VoiceEncoderComponent(
        std::shared_ptr<const VoiceEncoderWeights> weights,
        engine::core::BackendConfig backend);

    const engine::core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const VoiceEncoderWeights> & weights() const noexcept;
    const VoiceEncoderConfig & config() const noexcept;
    std::vector<float> embed_utterance_from_audio(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const VoiceEncoderWeights> weights_;
    engine::core::BackendConfig backend_;
};

class CAMPPlusEncoderComponent {
    struct State;

public:
    static CAMPPlusEncoderComponent load_from_source(
        std::shared_ptr<const engine::assets::TensorSource> source,
        const engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

    CAMPPlusEncoderComponent() = default;
    explicit CAMPPlusEncoderComponent(
        std::shared_ptr<State> state,
        const engine::core::ExecutionContext & execution_context);

    const engine::core::BackendConfig & backend() const noexcept;
    SpeakerEncoderOutputs embed_from_audio(const runtime::AudioBuffer & audio) const;
    SpeakerEncoderOutputs embed_from_features(
        const std::vector<float> & features,
        int64_t frames,
        int64_t dims) const;

private:
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};

class S3TokenizerComponent {
public:
    static S3TokenizerComponent load_from_source(
        const engine::assets::TensorSource & source,
        const engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

    S3TokenizerComponent(
        std::shared_ptr<const S3TokenizerComponentWeights> weights,
        const engine::core::ExecutionContext & execution_context);

    const engine::core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const S3TokenizerComponentWeights> & weights() const noexcept;
    TokenizerOutputs tokenize(
        const runtime::AudioBuffer & audio,
        std::optional<int64_t> max_len) const;
    EmbedReferenceOutputs embed_reference(
        const CAMPPlusEncoderComponent & speaker_encoder,
        const runtime::AudioBuffer & audio) const;
    EmbedReferenceOutputs embed_reference_from_wavs(
        const CAMPPlusEncoderComponent & speaker_encoder,
        const runtime::AudioBuffer & audio_24k,
        const runtime::AudioBuffer & audio_16k) const;

private:
    struct State;
    std::shared_ptr<const S3TokenizerComponentWeights> weights_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};


}  // namespace engine::models::chatterbox
