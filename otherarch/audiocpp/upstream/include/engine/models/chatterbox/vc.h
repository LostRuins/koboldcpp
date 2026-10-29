#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/chatterbox/components.h"
#include "engine/models/chatterbox/conditionals.h"
#include "engine/framework/codecs/s3gen_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::chatterbox {

struct ChatterboxVoiceConversionConfig {
    float s3gen_cfg_rate = 0.7f;
    int64_t num_inference_steps = 10;
    uint32_t seed = 0;
};

struct ChatterboxVoiceConversionOutputs {
    std::vector<int32_t> source_speech_tokens;
    int64_t source_speech_token_count = 0;
    std::vector<float> waveform;
    int64_t samples = 0;
    double target_ref_ms = 0.0;
    double source_tokenizer_ms = 0.0;
    double s3gen_ms = 0.0;
    double s3gen_token2mel_ms = 0.0;
    double s3gen_vocoder_ms = 0.0;
    engine::codecs::s3gen::S3GenTimingBreakdown s3gen_timing;
};

class ChatterboxVCComponent {
public:
    ChatterboxVCComponent(
        engine::models::chatterbox::S3TokenizerComponent tokenizer_component,
        engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder,
        std::unique_ptr<engine::codecs::s3gen::S3GenRuntime> s3gen,
        ChatterboxPromptPrepConfig prompt_prep_config,
        bool mem_saver = false);

    ChatterboxVoiceConversionOutputs convert(
        const runtime::AudioBuffer & source_audio,
        const runtime::AudioBuffer & target_voice,
        const ChatterboxVoiceConversionConfig & config = {}) const;

private:
    engine::models::chatterbox::S3TokenizerComponent tokenizer_;
    engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder_;
    std::unique_ptr<engine::codecs::s3gen::S3GenRuntime> s3gen_;
    ChatterboxPromptPrepConfig prompt_prep_config_;
    bool mem_saver_ = false;
};

runtime::AudioBuffer load_chatterbox_vc_audio_mono(
    const std::string & path,
    int sample_rate);

runtime::AudioBuffer normalize_chatterbox_vc_audio_mono(
    const runtime::AudioBuffer & audio,
    int sample_rate);

}  // namespace engine::models::chatterbox
