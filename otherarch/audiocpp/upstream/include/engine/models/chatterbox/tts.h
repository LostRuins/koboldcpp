#pragma once

#include "engine/models/chatterbox/components.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/chatterbox/conditionals.h"
#include "engine/framework/codecs/s3gen_runtime.h"
#include "engine/models/chatterbox/t3_component.h"
#include "engine/models/chatterbox/text_tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::chatterbox {

struct ChatterboxVoiceCloneConfig {
    float exaggeration = 0.5f;
    float guidance_scale = 0.5f;
    float temperature = 0.8f;
    float repetition_penalty = 1.2f;
    float min_p = 0.05f;
    float top_p = 1.0f;
    float s3gen_cfg_rate = 0.7f;
    int64_t max_new_tokens = 384;
    uint32_t seed = 0;
    std::string language = "en";
    bool do_sample = true;
    bool stop_on_eos = true;
};

struct ChatterboxVoiceCloneOutputs {
    std::string normalized_text;
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> generated_speech_tokens;
    std::vector<int32_t> cleaned_speech_tokens;
    int64_t cuda_memory_total_bytes = 0;
    double prompt_prep_ms = 0.0;
    double prompt_prep_gen_ms = 0.0;
    double prompt_prep_gen_prompt_mel_ms = 0.0;
    double prompt_prep_gen_speaker_ms = 0.0;
    double prompt_prep_gen_tokenizer_ms = 0.0;
    double prompt_prep_voice_encoder_ms = 0.0;
    double prompt_prep_tokenizer_ms = 0.0;
    int64_t prompt_prep_cuda_memory_used_before_bytes = 0;
    int64_t prompt_prep_cuda_memory_used_after_bytes = 0;
    double t3_ms = 0.0;
    double t3_prefix_cache_build_ms = 0.0;
    double t3_decoder_cache_clone_ms = 0.0;
    double t3_prefill_runner_ms = 0.0;
    double t3_decode_runner_ms = 0.0;
    double t3_logits_ms = 0.0;
    double t3_sampling_ms = 0.0;
    double t3_next_embed_ms = 0.0;
    int64_t t3_cuda_memory_used_before_bytes = 0;
    int64_t t3_cuda_memory_used_after_bytes = 0;
    double s3gen_ms = 0.0;
    double s3gen_token2mel_ms = 0.0;
    double s3gen_token2mel_embed_ms = 0.0;
    double s3gen_token2mel_encoder_ms = 0.0;
    double s3gen_token2mel_mu_ms = 0.0;
    double s3gen_token2mel_cfm_ms = 0.0;
    engine::codecs::s3gen::S3FlowCFMTimingBreakdown s3gen_token2mel_cfm_timing;
    double s3gen_vocoder_ms = 0.0;
    int64_t s3gen_token2mel_cuda_memory_used_before_bytes = 0;
    int64_t s3gen_token2mel_cuda_memory_used_after_bytes = 0;
    int64_t s3gen_vocoder_cuda_memory_used_before_bytes = 0;
    int64_t s3gen_vocoder_cuda_memory_used_after_bytes = 0;
    int64_t s3gen_cuda_memory_used_before_bytes = 0;
    int64_t s3gen_cuda_memory_used_after_bytes = 0;
    std::vector<float> waveform;
    int64_t samples = 0;
    std::vector<float> source;
    int64_t source_channels = 0;
    int64_t source_frames = 0;
    std::vector<float> post;
    int64_t post_frames = 0;
    std::vector<float> mel;
    int64_t mel_channels = 0;
    int64_t mel_frames = 0;
};

class ChatterboxTTSComponent {
public:
    ChatterboxTTSComponent(
        std::shared_ptr<const T3InferenceWeights> t3_weights,
        std::shared_ptr<const ChatterboxEnglishTokenizerModel> tokenizer,
        engine::models::chatterbox::VoiceEncoderComponent voice_encoder,
        engine::models::chatterbox::S3TokenizerComponent tokenizer_component,
        engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder,
        std::unique_ptr<engine::codecs::s3gen::S3GenRuntime> s3gen,
        ChatterboxPromptPrepConfig prompt_prep_config,
        const engine::core::ExecutionContext & execution_context,
        bool mem_saver = false);

    ChatterboxVoiceCloneOutputs synthesize_voice_clone(
        const std::string & text,
        const runtime::AudioBuffer & reference_audio,
        const ChatterboxVoiceCloneConfig & config = {}) const;

    ChatterboxConditionalsOutputs prepare_voice_clone_conditionals(
        const runtime::AudioBuffer & reference_audio,
        const ChatterboxVoiceCloneConfig & config = {}) const;

    ChatterboxVoiceCloneOutputs synthesize_voice_clone_with_conditionals(
        const std::string & text,
        const ChatterboxConditionalsOutputs & conditionals,
        const ChatterboxVoiceCloneConfig & config = {}) const;

private:
    ChatterboxVoiceCloneOutputs synthesize_voice_clone_impl(
        const std::string & text,
        const ChatterboxConditionalsOutputs & conditionals,
        const ChatterboxVoiceCloneConfig & config) const;

    T3InferenceComponent t3_;
    std::shared_ptr<const ChatterboxEnglishTokenizerModel> tokenizer_;
    ChatterboxConditionalsComponent conditionals_;
    std::unique_ptr<engine::codecs::s3gen::S3GenRuntime> s3gen_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    bool mem_saver_ = false;
};

}  // namespace engine::models::chatterbox
