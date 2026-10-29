#include "engine/models/chatterbox/conditionals.h"

#include "components/audio_processing.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::chatterbox {
namespace {

runtime::AudioBuffer to_mono_audio(const runtime::AudioBuffer & audio) {
    if (audio.channels == 1) {
        return audio;
    }
    const auto mono_samples =
        engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    return {audio.sample_rate, 1, mono_samples};
}

runtime::AudioBuffer trim_audio(
    const runtime::AudioBuffer & audio,
    int64_t max_samples) {
    runtime::AudioBuffer trimmed = audio;
    if (max_samples > 0 && static_cast<int64_t>(trimmed.samples.size()) > max_samples) {
        engine::audio::truncate_samples_to_count(trimmed.samples, static_cast<size_t>(max_samples));
    }
    return trimmed;
}

runtime::AudioBuffer resample_mono_audio(
    const runtime::AudioBuffer & audio,
    int target_sample_rate) {
    if (audio.channels != 1) {
        throw std::runtime_error("resample_mono_audio expects mono audio");
    }
    if (audio.sample_rate == target_sample_rate) {
        return audio;
    }
    return {
        target_sample_rate,
        1,
        components::resample_component_mono(audio.samples, audio.sample_rate, target_sample_rate),
    };
}

}  // namespace

ChatterboxConditionalsComponent::ChatterboxConditionalsComponent(
    engine::models::chatterbox::VoiceEncoderComponent voice_encoder,
    engine::models::chatterbox::S3TokenizerComponent tokenizer,
    engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder,
    ChatterboxPromptPrepConfig config)
    : voice_encoder_(std::move(voice_encoder)),
      tokenizer_(std::move(tokenizer)),
      speaker_encoder_(std::move(speaker_encoder)),
      config_(config) {}

const ChatterboxPromptPrepConfig & ChatterboxConditionalsComponent::config() const noexcept {
    return config_;
}

ChatterboxConditionalsOutputs ChatterboxConditionalsComponent::prepare(
    const runtime::AudioBuffer & audio,
    float exaggeration) const {
    const auto mono = to_mono_audio(audio);
    // Match Python prepare_conditionals():
    // 1. Build the reference path in 24 kHz first.
    // 2. Derive the 16 kHz path from that 24 kHz reference waveform.
    // 3. Use the full 16 kHz reference for VoiceEncoder speaker embedding.
    // 4. Use only the first ENC_COND_LEN samples for T3 prompt speech tokens.
    auto reference_audio_24k = resample_mono_audio(mono, static_cast<int>(config_.generator_sample_rate));
    auto reference_audio_16k = resample_mono_audio(reference_audio_24k, static_cast<int>(config_.tokenizer_sample_rate));

    auto generator_audio = trim_audio(reference_audio_24k, config_.decoder_condition_samples);
    runtime::AudioBuffer generator_audio_16k{
        static_cast<int>(config_.tokenizer_sample_rate),
        1,
        components::resample_component_torchaudio_hann_mono(
            generator_audio.samples,
            generator_audio.sample_rate,
            static_cast<int>(config_.tokenizer_sample_rate)),
    };

    auto tokenizer_audio = trim_audio(reference_audio_16k, config_.encoder_condition_samples);
    const auto & voice_encoder_audio = reference_audio_16k;

    ChatterboxConditionalsOutputs outputs;
    const auto gen_start = std::chrono::steady_clock::now();
    const auto prompt_mel_start = std::chrono::steady_clock::now();
    auto prompt_mel = components::compute_s3_prompt_mel(generator_audio);
    outputs.gen.prompt_mel_ms = engine::debug::elapsed_ms(prompt_mel_start);
    const auto speaker_start = std::chrono::steady_clock::now();
    auto speaker_embedding = speaker_encoder_.embed_from_audio(generator_audio_16k);
    outputs.gen.speaker_ms = engine::debug::elapsed_ms(speaker_start);
    const auto gen_tokenizer_start = std::chrono::steady_clock::now();
    auto gen_prompt_tokens = tokenizer_.tokenize(generator_audio_16k, std::nullopt);
    outputs.gen.tokenizer_ms = engine::debug::elapsed_ms(gen_tokenizer_start);
    const int64_t expected_token_count = prompt_mel.frames / 2;
    if (gen_prompt_tokens.token_count != expected_token_count) {
        const int64_t keep = std::min(gen_prompt_tokens.token_count, expected_token_count);
        gen_prompt_tokens.tokens.resize(static_cast<size_t>(keep));
        gen_prompt_tokens.token_count = keep;
    }
    outputs.gen.prompt_tokens = std::move(gen_prompt_tokens.tokens);
    outputs.gen.prompt_token_count = gen_prompt_tokens.token_count;
    outputs.gen.prompt_feat.assign(static_cast<size_t>(prompt_mel.frames * prompt_mel.n_mels), 0.0f);
    for (int64_t mel_bin = 0; mel_bin < prompt_mel.n_mels; ++mel_bin) {
        for (int64_t frame = 0; frame < prompt_mel.frames; ++frame) {
            outputs.gen.prompt_feat[static_cast<size_t>(frame * prompt_mel.n_mels + mel_bin)] =
                prompt_mel.mel[static_cast<size_t>(mel_bin * prompt_mel.frames + frame)];
        }
    }
    outputs.gen.prompt_feat_frames = prompt_mel.frames;
    outputs.gen.prompt_feat_dims = prompt_mel.n_mels;
    outputs.gen.embedding = std::move(speaker_embedding.embedding);
    outputs.gen.embedding_size = speaker_embedding.embedding_size;
    outputs.gen_ms = engine::debug::elapsed_ms(gen_start);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.prepare.gen.total_ms", outputs.gen_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.prepare.gen.prompt_mel_ms", outputs.gen.prompt_mel_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.prepare.gen.speaker_ms", outputs.gen.speaker_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.prepare.gen.tokenizer_ms", outputs.gen.tokenizer_ms);

    const auto voice_encoder_start = std::chrono::steady_clock::now();
    outputs.t3.speaker_embedding = voice_encoder_.embed_utterance_from_audio(voice_encoder_audio);
    outputs.voice_encoder_ms = engine::debug::elapsed_ms(voice_encoder_start);
    outputs.t3.speaker_embedding_size = static_cast<int64_t>(outputs.t3.speaker_embedding.size());
    engine::debug::timing_log_scalar("chatterbox.voice_clone.prepare.voice_encoder_ms", outputs.voice_encoder_ms);

    TokenizerOutputs prompt_tokens;
    if (config_.t3_speech_cond_prompt_len > 0) {
        const auto prompt_tokenizer_start = std::chrono::steady_clock::now();
        prompt_tokens = tokenizer_.tokenize(tokenizer_audio, config_.t3_speech_cond_prompt_len);
        outputs.tokenizer_ms = engine::debug::elapsed_ms(prompt_tokenizer_start);
    }
    engine::debug::timing_log_scalar("chatterbox.voice_clone.prepare.tokenizer_ms", outputs.tokenizer_ms);
    if (config_.t3_speech_cond_prompt_len > 0) {
        const int64_t keep = std::min(prompt_tokens.token_count, config_.t3_speech_cond_prompt_len);
        outputs.t3.cond_prompt_speech_tokens.assign(
            prompt_tokens.tokens.begin(),
            prompt_tokens.tokens.begin() + static_cast<ptrdiff_t>(keep));
        outputs.t3.cond_prompt_token_count = keep;
    }

    outputs.t3.emotion_adv = {exaggeration};
    outputs.t3.emotion_adv_count = 1;
    return outputs;
}

}  // namespace engine::models::chatterbox
