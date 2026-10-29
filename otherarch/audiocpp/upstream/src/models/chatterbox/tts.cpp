#include "engine/models/chatterbox/tts.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::chatterbox {
namespace {

constexpr int32_t kSpeechVocabSize = 6561;
constexpr int32_t kSpeechStartToken = kSpeechVocabSize;
constexpr int32_t kSpeechStopToken = kSpeechVocabSize + 1;

std::vector<int32_t> clean_generated_speech_tokens_like_python(const std::vector<int32_t> & tokens) {
    // Mirror Python's drop_invalid_tokens(): keep the span after the first SOS
    // up to the first EOS, then drop any remaining non-speech ids.
    auto start_it = std::find(tokens.begin(), tokens.end(), kSpeechStartToken);
    if (start_it != tokens.end()) {
        ++start_it;
    } else {
        start_it = tokens.begin();
    }

    const auto stop_it = std::find(start_it, tokens.end(), kSpeechStopToken);

    std::vector<int32_t> cleaned;
    cleaned.reserve(static_cast<size_t>(std::distance(start_it, stop_it)));
    for (auto it = start_it; it != stop_it; ++it) {
        const int32_t token = *it;
        if (token >= 0 && token < kSpeechVocabSize) {
            cleaned.push_back(token);
        }
    }
    return cleaned;
}

}  // namespace

void apply_s3_trim_fade(std::vector<float> & waveform, int sample_rate) {
    const int64_t n_trim = sample_rate / 50;
    const int64_t fade_size = 2 * n_trim;
    if (waveform.empty() || n_trim <= 0) {
        return;
    }
    const int64_t keep = std::min<int64_t>(static_cast<int64_t>(waveform.size()), fade_size);
    for (int64_t i = 0; i < std::min<int64_t>(keep, n_trim); ++i) {
        waveform[static_cast<size_t>(i)] = 0.0f;
    }
    constexpr double kPi = 3.141592653589793238462643383279502884;
    for (int64_t i = n_trim; i < keep; ++i) {
        const double alpha = static_cast<double>(i - n_trim) / static_cast<double>(std::max<int64_t>(n_trim - 1, 1));
        const float fade = static_cast<float>((std::cos(kPi * (1.0 - alpha)) + 1.0) / 2.0);
        waveform[static_cast<size_t>(i)] *= fade;
    }
}

namespace {

engine::core::BackendMemorySnapshot capture_backend_memory_snapshot(const engine::core::ExecutionContext * execution_context) {
    if (!engine::debug::trace_log_enabled() && !engine::debug::timing_log_enabled()) {
        return {};
    }
    return execution_context != nullptr ? execution_context->memory_snapshot() : engine::core::BackendMemorySnapshot{};
}

}  // namespace

ChatterboxTTSComponent::ChatterboxTTSComponent(
    std::shared_ptr<const T3InferenceWeights> t3_weights,
    std::shared_ptr<const ChatterboxEnglishTokenizerModel> tokenizer,
    engine::models::chatterbox::VoiceEncoderComponent voice_encoder,
    engine::models::chatterbox::S3TokenizerComponent tokenizer_component,
    engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder,
    std::unique_ptr<engine::codecs::s3gen::S3GenRuntime> s3gen,
    ChatterboxPromptPrepConfig prompt_prep_config,
    const engine::core::ExecutionContext & execution_context,
    bool mem_saver)
    : t3_(
          std::move(t3_weights),
          execution_context),
      tokenizer_(std::move(tokenizer)),
      conditionals_(
          std::move(voice_encoder),
          std::move(tokenizer_component),
          std::move(speaker_encoder),
          prompt_prep_config),
      s3gen_(std::move(s3gen)),
      execution_context_(&execution_context),
      mem_saver_(mem_saver) {
    if (!tokenizer_) {
        throw std::runtime_error("ChatterboxTtsComponent requires text tokenizer");
    }
}

ChatterboxVoiceCloneOutputs ChatterboxTTSComponent::synthesize_voice_clone(
    const std::string & text,
    const runtime::AudioBuffer & reference_audio,
    const ChatterboxVoiceCloneConfig & config) const {
    const auto prompt_prep_memory_before = capture_backend_memory_snapshot(execution_context_);
    const auto prompt_prep_started = std::chrono::steady_clock::now();
    const auto conds = prepare_voice_clone_conditionals(reference_audio, config);
    const double prompt_prep_ms =
        engine::debug::elapsed_ms(prompt_prep_started);
    const auto prompt_prep_memory_after = capture_backend_memory_snapshot(execution_context_);
    auto outputs = synthesize_voice_clone_impl(text, conds, config);
    outputs.prompt_prep_ms = prompt_prep_ms;
    if (prompt_prep_memory_after.available) {
        outputs.cuda_memory_total_bytes = prompt_prep_memory_after.total_bytes;
    } else if (prompt_prep_memory_before.available) {
        outputs.cuda_memory_total_bytes = prompt_prep_memory_before.total_bytes;
    }
    outputs.prompt_prep_cuda_memory_used_before_bytes = prompt_prep_memory_before.used_bytes;
    outputs.prompt_prep_cuda_memory_used_after_bytes = prompt_prep_memory_after.used_bytes;
    engine::debug::timing_log_scalar("chatterbox.voice_clone.prepare.total_ms", outputs.prompt_prep_ms);
    return outputs;
}

ChatterboxConditionalsOutputs ChatterboxTTSComponent::prepare_voice_clone_conditionals(
    const runtime::AudioBuffer & reference_audio,
    const ChatterboxVoiceCloneConfig & config) const {
    return conditionals_.prepare(reference_audio, config.exaggeration);
}

ChatterboxVoiceCloneOutputs ChatterboxTTSComponent::synthesize_voice_clone_with_conditionals(
    const std::string & text,
    const ChatterboxConditionalsOutputs & conditionals,
    const ChatterboxVoiceCloneConfig & config) const {
    return synthesize_voice_clone_impl(text, conditionals, config);
}

ChatterboxVoiceCloneOutputs ChatterboxTTSComponent::synthesize_voice_clone_impl(
    const std::string & text,
    const ChatterboxConditionalsOutputs & conds,
    const ChatterboxVoiceCloneConfig & config) const {
    ChatterboxVoiceCloneOutputs outputs;
    outputs.normalized_text = normalize_chatterbox_tts_text(text);
    const std::string language = normalize_chatterbox_language_code(config.language);
    outputs.text_tokens = chatterbox_language_uses_multilingual_t3(language)
        ? encode_chatterbox_multilingual_text(*tokenizer_, outputs.normalized_text, language)
        : encode_chatterbox_english_text(*tokenizer_, outputs.normalized_text);
    outputs.text_tokens.insert(outputs.text_tokens.begin(), chatterbox_english_start_token_id(*tokenizer_));
    outputs.text_tokens.push_back(chatterbox_english_stop_token_id(*tokenizer_));

    outputs.prompt_prep_gen_ms = conds.gen_ms;
    outputs.prompt_prep_gen_prompt_mel_ms = conds.gen.prompt_mel_ms;
    outputs.prompt_prep_gen_speaker_ms = conds.gen.speaker_ms;
    outputs.prompt_prep_gen_tokenizer_ms = conds.gen.tokenizer_ms;
    outputs.prompt_prep_voice_encoder_ms = conds.voice_encoder_ms;
    outputs.prompt_prep_tokenizer_ms = conds.tokenizer_ms;

    T3GenerateRequest request;
    request.speaker_embedding = conds.t3.speaker_embedding;
    request.cond_prompt_speech_tokens = conds.t3.cond_prompt_speech_tokens;
    request.emotion_adv = conds.t3.emotion_adv;
    request.text_tokens = outputs.text_tokens;
    request.max_new_tokens = config.max_new_tokens;
    request.stop_on_eos = config.stop_on_eos;
    request.do_sample = config.do_sample;
    request.temperature = config.temperature;
    request.top_p = config.top_p;
    request.min_p = config.min_p;
    request.repetition_penalty = config.repetition_penalty;
    request.guidance_scale = config.guidance_scale;
    request.seed = config.seed;

    const auto t3_memory_before = capture_backend_memory_snapshot(execution_context_);
    const auto t3_started = std::chrono::steady_clock::now();
    const auto t3_outputs = t3_.generate_speech_tokens(request);
    outputs.t3_ms =
        engine::debug::elapsed_ms(t3_started);
    const auto t3_memory_after = capture_backend_memory_snapshot(execution_context_);
    if (t3_memory_after.available) {
        outputs.cuda_memory_total_bytes = t3_memory_after.total_bytes;
    } else if (t3_memory_before.available) {
        outputs.cuda_memory_total_bytes = t3_memory_before.total_bytes;
    }
    outputs.t3_prefix_cache_build_ms = t3_outputs.prefix_cache_build_ms;
    outputs.t3_decoder_cache_clone_ms = t3_outputs.decoder_cache_clone_ms;
    outputs.t3_prefill_runner_ms = t3_outputs.prefill_runner_ms;
    outputs.t3_decode_runner_ms = t3_outputs.decode_runner_ms;
    outputs.t3_logits_ms = t3_outputs.logits_ms;
    outputs.t3_sampling_ms = t3_outputs.sampling_ms;
    outputs.t3_next_embed_ms = t3_outputs.next_embed_ms;
    outputs.t3_cuda_memory_used_before_bytes = t3_memory_before.used_bytes;
    outputs.t3_cuda_memory_used_after_bytes = t3_memory_after.used_bytes;
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.total_ms", outputs.t3_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.prefix_cache.build_ms", outputs.t3_prefix_cache_build_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.decoder_cache.clone_ms", outputs.t3_decoder_cache_clone_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.prefill_runner_ms", outputs.t3_prefill_runner_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.decode_runner_ms", outputs.t3_decode_runner_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.logits_ms", outputs.t3_logits_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.sampling_ms", outputs.t3_sampling_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.t3.next_embed_ms", outputs.t3_next_embed_ms);
    outputs.generated_speech_tokens = t3_outputs.predicted_tokens;
    outputs.cleaned_speech_tokens = clean_generated_speech_tokens_like_python(outputs.generated_speech_tokens);
    if (mem_saver_) {
        t3_.release_runtime_graphs();
    } else if (execution_context_ != nullptr && execution_context_->uses_host_graph_plan()) {
        t3_.release_runtime_cache();
    }

    const auto s3gen_memory_before = capture_backend_memory_snapshot(execution_context_);
    const auto s3gen_started = std::chrono::steady_clock::now();
    engine::codecs::s3gen::S3GenTimingBreakdown s3gen_timing;
    const auto token2mel_memory_before = capture_backend_memory_snapshot(execution_context_);
    const auto token2mel_started = std::chrono::steady_clock::now();
    const auto mel = s3gen_->token_to_mel(
        conds.gen,
        outputs.cleaned_speech_tokens,
        static_cast<int64_t>(outputs.cleaned_speech_tokens.size()),
        10,
        config.s3gen_cfg_rate,
        true,
        {},
        config.seed,
        &s3gen_timing);
    s3gen_timing.token2mel_ms =
        engine::debug::elapsed_ms(token2mel_started);
    const auto token2mel_memory_after = capture_backend_memory_snapshot(execution_context_);
    if (mem_saver_) {
        s3gen_->release_flow_graphs();
    }
    const uint64_t prior_noise_values =
        static_cast<uint64_t>(mel.channels * (conds.gen.prompt_feat_frames + mel.frames));
    const auto vocoder_memory_before = capture_backend_memory_snapshot(execution_context_);
    const auto vocoder_started = std::chrono::steady_clock::now();
    const auto vocoder_outputs = s3gen_->decode_waveform(
        mel.mel,
        mel.frames,
        config.seed,
        prior_noise_values);
    s3gen_timing.vocoder_ms =
        engine::debug::elapsed_ms(vocoder_started);
    if (mem_saver_) {
        s3gen_->release_vocoder_graphs();
    }
    const auto vocoder_memory_after = capture_backend_memory_snapshot(execution_context_);
    outputs.s3gen_ms =
        engine::debug::elapsed_ms(s3gen_started);
    const auto s3gen_memory_after = capture_backend_memory_snapshot(execution_context_);
    if (s3gen_memory_after.available) {
        outputs.cuda_memory_total_bytes = s3gen_memory_after.total_bytes;
    } else if (s3gen_memory_before.available && outputs.cuda_memory_total_bytes == 0) {
        outputs.cuda_memory_total_bytes = s3gen_memory_before.total_bytes;
    }
    outputs.waveform = vocoder_outputs.waveform;
    outputs.samples = vocoder_outputs.samples;
    outputs.mel = mel.mel;
    outputs.mel_channels = mel.channels;
    outputs.mel_frames = mel.frames;
    apply_s3_trim_fade(outputs.waveform, 24000);
    if (chatterbox_language_uses_multilingual_t3(language)) {
        const int64_t speech_token_audio_samples = 24000 / 25;
        const int64_t effective_speech_tokens =
            std::max<int64_t>(1, static_cast<int64_t>(outputs.cleaned_speech_tokens.size()) - 1);
        const int64_t keep_samples = effective_speech_tokens * speech_token_audio_samples;
        if (keep_samples < static_cast<int64_t>(outputs.waveform.size())) {
            outputs.waveform.resize(static_cast<size_t>(keep_samples));
            outputs.samples = keep_samples;
        }
    }
    outputs.s3gen_token2mel_ms = s3gen_timing.token2mel_ms;
    outputs.s3gen_token2mel_embed_ms = s3gen_timing.token2mel_embed_ms;
    outputs.s3gen_token2mel_encoder_ms = s3gen_timing.token2mel_encoder_ms;
    outputs.s3gen_token2mel_mu_ms = s3gen_timing.token2mel_mu_ms;
    outputs.s3gen_token2mel_cfm_ms = s3gen_timing.token2mel_cfm_ms;
    outputs.s3gen_token2mel_cfm_timing = s3gen_timing.token2mel_cfm;
    outputs.s3gen_vocoder_ms = s3gen_timing.vocoder_ms;
    outputs.s3gen_token2mel_cuda_memory_used_before_bytes = token2mel_memory_before.used_bytes;
    outputs.s3gen_token2mel_cuda_memory_used_after_bytes = token2mel_memory_after.used_bytes;
    outputs.s3gen_vocoder_cuda_memory_used_before_bytes = vocoder_memory_before.used_bytes;
    outputs.s3gen_vocoder_cuda_memory_used_after_bytes = vocoder_memory_after.used_bytes;
    outputs.s3gen_cuda_memory_used_before_bytes = s3gen_memory_before.used_bytes;
    outputs.s3gen_cuda_memory_used_after_bytes = s3gen_memory_after.used_bytes;
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.total_ms", outputs.s3gen_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.token2mel.total_ms", outputs.s3gen_token2mel_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.token2mel.embed_ms", outputs.s3gen_token2mel_embed_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.token2mel.encoder_ms", outputs.s3gen_token2mel_encoder_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.token2mel.mu_ms", outputs.s3gen_token2mel_mu_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.token2mel.cfm.total_ms", outputs.s3gen_token2mel_cfm_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.token2mel.cfm.host_update_ms", outputs.s3gen_token2mel_cfm_timing.host_update_ms);
    engine::debug::timing_log_scalar(
        "chatterbox.voice_clone.s3gen.token2mel.cfm.decoder.conditioned.graph.total_ms",
        outputs.s3gen_token2mel_cfm_timing.conditioned.graph_compute_ms);
    engine::debug::timing_log_scalar(
        "chatterbox.voice_clone.s3gen.token2mel.cfm.decoder.unconditioned.graph.total_ms",
        outputs.s3gen_token2mel_cfm_timing.unconditioned.graph_compute_ms);
    engine::debug::timing_log_scalar("chatterbox.voice_clone.s3gen.vocoder_ms", outputs.s3gen_vocoder_ms);
    return outputs;
}

}  // namespace engine::models::chatterbox
