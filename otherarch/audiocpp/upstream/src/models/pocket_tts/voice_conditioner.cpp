#include "engine/models/pocket_tts/voice_conditioner.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "engine/models/pocket_tts/voice_state_assets.h"
#include "graph_common.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::pocket_tts {
namespace {

std::vector<float> append_bos_voice_embedding(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSHostWeights & weights,
    const std::vector<float> & conditioning,
    int64_t frames) {
    const int64_t flow_hidden = manifest.model_config.flow_dim;
    if (!manifest.model_config.insert_bos_before_voice) {
        return conditioning;
    }
    if (!weights.bos_before_voice.has_value()) {
        throw std::runtime_error("flow_lm.bos_before_voice is required when insert_bos_before_voice is enabled");
    }
    const auto & bos = *weights.bos_before_voice;
    if (bos.shape.rank != 3 || bos.shape.dims[0] != 1 || bos.shape.dims[1] != 1 || bos.shape.dims[2] != flow_hidden) {
        throw std::runtime_error("flow_lm.bos_before_voice must have shape [1, 1, flow_dim]");
    }
    std::vector<float> output;
    output.reserve(static_cast<size_t>((frames + 1) * flow_hidden));
    output.insert(output.end(), bos.values.begin(), bos.values.end());
    output.insert(output.end(), conditioning.begin(), conditioning.end());
    return output;
}

MimiPromptEmbedding append_bos_voice_embedding(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSHostWeights & weights,
    MimiPromptEmbedding embedding) {
    const int64_t flow_hidden = manifest.model_config.flow_dim;
    embedding.values = append_bos_voice_embedding(manifest, weights, embedding.values, embedding.frames);
    embedding.frames = static_cast<int64_t>(embedding.values.size() / static_cast<size_t>(flow_hidden));
    return embedding;
}

std::vector<float> load_conditioning_audio(
    const std::filesystem::path & path,
    int target_sample_rate,
    int64_t frame_size,
    bool truncate) {
    auto resampled = audio::read_wav_f32_as_mono_linear_resampled(path, target_sample_rate);
    if (truncate) {
        const size_t max_samples = static_cast<size_t>(30 * target_sample_rate);
        if (resampled.size() > max_samples) {
            resampled.resize(max_samples);
        }
    }
    return audio::zero_pad_samples_to_multiple(resampled, frame_size);
}

std::vector<float> prepare_conditioning_audio(
    const runtime::AudioBuffer & audio_buffer,
    int target_sample_rate,
    int64_t frame_size,
    bool truncate) {
    auto resampled = audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio_buffer.samples,
        audio_buffer.sample_rate,
        audio_buffer.channels,
        target_sample_rate);
    if (truncate) {
        const size_t max_samples = static_cast<size_t>(30 * target_sample_rate);
        if (resampled.size() > max_samples) {
            resampled.resize(max_samples);
        }
    }
    return audio::zero_pad_samples_to_multiple(resampled, frame_size);
}

}  // namespace

VoiceConditioner::VoiceConditioner(FlowLMConfig flow_config) : flow_lm_(std::move(flow_config)) {}

VoiceConditioningResult VoiceConditioner::prepare(
    const models::pocket_tts::VoiceConditioningPlan & plan,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    ggml_backend_t backend,
    int threads,
    size_t mimi_encoder_graph_context_bytes,
    size_t flow_weights_view_context_bytes,
    size_t flow_step_graph_context_bytes) const {
    VoiceConditioningResult result;
    int64_t prompt_frames = 0;
    const double prepare_ms = engine::debug::measure_ms([&]() {
        if (plan.source != models::pocket_tts::VoiceSourceKind::CloneAudio) {
            const auto voice_assets = models::pocket_tts::load_voice_assets_for_plan(plan, manifest);
            result = VoiceConditioningResult{
                plan,
                flow_lm_.make_state(voice_assets),
            };
            return;
        }

        const int sample_rate = manifest.model_config.sample_rate;
        const auto frame_rate = manifest.model_config.mimi_frame_rate;
        const int64_t frame_size = static_cast<int64_t>(std::llround(static_cast<double>(sample_rate) / frame_rate));
        const auto audio = plan.clone_audio.has_value()
            ? prepare_conditioning_audio(*plan.clone_audio, sample_rate, frame_size, plan.truncate_clone_audio)
            : load_conditioning_audio(plan.clone_audio_path, sample_rate, frame_size, plan.truncate_clone_audio);
        auto prompt_embedding =
            mimi_encoder_.encode_prompt_embedding(backend, threads, manifest, weights, audio, mimi_encoder_graph_context_bytes);
        prompt_embedding = append_bos_voice_embedding(manifest, weights.host, std::move(prompt_embedding));
        prompt_frames = prompt_embedding.frames;
        result = VoiceConditioningResult{
            plan,
            flow_lm_.prompt_state(
                backend,
                threads,
                manifest,
                weights,
                prompt_embedding.values,
                prompt_embedding.frames,
                flow_lm_.make_empty_state(),
                flow_weights_view_context_bytes,
                flow_step_graph_context_bytes),
        };
    });
    auto source_name = std::string("unknown");
    switch (result.plan.source) {
        case models::pocket_tts::VoiceSourceKind::NamedPreset:
            source_name = "named_preset";
            break;
        case models::pocket_tts::VoiceSourceKind::PreparedEmbedding:
            source_name = "prepared_embedding";
            break;
        case models::pocket_tts::VoiceSourceKind::CloneAudio:
            source_name = "clone_audio";
            break;
    }
    engine::debug::timing_log_scalar("pocket_tts.voice.prepare_ms", prepare_ms);
    engine::debug::trace_log_scalar("pocket_tts.voice.source", source_name);
    if (prompt_frames > 0) {
        engine::debug::trace_log_scalar("pocket_tts.voice.prompt_frames", prompt_frames);
    }
    return result;
}

}  // namespace engine::models::pocket_tts
