#include "engine/community_models/mms_forced_aligner/emissions.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::community_models::mms_forced_aligner {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kSampleRate16k = 16000;
constexpr int64_t kFrameStrideSamples = 320;
constexpr double kVarianceEpsilon = 1.0e-7;

std::vector<float> encode_window(
    const modules::Wav2Vec2EncoderRuntime & mms_encoder,
    const std::vector<float> & window) {
    modules::Wav2Vec2EncoderRunConfig run_config;
    run_config.apply_final_projection = true;
    const auto output = mms_encoder.encode(window, 1, static_cast<int64_t>(window.size()), run_config);
    return std::move(output.hidden_states);
}

}  // namespace

std::vector<float> mms_normalize_waveform_16k(const std::vector<float> & waveform) {
    if (waveform.empty()) {
        throw std::runtime_error("MMS forced aligner cannot normalize an empty waveform");
    }
    double sum = 0.0;
    for (const float sample : waveform) {
        sum += static_cast<double>(sample);
    }
    const double mean = sum / static_cast<double>(waveform.size());
    double variance_sum = 0.0;
    for (const float sample : waveform) {
        const double centered = static_cast<double>(sample) - mean;
        variance_sum += centered * centered;
    }
    const double denom = std::sqrt(variance_sum / static_cast<double>(waveform.size()) + kVarianceEpsilon);
    std::vector<float> out(waveform.size());
    for (size_t index = 0; index < waveform.size(); ++index) {
        out[index] = static_cast<float>((static_cast<double>(waveform[index]) - mean) / denom);
    }
    return out;
}

std::vector<float> mms_log_softmax_and_star(const float * logits, int64_t frames, int64_t classes) {
    if (frames <= 0 || classes <= 0) {
        throw std::runtime_error("MMS forced aligner log-softmax requires positive frames and classes");
    }
    std::vector<float> out(static_cast<size_t>(frames) * static_cast<size_t>(classes + 1));
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = logits + frame * classes;
        float max_logit = row[0];
        for (int64_t cls = 1; cls < classes; ++cls) {
            max_logit = std::max(max_logit, row[cls]);
        }
        double log_sum = 0.0;
        for (int64_t cls = 0; cls < classes; ++cls) {
            log_sum += std::exp(static_cast<double>(row[cls]) - static_cast<double>(max_logit));
        }
        const double log_normalizer = static_cast<double>(max_logit) + std::log(log_sum);
        float * out_row = out.data() + frame * (classes + 1);
        for (int64_t cls = 0; cls < classes; ++cls) {
            const float log_prob = static_cast<float>(static_cast<double>(row[cls]) - log_normalizer);
            if (std::isnan(log_prob)) {
                throw std::runtime_error("MMS forced aligner log-softmax produced NaN");
            }
            out_row[cls] = log_prob;
        }
        out_row[classes] = 0.0F;  // virtual <star> class, log-probability 0.0
    }
    return out;
}

MmsWav2Vec2CtcRuntime::MmsWav2Vec2CtcRuntime(
    std::shared_ptr<const MmsForcedAlignerAssets> assets,
    core::BackendConfig backend,
    engine::assets::TensorStorageType weight_storage_type,
    MmsWav2Vec2CtcConfig config)
    : assets_(std::move(assets)),
      backend_(std::move(backend)),
      weight_storage_type_(weight_storage_type),
      config_(config) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MMS forced aligner emission runtime requires assets");
    }
    if (config_.window_sec <= 0.0 || config_.context_sec < 0.0 || config_.context_sec >= config_.window_sec) {
        throw std::runtime_error("MMS forced aligner requires 0 <= emission_context_sec < emission_window_sec");
    }
}

void MmsWav2Vec2CtcRuntime::load_encoder() const {
    if (mms_encoder_ != nullptr) {
        return;
    }
    if (assets_->model_weights == nullptr) {
        throw std::runtime_error("MMS forced aligner tensor source must not be null");
    }
    const auto & model = assets_->model_config;
    modules::Wav2Vec2EncoderConfig mms_config;
    mms_config.hidden_size = model.hidden_size;
    mms_config.intermediate_size = model.intermediate_size;
    mms_config.num_hidden_layers = model.num_hidden_layers;
    mms_config.output_hidden_layer = model.num_hidden_layers;
    mms_config.num_attention_heads = model.num_attention_heads;
    mms_config.conv_dim = model.conv_dim;
    mms_config.conv_kernel = model.conv_kernel;
    mms_config.conv_stride = model.conv_stride;
    mms_config.layer_norm_eps = model.layer_norm_eps;
    mms_config.num_conv_pos_embeddings = model.num_conv_pos_embeddings;
    mms_config.num_conv_pos_embedding_groups = model.num_conv_pos_embedding_groups;
    mms_config.final_projection_size = model.vocab_size;
    mms_config.feature_extractor_norm = modules::Wav2Vec2FeatureExtractorNorm::LayerNormEveryLayer;
    mms_config.encoder_layer_norm_order = modules::Wav2Vec2EncoderLayerNormOrder::PreNorm;
    mms_config.apply_encoder_input_layer_norm = false;
    mms_config.apply_final_layer_norm = true;

    modules::Wav2Vec2EncoderWeightBinding mms_binding;
    mms_binding.feature_extractor_layers = "wav2vec2.feature_extractor.conv_layers";
    mms_binding.feature_projection_layer_norm = "wav2vec2.feature_projection.layer_norm";
    mms_binding.feature_projection_projection = "wav2vec2.feature_projection.projection";
    mms_binding.positional_conv = "wav2vec2.encoder.pos_conv_embed.conv";
    mms_binding.encoder_layer_norm = "wav2vec2.encoder.layer_norm";
    mms_binding.encoder_layers = "wav2vec2.encoder.layers";
    mms_binding.final_projection = "lm_head";
    mms_binding.conv_storage_type = weight_storage_type_;
    mms_binding.positional_conv_storage_type = weight_storage_type_;
    mms_binding.projection_storage_type = weight_storage_type_;
    mms_binding.attention_storage_type = weight_storage_type_;
    mms_binding.feed_forward_storage_type = weight_storage_type_;
    mms_binding.final_projection_storage_type = weight_storage_type_;

    mms_encoder_ = std::make_unique<modules::Wav2Vec2EncoderRuntime>(
        modules::Wav2Vec2EncoderRuntime::load_from_tensor_source(
            assets_->model_weights,
            backend_,
            std::move(mms_config),
            std::move(mms_binding)));
}

MmsEmissionOutput MmsWav2Vec2CtcRuntime::compute(const runtime::AudioBuffer & audio) const {
    const auto frontend_start = Clock::now();
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty() ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("MMS forced aligner requires nonempty valid audio input");
    }
    const auto mono_16k = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        kSampleRate16k);
    if (mono_16k.empty()) {
        throw std::runtime_error("MMS forced aligner audio resampled to silence");
    }
    const auto frontend_end = Clock::now();

    const auto encoder_start = Clock::now();
    load_encoder();
    const auto & mms_encoder = *mms_encoder_;

    // Checked seconds-to-samples rounding: disqualify absurd values before
    // llround so a malformed configuration cannot overflow into invalid window
    // sizing or allocation attempts. Anything over an hour is not a usable
    // alignment window (the default is 30 s) and is rejected loudly.
    if (!(config_.window_sec > 0.0 && config_.window_sec <= 3600.0) ||
        !(config_.context_sec >= 0.0 && config_.context_sec <= 3600.0)) {
        throw std::runtime_error(
            "MMS forced aligner emission window/context must be in [0, 3600] seconds");
    }
    const int64_t win_samples = static_cast<int64_t>(
        std::llround(config_.window_sec * static_cast<double>(kSampleRate16k)));
    const int64_t ctx_samples = static_cast<int64_t>(
        std::llround(config_.context_sec * static_cast<double>(kSampleRate16k)));
    if (win_samples % kFrameStrideSamples != 0 || ctx_samples % kFrameStrideSamples != 0) {
        throw std::runtime_error("MMS forced aligner window and context must be multiples of 320 samples");
    }
    const int64_t center_frames = win_samples / static_cast<int64_t>(kFrameStrideSamples);
    const int64_t context_frames = ctx_samples / static_cast<int64_t>(kFrameStrideSamples);

    const int64_t audio_samples = static_cast<int64_t>(mono_16k.size());
    const int64_t vocab_size = assets_->model_config.vocab_size;

    const auto [log_probs, frames] = [&]() -> std::pair<std::vector<float>, int64_t> {
        if (audio_samples < win_samples) {
            // Short audio runs un-windowed, exactly like the reference.
            const auto normalized = mms_normalize_waveform_16k(mono_16k);
            const auto hidden = encode_window(mms_encoder, normalized);
            const int64_t tokens = static_cast<int64_t>(hidden.size()) / vocab_size;
            if (hidden.size() % static_cast<size_t>(vocab_size) != 0) {
                throw std::runtime_error("MMS forced aligner encoder output width mismatch");
            }
            return {mms_log_softmax_and_star(hidden.data(), tokens, vocab_size), tokens};
        }
        const int64_t n_windows = (audio_samples + win_samples - 1) / win_samples;
        const int64_t extension = n_windows * win_samples - audio_samples;
        const int64_t window_samples = win_samples + 2 * ctx_samples;
        const int64_t trimmed_tail_frames = extension / static_cast<int64_t>(kFrameStrideSamples);

        std::vector<float> stitched;
        stitched.reserve(static_cast<size_t>(n_windows * center_frames - trimmed_tail_frames) *
                         static_cast<size_t>(MmsVocabulary::kClassCount));
        std::vector<float> window(static_cast<size_t>(window_samples), 0.0F);
        for (int64_t w = 0; w < n_windows; ++w) {
            std::fill(window.begin(), window.end(), 0.0F);
            const int64_t window_start = w * win_samples;
            const auto copy_start = mono_16k.begin() + std::max<int64_t>(0, window_start - ctx_samples);
            const auto copy_end = mono_16k.begin() + std::min(window_start + win_samples + ctx_samples, audio_samples);
            std::copy(copy_start, copy_end, window.begin() + std::max<int64_t>(0, ctx_samples - window_start));
            const auto normalized = mms_normalize_waveform_16k(window);
            const auto hidden = encode_window(mms_encoder, normalized);
            const int64_t tokens = static_cast<int64_t>(hidden.size()) / vocab_size;
            // The in-tree feature-extractor convolutions use zero padding (vs the
            // reference's kernel//2 padding), so the last window can come back two
            // frames short. Only the center slice is kept and the recorded
            // extension tail is trimmed, so sufficiency of the center is the real
            // invariant (the reference's extra tail frames are dropped by trim).
            if (hidden.size() % static_cast<size_t>(vocab_size) != 0 ||
                tokens < context_frames + center_frames) {
                throw std::runtime_error(
                    "MMS forced aligner window frame count insufficient: got " + std::to_string(tokens) +
                    " frames, need " + std::to_string(context_frames + center_frames) + " for the center slice");
            }
            const auto window_log_probs = mms_log_softmax_and_star(
                hidden.data() + context_frames * vocab_size,
                center_frames,
                vocab_size);
            stitched.insert(
                stitched.end(),
                window_log_probs.begin(),
                window_log_probs.end());
        }
        if (trimmed_tail_frames > 0) {
            stitched.resize(
                stitched.size() - static_cast<size_t>(trimmed_tail_frames) *
                    static_cast<size_t>(MmsVocabulary::kClassCount));
        }
        return {std::move(stitched), n_windows * center_frames - trimmed_tail_frames};
    }();
    const auto encoder_end = Clock::now();

    debug::timing_log_scalar("mms_forced_aligner.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
    debug::timing_log_scalar("mms_forced_aligner.encoder_ms", engine::debug::elapsed_ms(encoder_start, encoder_end));

    MmsEmissionOutput output;
    output.log_probs = std::move(log_probs);
    output.frames = frames;
    output.classes = MmsVocabulary::kClassCount;
    return output;
}

}  // namespace engine::community_models::mms_forced_aligner
