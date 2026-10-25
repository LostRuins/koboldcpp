#include "engine/models/chatterbox/components.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace engine::models::chatterbox {

namespace {

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

std::vector<float> l2_normalize(const std::vector<float> & values) {
    float norm_sq = 0.0f;
    for (float value : values) {
        norm_sq += value * value;
    }
    const float norm = std::sqrt(std::max(norm_sq, 1.0e-12f));
    std::vector<float> out(values.size(), 0.0f);
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = values[i] / norm;
    }
    return out;
}

std::vector<float> resample_voice_encoder_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ClampToExpected;
    options.output_padding = 256;
    options.warning_context = "Chatterbox voice encoder";
    options.fallback_description = "linear resampling";
    return engine::audio::resample_mono_soxr_or_linear(input, input_sample_rate, output_sample_rate, options);
}

int64_t get_frame_step(const VoiceEncoderConfig & config) {
    const int64_t frame_step = static_cast<int64_t>(
        std::llround((static_cast<double>(config.sample_rate) / static_cast<double>(config.partial_rate)) /
                     static_cast<double>(config.partial_frames)));
    if (frame_step <= 0 || frame_step > config.partial_frames) {
        throw std::runtime_error("invalid voice encoder frame_step");
    }
    return frame_step;
}

std::pair<int64_t, int64_t> get_num_wins(int64_t n_frames, int64_t step, const VoiceEncoderConfig & config) {
    if (n_frames <= 0) {
        throw std::runtime_error("voice encoder requires positive frame count");
    }
    const int64_t win_size = config.partial_frames;
    const int64_t numerator = std::max<int64_t>(n_frames - win_size + step, 0);
    int64_t n_wins = numerator / step;
    const int64_t remainder = numerator % step;
    const float coverage =
        static_cast<float>(remainder + (win_size - step)) / static_cast<float>(win_size);
    if (n_wins == 0 || coverage >= config.min_coverage) {
        ++n_wins;
    }
    const int64_t target_n = win_size + step * (n_wins - 1);
    return {n_wins, target_n};
}

std::vector<float> trim_trailing_partial_padding(
    const std::vector<float> & mel_frames,
    int64_t frames,
    int64_t num_mels,
    int64_t target_frames) {
    std::vector<float> out(static_cast<size_t>(target_frames * num_mels), 0.0f);
    const int64_t copy_frames = std::min(frames, target_frames);
    std::copy(
        mel_frames.begin(),
        mel_frames.begin() + static_cast<ptrdiff_t>(copy_frames * num_mels),
        out.begin());
    return out;
}

std::vector<float> trim_voice_encoder_audio_librosa(
    const std::vector<float> & audio,
    float top_db) {
    if (audio.empty()) {
        return audio;
    }

    constexpr int64_t frame_length = 2048;
    constexpr int64_t hop_length = 512;
    const int64_t pad = frame_length / 2;
    std::vector<float> padded(static_cast<size_t>(audio.size() + 2 * pad), 0.0f);
    std::copy(audio.begin(), audio.end(), padded.begin() + pad);

    const int64_t frames = 1 + (static_cast<int64_t>(padded.size()) - frame_length) / hop_length;
    std::vector<float> rms(static_cast<size_t>(frames), 0.0f);
    float ref = 0.0f;
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t start = frame * hop_length;
        double power = 0.0;
        for (int64_t i = 0; i < frame_length; ++i) {
            const double sample = static_cast<double>(padded[static_cast<size_t>(start + i)]);
            power += sample * sample;
        }
        const float value = static_cast<float>(std::sqrt(power / static_cast<double>(frame_length)));
        rms[static_cast<size_t>(frame)] = value;
        ref = std::max(ref, value);
    }
    if (ref <= 0.0f) {
        return {};
    }

    int64_t first = -1;
    int64_t last = -1;
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float value = rms[static_cast<size_t>(frame)];
        const float db = 20.0f * std::log10(std::max(value, 1.0e-10f) / ref);
        if (db > -top_db) {
            if (first < 0) {
                first = frame;
            }
            last = frame;
        }
    }
    if (first < 0) {
        return {};
    }

    const int64_t start = first * hop_length;
    const int64_t end = std::min<int64_t>(static_cast<int64_t>(audio.size()), (last + 1) * hop_length);
    if (end <= start) {
        return {};
    }
    return std::vector<float>(
        audio.begin() + static_cast<ptrdiff_t>(start),
        audio.begin() + static_cast<ptrdiff_t>(end));
}

}  // namespace

std::vector<float> compute_voice_encoder_mel(
    const std::vector<float> & waveform,
    const VoiceEncoderConfig & config) {
    engine::audio::MelSpectrogramFrontendConfig mel_config;
    mel_config.sample_rate = config.sample_rate;
    mel_config.n_fft = config.n_fft;
    mel_config.hop_length = config.hop_size;
    mel_config.win_length = config.win_size;
    mel_config.n_mels = config.num_mels;
    mel_config.mel_fmax = static_cast<float>(config.sample_rate) * 0.5f;
    mel_config.window = engine::audio::MelHannWindow::Periodic;
    mel_config.stft_center = true;
    mel_config.waveform_padding = engine::audio::MelWaveformPadding::None;
    mel_config.spectrum_mode = engine::audio::MelSpectrumMode::PowerBeforeProjection;
    mel_config.value_transform = engine::audio::MelValueTransform::None;
    mel_config.layout = engine::audio::MelOutputLayout::TimeMajor;
    return engine::audio::get_cached_mel_spectrogram_frontend(mel_config)
        ->extract_mono(waveform).values;
}

namespace {

std::vector<float> run_lstm_layer(
    const VoiceEncoderLayerWeights & weights,
    int64_t input_size,
    int64_t hidden_size,
    const std::vector<float> & sequence,
    int64_t frames,
    std::vector<float> & final_hidden) {
    std::vector<float> hidden(static_cast<size_t>(hidden_size), 0.0f);
    std::vector<float> cell(static_cast<size_t>(hidden_size), 0.0f);
    std::vector<float> output(static_cast<size_t>(frames * hidden_size), 0.0f);
    std::vector<float> gates(static_cast<size_t>(4 * hidden_size), 0.0f);

    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * x = sequence.data() + static_cast<ptrdiff_t>(frame * input_size);
#ifdef _OPENMP
#pragma omp parallel for if (hidden_size >= 128)
#endif
        for (int64_t gate = 0; gate < 4 * hidden_size; ++gate) {
            double value = static_cast<double>(weights.bias_ih[static_cast<size_t>(gate)]) +
                static_cast<double>(weights.bias_hh[static_cast<size_t>(gate)]);
            const float * w_ih = weights.weight_ih.data() + static_cast<ptrdiff_t>(gate * input_size);
            const float * w_hh = weights.weight_hh.data() + static_cast<ptrdiff_t>(gate * hidden_size);
            for (int64_t i = 0; i < input_size; ++i) {
                value += static_cast<double>(w_ih[i]) * static_cast<double>(x[i]);
            }
            for (int64_t i = 0; i < hidden_size; ++i) {
                value += static_cast<double>(w_hh[i]) * static_cast<double>(hidden[static_cast<size_t>(i)]);
            }
            gates[static_cast<size_t>(gate)] = static_cast<float>(value);
        }

        for (int64_t i = 0; i < hidden_size; ++i) {
            const float in_gate = sigmoid(gates[static_cast<size_t>(i)]);
            const float forget_gate = sigmoid(gates[static_cast<size_t>(hidden_size + i)]);
            const float cell_gate = std::tanh(gates[static_cast<size_t>(2 * hidden_size + i)]);
            const float out_gate = sigmoid(gates[static_cast<size_t>(3 * hidden_size + i)]);
            cell[static_cast<size_t>(i)] = forget_gate * cell[static_cast<size_t>(i)] + in_gate * cell_gate;
            hidden[static_cast<size_t>(i)] = out_gate * std::tanh(cell[static_cast<size_t>(i)]);
            output[static_cast<size_t>(frame * hidden_size + i)] = hidden[static_cast<size_t>(i)];
        }
    }

    final_hidden = hidden;
    return output;
}

std::vector<float> project_embedding(const VoiceEncoderWeights & weights, const std::vector<float> & hidden) {
    const int64_t out_features = weights.config.speaker_embed_size;
    const int64_t in_features = weights.config.hidden_size;
    std::vector<float> raw(static_cast<size_t>(out_features), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for if (out_features >= 64)
#endif
    for (int64_t out = 0; out < out_features; ++out) {
        double value = static_cast<double>(weights.proj_bias[static_cast<size_t>(out)]);
        const float * row = weights.proj_weight.data() + static_cast<ptrdiff_t>(out * in_features);
        for (int64_t in = 0; in < in_features; ++in) {
            value += static_cast<double>(row[in]) * static_cast<double>(hidden[static_cast<size_t>(in)]);
        }
        if (weights.config.final_relu && value < 0.0) {
            value = 0.0;
        }
        raw[static_cast<size_t>(out)] = static_cast<float>(value);
    }
    return l2_normalize(raw);
}

}  // namespace

std::shared_ptr<const VoiceEncoderWeights> load_voice_encoder_weights(const engine::assets::TensorSource & source) {
    auto weights = std::make_shared<VoiceEncoderWeights>();
    weights->lstm_layers.resize(3);
    weights->lstm_layers[0].weight_ih = source.require_f32("lstm.weight_ih_l0", {1024, 40});
    weights->lstm_layers[0].weight_hh = source.require_f32("lstm.weight_hh_l0", {1024, 256});
    weights->lstm_layers[0].bias_ih = source.require_f32("lstm.bias_ih_l0", {1024});
    weights->lstm_layers[0].bias_hh = source.require_f32("lstm.bias_hh_l0", {1024});
    weights->lstm_layers[1].weight_ih = source.require_f32("lstm.weight_ih_l1", {1024, 256});
    weights->lstm_layers[1].weight_hh = source.require_f32("lstm.weight_hh_l1", {1024, 256});
    weights->lstm_layers[1].bias_ih = source.require_f32("lstm.bias_ih_l1", {1024});
    weights->lstm_layers[1].bias_hh = source.require_f32("lstm.bias_hh_l1", {1024});
    weights->lstm_layers[2].weight_ih = source.require_f32("lstm.weight_ih_l2", {1024, 256});
    weights->lstm_layers[2].weight_hh = source.require_f32("lstm.weight_hh_l2", {1024, 256});
    weights->lstm_layers[2].bias_ih = source.require_f32("lstm.bias_ih_l2", {1024});
    weights->lstm_layers[2].bias_hh = source.require_f32("lstm.bias_hh_l2", {1024});
    weights->proj_weight = source.require_f32("proj.weight", {256, 256});
    weights->proj_bias = source.require_f32("proj.bias", {256});
    weights->similarity_weight = source.require_f32("similarity_weight", {1});
    weights->similarity_bias = source.require_f32("similarity_bias", {1});
    return weights;
}

VoiceEncoderComponent VoiceEncoderComponent::load_from_source(
    const engine::assets::TensorSource & source,
    engine::core::BackendConfig backend) {
    return VoiceEncoderComponent(load_voice_encoder_weights(source), backend);
}

VoiceEncoderComponent::VoiceEncoderComponent(
    std::shared_ptr<const VoiceEncoderWeights> weights,
    engine::core::BackendConfig backend)
    : weights_(std::move(weights)),
      backend_(backend) {
    if (!weights_) {
        throw std::runtime_error("VoiceEncoderComponent requires weights");
    }
    if (weights_->lstm_layers.size() != 3) {
        throw std::runtime_error("VoiceEncoderComponent expects 3 LSTM layers");
    }
}

const engine::core::BackendConfig & VoiceEncoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const VoiceEncoderWeights> & VoiceEncoderComponent::weights() const noexcept {
    return weights_;
}

const VoiceEncoderConfig & VoiceEncoderComponent::config() const noexcept {
    return weights_->config;
}

std::vector<float> VoiceEncoderComponent::embed_utterance_from_audio(
    const runtime::AudioBuffer & audio) const {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("VoiceEncoderComponent requires non-empty audio");
    }
    std::vector<float> mono = audio.channels == 1
        ? audio.samples
        : engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != weights_->config.sample_rate) {
        mono = resample_voice_encoder_mono(
            mono,
            audio.sample_rate,
            static_cast<int>(weights_->config.sample_rate));
    }
    mono = trim_voice_encoder_audio_librosa(mono, 20.0f);
    if (mono.empty()) {
        return std::vector<float>(static_cast<size_t>(weights_->config.speaker_embed_size), 0.0f);
    }
    const std::vector<float> mel_frames = compute_voice_encoder_mel(mono, weights_->config);
    const int64_t frames = static_cast<int64_t>(mel_frames.size()) / weights_->config.num_mels;
    const int64_t frame_step = get_frame_step(weights_->config);
    const auto [n_partials, target_frames] = get_num_wins(frames, frame_step, weights_->config);
    const auto padded = trim_trailing_partial_padding(
        mel_frames,
        frames,
        weights_->config.num_mels,
        target_frames);
    std::vector<float> utterance_embeds;
    utterance_embeds.reserve(static_cast<size_t>(n_partials * weights_->config.speaker_embed_size));
    std::vector<float> partials(
        static_cast<size_t>(n_partials * weights_->config.partial_frames * weights_->config.num_mels),
        0.0f);
    for (int64_t partial_index = 0; partial_index < n_partials; ++partial_index) {
        const int64_t start = partial_index * frame_step;
        for (int64_t frame = 0; frame < weights_->config.partial_frames; ++frame) {
            const float * src = padded.data() + static_cast<ptrdiff_t>((start + frame) * weights_->config.num_mels);
            float * dst = partials.data() + static_cast<ptrdiff_t>(((partial_index * weights_->config.partial_frames) + frame) * weights_->config.num_mels);
            std::copy(src, src + weights_->config.num_mels, dst);
        }
    }
    for (int64_t partial_index = 0; partial_index < n_partials; ++partial_index) {
        const float * partial_ptr = partials.data() + static_cast<ptrdiff_t>(
            partial_index * weights_->config.partial_frames * weights_->config.num_mels);
        std::vector<float> partial(
            partial_ptr,
            partial_ptr + static_cast<ptrdiff_t>(weights_->config.partial_frames * weights_->config.num_mels));
        std::vector<float> hidden;
        auto layer_out = run_lstm_layer(
            weights_->lstm_layers[0],
            weights_->config.num_mels,
            weights_->config.hidden_size,
            partial,
            weights_->config.partial_frames,
            hidden);
        layer_out = run_lstm_layer(
            weights_->lstm_layers[1],
            weights_->config.hidden_size,
            weights_->config.hidden_size,
            layer_out,
            weights_->config.partial_frames,
            hidden);
        layer_out = run_lstm_layer(
            weights_->lstm_layers[2],
            weights_->config.hidden_size,
            weights_->config.hidden_size,
            layer_out,
            weights_->config.partial_frames,
            hidden);
        const auto embed = project_embedding(*weights_, hidden);
        utterance_embeds.insert(utterance_embeds.end(), embed.begin(), embed.end());
    }
    std::vector<float> mean_embed(static_cast<size_t>(weights_->config.speaker_embed_size), 0.0f);
    for (int64_t partial_index = 0; partial_index < n_partials; ++partial_index) {
        const float * embed = utterance_embeds.data() + static_cast<ptrdiff_t>(partial_index * weights_->config.speaker_embed_size);
        for (int64_t i = 0; i < weights_->config.speaker_embed_size; ++i) {
            mean_embed[static_cast<size_t>(i)] += embed[i];
        }
    }
    for (float & value : mean_embed) {
        value /= static_cast<float>(n_partials);
    }
    return l2_normalize(mean_embed);
}

}  // namespace engine::models::chatterbox
