#include "engine/models/cosyvoice3/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"
#include "engine/framework/modules/speech_encoders/s3_tokenizer.h"
#include "engine/framework/runtime/cache_slots.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::cosyvoice3 {
namespace {

uint64_t hash_audio(const engine::runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(audio.sample_rate));
    mix(static_cast<uint64_t>(audio.channels));
    mix(static_cast<uint64_t>(audio.samples.size()));
    for (float sample : audio.samples) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(sample));
        std::memcpy(&bits, &sample, sizeof(bits));
        mix(bits);
    }
    return hash;
}

std::vector<float> mono_resampled(const engine::runtime::AudioBuffer & audio, int sample_rate) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("CosyVoice3 requires non-empty reference audio");
    }
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate == sample_rate) {
        return mono;
    }
    engine::audio::TorchaudioSincHannResampleOptions options;
    options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat64;
    return engine::audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, sample_rate, options);
}

engine::runtime::AudioBuffer mono_buffer(const engine::runtime::AudioBuffer & audio, int sample_rate) {
    engine::runtime::AudioBuffer out;
    out.sample_rate = sample_rate;
    out.channels = 1;
    out.samples = mono_resampled(audio, sample_rate);
    return out;
}

int64_t reflect_index(int64_t index, int64_t length) {
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

void compute_prompt_mel(
    const engine::runtime::AudioBuffer & audio,
    std::vector<float> & values,
    int64_t & frames) {
    constexpr int64_t kSampleRate = 24000;
    constexpr int64_t kNfft = 1920;
    constexpr int64_t kHop = 480;
    constexpr int64_t kWin = 1920;
    constexpr int64_t kMels = 80;
    constexpr float kEps = 1.0e-9F;
    constexpr float kLogClamp = 1.0e-5F;
    const auto mono = mono_resampled(audio, kSampleRate);
    if (mono.size() < 2) {
        throw std::runtime_error("CosyVoice3 reference audio is too short for prompt mel");
    }
    const int64_t pad = (kNfft - kHop) / 2;
    const int64_t padded_samples = static_cast<int64_t>(mono.size()) + 2 * pad;
    std::vector<float> padded(static_cast<size_t>(padded_samples), 0.0F);
    for (int64_t index = 0; index < padded_samples; ++index) {
        padded[static_cast<size_t>(index)] =
            mono[static_cast<size_t>(reflect_index(index - pad, static_cast<int64_t>(mono.size())))];
    }
    const engine::audio::STFTConfig stft_config{
        kNfft,
        kHop,
        kWin,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        padded,
        window,
        1,
        padded_samples,
        stft_config);
    frames = magnitude.shape[2];
    const int64_t freq_bins = kNfft / 2 + 1;
    const auto filterbank = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{kSampleRate, kNfft, kMels, 0.0F, 0.0F, true});
    values.assign(static_cast<size_t>(frames * kMels), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for if (frames > 8)
#endif
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t mel = 0; mel < kMels; ++mel) {
            double sum = 0.0;
            for (int64_t bin = 0; bin < freq_bins; ++bin) {
                const float mag = magnitude.values[static_cast<size_t>(bin * frames + frame)];
                const float stabilized = std::sqrt(mag * mag + kEps);
                sum += static_cast<double>(filterbank.values[static_cast<size_t>(mel * freq_bins + bin)]) *
                    static_cast<double>(stabilized);
            }
            values[static_cast<size_t>(frame * kMels + mel)] =
                static_cast<float>(std::log(std::max(sum, static_cast<double>(kLogClamp))));
        }
    }
}

std::vector<float> compute_campplus_fbank(const engine::runtime::AudioBuffer & audio, int64_t & frames) {
    const auto mono = mono_resampled(audio, 16000);
    engine::audio::KaldiFbankOptions options;
    options.sample_rate = 16000;
    options.num_mels = 80;
    options.frame_length_ms = 25.0F;
    options.frame_shift_ms = 10.0F;
    options.window_type = engine::audio::KaldiFbankWindowType::Povey;
    options.lfr_m = 1;
    options.lfr_n = 1;
    options.preemphasis = 0.97F;
    options.low_frequency = 20.0F;
    options.high_frequency = 0.0F;
    options.remove_dc_offset = true;
    options.upscale_samples = false;
    options.apply_cmvn = false;
    auto fbank = engine::audio::extract_kaldi_fbank(mono, options);
    if (fbank.frames <= 0 || fbank.feature_dim != 80) {
        throw std::runtime_error("CosyVoice3 reference audio is too short for CAMPPlus");
    }
    frames = fbank.frames;
    for (int64_t dim = 0; dim < fbank.feature_dim; ++dim) {
        float mean = 0.0F;
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            mean += fbank.values[static_cast<size_t>(frame * fbank.feature_dim + dim)];
        }
        mean /= static_cast<float>(fbank.frames);
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            fbank.values[static_cast<size_t>(frame * fbank.feature_dim + dim)] -= mean;
        }
    }
    return std::move(fbank.values);
}

}  // namespace

class CosyVoice3Frontend::Impl {
public:
    Impl(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        size_t reference_cache_slots)
        : assets_(std::move(assets)),
          reference_cache_(reference_cache_slots) {
        (void) graph_arena_bytes;
        (void) weight_context_bytes;
        if (assets_ == nullptr) {
            throw std::runtime_error("CosyVoice3 frontend requires assets");
        }
        engine::modules::S3TokenizerConfig speech_tokenizer_config;
        speech_tokenizer_config.weight_storage_type = storage_type;
        speech_tokenizer_ = std::make_unique<engine::modules::S3TokenizerComponent>(
            engine::modules::S3TokenizerComponent::load_from_source(
                assets_->speech_tokenizer_weights,
                execution,
                speech_tokenizer_config));
        engine::modules::CampplusEncoderConfig campplus_config;
        campplus_config.feat_dim = 80;
        campplus_config.embedding_size = assets_->config.speaker_dim;
        campplus_config.weight_storage_type = storage_type;
        campplus_config.weight_layout = engine::modules::CampplusEncoderWeightLayout::Fused;
        campplus_config.normalize_partial_segment_by_full_length = true;
        campplus_ = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
            assets_->campplus_weights,
            execution.config(),
            campplus_config);
    }

    const CosyVoice3ReferenceFeatures & prepare_reference(const engine::runtime::AudioBuffer & audio) {
        const auto key = hash_audio(audio);
        const auto * cached = reference_cache_.find(key);
        if (cached != nullptr) {
            return *cached;
        }

        const auto started = std::chrono::steady_clock::now();
        CosyVoice3ReferenceFeatures features;
        auto audio16 = mono_buffer(audio, 16000);
        auto token_output = speech_tokenizer_->tokenize(audio16, std::nullopt);
        features.speech_tokens = std::move(token_output.tokens);
        features.speech_token_count = token_output.token_count;
        compute_prompt_mel(audio, features.prompt_mel, features.prompt_mel_frames);
        int64_t fbank_frames = 0;
        auto fbank = compute_campplus_fbank(audio, fbank_frames);
        auto speaker = campplus_.embed_from_features(fbank, fbank_frames, 80);
        features.speaker_embedding = std::move(speaker.embedding);
        if (static_cast<int64_t>(features.speaker_embedding.size()) != assets_->config.speaker_dim) {
            throw std::runtime_error("CosyVoice3 CAMPPlus speaker embedding size mismatch");
        }
        engine::debug::timing_log_scalar("cosyvoice3.frontend.reference_ms", engine::debug::elapsed_ms(started));

        if (reference_cache_.capacity() == 0) {
            uncached_ = std::move(features);
            return uncached_;
        }
        reference_cache_.put(key, std::move(features));
        const auto * inserted = reference_cache_.find(key);
        if (inserted == nullptr) {
            throw std::runtime_error("CosyVoice3 reference cache insert failed");
        }
        return *inserted;
    }

    void release_graphs() {
        if (speech_tokenizer_ != nullptr) {
            speech_tokenizer_->release_runtime_cache();
        }
        campplus_.release_runtime_graph();
    }

private:
    std::shared_ptr<const CosyVoice3Assets> assets_;
    std::unique_ptr<engine::modules::S3TokenizerComponent> speech_tokenizer_;
    engine::modules::CampplusEncoderComponent campplus_;
    engine::runtime::CacheSlots<uint64_t, CosyVoice3ReferenceFeatures> reference_cache_;
    CosyVoice3ReferenceFeatures uncached_;
};

CosyVoice3Frontend::CosyVoice3Frontend(
    std::shared_ptr<const CosyVoice3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    size_t reference_cache_slots)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          reference_cache_slots)) {}

CosyVoice3Frontend::~CosyVoice3Frontend() = default;

const CosyVoice3ReferenceFeatures & CosyVoice3Frontend::prepare_reference(const engine::runtime::AudioBuffer & audio) {
    return impl_->prepare_reference(audio);
}

void CosyVoice3Frontend::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::cosyvoice3
