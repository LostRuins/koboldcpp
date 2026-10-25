#include "engine/framework/audio/kaldi_fbank.h"

#include "engine/framework/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Framing and LFR structure adapted from handy-computer/transcribe.cpp at
// c87109304c42707867f926fb7b9c378d9f46df8a. SPDX-License-Identifier: MIT;
// Copyright (c) 2026 The transcribe.cpp authors.

namespace engine::audio {
namespace {

int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    if (result > std::numeric_limits<int>::max() / 2) {
      throw std::runtime_error("Kaldi fbank FFT size overflow");
    }
    result *= 2;
  }
  return result;
}

double hz_to_mel(double frequency) {
  return 1127.0 * std::log(1.0 + frequency / 700.0);
}

size_t checked_product(size_t left, size_t right, const char *context) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error(std::string("Kaldi fbank size overflow: ") +
                             context);
  }
  return left * right;
}

void validate_options(const KaldiFbankOptions &options) {
  if (options.sample_rate <= 0) {
    throw std::runtime_error("Kaldi fbank sample_rate must be > 0");
  }
  if (options.num_mels <= 3) {
    throw std::runtime_error("Kaldi fbank num_mels must be > 3");
  }
  if (!std::isfinite(options.frame_length_ms) ||
      !std::isfinite(options.frame_shift_ms) ||
      !(options.frame_length_ms > 0.0F) || !(options.frame_shift_ms > 0.0F)) {
    throw std::runtime_error("Kaldi fbank frame length and shift must be > 0");
  }
  if (options.lfr_m <= 0 || options.lfr_n <= 0) {
    throw std::runtime_error("Kaldi fbank LFR factors must be > 0");
  }
  if (!std::isfinite(options.preemphasis) || options.preemphasis < 0.0F ||
      options.preemphasis > 1.0F) {
    throw std::runtime_error("Kaldi fbank preemphasis must be in [0, 1]");
  }
  const float nyquist = static_cast<float>(options.sample_rate) * 0.5F;
  const float high_frequency = options.high_frequency <= 0.0F
                                   ? nyquist + options.high_frequency
                                   : options.high_frequency;
  if (!std::isfinite(options.low_frequency) ||
      !std::isfinite(options.high_frequency) ||
      !std::isfinite(high_frequency) || options.low_frequency < 0.0F ||
      options.low_frequency >= high_frequency || high_frequency > nyquist) {
    throw std::runtime_error("Kaldi fbank frequency bounds are invalid");
  }
}

std::vector<float> make_hamming_window(int size) {
  std::vector<float> window(static_cast<size_t>(size), 1.0F);
  if (size <= 1) {
    return window;
  }
  const double radians_per_sample =
      2.0 * std::acos(-1.0) / static_cast<double>(size - 1);
  for (int index = 0; index < size; ++index) {
    window[static_cast<size_t>(index)] =
        static_cast<float>(0.54 - 0.46 * std::cos(radians_per_sample *
                                                  static_cast<double>(index)));
  }
  return window;
}

std::vector<float> make_povey_window(int64_t window_size) {
  std::vector<float> window(static_cast<size_t>(window_size), 0.0F);
  constexpr float kPi = 3.14159265358979323846F;
  for (int64_t index = 0; index < window_size; ++index) {
    const float hann =
        0.5F - 0.5F * std::cos(2.0F * kPi * static_cast<float>(index) /
                               static_cast<float>(window_size - 1));
    window[static_cast<size_t>(index)] = std::pow(hann, 0.85F);
  }
  return window;
}

std::vector<float> make_hanning_window(int64_t window_size) {
  std::vector<float> window(static_cast<size_t>(window_size), 0.0F);
  if (window_size <= 1) {
    return window;
  }
  constexpr double kPi = 3.14159265358979323846;
  for (int64_t index = 0; index < window_size; ++index) {
    window[static_cast<size_t>(index)] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(index) /
                             static_cast<double>(window_size - 1)));
  }
  return window;
}

std::vector<float> make_mel_filterbank(int sample_rate, int fft_size,
                                       int num_mels, float low_frequency,
                                       float high_frequency) {
  const int spectrum_bins = fft_size / 2 + 1;
  const int filtered_bins = fft_size / 2;
  const double nyquist = static_cast<double>(sample_rate) * 0.5;
  const double high =
      high_frequency <= 0.0F ? nyquist + high_frequency : high_frequency;
  const double low_mel = hz_to_mel(low_frequency);
  const double high_mel = hz_to_mel(high);
  const double mel_step =
      (high_mel - low_mel) / static_cast<double>(num_mels + 1);
  const double bin_width =
      static_cast<double>(sample_rate) / static_cast<double>(fft_size);

  std::vector<float> filters(checked_product(static_cast<size_t>(num_mels),
                                             static_cast<size_t>(spectrum_bins),
                                             "mel filters"),
                             0.0F);
  for (int mel_bin = 0; mel_bin < num_mels; ++mel_bin) {
    const double left = low_mel + static_cast<double>(mel_bin) * mel_step;
    const double center = left + mel_step;
    const double right = center + mel_step;
    float *row = filters.data() + static_cast<size_t>(mel_bin) * spectrum_bins;
    for (int fft_bin = 0; fft_bin < filtered_bins; ++fft_bin) {
      const double mel = hz_to_mel(static_cast<double>(fft_bin) * bin_width);
      const double rising = (mel - left) / (center - left);
      const double falling = (right - mel) / (right - center);
      row[fft_bin] =
          static_cast<float>(std::max(0.0, std::min(rising, falling)));
    }
  }
  return filters;
}

std::vector<float> make_float_mel_filterbank(
    int64_t sample_rate, int64_t fft_size, int64_t num_mels,
    float low_frequency, float high_frequency) {
  const int64_t filtered_bins = fft_size / 2;
  const float nyquist = 0.5F * static_cast<float>(sample_rate);
  if (high_frequency <= 0.0F) high_frequency += nyquist;
  const float bin_width = static_cast<float>(sample_rate) / static_cast<float>(fft_size);
  const float mel_low = 1127.0F * std::log(1.0F + low_frequency / 700.0F);
  const float mel_high = 1127.0F * std::log(1.0F + high_frequency / 700.0F);
  const float mel_delta = (mel_high - mel_low) / static_cast<float>(num_mels + 1);
  std::vector<float> bank(static_cast<size_t>(num_mels * (filtered_bins + 1)), 0.0F);
  for (int64_t mel_bin = 0; mel_bin < num_mels; ++mel_bin) {
    const float left = mel_low + static_cast<float>(mel_bin) * mel_delta;
    const float center = mel_low + static_cast<float>(mel_bin + 1) * mel_delta;
    const float right = mel_low + static_cast<float>(mel_bin + 2) * mel_delta;
    for (int64_t fft_bin = 0; fft_bin < filtered_bins; ++fft_bin) {
      const float freq = bin_width * static_cast<float>(fft_bin);
      const float mel = 1127.0F * std::log(1.0F + freq / 700.0F);
      const float rising = (mel - left) / std::max(center - left, 1.0e-12F);
      const float falling = (right - mel) / std::max(right - center, 1.0e-12F);
      bank[static_cast<size_t>(mel_bin * (filtered_bins + 1) + fft_bin)] =
          std::max(0.0F, std::min(rising, falling));
    }
  }
  return bank;
}

struct SpeakerMelFilterbankKey {
  int64_t sample_rate = 0;
  int64_t padded_window_size = 0;
  int64_t num_mels = 0;
  float low_frequency = 0.0F;
  float high_frequency = 0.0F;

  bool operator==(const SpeakerMelFilterbankKey &other) const noexcept {
    return sample_rate == other.sample_rate &&
           padded_window_size == other.padded_window_size &&
           num_mels == other.num_mels &&
           low_frequency == other.low_frequency &&
           high_frequency == other.high_frequency;
  }
};

struct SpeakerMelFilterbankKeyHash {
  size_t operator()(const SpeakerMelFilterbankKey &key) const noexcept {
    size_t seed = 0;
    seed ^= std::hash<int64_t>{}(key.sample_rate) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
    seed ^= std::hash<int64_t>{}(key.padded_window_size) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
    seed ^= std::hash<int64_t>{}(key.num_mels) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<float>{}(key.low_frequency) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(key.high_frequency) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
    return seed;
  }
};

} // namespace

const std::vector<float> &cached_kaldi_povey_window(int64_t window_size) {
  if (window_size <= 0) {
    throw std::runtime_error("Kaldi Povey window size must be positive");
  }
  static std::mutex mutex;
  static std::unordered_map<int64_t, std::vector<float>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto it = cache.find(window_size);
  if (it != cache.end()) {
    return it->second;
  }
  return cache.emplace(window_size, make_povey_window(window_size))
      .first->second;
}

struct KaldiMelFilterbankCache::Impl {
  std::mutex mutex;
  std::unordered_map<SpeakerMelFilterbankKey, std::vector<float>,
                     SpeakerMelFilterbankKeyHash>
      values;
};

KaldiMelFilterbankCache::KaldiMelFilterbankCache()
    : impl_(std::make_unique<Impl>()) {}

KaldiMelFilterbankCache::~KaldiMelFilterbankCache() = default;

const std::vector<float> &KaldiMelFilterbankCache::get(
    int64_t sample_rate,
    int64_t padded_window_size,
    int64_t num_mels,
    float low_frequency,
    float high_frequency,
    const std::function<std::vector<float>()> &build_filterbank) {
  if (sample_rate <= 0 || padded_window_size <= 0 || num_mels <= 0) {
    throw std::runtime_error("Kaldi speaker mel filterbank options must be positive");
  }
  const float nyquist = 0.5F * static_cast<float>(sample_rate);
  const float high = high_frequency <= 0.0F ? high_frequency + nyquist
                                            : high_frequency;
  if (!std::isfinite(low_frequency) || !std::isfinite(high_frequency) ||
      !std::isfinite(high) || low_frequency < 0.0F || low_frequency >= high ||
      high > nyquist) {
    throw std::runtime_error("Kaldi speaker mel filterbank frequency bounds are invalid");
  }
  const SpeakerMelFilterbankKey key{
      sample_rate, padded_window_size, num_mels, low_frequency, high_frequency};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->values.find(key);
  if (it != impl_->values.end()) {
    return it->second;
  }
  return impl_->values.emplace(key, build_filterbank()).first->second;
}

const std::vector<float> &cached_kaldi_campplus_mel_filterbank_16k() {
  static const auto bank = make_float_mel_filterbank(16000, 512, 80, 20.0F, 0.0F);
  return bank;
}

KaldiFbankFeatures extract_kaldi_fbank(const std::vector<float> &audio,
                                       const KaldiFbankOptions &options) {
  validate_options(options);
  if (options.num_mels > std::numeric_limits<int>::max() / options.lfr_m) {
    throw std::runtime_error("Kaldi fbank feature dimension overflow");
  }
  const int feature_dim = options.num_mels * options.lfr_m;
  if (options.apply_cmvn &&
      (options.cmvn_shift.size() != static_cast<size_t>(feature_dim) ||
       options.cmvn_scale.size() != static_cast<size_t>(feature_dim))) {
    throw std::runtime_error(
        "Kaldi fbank CMVN vectors must match the LFR feature dimension");
  }

  const double window_size_f64 = static_cast<double>(options.sample_rate) *
                                 static_cast<double>(options.frame_length_ms) /
                                 1000.0;
  const double window_shift_f64 = static_cast<double>(options.sample_rate) *
                                  static_cast<double>(options.frame_shift_ms) /
                                  1000.0;
  if (window_size_f64 > std::numeric_limits<int>::max() ||
      window_shift_f64 > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "Kaldi fbank frame settings exceed the supported sample count");
  }
  const int window_size = static_cast<int>(window_size_f64);
  const int window_shift = static_cast<int>(window_shift_f64);
  if (window_size <= 0 || window_shift <= 0) {
    throw std::runtime_error(
        "Kaldi fbank frame settings round to zero samples");
  }

  KaldiFbankFeatures result;
  result.feature_dim = feature_dim;
  if (audio.size() < static_cast<size_t>(window_size)) {
    return result;
  }

  const size_t mel_frames_size =
      1 + (audio.size() - static_cast<size_t>(window_size)) /
              static_cast<size_t>(window_shift);
  if (mel_frames_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "Kaldi fbank frame count exceeds the supported range");
  }
  const int mel_frames = static_cast<int>(mel_frames_size);
  const int fft_size = next_power_of_two(window_size);
  const int spectrum_bins = fft_size / 2 + 1;
  const float sample_scale = options.upscale_samples ? 32768.0F : 1.0F;
  const auto window = options.window_type == KaldiFbankWindowType::Povey
                          ? cached_kaldi_povey_window(window_size)
                      : options.window_type == KaldiFbankWindowType::Hanning
                          ? make_hanning_window(window_size)
                          : make_hamming_window(window_size);
  const auto filters =
      make_mel_filterbank(options.sample_rate, fft_size, options.num_mels,
                          options.low_frequency, options.high_frequency);
  std::vector<std::vector<std::pair<int, float>>> sparse_filters;
  if (options.sparse_filterbank) {
    sparse_filters.resize(static_cast<size_t>(options.num_mels));
    for (int mel = 0; mel < options.num_mels; ++mel) {
      for (int bin = 0; bin < spectrum_bins; ++bin) {
        const float weight = filters[static_cast<size_t>(mel) * spectrum_bins + bin];
        if (weight != 0.0F) sparse_filters[static_cast<size_t>(mel)].emplace_back(bin, weight);
      }
    }
  }

  std::vector<float> framed(checked_product(static_cast<size_t>(mel_frames),
                                            static_cast<size_t>(fft_size),
                                            "framed audio"),
                            0.0F);
  for (int frame_index = 0; frame_index < mel_frames; ++frame_index) {
    float *frame = framed.data() + static_cast<size_t>(frame_index) * fft_size;
    const size_t sample_offset =
        static_cast<size_t>(frame_index) * window_shift;
    for (int sample = 0; sample < window_size; ++sample) {
      frame[sample] =
          audio[sample_offset + static_cast<size_t>(sample)] * sample_scale;
    }
    if (options.remove_dc_offset) {
      double sum = 0.0;
      for (int sample = 0; sample < window_size; ++sample) {
        sum += frame[sample];
      }
      const float mean =
          static_cast<float>(sum / static_cast<double>(window_size));
      for (int sample = 0; sample < window_size; ++sample) {
        frame[sample] -= mean;
      }
    }
    if (options.preemphasis != 0.0F) {
      for (int sample = window_size - 1; sample >= 1; --sample) {
        frame[sample] -= options.preemphasis * frame[sample - 1];
      }
      frame[0] *= 1.0F - options.preemphasis;
    }
    for (int sample = 0; sample < window_size; ++sample) {
      frame[sample] *= window[static_cast<size_t>(sample)];
    }
  }

  std::vector<std::complex<float>> spectrum(
      checked_product(static_cast<size_t>(mel_frames),
                      static_cast<size_t>(spectrum_bins), "spectrum"));
  real_fft_forward(
      {static_cast<size_t>(mel_frames), static_cast<size_t>(fft_size)},
      {
          static_cast<std::ptrdiff_t>(fft_size) *
              static_cast<std::ptrdiff_t>(sizeof(float)),
          static_cast<std::ptrdiff_t>(sizeof(float)),
      },
      {
          static_cast<std::ptrdiff_t>(spectrum_bins) *
              static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
          static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
      },
      1, framed.data(), spectrum.data());

  std::vector<float> mel_features(
      checked_product(static_cast<size_t>(mel_frames),
                      static_cast<size_t>(options.num_mels), "mel features"),
      0.0F);
  constexpr float kMelEpsilon = std::numeric_limits<float>::epsilon();
  for (int frame_index = 0; frame_index < mel_frames; ++frame_index) {
    const auto *frame_spectrum =
        spectrum.data() + static_cast<size_t>(frame_index) * spectrum_bins;
    float *mel_row = mel_features.data() +
                     static_cast<size_t>(frame_index) * options.num_mels;
    for (int mel_bin = 0; mel_bin < options.num_mels; ++mel_bin) {
      const float *filter =
          filters.data() + static_cast<size_t>(mel_bin) * spectrum_bins;
      float energy = 0.0F;
      if (options.sparse_filterbank) {
        for (const auto & [bin, weight] : sparse_filters[static_cast<size_t>(mel_bin)]) {
          energy += weight * std::norm(frame_spectrum[bin]);
        }
      } else {
        for (int fft_bin = 0; fft_bin < spectrum_bins; ++fft_bin) {
          energy += filter[fft_bin] * std::norm(frame_spectrum[fft_bin]);
        }
      }
      mel_row[mel_bin] = std::log(std::max(energy, kMelEpsilon));
    }
  }

  const int left_pad = (options.lfr_m - 1) / 2;
  result.frames = 1 + (mel_frames - 1) / options.lfr_n;
  result.values.assign(checked_product(static_cast<size_t>(result.frames),
                                       static_cast<size_t>(feature_dim),
                                       "LFR features"),
                       0.0F);
  for (int output_frame = 0; output_frame < result.frames; ++output_frame) {
    float *output =
        result.values.data() + static_cast<size_t>(output_frame) * feature_dim;
    for (int stacked_frame = 0; stacked_frame < options.lfr_m;
         ++stacked_frame) {
      const int64_t padded_index =
          static_cast<int64_t>(output_frame) * options.lfr_n + stacked_frame;
      const int source_frame = static_cast<int>(std::clamp<int64_t>(
          padded_index - left_pad, 0, static_cast<int64_t>(mel_frames - 1)));
      const float *source =
          mel_features.data() +
          static_cast<size_t>(source_frame) * options.num_mels;
      std::memcpy(
          output + static_cast<size_t>(stacked_frame) * options.num_mels,
          source, static_cast<size_t>(options.num_mels) * sizeof(float));
    }
    if (options.apply_cmvn) {
      for (int feature = 0; feature < feature_dim; ++feature) {
        output[feature] = (output[feature] +
                           options.cmvn_shift[static_cast<size_t>(feature)]) *
                          options.cmvn_scale[static_cast<size_t>(feature)];
      }
    }
  }
  return result;
}

} // namespace engine::audio
