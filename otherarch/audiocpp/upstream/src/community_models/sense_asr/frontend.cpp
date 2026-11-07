#include "engine/community_models/sense_asr/frontend.h"

#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::sense_asr {

namespace {

std::vector<float> to_16k_mono(const std::vector<float> &audio,
                               int sample_rate) {
  if (sample_rate <= 0) {
    throw std::runtime_error("SenseVoice frontend requires a positive sample rate");
  }
  if (sample_rate == 16000) {
    return audio;
  }
  return engine::audio::resample_mono_linear(audio, sample_rate, 16000);
}

void dump_fbank(const SenseAsrAudioFeatures &features, int frames_to_dump) {
  if (std::getenv("SENSE_ASR_DUMP_FBANK") == nullptr) {
    return;
  }
  int frames = std::min<int>(features.frames,
                             frames_to_dump > 0 ? frames_to_dump : features.frames);
  for (int i = 0; i < frames; ++i) {
    fprintf(stderr, "fb %d: ", i);
    for (int j = 0; j < features.feature_dim; ++j) {
      fprintf(stderr, "%.4f ",
              features.values[static_cast<size_t>(i) * features.feature_dim + j]);
    }
    fprintf(stderr, "\n");
  }
}

} // namespace

SenseAsrFrontend::SenseAsrFrontend(SenseAsrFrontendConfig config)
    : config_(std::move(config)) {
  if (config_.sample_rate <= 0 || config_.num_mels <= 0 ||
      config_.frame_length_ms <= 0 || config_.frame_shift_ms <= 0 ||
      config_.lfr_m <= 0 || config_.lfr_n <= 0) {
    throw std::runtime_error("SenseVoice frontend dimensions must be positive");
  }
}

SenseAsrAudioFeatures
SenseAsrFrontend::extract(const std::vector<float> &audio,
                          int sample_rate) const {
  const auto mono = to_16k_mono(audio, sample_rate);
  engine::audio::KaldiFbankOptions options;
  options.sample_rate = config_.sample_rate;
  options.num_mels = static_cast<int>(config_.num_mels);
  options.frame_length_ms = static_cast<float>(config_.frame_length_ms);
  options.frame_shift_ms = static_cast<float>(config_.frame_shift_ms);
  options.lfr_m = static_cast<int>(config_.lfr_m);
  options.lfr_n = static_cast<int>(config_.lfr_n);
  options.preemphasis = config_.preemphasis;
  options.low_frequency = config_.low_frequency;
  options.high_frequency = config_.high_frequency;
  options.remove_dc_offset = true;
  options.upscale_samples = true;
  options.apply_cmvn = false;

  const auto features = engine::audio::extract_kaldi_fbank(mono, options);
  SenseAsrAudioFeatures result;
  result.values = std::move(features.values);
  result.frames = features.frames;
  result.feature_dim = features.feature_dim;
  dump_fbank(result, 10);
  return result;
}

} // namespace engine::community_models::sense_asr
