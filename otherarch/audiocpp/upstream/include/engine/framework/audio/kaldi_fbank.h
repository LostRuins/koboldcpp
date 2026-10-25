#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace engine::audio {

enum class KaldiFbankWindowType {
  Hamming,
  Povey,
  Hanning,
};

struct KaldiFbankOptions {
  int sample_rate = 16000;
  int num_mels = 80;
  float frame_length_ms = 25.0F;
  float frame_shift_ms = 10.0F;
  KaldiFbankWindowType window_type = KaldiFbankWindowType::Hamming;
  int lfr_m = 7;
  int lfr_n = 6;
  float preemphasis = 0.97F;
  float low_frequency = 20.0F;
  float high_frequency = 0.0F;
  bool remove_dc_offset = true;
  bool upscale_samples = false;
  bool apply_cmvn = false;
  std::vector<float> cmvn_shift;
  std::vector<float> cmvn_scale;
  bool sparse_filterbank = false;
};

struct KaldiFbankFeatures {
  int frames = 0;
  int feature_dim = 0;
  std::vector<float> values;
};

const std::vector<float> &cached_kaldi_povey_window(int64_t window_size);

// Float triangle construction used by 16 kHz CAMPPlus speaker frontends.
// extract_kaldi_fbank uses a numerically different double-precision bank.
const std::vector<float> &cached_kaldi_campplus_mel_filterbank_16k();

class KaldiMelFilterbankCache {
public:
  KaldiMelFilterbankCache();
  ~KaldiMelFilterbankCache();

  KaldiMelFilterbankCache(const KaldiMelFilterbankCache &) = delete;
  KaldiMelFilterbankCache &operator=(const KaldiMelFilterbankCache &) = delete;

  const std::vector<float> &get(
      int64_t sample_rate,
      int64_t padded_window_size,
      int64_t num_mels,
      float low_frequency,
      float high_frequency,
      const std::function<std::vector<float>()> &build_filterbank);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Extracts snip-edges Kaldi log-mel features followed by centered LFR stacking.
KaldiFbankFeatures extract_kaldi_fbank(const std::vector<float> &audio,
                                       const KaldiFbankOptions &options = {});

} // namespace engine::audio
