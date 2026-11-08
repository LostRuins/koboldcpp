#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::community_models::sense_asr {

struct SenseAsrFrontendConfig {
  int sample_rate = 16000;
  int64_t num_mels = 80;
  int64_t frame_length_ms = 25;
  int64_t frame_shift_ms = 10;
  int64_t lfr_m = 7;
  int64_t lfr_n = 6;
  float preemphasis = 0.97F;
  float low_frequency = 20.0F;
  float high_frequency = 8000.0F;
};

struct SenseAsrEncoderConfig {
  int64_t input_size = 560;
  int64_t d_model = 512;
  int64_t attention_heads = 4;
  int64_t ffn_dim = 2048;
  int64_t num_blocks = 50;
  int64_t timestamp_prediction_layers = 20;
  int64_t kernel_size = 11;
  int64_t vocab_size = 25055;
  int64_t blank_id = 0;
  int64_t max_frames = 2048;
  std::vector<int32_t> query_tokens = {0, 1, 2, 14};
};

struct SenseAsrConfig {
  std::string model_type = "sensevoice-small";
  std::vector<std::string> vocab;
  SenseAsrFrontendConfig frontend;
  SenseAsrEncoderConfig encoder;
};

struct SenseAsrAssets {
  assets::ResourceBundle resources;
  SenseAsrConfig config;
  std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const SenseAsrAssets>
load_sense_asr_assets(const std::filesystem::path &model_path);
std::shared_ptr<const SenseAsrAssets>
load_sense_asr_assets(assets::ResourceBundle resources);

} // namespace engine::community_models::sense_asr
