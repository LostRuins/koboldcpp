#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::maya1 {

struct Maya1RopeConfig {
  float factor = 32.0F;
  float low_freq_factor = 1.0F;
  float high_freq_factor = 4.0F;
  int64_t original_max_position_embeddings = 8192;
};

struct Maya1Config {
  int64_t hidden_size = 3072;
  int64_t intermediate_size = 8192;
  int64_t layers = 28;
  int64_t attention_heads = 24;
  int64_t kv_heads = 8;
  int64_t head_dim = 128;
  int64_t vocab_size = 156960;
  int64_t max_position_embeddings = 131072;
  float rms_norm_eps = 1.0e-5F;
  float rope_theta = 500000.0F;
  Maya1RopeConfig rope;
};

struct Maya1GenerationConfig {
  int64_t max_tokens = 2048;
  int64_t min_tokens = 28;
  float temperature = 0.4F;
  float top_p = 0.9F;
  float repetition_penalty = 1.1F;
};

struct Maya1Assets {
  assets::ResourceBundle resources;
  Maya1Config config;
  Maya1GenerationConfig generation;
  std::shared_ptr<const assets::TensorSource> model_weights;
  std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const Maya1Assets>
load_maya1_assets(const std::filesystem::path &model_path);

} // namespace engine::models::maya1
