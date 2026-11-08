#include "engine/community_models/sense_asr/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/package.h"

#include <gguf.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::sense_asr {
namespace {

struct SenseAsrGgufMetadata {
  std::string architecture;
  int64_t input_size = 560;
  int64_t output_size = 512;
  int64_t attention_heads = 4;
  int64_t num_blocks = 50;
  int64_t tp_blocks = 20;
  int64_t kernel_size = 11;
  int64_t vocab_size = 25055;
  int64_t blank_id = 0;
  std::vector<int32_t> query_tokens = {0, 1, 2, 14};
  std::vector<std::string> vocab;
};

class GgufMetadataReader {
public:
  explicit GgufMetadataReader(const std::filesystem::path &path) {
    gguf_init_params params{};
    params.no_alloc = false;
    params.ctx = nullptr;
    gguf_context *gguf = gguf_init_from_file(path.string().c_str(), params);
    if (gguf == nullptr) {
      throw std::runtime_error(
          "SenseVoice failed to open GGUF metadata at " + path.string());
    }
    ctx_.reset(gguf);
  }

  int64_t kv_u32(const char *key, int64_t fallback) const {
    const int64_t id = gguf_find_key(ctx_.get(), key);
    return id < 0 ? fallback : static_cast<int64_t>(gguf_get_val_u32(ctx_.get(), id));
  }

  std::string kv_str(const char *key, std::string fallback) const {
    const int64_t id = gguf_find_key(ctx_.get(), key);
    if (id < 0) {
      return fallback;
    }
    const char *value = gguf_get_val_str(ctx_.get(), id);
    return value != nullptr ? std::string(value) : fallback;
  }

  std::vector<int32_t> arr_i32(const char *key,
                               std::vector<int32_t> fallback) const {
    const int64_t id = gguf_find_key(ctx_.get(), key);
    if (id < 0) {
      return fallback;
    }
    const size_t count = gguf_get_arr_n(ctx_.get(), id);
    std::vector<int32_t> values(count, 0);
    const void *data = gguf_get_arr_data(ctx_.get(), id);
    for (size_t i = 0; i < count; ++i) {
      values[i] = static_cast<const int32_t *>(data)[i];
    }
    return values;
  }

  std::vector<std::string> arr_str(const char *key) const {
    const int64_t id = gguf_find_key(ctx_.get(), key);
    if (id < 0) {
      return {};
    }
    const size_t count = gguf_get_arr_n(ctx_.get(), id);
    std::vector<std::string> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const char *value = gguf_get_arr_str(ctx_.get(), id, i);
      values.emplace_back(value != nullptr ? value : "");
    }
    return values;
  }

private:
  struct GgufDeleter {
    void operator()(gguf_context *ctx) const noexcept {
      if (ctx != nullptr) {
        gguf_free(ctx);
      }
    }
  };

  std::unique_ptr<gguf_context, GgufDeleter> ctx_;
};

SenseAsrConfig parse_config(const SenseAsrGgufMetadata &meta) {
  SenseAsrConfig config;
  config.model_type = meta.architecture.empty() ? "sensevoice-small"
                                                : meta.architecture;
  config.vocab = meta.vocab;

  config.frontend.sample_rate = 16000;
  config.frontend.num_mels = 80;
  config.frontend.frame_length_ms = 25;
  config.frontend.frame_shift_ms = 10;
  config.frontend.lfr_m = 7;
  config.frontend.lfr_n = 6;
  config.frontend.preemphasis = 0.97F;
  config.frontend.low_frequency = 20.0F;
  config.frontend.high_frequency = 8000.0F;

  config.encoder.input_size = meta.input_size;
  config.encoder.d_model = meta.output_size;
  config.encoder.attention_heads = meta.attention_heads;
  config.encoder.ffn_dim = 4 * meta.output_size;
  config.encoder.num_blocks = meta.num_blocks;
  config.encoder.timestamp_prediction_layers = meta.tp_blocks;
  config.encoder.kernel_size = meta.kernel_size;
  config.encoder.vocab_size = meta.vocab_size;
  config.encoder.blank_id = meta.blank_id;
  config.encoder.query_tokens = meta.query_tokens;

  if (config.encoder.input_size != 560 || config.encoder.d_model != 512 ||
      config.encoder.attention_heads != 4 || config.encoder.ffn_dim != 2048 ||
      config.encoder.num_blocks != 50 ||
      config.encoder.timestamp_prediction_layers != 20 ||
      config.encoder.kernel_size != 11 || config.encoder.vocab_size != 25055 ||
      config.encoder.blank_id != 0) {
    throw std::runtime_error(
        "SenseVoice-Small GGUF config does not match the published "
        "architecture: 560-wide ReLU SAN-M encoder required");
  }
  if (config.encoder.d_model % config.encoder.attention_heads != 0) {
    throw std::runtime_error(
        "SenseVoice-Small encoder width must be divisible by attention heads");
  }
  if (config.encoder.kernel_size % 2 == 0) {
    throw std::runtime_error("SenseVoice-Small FSMN kernel size must be odd");
  }
  if (!config.vocab.empty() &&
      static_cast<int64_t>(config.vocab.size()) != config.encoder.vocab_size) {
    throw std::runtime_error(
        "SenseVoice-Small GGUF vocab size does not match sv.vocab_size");
  }
  return config;
}

std::shared_ptr<const SenseAsrAssets>
make_assets(assets::ResourceBundle resources) {
  SenseAsrAssets assets;
  assets.resources = std::move(resources);
  assets.model_weights = assets.resources.open_tensor_source("weights");
  if (assets.model_weights == nullptr) {
    throw std::runtime_error("SenseVoice-Small is missing its GGUF weights");
  }

  const GgufMetadataReader reader(assets.model_weights->source_path());
  SenseAsrGgufMetadata meta;
  meta.architecture = reader.kv_str("general.architecture", "sensevoice-small");
  meta.input_size = reader.kv_u32("sv.input_size", 560);
  meta.output_size = reader.kv_u32("sv.output_size", 512);
  meta.attention_heads = reader.kv_u32("sv.attention_heads", 4);
  meta.num_blocks = reader.kv_u32("sv.num_blocks", 50);
  meta.tp_blocks = reader.kv_u32("sv.tp_blocks", 20);
  meta.kernel_size = reader.kv_u32("sv.kernel_size", 11);
  meta.vocab_size = reader.kv_u32("sv.vocab_size", 25055);
  meta.blank_id = reader.kv_u32("sv.blank_id", 0);
  meta.query_tokens = reader.arr_i32("sv.query_tokens", {0, 1, 2, 14});
  meta.vocab = reader.arr_str("sv.vocab");
  assets.config = parse_config(meta);

  for (const char *tensor : {
           "embed.weight",
           "encoder.encoders0.0.self_attn.linear_q_k_v.weight",
           "encoder.encoders.0.self_attn.linear_q_k_v.weight",
           "encoder.after_norm.weight",
           "encoder.tp_encoders.0.self_attn.linear_q_k_v.weight",
           "encoder.tp_norm.weight",
           "ctc.ctc_lo.weight",
           "ctc.ctc_lo.bias",
       }) {
    if (!assets.model_weights->has_tensor(tensor)) {
      throw std::runtime_error(
          std::string("SenseVoice-Small is missing required tensor: ") + tensor);
    }
  }
  return std::make_shared<SenseAsrAssets>(std::move(assets));
}

} // namespace

std::shared_ptr<const SenseAsrAssets>
load_sense_asr_assets(const std::filesystem::path &model_path) {
  return make_assets(engine::model_spec::load_resource_bundle_for_family(
      model_path, "sense_asr"));
}

std::shared_ptr<const SenseAsrAssets>
load_sense_asr_assets(assets::ResourceBundle resources) {
  return make_assets(std::move(resources));
}

} // namespace engine::community_models::sense_asr
