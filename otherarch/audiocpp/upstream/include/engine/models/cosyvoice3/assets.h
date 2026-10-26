#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::cosyvoice3 {

struct CosyVoice3Config {
    int64_t sample_rate = 24000;
    int64_t text_vocab_size = 151936;
    int64_t hidden_size = 896;
    int64_t intermediate_size = 4864;
    int64_t layers = 24;
    int64_t heads = 14;
    int64_t kv_heads = 2;
    int64_t head_dim = 64;
    int64_t speech_token_size = 6561;
    int64_t speech_reserved_tokens = 200;
    int64_t flow_mel_channels = 80;
    int64_t flow_hidden_size = 1024;
    int64_t flow_layers = 22;
    int64_t flow_heads = 16;
    int64_t flow_head_dim = 64;
    int64_t flow_ff_mult = 2;
    int64_t flow_input_channels = 240;
    int64_t flow_static_chunk_size = 50;
    int64_t token_mel_ratio = 2;
    int64_t pre_lookahead_len = 3;
    int64_t speaker_dim = 192;
};

struct CosyVoice3Assets {
    std::filesystem::path model_root;
    std::filesystem::path gguf_path;
    engine::assets::ResourceBundle resources;
    CosyVoice3Config config;
    std::shared_ptr<const engine::assets::TensorSource> llm_weights;
    std::shared_ptr<const engine::assets::TensorSource> flow_weights;
    std::shared_ptr<const engine::assets::TensorSource> hift_weights;
    std::shared_ptr<const engine::assets::TensorSource> campplus_weights;
    std::shared_ptr<const engine::assets::TensorSource> speech_tokenizer_weights;
    std::shared_ptr<const engine::assets::TensorSource> blank_en_weights;
};

std::shared_ptr<const CosyVoice3Assets> load_cosyvoice3_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::cosyvoice3
