#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSModelConfig {
    float default_temperature = 0.7F;
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    float mimi_frame_rate = 12.5F;
    int64_t flow_layers = 6;
    int64_t flow_depth = 6;
    int64_t flow_dim = 1024;
    int64_t flow_heads = 16;
    int64_t flow_hidden_size = 512;
    int64_t flow_intermediate_size = 4096;
    int64_t latent_dim = 32;
    int64_t mimi_layers = 2;
    int64_t mimi_dim = 512;
    int64_t mimi_heads = 8;
    int64_t mimi_intermediate_size = 2048;
    int64_t mimi_inner_dim = 32;
    int64_t mimi_outer_dim = 512;
    int64_t mimi_context = 250;
    int64_t mimi_seanet_dimension = 512;
    int64_t mimi_base_filters = 64;
    int64_t mimi_encoder_upsample_stride = 16;
    bool pad_with_spaces_for_short_inputs = false;
    bool remove_semicolons = false;
    bool insert_bos_before_voice = false;
    std::optional<int> model_recommended_frames_after_eos = std::nullopt;
};

struct PocketTTSHostWeights {
    assets::TensorDataF32 conditioner_embedding_table;
    std::vector<float> bos_emb;
    std::optional<assets::TensorDataF32> bos_before_voice;
    std::vector<float> emb_mean;
    std::vector<float> emb_std;
};

struct PocketTTSAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::filesystem::path voice_asset_root;
    PocketTTSHostWeights host_weights;
    PocketTTSModelConfig model_config;
    std::vector<tokenizers::SentencePiecePiece> tokenizer_pieces;
    std::string language;
};

std::shared_ptr<const PocketTTSAssets> load_pocket_tts_assets(
    const std::filesystem::path & model_path,
    std::string language);

}  // namespace engine::models::pocket_tts
