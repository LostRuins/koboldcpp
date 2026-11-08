#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::mms_forced_aligner {

// Parsed wav2vec2 model config from config.json. All fields are validated
// against the upstream MMS-300M forced-aligner contract on load.
struct MmsWav2Vec2Config {
    std::string model_type;
    std::vector<std::string> architectures;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    std::vector<int64_t> conv_dim;
    std::vector<int64_t> conv_kernel;
    std::vector<int64_t> conv_stride;
    float layer_norm_eps = 1.0e-5F;
    int64_t num_conv_pos_embeddings = 0;
    int64_t num_conv_pos_embedding_groups = 0;
    bool do_stable_layer_norm = false;
    bool conv_bias = true;
    std::string feat_extract_norm;
    std::string hidden_act;
    int64_t vocab_size = 0;
    int64_t pad_token_id = 0;
};

// Preprocessor defaults match Wav2Vec2FeatureExtractor when the optional
// preprocessor_config.json sidecar is absent (as in the pinned revision).
struct MmsPreprocessorConfig {
    int64_t sampling_rate = 16000;
    bool do_normalize = true;
};

struct MmsVocabulary {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string> id_to_token;
    int32_t blank_id = 0;

    // Virtual <star> class appended after the real classes as an extra
    // log-probability column at inference time; never present in vocab.json.
    static constexpr int32_t kStarId = 31;
    static constexpr int32_t kClassCount = kStarId + 1;
};

struct MmsForcedAlignerAssets {
    assets::ResourceBundle resources;
    MmsWav2Vec2Config model_config;
    MmsVocabulary vocabulary;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

// Pure config/vocabulary parsing and contract validation; unit-testable with
// temporary JSON files (no checkpoint required).
struct MmsParsedConfigs {
    MmsWav2Vec2Config model_config;
    MmsPreprocessorConfig preprocessor_config;
    MmsVocabulary vocabulary;
};

MmsParsedConfigs parse_mms_forced_aligner_configs(const assets::ResourceBundle & resources);

std::shared_ptr<const MmsForcedAlignerAssets> load_mms_forced_aligner_assets(const std::filesystem::path & model_root);

}  // namespace engine::community_models::mms_forced_aligner
