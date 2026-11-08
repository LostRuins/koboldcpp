#include "engine/community_models/mms_forced_aligner/assets.h"

#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::mms_forced_aligner {

namespace {

using engine::io::json::Value;

std::vector<int64_t> require_i64_array(const Value & value, const char * field) {
    std::vector<int64_t> out;
    const auto & array = value.require(field).as_array();
    out.reserve(array.size());
    for (const auto & element : array) {
        if (!element.is_number()) {
            throw std::runtime_error(std::string("MMS forced aligner config: ") + field + " must be an array of integers");
        }
        out.push_back(element.as_i64());
    }
    return out;
}

int64_t require_i64(const Value & value, const char * field) {
    return value.require(field).as_i64();
}

float require_f32(const Value & value, const char * field) {
    return value.require(field).as_f32();
}

bool require_bool(const Value & value, const char * field) {
    return value.require(field).as_bool();
}

std::string require_string(const Value & value, const char * field) {
    return value.require(field).as_string();
}

MmsWav2Vec2Config parse_model_config(const Value & root) {
    MmsWav2Vec2Config config;
    config.model_type = require_string(root, "model_type");
    const auto & architectures = root.require("architectures").as_array();
    config.architectures.reserve(architectures.size());
    for (const auto & element : architectures) {
        config.architectures.push_back(element.as_string());
    }
    config.hidden_size = require_i64(root, "hidden_size");
    config.intermediate_size = require_i64(root, "intermediate_size");
    config.num_hidden_layers = require_i64(root, "num_hidden_layers");
    config.num_attention_heads = require_i64(root, "num_attention_heads");
    config.conv_dim = require_i64_array(root, "conv_dim");
    config.conv_kernel = require_i64_array(root, "conv_kernel");
    config.conv_stride = require_i64_array(root, "conv_stride");
    config.layer_norm_eps = require_f32(root, "layer_norm_eps");
    config.num_conv_pos_embeddings = require_i64(root, "num_conv_pos_embeddings");
    config.num_conv_pos_embedding_groups = require_i64(root, "num_conv_pos_embedding_groups");
    config.do_stable_layer_norm = require_bool(root, "do_stable_layer_norm");
    config.conv_bias = require_bool(root, "conv_bias");
    config.feat_extract_norm = require_string(root, "feat_extract_norm");
    config.hidden_act = require_string(root, "hidden_act");
    config.vocab_size = require_i64(root, "vocab_size");
    config.pad_token_id = require_i64(root, "pad_token_id");
    return config;
}

MmsPreprocessorConfig parse_preprocessor_config(const assets::ResourceBundle & resources) {
    MmsPreprocessorConfig config;
    const auto * file = resources.find_file("preprocessor_config");
    if (file == nullptr) {
        return config;
    }
    const auto root = resources.parse_json("preprocessor_config");
    if (const auto * value = root.find("sampling_rate")) {
        config.sampling_rate = value->as_i64();
    }
    if (const auto * value = root.find("do_normalize")) {
        config.do_normalize = value->as_bool();
    }
    return config;
}

MmsVocabulary parse_vocabulary(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("vocab");
    MmsVocabulary vocab;
    for (const auto & [token, id_value] : root.as_object()) {
        if (!id_value.is_number()) {
            throw std::runtime_error("MMS forced aligner vocab.json contains a non-numeric id for token '" + token + "'");
        }
        const int32_t id = static_cast<int32_t>(id_value.as_i64());
        vocab.token_to_id.emplace(token, id);
    }
    if (vocab.token_to_id.empty()) {
        throw std::runtime_error("MMS forced aligner vocab.json is empty");
    }
    const int64_t vocab_size = static_cast<int64_t>(vocab.token_to_id.size());
    vocab.id_to_token.assign(static_cast<size_t>(vocab_size), "");
    for (const auto & [token, id] : vocab.token_to_id) {
        if (id < 0 || id >= vocab_size) {
            throw std::runtime_error("MMS forced aligner vocab.json id " + std::to_string(id) +
                                     " for token '" + token + "' is out of range");
        }
        auto & slot = vocab.id_to_token[static_cast<size_t>(id)];
        if (!slot.empty()) {
            throw std::runtime_error("MMS forced aligner vocab.json has duplicate id " + std::to_string(id));
        }
        slot = token;
    }
    for (int32_t id = 0; id < vocab_size; ++id) {
        if (vocab.id_to_token[static_cast<size_t>(id)].empty()) {
            throw std::runtime_error("MMS forced aligner vocab.json is missing id " + std::to_string(id));
        }
    }
    const auto blank = vocab.token_to_id.find("<blank>");
    if (blank == vocab.token_to_id.end()) {
        throw std::runtime_error("MMS forced aligner vocab.json is missing the <blank> token");
    }
    vocab.blank_id = blank->second;
    return vocab;
}

// The exact token->id mapping of the pinned checkpoint revision. The MMS-1130
// alphabet is letter-frequency-ordered (a,i,e,n,o,u,...) rather than
// alphabetical, so a shape-valid but re-sorted vocab.json would silently map
// transcript letters to the wrong CTC classes without this check.
constexpr std::pair<const char *, int32_t> kExpectedVocab[31] = {
    {"<blank>", 0}, {"<pad>", 1}, {"</s>", 2}, {"<unk>", 3},
    {"a", 4}, {"i", 5}, {"e", 6}, {"n", 7}, {"o", 8}, {"u", 9},
    {"t", 10}, {"s", 11}, {"r", 12}, {"m", 13}, {"k", 14}, {"l", 15},
    {"d", 16}, {"g", 17}, {"h", 18}, {"y", 19}, {"b", 20}, {"p", 21},
    {"w", 22}, {"c", 23}, {"v", 24}, {"j", 25}, {"z", 26}, {"f", 27},
    {"'", 28}, {"q", 29}, {"x", 30},
};

void validate_contract(
    const MmsWav2Vec2Config & config,
    const MmsPreprocessorConfig & preprocessor,
    const MmsVocabulary & vocab) {
    if (config.model_type != "wav2vec2") {
        throw std::runtime_error("MMS forced aligner requires model_type=wav2vec2, got '" + config.model_type + "'");
    }
    const bool has_ctc_head = std::find(
        config.architectures.begin(), config.architectures.end(), "Wav2Vec2ForCTC") != config.architectures.end();
    if (!has_ctc_head) {
        throw std::runtime_error("MMS forced aligner requires architectures to contain Wav2Vec2ForCTC");
    }
    if (!config.do_stable_layer_norm) {
        throw std::runtime_error("MMS forced aligner requires do_stable_layer_norm=true");
    }
    if (config.vocab_size != 31) {
        throw std::runtime_error("MMS forced aligner requires vocab_size=31, got " + std::to_string(config.vocab_size));
    }
    if (config.pad_token_id != 0) {
        throw std::runtime_error("MMS forced aligner requires pad_token_id=0, got " + std::to_string(config.pad_token_id));
    }
    if (config.conv_dim.empty() || config.conv_dim.size() != config.conv_kernel.size() ||
        config.conv_dim.size() != config.conv_stride.size()) {
        throw std::runtime_error("MMS forced aligner conv_dim/conv_kernel/conv_stride must be nonempty and equally sized");
    }
    int64_t stride_product = 1;
    for (const int64_t stride : config.conv_stride) {
        if (stride <= 0 || stride > 64) {
            throw std::runtime_error("MMS forced aligner conv strides must be in [1, 64]");
        }
        stride_product *= stride;
    }
    if (stride_product != 320) {
        throw std::runtime_error("MMS forced aligner conv stride product must be 320, got " + std::to_string(stride_product));
    }
    if (preprocessor.sampling_rate != 16000) {
        throw std::runtime_error(
            "MMS forced aligner requires a 16 kHz frontend, got " + std::to_string(preprocessor.sampling_rate));
    }
    if (!preprocessor.do_normalize) {
        throw std::runtime_error("MMS forced aligner requires do_normalize=true");
    }
    if (config.feat_extract_norm != "layer") {
        throw std::runtime_error("MMS forced aligner requires feat_extract_norm=layer, got '" + config.feat_extract_norm + "'");
    }
    if (!config.conv_bias) {
        throw std::runtime_error("MMS forced aligner requires conv_bias=true");
    }
    if (config.hidden_act != "gelu") {
        throw std::runtime_error("MMS forced aligner requires hidden_act=gelu, got '" + config.hidden_act + "'");
    }
    if (config.hidden_size <= 0 || config.num_attention_heads <= 0 ||
        config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("MMS forced aligner hidden_size must be divisible by num_attention_heads");
    }
    if (config.num_conv_pos_embedding_groups <= 0 ||
        config.hidden_size % config.num_conv_pos_embedding_groups != 0) {
        throw std::runtime_error("MMS forced aligner hidden_size must be divisible by num_conv_pos_embedding_groups");
    }
    if (vocab.blank_id != 0 || vocab.token_to_id.at("<blank>") != 0) {
        throw std::runtime_error("MMS forced aligner requires <blank> to be id 0");
    }
    if (static_cast<int64_t>(vocab.token_to_id.size()) != config.vocab_size) {
        throw std::runtime_error("MMS forced aligner vocab.json size must equal vocab_size");
    }
    // Exact pinned token->id mapping (defeats a re-sorted but shape-valid file).
    for (const auto & [token, id] : kExpectedVocab) {
        const auto found = vocab.token_to_id.find(token);
        if (found == vocab.token_to_id.end() || found->second != id) {
            throw std::runtime_error(
                "MMS forced aligner vocab.json must map '" + std::string(token) + "' to id " +
                std::to_string(id) + " at the pinned checkpoint revision");
        }
    }
}

}  // namespace

MmsParsedConfigs parse_mms_forced_aligner_configs(const assets::ResourceBundle & resources) {
    MmsParsedConfigs parsed;
    parsed.model_config = parse_model_config(resources.parse_json("config"));
    parsed.preprocessor_config = parse_preprocessor_config(resources);
    parsed.vocabulary = parse_vocabulary(resources);
    validate_contract(parsed.model_config, parsed.preprocessor_config, parsed.vocabulary);
    return parsed;
}

std::shared_ptr<const MmsForcedAlignerAssets> load_mms_forced_aligner_assets(const std::filesystem::path & model_root) {
    auto assets = std::make_shared<MmsForcedAlignerAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_root,
        engine::model_spec::default_spec_path("mms_forced_aligner"));
    auto parsed = parse_mms_forced_aligner_configs(assets->resources);
    assets->model_config = std::move(parsed.model_config);
    assets->vocabulary = std::move(parsed.vocabulary);
    assets->model_weights = assets->resources.open_tensor_source("weights");
    return assets;
}

}  // namespace engine::community_models::mms_forced_aligner
