#include "engine/models/qwen3_asr/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::qwen3_asr {
namespace json = engine::io::json;
namespace {

Qwen3ASRAudioEncoderConfig parse_audio_encoder_config(const json::Value & value) {
    Qwen3ASRAudioEncoderConfig config;
    config.num_mel_bins = json::require_i64(value, "num_mel_bins");
    config.encoder_layers = json::require_i64(value, "encoder_layers");
    config.encoder_attention_heads = json::require_i64(value, "encoder_attention_heads");
    config.encoder_ffn_dim = json::require_i64(value, "encoder_ffn_dim");
    config.d_model = json::require_i64(value, "d_model");
    // The Transformers checkpoint calls a small RoPE setting
    // max_position_embeddings. The audio tower still uses the same 1500-frame
    // sinusoidal table as the original Qwen checkpoint.
    config.max_source_positions = json::optional_i64(value, "max_source_positions", 1500);
    config.n_window = json::require_i64(value, "n_window");
    config.n_window_infer = json::require_i64(value, "n_window_infer");
    config.conv_chunksize = json::require_i64(value, "conv_chunksize");
    config.downsample_hidden_size = json::require_i64(value, "downsample_hidden_size");
    config.output_dim = json::require_i64(value, "output_dim");
    config.activation_function = json::require_string(value, "activation_function");
    if (config.activation_function != "gelu") {
        throw std::runtime_error("Qwen3 ASR currently supports gelu audio activation");
    }
    return config;
}

Qwen3ASRTextDecoderConfig parse_text_decoder_config(
    const json::Value & thinker_config,
    const json::Value & text_config) {
    Qwen3ASRTextDecoderConfig config;
    config.vocab_size = json::require_i64(text_config, "vocab_size");
    config.output_size = json::optional_i64(thinker_config, "classify_num", config.vocab_size);
    config.hidden_size = json::require_i64(text_config, "hidden_size");
    config.intermediate_size = json::require_i64(text_config, "intermediate_size");
    config.num_hidden_layers = json::require_i64(text_config, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(text_config, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(text_config, "num_key_value_heads");
    config.head_dim = json::optional_i64(text_config, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::require_i64(text_config, "max_position_embeddings");
    config.audio_token_id = json::require_i64(thinker_config, "audio_token_id");
    config.audio_start_token_id = json::optional_i64(thinker_config, "audio_start_token_id", 0);
    config.audio_end_token_id = json::optional_i64(thinker_config, "audio_end_token_id", 0);
    config.pad_token_id = json::optional_i64(thinker_config, "pad_token_id", config.pad_token_id);
    config.pad_token_id = json::optional_i64(text_config, "pad_token_id", config.pad_token_id);
    config.rms_norm_eps = json::optional_f32(text_config, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(text_config, "rope_theta", config.rope_theta);
    const auto * rope_parameters = text_config.find("rope_parameters");
    if (rope_parameters != nullptr && rope_parameters->is_object()) {
        config.rope_theta = json::optional_f32(*rope_parameters, "rope_theta", config.rope_theta);
    }
    const auto * rope_scaling = text_config.find("rope_scaling");
    if (rope_scaling != nullptr && rope_scaling->is_object()) {
        config.mrope_section = json::optional_i64_array_or_scalar(*rope_scaling, "mrope_section", config.mrope_section);
    }
    return config;
}

int64_t require_added_token_id(const assets::ResourceBundle & resources, std::string_view content) {
    const auto tokenizer = resources.parse_json("tokenizer_json");
    for (const auto & item : tokenizer.require("added_tokens").as_array()) {
        const auto * token_content = item.find("content");
        const auto * token_id = item.find("id");
        if (token_content != nullptr && token_content->is_string() &&
            token_id != nullptr && token_id->is_number() && token_content->as_string() == content) {
            return token_id->as_i64();
        }
    }
    throw std::runtime_error("Qwen3 ASR tokenizer.json is missing token: " + std::string(content));
}

void add_supported_languages(Qwen3ASRConfig & config, const json::Value & root) {
    static constexpr const char * kLanguages[] = {
        "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
        "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
        "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
        "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
    };
    config.supported_languages = {"Auto"};
    const auto * languages = root.find("support_languages");
    if (languages != nullptr && languages->is_array()) {
        for (const auto & language : languages->as_array()) {
            config.supported_languages.push_back(language.as_string());
        }
        return;
    }
    for (const char * language : kLanguages) {
        config.supported_languages.emplace_back(language);
    }
}

Qwen3ASRConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto * legacy_thinker = root.find("thinker_config");
    const bool hf_layout = legacy_thinker == nullptr;
    const auto & thinker_config = hf_layout ? root : *legacy_thinker;
    const auto & audio_config = thinker_config.require("audio_config");
    const auto & text_config = thinker_config.require("text_config");

    Qwen3ASRConfig config;
    config.hf_transformers_layout = hf_layout;
    config.audio_encoder_binding = hf_layout
        ? modules::Qwen3AudioEncoderWeightBinding{"model.audio_tower", "model.multi_modal_projector", "linear_1", "linear_2"}
        : modules::Qwen3AudioEncoderWeightBinding{"thinker.audio_tower", "thinker.audio_tower"};
    config.model_type = json::require_string(root, "model_type");
    config.thinker_model_type = json::optional_string(thinker_config, "model_type", config.model_type);
    config.model_size = json::optional_string(root, "model_size", hf_layout ? "Qwen3-ASR-1.7B-hf" : config.model_type);
    config.classify_num = json::optional_i64(thinker_config, "classify_num", 0);
    config.timestamp_token_id = json::optional_i64(root, "timestamp_token_id", 0);
    config.tie_word_embeddings = json::optional_bool(
        root,
        "tie_word_embeddings",
        json::optional_bool(text_config, "tie_word_embeddings", false));
    config.audio_encoder = parse_audio_encoder_config(audio_config);
    config.text_decoder = parse_text_decoder_config(thinker_config, text_config);
    if (hf_layout) {
        config.text_decoder.audio_start_token_id = require_added_token_id(resources, "<|audio_start|>");
        config.text_decoder.audio_end_token_id = require_added_token_id(resources, "<|audio_end|>");
    }

    const auto generation = resources.parse_json("generation_config");
    config.max_new_tokens = json::optional_i64(generation, "max_new_tokens", config.max_new_tokens);
    config.text_decoder.pad_token_id = json::optional_i64(generation, "pad_token_id", config.text_decoder.pad_token_id);
    config.text_decoder.eos_token_ids = json::require_i64_array_or_scalar(generation, "eos_token_id");

    const auto processor = resources.parse_json(resources.has_file("processor_config") ? "processor_config" : "preprocessor_config");
    const auto * feature_extractor = processor.find("feature_extractor");
    const auto & frontend = feature_extractor != nullptr && feature_extractor->is_object() ? *feature_extractor : processor;
    config.frontend.sample_rate = static_cast<int>(json::optional_i64(frontend, "sampling_rate", config.frontend.sample_rate));
    config.frontend.feature_size = json::require_i64(frontend, "feature_size");
    config.frontend.hop_length = json::require_i64(frontend, "hop_length");
    config.frontend.n_fft = json::require_i64(frontend, "n_fft");
    config.timestamp_segment_time_ms = json::optional_i64(processor, "timestamp_segment_time", 0);
    if (config.timestamp_segment_time_ms == 0) {
        config.timestamp_segment_time_ms = json::optional_i64(root, "timestamp_segment_time", 0);
    }
    config.sample_rate = config.frontend.sample_rate;
    if (config.frontend.feature_size != config.audio_encoder.num_mel_bins) {
        throw std::runtime_error("Qwen3 ASR frontend feature size does not match audio encoder config");
    }

    add_supported_languages(config, root);
    return config;
}

assets::ResourceBundle make_resource_bundle(
    const std::filesystem::path & model_path,
    std::string_view package_family) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, package_family);
    if (!resources.has_file("preprocessor_config") && !resources.has_file("processor_config")) {
        throw std::runtime_error("Qwen3 ASR requires preprocessor_config.json or processor_config.json");
    }
    const bool has_legacy_tokenizer = resources.has_file("vocab") && resources.has_file("merges");
    if (!has_legacy_tokenizer && !resources.has_file("tokenizer_json")) {
        throw std::runtime_error("Qwen3 ASR requires vocab.json plus merges.txt, or tokenizer.json");
    }
    return resources;
}

std::shared_ptr<const Qwen3ASRAssets> make_assets(
    assets::ResourceBundle resources) {
    if (!resources.has_file("preprocessor_config") &&
        !resources.has_file("processor_config")) {
        throw std::runtime_error(
            "Qwen3 ASR requires preprocessor_config.json or processor_config.json");
    }
    const bool has_legacy_tokenizer =
        resources.has_file("vocab") && resources.has_file("merges");
    if (!has_legacy_tokenizer && !resources.has_file("tokenizer_json")) {
        throw std::runtime_error(
            "Qwen3 ASR requires vocab.json plus merges.txt, or tokenizer.json");
    }
    Qwen3ASRAssets assets;
    assets.resources = std::move(resources);
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("weights");
    return std::make_shared<Qwen3ASRAssets>(std::move(assets));
}

}  // namespace

std::shared_ptr<const Qwen3ASRAssets> load_qwen3_asr_assets(const std::filesystem::path & model_path) {
    return load_qwen3_asr_assets(model_path, "qwen3_asr");
}

std::shared_ptr<const Qwen3ASRAssets> load_qwen3_asr_assets(
    const std::filesystem::path & model_path,
    std::string_view package_family) {
    return make_assets(make_resource_bundle(model_path, package_family));
}

std::shared_ptr<const Qwen3ASRAssets> load_qwen3_asr_assets(
    assets::ResourceBundle resources) {
    return make_assets(std::move(resources));
}

}  // namespace engine::models::qwen3_asr
