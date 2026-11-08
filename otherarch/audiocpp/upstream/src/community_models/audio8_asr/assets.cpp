#include "engine/community_models/audio8_asr/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/tokenizers/qwen_bpe_bundle.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::audio8_asr {
namespace json = engine::io::json;
namespace {

qwen3_asr::Qwen3ASRAudioEncoderConfig parse_audio_encoder_config(const json::Value & value) {
    qwen3_asr::Qwen3ASRAudioEncoderConfig config;
    config.num_mel_bins = json::require_i64(value, "num_mel_bins");
    config.encoder_layers = json::require_i64(value, "encoder_layers");
    config.encoder_attention_heads = json::require_i64(value, "encoder_attention_heads");
    config.encoder_ffn_dim = json::require_i64(value, "encoder_ffn_dim");
    config.d_model = json::require_i64(value, "d_model");
    config.max_source_positions = json::optional_i64(value, "max_source_positions", 1500);
    config.n_window = json::require_i64(value, "n_window");
    config.n_window_infer = json::require_i64(value, "n_window_infer");
    config.conv_chunksize = json::require_i64(value, "conv_chunksize");
    config.downsample_hidden_size = json::require_i64(value, "downsample_hidden_size");
    config.output_dim = json::require_i64(value, "output_dim");
    config.activation_function = json::require_string(value, "activation_function");
    if (config.activation_function != "gelu") {
        throw std::runtime_error("Audio8 ASR currently supports gelu audio activation");
    }
    return config;
}

Audio8ASRConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");

    Audio8ASRConfig config;
    const auto model_type = json::require_string(root, "model_type");
    if (model_type != "arkasr") {
        throw std::runtime_error(
            "Audio8 ASR requires model_type 'arkasr', got: " + model_type);
    }
    config.merge_factor = json::optional_i64(root, "merge_factor", 4);
    if (config.merge_factor <= 0) {
        throw std::runtime_error("Audio8 ASR merge_factor must be positive");
    }

    const auto & audio_config = root.require("qwen3_asr_audio_config");
    config.audio_encoder = parse_audio_encoder_config(audio_config);

    config.tower.input_size = config.audio_encoder.output_dim;
    config.tower.hidden_size = config.audio_encoder.output_dim;
    const auto tower_intermediate = json::optional_i64(root, "qwen3_asr_mlp_tower_hidden_size", 0);
    config.tower.intermediate_size =
        tower_intermediate > 0 ? tower_intermediate : config.tower.input_size * 4;
    config.tower.layers = json::optional_i64(root, "qwen3_asr_mlp_tower_layers", 4);
    config.tower.output_size = json::require_i64(root, "hidden_size");

    auto & decoder = config.text_decoder;
    decoder.vocab_size = json::require_i64(root, "vocab_size");
    decoder.hidden_size = config.tower.output_size;
    decoder.intermediate_size = json::require_i64(root, "intermediate_size");
    decoder.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    decoder.num_attention_heads = json::require_i64(root, "num_attention_heads");
    decoder.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    decoder.head_dim = json::optional_i64(
        root, "head_dim", decoder.hidden_size / decoder.num_attention_heads);
    decoder.max_position_embeddings = json::optional_i64(root, "max_position_embeddings", 32768);
    decoder.audio_token_id = json::require_i64(root, "audio_token_id");
    decoder.pad_token_id = json::require_i64(root, "pad_token_id");
    decoder.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", true);
    decoder.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", decoder.rms_norm_eps);
    decoder.rope_theta = json::optional_f32(root, "rope_theta", decoder.rope_theta);

    const auto generation = resources.parse_json("generation_config");
    config.max_audio_samples = json::optional_i64(generation, "max_audio_samples", config.max_audio_samples);
    decoder.max_new_tokens = json::optional_i64(generation, "max_new_tokens", decoder.max_new_tokens);
    decoder.pad_token_id = json::optional_i64(generation, "pad_token_id", decoder.pad_token_id);
    decoder.eos_token_ids = json::require_i64_array_or_scalar(generation, "eos_token_id");

    const auto processor = resources.parse_json("preprocessor_config");
    const auto * feature_extractor = processor.find("feature_extractor");
    const auto & frontend = feature_extractor != nullptr && feature_extractor->is_object() ? *feature_extractor : processor;
    config.frontend.sample_rate = static_cast<int>(json::optional_i64(frontend, "sampling_rate", config.frontend.sample_rate));
    config.frontend.feature_size = json::require_i64(frontend, "feature_size");
    config.frontend.hop_length = json::require_i64(frontend, "hop_length");
    config.frontend.n_fft = json::require_i64(frontend, "n_fft");
    if (config.frontend.feature_size != config.audio_encoder.num_mel_bins) {
        throw std::runtime_error("Audio8 ASR frontend feature size does not match audio encoder config");
    }

    // Prompt special tokens live in tokenizer.json added_tokens; the audio
    // token id from the config must match the tokenizer entry.
    config.user_token_id = engine::tokenizers::require_added_token_id(resources, "<|user|>");
    config.begin_audio_token_id = engine::tokenizers::require_added_token_id(resources, "<|begin_of_audio|>");
    config.end_audio_token_id = engine::tokenizers::require_added_token_id(resources, "<|end_of_audio|>");
    config.assistant_token_id = engine::tokenizers::require_added_token_id(resources, "<|assistant|>");
    config.text_decoder.audio_token_id = engine::tokenizers::require_added_token_id(resources, "<|audio|>");

    config.supported_languages = {
        "Chinese", "English", "Cantonese", "French", "German", "Japanese", "Korean"};
    return config;
}

void validate_config(const Audio8ASRConfig & config) {
    const auto & decoder = config.text_decoder;
    if (decoder.vocab_size <= 0 || decoder.hidden_size <= 0 || decoder.intermediate_size <= 0 ||
        decoder.num_hidden_layers <= 0 || decoder.num_attention_heads <= 0 ||
        decoder.num_key_value_heads <= 0 || decoder.head_dim <= 0 ||
        decoder.hidden_size % decoder.num_attention_heads != 0) {
        throw std::runtime_error("Audio8 ASR invalid decoder metadata");
    }
    if (config.audio_encoder.output_dim <= 0 || config.tower.layers <= 0 ||
        config.tower.intermediate_size <= 0) {
        throw std::runtime_error("Audio8 ASR invalid audio adapter metadata");
    }
    if (config.frontend.sample_rate != 16000) {
        throw std::runtime_error("Audio8 ASR requires 16 kHz audio frontend");
    }
}

}  // namespace

std::shared_ptr<const Audio8ASRAssets> load_audio8_asr_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, "audio8_asr");
    if (!resources.has_file("preprocessor_config")) {
        throw std::runtime_error("Audio8 ASR requires preprocessor_config.json");
    }
    if (!resources.has_file("tokenizer_json")) {
        throw std::runtime_error("Audio8 ASR requires tokenizer.json");
    }

    auto assets = std::make_shared<Audio8ASRAssets>();
    assets->resources = std::move(resources);
    assets->config = parse_config(assets->resources);
    validate_config(assets->config);

    const auto weights = assets->resources.open_tensor_source("weights");
    auto encoder_assets = std::make_shared<qwen3_asr::Qwen3ASRAssets>();
    encoder_assets->config.audio_encoder_binding = {"audio_encoder", "audio_encoder"};
    encoder_assets->config.frontend.sample_rate = assets->config.frontend.sample_rate;
    encoder_assets->config.frontend.feature_size = assets->config.frontend.feature_size;
    encoder_assets->config.frontend.hop_length = assets->config.frontend.hop_length;
    encoder_assets->config.frontend.n_fft = assets->config.frontend.n_fft;
    encoder_assets->config.audio_encoder = assets->config.audio_encoder;
    encoder_assets->model_weights = weights;
    assets->encoder_assets = std::move(encoder_assets);

    assets->tokenizer = engine::tokenizers::load_qwen_bpe_tokenizer(assets->resources);
    return assets;
}

}  // namespace engine::community_models::audio8_asr
