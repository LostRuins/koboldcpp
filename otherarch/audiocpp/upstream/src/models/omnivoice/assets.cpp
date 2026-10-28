#include "engine/models/omnivoice/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::omnivoice {
namespace json = engine::io::json;
namespace {

OmniVoiceConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    OmniVoiceConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.audio_vocab_size = json::require_i64(root, "audio_vocab_size");
    config.audio_mask_id = json::require_i64(root, "audio_mask_id");
    config.num_audio_codebook = json::require_i64(root, "num_audio_codebook");
    config.audio_codebook_weights = json::optional_f32_array(root, "audio_codebook_weights");
    config.eos_token_id = json::optional_i64(root, "eos_token_id", 0);
    config.pad_token_id = json::optional_i64(root, "pad_token_id", 0);

    const auto & llm = root.require("llm_config");
    config.llm.model_type = json::require_string(llm, "model_type");
    config.llm.vocab_size = json::require_i64(llm, "vocab_size");
    config.llm.hidden_size = json::require_i64(llm, "hidden_size");
    config.llm.intermediate_size = json::require_i64(llm, "intermediate_size");
    config.llm.num_hidden_layers = json::require_i64(llm, "num_hidden_layers");
    config.llm.num_attention_heads = json::require_i64(llm, "num_attention_heads");
    config.llm.num_key_value_heads = json::require_i64(llm, "num_key_value_heads");
    config.llm.head_dim = json::optional_i64(llm, "head_dim", config.llm.hidden_size / config.llm.num_attention_heads);
    config.llm.max_position_embeddings = json::require_i64(llm, "max_position_embeddings");
    config.llm.rms_norm_eps = json::optional_f32(llm, "rms_norm_eps", config.llm.rms_norm_eps);
    const auto * rope_parameters = llm.find("rope_parameters");
    if (rope_parameters != nullptr && rope_parameters->is_object()) {
        config.llm.rope_theta = json::optional_f32(*rope_parameters, "rope_theta", config.llm.rope_theta);
    }

    const auto audio_tokenizer_root = resources.parse_json("audio_tokenizer_config");
    config.audio_tokenizer.model_type = json::require_string(audio_tokenizer_root, "model_type");
    config.audio_tokenizer.sample_rate = static_cast<int>(json::optional_i64(audio_tokenizer_root, "sample_rate", 24000));
    config.audio_tokenizer.semantic_sample_rate =
        static_cast<int>(json::optional_i64(audio_tokenizer_root, "semantic_sample_rate", 16000));
    config.audio_tokenizer.downsample_factor = json::optional_i64(audio_tokenizer_root, "downsample_factor", 0);
    config.audio_tokenizer.codebook_size = json::optional_i64(audio_tokenizer_root, "codebook_size", 0);
    config.audio_tokenizer.num_codebooks = json::optional_i64(audio_tokenizer_root, "num_codebooks", 0);
    config.audio_tokenizer.hidden_size = json::optional_i64(audio_tokenizer_root, "hidden_size", 0);
    config.audio_tokenizer.codebook_dim = json::optional_i64(audio_tokenizer_root, "codebook_dim", 0);
    config.audio_tokenizer.kernel_size = json::optional_i64(audio_tokenizer_root, "kernel_size", 0);
    config.audio_tokenizer.unit_kernel_size = json::optional_i64(audio_tokenizer_root, "unit_kernel_size", 0);
    config.audio_tokenizer.target_bandwidths = json::optional_f32_array(audio_tokenizer_root, "target_bandwidths");
    config.audio_tokenizer.channel_ratios = json::optional_i64_array(audio_tokenizer_root, "channel_ratios");
    config.audio_tokenizer.strides = json::optional_i64_array(audio_tokenizer_root, "strides");
    config.audio_tokenizer.block_dilations = json::optional_i64_array(audio_tokenizer_root, "block_dilations");

    const auto & semantic = audio_tokenizer_root.require("semantic_model_config");
    config.audio_tokenizer.semantic_model.hidden_size = json::optional_i64(semantic, "hidden_size", 0);
    config.audio_tokenizer.semantic_model.intermediate_size = json::optional_i64(semantic, "intermediate_size", 0);
    config.audio_tokenizer.semantic_model.num_attention_heads = json::optional_i64(semantic, "num_attention_heads", 0);
    config.audio_tokenizer.semantic_model.num_hidden_layers = json::optional_i64(semantic, "num_hidden_layers", 0);
    config.audio_tokenizer.semantic_model.num_conv_pos_embeddings =
        json::optional_i64(semantic, "num_conv_pos_embeddings", 0);
    config.audio_tokenizer.semantic_model.num_conv_pos_embedding_groups =
        json::optional_i64(semantic, "num_conv_pos_embedding_groups", 0);
    config.audio_tokenizer.semantic_model.layer_norm_eps =
        json::optional_f32(semantic, "layer_norm_eps", config.audio_tokenizer.semantic_model.layer_norm_eps);
    config.audio_tokenizer.semantic_model.feat_proj_layer_norm =
        json::optional_bool(semantic, "feat_proj_layer_norm", config.audio_tokenizer.semantic_model.feat_proj_layer_norm);
    config.audio_tokenizer.semantic_model.do_stable_layer_norm =
        json::optional_bool(semantic, "do_stable_layer_norm", config.audio_tokenizer.semantic_model.do_stable_layer_norm);
    config.audio_tokenizer.semantic_model.conv_dim = json::optional_i64_array(semantic, "conv_dim");
    config.audio_tokenizer.semantic_model.conv_kernel = json::optional_i64_array(semantic, "conv_kernel");
    config.audio_tokenizer.semantic_model.conv_stride = json::optional_i64_array(semantic, "conv_stride");

    const auto & acoustic = audio_tokenizer_root.require("acoustic_model_config");
    config.audio_tokenizer.acoustic_codebooks = json::optional_i64(acoustic, "n_codebooks", 0);
    config.audio_tokenizer.hop_length = json::optional_i64(acoustic, "hop_length", 0);
    config.audio_tokenizer.acoustic_model.codebook_dim = json::optional_i64(acoustic, "codebook_dim", 0);
    config.audio_tokenizer.acoustic_model.encoder_hidden_size = json::optional_i64(acoustic, "encoder_hidden_size", 0);
    config.audio_tokenizer.acoustic_model.decoder_hidden_size = json::optional_i64(acoustic, "decoder_hidden_size", 0);
    config.audio_tokenizer.acoustic_model.hidden_size = json::optional_i64(acoustic, "hidden_size", 0);
    config.audio_tokenizer.acoustic_model.downsampling_ratios =
        json::optional_i64_array(acoustic, "downsampling_ratios");
    config.audio_tokenizer.acoustic_model.upsampling_ratios =
        json::optional_i64_array(acoustic, "upsampling_ratios");
    if (config.audio_tokenizer.hidden_size <= 0 &&
        config.audio_tokenizer.semantic_model.hidden_size > 0 &&
        config.audio_tokenizer.acoustic_model.hidden_size > 0) {
        config.audio_tokenizer.hidden_size =
            config.audio_tokenizer.semantic_model.hidden_size +
            config.audio_tokenizer.acoustic_model.hidden_size;
    }

    const auto preprocessor = resources.parse_json("audio_tokenizer_preprocessor");
    config.audio_tokenizer.sample_rate = static_cast<int>(
        json::optional_i64(preprocessor, "sampling_rate", config.audio_tokenizer.sample_rate));
    if (config.audio_tokenizer.num_codebooks <= 0 && config.audio_tokenizer.hop_length > 0 &&
        config.audio_tokenizer.codebook_size > 0 && !config.audio_tokenizer.target_bandwidths.empty()) {
        const float highest_bandwidth = config.audio_tokenizer.target_bandwidths.back();
        const double bits_per_codebook = std::log2(static_cast<double>(config.audio_tokenizer.codebook_size));
        const double frame_rate = std::ceil(
            static_cast<double>(config.audio_tokenizer.sample_rate) /
            static_cast<double>(config.audio_tokenizer.hop_length));
        config.audio_tokenizer.num_codebooks = static_cast<int64_t>(
            std::max(1.0, std::floor((1000.0 * static_cast<double>(highest_bandwidth)) / (frame_rate * bits_per_codebook))));
    }
    return config;
}

}  // namespace

std::shared_ptr<const OmniVoiceAssets> load_omnivoice_assets(const std::filesystem::path & model_path) {
    OmniVoiceAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("omnivoice"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("weights");
    assets.audio_tokenizer_weights = assets.resources.open_tensor_source("audio_tokenizer_weights");
    return std::make_shared<OmniVoiceAssets>(std::move(assets));
}

}  // namespace engine::models::omnivoice
