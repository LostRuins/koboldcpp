#include "engine/models/voxcpm2/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>

namespace engine::models::voxcpm2 {
namespace json = engine::io::json;
namespace {

VoxCPM2RopeScalingConfig parse_rope_scaling(const json::Value & value) {
    VoxCPM2RopeScalingConfig config;
    config.type = json::optional_string(value, "type", "");
    config.long_factor = json::optional_f32_array(value, "long_factor");
    config.short_factor = json::optional_f32_array(value, "short_factor");
    config.original_max_position_embeddings =
        json::optional_i64(value, "original_max_position_embeddings", 0);
    return config;
}

VoxCPM2MiniCPMConfig parse_lm_config(const json::Value & value) {
    VoxCPM2MiniCPMConfig config;
    config.bos_token_id = json::optional_i64(value, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(value, "eos_token_id", config.eos_token_id);
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.kv_channels = json::optional_i64(value, "kv_channels", config.hidden_size / config.num_attention_heads);
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.scale_emb = json::optional_i64(value, "scale_emb", config.scale_emb);
    config.dim_model_base = json::optional_i64(value, "dim_model_base", config.dim_model_base);
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    config.scale_depth = json::optional_f32(value, "scale_depth", config.scale_depth);
    config.use_mup = json::optional_bool(value, "use_mup", config.use_mup);
    if (const auto * rope_scaling = value.find("rope_scaling"); rope_scaling != nullptr) {
        config.rope_scaling = parse_rope_scaling(*rope_scaling);
    }
    engine::io::require_positive(config.hidden_size, "lm hidden_size");
    engine::io::require_positive(config.intermediate_size, "lm intermediate_size");
    engine::io::require_positive(config.max_position_embeddings, "lm max_position_embeddings");
    engine::io::require_positive(config.num_attention_heads, "lm num_attention_heads");
    engine::io::require_positive(config.num_hidden_layers, "lm num_hidden_layers");
    engine::io::require_positive(config.num_key_value_heads, "lm num_key_value_heads");
    engine::io::require_positive(config.kv_channels, "lm kv_channels");
    engine::io::require_positive(config.vocab_size, "lm vocab_size");
    engine::io::require_divisible(config.hidden_size, config.num_attention_heads, "lm hidden_size / num_attention_heads");
    engine::io::require_divisible(config.num_attention_heads, config.num_key_value_heads, "lm attention heads");
    if (!config.rope_scaling.type.empty()) {
        if (config.rope_scaling.type != "longrope") {
            throw std::runtime_error("VoxCPM2 currently expects longrope rope_scaling");
        }
        const int64_t expected = config.hidden_size / config.num_attention_heads / 2;
        if (static_cast<int64_t>(config.rope_scaling.long_factor.size()) != expected ||
            static_cast<int64_t>(config.rope_scaling.short_factor.size()) != expected) {
            throw std::runtime_error("VoxCPM2 rope_scaling factor length does not match head_dim / 2");
        }
    }
    return config;
}

VoxCPM2LocalTransformerConfig parse_local_transformer_config(
    const json::Value & value,
    const char * label) {
    VoxCPM2LocalTransformerConfig config;
    config.hidden_dim = json::require_i64(value, "hidden_dim");
    config.ffn_dim = json::require_i64(value, "ffn_dim");
    config.num_heads = json::require_i64(value, "num_heads");
    config.num_layers = json::require_i64(value, "num_layers");
    config.kv_channels = json::optional_i64(value, "kv_channels", config.hidden_dim / config.num_heads);
    engine::io::require_positive(config.hidden_dim, label);
    engine::io::require_positive(config.ffn_dim, label);
    engine::io::require_positive(config.num_heads, label);
    engine::io::require_positive(config.num_layers, label);
    engine::io::require_positive(config.kv_channels, label);
    engine::io::require_divisible(config.hidden_dim, config.num_heads, label);
    return config;
}

VoxCPM2DiTConfig parse_dit_config(const json::Value & value) {
    const auto base = parse_local_transformer_config(value, "dit transformer");
    VoxCPM2DiTConfig config;
    config.hidden_dim = base.hidden_dim;
    config.ffn_dim = base.ffn_dim;
    config.num_heads = base.num_heads;
    config.num_layers = base.num_layers;
    config.kv_channels = base.kv_channels;
    config.mean_mode = json::optional_bool(value, "dit_mean_mode", json::optional_bool(value, "mean_mode", false));
    const auto & cfm = value.require("cfm_config");
    config.cfm.sigma_min = json::optional_f32(cfm, "sigma_min", config.cfm.sigma_min);
    config.cfm.solver = json::optional_string(cfm, "solver", config.cfm.solver);
    config.cfm.t_scheduler = json::optional_string(cfm, "t_scheduler", config.cfm.t_scheduler);
    config.cfm.inference_cfg_rate = json::optional_f32(cfm, "inference_cfg_rate", config.cfm.inference_cfg_rate);
    if (config.cfm.solver != "euler") {
        throw std::runtime_error("VoxCPM2 CFM currently expects euler solver");
    }
    if (config.cfm.t_scheduler != "log-norm") {
        throw std::runtime_error("VoxCPM2 CFM currently expects log-norm scheduler");
    }
    return config;
}

VoxCPM2AudioVAEConfig parse_audio_vae_config(const json::Value & value) {
    VoxCPM2AudioVAEConfig config;
    config.encoder_dim = json::require_i64(value, "encoder_dim");
    config.encoder_rates = json::require_i64_array(value, "encoder_rates");
    config.latent_dim = json::require_i64(value, "latent_dim");
    config.decoder_dim = json::require_i64(value, "decoder_dim");
    config.decoder_rates = json::require_i64_array(value, "decoder_rates");
    config.sample_rate_bin_boundaries = json::optional_i64_array(value, "sr_bin_boundaries");
    config.sample_rate = static_cast<int>(json::require_i64(value, "sample_rate"));
    config.output_sample_rate = static_cast<int>(json::require_i64(value, "out_sample_rate"));
    engine::io::require_positive(config.encoder_dim, "AudioVAE encoder_dim");
    engine::io::require_positive(config.latent_dim, "AudioVAE latent_dim");
    engine::io::require_positive(config.decoder_dim, "AudioVAE decoder_dim");
    engine::io::require_positive(config.sample_rate, "AudioVAE sample_rate");
    engine::io::require_positive(config.output_sample_rate, "AudioVAE out_sample_rate");
    if (config.encoder_rates.empty() || config.decoder_rates.empty()) {
        throw std::runtime_error("VoxCPM2 AudioVAE rates must be non-empty");
    }
    for (const auto rate : config.encoder_rates) {
        engine::io::require_positive(rate, "AudioVAE encoder rate");
    }
    for (const auto rate : config.decoder_rates) {
        engine::io::require_positive(rate, "AudioVAE decoder rate");
    }
    return config;
}

VoxCPM2Config parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    VoxCPM2Config config;
    config.architecture = json::require_string(root, "architecture");
    if (config.architecture != "voxcpm2") {
        throw std::runtime_error("VoxCPM2 config architecture mismatch: " + config.architecture);
    }
    config.lm = parse_lm_config(root.require("lm_config"));
    config.patch_size = json::optional_i64(root, "patch_size", config.patch_size);
    config.feat_dim = json::optional_i64(root, "feat_dim", config.feat_dim);
    config.residual_lm_num_layers =
        json::optional_i64(root, "residual_lm_num_layers", config.residual_lm_num_layers);
    config.residual_lm_no_rope = json::optional_bool(root, "residual_lm_no_rope", config.residual_lm_no_rope);
    config.scalar_quantization_latent_dim =
        json::optional_i64(root, "scalar_quantization_latent_dim", config.scalar_quantization_latent_dim);
    config.scalar_quantization_scale =
        json::optional_i64(root, "scalar_quantization_scale", config.scalar_quantization_scale);
    config.encoder = parse_local_transformer_config(root.require("encoder_config"), "local encoder transformer");
    config.dit = parse_dit_config(root.require("dit_config"));
    config.audio_vae = parse_audio_vae_config(root.require("audio_vae_config"));
    config.max_length = json::optional_i64(root, "max_length", config.max_length);
    config.device = json::optional_string(root, "device", config.device);
    config.dtype = json::optional_string(root, "dtype", config.dtype);
    engine::io::require_positive(config.patch_size, "patch_size");
    engine::io::require_positive(config.feat_dim, "feat_dim");
    engine::io::require_positive(config.residual_lm_num_layers, "residual_lm_num_layers");
    engine::io::require_positive(config.scalar_quantization_latent_dim, "scalar_quantization_latent_dim");
    engine::io::require_positive(config.scalar_quantization_scale, "scalar_quantization_scale");
    engine::io::require_positive(config.max_length, "max_length");
    if (config.feat_dim != config.audio_vae.latent_dim) {
        throw std::runtime_error("VoxCPM2 feat_dim must match AudioVAE latent_dim");
    }
    if (config.residual_lm_num_layers > config.lm.num_hidden_layers) {
        throw std::runtime_error("VoxCPM2 residual_lm_num_layers exceeds lm num_hidden_layers");
    }
    return config;
}

void validate_weight_anchors(const VoxCPM2Assets & assets) {
    const auto & config = assets.config;
    const auto & weights = *assets.model_weights;
    assets::require_tensor_shape(weights, "base_lm.embed_tokens.weight", {config.lm.vocab_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.norm.weight", {config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.layers.0.self_attn.q_proj.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.layers.0.self_attn.k_proj.weight",
        {config.lm.num_key_value_heads * config.lm.kv_channels, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.layers.0.mlp.gate_proj.weight", {config.lm.intermediate_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "residual_lm.norm.weight", {config.lm.hidden_size});
    assets::require_tensor_shape(weights, "feat_encoder.special_token", {1, 1, 1, config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "feat_encoder.in_proj.weight", {config.encoder.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "feat_encoder.encoder.norm.weight", {config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.in_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.cond_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.out_proj.weight", {config.feat_dim, config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.decoder.norm.weight", {config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "fsq_layer.in_proj.weight", {config.scalar_quantization_latent_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "fsq_layer.out_proj.weight", {config.lm.hidden_size, config.scalar_quantization_latent_dim});
    assets::require_tensor_shape(weights, "enc_to_lm_proj.weight", {config.lm.hidden_size, config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "lm_to_dit_proj.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "res_to_dit_proj.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "fusion_concat_proj.weight", {config.lm.hidden_size, config.lm.hidden_size * 2});
    assets::require_tensor_shape(weights, "stop_proj.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "stop_head.weight", {2, config.lm.hidden_size});

    const auto & vae = *assets.audiovae_weights;
    assets::require_tensor_shape(vae, "encoder.fc_mu.weight_v", {config.audio_vae.latent_dim, config.audio_vae.decoder_dim, 3});
    assets::require_tensor_shape(vae, "encoder.fc_mu.bias", {config.audio_vae.latent_dim});
    assets::require_tensor_shape(vae, "decoder.model.0.weight_v", {config.audio_vae.latent_dim, 1, 7});
    assets::require_tensor_shape(vae, "decoder.model.1.weight_v", {config.audio_vae.decoder_dim, config.audio_vae.latent_dim, 1});
}

}

std::shared_ptr<const VoxCPM2Assets> load_voxcpm2_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<VoxCPM2Assets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("voxcpm2"));
    out->config = parse_config(out->resources);
    out->model_weights = out->resources.open_tensor_source("weights");
    out->audiovae_weights = out->resources.open_tensor_source("audiovae_weights");
    validate_weight_anchors(*out);
    return out;
}

}  // namespace engine::models::voxcpm2

