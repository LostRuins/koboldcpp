#include "engine/models/yue2/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>

namespace engine::models::yue2 {
namespace json = engine::io::json;
namespace {

Yue2SamplingConfig parse_sampling(const json::Value & value, Yue2SamplingConfig fallback) {
    fallback.temperature = json::optional_f32(value, "temperature", fallback.temperature);
    fallback.top_p = json::optional_f32(value, "top_p", fallback.top_p);
    fallback.top_k = json::optional_i64(value, "top_k", fallback.top_k);
    fallback.repetition_penalty = json::optional_f32(value, "repetition_penalty", fallback.repetition_penalty);
    fallback.penalty_window = json::optional_i64(value, "penalty_window", fallback.penalty_window);
    fallback.min_tokens = json::optional_i64(value, "min_tokens", fallback.min_tokens);
    fallback.max_tokens = json::optional_i64(value, "max_tokens", fallback.max_tokens);
    if (fallback.temperature < 0.0F || fallback.temperature > 5.0F ||
        fallback.top_p <= 0.0F || fallback.top_p > 1.0F ||
        fallback.top_k < 1 ||
        fallback.repetition_penalty <= 0.0F ||
        fallback.penalty_window < 1 ||
        fallback.min_tokens < 0 ||
        fallback.max_tokens < fallback.min_tokens) {
        throw std::runtime_error("Yue2 sampling config is invalid");
    }
    return fallback;
}

Yue2ModelConfig parse_model_config(const std::filesystem::path & path) {
    const auto root = json::parse_file(path);
    Yue2ModelConfig out;
    out.hidden_size = json::require_i64(root, "hidden_size");
    out.layers = json::require_i64(root, "num_hidden_layers");
    out.attention_heads = json::require_i64(root, "num_attention_heads");
    out.kv_heads = json::require_i64(root, "num_key_value_heads");
    out.head_dim = json::require_i64(root, "head_dim");
    out.intermediate_size = json::require_i64(root, "intermediate_size");
    out.vocab_size = json::require_i64(root, "vocab_size");
    out.max_position_embeddings = json::require_i64(root, "max_position_embeddings");
    out.latent_dim = json::optional_i64(root, "latent_dim", out.latent_dim);
    out.max_latent_frames = json::optional_i64(root, "max_latent_frames", out.max_latent_frames);
    out.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
    out.rope_theta = json::optional_f32(root, "rope_theta", out.rope_theta);
    out.timestep_shift = json::optional_f32(root, "timestep_shift", out.timestep_shift);
    engine::io::require_positive(out.hidden_size, "Yue2 hidden_size");
    engine::io::require_positive(out.layers, "Yue2 layers");
    engine::io::require_positive(out.attention_heads, "Yue2 attention heads");
    engine::io::require_positive(out.kv_heads, "Yue2 kv heads");
    engine::io::require_positive(out.head_dim, "Yue2 head_dim");
    engine::io::require_positive(out.intermediate_size, "Yue2 intermediate_size");
    engine::io::require_positive(out.vocab_size, "Yue2 vocab_size");
    engine::io::require_positive(out.max_position_embeddings, "Yue2 max_position_embeddings");
    if (out.attention_heads % out.kv_heads != 0) {
        throw std::runtime_error("Yue2 attention heads must be divisible by kv heads");
    }
    return out;
}

Yue2VAEConfig parse_vae_config(const std::filesystem::path & path) {
    const auto root = json::parse_file(path);
    Yue2VAEConfig out;
    out.sample_rate = static_cast<int>(json::optional_i64(root, "sample_rate", out.sample_rate));
    out.channels = json::optional_i64(root, "audio_channels", out.channels);
    out.latent_dim = json::optional_i64(root, "latent_dim", out.latent_dim);
    out.downsampling_ratio = json::optional_i64(root, "downsampling_ratio", out.downsampling_ratio);
    out.decode_core_frames = json::optional_i64(root, "decode_core_frames", out.decode_core_frames);
    out.decode_halo_frames = json::optional_i64(root, "decode_halo_frames", out.decode_halo_frames);
    engine::io::require_positive(out.sample_rate, "Yue2 VAE sample_rate");
    engine::io::require_positive(out.channels, "Yue2 VAE channels");
    engine::io::require_positive(out.latent_dim, "Yue2 VAE latent_dim");
    return out;
}

Yue2GenerationConfig parse_generation_config(const std::filesystem::path & path) {
    Yue2GenerationConfig out;
    if (!engine::io::is_existing_file(path)) {
        out.abc = Yue2SamplingConfig{0.7F, 0.9F, 30, 1.005F, 100, 32, 4096};
        return out;
    }
    const auto root = json::parse_file(path);
    if (const auto * abc = root.find("abc")) {
        out.abc = parse_sampling(*abc, Yue2SamplingConfig{0.7F, 0.9F, 30, 1.005F, 100, 32, 4096});
    }
    if (const auto * semantic = root.find("semantic")) {
        out.semantic = parse_sampling(*semantic, out.semantic);
    }
    out.ode_steps = json::optional_i64(root, "ode_steps", out.ode_steps);
    out.context = json::optional_i64(root, "context", out.context);
    if (out.ode_steps <= 0 || out.context != kContextTokens) {
        throw std::runtime_error("Yue2 generation config requires positive midpoint steps and context=24576");
    }
    return out;
}

}  // namespace

std::shared_ptr<const Yue2Assets> load_yue2_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<Yue2Assets>();
    assets->model_root = assets::prepare_model_directory(model_path).model_root;
    const auto sidecars = assets->model_root / "sidecars";
    assets->tiktoken_path = sidecars / "yue2-qwen.tiktoken";
    const auto model_config = sidecars / "yue2-model-config.json";
    const auto generation_config = sidecars / "yue2-generation-config.json";
    const auto vae_config = sidecars / "yue2-vae-config.json";
    if (!engine::io::is_existing_file(model_config)) {
        throw std::runtime_error("missing Yue2 model config: " + model_config.string());
    }
    if (!engine::io::is_existing_file(vae_config)) {
        throw std::runtime_error("missing Yue2 VAE config: " + vae_config.string());
    }
    if (!engine::io::is_existing_file(assets->tiktoken_path)) {
        throw std::runtime_error("missing Yue2 tiktoken file: " + assets->tiktoken_path.string());
    }
    assets->config.model = parse_model_config(model_config);
    assets->config.vae = parse_vae_config(vae_config);
    assets->config.generation = parse_generation_config(generation_config);
    return assets;
}

}  // namespace engine::models::yue2
