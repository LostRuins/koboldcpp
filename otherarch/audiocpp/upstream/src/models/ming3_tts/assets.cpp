#include "engine/models/ming3_tts/assets.h"

#include "engine/framework/model_spec/package.h"

namespace engine::models::ming3_tts {
namespace {

modules::MingBackboneConfig parse_backbone(const io::json::Value & value) {
    modules::MingBackboneConfig config;
    config.hidden_size = value.require("hidden_size").as_i64();
    config.intermediate_size = value.require("intermediate_size").as_i64();
    config.layers = value.require("num_hidden_layers").as_i64();
    config.attention_heads = value.require("num_attention_heads").as_i64();
    config.key_value_heads = value.require("num_key_value_heads").as_i64();
    config.rms_norm_eps = value.require("rms_norm_eps").as_f32();
    config.rope_theta = value.require("rope_theta").as_f32();
    return config;
}

modules::MingPatchNetworkConfig parse_patch_network(const io::json::Value & value) {
    modules::MingPatchNetworkConfig config;
    config.hidden_size = value.require("hidden_size").as_i64();
    config.heads = value.require("num_heads").as_i64();
    config.depth = value.require("depth").as_i64();
    config.intermediate_size = config.hidden_size * value.require("mlp_ratio").as_i64();
    return config;
}

}  // namespace

MingConfig parse_ming_config(const io::json::Value & value) {
    MingConfig config;
    const auto & llm = value.require("llm_config");
    config.backbone = parse_backbone(llm);
    config.vocabulary_size = llm.require("vocab_size").as_i64();
    const auto & codec = value.require("audio_tokenizer_config");
    config.codec.patch_size = codec.require("patch_size").as_i64();
    config.codec.sample_rate = codec.require("sample_rate").as_i64();
    const auto & encoder = codec.require("enc_kwargs");
    config.codec.encoder.backbone = parse_backbone(encoder.require("backbone"));
    config.codec.encoder.input_dim = encoder.require("input_dim").as_i64();
    config.codec.encoder.hop_size = encoder.require("hop_size").as_i64();
    config.codec.encoder.latent_dim = encoder.require("latent_dim").as_i64();
    config.codec.encoder.sliding_window = encoder.require("backbone").require("sliding_window").as_i64();
    const auto & decoder = codec.require("dec_kwargs");
    config.codec.decoder.backbone = parse_backbone(decoder.require("backbone"));
    config.codec.decoder.latent_dim = decoder.require("latent_dim").as_i64();
    config.codec.decoder.output_dim = decoder.require("output_dim").as_i64();
    config.codec.decoder.sliding_window = decoder.require("backbone").require("sliding_window").as_i64();
    config.flow.network = parse_patch_network(value.require("ditar_config"));
    config.flow.aggregator = parse_patch_network(value.require("aggregator_config"));
    config.flow.latent_dim = config.codec.encoder.latent_dim;
    config.flow.conditioning_dim = config.backbone.hidden_size;
    return config;
}

std::shared_ptr<const MingAssets> load_ming3_tts_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<MingAssets>();
    assets->resources = model_spec::load_resource_bundle_for_family(path, "ming3_tts");
    assets->config = parse_ming_config(assets->resources.parse_json("config"));
    assets->model_weights = assets->resources.open_tensor_source("weights");
    assets->speaker_weights = assets->resources.open_tensor_source("speaker");
    return assets;
}

}  // namespace engine::models::ming3_tts
