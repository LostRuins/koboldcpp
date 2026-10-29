#include "engine/models/pocket_tts/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/yaml.h"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace engine::models::pocket_tts {
namespace {

struct PocketTTSDescriptor {
    float default_temperature = 0.7F;
    bool pad_with_spaces_for_short_inputs = false;
    bool remove_semicolons = false;
    bool insert_bos_before_voice = false;
    std::optional<int> model_recommended_frames_after_eos;
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    int flow_layers = 6;
    // Residual blocks of the flow net (upstream `flow_lm.flow.depth`). Distinct from the
    // transformer depth: the *_24l models have 24 transformer layers but a 6-block flow net.
    int flow_depth = 6;
    int flow_dim = 1024;
    int flow_heads = 16;
    int flow_hidden_size = 512;
    int flow_intermediate_size = 4096;
    int latent_dim = 32;
    int mimi_layers = 2;
    int mimi_dim = 512;
    int mimi_heads = 8;
    int mimi_intermediate_size = 2048;
    int mimi_inner_dim = 32;
    int mimi_outer_dim = 512;
    int mimi_context = 250;
    int mimi_seanet_dimension = 512;
    int mimi_base_filters = 64;
    int mimi_encoder_upsample_stride = 16;
};

PocketTTSDescriptor descriptor_from_yaml(const io::yaml::FlattenedDocument & parsed) {
    PocketTTSDescriptor descriptor;
    descriptor.default_temperature =
        io::yaml::optional_float(parsed, "default_temperature").value_or(descriptor.default_temperature);
    descriptor.pad_with_spaces_for_short_inputs =
        io::yaml::optional_bool(parsed, "pad_with_spaces_for_short_inputs", false);
    descriptor.remove_semicolons = io::yaml::optional_bool(parsed, "remove_semicolons", false);
    descriptor.insert_bos_before_voice = io::yaml::optional_bool(parsed, "flow_lm.insert_bos_before_voice", false);
    descriptor.model_recommended_frames_after_eos = io::yaml::optional_int(parsed, "model_recommended_frames_after_eos");
    descriptor.sample_rate = io::yaml::require_int(parsed, "mimi.sample_rate");
    descriptor.frame_rate = io::yaml::require_float(parsed, "mimi.frame_rate");
    descriptor.flow_layers = io::yaml::require_int(parsed, "flow_lm.transformer.num_layers");
    descriptor.flow_depth = io::yaml::optional_int(parsed, "flow_lm.flow.depth").value_or(descriptor.flow_depth);
    descriptor.flow_dim = io::yaml::require_int(parsed, "flow_lm.transformer.d_model");
    descriptor.flow_heads = io::yaml::require_int(parsed, "flow_lm.transformer.num_heads");
    descriptor.flow_hidden_size = io::yaml::require_int(parsed, "flow_lm.flow.dim");
    descriptor.flow_intermediate_size =
        descriptor.flow_dim * io::yaml::require_int(parsed, "flow_lm.transformer.hidden_scale");
    descriptor.latent_dim = io::yaml::require_int(parsed, "mimi.quantizer.dimension");
    descriptor.mimi_layers = io::yaml::require_int(parsed, "mimi.transformer.num_layers");
    descriptor.mimi_dim = io::yaml::require_int(parsed, "mimi.transformer.d_model");
    descriptor.mimi_heads = io::yaml::require_int(parsed, "mimi.transformer.num_heads");
    descriptor.mimi_intermediate_size = io::yaml::require_int(parsed, "mimi.transformer.dim_feedforward");
    descriptor.mimi_inner_dim = io::yaml::require_int(parsed, "mimi.inner_dim");
    descriptor.mimi_outer_dim = io::yaml::require_int(parsed, "mimi.outer_dim");
    descriptor.mimi_context = io::yaml::require_int(parsed, "mimi.transformer.context");
    descriptor.mimi_seanet_dimension = io::yaml::require_int(parsed, "mimi.seanet.dimension");
    descriptor.mimi_base_filters = io::yaml::require_int(parsed, "mimi.seanet.n_filters");
    int hop_length = 1;
    for (const int ratio : io::yaml::require_list_int(parsed, "mimi.seanet.ratios")) {
        hop_length *= ratio;
    }
    const double encoder_frame_rate = static_cast<double>(descriptor.sample_rate) / static_cast<double>(hop_length);
    descriptor.mimi_encoder_upsample_stride =
        static_cast<int>(std::llround(encoder_frame_rate / static_cast<double>(descriptor.frame_rate)));
    return descriptor;
}

PocketTTSDescriptor descriptor_from_builtin_defaults(const std::string & language) {
    PocketTTSDescriptor descriptor;
    // Upstream pocket-tts defaults the english and english_2026-04 models to
    // temperature 0.3 (human evals preferred it at equal WER/similarity).
    if (language == "english" || language == "english_2026-04") {
        descriptor.default_temperature = 0.3F;
    }
    descriptor.pad_with_spaces_for_short_inputs = (language == "english_2026-01");
    descriptor.insert_bos_before_voice = (language != "english_2026-01");
    if (language.find("french_24l") != std::string::npos) {
        descriptor.remove_semicolons = true;
        descriptor.model_recommended_frames_after_eos = 8;
    }
    descriptor.flow_layers = language.find("_24l") != std::string::npos ? 24 : 6;
    descriptor.mimi_inner_dim = (language == "english_2026-01") ? 512 : 32;
    return descriptor;
}

PocketTTSDescriptor resolve_descriptor(
    const assets::ResourceBundle & resources,
    const std::string & language) {
    if (resources.has_file("config")) {
        return descriptor_from_yaml(resources.parse_flattened_yaml("config"));
    }
    return descriptor_from_builtin_defaults(language);
}

PocketTTSModelConfig make_model_config(const PocketTTSDescriptor & descriptor) {
    PocketTTSModelConfig config;
    config.default_temperature = descriptor.default_temperature;
    config.sample_rate = descriptor.sample_rate;
    config.frame_rate = descriptor.frame_rate;
    config.mimi_frame_rate = descriptor.frame_rate;
    config.flow_layers = descriptor.flow_layers;
    config.flow_depth = descriptor.flow_depth;
    config.flow_dim = descriptor.flow_dim;
    config.flow_heads = descriptor.flow_heads;
    config.flow_hidden_size = descriptor.flow_hidden_size;
    config.flow_intermediate_size = descriptor.flow_intermediate_size;
    config.latent_dim = descriptor.latent_dim;
    config.mimi_layers = descriptor.mimi_layers;
    config.mimi_dim = descriptor.mimi_dim;
    config.mimi_heads = descriptor.mimi_heads;
    config.mimi_intermediate_size = descriptor.mimi_intermediate_size;
    config.mimi_inner_dim = descriptor.mimi_inner_dim;
    config.mimi_outer_dim = descriptor.mimi_outer_dim;
    config.mimi_context = descriptor.mimi_context;
    config.mimi_seanet_dimension = descriptor.mimi_seanet_dimension;
    config.mimi_base_filters = descriptor.mimi_base_filters;
    config.mimi_encoder_upsample_stride = descriptor.mimi_encoder_upsample_stride;
    config.pad_with_spaces_for_short_inputs = descriptor.pad_with_spaces_for_short_inputs;
    config.remove_semicolons = descriptor.remove_semicolons;
    config.insert_bos_before_voice = descriptor.insert_bos_before_voice;
    config.model_recommended_frames_after_eos = descriptor.model_recommended_frames_after_eos;
    return config;
}

PocketTTSHostWeights load_host_weights(const assets::TensorSource & source) {
    PocketTTSHostWeights weights;
    weights.conditioner_embedding_table = source.require_f32_tensor("flow_lm.conditioner.embed.weight");
    weights.bos_emb = source.require_f32("flow_lm.bos_emb");
    weights.bos_before_voice = source.optional_f32_tensor("flow_lm.bos_before_voice");
    weights.emb_mean = source.require_f32("flow_lm.emb_mean");
    weights.emb_std = source.require_f32("flow_lm.emb_std");
    return weights;
}

PocketTTSAssets load_assets_from_resources(
    assets::ResourceBundle resources,
    std::string language) {
    const auto descriptor = resolve_descriptor(resources, language);
    const auto tensor_source = resources.open_tensor_source("weights");

    PocketTTSAssets manifest;
    manifest.resources = std::move(resources);
    manifest.model_weights = tensor_source;
    manifest.voice_asset_root = tensor_source->source_path().parent_path();
    manifest.host_weights = load_host_weights(*tensor_source);
    manifest.language = std::move(language);
    manifest.tokenizer_pieces = tokenizers::load_sentencepiece_model(manifest.resources.require_file("tokenizer"));
    manifest.model_config = make_model_config(descriptor);
    return manifest;
}

}  // namespace

std::shared_ptr<const PocketTTSAssets> load_pocket_tts_assets(
    const std::filesystem::path & model_path,
    std::string language) {
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("pocket_tts"));
    return std::make_shared<PocketTTSAssets>(load_assets_from_resources(std::move(resources), std::move(language)));
}

}  // namespace engine::models::pocket_tts
