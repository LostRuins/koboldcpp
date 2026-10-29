#include "engine/models/neutts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"
#include "engine/framework/model_spec/package.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace engine::models::neutts {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "neutts";

NeuTTSBackboneConfig parse_backbone_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    if (json::require_string(root, "model_type") != "qwen3") {
        throw std::runtime_error("NeuTTS backbone config must use model_type qwen3");
    }
    NeuTTSBackboneConfig out;
    out.hidden_size = json::require_i64(root, "hidden_size");
    out.intermediate_size = json::require_i64(root, "intermediate_size");
    out.layers = json::require_i64(root, "num_hidden_layers");
    out.attention_heads = json::require_i64(root, "num_attention_heads");
    out.kv_heads = json::require_i64(root, "num_key_value_heads");
    out.head_dim = json::require_i64(root, "head_dim");
    out.vocab_size = json::require_i64(root, "vocab_size");
    out.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
    if (const auto * rope = root.find("rope_parameters")) {
        out.rope_theta = json::optional_f32(*rope, "rope_theta", out.rope_theta);
    } else {
        out.rope_theta = json::optional_f32(root, "rope_theta", out.rope_theta);
    }
    out.eos_token_id = static_cast<int32_t>(json::require_i64(root, "eos_token_id"));
    out.pad_token_id = static_cast<int32_t>(json::require_i64(root, "pad_token_id"));
    out.max_context = json::optional_i64(root, "max_position_embeddings", 2048);

    const auto & neuphonic = root.require("neuphonic");
    const auto input_format = json::require_string(neuphonic, "input_format");
    if (input_format != "BPE") {
        throw std::runtime_error("NeuTTS native runtime expects BPE input format");
    }
    out.supported_languages = json::require_string_array(neuphonic, "supported_langs");
    out.supported_emotions = json::require_string_array(neuphonic, "supported_emotions");
    return out;
}

NeuTTSCodecConfig parse_codec_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("codec_config");
    if (json::require_string(root, "model_type") != "neucodec") {
        throw std::runtime_error("NeuTTS codec config must use model_type neucodec");
    }
    NeuTTSCodecConfig out;
    out.sample_rate = json::require_i64(root, "sampling_rate");
    out.output_sample_rate = json::require_i64(root, "output_sampling_rate");
    // NeuCodec's preprocessor hop_length belongs to the encoder/frontend path.
    // The decoder is constructed by the reference code with hop_length=480.
    out.hidden_size = json::require_i64(root, "hidden_size");
    out.intermediate_size = json::require_i64(root, "intermediate_size");
    out.layers = json::require_i64(root, "num_hidden_layers");
    out.attention_heads = json::require_i64(root, "num_attention_heads");
    out.kv_heads = json::require_i64(root, "num_key_value_heads");
    out.head_dim = json::require_i64(root, "head_dim");
    out.quantization_dim = json::require_i64(root, "quantization_dim");
    out.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
    if (const auto * rope = root.find("rope_parameters")) {
        out.rope_theta = json::optional_f32(*rope, "rope_theta", out.rope_theta);
    }
    out.quantization_levels = json::require_i64_array(root, "quantization_levels");
    return out;
}

std::vector<int32_t> read_i32_vector(const assets::TensorSource & source, const std::string & name) {
    const auto raw = source.require_tensor_data(name);
    if (raw.metadata.dtype != "I32" && raw.metadata.dtype != "i32") {
        throw std::runtime_error("NeuTTS speaker prompt tensor must be I32: " + name);
    }
    if (raw.metadata.shape.size() != 1 || raw.metadata.shape[0] <= 0) {
        throw std::runtime_error("NeuTTS speaker prompt tensor must be rank-1: " + name);
    }
    const size_t count = static_cast<size_t>(raw.metadata.shape[0]);
    if (raw.bytes.size() != count * sizeof(int32_t)) {
        throw std::runtime_error("NeuTTS speaker prompt byte size mismatch: " + name);
    }
    std::vector<int32_t> out(count);
    std::memcpy(out.data(), raw.bytes.data(), raw.bytes.size());
    return out;
}

void add_speaker(
    NeuTTSAssets & assets,
    const assets::TensorSource & speaker_source,
    const std::string & id) {
    NeuTTSSpeakerPrompt prompt;
    prompt.id = id;
    prompt.reference_text = engine::io::trim_ascii_whitespace(assets.resources.read_text("speaker_text_" + id));
    prompt.speech_codes = read_i32_vector(speaker_source, id);
    auto inserted = assets.speakers.emplace(id, std::move(prompt));
    if (!inserted.second) {
        throw std::runtime_error("duplicate NeuTTS speaker prompt: " + id);
    }
}

}  // namespace

std::shared_ptr<const NeuTTSAssets> load_neutts_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<NeuTTSAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path(kFamily));
    assets->backbone = parse_backbone_config(assets->resources);
    assets->codec = parse_codec_config(assets->resources);
    assets->backbone_weights = assets->resources.open_tensor_source("backbone");
    assets->codec_weights = assets->resources.open_tensor_source("codec");

    const auto speaker_source = assets->resources.open_tensor_source("speaker_prompts");
    for (const char * id : {"dave", "emily", "greta", "jo", "juliette", "mateo", "paul", "sophie", "steven"}) {
        add_speaker(*assets, *speaker_source, id);
    }
    return assets;
}

}  // namespace engine::models::neutts
