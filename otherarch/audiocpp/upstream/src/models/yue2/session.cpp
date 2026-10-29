#include "engine/models/yue2/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::yue2 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr const char * kFamily = "yue2";

std::shared_ptr<const Yue2Assets> require_assets(std::shared_ptr<const Yue2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Yue2 session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType parse_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    assets::TensorStorageType fallback) {
    return runtime::parse_tensor_storage_option(
        options.options,
        key,
        "yue2.weight_type",
        fallback,
        {
            assets::TensorStorageType::Native,
            assets::TensorStorageType::F32,
            assets::TensorStorageType::F16,
            assets::TensorStorageType::BF16,
            assets::TensorStorageType::Q8_0,
            assets::TensorStorageType::Q4_0,
            assets::TensorStorageType::Q4_K,
        });
}

core::AttentionPreference attention_preference_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"yue2.attention"})) {
        return core::parse_attention_preference(*value, "yue2.attention");
    }
    return core::AttentionPreference::Auto;
}

std::filesystem::path resolve_component_gguf_path(
    const Yue2Assets & assets,
    std::string_view option_name,
    const std::string & value) {
    if (value.empty()) {
        throw std::runtime_error(std::string(option_name) + " must not be empty");
    }
    const std::filesystem::path requested = std::filesystem::u8path(value);
    // kcpp: --musicllm and --musicvae are independent explicit file paths.
    const auto path = requested.is_absolute() ? requested : assets.model_root / requested;
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string(option_name) + " file does not exist: " + path.string());
    }
    if (path.extension() != ".gguf") {
        throw std::runtime_error(std::string(option_name) + " must point to a GGUF file");
    }
    return path;
}

void validate_component_anchors(const Yue2Assets & assets) {
    const auto & source = *assets.model_weights;
    assets::require_tensor_shape(
        source,
        "model.embed_tokens.weight",
        {assets.config.model.vocab_size, assets.config.model.hidden_size});
    assets::require_tensor_shape(
        source,
        "model.layers.0.self_attn.q_proj.weight",
        {assets.config.model.attention_heads * assets.config.model.head_dim, assets.config.model.hidden_size});
    assets::require_tensor_shape(
        source,
        "model.layers.0.nar_self_attn.q_proj.weight",
        {assets.config.model.attention_heads * assets.config.model.head_dim, assets.config.model.hidden_size});
    assets::require_tensor_shape(source, "vae2llm.weight", {assets.config.model.hidden_size, assets.config.model.latent_dim});
    assets::require_tensor_shape(source, "llm2vae.weight", {assets.config.model.latent_dim, assets.config.model.hidden_size});
    if (assets.vae_weights->has_tensor("decoder.layers.0.weight")) {
        assets::require_tensor_shape(*assets.vae_weights, "decoder.layers.0.weight", {2048, 64, 7});
    } else {
        assets::require_tensor_shape(*assets.vae_weights, "decoder.layers.0.weight_g", {2048, 1, 1});
    }
}

std::shared_ptr<const Yue2Assets> select_component_assets(
    std::shared_ptr<const Yue2Assets> base,
    const std::unordered_map<std::string, std::string> & options) {
    auto selected = std::make_shared<Yue2Assets>(*base);
    const std::string model_gguf =
        runtime::find_option(options, {"yue2.model_gguf"}).value_or("yue2-3b-q8_0.gguf");
    selected->model_weights = assets::open_tensor_source(
        resolve_component_gguf_path(*base, "yue2.model_gguf", model_gguf),
        "model_weights");
    const std::string vae_gguf =
        runtime::find_option(options, {"yue2.vae_gguf"}).value_or("yue2-vae-f16.gguf");
    selected->vae_weights = assets::open_tensor_source(
        resolve_component_gguf_path(*base, "yue2.vae_gguf", vae_gguf),
        "vae_weights");
    validate_component_anchors(*selected);
    auto adapter_path = std::filesystem::u8path(runtime::find_option(options, {"yue2.ar_lora"}).value_or(""));
    auto nar_adapter_path = std::filesystem::u8path(runtime::find_option(options, {"yue2.nar_lora"}).value_or(""));
    if (!adapter_path.empty() || !nar_adapter_path.empty()) {
        if (!adapter_path.empty() && adapter_path.is_relative()) adapter_path = selected->model_root / adapter_path;
        if (!nar_adapter_path.empty() && nar_adapter_path.is_relative()) nar_adapter_path = selected->model_root / nar_adapter_path;
        selected->model_weights = make_yue2_lora_source(
            selected->model_weights, adapter_path,
            runtime::parse_finite_float_option(options, {"yue2.ar_lora_scale"}).value_or(1.0F),
            selected->config.model.layers, nar_adapter_path,
            runtime::parse_finite_float_option(options, {"yue2.nar_lora_scale"}).value_or(1.0F));
    }
    return selected;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_yue2_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const Yue2Assets> assets) {
    return std::make_unique<Yue2Session>(
        task,
        options,
        std::move(assets));
}

const runtime::ModelMetadata & yue2_metadata() noexcept {
    static const runtime::ModelMetadata metadata{
        kFamily,
        "Yue2",
        "Yue2 music generation with lyrics, style, and optional ABC conditioning.",
        {}, // kcpp: fixed configs and bundled tokenizer; no user sidecars.
        {
            "yue2-3b-q8_0.gguf",
            "yue2-3b-q4_0.gguf",
            "yue2-3b-bf16.gguf",
            "yue2-vae-f16.gguf",
            "yue2-vae-f32.gguf",
        }};
    return metadata;
}

const runtime::CapabilitySet & yue2_capabilities() noexcept {
    static const runtime::CapabilitySet capabilities{
        {runtime::TaskCapability{
            runtime::VoiceTaskKind::AudioGeneration,
            {runtime::RunMode::Offline},
        }},
        {"auto"},
        false,
        true,
        false,
    };
    return capabilities;
}

runtime::ModelCliInterface yue2_cli_interface() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"style", "string", "Song style/tags.", true},
        {"lyrics", "string", "Lyrics text. Leave empty for instrumental generation; if omitted, the CLI text input is used."},
        {"abc", "string", "External ABC score text for melody/full routes."},
        {"abc_file", "string", "Path to an external ABC score file for melody/full routes."},
        {"cot", "off|melody|full", "Symbolic planning route. off skips ABC generation; melody/full generate or consume ABC before music tokens. Default: full.", false, "full"},
        {"export_semantic", "bool", "Attach the semantic token stream as a semantic artifact (JSON array of codec indices); default false.", false, "false"},
        {"stop_after", "abc|semantic|audio", "Last stage to run. abc writes the generated score only; semantic also writes the semantic token export; audio renders the full song. Default: audio.", false, "audio"},
        {"seed", "int", "Generation seed. Must be nonnegative and less than 2^63. Default: 1234.", false, "1234", "0"},
        {"guidance_scale", "float", "Semantic classifier-free guidance scale. Default follows the upstream route defaults. Omitted defaults to 1.01 with cot=off, otherwise 1.0.", false, std::nullopt, "0", "20"},
        {"num_inference_steps", "int", "NAR midpoint ODE steps. Default: 8.", false, "8", "1"},
        {"semantic_prefix", "string", "JSON array of semantic codec indices to force as the start of the music stream."},
        {"semantic_prefix_file", "string", "Path to a JSON array of semantic codec indices to force as the start of the music stream."},
        {"nar_noise_file", "string", "Path to a raw float32 noise file for NAR generation, shaped [frames,64]."},
        {"abc_temperature", "float", "ABC planner sampling temperature.", false, std::nullopt, "0", "5"},
        {"abc_top_p", "float", "ABC planner nucleus sampling probability. Must be in (0, 1].", false, std::nullopt, std::nullopt, "1"},
        {"abc_top_k", "int", "ABC planner top-k sampling limit.", false, std::nullopt, "1"},
        {"abc_repetition_penalty", "float", "ABC planner repetition penalty. Must be positive."},
        {"abc_penalty_window", "int", "ABC planner repetition penalty window.", false, std::nullopt, "1"},
        {"abc_min_tokens", "int", "Minimum ABC planner tokens before EOS is accepted.", false, std::nullopt, "0"},
        {"abc_max_tokens", "int", "Maximum ABC planner tokens. Zero is allowed when min_tokens is also zero.", false, std::nullopt, "0"},
        {"semantic_temperature", "float", "Semantic codec sampling temperature.", false, std::nullopt, "0", "5"},
        {"semantic_top_p", "float", "Semantic codec nucleus sampling probability. Must be in (0, 1].", false, std::nullopt, std::nullopt, "1"},
        {"semantic_top_k", "int", "Semantic codec top-k sampling limit.", false, std::nullopt, "1"},
        {"semantic_repetition_penalty", "float", "Semantic codec repetition penalty. Must be positive."},
        {"semantic_penalty_window", "int", "Semantic codec repetition penalty window.", false, std::nullopt, "1"},
        {"semantic_min_tokens", "int", "Minimum semantic tokens before EOS is accepted.", false, std::nullopt, "0"},
        {"semantic_max_tokens", "int", "Maximum semantic codec tokens. Zero is allowed when min_tokens is also zero.", false, std::nullopt, "0"},
    };
    out.session_options = {
        {"yue2.ar_lora", "path", "AR adapter safetensors file; native, PEFT, and ComfyUI LoRA layouts. Relative paths use the model root."},
        {"yue2.ar_lora_scale", "float", "AR LoRA delta scale; zero disables the adapter.", false, "1.0"},
        {"yue2.nar_lora", "path", "NAR adapter safetensors file; relative paths use the model root."},
        {"yue2.nar_lora_scale", "float", "NAR delta scale; full projection replacements are unscaled. Zero disables the adapter.", false, "1.0"},
        {"yue2.model_gguf", "string", "Yue2 main AR/NAR component GGUF file relative to the model root.", false, "yue2-3b-q8_0.gguf"},
        {"yue2.vae_gguf", "string", "Yue2 VAE component GGUF file relative to the model root.", false, "yue2-vae-f16.gguf"},
        {"yue2.model_weight_type", "native|f32|f16|bf16|q8_0|q4_0|q4_k", "YuE2 MoT weight storage type. Omitted inherits weight_type."},
        {"yue2.vae_weight_type", "native|f32|f16|bf16|q8_0|q4_0|q4_k", "YuE2 VAE weight storage type. Omitted inherits weight_type."},
        {"yue2.model_weight_context_mb", "int", "Yue2 main model weight context size in MiB.", false, "32", "1"},
        {"yue2.vae_weight_context_mb", "int", "Yue2 VAE weight context size in MiB.", false, "32", "1"},
        {"yue2.ar_prefill_graph_arena_mb", "int", "AR prefill graph arena size in MiB.", false, "32", "1"},
        {"yue2.ar_decode_graph_arena_mb", "int", "AR one-token decode graph arena size in MiB.", false, "32", "1"},
        {"yue2.nar_graph_arena_mb", "int", "NAR acoustic flow graph arena size in MiB.", false, "32", "1"},
        {"yue2.vae_graph_arena_mb", "int", "VAE decode graph arena size in MiB.", false, "32", "1"},
        {"yue2.attention", "auto|flash|eager", "NAR acoustic-flow attention lowering; auto uses flash except on CUDA sm70 (no kernel) and Intel Vulkan (eager measured faster).", false, "auto"},
        {"yue2.attention_tile_rows", "int", "Query rows per tile in eager NAR attention; nonpositive values choose automatic tiling to keep each score matrix under 3 GiB.", false, "0", std::nullopt},
        {"yue2.weight_type", "native|f32|f16|bf16|q8_0|q4_0|q4_k", "Shared weight storage type. Default: native.", false, "native"},
    };
    return out;
}

}  // namespace

Yue2Session::Yue2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Yue2Assets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(select_component_assets(require_assets(std::move(assets)), options.options)) {
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Yue2 supports only offline gen/music");
    }
    pipeline_ = std::make_unique<Yue2PipelineRuntime>(
        execution_context(),
        assets_,
        parse_weight_type(options, "yue2.model_weight_type", assets::TensorStorageType::Native),
        parse_weight_type(options, "yue2.vae_weight_type", assets::TensorStorageType::Native),
        runtime::parse_size_mb_option(options.options, {"yue2.model_weight_context_mb"}, 32ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.vae_weight_context_mb"}, 32ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.ar_prefill_graph_arena_mb"}, 32ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.ar_decode_graph_arena_mb"}, 32ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.nar_graph_arena_mb"}, 32ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.vae_graph_arena_mb"}, 32ull * 1024ull * 1024ull),
        attention_preference_from_options(options),
        runtime::parse_i64_option(options.options, {"yue2.attention_tile_rows"}).value_or(0));
}

Yue2Session::~Yue2Session() = default;

std::string Yue2Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Yue2Session::task_kind() const {
    return task_.task;
}

runtime::RunMode Yue2Session::run_mode() const {
    return task_.mode;
}

void Yue2Session::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.text.has_value() || !request.options.empty()) {
        (void) parse_yue2_preparation_request(request, assets_->config.generation);
    }
    mark_prepared();
}

runtime::TaskResult Yue2Session::run(const runtime::TaskRequest & request) {
    require_prepared("Yue2 run");
    const auto wall_start = Clock::now();
    const auto parsed = parse_yue2_request(request, assets_->config.generation);
    runtime::TaskResult result;
    auto run_result = pipeline_->run(parsed);
    result.audio_output = std::move(run_result.audio);
    if (!run_result.plan_abc_text.empty()) {
        result.output_artifacts.push_back(runtime::make_text_artifact(
            runtime::ArtifactKind::Custom,
            "score",
            run_result.plan_abc_text,
            {
                {"mime", "text/vnd.abc"},
                {"format", "abc"},
                {"extension", "abc"},
                {"source", "generated"},
                {"truncated", run_result.plan_abc_truncated ? "true" : "false"},
            }));
    }
    if (!run_result.semantic_codes.empty()) {
        result.output_artifacts.push_back(runtime::make_text_artifact(
            runtime::ArtifactKind::Custom,
            "semantic",
            semantic_codes_to_json(run_result.semantic_codes),
            {
                {"mime", "application/vnd.yue2.semantic+json"},
                {"format", "yue2-semantic-codec"},
                {"extension", "json"},
                {"frames", std::to_string(run_result.semantic_codes.size())},
                {"truncated", run_result.semantic_truncated ? "true" : "false"},
            }));
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_yue2_loader() {
    class LoadedModel final : public runtime::ILoadedVoiceModel {
    public:
        explicit LoadedModel(std::shared_ptr<const Yue2Assets> assets)
            : assets_(require_assets(std::move(assets))) {}

        const runtime::ModelMetadata & metadata() const noexcept override {
            return yue2_metadata();
        }

        const runtime::CapabilitySet & capabilities() const noexcept override {
            return yue2_capabilities();
        }

        std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
            const runtime::TaskSpec & task,
            const runtime::SessionOptions & options) const override {
            return create_yue2_session(task, options, assets_);
        }

    private:
        std::shared_ptr<const Yue2Assets> assets_;
    };

    class Loader final : public runtime::IVoiceModelLoader {
    public:
        std::string family() const override {
            return kFamily;
        }

        bool can_load(const runtime::ModelLoadRequest & request) const override {
            if (request.family_hint.has_value() && *request.family_hint != kFamily) {
                return false;
            }
            try {
                (void) load_yue2_assets(request.model_path, request.options);
                return true;
            } catch (const std::exception &) {
                return false;
            }
        }

        runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
            auto assets = load_yue2_assets(request.model_path, request.options);
            runtime::ModelInspection inspection;
            inspection.model_root = assets->model_root;
            inspection.metadata = yue2_metadata();
            inspection.capabilities = yue2_capabilities();
            inspection.cli = yue2_cli_interface();
            inspection.discovered_configs = runtime::discover_named_assets(
                inspection.model_root,
                inspection.metadata.config_candidates);
            inspection.discovered_weights = runtime::discover_named_assets(
                inspection.model_root,
                inspection.metadata.weight_candidates);
            return inspection;
        }

        std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
            return std::make_unique<LoadedModel>(load_yue2_assets(request.model_path, request.options));
        }

        runtime::CapabilitySet advertised_capabilities() const override {
            return yue2_capabilities();
        }
    };

    return std::make_shared<Loader>();
}

}  // namespace engine::models::yue2
