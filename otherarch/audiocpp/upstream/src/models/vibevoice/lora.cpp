#include "engine/models/vibevoice/lora.h"

#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::vibevoice {
namespace {

constexpr std::string_view kLanguageModelPrefix = "model.language_model.";
constexpr std::string_view kPredictionHeadPrefix = "model.prediction_head.";
constexpr std::string_view kAcousticConnectorPrefix = "model.acoustic_connector.";
constexpr std::string_view kSemanticConnectorPrefix = "model.semantic_connector.";

void log_line(const std::string & message, engine::debug::LogLevel level = engine::debug::LogLevel::Info) {
    engine::debug::log_message(level, "vibevoice", message);
}

using OverrideTensor = assets::WeightReplacement;

std::filesystem::path resolve_adapter_dir(const std::filesystem::path & adapter_path) {
    if (engine::io::is_existing_directory(adapter_path)) {
        return adapter_path;
    }
    return adapter_path.parent_path();
}

// Preserve the upstream package locations; shared I/O selects the container.
std::shared_ptr<const assets::TensorSource> open_diffusion_head_source(const std::filesystem::path & adapter_dir) {
    for (const auto * name : {"diffusion_head/model", "diffusion_head_full", "diffusion_head/diffusion_head_full"}) {
        const auto logical = adapter_dir / name;
        if (const auto file = assets::find_tensor_file(logical.parent_path(), logical.filename().string())) {
            return assets::open_tensor_source(*file);
        }
    }
    return nullptr;
}

// Registers full-weight overrides for every adapter tensor that has a matching base tensor. A shape
// mismatch is a hard error (wrong model size); a missing base tensor is skipped like strict=False.
int add_full_overrides(
    const assets::TensorSource & base,
    const assets::TensorSource & source,
    std::string_view base_prefix,
    std::map<std::string, OverrideTensor> & overrides) {
    int count = 0;
    for (const auto & metadata : source.tensors()) {
        const std::string base_name = std::string(base_prefix) + metadata.name;
        if (!base.has_tensor(base_name)) {
            log_line("skip " + base_name + " (no matching base tensor)", engine::debug::LogLevel::Warning);
            continue;
        }
        if (base.require_metadata(base_name).shape != metadata.shape) {
            throw std::runtime_error(
                "VibeVoice fine-tune shape mismatch for " + base_name +
                " (adapter is trained for a different model size)");
        }
        OverrideTensor override_tensor;
        override_tensor.shape = metadata.shape;
        override_tensor.values = source.require_f32(metadata.name);
        overrides[base_name] = std::move(override_tensor);
        ++count;
    }
    return count;
}

void add_connector_overrides(
    const assets::TensorSource & base,
    const std::filesystem::path & adapter_dir,
    const char * subdir,
    std::string_view base_prefix,
    const char * label,
    std::map<std::string, OverrideTensor> & overrides) {
    const auto path = assets::find_tensor_file(adapter_dir / subdir, "pytorch_model");
    if (!path) {
        log_line(std::string(label) + ": not present, skipped");
        return;
    }
    const auto source = assets::open_tensor_source(*path);
    const int count = add_full_overrides(base, *source, base_prefix, overrides);
    log_line(std::string(label) + ": overrode " + std::to_string(count) + " tensors");
}

}  // namespace

std::shared_ptr<const assets::TensorSource> make_vibevoice_finetune_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path,
    float scale_override) {
    if (base == nullptr) {
        throw std::runtime_error("VibeVoice LoRA merge requires a base tensor source");
    }
    const auto prepare_started = std::chrono::steady_clock::now();
    const auto adapter_open_started = std::chrono::steady_clock::now();
    assets::WeightAdapterReadOptions read_options;
    read_options.require_scaling_metadata = true;
    if (scale_override > 0.0F) read_options.scale_override = scale_override;
    auto adapter = assets::read_weight_adapter(adapter_path, read_options);
    engine::debug::timing_log_scalar("vibevoice.lora.adapter_open_ms", engine::debug::elapsed_ms(adapter_open_started));
    if (adapter.updates.empty()) {
        throw std::runtime_error("VibeVoice adapter contains no weight updates: " + adapter_path.string());
    }
    if (!adapter.replacements.empty()) {
        throw std::runtime_error("VibeVoice full replacements must use the diffusion-head/connector sidecars");
    }
    const auto delta_load_started = std::chrono::steady_clock::now();
    assets::WeightAdapterBindings bindings;
    for (const auto & tensor : base->tensors()) {
        bindings.add(tensor.name, tensor.name);
        if (tensor.name.rfind(kLanguageModelPrefix, 0) == 0) {
            bindings.add(tensor.name.substr(kLanguageModelPrefix.size()), tensor.name);
        }
    }
    auto plan = assets::bind_weight_adapter(*base, std::move(adapter), bindings, assets::WeightMergeMode::RankWiseF32);
    engine::debug::timing_log_scalar("vibevoice.lora.delta_load_ms", engine::debug::elapsed_ms(delta_load_started));

    log_line("applying fine-tune adapter: " + adapter_path.string());
    log_line("LM adapter: merging " + std::to_string(plan.updates.size()) + " decoder modules");

    const auto adapter_dir = resolve_adapter_dir(adapter_path);
    auto & overrides = plan.replacements;
    if (const auto head = open_diffusion_head_source(adapter_dir)) {
        const auto override_started = std::chrono::steady_clock::now();
        const int count = add_full_overrides(*base, *head, kPredictionHeadPrefix, overrides);
        engine::debug::timing_log_scalar(
            "vibevoice.lora.diffusion_head_override_load_ms",
            engine::debug::elapsed_ms(override_started));
        log_line("diffusion head: overrode " + std::to_string(count) + " tensors");
    } else {
        log_line("diffusion head: not present, skipped");
    }
    const auto acoustic_started = std::chrono::steady_clock::now();
    add_connector_overrides(*base, adapter_dir, "acoustic_connector", kAcousticConnectorPrefix,
        "acoustic connector", overrides);
    engine::debug::timing_log_scalar(
        "vibevoice.lora.acoustic_connector_override_load_ms",
        engine::debug::elapsed_ms(acoustic_started));
    const auto semantic_started = std::chrono::steady_clock::now();
    add_connector_overrides(*base, adapter_dir, "semantic_connector", kSemanticConnectorPrefix,
        "semantic connector", overrides);
    engine::debug::timing_log_scalar(
        "vibevoice.lora.semantic_connector_override_load_ms",
        engine::debug::elapsed_ms(semantic_started));
    engine::debug::timing_log_scalar("vibevoice.lora.prepare_ms", engine::debug::elapsed_ms(prepare_started));
    engine::debug::timing_log_scalar("vibevoice.lora.decoder_delta_tensors", plan.updates.size());
    engine::debug::timing_log_scalar("vibevoice.lora.full_override_tensors", overrides.size());

    return assets::make_weight_adapter_source(std::move(base), std::move(plan), "vibevoice.lora");
}

std::shared_ptr<const VibeVoiceAssets> apply_vibevoice_finetune_options(
    std::shared_ptr<const VibeVoiceAssets> assets,
    const std::unordered_map<std::string, std::string> & options) {
    const auto lora_path = runtime::find_option(options, {"vibevoice.lora"});
    if (!lora_path.has_value() || lora_path->empty()) {
        return assets;
    }
    if (assets == nullptr) {
        throw std::runtime_error("VibeVoice fine-tune requires assets");
    }
    if (assets->fine_tune_applied) {
        throw std::runtime_error(
            "VibeVoice LoRA already applied via --load-option; do not also pass it via --session-option");
    }
    float scale_override = -1.0F;
    if (const auto value = runtime::parse_finite_float_option(options, {"vibevoice.lora_scale"})) {
        if (*value <= 0.0F) {
            throw std::runtime_error("VibeVoice vibevoice.lora_scale must be positive");
        }
        scale_override = *value;
    }
    auto updated = std::make_shared<VibeVoiceAssets>(*assets);
    updated->model_weights = make_vibevoice_finetune_source(assets->model_weights, *lora_path, scale_override);
    updated->fine_tune_applied = true;
    return updated;
}

}  // namespace engine::models::vibevoice
