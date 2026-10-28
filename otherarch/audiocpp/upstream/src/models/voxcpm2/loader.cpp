#include "engine/models/voxcpm2/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/voxcpm2/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::voxcpm2 {
namespace {

runtime::CapabilitySet capabilities(const VoxCPM2Assets &) {
  runtime::CapabilitySet out;
  out.supported_tasks = {
      {runtime::VoiceTaskKind::Tts,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
      {runtime::VoiceTaskKind::VoiceCloning,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
      {runtime::VoiceTaskKind::VoiceDesign,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
  };
  out.languages = {"Auto"};
  out.supports_speaker_reference = true;
  return out;
}

runtime::ModelMetadata metadata(const VoxCPM2Assets &assets) {
  runtime::ModelMetadata out;
  out.family = "voxcpm2";
  out.variant = assets.config.architecture;
  out.description = "VoxCPM2 loaded from local safetensors assets.";
  return out;
}

runtime::ModelCliInterface cli(const VoxCPM2Assets &) {
  runtime::ModelCliInterface out;
  out.request_options = {
      {"min_tokens", "int", "Minimum generated audio tokens.", false, "2", "0"},
      {"max_tokens", "int", "Maximum generated audio tokens. Zero resolves to model max_length; requires min_tokens <= resolved max_tokens <= model limit. Default: 4096.", false, "4096", "0"},
      {"num_inference_steps", "int", "Flow sampling steps. Must be positive. Default: 10.", false, "10"},
      {"guidance_scale", "float", "Classifier-free guidance scale. Must be nonnegative. Default: 2.", false, "2"},
      {"retry_badcase", "bool", "Retry detected bad generations. Offline only; streaming requires false. Default: true.", false, "true"},
      {"retry_badcase_max_times", "int", "Total attempts including the initial attempt, not extra retries; must be positive. Default: 3.", false, "3"},
      {"retry_badcase_ratio_threshold", "float", "Bad-case audio/text length ratio threshold. Must be positive. Default: 6.", false, "6"},
      {"seed", "int", "Sampling seed.", false, "1234", "0", "4294967295"},
      {"cfm_noise_file", "path", "Precomputed flow noise file.", false},
      {"stream_chunk_policy", "grow|fixed", "Streaming decoder cadence: grow emits one patch first, then doubles toward stream_frames_per_event; fixed uses the configured size. Ignored offline.", false, "grow"},
      {"stream_frames_per_event", "int", "Maximum generated latent patches per streaming decoder event. Set to 1 for one-patch emission. Ignored offline.", false, "4", "1"},
      {"stream_left_context", "int", "Previous patches decoded as streaming left context.", false, "3", "0", "8"},
      {"prompt_text", "string", "Transcript for continuation prompt audio, distinct from the base voice reference.", false},
      {"chunk_strategy", "continuation|stateless", "Long-form continuation or independent generation per chunk. Offline only. Default: continuation.", false, "continuation"},
      {"text_chunk_size", "int", "Framework long-form text chunk size. Offline only. Default: 2048.", false, "2048"},
      {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode. Offline only. Default: tag_aware.", false, "tag_aware"},
  };
  out.session_options = {
      {"voxcpm2.lora", "path", "Native LoRA directory or checkpoint file with adjacent lora_config.json; merged into generator weights at session creation.", false},
      {"voxcpm2.lora_scale", "float", "Multiplier for the adapter's configured alpha/r scaling; default 1.", false, "1"},
      {"voxcpm2.weight_type", "native|f32|f16|bf16|q8_0", "Language-model weight storage; does not override AudioVAE storage.", false},
      {"voxcpm2.audiovae_weight_type", "native|f32|f16|bf16|q8_0", "AudioVAE weight storage; defaults independently to native, not the shared weight_type."},
      {"voxcpm2.prompt_cache_slots", "int", "Prompt and prompt-audio embedding cache slots; default 1.", false, "1", "0"},
      {"voxcpm2.mem_saver", "bool", "Use tighter graph workspaces and release request runtime graphs; default true.", false, "true"},
      {"voxcpm2.perf_mode", "enum", "off or optimized; optimized packs QKV and gate/up weights without changing their dtype. May change numerical results and generated audio.", false, "off"},
      {"voxcpm2.audiovae_tile_frames", "int", "Opt-in causal AudioVAE tiling; zero preserves full decoding.", false, "0", "0"},
      {"voxcpm2.weight_context_mb", "int", "Context or graph arena size in MiB.", false, "3072", "1"},
      {"voxcpm2.text_embedding_graph_arena_mb", "int", "Text embedding host graph-metadata arena size in MiB, not device workspace.", false, "64", "1"},
      {"voxcpm2.lm_step_graph_arena_mb", "int", "LM step host graph-metadata arena size in MiB, not device workspace.", false, "1024", "1"},
      {"voxcpm2.projection_graph_arena_mb", "int", "Projection host graph-metadata arena size in MiB, not device workspace.", false, "256", "1"},
      {"voxcpm2.local_encoder_graph_arena_mb", "int", "Local encoder host graph-metadata arena size in MiB, not device workspace.", false, "512", "1"},
      {"voxcpm2.dit_graph_arena_mb", "int", "DiT estimator host graph-metadata arena size in MiB, not device workspace.", false, "1024", "1"},
      {"voxcpm2.audiovae_weight_context_mb", "int", "Context or graph arena size in MiB.", false, "768", "1"},
      {"voxcpm2.audiovae_graph_arena_mb", "int", "AudioVAE decoder host graph-metadata arena size in MiB, not device workspace.", false, "1024", "1"},
      {"voxcpm2.audiovae_encoder_graph_arena_mb", "int", "AudioVAE encoder host graph-metadata arena size in MiB, not device workspace.", false, "1024", "1"},
      {"voxcpm2.audiovae_latent_capacity", "int", "Optional decoder graph reservation in latent frames, reused by shorter requests and doubled when exceeded; omitted builds the graph at each request's own frame count.", false, std::nullopt, "1"},
      {"voxcpm2.audiovae_encoder_sample_capacity", "int", "Reserved encoder sample capacity.", false, "240000", "1"},
      {"voxcpm2.exec_plan", "string", "JSON component execution plans. audiovae: impl=default/ggml; backend must match the session; no artifacts. params.threads is a nonnegative int: positive values require a CPU session and enable a dedicated AudioVAE backend with streaming decode overlap; omitted/0 shares the session backend.", false, "{}"},
  };
  return out;
}

class VoxCPM2Loader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "voxcpm2"; }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts,
         {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceCloning,
         {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceDesign,
         {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.supports_speaker_reference = true;
    return out;
  }

  std::string advertised_instructions_policy() const override {
    return "text_prefix";
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    try {
      (void)engine::model_spec::load_resource_bundle(
          request.model_path,
          engine::model_spec::default_spec_path(family()));
      return !request.family_hint.has_value() || *request.family_hint == family();
    } catch (...) {
      return false;
    }
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    const auto assets =
        load_voxcpm2_assets(request.model_path);
    runtime::ModelInspection inspection;
    inspection.model_root = assets->resources.model_root();
    inspection.metadata = metadata(*assets);
    inspection.capabilities = capabilities(*assets);
    inspection.cli = cli(*assets);
    const auto spec_path = engine::model_spec::default_spec_path(family());
    inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
        request.model_path,
        spec_path,
        engine::model_spec::ResourceKind::Files);
    inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
        request.model_path,
        spec_path,
        engine::model_spec::ResourceKind::Tensors);
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    return load_voxcpm2_model(request.model_path);
  }
};

} // namespace

VoxCPM2LoadedModel::VoxCPM2LoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const VoxCPM2Assets> assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata &VoxCPM2LoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
VoxCPM2LoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
VoxCPM2LoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  if (task.mode != runtime::RunMode::Offline &&
      task.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "VoxCPM2 only supports offline and streaming sessions");
  }
  if (task.task != runtime::VoiceTaskKind::Tts &&
      task.task != runtime::VoiceTaskKind::VoiceCloning &&
      task.task != runtime::VoiceTaskKind::VoiceDesign) {
    throw std::runtime_error("VoxCPM2 supports TTS, voice cloning, and voice design");
  }
  if (task.mode == runtime::RunMode::Streaming) {
    return std::make_unique<VoxCPM2StreamingSession>(task, options, assets_);
  }
  return std::make_unique<VoxCPM2OfflineSession>(task, options, assets_);
}

std::unique_ptr<VoxCPM2LoadedModel>
load_voxcpm2_model(const std::filesystem::path &model_path) {
  auto assets = load_voxcpm2_assets(model_path);
  return std::make_unique<VoxCPM2LoadedModel>(
      metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_voxcpm2_loader() {
  return std::make_shared<VoxCPM2Loader>();
}

} // namespace engine::models::voxcpm2

