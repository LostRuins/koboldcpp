#include "diffusion_engine.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <list>
#include <mutex>
#include <set>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/ggml_graph_cut.h"
#include "core/ggml_runner.h"
#include "core/ggml_tensor_utils.h"
#include "core/layer_split_partition.h"
#include "model.h"

#include "core/rng.hpp"
#include "core/rng_mt19937.hpp"
#include "core/rng_philox.hpp"
#include "core/util.h"
#include "model_builders.h"
#include "model_loader.h"
#include "model_manager.h"
#include "stable-diffusion.h"

#include "conditioning/conditioner.hpp"
#include "core/backend_fit.h"
#include "extensions/generation_extension.h"
#include "model/adapter/ip_adapter.hpp"
#include "model/adapter/lora.hpp"
#include "model/diffusion/animatediff.hpp"
#include "model/diffusion/control.hpp"
#include "model/diffusion/model.hpp"
#include "model/vae/audio_vae.hpp"
#include "model/vae/ltx_vae.hpp"
#include "model/vae/vae.hpp"
#include "runtime/denoiser.hpp"
#include "runtime/guidance.h"
#include "runtime/preview_interval.h"
#include "runtime/sample-cache.h"

#include "name_conversion.h"
#include "runtime/latent-preview.h"

#include <atomic>

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 1.x Inpaint",
    "Instruct-Pix2Pix",
    "SD 1.x Tiny UNet",
    "SD 2.x",
    "SD 2.x Inpaint",
    "SD 2.x Tiny UNet",
    "SDXS (512-DS)",
    "SDXS (09)",
    "SDXL",
    "SDXL Inpaint",
    "SDXL Instruct-Pix2Pix",
    "SDXL (Vega)",
    "SDXL (SSD1B)",
    "SVD",
    "SD3.x",
    "Flux",
    "Flux Fill",
    "Flux Control",
    "Flex.2",
    "Chroma Radiance",
    "Wan 2.x",
    "Wan 2.2 I2V",
    "Wan 2.2 TI2V",
    "LingBot Video",
    "Qwen Image",
    "Qwen Image Layered",
    "Hunyuan Video",
    "Anima",
    "Flux.2",
    "Flux.2 klein",
    "LTXAV",
    "MiniMax-H3",
    "HiDream O1",
    "Z-Image",
    "Boogu Image",
    "Ovis Image",
    "Ernie Image",
    "Lens",
    "MiniT2I",
    "Longcat-Image",
    "PiD",
    "Ideogram 4",
    "SeFi-Image",
    "Krea2",
    "Mage Flow",
    "SenseNova U1.5",
    "ESRGAN",
};

static_assert(VERSION_COUNT == sizeof(model_version_to_str) / sizeof(model_version_to_str[0]),
              "\nnumber of elements in model_version_to_str[] != VERSION_COUNT");

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120f,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

template <typename T, typename = void>
struct has_set_runtime_backends : std::false_type {};
template <typename T>
struct has_set_runtime_backends<T,
                                std::void_t<decltype(std::declval<T&>().set_runtime_backends(
                                    std::declval<const std::vector<ggml_backend_t>&>()))>> : std::true_type {};

static_assert(std::atomic<sd_cancel_mode_t>::is_always_lock_free,
              "sd_cancel_mode_t must be lock-free");

StableDiffusionGGML::StableDiffusionGGML()
    : rng(std::make_shared<PhiloxRNG>()),
      denoiser(std::make_shared<CompVisDenoiser>()) {}

StableDiffusionGGML::~StableDiffusionGGML() = default;

const std::map<StableDiffusionGGML::RunnerGroup, std::set<ModelComponent>>& StableDiffusionGGML::runner_components() {
    static const std::map<RunnerGroup, std::set<ModelComponent>> components{
        {RunnerGroup::Core, {ModelComponent::Conditioner, ModelComponent::Diffusion, ModelComponent::HighNoiseDiffusion, ModelComponent::CLIPVision, ModelComponent::IPAdapter}},
        {RunnerGroup::VAE, {ModelComponent::VAE, ModelComponent::PreviewVAE, ModelComponent::AudioVAE}},
        {RunnerGroup::ControlNet, {ModelComponent::ControlNet}},
        {RunnerGroup::Extensions, {ModelComponent::PhotoMaker, ModelComponent::PuLID}},
    };
    return components;
}

StableDiffusionGGML::RunnerGroups StableDiffusionGGML::all_runner_groups() {
    RunnerGroups groups;
    for (const auto& entry : runner_components()) {
        groups.insert(entry.first);
    }
    return groups;
}

ModelLoader::FileVersions StableDiffusionGGML::runner_source_versions(RunnerGroup group, const ModelLoader& loader) const {
    auto sources = model_manager->source_versions(runner_components().at(group), loader);
    if (group == RunnerGroup::Core) {
        // PhotoMaker's LoRA may already be merged into resident core weights.
        auto extra = loader.file_versions({"alphas_cumprod", "v_pred", "edm_vpred.", "pmid."});
        sources.insert(extra.begin(), extra.end());
    }
    return sources;
}

void StableDiffusionGGML::capture_runner_sources() {
    RunnerState state;
    state.catalog_revision = model_manager->loader().revision();
    for (const auto& entry : runner_components()) {
        state.sources[entry.first] = runner_source_versions(entry.first, model_manager->loader());
    }
    state.ready   = true;
    runner_state_ = std::move(state);
}

void StableDiffusionGGML::end_runners() {
    if (cond_stage_model)
        cond_stage_model->runner_end();
    if (diffusion_model)
        diffusion_model->runner_end();
    if (high_noise_diffusion_model)
        high_noise_diffusion_model->runner_end();
    if (clip_vision)
        clip_vision->runner_end();
    if (ip_adapter)
        ip_adapter->runner_end();
    if (first_stage_model)
        first_stage_model->runner_end();
    if (preview_vae)
        preview_vae->runner_end();
    if (audio_vae_model)
        audio_vae_model->runner_end();
    if (control_net)
        control_net->runner_end();
    for (auto& extension : generation_extensions)
        extension->runner_end();
    for (auto& lora : runtime_lora_models)
        if (lora.model)
            lora.model->runner_end();
}

bool StableDiffusionGGML::reset_runners(const RunnerGroups& groups) {
    end_runners();
    clear_lora_adapters();
    runtime_lora_models.clear();
    for (auto group : groups) {
        for (auto component : runner_components().at(group)) {
            if (!model_manager->unregister_param_tensors(component)) {
                return false;
            }
        }
    }
    for (auto group : groups) {
        switch (group) {
            case RunnerGroup::Core:
                cond_stage_model.reset();
                diffusion_model.reset();
                high_noise_diffusion_model.reset();
                clip_vision.reset();
                ip_adapter.reset();
                ip_adapter_tokens        = {};
                ip_adapter_uncond_tokens = {};
                runtime_lora_models.clear();
                break;
            case RunnerGroup::VAE:
                first_stage_model.reset();
                preview_vae.reset();
                audio_vae_model.reset();
                break;
            case RunnerGroup::ControlNet:
                control_net.reset();
                break;
            case RunnerGroup::Extensions:
                generation_extensions.clear();
                break;
        }
    }
    return true;
}

bool StableDiffusionGGML::refresh_model_sources() {
    bool changed;
    if (!model_manager->loader().files_changed(changed, false)) {
        return false;
    }
    if (!changed && runner_state_.ready && runner_state_.catalog_revision == model_manager->loader().revision()) {
        return true;
    }
    ModelLoader candidate = model_manager->loader();
    return candidate.refresh_files(false) && apply_model_update(std::move(candidate));
}

bool StableDiffusionGGML::apply_model_update(ModelLoader candidate,
                                             std::unique_ptr<ModelConfig> next_config,
                                             RunnerGroups groups) {
    const SDVersion next_version = candidate.get_sd_version();
    if (next_version == VERSION_COUNT) {
        LOG_ERROR("cannot identify updated diffusion model");
        return false;
    }
    if (!runner_state_.ready || next_version != version) {
        groups = all_runner_groups();
    } else {
        for (const auto& entry : runner_components()) {
            if (runner_state_.sources.at(entry.first) != runner_source_versions(entry.first, candidate)) {
                groups.insert(entry.first);
            }
        }
    }
    runner_state_.ready = false;
    if (!reset_runners(groups)) {
        return false;
    }
    if (!model_manager->set_loader(std::move(candidate))) {
        reset_runners(all_runner_groups());
        return false;
    }
    if (next_config) {
        config_ = std::move(next_config);
    }
    version = next_version;
    if (!build_runners(groups)) {
        reset_runners(all_runner_groups());
        return false;
    }
    capture_runner_sources();
    return true;
}

ggml_backend_t StableDiffusionGGML::backend_for(SDBackendModule module) {
    ggml_backend_t module_backend = backend_manager.runtime_backend(module);
    if (module_backend == nullptr) {
        LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
    }
    return module_backend;
}

ggml_backend_t StableDiffusionGGML::params_backend_for(SDBackendModule module) {
    ggml_backend_t module_backend = backend_manager.params_backend(module);
    if (module_backend == nullptr) {
        LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
    }
    return module_backend;
}

void StableDiffusionGGML::set_cancel_flag(enum sd_cancel_mode_t flag) {
    cancellation_flag.store(flag, std::memory_order_release);
}

void StableDiffusionGGML::reset_cancel_flag() {
    set_cancel_flag(SD_CANCEL_RESET);
}

enum sd_cancel_mode_t StableDiffusionGGML::get_cancel_flag() {
    return cancellation_flag.load(std::memory_order_acquire);
}

size_t StableDiffusionGGML::max_graph_vram_bytes_for_module(SDBackendModule module) {
    return max_vram_assignment.bytes_for_backend(backend_for(module));
}

std::vector<size_t> StableDiffusionGGML::layer_split_vram_limits_for_backends(const std::vector<ggml_backend_t>& backends) {
    std::vector<size_t> limits;
    limits.reserve(backends.size());
    for (ggml_backend_t backend : backends) {
        limits.push_back(max_vram_assignment.bytes_for_backend(backend));
    }
    return limits;
}

bool StableDiffusionGGML::ensure_backend_pair(SDBackendModule module) {
    if (backend_for(module) == nullptr) {
        return false;
    }
    return params_backend_for(module) != nullptr;
}

template <typename T>
bool StableDiffusionGGML::register_runner_params(ModelComponent component,
                                                 const std::shared_ptr<T>& model,
                                                 SDBackendModule module,
                                                 size_t* params_mem_size) {
    if (model == nullptr) {
        return true;
    }
    std::map<std::string, ggml_tensor*> group_tensors;
    std::map<ggml_tensor*, enum ggml_op> tensor_ops;
    model->get_param_tensors(group_tensors);
    if constexpr (std::is_base_of_v<Conditioner, T>) {
        model->get_param_tensor_ops(tensor_ops);
    }
    if (model_manager == nullptr) {
        return true;
    }
    ModelManager::ResidencyMode residency_mode =
        backend_manager.params_backend_is_disk(module) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend;

    std::vector<ggml_backend_t> module_backends = backend_manager.runtime_backends(module);
    if (module_backends.size() > 1) {
        if constexpr (has_set_runtime_backends<T>::value) {
            if (module == SDBackendModule::DIFFUSION || module == SDBackendModule::TE) {
                if (backend_manager.split_mode(module) == SDSplitMode::ROW) {
                    return register_row_split_runner_params(component,
                                                            model,
                                                            module,
                                                            module_backends,
                                                            std::move(group_tensors),
                                                            tensor_ops,
                                                            residency_mode,
                                                            params_mem_size);
                }
                return register_layer_split_runner_params(component,
                                                          model,
                                                          module,
                                                          module_backends,
                                                          std::move(group_tensors),
                                                          tensor_ops,
                                                          residency_mode,
                                                          params_mem_size);
            }
        }
        LOG_WARN("%s module does not support multiple runtime backends; using %s",
                 sd_backend_module_name(module),
                 sd::layer_split_backend_device_display_name(module_backends[0]).c_str());
    }
    return model_manager->register_param_tensors(component,
                                                 std::move(group_tensors),
                                                 residency_mode,
                                                 backend_for(module),
                                                 params_backend_for(module),
                                                 params_mem_size,
                                                 false,
                                                 false,
                                                 &tensor_ops);
}

template <typename T>
bool StableDiffusionGGML::register_row_split_runner_params(ModelComponent component,
                                                           const std::shared_ptr<T>& model,
                                                           SDBackendModule module,
                                                           const std::vector<ggml_backend_t>& module_backends,
                                                           std::map<std::string, ggml_tensor*> group_tensors,
                                                           const std::map<ggml_tensor*, enum ggml_op>& tensor_ops,
                                                           ModelManager::ResidencyMode residency_mode,
                                                           size_t* params_mem_size) {
    ggml_backend_t main_backend = module_backends[0];

    auto fall_back_to_layer_split = [&](const char* reason) {
        LOG_WARN("%s: row split unavailable (%s); falling back to layer split", model_component_name(component), reason);
        return register_layer_split_runner_params(component,
                                                  model,
                                                  module,
                                                  module_backends,
                                                  std::move(group_tensors),
                                                  tensor_ops,
                                                  residency_mode,
                                                  params_mem_size);
    };

    ggml_backend_dev_t main_dev = ggml_backend_get_device(main_backend);
    ggml_backend_reg_t reg      = main_dev != nullptr ? ggml_backend_dev_backend_reg(main_dev) : nullptr;
    if (reg == nullptr) {
        return fall_back_to_layer_split("no backend registry");
    }
    const size_t reg_dev_count = ggml_backend_reg_dev_count(reg);
    std::vector<float> tensor_split(reg_dev_count, 0.0f);
    constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
    for (ggml_backend_t backend : module_backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        int reg_index          = -1;
        for (size_t i = 0; i < reg_dev_count; i++) {
            if (ggml_backend_reg_dev_get(reg, i) == dev) {
                reg_index = (int)i;
                break;
            }
        }
        if (reg_index < 0) {
            return fall_back_to_layer_split("devices span different backend registries");
        }
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        int64_t usable_bytes    = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes,
                                                 (int64_t)free_bytes / 8);
        tensor_split[reg_index] = usable_bytes > 0 ? (float)((double)usable_bytes / (1024.0 * 1024.0)) : 1.0f;
    }

    ggml_backend_buffer_type_t split_buft = backend_manager.split_buffer_type(main_backend, tensor_split);
    if (split_buft == nullptr) {
        return fall_back_to_layer_split("backend has no split buffer type");
    }
    std::vector<std::pair<ggml_backend_t, size_t>> split_device_limits;
    for (auto backend : module_backends) {
        split_device_limits.emplace_back(backend, max_vram_assignment.bytes_for_backend(backend));
    }
    model_manager->set_split_buffer_type(main_backend, split_buft, split_device_limits);

    std::map<std::string, ggml_tensor*> split_tensors;
    if constexpr (std::is_base_of_v<Conditioner, T>) {
        model->get_layer_split_param_tensors(split_tensors);
    } else {
        split_tensors = group_tensors;
    }

    std::map<std::string, ggml_tensor*> row_split_map;
    std::map<std::string, ggml_tensor*> regular_map;
    size_t row_split_bytes = 0;
    for (const auto& kv : group_tensors) {
        if (split_tensors.count(kv.first) != 0 &&
            sd::layer_split_tensor_block_index(kv.first) >= 0 &&
            ModelManager::tensor_shape_supports_split_buffer(kv.second)) {
            row_split_map[kv.first] = kv.second;
            row_split_bytes += ggml_nbytes(kv.second);
        } else {
            regular_map[kv.first] = kv.second;
        }
    }
    if (row_split_map.empty()) {
        return fall_back_to_layer_split("no row-splittable transformer block weights found");
    }

    LOG_INFO("%s row split: %zu tensors (%.1f MB) split across %zu devices (main %s)",
             model_component_name(component),
             row_split_map.size(),
             row_split_bytes / (1024.f * 1024.f),
             module_backends.size(),
             sd::layer_split_backend_device_display_name(main_backend).c_str());

    if (!model_manager->register_param_tensors(component,
                                               std::move(row_split_map),
                                               residency_mode,
                                               main_backend,
                                               params_backend_for(module),
                                               params_mem_size,
                                               /*allow_split_buffer=*/true,
                                               false,
                                               &tensor_ops)) {
        return false;
    }
    return model_manager->register_param_tensors(component,
                                                 std::move(regular_map),
                                                 residency_mode,
                                                 main_backend,
                                                 params_backend_for(module),
                                                 params_mem_size,
                                                 false,
                                                 false,
                                                 &tensor_ops);
}

template <typename T>
bool StableDiffusionGGML::register_layer_split_runner_params(ModelComponent component,
                                                             const std::shared_ptr<T>& model,
                                                             SDBackendModule module,
                                                             const std::vector<ggml_backend_t>& module_backends,
                                                             std::map<std::string, ggml_tensor*> group_tensors,
                                                             const std::map<ggml_tensor*, enum ggml_op>& tensor_ops,
                                                             ModelManager::ResidencyMode residency_mode,
                                                             size_t* params_mem_size) {
    bool has_cpu_device = false;
    for (ggml_backend_t backend : module_backends) {
        has_cpu_device = has_cpu_device || sd_backend_is_cpu(backend);
    }
    if (has_cpu_device) {
        // The scheduler reserves the CPU slot for its fallback backend, and
        // CPU weight participation is what --params-backend <module>=cpu is
        // for; a CPU device in a split list is almost certainly a mistake.
        LOG_WARN(
            "%s: layer split across a CPU device is not supported; using %s "
            "(use --params-backend %s=cpu to keep weights in RAM)",
            model_component_name(component),
            sd::layer_split_backend_device_display_name(module_backends[0]).c_str(),
            sd_backend_module_name(module));
        return model_manager->register_param_tensors(component,
                                                     std::move(group_tensors),
                                                     residency_mode,
                                                     module_backends[0],
                                                     params_backend_for(module),
                                                     params_mem_size,
                                                     false,
                                                     false,
                                                     &tensor_ops);
    }

    model->set_runtime_backends(module_backends);
    model->set_graph_cut_layer_split_backend_vram_limits(layer_split_vram_limits_for_backends(module_backends));
    model->set_graph_cut_layer_split_enabled(true);
    const bool params_follow_runtime = backend_manager.params_backend_follows_runtime(module) ||
                                       backend_manager.params_backend_is_disk(module);
    ggml_backend_t initial_params_backend = params_follow_runtime ? module_backends[0] : params_backend_for(module);
    if (initial_params_backend == nullptr) {
        return false;
    }

    LOG_INFO("%s graph-cut layer split: deferring %zu tensors across %zu runtime backends until first graph",
             model_component_name(component),
             group_tensors.size(),
             module_backends.size());

    return model_manager->register_param_tensors(component,
                                                 std::move(group_tensors),
                                                 residency_mode,
                                                 module_backends[0],
                                                 initial_params_backend,
                                                 params_mem_size,
                                                 false,
                                                 params_follow_runtime,
                                                 &tensor_ops);
}

bool StableDiffusionGGML::unload_control_net() {
    ContextOperation operation(*this);
    if (!operation.acquired) {
        return false;
    }
    if (model_manager == nullptr || config_ == nullptr) {
        LOG_ERROR("cannot unload ControlNet: context is not initialized");
        return false;
    }
    ModelLoader candidate = model_manager->loader();
    if (config_->control_net_file != 0 && !candidate.del_file(config_->control_net_file)) {
        return false;
    }
    auto next_config = std::make_unique<ModelConfig>(*config_);
    next_config->set_control_net(0, "");
    return apply_model_update(std::move(candidate), std::move(next_config), {RunnerGroup::ControlNet});
}

bool StableDiffusionGGML::load_control_net_from_file(const std::string& path) {
    ContextOperation operation(*this);
    if (!operation.acquired) {
        return false;
    }
    if (path.empty() || model_manager == nullptr || config_ == nullptr) {
        LOG_ERROR("cannot load ControlNet: invalid path or uninitialized context");
        return false;
    }
    ModelLoader candidate = model_manager->loader();
    ModelLoader::FileId file_id;
    if (!candidate.add_file(path, "", &file_id)) {
        return false;
    }
    if (config_->control_net_file != 0 && config_->control_net_file != file_id && !candidate.del_file(config_->control_net_file)) {
        return false;
    }
    auto next_config = std::make_unique<ModelConfig>(*config_);
    next_config->set_control_net(file_id, path);
    return apply_model_update(std::move(candidate), std::move(next_config), {RunnerGroup::ControlNet});
}

bool StableDiffusionGGML::init_backend() {
    std::string error;
    if (!backend_manager.init(backend_spec.c_str(),
                              params_backend_spec.c_str(),
                              split_mode_spec.c_str(),
                              &error)) {
        LOG_ERROR("backend config failed: %s", error.c_str());
        return false;
    }
    return ensure_backend_pair(SDBackendModule::DIFFUSION);
}

bool StableDiffusionGGML::row_split_active() {
    for (SDBackendModule module : {SDBackendModule::DIFFUSION, SDBackendModule::TE}) {
        if (backend_manager.split_mode(module) == SDSplitMode::ROW &&
            backend_manager.runtime_backends(module).size() > 1) {
            return true;
        }
    }
    return false;
}

bool StableDiffusionGGML::graph_cut_layer_split_active() {
    for (SDBackendModule module : {SDBackendModule::DIFFUSION, SDBackendModule::TE}) {
        if (backend_manager.split_mode(module) == SDSplitMode::LAYER &&
            backend_manager.runtime_backends(module).size() > 1) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<RNG> StableDiffusionGGML::get_rng(rng_type_t rng_type) {
    if (rng_type == STD_DEFAULT_RNG) {
        return std::make_shared<STDDefaultRNG>();
    } else if (rng_type == CPU_RNG) {
        return std::make_shared<MT19937RNG>();
    } else {  // default: CUDA_RNG
        return std::make_shared<PhiloxRNG>();
    }
}

void StableDiffusionGGML::refresh_compvis_denoiser_sigmas() {
    auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
    if (!comp_vis_denoiser) {
        return;
    }
    std::vector<float> alphas_cumprod(TIMESTEPS);
    if (file_alphas_cumprod.size() == TIMESTEPS) {
        alphas_cumprod = file_alphas_cumprod;
    } else {
        calculate_alphas_cumprod(alphas_cumprod.data());
    }
    for (int i = 0; i < TIMESTEPS; i++) {
        comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
        comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
    }
}

void StableDiffusionGGML::load_alphas_cumprod() {
    file_alphas_cumprod.clear();

    std::vector<float> loaded_alphas;
    if (!model_manager->load_float_tensor("alphas_cumprod", loaded_alphas)) {
        return;
    }
    if (loaded_alphas.size() != TIMESTEPS) {
        LOG_WARN("ignore alphas_cumprod from model file: expected %d values, got %zu",
                 TIMESTEPS,
                 loaded_alphas.size());
        return;
    }
    for (float alpha : loaded_alphas) {
        if (!std::isfinite(alpha) || alpha <= 0.0f || alpha > 1.0f) {
            LOG_WARN("ignore invalid alphas_cumprod from model file");
            return;
        }
    }

    file_alphas_cumprod = std::move(loaded_alphas);
    LOG_VERBOSE("loaded alphas_cumprod from model file");
}

bool StableDiffusionGGML::init_model_loader(ModelLoader& model_loader, ModelConfig& configuration) {
    auto* sd_ctx_params       = &configuration.params; // kcpp make ctx params mutable
    auto& use_tae             = configuration.use_tae;
    auto& use_audio_vae       = configuration.use_audio_vae;

    if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
        LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
        if (!model_loader.init_from_file(sd_ctx_params->model_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", sd_ctx_params->model_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) > 0) {
        LOG_INFO("loading diffusion model from '%s'", sd_ctx_params->diffusion_model_path);
        if (!model_loader.init_from_file(sd_ctx_params->diffusion_model_path, "model.diffusion_model.")) {
            LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->diffusion_model_path);
        }
    }

    {   // begin kcpp replacements
        auto& p = *sd_ctx_params;
        SDVersion tempver = model_loader.get_sd_version();
        bool fallback_swapped = false;

        auto path_empty = [](const char* path) -> bool {
            return path == nullptr || *path == '\0';
        };

        // kcpp fallback to separate diffusion model passed as model
        if (!path_empty(p.model_path) &&
        path_empty(p.diffusion_model_path) &&
        (!path_empty(p.t5xxl_path)||!path_empty(p.clip_l_path)))
        {
            bool endswithsafetensors = ends_with(p.model_path, ".safetensors");
            if(endswithsafetensors && !model_loader.has_diffusion_model_tensors())
            {
                LOG_INFO("SD Diffusion Model tensors missing! Fallback trying alternative tensor names...\n");
                if (!model_loader.init_from_file(p.model_path, "model.diffusion_model.")) {
                    LOG_WARN("loading diffusion model from '%s' failed", p.model_path);
                }
                fallback_swapped = true;
                tempver = model_loader.get_sd_version();
            }
        }

        if (tempver == VERSION_ANIMA && !fallback_swapped &&
            !path_empty(p.model_path) &&
            path_empty(p.diffusion_model_path) &&
            !model_loader.has_diffusion_model_tensors()
            )
        {
            LOG_INFO("Anima: SD Diffusion Model tensors missing! Fallback trying alternative tensor names...\n");
            if (!model_loader.init_from_file(p.model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", p.model_path);
            }
            tempver = model_loader.get_sd_version();
        }

        auto toLowerCase = [](const std::string& str) -> std::string {
            std::string result;
            std::locale loc;
            for (char ch : str) {
                result += std::tolower(ch, loc); // Use locale-aware tolower
            }
            return result;
        };

        bool iswan = sd_version_is_wan(tempver);
        bool is_wan21 = sd_version_is_wan(tempver) && tempver != VERSION_WAN2_2_TI2V;
        bool is_qwenimg = sd_version_is_qwen_image(tempver);
        bool iszimg = sd_version_is_z_image(tempver);
        bool isflux2 = sd_version_is_flux2(tempver);
        bool is_ovis =  (tempver==VERSION_OVIS_IMAGE);
        bool is_anima = sd_version_is_anima(tempver);
        bool is_ernie = sd_version_is_ernie_image(tempver);
        bool is_longcat = sd_version_is_longcat(tempver);
        bool is_lens = sd_version_is_lens(tempver);
        bool is_ltx = sd_version_is_ltxav(tempver);
        bool is_ideogram = sd_version_is_ideogram4(tempver);
        bool is_boogu = sd_version_is_boogu_image(tempver);
        bool is_krea2 = sd_version_is_krea2(tempver);
        bool is_sefi = sd_version_is_sefi_image(tempver);
        bool is_mageflow = sd_version_is_mage_flow(tempver);
        bool is_minimaxh3 = sd_version_is_minimax_h3(tempver);
        bool conditioner_is_llm = (is_qwenimg || iszimg || isflux2 || is_ovis || is_anima || is_ernie || is_longcat || is_lens || is_ltx || is_ideogram || is_boogu || is_krea2 || is_sefi || is_mageflow || is_minimaxh3);
        bool has_llm_vision = (is_qwenimg || is_longcat || is_boogu);

        //kcpp qol fallback: if a llm was loaded as t5 by mistake
        if(conditioner_is_llm && !path_empty(p.t5xxl_path))
        {
            if(path_empty(p.clip_l_path))
            {
                std::swap(p.clip_l_path, p.t5xxl_path);
            }
            else if(path_empty(p.clip_g_path))
            {
                //very tricky case. see if we can tell if clipl is an mmproj, if so move to right place
                if(toLowerCase(p.clip_l_path).find("mmproj") != std::string::npos || is_ltx)
                {
                    // g = l, l = t, t = ""
                    std::swap(p.clip_g_path, p.clip_l_path);
                    std::swap(p.clip_l_path, p.t5xxl_path);
                }
            }
        }

        //settle clip-l replacements
        if (!path_empty(p.clip_l_path))
        {
            if(conditioner_is_llm && path_empty(p.llm_path))
            {
                std::swap(p.llm_path, p.clip_l_path);
            }
            else if(iswan)
            {
                if(path_empty(p.t5xxl_path))
                {
                    std::swap(p.t5xxl_path, p.clip_l_path);
                } else if (path_empty(p.clip_vision_path)) {
                    std::swap(p.clip_vision_path, p.clip_l_path);
                }
            }
        }

        //settle clip-g replacements
        if (!path_empty(p.clip_g_path))
        {
            if(iswan && path_empty(p.clip_vision_path))
            {
                std::swap(p.clip_vision_path, p.clip_g_path);
            }
            else if(has_llm_vision && path_empty(p.llm_vision_path))
            {
                std::swap(p.llm_vision_path, p.clip_g_path);
            }
            else if(is_ltx)
            {
                std::swap(p.embeddings_connectors_path, p.clip_g_path);
            }
            else if(is_ideogram)
            {
                std::swap(p.uncond_diffusion_model_path, p.clip_g_path);
            }
        }

        //settle possible inversions for mmproj
        if(!path_empty(p.llm_vision_path) && !path_empty(p.llm_path))
        {
            if(toLowerCase(p.llm_vision_path).find("mmproj") == std::string::npos &&
            toLowerCase(p.llm_path).find("mmproj") != std::string::npos)
            {
                std::swap(p.llm_path, p.llm_vision_path);
            }
        }

        //settle tae replacements
        kcpp_taesd_path = SAFE_STR(sd_ctx_params->taesd_path);
        if(kcpp_taesd_path != "")
        {
            std::string to_search = "taesd.embd";
            std::string to_replace = "";
            if(sd_version_is_sd1(tempver) || sd_version_is_sd2(tempver))
            {
                to_replace = "taesd.embd";
            }
            else if(sd_version_is_sdxl(tempver))
            {
                to_replace = "taesd_xl.embd";
            }
            else if(sd_version_uses_flux_vae(tempver))
            {
                to_replace = "taesd_f.embd";
            }
            else if(sd_version_is_sd3(tempver))
            {
                to_replace = "taesd_3.embd";
            }
            else if(sd_version_uses_flux2_vae(tempver))
            {
                to_replace = "taesd_f2.embd";
            }
            else if(sd_version_uses_wan_vae(tempver))
            {
                to_replace = "taesd_w21.embd";
            }

            if(to_replace!="")
            {
                size_t pos = kcpp_taesd_path.find(to_search);
                if (pos != std::string::npos) {
                    kcpp_taesd_path.replace(pos, to_search.length(), to_replace);
                }
            }
            else
            {
                printf("\nCannot use TAESD: Unknown tempver %d. TAESD Disabled!\n",tempver);
                kcpp_taesd_path = "";
            }
            if (kcpp_taesd_path != "" && !file_exists(kcpp_taesd_path))
            {
                printf("\nCannot use TAESD: \"%s\" not found. TAESD Disabled!\n", kcpp_taesd_path.c_str());
                kcpp_taesd_path = "";
            }
        }
        if (!path_empty(p.photo_maker_path) > 0 && tempver != VERSION_SDXL) {
            printf("\nWARNING: PhotoMaker is only compatible with SDXL models. PhotoMaker will be disabled!\n");
            p.photo_maker_path = "";
        }

        //for models with TAE suppport, if vae is set, tae is off, and it looks like a tae, swap to tae
        if(kcpp_taesd_path=="" && !path_empty(p.vae_path) && toLowerCase(p.vae_path).rfind("tae")!=std::string::npos)
        {
            try {
                const uintmax_t tae_size_limit = 64 * 1024 * 1024; //if its less than 64mb, it might be a TAE
                // Get the file size in bytes cross-platform
                uintmax_t size = std::filesystem::file_size(p.vae_path);
                if (size > 0 && size < tae_size_limit) {
                    printf("\nVAE appears to be a TAE, loading as TAE instead!\n");
                    kcpp_taesd_path = p.vae_path;
                    p.vae_path = "";
                }
            }
            catch (const std::filesystem::filesystem_error& e) {
                std::printf("Error accessing file: %s\n", e.what());
            }
        }

        p.taesd_path = kcpp_taesd_path.c_str();

        // patch hidream to fix broken images on vulkan
        // https://github.com/leejet/stable-diffusion.cpp/issues/1496
        if(tempver == VERSION_HIDREAM_O1 && path_empty(p.tensor_type_rules))
        {
            p.tensor_type_rules = "^model.language_model.layers.[0-9]+.mlp.down_proj.weight=bf16";
        }

        //debug print
        // printf("\n\nclip_g: %s\nclip_l: %s\nclip_vision: %s\nllm: %s\nllm_vision: %s\nt5xxl: %s\ntaesd: %s\n",
        // p.clip_g_path, p.clip_l_path, p.clip_vision_path,
        // p.llm_path, p.llm_vision_path, p.t5xxl_path,
        // p.taesd_path);
        // end kcpp replacements
    }

    if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
        LOG_INFO("loading high noise diffusion model from '%s'", sd_ctx_params->high_noise_diffusion_model_path);
        if (!model_loader.init_from_file(sd_ctx_params->high_noise_diffusion_model_path, "model.high_noise_diffusion_model.")) {
            LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->high_noise_diffusion_model_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->uncond_diffusion_model_path)) > 0) {
        LOG_INFO("loading unconditional diffusion model from '%s'", sd_ctx_params->uncond_diffusion_model_path);
        if (!model_loader.init_from_file(sd_ctx_params->uncond_diffusion_model_path, "model.diffusion_model.uncond.")) {
            LOG_WARN("loading unconditional diffusion model from '%s' failed", sd_ctx_params->uncond_diffusion_model_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
        LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
        if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, "clip_l.")) {
            LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
        LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
        if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, "clip_g.")) {
            LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
        LOG_INFO("loading clip_vision from '%s'", sd_ctx_params->clip_vision_path);
        if (!model_loader.init_from_file(sd_ctx_params->clip_vision_path, "clip_vision.")) {
            LOG_WARN("loading clip_vision from '%s' failed", sd_ctx_params->clip_vision_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
        LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
        if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
            LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->pulid_weights_path)) > 0) {
        LOG_INFO("loading PuLID weights from '%s'", sd_ctx_params->pulid_weights_path);
        if (!model_loader.init_from_file(sd_ctx_params->pulid_weights_path,
                                         "model.diffusion_model.")) {
            LOG_WARN("loading PuLID weights from '%s' failed", sd_ctx_params->pulid_weights_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->llm_path)) > 0) {
        LOG_INFO("loading llm from '%s'", sd_ctx_params->llm_path);
        if (!model_loader.init_from_file(sd_ctx_params->llm_path, "text_encoders.llm.")) {
            LOG_WARN("loading llm from '%s' failed", sd_ctx_params->llm_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->llm_vision_path)) > 0) {
        LOG_INFO("loading llm vision from '%s'", sd_ctx_params->llm_vision_path);
        if (!model_loader.init_from_file(sd_ctx_params->llm_vision_path, "text_encoders.llm.visual.")) {
            LOG_WARN("loading llm vision from '%s' failed", sd_ctx_params->llm_vision_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
        LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
        if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
            LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
            external_vae_is_invalid = true;
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0) {
        LOG_INFO("loading tae from '%s'", sd_ctx_params->taesd_path);
        if (!model_loader.init_from_file(sd_ctx_params->taesd_path, "tae.")) {
            LOG_WARN("loading tae from '%s' failed", sd_ctx_params->taesd_path);
        } else {
            use_tae = true;
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->embeddings_connectors_path)) > 0) {
        LOG_INFO("loading embeddings connectors from '%s'", sd_ctx_params->embeddings_connectors_path);
        if (!model_loader.init_from_file(sd_ctx_params->embeddings_connectors_path)) {
            LOG_WARN("loading embeddings connectors from '%s' failed", sd_ctx_params->embeddings_connectors_path);
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->audio_vae_path)) > 0) {
        LOG_INFO("loading audio VAE from '%s'", sd_ctx_params->audio_vae_path);
        if (!model_loader.init_from_file(sd_ctx_params->audio_vae_path)) {
            LOG_WARN("loading audio VAE weights from '%s' failed", sd_ctx_params->audio_vae_path);
        } else {
            use_audio_vae = true;
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->motion_module_path)) > 0) {
        LOG_INFO("loading motion module (AnimateDiff) from '%s'", sd_ctx_params->motion_module_path);
        if (!model_loader.init_from_file(sd_ctx_params->motion_module_path,
                                         "model.diffusion_model.motion_module.")) {
            LOG_WARN("loading motion module from '%s' failed", sd_ctx_params->motion_module_path);
        } else {
            configuration.animatediff_loaded = true;
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
        if (!model_loader.add_file(sd_ctx_params->control_net_path, "", &configuration.control_net_file)) {
            LOG_ERROR("init control net model loader from file failed: '%s'", sd_ctx_params->control_net_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->ip_adapter_path)) > 0) {
        if (!model_loader.init_from_file(sd_ctx_params->ip_adapter_path)) {
            LOG_ERROR("init ip-adapter model loader from file failed: '%s'", sd_ctx_params->ip_adapter_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(sd_ctx_params->photo_maker_path)) > 0) {
        configuration.photomaker_source_available = model_loader.add_file(sd_ctx_params->photo_maker_path, "pmid.");
        if (!configuration.photomaker_source_available) {
            LOG_WARN("loading stacked ID embedding from '%s' failed", sd_ctx_params->photo_maker_path);
        }
    }

    model_loader.convert_tensors_name();

    ggml_type wtype               = sd_type_to_ggml_type(sd_ctx_params->wtype);
    std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
    if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
        model_loader.set_wtype_override(wtype, tensor_type_rules);
    }

    return true;
}

bool StableDiffusionGGML::init(const sd_ctx_params_t* sd_ctx_params) {
    auto configuration        = std::make_unique<ModelConfig>(*sd_ctx_params);
    n_threads                 = sd_ctx_params->n_threads;
    enable_mmap               = sd_ctx_params->enable_mmap;
    disable_prefetch          = sd_ctx_params->disable_prefetch;
    disable_segmented_compute = sd_ctx_params->disable_segmented_compute;
    eager_load                = sd_ctx_params->eager_load;
    backend_spec              = SAFE_STR(sd_ctx_params->backend);
    params_backend_spec       = SAFE_STR(sd_ctx_params->params_backend);
    split_mode_spec           = SAFE_STR(sd_ctx_params->split_mode);
    auto_fit_enabled          = sd_ctx_params->auto_fit && backend_spec.empty() && params_backend_spec.empty();
    max_vram_assignment.reset(0.f);
    {
        std::string error;
        if (!max_vram_assignment.parse(SAFE_STR(sd_ctx_params->max_vram), &error)) {
            LOG_ERROR("%s", error.c_str());
            return false;
        }
    }

    std::string rpc_servers_spec = SAFE_STR(sd_ctx_params->rpc_servers);
    add_rpc_devices(rpc_servers_spec);

    rng = get_rng(sd_ctx_params->rng_type);
    if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
        sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
    } else {
        sampler_rng = rng;
    }

    ggml_log_set(sd_ggml_log_callback, nullptr);

    model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(n_threads);
    model_manager->set_enable_mmap(enable_mmap);
    model_manager->set_segmented_compute_disabled(disable_segmented_compute);
    model_manager->set_prefetch_disabled(disable_prefetch);
    ModelLoader model_loader;

    if (!init_model_loader(model_loader, *configuration)) {
        return false;
    }

    version = model_loader.get_sd_version();
    if (version == VERSION_COUNT) {
        LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
        return false;
    } else {
        LOG_INFO("Version: %s ", model_version_to_str[version]);
    }

    if (auto_fit_enabled) {
        if (!sd::backend_fit::derive_backend_specs(model_loader,
                                                   sd_type_to_ggml_type(sd_ctx_params->wtype),
                                                   max_vram_assignment,
                                                   backend_spec,
                                                   params_backend_spec)) {
            return false;
        }
    }

    if (!init_backend()) {
        return false;
    }
    {
        std::string error;
        if (!max_vram_assignment.canonicalize_backend_keys(&error)) {
            LOG_ERROR("%s", error.c_str());
            return false;
        }
    }
    if (eager_load && graph_cut_layer_split_active()) {
        LOG_WARN("--eager-load is not supported with graph-cut layer split; weights will be prepared lazily");
        eager_load = false;
    }

    diffusion_conv_direct = sd_ctx_params->diffusion_conv_direct;
    return apply_model_update(std::move(model_loader), std::move(configuration), all_runner_groups());
}

bool StableDiffusionGGML::uses_tae() const {
    return config_->use_tae || version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09;
}

bool StableDiffusionGGML::tae_preview_only() const {
    return config_->params.tae_preview_only && version != VERSION_SDXS_512_DS && version != VERSION_SDXS_09;
}

void StableDiffusionGGML::configure_weight_loading() {
    const auto* sd_ctx_params = &config_->params;
    const auto& model_loader  = model_manager->loader();
    const auto wtype_stat     = model_loader.get_wtype_stat();
    bool have_int8_tensorwise = false;
    for (const auto& [_, tensor_storage] : model_loader.get_tensor_storage_map()) {
        if (tensor_storage.is_int8_tensorwise) {
            have_int8_tensorwise = true;
            break;
        }
    }

    if (sd_ctx_params->lora_apply_mode == LORA_APPLY_AUTO) {
        bool have_quantized_weight = have_int8_tensorwise;
        for (const auto& [type, _] : wtype_stat) {
            if (ggml_is_quantized(type)) {
                have_quantized_weight = true;
                break;
            }
        }
        // Avoid full-model LoRA merge buffers on constrained setups.
        const bool params_offloaded      = params_backend_for(SDBackendModule::DIFFUSION) != backend_for(SDBackendModule::DIFFUSION);
        const bool streaming_constrained = params_offloaded ||
                                           backend_manager.params_backend_is_disk(SDBackendModule::DIFFUSION);
        if (have_quantized_weight || streaming_constrained || row_split_active()) {
            apply_lora_immediately = false;
        } else {
            apply_lora_immediately = true;
        }
    } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
        if (have_int8_tensorwise) {
            LOG_WARN(
                "INT8 tensorwise weights do not support the immediately LoRA apply mode; "
                "using at_runtime instead");
            apply_lora_immediately = false;
        } else if (row_split_active()) {
            LOG_WARN(
                "row-split tensors do not support the immediately LoRA apply mode; "
                "LoRAs will not be applied to them (use --lora-apply-mode at_runtime)");
            apply_lora_immediately = false;
        } else {
            apply_lora_immediately = true;
        }
    } else {
        apply_lora_immediately = false;
    }

    bool needs_writable_mmap = enable_mmap && apply_lora_immediately;
    model_manager->set_writable_mmap(needs_writable_mmap);
    if (enable_mmap && apply_lora_immediately) {
        LOG_WARN("in mode 'immediately', LoRAs will cause extra memory usage with mmap");
    }
    model_manager->prepare_file_io();
    load_alphas_cumprod();
}

sd::model_builders::Context StableDiffusionGGML::model_build_context() {
    return {config_->params, version, model_manager->loader().get_tensor_storage_map(), backend_manager, model_manager};
}

bool StableDiffusionGGML::build_core_runners() {
    sd::model_builders::CoreRunners runners;
    if (!sd::model_builders::build_core_runners(model_build_context(), runners)) {
        return false;
    }
    cond_stage_model           = std::move(runners.conditioner);
    diffusion_model            = std::move(runners.diffusion);
    high_noise_diffusion_model = std::move(runners.high_noise_diffusion);
    clip_vision                = std::move(runners.clip_vision);
    ip_adapter                 = std::move(runners.ip_adapter);

    cond_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::TE));
    diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::DIFFUSION));
    if (high_noise_diffusion_model) {
        high_noise_diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::DIFFUSION));
    }
    if (clip_vision) {
        clip_vision->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::CLIP_VISION));
    }
    return register_runner_params(ModelComponent::Conditioner, cond_stage_model, SDBackendModule::TE) &&
           register_runner_params(ModelComponent::Diffusion, diffusion_model, SDBackendModule::DIFFUSION) &&
           register_runner_params(ModelComponent::HighNoiseDiffusion, high_noise_diffusion_model, SDBackendModule::DIFFUSION) &&
           register_runner_params(ModelComponent::CLIPVision, clip_vision, SDBackendModule::CLIP_VISION) &&
           register_runner_params(ModelComponent::IPAdapter, ip_adapter, SDBackendModule::DIFFUSION);
}

bool StableDiffusionGGML::build_vae_runners() {
    sd::model_builders::VAEOptions options;
    options.use_tae                 = uses_tae();
    options.tae_preview_only        = tae_preview_only();
    options.use_audio_vae           = config_->use_audio_vae;
    options.external_vae_is_invalid = external_vae_is_invalid;
    sd::model_builders::VAERunners runners;
    if (!sd::model_builders::build_vae_runners(model_build_context(), options, runners)) {
        return false;
    }
    first_stage_model = std::move(runners.vae);
    preview_vae       = std::move(runners.preview);
    audio_vae_model   = std::move(runners.audio);

    first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::VAE));
    if (preview_vae) {
        preview_vae->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::VAE));
    }
    return register_runner_params(ModelComponent::VAE, first_stage_model, SDBackendModule::VAE) &&
           register_runner_params(ModelComponent::PreviewVAE, preview_vae, SDBackendModule::VAE) &&
           register_runner_params(ModelComponent::AudioVAE, audio_vae_model, SDBackendModule::VAE);
}

bool StableDiffusionGGML::build_control_net_runner() {
    if (config_->control_net_file == 0) {
        return true;
    }
    if (!sd::model_builders::build_control_net_runner(model_build_context(), control_net)) {
        return false;
    }
    control_net->set_max_graph_vram_bytes(max_graph_vram_bytes_for_module(SDBackendModule::CONTROL_NET));
    return register_runner_params(ModelComponent::ControlNet, control_net, SDBackendModule::CONTROL_NET);
}

bool StableDiffusionGGML::build_extension_runners() {
    GenerationExtensionInitContext extension_ctx{
        &config_->params,
        version,
        model_manager->loader().get_tensor_storage_map(),
        config_->photomaker_source_available,
        model_manager,
        n_threads,
        [this](SDBackendModule module) { return ensure_backend_pair(module); },
        [this](SDBackendModule module) { return backend_for(module); },
        [this](SDBackendModule module) { return params_backend_for(module); },
    };
    if (!sd::model_builders::build_extension_runners(extension_ctx, generation_extensions)) {
        return false;
    }
    for (auto& extension : generation_extensions) {
        if (!register_runner_params(extension->component(), extension, SDBackendModule::PHOTOMAKER)) {
            return false;
        }
    }
    return true;
}

bool StableDiffusionGGML::validate_and_load_runners() {
    const auto* sd_ctx_params   = &config_->params;
    const bool use_tae          = uses_tae();
    const bool tae_preview_only = this->tae_preview_only();
    if (sd_ctx_params->flash_attn) {
        LOG_INFO("Using flash attention");
        if(!sd_version_is_qwen_image(version)) //kcpp: edit 14feb, breaks qwen image edit
        {
        cond_stage_model->set_flash_attention_enabled(true);
        }
        if (clip_vision) {
            clip_vision->set_flash_attention_enabled(true);
        }
        if (first_stage_model) {
            first_stage_model->set_flash_attention_enabled(true);
        }
        if (preview_vae) {
            preview_vae->set_flash_attention_enabled(true);
        }
    }

    if (sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) {
        LOG_INFO("Using flash attention in the diffusion model");
        diffusion_model->set_flash_attention_enabled(true);
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->set_flash_attention_enabled(true);
        }
    }
    LOG_VERBOSE("validating model metadata");

    std::set<std::string> ignore_tensors;
    if (use_tae && !tae_preview_only) {
        ignore_tensors.insert("first_stage_model.");
    }
    for (auto& extension : generation_extensions) {
        extension->add_ignore_tensors(ignore_tensors);
    }
    ignore_tensors.insert("model.diffusion_model.__x0__");
    ignore_tensors.insert("model.diffusion_model.__32x32__");
    ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

    if (audio_vae_model) {
        if (!sd_version_is_minimax_h3(version) && !sd_version_is_ltxav(version)) { // kcpp
            ignore_tensors.insert("audio_vae.encoder");
        }
    }
    if (version == VERSION_OVIS_IMAGE) {
        ignore_tensors.insert("text_encoders.llm.vision_model.");
        ignore_tensors.insert("text_encoders.llm.visual_tokenizer.");
        ignore_tensors.insert("text_encoders.llm.vte.");
    }
    if (version == VERSION_SVD) {
        ignore_tensors.insert("conditioner.embedders.3");
    }
    if (sd_version_is_ernie_image(version)) {
        ignore_tensors.insert("text_encoders.llm.vision_tower.");
        ignore_tensors.insert("text_encoders.llm.multi_modal_projector.");
    }
    if (sd_version_is_lens(version)) {
        ignore_tensors.insert("text_encoders.llm.tokenizer_json");
        ignore_tensors.insert("text_encoders.llm.model.layers.0.mlp.experts.gate_up_proj.weight_scale_2");
        ignore_tensors.insert("text_encoders.llm.model.layers.0.mlp.experts.down_proj.weight_scale_2");
    }
    if (sd_version_is_ideogram4(version)) {
        ignore_tensors.insert("text_encoders.llm.lm_head.");
        ignore_tensors.insert("text_encoders.llm.visual.");
        ignore_tensors.insert("text_encoders.llm.vision_model.");
        ignore_tensors.insert("text_encoders.llm.tokenizer_json");
    }
    if (version == VERSION_HIDREAM_O1) {
        ignore_tensors.insert("lm_head.");
        ignore_tensors.insert("model.visual.deepstack_merger_list.");
    }

    model_manager->set_common_ignore_tensors(ignore_tensors);
    if (!model_manager->validate_registered_tensors()) {
        LOG_ERROR("model metadata validation failed");
        return false;
    }

    if (eager_load) {
        if (!model_manager->load_all_params_eagerly()) {
            LOG_ERROR("model params eager load failed");
            return false;
        }
        LOG_VERBOSE("model metadata validated; weights pre-loaded to params backend");
    } else {
        LOG_VERBOSE("model metadata validated; weights will be prepared lazily");
    }

    {
        size_t text_encoder_params_mem_size = model_manager->registered_params_size({ModelComponent::Conditioner});
        size_t unet_params_mem_size         = model_manager->registered_params_size({ModelComponent::Diffusion, ModelComponent::HighNoiseDiffusion});
        size_t vae_params_mem_size          = model_manager->registered_params_size(runner_components().at(RunnerGroup::VAE));
        size_t control_net_params_mem_size  = model_manager->registered_params_size({ModelComponent::ControlNet});
        size_t extension_params_mem_size    = model_manager->registered_params_size(runner_components().at(RunnerGroup::Extensions));
        size_t total_params_ram_size        = 0;
        size_t total_params_vram_size       = 0;
        auto add_params_memory              = [&](size_t size, SDBackendModule module) {
            if (size == 0) {
                return true;
            }
            ggml_backend_t module_backend = params_backend_for(module);
            if (module_backend == nullptr) {
                return false;
            }
            if (sd_backend_is_cpu(module_backend)) {
                total_params_ram_size += size;
            } else {
                total_params_vram_size += size;
            }
            return true;
        };
        auto params_memory_location = [&](size_t size, SDBackendModule module) {
            if (size == 0) {
                return "N/A";
            }
            ggml_backend_t module_backend = params_backend_for(module);
            if (module_backend == nullptr) {
                return "N/A";
            }
            return sd_backend_is_cpu(module_backend) ? "RAM" : "VRAM";
        };

        if (!add_params_memory(text_encoder_params_mem_size, SDBackendModule::TE) ||
            !add_params_memory(extension_params_mem_size, SDBackendModule::PHOTOMAKER) ||
            !add_params_memory(unet_params_mem_size, SDBackendModule::DIFFUSION) ||
            !add_params_memory(vae_params_mem_size, SDBackendModule::VAE) ||
            !add_params_memory(control_net_params_mem_size, SDBackendModule::CONTROL_NET)) {
            return false;
        }

        size_t total_params_size = total_params_ram_size + total_params_vram_size;
        LOG_INFO(
            "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
            "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), extensions %.2fMB(%s)",
            total_params_size / 1024.0 / 1024.0,
            total_params_vram_size / 1024.0 / 1024.0,
            total_params_ram_size / 1024.0 / 1024.0,
            text_encoder_params_mem_size / 1024.0 / 1024.0,
            params_memory_location(text_encoder_params_mem_size, SDBackendModule::TE),
            unet_params_mem_size / 1024.0 / 1024.0,
            params_memory_location(unet_params_mem_size, SDBackendModule::DIFFUSION),
            vae_params_mem_size / 1024.0 / 1024.0,
            params_memory_location(vae_params_mem_size, SDBackendModule::VAE),
            control_net_params_mem_size / 1024.0 / 1024.0,
            params_memory_location(control_net_params_mem_size, SDBackendModule::CONTROL_NET),
            extension_params_mem_size / 1024.0 / 1024.0,
            params_memory_location(extension_params_mem_size, SDBackendModule::PHOTOMAKER));
    }
    return true;
}

bool StableDiffusionGGML::build_denoiser() {
    const auto* sd_ctx_params      = &config_->params;
    const auto& model_loader       = model_manager->loader();
    const auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    denoiser                       = std::make_shared<CompVisDenoiser>();
    default_flow_shift             = INFINITY;
    prediction_t pred_type         = sd_ctx_params->prediction;

    if (pred_type == PREDICTION_COUNT) {
        if (sd_version_is_sd2(version)) {
            pred_type = is_using_v_parameterization_for_sd2(sd_version_is_inpaint(version)) ? V_PRED : EPS_PRED;
        } else if (sd_version_is_sdxl(version)) {
            if (tensor_storage_map.find("edm_vpred.sigma_max") != tensor_storage_map.end()) {
                // CosXL models
                // TODO: get sigma_min and sigma_max values from file
                pred_type = EDM_V_PRED;
            } else if (tensor_storage_map.find("v_pred") != tensor_storage_map.end()) {
                pred_type = V_PRED;
            } else {
                pred_type = EPS_PRED;
            }
        } else if (sd_version_is_sd3(version) ||
                   sd_version_is_wan(version) ||
                   sd_version_is_hunyuan_video(version) ||
                   sd_version_is_lingbot_video(version) ||
                   sd_version_is_minimax_h3(version) ||
                   sd_version_is_qwen_image(version) ||
                   sd_version_is_mage_flow(version) ||
                   version == VERSION_HIDREAM_O1 ||
                   sd_version_is_anima(version) ||
                   sd_version_is_ernie_image(version) ||
                   sd_version_is_z_image(version) ||
                   sd_version_is_boogu_image(version) ||
                   sd_version_is_pid(version) ||
                   sd_version_is_ideogram4(version)) {
            pred_type = FLOW_PRED;
            if (sd_version_is_wan(version)) {
                default_flow_shift = 5.f;
            } else if (sd_version_is_hunyuan_video(version)) {
                default_flow_shift = 7.f;
            } else if (sd_version_is_minimax_h3(version)) {
                default_flow_shift = 12.f;
            } else if (sd_version_is_ernie_image(version)) {
                default_flow_shift = 4.f;
            } else if (sd_version_is_pid(version)) {
                default_flow_shift = 1.5f;
            } else if (sd_version_is_ideogram4(version)) {
                default_flow_shift = 1.0f;
            } else if (sd_version_is_boogu_image(version)) {
                default_flow_shift = 3.16f;
            } else if (sd_version_is_mage_flow(version)) {
                default_flow_shift = 6.f;
            } else {
                default_flow_shift = 3.f;
            }
        } else if (sd_version_is_flux(version) ||
                   sd_version_is_flux2(version) ||
                   sd_version_is_longcat(version) ||
                   sd_version_is_lens(version) ||
                   sd_version_is_ltxav(version) ||
                   sd_version_is_krea2(version)) {
            pred_type = FLUX_FLOW_PRED;

            default_flow_shift = 1.0f;  // TODO: validate
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (starts_with(name, "model.diffusion_model.guidance_in.in_layer.weight")) {
                    default_flow_shift = 1.15f;
                    break;
                }
            }
            if (sd_version_is_longcat(version)) {
                default_flow_shift = 3.0f;
            } else if (sd_version_is_lens(version)) {
                default_flow_shift = 1.83f;
            } else if (sd_version_is_ltxav(version)) {
                default_flow_shift = 2.37f;
            } else if (sd_version_is_krea2(version)) {
                default_flow_shift = 1.15f;
            }
        } else if (sd_version_is_sefi_image(version)) {
            pred_type = SEFI_FLOW_PRED;
        } else if (sd_version_is_minit2i(version)) {
            pred_type = MINIT2I_FLOW_PRED;
        } else if (sd_version_is_sensenova_u1(version)) {
            pred_type          = SENSENOVA_U1_FLOW_PRED;
            default_flow_shift = 3.f;
        } else {
            pred_type = EPS_PRED;
        }
    }

    switch (pred_type) {
        case EPS_PRED:
            LOG_INFO("running in eps-prediction mode");
            break;
        case V_PRED:
            LOG_INFO("running in v-prediction mode");
            denoiser = std::make_shared<CompVisVDenoiser>();
            break;
        case EDM_V_PRED:
            LOG_INFO("running in v-prediction EDM mode");
            denoiser = std::make_shared<EDMVDenoiser>();
            break;
        case FLOW_PRED: {
            if (sd_version_is_ltxav(version)) {
                LOG_INFO("running in LTXAV FLOW mode");
                denoiser = std::make_shared<FluxFlowDenoiser>();
            } else if (sd_version_is_minimax_h3(version)) {
                LOG_INFO("running in MiniMax H3 AV FLOW mode");
                denoiser = std::make_shared<H3AVFlowDenoiser>(default_flow_shift, 3.f, get_latent_channel());
            } else {
                LOG_INFO("running in FLOW mode");
                denoiser = std::make_shared<DiscreteFlowDenoiser>();
            }
            break;
        }
        case FLUX_FLOW_PRED: {
            LOG_INFO("running in Flux FLOW mode");
            denoiser = std::make_shared<FluxFlowDenoiser>();
            break;
        }
        case SEFI_FLOW_PRED: {
            LOG_INFO("running in SeFi-Image dual-time FLOW mode");
            denoiser = std::make_shared<SefiFlowDenoiser>();
            break;
        }
        case MINIT2I_FLOW_PRED: {
            LOG_INFO("running in MiniT2I FLOW mode");
            denoiser = std::make_shared<MiniT2IFlowDenoiser>();
            break;
        }
        case SENSENOVA_U1_FLOW_PRED: {
            LOG_INFO("running in SenseNova U1.5 FLOW mode");
            denoiser = std::make_shared<SenseNovaU1FlowDenoiser>(default_flow_shift);
            break;
        }
        default: {
            LOG_ERROR("Unknown predition type %i", pred_type);
            return false;
        }
    }

    refresh_compvis_denoiser_sigmas();
    return true;
}

bool StableDiffusionGGML::build_runners(const RunnerGroups& groups) {
    const auto& model_loader                                 = model_manager->loader();
    std::map<ggml_type, uint32_t> wtype_stat                 = model_loader.get_wtype_stat();
    std::map<ggml_type, uint32_t> conditioner_wtype_stat     = model_loader.get_conditioner_wtype_stat();
    std::map<ggml_type, uint32_t> diffusion_model_wtype_stat = model_loader.get_diffusion_model_wtype_stat();
    std::map<ggml_type, uint32_t> vae_wtype_stat             = model_loader.get_vae_wtype_stat();

    auto wtype_stat_to_str = [](const std::map<ggml_type, uint32_t>& m, int key_width = 8, int value_width = 5) -> std::string {
        std::ostringstream oss;
        bool first = true;
        for (const auto& [type, count] : m) {
            if (!first)
                oss << "|";
            first = false;
            oss << std::right << std::setw(key_width) << ggml_type_name(type)
                << ": "
                << std::left << std::setw(value_width) << count;
        }
        return oss.str();
    };

    LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
    LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
    LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
    LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

    LOG_VERBOSE("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

    configure_weight_loading();
    for (auto group : groups) {
        bool success = false;
        switch (group) {
            case RunnerGroup::Core:
                success = build_core_runners();
                break;
            case RunnerGroup::VAE:
                success = build_vae_runners();
                break;
            case RunnerGroup::ControlNet:
                success = build_control_net_runner();
                break;
            case RunnerGroup::Extensions:
                success = build_extension_runners();
                break;
        }
        if (!success) {
            return false;
        }
    }
    if (!validate_and_load_runners()) {
        return false;
    }
    return groups.count(RunnerGroup::Core) == 0 || build_denoiser();
}

bool StableDiffusionGGML::is_using_v_parameterization_for_sd2(bool is_inpaint) {
    struct RunnerEndOnExit {
        GGMLRunner* runner = nullptr;
        ~RunnerEndOnExit() {
            if (runner != nullptr) {
                runner->runner_end();
            }
        }
    };
    RunnerEndOnExit diffusion_runner_end{diffusion_model.get()};

    sd::Tensor<float> x_t   = sd::full<float>({8, 8, 4, 1}, 0.5f);
    sd::Tensor<float> c     = sd::full<float>({1024, 2, 1, 1}, 0.5f);
    sd::Tensor<float> steps = sd::full<float>({1}, 999.0f);
    sd::Tensor<float> concat;
    if (is_inpaint) {
        concat = sd::zeros<float>({8, 8, 5, 1});
    }

    int64_t t0 = ggml_time_ms();
    sd::Tensor<float> out;
    DiffusionParams diffusion_params;
    diffusion_params.x         = &x_t;
    diffusion_params.timesteps = &steps;
    diffusion_params.context   = &c;
    diffusion_params.extra     = UNetDiffusionExtra{};
    if (!concat.empty()) {
        diffusion_params.c_concat = &concat;
    }
    auto out_opt = diffusion_model->compute(n_threads, diffusion_params);
    GGML_ASSERT(!out_opt.empty());
    out = std::move(out_opt);

    double result = static_cast<double>((out - x_t).mean());
    int64_t t1    = ggml_time_ms();
    LOG_VERBOSE("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    return result < -1;
}

std::string StableDiffusionGGML::lora_log_id(const ModelManager::LoraSpec& lora) {
    return lora.is_high_noise ? "|high_noise|" + lora.path : lora.path;
}

std::shared_ptr<LoraModel> StableDiffusionGGML::load_lora_model(const ModelManager::LoraSpec& lora_spec,
                                                                SDBackendModule module,
                                                                LoraModel::filter_t module_filter) {
    if (!ensure_backend_pair(module)) {
        return nullptr;
    }
    if (lora_spec.is_high_noise) {
        LOG_VERBOSE("high noise lora: %s", lora_spec.path.c_str());
    }
    // kcpp first check the cache
    std::string lora_key = "|" + std::to_string(static_cast<int>(module)) + "|" + lora_log_id(lora_spec);
    if (!apply_lora_immediately) {
        auto it = kcpp_lora_cache.find(lora_key);
        if (it != kcpp_lora_cache.end()) {
            if (it->second) {
                it->second->multiplier = lora_spec.multiplier;
            }
            return it->second;
        }
    }
    const auto mode                        = backend_manager.params_backend_is_disk(module)
                                                 ? ModelManager::ResidencyMode::Disk
                                                 : ModelManager::ResidencyMode::ParamBackend;
    auto lora                              = std::make_shared<LoraModel>(lora_log_id(lora_spec), backend_for(module), params_backend_for(module),
                                            model_manager, lora_spec.file_id, version, mode,
                                            backend_manager.params_backend_follows_runtime(module));
    LoraModel::filter_t lora_tensor_filter = module_filter;
    if (!lora_spec.tensor_name_prefix_filter.empty()) {
        lora_tensor_filter = [module_filter, prefix = lora_spec.tensor_name_prefix_filter](const std::string& tensor_name) {
            return starts_with(tensor_name, prefix) && (!module_filter || module_filter(tensor_name));
        };
    }
    if (!lora->init_params(n_threads, lora_tensor_filter)) {
        LOG_WARN("load lora tensors from %s failed", lora_spec.path.c_str());
        // kcpp also cache negatives to avoid I/O at runtime
        lora = nullptr;
        if (!apply_lora_immediately && kcpp_lora_cache_populate)
            kcpp_lora_cache[lora_key] = lora;
        return lora;
    }

    lora->multiplier = lora_spec.multiplier;
    // kcpp add to cache
    if (!apply_lora_immediately && kcpp_lora_cache_populate)
        kcpp_lora_cache[lora_key] = lora;
    return lora;
}

void StableDiffusionGGML::clear_lora_adapters() {
    if (cond_stage_model) {
        cond_stage_model->set_weight_adapter(nullptr);
    }
    if (diffusion_model) {
        diffusion_model->set_weight_adapter(nullptr);
    }
    if (high_noise_diffusion_model) {
        high_noise_diffusion_model->set_weight_adapter(nullptr);
    }
    if (first_stage_model) {
        first_stage_model->set_weight_adapter(nullptr);
    }
}

std::vector<std::shared_ptr<LoraModel>> StableDiffusionGGML::load_runtime_loras_for_module(const std::vector<ModelManager::LoraSpec>& loras,
                                                                                           const std::set<std::string>& model_tensor_names,
                                                                                           SDBackendModule module,
                                                                                           LoraModel::filter_t module_filter,
                                                                                           bool& success,
                                                                                           std::vector<RuntimeLora>& next_models) {
    std::vector<std::shared_ptr<LoraModel>> module_lora_models;
    for (const auto& lora_spec : loras) {
        auto cached = std::find_if(runtime_lora_models.begin(), runtime_lora_models.end(), [&](const RuntimeLora& entry) {
            return entry.model != nullptr && entry.module == module && entry.matches(lora_spec);
        });
        auto lora   = cached == runtime_lora_models.end() ? load_lora_model(lora_spec, module, module_filter)
                                                          : std::move(cached->model);
        if (lora == nullptr) {
            if (lora_spec.required) {
                LOG_ERROR("required lora load failed: %s", lora_spec.path.c_str());
                success = false;
            }
            continue;
        }
        if (lora->lora_tensors.empty()) {
            continue;
        }

        lora->preprocess_lora_tensors(model_tensor_names);
        lora->multiplier = lora_spec.multiplier;
        next_models.push_back({lora_spec, module, lora});
        module_lora_models.push_back(std::move(lora));
    }
    return module_lora_models;
}

bool StableDiffusionGGML::apply_loras_immediately(const std::vector<ModelManager::LoraSpec>& loras) {
    if (model_manager == nullptr) {
        if (!loras.empty()) {
            LOG_WARN("model manager is not available for immediate lora");
        }
        return false;
    }

    clear_lora_adapters();
    runtime_lora_models.clear();

    if (!loras.empty()) {
        LOG_INFO("apply lora immediately");
    }
    return model_manager->set_loras(loras, version);
}

bool StableDiffusionGGML::apply_loras_at_runtime(const std::vector<ModelManager::LoraSpec>& loras) {
    if (model_manager != nullptr) {
        if (!model_manager->set_loras({}, version))
            return false;
    }
    clear_lora_adapters();
    if (loras.empty()) {
        runtime_lora_models.clear();
        return true;
    }

    bool success = true;
    std::vector<RuntimeLora> next_models;
    std::set<std::string> model_tensor_names;
    if (model_manager != nullptr) {
        model_tensor_names = model_manager->tensor_names();
    }

    LOG_INFO("apply lora at runtime");
    if (cond_stage_model) {
        auto lora_tensor_filter = [&](const std::string& tensor_name) {
            if (is_cond_stage_model_name(tensor_name)) {
                return true;
            }
            return false;
        };
        auto cond_stage_lora_models =
            load_runtime_loras_for_module(loras,
                                          model_tensor_names,
                                          SDBackendModule::TE,
                                          lora_tensor_filter, success, next_models);
        // Only attach the adapter when there are LoRAs targeting the cond_stage model.
        // An empty MultiLoraAdapter still routes every linear/conv through
        // forward_with_lora() instead of the direct kernel path — slower for no benefit.
        if (!cond_stage_lora_models.empty()) {
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(cond_stage_lora_models);
            cond_stage_model->set_weight_adapter(multi_lora_adapter);
        }
    }
    if (diffusion_model) {
        auto lora_tensor_filter = [&](const std::string& tensor_name) {
            if (is_diffusion_model_name(tensor_name)) {
                return true;
            }
            return false;
        };
        auto diffusion_lora_models =
            load_runtime_loras_for_module(loras,
                                          model_tensor_names,
                                          SDBackendModule::DIFFUSION,
                                          lora_tensor_filter, success, next_models);
        if (!diffusion_lora_models.empty()) {
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(diffusion_lora_models);
            diffusion_model->set_weight_adapter(multi_lora_adapter);
            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_weight_adapter(multi_lora_adapter);
            }
        }
    }

    if (first_stage_model) {
        auto lora_tensor_filter = [&](const std::string& tensor_name) {
            if (is_first_stage_model_name(tensor_name)) {
                return true;
            }
            return false;
        };
        auto first_stage_lora_models =
            load_runtime_loras_for_module(loras,
                                          model_tensor_names,
                                          SDBackendModule::VAE,
                                          lora_tensor_filter, success, next_models);
        if (!first_stage_lora_models.empty()) {
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(first_stage_lora_models);
            first_stage_model->set_weight_adapter(multi_lora_adapter);
        }
    }
    runtime_lora_models = std::move(next_models);
    return success;
}

void StableDiffusionGGML::lora_stat() {
    if (!runtime_lora_models.empty()) {
        LOG_INFO("runtime_lora_models:");
        for (auto& lora_model : runtime_lora_models) {
            lora_model.model->stat();
        }
    }
}

bool StableDiffusionGGML::apply_loras(const sd_lora_t* loras, uint32_t lora_count) {
    std::vector<ModelManager::LoraSpec> all_loras;
    all_loras.reserve(lora_count);
    for (uint32_t i = 0; i < lora_count; i++) {
        std::string lora_id = SAFE_STR(loras[i].path);
        ModelManager::LoraSpec lora_spec;
        lora_spec.path          = lora_id;
        lora_spec.multiplier    = loras[i].multiplier;
        lora_spec.is_high_noise = loras[i].is_high_noise;
        all_loras.push_back(std::move(lora_spec));
        if (loras[i].is_high_noise) {
            lora_id = "|high_noise|" + lora_id;
        }
        LOG_VERBOSE("lora %s:%.2f", lora_id.c_str(), loras[i].multiplier);
    }

    for (auto& extension : generation_extensions) {
        extension->collect_loras(all_loras);
    }

    int64_t t0 = ggml_time_ms();
    end_runners();
    clear_lora_adapters();
    if (!model_manager->prepare_lora_sources(all_loras))
        return false;
    runtime_lora_models.erase(std::remove_if(runtime_lora_models.begin(), runtime_lora_models.end(), [&](const RuntimeLora& entry) {
                                  return std::none_of(all_loras.begin(), all_loras.end(), [&](const ModelManager::LoraSpec& spec) {
                                      return entry.matches(spec);
                                  });
                              }),
                              runtime_lora_models.end());
    const bool success = apply_lora_immediately ? apply_loras_immediately(all_loras)
                                                : apply_loras_at_runtime(all_loras);
    if (!success) {
        clear_lora_adapters();
        runtime_lora_models.clear();
        return false;
    }
    runner_state_.catalog_revision = model_manager->loader().revision();
    int64_t t1                     = ggml_time_ms();
    if (!all_loras.empty()) {
        LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    }
    return true;
}

void StableDiffusionGGML::reset_generation_extensions() {
    for (auto& extension : generation_extensions) {
        extension->reset_runtime_condition();
    }
}

void StableDiffusionGGML::prepare_generation_extensions(const sd_pm_params_t& pm_params,
                                                        const sd_pulid_params_t& pulid_params,
                                                        ConditionerParams& condition_params,
                                                        int total_steps) {
    reset_generation_extensions();
    GenerationExtensionConditionContext ctx{
        cond_stage_model.get(),
        condition_params,
        pm_params,
        pulid_params,
        n_threads,
        total_steps,
    };

    for (auto& extension : generation_extensions) {
        extension->prepare_condition(ctx);
    }
}

sd::Tensor<float> StableDiffusionGGML::get_clip_vision_output(const sd::Tensor<float>& image,
                                                              bool return_pooled,
                                                              int clip_skip,
                                                              bool zero_out_masked) {
    sd::Tensor<float> output;
    if (zero_out_masked) {
        if (return_pooled) {
            output = sd::zeros<float>({clip_vision->vision_model.projection_dim});
        } else {
            output = sd::zeros<float>({clip_vision->vision_model.hidden_size, 257});
        }
    } else {
        auto pixel_values = clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);
        auto output_opt   = clip_vision->compute(n_threads, pixel_values, return_pooled, clip_skip);
        if (output_opt.empty()) {
            LOG_ERROR("clip_vision compute failed");
            return {};
        }
        output = std::move(output_opt);
    }
    return output;
}

void StableDiffusionGGML::compute_ip_adapter_tokens(const sd_image_t& image, float strength) {
    ip_adapter_tokens        = {};
    ip_adapter_uncond_tokens = {};
    ip_adapter_strength      = strength;
    if (ip_adapter == nullptr || clip_vision == nullptr || image.data == nullptr) {
        return;
    }
    auto image_tensor = sd_image_to_tensor(image);
    auto embed        = ip_adapter->is_plus
                            ? get_clip_vision_output(image_tensor, false, 2)
                            : get_clip_vision_output(image_tensor, true, -1);
    if (embed.empty()) {
        return;
    }
    ip_adapter_tokens = ip_adapter->compute(n_threads, embed);
    if (ip_adapter_tokens.empty()) {
        LOG_ERROR("IP-Adapter conditional image projection failed");
        return;
    }
    auto uncond_embed        = sd::Tensor<float>::zeros_like(embed);
    ip_adapter_uncond_tokens = ip_adapter->compute(n_threads, uncond_embed);
    if (ip_adapter_uncond_tokens.empty()) {
        LOG_ERROR("IP-Adapter unconditional image projection failed");
        ip_adapter_tokens = {};
        return;
    }
    LOG_INFO("IP-Adapter: %lld image tokens, strength %.2f",
             (long long)ip_adapter_tokens.shape()[1], strength);
}

std::vector<float> StableDiffusionGGML::process_timesteps(const std::vector<float>& timesteps,
                                                          const sd::Tensor<float>& init_latent,
                                                          const sd::Tensor<float>& denoise_mask,
                                                          int step) {
    if (auto sefi_denoiser = std::dynamic_pointer_cast<SefiFlowDenoiser>(denoiser)) {
        int sched_idx = step > 0 ? step - 1 : 0;
        if (sched_idx >= static_cast<int>(sefi_denoiser->tex_timesteps.size())) {
            sched_idx = static_cast<int>(sefi_denoiser->tex_timesteps.size()) - 1;
        }
        return {sefi_denoiser->sem_timesteps[sched_idx],
                sefi_denoiser->tex_timesteps[sched_idx]};
    }
    if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B") {
        int64_t frame_count = init_latent.shape()[2];
        auto new_timesteps  = std::vector<float>(static_cast<size_t>(frame_count), timesteps[0]);

        if (!denoise_mask.empty() && denoise_mask.dim() >= 4 && denoise_mask.shape()[2] == frame_count) {
            for (int64_t frame = 0; frame < frame_count; ++frame) {
                float value = denoise_mask.dim() == 5 ? denoise_mask.index(0, 0, frame, 0, 0) : denoise_mask.index(0, 0, frame, 0);
                if (value == 0.f) {
                    new_timesteps[static_cast<size_t>(frame)] = 0.f;
                }
            }
        }
        return new_timesteps;
    } else {
        return timesteps;
    }
}

std::vector<float> StableDiffusionGGML::process_ltxav_video_timesteps(const std::vector<float>& timesteps,
                                                                      const sd::Tensor<float>& init_latent,
                                                                      const sd::Tensor<float>& denoise_mask) {
    if (timesteps.empty() || denoise_mask.empty() || init_latent.dim() < 4 || denoise_mask.dim() < 4) {
        return timesteps;
    }

    int64_t width  = init_latent.shape()[0];
    int64_t height = init_latent.shape()[1];
    int64_t frames = init_latent.shape()[2];
    if (denoise_mask.shape()[0] != width ||
        denoise_mask.shape()[1] != height ||
        denoise_mask.shape()[2] != frames ||
        denoise_mask.shape()[3] < 1) {
        LOG_WARN("unexpected LTXAV denoise mask shape for timestep processing");
        return timesteps;
    }

    std::vector<float> video_timesteps(static_cast<size_t>(width * height * frames));
    size_t idx = 0;
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                float mask             = denoise_mask.dim() == 5 ? denoise_mask.index(w, h, t, 0, 0)
                                                                 : denoise_mask.index(w, h, t, 0);
                video_timesteps[idx++] = mask * timesteps[0];
            }
        }
    }
    return video_timesteps;
}

void StableDiffusionGGML::preview_image(int step,
                                        const sd::Tensor<float>& latents,
                                        enum SDVersion version,
                                        preview_t preview_mode,
                                        std::function<void(int, int, sd_image_t*, bool, void*)> step_callback,
                                        void* step_callback_data,
                                        bool is_noisy) {
    bool is_video = preview_latent_tensor_is_video(latents);
    uint32_t dim  = is_video ? static_cast<uint32_t>(latents.shape()[3]) : static_cast<uint32_t>(latents.shape()[2]);
    int channels  = get_latent_channel();
    auto _latents = channels != dim ? is_video ? sd::ops::slice(latents, 3, 0, channels)
                                               : sd::ops::slice(latents, 2, 0, channels)
                                    : latents;
    if (preview_mode == PREVIEW_PROJ) {
        int patch_sz                     = 1;
        const float(*latent_rgb_proj)[3] = nullptr;
        float* latent_rgb_bias           = nullptr;

        if (channels == 128) {
            if (sd_version_uses_flux2_vae(version)) {
                latent_rgb_proj = flux2_latent_rgb_proj;
                latent_rgb_bias = flux2_latent_rgb_bias;
                patch_sz        = 2;
            } else if (version == VERSION_LTXAV) {
                latent_rgb_proj = ltxav_latent_rgb_proj;
                latent_rgb_bias = ltxav_latent_rgb_bias;
            } else {
                LOG_WARN("No latent to RGB projection known for this model");
                return;
            }
        } else if (channels == 48) {
            if (sd_version_is_wan(version)) {
                latent_rgb_proj = wan_22_latent_rgb_proj;
                latent_rgb_bias = wan_22_latent_rgb_bias;
            } else {
                LOG_WARN("No latent to RGB projection known for this model");
                return;
            }
        } else if (channels == 24) {
            if (sd_version_is_minimax_h3(version)) {
                latent_rgb_proj = minimax_latent_rgb_proj;
                latent_rgb_bias = minimax_latent_rgb_bias;
            } else {
                LOG_WARN("No latent to RGB projection known for this model");
                return;
            }
        } else if (channels == 16) {
            if (sd_version_is_sd3(version)) {
                latent_rgb_proj = sd3_latent_rgb_proj;
                latent_rgb_bias = sd3_latent_rgb_bias;
            } else if (sd_version_uses_flux_vae(version)) {
                latent_rgb_proj = flux_latent_rgb_proj;
                latent_rgb_bias = flux_latent_rgb_bias;
            } else if (sd_version_uses_wan_vae(version)) {
                latent_rgb_proj = wan_21_latent_rgb_proj;
                latent_rgb_bias = wan_21_latent_rgb_bias;
            } else {
                LOG_WARN("No latent to RGB projection known for this model");
                return;
            }
        } else if (channels == 4) {
            if (sd_version_is_sdxl(version)) {
                latent_rgb_proj = sdxl_latent_rgb_proj;
                latent_rgb_bias = sdxl_latent_rgb_bias;
            } else if (sd_version_is_sd1(version) || sd_version_is_sd2(version)) {
                latent_rgb_proj = sd_latent_rgb_proj;
                latent_rgb_bias = sd_latent_rgb_bias;
            } else {
                LOG_WARN("No latent to RGB projection known for this model");
                return;
            }
        } else if (channels != 3) {
            LOG_WARN("No latent to RGB projection known for this model (dim = %d)", dim);
            return;
        }

        uint32_t frames     = is_video ? static_cast<uint32_t>(_latents.shape()[2]) : 1;
        uint32_t img_width  = static_cast<uint32_t>(_latents.shape()[0]) * patch_sz;
        uint32_t img_height = static_cast<uint32_t>(_latents.shape()[1]) * patch_sz;

        uint8_t* data = (uint8_t*)malloc(frames * img_width * img_height * 3 * sizeof(uint8_t));
        GGML_ASSERT(data != nullptr);
        preview_latent_video(data, _latents, latent_rgb_proj, latent_rgb_bias, patch_sz);
        sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
        GGML_ASSERT(images != nullptr);
        for (uint32_t i = 0; i < frames; i++) {
            images[i] = {img_width, img_height, 3, data + i * img_width * img_height * 3};
        }
        step_callback(step, frames, images, is_noisy, step_callback_data);
        free(data);
        free(images);
        return;
    }

    if (preview_mode == PREVIEW_VAE || preview_mode == PREVIEW_TAE) {
        sd::Tensor<float> vae_latents;
        sd::Tensor<float> decoded;
        if (preview_vae) {
            vae_latents = preview_vae->diffusion_to_vae_latents(_latents);
            decoded     = preview_vae->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
        } else {
            vae_latents = first_stage_model->diffusion_to_vae_latents(_latents);
            decoded     = first_stage_model->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
        }
        if (decoded.empty()) {
            LOG_ERROR("preview decode failed at step %d", step);
            return;
        }

        is_video           = preview_latent_tensor_is_video(decoded);
        uint32_t frames    = is_video ? static_cast<uint32_t>(decoded.shape()[2]) : 1;
        sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
        GGML_ASSERT(images != nullptr);
        for (uint32_t i = 0; i < frames; ++i) {
            images[i] = tensor_to_sd_image(decoded, static_cast<int>(i));
        }

        step_callback(step, frames, images, is_noisy, step_callback_data);
        for (uint32_t i = 0; i < frames; ++i) {
            free(images[i].data);
        }
        free(images);
        return;
    }

    if (preview_mode != PREVIEW_NONE) {
        LOG_WARN("Unsupported preview mode: %d", static_cast<int>(preview_mode));
    }
}

std::vector<float> StableDiffusionGGML::prepare_sample_timesteps(float sigma,
                                                                 int shifted_timestep) {
    float t = denoiser->sigma_to_t(sigma);
    if (shifted_timestep > 0) {
        float shifted_t_float = t * (float(shifted_timestep) / float(TIMESTEPS));
        int64_t shifted_t     = static_cast<int64_t>(roundf(shifted_t_float));
        shifted_t             = std::max((int64_t)0, std::min((int64_t)(TIMESTEPS - 1), shifted_t));
        LOG_VERBOSE("shifting timestep from %.2f to %" PRId64 " (sigma: %.4f)", t, shifted_t, sigma);
        return std::vector<float>{(float)shifted_t};
    }
    if (sd_version_is_anima(version)) {
        return std::vector<float>{t / static_cast<float>(TIMESTEPS)};
    }
    if (sd_version_is_boogu_image(version)) {
        return std::vector<float>{t / static_cast<float>(TIMESTEPS)};
    }
    if (version == VERSION_HIDREAM_O1) {
        return std::vector<float>{1.0f - (t / static_cast<float>(TIMESTEPS))};
    }
    if (sd_version_is_z_image(version) || sd_version_is_ideogram4(version)) {
        return std::vector<float>{1000.f - t};
    }
    return std::vector<float>{t};
}

void StableDiffusionGGML::adjust_sample_step_scalings(int shifted_timestep,
                                                      const std::vector<float>& timesteps_vec,
                                                      float c_in,
                                                      float* c_skip,
                                                      float* c_out) {
    GGML_ASSERT(c_skip != nullptr);
    GGML_ASSERT(c_out != nullptr);
    if (shifted_timestep <= 0) {
        return;
    }

    int64_t shifted_t_idx              = static_cast<int64_t>(roundf(timesteps_vec[0]));
    float shifted_sigma                = denoiser->t_to_sigma((float)shifted_t_idx);
    std::vector<float> shifted_scaling = denoiser->get_scalings(shifted_sigma);
    float shifted_c_skip               = shifted_scaling[0];
    float shifted_c_out                = shifted_scaling[1];
    float shifted_c_in                 = shifted_scaling[2];

    *c_skip = shifted_c_skip * c_in / shifted_c_in;
    *c_out  = shifted_c_out;
}

StableDiffusionGGML::SamplePreviewContext StableDiffusionGGML::prepare_sample_preview_context() {
    return SamplePreviewContext{sd_get_preview_callback(),
                                sd_get_preview_callback_data(),
                                sd_get_preview_mode()};
}

void StableDiffusionGGML::report_sample_progress(int step,
                                                 size_t total_steps,
                                                 bool terminal_sigma_is_zero,
                                                 int64_t* last_progress_us) {
    if (sd::preview::sample_step_is_complete(step, total_steps, terminal_sigma_is_zero)) {
        int64_t now        = ggml_time_us();
        int showstep       = std::abs(step);
        float step_seconds = last_progress_us != nullptr && *last_progress_us > 0
                                 ? (now - *last_progress_us) / 1000000.f
                                 : 0.f;
        pretty_progress(showstep, (int)total_steps, step_seconds);
        if (last_progress_us != nullptr) {
            *last_progress_us = now;
        }
    }
}

void StableDiffusionGGML::compute_sample_controls(const sd::Tensor<float>& control_image,
                                                  const sd::Tensor<float>& noised_input,
                                                  const sd::Tensor<float>& timesteps_tensor,
                                                  const SDCondition& condition,
                                                  std::vector<sd::Tensor<float>>* controls) {
    GGML_ASSERT(controls != nullptr);
    controls->clear();
    if (control_image.empty() || control_net == nullptr) {
        return;
    }

    auto control_result = control_net->compute(n_threads,
                                               noised_input,
                                               control_image,
                                               timesteps_tensor,
                                               condition.c_crossattn,
                                               condition.c_vector);
    if (!control_result.has_value()) {
        LOG_ERROR("controlnet compute failed");
        return;
    }

    *controls = std::move(*control_result);
}

sd::Tensor<float> StableDiffusionGGML::sample(const std::shared_ptr<DiffusionModelRunner>& work_diffusion_model,
                                              bool inverse_noise_scaling,
                                              const sd::Tensor<float>& init_latent,
                                              sd::Tensor<float> noise,
                                              const SDCondition& cond,
                                              const SDCondition& uncond,
                                              const SDCondition& img_uncond,
                                              const sd::Tensor<float>& control_image,
                                              float control_strength,
                                              const sd_guidance_params_t& guidance,
                                              float eta,
                                              int shifted_timestep,
                                              sample_method_t method,
                                              bool is_flow_denoiser,
                                              const char* extra_sample_args,
                                              const std::vector<float>& sigmas,
                                              const std::vector<sd::Tensor<float>>& ref_latents,
                                              const RefImageParams& ref_image_params,
                                              const sd::Tensor<float>& denoise_mask,
                                              const sd::Tensor<float>& vace_context,
                                              float vace_strength,
                                              int audio_length,
                                              float frame_rate,
                                              const sd_cache_params_t* cache_params,
                                              bool preview_final_step,
                                              const sd::Tensor<float>& video_positions) {
    struct RunnerEndOnExit {
        GGMLRunner* runner = nullptr;
        ~RunnerEndOnExit() {
            if (runner != nullptr) {
                runner->runner_end();
            }
        }
    };
    RunnerEndOnExit sample_diffusion_runner_end{work_diffusion_model.get()};

    RunnerEndOnExit sample_control_runner_end{!control_image.empty() && control_net != nullptr ? control_net.get() : nullptr};

    std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);
    float cfg_scale     = guidance.txt_cfg;
    float img_cfg_scale = guidance.img_cfg;
    float slg_scale     = guidance.slg.scale;
    bool slg_uncond     = sd::guidance::parse_skip_layer_guidance_uncond_arg(extra_sample_args);

    std::vector<float> guidance_schedule = sd::guidance::parse_guidance_schedule(extra_sample_args);
    if (!guidance_schedule.empty() && guidance_schedule.size() != sigmas.size() - 1) {
        if (guidance_schedule.size() > sigmas.size()) {
            LOG_WARN("guidance_schedule length (%zu) is greater than number of steps (%zu)", guidance_schedule.size(), sigmas.size() - 1);
            LOG_WARN("truncating guidance_schedule to match step count");
            guidance_schedule.resize(sigmas.size() - 1);
        } else {
            LOG_INFO("padding guidance_schedule with cfg_scale");
            while (guidance_schedule.size() < sigmas.size() - 1) {
                guidance_schedule.push_back(cfg_scale);
            }
        }
    }

    if (!guidance_schedule.empty()) {
        std::string schedule_str = "[";
        for (size_t i = 0; i < guidance_schedule.size(); ++i) {
            schedule_str += std::to_string(guidance_schedule[i]);
            if (i < guidance_schedule.size() - 1) {
                schedule_str += ", ";
            }
        }
        schedule_str += "]";
        LOG_VERBOSE("using guidance schedule: %s", schedule_str.c_str());
    }

    sd_sample::SampleCacheRuntime cache_runtime = sd_sample::init_sample_cache_runtime(version,
                                                                                       cache_params,
                                                                                       denoiser.get(),
                                                                                       sigmas);

    bool needs_uncond_denoised = method == EULER_CFG_PP_SAMPLE_METHOD || method == EULER_A_CFG_PP_SAMPLE_METHOD;
    // Spectrum cache is not supported for CFG++ samplers
    if (needs_uncond_denoised) {
        if (cache_runtime.spectrum_enabled) {
            LOG_WARN("Spectrum cache requested but not supported for CFG++ samplers");
            cache_runtime.spectrum_enabled = false;
        }
    }

    size_t steps                = sigmas.size() - 1;
    bool terminal_sigma_is_zero = sigmas.back() == 0.f;
    bool has_skiplayer          = (slg_scale != 0.0f || slg_uncond) && !skip_layers.empty();
    if (has_skiplayer && !sd_version_is_dit(version)) {
        has_skiplayer = false;
        LOG_WARN("SLG is incompatible with this model type");
    }
    sd::guidance::AdaptiveProjectedGuidanceParams apg_params = sd::guidance::parse_adaptive_projected_guidance_args(extra_sample_args);
    bool use_apg_guidance                                    = sd::guidance::is_adaptive_projected_guidance_enabled(apg_params);
    if (use_apg_guidance) {
        LOG_INFO("using Adaptive Projected Guidance (APG)");
    }
    sd::guidance::ClassifierFreeGuidance classifier_free_guidance(cfg_scale, img_cfg_scale);
    sd::guidance::AdaptiveProjectedGuidance adaptive_projected_guidance(cfg_scale, img_cfg_scale, apg_params);
    const sd::guidance::BaseGuidance& primary_guidance = use_apg_guidance
                                                             ? static_cast<const sd::guidance::BaseGuidance&>(adaptive_projected_guidance)
                                                             : static_cast<const sd::guidance::BaseGuidance&>(classifier_free_guidance);
    sd::guidance::SkipLayerGuidance skip_layer_guidance(has_skiplayer ? skip_layers : std::vector<int>(),
                                                        has_skiplayer ? slg_scale : 0.0f,
                                                        guidance.slg.layer_start,
                                                        guidance.slg.layer_end);

    if (version == VERSION_HIDREAM_O1 && !noise.empty()) {
        noise *= eta;
    }

    int64_t last_progress_us     = ggml_time_us();
    SamplePreviewContext preview = prepare_sample_preview_context();

    sd::Tensor<float> processed_init_latent       = denoiser->process_latent_in(init_latent);
    const sd::Tensor<float>& sampling_init_latent = processed_init_latent.empty()
                                                        ? init_latent
                                                        : processed_init_latent;
    sd::Tensor<float> x_t                         = !noise.empty()
                                                        ? denoiser->noise_scaling(sigmas[0], noise, sampling_init_latent)
                                                        : sampling_init_latent;
    sd::Tensor<float> denoised                    = x_t;

    auto denoise = [&](const sd::Tensor<float>& x, float sigma, int step) -> sd::guidance::GuiderOutput {
        if (get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_VERBOSE("cancelling generation");
            return {};
        }

        if (step == 1 || step == -1) {
            pretty_progress(0, (int)steps, 0);
            last_progress_us = ggml_time_us();
        }

        std::vector<float> scaling = denoiser->get_scalings(sigma);
        GGML_ASSERT(scaling.size() == 3);
        float c_skip = scaling[0];
        float c_out  = scaling[1];
        float c_in   = scaling[2];

        bool preview_needed = preview.callback != nullptr &&
                              sd::preview::should_preview_sample_step(step,
                                                                      steps,
                                                                      terminal_sigma_is_zero,
                                                                      sd_get_preview_interval(),
                                                                      preview_final_step);

        std::vector<float> base_timesteps_vec = prepare_sample_timesteps(sigma, shifted_timestep);
        std::vector<float> timesteps_vec      = base_timesteps_vec;
        sd::Tensor<float> audio_timesteps_tensor;
        if (sd_version_is_ltxav(version) && !denoise_mask.empty()) {
            timesteps_vec          = process_ltxav_video_timesteps(base_timesteps_vec, sampling_init_latent, denoise_mask);
            audio_timesteps_tensor = sd::Tensor<float>({static_cast<int64_t>(base_timesteps_vec.size())}, base_timesteps_vec);
        } else {
            timesteps_vec = process_timesteps(timesteps_vec, sampling_init_latent, denoise_mask, step);
        }
        const std::vector<float>& scaling_timesteps_vec = (sd_version_is_ltxav(version) && !denoise_mask.empty())
                                                              ? base_timesteps_vec
                                                              : timesteps_vec;
        adjust_sample_step_scalings(shifted_timestep, scaling_timesteps_vec, c_in, &c_skip, &c_out);

        sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
        sd::Tensor<float> guidance_tensor({1}, std::vector<float>{guidance.distilled_guidance});
        sd::Tensor<float> hunyuan_timestep_r_tensor;
        if (sd_version_is_hunyuan_video(version) && step + 1 < sigmas.size()) {
            hunyuan_timestep_r_tensor = sd::Tensor<float>::from_vector({sigmas[step + 1]});
        }
        sd::Tensor<float> noised_input = x * c_in;
        if (!denoise_mask.empty() && (version == VERSION_WAN2_2_TI2V || sd_version_is_ltxav(version) || sd_version_is_lingbot_video(version))) {
            noised_input = noised_input * denoise_mask + sampling_init_latent * (1.0f - denoise_mask);
        }

        if (cache_runtime.spectrum_enabled && cache_runtime.spectrum.should_predict()) {
            cache_runtime.spectrum.predict(&denoised);
            if (!denoise_mask.empty()) {
                denoised = denoised * denoise_mask + sampling_init_latent * (1.0f - denoise_mask);
            }
            if (preview_needed && sd_should_preview_denoised()) {
                preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
            }
            report_sample_progress(step, steps, terminal_sigma_is_zero, &last_progress_us);
            sd::guidance::GuiderOutput output;
            output.pred = denoised;
            return output;
        }

        if (preview_needed && sd_should_preview_noisy()) {
            preview_image(step, noised_input, version, preview.mode, preview.callback, preview.data, true);
        }

        sd::Tensor<float> cond_out;
        sd::Tensor<float> uncond_out;
        sd::Tensor<float> img_uncond_out;
        sd_sample::SampleStepCacheDispatcher step_cache(cache_runtime, step, sigma);
        std::vector<sd::Tensor<float>> controls;
        DiffusionParams diffusion_params;
        diffusion_params.x                = &noised_input;
        diffusion_params.timesteps        = &timesteps_tensor;
        diffusion_params.ref_image_params = ref_image_params;
        sd::guidance::GuidanceInput step_guidance_input;
        step_guidance_input.step          = step;
        step_guidance_input.schedule_size = sigmas.size();
        bool is_skiplayer_step            = skip_layer_guidance.is_enabled_for_step(step_guidance_input);

        compute_sample_controls(control_image,
                                noised_input,
                                timesteps_tensor,
                                cond,
                                &controls);

        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        bool uncond_without_ref_latents = !img_uncond.empty() &&
                                          !ref_latents.empty() &&
                                          sd_version_supports_ref_latent_img_cfg(version);

        auto run_condition = [&](const SDCondition& condition,
                                 const sd::Tensor<float>* c_concat_override                 = nullptr,
                                 const std::vector<int>* local_skip_layers                  = nullptr,
                                 const std::vector<sd::Tensor<float>>* ref_latents_override = nullptr,
                                 bool use_uncond_ip                                         = false) -> sd::Tensor<float> {
            diffusion_params.context     = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
            diffusion_params.c_concat    = c_concat_override != nullptr ? c_concat_override : (condition.c_concat.empty() ? nullptr : &condition.c_concat);
            diffusion_params.y           = condition.c_vector.empty() ? nullptr : &condition.c_vector;
            diffusion_params.ref_latents = ref_latents_override != nullptr ? ref_latents_override : (condition.c_ref_images.empty() ? &ref_latents : &condition.c_ref_images);

            if (sd_version_is_unet(version)) {
                int nvf = -1;
                if (config_->animatediff_loaded && noised_input.dim() >= 4 && noised_input.shape()[3] > 1) {
                    nvf = static_cast<int>(noised_input.shape()[3]);
                }
                UNetDiffusionExtra unet_extra{nvf, &controls, control_strength};
                const auto& ip_tokens = use_uncond_ip ? ip_adapter_uncond_tokens : ip_adapter_tokens;
                if (!ip_tokens.empty()) {
                    unet_extra.ip_context = &ip_tokens;
                    unet_extra.ip_scale   = ip_adapter_strength;
                }
                diffusion_params.extra = unet_extra;
            } else if (sd_version_is_sd3(version)) {
                diffusion_params.extra = SkipLayerDiffusionExtra{local_skip_layers};
            } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) || sd_version_is_longcat(version) || sd_version_is_sefi_image(version)) {
                diffusion_params.extra = FluxDiffusionExtra{&guidance_tensor,
                                                            local_skip_layers};
            } else if (sd_version_is_anima(version)) {
                diffusion_params.extra = AnimaDiffusionExtra{condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids,
                                                             condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights};
            } else if (sd_version_is_wan(version)) {
                diffusion_params.extra = WanDiffusionExtra{vace_context.empty() ? nullptr : &vace_context,
                                                           vace_strength};
            } else if (sd_version_is_hunyuan_video(version)) {
                diffusion_params.extra = HunyuanVideoDiffusionExtra{
                    &guidance_tensor,
                    condition.extra_c_crossattns.empty() ? nullptr : &condition.extra_c_crossattns[0],
                    condition.c_vector.empty() ? nullptr : &condition.c_vector,
                    hunyuan_timestep_r_tensor.empty() ? nullptr : &hunyuan_timestep_r_tensor};
            } else if (version == VERSION_HIDREAM_O1) {
                diffusion_params.extra = HiDreamO1DiffusionExtra{
                    condition.c_input_ids.empty() ? nullptr : &condition.c_input_ids,
                    condition.c_position_ids.empty() ? nullptr : &condition.c_position_ids,
                    condition.c_token_types.empty() ? nullptr : &condition.c_token_types,
                    condition.c_vinput_mask.empty() ? nullptr : &condition.c_vinput_mask,
                    condition.c_image_embeds.empty() ? nullptr : &condition.c_image_embeds};
            } else if (sd_version_is_minimax_h3(version)) {
                diffusion_params.extra = MiniMaxH3DiffusionExtra{
                    condition.c_token_types.empty() ? nullptr : &condition.c_token_types,
                    condition.c_position_ids.empty() ? nullptr : &condition.c_position_ids,
                    condition.c_ref_audios.empty() ? nullptr : &condition.c_ref_audios,
                    condition.c_reference_blocks.empty() ? nullptr : &condition.c_reference_blocks,
                    audio_length,
                    std::isfinite(active_flow_shift) ? active_flow_shift : 12.f,
                    3.f};
            } else if (sd_version_is_ltxav(version)) {
                diffusion_params.extra = LTXAVDiffusionExtra{
                    nullptr,
                    audio_timesteps_tensor.empty() ? nullptr : &audio_timesteps_tensor,
                    audio_length,
                    frame_rate,
                    video_positions.empty() ? nullptr : &video_positions};
            } else if (sd_version_is_minit2i(version)) {
                diffusion_params.extra = MiniT2IDiffusionExtra{
                    condition.c_vector.empty() ? nullptr : &condition.c_vector};
            } else if (sd_version_is_sensenova_u1(version)) {
                diffusion_params.extra = SenseNovaU1DiffusionExtra{
                    condition.c_input_ids.empty() ? nullptr : &condition.c_input_ids};
            } else {
                diffusion_params.extra = std::monostate{};
            }

            sd::Tensor<float> cached_output;
            if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                return std::move(cached_output);
            }

            for (const auto& extension : generation_extensions) {
                extension->before_diffusion(diffusion_params, step);
            }

            auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
            if (output_opt.empty()) {
                LOG_ERROR("diffusion model compute failed");
                return sd::Tensor<float>();
            }

            step_cache.after_condition(&condition, noised_input, output_opt);
            return output_opt;
        };

        const SDCondition* positive_condition      = &cond;
        const sd::Tensor<float>* c_concat_override = nullptr;
        for (const auto& extension : generation_extensions) {
            const SDCondition& next_condition = extension->before_condition(step, *positive_condition);
            if (&next_condition != positive_condition) {
                positive_condition = &next_condition;
                if (positive_condition != &cond) {
                    c_concat_override = cond.c_concat.empty() ? nullptr : &cond.c_concat;
                }
                break;
            }
        }

        cond_out = run_condition(*positive_condition, c_concat_override);
        if (cond_out.empty()) {
            return {};
        }

        if (!uncond.empty()) {
            if (!step_cache.is_step_skipped()) {
                compute_sample_controls(control_image,
                                        noised_input,
                                        timesteps_tensor,
                                        uncond,
                                        &controls);
            }
            const std::vector<int>* uncond_skip_layers = nullptr;
            if (is_skiplayer_step && slg_uncond) {
                LOG_VERBOSE("Skipping layers at uncond step %d\n", step);
                uncond_skip_layers = &skip_layer_guidance.layers();
            }
            uncond_out = run_condition(uncond,
                                       uncond.c_concat.empty() ? nullptr : &uncond.c_concat,
                                       uncond_skip_layers,
                                       nullptr,
                                       true);
            if (uncond_out.empty()) {
                return {};
            }
        }
        if (!img_uncond.empty()) {
            img_uncond_out = run_condition(img_uncond,
                                           img_uncond.c_concat.empty() ? nullptr : &img_uncond.c_concat,
                                           nullptr,
                                           uncond_without_ref_latents ? &empty_ref_latents : nullptr,
                                           true);
            if (img_uncond_out.empty()) {
                return {};
            }
        }
        sd::guidance::GuidanceInput guidance_input;
        guidance_input.step            = step;
        guidance_input.schedule_size   = sigmas.size();
        guidance_input.pred_cond       = &cond_out;
        guidance_input.pred_uncond     = uncond_out.empty() ? nullptr : &uncond_out;
        guidance_input.pred_img_uncond = img_uncond_out.empty() ? nullptr : &img_uncond_out;

        sd::guidance::GuiderOutput guided = guidance_schedule.empty() ? primary_guidance.forward(guidance_input, {}) : primary_guidance.forward(guidance_input, {}, guidance_schedule[guidance_schedule.size() - 1 - step]);
        if (guided.pred.empty()) {
            return {};
        }

        if (is_skiplayer_step && slg_scale != 0.0f) {
            LOG_VERBOSE("Skipping layers at step %d\n", step);
            if (!step_cache.is_step_skipped()) {
                guidance_input.predict_skip_layer = [&]() -> sd::Tensor<float> {
                    return run_condition(cond,
                                         cond.c_concat.empty() ? nullptr : &cond.c_concat,
                                         &skip_layer_guidance.layers());
                };
            }
        }

        guided = skip_layer_guidance.forward(guidance_input, std::move(guided));
        if (guided.pred.empty()) {
            return {};
        }

        denoised = guided.pred * c_out + x * c_skip;
        sd::guidance::GuiderOutput output;
        output.pred = denoised;
        if (needs_uncond_denoised) {
            const sd::Tensor<float>& base_uncond = !img_uncond_out.empty()
                                                       ? img_uncond_out
                                                       : (!uncond_out.empty() ? uncond_out : cond_out);
            output.pred_uncond                   = base_uncond * c_out + x * c_skip;
        }
        if (cache_runtime.spectrum_enabled) {
            cache_runtime.spectrum.update(denoised);
        }
        if (!denoise_mask.empty()) {
            denoised = denoised * denoise_mask + sampling_init_latent * (1.0f - denoise_mask);
        }
        if (preview_needed && sd_should_preview_denoised()) {
            preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
        }
        report_sample_progress(step, steps, terminal_sigma_is_zero, &last_progress_us);
        output.pred = denoised;
        return output;
    };

    auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser, extra_sample_args, denoiser);
    if (x0_opt.empty()) {
        LOG_ERROR("Diffusion model sampling failed");
        if (control_net) {
            control_net->free_control_ctx();
        }
        return {};
    }

    auto x0 = std::move(x0_opt);
    sd_sample::log_sample_cache_summary(cache_runtime, steps);
    if (inverse_noise_scaling) {
        x0 = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x0);
    }
    x0 = denoiser->process_latent_out(std::move(x0));

    if (control_net) {
        control_net->free_control_ctx();
    }
    return x0;
}

int StableDiffusionGGML::get_vae_scale_factor() {
    if (sd_version_is_pid(version)) {
        return 1;
    }
    return first_stage_model->get_scale_factor();
}

int StableDiffusionGGML::get_diffusion_model_down_factor() {
    int down_factor = 8;  // unet
    if (sd_version_is_dit(version)) {
        if (sd_version_is_sensenova_u1(version)) {
            down_factor = 32;
        } else if (sd_version_is_wan(version) || sd_version_is_lingbot_video(version) || sd_version_is_minimax_h3(version)) {
            down_factor = 2;
        } else {
            down_factor = 1;
        }
    }
    return down_factor;
}

int StableDiffusionGGML::get_latent_channel() {
    int latent_channel = 4;
    if (sd_version_is_dit(version)) {
        if (sd_version_is_ltxav(version)) {
            latent_channel = 128;
        } else if (sd_version_is_minimax_h3(version)) {
            latent_channel = 24;
        } else if (version == VERSION_WAN2_2_TI2V) {
            latent_channel = 48;
        } else if (sd_version_is_hunyuan_video(version)) {
            latent_channel = 32;
        } else if (version == VERSION_HIDREAM_O1) {
            latent_channel = 3;
        } else if (version == VERSION_CHROMA_RADIANCE) {
            latent_channel = 3;
        } else if (sd_version_is_minit2i(version)) {
            latent_channel = 3;
        } else if (sd_version_is_sensenova_u1(version)) {
            latent_channel = 3;
        } else if (sd_version_is_pid(version)) {
            latent_channel = 3;
        } else if (sd_version_is_sefi_image(version)) {
            latent_channel = 144;
        } else if (sd_version_uses_flux2_vae(version)) {
            latent_channel = 128;
        } else if (sd_version_is_mage_flow(version)) {
            latent_channel = 128;
        } else {
            latent_channel = 16;
        }
    }
    return latent_channel;
}

int StableDiffusionGGML::get_image_channels() const {
    return version == VERSION_QWEN_IMAGE_LAYERED ? 4 : 3;
}

int StableDiffusionGGML::get_image_seq_len(int h, int w) {
    int vae_scale_factor = get_vae_scale_factor();
    return (h / vae_scale_factor) * (w / vae_scale_factor);
}

sd::Tensor<float> StableDiffusionGGML::generate_init_latent(int width,
                                                            int height,
                                                            int frames,
                                                            bool video) {
    int vae_scale_factor = get_vae_scale_factor();
    int W                = width / vae_scale_factor;
    int H                = height / vae_scale_factor;
    int T                = video_frames_to_latent_frames(frames);
    int C                = get_latent_channel();
    if (video) {
        return sd::zeros<float>({W, H, T, C, 1});
    }
    return sd::zeros<float>({W, H, C, 1});
}

int StableDiffusionGGML::video_frames_to_latent_frames(int frames) {
    int latent_frames = frames;
    if (sd_version_is_ltxav(version)) {
        latent_frames = ((frames - 1) / 8) + 1;
    } else if (sd_version_is_minimax_h3(version)) {
        latent_frames = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
    } else if (sd_version_is_wan(version) || sd_version_is_lingbot_video(version) || sd_version_is_hunyuan_video(version)) {
        latent_frames = ((frames - 1) / 4) + 1;
    }
    return latent_frames;
}

int StableDiffusionGGML::latent_frames_to_video_frames(int latent_frames) {
    if (latent_frames <= 0) {
        return latent_frames;
    }
    if (sd_version_is_ltxav(version)) {
        return (latent_frames - 1) * 8 + 1;
    }
    if (sd_version_is_minimax_h3(version)) {
        return latent_frames <= 2 ? 5 : ((latent_frames - 2) / 5) * 17 + 5;
    }
    if (sd_version_is_wan(version) || sd_version_is_lingbot_video(version) || sd_version_is_hunyuan_video(version)) {
        return (latent_frames - 1) * 4 + 1;
    }
    return latent_frames;
}

int StableDiffusionGGML::align_video_frames(int frames) {
    if (sd_version_is_minimax_h3(version)) {
        frames = std::max(frames, 5);
        while (frames % 17 != 5) {
            ++frames;
        }
        return frames;
    }
    return latent_frames_to_video_frames(video_frames_to_latent_frames(frames));
}

sd::Tensor<float> StableDiffusionGGML::encode_to_vae_latents(const sd::Tensor<float>& x) {
    auto latents = first_stage_model->encode(n_threads, x, vae_tiling_params, circular_x, circular_y);
    if (latents.empty()) {
        return {};
    }
    latents = first_stage_model->vae_output_to_latents(latents, rng);
    return latents;
}

sd::Tensor<float> StableDiffusionGGML::encode_first_stage(const sd::Tensor<float>& x) {
    auto latents = encode_to_vae_latents(x);
    if (latents.empty()) {
        return {};
    }
    if (version != VERSION_SD1_PIX2PIX) {
        latents = first_stage_model->vae_to_diffusion_latents(latents);
    }
    return latents;
}

sd::Tensor<float> StableDiffusionGGML::decode_first_stage(const sd::Tensor<float>& x, bool decode_video) {
    if (sd_version_is_pid(version) || sd_version_is_minit2i(version)) {
        return sd::ops::clamp((x + 1.f) * 0.5f, 0.0f, 1.0f);
    }
    auto latents                      = first_stage_model->diffusion_to_vae_latents(x);
    auto decoded                      = first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
    const bool prefer_temporal_tiling = decode_video && first_stage_model->can_temporal_tile_decode();
    while (decoded.empty() &&
           sd::backend_fit::prepare_vae_decode_retry_tiling(vae_tiling_params, prefer_temporal_tiling)) {
        decoded = first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
    }
    return decoded;
}

sd::Tensor<float> StableDiffusionGGML::normalize_ltx_video_latents(const sd::Tensor<float>& x) {
    auto ltx_vae = std::dynamic_pointer_cast<LTXVideoVAE>(first_stage_model);
    if (!ltx_vae) {
        LOG_ERROR("LTX latent normalization requires LTX video VAE");
        return {};
    }
    return ltx_vae->normalize_latents(n_threads, x);
}

sd::Tensor<float> StableDiffusionGGML::un_normalize_ltx_video_latents(const sd::Tensor<float>& x) {
    auto ltx_vae = std::dynamic_pointer_cast<LTXVideoVAE>(first_stage_model);
    if (!ltx_vae) {
        LOG_ERROR("LTX latent un-normalization requires LTX video VAE");
        return {};
    }
    return ltx_vae->un_normalize_latents(n_threads, x);
}

sd::Tensor<float> StableDiffusionGGML::decode_ltx_audio_latent(const sd::Tensor<float>& audio_latent) {
    if (audio_vae_model == nullptr || audio_latent.empty()) {
        return {};
    }
    auto waveform = audio_vae_model->decode(n_threads, audio_latent);
    return waveform;
}

void StableDiffusionGGML::set_flow_shift(float flow_shift) {
    auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
    if (flow_denoiser) {
        if (flow_shift == INFINITY) {
            flow_shift = default_flow_shift;
        }
        flow_denoiser->set_shift(flow_shift);
        active_flow_shift = flow_shift;
    }
}

bool StableDiffusionGGML::is_flow_denoiser() {
    auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
    return !!flow_denoiser;
}

std::string StableDiffusionGGML::get_default_ref_image_preset(SDVersion version) const {
    if (sd_version_is_longcat(version)) {
        return "longcat";
    } else if (sd_version_is_flux(version)) {
        return "flux_kontext";
    } else if (sd_version_is_flux2(version) || sd_version_is_sefi_image(version)) {
        return "flux2";
    } else if (version == VERSION_QWEN_IMAGE_LAYERED) {
        return "qwen_layered";
    } else if (sd_version_is_qwen_image(version)) {
        return "qwen";
    } else if (sd_version_is_mage_flow(version)) {
        return "mage_flow";
    } else if (sd_version_is_z_image(version) || sd_version_is_boogu_image(version)) {
        return "z_image_omni";
    } else if (sd_version_is_krea2(version)) {
        // have to make a choice between "krea2_edit" mode (for lbouaraba/krea2edit)
        // and "krea2_ostris_edit" (for krea2 ostris edit)
        // since krea2 ostris edit support predates, it should probably be default
        return "krea2_ostris_edit";
    } else if (sd_version_is_anima(version)) {
        return "cosmos_reference";
    }
    return "default";
}

RefImageParams StableDiffusionGGML::resolve_ref_image_params(const char* ref_image_args) const {
    RefImageParams params;
    std::string preset_name = get_default_ref_image_preset(version);

    for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
        if (key == "preset") {
            std::string requested_preset_name = value;
            if (REF_IMAGE_PRESETS.count(requested_preset_name)) {
                preset_name = requested_preset_name;
            } else if (value != "default") {
                std::string valid_list;
                for (auto const& [name, _] : REF_IMAGE_PRESETS) {
                    valid_list += (valid_list.empty() ? "" : ", ") + name;
                }
                LOG_WARN("ignoring invalid reference image preset '%s'. Valid options: [%s]", value.c_str(), valid_list.c_str());
            }
            break;
        }
    }
    if (preset_name != "default") {
        LOG_INFO("Using '%s' preset for reference images", preset_name.c_str());
        params = REF_IMAGE_PRESETS.at(preset_name);
    }

    for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
        if (key == "pass_to_vlm") {
            if (!parse_strict_bool(value, params.pass_to_vlm)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "pass_to_dit") {
            if (!parse_strict_bool(value, params.pass_to_dit)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "ref_index_mode") {
            if (value == "fixed") {
                params.ref_index_mode = Rope::RefIndexMode::FIXED;
            } else if (value == "increase") {
                params.ref_index_mode = Rope::RefIndexMode::INCREASE;
            } else if (value == "decrease") {
                params.ref_index_mode = Rope::RefIndexMode::DECREASE;
            } else {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "force_ref_timestep_zero") {
            if (!parse_strict_bool(value, params.force_ref_timestep_zero)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "resize_before_vae") {
            if (!parse_strict_bool(value, params.resize_before_vae)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "vae_input_max_pixels") {
            if (!parse_strict_int(value, params.vae_input_max_pixels)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "vlm_resize_mode") {
            if (value == "longest_side") {
                params.vlm_resize_mode = RefImageResizeMode::LONGEST_SIDE;
            } else if (value == "area") {
                params.vlm_resize_mode = RefImageResizeMode::AREA;
            } else if (value == "none") {
                params.vlm_resize_mode = RefImageResizeMode::NONE;
            } else {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "vlm_max_size") {
            if (!parse_strict_int(value, params.vlm_max_size)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key == "vlm_min_size") {
            if (!parse_strict_int(value, params.vlm_min_size)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            }
        } else if (key != "preset" && key != "vlm_size") {
            LOG_WARN("ignoring unknown reference image arg '%s'", key.c_str());
        }
    }
    for (const auto& [key, value] : parse_key_value_args(ref_image_args, "reference image args")) {
        if (key == "vlm_size") {
            int vlm_size;
            if (!parse_strict_int(value, vlm_size)) {
                LOG_WARN("ignoring invalid reference image arg '%s=%s'", key.c_str(), value.c_str());
            } else {
                LOG_INFO("vlm_size override: setting both min and max size to %ld", (long)vlm_size);
                params.vlm_min_size = vlm_size;
                params.vlm_max_size = vlm_size;
            }
            break;
        }
    }
    if (params.force_ref_timestep_zero && !sd_version_is_krea2(version)) {
        LOG_WARN("force_ref_timestep_zero is only supported by Krea2 architecture for now");
    }
    return params;
}

void StableDiffusionGGML::apply_circular_axes(bool circular_x, bool circular_y) {
    this->circular_x = circular_x;
    this->circular_y = circular_y;
    if (this->diffusion_model) {
        this->diffusion_model->set_circular_axes(circular_x, circular_y);
    }
    if (this->high_noise_diffusion_model) {
        this->high_noise_diffusion_model->set_circular_axes(circular_x, circular_y);
    }
    if (this->control_net) {
        this->control_net->set_circular_axes(circular_x, circular_y);
    }
    if (circular_x || circular_y) {
        LOG_INFO("Using circular padding for convolutions (x=%s, y=%s)",
                 circular_x ? "true" : "false",
                 circular_y ? "true" : "false");
    }
}
