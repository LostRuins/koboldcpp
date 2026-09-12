#ifndef __SD_PIPELINE_DIFFUSION_ENGINE_H__
#define __SD_PIPELINE_DIFFUSION_ENGINE_H__

#include <atomic>
#include <cmath>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/ggml_graph_cut.h"
#include "core/tensor.hpp"
#include "core/util.h"
#include "model/adapter/lora.hpp"
#include "model_builders.h"
#include "model_manager.h"
#include "stable-diffusion.h"

class RNG;
struct Denoiser;
struct LoraModel;
struct ConditionerParams;
struct SDCondition;
struct RefImageParams;

extern const char* model_version_to_str[];

static inline bool sd_version_supports_ref_latent_img_cfg(SDVersion version) {
    return version == VERSION_FLUX ||
           sd_version_is_flux2(version) ||
           sd_version_is_qwen_image(version) ||
           sd_version_is_mage_flow(version) ||
           sd_version_is_longcat(version) ||
           sd_version_is_z_image(version) ||
           sd_version_is_boogu_image(version);
}

class StableDiffusionGGML {
public:
    SDBackendManager backend_manager;

    SDVersion version;
    bool external_vae_is_invalid = false;

    bool circular_x = false;
    bool circular_y = false;

    std::shared_ptr<RNG> rng;
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float default_flow_shift         = INFINITY;
    float active_flow_shift          = INFINITY;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd or wan2.1 i2v
    std::shared_ptr<DiffusionModelRunner> diffusion_model;
    std::shared_ptr<DiffusionModelRunner> high_noise_diffusion_model;
    std::shared_ptr<VAE> first_stage_model;
    std::shared_ptr<VAE> preview_vae;
    std::shared_ptr<AudioVAERunner> audio_vae_model;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<IPAdapter::IPAdapterRunner> ip_adapter;
    sd::Tensor<float> ip_adapter_tokens;
    sd::Tensor<float> ip_adapter_uncond_tokens;
    float ip_adapter_strength = 1.0f;
    std::vector<std::shared_ptr<GenerationExtension>> generation_extensions;
    struct RuntimeLora {
        ModelManager::LoraSpec spec;
        SDBackendModule module;
        std::shared_ptr<LoraModel> model;

        bool matches(const ModelManager::LoraSpec& other) const {
            return spec.file_id == other.file_id && spec.file_revision == other.file_revision &&
                   spec.tensor_name_prefix_filter == other.tensor_name_prefix_filter;
        }
    };
    std::vector<RuntimeLora> runtime_lora_models;
    bool apply_lora_immediately = false;
    int animatediff_num_frames  = 0;
    // kcpp
    std::map<std::string, std::shared_ptr<LoraModel>> kcpp_lora_cache;
    bool kcpp_lora_cache_populate = false;
    std::string kcpp_taesd_path;
    // kcpp

    std::string taesd_path;
    sd_tiling_params_t vae_tiling_params = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
    bool enable_mmap                     = false;
    sd::ggml_graph_cut::MaxVramAssignment max_vram_assignment;
    bool disable_prefetch          = false;
    bool disable_segmented_compute = false;
    bool eager_load                = false;
    std::string backend_spec;
    std::string params_backend_spec;
    std::string split_mode_spec;
    bool auto_fit_enabled = false;

    bool diffusion_conv_direct = false;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    std::shared_ptr<ModelManager> model_manager;

    enum class RunnerGroup { Core,
                             VAE,
                             ControlNet,
                             Extensions };
    using RunnerGroups = std::set<RunnerGroup>;

    struct ModelConfig {
        sd_ctx_params_t params{};
        std::list<std::string> strings;
        std::vector<sd_embedding_t> embeddings;
        ModelLoader::FileId control_net_file = 0;
        bool use_tae                         = false;
        bool use_audio_vae                   = false;
        bool photomaker_source_available     = false;
        bool animatediff_loaded              = false;

        explicit ModelConfig(const sd_ctx_params_t& initial)
            : params(initial) {
            for (auto member : {&sd_ctx_params_t::model_path, &sd_ctx_params_t::clip_l_path,
                                &sd_ctx_params_t::clip_g_path, &sd_ctx_params_t::clip_vision_path,
                                &sd_ctx_params_t::t5xxl_path, &sd_ctx_params_t::llm_path,
                                &sd_ctx_params_t::llm_vision_path, &sd_ctx_params_t::diffusion_model_path,
                                &sd_ctx_params_t::high_noise_diffusion_model_path, &sd_ctx_params_t::uncond_diffusion_model_path,
                                &sd_ctx_params_t::embeddings_connectors_path, &sd_ctx_params_t::vae_path,
                                &sd_ctx_params_t::audio_vae_path, &sd_ctx_params_t::taesd_path,
                                &sd_ctx_params_t::control_net_path, &sd_ctx_params_t::ip_adapter_path,
                                &sd_ctx_params_t::motion_module_path, &sd_ctx_params_t::photo_maker_path,
                                &sd_ctx_params_t::pulid_weights_path, &sd_ctx_params_t::tensor_type_rules,
                                &sd_ctx_params_t::max_vram, &sd_ctx_params_t::backend,
                                &sd_ctx_params_t::params_backend, &sd_ctx_params_t::split_mode,
                                &sd_ctx_params_t::rpc_servers, &sd_ctx_params_t::model_args}) {
                strings.emplace_back(SAFE_STR(initial.*member));
                params.*member = strings.back().c_str();
            }
            for (uint32_t i = 0; i < initial.embedding_count; ++i) {
                strings.emplace_back(SAFE_STR(initial.embeddings[i].name));
                const char* name = strings.back().c_str();
                strings.emplace_back(SAFE_STR(initial.embeddings[i].path));
                embeddings.push_back({name, strings.back().c_str()});
            }
            params.embeddings = embeddings.data();
        }

        ModelConfig(const ModelConfig& other)
            : ModelConfig(other.params) {
            control_net_file            = other.control_net_file;
            use_tae                     = other.use_tae;
            use_audio_vae               = other.use_audio_vae;
            photomaker_source_available = other.photomaker_source_available;
            animatediff_loaded          = other.animatediff_loaded;
        }
        ModelConfig& operator=(const ModelConfig&) = delete;

        void set_control_net(ModelLoader::FileId id, const std::string& path) {
            control_net_file = id;
            strings.push_back(path);
            params.control_net_path = strings.back().c_str();
        }
    };

    struct RunnerState {
        bool ready                = false;
        uint64_t catalog_revision = 0;
        std::map<RunnerGroup, ModelLoader::FileVersions> sources;
    };

    std::recursive_mutex execution_mutex;
    std::unique_ptr<ModelConfig> config_;
    RunnerState runner_state_;
    bool executing_ = false;

    std::shared_ptr<Denoiser> denoiser;
    std::vector<float> file_alphas_cumprod;

    StableDiffusionGGML();
    ~StableDiffusionGGML();

    static const std::map<RunnerGroup, std::set<ModelComponent>>& runner_components();

    static RunnerGroups all_runner_groups();

    ModelLoader::FileVersions runner_source_versions(RunnerGroup group, const ModelLoader& loader) const;

    void capture_runner_sources();

    void end_runners();

    bool reset_runners(const RunnerGroups& groups);

    bool refresh_model_sources();

    bool apply_model_update(ModelLoader candidate,
                            std::unique_ptr<ModelConfig> next_config = nullptr,
                            RunnerGroups groups                      = {});

    struct ContextOperation {
        StableDiffusionGGML& sd;
        std::unique_lock<std::recursive_mutex> lock;
        bool acquired = false;

        explicit ContextOperation(StableDiffusionGGML& sd)
            : sd(sd), lock(sd.execution_mutex, std::try_to_lock) {
            if (!lock.owns_lock() || sd.executing_) {
                // The caller may be a log callback, so rejecting it must not log.
                return;
            }
            sd.executing_ = true;
            acquired      = true;
        }

        ~ContextOperation() {
            if (acquired) {
                sd.executing_ = false;
            }
        }
    };

    struct ExecutionScope {
        ContextOperation operation;
        bool ready = false;

        explicit ExecutionScope(StableDiffusionGGML& sd)
            : operation(sd) {
            ready = operation.acquired && sd.refresh_model_sources();
        }

        ~ExecutionScope() {
            if (ready) {
                operation.sd.end_runners();
            }
        }
    };

    ggml_backend_t backend_for(SDBackendModule module);

    ggml_backend_t params_backend_for(SDBackendModule module);

    std::atomic<sd_cancel_mode_t> cancellation_flag = SD_CANCEL_RESET;

    void set_cancel_flag(enum sd_cancel_mode_t flag);

    void reset_cancel_flag();

    enum sd_cancel_mode_t get_cancel_flag();

    size_t max_graph_vram_bytes_for_module(SDBackendModule module);

    std::vector<size_t> layer_split_vram_limits_for_backends(const std::vector<ggml_backend_t>& backends);

    bool ensure_backend_pair(SDBackendModule module);

    template <typename T>
    bool register_runner_params(ModelComponent component,
                                const std::shared_ptr<T>& model,
                                SDBackendModule module,
                                size_t* params_mem_size = nullptr);

    template <typename T>
    bool register_row_split_runner_params(ModelComponent component,
                                          const std::shared_ptr<T>& model,
                                          SDBackendModule module,
                                          const std::vector<ggml_backend_t>& module_backends,
                                          std::map<std::string, ggml_tensor*> group_tensors,
                                          const std::map<ggml_tensor*, enum ggml_op>& tensor_ops,
                                          ModelManager::ResidencyMode residency_mode,
                                          size_t* params_mem_size);

    // Register graph-cut layer-split tensors on the primary backend first.
    // The first real graph assigns each param tensor to a runtime backend
    // before weights are loaded or staged.
    template <typename T>
    bool register_layer_split_runner_params(ModelComponent component,
                                            const std::shared_ptr<T>& model,
                                            SDBackendModule module,
                                            const std::vector<ggml_backend_t>& module_backends,
                                            std::map<std::string, ggml_tensor*> group_tensors,
                                            const std::map<ggml_tensor*, enum ggml_op>& tensor_ops,
                                            ModelManager::ResidencyMode residency_mode,
                                            size_t* params_mem_size);

    bool unload_control_net();

    bool load_control_net_from_file(const std::string& path);

    void apply_circular_axes(bool circular_x, bool circular_y);

    bool init_backend();

    bool row_split_active();

    bool graph_cut_layer_split_active();

    std::shared_ptr<RNG> get_rng(rng_type_t rng_type);

    void refresh_compvis_denoiser_sigmas();

    void load_alphas_cumprod();

    bool init_model_loader(ModelLoader& model_loader, ModelConfig& configuration);

    bool init(const sd_ctx_params_t* sd_ctx_params);

    bool uses_tae() const;

    bool tae_preview_only() const;

    void configure_weight_loading();

    sd::model_builders::Context model_build_context();

    bool build_core_runners();

    bool build_vae_runners();

    bool build_control_net_runner();

    bool build_extension_runners();

    bool validate_and_load_runners();

    bool build_denoiser();

    bool build_runners(const RunnerGroups& groups);

    bool is_using_v_parameterization_for_sd2(bool is_inpaint = false);

    static std::string lora_log_id(const ModelManager::LoraSpec& lora);

    std::shared_ptr<LoraModel> load_lora_model(const ModelManager::LoraSpec& lora_spec,
                                               SDBackendModule module,
                                               LoraModel::filter_t module_filter = nullptr);

    void clear_lora_adapters();

    std::vector<std::shared_ptr<LoraModel>> load_runtime_loras_for_module(const std::vector<ModelManager::LoraSpec>& loras,
                                                                          const std::set<std::string>& model_tensor_names,
                                                                          SDBackendModule module,
                                                                          LoraModel::filter_t module_filter,
                                                                          bool& success,
                                                                          std::vector<RuntimeLora>& next_models);

    bool apply_loras_immediately(const std::vector<ModelManager::LoraSpec>& loras);

    bool apply_loras_at_runtime(const std::vector<ModelManager::LoraSpec>& loras);

    void lora_stat();

    bool apply_loras(const sd_lora_t* loras, uint32_t lora_count);

    void reset_generation_extensions();

    void prepare_generation_extensions(const sd_pm_params_t& pm_params,
                                       const sd_pulid_params_t& pulid_params,
                                       ConditionerParams& condition_params,
                                       int total_steps);

    sd::Tensor<float> get_clip_vision_output(const sd::Tensor<float>& image,
                                             bool return_pooled   = true,
                                             int clip_skip        = -1,
                                             bool zero_out_masked = false);

    void compute_ip_adapter_tokens(const sd_image_t& image, float strength);

    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask,
                                         int step);

    std::vector<float> process_ltxav_video_timesteps(const std::vector<float>& timesteps,
                                                     const sd::Tensor<float>& init_latent,
                                                     const sd::Tensor<float>& denoise_mask);

    void preview_image(int step,
                       const sd::Tensor<float>& latents,
                       enum SDVersion version,
                       preview_t preview_mode,
                       std::function<void(int, int, sd_image_t*, bool, void*)> step_callback,
                       void* step_callback_data,
                       bool is_noisy);

    std::vector<float> prepare_sample_timesteps(float sigma,
                                                int shifted_timestep);

    void adjust_sample_step_scalings(int shifted_timestep,
                                     const std::vector<float>& timesteps_vec,
                                     float c_in,
                                     float* c_skip,
                                     float* c_out);

    struct SamplePreviewContext {
        sd_preview_cb_t callback = nullptr;
        void* data               = nullptr;
        preview_t mode           = PREVIEW_NONE;
    };

    SamplePreviewContext prepare_sample_preview_context();

    void report_sample_progress(int step,
                                size_t total_steps,
                                bool terminal_sigma_is_zero,
                                int64_t* last_progress_us);

    void compute_sample_controls(const sd::Tensor<float>& control_image,
                                 const sd::Tensor<float>& noised_input,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const SDCondition& condition,
                                 std::vector<sd::Tensor<float>>* controls);

    sd::Tensor<float> sample(const std::shared_ptr<DiffusionModelRunner>& work_diffusion_model,
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
                             const sd::Tensor<float>& video_positions = {});

    int get_vae_scale_factor();

    int get_diffusion_model_down_factor();

    int get_latent_channel();

    int get_image_channels() const;

    int get_image_seq_len(int h, int w);

    sd::Tensor<float> generate_init_latent(int width,
                                           int height,
                                           int frames = 1,
                                           bool video = false);

    int video_frames_to_latent_frames(int frames);

    int latent_frames_to_video_frames(int latent_frames);

    int align_video_frames(int frames);

    sd::Tensor<float> encode_to_vae_latents(const sd::Tensor<float>& x);

    sd::Tensor<float> encode_first_stage(const sd::Tensor<float>& x);

    sd::Tensor<float> decode_first_stage(const sd::Tensor<float>& x, bool decode_video = false);

    sd::Tensor<float> normalize_ltx_video_latents(const sd::Tensor<float>& x);

    sd::Tensor<float> un_normalize_ltx_video_latents(const sd::Tensor<float>& x);

    sd::Tensor<float> decode_ltx_audio_latent(const sd::Tensor<float>& audio_latent);

    void set_flow_shift(float flow_shift = INFINITY);

    bool is_flow_denoiser();

    std::string get_default_ref_image_preset(SDVersion version) const;

    RefImageParams resolve_ref_image_params(const char* ref_image_args) const;
};

#endif  // __SD_PIPELINE_DIFFUSION_ENGINE_H__
