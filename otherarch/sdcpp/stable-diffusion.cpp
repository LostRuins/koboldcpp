#include "ggml_extend.hpp"
#include "ggml_graph_cut.h"

#include "model.h"
#include "rng.hpp"
#include "rng_mt19937.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "anima.hpp"
#include "auto_encoder_kl.hpp"
#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "ernie_image.hpp"
#include "esrgan.hpp"
#include "flux.hpp"
#include "guidance.h"
#include "hidream_o1.hpp"
#include "lens.hpp"
#include "lora.hpp"
#include "ltx_audio_vae.h"
#include "ltx_latent_upscaler.hpp"
#include "ltx_vae.hpp"
#include "ltxv.hpp"
#include "mmdit.hpp"
#include "pmid.hpp"
#include "qwen_image.hpp"
#include "sample-cache.h"
#include "tae.hpp"
#include "unet.hpp"
#include "upscaler.h"
#include "vae.hpp"
#include "wan.hpp"
#include "z_image.hpp"

#include "latent-preview.h"
#include "name_conversion.h"

#include <filesystem>

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
    "Qwen Image",
    "Anima",
    "Flux.2",
    "Flux.2 klein",
    "LTXAV",
    "HiDream O1",
    "Z-Image",
    "Ovis Image",
    "Ernie Image",
    "Lens",
    "Longcat-Image",
};

const char* sampling_methods_str[] = {
    "Euler",
    "Euler A",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD",
    "Res Multistep",
    "Res 2s",
    "ER-SDE",
    "Euler CFG++",
    "Euler A CFG++",
    "Euler GE",
};

/*================================================== Helper Functions ================================================*/

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

static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
    float reuse_threshold = params.reuse_threshold;
    if (reuse_threshold == INFINITY) {
        if (params.mode == SD_CACHE_EASYCACHE) {
            reuse_threshold = 0.2f;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0f;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    std::vector<MmapTensorStore> mmap_tensor_store;
    SDBackendManager backend_manager;

    SDVersion version;
    bool vae_decode_only         = false;
    bool external_vae_is_invalid = false;
    bool free_params_immediately = false;

    bool circular_x = false;
    bool circular_y = false;

    std::shared_ptr<RNG> rng         = std::make_shared<PhiloxRNG>();
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float default_flow_shift         = INFINITY;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd or wan2.1 i2v
    std::shared_ptr<DiffusionModelRunner> diffusion_model;
    std::shared_ptr<DiffusionModelRunner> high_noise_diffusion_model;
    std::shared_ptr<VAE> first_stage_model;
    std::shared_ptr<VAE> preview_vae;
    std::shared_ptr<LTXV::LTXAudioVAERunner> audio_vae_model;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;
    std::shared_ptr<PhotoMakerIDEmbed> pmid_id_embeds;
    std::vector<std::shared_ptr<LoraModel>> cond_stage_lora_models;
    std::vector<std::shared_ptr<LoraModel>> diffusion_lora_models;
    std::vector<std::shared_ptr<LoraModel>> first_stage_lora_models;
    bool apply_lora_immediately = false;
    std::map<std::string, std::shared_ptr<LoraModel>> kcpp_lora_cache;
    bool kcpp_lora_cache_populate = false;

    std::string taesd_path;
    sd_tiling_params_t vae_tiling_params = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
    bool offload_params_to_cpu           = false;
    float max_vram                       = 0.f;
    bool use_pmid                        = false;
    std::string backend_spec;
    std::string params_backend_spec;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    std::map<std::string, ggml_tensor*> tensors;

    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() = default;

    ggml_backend_t backend_for(SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.runtime_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    ggml_backend_t params_backend_for(SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.params_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    std::string toLowerCase(const std::string& str) {
        std::string result;
        std::locale loc;

        for (char ch : str) {
            result += std::tolower(ch, loc); // Use locale-aware tolower
        }

        return result;
    }

    bool ensure_backend_pair(SDBackendModule module) {
        if (backend_for(module) == nullptr) {
            return false;
        }
        return params_backend_for(module) != nullptr;
    }

    bool init_backend(const sd_ctx_params_t* sd_ctx_params) {
        std::string error;
        if (!backend_manager.init(sd_ctx_params->backend,
                                  sd_ctx_params->params_backend,
                                  sd_ctx_params->offload_params_to_cpu,
                                  sd_ctx_params->keep_clip_on_cpu,
                                  sd_ctx_params->keep_vae_on_cpu,
                                  sd_ctx_params->keep_control_net_on_cpu,
                                  &error)) {
            LOG_ERROR("backend config failed: %s", error.c_str());
            return false;
        }
        return ensure_backend_pair(SDBackendModule::DIFFUSION);
    }

    std::shared_ptr<RNG> get_rng(rng_type_t rng_type) {
        if (rng_type == STD_DEFAULT_RNG) {
            return std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CPU_RNG) {
            return std::make_shared<MT19937RNG>();
        } else {  // default: CUDA_RNG
            return std::make_shared<PhiloxRNG>();
        }
    }

    bool init(const sd_ctx_params_t* sd_ctx_params_kcpp) {
        // kcpp make sd_ctx_params mutable
        sd_ctx_params_t sd_ctx_params_local = *sd_ctx_params_kcpp;
        sd_ctx_params_t *sd_ctx_params = &sd_ctx_params_local;
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;
        max_vram                = sd_ctx_params->max_vram;
        backend_spec            = SAFE_STR(sd_ctx_params->backend);
        params_backend_spec     = SAFE_STR(sd_ctx_params->params_backend);

        bool use_tae       = false;
        bool use_audio_vae = false;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        if (!init_backend(sd_ctx_params)) {
            return false;
        }
        max_vram = sd::ggml_graph_cut::resolve_max_vram_gib(max_vram, backend_for(SDBackendModule::DIFFUSION));

        std::string clip_vision_fixed = SAFE_STR(sd_ctx_params->clip_vision_path);
        std::string clipg_path_fixed  = SAFE_STR(sd_ctx_params->clip_g_path);
        std::string clipl_path_fixed  = SAFE_STR(sd_ctx_params->clip_l_path);
        std::string llm_path_fixed    = SAFE_STR(sd_ctx_params->llm_path);
        std::string llm_vision_path_fixed = SAFE_STR(sd_ctx_params->llm_vision_path);
        std::string t5_path_fixed     = SAFE_STR(sd_ctx_params->t5xxl_path);
        std::string taesd_path_fixed  = SAFE_STR(sd_ctx_params->taesd_path);

        ModelLoader model_loader;

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

        if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
            LOG_INFO("loading high noise diffusion model from '%s'", sd_ctx_params->high_noise_diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->high_noise_diffusion_model_path, "model.high_noise_diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->high_noise_diffusion_model_path);
            }
        }

        bool is_unet = sd_version_is_unet(model_loader.get_sd_version());

        // begin kcpp replacements
        SDVersion tempver = model_loader.get_sd_version();

        // kcpp fallback to separate diffusion model passed as model
        if (tempver == VERSION_COUNT &&
        strlen(SAFE_STR(sd_ctx_params->model_path)) > 0 &&
        strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) == 0 &&
        (t5_path_fixed!=""||clipl_path_fixed!=""))
        {
            bool endswithsafetensors = ends_with(sd_ctx_params->model_path, ".safetensors");
            if(endswithsafetensors && !model_loader.has_diffusion_model_tensors())
            {
                LOG_INFO("SD Diffusion Model tensors missing! Fallback trying alternative tensor names...\n");
                if (!model_loader.init_from_file(sd_ctx_params->model_path, "model.diffusion_model.")) {
                    LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->model_path);
                }
                tempver = model_loader.get_sd_version();
            }
        }

        if (tempver == VERSION_ANIMA &&
            strlen(SAFE_STR(sd_ctx_params->model_path)) > 0 &&
            strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) == 0 &&
            !model_loader.has_diffusion_model_tensors()
            )
        {
            LOG_INFO("Anima: SD Diffusion Model tensors missing! Fallback trying alternative tensor names...\n");
            if (!model_loader.init_from_file(sd_ctx_params->model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->model_path);
            }
            tempver = model_loader.get_sd_version();
        }

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
        bool conditioner_is_llm = (is_qwenimg || iszimg || isflux2 || is_ovis || is_anima || is_ernie || is_longcat || is_lens);

        //kcpp qol fallback: if a llm was loaded as t5 by mistake
        if(conditioner_is_llm && t5_path_fixed!="")
        {
            if(clipl_path_fixed=="")
            {
                clipl_path_fixed = t5_path_fixed;
                t5_path_fixed = "";
            }
            else if(clipl_path_fixed!="" && clipg_path_fixed=="")
            {
                //very tricky case. see if we can tell if clipl is an mmproj, if so move to right place
                if(toLowerCase(clipl_path_fixed).find("mmproj") != std::string::npos)
                {
                    clipg_path_fixed = clipl_path_fixed;
                    clipl_path_fixed = t5_path_fixed;
                    t5_path_fixed = "";
                }
            }
        }

        //settle clip-l replacements
        if (clipl_path_fixed!="")
        {
            if(conditioner_is_llm && llm_path_fixed=="")
            {
                llm_path_fixed = clipl_path_fixed;
                clipl_path_fixed = "";
            }
            else if(iswan)
            {
                if(t5_path_fixed=="")
                {
                    t5_path_fixed    = clipl_path_fixed;
                    clipl_path_fixed = "";
                } else if (t5_path_fixed != "" && clip_vision_fixed == "") {
                    clip_vision_fixed = clipl_path_fixed;
                    clipl_path_fixed  = "";
                }
            }
        }

        //settle clip-g replacements
        if (clipg_path_fixed!="")
        {
            if(iswan && clip_vision_fixed=="")
            {
                clip_vision_fixed = clipg_path_fixed;
                clipg_path_fixed = "";
            }
            else if(is_qwenimg && llm_vision_path_fixed=="")
            {
                llm_vision_path_fixed = clipg_path_fixed;
                clipg_path_fixed = "";
            }
        }

        //settle possible inversions for mmproj
        if(llm_vision_path_fixed!="" && llm_path_fixed!="")
        {
            if(toLowerCase(llm_vision_path_fixed).find("mmproj") == std::string::npos &&
            toLowerCase(llm_path_fixed).find("mmproj") != std::string::npos)
            {
                std::string tmp = llm_path_fixed;
                llm_path_fixed = llm_vision_path_fixed;
                llm_vision_path_fixed = tmp;
            }
        }

        //settle tae replacements
        if(taesd_path_fixed != "")
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
            else if(sd_version_is_flux(tempver)||sd_version_is_z_image(tempver)||tempver == VERSION_OVIS_IMAGE||is_longcat)
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
            else if(is_wan21||is_qwenimg||sd_version_is_anima(tempver))
            {
                to_replace = "taesd_w21.embd";
            }

            if(to_replace!="")
            {
                size_t pos = taesd_path_fixed.find(to_search);
                if (pos != std::string::npos) {
                    taesd_path_fixed.replace(pos, to_search.length(), to_replace);
                }
            }
            else
            {
                printf("\nCannot use TAESD: Unknown tempver %d. TAESD Disabled!\n",tempver);
                taesd_path_fixed = "";
            }
            if (taesd_path_fixed != "" && !file_exists(taesd_path_fixed))
            {
                printf("\nCannot use TAESD: \"%s\" not found. TAESD Disabled!\n", taesd_path_fixed.c_str());
                taesd_path_fixed = "";
            }
        }
        if (strlen(SAFE_STR(sd_ctx_params->photo_maker_path)) > 0 && tempver != VERSION_SDXL) {
            printf("\nWARNING: PhotoMaker is only compatible with SDXL models. PhotoMaker will be disabled!\n");
            sd_ctx_params->photo_maker_path = "";
        }

        sd_ctx_params->clip_g_path      = clipg_path_fixed.c_str();
        sd_ctx_params->clip_l_path      = clipl_path_fixed.c_str();
        sd_ctx_params->clip_vision_path = clip_vision_fixed.c_str();
        sd_ctx_params->llm_path         = llm_path_fixed.c_str();
        sd_ctx_params->llm_vision_path  = llm_vision_path_fixed.c_str();
        sd_ctx_params->t5xxl_path       = t5_path_fixed.c_str();
        sd_ctx_params->taesd_path       = taesd_path_fixed.c_str();
        //debug print
        // printf("\n\nclip_g: %s\nclip_l: %s\nclip_vision: %s\nllm: %s\nllm_vision: %s\nt5xxl: %s\ntaesd: %s\n",
        // sd_ctx_params->clip_g_path, sd_ctx_params->clip_l_path, sd_ctx_params->clip_vision_path,
        // sd_ctx_params->llm_path, sd_ctx_params->llm_vision_path, sd_ctx_params->t5xxl_path,
        // taesd_path.c_str());
        // end kcpp replacements

        if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
            LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
            std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, prefix)) {
                LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
            LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
            std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, prefix)) {
                LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
            LOG_INFO("loading clip_vision from '%s'", sd_ctx_params->clip_vision_path);
            std::string prefix = "cond_stage_model.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_vision_path, prefix)) {
                LOG_WARN("loading clip_vision from '%s' failed", sd_ctx_params->clip_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
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
            LOG_INFO("loading LTX audio VAE from '%s'", sd_ctx_params->audio_vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->audio_vae_path)) {
                LOG_WARN("loading LTX audio VAE weights from '%s' failed", sd_ctx_params->audio_vae_path);
            } else {
                use_audio_vae = true;
            }
        }

        model_loader.convert_tensors_name();

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
            return false;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        ggml_type wtype               = (int)sd_ctx_params->wtype < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)
                                            ? (ggml_type)sd_ctx_params->wtype
                                            : GGML_TYPE_COUNT;
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        //kcpp: patch hidream to fix broken images on vulkan https://github.com/leejet/stable-diffusion.cpp/issues/1496
        if(version == VERSION_HIDREAM_O1 && tensor_type_rules.size()==0)
        {
            tensor_type_rules = "^model.language_model.layers.[0-9]+.mlp.down_proj.weight=bf16";
        }
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }

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

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (sd_ctx_params->lora_apply_mode == LORA_APPLY_AUTO) {
            bool have_quantized_weight = false;
            if (wtype != GGML_TYPE_COUNT && ggml_is_quantized(wtype)) {
                have_quantized_weight = true;
            } else {
                for (const auto& [type, _] : wtype_stat) {
                    if (ggml_is_quantized(type)) {
                        have_quantized_weight = true;
                        break;
                    }
                }
            }
            if (have_quantized_weight) {
                apply_lora_immediately = false;
            } else {
                apply_lora_immediately = true;
            }
        } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
            apply_lora_immediately = true;
        } else {
            apply_lora_immediately = false;
        }

        std::map<std::string, ggml_tensor*> mmap_able_tensors;
        bool enable_mmap_tensors = false;
        bool needs_writable_mmap = false;
        if (sd_ctx_params->enable_mmap) {
            if (apply_lora_immediately) {
                needs_writable_mmap = true;
                LOG_WARN("in mode 'immediately', LoRAs will cause extra memory usage with mmap");
            }
            enable_mmap_tensors = true;
        }

        // split definition to avoid msvc choking on the extra parameter handling
        auto module_can_mmap = [&](SDBackendModule module) {
            return enable_mmap_tensors &&
                   (backend_manager.runtime_backend_is_cpu(module) ||
                    backend_manager.params_backend_is_cpu(module) ||
                    backend_manager.runtime_backend_supports_host_buffer(module));
        };

        auto get_param_tensors_p = [&](auto&& model, bool do_mmap, const char* prefix) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp, prefix);
            for (const auto& [key, tensor] : temp) {
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        auto get_param_tensors = [&](auto&& model, bool do_mmap) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp);
            for (const auto& [key, tensor] : temp) {
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        if (sd_version_is_control(version)) {
            // Might need vae encode for control cond
            vae_decode_only = false;
        }
        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            tae_preview_only = false;
            use_tae          = true;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(max_vram);

        {
            if (!ensure_backend_pair(SDBackendModule::TE) ||
                !ensure_backend_pair(SDBackendModule::DIFFUSION)) {
                return false;
            }

            if (sd_version_is_sd3(version)) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                     params_backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map);
                diffusion_model  = std::make_shared<MMDiTRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                params_backend_for(SDBackendModule::DIFFUSION),
                                                                tensor_storage_map,
                                                                "model.diffusion_model");
            } else if (sd_version_is_flux(version)) {
                bool is_chroma = false;
                for (auto pair : tensor_storage_map) {
                    if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                        is_chroma = true;
                        break;
                    }
                }
                if (is_chroma) {
                    if ((sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) && sd_ctx_params->chroma_use_dit_mask) {
                        LOG_WARN(
                            "!!!It looks like you are using Chroma with flash attention. "
                            "This is currently unsupported. "
                            "If you find that the generated images are broken, "
                            "try either disabling flash attention or specifying "
                            "--chroma-disable-dit-mask as a workaround.");
                    }

                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                        params_backend_for(SDBackendModule::TE),
                                                                        tensor_storage_map,
                                                                        sd_ctx_params->chroma_use_t5_mask,
                                                                        sd_ctx_params->chroma_t5_mask_pad);
                } else if (version == VERSION_OVIS_IMAGE) {
                    cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                     params_backend_for(SDBackendModule::TE),
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     false);
                } else {
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                          params_backend_for(SDBackendModule::TE),
                                                                          tensor_storage_map);
                }
                diffusion_model = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_ltxav(version)) {
                cond_stage_model = std::make_shared<LTXAVEmbedder>(backend_for(SDBackendModule::TE),
                                                                   params_backend_for(SDBackendModule::TE),
                                                                   tensor_storage_map);
                diffusion_model  = std::make_shared<LTXV::LTXAVRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                      params_backend_for(SDBackendModule::DIFFUSION),
                                                                      tensor_storage_map,
                                                                      "model.diffusion_model");
            } else if (sd_version_is_wan(version)) {
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(backend_for(SDBackendModule::TE),
                                                                    params_backend_for(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true);
                diffusion_model  = std::make_shared<WAN::WanRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                   params_backend_for(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model",
                                                                   version);
                if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                    high_noise_diffusion_model = std::make_shared<WAN::WanRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                  params_backend_for(SDBackendModule::DIFFUSION),
                                                                                  tensor_storage_map,
                                                                                  "model.high_noise_diffusion_model",
                                                                                  version);
                }
                if (diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-FLF2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-I2V-1.3B") {
                    if (!ensure_backend_pair(SDBackendModule::CLIP_VISION)) {
                        return false;
                    }
                    clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend_for(SDBackendModule::CLIP_VISION),
                                                                             params_backend_for(SDBackendModule::CLIP_VISION),
                                                                             tensor_storage_map);
                    clip_vision->set_max_graph_vram_bytes(max_graph_vram_bytes);
                    get_param_tensors(clip_vision, module_can_mmap(SDBackendModule::CLIP_VISION));
                }
            } else if (sd_version_is_qwen_image(version)) {
                bool enable_vision = false;
                if (!vae_decode_only) {
                    enable_vision = true;
                }
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                diffusion_model  = std::make_shared<Qwen::QwenImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                          params_backend_for(SDBackendModule::DIFFUSION),
                                                                          tensor_storage_map,
                                                                          "model.diffusion_model",
                                                                          version,
                                                                          sd_ctx_params->qwen_image_zero_cond_t);
            } else if (sd_version_is_longcat(version)) {
                bool enable_vision = false;
                if (!vae_decode_only) {
                    enable_vision = true;
                }
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            } else if (version == VERSION_HIDREAM_O1) {
                cond_stage_model = std::make_shared<HiDreamO1::HiDreamO1Conditioner>(backend_for(SDBackendModule::TE),
                                                                                     params_backend_for(SDBackendModule::TE),
                                                                                     tensor_storage_map);
                diffusion_model  = std::make_shared<HiDreamO1::HiDreamO1Runner>(backend_for(SDBackendModule::DIFFUSION),
                                                                               params_backend_for(SDBackendModule::DIFFUSION),
                                                                               tensor_storage_map,
                                                                               "model");
            } else if (sd_version_is_anima(version)) {
                cond_stage_model = std::make_shared<AnimaConditioner>(backend_for(SDBackendModule::TE),
                                                                      params_backend_for(SDBackendModule::TE),
                                                                      tensor_storage_map);
                diffusion_model  = std::make_shared<Anima::AnimaRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                       params_backend_for(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.diffusion_model");
            } else if (sd_version_is_z_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<ZImage::ZImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                         params_backend_for(SDBackendModule::DIFFUSION),
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         version);
            } else if (sd_version_is_ernie_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<ErnieImage::ErnieImageRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                                 params_backend_for(SDBackendModule::DIFFUSION),
                                                                                 tensor_storage_map,
                                                                                 "model.diffusion_model");
            } else if (sd_version_is_lens(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Lens::LensRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model");
            } else {  // SD1.x SD2.x SDXL
                std::map<std::string, std::string> embbeding_map;
                for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                    embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
                }
                if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(backend_for(SDBackendModule::TE),
                                                                                           params_backend_for(SDBackendModule::TE),
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version,
                                                                                           PM_VERSION_2);
                } else {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(backend_for(SDBackendModule::TE),
                                                                                           params_backend_for(SDBackendModule::TE),
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version);
                }
                diffusion_model = std::make_shared<UNetModelRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                    params_backend_for(SDBackendModule::DIFFUSION),
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model",
                                                                    version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the diffusion model");
                    diffusion_model->set_conv2d_direct_enabled(true);
                }
            }

            cond_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
            get_param_tensors(cond_stage_model, module_can_mmap(SDBackendModule::TE));

            diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
            get_param_tensors(diffusion_model, module_can_mmap(SDBackendModule::DIFFUSION));

            if (sd_version_is_unet_edit(version)) {
                vae_decode_only = false;
            }

            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
                get_param_tensors(high_noise_diffusion_model, module_can_mmap(SDBackendModule::DIFFUSION));
            }

            if (!ensure_backend_pair(SDBackendModule::VAE)) {
                return false;
            }

            auto create_tae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version) ||
                    sd_version_is_ltxav(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(backend_for(SDBackendModule::VAE),
                                                                  params_backend_for(SDBackendModule::VAE),
                                                                  tensor_storage_map,
                                                                  "decoder",
                                                                  vae_decode_only,
                                                                  version);

                } else {
                    auto model = std::make_shared<TinyImageAutoEncoder>(backend_for(SDBackendModule::VAE),
                                                                        params_backend_for(SDBackendModule::VAE),
                                                                        tensor_storage_map,
                                                                        "decoder.layers",
                                                                        vae_decode_only,
                                                                        version);
                    return model;
                }
            };

            auto create_vae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_ltxav(version)) {
                    return std::make_shared<LTXVideoVAE>(backend_for(SDBackendModule::VAE),
                                                         params_backend_for(SDBackendModule::VAE),
                                                         tensor_storage_map,
                                                         "first_stage_model",
                                                         vae_decode_only,
                                                         version);
                } else if (sd_version_is_wan(version) ||
                           sd_version_is_qwen_image(version) ||
                           sd_version_is_anima(version)) {
                    return std::make_shared<WAN::WanVAERunner>(backend_for(SDBackendModule::VAE),
                                                               params_backend_for(SDBackendModule::VAE),
                                                               tensor_storage_map,
                                                               "first_stage_model",
                                                               vae_decode_only,
                                                               version);
                } else {
                    auto model = std::make_shared<AutoEncoderKL>(backend_for(SDBackendModule::VAE),
                                                                 params_backend_for(SDBackendModule::VAE),
                                                                 tensor_storage_map,
                                                                 "first_stage_model",
                                                                 vae_decode_only,
                                                                 false,
                                                                 version);
                    if (sd_version_is_sdxl(version) &&
                        (strlen(SAFE_STR(sd_ctx_params->vae_path)) == 0 || sd_ctx_params->force_sdxl_vae_conv_scale || external_vae_is_invalid)) {
                        float vae_conv_2d_scale = 1.f / 32.f;
                        LOG_WARN(
                            "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, "
                            "using Conv2D scale %.3f",
                            vae_conv_2d_scale);
                        model->set_conv2d_scale(vae_conv_2d_scale);
                    }
                    return model;
                }
            };

            bool vae_mmap = module_can_mmap(SDBackendModule::VAE);

            if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1) {
                LOG_INFO("using FakeVAE");
                first_stage_model = std::make_shared<FakeVAE>(version,
                                                              backend_for(SDBackendModule::VAE),
                                                              params_backend_for(SDBackendModule::VAE));
            } else if (use_tae && !tae_preview_only) {
                LOG_INFO("using TAE for encoding / decoding");
                first_stage_model = create_tae();
                first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
                get_param_tensors_p(first_stage_model, vae_mmap, "tae");
            } else {
                LOG_INFO("using VAE for encoding / decoding");
                first_stage_model = create_vae();
                first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
                get_param_tensors_p(first_stage_model, vae_mmap, "first_stage_model");
                if (use_tae && tae_preview_only) {
                    LOG_INFO("using TAE for preview");
                    preview_vae = create_tae();
                    preview_vae->set_max_graph_vram_bytes(max_graph_vram_bytes);
                    get_param_tensors_p(preview_vae, vae_mmap, "tae");
                }
            }

            if (use_audio_vae) {
                audio_vae_model = std::make_shared<LTXV::LTXAudioVAERunner>(backend_for(SDBackendModule::VAE),
                                                                            params_backend_for(SDBackendModule::VAE),
                                                                            tensor_storage_map);
                get_param_tensors_p(audio_vae_model, vae_mmap, "");
            }

            if (sd_ctx_params->vae_conv_direct) {
                LOG_INFO("Using Conv2d direct in the vae model");
                first_stage_model->set_conv2d_direct_enabled(true);
                if (preview_vae) {
                    preview_vae->set_conv2d_direct_enabled(true);
                }
            }

            if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
                if (!ensure_backend_pair(SDBackendModule::CONTROL_NET)) {
                    return false;
                }
                control_net = std::make_shared<ControlNet>(backend_for(SDBackendModule::CONTROL_NET),
                                                           params_backend_for(SDBackendModule::CONTROL_NET),
                                                           tensor_storage_map,
                                                           version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the control net");
                    control_net->set_conv2d_direct_enabled(true);
                }
            }
            if (strlen(SAFE_STR(sd_ctx_params->photo_maker_path)) > 0) {
                if (!ensure_backend_pair(SDBackendModule::PHOTOMAKER)) {
                    return false;
                }
                if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                    pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend_for(SDBackendModule::PHOTOMAKER),
                                                                       params_backend_for(SDBackendModule::PHOTOMAKER),
                                                                       tensor_storage_map,
                                                                       "pmid",
                                                                       version,
                                                                       PM_VERSION_2);
                    LOG_INFO("using PhotoMaker Version 2");
                } else {
                    pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend_for(SDBackendModule::PHOTOMAKER),
                                                                       params_backend_for(SDBackendModule::PHOTOMAKER),
                                                                       tensor_storage_map,
                                                                       "pmid",
                                                                       version);
                }
                pmid_lora               = std::make_shared<LoraModel>("pmid",
                                                        backend_for(SDBackendModule::PHOTOMAKER),
                                                        params_backend_for(SDBackendModule::PHOTOMAKER),
                                                        sd_ctx_params->photo_maker_path,
                                                        "",
                                                        version);
                auto lora_tensor_filter = [&](const std::string& tensor_name) {
                    if (starts_with(tensor_name, "lora.model")) {
                        return true;
                    }
                    return false;
                };
                if (!pmid_lora->load_from_file(n_threads, lora_tensor_filter)) {
                    LOG_WARN("load photomaker lora tensors from %s failed", sd_ctx_params->photo_maker_path);
                    return false;
                }
                LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", sd_ctx_params->photo_maker_path);
                if (!model_loader.init_from_file_and_convert_name(sd_ctx_params->photo_maker_path, "pmid.")) {
                    LOG_WARN("loading stacked ID embedding from '%s' failed", sd_ctx_params->photo_maker_path);
                } else {
                    use_pmid = true;
                }
            }
            if (use_pmid) {
                get_param_tensors_p(pmid_model, module_can_mmap(SDBackendModule::PHOTOMAKER), "pmid");
            }

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

            diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            if (control_net) {
                control_net->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            circular_x = sd_ctx_params->circular_x;
            circular_y = sd_ctx_params->circular_y;
        }

        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tae && !tae_preview_only) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (use_pmid) {
            ignore_tensors.insert("pmid.unet.");
        }
        ignore_tensors.insert("model.diffusion_model.__x0__");
        ignore_tensors.insert("model.diffusion_model.__32x32__");
        ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.conv1");
            ignore_tensors.insert("first_stage_model.quant");
            ignore_tensors.insert("tae.encoder");
            ignore_tensors.insert("text_encoders.llm.visual.");
        }
        if (audio_vae_model) {
            ignore_tensors.insert("audio_vae.encoder");
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
        if (version == VERSION_HIDREAM_O1) {
            ignore_tensors.insert("lm_head.");
            ignore_tensors.insert("model.visual.deepstack_merger_list.");
        }

        if (enable_mmap_tensors) {
            if (mmap_able_tensors.empty()) {
                LOG_DEBUG("no tensors could be memory-mapped");
            } else {
                mmap_tensor_store = model_loader.mmap_tensors(mmap_able_tensors, ignore_tensors, needs_writable_mmap);
            }
        }

        if (clip_vision && !clip_vision->alloc_params_buffer()) {
            LOG_ERROR("CLIP vision params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (cond_stage_model) {
            cond_stage_model->alloc_params_buffer();
        }
        if (diffusion_model) {
            diffusion_model->alloc_params_buffer();
        }
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->alloc_params_buffer();
        }
        if (first_stage_model && !first_stage_model->alloc_params_buffer()) {
            LOG_ERROR("VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (preview_vae && !preview_vae->alloc_params_buffer()) {
            LOG_ERROR("preview VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (audio_vae_model && !audio_vae_model->alloc_params_buffer()) {
            LOG_ERROR("LTX audio VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (use_pmid && pmid_model && !pmid_model->alloc_params_buffer()) {
            LOG_ERROR("PhotoMaker params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }

        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads, sd_ctx_params->enable_mmap);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("finished loaded file");

        {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            if (high_noise_diffusion_model) {
                unet_params_mem_size += high_noise_diffusion_model->get_params_buffer_size();
            }
            size_t vae_params_mem_size = 0;
            vae_params_mem_size        = first_stage_model->get_params_buffer_size();
            if (preview_vae) {
                vae_params_mem_size += preview_vae->get_params_buffer_size();
            }
            if (audio_vae_model) {
                vae_params_mem_size += audio_vae_model->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(SAFE_STR(sd_ctx_params->control_net_path), n_threads)) {
                    ggml_free(ctx);
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (use_pmid) {
                pmid_params_mem_size = pmid_model->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            auto add_params_memory        = [&](size_t size, SDBackendModule module) {
                if (size == 0) {
                    return true;
                }
                ggml_backend_t module_backend = params_backend_for(module);
                if (module_backend == nullptr) {
                    return false;
                }
                if (ggml_backend_is_cpu(module_backend)) {
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
                return ggml_backend_is_cpu(module_backend) ? "RAM" : "VRAM";
            };

            if (!add_params_memory(clip_params_mem_size, SDBackendModule::TE) ||
                !add_params_memory(pmid_params_mem_size, SDBackendModule::PHOTOMAKER) ||
                !add_params_memory(unet_params_mem_size, SDBackendModule::DIFFUSION) ||
                !add_params_memory(vae_params_mem_size, SDBackendModule::VAE) ||
                !add_params_memory(control_net_params_mem_size, SDBackendModule::CONTROL_NET)) {
                ggml_free(ctx);
                return false;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(clip_params_mem_size, SDBackendModule::TE),
                unet_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(unet_params_mem_size, SDBackendModule::DIFFUSION),
                vae_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(vae_params_mem_size, SDBackendModule::VAE),
                control_net_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(control_net_params_mem_size, SDBackendModule::CONTROL_NET),
                pmid_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(pmid_params_mem_size, SDBackendModule::PHOTOMAKER));
        }

        // init denoiser
        {
            prediction_t pred_type = sd_ctx_params->prediction;

            if (pred_type == PREDICTION_COUNT) {
                if (sd_version_is_sd2(version)) {
                    // check is_using_v_parameterization_for_sd2
                    if (is_using_v_parameterization_for_sd2(sd_version_is_inpaint(version))) {
                        pred_type = V_PRED;
                    } else {
                        pred_type = EPS_PRED;
                    }
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
                           sd_version_is_qwen_image(version) ||
                           version == VERSION_HIDREAM_O1 ||
                           sd_version_is_anima(version) ||
                           sd_version_is_ernie_image(version) ||
                           sd_version_is_z_image(version)) {
                    pred_type = FLOW_PRED;
                    if (sd_version_is_wan(version)) {
                        default_flow_shift = 5.f;
                    } else if (sd_version_is_ernie_image(version)) {
                        default_flow_shift = 4.f;
                    } else {
                        default_flow_shift = 3.f;
                    }
                } else if (sd_version_is_flux(version) ||
                           sd_version_is_longcat(version) ||
                           sd_version_is_lens(version) ||
                           sd_version_is_ltxav(version)) {
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
                    }
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
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
                case FLUX2_FLOW_PRED: {
                    LOG_INFO("running in Flux2 FLOW mode");
                    denoiser = std::make_shared<Flux2FlowDenoiser>();
                    break;
                }
                default: {
                    LOG_ERROR("Unknown predition type %i", pred_type);
                    ggml_free(ctx);
                    return false;
                }
            }

            auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
            if (comp_vis_denoiser) {
                for (int i = 0; i < TIMESTEPS; i++) {
                    comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                    comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
                }
            }
        }

        ggml_free(ctx);
        return true;
    }

    bool is_using_v_parameterization_for_sd2(bool is_inpaint = false) {
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
        diffusion_model->free_compute_buffer();

        double result = static_cast<double>((out - x_t).mean());
        int64_t t1    = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    std::shared_ptr<LoraModel> load_lora_model_from_file(const std::string& lora_id,
                                                         float multiplier,
                                                         SDBackendModule module,
                                                         LoraModel::filter_t lora_tensor_filter = nullptr) {
        // kcpp
        // first check the cache
        std::string lora_key = "|" + std::to_string(static_cast<int>(module)) + "|" + lora_id;
        if (!apply_lora_immediately) {
            auto it = kcpp_lora_cache.find(lora_key);
            if (it != kcpp_lora_cache.end()) {
                if (it->second) {
                    it->second->multiplier = multiplier;
                }
                return it->second;
            }
        }

        std::string lora_path             = lora_id;
        static std::string high_noise_tag = "|high_noise|";
        bool is_high_noise                = false;
        if (starts_with(lora_path, high_noise_tag)) {
            lora_path     = lora_path.substr(high_noise_tag.size());
            is_high_noise = true;
            LOG_DEBUG("high noise lora: %s", lora_path.c_str());
        }
        if (!ensure_backend_pair(module)) {
            return nullptr;
        }
        auto lora = std::make_shared<LoraModel>(lora_id,
                                                backend_for(module),
                                                backend_for(module),
                                                lora_path,
                                                is_high_noise ? "model.high_noise_" : "",
                                                version);
        if (!lora->load_from_file(n_threads, lora_tensor_filter)) {
            LOG_WARN("load lora tensors from %s failed", lora_path.c_str());
            // also cache negatives to avoid I/O at runtime
            lora = nullptr;
            if (!apply_lora_immediately && kcpp_lora_cache_populate)
                kcpp_lora_cache[lora_key] = lora;
            return lora;
        }

        lora->multiplier = multiplier;
        if (!apply_lora_immediately && kcpp_lora_cache_populate)
            kcpp_lora_cache[lora_key] = lora;
        return lora;
    }

    void apply_loras_immediately(const std::unordered_map<std::string, float>& lora_state) {
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;
            lora_state_diff[lora_name] += multiplier;
        }
        for (auto& kv : curr_lora_state) {
            const std::string& lora_name = kv.first;
            float curr_multiplier        = kv.second;
            lora_state_diff[lora_name] -= curr_multiplier;
        }

        if (lora_state_diff.empty()) {
            return;
        }

        LOG_INFO("apply lora immediately");

        size_t rm = lora_state_diff.size() - lora_state.size();
        if (rm != 0) {
            LOG_INFO("attempting to apply %lu LoRAs (removing %lu applied LoRAs)", lora_state.size(), rm);
        } else {
            LOG_INFO("attempting to apply %lu LoRAs", lora_state.size());
        }

        for (auto& kv : lora_state_diff) {
            int64_t t0 = ggml_time_ms();

            auto lora = load_lora_model_from_file(kv.first, kv.second, SDBackendModule::DIFFUSION);
            if (!lora || lora->lora_tensors.empty()) {
                continue;
            }
            lora->apply(tensors, version, n_threads);
            lora->free_params_buffer();

            int64_t t1 = ggml_time_ms();

            LOG_INFO("lora '%s' applied, taking %.2fs", kv.first.c_str(), (t1 - t0) * 1.0f / 1000);
        }

        curr_lora_state = lora_state;
    }

    void apply_loras_at_runtime(const std::unordered_map<std::string, float>& lora_state) {
        cond_stage_lora_models.clear();
        diffusion_lora_models.clear();
        first_stage_lora_models.clear();
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
        if (lora_state.empty()) {
            return;
        }
        LOG_INFO("apply lora at runtime");
        if (cond_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : cond_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            cond_stage_lora_models  = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_cond_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_id = kv.first;
                float multiplier           = kv.second;

                auto lora = load_lora_model_from_file(lora_id, multiplier, SDBackendModule::TE, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    cond_stage_lora_models.push_back(lora);
                }
            }
            // Only attach the adapter when there are LoRAs targeting the cond_stage model.
            // An empty MultiLoraAdapter still routes every linear/conv through
            // forward_with_lora() instead of the direct kernel path — slower for no benefit.
            if (!cond_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(cond_stage_lora_models);
                cond_stage_model->set_weight_adapter(multi_lora_adapter);
            }
        }
        if (diffusion_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : diffusion_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            diffusion_lora_models   = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_diffusion_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, SDBackendModule::DIFFUSION, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    diffusion_lora_models.push_back(lora);
                }
            }
            if (!diffusion_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(diffusion_lora_models);
                diffusion_model->set_weight_adapter(multi_lora_adapter);
                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->set_weight_adapter(multi_lora_adapter);
                }
            }
        }

        if (first_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : first_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            first_stage_lora_models = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_first_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, SDBackendModule::VAE, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    first_stage_lora_models.push_back(lora);
                }
            }
            if (!first_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(first_stage_lora_models);
                first_stage_model->set_weight_adapter(multi_lora_adapter);
            }
        }
    }

    void lora_stat() {
        if (!cond_stage_lora_models.empty()) {
            LOG_INFO("cond_stage_lora_models:");
            for (auto& lora_model : cond_stage_lora_models) {
                lora_model->stat();
            }
        }

        if (!diffusion_lora_models.empty()) {
            LOG_INFO("diffusion_lora_models:");
            for (auto& lora_model : diffusion_lora_models) {
                lora_model->stat();
            }
        }

        if (!first_stage_lora_models.empty()) {
            LOG_INFO("first_stage_lora_models:");
            for (auto& lora_model : first_stage_lora_models) {
                lora_model->stat();
            }
        }
    }

    void apply_loras(const sd_lora_t* loras, uint32_t lora_count) {
        std::unordered_map<std::string, float> lora_f2m;
        for (uint32_t i = 0; i < lora_count; i++) {
            std::string lora_id = SAFE_STR(loras[i].path);
            if (loras[i].is_high_noise) {
                lora_id = "|high_noise|" + lora_id;
            }
            lora_f2m[lora_id] = loras[i].multiplier;
            LOG_DEBUG("lora %s:%.2f", lora_id.c_str(), loras[i].multiplier);
        }
        int64_t t0 = ggml_time_ms();
        if (apply_lora_immediately) {
            apply_loras_immediately(lora_f2m);
        } else {
            apply_loras_at_runtime(lora_f2m);
        }
        int64_t t1 = ggml_time_ms();
        if (!lora_f2m.empty()) {
            LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        }
    }

    SDCondition get_pmid_conditon(sd_pm_params_t pm_params,
                                  ConditionerParams& condition_params) {
        SDCondition id_cond;
        if (use_pmid) {
            if (!pmid_lora->applied) {
                int64_t t0 = ggml_time_ms();
                pmid_lora->apply(tensors, version, n_threads);
                int64_t t1         = ggml_time_ms();
                pmid_lora->applied = true;
                LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
                if (free_params_immediately) {
                    pmid_lora->free_params_buffer();
                }
            }
            // preprocess input id images
            bool pmv2 = pmid_model->get_version() == PM_VERSION_2;
            if (pm_params.id_images_count > 0) {
                int clip_image_size        = 224;
                pmid_model->style_strength = pm_params.style_strength;
                sd::Tensor<float> id_image_tensor;
                for (int i = 0; i < pm_params.id_images_count; i++) {
                    auto id_image           = sd_image_to_tensor(pm_params.id_images[i]);
                    auto processed_id_image = clip_preprocess(id_image, clip_image_size, clip_image_size);
                    if (id_image_tensor.empty()) {
                        id_image_tensor = processed_id_image;
                    } else {
                        id_image_tensor = sd::ops::concat(id_image_tensor, processed_id_image, 3);
                    }
                }

                int64_t t0                      = ggml_time_ms();
                condition_params.num_input_imgs = pm_params.id_images_count;
                auto cond_tup                   = cond_stage_model->get_learned_condition_with_trigger(n_threads,
                                                                                                       condition_params);
                id_cond                         = std::get<0>(cond_tup);
                auto class_tokens_mask          = std::get<1>(cond_tup);
                sd::Tensor<float> id_embeds;
                if (pmv2 && pm_params.id_embed_path != nullptr) {
                    try {
                        id_embeds = sd::load_tensor_from_file_as_tensor<float>(pm_params.id_embed_path);
                    } catch (const std::exception&) {
                        id_embeds = {};
                    }
                }
                if (pmv2 && id_embeds.empty()) {
                    LOG_WARN("Provided PhotoMaker images, but NO valid ID embeds file for PM v2");
                    LOG_WARN("Turn off PhotoMaker");
                    use_pmid = false;
                } else {
                    if (pmv2 && pm_params.id_images_count != id_embeds.shape()[1]) {
                        LOG_WARN("PhotoMaker image count (%d) does NOT match ID embeds (%d). You should run face_detect.py again.", pm_params.id_images_count, static_cast<int>(id_embeds.shape()[1]));
                        LOG_WARN("Turn off PhotoMaker");
                        use_pmid = false;
                    } else {
                        auto res = pmid_model->compute(n_threads,
                                                       id_image_tensor,
                                                       id_cond.c_crossattn,
                                                       id_embeds,
                                                       class_tokens_mask);
                        if (res.empty()) {
                            LOG_ERROR("Photomaker ID Stacking failed");
                            LOG_WARN("Turn off PhotoMaker");
                            use_pmid = false;
                        } else {
                            id_cond.c_crossattn = std::move(res);
                            int64_t t1          = ggml_time_ms();
                            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
                            // Encode input prompt without the trigger word for delayed conditioning
                            condition_params.text = cond_stage_model->remove_trigger_from_prompt(condition_params.text);
                        }
                        if (free_params_immediately) {
                            pmid_model->free_params_buffer();
                        }
                    }
                }
            } else {
                LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
                LOG_WARN("Turn off PhotoMaker");
                use_pmid = false;
            }
        }
        return id_cond;
    }

    sd::Tensor<float> get_clip_vision_output(const sd::Tensor<float>& image,
                                             bool return_pooled   = true,
                                             int clip_skip        = -1,
                                             bool zero_out_masked = false) {
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

    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask) {
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B") {
            auto new_timesteps = std::vector<float>(static_cast<size_t>(init_latent.shape()[2]), timesteps[0]);

            if (!denoise_mask.empty()) {
                float value = denoise_mask.dim() == 5 ? denoise_mask.index(0, 0, 0, 0, 0) : denoise_mask.index(0, 0, 0, 0);
                if (value == 0.f) {
                    new_timesteps[0] = 0.f;
                }
            }
            return new_timesteps;
        } else {
            return timesteps;
        }
    }

    std::vector<float> process_ltxav_video_timesteps(const std::vector<float>& timesteps,
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

    void preview_image(int step,
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
            } else if (channels == 16) {
                if (sd_version_is_sd3(version)) {
                    latent_rgb_proj = sd3_latent_rgb_proj;
                    latent_rgb_bias = sd3_latent_rgb_bias;
                } else if (sd_version_is_flux(version) || sd_version_is_z_image(version) || sd_version_is_longcat(version)) {
                    latent_rgb_proj = flux_latent_rgb_proj;
                    latent_rgb_bias = flux_latent_rgb_bias;
                } else if (sd_version_is_wan(version) || sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
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
                preview_vae->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
                vae_latents = preview_vae->diffusion_to_vae_latents(_latents);
                decoded     = preview_vae->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
            } else {
                first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
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

    std::vector<float> prepare_sample_timesteps(float sigma,
                                                int shifted_timestep) {
        float t = denoiser->sigma_to_t(sigma);
        if (shifted_timestep > 0) {
            float shifted_t_float = t * (float(shifted_timestep) / float(TIMESTEPS));
            int64_t shifted_t     = static_cast<int64_t>(roundf(shifted_t_float));
            shifted_t             = std::max((int64_t)0, std::min((int64_t)(TIMESTEPS - 1), shifted_t));
            LOG_DEBUG("shifting timestep from %.2f to %" PRId64 " (sigma: %.4f)", t, shifted_t, sigma);
            return std::vector<float>{(float)shifted_t};
        }
        if (sd_version_is_anima(version)) {
            return std::vector<float>{t / static_cast<float>(TIMESTEPS)};
        }
        if (version == VERSION_HIDREAM_O1) {
            return std::vector<float>{1.0f - (t / static_cast<float>(TIMESTEPS))};
        }
        if (sd_version_is_z_image(version)) {
            return std::vector<float>{1000.f - t};
        }
        return std::vector<float>{t};
    }

    void adjust_sample_step_scalings(int shifted_timestep,
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

    struct SamplePreviewContext {
        sd_preview_cb_t callback = nullptr;
        void* data               = nullptr;
        preview_t mode           = PREVIEW_NONE;
    };

    SamplePreviewContext prepare_sample_preview_context() {
        return SamplePreviewContext{sd_get_preview_callback(),
                                    sd_get_preview_callback_data(),
                                    sd_get_preview_mode()};
    }

    void report_sample_progress(int step, size_t total_steps, int64_t t0) {
        int64_t t1 = ggml_time_us();
        if (step > 0 || step == -(int)total_steps) {
            int showstep = std::abs(step);
            pretty_progress(showstep, (int)total_steps, (t1 - t0) / 1000000.f / showstep);
        }
    }

    void compute_sample_controls(const sd::Tensor<float>& control_image,
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

    sd::Tensor<float> sample(const std::shared_ptr<DiffusionModelRunner>& work_diffusion_model,
                             bool inverse_noise_scaling,
                             const sd::Tensor<float>& init_latent,
                             sd::Tensor<float> noise,
                             const SDCondition& cond,
                             const SDCondition& uncond,
                             const SDCondition& img_cond,
                             const SDCondition& id_cond,
                             const sd::Tensor<float>& control_image,
                             float control_strength,
                             const sd_guidance_params_t& guidance,
                             float eta,
                             int shifted_timestep,
                             sample_method_t method,
                             bool is_flow_denoiser,
                             const char* extra_sample_args,
                             const std::vector<float>& sigmas,
                             int start_merge_step,
                             const std::vector<sd::Tensor<float>>& ref_latents,
                             bool increase_ref_index,
                             const sd::Tensor<float>& denoise_mask,
                             const sd::Tensor<float>& vace_context,
                             float vace_strength,
                             int audio_length,
                             float frame_rate,
                             const sd_cache_params_t* cache_params,
                             const sd::Tensor<float>& video_positions = {}) {
        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);
        float cfg_scale     = guidance.txt_cfg;
        float img_cfg_scale = guidance.img_cfg;
        float slg_scale     = guidance.slg.scale;

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

        size_t steps       = sigmas.size() - 1;
        bool has_skiplayer = slg_scale != 0.0f && !skip_layers.empty();
        if (has_skiplayer && !sd_version_is_dit(version)) {
            has_skiplayer = false;
            LOG_WARN("SLG is incompatible with this model type");
        }
        sd::guidance::ClassifierFreeGuidance classifier_free_guidance(cfg_scale, img_cfg_scale);
        sd::guidance::SkipLayerGuidance skip_layer_guidance(has_skiplayer ? skip_layers : std::vector<int>(),
                                                            has_skiplayer ? slg_scale : 0.0f,
                                                            guidance.slg.layer_start,
                                                            guidance.slg.layer_end);

        if (version == VERSION_HIDREAM_O1 && !noise.empty()) {
            noise *= eta;
        }

        int64_t t0                   = ggml_time_us();
        sd::Tensor<float> x_t        = !noise.empty()
                                           ? denoiser->noise_scaling(sigmas[0], noise, init_latent)
                                           : init_latent;
        sd::Tensor<float> denoised   = x_t;
        SamplePreviewContext preview = prepare_sample_preview_context();

        auto denoise = [&](const sd::Tensor<float>& x, float sigma, int step) -> sd::guidance::GuiderOutput {
            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
            }

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            std::vector<float> base_timesteps_vec = prepare_sample_timesteps(sigma, shifted_timestep);
            std::vector<float> timesteps_vec      = base_timesteps_vec;
            sd::Tensor<float> audio_timesteps_tensor;
            if (sd_version_is_ltxav(version) && !denoise_mask.empty()) {
                timesteps_vec          = process_ltxav_video_timesteps(base_timesteps_vec, init_latent, denoise_mask);
                audio_timesteps_tensor = sd::Tensor<float>({static_cast<int64_t>(base_timesteps_vec.size())}, base_timesteps_vec);
            } else {
                timesteps_vec = process_timesteps(timesteps_vec, init_latent, denoise_mask);
            }
            const std::vector<float>& scaling_timesteps_vec = (sd_version_is_ltxav(version) && !denoise_mask.empty())
                                                                  ? base_timesteps_vec
                                                                  : timesteps_vec;
            adjust_sample_step_scalings(shifted_timestep, scaling_timesteps_vec, c_in, &c_skip, &c_out);

            sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
            sd::Tensor<float> guidance_tensor({1}, std::vector<float>{guidance.distilled_guidance});
            sd::Tensor<float> noised_input = x * c_in;
            if (!denoise_mask.empty() && (version == VERSION_WAN2_2_TI2V || sd_version_is_ltxav(version))) {
                noised_input = noised_input * denoise_mask + init_latent * (1.0f - denoise_mask);
            }

            if (cache_runtime.spectrum_enabled && cache_runtime.spectrum.should_predict()) {
                cache_runtime.spectrum.predict(&denoised);
                if (!denoise_mask.empty()) {
                    denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
                }
                if (sd_should_preview_denoised() && preview.callback != nullptr) {
                    preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
                }
                report_sample_progress(step, steps, t0);
                sd::guidance::GuiderOutput output;
                output.pred = denoised;
                return output;
            }

            if (sd_should_preview_noisy() && preview.callback != nullptr) {
                preview_image(step, noised_input, version, preview.mode, preview.callback, preview.data, true);
            }

            sd::Tensor<float> cond_out;
            sd::Tensor<float> uncond_out;
            sd::Tensor<float> img_cond_out;
            sd_sample::SampleStepCacheDispatcher step_cache(cache_runtime, step, sigma);
            std::vector<sd::Tensor<float>> controls;
            DiffusionParams diffusion_params;
            diffusion_params.x                  = &noised_input;
            diffusion_params.timesteps          = &timesteps_tensor;
            diffusion_params.increase_ref_index = increase_ref_index;

            compute_sample_controls(control_image,
                                    noised_input,
                                    timesteps_tensor,
                                    cond,
                                    &controls);

            auto run_condition = [&](const SDCondition& condition,
                                     const sd::Tensor<float>* c_concat_override = nullptr,
                                     const std::vector<int>* local_skip_layers  = nullptr) -> sd::Tensor<float> {
                diffusion_params.context     = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
                diffusion_params.c_concat    = c_concat_override != nullptr ? c_concat_override : (condition.c_concat.empty() ? nullptr : &condition.c_concat);
                diffusion_params.y           = condition.c_vector.empty() ? nullptr : &condition.c_vector;
                diffusion_params.ref_latents = condition.c_ref_images.empty() ? &ref_latents : &condition.c_ref_images;

                if (sd_version_is_unet(version)) {
                    diffusion_params.extra = UNetDiffusionExtra{-1, &controls, control_strength};
                } else if (sd_version_is_sd3(version)) {
                    diffusion_params.extra = SkipLayerDiffusionExtra{local_skip_layers};
                } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) || sd_version_is_longcat(version)) {
                    diffusion_params.extra = FluxDiffusionExtra{&guidance_tensor,
                                                                local_skip_layers};
                } else if (sd_version_is_anima(version)) {
                    diffusion_params.extra = AnimaDiffusionExtra{condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids,
                                                                 condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights};
                } else if (sd_version_is_wan(version)) {
                    diffusion_params.extra = WanDiffusionExtra{vace_context.empty() ? nullptr : &vace_context,
                                                               vace_strength};
                } else if (version == VERSION_HIDREAM_O1) {
                    diffusion_params.extra = HiDreamO1DiffusionExtra{
                        condition.c_input_ids.empty() ? nullptr : &condition.c_input_ids,
                        condition.c_position_ids.empty() ? nullptr : &condition.c_position_ids,
                        condition.c_token_types.empty() ? nullptr : &condition.c_token_types,
                        condition.c_vinput_mask.empty() ? nullptr : &condition.c_vinput_mask,
                        condition.c_image_embeds.empty() ? nullptr : &condition.c_image_embeds};
                } else if (sd_version_is_ltxav(version)) {
                    diffusion_params.extra = LTXAVDiffusionExtra{
                        nullptr,
                        audio_timesteps_tensor.empty() ? nullptr : &audio_timesteps_tensor,
                        audio_length,
                        frame_rate,
                        video_positions.empty() ? nullptr : &video_positions};
                } else {
                    diffusion_params.extra = std::monostate{};
                }

                sd::Tensor<float> cached_output;
                if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                    return std::move(cached_output);
                }

                auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
                if (output_opt.empty()) {
                    LOG_ERROR("diffusion model compute failed");
                    return sd::Tensor<float>();
                }

                step_cache.after_condition(&condition, noised_input, output_opt);
                return output_opt;
            };

            if (start_merge_step == -1 || step <= start_merge_step) {
                cond_out = run_condition(cond);
                if (cond_out.empty()) {
                    return {};
                }
            } else {
                GGML_ASSERT(!id_cond.empty());
                cond_out = run_condition(id_cond,
                                         cond.c_concat.empty() ? nullptr : &cond.c_concat);
                if (cond_out.empty()) {
                    return {};
                }
            }

            if (!uncond.empty()) {
                if (!step_cache.is_step_skipped()) {
                    compute_sample_controls(control_image,
                                            noised_input,
                                            timesteps_tensor,
                                            uncond,
                                            &controls);
                }
                uncond_out = run_condition(uncond);
                if (uncond_out.empty()) {
                    return {};
                }
            }
            if (!img_cond.empty()) {
                img_cond_out = run_condition(img_cond,
                                             cond.c_concat.empty() ? nullptr : &cond.c_concat);
                if (img_cond_out.empty()) {
                    return {};
                }
            }
            sd::guidance::GuidanceInput guidance_input;
            guidance_input.step          = step;
            guidance_input.schedule_size = sigmas.size();
            guidance_input.pred_cond     = &cond_out;
            guidance_input.pred_uncond   = uncond_out.empty() ? nullptr : &uncond_out;
            guidance_input.pred_img_cond = img_cond_out.empty() ? nullptr : &img_cond_out;

            sd::guidance::GuiderOutput guided = classifier_free_guidance.forward(guidance_input, {});
            if (guided.pred.empty()) {
                return {};
            }

            if (skip_layer_guidance.is_enabled_for_step(guidance_input)) {
                LOG_DEBUG("Skipping layers at step %d\n", step);
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
                const sd::Tensor<float>& base_uncond = !uncond_out.empty() ? uncond_out : cond_out;
                output.pred_uncond                   = base_uncond * c_out + x * c_skip;
            }
            if (cache_runtime.spectrum_enabled) {
                cache_runtime.spectrum.update(denoised);
            }
            if (!denoise_mask.empty()) {
                denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
            }
            if (sd_should_preview_denoised() && preview.callback != nullptr) {
                preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
            }
            report_sample_progress(step, steps, t0);
            output.pred = denoised;
            return output;
        };

        auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser, extra_sample_args);
        if (x0_opt.empty()) {
            LOG_ERROR("Diffusion model sampling failed");
            if (control_net) {
                control_net->free_control_ctx();
                control_net->free_compute_buffer();
            }
            if (work_diffusion_model) {
                work_diffusion_model->free_compute_buffer();
            }
            return {};
        }

        auto x0 = std::move(x0_opt);
        sd_sample::log_sample_cache_summary(cache_runtime, steps);
        if (inverse_noise_scaling) {
            x0 = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x0);
        }

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        if (work_diffusion_model) {
            work_diffusion_model->free_compute_buffer();
        }
        return x0;
    }

    int get_vae_scale_factor() {
        return first_stage_model->get_scale_factor();
    }

    int get_diffusion_model_down_factor() {
        int down_factor = 8;  // unet
        if (sd_version_is_dit(version)) {
            if (sd_version_is_wan(version)) {
                down_factor = 2;
            } else {
                down_factor = 1;
            }
        }
        return down_factor;
    }

    int get_latent_channel() {
        int latent_channel = 4;
        if (sd_version_is_dit(version)) {
            if (sd_version_is_ltxav(version)) {
                latent_channel = 128;
            } else if (version == VERSION_WAN2_2_TI2V) {
                latent_channel = 48;
            } else if (version == VERSION_HIDREAM_O1) {
                latent_channel = 3;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                latent_channel = 3;
            } else if (sd_version_uses_flux2_vae(version)) {
                latent_channel = 128;
            } else {
                latent_channel = 16;
            }
        }
        return latent_channel;
    }

    int get_image_seq_len(int h, int w) {
        int vae_scale_factor = get_vae_scale_factor();
        return (h / vae_scale_factor) * (w / vae_scale_factor);
    }

    sd::Tensor<float> generate_init_latent(int width,
                                           int height,
                                           int frames = 1,
                                           bool video = false) {
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

    int video_frames_to_latent_frames(int frames) {
        int latent_frames = frames;
        if (sd_version_is_ltxav(version)) {
            latent_frames = ((frames - 1) / 8) + 1;
        } else if (sd_version_is_wan(version)) {
            latent_frames = ((frames - 1) / 4) + 1;
        }
        return latent_frames;
    }

    int latent_frames_to_video_frames(int latent_frames) {
        if (latent_frames <= 0) {
            return latent_frames;
        }
        if (sd_version_is_ltxav(version)) {
            return (latent_frames - 1) * 8 + 1;
        }
        if (sd_version_is_wan(version)) {
            return (latent_frames - 1) * 4 + 1;
        }
        return latent_frames;
    }

    int align_video_frames(int frames) {
        return latent_frames_to_video_frames(video_frames_to_latent_frames(frames));
    }

    sd::Tensor<float> encode_to_vae_latents(const sd::Tensor<float>& x) {
        auto latents = first_stage_model->encode(n_threads, x, vae_tiling_params, circular_x, circular_y);
        if (latents.empty()) {
            return {};
        }
        latents = first_stage_model->vae_output_to_latents(latents, rng);
        return latents;
    }

    sd::Tensor<float> encode_first_stage(const sd::Tensor<float>& x) {
        auto latents = encode_to_vae_latents(x);
        if (latents.empty()) {
            return {};
        }
        if (version != VERSION_SD1_PIX2PIX) {
            latents = first_stage_model->vae_to_diffusion_latents(latents);
        }
        return latents;
    }

    sd::Tensor<float> decode_first_stage(const sd::Tensor<float>& x, bool decode_video = false) {
        auto latents = first_stage_model->diffusion_to_vae_latents(x);
        first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
        return first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
    }

    sd::Tensor<float> normalize_ltx_video_latents(const sd::Tensor<float>& x) {
        auto ltx_vae = std::dynamic_pointer_cast<LTXVideoVAE>(first_stage_model);
        if (!ltx_vae) {
            LOG_ERROR("LTX latent normalization requires LTX video VAE");
            return {};
        }
        return ltx_vae->normalize_latents(n_threads, x);
    }

    sd::Tensor<float> un_normalize_ltx_video_latents(const sd::Tensor<float>& x) {
        auto ltx_vae = std::dynamic_pointer_cast<LTXVideoVAE>(first_stage_model);
        if (!ltx_vae) {
            LOG_ERROR("LTX latent un-normalization requires LTX video VAE");
            return {};
        }
        return ltx_vae->un_normalize_latents(n_threads, x);
    }

    sd::Tensor<float> decode_ltx_audio_latent(const sd::Tensor<float>& audio_latent) {
        if (audio_vae_model == nullptr || audio_latent.empty()) {
            return {};
        }
        auto waveform = audio_vae_model->decode(n_threads, audio_latent);
        if (free_params_immediately) {
            audio_vae_model->free_params_buffer();
        }
        return waveform;
    }

    void set_flow_shift(float flow_shift = INFINITY) {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        if (flow_denoiser) {
            if (flow_shift == INFINITY) {
                flow_shift = default_flow_shift;
            }
            flow_denoiser->set_shift(flow_shift);
        }
    }

    bool is_flow_denoiser() {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        return !!flow_denoiser;
    }

    //added for kcpp
    void SetCircularAxesAll(bool circular_x, bool circular_y)
    {
        diffusion_model->set_circular_axes(circular_x, circular_y);
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->set_circular_axes(circular_x, circular_y);
        }
        if (control_net) {
            control_net->set_circular_axes(circular_x, circular_y);
        }
        if (first_stage_model) {
            first_stage_model->set_circular_axes(circular_x, circular_y);
        }
    }
    //end added for kcpp

};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

const char* sd_type_name(enum sd_type_t type) {
    if ((int)type < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)) {
        return ggml_type_name((ggml_type)type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "ltx2",
};

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "flux2_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return NONE_STR;
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

const char* hires_upscaler_to_str[] = {
    "None",
    "Latent",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Lanczos",
    "Nearest",
    "Model",
};

const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler) {
    if (upscaler >= SD_HIRES_UPSCALER_NONE && upscaler < SD_HIRES_UPSCALER_COUNT) {
        return hires_upscaler_to_str[upscaler];
    }
    return NONE_STR;
}

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str) {
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; i++) {
        if (!strcmp(str, hires_upscaler_to_str[i])) {
            return (enum sd_hires_upscaler_t)i;
        }
    }
    return SD_HIRES_UPSCALER_COUNT;
}

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = INFINITY;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
    cache_params->spectrum_w                  = 0.40f;
    cache_params->spectrum_m                  = 3;
    cache_params->spectrum_lam                = 1.0f;
    cache_params->spectrum_window_size        = 2;
    cache_params->spectrum_flex_window        = 0.50f;
    cache_params->spectrum_warmup_steps       = 4;
    cache_params->spectrum_stop_percent       = 0.9f;
}

void sd_hires_params_init(sd_hires_params_t* hires_params) {
    *hires_params                     = {};
    hires_params->enabled             = false;
    hires_params->upscaler            = SD_HIRES_UPSCALER_LATENT;
    hires_params->model_path          = nullptr;
    hires_params->scale               = 2.0f;
    hires_params->target_width        = 0;
    hires_params->target_height       = 0;
    hires_params->steps               = 0;
    hires_params->denoising_strength  = 0.7f;
    hires_params->upscale_tile_size   = 128;
    hires_params->custom_sigmas       = nullptr;
    hires_params->custom_sigmas_count = 0;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->sampler_rng_type        = RNG_TYPE_COUNT;
    sd_ctx_params->prediction              = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode         = LORA_APPLY_AUTO;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->max_vram                = 0.f;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
    sd_ctx_params->backend                 = nullptr;
    sd_ctx_params->params_backend          = nullptr;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(8192);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 8192 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "clip_vision_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "diffusion_model_path: %s\n"
             "high_noise_diffusion_model_path: %s\n"
             "embeddings_connectors_path: %s\n"
             "vae_path: %s\n"
             "audio_vae_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "photo_maker_path: %s\n"
             "tensor_type_rules: %s\n"
             "vae_decode_only: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "offload_params_to_cpu: %s\n"
             "max_vram: %.3f\n"
             "backend: %s\n"
             "params_backend: %s\n"
             "keep_clip_on_cpu: %s\n"
             "keep_control_net_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "chroma_use_dit_mask: %s\n"
             "chroma_use_t5_mask: %s\n"
             "chroma_t5_mask_pad: %d\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->clip_vision_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path),
             SAFE_STR(sd_ctx_params->embeddings_connectors_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->audio_vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->photo_maker_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             BOOL_STR(sd_ctx_params->offload_params_to_cpu),
             sd_ctx_params->max_vram,
             SAFE_STR(sd_ctx_params->backend),
             SAFE_STR(sd_ctx_params->params_backend),
             BOOL_STR(sd_ctx_params->keep_clip_on_cpu),
             BOOL_STR(sd_ctx_params->keep_control_net_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->circular_x),
             BOOL_STR(sd_ctx_params->circular_y),
             BOOL_STR(sd_ctx_params->chroma_use_dit_mask),
             BOOL_STR(sd_ctx_params->chroma_use_t5_mask),
             sd_ctx_params->chroma_t5_mask_pad);

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
    sample_params->extra_sample_args           = nullptr;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "shifted_timestep: %d, "
             "flow_shift: %.2f, "
             "extra_sample_args: %s)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->shifted_timestep,
             sample_params->flow_shift,
             SAFE_STR(sample_params->extra_sample_args));

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip         = -1;
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->vae_tiling_params = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_cache_params_init(&sd_img_gen_params->cache);
    sd_hires_params_init(&sd_img_gen_params->hires);
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "auto_resize_ref_image: %s\n"
             "increase_ref_index: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s (temporal=%s, extra_tiling_args=%s)\n"
             "hires: {enabled=%s, upscaler=%s, model_path=%s, scale=%.2f, target=%dx%d, steps=%d, denoising_strength=%.2f}\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             BOOL_STR(sd_img_gen_params->auto_resize_ref_image),
             BOOL_STR(sd_img_gen_params->increase_ref_index),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.temporal_tiling),
             SAFE_STR(sd_img_gen_params->vae_tiling_params.extra_tiling_args),
             BOOL_STR(sd_img_gen_params->hires.enabled),
             sd_hires_upscaler_name(sd_img_gen_params->hires.upscaler),
             SAFE_STR(sd_img_gen_params->hires.model_path),
             sd_img_gen_params->hires.scale,
             sd_img_gen_params->hires.target_width,
             sd_img_gen_params->hires.target_height,
             sd_img_gen_params->hires.steps,
             sd_img_gen_params->hires.denoising_strength);
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    } else if (sd_img_gen_params->cache.mode == SD_CACHE_UCACHE) {
        cache_mode_str = "ucache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             get_cache_reuse_threshold(sd_img_gen_params->cache),
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}

void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    *sd_vid_gen_params = {};
    sd_sample_params_init(&sd_vid_gen_params->sample_params);
    sd_sample_params_init(&sd_vid_gen_params->high_noise_sample_params);
    sd_vid_gen_params->high_noise_sample_params.sample_steps = -1;
    sd_vid_gen_params->width                                 = 512;
    sd_vid_gen_params->height                                = 512;
    sd_vid_gen_params->strength                              = 0.75f;
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->fps                                   = 16;
    sd_vid_gen_params->moe_boundary                          = 0.875f;
    sd_vid_gen_params->vace_strength                         = 1.f;
    sd_vid_gen_params->vae_tiling_params                     = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_vid_gen_params->hires.enabled                         = false;
    sd_vid_gen_params->hires.upscaler                        = SD_HIRES_UPSCALER_LATENT;
    sd_vid_gen_params->hires.scale                           = 2.f;
    sd_vid_gen_params->hires.target_width                    = 0;
    sd_vid_gen_params->hires.target_height                   = 0;
    sd_vid_gen_params->hires.steps                           = 0;
    sd_vid_gen_params->hires.denoising_strength              = 0.7f;
    sd_vid_gen_params->hires.upscale_tile_size               = 128;
    sd_vid_gen_params->hires.custom_sigmas                   = nullptr;
    sd_vid_gen_params->hires.custom_sigmas_count             = 0;
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

static bool sd_version_supports_video_generation(SDVersion version) {
    return version == VERSION_SVD || sd_version_is_wan(version) || sd_version_is_ltxav(version);
}

static bool sd_version_supports_image_generation(SDVersion version) {
    return !sd_version_supports_video_generation(version);
}

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

static sd_audio_t* waveform_to_sd_audio(const StableDiffusionGGML* sd,
                                        const sd::Tensor<float>& waveform) {
    if (sd == nullptr || waveform.empty()) {
        return nullptr;
    }

    int64_t sample_count = waveform.shape()[0];
    int64_t channels     = waveform.shape().size() > 1 ? waveform.shape()[1] : 1;
    if (sample_count <= 0 || channels <= 0) {
        return nullptr;
    }

    sd_audio_t* audio = (sd_audio_t*)malloc(sizeof(sd_audio_t));
    if (audio == nullptr) {
        return nullptr;
    }

    audio->sample_rate  = static_cast<uint32_t>(sd->audio_vae_model != nullptr ? sd->audio_vae_model->config.output_sample_rate() : 0);
    audio->channels     = static_cast<uint32_t>(channels);
    audio->sample_count = static_cast<uint64_t>(sample_count);
    size_t sample_bytes = waveform.numel() * sizeof(float);
    audio->data         = (float*)malloc(sample_bytes);
    if (audio->data == nullptr) {
        free(audio);
        return nullptr;
    }

    auto wavaform_t = waveform.permute({1, 0, 2, 3});
    std::memcpy(audio->data, wavaform_t.data(), sample_bytes);

    return audio;
}

void free_sd_audio(sd_audio_t* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_video_generation(sd_ctx->sd->version);
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        if (sd_version_is_dit(sd_ctx->sd->version)) {
            return EULER_SAMPLE_METHOD;
        }
    }
    return EULER_A_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        auto edm_v_denoiser = std::dynamic_pointer_cast<EDMVDenoiser>(sd_ctx->sd->denoiser);
        if (edm_v_denoiser) {
            return EXPONENTIAL_SCHEDULER;
        }
    }
    if (sample_method == LCM_SAMPLE_METHOD || sample_method == TCD_SAMPLE_METHOD) {
        return LCM_SCHEDULER;
    } else if (sample_method == DDIM_TRAILING_SAMPLE_METHOD) {
        return SIMPLE_SCHEDULER;
    } else if (sd_ctx != nullptr && sd_ctx->sd != nullptr && sd_version_is_ltxav(sd_ctx->sd->version)) {
        return LTX2_SCHEDULER;
    }
    return DISCRETE_SCHEDULER;
}

static int64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    srand((int)time(nullptr));
    return rand();
}

static enum sample_method_t resolve_sample_method(sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sample_method == SAMPLE_METHOD_COUNT) {
        return sd_get_default_sample_method(sd_ctx);
    }
    return sample_method;
}

static scheduler_t resolve_scheduler(sd_ctx_t* sd_ctx,
                                     scheduler_t scheduler,
                                     enum sample_method_t sample_method) {
    if (scheduler == SCHEDULER_COUNT) {
        return sd_get_default_scheduler(sd_ctx, sample_method);
    }
    return scheduler;
}

static float resolve_eta(sd_ctx_t* sd_ctx,
                         float eta,
                         enum sample_method_t sample_method) {
    if (eta == INFINITY) {
        if (sd_ctx->sd->version == VERSION_HIDREAM_O1) {
            return 8.f;
        }
        switch (sample_method) {
            case DDIM_TRAILING_SAMPLE_METHOD:
            case TCD_SAMPLE_METHOD:
            case RES_MULTISTEP_SAMPLE_METHOD:
            case RES_2S_SAMPLE_METHOD:
                return 0.0f;
            case EULER_A_SAMPLE_METHOD:
            case DPMPP2S_A_SAMPLE_METHOD:
            case ER_SDE_SAMPLE_METHOD:
            case EULER_A_CFG_PP_SAMPLE_METHOD:
                return 1.0f;
            default:;
        }
        return 0.0f;
    }
    return eta;
}

struct GenerationRequest {
    std::string prompt;
    std::string negative_prompt;
    int width                                = -1;
    int height                               = -1;
    int clip_skip                            = -1;
    int vae_scale_factor                     = -1;
    int diffusion_model_down_factor          = -1;
    int64_t seed                             = -1;
    bool use_uncond                          = false;
    bool use_img_cond                        = false;
    bool use_high_noise_uncond               = false;
    bool use_high_noise_img_cond             = false;
    const sd_cache_params_t* cache_params    = nullptr;
    int batch_count                          = 1;
    int shifted_timestep                     = 0;
    float strength                           = 1.f;
    float control_strength                   = 0.f;
    float eta                                = 0.f;
    bool increase_ref_index                  = false;
    bool auto_resize_ref_image               = false;
    sd_guidance_params_t guidance            = {};
    sd_guidance_params_t high_noise_guidance = {};
    sd_pm_params_t pm_params                 = {};
    sd_hires_params_t hires                  = {};
    int frames                               = -1;
    int requested_frames                     = -1;
    int fps                                  = 16;
    float vace_strength                      = 1.f;

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
        prompt                      = SAFE_STR(sd_img_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_img_gen_params->negative_prompt);
        width                       = sd_img_gen_params->width;
        height                      = sd_img_gen_params->height;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_img_gen_params->seed;
        batch_count                 = sd_img_gen_params->batch_count;
        clip_skip                   = sd_img_gen_params->clip_skip;
        shifted_timestep            = sd_img_gen_params->sample_params.shifted_timestep;
        strength                    = sd_img_gen_params->strength;
        control_strength            = sd_img_gen_params->control_strength;
        eta                         = sd_img_gen_params->sample_params.eta;
        increase_ref_index          = sd_img_gen_params->increase_ref_index;
        auto_resize_ref_image       = sd_img_gen_params->auto_resize_ref_image;
        guidance                    = sd_img_gen_params->sample_params.guidance;
        pm_params                   = sd_img_gen_params->pm_params;
        hires                       = sd_img_gen_params->hires;
        cache_params                = &sd_img_gen_params->cache;
        resolve(sd_ctx);
    }

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params) {
        prompt                      = SAFE_STR(sd_vid_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_vid_gen_params->negative_prompt);
        width                       = sd_vid_gen_params->width;
        height                      = sd_vid_gen_params->height;
        requested_frames            = std::max(1, sd_vid_gen_params->video_frames);
        frames                      = sd_ctx->sd->align_video_frames(requested_frames);
        clip_skip                   = sd_vid_gen_params->clip_skip;
        fps                         = std::max(1, sd_vid_gen_params->fps);
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_vid_gen_params->seed;
        strength                    = sd_vid_gen_params->strength;
        cache_params                = &sd_vid_gen_params->cache;
        vace_strength               = sd_vid_gen_params->vace_strength;
        guidance                    = sd_vid_gen_params->sample_params.guidance;
        high_noise_guidance         = sd_vid_gen_params->high_noise_sample_params.guidance;
        hires                       = sd_vid_gen_params->hires;
        resolve(sd_ctx);
        if (frames != requested_frames) {
            LOG_WARN("align video frames from %d to %d for %s",
                     requested_frames,
                     frames,
                     model_version_to_str[sd_ctx->sd->version]);
        }
    }

    void align_generation_request_size() {
        align_image_size(&width, &height, "generation request");
    }

    void align_image_size(int* target_width, int* target_height, const char* label) {
        int spatial_multiple = vae_scale_factor * diffusion_model_down_factor;
        int width_offset     = align_up_offset(*target_width, spatial_multiple);
        int height_offset    = align_up_offset(*target_height, spatial_multiple);
        if (width_offset <= 0 && height_offset <= 0) {
            return;
        }

        int original_width  = *target_width;
        int original_height = *target_height;

        *target_width += width_offset;
        *target_height += height_offset;
        LOG_WARN("align %s up %dx%d to %dx%d (multiple=%d)",
                 label,
                 original_width,
                 original_height,
                 *target_width,
                 *target_height,
                 spatial_multiple);
    }

    void resolve_hires() {
        if (!hires.enabled) {
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_NONE) {
            hires.enabled = false;
            return;
        }
        if (hires.upscaler < SD_HIRES_UPSCALER_NONE || hires.upscaler >= SD_HIRES_UPSCALER_COUNT) {
            LOG_WARN("hires upscaler '%d' is invalid, disabling hires", hires.upscaler);
            hires.enabled = false;
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_MODEL && strlen(SAFE_STR(hires.model_path)) == 0) {
            LOG_WARN("hires model upscaler requires a model path, disabling hires");
            hires.enabled = false;
            return;
        }
        if (hires.scale <= 0.f && hires.target_width <= 0 && hires.target_height <= 0) {
            LOG_WARN("hires scale must be positive when no target size is set, disabling hires");
            hires.enabled = false;
            return;
        }
        if (hires.custom_sigmas_count < 0) {
            LOG_WARN("hires custom sigmas count is negative, ignoring custom sigmas");
            hires.custom_sigmas       = nullptr;
            hires.custom_sigmas_count = 0;
        }
        if (hires.custom_sigmas_count > 0 && hires.custom_sigmas == nullptr) {
            LOG_WARN("hires custom sigmas count is positive but custom sigmas are null, ignoring custom sigmas");
            hires.custom_sigmas_count = 0;
        }
        if (hires.custom_sigmas_count == 1) {
            LOG_WARN("hires custom sigmas requires at least two values, ignoring custom sigmas");
            hires.custom_sigmas       = nullptr;
            hires.custom_sigmas_count = 0;
        }
        hires.denoising_strength = std::clamp(hires.denoising_strength, 0.0001f, 1.f);
        hires.steps              = std::max(0, hires.steps);

        if (hires.target_width > 0 && hires.target_height > 0) {
            // pass
        } else if (hires.target_width > 0) {
            hires.target_height = hires.target_width;
        } else if (hires.target_height > 0) {
            hires.target_width = hires.target_height;
        } else {
            hires.target_width  = static_cast<int>(std::round(width * hires.scale));
            hires.target_height = static_cast<int>(std::round(height * hires.scale));
        }

        if (hires.target_width <= 0 || hires.target_height <= 0) {
            LOG_WARN("hires target size is not positive, disabling hires");
            hires.enabled = false;
            return;
        }
        align_image_size(&hires.target_width, &hires.target_height, "hires target");
    }

    static void resolve_guidance(sd_ctx_t* sd_ctx,
                                 sd_guidance_params_t* guidance,
                                 bool* use_uncond,
                                 bool* use_img_cond,
                                 const char* stage_name = nullptr) {
        GGML_ASSERT(guidance != nullptr);
        GGML_ASSERT(use_uncond != nullptr);
        GGML_ASSERT(use_img_cond != nullptr);
        // out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        // img_cfg == txt_cfg means that img_cfg is not used
        if (!std::isfinite(guidance->img_cfg)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (!sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (guidance->txt_cfg != 1.f) {
            *use_uncond = true;
        }

        if (guidance->img_cfg != guidance->txt_cfg) {
            *use_img_cond = true;
            *use_uncond   = true;
        }

        if (guidance->txt_cfg < 1.f) {
            const char* prefix = stage_name == nullptr ? "" : stage_name;
            if (guidance->txt_cfg == 0.f) {
                LOG_WARN("%sunconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)",
                         prefix);
            } else {
                LOG_WARN("%scfg value out of expected range may produce unexpected results", prefix);
            }
        }
    }

    void resolve(sd_ctx_t* sd_ctx) {
        align_generation_request_size();
        resolve_hires();
        seed = resolve_seed(seed);

        resolve_guidance(sd_ctx, &guidance, &use_uncond, &use_img_cond);
        if (sd_ctx->sd->high_noise_diffusion_model) {
            resolve_guidance(sd_ctx,
                             &high_noise_guidance,
                             &use_high_noise_uncond,
                             &use_high_noise_img_cond,
                             "high noise: ");
        }

        if (shifted_timestep > 0 && !sd_version_is_sdxl(sd_ctx->sd->version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
    }
};

struct SamplePlan {
    enum sample_method_t sample_method            = SAMPLE_METHOD_COUNT;
    enum sample_method_t high_noise_sample_method = SAMPLE_METHOD_COUNT;
    const char* extra_sample_args                 = nullptr;
    const char* high_noise_extra_sample_args      = nullptr;
    float eta                                     = 0.f;
    float high_noise_eta                          = 0.f;
    int sample_steps                              = 0;
    int high_noise_sample_steps                   = 0;
    int total_steps                               = 0;
    float moe_boundary                            = 0.f;
    int start_merge_step                          = -1;
    std::vector<float> sigmas;

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_img_gen_params_t* sd_img_gen_params,
               const GenerationRequest& request) {
        sample_method     = sd_img_gen_params->sample_params.sample_method;
        extra_sample_args = sd_img_gen_params->sample_params.extra_sample_args;
        eta               = sd_img_gen_params->sample_params.eta;
        sample_steps      = sd_img_gen_params->sample_params.sample_steps;
        resolve(sd_ctx, &request, &sd_img_gen_params->sample_params);
    }

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_vid_gen_params_t* sd_vid_gen_params,
               const GenerationRequest& request) {
        sample_method     = sd_vid_gen_params->sample_params.sample_method;
        extra_sample_args = sd_vid_gen_params->sample_params.extra_sample_args;
        eta               = sd_vid_gen_params->sample_params.eta;
        sample_steps      = sd_vid_gen_params->sample_params.sample_steps;
        if (sd_ctx->sd->high_noise_diffusion_model) {
            high_noise_sample_steps      = sd_vid_gen_params->high_noise_sample_params.sample_steps;
            high_noise_sample_method     = sd_vid_gen_params->high_noise_sample_params.sample_method;
            high_noise_extra_sample_args = sd_vid_gen_params->high_noise_sample_params.extra_sample_args;
            high_noise_eta               = sd_vid_gen_params->high_noise_sample_params.eta;
        }
        moe_boundary = sd_vid_gen_params->moe_boundary;
        resolve(sd_ctx, &request, &sd_vid_gen_params->sample_params);
    }

    void resolve(sd_ctx_t* sd_ctx,
                 const GenerationRequest* request,
                 const sd_sample_params_t* sample_params) {
        sample_method = resolve_sample_method(sd_ctx, sample_method);

        total_steps = sample_steps + std::max(0, high_noise_sample_steps);

        if (sample_params->custom_sigmas_count > 0) {
            sigmas      = std::vector<float>(sample_params->custom_sigmas,
                                        sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set total_steps to %d", total_steps);
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set high_noise_sample_steps to %d", high_noise_sample_steps);
            }
        } else {
            scheduler_t scheduler = resolve_scheduler(sd_ctx,
                                                      sample_params->scheduler,
                                                      sample_method);
            int sample_seq_len    = sd_ctx->sd->get_image_seq_len(request->height, request->width);
            if (sd_version_is_ltxav(sd_ctx->sd->version) && request->frames > 0) {
                int latent_frames = ((request->frames - 1) / 8) + 1;
                sample_seq_len *= latent_frames;
            }
            sigmas = sd_ctx->sd->denoiser->get_sigmas(total_steps,
                                                      sample_seq_len,
                                                      scheduler,
                                                      sd_ctx->sd->version,
                                                      sample_params->extra_sample_args);
        }

        eta = resolve_eta(sd_ctx, eta, sample_method);

        if (high_noise_sample_steps < 0) {
            for (size_t i = 0; i < sigmas.size(); ++i) {
                if (sigmas[i] < moe_boundary) {
                    high_noise_sample_steps = static_cast<int>(i);
                    break;
                }
            }
            LOG_DEBUG("switching from high noise model at step %d", high_noise_sample_steps);
        }

        LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
        if (high_noise_sample_steps > 0) {
            high_noise_sample_method = resolve_sample_method(sd_ctx,
                                                             high_noise_sample_method);
            high_noise_eta           = resolve_eta(sd_ctx, high_noise_eta, high_noise_sample_method);
            LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);
        }

        if (sd_ctx->sd->use_pmid) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * total_steps);
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }
    }
};

struct ImageGenerationLatents {
    sd::Tensor<float> init_latent;
    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    sd::Tensor<float> audio_latent;
    sd::Tensor<float> video_positions;
    sd::Tensor<float> control_image;
    std::vector<sd::Tensor<float>> ref_images;
    std::vector<sd::Tensor<float>> ref_latents;
    sd::Tensor<float> denoise_mask;
    sd::Tensor<float> clip_vision_output;
    sd::Tensor<float> vace_context;
    int64_t ref_image_num                  = 0;
    int64_t video_conditioning_frame_count = 0;
    int64_t video_target_frame_count       = 0;
    int audio_length                       = 0;
};

static float ltxv_latent_corner_to_pixel_frame(int64_t corner_index,
                                               int temporal_scale,
                                               bool causal_temporal_positioning) {
    float pixel_t = static_cast<float>(corner_index * temporal_scale);
    if (causal_temporal_positioning) {
        pixel_t = std::max(0.f, pixel_t + 1.f - static_cast<float>(temporal_scale));
    }
    return pixel_t;
}

static void set_ltxv_video_position(sd::Tensor<float>* positions,
                                    int64_t token,
                                    float t_start,
                                    float t_end,
                                    float h_start,
                                    float h_end,
                                    float w_start,
                                    float w_end) {
    positions->index(0, 0, token, 0) = t_start;
    positions->index(1, 0, token, 0) = t_end;
    positions->index(0, 1, token, 0) = h_start;
    positions->index(1, 1, token, 0) = h_end;
    positions->index(0, 2, token, 0) = w_start;
    positions->index(1, 2, token, 0) = w_end;
}

static sd::Tensor<float> build_ltxv_video_positions(int64_t width,
                                                    int64_t height,
                                                    int64_t target_latent_frames,
                                                    int64_t keyframe_latent_frames,
                                                    int keyframe_frame_idx,
                                                    int keyframe_pixel_frames,
                                                    int fps,
                                                    int spatial_scale,
                                                    int temporal_scale,
                                                    bool causal_temporal_positioning) {
    GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0);
    GGML_ASSERT(keyframe_latent_frames > 0);
    GGML_ASSERT(fps > 0);

    int64_t total_tokens = width * height * (target_latent_frames + keyframe_latent_frames);
    sd::Tensor<float> positions({2, 3, total_tokens, 1});
    int64_t token = 0;

    for (int64_t t = 0; t < target_latent_frames; t++) {
        float t_start = ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        float t_end   = ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
        for (int64_t h = 0; h < height; h++) {
            float h_start = static_cast<float>(h * spatial_scale);
            float h_end   = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; w++) {
                float w_start = static_cast<float>(w * spatial_scale);
                float w_end   = static_cast<float>((w + 1) * spatial_scale);
                set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
            }
        }
    }

    for (int64_t t = 0; t < keyframe_latent_frames; t++) {
        float t_start = static_cast<float>(keyframe_frame_idx + t * temporal_scale);
        float t_end   = static_cast<float>(keyframe_frame_idx + (t + 1) * temporal_scale);
        if (keyframe_pixel_frames == 1) {
            t_end = t_start + 1.f;
        }
        t_start /= static_cast<float>(fps);
        t_end /= static_cast<float>(fps);
        for (int64_t h = 0; h < height; h++) {
            float h_start = static_cast<float>(h * spatial_scale);
            float h_end   = static_cast<float>((h + 1) * spatial_scale);
            for (int64_t w = 0; w < width; w++) {
                float w_start = static_cast<float>(w * spatial_scale);
                float w_end   = static_cast<float>((w + 1) * spatial_scale);
                set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
            }
        }
    }

    return positions;
}

static sd::Tensor<float> pack_ltxav_audio_and_video_latents(const sd::Tensor<float>& video_latent,
                                                            const sd::Tensor<float>& audio_latent) {
    if (audio_latent.empty()) {
        return video_latent;
    }

    GGML_ASSERT(video_latent.dim() == 4 || video_latent.dim() == 5);
    GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
    if (video_latent.dim() == 5) {
        GGML_ASSERT(video_latent.shape()[4] == 1);
    }
    if (audio_latent.dim() == 4) {
        GGML_ASSERT(audio_latent.shape()[3] == 1);
    }

    int64_t width        = video_latent.shape()[0];
    int64_t height       = video_latent.shape()[1];
    int64_t frames       = video_latent.shape()[2];
    int64_t video_ch     = video_latent.shape()[3];
    int64_t spatial_size = width * height * frames;
    int64_t audio_values = audio_latent.numel();
    int64_t extra_ch     = (audio_values + spatial_size - 1) / spatial_size;

    std::vector<int64_t> packed_shape = video_latent.shape();
    packed_shape[3]                   = video_ch + extra_ch;
    sd::Tensor<float> packed          = sd::zeros<float>(packed_shape);

    std::copy_n(video_latent.data(), video_latent.numel(), packed.data());
    std::copy_n(audio_latent.data(), audio_latent.numel(), packed.data() + video_latent.numel());
    return packed;
}

static sd::Tensor<float> pack_ltxav_audio_and_video_denoise_mask(const sd::Tensor<float>& video_mask,
                                                                 const sd::Tensor<float>& video_latent,
                                                                 const sd::Tensor<float>& audio_latent) {
    if (video_mask.empty() || audio_latent.empty()) {
        return video_mask;
    }

    GGML_ASSERT(video_latent.dim() == 4 || video_latent.dim() == 5);
    GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
    if (video_latent.dim() == 5) {
        GGML_ASSERT(video_latent.shape()[4] == 1);
    }
    if (audio_latent.dim() == 4) {
        GGML_ASSERT(audio_latent.shape()[3] == 1);
    }

    int64_t width        = video_latent.shape()[0];
    int64_t height       = video_latent.shape()[1];
    int64_t frames       = video_latent.shape()[2];
    int64_t video_ch     = video_latent.shape()[3];
    int64_t spatial_size = width * height * frames;
    int64_t audio_values = audio_latent.numel();
    int64_t extra_ch     = (audio_values + spatial_size - 1) / spatial_size;

    GGML_ASSERT(video_mask.dim() == video_latent.dim());
    GGML_ASSERT(video_mask.shape()[0] == width);
    GGML_ASSERT(video_mask.shape()[1] == height);
    GGML_ASSERT(video_mask.shape()[2] == frames);
    if (video_mask.dim() == 5) {
        GGML_ASSERT(video_mask.shape()[4] == video_latent.shape()[4]);
    }

    int64_t mask_ch = video_mask.shape()[3];
    if (mask_ch == video_ch + extra_ch) {
        return video_mask;
    }
    GGML_ASSERT(mask_ch == 1 || mask_ch == video_ch);

    sd::Tensor<float> video_mask_full = video_mask;
    if (mask_ch == 1 && video_ch != 1) {
        video_mask_full = video_mask * sd::Tensor<float>::ones(video_latent.shape());
    }

    std::vector<int64_t> audio_mask_shape = video_latent.shape();
    audio_mask_shape[3]                   = extra_ch;
    auto audio_mask                       = sd::Tensor<float>::ones(audio_mask_shape);
    return sd::ops::concat(video_mask_full, audio_mask, 3);
}

static sd::Tensor<float> make_ltxav_video_denoise_mask(const sd::Tensor<float>& video_latent, float value = 1.f) {
    if (video_latent.empty()) {
        return {};
    }
    return sd::full<float>({video_latent.shape()[0],
                            video_latent.shape()[1],
                            video_latent.shape()[2],
                            1,
                            1},
                           value);
}

static sd::Tensor<float> encode_ltxav_condition_image(sd_ctx_t* sd_ctx,
                                                      const sd::Tensor<float>& image,
                                                      const char* name) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image.empty()) {
        return {};
    }
    auto condition_image  = image.reshape({image.shape()[0],
                                           image.shape()[1],
                                           1,
                                           image.shape()[2],
                                           image.shape()[3]});
    auto condition_latent = sd_ctx->sd->encode_first_stage(condition_image);
    if (condition_latent.empty()) {
        LOG_ERROR("failed to encode LTXAV %s image", name);
    }
    return condition_latent;
}

static bool apply_ltxav_condition_by_latent_index(sd::Tensor<float>* video_latent,
                                                  sd::Tensor<float>* video_mask,
                                                  const sd::Tensor<float>& condition_latent,
                                                  int64_t latent_idx,
                                                  const char* name,
                                                  float conditioned_mask) {
    if (video_latent == nullptr || video_mask == nullptr || video_latent->empty() || video_mask->empty()) {
        return false;
    }
    if (condition_latent.empty() ||
        condition_latent.shape()[0] != video_latent->shape()[0] ||
        condition_latent.shape()[1] != video_latent->shape()[1] ||
        condition_latent.shape()[3] != video_latent->shape()[3]) {
        LOG_ERROR("invalid LTXAV %s condition latent shape", name);
        return false;
    }
    int64_t latent_frames    = video_latent->shape()[2];
    int64_t condition_frames = condition_latent.shape()[2];
    if (latent_idx < 0 || condition_frames <= 0 || latent_idx + condition_frames > latent_frames) {
        LOG_ERROR("invalid LTXAV %s image latent range: start=%" PRId64 ", length=%" PRId64 ", latent_frames=%" PRId64,
                  name,
                  latent_idx,
                  condition_frames,
                  latent_frames);
        return false;
    }

    sd::ops::slice_assign(video_latent, 2, latent_idx, latent_idx + condition_frames, condition_latent);
    sd::ops::fill_slice(video_mask, 2, latent_idx, latent_idx + condition_frames, conditioned_mask);
    return true;
}

static bool apply_ltxav_condition_image_by_latent_index(sd_ctx_t* sd_ctx,
                                                        const sd::Tensor<float>& image,
                                                        sd::Tensor<float>* video_latent,
                                                        sd::Tensor<float>* video_mask,
                                                        int64_t latent_idx,
                                                        const char* name,
                                                        float strength) {
    auto condition_latent = encode_ltxav_condition_image(sd_ctx, image, name);
    return !condition_latent.empty() &&
           apply_ltxav_condition_by_latent_index(video_latent,
                                                 video_mask,
                                                 condition_latent,
                                                 latent_idx,
                                                 name,
                                                 1.0f - std::clamp(strength, 0.f, 1.f));
}

static sd::Tensor<float> unpack_ltxav_audio_latent(const sd::Tensor<float>& packed_latent,
                                                   int audio_length,
                                                   int video_channels) {
    if (packed_latent.empty() || audio_length <= 0) {
        return {};
    }

    GGML_ASSERT(packed_latent.dim() == 4 || packed_latent.dim() == 5);
    int64_t width          = packed_latent.shape()[0];
    int64_t height         = packed_latent.shape()[1];
    int64_t frames         = packed_latent.shape()[2];
    int64_t total_channels = packed_latent.shape()[3];
    int64_t spatial_size   = width * height * frames;
    if (total_channels <= video_channels) {
        return {};
    }

    constexpr int kLtxavAudioFrequencyBins = 16;
    constexpr int kLtxavAudioChannels      = 8;
    int64_t required_values                = static_cast<int64_t>(audio_length) * kLtxavAudioFrequencyBins * kLtxavAudioChannels;
    int64_t packed_values                  = (total_channels - video_channels) * spatial_size;
    if (packed_values < required_values) {
        return {};
    }

    sd::Tensor<float> audio_latent({kLtxavAudioFrequencyBins, audio_length, kLtxavAudioChannels, 1});
    const float* audio_src = packed_latent.data() + static_cast<size_t>(video_channels) * static_cast<size_t>(spatial_size);
    std::copy_n(audio_src, static_cast<size_t>(required_values), audio_latent.data());
    return audio_latent;
}

static sd::Tensor<float> make_ltxav_empty_audio_latent(int audio_length) {
    if (audio_length <= 0) {
        return {};
    }
    constexpr int kLtxavAudioFrequencyBins = 16;
    constexpr int kLtxavAudioChannels      = 8;
    return sd::zeros<float>({kLtxavAudioFrequencyBins, audio_length, kLtxavAudioChannels, 1});
}

static sd::Tensor<float> resize_ltxav_audio_latent(const sd::Tensor<float>& audio_latent,
                                                   int target_audio_length) {
    auto resized = make_ltxav_empty_audio_latent(target_audio_length);
    if (resized.empty() || audio_latent.empty()) {
        return resized;
    }
    GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
    int copy_length = std::min(static_cast<int>(audio_latent.shape()[1]), target_audio_length);
    if (copy_length > 0) {
        auto copied = sd::ops::slice(audio_latent, 1, 0, copy_length);
        sd::ops::slice_assign(&resized, 1, 0, copy_length, copied);
    }
    return resized;
}

static int get_ltxav_num_audio_latents(int frames, int fps) {
    GGML_ASSERT(frames > 0);
    GGML_ASSERT(fps > 0);
    constexpr float kSampleRate            = 16000.0f;
    constexpr float kMelHopLength          = 160.0f;
    constexpr float kAudioLatentDownsample = 4.0f;
    constexpr float kLatentsPerSecond      = kSampleRate / kMelHopLength / kAudioLatentDownsample;
    return static_cast<int>(std::ceil((static_cast<float>(frames) / static_cast<float>(fps)) * kLatentsPerSecond));
}

struct ImageGenerationEmbeds {
    SDCondition cond;
    SDCondition uncond;
    SDCondition img_cond;
    SDCondition id_cond;
};

struct CircularAxesState {
    bool circular_x = false;
    bool circular_y = false;
};

static CircularAxesState configure_image_vae_axes(sd_ctx_t* sd_ctx,
                                                  const sd_img_gen_params_t* sd_img_gen_params,
                                                  const GenerationRequest& request) {
    CircularAxesState original_axes = {sd_ctx->sd->circular_x, sd_ctx->sd->circular_y};

    if (!sd_img_gen_params->vae_tiling_params.enabled) {
        if (sd_ctx->sd->first_stage_model) {
            sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        if (sd_ctx->sd->preview_vae) {
            sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        return original_axes;
    }

    int tile_size_x, tile_size_y;
    float overlap;
    int latent_size_x = request.width / request.vae_scale_factor;
    int latent_size_y = request.height / request.vae_scale_factor;
    sd_ctx->sd->first_stage_model->get_tile_sizes(tile_size_x,
                                                  tile_size_y,
                                                  overlap,
                                                  sd_img_gen_params->vae_tiling_params,
                                                  latent_size_x,
                                                  latent_size_y);

    sd_ctx->sd->circular_x = sd_ctx->sd->circular_x && (tile_size_x >= latent_size_x);
    sd_ctx->sd->circular_y = sd_ctx->sd->circular_y && (tile_size_y >= latent_size_y);

    if (sd_ctx->sd->first_stage_model) {
        sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }
    if (sd_ctx->sd->preview_vae) {
        sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }

    sd_ctx->sd->circular_x = original_axes.circular_x && (tile_size_x < latent_size_x);
    sd_ctx->sd->circular_y = original_axes.circular_y && (tile_size_y < latent_size_y);

    return original_axes;
}

static void restore_image_vae_axes(sd_ctx_t* sd_ctx, const CircularAxesState& original_axes) {
    sd_ctx->sd->circular_x = original_axes.circular_x;
    sd_ctx->sd->circular_y = original_axes.circular_y;
}

class ImageVaeAxesGuard {
private:
    sd_ctx_t* sd_ctx = nullptr;
    CircularAxesState original_axes;

public:
    ImageVaeAxesGuard(sd_ctx_t* sd_ctx,
                      const sd_img_gen_params_t* sd_img_gen_params,
                      const GenerationRequest& request)
        : sd_ctx(sd_ctx),
          original_axes(configure_image_vae_axes(sd_ctx, sd_img_gen_params, request)) {}

    ~ImageVaeAxesGuard() {
        restore_image_vae_axes(sd_ctx, original_axes);
    }

    ImageVaeAxesGuard(const ImageVaeAxesGuard&)            = delete;
    ImageVaeAxesGuard& operator=(const ImageVaeAxesGuard&) = delete;
};

static std::optional<ImageGenerationLatents> prepare_image_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_img_gen_params_t* sd_img_gen_params,
                                                                              GenerationRequest* request,
                                                                              SamplePlan* plan) {
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> init_image_tensor;
    sd::Tensor<float> control_image_tensor;
    sd::Tensor<float> mask_image_tensor;

    if (sd_img_gen_params->init_image.data != nullptr) {
        LOG_INFO("IMG2IMG");

        if (request->strength < 1.f) {
            size_t t_enc = static_cast<size_t>(plan->sample_steps * request->strength);
            if (t_enc == static_cast<size_t>(plan->sample_steps)) {
                t_enc--;
            }
            LOG_INFO("target t_enc is %zu steps", t_enc);
            std::vector<float> sigma_sched;
            sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());
            plan->sigmas       = std::move(sigma_sched);
            plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
        }

        init_image_tensor = sd_image_to_tensor(sd_img_gen_params->init_image, request->width, request->height);
    }

    if (sd_img_gen_params->mask_image.data != nullptr) {
        mask_image_tensor = sd_image_to_tensor(sd_img_gen_params->mask_image, request->width, request->height);
        mask_image_tensor = sd::ops::round(mask_image_tensor);
    }

    if (sd_img_gen_params->control_image.data != nullptr) {
        control_image_tensor = sd_image_to_tensor(sd_img_gen_params->control_image, request->width, request->height);
    }

    if (init_image_tensor.empty() || mask_image_tensor.empty()) {
        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            LOG_WARN("inpainting model requires both an init image and a mask image.");
        }
    }

    if (mask_image_tensor.empty()) {
        mask_image_tensor = sd::full<float>({request->width, request->height, 1, 1}, 1.f);
    }

    sd::Tensor<float> latent_mask = sd::ops::interpolate(mask_image_tensor,
                                                         {request->width / request->vae_scale_factor,
                                                          request->height / request->vae_scale_factor,
                                                          1,
                                                          1},
                                                         sd::ops::InterpolateMode::NearestMax);

    sd::Tensor<float> init_latent;
    sd::Tensor<float> control_latent;
    if (init_image_tensor.empty()) {
        init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(init_image_tensor);
        if (init_latent.empty()) {
            LOG_ERROR("failed to encode init image");
            return std::nullopt;
        }
    }

    if (!control_image_tensor.empty() && !sd_ctx->sd->vae_decode_only) {
        control_latent = sd_ctx->sd->encode_first_stage(control_image_tensor);
        if (control_latent.empty()) {
            LOG_ERROR("failed to encode control image");
            return std::nullopt;
        }
    }

    std::vector<sd::Tensor<float>> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(sd_image_to_tensor(sd_img_gen_params->ref_images[i]));
    }

    if (ref_images.empty() && sd_version_is_unet_edit(sd_ctx->sd->version)) {
        LOG_WARN("This model needs at least one reference image; using an empty reference");
        ref_images.push_back(sd::zeros<float>({request->width, request->height, 3, 1}));
        request->guidance.img_cfg = request->guidance.txt_cfg;
    }

    if (!ref_images.empty()) {
        LOG_INFO("EDIT mode");
    }

    std::vector<sd::Tensor<float>> ref_latents;
    for (size_t i = 0; i < ref_images.size(); i++) {
        if (sd_ctx->sd->version == VERSION_HIDREAM_O1) {
            continue;
        }
        sd::Tensor<float> ref_latent;
        if (request->auto_resize_ref_image) {
            LOG_DEBUG("auto resize ref images");
            int vae_image_size = std::min(1024 * 1024, request->width * request->height);
            double vae_width   = sqrt(vae_image_size * ref_images[i].shape()[0] / ref_images[i].shape()[1]);
            double vae_height  = vae_width * ref_images[i].shape()[1] / ref_images[i].shape()[0];

            int factor = sd_version_is_qwen_image(sd_ctx->sd->version) ? 32 : 16;
            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            auto resized_ref_img = sd::ops::interpolate(ref_images[i],
                                                        {static_cast<int>(vae_width), static_cast<int>(vae_height), 3, 1});

            LOG_DEBUG("resize vae ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64,
                      static_cast<int>(i),
                      ref_images[i].shape()[1],
                      ref_images[i].shape()[0],
                      resized_ref_img.shape()[1],
                      resized_ref_img.shape()[0]);

            ref_latent = sd_ctx->sd->encode_first_stage(resized_ref_img);
        } else {
            ref_latent = sd_ctx->sd->encode_first_stage(ref_images[i]);
        }
        if (ref_latent.empty()) {
            LOG_ERROR("failed to encode reference image %d", static_cast<int>(i));
            return std::nullopt;
        }

        ref_latents.push_back(std::move(ref_latent));
    }

    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        sd::Tensor<float> masked_init_latent;

        if (sd_ctx->sd->version != VERSION_FLEX_2) {
            if (!init_image_tensor.empty()) {
                auto masked_image  = ((1.0f - mask_image_tensor) * (init_image_tensor - 0.5f)) + 0.5f;
                masked_init_latent = sd_ctx->sd->encode_first_stage(masked_image);
                if (masked_init_latent.empty()) {
                    LOG_ERROR("failed to encode masked init image");
                    return std::nullopt;
                }
            } else {
                masked_init_latent = sd::Tensor<float>::zeros_like(init_latent);
            }
        } else {
            masked_init_latent = ((1.0f - latent_mask) * init_latent);
        }

        auto uncond_masked_init_latent = sd::Tensor<float>::zeros_like(masked_init_latent);

        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            auto mask = mask_image_tensor.reshape({request->vae_scale_factor,
                                                   request->width / request->vae_scale_factor,
                                                   request->vae_scale_factor,
                                                   request->height / request->vae_scale_factor});
            mask      = mask.permute({1, 3, 0, 2}).reshape({request->width / request->vae_scale_factor, request->height / request->vae_scale_factor, request->vae_scale_factor * request->vae_scale_factor, 1});

            concat_latent        = sd::ops::concat(masked_init_latent, mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, mask, 2);
        } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
            concat_latent = sd::ops::concat(masked_init_latent, latent_mask, 2);
            if (!control_latent.empty()) {
                concat_latent = sd::ops::concat(concat_latent, control_latent, 2);
            } else {
                concat_latent = sd::ops::concat(concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
            }

            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, latent_mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
        } else {  // SD1.x SD2.x SDXL inpaint
            concat_latent        = sd::ops::concat(latent_mask, masked_init_latent, 2);
            uncond_concat_latent = sd::ops::concat(latent_mask, uncond_masked_init_latent, 2);
        }
    }
    if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        concat_latent        = sd::ops::interpolate<float>(ref_latents[0], init_latent.shape());
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }
    if (sd_ctx->sd->version == VERSION_FLUX_CONTROLS) {
        if (!control_latent.empty()) {
            concat_latent = control_latent;
        } else {
            concat_latent = sd::Tensor<float>::zeros_like(init_latent);
        }
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }

    if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0) {
        int64_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);
    }

    ImageGenerationLatents latents;
    latents.init_latent          = std::move(init_latent);
    latents.concat_latent        = std::move(concat_latent);
    latents.uncond_concat_latent = std::move(uncond_concat_latent);
    latents.control_image        = std::move(control_image_tensor);
    latents.ref_images           = std::move(ref_images);
    latents.ref_latents          = std::move(ref_latents);

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        latent_mask = sd::ops::max_pool_2d(latent_mask,
                                           {3, 3},
                                           {1, 1},
                                           {1, 1});
    }
    latents.denoise_mask = std::move(latent_mask);

    return latents;
}

static std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds(sd_ctx_t* sd_ctx,
                                                                            const sd_img_gen_params_t* sd_img_gen_params,
                                                                            GenerationRequest* request,
                                                                            SamplePlan* plan,
                                                                            ImageGenerationLatents* latents) {
    ConditionerParams condition_params;
    condition_params.text       = request->prompt;
    condition_params.clip_skip  = request->clip_skip;
    condition_params.width      = request->width;
    condition_params.height     = request->height;
    condition_params.ref_images = &latents->ref_images;

    auto id_cond                     = sd_ctx->sd->get_pmid_conditon(request->pm_params, condition_params);
    int64_t prepare_start_ms         = ggml_time_ms();
    condition_params.zero_out_masked = false;
    auto cond                        = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                           condition_params);
    if (cond.c_concat.empty()) {
        cond.c_concat = latents->concat_latent;  // TODO: optimize
    }

    SDCondition uncond;
    if (request->use_uncond || request->use_high_noise_uncond) {
        bool zero_out_masked = false;
        if (sd_version_is_sdxl(sd_ctx->sd->version) &&
            request->negative_prompt.empty() &&
            !sd_ctx->sd->is_using_edm_v_parameterization) {
            zero_out_masked = true;
        }
        condition_params.text            = request->negative_prompt;
        condition_params.zero_out_masked = zero_out_masked;
        uncond                           = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                               condition_params);
        if (uncond.c_concat.empty()) {
            uncond.c_concat = latents->uncond_concat_latent;  // TODO: optimize
        }
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    ImageGenerationEmbeds embeds;
    if (request->use_img_cond) {
        embeds.img_cond = SDCondition(uncond.c_crossattn, uncond.c_vector, cond.c_concat);
    }
    embeds.cond    = std::move(cond);
    embeds.uncond  = std::move(uncond);
    embeds.id_cond = std::move(id_cond);

    return embeds;
}

static sd_image_t* decode_image_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const std::vector<sd::Tensor<float>>& final_latents) {
    if (final_latents.size() != static_cast<size_t>(request.batch_count)) {
        LOG_ERROR("expected %d latents, got %zu", request.batch_count, final_latents.size());
        return nullptr;
    }
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<sd::Tensor<float>> decoded_images;
    int64_t t0 = ggml_time_ms();

    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);
        if (image.empty()) {
            LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->first_stage_model->free_params_buffer();
            }
            return nullptr;
        }
        decoded_images.push_back(std::move(image));
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %zu decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t0) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_image_t* result_images = (sd_image_t*)calloc(request.batch_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    memset(result_images, 0, request.batch_count * sizeof(sd_image_t));

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i] = tensor_to_sd_image(decoded_images[i]);
    }

    return result_images;
}

static sd::Tensor<float> upscale_hires_latent(sd_ctx_t* sd_ctx,
                                              const sd::Tensor<float>& latent,
                                              const GenerationRequest& request,
                                              UpscalerGGML* upscaler) {
    auto get_hires_latent_target_shape = [&]() {
        std::vector<int64_t> target_shape = latent.shape();
        if (target_shape.size() < 2) {
            target_shape.clear();
            return target_shape;
        }
        target_shape[0] = request.hires.target_width / request.vae_scale_factor;
        target_shape[1] = request.hires.target_height / request.vae_scale_factor;
        return target_shape;
    };

    if (request.hires.upscaler == SD_HIRES_UPSCALER_LATENT ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_ANTIALIASED ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED) {
        std::vector<int64_t> target_shape = get_hires_latent_target_shape();
        if (target_shape.empty()) {
            LOG_ERROR("latent has invalid shape for hires upscale");
            return {};
        }

        sd::ops::InterpolateMode mode = sd::ops::InterpolateMode::Nearest;
        bool antialias                = false;
        switch (request.hires.upscaler) {
            case SD_HIRES_UPSCALER_LATENT:
                mode = sd::ops::InterpolateMode::Bilinear;
                break;
            case SD_HIRES_UPSCALER_LATENT_NEAREST:
                mode = sd::ops::InterpolateMode::Nearest;
                break;
            case SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT:
                mode = sd::ops::InterpolateMode::NearestExact;
                break;
            case SD_HIRES_UPSCALER_LATENT_ANTIALIASED:
                mode      = sd::ops::InterpolateMode::Bilinear;
                antialias = true;
                break;
            case SD_HIRES_UPSCALER_LATENT_BICUBIC:
                mode = sd::ops::InterpolateMode::Bicubic;
                break;
            case SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED:
                mode      = sd::ops::InterpolateMode::Bicubic;
                antialias = true;
                break;
            default:
                break;
        }

        LOG_INFO("hires %s upscale %" PRId64 "x%" PRId64 " -> %" PRId64 "x%" PRId64,
                 sd_hires_upscaler_name(request.hires.upscaler),
                 latent.shape()[0],
                 latent.shape()[1],
                 target_shape[0],
                 target_shape[1]);

        return sd::ops::interpolate(latent, target_shape, mode, false, antialias);
    } else if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL ||
               request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS ||
               request.hires.upscaler == SD_HIRES_UPSCALER_NEAREST) {
        if (sd_ctx->sd->vae_decode_only) {
            LOG_ERROR("hires %s upscaler requires VAE encoder weights; create the context with vae_decode_only=false",
                      sd_hires_upscaler_name(request.hires.upscaler));
            return {};
        }
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL && upscaler == nullptr) {
            LOG_ERROR("hires model upscaler context is null");
            return {};
        }

        sd::Tensor<float> decoded = sd_ctx->sd->decode_first_stage(latent);
        if (decoded.empty()) {
            LOG_ERROR("decode_first_stage failed before hires %s upscale",
                      sd_hires_upscaler_name(request.hires.upscaler));
            return {};
        }

        sd::Tensor<float> upscaled_tensor;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            upscaled_tensor = upscaler->upscale_tensor(decoded);
            if (upscaled_tensor.empty()) {
                LOG_ERROR("hires model upscale failed");
                return {};
            }

            if (upscaled_tensor.shape()[0] != request.hires.target_width ||
                upscaled_tensor.shape()[1] != request.hires.target_height) {
                upscaled_tensor = sd::ops::interpolate(upscaled_tensor,
                                                       {request.hires.target_width,
                                                        request.hires.target_height,
                                                        upscaled_tensor.shape()[2],
                                                        upscaled_tensor.shape()[3]});
            }
        } else {
            sd::ops::InterpolateMode mode = request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS
                                                ? sd::ops::InterpolateMode::Lanczos
                                                : sd::ops::InterpolateMode::Nearest;
            LOG_INFO("hires %s image upscale %" PRId64 "x%" PRId64 " -> %dx%d",
                     sd_hires_upscaler_name(request.hires.upscaler),
                     decoded.shape()[0],
                     decoded.shape()[1],
                     request.hires.target_width,
                     request.hires.target_height);
            upscaled_tensor = sd::ops::interpolate(decoded,
                                                   {request.hires.target_width,
                                                    request.hires.target_height,
                                                    decoded.shape()[2],
                                                    decoded.shape()[3]},
                                                   mode);
            upscaled_tensor = sd::ops::clamp(upscaled_tensor, 0.0f, 1.0f);
        }

        sd::Tensor<float> upscaled_latent = sd_ctx->sd->encode_first_stage(upscaled_tensor);
        if (upscaled_latent.empty()) {
            LOG_ERROR("encode_first_stage failed after hires %s upscale",
                      sd_hires_upscaler_name(request.hires.upscaler));
        }
        return upscaled_latent;
    }

    LOG_ERROR("unsupported hires upscaler '%s'", sd_hires_upscaler_name(request.hires.upscaler));
    return {};
}

static std::vector<float> make_hires_sigma_schedule(sd_ctx_t* sd_ctx,
                                                    const sd_hires_params_t& hires,
                                                    const sd_sample_params_t& sample_params,
                                                    sample_method_t sample_method,
                                                    int default_steps,
                                                    int sample_seq_len,
                                                    int* scheduler_steps_out) {
    if (scheduler_steps_out != nullptr) {
        *scheduler_steps_out = 0;
    }

    if (hires.custom_sigmas_count > 0 && hires.custom_sigmas != nullptr) {
        std::vector<float> custom_sigmas(hires.custom_sigmas,
                                         hires.custom_sigmas + hires.custom_sigmas_count);
        if (scheduler_steps_out != nullptr) {
            *scheduler_steps_out = static_cast<int>(custom_sigmas.size()) - 1;
        }
        return custom_sigmas;
    }

    int effective_steps = hires.steps > 0 ? hires.steps : default_steps;
    effective_steps     = std::max(1, effective_steps);

    // sd-webui behavior: scale up total steps so trimming by denoising_strength yields exactly hires_steps effective steps,
    // unlike img2img which trims from a fixed step count.
    int scheduler_steps = static_cast<int>(effective_steps / hires.denoising_strength);
    scheduler_steps     = std::max(1, scheduler_steps);

    scheduler_t scheduler     = resolve_scheduler(sd_ctx,
                                                  sample_params.scheduler,
                                                  sample_method);
    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(scheduler_steps,
                                                                 sample_seq_len,
                                                                 scheduler,
                                                                 sd_ctx->sd->version,
                                                                 sample_params.extra_sample_args);
    size_t t_enc              = static_cast<size_t>(scheduler_steps * hires.denoising_strength);
    if (t_enc >= static_cast<size_t>(scheduler_steps)) {
        t_enc = static_cast<size_t>(scheduler_steps) - 1;
    }
    if (scheduler_steps_out != nullptr) {
        *scheduler_steps_out = scheduler_steps;
    }
    return std::vector<float>(sigmas.begin() + scheduler_steps - static_cast<int>(t_enc) - 1,
                              sigmas.end());
}

SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    LOG_INFO("generate_image %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan);
    if (!latents_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents);
    if (!embeds_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    std::vector<sd::Tensor<float>> final_latents;
    int64_t denoise_start = ggml_time_ms();
    for (int b = 0; b < request.batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = request.seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

        sd::Tensor<float> x_0 = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                   true,
                                                   latents.init_latent,
                                                   std::move(noise),
                                                   embeds.cond,
                                                   embeds.uncond,
                                                   embeds.img_cond,
                                                   embeds.id_cond,
                                                   latents.control_image,
                                                   request.control_strength,
                                                   request.guidance,
                                                   plan.eta,
                                                   request.shifted_timestep,
                                                   plan.sample_method,
                                                   sd_ctx->sd->is_flow_denoiser(),
                                                   plan.extra_sample_args,
                                                   plan.sigmas,
                                                   plan.start_merge_step,
                                                   latents.ref_latents,
                                                   request.increase_ref_index,
                                                   latents.denoise_mask,
                                                   sd::Tensor<float>(),
                                                   1.f,
                                                   0,
                                                   static_cast<float>(request.fps),
                                                   request.cache_params);
        int64_t sampling_end  = ggml_time_ms();
        if (!x_0.empty()) {
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(std::move(x_0));
            continue;
        }

        LOG_ERROR("sampling for image %d/%d failed after %.2fs",
                  b + 1,
                  request.batch_count,
                  (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        return nullptr;
    }
    if (sd_ctx->sd->free_params_immediately && !request.hires.enabled) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %zu latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);

    if (request.hires.enabled && request.hires.target_width > 0) {
        LOG_INFO("hires fix: upscaling to %dx%d", request.hires.target_width, request.hires.target_height);

        std::unique_ptr<UpscalerGGML> hires_upscaler;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            LOG_INFO("hires fix: loading model upscaler from '%s'", request.hires.model_path);
            hires_upscaler                    = std::make_unique<UpscalerGGML>(sd_ctx->sd->n_threads,
                                                            false,
                                                            request.hires.upscale_tile_size,
                                                            sd_ctx->sd->backend_spec,
                                                            sd_ctx->sd->params_backend_spec);
            const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(sd_ctx->sd->max_vram);
            hires_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
            if (!hires_upscaler->load_from_file(request.hires.model_path,
                                                sd_ctx->sd->offload_params_to_cpu,
                                                sd_ctx->sd->n_threads)) {
                LOG_ERROR("load hires model upscaler failed");
                if (sd_ctx->sd->free_params_immediately) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }
        }

        int hires_scheduler_steps = 0;
        std::vector<float> hires_sigma_sched =
            make_hires_sigma_schedule(sd_ctx,
                                      request.hires,
                                      sd_img_gen_params->sample_params,
                                      plan.sample_method,
                                      plan.sample_steps,
                                      sd_ctx->sd->get_image_seq_len(request.hires.target_height, request.hires.target_width),
                                      &hires_scheduler_steps);
        LOG_INFO("hires fix: scheduler_steps=%d, denoising_strength=%.2f, sigma_sched_size=%zu%s",
                 hires_scheduler_steps,
                 request.hires.denoising_strength,
                 hires_sigma_sched.size(),
                 request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");

        std::vector<sd::Tensor<float>> hires_final_latents;
        int64_t hires_denoise_start = ggml_time_ms();
        for (int b = 0; b < (int)final_latents.size(); b++) {
            int64_t cur_seed = request.seed + b;
            sd_ctx->sd->rng->manual_seed(cur_seed);
            sd_ctx->sd->sampler_rng->manual_seed(cur_seed);

            sd::Tensor<float> upscaled = upscale_hires_latent(sd_ctx,
                                                              final_latents[b],
                                                              request,
                                                              hires_upscaler.get());
            if (upscaled.empty()) {
                if (sd_ctx->sd->free_params_immediately) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }

            sd::Tensor<float> noise = sd::randn_like<float>(upscaled, sd_ctx->sd->rng);

            sd::Tensor<float> hires_denoise_mask;
            if (!latents.denoise_mask.empty()) {
                std::vector<int64_t> mask_shape = latents.denoise_mask.shape();
                mask_shape[0]                   = upscaled.shape()[0];
                mask_shape[1]                   = upscaled.shape()[1];
                hires_denoise_mask              = sd::ops::interpolate(latents.denoise_mask,
                                                                       mask_shape,
                                                                       sd::ops::InterpolateMode::NearestMax);
            }

            int64_t hires_sample_start = ggml_time_ms();
            sd::Tensor<float> x_0      = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                            true,
                                                            upscaled,
                                                            std::move(noise),
                                                            embeds.cond,
                                                            embeds.uncond,
                                                            embeds.img_cond,
                                                            embeds.id_cond,
                                                            latents.control_image,
                                                            request.control_strength,
                                                            request.guidance,
                                                            plan.eta,
                                                            request.shifted_timestep,
                                                            plan.sample_method,
                                                            sd_ctx->sd->is_flow_denoiser(),
                                                            plan.extra_sample_args,
                                                            hires_sigma_sched,
                                                            plan.start_merge_step,
                                                            latents.ref_latents,
                                                            request.increase_ref_index,
                                                            hires_denoise_mask,
                                                            sd::Tensor<float>(),
                                                            1.f,
                                                            0,
                                                            static_cast<float>(request.fps),
                                                            request.cache_params);
            int64_t hires_sample_end   = ggml_time_ms();
            if (!x_0.empty()) {
                LOG_INFO("hires sampling %d/%d completed, taking %.2fs",
                         b + 1,
                         (int)final_latents.size(),
                         (hires_sample_end - hires_sample_start) * 1.0f / 1000);
                hires_final_latents.push_back(std::move(x_0));
                continue;
            }

            LOG_ERROR("hires sampling for image %d/%d failed after %.2fs",
                      b + 1,
                      (int)final_latents.size(),
                      (hires_sample_end - hires_sample_start) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return nullptr;
        }
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        int64_t hires_denoise_end = ggml_time_ms();
        LOG_INFO("hires fix completed, taking %.2fs", (hires_denoise_end - hires_denoise_start) * 1.0f / 1000);

        final_latents = std::move(hires_final_latents);
    }

    auto result = decode_image_outputs(sd_ctx, request, final_latents);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}

static std::optional<ImageGenerationLatents> prepare_video_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_vid_gen_params_t* sd_vid_gen_params,
                                                                              GenerationRequest* request) {
    ImageGenerationLatents latents;
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> start_image;
    sd::Tensor<float> end_image;

    if (sd_vid_gen_params->init_image.data) {
        start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, request->width, request->height);
    }

    if (sd_vid_gen_params->end_image.data) {
        end_image = sd_image_to_tensor(sd_vid_gen_params->end_image, request->width, request->height);
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version)) {
        latents.audio_length = get_ltxav_num_audio_latents(request->frames, request->fps);
        latents.audio_latent = make_ltxav_empty_audio_latent(latents.audio_length);
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version)) {
        if (sd_vid_gen_params->control_frames_size > 0) {
            LOG_ERROR("LTXAV control_frames are not implemented");
            return std::nullopt;
        }

        if (!start_image.empty() || !end_image.empty()) {
            if (sd_ctx->sd->vae_decode_only) {
                LOG_ERROR("LTXAV image conditioning requires VAE encoder weights; create the context with vae_decode_only=false");
                return std::nullopt;
            }

            if (!start_image.empty() && !end_image.empty()) {
                LOG_INFO("FLF2V");
            } else if (!start_image.empty()) {
                LOG_INFO("IMG2VID");
            } else {
                LOG_INFO("END2VID");
            }

            int64_t t1          = ggml_time_ms();
            latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);

            float conditioning_strength = std::clamp(request->strength, 0.f, 1.f);
            float conditioned_mask      = 1.0f - conditioning_strength;
            latents.denoise_mask        = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);

            auto apply_video_condition_by_keyframe_index = [&](const sd::Tensor<float>& keyframes,
                                                               int frame_idx,
                                                               const char* name) -> bool {
                int64_t keyframe_frames = keyframes.shape()[2];
                if (keyframe_frames <= 0 || keyframes.shape()[0] != latents.init_latent.shape()[0] ||
                    keyframes.shape()[1] != latents.init_latent.shape()[1] ||
                    keyframes.shape()[3] != latents.init_latent.shape()[3]) {
                    LOG_ERROR("invalid LTXAV %s keyframe latent shape", name);
                    return false;
                }

                latents.video_target_frame_count       = latents.init_latent.shape()[2];
                latents.video_conditioning_frame_count = keyframe_frames;
                latents.init_latent                    = sd::ops::concat(latents.init_latent, keyframes, 2);

                auto keyframe_mask      = sd::full<float>({keyframes.shape()[0],
                                                           keyframes.shape()[1],
                                                           keyframes.shape()[2],
                                                           1,
                                                           1},
                                                     conditioned_mask);
                latents.denoise_mask    = sd::ops::concat(latents.denoise_mask, keyframe_mask, 2);
                latents.video_positions = build_ltxv_video_positions(latents.init_latent.shape()[0],
                                                                     latents.init_latent.shape()[1],
                                                                     latents.video_target_frame_count,
                                                                     keyframe_frames,
                                                                     frame_idx,
                                                                     1,
                                                                     request->fps,
                                                                     request->vae_scale_factor,
                                                                     8,
                                                                     true);
                return true;
            };

            if (!start_image.empty()) {
                if (!apply_ltxav_condition_image_by_latent_index(sd_ctx,
                                                                 start_image,
                                                                 &latents.init_latent,
                                                                 &latents.denoise_mask,
                                                                 0,
                                                                 "init",
                                                                 conditioning_strength)) {
                    return std::nullopt;
                }
            }

            if (!end_image.empty()) {
                auto end_image_latent = encode_ltxav_condition_image(sd_ctx, end_image, "end");
                if (end_image_latent.empty()) {
                    return std::nullopt;
                }

                int frame_idx = request->frames - 1;
                bool ok       = frame_idx == 0 ? apply_ltxav_condition_by_latent_index(&latents.init_latent,
                                                                                       &latents.denoise_mask,
                                                                                       end_image_latent,
                                                                                       0,
                                                                                       "end",
                                                                                       conditioned_mask)
                                               : apply_video_condition_by_keyframe_index(end_image_latent, frame_idx, "end");
                if (!ok) {
                    return std::nullopt;
                }
            }

            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
        }
    }

    if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
        LOG_INFO("IMG2VID");

        if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
            if (!start_image.empty()) {
                auto clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2);
                if (clip_vision_output.empty()) {
                    LOG_ERROR("failed to compute clip vision output for init image");
                    return std::nullopt;
                }
                latents.clip_vision_output = std::move(clip_vision_output);
            } else {
                latents.clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2, true);
            }

            if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
                sd::Tensor<float> end_image_clip_vision_output;
                if (!end_image.empty()) {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2);
                    if (end_image_clip_vision_output.empty()) {
                        LOG_ERROR("failed to compute clip vision output for end image");
                        return std::nullopt;
                    }
                } else {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2, true);
                }
                latents.clip_vision_output = sd::ops::concat(latents.clip_vision_output, end_image_clip_vision_output, 1);
            }

            int64_t t1 = ggml_time_ms();
            LOG_INFO("get_clip_vision_output completed, taking %" PRId64 " ms", t1 - prepare_start_ms);
        }

        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        if (!start_image.empty()) {
            sd::ops::slice_assign(&image, 2, 0, 1, start_image.unsqueeze(2));
        }
        if (!end_image.empty()) {
            sd::ops::slice_assign(&image, 2, request->frames - 1, request->frames, end_image.unsqueeze(2));
        }

        auto concat_latent = sd_ctx->sd->encode_first_stage(image);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (concat_latent.empty()) {
            LOG_ERROR("failed to encode video conditioning frames");
            return std::nullopt;
        }
        latents.concat_latent = std::move(concat_latent);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);

        sd::Tensor<float> concat_mask = sd::zeros<float>({latents.concat_latent.shape()[0],
                                                          latents.concat_latent.shape()[1],
                                                          latents.concat_latent.shape()[2],
                                                          4,
                                                          1});  // [b, 4, t, h/vae_scale_factor, w/vae_scale_factor]
        if (!start_image.empty()) {
            sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
        }
        if (!end_image.empty()) {
            auto last_channel = sd::ops::slice(concat_mask, 3, 3, 4);
            sd::ops::fill_slice(&last_channel, 2, last_channel.shape()[2] - 1, last_channel.shape()[2], 1.0f);
            sd::ops::slice_assign(&concat_mask, 3, 3, 4, last_channel);
        }
        latents.concat_latent = sd::ops::concat(concat_mask, latents.concat_latent, 3);  // [b, 4+c, t, h/vae_scale_factor, w/vae_scale_factor]
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-TI2V-5B" && !start_image.empty()) {
        LOG_INFO("IMG2VID");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode init video frame");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
               sd_ctx->sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B") {
        LOG_INFO("VACE");
        int64_t t1 = ggml_time_ms();
        sd::Tensor<float> ref_image_latent;
        if (!start_image.empty()) {
            auto ref_img     = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
            auto encoded_ref = sd_ctx->sd->encode_first_stage(ref_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
            if (encoded_ref.empty()) {
                LOG_ERROR("failed to encode VACE reference image");
                return std::nullopt;
            }
            ref_image_latent = sd::ops::concat(encoded_ref, sd::zeros<float>(encoded_ref.shape()), 3);  // [b, 2*c, 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        sd::Tensor<float> control_video = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        int64_t control_frame_count     = std::min<int64_t>(request->frames, sd_vid_gen_params->control_frames_size);
        for (int64_t i = 0; i < control_frame_count; ++i) {
            auto control_frame = sd_image_to_tensor(sd_vid_gen_params->control_frames[i], request->width, request->height);
            sd::ops::slice_assign(&control_video, 2, i, i + 1, control_frame.unsqueeze(2));
        }

        sd::Tensor<float> mask = sd::full<float>({request->width, request->height, request->frames, 1, 1}, 1.0f);

        control_video              = control_video - 0.5f;
        sd::Tensor<float> inactive = control_video * (1.0f - mask) + 0.5f;
        sd::Tensor<float> reactive = control_video * mask + 0.5f;

        inactive = sd_ctx->sd->encode_first_stage(inactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (inactive.empty()) {
            LOG_ERROR("failed to encode VACE inactive context");
            return std::nullopt;
        }

        reactive = sd_ctx->sd->encode_first_stage(reactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (reactive.empty()) {
            LOG_ERROR("failed to encode VACE reactive context");
            return std::nullopt;
        }

        int64_t length = inactive.shape()[2];
        if (!ref_image_latent.empty()) {
            length += 1;
            request->frames       = static_cast<int>((length - 1) * 4 + 1);
            latents.ref_image_num = 1;
        }
        auto vace_context = sd::ops::concat(inactive, reactive, 3);  // [b, 2*c, t, h/vae_scale_factor, w/vae_scale_factor]

        mask              = sd::full<float>({request->width, request->height, inactive.shape()[2], 1, 1}, 1.0f);
        auto mask_context = mask.reshape({request->vae_scale_factor,
                                          inactive.shape()[0],
                                          request->vae_scale_factor,
                                          inactive.shape()[1],
                                          inactive.shape()[2]});   // [t, h/vae_scale_factor, vae_scale_factor, w/vae_scale_factor, vae_scale_factor]
        mask_context      = mask_context.permute({1, 3, 4, 0, 2})  // [vae_scale_factor, vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]
                           .reshape({inactive.shape()[0],
                                     inactive.shape()[1],
                                     inactive.shape()[2],
                                     request->vae_scale_factor * request->vae_scale_factor});  // [vae_scale_factor*vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]

        if (!ref_image_latent.empty()) {
            vace_context  = sd::ops::concat(ref_image_latent, vace_context, 2);  // [b, 2*c, t+1, h/vae_scale_factor, w/vae_scale_factor]
            auto mask_pad = sd::zeros<float>({mask_context.shape()[0],
                                              mask_context.shape()[1],
                                              1,
                                              mask_context.shape()[3]});  // [vae_scale_factor*vae_scale_factor, 1, h/vae_scale_factor, w/vae_scale_factor]
            mask_context  = sd::ops::concat(mask_pad, mask_context, 2);   // [vae_scale_factor*vae_scale_factor, t + 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        mask_context.unsqueeze_(mask_context.dim());  // [b, vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]

        latents.vace_context = sd::ops::concat(vace_context, mask_context, 3);  // [b, 2*c + vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]
        int64_t t2           = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    }

    if (latents.init_latent.empty()) {
        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
    }

    if (sd_version_is_ltxav(sd_ctx->sd->version) && !latents.audio_latent.empty()) {
        if (!latents.denoise_mask.empty()) {
            latents.denoise_mask = pack_ltxav_audio_and_video_denoise_mask(latents.denoise_mask,
                                                                           latents.init_latent,
                                                                           latents.audio_latent);
        }
        latents.init_latent = pack_ltxav_audio_and_video_latents(latents.init_latent, latents.audio_latent);
    }

    return latents;
}

static ImageGenerationEmbeds prepare_video_generation_embeds(sd_ctx_t* sd_ctx,
                                                             const sd_vid_gen_params_t* sd_vid_gen_params,
                                                             const GenerationRequest& request,
                                                             const ImageGenerationLatents& latents) {
    ImageGenerationEmbeds embeds;
    ConditionerParams condition_params;
    condition_params.clip_skip       = request.clip_skip;
    condition_params.text            = request.prompt;
    condition_params.zero_out_masked = true;

    int64_t prepare_start_ms = ggml_time_ms();
    embeds.cond              = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                   condition_params);
    embeds.cond.c_concat     = latents.concat_latent;
    embeds.cond.c_vector     = latents.clip_vision_output;
    if (request.use_uncond) {
        condition_params.text  = request.negative_prompt;
        embeds.uncond          = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                     condition_params);
        embeds.uncond.c_concat = latents.concat_latent;
        embeds.uncond.c_vector = latents.clip_vision_output;
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }
    return embeds;
}

static sd_image_t* decode_video_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const sd::Tensor<float>& final_latent,
                                        int* num_frames_out) {
    if (final_latent.empty()) {
        LOG_ERROR("no latent video to decode");
        return nullptr;
    }
    sd::Tensor<float> video_latent = final_latent;
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        video_latent.shape()[3] > sd_ctx->sd->get_latent_channel()) {
        video_latent = sd::ops::slice(video_latent, 3, 0, sd_ctx->sd->get_latent_channel());
    }
    LOG_DEBUG("decode_video_outputs latent %dx%dx%dx%d",
              (int)video_latent.shape()[0],
              (int)video_latent.shape()[1],
              (int)video_latent.shape()[2],
              (int)video_latent.shape()[3]);
    // auto z = sd::load_tensor_from_file_as_tensor<float>("ltx_vae_z.bin");
    int64_t t4            = ggml_time_ms();
    sd::Tensor<float> vid = sd_ctx->sd->decode_first_stage(video_latent, true);
    int64_t t5            = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t5 - t4) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (vid.empty()) {
        LOG_ERROR("decode_first_stage failed for video");
        return nullptr;
    }
    LOG_DEBUG("decode_video_outputs decoded %dx%dx%dx%d",
              (int)vid.shape()[0],
              (int)vid.shape()[1],
              (int)vid.shape()[2],
              (int)vid.shape()[3]);
    if (request.frames > 0 &&
        vid.shape()[2] > request.frames) {
        vid = sd::ops::slice(vid, 2, 0, request.frames);
    }

    sd_image_t* result_images = (sd_image_t*)calloc(vid.shape()[2], sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = static_cast<int>(vid.shape()[2]);
    }

    for (int64_t i = 0; i < vid.shape()[2]; i++) {
        result_images[i] = tensor_to_sd_image(vid, static_cast<int>(i));
    }

    return result_images;
}

static sd::Tensor<float> upscale_ltx_spatial_video_latent(sd_ctx_t* sd_ctx,
                                                          const char* model_path,
                                                          const sd::Tensor<float>& packed_latent,
                                                          int audio_length) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || packed_latent.empty()) {
        return {};
    }
    if (strlen(SAFE_STR(model_path)) == 0) {
        LOG_ERROR("LTX latent spatial upscale requires a model path");
        return {};
    }
    if (!sd_ctx->sd->ensure_backend_pair(SDBackendModule::UPSCALER)) {
        return {};
    }

    int latent_channels            = sd_ctx->sd->get_latent_channel();
    sd::Tensor<float> video_latent = packed_latent;
    sd::Tensor<float> audio_latent;
    if (packed_latent.shape()[3] > latent_channels) {
        video_latent = sd::ops::slice(packed_latent, 3, 0, latent_channels);
        audio_latent = unpack_ltxav_audio_latent(packed_latent, audio_length, latent_channels);
    }

    LOG_INFO("LTX latent spatial upscale: latent %dx%dx%dx%d -> model output",
             (int)video_latent.shape()[0],
             (int)video_latent.shape()[1],
             (int)video_latent.shape()[2],
             (int)video_latent.shape()[3]);

    sd::Tensor<float> unnormalized = sd_ctx->sd->un_normalize_ltx_video_latents(video_latent);
    if (unnormalized.empty()) {
        LOG_ERROR("LTX latent un-normalization failed before spatial upscale");
        return {};
    }

    std::unique_ptr<LTXVUpsampler::LatentUpsamplerRunner> upsampler =
        std::make_unique<LTXVUpsampler::LatentUpsamplerRunner>(sd_ctx->sd->backend_for(SDBackendModule::UPSCALER),
                                                               sd_ctx->sd->params_backend_for(SDBackendModule::UPSCALER));
    const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(sd_ctx->sd->max_vram);
    upsampler->set_max_graph_vram_bytes(max_graph_vram_bytes);
    if (!upsampler->load_from_file(model_path, sd_ctx->sd->n_threads)) {
        LOG_ERROR("load LTX latent upsampler failed");
        return {};
    }

    sd::Tensor<float> upscaled = upsampler->compute(sd_ctx->sd->n_threads, unnormalized);
    upsampler.reset();
    if (upscaled.empty()) {
        LOG_ERROR("LTX latent spatial upscale failed");
        return {};
    }

    upscaled = sd_ctx->sd->normalize_ltx_video_latents(upscaled);
    if (upscaled.empty()) {
        LOG_ERROR("LTX latent normalization failed after spatial upscale");
        return {};
    }

    if (!audio_latent.empty()) {
        upscaled = pack_ltxav_audio_and_video_latents(upscaled, audio_latent);
    }
    return upscaled;
}

static bool apply_ltxv_refine_image_conditioning(sd_ctx_t* sd_ctx,
                                                 const sd_vid_gen_params_t* sd_vid_gen_params,
                                                 const GenerationRequest& request,
                                                 const ImageGenerationLatents& latents,
                                                 sd::Tensor<float>* latent,
                                                 sd::Tensor<float>* denoise_mask,
                                                 sd::Tensor<float>* video_positions) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_vid_gen_params == nullptr ||
        latent == nullptr || latent->empty() || denoise_mask == nullptr || video_positions == nullptr) {
        return true;
    }
    if (sd_vid_gen_params->init_image.data == nullptr &&
        sd_vid_gen_params->end_image.data == nullptr) {
        return true;
    }
    if (sd_ctx->sd->vae_decode_only) {
        LOG_ERROR("LTXV refine image conditioning requires VAE encoder weights; create the context with vae_decode_only=false");
        return false;
    }

    constexpr float conditioning_strength = 1.f;
    int latent_channels                   = sd_ctx->sd->get_latent_channel();
    sd::Tensor<float> video_latent        = *latent;
    sd::Tensor<float> audio_latent;
    if (latent->shape()[3] > latent_channels) {
        video_latent = sd::ops::slice(*latent, 3, 0, latent_channels);
        audio_latent = unpack_ltxav_audio_latent(*latent, latents.audio_length, latent_channels);
        if (audio_latent.empty()) {
            LOG_ERROR("failed to unpack LTXAV audio latent before image-to-video inplace conditioning");
            return false;
        }
    }

    int image_width              = static_cast<int>(video_latent.shape()[0]) * request.vae_scale_factor;
    int image_height             = static_cast<int>(video_latent.shape()[1]) * request.vae_scale_factor;
    sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video_latent, 1.f);

    if (sd_vid_gen_params->init_image.data != nullptr) {
        sd::Tensor<float> start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, image_width, image_height);
        if (!apply_ltxav_condition_image_by_latent_index(sd_ctx,
                                                         start_image,
                                                         &video_latent,
                                                         &video_mask,
                                                         0,
                                                         "init",
                                                         conditioning_strength)) {
            return false;
        }
    }

    if (sd_vid_gen_params->end_image.data != nullptr) {
        sd::Tensor<float> end_image        = sd_image_to_tensor(sd_vid_gen_params->end_image, image_width, image_height);
        sd::Tensor<float> end_image_latent = encode_ltxav_condition_image(sd_ctx, end_image, "end");
        if (end_image_latent.empty()) {
            return false;
        }

        int frame_idx = request.frames - 1;
        if (frame_idx == 0) {
            if (!apply_ltxav_condition_by_latent_index(&video_latent,
                                                       &video_mask,
                                                       end_image_latent,
                                                       0,
                                                       "end",
                                                       1.f - conditioning_strength)) {
                return false;
            }
        } else {
            if (latents.video_conditioning_frame_count <= 0 || latents.video_target_frame_count <= 0) {
                LOG_ERROR("LTXV FLF2V refine conditioning requires low-resolution keyframe conditioning metadata");
                return false;
            }
            int64_t target_latent_frames = latents.video_target_frame_count;
            if (!apply_ltxav_condition_by_latent_index(&video_latent,
                                                       &video_mask,
                                                       end_image_latent,
                                                       target_latent_frames,
                                                       "end",
                                                       1.f - conditioning_strength)) {
                return false;
            }
            *video_positions = build_ltxv_video_positions(video_latent.shape()[0],
                                                          video_latent.shape()[1],
                                                          target_latent_frames,
                                                          end_image_latent.shape()[2],
                                                          frame_idx,
                                                          1,
                                                          request.fps,
                                                          request.vae_scale_factor,
                                                          8,
                                                          true);
        }
    }

    if (!audio_latent.empty()) {
        *latent       = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
        *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent);
    } else {
        *latent       = std::move(video_latent);
        *denoise_mask = std::move(video_mask);
    }
    LOG_INFO("LTXV refine image conditioning applied at %dx%d", image_width, image_height);
    return true;
}

SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out) {
    if (sd_ctx == nullptr || sd_vid_gen_params == nullptr) {
        return false;
    }
    if (frames_out != nullptr) {
        *frames_out = nullptr;
    }
    if (audio_out != nullptr) {
        *audio_out = nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = 0;
    }
    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_vid_gen_params);
    bool latent_upscale_enabled     = request.hires.enabled;
    GenerationRequest hires_request = request;
    if (latent_upscale_enabled) {
        if (!sd_version_is_ltxav(sd_ctx->sd->version)) {
            LOG_ERROR("LTX latent spatial upscale is only supported for LTX video models");
            return false;
        }
        if (request.hires.upscaler != SD_HIRES_UPSCALER_MODEL) {
            LOG_ERROR("LTX latent spatial upscale currently requires hires upscaler MODEL");
            return false;
        }
        if (strlen(SAFE_STR(request.hires.model_path)) == 0) {
            LOG_ERROR("LTX latent spatial upscale is enabled but hires model path was not provided");
            return false;
        }
    }

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_vid_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count);

    SamplePlan plan(sd_ctx, sd_vid_gen_params, request);
    auto latent_inputs_opt = prepare_video_generation_latents(sd_ctx, sd_vid_gen_params, &request);
    if (!latent_inputs_opt.has_value()) {
        return false;
    }
    ImageGenerationLatents latents = std::move(*latent_inputs_opt);

    ImageGenerationEmbeds embeds = prepare_video_generation_embeds(sd_ctx,
                                                                   sd_vid_gen_params,
                                                                   request,
                                                                   latents);
    if (latent_upscale_enabled) {
        LOG_INFO("generate_video %dx%dx%d -> LTX latent spatial upscale",
                 request.width,
                 request.height,
                 request.frames);
    } else {
        LOG_INFO("generate_video %dx%dx%d",
                 request.width,
                 request.height,
                 request.frames);
    }

    int64_t latent_start = ggml_time_ms();
    int W                = request.width / request.vae_scale_factor;
    int H                = request.height / request.vae_scale_factor;
    int T                = static_cast<int>(latents.init_latent.shape()[2]);

    sd::Tensor<float> x_t   = latents.init_latent;
    sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);

    if (plan.high_noise_sample_steps > 0) {
        LOG_DEBUG("sample(high noise) %dx%dx%d", W, H, T);

        int64_t sampling_start = ggml_time_ms();
        std::vector<float> high_noise_sigmas(plan.sigmas.begin(), plan.sigmas.begin() + plan.high_noise_sample_steps + 1);
        plan.sigmas = std::vector<float>(plan.sigmas.begin() + plan.high_noise_sample_steps, plan.sigmas.end());

        sd::Tensor<float> x_t_sampled = sd_ctx->sd->sample(sd_ctx->sd->high_noise_diffusion_model,
                                                           false,
                                                           x_t,
                                                           std::move(noise),
                                                           embeds.cond,
                                                           request.use_high_noise_uncond ? embeds.uncond : SDCondition(),
                                                           embeds.img_cond,
                                                           embeds.id_cond,
                                                           sd::Tensor<float>(),
                                                           0.f,
                                                           request.high_noise_guidance,
                                                           plan.high_noise_eta,
                                                           request.shifted_timestep,
                                                           plan.high_noise_sample_method,
                                                           sd_ctx->sd->is_flow_denoiser(),
                                                           plan.high_noise_extra_sample_args,
                                                           high_noise_sigmas,
                                                           -1,
                                                           std::vector<sd::Tensor<float>>{},
                                                           false,
                                                           latents.denoise_mask,
                                                           latents.vace_context,
                                                           request.vace_strength,
                                                           latents.audio_length,
                                                           static_cast<float>(request.fps),
                                                           request.cache_params,
                                                           latents.video_positions);
        int64_t sampling_end          = ggml_time_ms();
        if (x_t_sampled.empty()) {
            LOG_ERROR("sampling(high noise) failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
            }
            return false;
        }

        x_t   = std::move(x_t_sampled);
        noise = {};
        LOG_INFO("sampling(high noise) completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
        }
    }

    LOG_DEBUG("sample %dx%dx%d", W, H, T);
    int64_t sampling_start         = ggml_time_ms();
    sd::Tensor<float> final_latent = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                        true,
                                                        x_t,
                                                        std::move(noise),
                                                        embeds.cond,
                                                        request.use_uncond ? embeds.uncond : SDCondition(),
                                                        embeds.img_cond,
                                                        embeds.id_cond,
                                                        sd::Tensor<float>(),
                                                        0.f,
                                                        sd_vid_gen_params->sample_params.guidance,
                                                        plan.eta,
                                                        sd_vid_gen_params->sample_params.shifted_timestep,
                                                        plan.sample_method,
                                                        sd_ctx->sd->is_flow_denoiser(),
                                                        plan.extra_sample_args,
                                                        plan.sigmas,
                                                        -1,
                                                        std::vector<sd::Tensor<float>>{},
                                                        false,
                                                        latents.denoise_mask,
                                                        latents.vace_context,
                                                        request.vace_strength,
                                                        latents.audio_length,
                                                        static_cast<float>(request.fps),
                                                        request.cache_params,
                                                        latents.video_positions);

    int64_t sampling_end = ggml_time_ms();
    if (final_latent.empty()) {
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        LOG_ERROR("sampling failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        return false;
    }
    LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);

    if (latent_upscale_enabled) {
        int64_t upscale_start             = ggml_time_ms();
        sd::Tensor<float> upscaled_latent = upscale_ltx_spatial_video_latent(sd_ctx,
                                                                             request.hires.model_path,
                                                                             final_latent,
                                                                             latents.audio_length);
        int64_t upscale_end               = ggml_time_ms();
        if (upscaled_latent.empty()) {
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return false;
        }
        LOG_INFO("LTX latent spatial upscale completed, taking %.2fs",
                 (upscale_end - upscale_start) * 1.0f / 1000);

        x_t                        = std::move(upscaled_latent);
        hires_request.width        = static_cast<int>(x_t.shape()[0]) * hires_request.vae_scale_factor;
        hires_request.height       = static_cast<int>(x_t.shape()[1]) * hires_request.vae_scale_factor;
        int upscaled_latent_frames = static_cast<int>(x_t.shape()[2]);
        int upscaled_frames        = sd_ctx->sd->latent_frames_to_video_frames(upscaled_latent_frames);
        if (upscaled_frames != hires_request.frames) {
            LOG_INFO("LTX latent upsampler output latent frames %d, frames %d -> %d",
                     upscaled_latent_frames,
                     hires_request.frames,
                     upscaled_frames);
            hires_request.frames = upscaled_frames;
        }
        if (sd_version_is_ltxav(sd_ctx->sd->version) && latents.audio_length > 0) {
            int target_audio_length = get_ltxav_num_audio_latents(hires_request.frames, hires_request.fps);
            if (target_audio_length != latents.audio_length) {
                int latent_channels            = sd_ctx->sd->get_latent_channel();
                sd::Tensor<float> video_latent = x_t;
                sd::Tensor<float> audio_latent = latents.audio_latent;
                if (x_t.shape()[3] > latent_channels) {
                    video_latent = sd::ops::slice(x_t, 3, 0, latent_channels);
                    audio_latent = unpack_ltxav_audio_latent(x_t, latents.audio_length, latent_channels);
                }
                audio_latent = resize_ltxav_audio_latent(audio_latent, target_audio_length);
                if (audio_latent.empty()) {
                    LOG_ERROR("failed to resize LTX audio latent for latent upscale: %d -> %d",
                              latents.audio_length,
                              target_audio_length);
                    if (sd_ctx->sd->free_params_immediately) {
                        sd_ctx->sd->diffusion_model->free_params_buffer();
                    }
                    return false;
                }
                x_t                  = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
                latents.audio_latent = std::move(audio_latent);
                LOG_INFO("LTX audio latent length adjusted for latent upscale: %d -> %d",
                         latents.audio_length,
                         target_audio_length);
                latents.audio_length = target_audio_length;
            }
        }
        if ((request.hires.target_width > 0 || request.hires.target_height > 0) &&
            (request.hires.target_width != hires_request.width || request.hires.target_height != hires_request.height)) {
            LOG_WARN("LTX latent spatial upsampler output is %dx%d; ignoring hires target %dx%d",
                     hires_request.width,
                     hires_request.height,
                     request.hires.target_width,
                     request.hires.target_height);
        }
        sd::Tensor<float> hires_denoise_mask;
        sd::Tensor<float> hires_video_positions;
        if (!apply_ltxv_refine_image_conditioning(sd_ctx,
                                                  sd_vid_gen_params,
                                                  hires_request,
                                                  latents,
                                                  &x_t,
                                                  &hires_denoise_mask,
                                                  &hires_video_positions)) {
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return false;
        }
        noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);

        W                                   = hires_request.width / hires_request.vae_scale_factor;
        H                                   = hires_request.height / hires_request.vae_scale_factor;
        T                                   = static_cast<int>(x_t.shape()[2]);
        sample_method_t hires_sample_method = plan.sample_method;
        int hires_scheduler_steps           = 0;
        std::vector<float> hires_sigma_sched =
            make_hires_sigma_schedule(sd_ctx,
                                      request.hires,
                                      sd_vid_gen_params->sample_params,
                                      hires_sample_method,
                                      plan.sample_steps,
                                      sd_ctx->sd->get_image_seq_len(hires_request.height, hires_request.width) * T,
                                      &hires_scheduler_steps);
        float hires_eta = resolve_eta(sd_ctx,
                                      sd_vid_gen_params->sample_params.eta,
                                      hires_sample_method);

        LOG_DEBUG("sample(latent upscale) %dx%dx%d", W, H, T);
        LOG_INFO("LTX latent spatial upscale refine: scheduler_steps=%d, denoising_strength=%.2f, sampler=%s, sigma_sched_size=%zu%s",
                 hires_scheduler_steps,
                 request.hires.denoising_strength,
                 sampling_methods_str[hires_sample_method],
                 hires_sigma_sched.size(),
                 request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");

        sampling_start = ggml_time_ms();
        final_latent   = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                            true,
                                            x_t,
                                            std::move(noise),
                                            embeds.cond,
                                          hires_request.use_uncond ? embeds.uncond : SDCondition(),
                                            embeds.img_cond,
                                            embeds.id_cond,
                                            sd::Tensor<float>(),
                                            0.f,
                                            sd_vid_gen_params->sample_params.guidance,
                                            hires_eta,
                                            sd_vid_gen_params->sample_params.shifted_timestep,
                                            hires_sample_method,
                                            sd_ctx->sd->is_flow_denoiser(),
                                            plan.extra_sample_args,
                                            hires_sigma_sched,
                                            -1,
                                            std::vector<sd::Tensor<float>>{},
                                            false,
                                            hires_denoise_mask,
                                            sd::Tensor<float>(),
                                            hires_request.vace_strength,
                                            latents.audio_length,
                                            static_cast<float>(hires_request.fps),
                                            hires_request.cache_params,
                                            hires_video_positions);
        sampling_end   = ggml_time_ms();
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        if (final_latent.empty()) {
            LOG_ERROR("sampling(latent upscale) failed after %.2fs",
                      (sampling_end - sampling_start) * 1.0f / 1000);
            return false;
        }
        LOG_INFO("sampling(latent upscale) completed, taking %.2fs",
                 (sampling_end - sampling_start) * 1.0f / 1000);
    } else if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }

    int64_t latent_end = ggml_time_ms();
    LOG_INFO("generating latent video completed, taking %.2fs", (latent_end - latent_start) * 1.0f / 1000);

    sd_audio_t* generated_audio = nullptr;
    if (sd_version_is_ltxav(sd_ctx->sd->version) &&
        latents.audio_length > 0 &&
        sd_ctx->sd->audio_vae_model != nullptr) {
        int64_t audio_latent_decode_start = ggml_time_ms();

        auto audio_latent = unpack_ltxav_audio_latent(final_latent,
                                                      latents.audio_length,
                                                      sd_ctx->sd->get_latent_channel());
        if (!audio_latent.empty()) {
            LOG_DEBUG("decode audio latent %dx%dx%dx%d",
                      (int)audio_latent.shape()[0],
                      (int)audio_latent.shape()[1],
                      (int)audio_latent.shape()[2],
                      (int)audio_latent.shape()[3]);
            auto waveform = sd_ctx->sd->decode_ltx_audio_latent(audio_latent);
            if (!waveform.empty()) {
                generated_audio = waveform_to_sd_audio(sd_ctx->sd, waveform);
            } else {
                LOG_WARN("LTX audio latent decode failed; continuing with silent video output");
            }
        }
        int64_t audio_latent_decode_end = ggml_time_ms();
        LOG_INFO("decoding audio latent completed, taking %.2fs", (audio_latent_decode_end - audio_latent_decode_start) * 1.0f / 1000);
    }

    if (latents.video_conditioning_frame_count > 0) {
        int64_t target_frames = latents.video_target_frame_count > 0 ? latents.video_target_frame_count
                                                                     : final_latent.shape()[2] - latents.video_conditioning_frame_count;
        final_latent          = sd::ops::slice(final_latent, 2, 0, target_frames);
    }

    if (latents.ref_image_num > 0) {
        final_latent = sd::ops::slice(final_latent, 2, latents.ref_image_num, final_latent.shape()[2]);
    }

    auto result = decode_video_outputs(sd_ctx, latent_upscale_enabled ? hires_request : request, final_latent, num_frames_out);
    if (result == nullptr) {
        free_sd_audio(generated_audio);
        return false;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_video completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    if (frames_out != nullptr) {
        *frames_out = result;
    }
    if (audio_out != nullptr) {
        *audio_out = generated_audio;
    } else {
        free_sd_audio(generated_audio);
    }
    return true;
}


#include "kcpp_sd_extensions.h"

namespace kcpp_sd {

    static_assert((int)SD_TYPE_COUNT == (int)GGML_TYPE_COUNT,
            "inconsistency between SD_TYPE_COUNT and GGML_TYPE_COUNT");

    int get_loaded_sd_version(sd_ctx_t* ctx) {
        return ctx->sd->version;
    }

    bool loaded_model_is_chroma(sd_ctx_t* ctx) {
        if (ctx != nullptr && ctx->sd != nullptr) {
            auto maybe_flux = std::dynamic_pointer_cast<Flux::FluxRunner>(ctx->sd->diffusion_model);
            if (maybe_flux != nullptr) {
                return maybe_flux->flux_params.is_chroma;
            }
        }
        return false;
    }

    int get_spatial_multiple(sd_ctx_t* ctx) {
        return ctx->sd->get_vae_scale_factor() * ctx->sd->get_diffusion_model_down_factor();
    }

    model_info get_model_info(sd_ctx_t* ctx)
    {
        model_info res = {};
        auto loadedsdver = get_loaded_sd_version(ctx);
        res.is_wan = (loadedsdver == SDVersion::VERSION_WAN2 || loadedsdver == SDVersion::VERSION_WAN2_2_I2V || loadedsdver == SDVersion::VERSION_WAN2_2_TI2V);
        res.is_qwenimg = (loadedsdver == SDVersion::VERSION_QWEN_IMAGE);
        res.is_chroma = loaded_model_is_chroma(ctx);
        res.is_kontext = (loadedsdver==SDVersion::VERSION_FLUX && !res.is_chroma);
        res.is_flux2 = (loadedsdver == SDVersion::VERSION_FLUX2 || loadedsdver == SDVersion::VERSION_FLUX2_KLEIN);
        res.is_flux1 = (loadedsdver == SDVersion::VERSION_FLUX);
        res.is_zimage = (loadedsdver == SDVersion::VERSION_Z_IMAGE);
        res.is_sdxs = (loadedsdver == SDVersion::VERSION_SDXS_512_DS || loadedsdver == SDVersion::VERSION_SDXS_09);
        res.is_sd1 = (loadedsdver == SDVersion::VERSION_SD1);
        res.is_sd2 = (loadedsdver == SDVersion::VERSION_SD2);
        res.is_sdxl = sd_version_is_sdxl((SDVersion)loadedsdver);
        res.vae_scale_factor = ctx->sd->get_vae_scale_factor();
        res.spatial_multiple = get_spatial_multiple(ctx);
        return res;
    }

    void SetCircularAxesAll(sd_ctx_t* ctx, bool circular_x, bool circular_y) {
        ctx->sd->SetCircularAxesAll(circular_x, circular_y);
    }

    void set_lora_cache(sd_ctx_t *ctx, bool enable) {
        ctx->sd->kcpp_lora_cache_populate = enable;
    }

    void apply_loras(sd_ctx_t *ctx, const std::vector<sd_lora_t>& lora_specs)
    {
        ctx->sd->apply_loras(lora_specs.data(), lora_specs.size());
    }

}

