#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "tae.hpp"
#include "vae.hpp"

// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_STATIC
#include "stb_image.h"

#include <inttypes.h>
#include <cinttypes>

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_STATIC
// #include "stb_image_write.h"

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 1.x Inpaint",
    "Instruct-Pix2Pix",
    "SD 2.x",
    "SD 2.x Inpaint",
    "SDXL",
    "SDXL Inpaint",
    "SDXL Instruct-Pix2Pix",
    "SVD",
    "SD3.x",
    "Flux",
    "Flux Fill"};

const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD"};

/*================================================== Helper Functions ================================================*/

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120,
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

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = NULL;  // general backend
    ggml_backend_t clip_backend        = NULL;
    ggml_backend_t control_net_backend = NULL;
    ggml_backend_t vae_backend         = NULL;
    ggml_type model_wtype              = GGML_TYPE_COUNT;
    ggml_type conditioner_wtype        = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype    = GGML_TYPE_COUNT;
    ggml_type vae_wtype                = GGML_TYPE_COUNT;

    SDVersion version;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<AutoEncoderKL> first_stage_model;
    std::shared_ptr<TinyAutoEncoder> tae_first_stage;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;
    std::shared_ptr<PhotoMakerIDEmbed> pmid_id_embeds;

    std::string taesd_path;
    bool use_tiny_autoencoder = false;
    bool vae_tiling           = false;
    bool stacked_id           = false;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::string lora_model_dir;
    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() {
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);
    }

    void init_backend() {
#ifdef SD_USE_CUDA
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        ggml_log_set(ggml_log_callback_default, nullptr);
        backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        for (int device = 0; device < ggml_backend_vk_get_device_count(); ++device) {
            backend = ggml_backend_vk_init(device);
        }
        if (!backend) {
            LOG_WARN("Failed to initialize Vulkan backend");
        }
#endif
#ifdef SD_USE_OPENCL
        LOG_DEBUG("Using OpenCL backend");
        // ggml_log_set(ggml_log_callback_default, nullptr); // Optional ggml logs
        backend = ggml_backend_opencl_init();
        if (!backend) {
            LOG_WARN("Failed to initialize OpenCL backend");
        }
#endif
#ifdef SD_USE_SYCL
        LOG_DEBUG("Using SYCL backend");
        backend = ggml_backend_sycl_init(0);
#endif

        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }
    }

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        lora_model_dir          = SAFE_STR(sd_ctx_params->lora_model_dir);
        taesd_path              = SAFE_STR(sd_ctx_params->taesd_path);
        use_tiny_autoencoder    = taesd_path.size() > 0;
        vae_tiling              = sd_ctx_params->vae_tiling;

        if (sd_ctx_params->rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (sd_ctx_params->rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }

        init_backend();

        std::string taesd_path_fixed = taesd_path;

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

        bool is_unet = model_loader.model_is_unet();

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

        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
            LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
            }
        }

        version = model_loader.get_sd_version();

        // kcpp fallback to separate diffusion model passed as model
        if (version == VERSION_COUNT &&
            strlen(SAFE_STR(sd_ctx_params->model_path)) > 0 &&
            strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) == 0 &&
            strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0 )
        {
            bool endswithsafetensors = ends_with(sd_ctx_params->model_path, ".safetensors");
            if(endswithsafetensors && !model_loader.has_diffusion_model_tensors())
            {
                LOG_INFO("SD Diffusion Model tensors missing! Fallback trying alternative tensor names...\n");
                if (!model_loader.init_from_file(sd_ctx_params->model_path, "model.diffusion_model.")) {
                    LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->model_path);
                }
                version = model_loader.get_sd_version();
            }
        }

        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
            return false;
        }

        LOG_INFO("Version: %s ", model_version_to_str[version]);

        if(use_tiny_autoencoder)
        {
            std::string to_search = "taesd.embd";
            std::string to_replace = "";
            if(version==VERSION_SDXL)
            {
                to_replace = "taesd_xl.embd";
            }
            else if(version==VERSION_FLUX)
            {
                to_replace = "taesd_f.embd";
            }
            else if(version==VERSION_SD3)
            {
                to_replace = "taesd_3.embd";
            }

            if(to_replace!="")
            {
                size_t pos = taesd_path_fixed.find(to_search);
                if (pos != std::string::npos) {
                    taesd_path_fixed.replace(pos, to_search.length(), to_replace);
                }
            }
        }

        ggml_type wtype = (ggml_type)sd_ctx_params->wtype;
        if (wtype == GGML_TYPE_COUNT) {
            model_wtype = model_loader.get_sd_wtype();
            if (model_wtype == GGML_TYPE_COUNT) {
                model_wtype = GGML_TYPE_F32;
                LOG_WARN("can not get mode wtype frome weight, use f32");
            }
            conditioner_wtype = model_loader.get_conditioner_wtype();
            if (conditioner_wtype == GGML_TYPE_COUNT) {
                conditioner_wtype = wtype;
            }
            diffusion_model_wtype = model_loader.get_diffusion_model_wtype();
            if (diffusion_model_wtype == GGML_TYPE_COUNT) {
                diffusion_model_wtype = wtype;
            }
            vae_wtype = model_loader.get_vae_wtype();

            if (vae_wtype == GGML_TYPE_COUNT) {
                vae_wtype = wtype;
            }
        } else {
            model_wtype           = wtype;
            conditioner_wtype     = wtype;
            diffusion_model_wtype = wtype;
            vae_wtype             = wtype;
            model_loader.set_wtype_override(wtype);
        }

        if (sd_version_is_sdxl(version)) {
            vae_wtype = GGML_TYPE_F32;
            model_loader.set_wtype_override(GGML_TYPE_F32, "vae.");
        }

        LOG_INFO("Weight type:                 %s", model_wtype != GGML_TYPE_COUNT ? ggml_type_name(model_wtype) : "??");
        LOG_INFO("Conditioner weight type:     %s", conditioner_wtype != GGML_TYPE_COUNT ? ggml_type_name(conditioner_wtype) : "??");
        LOG_INFO("Diffusion model weight type: %s", diffusion_model_wtype != GGML_TYPE_COUNT ? ggml_type_name(diffusion_model_wtype) : "??");
        LOG_INFO("VAE weight type:             %s", vae_wtype != GGML_TYPE_COUNT ? ggml_type_name(vae_wtype) : "??");

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (sd_version_is_sdxl(version)) {
            scale_factor = 0.13025f;
            if (strlen(SAFE_STR(sd_ctx_params->vae_path)) == 0 && taesd_path_fixed.size() == 0) {
                LOG_WARN(
                    "!!!It looks like you are using SDXL model. "
                    "If you find that the generated images are completely black, "
                    "try specifying a different VAE. "
                    "You can find it here: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors");
            }
        } else if (sd_version_is_sd3(version)) {
            scale_factor = 1.5305f;
        } else if (sd_version_is_flux(version)) {
            scale_factor = 0.3611;
            // TODO: shift_factor
        }

        bool clip_on_cpu = sd_ctx_params->keep_clip_on_cpu;

        if (version == VERSION_SVD) {
            clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, model_loader.tensor_storages_types);
            clip_vision->alloc_params_buffer();
            clip_vision->get_param_tensors(tensors);

            diffusion_model = std::make_shared<UNetModel>(backend, model_loader.tensor_storages_types, version);
            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            first_stage_model = std::make_shared<AutoEncoderKL>(backend, model_loader.tensor_storages_types, "first_stage_model", vae_decode_only, true, version);
            LOG_DEBUG("vae_decode_only %d", vae_decode_only);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");
        } else {
            clip_backend   = backend;
            bool use_t5xxl = false;
            if (sd_version_is_dit(version)) {
                use_t5xxl = true;
            }
            if (!ggml_backend_is_cpu(backend) && use_t5xxl && conditioner_wtype != GGML_TYPE_F32) {
                clip_on_cpu = true;
                LOG_INFO("set clip_on_cpu to true");
            }
            if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("CLIP: Using CPU backend");
                clip_backend = ggml_backend_cpu_init();
            }
            if (sd_ctx_params->diffusion_flash_attn) {
                LOG_INFO("Using flash attention in the diffusion model");
            }
            if (sd_version_is_sd3(version)) {
                if (sd_ctx_params->diffusion_flash_attn) {
                    LOG_WARN("flash attention in this diffusion model is currently unsupported!");
                }
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backend, model_loader.tensor_storages_types);
                diffusion_model  = std::make_shared<MMDiTModel>(backend, model_loader.tensor_storages_types);
            } else if (sd_version_is_flux(version)) {
                bool is_chroma = false;
                for (auto pair : model_loader.tensor_storages_types) {
                    if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                        is_chroma = true;
                        break;
                    }
                }
                if (is_chroma) {
                    cond_stage_model = std::make_shared<PixArtCLIPEmbedder>(clip_backend,
                                                                            model_loader.tensor_storages_types,
                                                                            -1,
                                                                            sd_ctx_params->chroma_use_t5_mask,
                                                                            sd_ctx_params->chroma_t5_mask_pad);
                } else {
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(clip_backend, model_loader.tensor_storages_types);
                }
                diffusion_model = std::make_shared<FluxModel>(backend,
                                                              model_loader.tensor_storages_types,
                                                              version,
                                                              sd_ctx_params->diffusion_flash_attn,
                                                              sd_ctx_params->chroma_use_dit_mask);
            } else {
                if (strstr(SAFE_STR(sd_ctx_params->stacked_id_embed_dir), "v2")) {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                           model_loader.tensor_storages_types,
                                                                                           SAFE_STR(sd_ctx_params->embedding_dir),
                                                                                           version,
                                                                                           PM_VERSION_2);
                } else {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                           model_loader.tensor_storages_types,
                                                                                           SAFE_STR(sd_ctx_params->embedding_dir),
                                                                                           version);
                }
                diffusion_model = std::make_shared<UNetModel>(backend,
                                                              model_loader.tensor_storages_types,
                                                              version,
                                                              sd_ctx_params->diffusion_flash_attn);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the diffusion model");
                    std::dynamic_pointer_cast<UNetModel>(diffusion_model)->unet.enable_conv2d_direct();
                }
            }

            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors);

            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            if (!use_tiny_autoencoder) {
                if (sd_ctx_params->keep_vae_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_INFO("VAE Autoencoder: Using CPU backend");
                    vae_backend = ggml_backend_cpu_init();
                } else {
                    vae_backend = backend;
                }
                first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend,
                                                                    model_loader.tensor_storages_types,
                                                                    "first_stage_model",
                                                                    vae_decode_only,
                                                                    false,
                                                                    version);
                if (sd_ctx_params->vae_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the vae model");
                    first_stage_model->enable_conv2d_direct();
                }
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
            } else {
                tae_first_stage = std::make_shared<TinyAutoEncoder>(backend,
                                                                    model_loader.tensor_storages_types,
                                                                    "decoder.layers",
                                                                    vae_decode_only,
                                                                    version);
                if (sd_ctx_params->vae_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the tae model");
                    tae_first_stage->enable_conv2d_direct();
                }
            }
            // first_stage_model->get_param_tensors(tensors, "first_stage_model.");

            if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
                ggml_backend_t controlnet_backend = NULL;
                if (sd_ctx_params->keep_control_net_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_DEBUG("ControlNet: Using CPU backend");
                    controlnet_backend = ggml_backend_cpu_init();
                } else {
                    controlnet_backend = backend;
                }
                control_net = std::make_shared<ControlNet>(controlnet_backend, model_loader.tensor_storages_types, version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the control net");
                    control_net->enable_conv2d_direct();
                }
            }

            if (strstr(SAFE_STR(sd_ctx_params->stacked_id_embed_dir), "v2")) {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend, model_loader.tensor_storages_types, "pmid", version, PM_VERSION_2);
                LOG_INFO("using PhotoMaker Version 2");
            } else {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend, model_loader.tensor_storages_types, "pmid", version);
            }
            if (strlen(SAFE_STR(sd_ctx_params->stacked_id_embed_dir)) > 0) {
              if (version != VERSION_SDXL) {
                printf("\n!!!!\nWARNING: PhotoMaker is only compatible with SDXL models. PhotoMaker will be disabled!\n!!!!\n");
              } else {
                pmid_lora = std::make_shared<LoraModel>(backend, sd_ctx_params->stacked_id_embed_dir, "");
                if (!pmid_lora->load_from_file(true)) {
                    LOG_WARN("load photomaker lora tensors from %s failed", sd_ctx_params->stacked_id_embed_dir);
                    return false;
                }
                LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", sd_ctx_params->stacked_id_embed_dir);
                if (!model_loader.init_from_file(sd_ctx_params->stacked_id_embed_dir, "pmid.")) {
                    LOG_WARN("loading stacked ID embedding from '%s' failed", sd_ctx_params->stacked_id_embed_dir);
                } else {
                    stacked_id = true;
                }
              }
            }
            if (stacked_id) {
                if (!pmid_model->alloc_params_buffer()) {
                    LOG_ERROR(" pmid model params buffer allocation failed");
                    return false;
                }
                pmid_model->get_param_tensors(tensors, "pmid");
            }
        }

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        struct ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != NULL);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        int64_t t0 = ggml_time_ms();

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tiny_autoencoder) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (stacked_id) {
            ignore_tensors.insert("lora.");
        }

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.quant");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success = model_loader.load_tensors(tensors, backend, ignore_tensors);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        // LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);

        if (version == VERSION_SVD) {
            // diffusion_model->test();
            // first_stage_model->test();
            // return false;
        } else {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            size_t vae_params_mem_size  = 0;
            if (!use_tiny_autoencoder) {
                vae_params_mem_size = first_stage_model->get_params_buffer_size();
            } else {
                if (!tae_first_stage->load_from_file(taesd_path_fixed)) {
                    return false;
                }
                vae_params_mem_size = tae_first_stage->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(SAFE_STR(sd_ctx_params->control_net_path))) {
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (stacked_id) {
                pmid_params_mem_size = pmid_model->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(clip_backend)) {
                total_params_ram_size += clip_params_mem_size + pmid_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size + pmid_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += unet_params_mem_size;
            } else {
                total_params_vram_size += unet_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }

            if (ggml_backend_is_cpu(control_net_backend)) {
                total_params_ram_size += control_net_params_mem_size;
            } else {
                total_params_vram_size += control_net_params_mem_size;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "clip %.2fMB(%s), unet %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                control_net_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(control_net_backend) ? "RAM" : "VRAM",
                pmid_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM");
        }

        int64_t t1 = ggml_time_ms();
        LOG_INFO("loading model from '%s' completed, taking %.2fs", SAFE_STR(sd_ctx_params->model_path), (t1 - t0) * 1.0f / 1000);

        // check is_using_v_parameterization_for_sd2

        if (sd_version_is_sd2(version)) {
            if (is_using_v_parameterization_for_sd2(ctx, sd_version_is_inpaint(version))) {
                is_using_v_parameterization = true;
            }
        } else if (sd_version_is_sdxl(version)) {
            if (model_loader.tensor_storages_types.find("edm_vpred.sigma_max") != model_loader.tensor_storages_types.end()) {
                // CosXL models
                // TODO: get sigma_min and sigma_max values from file
                is_using_edm_v_parameterization = true;
            }
            if (model_loader.tensor_storages_types.find("v_pred") != model_loader.tensor_storages_types.end()) {
                is_using_v_parameterization = true;
            }
        } else if (version == VERSION_SVD) {
            // TODO: V_PREDICTION_EDM
            is_using_v_parameterization = true;
        }

        if (sd_version_is_sd3(version)) {
            LOG_INFO("running in FLOW mode");
            denoiser = std::make_shared<DiscreteFlowDenoiser>();
        } else if (sd_version_is_flux(version)) {
            LOG_INFO("running in Flux FLOW mode");
            float shift = 1.0f;  // TODO: validate
            for (auto pair : model_loader.tensor_storages_types) {
                if (pair.first.find("model.diffusion_model.guidance_in.in_layer.weight") != std::string::npos) {
                    shift = 1.15f;
                    break;
                }
            }
            denoiser = std::make_shared<FluxFlowDenoiser>(shift);
        } else if (is_using_v_parameterization) {
            LOG_INFO("running in v-prediction mode");
            denoiser = std::make_shared<CompVisVDenoiser>();
        } else if (is_using_edm_v_parameterization) {
            LOG_INFO("running in v-prediction EDM mode");
            denoiser = std::make_shared<EDMVDenoiser>();
        } else {
            LOG_INFO("running in eps-prediction mode");
        }

        if (sd_ctx_params->schedule != DEFAULT) {
            switch (sd_ctx_params->schedule) {
                case DISCRETE:
                    LOG_INFO("running with discrete schedule");
                    denoiser->schedule = std::make_shared<DiscreteSchedule>();
                    break;
                case KARRAS:
                    LOG_INFO("running with Karras schedule");
                    denoiser->schedule = std::make_shared<KarrasSchedule>();
                    break;
                case EXPONENTIAL:
                    LOG_INFO("running exponential schedule");
                    denoiser->schedule = std::make_shared<ExponentialSchedule>();
                    break;
                case AYS:
                    LOG_INFO("Running with Align-Your-Steps schedule");
                    denoiser->schedule          = std::make_shared<AYSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case GITS:
                    LOG_INFO("Running with GITS schedule");
                    denoiser->schedule          = std::make_shared<GITSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case DEFAULT:
                    // Don't touch anything.
                    break;
                default:
                    LOG_ERROR("Unknown schedule %i", sd_ctx_params->schedule);
                    abort();
            }
        }

        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
        if (comp_vis_denoiser) {
            for (int i = 0; i < TIMESTEPS; i++) {
                comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
            }
        }

        LOG_DEBUG("finished loaded file");
        ggml_free(ctx);
        return true;
    }

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx, bool is_inpaint = false) {
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        struct ggml_tensor* c = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        struct ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(timesteps, 999);

        struct ggml_tensor* concat = is_inpaint ? ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 5, 1) : NULL;
        if (concat != NULL) {
            ggml_set_f32(concat, 0);
        }

        int64_t t0              = ggml_time_ms();
        struct ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        diffusion_model->compute(n_threads, x_t, timesteps, c, concat, NULL, NULL, {}, -1, {}, 0.f, &out);
        diffusion_model->free_compute_buffer();

        double result = 0.f;
        {
            float* vec_x   = (float*)x_t->data;
            float* vec_out = (float*)out->data;

            int64_t n = ggml_nelements(out);

            for (int i = 0; i < n; i++) {
                result += ((double)vec_out[i] - (double)vec_x[i]);
            }
            result /= n;
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    void apply_lora_from_file(const std::string& lora_path, float multiplier) {
        int64_t t0                 = ggml_time_ms();
        std::string st_file_path   = lora_path;
        std::string file_path;
        if (file_exists(st_file_path)) {
            file_path = st_file_path;
        } else {
            LOG_WARN("can not find %s for lora %s", st_file_path.c_str(), lora_path.c_str());
            return;
        }
        LoraModel lora(backend, file_path);
        if (!lora.load_from_file()) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        lora.apply(tensors, version, n_threads);
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        LOG_INFO("lora '%s' applied, taking %.2fs",
                 lora_path.c_str(),
                 (t1 - t0) * 1.0f / 1000);
    }

    void apply_lora(const std::string& lora_name, float multiplier) {
        int64_t t0                 = ggml_time_ms();
        std::string st_file_path   = path_join(lora_model_dir, lora_name + ".safetensors");
        std::string ckpt_file_path = path_join(lora_model_dir, lora_name + ".ckpt");
        std::string file_path;
        if (file_exists(st_file_path)) {
            file_path = st_file_path;
        } else if (file_exists(ckpt_file_path)) {
            file_path = ckpt_file_path;
        } else {
            LOG_WARN("can not find %s or %s for lora %s", st_file_path.c_str(), ckpt_file_path.c_str(), lora_name.c_str());
            return;
        }
        LoraModel lora(backend, file_path);
        if (!lora.load_from_file()) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        // TODO: send version?
        lora.apply(tensors, version, n_threads);
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        LOG_INFO("lora '%s' applied, taking %.2fs", lora_name.c_str(), (t1 - t0) * 1.0f / 1000);
    }

    void apply_loras(const std::unordered_map<std::string, float>& lora_state) {
        if (lora_state.size() > 0 && model_wtype != GGML_TYPE_F16 && model_wtype != GGML_TYPE_F32) {
            LOG_WARN("In quantized models when applying LoRA, the images have poor quality.");
        }
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

        size_t rm = lora_state_diff.size() - lora_state.size();
        if (rm != 0) {
            LOG_INFO("Attempting to apply %lu LoRAs (removing %lu applied LoRAs)", lora_state.size(), rm);
        } else {
            LOG_INFO("Attempting to apply %lu LoRAs", lora_state.size());
        }

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

    ggml_tensor* id_encoder(ggml_context* work_ctx,
                            ggml_tensor* init_img,
                            ggml_tensor* prompts_embeds,
                            ggml_tensor* id_embeds,
                            std::vector<bool>& class_tokens_mask) {
        ggml_tensor* res = NULL;
        pmid_model->compute(n_threads, init_img, prompts_embeds, id_embeds, class_tokens_mask, &res, work_ctx);
        return res;
    }

    SDCondition get_svd_condition(ggml_context* work_ctx,
                                  sd_image_t init_image,
                                  int width,
                                  int height,
                                  int fps                    = 6,
                                  int motion_bucket_id       = 127,
                                  float augmentation_level   = 0.f,
                                  bool force_zero_embeddings = false) {
        // c_crossattn
        int64_t t0                      = ggml_time_ms();
        struct ggml_tensor* c_crossattn = NULL;
        {
            if (force_zero_embeddings) {
                c_crossattn = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, clip_vision->vision_model.projection_dim);
                ggml_set_f32(c_crossattn, 0.f);
            } else {
                sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                sd_image_f32_t resized_image = clip_preprocess(image, clip_vision->vision_model.image_size);
                free(image.data);
                image.data = NULL;

                ggml_tensor* pixel_values = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
                sd_image_f32_to_tensor(resized_image.data, pixel_values, false);
                free(resized_image.data);
                resized_image.data = NULL;

                // print_ggml_tensor(pixel_values);
                clip_vision->compute(n_threads, pixel_values, &c_crossattn, work_ctx);
                // print_ggml_tensor(c_crossattn);
            }
        }

        // c_concat
        struct ggml_tensor* c_concat = NULL;
        {
            if (force_zero_embeddings) {
                c_concat = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 4, 1);
                ggml_set_f32(c_concat, 0.f);
            } else {
                ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);

                if (width != init_image.width || height != init_image.height) {
                    sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                    sd_image_f32_t resized_image = resize_sd_image_f32_t(image, width, height);
                    free(image.data);
                    image.data = NULL;
                    sd_image_f32_to_tensor(resized_image.data, init_img, false);
                    free(resized_image.data);
                    resized_image.data = NULL;
                } else {
                    sd_image_to_tensor(init_image.data, init_img);
                }
                if (augmentation_level > 0.f) {
                    struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_img);
                    ggml_tensor_set_f32_randn(noise, rng);
                    // encode_pixels += torch.randn_like(pixels) * augmentation_level
                    ggml_tensor_scale(noise, augmentation_level);
                    ggml_tensor_add(init_img, noise);
                }
                ggml_tensor* moments = encode_first_stage(work_ctx, init_img);
                c_concat             = get_first_stage_encoding(work_ctx, moments);
            }
        }

        // y
        struct ggml_tensor* y = NULL;
        {
            y                            = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model->get_adm_in_channels());
            int out_dim                  = 256;
            int fps_id                   = fps - 1;
            std::vector<float> timesteps = {(float)fps_id, (float)motion_bucket_id, augmentation_level};
            set_timestep_embedding(timesteps, y, out_dim);
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing svd condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {c_crossattn, y, c_concat};
    }

    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* init_latent,
                        ggml_tensor* noise,
                        SDCondition cond,
                        SDCondition uncond,
                        SDCondition img_cond,
                        ggml_tensor* control_hint,
                        float control_strength,
                        sd_guidance_params_t guidance,
                        float eta,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        int start_merge_step,
                        SDCondition id_cond,
                        std::vector<ggml_tensor*> ref_latents = {},
                        ggml_tensor* denoise_mask             = nullptr) {
        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);

        float cfg_scale     = guidance.txt_cfg;
        float img_cfg_scale = guidance.img_cfg;
        float slg_scale     = guidance.slg.scale;

        float min_cfg = guidance.min_cfg;

        if (img_cfg_scale != cfg_scale && !sd_version_is_inpaint_or_unet_edit(version)) {
            LOG_WARN("2-conditioning CFG is not supported with this model, disabling it for better performance...");
            img_cfg_scale = cfg_scale;
        }

        LOG_DEBUG("Sample");
        struct ggml_init_params params;
        size_t data_size = ggml_row_size(init_latent->type, init_latent->ne[0]);
        for (int i = 1; i < 4; i++) {
            data_size *= init_latent->ne[i];
        }
        data_size += 1024;
        params.mem_size       = data_size * 3;
        params.mem_buffer     = NULL;
        params.no_alloc       = false;
        ggml_context* tmp_ctx = ggml_init(params);

        size_t steps = sigmas.size() - 1;
        // noise = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(noise);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, init_latent);
        copy_ggml_tensor(x, init_latent);
        x = denoiser->noise_scaling(sigmas[0], noise, x);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, noise);

        bool has_unconditioned = img_cfg_scale != 1.0 && uncond.c_crossattn != NULL;
        bool has_img_cond      = cfg_scale != img_cfg_scale && img_cond.c_crossattn != NULL;
        bool has_skiplayer     = slg_scale != 0.0 && skip_layers.size() > 0;

        // denoise wrapper
        struct ggml_tensor* out_cond     = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond   = NULL;
        struct ggml_tensor* out_skip     = NULL;
        struct ggml_tensor* out_img_cond = NULL;

        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        if (has_skiplayer) {
            if (sd_version_is_dit(version)) {
                out_skip = ggml_dup_tensor(work_ctx, x);
            } else {
                has_skiplayer = false;
                LOG_WARN("SLG is incompatible with %s models", model_version_to_str[version]);
            }
        }
        if (has_img_cond) {
            out_img_cond = ggml_dup_tensor(work_ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        auto denoise = [&](ggml_tensor* input, float sigma, int step) -> ggml_tensor* {
            if (step == 1) {
                pretty_progress(0, (int)steps, 0);
            }
            int64_t t0 = ggml_time_us();

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            float t = denoiser->sigma_to_t(sigma);
            std::vector<float> timesteps_vec(x->ne[3], t);  // [N, ]
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            std::vector<float> guidance_vec(x->ne[3], guidance.distilled_guidance);
            auto guidance_tensor = vector_to_ggml_tensor(work_ctx, guidance_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_tensor_scale(noised_input, c_in);

            std::vector<struct ggml_tensor*> controls;

            if (control_hint != NULL) {
                control_net->compute(n_threads, noised_input, control_hint, timesteps, cond.c_crossattn, cond.c_vector);
                controls = control_net->controls;
                // print_ggml_tensor(controls[12]);
                // GGML_ASSERT(0);
            }

            if (start_merge_step == -1 || step <= start_merge_step) {
                // cond
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         cond.c_crossattn,
                                         cond.c_concat,
                                         cond.c_vector,
                                         guidance_tensor,
                                         ref_latents,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            } else {
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         id_cond.c_crossattn,
                                         cond.c_concat,
                                         id_cond.c_vector,
                                         guidance_tensor,
                                         ref_latents,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            }

            float* negative_data = NULL;
            if (has_unconditioned) {
                // uncond
                if (control_hint != NULL) {
                    control_net->compute(n_threads, noised_input, control_hint, timesteps, uncond.c_crossattn, uncond.c_vector);
                    controls = control_net->controls;
                }
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         uncond.c_crossattn,
                                         uncond.c_concat,
                                         uncond.c_vector,
                                         guidance_tensor,
                                         ref_latents,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_uncond);
                negative_data = (float*)out_uncond->data;
            }

            float* img_cond_data = NULL;
            if (has_img_cond) {
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         img_cond.c_crossattn,
                                         img_cond.c_concat,
                                         img_cond.c_vector,
                                         guidance_tensor,
                                         ref_latents,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_img_cond);
                img_cond_data = (float*)out_img_cond->data;
            }

            int step_count         = sigmas.size();
            bool is_skiplayer_step = has_skiplayer && step > (int)(guidance.slg.layer_start * step_count) && step < (int)(guidance.slg.layer_end * step_count);
            float* skip_layer_data = NULL;
            if (is_skiplayer_step) {
                LOG_DEBUG("Skipping layers at step %d\n", step);
                // skip layer (same as conditionned)
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         cond.c_crossattn,
                                         cond.c_concat,
                                         cond.c_vector,
                                         guidance_tensor,
                                         ref_latents,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_skip,
                                         NULL,
                                         skip_layers);
                skip_layer_data = (float*)out_skip->data;
            }
            float* vec_denoised  = (float*)denoised->data;
            float* vec_input     = (float*)input->data;
            float* positive_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised);
            for (int i = 0; i < ne_elements; i++) {
                float latent_result = positive_data[i];
                if (has_unconditioned) {
                    // out_uncond + cfg_scale * (out_cond - out_uncond)
                    int64_t ne3 = out_cond->ne[3];
                    if (min_cfg != cfg_scale && ne3 != 1) {
                        int64_t i3  = i / out_cond->ne[0] * out_cond->ne[1] * out_cond->ne[2];
                        float scale = min_cfg + (cfg_scale - min_cfg) * (i3 * 1.0f / ne3);
                    } else {
                        if (has_img_cond) {
                            // out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
                            latent_result = negative_data[i] + img_cfg_scale * (img_cond_data[i] - negative_data[i]) + cfg_scale * (positive_data[i] - img_cond_data[i]);
                        } else {
                            // img_cfg_scale == cfg_scale
                            latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                        }
                    }
                } else if (has_img_cond) {
                    // img_cfg_scale == 1
                    latent_result = img_cond_data[i] + cfg_scale * (positive_data[i] - img_cond_data[i]);
                }
                if (is_skiplayer_step) {
                    latent_result = latent_result + (positive_data[i] - skip_layer_data[i]) * slg_scale;
                }
                // v = latent_result, eps = latent_result
                // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
                vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
            }
            int64_t t1 = ggml_time_us();
            if (step > 0) {
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                // LOG_INFO("step %d sampling completed taking %.2fs", step, (t1 - t0) * 1.0f / 1000000);
            }
            if (denoise_mask != nullptr) {
                for (int64_t x = 0; x < denoised->ne[0]; x++) {
                    for (int64_t y = 0; y < denoised->ne[1]; y++) {
                        float mask = ggml_tensor_get_f32(denoise_mask, x, y);
                        for (int64_t k = 0; k < denoised->ne[2]; k++) {
                            float init = ggml_tensor_get_f32(init_latent, x, y, k);
                            float den  = ggml_tensor_get_f32(denoised, x, y, k);
                            ggml_tensor_set_f32(denoised, init + mask * (den - init), x, y, k);
                        }
                    }
                }
            }

            return denoised;
        };

        sample_k_diffusion(method, denoise, work_ctx, x, sigmas, rng, eta);

        x = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x);

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        diffusion_model->free_compute_buffer();
        return x;
    }

    // ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        ggml_tensor* latent       = ggml_new_tensor_4d(work_ctx, moments->type, moments->ne[0], moments->ne[1], moments->ne[2] / 2, moments->ne[3]);
        struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, latent);
        ggml_tensor_set_f32_randn(noise, rng);
        // noise = load_tensor_from_file(work_ctx, "noise.bin");
        {
            float mean   = 0;
            float logvar = 0;
            float value  = 0;
            float std_   = 0;
            for (int i = 0; i < latent->ne[3]; i++) {
                for (int j = 0; j < latent->ne[2]; j++) {
                    for (int k = 0; k < latent->ne[1]; k++) {
                        for (int l = 0; l < latent->ne[0]; l++) {
                            mean   = ggml_tensor_get_f32(moments, l, k, j, i);
                            logvar = ggml_tensor_get_f32(moments, l, k, j + (int)latent->ne[2], i);
                            logvar = std::max(-30.0f, std::min(logvar, 20.0f));
                            std_   = std::exp(0.5f * logvar);
                            value  = mean + std_ * ggml_tensor_get_f32(noise, l, k, j, i);
                            value  = value * scale_factor;
                            // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                            ggml_tensor_set_f32(latent, value, l, k, j, i);
                        }
                    }
                }
            }
        }
        return latent;
    }

    ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode) {
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];
        int64_t C = 8;
        if (use_tiny_autoencoder) {
            C = 4;
        } else {
            if (sd_version_is_sd3(version)) {
                C = 32;
            } else if (sd_version_is_flux(version)) {
                C = 32;
            }
        }
        ggml_tensor* result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                 decode ? (W * 8) : (W / 8),  // width
                                                 decode ? (H * 8) : (H / 8),  // height
                                                 decode ? 3 : C,
                                                 x->ne[3]);  // channels
        int64_t t0          = ggml_time_ms();

        // TODO: args instead of env for tile size / overlap?

        float tile_overlap = 0.5f;
        const char* SD_TILE_OVERLAP = getenv("SD_TILE_OVERLAP");
        if (SD_TILE_OVERLAP != nullptr) {
            std::string sd_tile_overlap_str = SD_TILE_OVERLAP;
            try {
                tile_overlap = std::stof(sd_tile_overlap_str);
                if (tile_overlap < 0.0) {
                    LOG_WARN("SD_TILE_OVERLAP too low, setting it to 0.0");
                    tile_overlap = 0.0;
                }
                else if (tile_overlap > 0.5) {
                    LOG_WARN("SD_TILE_OVERLAP too high, setting it to 0.5");
                    tile_overlap = 0.5;
                }
            } catch (const std::invalid_argument&) {
                LOG_WARN("SD_TILE_OVERLAP is invalid, keeping the default");
            } catch (const std::out_of_range&) {
                LOG_WARN("SD_TILE_OVERLAP is out of range, keeping the default");
            }
        }

        int tile_size_x = 32;
        int tile_size_y = 32;
        const char* SD_TILE_SIZE = getenv("SD_TILE_SIZE");
        if (SD_TILE_SIZE != nullptr) {
            // format is AxB, or just A (equivalent to AxA)
            // A and B can be integers (tile size) or floating point
            // floating point <= 1 means simple fraction of the latent dimension
            // floating point > 1 means number of tiles across that dimension
            // a single number gets applied to both
            auto get_tile_factor = [tile_overlap](const std::string& factor_str) {
                float factor = std::stof(factor_str);
                if (factor > 1.0)
                    factor = 1 / (factor - factor * tile_overlap + tile_overlap);
                return factor;
            };
            const int latent_x = W / (decode ? 1 : 8);
            const int latent_y = H / (decode ? 1 : 8);
            const int min_tile_dimension = 4;
            std::string sd_tile_size_str = SD_TILE_SIZE;
            size_t x_pos = sd_tile_size_str.find('x');
            try {
                int tmp_x = tile_size_x, tmp_y = tile_size_y;
                if (x_pos != std::string::npos) {
                    std::string tile_x_str = sd_tile_size_str.substr(0, x_pos);
                    std::string tile_y_str = sd_tile_size_str.substr(x_pos + 1);
                    if (tile_x_str.find('.') != std::string::npos) {
                        tmp_x = std::round(latent_x * get_tile_factor(tile_x_str));
                    }
                    else {
                        tmp_x = std::stoi(tile_x_str);
                    }
                    if (tile_y_str.find('.') != std::string::npos) {
                        tmp_y = std::round(latent_y * get_tile_factor(tile_y_str));
                    }
                    else {
                        tmp_y = std::stoi(tile_y_str);
                    }
                }
                else {
                    if (sd_tile_size_str.find('.') != std::string::npos) {
                        float tile_factor = get_tile_factor(sd_tile_size_str);
                        tmp_x = std::round(latent_x * tile_factor);
                        tmp_y = std::round(latent_y * tile_factor);
                    }
                    else {
                        tmp_x = tmp_y = std::stoi(sd_tile_size_str);
                    }
                }
                tile_size_x = std::max(std::min(tmp_x, latent_x), min_tile_dimension);
                tile_size_y = std::max(std::min(tmp_y, latent_y), min_tile_dimension);
            } catch (const std::invalid_argument&) {
                LOG_WARN("SD_TILE_SIZE is invalid, keeping the default");
            } catch (const std::out_of_range&) {
                LOG_WARN("SD_TILE_SIZE is out of range, keeping the default");
            }
        }

        if(!decode){
            // TODO: also use and arg for this one?
            // to keep the compute buffer size consistent
            tile_size_x*=1.30539;
            tile_size_y*=1.30539;
        }
        if (!use_tiny_autoencoder) {
            if (decode) {
                ggml_tensor_scale(x, 1.0f / scale_factor);
            } else {
                ggml_tensor_scale_input(x);
            }
            if (vae_tiling) {
                if (SD_TILE_SIZE != nullptr) {
                    LOG_INFO("VAE Tile size: %dx%d", tile_size_x, tile_size_y);
                }
                if (SD_TILE_OVERLAP != nullptr) {
                    LOG_INFO("VAE Tile overlap: %.2f", tile_overlap);
                }
                // split latent in 32x32 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    first_stage_model->compute(n_threads, in, decode, &out);
                };
                sd_tiling_non_square(x, result, 8, tile_size_x, tile_size_y, tile_overlap, on_tiling);
            } else {
                first_stage_model->compute(n_threads, x, decode, &result);
            }
            first_stage_model->free_compute_buffer();
            if (decode) {
                ggml_tensor_scale_output(result);
            }
        } else {
            if (vae_tiling) {
                // split latent in 64x64 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    tae_first_stage->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 64, 0.5f, on_tiling);
            } else {
                tae_first_stage->compute(n_threads, x, decode, &result);
            }
            tae_first_stage->free_compute_buffer();
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae [mode: %s] graph completed, taking %.2fs", decode ? "DECODE" : "ENCODE", (t1 - t0) * 1.0f / 1000);
        if (decode) {
            ggml_tensor_clamp(result, 0.0f, 1.0f);
        }
        return result;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, false);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, true);
    }
};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

const char* sd_type_name(enum sd_type_t type) {
    return ggml_type_name((ggml_type)type);
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < SD_TYPE_COUNT; i++) {
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
    "euler_a",
    "euler",
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

const char* schedule_to_str[] = {
    "default",
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
};

const char* sd_schedule_name(enum schedule_t schedule) {
    if (schedule < SCHEDULE_COUNT) {
        return schedule_to_str[schedule];
    }
    return NONE_STR;
}

enum schedule_t str_to_schedule(const char* str) {
    for (int i = 0; i < SCHEDULE_COUNT; i++) {
        if (!strcmp(str, schedule_to_str[i])) {
            return (enum schedule_t)i;
        }
    }
    return SCHEDULE_COUNT;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    memset((void*)sd_ctx_params, 0, sizeof(sd_ctx_params_t));
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->vae_tiling              = false;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->schedule                = DEFAULT;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return NULL;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "t5xxl_path: %s\n"
             "diffusion_model_path: %s\n"
             "vae_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "lora_model_dir: %s\n"
             "embedding_dir: %s\n"
             "stacked_id_embed_dir: %s\n"
             "vae_decode_only: %s\n"
             "vae_tiling: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "schedule: %s\n"
             "keep_clip_on_cpu: %s\n"
             "keep_control_net_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "diffusion_flash_attn: %s\n"
             "chroma_use_dit_mask: %s\n"
             "chroma_use_t5_mask: %s\n"
             "chroma_t5_mask_pad: %d\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->lora_model_dir),
             SAFE_STR(sd_ctx_params->embedding_dir),
             SAFE_STR(sd_ctx_params->stacked_id_embed_dir),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->vae_tiling),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_schedule_name(sd_ctx_params->schedule),
             BOOL_STR(sd_ctx_params->keep_clip_on_cpu),
             BOOL_STR(sd_ctx_params->keep_control_net_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->chroma_use_dit_mask),
             BOOL_STR(sd_ctx_params->chroma_use_t5_mask),
             sd_ctx_params->chroma_t5_mask_pad);

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    memset((void*)sd_img_gen_params, 0, sizeof(sd_img_gen_params_t));
    sd_img_gen_params->clip_skip                   = -1;
    sd_img_gen_params->guidance.txt_cfg            = 7.0f;
    sd_img_gen_params->guidance.min_cfg            = 1.0f;
    sd_img_gen_params->guidance.img_cfg            = INFINITY;
    sd_img_gen_params->guidance.distilled_guidance = 3.5f;
    sd_img_gen_params->guidance.slg.layer_count    = 0;
    sd_img_gen_params->guidance.slg.layer_start    = 0.01f;
    sd_img_gen_params->guidance.slg.layer_end      = 0.2f;
    sd_img_gen_params->guidance.slg.scale          = 0.f;
    sd_img_gen_params->ref_images_count            = 0;
    sd_img_gen_params->width                       = 512;
    sd_img_gen_params->height                      = 512;
    sd_img_gen_params->sample_method               = EULER_A;
    sd_img_gen_params->sample_steps                = 20;
    sd_img_gen_params->eta                         = 0.f;
    sd_img_gen_params->strength                    = 0.75f;
    sd_img_gen_params->seed                        = -1;
    sd_img_gen_params->batch_count                 = 1;
    sd_img_gen_params->control_strength            = 0.9f;
    sd_img_gen_params->style_strength              = 20.f;
    sd_img_gen_params->normalize_input             = false;
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return NULL;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "txt_cfg: %.2f\n"
             "img_cfg: %.2f\n"
             "min_cfg: %.2f\n"
             "distilled_guidance: %.2f\n"
             "slg.layer_count: %zu\n"
             "slg.layer_start: %.2f\n"
             "slg.layer_end: %.2f\n"
             "slg.scale: %.2f\n"
             "width: %d\n"
             "height: %d\n"
             "sample_method: %s\n"
             "sample_steps: %d\n"
             "eta: %.2f\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "control_strength: %.2f\n"
             "style_strength: %.2f\n"
             "normalize_input: %s\n"
             "input_id_images_path: %s\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->guidance.txt_cfg,
             sd_img_gen_params->guidance.img_cfg,
             sd_img_gen_params->guidance.min_cfg,
             sd_img_gen_params->guidance.distilled_guidance,
             sd_img_gen_params->guidance.slg.layer_count,
             sd_img_gen_params->guidance.slg.layer_start,
             sd_img_gen_params->guidance.slg.layer_end,
             sd_img_gen_params->guidance.slg.scale,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             sd_sample_method_name(sd_img_gen_params->sample_method),
             sd_img_gen_params->sample_steps,
             sd_img_gen_params->eta,
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             sd_img_gen_params->control_strength,
             sd_img_gen_params->style_strength,
             BOOL_STR(sd_img_gen_params->normalize_input),
             SAFE_STR(sd_img_gen_params->input_id_images_path));

    return buf;
}

void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    memset((void*)sd_vid_gen_params, 0, sizeof(sd_vid_gen_params_t));
    sd_vid_gen_params->guidance.txt_cfg            = 7.0f;
    sd_vid_gen_params->guidance.min_cfg            = 1.0f;
    sd_vid_gen_params->guidance.img_cfg            = INFINITY;
    sd_vid_gen_params->guidance.distilled_guidance = 3.5f;
    sd_vid_gen_params->guidance.slg.layer_count    = 0;
    sd_vid_gen_params->guidance.slg.layer_start    = 0.01f;
    sd_vid_gen_params->guidance.slg.layer_end      = 0.2f;
    sd_vid_gen_params->guidance.slg.scale          = 0.f;
    sd_vid_gen_params->width                       = 512;
    sd_vid_gen_params->height                      = 512;
    sd_vid_gen_params->sample_method               = EULER_A;
    sd_vid_gen_params->sample_steps                = 20;
    sd_vid_gen_params->strength                    = 0.75f;
    sd_vid_gen_params->seed                        = -1;
    sd_vid_gen_params->video_frames                = 6;
    sd_vid_gen_params->motion_bucket_id            = 127;
    sd_vid_gen_params->fps                         = 6;
    sd_vid_gen_params->augmentation_level          = 0.f;
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = NULL;
};

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == NULL) {
        return NULL;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == NULL) {
        return NULL;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
        free(sd_ctx);
        return NULL;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != NULL) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
    }
    free(sd_ctx);
}

sd_image_t* generate_image_internal(sd_ctx_t* sd_ctx,
                                    struct ggml_context* work_ctx,
                                    ggml_tensor* init_latent,
                                    std::string prompt,
                                    std::string negative_prompt,
                                    int clip_skip,
                                    sd_guidance_params_t guidance,
                                    float eta,
                                    int width,
                                    int height,
                                    enum sample_method_t sample_method,
                                    const std::vector<float>& sigmas,
                                    int64_t seed,
                                    int batch_count,
                                    const sd_image_t* control_cond,
                                    float control_strength,
                                    float style_ratio,
                                    bool normalize_input,
                                    std::string input_id_images_path,
                                    std::vector<ggml_tensor*> ref_latents,
                                    ggml_tensor* concat_latent = NULL,
                                    ggml_tensor* denoise_mask  = NULL,
                                    const kcpp_img_gen_params_t* kcpp_img_gen_params = NULL) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(NULL));
        seed = rand();
    }

    // for (auto v : sigmas) {
    //     std::cout << v << " ";
    // }
    // std::cout << std::endl;

    int sample_steps = sigmas.size() - 1;

    // Apply lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    //only use hardcoded lora for kcpp
    if (!lora_f2m.empty()) {
        lora_f2m.clear();
        printf("\nWarning: not applying LoRAs requested by prompt!\n");
    }

    int64_t t0 = ggml_time_ms();
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    // Photo Maker
    std::string prompt_text_only;
    ggml_tensor* init_img = NULL;
    SDCondition id_cond;
    std::vector<bool> class_tokens_mask;
    if (sd_ctx->sd->pmid_lora && kcpp_img_gen_params && kcpp_img_gen_params->photomaker_reference_count>0)
    {
        sd_ctx->sd->stacked_id = true; //turn on photomaker if needed
    }
    if (sd_ctx->sd->stacked_id) {
        if (!sd_ctx->sd->pmid_lora->applied) {
            t0 = ggml_time_ms();
            sd_ctx->sd->pmid_lora->apply(sd_ctx->sd->tensors, sd_ctx->sd->version, sd_ctx->sd->n_threads);
            t1                             = ggml_time_ms();
            sd_ctx->sd->pmid_lora->applied = true;
            LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_lora->free_params_buffer();
            }
        }
        // preprocess input id images
        std::vector<sd_image_t*> input_id_images;
        bool pmv2 = sd_ctx->sd->pmid_model->get_version() == PM_VERSION_2;
        if (sd_ctx->sd->pmid_model && input_id_images_path.size() > 0) {
            std::vector<std::string> img_files = get_files_from_dir(input_id_images_path);
            for (std::string img_file : img_files) {
                int c = 0;
                int width, height;
                if (ends_with(img_file, "safetensors")) {
                    continue;
                }
                uint8_t* input_image_buffer = stbi_load(img_file.c_str(), &width, &height, &c, 3);
                if (input_image_buffer == NULL) {
                    LOG_ERROR("PhotoMaker load image from '%s' failed", img_file.c_str());
                    continue;
                } else {
                    LOG_INFO("PhotoMaker loaded image from '%s'", img_file.c_str());
                }
                sd_image_t* input_image = NULL;
                input_image             = new sd_image_t{(uint32_t)width,
                                             (uint32_t)height,
                                             3,
                                             input_image_buffer};
                input_image             = preprocess_id_image(input_image);
                if (input_image == NULL) {
                    LOG_ERROR("preprocess input id image from '%s' failed", img_file.c_str());
                    continue;
                }
                input_id_images.push_back(input_image);
            }
        }

        // handle multiple photomaker image passed in by kcpp
        if (sd_ctx->sd->pmid_lora && kcpp_img_gen_params)
        {
            for(int i=0;i<kcpp_img_gen_params->photomaker_reference_count;++i)
            {
                int c = 0;
                int width, height;
                width = kcpp_img_gen_params->photomaker_references[i].width;
                height = kcpp_img_gen_params->photomaker_references[i].height;
                c = kcpp_img_gen_params->photomaker_references[i].channel;
                uint8_t* input_image_buffer = kcpp_img_gen_params->photomaker_references[i].data;
                sd_image_t* input_image = NULL;
                input_image  = new sd_image_t{(uint32_t)width,
                                                (uint32_t)height,
                                                3,
                                                input_image_buffer};
                input_image   = preprocess_id_image(input_image);
                if (input_image == NULL) {
                    LOG_ERROR("\npreprocess input id image from kcpp photomaker failed\n");
                } else {
                    LOG_INFO("\nPhotoMaker loaded image from kcpp\n");
                    input_id_images.push_back(input_image);
                }
            }
        }

        if (input_id_images.size() > 0) {
            sd_ctx->sd->pmid_model->style_strength = style_ratio;
            int32_t w                              = input_id_images[0]->width;
            int32_t h                              = input_id_images[0]->height;
            int32_t channels                       = input_id_images[0]->channel;
            int32_t num_input_images               = (int32_t)input_id_images.size();
            init_img                               = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, channels, num_input_images);
            // TODO: move these to somewhere else and be user settable
            float mean[] = {0.48145466f, 0.4578275f, 0.40821073f};
            float std[]  = {0.26862954f, 0.26130258f, 0.27577711f};
            for (int i = 0; i < num_input_images; i++) {
                sd_image_t* init_image = input_id_images[i];
                if (normalize_input)
                    sd_mul_images_to_tensor(init_image->data, init_img, i, mean, std);
                else
                    sd_mul_images_to_tensor(init_image->data, init_img, i, NULL, NULL);
            }
            t0                            = ggml_time_ms();
            auto cond_tup                 = sd_ctx->sd->cond_stage_model->get_learned_condition_with_trigger(work_ctx,
                                                                                                             sd_ctx->sd->n_threads, prompt,
                                                                                                             clip_skip,
                                                                                                             width,
                                                                                                             height,
                                                                                                             num_input_images,
                                                                                                             sd_ctx->sd->diffusion_model->get_adm_in_channels());
            id_cond                       = std::get<0>(cond_tup);
            class_tokens_mask             = std::get<1>(cond_tup);  //
            struct ggml_tensor* id_embeds = NULL;
            if (pmv2) {
                // id_embeds = sd_ctx->sd->pmid_id_embeds->get();
                id_embeds = load_tensor_from_file(work_ctx, path_join(input_id_images_path, "id_embeds.bin"));
                // print_ggml_tensor(id_embeds, true, "id_embeds:");
            }
            id_cond.c_crossattn = sd_ctx->sd->id_encoder(work_ctx, init_img, id_cond.c_crossattn, id_embeds, class_tokens_mask);
            t1                  = ggml_time_ms();
            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_model->free_params_buffer();
            }
            // Encode input prompt without the trigger word for delayed conditioning
            prompt_text_only = sd_ctx->sd->cond_stage_model->remove_trigger_from_prompt(work_ctx, prompt);
            // printf("%s || %s \n", prompt.c_str(), prompt_text_only.c_str());
            prompt = prompt_text_only;  //
            // if (sample_steps < 50) {
            //     LOG_INFO("sampling steps increases from %d to 50 for PHOTOMAKER", sample_steps);
            //     sample_steps = 50;
            // }
        } else {
            LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
            LOG_WARN("Turn off PhotoMaker");
            sd_ctx->sd->stacked_id = false;
        }
        for (sd_image_t* img : input_id_images) {
            free(img->data);
        }
        input_id_images.clear();
    }

    // Get learned condition
    t0               = ggml_time_ms();
    SDCondition cond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                           sd_ctx->sd->n_threads,
                                                                           prompt,
                                                                           clip_skip,
                                                                           width,
                                                                           height,
                                                                           sd_ctx->sd->diffusion_model->get_adm_in_channels());

    SDCondition uncond;
    if (guidance.txt_cfg != 1.0 ||
        (sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version) && guidance.txt_cfg != guidance.img_cfg)) {
        bool force_zero_embeddings = false;
        if (sd_version_is_sdxl(sd_ctx->sd->version) && negative_prompt.size() == 0 && !sd_ctx->sd->is_using_edm_v_parameterization) {
            force_zero_embeddings = true;
        }
        uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                     sd_ctx->sd->n_threads,
                                                                     negative_prompt,
                                                                     clip_skip,
                                                                     width,
                                                                     height,
                                                                     sd_ctx->sd->diffusion_model->get_adm_in_channels(),
                                                                     force_zero_embeddings);
    }
    t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    struct ggml_tensor* image_hint = NULL;
    if (control_cond != NULL) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_tensor(control_cond->data, image_hint);
    }

    // Sample
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        C = 16;
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        C = 16;
    }
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        int64_t mask_channels = 1;
        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            mask_channels = 8 * 8;  // flatten the whole mask
        }
        auto empty_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], mask_channels + init_latent->ne[2], 1);
        // no mask, set the whole image as masked
        for (int64_t x = 0; x < empty_latent->ne[0]; x++) {
            for (int64_t y = 0; y < empty_latent->ne[1]; y++) {
                if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                    // TODO: this might be wrong
                    for (int64_t c = 0; c < init_latent->ne[2]; c++) {
                        ggml_tensor_set_f32(empty_latent, 0, x, y, c);
                    }
                    for (int64_t c = init_latent->ne[2]; c < empty_latent->ne[2]; c++) {
                        ggml_tensor_set_f32(empty_latent, 1, x, y, c);
                    }
                } else {
                    ggml_tensor_set_f32(empty_latent, 1, x, y, 0);
                    for (int64_t c = 1; c < empty_latent->ne[2]; c++) {
                        ggml_tensor_set_f32(empty_latent, 0, x, y, c);
                    }
                }
            }
        }
        if (concat_latent == NULL) {
            concat_latent = empty_latent;
        }
        cond.c_concat   = concat_latent;
        uncond.c_concat = empty_latent;
        denoise_mask    = NULL;
    } else if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        auto empty_latent = ggml_dup_tensor(work_ctx, init_latent);
        ggml_set_f32(empty_latent, 0);
        uncond.c_concat = empty_latent;
        if (concat_latent == NULL) {
            concat_latent = empty_latent;
        }
        cond.c_concat = ref_latents[0];
    }
    SDCondition img_cond;
    if (uncond.c_crossattn != NULL &&
        (sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version) && guidance.txt_cfg != guidance.img_cfg)) {
        img_cond = SDCondition(uncond.c_crossattn, uncond.c_vector, cond.c_concat);
    }
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

        int start_merge_step = -1;
        if (sd_ctx->sd->stacked_id) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * sample_steps);
            // if (start_merge_step > 30)
            //     start_merge_step = 30;
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }

        // Disable min_cfg
        guidance.min_cfg = guidance.txt_cfg;

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     x_t,
                                                     noise,
                                                     cond,
                                                     uncond,
                                                     img_cond,
                                                     image_hint,
                                                     control_strength,
                                                     guidance,
                                                     eta,
                                                     sample_method,
                                                     sigmas,
                                                     start_merge_step,
                                                     id_cond,
                                                     ref_latents,
                                                     denoise_mask);

        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    // Decode to image
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1                      = ggml_time_ms();
        struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        // print_ggml_tensor(img);
        if (img != NULL) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    sd_image_t* result_images = (sd_image_t*)calloc(batch_count, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(decoded_images[i]);
    }
    ggml_free(work_ctx);

    return result_images;
}

ggml_tensor* generate_init_latent(sd_ctx_t* sd_ctx,
                                  ggml_context* work_ctx,
                                  int width,
                                  int height) {
    int C = 4;
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        C = 16;
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        C = 16;
    }
    int W                    = width / 8;
    int H                    = height / 8;
    ggml_tensor* init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        ggml_set_f32(init_latent, 0.0609f);
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        ggml_set_f32(init_latent, 0.1159f);
    } else {
        ggml_set_f32(init_latent, 0.f);
    }
    return init_latent;
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params, const kcpp_img_gen_params_t* kcpp_img_gen_params) {
    int width  = sd_img_gen_params->width;
    int height = sd_img_gen_params->height;
    if (sd_version_is_dit(sd_ctx->sd->version)) {
        if (width % 16 || height % 16) {
            LOG_ERROR("Image dimensions must be must be a multiple of 16 on each axis for %s models. (Got %dx%d)", model_version_to_str[sd_ctx->sd->version], width, height);
            return NULL;
        }
    } else if (width % 64 || height % 64) {
        LOG_ERROR("Image dimensions must be must be a multiple of 64 on each axis for %s models. (Got %dx%d)", model_version_to_str[sd_ctx->sd->version], width, height);
        return NULL;
    }
    LOG_DEBUG("generate_image %dx%d", width, height);
    if (sd_ctx == NULL || sd_img_gen_params == NULL) {
        return NULL;
    }

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(20 * 1024 * 1024);  // 20 MB increased by kcpp
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        params.mem_size *= 2; //readjust by kcpp as above changed
    }
    if (sd_version_is_flux(sd_ctx->sd->version)) {
        params.mem_size *= 3; //readjust by kcpp as above changed
    }
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(15 * 1024 * 1024);  // 15 MB increased by kcpp
    }
    params.mem_size += width * height * 3 * sizeof(float) * 3;
    params.mem_size += width * height * 3 * sizeof(float) * 3 * sd_img_gen_params->ref_images_count;
    params.mem_size *= sd_img_gen_params->batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    int64_t seed = sd_img_gen_params->seed;
    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);

    size_t t0 = ggml_time_ms();

    ggml_tensor* init_latent   = NULL;
    ggml_tensor* concat_latent = NULL;
    ggml_tensor* denoise_mask  = NULL;
    std::vector<float> sigmas  = sd_ctx->sd->denoiser->get_sigmas(sd_img_gen_params->sample_steps);

    if (sd_img_gen_params->init_image.data) {
        LOG_INFO("IMG2IMG");

        size_t t_enc = static_cast<size_t>(sd_img_gen_params->sample_steps * sd_img_gen_params->strength);
        if (t_enc == sd_img_gen_params->sample_steps)
            t_enc--;
        LOG_INFO("target t_enc is %zu steps", t_enc);
        std::vector<float> sigma_sched;
        sigma_sched.assign(sigmas.begin() + sd_img_gen_params->sample_steps - t_enc - 1, sigmas.end());
        sigmas = sigma_sched;

        ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        ggml_tensor* mask_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 1, 1);

        sd_mask_to_tensor(sd_img_gen_params->mask_image.data, mask_img);
        sd_image_to_tensor(sd_img_gen_params->init_image.data, init_img);

        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            int64_t mask_channels = 1;
            if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                mask_channels = 8 * 8;  // flatten the whole mask
            }
            ggml_tensor* masked_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
            sd_apply_mask(init_img, mask_img, masked_img);
            ggml_tensor* masked_latent = NULL;
            if (!sd_ctx->sd->use_tiny_autoencoder) {
                ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, masked_img);
                masked_latent        = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
            } else {
                masked_latent = sd_ctx->sd->encode_first_stage(work_ctx, masked_img);
            }
            concat_latent = ggml_new_tensor_4d(work_ctx,
                                               GGML_TYPE_F32,
                                               masked_latent->ne[0],
                                               masked_latent->ne[1],
                                               mask_channels + masked_latent->ne[2],
                                               1);
            for (int ix = 0; ix < masked_latent->ne[0]; ix++) {
                for (int iy = 0; iy < masked_latent->ne[1]; iy++) {
                    int mx = ix * 8;
                    int my = iy * 8;
                    if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                        for (int k = 0; k < masked_latent->ne[2]; k++) {
                            float v = ggml_tensor_get_f32(masked_latent, ix, iy, k);
                            ggml_tensor_set_f32(concat_latent, v, ix, iy, k);
                        }
                        // "Encode" 8x8 mask chunks into a flattened 1x64 vector, and concatenate to masked image
                        for (int x = 0; x < 8; x++) {
                            for (int y = 0; y < 8; y++) {
                                float m = ggml_tensor_get_f32(mask_img, mx + x, my + y);
                                // TODO: check if the way the mask is flattened is correct (is it supposed to be x*8+y or x+8*y?)
                                // python code was using "b (h 8) (w 8) -> b (8 8) h w"
                                ggml_tensor_set_f32(concat_latent, m, ix, iy, masked_latent->ne[2] + x * 8 + y);
                            }
                        }
                    } else {
                        float m = ggml_tensor_get_f32(mask_img, mx, my);
                        ggml_tensor_set_f32(concat_latent, m, ix, iy, 0);
                        for (int k = 0; k < masked_latent->ne[2]; k++) {
                            float v = ggml_tensor_get_f32(masked_latent, ix, iy, k);
                            ggml_tensor_set_f32(concat_latent, v, ix, iy, k + mask_channels);
                        }
                    }
                }
            }
        }

        {
            // LOG_WARN("Inpainting with a base model is not great");
            denoise_mask = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 1, 1);
            for (int ix = 0; ix < denoise_mask->ne[0]; ix++) {
                for (int iy = 0; iy < denoise_mask->ne[1]; iy++) {
                    int mx  = ix * 8;
                    int my  = iy * 8;
                    float m = ggml_tensor_get_f32(mask_img, mx, my);
                    ggml_tensor_set_f32(denoise_mask, m, ix, iy);
                }
            }
        }

        if (!sd_ctx->sd->use_tiny_autoencoder) {
            ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
            init_latent          = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
        } else {
            init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
        }
    } else {
        LOG_INFO("TXT2IMG");
        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            LOG_WARN("This is an inpainting model, this should only be used in img2img mode with a mask");
        }
        init_latent = generate_init_latent(sd_ctx, work_ctx, width, height);
    }

    if (sd_img_gen_params->ref_images_count > 0) {
        LOG_INFO("EDIT mode");
    }

    std::vector<struct ggml_tensor*> ref_latents;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ggml_tensor* img = ggml_new_tensor_4d(work_ctx,
                                              GGML_TYPE_F32,
                                              sd_img_gen_params->ref_images[i].width,
                                              sd_img_gen_params->ref_images[i].height,
                                              3,
                                              1);
        sd_image_to_tensor(sd_img_gen_params->ref_images[i].data, img);

        ggml_tensor* latent = NULL;
        if (sd_ctx->sd->use_tiny_autoencoder) {
            latent = sd_ctx->sd->encode_first_stage(work_ctx, img);
        } else if (sd_ctx->sd->version == VERSION_SD1_PIX2PIX) {
            latent = sd_ctx->sd->encode_first_stage(work_ctx, img);
            latent = ggml_view_3d(work_ctx,
                                  latent,
                                  latent->ne[0],
                                  latent->ne[1],
                                  latent->ne[2] / 2,
                                  latent->nb[1],
                                  latent->nb[2],
                                  0);
        } else {
            ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, img);
            latent               = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
        }
        ref_latents.push_back(latent);
    }

    if (sd_img_gen_params->init_image.data != NULL || sd_img_gen_params->ref_images_count > 0) {
        size_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    }

    sd_image_t* result_images = generate_image_internal(sd_ctx,
                                                        work_ctx,
                                                        init_latent,
                                                        SAFE_STR(sd_img_gen_params->prompt),
                                                        SAFE_STR(sd_img_gen_params->negative_prompt),
                                                        sd_img_gen_params->clip_skip,
                                                        sd_img_gen_params->guidance,
                                                        sd_img_gen_params->eta,
                                                        width,
                                                        height,
                                                        sd_img_gen_params->sample_method,
                                                        sigmas,
                                                        seed,
                                                        sd_img_gen_params->batch_count,
                                                        sd_img_gen_params->control_cond,
                                                        sd_img_gen_params->control_strength,
                                                        sd_img_gen_params->style_strength,
                                                        sd_img_gen_params->normalize_input,
                                                        sd_img_gen_params->input_id_images_path,
                                                        ref_latents,
                                                        concat_latent,
                                                        denoise_mask,
                                                        kcpp_img_gen_params);

    size_t t2 = ggml_time_ms();

    LOG_INFO("generate_image completed in %.2fs", (t2 - t0) * 1.0f / 1000);

    return result_images;
}

SD_API sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params) {
    if (sd_ctx == NULL || sd_vid_gen_params == NULL) {
        return NULL;
    }

    int width  = sd_vid_gen_params->width;
    int height = sd_vid_gen_params->height;
    LOG_INFO("img2vid %dx%d", width, height);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sd_vid_gen_params->sample_steps);

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024) * 1024;  // 10 MB
    params.mem_size += width * height * 3 * sizeof(float) * sd_vid_gen_params->video_frames;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    // draft context
    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    int64_t seed = sd_vid_gen_params->seed;
    if (seed < 0) {
        seed = (int)time(NULL);
    }

    sd_ctx->sd->rng->manual_seed(seed);

    int64_t t0 = ggml_time_ms();

    SDCondition cond = sd_ctx->sd->get_svd_condition(work_ctx,
                                                     sd_vid_gen_params->init_image,
                                                     width,
                                                     height,
                                                     sd_vid_gen_params->fps,
                                                     sd_vid_gen_params->motion_bucket_id,
                                                     sd_vid_gen_params->augmentation_level);

    auto uc_crossattn = ggml_dup_tensor(work_ctx, cond.c_crossattn);
    ggml_set_f32(uc_crossattn, 0.f);

    auto uc_concat = ggml_dup_tensor(work_ctx, cond.c_concat);
    ggml_set_f32(uc_concat, 0.f);

    auto uc_vector = ggml_dup_tensor(work_ctx, cond.c_vector);

    SDCondition uncond = SDCondition(uc_crossattn, uc_vector, uc_concat);

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->clip_vision->free_params_buffer();
    }

    sd_ctx->sd->rng->manual_seed(seed);
    int C                   = 4;
    int W                   = width / 8;
    int H                   = height / 8;
    struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, sd_vid_gen_params->video_frames);
    ggml_set_f32(x_t, 0.f);

    struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, sd_vid_gen_params->video_frames);
    ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

    LOG_INFO("sampling using %s method", sampling_methods_str[sd_vid_gen_params->sample_method]);
    struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                 x_t,
                                                 noise,
                                                 cond,
                                                 uncond,
                                                 {},
                                                 {},
                                                 0.f,
                                                 sd_vid_gen_params->guidance,
                                                 0.f,
                                                 sd_vid_gen_params->sample_method,
                                                 sigmas,
                                                 -1,
                                                 SDCondition(NULL, NULL, NULL));

    int64_t t2 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t2 - t1) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }

    struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, x_0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (img == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(sd_vid_gen_params->video_frames, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < sd_vid_gen_params->video_frames; i++) {
        auto img_i = ggml_view_3d(work_ctx, img, img->ne[0], img->ne[1], img->ne[2], img->nb[1], img->nb[2], img->nb[3] * i);

        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(img_i);
    }
    ggml_free(work_ctx);

    int64_t t3 = ggml_time_ms();

    LOG_INFO("img2vid completed in %.2fs", (t3 - t0) * 1.0f / 1000);

    return result_images;
}
