#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <mutex>
#include <random>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include <nlohmann/json.hpp>
#include <inttypes.h>
#include <cinttypes>
#include <algorithm>
#include <filesystem>

#include "otherarch/utils.h"
#include "model_adapter.h"

#include "stable-diffusion.h"
#include "src/kcpp_sd_extensions.h"
#include "src/core/util.h"
#include "ggml-backend.h"

using namespace kcpp_sd;

//#define STB_IMAGE_IMPLEMENTATION //already defined in llava
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

// #define STB_IMAGE_RESIZE_IMPLEMENTATION //already defined in llava
#include "stb_image_resize.h"

#include "avi_writer.h"

struct LoraMap {
    std::vector<std::pair<std::string, float>> items;
    std::unordered_map<std::string, std::size_t> index;

    void add_lora(const std::string& k, float v) {
        auto it = index.find(k);
        if (it == index.end()) {
            index[k] = items.size();
            items.emplace_back(k, v);
        } else {
            items[it->second].second += v;
        }
    }

    float check_small_mult(float mult) {
        if (mult > 1e-6 || mult < -1e-6)
            return mult;
        return 0.f;
    }

    float get_mult(const std::string& k) {
        auto lora = index.find(k);
        if (lora == index.end()) return 0.f;
        return check_small_mult(items[lora->second].second);
    }

    std::vector<sd_lora_t> get_lora_specs(bool include_zeroes = false) {
        std::vector<sd_lora_t> lora_specs;
        for (const auto & lora: items) {
            float multiplier = check_small_mult(lora.second);
            if (include_zeroes || multiplier != 0.f) {
                sd_lora_t spec = {};
                spec.path = lora.first.c_str();
                spec.multiplier = multiplier;
                lora_specs.push_back(spec);
            }
        }
        return lora_specs;
    }

    std::string get_lora_meta() {
        std::stringstream lora_meta;
        lora_meta << std::setprecision(6);
        for (const auto & lora: items) {
            float multiplier = check_small_mult(lora.second);
            if (multiplier != 0.f) {
                std::string lora_name = std::filesystem::path(lora.first).stem().string();
                lora_meta << "<lora:" << lora_name << ":" << multiplier << ">";
            }
        }
        return lora_meta.str();
    }

};


struct SDParams {
    int n_threads = -1;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string audio_vae_path;
    std::string taesd_path;
    std::string stacked_id_embeddings_path;
    sd_type_t wtype = SD_TYPE_COUNT;

    std::string prompt;
    std::string negative_prompt;
    float cfg_scale   = 7.0f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;

    sample_method_t sample_method = sample_method_t::SAMPLE_METHOD_COUNT;
    scheduler_t scheduler         = scheduler_t::SCHEDULER_COUNT;
    int sample_steps              = 20;
    float distilled_guidance      = -1.0f;
    float shifted_timestep        = 0;
    float flow_shift              = -1.0f;
    std::string extra_sample_args = "";
    float eta                     = -1.0f;
    float strength                = 0.75f;
    int64_t seed                  = 42;
    bool diffusion_flash_attn     = false;
    bool diffusion_conv_direct    = false;
    bool vae_conv_direct          = false;
    std::string ref_image_args    = "";

    LoraMap lora_map;
    bool lora_dynamic = false;

    std::string cache_mode;
    std::string cache_options;
};

//shared
int total_img_gens = 0;

//global static vars for SD
static SDParams * sd_params = nullptr;
static sd_ctx_t * sd_ctx = nullptr;
static upscaler_ctx_t* upscaler_ctx = nullptr;
static int sddebugmode = 0;
static uint8_t * input_image_buffer = NULL;
static uint8_t * input_mask_buffer = NULL;
static uint8_t * upscale_src_buffer = NULL;
static std::vector<uint8_t *> input_extraimage_buffers;
const int max_extra_images = 4;

static std::string sdmaingpuenv;
static int cfg_tiled_vae_threshold = 0;
static int cfg_square_limit = 0;
static int cfg_side_limit = 0;
static bool sd_is_quiet = false;
static bool photomaker_enabled = false;

static bool is_vid_model = false;
static bool remove_limits = false;

struct gendata_st {
    int status = 0;
    int step = 0;
    double step_time = 0.0;
    std::string preview;
};

struct {
    std::mutex mux;
    std::chrono::steady_clock::time_point start_time;
    int steps = 0;
    bool preview_requested = false;
    bool preview_enabled = false;
    gendata_st gendata;
} geninfo;

static struct {
    std::string data;
    std::string data_extra;
    std::string final_frame;
    std::string info;
    bool animated;
    void reset() {
        data = "";
        data_extra = "";
        final_frame = "";
        info = "{}";
        animated = false;
    }
    sd_generation_outputs outputs(int status) {
        sd_generation_outputs output;
        output.status = status;
        output.data = data.c_str();
        output.data_extra = data_extra.c_str();
        output.final_frame = final_frame.c_str();
        output.info = info.c_str();
        output.animated = animated;
        return output;
    }
    sd_generation_outputs error(const char* message) {
        reset();
        printf("\n%s\n", message);
        return outputs(0);
    }
} sd_generation;

static std::string read_str_from_disk(std::string filepath)
{
    std::string output;
    std::cout << "\nTry read vocab from " << filepath << std::endl;

    std::ifstream file(sd_get_u8path(filepath));  // text mode
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    output.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

    return output;
}

static std::string load_embd_file(std::string& cache, const char* filename)
{
    if (cache.empty()) {
        std::string filepath = executable_path + filename;
        cache = read_str_from_disk(filepath);
    }
    return cache;
}

std::string load_clip_merges()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/merges_utf8_c_str.embd");
}
std::string load_qwen2_merges()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/qwen2_merges_utf8_c_str.embd");
}
std::string load_gemma_merges()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/gemma_merges_utf8_c_str.embd");
}
std::string load_gemma_vocab_json()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/gemma_vocab_json.embd");
}
std::string load_gemma2_merges()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/gemma2_merges_utf8_c_str.embd");
}
std::string load_gemma2_vocab_json()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/gemma2_vocab_json.embd");
}
std::string load_mistral_merges()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/mistral2_merges_utf8_c_str.embd");
}
std::string load_mistral_vocab_json()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/mistral2_vocab_json.embd");
}
std::string load_t5_tokenizer_json()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/t5_tokenizer_json.embd");
}
std::string load_umt5_tokenizer_json()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/umt5_tokenizer_json.embd");
}
std::string load_gpt_oss_merges()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/gpt_oss_merges_utf8_c_str.embd");
}
std::string load_gpt_oss_vocab_json()
{
    static std::string cache;
    return load_embd_file(cache, "embd_res/gpt_oss_vocab_json.embd");
}

static void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data);

// 0 disable, 1 initial, 2 denoised
// the first noisy call is used to detect the inference phase
static void set_preview_images(int enable) {
    bool denoised = (enable == 2);
    bool noisy = (enable == 1);
    sd_set_preview_callback(step_callback, PREVIEW_PROJ, 1, denoised, noisy, nullptr);
}

static inline double get_time_delta(const std::chrono::steady_clock::time_point& start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

static void progress_callback(int step, int steps, float time, void* data)
{
    (void) data;
    const char* phase = "Encoding";
    {
        std::lock_guard<std::mutex> lock(geninfo.mux);
        if (geninfo.preview_requested && !geninfo.preview_enabled) {
            geninfo.preview_enabled = true;
            set_preview_images(2);
        }
        /* progress_callback is also called for model loading and VAE encoding,
        so ignore the step count unless step_callback signals we are diffusing */
        if (geninfo.gendata.status == 2) {
            /* adjust the max step count (may change for img2img) */
            geninfo.steps = steps;
            if (step != geninfo.gendata.step) {
                geninfo.gendata.step = step;
                geninfo.gendata.preview = "";
            }
            if (step == steps) {
                geninfo.gendata.status = 3;
            }
            phase = "Generating image";
        } else if (geninfo.gendata.status == 3) {
            phase = "Decoding";
        }
        /* let the terminal report tiling progress */
    }

    if(sd_is_quiet || step == 0) return;
    const char* unit = "s/it";
    float speed = time;
    if (speed < 1.0f && speed > 0.f) {
        speed = 1.0f / speed;
        unit  = "it/s";
    }
    if(step==1)
    {
        printf("\n");
    }
    printf("\r%s: %d/%d steps, %.2f %s\033[K%s", phase, step, steps, speed, unit, (step == steps || sddebugmode==1) ? "\n" : "");
    fflush(stdout);
}

static bool is_video_model(kcpp_sd::model_info info)
{
    return info.is_wan || info.is_ltx || info.is_minimaxh3;
}

bool sdtype_load_model(const sd_load_model_inputs inputs) {

    sd_is_quiet = inputs.quiet;
    set_sd_quiet(sd_is_quiet);
    executable_path = sd_get_u8path(inputs.executable_path);
    std::string taesdpath = "";
    LoraMap lora_map;
    for(int i=0;i<inputs.lora_len;++i)
    {
        lora_map.add_lora(inputs.lora_filenames[i], inputs.lora_multipliers[i]);
    }
    std::string vaefilename = inputs.vae_filename;
    std::string audiovaefilename = inputs.audio_vae_filename;
    std::string t5xxl_filename = inputs.t5xxl_filename;
    std::string clip1_filename = inputs.clip1_filename;
    std::string clip2_filename = inputs.clip2_filename;
    std::string photomaker_filename = inputs.photomaker_filename;
    std::string upscaler_filename = inputs.upscaler_filename;
    cfg_tiled_vae_threshold = inputs.tiled_vae_threshold;
    cfg_tiled_vae_threshold = (cfg_tiled_vae_threshold > 8192 ? 8192 : cfg_tiled_vae_threshold);
    cfg_tiled_vae_threshold = (cfg_tiled_vae_threshold <= 0 ? 8192 : cfg_tiled_vae_threshold); //if negative dont tile
    cfg_side_limit = inputs.img_hard_limit;
    cfg_square_limit = inputs.img_soft_limit;
    printf("\nImageGen Init - Load Model: %s\n",inputs.model_filename);

    std::string backend = inputs.backend ? inputs.backend : "";
    std::string params_backend = inputs.params_backend ? inputs.params_backend : "";
    std::string split_mode = inputs.split_mode ? inputs.split_mode : "";

    int lora_apply_mode = LORA_APPLY_AT_RUNTIME;
    bool lora_dynamic = false;
    bool lora_cache = false;
    if(inputs.lora_apply_mode >= 1 && inputs.lora_apply_mode <= 2) {
        lora_apply_mode = inputs.lora_apply_mode;
    }
    else {
        // bit 3: LoRAs can be changed dynamically
        // bit 4: cache the initial LoRA list in VRAM
        lora_dynamic = !!(inputs.lora_apply_mode & (1<<3));
        lora_cache   = lora_dynamic && !!(inputs.lora_apply_mode & (1<<4));
    }

    if(lora_map.items.size() > 0)
    {
        const char* lora_apply_mode_name = lora_apply_mode == 1 ? "immediately"
                                         : lora_apply_mode == 2 ? "at runtime"
                                         : "auto";
        const char * lora_dynamic_name = lora_dynamic ? ", dynamic" : "";
        const char * lora_cache_name = lora_cache ? ", with caching" : "";
        printf("With LoRAs in apply mode %s%s%s:\n", lora_apply_mode_name, lora_dynamic_name, lora_cache_name);
        for(auto lora: lora_map.items)
        {
            printf("  %s at %f power\n", lora.first.c_str(), lora.second);
        }
    }

    if(inputs.taesd)
    {
        taesdpath = executable_path + "embd_res/taesd.embd";
        printf("With TAE SD VAE: %s\n",taesdpath.c_str());
        if (cfg_tiled_vae_threshold < 8192) {
            printf("  disabling VAE tiling for TAESD\n");
            cfg_tiled_vae_threshold = 8192;
        }
    }
    else if(vaefilename!="")
    {
        printf("With Custom VAE: %s\n",vaefilename.c_str());
    }
    if(audiovaefilename!="")
    {
        printf("With Audio VAE: %s\n",audiovaefilename.c_str());
    }
    if(t5xxl_filename!="")
    {
        printf("With Custom T5-XXL Model: %s\n",t5xxl_filename.c_str());
    }
    if(clip1_filename!="")
    {
        printf("With Custom Clip-1 Model: %s\n",clip1_filename.c_str());
    }
    if(clip2_filename!="")
    {
        printf("With Custom Clip-2 Model: %s\n",clip2_filename.c_str());
    }
    if(photomaker_filename!="")
    {
        printf("With PhotoMaker Model: %s\n",photomaker_filename.c_str());
        photomaker_enabled = true;
    }
    if(upscaler_filename!="")
    {
        printf("With Upscaler Model: %s\n",upscaler_filename.c_str());
    }
    if(inputs.flash_attention)
    {
        printf("Flash Attention is enabled\n");
    }
    if(inputs.diffusion_conv_direct)
    {
        printf("Conv2D Direct for diffusion model is enabled\n");
    }
    if(inputs.vae_conv_direct)
    {
        printf("Conv2D Direct for VAE model is enabled\n");
    }
    if(backend != "")
    {
        printf("Backend assignment: \"%s\"\n", backend.c_str());
    }
    if (inputs.use_mmap && params_backend == "CPU") {
        printf("Offloading weights to system RAM with mmap\n");
        if (!lora_dynamic && inputs.lora_len > 0) {
            printf("Note: static LoRAs can reduce mmap memory savings!\n");
        }
    } else if (inputs.params_backend == "CPU") {
        printf("Offloading weights to system RAM\n");
    } else if (inputs.use_mmap) {
        printf("Using mmap for I/O\n");
    }
    if(inputs.auto_fit) {
        printf("Using auto-fit");
    }
    if(params_backend != "" && params_backend != "CPU") {
        printf("Parameters backend assignment: \"%s\"\n", params_backend.c_str());
    }
    if(split_mode != "") {
        printf("Using split mode: \"%s\"\n", split_mode.c_str());
    }
    std::string max_vram;
    if(inputs.max_vram && *inputs.max_vram) {
        max_vram = inputs.max_vram;
        printf("Using max VRAM = %s GB\n", max_vram.c_str());
    }
    if(inputs.quant > 0)
    {
        printf("Note: Loading a pre-quantized model is always faster than using compress weights!\n");
    }

    sd_params = new SDParams();
    sd_params->model_path = inputs.model_filename;
    sd_params->wtype = SD_TYPE_COUNT;
    if (inputs.quant > 0) {
        sd_params->wtype = (inputs.quant==1?SD_TYPE_Q8_0:SD_TYPE_Q4_0);
        printf("Diffusion Model quantized to %s\n", sd_type_name(sd_params->wtype));
    }
    sd_params->n_threads = inputs.threads; //if -1 use physical cores
    sd_params->diffusion_flash_attn = inputs.flash_attention;
    sd_params->diffusion_conv_direct = inputs.diffusion_conv_direct;
    sd_params->vae_conv_direct = inputs.vae_conv_direct;
    sd_params->vae_path = vaefilename;
    sd_params->audio_vae_path = audiovaefilename;
    sd_params->taesd_path = taesdpath;
    sd_params->t5xxl_path = t5xxl_filename;
    sd_params->clip_l_path = clip1_filename;
    sd_params->clip_g_path = clip2_filename;
    sd_params->stacked_id_embeddings_path = photomaker_filename;
    sd_params->lora_map = lora_map;
    sd_params->lora_dynamic = lora_dynamic;
    //if t5 is set, and model is a gguf, load it as a diffusion model path
    bool endswithgguf = (sd_params->model_path.rfind(".gguf") == sd_params->model_path.size() - 5);
    if((sd_params->t5xxl_path!="" || sd_params->clip_l_path!="" || sd_params->clip_g_path!="") && endswithgguf)
    {
        //extra check - make sure there is no diffusion model prefix already inside!
        if(!gguf_tensor_exists(sd_params->model_path,"model.diffusion_model.",false))
        {
            printf("\nSwap to Diffusion Model Path:%s",sd_params->model_path.c_str());
            sd_params->diffusion_model_path = sd_params->model_path;
            sd_params->model_path = "";
        }
    }

    sddebugmode = inputs.debugmode;

    set_sd_log_level(sddebugmode);

    sd_ctx_params_t params = {};
    sd_ctx_params_init(&params);

    params.model_path = sd_params->model_path.c_str();
    params.clip_l_path = sd_params->clip_l_path.c_str();
    params.clip_g_path = sd_params->clip_g_path.c_str();
    params.t5xxl_path = sd_params->t5xxl_path.c_str();
    params.diffusion_model_path = sd_params->diffusion_model_path.c_str();
    params.vae_path = sd_params->vae_path.c_str();
    params.audio_vae_path = sd_params->audio_vae_path.c_str();
    params.taesd_path = sd_params->taesd_path.c_str();
    params.photo_maker_path = sd_params->stacked_id_embeddings_path.c_str();

    params.rng_type = CUDA_RNG;

    params.n_threads = sd_params->n_threads;
    params.wtype = sd_params->wtype;
    params.diffusion_flash_attn = sd_params->diffusion_flash_attn;
    params.diffusion_conv_direct = sd_params->diffusion_conv_direct;
    params.vae_conv_direct = sd_params->vae_conv_direct;
    params.model_args = "chroma_use_dit_mask=true";
    params.max_vram = max_vram.c_str();
    params.stream_layers = inputs.stream_layers;
    params.eager_load = true; //kcpp should preload everything
    params.enable_mmap = inputs.use_mmap;
    params.backend = backend.c_str();
    params.params_backend = params_backend.c_str();
    params.split_mode = split_mode.c_str();
    params.auto_fit = inputs.auto_fit;
    params.lora_apply_mode = (lora_apply_mode_t)lora_apply_mode;

    // also switches flash attn for the vae and conditioner
    params.flash_attn = params.diffusion_flash_attn;

    if(inputs.debugmode==1)
    {
        char* buf = sd_ctx_params_to_str(&params);
        if(buf)
        {
            printf("\n%s\n", buf);
            free(buf);
        }
    }

    sd_ctx = new_sd_ctx(&params);

    if (sd_ctx == NULL) {
        printf("\nError: Image generation failed to setup!\nMake sure you have ALL files required (e.g. VAE, T5, Clip...) or baked into the model!\n");
        return false;
    }

    auto info = get_model_info(sd_ctx);

    if (is_video_model(info))
    {
        printf("\nSetting to Video Generation Mode!\n");
        is_vid_model = true;
    }

    // preload the LoRAs with the initial multipliers
    std::vector<sd_lora_t> lora_specs = sd_params->lora_map.get_lora_specs(lora_dynamic&& lora_cache);
    if(lora_specs.size()>0)
    {
        printf("  applying %zu LoRAs...\n", lora_specs.size());
        set_lora_cache(sd_ctx, lora_cache);
        apply_loras(sd_ctx, lora_specs);
        set_lora_cache(sd_ctx, false);
    }

    input_extraimage_buffers.reserve(max_extra_images);

    //load upscaler if provided
    if (upscaler_filename!="") {
        const int upscale_tile_size = 128;
        upscaler_ctx = new_upscaler_ctx(upscaler_filename.c_str(),
                                        params.diffusion_conv_direct,
                                        params.n_threads,
                                        upscale_tile_size,
                                        params.backend,
                                        params.params_backend);

        if (upscaler_ctx == nullptr) {
             printf("\nError: KCPP failed to load upscaler!\n");
        } else {
            printf("\nUpscaler has been loaded.\n");
        }
    }

    set_preview_images(0);

    sd_set_progress_callback(progress_callback, nullptr);

    return true;
}

static std::string friendly_model_name(std::filesystem::path model_path) {
    return model_path.filename().string();
}

static std::string get_scheduler_name(scheduler_t scheduler, bool as_sampler_suffix = false)
{
    if (scheduler == scheduler_t::SCHEDULER_COUNT) {
        return as_sampler_suffix ? "" : "default";
    } else {
        std::string prefix = as_sampler_suffix ? " " : "";
        return prefix + sd_scheduler_name(scheduler);
    }
}

static std::string get_image_params(const sd_img_gen_params_t & params, const std::string& lora_meta, int seed_offset) {
    std::string model = sd_params->model_path;
    if (model.empty())
        model = sd_params->diffusion_model_path;
    model = friendly_model_name(model);
    std::stringstream ss;
    ss << std::setprecision(3)
        <<    "Prompt: " << params.prompt << lora_meta
        << " | NegativePrompt: " << params.negative_prompt
        << " | Steps: " << params.sample_params.sample_steps
        << " | CFGScale: " << params.sample_params.guidance.txt_cfg
        << " | Guidance: " << params.sample_params.guidance.distilled_guidance
        << " | Seed: " << (params.seed + seed_offset)
        << " | Size: " << params.width << "x" << params.height
        << " | Sampler: " << sd_sample_method_name(params.sample_params.sample_method)
        << get_scheduler_name(params.sample_params.scheduler, true);
    if (params.sample_params.eta != -1.0f)
        ss << "| Eta: " << params.sample_params.eta;
    if (params.sample_params.shifted_timestep != 0)
        ss << "| Timestep Shift: " << params.sample_params.shifted_timestep;
    if (params.sample_params.flow_shift > 0.f && params.sample_params.flow_shift != INFINITY)
        ss << "| Flow Shift: " << params.sample_params.flow_shift;
    ss  << " | Clip skip: " << params.clip_skip
        << " | Model: " << model;
    if (sd_params->vae_path != "")
        ss << " | VAE: " << friendly_model_name(sd_params->vae_path);
    ss << " | Version: KoboldCpp";
    return ss.str();
}

static inline int rounddown_to(int n, int fac) {
    return n - n % fac;
}

static inline int roundup_to(int n, int fac) {
    return ((n + fac - 1) / fac) * fac;
}

const int img_side_min = 64;

//scale dimensions to ensure width and height stay within limits
//img_hard_limit = sdclamped, hard size limit per side, no side can exceed this
//square limit = total NxN resolution based limit to also apply
static void sd_fix_resolution(int &width, int &height, int img_hard_limit, int img_soft_limit, int spatial_multiple) {

    // sanitize the original values
    width = std::max(std::min(width, 8192), img_side_min);
    height = std::max(std::min(height, 8192), img_side_min);

    bool is_landscape = (width > height);
    int long_side = is_landscape ? width : height;
    int short_side = is_landscape ? height : width;
    float original_ratio = static_cast<float>(long_side) / short_side;

    // for the initial rounding, don't bother comparing to the original
    // requested ratio, since the user can choose those values directly
    long_side = rounddown_to(long_side, spatial_multiple);
    short_side = rounddown_to(short_side, spatial_multiple);
    img_hard_limit = rounddown_to(img_hard_limit, spatial_multiple);

    //enforce sdclamp side limit
    if (long_side > img_hard_limit) {
        short_side = static_cast<int>(short_side * img_hard_limit / static_cast<float>(long_side));
        long_side = img_hard_limit;
        if (short_side <= img_side_min) {
            short_side = img_side_min;
        } else {
            int down = rounddown_to(short_side, spatial_multiple);
            int up = roundup_to(short_side, spatial_multiple);
            float longf = static_cast<float>(long_side);
            // Choose better ratio match between rounding up or down
            short_side = (longf / down - original_ratio < original_ratio - longf / up) ? down : up;
        }
    }

    //enforce sd_restrict_square area limit
    int area_limit = img_soft_limit * img_soft_limit;
    if (long_side * short_side > area_limit) {
        float scale = std::sqrt(static_cast<float>(area_limit) / (long_side * short_side));
        int new_short = static_cast<int>(short_side * scale);
        int new_long = static_cast<int>(long_side * scale);

        if (new_short <= img_side_min) {
            short_side = img_side_min;
            long_side = rounddown_to(area_limit / short_side, spatial_multiple);
        } else {
            int new_long_down = rounddown_to(new_long, spatial_multiple);
            int new_short_down = rounddown_to(new_short, spatial_multiple);
            int new_short_up = roundup_to(new_short, spatial_multiple);
            int new_long_up = roundup_to(new_long, spatial_multiple);
            long_side = new_long_down;
            short_side = new_short_down;

            // we may get a ratio closer to the original if we still stay below the
            // limit when rounding up one of the dimensions, so check both cases
            float rdiff = std::fabs(static_cast<float>(new_long_down) / new_short_down - original_ratio);

            if (new_long_down * new_short_up < area_limit) {
                float newrdiff = std::fabs(static_cast<float>(new_long_down) / new_short_up - original_ratio);
                if (newrdiff < rdiff) {
                    long_side = new_long_down;
                    short_side = new_short_up;
                    rdiff = newrdiff;
                }
            }

            if (new_long_up * new_short_down < area_limit) {
                float newrdiff = std::fabs(static_cast<float>(new_long_up) / new_short_down - original_ratio);
                if (newrdiff < rdiff) {
                    long_side = new_long_up;
                    short_side = new_short_down;
                }
            }
        }
    }

    if (is_landscape) {
        width = long_side;
        height = short_side;
    } else {
        width = short_side;
        height = long_side;
    }
}

static enum sample_method_t sampler_from_name(const std::string& sampler)
{
    // all lowercase
    enum sample_method_t result = str_to_sample_method(sampler.c_str());
    if (result != sample_method_t::SAMPLE_METHOD_COUNT) {
        return result;
    } else {
        return sample_method_t::SAMPLE_METHOD_COUNT;
    }
}

uint8_t* resize_image(uint8_t * image_buffer, int& width, int& height, int expected_width = 0, int expected_height = 0, int expected_channel = 3)
{
    if ((expected_width > 0 && expected_height > 0) && (height != expected_height || width != expected_width)) {
        float dst_aspect = (float)expected_width / (float)expected_height;
        float src_aspect = (float)width / (float)height;

        int crop_x = 0, crop_y = 0;
        int crop_w = width, crop_h = height;

        if (src_aspect > dst_aspect) {
            crop_w = (int)(height * dst_aspect);
            crop_x = (width - crop_w) / 2;
        } else if (src_aspect < dst_aspect) {
            crop_h = (int)(width / dst_aspect);
            crop_y = (height - crop_h) / 2;
        }

        if (crop_x != 0 || crop_y != 0) {
            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("\ncrop input image from %dx%d to %dx%d\n", width, height, crop_w, crop_h);
            }
            uint8_t* cropped_image_buffer = (uint8_t*)malloc(crop_w * crop_h * expected_channel);
            if (cropped_image_buffer == NULL) {
                fprintf(stderr, "\nerror: allocate memory for crop\n");
                free(image_buffer);
                return NULL;
            }
            for (int row = 0; row < crop_h; row++) {
                uint8_t* src = image_buffer + ((crop_y + row) * width + crop_x) * expected_channel;
                uint8_t* dst = cropped_image_buffer + (row * crop_w) * expected_channel;
                memcpy(dst, src, crop_w * expected_channel);
            }

            width  = crop_w;
            height = crop_h;
            free(image_buffer);
            image_buffer = cropped_image_buffer;
        }

        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nresize input image from %dx%d to %dx%d\n", width, height, expected_width, expected_height);
        }
        int resized_height = expected_height;
        int resized_width  = expected_width;

        uint8_t* resized_image_buffer = (uint8_t*)malloc(resized_height * resized_width * expected_channel);
        if (resized_image_buffer == NULL) {
            fprintf(stderr, "\nerror: allocate memory for resize input image\n");
            free(image_buffer);
            return NULL;
        }
        stbir_resize(image_buffer, width, height, 0,
                     resized_image_buffer, resized_width, resized_height, 0, STBIR_TYPE_UINT8,
                     expected_channel, STBIR_ALPHA_CHANNEL_NONE, 0,
                     STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                     STBIR_COLORSPACE_SRGB, nullptr);
        width  = resized_width;
        height = resized_height;
        free(image_buffer);
        image_buffer = resized_image_buffer;
    }
    return image_buffer;
}

uint8_t* load_image_from_b64(const std::string & b64str, int& width, int& height, int expected_width = 0, int expected_height = 0, int expected_channel = 3)
{
    std::vector<uint8_t> decoded_buf = kcpp_base64_decode(b64str);
    int c = 0;
    uint8_t* image_buffer = (uint8_t*)stbi_load_from_memory(decoded_buf.data(), decoded_buf.size(), &width, &height, &c, expected_channel);

    if (image_buffer == NULL) {
        fprintf(stderr, "load_image_from_b64 failed\n");
        return NULL;
    }
    if (c < expected_channel) {
        fprintf(stderr, "load_image_from_b64: the number of channels for the input image must be >= %d, but got %d channels\n", expected_channel, c);
        free(image_buffer);
        return NULL;
    }
    if (width <= 0) {
        fprintf(stderr, "load_image_from_b64 error: the width of image must be greater than 0\n");
        free(image_buffer);
        return NULL;
    }
    if (height <= 0) {
        fprintf(stderr, "load_image_from_b64 error: the height of image must be greater than 0\n");
        free(image_buffer);
        return NULL;
    }

    // Resize input image ...
    image_buffer = resize_image(image_buffer,width,height,expected_width,expected_height,expected_channel);
    return image_buffer;
}

static enum scheduler_t scheduler_from_name(const char * scheduler)
{
    if (scheduler) {
        enum scheduler_t result = str_to_scheduler(scheduler);
        if (result != scheduler_t::SCHEDULER_COUNT)
        {
            return result;
        }
    }
    return scheduler_t::SCHEDULER_COUNT;
}

static void parse_cache_options(sd_cache_params_t & params, const std::string& cache_mode,
    const std::string& cache_options) {

    sd_cache_params_init(&params);
    if (cache_mode == "easycache") {
        params.mode = SD_CACHE_EASYCACHE;
    } else if (cache_mode == "ucache") {
        params.mode = SD_CACHE_UCACHE;
    } else if (cache_mode == "dbcache") {
        params.mode  = SD_CACHE_DBCACHE;
    } else if (cache_mode == "taylorseer") {
        params.mode  = SD_CACHE_TAYLORSEER;
    } else if (cache_mode == "cache-dit") {
        params.mode  = SD_CACHE_CACHE_DIT;
    } else if (cache_mode == "spectrum") {
        params.mode  = SD_CACHE_SPECTRUM;
    } else if (cache_mode != "" && cache_mode != "disabled") {
        printf("warning: unknown cache mode '%s'", cache_mode.c_str());
    }

    if (params.mode == SD_CACHE_DISABLED)
        return;

    if (cache_options == "")
        return;

    sd_cache_params_t cache_params = params;

    // from examples/common/common.hpp
    auto parse_named_params = [&](const std::string& opt_str) -> bool {
        std::stringstream ss(opt_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            size_t eq_pos = token.find('=');
            if (eq_pos == std::string::npos) {
                printf("error: cache option '%s' missing '=' separator", token.c_str());
                return false;
            }
            std::string key = token.substr(0, eq_pos);
            std::string val = token.substr(eq_pos + 1);
            try {
                if (key == "threshold") {
                    if (cache_mode == "easycache" || cache_mode == "ucache") {
                        cache_params.reuse_threshold = std::stof(val);
                    } else {
                        cache_params.residual_diff_threshold = std::stof(val);
                    }
                } else if (key == "start") {
                    cache_params.start_percent = std::stof(val);
                } else if (key == "end") {
                    cache_params.end_percent = std::stof(val);
                } else if (key == "decay") {
                    cache_params.error_decay_rate = std::stof(val);
                } else if (key == "relative") {
                    cache_params.use_relative_threshold = (std::stof(val) != 0.0f);
                } else if (key == "reset") {
                    cache_params.reset_error_on_compute = (std::stof(val) != 0.0f);
                } else if (key == "Fn" || key == "fn") {
                    cache_params.Fn_compute_blocks = std::stoi(val);
                } else if (key == "Bn" || key == "bn") {
                    cache_params.Bn_compute_blocks = std::stoi(val);
                } else if (key == "warmup") {
                    if (cache_mode == "spectrum") {
                        cache_params.spectrum_warmup_steps = std::stoi(val);
                    } else {
                        cache_params.max_warmup_steps = std::stoi(val);
                    }
                } else if (key == "w") {
                    cache_params.spectrum_w = std::stof(val);
                } else if (key == "m") {
                    cache_params.spectrum_m = std::stoi(val);
                } else if (key == "lam") {
                    cache_params.spectrum_lam = std::stof(val);
                } else if (key == "window") {
                    cache_params.spectrum_window_size = std::stoi(val);
                } else if (key == "flex") {
                    cache_params.spectrum_flex_window = std::stof(val);
                } else if (key == "stop") {
                    cache_params.spectrum_stop_percent = std::stof(val);
                } else {
                    printf("error: unknown cache parameter '%s'", key.c_str());
                    return false;
                }
            } catch (const std::exception&) {
                printf("error: invalid value '%s' for parameter '%s'", val.c_str(), key.c_str());
                return false;
            }
        }

        switch (cache_params.mode) {
            case SD_CACHE_EASYCACHE:
            case SD_CACHE_UCACHE:
                if (cache_params.reuse_threshold < 0.0f) {
                    printf("error: cache threshold must be non-negative");
                    return false;
                }
                if (cache_params.start_percent < 0.0f || cache_params.start_percent >= 1.0f ||
                    cache_params.end_percent <= 0.0f || cache_params.end_percent > 1.0f ||
                    cache_params.start_percent >= cache_params.end_percent) {
                    printf("error: cache start/end percents must satisfy 0.0 <= start < end <= 1.0");
                    return false;
                }
                break;
            default: ;
        }
        return true;
    };

    if (parse_named_params(cache_options)) {
        params = cache_params;
    }
}

static std::string raw_image_to_png_base64(const sd_image_t& img, std::string parameters = "") {
    std::string result;
    int out_data_len = 0;
    unsigned char * png = stbi_write_png_to_mem(img.data, 0, img.width, img.height, img.channel, &out_data_len, parameters != "" ? parameters.c_str() : nullptr);
    if (png != NULL) {
        result = kcpp_base64_encode(png,out_data_len);
        free(png);
    }
    return result;
}

static sd_audio_t load_audio_from_b64(const std::string& b64audio) {
    sd_audio_t audio = {0, 0, 0, nullptr};
    if (b64audio.empty()) {
        return audio;
    }

    std::vector<uint8_t> audio_data = kcpp_base64_decode(b64audio);
    std::vector<float> decoded_samples;
    int sample_rate = 0;
    int channels = 0;
    if (!kcpp_decode_audio_file_from_buf(audio_data.data(), audio_data.size(), sample_rate, channels, decoded_samples)) {
        printf("KCPP SD: failed to decode input audio\n");
        return audio;
    }

    uint64_t sample_count = static_cast<uint64_t>(decoded_samples.size() / static_cast<size_t>(channels));
    size_t float_count = decoded_samples.size();
    float* samples = (float*)malloc(float_count * sizeof(float));
    if (samples == nullptr || sample_count == 0) {
        free(samples);
        return audio;
    }

    std::memcpy(samples, decoded_samples.data(), float_count * sizeof(float));
    audio.sample_rate = static_cast<uint32_t>(sample_rate);
    audio.channels = static_cast<uint32_t>(channels);
    audio.sample_count = sample_count;
    audio.data = samples;
    return audio;
}

static bool supports_reference_images(kcpp_sd::model_info info)
{
    bool supported = (is_video_model(info) || info.supports_ref_image || info.is_kontext || photomaker_enabled) && !info.is_zimage;
    return supported;
}

static std::string upscale_image_to_png_base64(upscaler_ctx_t* upscaler_ctx, const sd_image_t& input_image, int upscale_factor = 2, const std::string& meta_image_info = "")
{
    std::string gen_data;
    sd_image_t* upscaled = nullptr;
    int upscaled_count = 0;
    if (upscale(upscaler_ctx, input_image, upscale_factor, &upscaled, &upscaled_count)) {
        gen_data = raw_image_to_png_base64(*upscaled, meta_image_info);
        free_sd_images(upscaled, upscaled_count);
    } else {
        printf("Upscaling failed!\n");
        gen_data = raw_image_to_png_base64(input_image, meta_image_info);
    }
    return gen_data;
}

void sdtype_abort_generation() {
    sd_cancel_generation(sd_ctx, SD_CANCEL_ALL);
}

sd_generation_outputs sdtype_generate(const sd_generation_inputs inputs)
{
    struct CleanupInfoOnExit {
        ~CleanupInfoOnExit() {
            {
                std::lock_guard<std::mutex> lock(geninfo.mux);
                geninfo.gendata.status = 0;
                geninfo.preview_requested = false;
                geninfo.preview_enabled = false;
            }
            set_preview_images(0);
        }
    } cleanup_info_on_exit;

    if(sd_ctx == nullptr || sd_params == nullptr)
    {
        return sd_generation.error("Warning: KCPP image generation not initialized!");
    }

    {
        std::lock_guard<std::mutex> lock(geninfo.mux);
        geninfo.start_time = std::chrono::steady_clock::now();
        geninfo.steps = inputs.sample_steps;
        geninfo.preview_requested = false;
        geninfo.preview_enabled = false;
        geninfo.gendata = {};
        geninfo.gendata.status = 1;
        set_preview_images(1);
    }

    sd_image_t * results = nullptr;
    int generated_num_results = 0;

    std::string img2img_data = std::string(inputs.init_images);
    std::string img2img_mask = std::string(inputs.mask);
    std::string input_audio_data = std::string(inputs.audio_data ? inputs.audio_data : "");
    std::vector<std::string> extra_image_data;
    for(int i=0;i<inputs.extra_images_len;++i)
    {
        extra_image_data.push_back(std::string(inputs.extra_images[i]));
    }
    sd_params->prompt = inputs.prompt;
    sd_params->negative_prompt = inputs.negative_prompt;
    sd_params->cfg_scale = inputs.cfg_scale;
    sd_params->distilled_guidance = inputs.distilled_guidance;
    sd_params->sample_steps = inputs.sample_steps;
    sd_params->shifted_timestep = inputs.shifted_timestep;
    sd_params->flow_shift = inputs.flow_shift;
    sd_params->extra_sample_args = inputs.extra_sample_args ? inputs.extra_sample_args : "";
    bool force_image_edit = false;
    sd_params->ref_image_args = "resize_before_vae=on"; // auto_resize_ref_image = true;
    if (inputs.ref_image_args && *inputs.ref_image_args) {
        sd_params->ref_image_args += ",";
        sd_params->ref_image_args += inputs.ref_image_args;
        if (sd_params->ref_image_args.find("preset") != std::string::npos) {
            force_image_edit = true;
            if(!sd_is_quiet && sddebugmode==1) {
                printf("ref_image_args=\"%s\", forcing edit mode", inputs.ref_image_args);
            }
        }
    }
    sd_params->eta = inputs.eta;
    sd_params->seed = inputs.seed;
    sd_params->width = inputs.width;
    sd_params->height = inputs.height;
    sd_params->strength = inputs.denoising_strength;
    sd_params->clip_skip = inputs.clip_skip;
    sd_params->sample_method = sampler_from_name(inputs.sample_method);
    sd_params->scheduler = scheduler_from_name(inputs.scheduler);

    if (sd_params->sample_method == sample_method_t::SAMPLE_METHOD_COUNT) {
        sd_params->sample_method = sd_get_default_sample_method(sd_ctx);
    }

    sd_params->cache_mode    = inputs.cache_mode ? inputs.cache_mode : "";
    sd_params->cache_options = inputs.cache_options ? inputs.cache_options : "";

    auto info = get_model_info(sd_ctx);
    bool is_img2img = img2img_data != "";

    if (info.is_flux1)
    {
        if (!info.is_chroma && sd_params->cfg_scale != 1.0f) {
            //non chroma clamp cfg scale
            if (!sd_is_quiet && sddebugmode) {
                printf("Flux: clamping CFG Scale to 1\n");
            }
            sd_params->cfg_scale = 1.0f;
        }
    }

    if(!remove_limits && info.is_zimage)
    {
        if(sd_params->cfg_scale > 4.0f)
        {
            if (!sd_is_quiet && sddebugmode) {
                printf("Z-Image: clamping CFG Scale to 4.0 to preserve quality\n");
            }
            sd_params->cfg_scale = 4.0f;
        }
    }

    if(info.is_sdxs)
    {
        if(sd_params->cfg_scale > 1.0f || sd_params->sample_steps > 1)
        {
            if (!sd_is_quiet && sddebugmode) {
                printf("SDXS: clamping steps and cfg to 1\n");
            }
            sd_params->cfg_scale = 1.0f;
            sd_params->sample_steps = 1;
        }
    }

    //if a single extra image is provided, mask is NOT provided, and img2img image is NOT provided
    //and it's a (SD1.5, SDXL) model that doesn't support extra images (see extra_image_data later)
    //swap extra image data into img2img instead (graceful fallback)
    if(!supports_reference_images(info) && !force_image_edit && extra_image_data.size()==1 && !is_img2img && img2img_mask=="")
    {
        is_img2img = true;
        img2img_data = extra_image_data[0];
        extra_image_data.clear();
        if (!sd_is_quiet && sddebugmode == 1) {
            printf("Switching reference image to img2img\n");
        }
    }

    if (is_video_model(info) && extra_image_data.size() == 0 && is_img2img)
    {
        extra_image_data.push_back(img2img_data);
    }

    // limit by image side
    int img_hard_limit = 8192; // "large enough", just to simplify the code
    if (cfg_side_limit > 0) {
        img_hard_limit = std::max(std::min(cfg_side_limit, img_hard_limit), img_side_min);
    }

    // limit by image area: avoid crashes due to bugs/limitations on certain models
    // a single side can be larger, but width*height are limited by img_soft_limit²
    int img_soft_limit;
    int hard_megapixel_res_limit = 2048; // hard area limit, no matter the config
    if (cfg_square_limit <= 0) {
        // default limit is model dependent: ~0.66 megapixel for SD1.5/SD2, 1 megapixel for most models
        img_soft_limit = (info.is_sd1 || info.is_sd2)?832:1024;
    } else {
        // force img_side_min <= limit <= hard_megapixel_res_limit
        img_soft_limit = std::max(std::min(cfg_square_limit, hard_megapixel_res_limit), img_side_min);
    }

    // unet is limited to multiples of 64; dit models vary
    int spatial_multiple = info.spatial_multiple;

    sd_fix_resolution(sd_params->width, sd_params->height, img_hard_limit, img_soft_limit, spatial_multiple);
    if (inputs.width != sd_params->width || inputs.height != sd_params->height) {
        printf("\nKCPP SD: Requested dimensions %dx%d changed to %dx%d\n",
            inputs.width, inputs.height, sd_params->width, sd_params->height);
    }

    // trigger tiling by image area, the memory used for the VAE buffer is 6656 bytes per image pixel, default 768x768
    bool dotile = (sd_params->width*sd_params->height > cfg_tiled_vae_threshold*cfg_tiled_vae_threshold);

    int vae_tile_size = -1;
    if (dotile) {
        int new_vae_tile_size = cfg_tiled_vae_threshold / info.vae_scale_factor;
        new_vae_tile_size = new_vae_tile_size / 2;
        new_vae_tile_size -= new_vae_tile_size % 2;
        if (new_vae_tile_size > vae_tile_size) {
            vae_tile_size = new_vae_tile_size;
        }
    }

    //for img2img
    sd_image_t input_image = {0,0,0,nullptr};
    std::vector<sd_image_t> reference_imgs;
    std::vector<sd_image_t> wan_imgs;
    std::vector<sd_image_t> photomaker_imgs;

    int nx, ny, nc;
    int img2imgW = sd_params->width; //for img2img input
    int img2imgH = sd_params->height;
    int img2imgC = 3; // Assuming RGB image

    std::string ts = get_timestamp_str();
    if(!sd_is_quiet)
    {
        printf("\n[%s] Generating Image (%d steps)\n",ts.c_str(),inputs.sample_steps);
    }else{
        printf("\n[%s] Generating (%d st.)\n",ts.c_str(),inputs.sample_steps);
    }

    fflush(stdout);

    if(extra_image_data.size()>0)
    {
        if(input_extraimage_buffers.size()>0) //just in time free old buffer
        {
            for(int i=0;i<input_extraimage_buffers.size();++i)
            {
                stbi_image_free(input_extraimage_buffers[i]);
            }
            input_extraimage_buffers.clear();
        }
        for(int i=0;i<extra_image_data.size() && i<max_extra_images;++i)
        {
            int nx2, ny2, nc2;
            int desiredchannels = 3;
            if(supports_reference_images(info)||force_image_edit)
            {
                if(is_video_model(info))
                {
                    uint8_t * loaded = load_image_from_b64(extra_image_data[i],nx2,ny2,img2imgW,img2imgH,3);
                    if(loaded)
                    {
                        input_extraimage_buffers.push_back(loaded);
                        sd_image_t extraimage_reference;
                        extraimage_reference.width = nx2;
                        extraimage_reference.height = ny2;
                        extraimage_reference.channel = desiredchannels;
                        extraimage_reference.data = loaded;
                        wan_imgs.push_back(extraimage_reference);
                    }
                }
                else if(info.supports_ref_image||force_image_edit)
                {
                    uint8_t * loaded = load_image_from_b64(extra_image_data[i],nx2,ny2);
                    if(loaded)
                    {
                        //kcpp fix: qwen image can stack overflow and crash when ref images exceed
                        // a total res of 512x512 = 262144, so we downscale if that's the case
                        // kcpp edit 2mar2026: this seems to be better now, so limit to 1024x1024 instead
                        int tgtx = nx2;
                        int tgty = ny2;
                        int res_lim_crash = 1024 * 1024;
                        if (nx2 * ny2 > res_lim_crash)
                        {
                            float factor = sqrtf((float)res_lim_crash / ((float)nx2 * (float)ny2));
                            tgtx = (int)(nx2 * factor);
                            tgty = (int)(ny2 * factor);
                            if (!sd_is_quiet && sddebugmode == 1)
                            {
                                printf("\nResized RefImg %dx%d to %dx%d", nx2, ny2, tgtx, tgty);
                            }
                            loaded = resize_image(loaded, nx2, ny2, tgtx, tgty);
                        }
                        if(loaded)
                        {
                            input_extraimage_buffers.push_back(loaded);
                            sd_image_t extraimage_reference;
                            extraimage_reference.width = nx2;
                            extraimage_reference.height = ny2;
                            extraimage_reference.channel = desiredchannels;
                            extraimage_reference.data = loaded;
                            reference_imgs.push_back(extraimage_reference);
                        }
                    }
                }
                else if (info.is_kontext || photomaker_enabled)
                {
                    uint8_t * loaded = load_image_from_b64(extra_image_data[i],nx2,ny2);
                    if(loaded)
                    {
                        input_extraimage_buffers.push_back(loaded);
                        sd_image_t extraimage_reference;
                        extraimage_reference.width = nx2;
                        extraimage_reference.height = ny2;
                        extraimage_reference.channel = desiredchannels;
                        extraimage_reference.data = loaded;
                        if(info.is_kontext)
                        {
                            reference_imgs.push_back(extraimage_reference);
                        }
                        else
                        {
                            photomaker_imgs.push_back(extraimage_reference);
                        }
                    }
                }
            }
        }

        //ensure prompt has img keyword, otherwise append it
        if(photomaker_enabled)
        {
            if (sd_params->prompt.find("img") == std::string::npos) {
                sd_params->prompt += " img";
            } else if (sd_params->prompt.rfind("img", 0) == 0) {
                // "img" found at the start of the string (position 0), which is not allowed. Add some text before it
                sd_params->prompt = "person " + sd_params->prompt;
            }
        }

        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nImageGen References: RefImg=%zu Wan=%zu Photomaker=%zu\n",reference_imgs.size(),wan_imgs.size(),photomaker_imgs.size());
        }
    }

    sd_img_gen_params_t params = {};
    sd_img_gen_params_init (&params);
    params.batch_count = 1;
    params.ref_image_args = sd_params->ref_image_args.c_str();
    params.prompt = sd_params->prompt.c_str();
    params.negative_prompt = sd_params->negative_prompt.c_str();
    params.clip_skip = sd_params->clip_skip;
    params.sample_params.guidance.txt_cfg = sd_params->cfg_scale;
    // params.sample_params.guidance.img_cfg = sd_params->cfg_scale; //removed, breaks qwen img edit and more
    if (sd_params->distilled_guidance >= 0.f) {
        params.sample_params.guidance.distilled_guidance = sd_params->distilled_guidance;
    }
    params.width = sd_params->width;
    params.height = sd_params->height;
    params.sample_params.sample_method = sd_params->sample_method;
    params.sample_params.scheduler = sd_params->scheduler;
    params.sample_params.sample_steps = sd_params->sample_steps;
    params.sample_params.shifted_timestep = sd_params->shifted_timestep;
    if (sd_params->eta >= 0.f && sd_params->eta <= 1.f) {
        params.sample_params.eta = sd_params->eta;
    }
    if (sd_params->flow_shift > 0.f && sd_params->flow_shift != INFINITY) {
        params.sample_params.flow_shift = sd_params->flow_shift;
    }
    params.sample_params.extra_sample_args = sd_params->extra_sample_args.c_str();
    params.seed = sd_params->seed;
    params.strength = sd_params->strength;
    params.vae_tiling_params.enabled = dotile;
    if (vae_tile_size > 0) {
        params.vae_tiling_params.tile_size_x = vae_tile_size;
        params.vae_tiling_params.tile_size_y = vae_tile_size;
    }
    if(dotile)
    {
        params.vae_tiling_params.temporal_tiling = true;
    }
    parse_cache_options(params.cache, sd_params->cache_mode, sd_params->cache_options);
    params.circular_x = inputs.circular_x;
    params.circular_y = inputs.circular_y;

    LoraMap lora_map = sd_params->lora_map;
    if (sd_params->lora_dynamic) {
        for (int i = 0; i < inputs.lora_len; i++) {
            std::string path = inputs.lora_filenames[i];
            float preloaded_mult = sd_params->lora_map.get_mult(path);
            lora_map.add_lora(path, inputs.lora_multipliers[i]);
        }
    }

    std::vector<sd_lora_t> lora_specs = lora_map.get_lora_specs();
    std::string lora_meta = lora_map.get_lora_meta();

    if(!sd_is_quiet && sddebugmode==1) {
        if (lora_specs.size() > 0) {
            printf("Applying LoRAs:\n");
            for(size_t i=0;i<lora_specs.size();++i)
            {
                printf("  %s @ %.3f\n", lora_specs[i].path, lora_specs[i].multiplier);
            }
        }
    }

    // note sdcpp tracks previously applied LoRAs and weights,
    // and apply/unapply the differences at each gen
    params.loras = lora_specs.data();
    params.lora_count = lora_specs.size();

    params.ref_images = reference_imgs.data();
    params.ref_images_count = reference_imgs.size();
    params.pm_params.id_images = photomaker_imgs.data();
    params.pm_params.id_images_count = photomaker_imgs.size();

    //the below params are only used in video models. May move into standalone object in future
    int vid_req_frames = inputs.vid_req_frames;
    int video_output_type = inputs.video_output_type;
    int vid_fps = inputs.vid_fps;
    remove_limits = inputs.remove_limits;

     //special case, is img2img and denoise strength is 0 and steps is 1, do a passthru
    bool is_passthrough = (sd_params->sample_steps<=1 && sd_params->strength<=0 && is_img2img && vid_req_frames<=1 && extra_image_data.size()==0);
    sd_audio_t* generated_audio = nullptr;
    sd_audio_t input_audio = {0, 0, 0, nullptr};

    if(is_vid_model)
    {
        std::vector<sd_image_t> control_frames; //empty for now
        sd_vid_gen_params_t vid_gen_params = {};
        sd_vid_gen_params_init (&vid_gen_params);
        vid_gen_params.prompt = params.prompt;
        vid_gen_params.negative_prompt = params.negative_prompt;
        vid_gen_params.clip_skip = params.clip_skip;
        vid_gen_params.control_frames = control_frames.data();
        vid_gen_params.control_frames_size = (int)control_frames.size();
        vid_gen_params.width = params.width;
        vid_gen_params.height = params.height;
        vid_gen_params.sample_params = params.sample_params;
        vid_gen_params.strength = params.strength;
        vid_gen_params.seed = params.seed;
        vid_gen_params.video_frames = vid_req_frames;
        vid_gen_params.fps = vid_fps;
        vid_gen_params.vae_tiling_params = params.vae_tiling_params;
        if (!input_audio_data.empty()) {
            input_audio = load_audio_from_b64(input_audio_data);
            if (input_audio.data == nullptr) {
                return sd_generation.error("KCPP SD: load audio from base64 failed!");
            }
            vid_gen_params.input_audio = &input_audio;
        }
        if (wan_imgs.size() > 0) {
            if (wan_imgs.size() >= 2) {
                vid_gen_params.init_image = wan_imgs[0];
                vid_gen_params.end_image  = wan_imgs[1];
            } else if (wan_imgs.size() == 1) {
                if (inputs.reverse_refimg) {
                    vid_gen_params.end_image = wan_imgs[0];
                } else {
                    vid_gen_params.init_image = wan_imgs[0];
                }
            }
        }
        if(!sd_is_quiet && sddebugmode==1)
        {
            std::stringstream ss;
            ss  << "\nVID PROMPT:" << vid_gen_params.prompt
            << "\nNPROMPT:"   << vid_gen_params.negative_prompt
            << "\nCLPSKP:"   << vid_gen_params.clip_skip
            << "\nSIZE:"     << vid_gen_params.width << "x" << vid_gen_params.height
            << "\nSTEP:"     << vid_gen_params.sample_params.sample_steps
            << "\nSEED:"     << vid_gen_params.seed
            << "\nSTRENGTH:" << vid_gen_params.strength
            << "\nFRAMES:"   << vid_gen_params.video_frames
            << "\nCTRL_FRM:" << vid_gen_params.control_frames_size
            << "\nINIT_IMGS:" << wan_imgs.size()
            << "\nINPUT_AUDIO:" << (vid_gen_params.input_audio ? "true" : "false")
            << "\n\n";
            printf("%s", ss.str().c_str());
        }

        fflush(stdout);

        results = nullptr;
        if (!generate_video(sd_ctx, &vid_gen_params, &results, &generated_num_results, &generated_audio)) {
            results = nullptr;
            generated_audio = nullptr;
        }
        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nRequested Vid Frames: %d, Generated Vid Frames: %d\n",vid_req_frames, generated_num_results);
        }
    }
    else if (!is_img2img)
    {
        if(!sd_is_quiet && sddebugmode==1)
        {
            char* buf = sd_img_gen_params_to_str(&params);
            if(buf)
            {
                printf("\n%s\n", buf);
                free(buf);
            }
        }

        fflush(stdout);

        if (!generate_image(sd_ctx, &params, &results, &generated_num_results)) {
            results = nullptr;
            generated_num_results = 0;
        }

    } else {

        if(input_image_buffer!=nullptr) //just in time free old buffer
        {
             stbi_image_free(input_image_buffer);
             input_image_buffer = nullptr;
        }

        input_image_buffer = load_image_from_b64(img2img_data,nx,ny,img2imgW,img2imgH,3);

        if (!input_image_buffer) {
            return sd_generation.error("KCPP SD: load image from memory failed!");
        }

        if(img2img_mask!="")
        {
            int nx2, ny2, nc2;
            if(input_mask_buffer!=nullptr) //just in time free old buffer
            {
                stbi_image_free(input_mask_buffer);
                input_mask_buffer = nullptr;
            }
            input_mask_buffer = load_image_from_b64(img2img_mask,nx2,ny2,img2imgW,img2imgH,1);

            if(inputs.flip_mask)
            {
                int bufsiz = nx2 * ny2 * 1; //1 channel
                for (int i = 0; i < bufsiz; ++i) {
                    input_mask_buffer[i] = 255 - input_mask_buffer[i];
                }
            }
        }

        input_image.width = img2imgW;
        input_image.height = img2imgH;
        input_image.channel = img2imgC;
        input_image.data = input_image_buffer;

        uint8_t* mask_image_buffer    = NULL;
        std::vector<uint8_t> default_mask_image_vec(img2imgW * img2imgH * img2imgC, 255);
        if (img2img_mask != "") {
            mask_image_buffer = input_mask_buffer;
        } else {
            mask_image_buffer = default_mask_image_vec.data();
        }
        sd_image_t mask_image = { (uint32_t) img2imgW, (uint32_t) img2imgH, 1, mask_image_buffer };

        params.init_image = input_image;
        params.mask_image = mask_image;

        if(!sd_is_quiet && sddebugmode==1)
        {
            char* buf = sd_img_gen_params_to_str(&params);
            if(buf)
            {
                printf("\n%s\n", buf);
                free(buf);
            }
        }

        fflush(stdout);

        if (is_passthrough) {
            printf("No generation triggered, passthrough mode.\n");
        } else {
            if (!generate_image(sd_ctx, &params, &results, &generated_num_results)) {
                results = nullptr;
                generated_num_results = 0;
            }
        }
    }

    if (!is_passthrough && results == NULL) {
        if (input_audio.data) {
            free(input_audio.data);
            input_audio.data = nullptr;
        }
        return sd_generation.error("KCPP SD generate failed!");
    }

    bool isanim = (vid_req_frames>1 && generated_num_results>1 && is_vid_model);
    nlohmann::json jsoninfo = nlohmann::json::object();
    if (!isanim) {
        jsoninfo["prompt"] = params.prompt + lora_meta;
        if (*params.negative_prompt)
            jsoninfo["negative_prompt"] = params.negative_prompt;
        jsoninfo["seed"] = params.seed;
        jsoninfo["cfg_scale"] = params.sample_params.guidance.txt_cfg;
        jsoninfo["width"] = params.width;
        jsoninfo["height"] = params.height;
        jsoninfo["steps"] = params.sample_params.sample_steps;
        jsoninfo["sampler_name"] = sd_sample_method_name(params.sample_params.sample_method);
        if (params.clip_skip > 0)
            jsoninfo["clip_skip"] = params.clip_skip;
        jsoninfo["extra_generation_params"] = nlohmann::json::object();
        if (params.sample_params.scheduler != scheduler_t::SCHEDULER_COUNT)
            jsoninfo["extra_generation_params"]["Schedule type"] = get_scheduler_name(params.sample_params.scheduler);
        if (params.sample_params.eta >= 0 && params.sample_params.eta <= 1)
            jsoninfo["eta"] = params.sample_params.eta;
        if (is_img2img)
            jsoninfo["denoising_strength"] = params.strength;
        if (sd_params->model_path.empty())
            jsoninfo["sd_model_name"] = friendly_model_name(sd_params->diffusion_model_path);
        else
            jsoninfo["sd_model_name"] = friendly_model_name(sd_params->model_path);
        if (sd_params->vae_path != "")
            jsoninfo["sd_vae_name"] = friendly_model_name(sd_params->vae_path);
        jsoninfo["infotexts"] = nlohmann::json::array();
        jsoninfo["all_prompts"] = nlohmann::json::array();
        jsoninfo["all_negative_prompts"] = nlohmann::json::array();
        jsoninfo["all_seeds"] = nlohmann::json::array();
        jsoninfo["lora_meta"] = lora_meta;
        jsoninfo["version"] = "KoboldCpp";
    }
    sd_image_t* upscaled_image = nullptr;
    std::string gen_data;
    std::string gen_data2;
    std::string final_frame_data;

    if (is_passthrough)
    {
        if(inputs.upscale && upscaler_ctx != nullptr)
        {
            printf("Upscaling original image (passthrough)...\n");
            gen_data = upscale_image_to_png_base64(upscaler_ctx, input_image, 2);
        }
        else {
            gen_data = raw_image_to_png_base64(input_image);
        }
    }
    else if (isanim)
    {
        //if multiframe, make a video
        if (generated_num_results > 0 && results && results->data)
        {
            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("\nSaving video buffer, VIDEO_OUTPUT_TYPE=%d...",video_output_type);
            }
            uint8_t * out_data = nullptr;
            uint8_t * out_data2 = nullptr;
            size_t out_len = 0;
            size_t out_len2 = 0;
            int status = 0;
            int status2 = 0;

            if(video_output_type==0 || video_output_type==2)
            {
                status = create_gif_buf_from_sd_images_msf(results, generated_num_results, vid_fps, &out_data,&out_len);
            }
            if(video_output_type==1 || video_output_type==2)
            {
                status2 = create_mjpg_avi_membuf_from_sd_images(results, generated_num_results, vid_fps, 40, &out_data2,&out_len2, generated_audio);
            }

            if(generated_num_results>1)
            {
                sd_image_t *final_frame_image = &results[generated_num_results-1];
                final_frame_data = raw_image_to_png_base64(*final_frame_image);
            }

            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("Video Output Sizes: GIF=%zu AVI=%zu\n",out_len,out_len2);
                if(status==0 && status2==0)
                {
                    printf("Video(s) Saved (Len %zu)!\n",out_len);
                } else {
                    printf("Save Failed!\n");
                }
            }
            if(status==0 && out_len>0)
            {
                gen_data = kcpp_base64_encode(out_data, out_len);
                free(out_data);
            }
            if (status2 == 0 && out_len2 > 0) {
                if (gen_data == "") {
                    gen_data = kcpp_base64_encode(out_data2, out_len2);
                } else {
                    gen_data2 = kcpp_base64_encode(out_data2, out_len2);
                }
                free(out_data2);
            }
        }
        free_sd_images(results, generated_num_results);
    }
    else
    {
        for (int i = 0; i < generated_num_results; i++)
        {
            sd_image_t& result_image = results[i];
            if (result_image.data == NULL) {
                continue;
            }
            std::string meta_image_info = get_image_params(params, lora_meta, i);
            if(inputs.upscale && upscaler_ctx != nullptr)
            {
                printf("Upscaling output image...\n");
                gen_data = upscale_image_to_png_base64(upscaler_ctx, result_image, 2, meta_image_info);
            } else {
                gen_data = raw_image_to_png_base64(result_image, meta_image_info);
            }
            jsoninfo["infotexts"][i] = meta_image_info;
            jsoninfo["all_seeds"][i] = params.seed + i;
            jsoninfo["all_prompts"][i] = params.prompt;
            jsoninfo["all_negative_prompts"][i] = params.negative_prompt;
        }
        free_sd_images(results, generated_num_results);
        results = nullptr;
    }

    if (generated_audio) {
        free_sd_audio(generated_audio);
        generated_audio = nullptr;
    }
    if (input_audio.data) {
        free(input_audio.data);
        input_audio.data = nullptr;
    }

    total_img_gens += 1;
    if(!sd_is_quiet)
    {
        std::string ts = get_timestamp_str();
        printf("[%s] Generating Media Complete\n",ts.c_str());
    }

    sd_generation.data = gen_data;
    sd_generation.data_extra = gen_data2;
    sd_generation.final_frame = final_frame_data;
    sd_generation.animated = isanim;
    sd_generation.info = jsoninfo.dump();
    return sd_generation.outputs(1);
}

static void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data)
{
    std::string preview;
    if (image != nullptr) {
        if (frame_count == 1) {
            preview = raw_image_to_png_base64(*image);
        } else {
            uint8_t * out_data = nullptr;
            size_t out_len = 0;
            if (create_gif_buf_from_sd_images_msf(image, frame_count, 16, &out_data,&out_len) == 0 && out_data && out_len > 0) {
                preview = kcpp_base64_encode(out_data, out_len);
            }
            if (out_data) {
                free(out_data);
            }
        }
    }
    double step_time = get_time_delta(geninfo.start_time);

    std::lock_guard<std::mutex> lock(geninfo.mux);
    if (geninfo.gendata.status <= 1) {
        geninfo.gendata.status = 2;
        if (geninfo.preview_requested && !geninfo.preview_enabled) {
            geninfo.preview_enabled = true;
        }
        if (geninfo.preview_enabled) {
            set_preview_images(2);
        } else {
            set_preview_images(0);
        }
    }
    geninfo.gendata.step = step;
    geninfo.gendata.step_time = step_time;
    geninfo.gendata.preview = preview;
}

void sdtype_request_ongoing_generation_preview()
{
    std::lock_guard<std::mutex> lock(geninfo.mux);
    if (geninfo.gendata.status != 0 && !geninfo.preview_requested) {
        geninfo.preview_requested = true;
    }
}

sd_generation_outputs sdtype_upscale(const sd_upscale_inputs inputs)
{
    sd_generation.reset();

    if(sd_ctx == nullptr || upscaler_ctx == nullptr || sd_params == nullptr)
    {
        return sd_generation.error("Warning: KCPP image upscaling not initialized!");
    }

    std::string rawb64 = inputs.init_images;
    int nx, ny;
    if(upscale_src_buffer!=nullptr) //just in time free old buffer
    {
        stbi_image_free(upscale_src_buffer);
        upscale_src_buffer = nullptr;
    }
    upscale_src_buffer = load_image_from_b64(rawb64,nx,ny);
    sd_image_t source_img;
    source_img.data = nullptr;
    std::string result;
    if(upscale_src_buffer)
    {
        source_img.width = nx;
        source_img.height = ny;
        source_img.channel = 3;
        source_img.data = upscale_src_buffer;

        result = upscale_image_to_png_base64(upscaler_ctx, source_img, inputs.upscaling_resize);
    }

    if (result == "") {
        return sd_generation.error("Warning: KCPP failed to upscale image");
    }

    sd_generation.data = result;
    return sd_generation.outputs(1);
}

sd_info_outputs sdtype_get_info()
{
    using json = nlohmann::json;
    json j;

    auto available_schedulers = json::array();
    available_schedulers.push_back("default");
    for (int i = 0; i < scheduler_t::SCHEDULER_COUNT; i++) {
        std::string name = sd_scheduler_name((scheduler_t)i);
        if (name != "NONE") {
            available_schedulers.push_back(name);
        }
    }
    j["available_schedulers"] = available_schedulers;

    auto available_samplers = json::array();
    available_samplers.push_back("default");
    for (int i = 0; i < sample_method_t::SAMPLE_METHOD_COUNT; i++) {
        std::string name = sd_sample_method_name((sample_method_t)i);
        if (name != "NONE") {
            available_samplers.push_back(name);
        }
    }
    j["available_samplers"] = available_samplers;

    auto get_dev_type_name = [](auto dev_type) -> std::string {
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU)
            return "CPU";
        else if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU)
            return "GPU";
        else if (dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU)
            return "IGPU";
        return "TYPE_" + std::to_string(dev_type);
    };

    auto devices = json::array();
    size_t dev_count = ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; ++i) {
        auto dev = ggml_backend_dev_get(i);
        json jdev;
        jdev["name"]        = ggml_backend_dev_name(dev);
        jdev["description"] = ggml_backend_dev_description(dev);
        jdev["type"]        = get_dev_type_name(ggml_backend_dev_type(dev));
        devices.push_back(jdev);
    }
    j["devices"] = devices;

    static std::string recent_info = j.dump();
    sd_info_outputs output;
    output.status = 0;
    output.data = recent_info.c_str();
    return output;
}

sd_info_outputs sdtype_get_ongoing_generation_info()
{
    double elapsed_time = 0.0;
    int steps = 0;
    gendata_st gendata;
    {
        std::lock_guard<std::mutex> lock(geninfo.mux);
        gendata = geninfo.gendata;
        steps = geninfo.steps;
        if (gendata.status != 0) {
            elapsed_time = get_time_delta(geninfo.start_time);
        }
    }

    nlohmann::json j;
    if (gendata.status == 0) {
        j["status"] = "idle";
    } else {
        if (gendata.status == 1)
            j["status"] = "conditioning";
        else if (gendata.status == 2)
            j["status"] = "diffusing";
        else if (gendata.status == 3)
            j["status"] = "decoding";
        else
            j["status"] = "UNKNOWN";
        j["steps"] = steps;
        j["elapsed_time"] = elapsed_time;
        j["step"] = gendata.step;
        j["step_time"] = gendata.step_time;
        j["preview"] = gendata.preview;
    }

    static thread_local std::string recent_info;
    recent_info = j.dump();
    sd_info_outputs output;
    output.status = 0;
    output.data = recent_info.c_str();
    return output;
}

