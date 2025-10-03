#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <inttypes.h>
#include <cinttypes>
#include <algorithm>
#include <filesystem>

#include "model_adapter.h"

std::string sd_load_merges();
std::string sd_load_t5();
std::string sd_load_umt5();
std::string sd_load_qwen2_merges();

#include "flux.hpp"
#include "stable-diffusion.cpp"
#include "util.cpp"
#include "upscaler.cpp"
#include "model.cpp"
#include "tokenize_util.cpp"
#include "zip.c"

#include "otherarch/utils.h"

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

//#define STB_IMAGE_IMPLEMENTATION //already defined in llava
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

// #define STB_IMAGE_RESIZE_IMPLEMENTATION //already defined in llava
#include "stb_image_resize.h"

#include "avi_writer.h"

static_assert((int)SD_TYPE_COUNT == (int)GGML_TYPE_COUNT,
              "inconsistency between SD_TYPE_COUNT and GGML_TYPE_COUNT");

struct SDParams {
    int n_threads = -1;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string stacked_id_embeddings_path;
    sd_type_t wtype = SD_TYPE_COUNT;

    std::string prompt;
    std::string negative_prompt;
    float cfg_scale   = 7.0f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;

    sample_method_t sample_method = EULER_A;
    int sample_steps              = 20;
    float strength                = 0.75f;
    int64_t seed                  = 42;
    bool clip_on_cpu              = false;
    bool diffusion_flash_attn     = false;
    bool diffusion_conv_direct    = false;
    bool vae_conv_direct          = false;

    bool chroma_use_dit_mask     = true;
};

//shared
int total_img_gens = 0;

//global static vars for SD
static SDParams * sd_params = nullptr;
static sd_ctx_t * sd_ctx = nullptr;
static int sddebugmode = 0;
static std::string recent_data = "";
static uint8_t * input_image_buffer = NULL;
static uint8_t * input_mask_buffer = NULL;
static std::vector<uint8_t *> input_extraimage_buffers;
const int max_extra_images = 4;

static std::string sdplatformenv, sddeviceenv, sdvulkandeviceenv;
static int cfg_tiled_vae_threshold = 0;
static int cfg_square_limit = 0;
static int cfg_side_limit = 0;
static bool sd_is_quiet = false;
static std::string sdmodelfilename = "";
static bool photomaker_enabled = false;

static bool is_vid_model = false;

static int get_loaded_sd_version(sd_ctx_t* ctx)
{
    return ctx->sd->version;
}

static bool loaded_model_is_chroma(sd_ctx_t* ctx)
{
    if (ctx != nullptr && ctx->sd != nullptr) {
        auto maybe_flux = std::dynamic_pointer_cast<FluxModel>(ctx->sd->diffusion_model);
        if (maybe_flux != nullptr) {
            return maybe_flux->flux.flux_params.is_chroma;
        }
    }
    return false;
}

static std::string read_str_from_disk(std::string filepath)
{
    std::string output;
    std::cout << "\nTry read vocab from " << filepath << std::endl;

    std::ifstream file(filepath);  // text mode
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    output.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

    return output;
}

std::string sd_load_merges()
{
    static std::string mergesstr;  // cached string
    if (!mergesstr.empty()) {
        return mergesstr;  // already loaded
    }
    std::string filepath = executable_path + "embd_res/merges_utf8_c_str.embd";
    mergesstr = read_str_from_disk(filepath);
    return mergesstr;
}
std::string sd_load_qwen2_merges()
{
    static std::string qwenmergesstr;  // cached string
    if (!qwenmergesstr.empty()) {
        return qwenmergesstr;  // already loaded
    }
    std::string filepath = executable_path + "embd_res/qwen2_merges_utf8_c_str.embd";
    qwenmergesstr = read_str_from_disk(filepath);
    return qwenmergesstr;
}
std::string sd_load_t5()
{
    static std::string t5str = "";
    if (!t5str.empty()) {
        return t5str;  // already loaded
    }
    std::string filepath = executable_path + "embd_res/t5_tokenizer_json.embd";
    t5str = read_str_from_disk(filepath);
    return t5str;
}
std::string sd_load_umt5()
{
    static std::string umt5str = "";
    if (!umt5str.empty()) {
        return umt5str;  // already loaded
    }
    std::string filepath = executable_path + "embd_res/umt5_tokenizer_json.embd";
    umt5str = read_str_from_disk(filepath);
    return umt5str;
}

bool sdtype_load_model(const sd_load_model_inputs inputs) {
    sd_is_quiet = inputs.quiet;
    set_sd_quiet(sd_is_quiet);
    executable_path = inputs.executable_path;
    std::string taesdpath = "";
    std::string lorafilename = inputs.lora_filename;
    std::string vaefilename = inputs.vae_filename;
    std::string t5xxl_filename = inputs.t5xxl_filename;
    std::string clip1_filename = inputs.clip1_filename;
    std::string clip2_filename = inputs.clip2_filename;
    std::string photomaker_filename = inputs.photomaker_filename;
    cfg_tiled_vae_threshold = inputs.tiled_vae_threshold;
    cfg_tiled_vae_threshold = (cfg_tiled_vae_threshold > 8192 ? 8192 : cfg_tiled_vae_threshold);
    cfg_tiled_vae_threshold = (cfg_tiled_vae_threshold <= 0 ? 8192 : cfg_tiled_vae_threshold); //if negative dont tile
    cfg_side_limit = inputs.img_hard_limit;
    cfg_square_limit = inputs.img_soft_limit;
    printf("\nImageGen Init - Load Model: %s\n",inputs.model_filename);

    if(lorafilename!="")
    {
        printf("With LoRA: %s at %f power\n",lorafilename.c_str(),inputs.lora_multiplier);
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
    if(inputs.quant > 0)
    {
        printf("Note: Loading a pre-quantized model is always faster than using compress weights!\n");
    }

    //duplicated from expose.cpp
    int cl_parseinfo = inputs.clblast_info; //first digit is whether configured, second is platform, third is devices
    std::string usingclblast = "GGML_OPENCL_CONFIGURED="+std::to_string(cl_parseinfo>0?1:0);
    putenv((char*)usingclblast.c_str());
    cl_parseinfo = cl_parseinfo%100; //keep last 2 digits
    int platform = cl_parseinfo/10;
    int devices = cl_parseinfo%10;
    sdplatformenv = "GGML_OPENCL_PLATFORM="+std::to_string(platform);
    sddeviceenv = "GGML_OPENCL_DEVICE="+std::to_string(devices);
    putenv((char*)sdplatformenv.c_str());
    putenv((char*)sddeviceenv.c_str());
    std::string vulkan_info_raw = inputs.vulkan_info;
    std::string vulkan_info_str = "";
    for (size_t i = 0; i < vulkan_info_raw.length(); ++i) {
        vulkan_info_str += vulkan_info_raw[i];
        if (i < vulkan_info_raw.length() - 1) {
            vulkan_info_str += ",";
        }
    }
    if(vulkan_info_str!="")
    {
        sdvulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
        putenv((char*)sdvulkandeviceenv.c_str());
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
    sd_params->taesd_path = taesdpath;
    sd_params->t5xxl_path = t5xxl_filename;
    sd_params->clip_l_path = clip1_filename;
    sd_params->clip_g_path = clip2_filename;
    sd_params->stacked_id_embeddings_path = photomaker_filename;
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
    params.taesd_path = sd_params->taesd_path.c_str();
    params.photo_maker_path = sd_params->stacked_id_embeddings_path.c_str();

    params.vae_decode_only = false;
    params.free_params_immediately = false;
    params.rng_type = CUDA_RNG;

    params.n_threads = sd_params->n_threads;
    params.wtype = sd_params->wtype;
    params.keep_clip_on_cpu = sd_params->clip_on_cpu;
    params.diffusion_flash_attn = sd_params->diffusion_flash_attn;
    params.diffusion_conv_direct = sd_params->diffusion_conv_direct;
    params.vae_conv_direct = sd_params->vae_conv_direct;
    params.chroma_use_dit_mask = sd_params->chroma_use_dit_mask;
    params.offload_params_to_cpu = inputs.offload_cpu;
    params.keep_vae_on_cpu = inputs.vae_cpu;
    params.keep_clip_on_cpu = inputs.clip_cpu;
    // params.flow_shift = 5.0f;

    if (params.chroma_use_dit_mask && params.diffusion_flash_attn) {
        // note we don't know yet if it's a Chroma model
        params.chroma_use_dit_mask = false;
    }

    if(inputs.debugmode==1)
    {
        std::stringstream ss;
        ss  << "\nMODEL:"      << params.model_path
            << "\nDIFFUSION:"  << params.diffusion_model_path
            << "\nVAE:"        << params.vae_path
            << "\nTAESD:"      << params.taesd_path
            << "\nPHOTOMAKER:" << params.photo_maker_path
            << "\nTHREADS:"    << params.n_threads
            << "\nWTYPE:"      << params.wtype
            << "\nDIFFUSIONFLASHATTN:"  << (params.diffusion_flash_attn ? 1 : 0)
            << "\nDIFFUSIONCONVDIRECT:" << (params.diffusion_conv_direct ? 1 : 0)
            << "\nVAECONVDIRECT:"       << (params.vae_conv_direct ? 1 : 0)
            << "\n";
        printf("%s", ss.str().c_str());
    }

    sd_ctx = new_sd_ctx(&params);

    if (sd_ctx == NULL) {
        printf("\nError: KCPP SD Failed to create context!\nIf using Flux/SD3.5, make sure you have ALL files required (e.g. VAE, T5, Clip...) or baked in!\n");
        printf("Otherwise, if you are using GGUF format, you can try the original .safetensors instead (Comfy GGUF not supported)\n");
        return false;
    }

    if (!sd_is_quiet) {
        if (loaded_model_is_chroma(sd_ctx) && sd_params->diffusion_flash_attn && sd_params->chroma_use_dit_mask)
        {
            printf("Chroma: flash attention is on, disabling DiT mask (this will lower image quality)\n");
            // disabled before loading
        }
    }

    auto loadedsdver = get_loaded_sd_version(sd_ctx);
    if (loadedsdver == SDVersion::VERSION_WAN2 || loadedsdver == SDVersion::VERSION_WAN2_2_I2V || loadedsdver == SDVersion::VERSION_WAN2_2_TI2V)
    {
        printf("\nVer %d, Setting to Video Generation Mode!\n",loadedsdver);
        is_vid_model = true;
    }

    std::filesystem::path mpath(inputs.model_filename);
    sdmodelfilename = mpath.filename().string();

    if(lorafilename!="" && inputs.lora_multiplier>0)
    {
        printf("\nApply LoRA...\n");
        sd_ctx->sd->apply_lora_from_file(lorafilename,inputs.lora_multiplier);
    }

    input_extraimage_buffers.reserve(max_extra_images);

    return true;
}

std::string clean_input_prompt(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    for (char ch : input) {
        // Check if the character is an ASCII or extended ASCII character
        if (static_cast<unsigned char>(ch) <= 0x7F || (ch >= 0xC2 && ch <= 0xF4)) {
            result.push_back(ch);
        }
    }
    //limit to max 800 chars
    result = result.substr(0, 800);
    return result;
}

static std::string get_image_params(const sd_img_gen_params_t & params) {
    std::stringstream parameter_string;
    parameter_string << std::setprecision(3)
        <<    "Prompt: " << params.prompt
        << " | NegativePrompt: " << params.negative_prompt
        << " | Steps: " << params.sample_params.sample_steps
        << " | CFGScale: " << params.sample_params.guidance.txt_cfg
        << " | Guidance: " << params.sample_params.guidance.distilled_guidance
        << " | Seed: " << params.seed
        << " | Size: " << params.width << "x" << params.height
        << " | Sampler: " << sd_sample_method_name(params.sample_params.sample_method)
        << " | Clip skip: " << params.clip_skip
        << " | Model: " << sdmodelfilename
        << " | Version: KoboldCpp";
    return parameter_string.str();
}

static inline int rounddown_64(int n) {
    return n - n % 64;
}

static inline int roundup_64(int n) {
    return ((n + 63) / 64) * 64;
}

static inline int roundnearest(int multiple, int n) {
    return ((n + (multiple/2)) / multiple) * multiple;
}

//scale dimensions to ensure width and height stay within limits
//img_hard_limit = sdclamped, hard size limit per side, no side can exceed this
//square limit = total NxN resolution based limit to also apply
static void sd_fix_resolution(int &width, int &height, int img_hard_limit, int img_soft_limit) {

    // sanitize the original values
    width = std::max(std::min(width, 8192), 64);
    height = std::max(std::min(height, 8192), 64);

    bool is_landscape = (width > height);
    int long_side = is_landscape ? width : height;
    int short_side = is_landscape ? height : width;
    float original_ratio = static_cast<float>(long_side) / short_side;

    // for the initial rounding, don't bother comparing to the original
    // requested ratio, since the user can choose those values directly
    long_side = rounddown_64(long_side);
    short_side = rounddown_64(short_side);
    img_hard_limit = rounddown_64(img_hard_limit);

    //enforce sdclamp side limit
    if (long_side > img_hard_limit) {
        short_side = static_cast<int>(short_side * img_hard_limit / static_cast<float>(long_side));
        long_side = img_hard_limit;
        if (short_side <= 64) {
            short_side = 64;
        } else {
            int down = rounddown_64(short_side);
            int up = roundup_64(short_side);
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

        if (new_short <= 64) {
            short_side = 64;
            long_side = rounddown_64(area_limit / short_side);
        } else {
            int new_long_down = rounddown_64(new_long);
            int new_short_down = rounddown_64(new_short);
            int new_short_up = roundup_64(new_short);
            int new_long_up = roundup_64(new_long);
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
    if(sampler=="euler a"||sampler=="k_euler_a"||sampler=="euler_a") //all lowercase
    {
        return sample_method_t::EULER_A;
    }
    else if(sampler=="euler"||sampler=="k_euler")
    {
        return sample_method_t::EULER;
    }
    else if(sampler=="heun"||sampler=="k_heun")
    {
        return sample_method_t::HEUN;
    }
    else if(sampler=="dpm2"||sampler=="k_dpm_2")
    {
        return sample_method_t::DPM2;
    }
    else if(sampler=="lcm"||sampler=="k_lcm")
    {
        return sample_method_t::LCM;
    }
    else if(sampler=="ddim")
    {
        return sample_method_t::DDIM_TRAILING;
    }
    else if(sampler=="dpm++ 2m karras" || sampler=="dpm++ 2m" || sampler=="k_dpmpp_2m")
    {
        return sample_method_t::DPMPP2M;
    }
    else
    {
        return sample_method_t::EULER_A;
    }
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

sd_generation_outputs sdtype_generate(const sd_generation_inputs inputs)
{
    sd_generation_outputs output;

    if(sd_ctx == nullptr || sd_params == nullptr)
    {
        printf("\nWarning: KCPP image generation not initialized!\n");
        output.data = "";
        output.animated = 0;
        output.status = 0;
        return output;
    }
    sd_image_t * results;

    //sanitize prompts, remove quotes and limit lengths
    std::string cleanprompt = clean_input_prompt(inputs.prompt);
    std::string cleannegprompt = clean_input_prompt(inputs.negative_prompt);
    std::string img2img_data = std::string(inputs.init_images);
    std::string img2img_mask = std::string(inputs.mask);
    std::vector<std::string> extra_image_data;
    for(int i=0;i<inputs.extra_images_len;++i)
    {
        extra_image_data.push_back(std::string(inputs.extra_images[i]));
    }

    sd_params->prompt = cleanprompt;
    sd_params->negative_prompt = cleannegprompt;
    sd_params->cfg_scale = inputs.cfg_scale;
    sd_params->sample_steps = inputs.sample_steps;
    sd_params->seed = inputs.seed;
    sd_params->width = inputs.width;
    sd_params->height = inputs.height;
    sd_params->strength = inputs.denoising_strength;
    sd_params->clip_skip = inputs.clip_skip;
    sd_params->sample_method = sampler_from_name(inputs.sample_method);

    bool is_img2img = img2img_data != "";

    auto loadedsdver = get_loaded_sd_version(sd_ctx);
    if (loadedsdver == SDVersion::VERSION_FLUX)
    {
        if (!loaded_model_is_chroma(sd_ctx) && sd_params->cfg_scale != 1.0f) {
            //non chroma clamp cfg scale
            if (!sd_is_quiet && sddebugmode) {
                printf("Flux: clamping CFG Scale to 1\n");
            }
            sd_params->cfg_scale = 1.0f;
        }
        if (sd_params->sample_method == sample_method_t::EULER_A) {
            //euler a broken on flux
            if (!sd_is_quiet && sddebugmode) {
                printf("%s: switching Euler A to Euler\n", loaded_model_is_chroma(sd_ctx) ? "Chroma" : "Flux");
            }
            sd_params->sample_method = sample_method_t::EULER;
        }
    }

    const int default_res_limit = 8192; // arbitrary, just to simplify the code
    // avoid crashes due to bugs/limitations on certain models
    // although it can be possible for a single side to exceed 1024, the total resolution of the image
    // cannot exceed (832x832) for sd1/sd2 or (1024x1024) for sdxl/sd3/flux, to prevent crashing the server
    const int hard_megapixel_res_limit = (loadedsdver==SDVersion::VERSION_SD1 || loadedsdver==SDVersion::VERSION_SD2)?832:1024;

    int img_hard_limit = default_res_limit;
    if (cfg_side_limit > 0) {
        img_hard_limit = std::max(std::min(cfg_side_limit, default_res_limit), 64);
    }

    int img_soft_limit = default_res_limit;
    if (cfg_square_limit > 0) {
        img_soft_limit = std::max(std::min(cfg_square_limit, default_res_limit), 64);
    }

    if (cfg_square_limit > 0 && sddebugmode == 1) {
        img_soft_limit = std::min(hard_megapixel_res_limit * 2, img_soft_limit);  //double the limit for debugmode if cfg_square_limit is set
    } else {
        img_soft_limit = std::min(hard_megapixel_res_limit, img_soft_limit);
    }

    sd_fix_resolution(sd_params->width, sd_params->height, img_hard_limit, img_soft_limit);
    if (inputs.width != sd_params->width || inputs.height != sd_params->height) {
        printf("\nKCPP SD: Requested dimensions %dx%d changed to %dx%d\n",
            inputs.width, inputs.height, sd_params->width, sd_params->height);
    }

    // trigger tiling by image area, the memory used for the VAE buffer is 6656 bytes per image pixel, default 768x768
    bool dotile = (sd_params->width*sd_params->height > cfg_tiled_vae_threshold*cfg_tiled_vae_threshold);

    //for img2img
    sd_image_t input_image = {0,0,0,nullptr};
    std::vector<sd_image_t> kontext_imgs;
    std::vector<sd_image_t> wan_imgs;
    std::vector<sd_image_t> photomaker_imgs;
    bool is_wan = (loadedsdver == SDVersion::VERSION_WAN2 || loadedsdver == SDVersion::VERSION_WAN2_2_I2V || loadedsdver == SDVersion::VERSION_WAN2_2_TI2V);
    bool is_kontext = (loadedsdver==SDVersion::VERSION_FLUX && !loaded_model_is_chroma(sd_ctx));

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
            if(is_wan)
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
            else if (is_kontext || photomaker_enabled)
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
                    if(is_kontext)
                    {
                        kontext_imgs.push_back(extraimage_reference);
                    }
                    else
                    {
                        photomaker_imgs.push_back(extraimage_reference);
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
            printf("\nImageGen References: Kontext=%d Wan=%d Photomaker=%d\n",kontext_imgs.size(),wan_imgs.size(),photomaker_imgs.size());
        }
    }

    sd_img_gen_params_t params = {};
    sd_img_gen_params_init (&params);

    params.batch_count = 1;

    params.prompt = sd_params->prompt.c_str();
    params.negative_prompt = sd_params->negative_prompt.c_str();
    params.clip_skip = sd_params->clip_skip;
    params.sample_params.guidance.txt_cfg = sd_params->cfg_scale;
    params.sample_params.guidance.img_cfg = sd_params->cfg_scale;
    params.width = sd_params->width;
    params.height = sd_params->height;
    params.sample_params.sample_method = sd_params->sample_method;
    params.sample_params.sample_steps = sd_params->sample_steps;
    params.seed = sd_params->seed;
    params.strength = sd_params->strength;
    params.vae_tiling_params.enabled = dotile;
    params.batch_count = 1;

    params.ref_images = kontext_imgs.data();
    params.ref_images_count = kontext_imgs.size();
    params.pm_params.id_images = photomaker_imgs.data();
    params.pm_params.id_images_count = photomaker_imgs.size();

    //the below params are only used in video models. May move into standalone object in future
    int vid_req_frames = inputs.vid_req_frames;
    int vid_req_avi = inputs.vid_req_avi;
    int generated_num_results = 1;

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
        if(wan_imgs.size()>0)
        {
            if(wan_imgs.size()>=1)
            {
                vid_gen_params.init_image = wan_imgs[0];
            }
            if(wan_imgs.size()>=2)
            {
                vid_gen_params.end_image = wan_imgs[1];
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
            << "\n\n";
            printf("%s", ss.str().c_str());
        }

        fflush(stdout);
        results = generate_video(sd_ctx, &vid_gen_params, &generated_num_results);
        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nRequested Vid Frames: %d, Generated Vid Frames: %d\n",vid_req_frames, generated_num_results);
        }
    }
    else if (!is_img2img)
    {
        if(!sd_is_quiet && sddebugmode==1)
        {
            std::stringstream ss;
            ss  << "\nTXT2IMG PROMPT:" << params.prompt
                << "\nNPROMPT:" << params.negative_prompt
                << "\nCLPSKP:" << params.clip_skip
                << "\nCFGSCLE:" << params.sample_params.guidance.txt_cfg
                << "\nSIZE:" << params.width << "x" << params.height
                << "\nSM:" << sd_sample_method_name(params.sample_params.sample_method)
                << "\nSTEP:" << params.sample_params.sample_steps
                << "\nSEED:" << params.seed
                << "\nBATCH:" << params.batch_count
                << "\n\n";
            printf("%s", ss.str().c_str());
        }

        fflush(stdout);

        results = generate_image(sd_ctx, &params);

    } else {

        if (params.width <= 0 || params.width % 64 != 0 || params.height <= 0 || params.height % 64 != 0) {
            printf("\nKCPP SD: bad request image dimensions!\n");
            output.data = "";
            output.animated = 0;
            output.status = 0;
            return output;
        }

        if(input_image_buffer!=nullptr) //just in time free old buffer
        {
             stbi_image_free(input_image_buffer);
             input_image_buffer = nullptr;
        }

        input_image_buffer = load_image_from_b64(img2img_data,nx,ny,img2imgW,img2imgH,3);

        if (!input_image_buffer) {
            printf("\nKCPP SD: load image from memory failed!\n");
            output.data = "";
            output.animated = 0;
            output.status = 0;
            return output;
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
            std::stringstream ss;
            ss  << "\nIMG2IMG PROMPT:" << params.prompt
                << "\nNPROMPT:" << params.negative_prompt
                << "\nCLPSKP:" << params.clip_skip
                << "\nCFGSCLE:" << params.sample_params.guidance.txt_cfg
                << "\nSIZE:" << params.width << "x" << params.height
                << "\nSM:" << sd_sample_method_name(params.sample_params.sample_method)
                << "\nSTEP:" << params.sample_params.sample_steps
                << "\nSEED:" << params.seed
                << "\nSTRENGTH:" << params.strength
                << "\nBATCH:" << params.batch_count
                << "\n\n";
            printf("%s", ss.str().c_str());
        }

        fflush(stdout);

        results = generate_image(sd_ctx, &params);

    }

    if (results == NULL) {
        printf("\nKCPP SD generate failed!\n");
        output.data = "";
        output.animated = 0;
        output.status = 0;
        return output;
    }

    bool wasanim = false;

    for (int i = 0; i < params.batch_count; i++) {
        if (results[i].data == NULL) {
            continue;
        }

        //if multiframe, make a video
        if(vid_req_frames>1 && generated_num_results>1 && is_vid_model)
        {
            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("\nSaving video buffer, AVI=%d...",vid_req_avi);
            }
            uint8_t * out_data = nullptr;
            size_t out_len = 0;
            int status = 0;
            wasanim = true;

            if(vid_req_avi==1)
            {
                status = create_mjpg_avi_membuf_from_sd_images(results, generated_num_results, 16, 40, &out_data,&out_len);
            }
            else
            {
                uint8_t * out_data_a = nullptr;
                uint8_t * out_data_b = nullptr;
                int status_a = 0;
                int status_b = 0;
                size_t out_len_a = 0;
                size_t out_len_b = 0;
                status_a = create_gif_buf_from_sd_images_gifh(results, generated_num_results, 16, &out_data_a,&out_len_a);
                status_b = create_gif_buf_from_sd_images_msf(results, generated_num_results, 16, &out_data_b,&out_len_b);
                if(!sd_is_quiet && sddebugmode==1)
                {
                    printf("GIF-H Len: %zu, MSF Len: %zu\n",out_len_a,out_len_b);
                }
                if(status_a==0 && out_len_a < out_len_b)
                {
                    free(out_data_b);
                    out_len = out_len_a;
                    out_data = out_data_a;
                    status = status_a;
                }
                else
                {
                    free(out_data_a);
                    out_len = out_len_b;
                    out_data = out_data_b;
                    status = status_b;
                }
            }

            if(!sd_is_quiet && sddebugmode==1)
            {
                if(status==0)
                {
                    printf("Video Saved (Len %zu)!\n",out_len);
                }else{
                    printf("Save Failed!\n");
                }

            }
            if(status==0)
            {
                recent_data = kcpp_base64_encode(out_data, out_len);
                free(out_data);
            }
        }
        else
        {
            int out_data_len;
            unsigned char * png = stbi_write_png_to_mem(results[i].data, 0, results[i].width, results[i].height, results[i].channel, &out_data_len, get_image_params(params).c_str());
            if (png != NULL)
            {
                recent_data = kcpp_base64_encode(png,out_data_len);
                free(png);
            }
        }

        free(results[i].data);
        results[i].data = NULL;
    }

    free(results);
    output.data = recent_data.c_str();
    output.animated = (wasanim?1:0);
    output.status = 1;
    total_img_gens += 1;
    return output;
}
