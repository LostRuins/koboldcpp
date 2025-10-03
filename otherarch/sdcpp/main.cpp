#include <stdio.h>
#include <string.h>
#include <time.h>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <vector>

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
//#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#include "avi_writer.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif  // _WIN32

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

namespace fs = std::filesystem;

const char* modes_str[] = {
    "img_gen",
    "vid_gen",
    "convert",
};
#define SD_ALL_MODES_STR "img_gen, vid_gen, convert"

enum SDMode {
    IMG_GEN,
    VID_GEN,
    CONVERT,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = IMG_GEN;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string qwen2vl_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string control_net_path;
    std::string embedding_dir;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string init_image_path;
    std::string end_image_path;
    std::string mask_image_path;
    std::string control_image_path;
    std::vector<std::string> ref_image_paths;
    std::string control_video_path;
    bool increase_ref_index = false;

    std::string prompt;
    std::string negative_prompt;
    int clip_skip   = -1;  // <= 0 represents unspecified
    int width       = 512;
    int height      = 512;
    int batch_count = 1;

    std::vector<int> skip_layers = {7, 8, 9};
    sd_sample_params_t sample_params;

    std::vector<int> high_noise_skip_layers = {7, 8, 9};
    sd_sample_params_t high_noise_sample_params;

    float moe_boundary  = 0.875f;
    int video_frames    = 1;
    int fps             = 16;
    float vace_strength = 1.f;

    float strength             = 0.75f;
    float control_strength     = 0.9f;
    rng_type_t rng_type        = CUDA_RNG;
    int64_t seed               = 42;
    bool verbose               = false;
    bool offload_params_to_cpu = false;
    bool control_net_cpu       = false;
    bool clip_on_cpu           = false;
    bool vae_on_cpu            = false;
    bool diffusion_flash_attn  = false;
    bool diffusion_conv_direct = false;
    bool vae_conv_direct       = false;
    bool canny_preprocess      = false;
    bool color                 = false;
    int upscale_repeats        = 1;

    // Photo Maker
    std::string photo_maker_path;
    std::string pm_id_images_dir;
    std::string pm_id_embed_path;
    float pm_style_strength = 20.f;

    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask  = false;
    int chroma_t5_mask_pad   = 1;
    float flow_shift         = INFINITY;

    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};

    SDParams() {
        sd_sample_params_init(&sample_params);
        sd_sample_params_init(&high_noise_sample_params);
        high_noise_sample_params.sample_steps = -1;
    }
};

void print_params(SDParams params) {
    char* sample_params_str            = sd_sample_params_to_str(&params.sample_params);
    char* high_noise_sample_params_str = sd_sample_params_to_str(&params.high_noise_sample_params);
    printf("Option: \n");
    printf("    n_threads:                         %d\n", params.n_threads);
    printf("    mode:                              %s\n", modes_str[params.mode]);
    printf("    model_path:                        %s\n", params.model_path.c_str());
    printf("    wtype:                             %s\n", params.wtype < SD_TYPE_COUNT ? sd_type_name(params.wtype) : "unspecified");
    printf("    clip_l_path:                       %s\n", params.clip_l_path.c_str());
    printf("    clip_g_path:                       %s\n", params.clip_g_path.c_str());
    printf("    clip_vision_path:                  %s\n", params.clip_vision_path.c_str());
    printf("    t5xxl_path:                        %s\n", params.t5xxl_path.c_str());
    printf("    qwen2vl_path:                      %s\n", params.qwen2vl_path.c_str());
    printf("    diffusion_model_path:              %s\n", params.diffusion_model_path.c_str());
    printf("    high_noise_diffusion_model_path:   %s\n", params.high_noise_diffusion_model_path.c_str());
    printf("    vae_path:                          %s\n", params.vae_path.c_str());
    printf("    taesd_path:                        %s\n", params.taesd_path.c_str());
    printf("    esrgan_path:                       %s\n", params.esrgan_path.c_str());
    printf("    control_net_path:                  %s\n", params.control_net_path.c_str());
    printf("    embedding_dir:                     %s\n", params.embedding_dir.c_str());
    printf("    photo_maker_path:                  %s\n", params.photo_maker_path.c_str());
    printf("    pm_id_images_dir:                  %s\n", params.pm_id_images_dir.c_str());
    printf("    pm_id_embed_path:                  %s\n", params.pm_id_embed_path.c_str());
    printf("    pm_style_strength:                 %.2f\n", params.pm_style_strength);
    printf("    output_path:                       %s\n", params.output_path.c_str());
    printf("    init_image_path:                   %s\n", params.init_image_path.c_str());
    printf("    end_image_path:                    %s\n", params.end_image_path.c_str());
    printf("    mask_image_path:                   %s\n", params.mask_image_path.c_str());
    printf("    control_image_path:                %s\n", params.control_image_path.c_str());
    printf("    ref_images_paths:\n");
    for (auto& path : params.ref_image_paths) {
        printf("        %s\n", path.c_str());
    };
    printf("    control_video_path:                %s\n", params.control_video_path.c_str());
    printf("    increase_ref_index:                %s\n", params.increase_ref_index ? "true" : "false");
    printf("    offload_params_to_cpu:             %s\n", params.offload_params_to_cpu ? "true" : "false");
    printf("    clip_on_cpu:                       %s\n", params.clip_on_cpu ? "true" : "false");
    printf("    control_net_cpu:                   %s\n", params.control_net_cpu ? "true" : "false");
    printf("    vae_on_cpu:                        %s\n", params.vae_on_cpu ? "true" : "false");
    printf("    diffusion flash attention:         %s\n", params.diffusion_flash_attn ? "true" : "false");
    printf("    diffusion Conv2d direct:           %s\n", params.diffusion_conv_direct ? "true" : "false");
    printf("    vae_conv_direct:                   %s\n", params.vae_conv_direct ? "true" : "false");
    printf("    control_strength:                  %.2f\n", params.control_strength);
    printf("    prompt:                            %s\n", params.prompt.c_str());
    printf("    negative_prompt:                   %s\n", params.negative_prompt.c_str());
    printf("    clip_skip:                         %d\n", params.clip_skip);
    printf("    width:                             %d\n", params.width);
    printf("    height:                            %d\n", params.height);
    printf("    sample_params:                     %s\n", SAFE_STR(sample_params_str));
    printf("    high_noise_sample_params:          %s\n", SAFE_STR(high_noise_sample_params_str));
    printf("    moe_boundary:                      %.3f\n", params.moe_boundary);
    printf("    flow_shift:                        %.2f\n", params.flow_shift);
    printf("    strength(img2img):                 %.2f\n", params.strength);
    printf("    rng:                               %s\n", sd_rng_type_name(params.rng_type));
    printf("    seed:                              %zd\n", params.seed);
    printf("    batch_count:                       %d\n", params.batch_count);
    printf("    vae_tiling:                        %s\n", params.vae_tiling_params.enabled ? "true" : "false");
    printf("    upscale_repeats:                   %d\n", params.upscale_repeats);
    printf("    chroma_use_dit_mask:               %s\n", params.chroma_use_dit_mask ? "true" : "false");
    printf("    chroma_use_t5_mask:                %s\n", params.chroma_use_t5_mask ? "true" : "false");
    printf("    chroma_t5_mask_pad:                %d\n", params.chroma_t5_mask_pad);
    printf("    video_frames:                      %d\n", params.video_frames);
    printf("    vace_strength:                     %.2f\n", params.vace_strength);
    printf("    fps:                               %d\n", params.fps);
    free(sample_params_str);
    free(high_noise_sample_params_str);
}

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -M, --mode [MODE]                  run mode, one of: [img_gen, vid_gen, convert], default: img_gen\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1)\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  --offload-to-cpu                   place the weights in RAM to save VRAM, and automatically load them into VRAM when needed\n");
    printf("  -m, --model [MODEL]                path to full model\n");
    printf("  --diffusion-model                  path to the standalone diffusion model\n");
    printf("  --high-noise-diffusion-model       path to the standalone high noise diffusion model\n");
    printf("  --clip_l                           path to the clip-l text encoder\n");
    printf("  --clip_g                           path to the clip-g text encoder\n");
    printf("  --clip_vision                      path to the clip-vision encoder\n");
    printf("  --t5xxl                            path to the t5xxl text encoder\n");
    printf("  --qwen2vl                          path to the qwen2vl text encoder\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)\n");
    printf("  --control-net [CONTROL_PATH]       path to control net model\n");
    printf("  --embd-dir [EMBEDDING_PATH]        path to embeddings\n");
    printf("  --upscale-model [ESRGAN_PATH]      path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now\n");
    printf("  --upscale-repeats                  Run the ESRGAN upscaler this many times (default 1)\n");
    printf("  --type [TYPE]                      weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K)\n");
    printf("                                     If not specified, the default is the type of the weight file\n");
    printf("  --tensor-type-rules [EXPRESSION]   weight type per tensor pattern (example: \"^vae\\.=f16,model\\.=q8_0\")\n");
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  -i, --init-img [IMAGE]             path to the init image, required by img2img\n");
    printf("  --mask [MASK]                      path to the mask image, required by img2img with mask\n");
    printf("  -i, --end-img [IMAGE]              path to the end image, required by flf2v\n");
    printf("  --control-image [IMAGE]            path to image condition, control net\n");
    printf("  -r, --ref-image [PATH]             reference image for Flux Kontext models (can be used multiple times) \n");
    printf("  --control-video [PATH]             path to control video frames, It must be a directory path.\n");
    printf("                                     The video frames inside should be stored as images in lexicographical (character) order\n");
    printf("                                     For example, if the control video path is `frames`, the directory contain images such as 00.png, 01.png, … etc.\n");
    printf("  --increase-ref-index               automatically increase the indices of references images based on the order they are listed (starting with 1).\n");
    printf("  -o, --output OUTPUT                path to write result image to (default: ./output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       the negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)\n");
    printf("  --img-cfg-scale SCALE              image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)\n");
    printf("  --guidance SCALE                   distilled guidance scale for models with guidance input (default: 3.5)\n");
    printf("  --slg-scale SCALE                  skip layer guidance (SLG) scale, only for DiT models: (default: 0)\n");
    printf("                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium\n");
    printf("  --eta SCALE                        eta in DDIM, only for DDIM and TCD: (default: 0)\n");
    printf("  --skip-layers LAYERS               Layers to skip for SLG steps: (default: [7,8,9])\n");
    printf("  --skip-layer-start START           SLG enabling point: (default: 0.01)\n");
    printf("  --skip-layer-end END               SLG disabling point: (default: 0.2)\n");
    printf("  --scheduler {discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple} Denoiser sigma scheduler (default: discrete)\n");
    printf("  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd}\n");
    printf("                                     sampling method (default: \"euler\" for Flux/SD3/Wan, \"euler_a\" otherwise)\n");
    printf("  --timestep-shift N                 shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant\n");
    printf("  --steps  STEPS                     number of sample steps (default: 20)\n");
    printf("  --high-noise-cfg-scale SCALE       (high noise) unconditional guidance scale: (default: 7.0)\n");
    printf("  --high-noise-img-cfg-scale SCALE   (high noise) image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)\n");
    printf("  --high-noise-guidance SCALE        (high noise) distilled guidance scale for models with guidance input (default: 3.5)\n");
    printf("  --high-noise-slg-scale SCALE       (high noise) skip layer guidance (SLG) scale, only for DiT models: (default: 0)\n");
    printf("                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium\n");
    printf("  --high-noise-eta SCALE             (high noise) eta in DDIM, only for DDIM and TCD: (default: 0)\n");
    printf("  --high-noise-skip-layers LAYERS    (high noise) Layers to skip for SLG steps: (default: [7,8,9])\n");
    printf("  --high-noise-skip-layer-start      (high noise) SLG enabling point: (default: 0.01)\n");
    printf("  --high-noise-skip-layer-end END    (high noise) SLG disabling point: (default: 0.2)\n");
    printf("  --high-noise-scheduler {discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple} Denoiser sigma scheduler (default: discrete)\n");
    printf("  --high-noise-sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd}\n");
    printf("                                     (high noise) sampling method (default: \"euler_a\")\n");
    printf("  --high-noise-steps  STEPS          (high noise) number of sample steps (default: -1 = auto)\n");
    printf("                                     SLG will be enabled at step int([STEPS]*[START]) and disabled at int([STEPS]*[END])\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)\n");
    printf("                                     1.0 corresponds to full destruction of information in init image\n");
    printf("  -H, --height H                     image height, in pixel space (default: 512)\n");
    printf("  -W, --width W                      image width, in pixel space (default: 512)\n");
    printf("  --rng {std_default, cuda}          RNG (default: cuda)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)\n");
    printf("  -b, --batch-count COUNT            number of images to generate\n");
    printf("  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)\n");
    printf("                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x\n");
    printf("  --vae-tiling                       process vae in tiles to reduce memory usage\n");
    printf("  --vae-tile-size [X]x[Y]            tile size for vae tiling (default: 32x32)\n");
    printf("  --vae-relative-tile-size [X]x[Y]   relative tile size for vae tiling, in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides --vae-tile-size)\n");
    printf("  --vae-tile-overlap OVERLAP         tile overlap for vae tiling, in fraction of tile size (default: 0.5)\n");
    printf("  --vae-on-cpu                       keep vae in cpu (for low vram)\n");
    printf("  --clip-on-cpu                      keep clip in cpu (for low vram)\n");
    printf("  --diffusion-fa                     use flash attention in the diffusion model (for low vram)\n");
    printf("                                     Might lower quality, since it implies converting k and v to f16.\n");
    printf("                                     This might crash if it is not supported by the backend.\n");
    printf("  --diffusion-conv-direct            use Conv2d direct in the diffusion model\n");
    printf("                                     This might crash if it is not supported by the backend.\n");
    printf("  --vae-conv-direct                  use Conv2d direct in the vae model (should improve the performance)\n");
    printf("                                     This might crash if it is not supported by the backend.\n");
    printf("  --control-net-cpu                  keep controlnet in cpu (for low vram)\n");
    printf("  --canny                            apply canny preprocessor (edge detection)\n");
    printf("  --color                            colors the logging tags according to level\n");
    printf("  --chroma-disable-dit-mask          disable dit mask for chroma\n");
    printf("  --chroma-enable-t5-mask            enable t5 mask for chroma\n");
    printf("  --chroma-t5-mask-pad  PAD_SIZE     t5 mask pad size of chroma\n");
    printf("  --video-frames                     video frames (default: 1)\n");
    printf("  --fps                              fps (default: 24)\n");
    printf("  --moe-boundary BOUNDARY            timestep boundary for Wan2.2 MoE model. (default: 0.875)\n");
    printf("                                     only enabled if `--high-noise-steps` is set to -1\n");
    printf("  --flow-shift SHIFT                 shift value for Flow models like SD3.x or WAN (default: auto)\n");
    printf("  --vace-strength                    wan vace strength\n");
    printf("  --photo-maker                      path to PHOTOMAKER model\n");
    printf("  --pm-id-images-dir [DIR]           path to PHOTOMAKER input id images dir\n");
    printf("  --pm-id-embed-path [PATH]          path to PHOTOMAKER v2 id embed\n");
    printf("  --pm-style-strength                strength for keeping PHOTOMAKER input identity (default: 20)\n");
    printf("  -v, --verbose                      print extra info\n");
}

#if defined(_WIN32)
static std::string utf16_to_utf8(const std::wstring& wstr) {
    if (wstr.empty())
        return {};
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(),
                                          nullptr, 0, nullptr, nullptr);
    if (size_needed <= 0)
        throw std::runtime_error("UTF-16 to UTF-8 conversion failed");

    std::string utf8(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(),
                        (char*)utf8.data(), size_needed, nullptr, nullptr);
    return utf8;
}

static std::string argv_to_utf8(int index, const char** argv) {
    int argc;
    wchar_t** argv_w = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv_w)
        throw std::runtime_error("Failed to parse command line");

    std::string result;
    if (index < argc) {
        result = utf16_to_utf8(argv_w[index]);
    }
    LocalFree(argv_w);
    return result;
}

#else  // Linux / macOS
static std::string argv_to_utf8(int index, const char** argv) {
    return std::string(argv[index]);
}

#endif

struct StringOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::string* target;
};

struct IntOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    int* target;
};

struct FloatOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    float* target;
};

struct BoolOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    bool keep_true;
    bool* target;
};

struct ManualOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::function<int(int argc, const char** argv, int index)> cb;
};

struct ArgOptions {
    std::vector<StringOption> string_options;
    std::vector<IntOption> int_options;
    std::vector<FloatOption> float_options;
    std::vector<BoolOption> bool_options;
    std::vector<ManualOption> manual_options;
};

bool parse_options(int argc, const char** argv, ArgOptions& options) {
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        bool found_arg = false;
        arg            = argv[i];

        for (auto& option : options.string_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                *option.target = argv_to_utf8(i, argv);
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.int_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                *option.target = std::stoi(argv[i]);
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.float_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                *option.target = std::stof(argv[i]);
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.bool_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (option.keep_true) {
                    *option.target = true;
                } else {
                    *option.target = false;
                }
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.manual_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                int ret   = option.cb(argc, argv, i);
                if (ret < 0) {
                    invalid_arg = true;
                    break;
                }
                i += ret;
            }
        }
        if (invalid_arg) {
            break;
        }
        if (!found_arg) {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (invalid_arg) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return false;
    }
    return true;
}

void parse_args(int argc, const char** argv, SDParams& params) {
    ArgOptions options;
    options.string_options = {
        {"-m", "--model", "", &params.model_path},
        {"", "--clip_l", "", &params.clip_l_path},
        {"", "--clip_g", "", &params.clip_g_path},
        {"", "--clip_vision", "", &params.clip_vision_path},
        {"", "--t5xxl", "", &params.t5xxl_path},
        {"", "--qwen2vl", "", &params.qwen2vl_path},
        {"", "--diffusion-model", "", &params.diffusion_model_path},
        {"", "--high-noise-diffusion-model", "", &params.high_noise_diffusion_model_path},
        {"", "--vae", "", &params.vae_path},
        {"", "--taesd", "", &params.taesd_path},
        {"", "--control-net", "", &params.control_net_path},
        {"", "--embd-dir", "", &params.embedding_dir},
        {"", "--lora-model-dir", "", &params.lora_model_dir},
        {"-i", "--init-img", "", &params.init_image_path},
        {"", "--end-img", "", &params.end_image_path},
        {"", "--tensor-type-rules", "", &params.tensor_type_rules},
        {"", "--photo-maker", "", &params.photo_maker_path},
        {"", "--pm-id-images-dir", "", &params.pm_id_images_dir},
        {"", "--pm-id-embed-path", "", &params.pm_id_embed_path},
        {"", "--mask", "", &params.mask_image_path},
        {"", "--control-image", "", &params.control_image_path},
        {"", "--control-video", "", &params.control_video_path},
        {"-o", "--output", "", &params.output_path},
        {"-p", "--prompt", "", &params.prompt},
        {"-n", "--negative-prompt", "", &params.negative_prompt},
        {"", "--upscale-model", "", &params.esrgan_path},
    };

    options.int_options = {
        {"-t", "--threads", "", &params.n_threads},
        {"", "--upscale-repeats", "", &params.upscale_repeats},
        {"-H", "--height", "", &params.height},
        {"-W", "--width", "", &params.width},
        {"", "--steps", "", &params.sample_params.sample_steps},
        {"", "--high-noise-steps", "", &params.high_noise_sample_params.sample_steps},
        {"", "--clip-skip", "", &params.clip_skip},
        {"-b", "--batch-count", "", &params.batch_count},
        {"", "--chroma-t5-mask-pad", "", &params.chroma_t5_mask_pad},
        {"", "--video-frames", "", &params.video_frames},
        {"", "--fps", "", &params.fps},
        {"", "--timestep-shift", "", &params.sample_params.shifted_timestep},
    };

    options.float_options = {
        {"", "--cfg-scale", "", &params.sample_params.guidance.txt_cfg},
        {"", "--img-cfg-scale", "", &params.sample_params.guidance.img_cfg},
        {"", "--guidance", "", &params.sample_params.guidance.distilled_guidance},
        {"", "--slg-scale", "", &params.sample_params.guidance.slg.scale},
        {"", "--skip-layer-start", "", &params.sample_params.guidance.slg.layer_start},
        {"", "--skip-layer-end", "", &params.sample_params.guidance.slg.layer_end},
        {"", "--eta", "", &params.sample_params.eta},
        {"", "--high-noise-cfg-scale", "", &params.high_noise_sample_params.guidance.txt_cfg},
        {"", "--high-noise-img-cfg-scale", "", &params.high_noise_sample_params.guidance.img_cfg},
        {"", "--high-noise-guidance", "", &params.high_noise_sample_params.guidance.distilled_guidance},
        {"", "--high-noise-slg-scale", "", &params.high_noise_sample_params.guidance.slg.scale},
        {"", "--high-noise-skip-layer-start", "", &params.high_noise_sample_params.guidance.slg.layer_start},
        {"", "--high-noise-skip-layer-end", "", &params.high_noise_sample_params.guidance.slg.layer_end},
        {"", "--high-noise-eta", "", &params.high_noise_sample_params.eta},
        {"", "--strength", "", &params.strength},
        {"", "--pm-style-strength", "", &params.pm_style_strength},
        {"", "--control-strength", "", &params.control_strength},
        {"", "--moe-boundary", "", &params.moe_boundary},
        {"", "--flow-shift", "", &params.flow_shift},
        {"", "--vace-strength", "", &params.vace_strength},
        {"", "--vae-tile-overlap", "", &params.vae_tiling_params.target_overlap},
    };

    options.bool_options = {
        {"", "--vae-tiling", "", true, &params.vae_tiling_params.enabled},
        {"", "--offload-to-cpu", "", true, &params.offload_params_to_cpu},
        {"", "--control-net-cpu", "", true, &params.control_net_cpu},
        {"", "--clip-on-cpu", "", true, &params.clip_on_cpu},
        {"", "--vae-on-cpu", "", true, &params.vae_on_cpu},
        {"", "--diffusion-fa", "", true, &params.diffusion_flash_attn},
        {"", "--diffusion-conv-direct", "", true, &params.diffusion_conv_direct},
        {"", "--vae-conv-direct", "", true, &params.vae_conv_direct},
        {"", "--canny", "", true, &params.canny_preprocess},
        {"-v", "--verbose", "", true, &params.verbose},
        {"", "--color", "", true, &params.color},
        {"", "--chroma-disable-dit-mask", "", false, &params.chroma_use_dit_mask},
        {"", "--chroma-enable-t5-mask", "", true, &params.chroma_use_t5_mask},
        {"", "--increase-ref-index", "", true, &params.increase_ref_index},
    };

    auto on_mode_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* mode = argv[index];
        if (mode != NULL) {
            int mode_found = -1;
            for (int i = 0; i < MODE_COUNT; i++) {
                if (!strcmp(mode, modes_str[i])) {
                    mode_found = i;
                }
            }
            if (mode_found == -1) {
                fprintf(stderr,
                        "error: invalid mode %s, must be one of [%s]\n",
                        mode, SD_ALL_MODES_STR);
                exit(1);
            }
            params.mode = (SDMode)mode_found;
        }
        return 1;
    };

    auto on_type_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg = argv[index];
        params.wtype    = str_to_sd_type(arg);
        if (params.wtype == SD_TYPE_COUNT) {
            fprintf(stderr, "error: invalid weight format %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_rng_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg = argv[index];
        params.rng_type = str_to_rng_type(arg);
        if (params.rng_type == RNG_TYPE_COUNT) {
            fprintf(stderr, "error: invalid rng type %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_schedule_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg                = argv[index];
        params.sample_params.scheduler = str_to_schedule(arg);
        if (params.sample_params.scheduler == SCHEDULE_COUNT) {
            fprintf(stderr, "error: invalid scheduler %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_high_noise_schedule_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg                           = argv[index];
        params.high_noise_sample_params.scheduler = str_to_schedule(arg);
        if (params.high_noise_sample_params.scheduler == SCHEDULE_COUNT) {
            fprintf(stderr, "error: invalid high noise scheduler %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_sample_method_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg                    = argv[index];
        params.sample_params.sample_method = str_to_sample_method(arg);
        if (params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            fprintf(stderr, "error: invalid sample method %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_high_noise_sample_method_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg                               = argv[index];
        params.high_noise_sample_params.sample_method = str_to_sample_method(arg);
        if (params.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            fprintf(stderr, "error: invalid high noise sample method %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_seed_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        params.seed = std::stoll(argv[index]);
        return 1;
    };

    auto on_help_arg = [&](int argc, const char** argv, int index) {
        print_usage(argc, argv);
        exit(0);
        return 0;
    };

    auto on_skip_layers_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        std::string layers_str = argv[index];
        if (layers_str[0] != '[' || layers_str[layers_str.size() - 1] != ']') {
            return -1;
        }

        layers_str = layers_str.substr(1, layers_str.size() - 2);

        std::regex regex("[, ]+");
        std::sregex_token_iterator iter(layers_str.begin(), layers_str.end(), regex, -1);
        std::sregex_token_iterator end;
        std::vector<std::string> tokens(iter, end);
        std::vector<int> layers;
        for (const auto& token : tokens) {
            try {
                layers.push_back(std::stoi(token));
            } catch (const std::invalid_argument& e) {
                return -1;
            }
        }
        params.skip_layers = layers;
        return 1;
    };

    auto on_high_noise_skip_layers_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        std::string layers_str = argv[index];
        if (layers_str[0] != '[' || layers_str[layers_str.size() - 1] != ']') {
            return -1;
        }

        layers_str = layers_str.substr(1, layers_str.size() - 2);

        std::regex regex("[, ]+");
        std::sregex_token_iterator iter(layers_str.begin(), layers_str.end(), regex, -1);
        std::sregex_token_iterator end;
        std::vector<std::string> tokens(iter, end);
        std::vector<int> layers;
        for (const auto& token : tokens) {
            try {
                layers.push_back(std::stoi(token));
            } catch (const std::invalid_argument& e) {
                return -1;
            }
        }
        params.high_noise_skip_layers = layers;
        return 1;
    };

    auto on_ref_image_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        params.ref_image_paths.push_back(argv[index]);
        return 1;
    };

    auto on_tile_size_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        std::string tile_size_str = argv[index];
        size_t x_pos              = tile_size_str.find('x');
        try {
            if (x_pos != std::string::npos) {
                std::string tile_x_str               = tile_size_str.substr(0, x_pos);
                std::string tile_y_str               = tile_size_str.substr(x_pos + 1);
                params.vae_tiling_params.tile_size_x = std::stoi(tile_x_str);
                params.vae_tiling_params.tile_size_y = std::stoi(tile_y_str);
            } else {
                params.vae_tiling_params.tile_size_x = params.vae_tiling_params.tile_size_y = std::stoi(tile_size_str);
            }
        } catch (const std::invalid_argument& e) {
            return -1;
        } catch (const std::out_of_range& e) {
            return -1;
        }
        return 1;
    };

    auto on_relative_tile_size_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        std::string rel_size_str = argv[index];
        size_t x_pos             = rel_size_str.find('x');
        try {
            if (x_pos != std::string::npos) {
                std::string rel_x_str               = rel_size_str.substr(0, x_pos);
                std::string rel_y_str               = rel_size_str.substr(x_pos + 1);
                params.vae_tiling_params.rel_size_x = std::stof(rel_x_str);
                params.vae_tiling_params.rel_size_y = std::stof(rel_y_str);
            } else {
                params.vae_tiling_params.rel_size_x = params.vae_tiling_params.rel_size_y = std::stof(rel_size_str);
            }
        } catch (const std::invalid_argument& e) {
            return -1;
        } catch (const std::out_of_range& e) {
            return -1;
        }
        return 1;
    };

    options.manual_options = {
        {"-M", "--mode", "", on_mode_arg},
        {"", "--type", "", on_type_arg},
        {"", "--rng", "", on_rng_arg},
        {"-s", "--seed", "", on_seed_arg},
        {"", "--sampling-method", "", on_sample_method_arg},
        {"", "--scheduler", "", on_schedule_arg},
        {"", "--skip-layers", "", on_skip_layers_arg},
        {"", "--high-noise-sampling-method", "", on_high_noise_sample_method_arg},
        {"", "--high-noise-scheduler", "", on_high_noise_schedule_arg},
        {"", "--high-noise-skip-layers", "", on_high_noise_skip_layers_arg},
        {"-r", "--ref-image", "", on_ref_image_arg},
        {"-h", "--help", "", on_help_arg},
        {"", "--vae-tile-size", "", on_tile_size_arg},
        {"", "--vae-relative-tile-size", "", on_relative_tile_size_arg},
    };

    if (!parse_options(argc, argv, options)) {
        print_usage(argc, argv);
        exit(1);
    }

    if (params.n_threads <= 0) {
        params.n_threads = sd_get_num_physical_cores();
    }

    if (params.mode != CONVERT && params.mode != VID_GEN && params.prompt.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: prompt\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.model_path.length() == 0 && params.diffusion_model_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: model_path/diffusion_model\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.output_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: output_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.width <= 0) {
        fprintf(stderr, "error: the width must be greater than 0\n");
        exit(1);
    }

    if (params.height <= 0) {
        fprintf(stderr, "error: the height must be greater than 0\n");
        exit(1);
    }

    if (params.sample_params.sample_steps <= 0) {
        fprintf(stderr, "error: the sample_steps must be greater than 0\n");
        exit(1);
    }

    if (params.high_noise_sample_params.sample_steps <= 0) {
        params.high_noise_sample_params.sample_steps = -1;
    }

    if (params.strength < 0.f || params.strength > 1.f) {
        fprintf(stderr, "error: can only work with strength in [0.0, 1.0]\n");
        exit(1);
    }

    if (params.mode != CONVERT && params.tensor_type_rules.size() > 0) {
        fprintf(stderr, "warning: --tensor-type-rules is currently supported only for conversion\n");
    }

    if (params.mode == VID_GEN && params.video_frames <= 0) {
        fprintf(stderr, "warning: --video-frames must be at least 1\n");
        exit(1);
    }

    if (params.mode == VID_GEN && params.fps <= 0) {
        fprintf(stderr, "warning: --fps must be at least 1\n");
        exit(1);
    }

    if (params.sample_params.shifted_timestep < 0 || params.sample_params.shifted_timestep > 1000) {
        fprintf(stderr, "error: timestep-shift must be between 0 and 1000\n");
        exit(1);
    }

    if (params.upscale_repeats < 1) {
        fprintf(stderr, "error: upscale multiplier must be at least 1\n");
        exit(1);
    }

    if (params.seed < 0) {
        srand((int)time(NULL));
        params.seed = rand();
    }

    if (params.mode == CONVERT) {
        if (params.output_path == "output.png") {
            params.output_path = "output.gguf";
        }
    }

    if (!isfinite(params.sample_params.guidance.img_cfg)) {
        params.sample_params.guidance.img_cfg = params.sample_params.guidance.txt_cfg;
    }

    if (!isfinite(params.high_noise_sample_params.guidance.img_cfg)) {
        params.high_noise_sample_params.guidance.img_cfg = params.high_noise_sample_params.guidance.txt_cfg;
    }
}

static std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

std::string get_image_params(SDParams params, int64_t seed) {
    std::string parameter_string = params.prompt + "\n";
    if (params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.sample_params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.sample_params.guidance.txt_cfg) + ", ";
    if (params.sample_params.guidance.slg.scale != 0 && params.skip_layers.size() != 0) {
        parameter_string += "SLG scale: " + std::to_string(params.sample_params.guidance.txt_cfg) + ", ";
        parameter_string += "Skip layers: [";
        for (const auto& layer : params.skip_layers) {
            parameter_string += std::to_string(layer) + ", ";
        }
        parameter_string += "], ";
        parameter_string += "Skip layer start: " + std::to_string(params.sample_params.guidance.slg.layer_start) + ", ";
        parameter_string += "Skip layer end: " + std::to_string(params.sample_params.guidance.slg.layer_end) + ", ";
    }
    parameter_string += "Guidance: " + std::to_string(params.sample_params.guidance.distilled_guidance) + ", ";
    parameter_string += "Eta: " + std::to_string(params.sample_params.eta) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(sd_rng_type_name(params.rng_type)) + ", ";
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(params.sample_params.sample_method));
    if (params.sample_params.scheduler != DEFAULT) {
        parameter_string += " " + std::string(sd_schedule_name(params.sample_params.scheduler));
    }
    parameter_string += ", ";
    for (const auto& te : {params.clip_l_path, params.clip_g_path, params.t5xxl_path, params.qwen2vl_path}) {
        if (!te.empty()) {
            parameter_string += "TE: " + sd_basename(te) + ", ";
        }
    }
    if (!params.diffusion_model_path.empty()) {
        parameter_string += "Unet: " + sd_basename(params.diffusion_model_path) + ", ";
    }
    if (!params.vae_path.empty()) {
        parameter_string += "VAE: " + sd_basename(params.vae_path) + ", ";
    }
    if (params.clip_skip != -1) {
        parameter_string += "Clip skip: " + std::to_string(params.clip_skip) + ", ";
    }
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}

/* Enables Printing the log level tag in color using ANSI escape codes */
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDParams* params = (SDParams*)data;
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!params->verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            tag_color = 33;
            level_str = "?????";
            break;
    }

    if (params->color == true) {
        fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
    } else {
        fprintf(out_stream, "[%-5s] ", level_str);
    }
    fputs(log, out_stream);
    fflush(out_stream);
}

uint8_t* load_image(const char* image_path, int& width, int& height, int expected_width = 0, int expected_height = 0, int expected_channel = 3) {
    int c                 = 0;
    uint8_t* image_buffer = (uint8_t*)stbi_load(image_path, &width, &height, &c, expected_channel);
    if (image_buffer == NULL) {
        fprintf(stderr, "load image from '%s' failed\n", image_path);
        return NULL;
    }
    if (c < expected_channel) {
        fprintf(stderr,
                "the number of channels for the input image must be >= %d,"
                "but got %d channels, image_path = %s\n",
                expected_channel,
                c,
                image_path);
        free(image_buffer);
        return NULL;
    }
    if (width <= 0) {
        fprintf(stderr, "error: the width of image must be greater than 0, image_path = %s\n", image_path);
        free(image_buffer);
        return NULL;
    }
    if (height <= 0) {
        fprintf(stderr, "error: the height of image must be greater than 0, image_path = %s\n", image_path);
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
            printf("crop input image from %dx%d to %dx%d, image_path = %s\n", width, height, crop_w, crop_h, image_path);
            uint8_t* cropped_image_buffer = (uint8_t*)malloc(crop_w * crop_h * expected_channel);
            if (cropped_image_buffer == NULL) {
                fprintf(stderr, "error: allocate memory for crop\n");
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

        printf("resize input image from %dx%d to %dx%d\n", width, height, expected_width, expected_height);
        int resized_height = expected_height;
        int resized_width  = expected_width;

        uint8_t* resized_image_buffer = (uint8_t*)malloc(resized_height * resized_width * expected_channel);
        if (resized_image_buffer == NULL) {
            fprintf(stderr, "error: allocate memory for resize input image\n");
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

bool load_images_from_dir(const std::string dir,
                          std::vector<sd_image_t>& images,
                          int expected_width  = 0,
                          int expected_height = 0,
                          int max_image_num   = 0,
                          bool verbose        = false) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        fprintf(stderr, "'%s' is not a valid directory\n", dir.c_str());
        return false;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file())
            continue;

        std::string path = entry.path().string();
        std::string ext  = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            if (verbose) {
                printf("load image %zu from '%s'\n", images.size(), path.c_str());
            }
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = load_image(path.c_str(), width, height, expected_width, expected_height);
            if (image_buffer == NULL) {
                fprintf(stderr, "load image from '%s' failed\n", path.c_str());
                return false;
            }

            images.push_back({(uint32_t)width,
                              (uint32_t)height,
                              3,
                              image_buffer});

            if (max_image_num > 0 && images.size() >= max_image_num) {
                break;
            }
        }
    }
    return true;
}

int main(int argc, const char* argv[]) {
    SDParams params;
    parse_args(argc, argv, params);
    params.sample_params.guidance.slg.layers                 = params.skip_layers.data();
    params.sample_params.guidance.slg.layer_count            = params.skip_layers.size();
    params.high_noise_sample_params.guidance.slg.layers      = params.high_noise_skip_layers.data();
    params.high_noise_sample_params.guidance.slg.layer_count = params.high_noise_skip_layers.size();

    sd_set_log_callback(sd_log_cb, (void*)&params);

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info());
    }

    if (params.mode == CONVERT) {
        bool success = convert(params.model_path.c_str(), params.vae_path.c_str(), params.output_path.c_str(), params.wtype, params.tensor_type_rules.c_str());
        if (!success) {
            fprintf(stderr,
                    "convert '%s'/'%s' to '%s' failed\n",
                    params.model_path.c_str(),
                    params.vae_path.c_str(),
                    params.output_path.c_str());
            return 1;
        } else {
            printf("convert '%s'/'%s' to '%s' success\n",
                   params.model_path.c_str(),
                   params.vae_path.c_str(),
                   params.output_path.c_str());
            return 0;
        }
    }

    bool vae_decode_only     = true;
    sd_image_t init_image    = {(uint32_t)params.width, (uint32_t)params.height, 3, NULL};
    sd_image_t end_image     = {(uint32_t)params.width, (uint32_t)params.height, 3, NULL};
    sd_image_t control_image = {(uint32_t)params.width, (uint32_t)params.height, 3, NULL};
    sd_image_t mask_image    = {(uint32_t)params.width, (uint32_t)params.height, 1, NULL};
    std::vector<sd_image_t> ref_images;
    std::vector<sd_image_t> pmid_images;
    std::vector<sd_image_t> control_frames;

    auto release_all_resources = [&]() {
        free(init_image.data);
        free(end_image.data);
        free(control_image.data);
        free(mask_image.data);
        for (auto image : ref_images) {
            free(image.data);
            image.data = NULL;
        }
        ref_images.clear();
        for (auto image : pmid_images) {
            free(image.data);
            image.data = NULL;
        }
        pmid_images.clear();
        for (auto image : control_frames) {
            free(image.data);
            image.data = NULL;
        }
        control_frames.clear();
    };

    if (params.init_image_path.size() > 0) {
        vae_decode_only = false;

        int width       = 0;
        int height      = 0;
        init_image.data = load_image(params.init_image_path.c_str(), width, height, params.width, params.height);
        if (init_image.data == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.init_image_path.c_str());
            release_all_resources();
            return 1;
        }
    }

    if (params.end_image_path.size() > 0) {
        vae_decode_only = false;

        int width      = 0;
        int height     = 0;
        end_image.data = load_image(params.end_image_path.c_str(), width, height, params.width, params.height);
        if (end_image.data == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.end_image_path.c_str());
            release_all_resources();
            return 1;
        }
    }

    if (params.mask_image_path.size() > 0) {
        int c           = 0;
        int width       = 0;
        int height      = 0;
        mask_image.data = load_image(params.mask_image_path.c_str(), width, height, params.width, params.height, 1);
        if (mask_image.data == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.mask_image_path.c_str());
            release_all_resources();
            return 1;
        }
    } else {
        mask_image.data = (uint8_t*)malloc(params.width * params.height);
        memset(mask_image.data, 255, params.width * params.height);
        if (mask_image.data == NULL) {
            fprintf(stderr, "malloc mask image failed\n");
            release_all_resources();
            return 1;
        }
    }

    if (params.control_net_path.size() > 0 && params.control_image_path.size() > 0) {
        int width          = 0;
        int height         = 0;
        control_image.data = load_image(params.control_image_path.c_str(), width, height, params.width, params.height);
        if (control_image.data == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.control_image_path.c_str());
            release_all_resources();
            return 1;
        }
        if (params.canny_preprocess) {  // apply preprocessor
            preprocess_canny(control_image,
                             0.08f,
                             0.08f,
                             0.8f,
                             1.0f,
                             false);
        }
    }

    if (params.ref_image_paths.size() > 0) {
        vae_decode_only = false;
        for (auto& path : params.ref_image_paths) {
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = load_image(path.c_str(), width, height);
            if (image_buffer == NULL) {
                fprintf(stderr, "load image from '%s' failed\n", path.c_str());
                release_all_resources();
                return 1;
            }
            ref_images.push_back({(uint32_t)width,
                                  (uint32_t)height,
                                  3,
                                  image_buffer});
        }
    }

    if (!params.control_video_path.empty()) {
        if (!load_images_from_dir(params.control_video_path,
                                  control_frames,
                                  params.width,
                                  params.height,
                                  params.video_frames,
                                  params.verbose)) {
            release_all_resources();
            return 1;
        }
    }

    if (!params.pm_id_images_dir.empty()) {
        if (!load_images_from_dir(params.pm_id_images_dir,
                                  pmid_images,
                                  0,
                                  0,
                                  0,
                                  params.verbose)) {
            release_all_resources();
            return 1;
        }
    }

    if (params.mode == VID_GEN) {
        vae_decode_only = false;
    }

    sd_ctx_params_t sd_ctx_params = {
        params.model_path.c_str(),
        params.clip_l_path.c_str(),
        params.clip_g_path.c_str(),
        params.clip_vision_path.c_str(),
        params.t5xxl_path.c_str(),
        params.qwen2vl_path.c_str(),
        params.diffusion_model_path.c_str(),
        params.high_noise_diffusion_model_path.c_str(),
        params.vae_path.c_str(),
        params.taesd_path.c_str(),
        params.control_net_path.c_str(),
        params.lora_model_dir.c_str(),
        params.embedding_dir.c_str(),
        params.photo_maker_path.c_str(),
        vae_decode_only,
        true,
        params.n_threads,
        params.wtype,
        params.rng_type,
        params.offload_params_to_cpu,
        params.clip_on_cpu,
        params.control_net_cpu,
        params.vae_on_cpu,
        params.diffusion_flash_attn,
        params.diffusion_conv_direct,
        params.vae_conv_direct,
        params.chroma_use_dit_mask,
        params.chroma_use_t5_mask,
        params.chroma_t5_mask_pad,
        params.flow_shift,
    };

    sd_ctx_t* sd_ctx = new_sd_ctx(&sd_ctx_params);

    if (sd_ctx == NULL) {
        printf("new_sd_ctx_t failed\n");
        release_all_resources();
        return 1;
    }

    if (params.sample_params.sample_method == SAMPLE_METHOD_DEFAULT) {
        params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
    }

    sd_image_t* results;
    int num_results = 1;
    if (params.mode == IMG_GEN) {
        sd_img_gen_params_t img_gen_params = {
            params.prompt.c_str(),
            params.negative_prompt.c_str(),
            params.clip_skip,
            init_image,
            ref_images.data(),
            (int)ref_images.size(),
            params.increase_ref_index,
            mask_image,
            params.width,
            params.height,
            params.sample_params,
            params.strength,
            params.seed,
            params.batch_count,
            control_image,
            params.control_strength,
            {
                pmid_images.data(),
                (int)pmid_images.size(),
                params.pm_id_embed_path.c_str(),
                params.pm_style_strength,
            },  // pm_params
            params.vae_tiling_params,
        };

        results     = generate_image(sd_ctx, &img_gen_params);
        num_results = params.batch_count;
    } else if (params.mode == VID_GEN) {
        sd_vid_gen_params_t vid_gen_params = {
            params.prompt.c_str(),
            params.negative_prompt.c_str(),
            params.clip_skip,
            init_image,
            end_image,
            control_frames.data(),
            (int)control_frames.size(),
            params.width,
            params.height,
            params.sample_params,
            params.high_noise_sample_params,
            params.moe_boundary,
            params.strength,
            params.seed,
            params.video_frames,
            params.vace_strength,
        };

        results = generate_video(sd_ctx, &vid_gen_params, &num_results);
    }

    if (results == NULL) {
        printf("generate failed\n");
        free_sd_ctx(sd_ctx);
        return 1;
    }

    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (params.esrgan_path.size() > 0 && params.upscale_repeats > 0) {
        upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(params.esrgan_path.c_str(),
                                                        params.offload_params_to_cpu,
                                                        params.diffusion_conv_direct,
                                                        params.n_threads);

        if (upscaler_ctx == NULL) {
            printf("new_upscaler_ctx failed\n");
        } else {
            for (int i = 0; i < params.batch_count; i++) {
                if (results[i].data == NULL) {
                    continue;
                }
                sd_image_t current_image = results[i];
                for (int u = 0; u < params.upscale_repeats; ++u) {
                    sd_image_t upscaled_image = upscale(upscaler_ctx, current_image, upscale_factor);
                    if (upscaled_image.data == NULL) {
                        printf("upscale failed\n");
                        break;
                    }
                    free(current_image.data);
                    current_image = upscaled_image;
                }
                results[i] = current_image;  // Set the final upscaled image as the result
            }
        }
    }

    // create directory if not exists
    {
        const fs::path out_path = params.output_path;
        if (const fs::path out_dir = out_path.parent_path(); !out_dir.empty()) {
            std::error_code ec;
            fs::create_directories(out_dir, ec);  // OK if already exists
            if (ec) {
                fprintf(stderr, "failed to create directory '%s': %s\n",
                        out_dir.string().c_str(), ec.message().c_str());
                return 1;
            }
        }
    }

    std::string base_path;
    std::string file_ext;
    std::string file_ext_lower;
    bool is_jpg;
    size_t last_dot_pos   = params.output_path.find_last_of(".");
    size_t last_slash_pos = std::min(params.output_path.find_last_of("/"),
                                     params.output_path.find_last_of("\\"));
    if (last_dot_pos != std::string::npos && (last_slash_pos == std::string::npos || last_dot_pos > last_slash_pos)) {  // filename has extension
        base_path = params.output_path.substr(0, last_dot_pos);
        file_ext = file_ext_lower = params.output_path.substr(last_dot_pos);
        std::transform(file_ext.begin(), file_ext.end(), file_ext_lower.begin(), ::tolower);
        is_jpg = (file_ext_lower == ".jpg" || file_ext_lower == ".jpeg" || file_ext_lower == ".jpe");
    } else {
        base_path = params.output_path;
        file_ext = file_ext_lower = "";
        is_jpg                    = false;
    }

    if (params.mode == VID_GEN && num_results > 1) {
        std::string vid_output_path = params.output_path;
        if (file_ext_lower == ".png") {
            vid_output_path = base_path + ".avi";
        }
        create_mjpg_avi_from_sd_images(vid_output_path.c_str(), results, num_results, params.fps);
        printf("save result MJPG AVI video to '%s'\n", vid_output_path.c_str());
    } else {
        // appending ".png" to absent or unknown extension
        if (!is_jpg && file_ext_lower != ".png") {
            base_path += file_ext;
            file_ext = ".png";
        }
        for (int i = 0; i < num_results; i++) {
            if (results[i].data == NULL) {
                continue;
            }
            std::string final_image_path = i > 0 ? base_path + "_" + std::to_string(i + 1) + file_ext : base_path + file_ext;
            if (is_jpg) {
                stbi_write_jpg(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                               results[i].data, 90, get_image_params(params, params.seed + i).c_str());
                printf("save result JPEG image to '%s'\n", final_image_path.c_str());
            } else {
                stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                               results[i].data, 0, get_image_params(params, params.seed + i).c_str());
                printf("save result PNG image to '%s'\n", final_image_path.c_str());
            }
        }
    }

    for (int i = 0; i < num_results; i++) {
        free(results[i].data);
        results[i].data = NULL;
    }
    free(results);
    free_sd_ctx(sd_ctx);

    release_all_resources();

    return 0;
}
