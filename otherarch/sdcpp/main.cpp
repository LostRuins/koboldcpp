#include <stdio.h>
#include <string.h>
#include <time.h>
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
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

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
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string control_net_path;
    std::string embedding_dir;
    std::string stacked_id_embed_dir;
    std::string input_id_images_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string mask_path;
    std::string control_image_path;
    std::vector<std::string> ref_image_paths;

    std::string prompt;
    std::string negative_prompt;
    float min_cfg       = 1.0f;
    float cfg_scale     = 7.0f;
    float img_cfg_scale = INFINITY;
    float guidance      = 3.5f;
    float eta           = 0.f;
    float style_ratio   = 20.f;
    int clip_skip       = -1;  // <= 0 represents unspecified
    int width           = 512;
    int height          = 512;
    int batch_count     = 1;

    int video_frames         = 6;
    int motion_bucket_id     = 127;
    int fps                  = 6;
    float augmentation_level = 0.f;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool control_net_cpu          = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool diffusion_flash_attn     = false;
    bool diffusion_conv_direct    = false;
    bool vae_conv_direct          = false;
    bool canny_preprocess         = false;
    bool color                    = false;
    int upscale_repeats           = 1;

    std::vector<int> skip_layers = {7, 8, 9};
    float slg_scale              = 0.f;
    float skip_layer_start       = 0.01f;
    float skip_layer_end         = 0.2f;

    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask  = false;
    int chroma_t5_mask_pad   = 1;
};

void print_params(SDParams params) {
    printf("Option: \n");
    printf("    n_threads:         %d\n", params.n_threads);
    printf("    mode:              %s\n", modes_str[params.mode]);
    printf("    model_path:        %s\n", params.model_path.c_str());
    printf("    wtype:             %s\n", params.wtype < SD_TYPE_COUNT ? sd_type_name(params.wtype) : "unspecified");
    printf("    clip_l_path:       %s\n", params.clip_l_path.c_str());
    printf("    clip_g_path:       %s\n", params.clip_g_path.c_str());
    printf("    t5xxl_path:        %s\n", params.t5xxl_path.c_str());
    printf("    diffusion_model_path:   %s\n", params.diffusion_model_path.c_str());
    printf("    vae_path:          %s\n", params.vae_path.c_str());
    printf("    taesd_path:        %s\n", params.taesd_path.c_str());
    printf("    esrgan_path:       %s\n", params.esrgan_path.c_str());
    printf("    control_net_path:   %s\n", params.control_net_path.c_str());
    printf("    embedding_dir:   %s\n", params.embedding_dir.c_str());
    printf("    stacked_id_embed_dir:   %s\n", params.stacked_id_embed_dir.c_str());
    printf("    input_id_images_path:   %s\n", params.input_id_images_path.c_str());
    printf("    style ratio:       %.2f\n", params.style_ratio);
    printf("    normalize input image :  %s\n", params.normalize_input ? "true" : "false");
    printf("    output_path:       %s\n", params.output_path.c_str());
    printf("    init_img:          %s\n", params.input_path.c_str());
    printf("    mask_img:          %s\n", params.mask_path.c_str());
    printf("    control_image:     %s\n", params.control_image_path.c_str());
    printf("    ref_images_paths:\n");
    for (auto& path : params.ref_image_paths) {
        printf("        %s\n", path.c_str());
    };
    printf("    clip on cpu:       %s\n", params.clip_on_cpu ? "true" : "false");
    printf("    controlnet cpu:    %s\n", params.control_net_cpu ? "true" : "false");
    printf("    vae decoder on cpu:%s\n", params.vae_on_cpu ? "true" : "false");
    printf("    diffusion flash attention:%s\n", params.diffusion_flash_attn ? "true" : "false");
    printf("    diffusion Conv2d direct:%s\n", params.diffusion_conv_direct ? "true" : "false");
    printf("    vae Conv2d direct:%s\n", params.vae_conv_direct ? "true" : "false");
    printf("    strength(control): %.2f\n", params.control_strength);
    printf("    prompt:            %s\n", params.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.negative_prompt.c_str());
    printf("    min_cfg:           %.2f\n", params.min_cfg);
    printf("    cfg_scale:         %.2f\n", params.cfg_scale);
    printf("    img_cfg_scale:     %.2f\n", params.img_cfg_scale);
    printf("    slg_scale:         %.2f\n", params.slg_scale);
    printf("    guidance:          %.2f\n", params.guidance);
    printf("    eta:               %.2f\n", params.eta);
    printf("    clip_skip:         %d\n", params.clip_skip);
    printf("    width:             %d\n", params.width);
    printf("    height:            %d\n", params.height);
    printf("    sample_method:     %s\n", sd_sample_method_name(params.sample_method));
    printf("    schedule:          %s\n", sd_schedule_name(params.schedule));
    printf("    sample_steps:      %d\n", params.sample_steps);
    printf("    strength(img2img): %.2f\n", params.strength);
    printf("    rng:               %s\n", sd_rng_type_name(params.rng_type));
    printf("    seed:              %ld\n", params.seed);
    printf("    batch_count:       %d\n", params.batch_count);
    printf("    vae_tiling:        %s\n", params.vae_tiling ? "true" : "false");
    printf("    upscale_repeats:   %d\n", params.upscale_repeats);
    printf("    chroma_use_dit_mask:   %s\n", params.chroma_use_dit_mask ? "true" : "false");
    printf("    chroma_use_t5_mask:    %s\n", params.chroma_use_t5_mask ? "true" : "false");
    printf("    chroma_t5_mask_pad:    %d\n", params.chroma_t5_mask_pad);
}

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -M, --mode [MODE]                  run mode, one of: [img_gen, convert], default: img_gen\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1)\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model [MODEL]                path to full model\n");
    printf("  --diffusion-model                  path to the standalone diffusion model\n");
    printf("  --clip_l                           path to the clip-l text encoder\n");
    printf("  --clip_g                           path to the clip-g text encoder\n");
    printf("  --t5xxl                            path to the t5xxl text encoder\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)\n");
    printf("  --control-net [CONTROL_PATH]       path to control net model\n");
    printf("  --embd-dir [EMBEDDING_PATH]        path to embeddings\n");
    printf("  --stacked-id-embd-dir [DIR]        path to PHOTOMAKER stacked id embeddings\n");
    printf("  --input-id-images-dir [DIR]        path to PHOTOMAKER input id images dir\n");
    printf("  --normalize-input                  normalize PHOTOMAKER input id images\n");
    printf("  --upscale-model [ESRGAN_PATH]      path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now\n");
    printf("  --upscale-repeats                  Run the ESRGAN upscaler this many times (default 1)\n");
    printf("  --type [TYPE]                      weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K)\n");
    printf("                                     If not specified, the default is the type of the weight file\n");
    printf("  --tensor-type-rules [EXPRESSION]   weight type per tensor pattern (example: \"^vae\\.=f16,model\\.=q8_0\")\n");
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  -i, --init-img [IMAGE]             path to the input image, required by img2img\n");
    printf("  --mask [MASK]                      path to the mask image, required by img2img with mask\n");
    printf("  --control-image [IMAGE]            path to image condition, control net\n");
    printf("  -r, --ref-image [PATH]             reference image for Flux Kontext models (can be used multiple times) \n");
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
    printf("                                     SLG will be enabled at step int([STEPS]*[START]) and disabled at int([STEPS]*[END])\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("  --style-ratio STYLE-RATIO          strength for keeping input identity (default: 20)\n");
    printf("  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)\n");
    printf("                                     1.0 corresponds to full destruction of information in init image\n");
    printf("  -H, --height H                     image height, in pixel space (default: 512)\n");
    printf("  -W, --width W                      image width, in pixel space (default: 512)\n");
    printf("  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd}\n");
    printf("                                     sampling method (default: \"euler_a\")\n");
    printf("  --steps  STEPS                     number of sample steps (default: 20)\n");
    printf("  --rng {std_default, cuda}          RNG (default: cuda)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)\n");
    printf("  -b, --batch-count COUNT            number of images to generate\n");
    printf("  --schedule {discrete, karras, exponential, ays, gits} Denoiser sigma schedule (default: discrete)\n");
    printf("  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)\n");
    printf("                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x\n");
    printf("  --vae-tiling                       process vae in tiles to reduce memory usage\n");
    printf("  --vae-on-cpu                       keep vae in cpu (for low vram)\n");
    printf("  --clip-on-cpu                      keep clip in cpu (for low vram)\n");
    printf("  --diffusion-fa                     use flash attention in the diffusion model (for low vram)\n");
    printf("                                     Might lower quality, since it implies converting k and v to f16.\n");
    printf("                                     This might crash if it is not supported by the backend.\n");
    printf("  --diffusion-conv-direct            use Conv2d direct in the diffusion model");
    printf("                                     This might crash if it is not supported by the backend.\n");
    printf("  --vae-conv-direct                  use Conv2d direct in the vae model (should improve the performance)");
    printf("                                     This might crash if it is not supported by the backend.\n");
    printf("  --control-net-cpu                  keep controlnet in cpu (for low vram)\n");
    printf("  --canny                            apply canny preprocessor (edge detection)\n");
    printf("  --color                            colors the logging tags according to level\n");
    printf("  --chroma-disable-dit-mask          disable dit mask for chroma\n");
    printf("  --chroma-enable-t5-mask            enable t5 mask for chroma\n");
    printf("  --chroma-t5-mask-pad  PAD_SIZE     t5 mask pad size of chroma\n");
    printf("  -v, --verbose                      print extra info\n");
}

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
        arg = argv[i];

        for (auto& option : options.string_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                *option.target = std::string(argv[i]);
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.int_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
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
                int ret = option.cb(argc, argv, i);
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
        {"", "--t5xxl", "", &params.t5xxl_path},
        {"", "--diffusion-model", "", &params.diffusion_model_path},
        {"", "--vae", "", &params.vae_path},
        {"", "--taesd", "", &params.taesd_path},
        {"", "--control-net", "", &params.control_net_path},
        {"", "--embd-dir", "", &params.embedding_dir},
        {"", "--stacked-id-embd-dir", "", &params.stacked_id_embed_dir},
        {"", "--lora-model-dir", "", &params.lora_model_dir},
        {"-i", "--init-img", "", &params.input_path},
        {"", "--tensor-type-rules", "", &params.tensor_type_rules},
        {"", "--input-id-images-dir", "", &params.input_id_images_path},
        {"", "--mask", "", &params.mask_path},
        {"", "--control-image", "", &params.control_image_path},
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
        {"", "--steps", "", &params.sample_steps},
        {"", "--clip-skip", "", &params.clip_skip},
        {"-b", "--batch-count", "", &params.batch_count},
        {"", "--chroma-t5-mask-pad", "", &params.chroma_t5_mask_pad},
    };

    options.float_options = {
        {"", "--cfg-scale", "", &params.cfg_scale},
        {"", "--img-cfg-scale", "", &params.img_cfg_scale},
        {"", "--guidance", "", &params.guidance},
        {"", "--eta", "", &params.eta},
        {"", "--strength", "", &params.strength},
        {"", "--style-ratio", "", &params.style_ratio},
        {"", "--control-strength", "", &params.control_strength},
        {"", "--slg-scale", "", &params.slg_scale},
        {"", "--skip-layer-start", "", &params.skip_layer_start},
        {"", "--skip-layer-end", "", &params.skip_layer_end},

    };

    options.bool_options = {
        {"", "--vae-tiling", "", true, &params.vae_tiling},
        {"", "--control-net-cpu", "", true, &params.control_net_cpu},
        {"", "--normalize-input", "", true, &params.normalize_input},
        {"", "--clip-on-cpu", "", true, &params.clip_on_cpu},
        {"", "--vae-on-cpu", "", true, &params.vae_on_cpu},
        {"", "--diffusion-fa", "", true, &params.diffusion_flash_attn},
        {"", "--diffusion-conv-direct", "", true, &params.diffusion_conv_direct},
        {"", "--vae-conv-direct", "", true, &params.vae_conv_direct},
        {"", "--canny", "", true, &params.canny_preprocess},
        {"-v", "--verbos", "", true, &params.verbose},
        {"", "--color", "", true, &params.color},
        {"", "--chroma-disable-dit-mask", "", false, &params.chroma_use_dit_mask},
        {"", "--chroma-enable-t5-mask", "", true, &params.chroma_use_t5_mask},
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
        const char* arg = argv[index];
        params.schedule = str_to_schedule(arg);
        if (params.schedule == SCHEDULE_COUNT) {
            fprintf(stderr, "error: invalid schedule %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_sample_method_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg      = argv[index];
        params.sample_method = str_to_sample_method(arg);
        if (params.sample_method == SAMPLE_METHOD_COUNT) {
            fprintf(stderr, "error: invalid sample method %s\n",
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

    auto on_ref_image_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        params.ref_image_paths.push_back(argv[index]);
        return 1;
    };

    options.manual_options = {
        {"-M", "--mode", "", on_mode_arg},
        {"", "--type", "", on_type_arg},
        {"", "--rng", "", on_rng_arg},
        {"-s", "--seed", "", on_seed_arg},
        {"", "--sampling-method", "", on_sample_method_arg},
        {"", "--schedule", "", on_schedule_arg},
        {"", "--skip-layers", "", on_skip_layers_arg},
        {"-r", "--ref-image", "", on_ref_image_arg},
        {"-h", "--help", "", on_help_arg},
    };

    if (!parse_options(argc, argv, options)) {
        print_usage(argc, argv);
        exit(1);
    }

    if (params.n_threads <= 0) {
        params.n_threads = get_num_physical_cores();
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

    if (params.sample_steps <= 0) {
        fprintf(stderr, "error: the sample_steps must be greater than 0\n");
        exit(1);
    }

    if (params.strength < 0.f || params.strength > 1.f) {
        fprintf(stderr, "error: can only work with strength in [0.0, 1.0]\n");
        exit(1);
    }

    if (params.mode != CONVERT && params.tensor_type_rules.size() > 0) {
        fprintf(stderr, "warning: --tensor-type-rules is currently supported only for conversion\n");
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

    if (!isfinite(params.img_cfg_scale)) {
        params.img_cfg_scale = params.cfg_scale;
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
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    if (params.slg_scale != 0 && params.skip_layers.size() != 0) {
        parameter_string += "SLG scale: " + std::to_string(params.cfg_scale) + ", ";
        parameter_string += "Skip layers: [";
        for (const auto& layer : params.skip_layers) {
            parameter_string += std::to_string(layer) + ", ";
        }
        parameter_string += "], ";
        parameter_string += "Skip layer start: " + std::to_string(params.skip_layer_start) + ", ";
        parameter_string += "Skip layer end: " + std::to_string(params.skip_layer_end) + ", ";
    }
    parameter_string += "Guidance: " + std::to_string(params.guidance) + ", ";
    parameter_string += "Eta: " + std::to_string(params.eta) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(sd_rng_type_name(params.rng_type)) + ", ";
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(params.sample_method));
    if (params.schedule != DEFAULT) {
        parameter_string += " " + std::string(sd_schedule_name(params.schedule));
    }
    parameter_string += ", ";
    for (const auto& te : {params.clip_l_path, params.clip_g_path, params.t5xxl_path}) {
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

int main(int argc, const char* argv[]) {
    SDParams params;

    parse_args(argc, argv, params);

    sd_guidance_params_t guidance_params = {params.cfg_scale,
                                            params.img_cfg_scale,
                                            params.min_cfg,
                                            params.guidance,
                                            {
                                                params.skip_layers.data(),
                                                params.skip_layers.size(),
                                                params.skip_layer_start,
                                                params.skip_layer_end,
                                                params.slg_scale,
                                            }};

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

    if (params.mode == VID_GEN) {
        fprintf(stderr, "SVD support is broken, do not use it!!!\n");
        return 1;
    }

    bool vae_decode_only          = true;
    uint8_t* input_image_buffer   = NULL;
    uint8_t* control_image_buffer = NULL;
    uint8_t* mask_image_buffer    = NULL;
    std::vector<sd_image_t> ref_images;

    if (params.input_path.size() > 0) {
        vae_decode_only = false;

        int c              = 0;
        int width          = 0;
        int height         = 0;
        input_image_buffer = stbi_load(params.input_path.c_str(), &width, &height, &c, 3);
        if (input_image_buffer == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.input_path.c_str());
            return 1;
        }
        if (c < 3) {
            fprintf(stderr, "the number of channels for the input image must be >= 3, but got %d channels\n", c);
            free(input_image_buffer);
            return 1;
        }
        if (width <= 0) {
            fprintf(stderr, "error: the width of image must be greater than 0\n");
            free(input_image_buffer);
            return 1;
        }
        if (height <= 0) {
            fprintf(stderr, "error: the height of image must be greater than 0\n");
            free(input_image_buffer);
            return 1;
        }

        // Resize input image ...
        if (params.height != height || params.width != width) {
            printf("resize input image from %dx%d to %dx%d\n", width, height, params.width, params.height);
            int resized_height = params.height;
            int resized_width  = params.width;

            uint8_t* resized_image_buffer = (uint8_t*)malloc(resized_height * resized_width * 3);
            if (resized_image_buffer == NULL) {
                fprintf(stderr, "error: allocate memory for resize input image\n");
                free(input_image_buffer);
                return 1;
            }
            stbir_resize(input_image_buffer, width, height, 0,
                         resized_image_buffer, resized_width, resized_height, 0, STBIR_TYPE_UINT8,
                         3 /*RGB channel*/, STBIR_ALPHA_CHANNEL_NONE, 0,
                         STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                         STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                         STBIR_COLORSPACE_SRGB, nullptr);

            // Save resized result
            free(input_image_buffer);
            input_image_buffer = resized_image_buffer;
        }
    } else if (params.ref_image_paths.size() > 0) {
        vae_decode_only = false;
        for (auto& path : params.ref_image_paths) {
            int c                 = 0;
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = stbi_load(path.c_str(), &width, &height, &c, 3);
            if (image_buffer == NULL) {
                fprintf(stderr, "load image from '%s' failed\n", path.c_str());
                return 1;
            }
            if (c < 3) {
                fprintf(stderr, "the number of channels for the input image must be >= 3, but got %d channels\n", c);
                free(image_buffer);
                return 1;
            }
            if (width <= 0) {
                fprintf(stderr, "error: the width of image must be greater than 0\n");
                free(image_buffer);
                return 1;
            }
            if (height <= 0) {
                fprintf(stderr, "error: the height of image must be greater than 0\n");
                free(image_buffer);
                return 1;
            }
            ref_images.push_back({(uint32_t)width,
                                  (uint32_t)height,
                                  3,
                                  image_buffer});
        }
    }

    sd_ctx_params_t sd_ctx_params = {
        params.model_path.c_str(),
        params.clip_l_path.c_str(),
        params.clip_g_path.c_str(),
        params.t5xxl_path.c_str(),
        params.diffusion_model_path.c_str(),
        params.vae_path.c_str(),
        params.taesd_path.c_str(),
        params.control_net_path.c_str(),
        params.lora_model_dir.c_str(),
        params.embedding_dir.c_str(),
        params.stacked_id_embed_dir.c_str(),
        vae_decode_only,
        params.vae_tiling,
        true,
        params.n_threads,
        params.wtype,
        params.rng_type,
        params.schedule,
        params.clip_on_cpu,
        params.control_net_cpu,
        params.vae_on_cpu,
        params.diffusion_flash_attn,
        params.diffusion_conv_direct,
        params.vae_conv_direct,
        params.chroma_use_dit_mask,
        params.chroma_use_t5_mask,
        params.chroma_t5_mask_pad,
    };

    sd_ctx_t* sd_ctx = new_sd_ctx(&sd_ctx_params);

    if (sd_ctx == NULL) {
        printf("new_sd_ctx_t failed\n");
        return 1;
    }

    sd_image_t input_image = {(uint32_t)params.width,
                              (uint32_t)params.height,
                              3,
                              input_image_buffer};

    sd_image_t* control_image = NULL;
    if (params.control_net_path.size() > 0 && params.control_image_path.size() > 0) {
        int c                = 0;
        control_image_buffer = stbi_load(params.control_image_path.c_str(), &params.width, &params.height, &c, 3);
        if (control_image_buffer == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.control_image_path.c_str());
            return 1;
        }
        control_image = new sd_image_t{(uint32_t)params.width,
                                       (uint32_t)params.height,
                                       3,
                                       control_image_buffer};
        if (params.canny_preprocess) {  // apply preprocessor
            control_image->data = preprocess_canny(control_image->data,
                                                   control_image->width,
                                                   control_image->height,
                                                   0.08f,
                                                   0.08f,
                                                   0.8f,
                                                   1.0f,
                                                   false);
        }
    }

    std::vector<uint8_t> default_mask_image_vec(params.width * params.height, 255);
    if (params.mask_path != "") {
        int c             = 0;
        mask_image_buffer = stbi_load(params.mask_path.c_str(), &params.width, &params.height, &c, 1);
    } else {
        mask_image_buffer = default_mask_image_vec.data();
    }
    sd_image_t mask_image = {(uint32_t)params.width,
                             (uint32_t)params.height,
                             1,
                             mask_image_buffer};

    sd_image_t* results;
    int expected_num_results = 1;
    if (params.mode == IMG_GEN) {
        sd_img_gen_params_t img_gen_params = {
            params.prompt.c_str(),
            params.negative_prompt.c_str(),
            params.clip_skip,
            guidance_params,
            input_image,
            ref_images.data(),
            (int)ref_images.size(),
            mask_image,
            params.width,
            params.height,
            params.sample_method,
            params.sample_steps,
            params.eta,
            params.strength,
            params.seed,
            params.batch_count,
            control_image,
            params.control_strength,
            params.style_ratio,
            params.normalize_input,
            params.input_id_images_path.c_str(),
        };

        results              = generate_image(sd_ctx, &img_gen_params);
        expected_num_results = params.batch_count;
    } else if (params.mode == VID_GEN) {
        sd_vid_gen_params_t vid_gen_params = {
            input_image,
            params.width,
            params.height,
            guidance_params,
            params.sample_method,
            params.sample_steps,
            params.strength,
            params.seed,
            params.video_frames,
            params.motion_bucket_id,
            params.fps,
            params.augmentation_level,
        };

        results              = generate_video(sd_ctx, &vid_gen_params);
        expected_num_results = params.video_frames;
    }

    if (results == NULL) {
        printf("generate failed\n");
        free_sd_ctx(sd_ctx);
        return 1;
    }

    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (params.esrgan_path.size() > 0 && params.upscale_repeats > 0) {
        upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(params.esrgan_path.c_str(),
                                                        params.n_threads,
                                                        params.diffusion_conv_direct);

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

    std::string dummy_name, ext, lc_ext;
    bool is_jpg;
    size_t last      = params.output_path.find_last_of(".");
    size_t last_path = std::min(params.output_path.find_last_of("/"),
                                params.output_path.find_last_of("\\"));
    if (last != std::string::npos  // filename has extension
        && (last_path == std::string::npos || last > last_path)) {
        dummy_name = params.output_path.substr(0, last);
        ext = lc_ext = params.output_path.substr(last);
        std::transform(ext.begin(), ext.end(), lc_ext.begin(), ::tolower);
        is_jpg = lc_ext == ".jpg" || lc_ext == ".jpeg" || lc_ext == ".jpe";
    } else {
        dummy_name = params.output_path;
        ext = lc_ext = "";
        is_jpg       = false;
    }
    // appending ".png" to absent or unknown extension
    if (!is_jpg && lc_ext != ".png") {
        dummy_name += ext;
        ext = ".png";
    }
    for (int i = 0; i < expected_num_results; i++) {
        if (results[i].data == NULL) {
            continue;
        }
        std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ext : dummy_name + ext;
        if (is_jpg) {
            stbi_write_jpg(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                           results[i].data, 90, get_image_params(params, params.seed + i).c_str());
            printf("save result JPEG image to '%s'\n", final_image_path.c_str());
        } else {
            stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                           results[i].data, 0, get_image_params(params, params.seed + i).c_str());
            printf("save result PNG image to '%s'\n", final_image_path.c_str());
        }
        free(results[i].data);
        results[i].data = NULL;
    }
    free(results);
    free_sd_ctx(sd_ctx);
    free(control_image_buffer);
    free(input_image_buffer);

    return 0;
}
