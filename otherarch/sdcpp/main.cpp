#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// #include "preprocessing.hpp"
#include "flux.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
};

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
const char* sample_method_str[] = {
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

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const char* schedule_str[] = {
    "default",
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
};

const char* modes_str[] = {
    "txt2img",
    "img2img",
    "img2vid",
    "convert",
};

enum SDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    std::string input_id_images_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string mask_path;
    std::string control_image_path;

    std::vector<std::string> kontext_image_paths;

    std::string prompt;
    std::string negative_prompt;
    float min_cfg     = 1.0f;
    float cfg_scale   = 7.0f;
    float guidance    = 3.5f;
    float eta         = 0.f;
    float style_ratio = 20.f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;
    int batch_count   = 1;

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
    bool canny_preprocess         = false;
    bool color                    = false;
    int upscale_repeats           = 1;

    std::vector<int> skip_layers = {7, 8, 9};
    float slg_scale              = 0.f;
    float skip_layer_start       = 0.01f;
    float skip_layer_end         = 0.2f;

    bool chroma_use_dit_mask     = true;
    bool chroma_use_t5_mask      = false;
    int  chroma_t5_mask_pad      = 1;
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
    printf("    controlnet_path:   %s\n", params.controlnet_path.c_str());
    printf("    embeddings_path:   %s\n", params.embeddings_path.c_str());
    printf("    stacked_id_embeddings_path:   %s\n", params.stacked_id_embeddings_path.c_str());
    printf("    input_id_images_path:   %s\n", params.input_id_images_path.c_str());
    printf("    style ratio:       %.2f\n", params.style_ratio);
    printf("    normalize input image :  %s\n", params.normalize_input ? "true" : "false");
    printf("    output_path:       %s\n", params.output_path.c_str());
    printf("    init_img:          %s\n", params.input_path.c_str());
    printf("    mask_img:          %s\n", params.mask_path.c_str());
    printf("    control_image:     %s\n", params.control_image_path.c_str());
    printf("    clip on cpu:       %s\n", params.clip_on_cpu ? "true" : "false");
    printf("    controlnet cpu:    %s\n", params.control_net_cpu ? "true" : "false");
    printf("    vae decoder on cpu:%s\n", params.vae_on_cpu ? "true" : "false");
    printf("    diffusion flash attention:%s\n", params.diffusion_flash_attn ? "true" : "false");
    printf("    strength(control): %.2f\n", params.control_strength);
    printf("    prompt:            %s\n", params.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.negative_prompt.c_str());
    printf("    min_cfg:           %.2f\n", params.min_cfg);
    printf("    cfg_scale:         %.2f\n", params.cfg_scale);
    printf("    slg_scale:         %.2f\n", params.slg_scale);
    printf("    guidance:          %.2f\n", params.guidance);
    printf("    eta:               %.2f\n", params.eta);
    printf("    clip_skip:         %d\n", params.clip_skip);
    printf("    width:             %d\n", params.width);
    printf("    height:            %d\n", params.height);
    printf("    sample_method:     %s\n", sample_method_str[params.sample_method]);
    printf("    schedule:          %s\n", schedule_str[params.schedule]);
    printf("    sample_steps:      %d\n", params.sample_steps);
    printf("    strength(img2img): %.2f\n", params.strength);
    printf("    rng:               %s\n", rng_type_to_str[params.rng_type]);
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
    printf("  -M, --mode [MODEL]                 run mode (txt2img or img2img or convert, default: txt2img)\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1)\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model [MODEL]                path to full model\n");
    printf("  --diffusion-model                  path to the standalone diffusion model\n");
    printf("  --clip_l                           path to the clip-l text encoder\n");
    printf("  --clip_g                           path to the clip-g text encoder\n");
    printf("  --t5xxl                            path to the the t5xxl text encoder\n");
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
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  -i, --init-img [IMAGE]             path to the input image, required by img2img\n");
    printf("  --mask [MASK]                      path to the mask image, required by img2img with mask\n");
    printf("  --control-image [IMAGE]            path to image condition, control net\n");
    printf("  -o, --output OUTPUT                path to write result image to (default: ./output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       the negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)\n");
    printf("  --guidance SCALE                   guidance scale for img2img (default: 3.5)\n");
    printf("  --slg-scale SCALE                  skip layer guidance (SLG) scale, only for DiT models: (default: 0)\n");
    printf("                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium\n");
    printf("  --eta SCALE                        eta in DDIM, only for DDIM and TCD: (default: 0)\n");
    printf("  --skip-layers LAYERS               Layers to skip for SLG steps: (default: [7,8,9])\n");
    printf("  --skip-layer-start START           SLG enabling point: (default: 0.01)\n");
    printf("  --skip-layer-end END               SLG disabling point: (default: 0.2)\n");
    printf("                                     SLG will be enabled at step int([STEPS]*[START]) and disabled at int([STEPS]*[END])\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("  --style-ratio STYLE-RATIO          strength for keeping input identity (default: 20%%)\n");
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
    printf("  --control-net-cpu                  keep controlnet in cpu (for low vram)\n");
    printf("  --canny                            apply canny preprocessor (edge detection)\n");
    printf("  --color                            Colors the logging tags according to level\n");
    printf("  --chroma-disable-dit-mask          disable dit mask for chroma\n");
    printf("  --chroma-enable-t5-mask            enable t5 mask for chroma\n");
    printf("  --chroma-t5-mask-pad  PAD_SIZE     t5 mask pad size of chroma\n");
    printf("  -v, --verbose                      print extra info\n");
    printf("  -ki, --kontext_img [PATH]        Reference image for Flux Kontext models (can be used multiple times) \n");
}

void parse_args(int argc, const char** argv, SDParams& params) {
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-M" || arg == "--mode") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* mode_selected = argv[i];
            int mode_found            = -1;
            for (int d = 0; d < MODE_COUNT; d++) {
                if (!strcmp(mode_selected, modes_str[d])) {
                    mode_found = d;
                }
            }
            if (mode_found == -1) {
                fprintf(stderr,
                        "error: invalid mode %s, must be one of [txt2img, img2img, img2vid, convert]\n",
                        mode_selected);
                exit(1);
            }
            params.mode = (SDMode)mode_found;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.model_path = argv[i];
        } else if (arg == "--clip_l") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_l_path = argv[i];
        } else if (arg == "--clip_g") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_g_path = argv[i];
        } else if (arg == "--t5xxl") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.t5xxl_path = argv[i];
        } else if (arg == "--diffusion-model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.diffusion_model_path = argv[i];
        } else if (arg == "--vae") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.vae_path = argv[i];
        } else if (arg == "--taesd") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.taesd_path = argv[i];
        } else if (arg == "--control-net") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.controlnet_path = argv[i];
        } else if (arg == "--upscale-model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.esrgan_path = argv[i];
        } else if (arg == "--embd-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.embeddings_path = argv[i];
        } else if (arg == "--stacked-id-embd-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.stacked_id_embeddings_path = argv[i];
        } else if (arg == "--input-id-images-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.input_id_images_path = argv[i];
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            std::string type        = argv[i];
            bool found              = false;
            std::string valid_types = "";
            for (size_t i = 0; i < SD_TYPE_COUNT; i++) {
                auto trait = ggml_get_type_traits((ggml_type)i);
                std::string name(trait->type_name);
                if (name == "f32" || trait->to_float && trait->type_size) {
                    if (i)
                        valid_types += ", ";
                    valid_types += name;
                    if (type == name) {
                        if (ggml_quantize_requires_imatrix((ggml_type)i)) {
                            printf("\033[35;1m[WARNING]\033[0m: type %s requires imatrix to work properly. A dummy imatrix will be used, expect poor quality.\n", trait->type_name);
                        }
                        params.wtype = (enum sd_type_t)i;
                        found        = true;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr, "error: invalid weight format %s, must be one of [%s]\n",
                        type.c_str(),
                        valid_types.c_str());
                exit(1);
            }
        } else if (arg == "--lora-model-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lora_model_dir = argv[i];
        } else if (arg == "-i" || arg == "--init-img") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.input_path = argv[i];
        } else if (arg == "--mask") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.mask_path = argv[i];
        } else if (arg == "--control-image") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.control_image_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.output_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.prompt = argv[i];
        } else if (arg == "--upscale-repeats") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.upscale_repeats = std::stoi(argv[i]);
            if (params.upscale_repeats < 1) {
                fprintf(stderr, "error: upscale multiplier must be at least 1\n");
                exit(1);
            }
        } else if (arg == "-n" || arg == "--negative-prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.negative_prompt = argv[i];
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.cfg_scale = std::stof(argv[i]);
        } else if (arg == "--guidance") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.guidance = std::stof(argv[i]);
        } else if (arg == "--eta") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.eta = std::stof(argv[i]);
        } else if (arg == "--strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.strength = std::stof(argv[i]);
        } else if (arg == "--style-ratio") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.style_ratio = std::stof(argv[i]);
        } else if (arg == "--control-strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.control_strength = std::stof(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.height = std::stoi(argv[i]);
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.width = std::stoi(argv[i]);
        } else if (arg == "--steps") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sample_steps = std::stoi(argv[i]);
        } else if (arg == "--clip-skip") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_skip = std::stoi(argv[i]);
        } else if (arg == "--vae-tiling") {
            params.vae_tiling = true;
        } else if (arg == "--control-net-cpu") {
            params.control_net_cpu = true;
        } else if (arg == "--normalize-input") {
            params.normalize_input = true;
        } else if (arg == "--clip-on-cpu") {
            params.clip_on_cpu = true;  // will slow down get_learned_condiotion but necessary for low MEM GPUs
        } else if (arg == "--vae-on-cpu") {
            params.vae_on_cpu = true;  // will slow down latent decoding but necessary for low MEM GPUs
        } else if (arg == "--diffusion-fa") {
            params.diffusion_flash_attn = true;  // can reduce MEM significantly
        } else if (arg == "--canny") {
            params.canny_preprocess = true;
        } else if (arg == "-b" || arg == "--batch-count") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.batch_count = std::stoi(argv[i]);
        } else if (arg == "--rng") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            std::string rng_type_str = argv[i];
            if (rng_type_str == "std_default") {
                params.rng_type = STD_DEFAULT_RNG;
            } else if (rng_type_str == "cuda") {
                params.rng_type = CUDA_RNG;
            } else {
                invalid_arg = true;
                break;
            }
        } else if (arg == "--schedule") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* schedule_selected = argv[i];
            int schedule_found            = -1;
            for (int d = 0; d < N_SCHEDULES; d++) {
                if (!strcmp(schedule_selected, schedule_str[d])) {
                    schedule_found = d;
                }
            }
            if (schedule_found == -1) {
                invalid_arg = true;
                break;
            }
            params.schedule = (schedule_t)schedule_found;
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.seed = std::stoll(argv[i]);
        } else if (arg == "--sampling-method") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* sample_method_selected = argv[i];
            int sample_method_found            = -1;
            for (int m = 0; m < N_SAMPLE_METHODS; m++) {
                if (!strcmp(sample_method_selected, sample_method_str[m])) {
                    sample_method_found = m;
                }
            }
            if (sample_method_found == -1) {
                invalid_arg = true;
                break;
            }
            params.sample_method = (sample_method_t)sample_method_found;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--color") {
            params.color = true;
        } else if (arg == "--slg-scale") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.slg_scale = std::stof(argv[i]);
        } else if (arg == "--skip-layers") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            if (argv[i][0] != '[') {
                invalid_arg = true;
                break;
            }
            std::string layers_str = argv[i];
            while (layers_str.back() != ']') {
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                layers_str += " " + std::string(argv[i]);
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
                    invalid_arg = true;
                    break;
                }
            }
            params.skip_layers = layers;

            if (invalid_arg) {
                break;
            }
        } else if (arg == "--skip-layer-start") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.skip_layer_start = std::stof(argv[i]);
        } else if (arg == "--skip-layer-end") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.skip_layer_end = std::stof(argv[i]);
        } else if (arg == "-ki" || arg == "--kontext-img") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.kontext_image_paths.push_back(argv[i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
    }
    if (invalid_arg) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }
    if (params.n_threads <= 0) {
        params.n_threads = sd_get_num_physical_cores();
    }

    if (params.mode != CONVERT && params.mode != IMG2VID && params.prompt.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: prompt\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.model_path.length() == 0 && params.diffusion_model_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: model_path/diffusion_model\n");
        print_usage(argc, argv);
        exit(1);
    }

    if ((params.mode == IMG2IMG || params.mode == IMG2VID) && params.input_path.length() == 0) {
        fprintf(stderr, "error: when using the img2img mode, the following arguments are required: init-img\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.output_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: output_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.width <= 0 || params.width % 64 != 0) {
        fprintf(stderr, "error: the width must be a multiple of 64\n");
        exit(1);
    }

    if (params.height <= 0 || params.height % 64 != 0) {
        fprintf(stderr, "error: the height must be a multiple of 64\n");
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

    if (params.seed < 0) {
        srand((int)time(NULL));
        params.seed = rand();
    }

    if (params.mode == CONVERT) {
        if (params.output_path == "output.png") {
            params.output_path = "output.gguf";
        }
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
    parameter_string += "RNG: " + std::string(rng_type_to_str[params.rng_type]) + ", ";
    parameter_string += "Sampler: " + std::string(sample_method_str[params.sample_method]);
    if (params.schedule == KARRAS) {
        parameter_string += " karras";
    }
    parameter_string += ", ";
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

    sd_set_log_callback(sd_log_cb, (void*)&params);

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info());
    }

    if (params.mode == CONVERT) {
        bool success = convert(params.model_path.c_str(), params.vae_path.c_str(), params.output_path.c_str(), params.wtype);
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

    if (params.mode == IMG2VID) {
        fprintf(stderr, "SVD support is broken, do not use it!!!\n");
        return 1;
    }
    bool vae_decode_only = true;

    std::vector<sd_image_t> kontext_imgs;
    for (auto& path : params.kontext_image_paths) {
        vae_decode_only       = false;
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
        kontext_imgs.push_back({(uint32_t)width,
                                (uint32_t)height,
                                3,
                                image_buffer});
    }

    uint8_t* input_image_buffer   = NULL;
    uint8_t* control_image_buffer = NULL;
    uint8_t* mask_image_buffer    = NULL;

    if (params.mode == IMG2IMG || params.mode == IMG2VID) {
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
    }

    sd_ctx_t* sd_ctx = new_sd_ctx(params.model_path.c_str(),
                                  params.clip_l_path.c_str(),
                                  params.clip_g_path.c_str(),
                                  params.t5xxl_path.c_str(),
                                  params.diffusion_model_path.c_str(),
                                  params.vae_path.c_str(),
                                  params.taesd_path.c_str(),
                                  params.controlnet_path.c_str(),
                                  params.lora_model_dir.c_str(),
                                  params.embeddings_path.c_str(),
                                  params.stacked_id_embeddings_path.c_str(),
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
                                  params.chroma_use_dit_mask,
                                  params.chroma_use_t5_mask,
                                  params.chroma_t5_mask_pad);

    if (sd_ctx == NULL) {
        printf("new_sd_ctx_t failed\n");
        return 1;
    }

    sd_image_t* control_image = NULL;
    if (params.controlnet_path.size() > 0 && params.control_image_path.size() > 0) {
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
    if (params.mode == TXT2IMG) {
        results = txt2img(sd_ctx,
                          params.prompt.c_str(),
                          params.negative_prompt.c_str(),
                          params.clip_skip,
                          params.cfg_scale,
                          params.guidance,
                          params.eta,
                          params.width,
                          params.height,
                          params.sample_method,
                          params.sample_steps,
                          params.seed,
                          params.batch_count,
                          control_image,
                          params.control_strength,
                          params.style_ratio,
                          params.normalize_input,
                          params.input_id_images_path.c_str(),
                          kontext_imgs.data(), kontext_imgs.size(),
                          params.skip_layers.data(),
                          params.skip_layers.size(),
                          params.slg_scale,
                          params.skip_layer_start,
                          params.skip_layer_end,
                          std::vector<sd_image_t*>());
    } else {
        sd_image_t input_image = {(uint32_t)params.width,
                                  (uint32_t)params.height,
                                  3,
                                  input_image_buffer};

        if (params.mode == IMG2VID) {
            results = img2vid(sd_ctx,
                              input_image,
                              params.width,
                              params.height,
                              params.video_frames,
                              params.motion_bucket_id,
                              params.fps,
                              params.augmentation_level,
                              params.min_cfg,
                              params.cfg_scale,
                              params.sample_method,
                              params.sample_steps,
                              params.strength,
                              params.seed);
            if (results == NULL) {
                printf("generate failed\n");
                free_sd_ctx(sd_ctx);
                return 1;
            }
            size_t last            = params.output_path.find_last_of(".");
            std::string dummy_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
            for (int i = 0; i < params.video_frames; i++) {
                if (results[i].data == NULL) {
                    continue;
                }
                std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ".png" : dummy_name + ".png";
                stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                               results[i].data, 0, get_image_params(params, params.seed + i).c_str());
                printf("save result image to '%s'\n", final_image_path.c_str());
                free(results[i].data);
                results[i].data = NULL;
            }
            free(results);
            free_sd_ctx(sd_ctx);
            return 0;
        } else {
            results = img2img(sd_ctx,
                              input_image,
                              mask_image,
                              params.prompt.c_str(),
                              params.negative_prompt.c_str(),
                              params.clip_skip,
                              params.cfg_scale,
                              params.guidance,
                              params.eta,
                              params.width,
                              params.height,
                              params.sample_method,
                              params.sample_steps,
                              params.strength,
                              params.seed,
                              params.batch_count,
                              control_image,
                              params.control_strength,
                              params.style_ratio,
                              params.normalize_input,
                              params.input_id_images_path.c_str(),
                              kontext_imgs.data(), kontext_imgs.size(),
                              params.skip_layers.data(),
                              params.skip_layers.size(),
                              params.slg_scale,
                              params.skip_layer_start,
                              params.skip_layer_end,
                              std::vector<sd_image_t*>());
        }
    }

    if (results == NULL) {
        printf("generate failed\n");
        free_sd_ctx(sd_ctx);
        return 1;
    }

    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (params.esrgan_path.size() > 0 && params.upscale_repeats > 0) {
        upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(params.esrgan_path.c_str(),
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
    for (int i = 0; i < params.batch_count; i++) {
        if (results[i].data == NULL) {
            continue;
        }
        std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ext : dummy_name + ext;
        if (is_jpg) {
            stbi_write_jpg(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                           results[i].data, 90);
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