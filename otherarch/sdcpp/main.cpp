#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cctype>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
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
    "upscale",
};
#define SD_ALL_MODES_STR "img_gen, vid_gen, convert, upscale"

const char* previews_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

enum SDMode {
    IMG_GEN,
    VID_GEN,
    CONVERT,
    UPSCALE,
    MODE_COUNT
};

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

    static std::string wrap_text(const std::string& text, size_t width, size_t indent) {
        std::ostringstream oss;
        size_t line_len = 0;
        size_t pos      = 0;

        while (pos < text.size()) {
            // Preserve manual newlines
            if (text[pos] == '\n') {
                oss << '\n'
                    << std::string(indent, ' ');
                line_len = indent;
                ++pos;
                continue;
            }

            // Add the character
            oss << text[pos];
            ++line_len;
            ++pos;

            // If the current line exceeds width, try to break at the last space
            if (line_len >= width) {
                std::string current = oss.str();
                size_t back         = current.size();

                // Find the last space (for a clean break)
                while (back > 0 && current[back - 1] != ' ' && current[back - 1] != '\n')
                    --back;

                // If found a space to break on
                if (back > 0 && current[back - 1] != '\n') {
                    std::string before = current.substr(0, back - 1);
                    std::string after  = current.substr(back);
                    oss.str("");
                    oss.clear();
                    oss << before << "\n"
                        << std::string(indent, ' ') << after;
                } else {
                    // If no space found, just break at width
                    oss << "\n"
                        << std::string(indent, ' ');
                }
                line_len = indent;
            }
        }

        return oss.str();
    }

    void print() const {
        constexpr size_t max_line_width = 120;

        struct Entry {
            std::string names;
            std::string desc;
        };
        std::vector<Entry> entries;

        auto add_entry = [&](const std::string& s, const std::string& l,
                             const std::string& desc, const std::string& hint = "") {
            std::ostringstream ss;
            if (!s.empty())
                ss << s;
            if (!s.empty() && !l.empty())
                ss << ", ";
            if (!l.empty())
                ss << l;
            if (!hint.empty())
                ss << " " << hint;
            entries.push_back({ss.str(), desc});
        };

        for (auto& o : string_options)
            add_entry(o.short_name, o.long_name, o.desc, "<string>");
        for (auto& o : int_options)
            add_entry(o.short_name, o.long_name, o.desc, "<int>");
        for (auto& o : float_options)
            add_entry(o.short_name, o.long_name, o.desc, "<float>");
        for (auto& o : bool_options)
            add_entry(o.short_name, o.long_name, o.desc, "");
        for (auto& o : manual_options)
            add_entry(o.short_name, o.long_name, o.desc);

        size_t max_name_width = 0;
        for (auto& e : entries)
            max_name_width = std::max(max_name_width, e.names.size());

        for (auto& e : entries) {
            size_t indent            = 2 + max_name_width + 4;
            size_t desc_width        = (max_line_width > indent ? max_line_width - indent : 40);
            std::string wrapped_desc = wrap_text(e.desc, max_line_width, indent);
            std::cout << "  " << std::left << std::setw(static_cast<int>(max_name_width) + 4)
                      << e.names << wrapped_desc << "\n";
        }
    }
};

bool parse_options(int argc, const char** argv, const std::vector<ArgOptions>& options_list) {
    bool invalid_arg = false;
    std::string arg;

    auto match_and_apply = [&](auto& opts, auto&& apply_fn) -> bool {
        for (auto& option : opts) {
            if ((option.short_name.size() > 0 && arg == option.short_name) ||
                (option.long_name.size() > 0 && arg == option.long_name)) {
                apply_fn(option);
                return true;
            }
        }
        return false;
    };

    for (int i = 1; i < argc; i++) {
        arg            = argv[i];
        bool found_arg = false;

        for (auto& options : options_list) {
            if (match_and_apply(options.string_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    *option.target = argv_to_utf8(i, argv);
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.int_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    *option.target = std::stoi(argv[i]);
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.float_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    *option.target = std::stof(argv[i]);
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.bool_options, [&](auto& option) {
                    *option.target = option.keep_true ? true : false;
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.manual_options, [&](auto& option) {
                    int ret = option.cb(argc, argv, i);
                    if (ret < 0) {
                        invalid_arg = true;
                        return;
                    }
                    i += ret;
                    found_arg = true;
                }))
                break;
        }

        if (invalid_arg) {
            fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
            return false;
        }
        if (!found_arg) {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    return true;
}

struct SDCliParams {
    SDMode mode             = IMG_GEN;
    std::string output_path = "output.png";

    bool verbose          = false;
    bool canny_preprocess = false;

    preview_t preview_method = PREVIEW_NONE;
    int preview_interval     = 1;
    std::string preview_path = "preview.png";
    int preview_fps          = 16;
    bool taesd_preview       = false;
    bool preview_noisy       = false;
    bool color               = false;

    bool normal_exit = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-o",
             "--output",
             "path to write result image to (default: ./output.png)",
             &output_path},
            {"",
             "--preview-path",
             "path to write preview image to (default: ./preview.png)",
             &preview_path},
        };

        options.int_options = {
            {"",
             "--preview-interval",
             "interval in denoising steps between consecutive updates of the image preview file (default is 1, meaning updating at every step)",
             &preview_interval},
        };

        options.bool_options = {
            {"",
             "--canny",
             "apply canny preprocessor (edge detection)",
             true, &canny_preprocess},
            {"-v",
             "--verbose",
             "print extra info",
             true, &verbose},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
            {"",
             "--taesd-preview-only",
             std::string("prevents usage of taesd for decoding the final image. (for use with --preview ") + previews_str[PREVIEW_TAE] + ")",
             true, &taesd_preview},
            {"",
             "--preview-noisy",
             "enables previewing noisy inputs of the models rather than the denoised outputs",
             true, &preview_noisy},

        };

        auto on_mode_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* mode_c_str = argv[index];
            if (mode_c_str != nullptr) {
                int mode_found = -1;
                for (int i = 0; i < MODE_COUNT; i++) {
                    if (!strcmp(mode_c_str, modes_str[i])) {
                        mode_found = i;
                    }
                }
                if (mode_found == -1) {
                    fprintf(stderr,
                            "error: invalid mode %s, must be one of [%s]\n",
                            mode_c_str, SD_ALL_MODES_STR);
                    exit(1);
                }
                mode = (SDMode)mode_found;
            }
            return 1;
        };

        auto on_preview_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* preview = argv[index];
            int preview_method  = -1;
            for (int m = 0; m < PREVIEW_COUNT; m++) {
                if (!strcmp(preview, previews_str[m])) {
                    preview_method = m;
                }
            }
            if (preview_method == -1) {
                fprintf(stderr, "error: preview method %s\n",
                        preview);
                return -1;
            }
            preview_method = (preview_t)preview_method;
            return 1;
        };

        auto on_help_arg = [&](int argc, const char** argv, int index) {
            normal_exit = true;
            return -1;
        };

        options.manual_options = {
            {"-M",
             "--mode",
             "run mode, one of [img_gen, vid_gen, upscale, convert], default: img_gen",
             on_mode_arg},
            {"",
             "--preview",
             std::string("preview method. must be one of the following [") + previews_str[0] + ", " + previews_str[1] + ", " + previews_str[2] + ", " + previews_str[3] + "] (default is " + previews_str[PREVIEW_NONE] + ")",
             on_preview_arg},
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
        };

        return options;
    };

    bool process_and_check() {
        if (output_path.length() == 0) {
            fprintf(stderr, "error: the following arguments are required: output_path\n");
            return false;
        }

        if (mode == CONVERT) {
            if (output_path == "output.png") {
                output_path = "output.gguf";
            }
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDCliParams {\n"
            << "  mode: " << modes_str[mode] << ",\n"
            << "  output_path: \"" << output_path << "\",\n"
            << "  verbose: " << (verbose ? "true" : "false") << ",\n"
            << "  color: " << (color ? "true" : "false") << ",\n"
            << "  canny_preprocess: " << (canny_preprocess ? "true" : "false") << ",\n"
            << "  preview_method: " << previews_str[preview_method] << ",\n"
            << "  preview_interval: " << preview_interval << ",\n"
            << "  preview_path: \"" << preview_path << "\",\n"
            << "  preview_fps: " << preview_fps << ",\n"
            << "  taesd_preview: " << (taesd_preview ? "true" : "false") << ",\n"
            << "  preview_noisy: " << (preview_noisy ? "true" : "false") << "\n"
            << "}";
        return oss.str();
    }
};

struct SDContextParams {
    int n_threads = -1;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string control_net_path;
    std::string embedding_dir;
    std::string photo_maker_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    std::string lora_model_dir;

    rng_type_t rng_type         = CUDA_RNG;
    rng_type_t sampler_rng_type = RNG_TYPE_COUNT;
    bool offload_params_to_cpu  = false;
    bool control_net_cpu        = false;
    bool clip_on_cpu            = false;
    bool vae_on_cpu             = false;
    bool diffusion_flash_attn   = false;
    bool diffusion_conv_direct  = false;
    bool vae_conv_direct        = false;

    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask  = false;
    int chroma_t5_mask_pad   = 1;

    prediction_t prediction           = DEFAULT_PRED;
    lora_apply_mode_t lora_apply_mode = LORA_APPLY_AUTO;

    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    bool force_sdxl_vae_conv_scale       = false;

    float flow_shift = INFINITY;

    ArgOptions get_options() {
        ArgOptions options;
        options.string_options = {
            {"-m",
             "--model",
             "path to full model",
             &model_path},
            {"",
             "--clip_l",
             "path to the clip-l text encoder", &clip_l_path},
            {"", "--clip_g",
             "path to the clip-g text encoder",
             &clip_g_path},
            {"",
             "--clip_vision",
             "path to the clip-vision encoder",
             &clip_vision_path},
            {"",
             "--t5xxl",
             "path to the t5xxl text encoder",
             &t5xxl_path},
            {"",
             "--llm",
             "path to the llm text encoder. For example: (qwenvl2.5 for qwen-image, mistral-small3.2 for flux2, ...)",
             &llm_path},
            {"",
             "--llm_vision",
             "path to the llm vit",
             &llm_vision_path},
            {"",
             "--qwen2vl",
             "alias of --llm. Deprecated.",
             &llm_path},
            {"",
             "--qwen2vl_vision",
             "alias of --llm_vision. Deprecated.",
             &llm_vision_path},
            {"",
             "--diffusion-model",
             "path to the standalone diffusion model",
             &diffusion_model_path},
            {"",
             "--high-noise-diffusion-model",
             "path to the standalone high noise diffusion model",
             &high_noise_diffusion_model_path},
            {"",
             "--vae",
             "path to standalone vae model",
             &vae_path},
            {"",
             "--taesd",
             "path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)",
             &taesd_path},
            {"",
             "--control-net",
             "path to control net model",
             &control_net_path},
            {"",
             "--embd-dir",
             "embeddings directory",
             &embedding_dir},
            {"",
             "--lora-model-dir",
             "lora model directory",
             &lora_model_dir},

            {"",
             "--tensor-type-rules",
             "weight type per tensor pattern (example: \"^vae\\.=f16,model\\.=q8_0\")",
             &tensor_type_rules},
            {"",
             "--photo-maker",
             "path to PHOTOMAKER model",
             &photo_maker_path},
            {"",
             "--upscale-model",
             "path to esrgan model.",
             &esrgan_path},
        };

        options.int_options = {
            {"-t",
             "--threads",
             "number of threads to use during computation (default: -1). "
             "If threads <= 0, then threads will be set to the number of CPU physical cores",
             &n_threads},
            {"",
             "--chroma-t5-mask-pad",
             "t5 mask pad size of chroma",
             &chroma_t5_mask_pad},
        };

        options.float_options = {
            {"",
             "--vae-tile-overlap",
             "tile overlap for vae tiling, in fraction of tile size (default: 0.5)",
             &vae_tiling_params.target_overlap},
            {"",
             "--flow-shift",
             "shift value for Flow models like SD3.x or WAN (default: auto)",
             &flow_shift},
        };

        options.bool_options = {
            {"",
             "--vae-tiling",
             "process vae in tiles to reduce memory usage",
             true, &vae_tiling_params.enabled},
            {"",
             "--force-sdxl-vae-conv-scale",
             "force use of conv scale on sdxl vae",
             true, &force_sdxl_vae_conv_scale},
            {"",
             "--offload-to-cpu",
             "place the weights in RAM to save VRAM, and automatically load them into VRAM when needed",
             true, &offload_params_to_cpu},
            {"",
             "--control-net-cpu",
             "keep controlnet in cpu (for low vram)",
             true, &control_net_cpu},
            {"",
             "--clip-on-cpu",
             "keep clip in cpu (for low vram)",
             true, &clip_on_cpu},
            {"",
             "--vae-on-cpu",
             "keep vae in cpu (for low vram)",
             true, &vae_on_cpu},
            {"",
             "--diffusion-fa",
             "use flash attention in the diffusion model",
             true, &diffusion_flash_attn},
            {"",
             "--diffusion-conv-direct",
             "use ggml_conv2d_direct in the diffusion model",
             true, &diffusion_conv_direct},
            {"",
             "--vae-conv-direct",
             "use ggml_conv2d_direct in the vae model",
             true, &vae_conv_direct},
            {"",
             "--chroma-disable-dit-mask",
             "disable dit mask for chroma",
             false, &chroma_use_dit_mask},
            {"",
             "--chroma-enable-t5-mask",
             "enable t5 mask for chroma",
             true, &chroma_use_t5_mask},
        };

        auto on_type_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg = argv[index];
            wtype           = str_to_sd_type(arg);
            if (wtype == SD_TYPE_COUNT) {
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
            rng_type        = str_to_rng_type(arg);
            if (rng_type == RNG_TYPE_COUNT) {
                fprintf(stderr, "error: invalid rng type %s\n",
                        arg);
                return -1;
            }
            return 1;
        };

        auto on_sampler_rng_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg  = argv[index];
            sampler_rng_type = str_to_rng_type(arg);
            if (sampler_rng_type == RNG_TYPE_COUNT) {
                fprintf(stderr, "error: invalid sampler rng type %s\n",
                        arg);
                return -1;
            }
            return 1;
        };

        auto on_prediction_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg = argv[index];
            prediction      = str_to_prediction(arg);
            if (prediction == PREDICTION_COUNT) {
                fprintf(stderr, "error: invalid prediction type %s\n",
                        arg);
                return -1;
            }
            return 1;
        };

        auto on_lora_apply_mode_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg = argv[index];
            lora_apply_mode = str_to_lora_apply_mode(arg);
            if (lora_apply_mode == LORA_APPLY_MODE_COUNT) {
                fprintf(stderr, "error: invalid lora apply model %s\n",
                        arg);
                return -1;
            }
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
                    std::string tile_x_str        = tile_size_str.substr(0, x_pos);
                    std::string tile_y_str        = tile_size_str.substr(x_pos + 1);
                    vae_tiling_params.tile_size_x = std::stoi(tile_x_str);
                    vae_tiling_params.tile_size_y = std::stoi(tile_y_str);
                } else {
                    vae_tiling_params.tile_size_x = vae_tiling_params.tile_size_y = std::stoi(tile_size_str);
                }
            } catch (const std::invalid_argument&) {
                return -1;
            } catch (const std::out_of_range&) {
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
                    std::string rel_x_str        = rel_size_str.substr(0, x_pos);
                    std::string rel_y_str        = rel_size_str.substr(x_pos + 1);
                    vae_tiling_params.rel_size_x = std::stof(rel_x_str);
                    vae_tiling_params.rel_size_y = std::stof(rel_y_str);
                } else {
                    vae_tiling_params.rel_size_x = vae_tiling_params.rel_size_y = std::stof(rel_size_str);
                }
            } catch (const std::invalid_argument&) {
                return -1;
            } catch (const std::out_of_range&) {
                return -1;
            }
            return 1;
        };

        options.manual_options = {
            {"",
             "--type",
             "weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K). "
             "If not specified, the default is the type of the weight file",
             on_type_arg},
            {"",
             "--rng",
             "RNG, one of [std_default, cuda, cpu], default: cuda(sd-webui), cpu(comfyui)",
             on_rng_arg},
            {"",
             "--sampler-rng",
             "sampler RNG, one of [std_default, cuda, cpu]. If not specified, use --rng",
             on_sampler_rng_arg},
            {"",
             "--prediction",
             "prediction type override, one of [eps, v, edm_v, sd3_flow, flux_flow, flux2_flow]",
             on_prediction_arg},
            {"",
             "--lora-apply-mode",
             "the way to apply LoRA, one of [auto, immediately, at_runtime], default is auto. "
             "In auto mode, if the model weights contain any quantized parameters, the at_runtime mode will be used; otherwise, immediately will be used."
             "The immediately mode may have precision and compatibility issues with quantized parameters, "
             "but it usually offers faster inference speed and, in some cases, lower memory usage. "
             "The at_runtime mode, on the other hand, is exactly the opposite.",
             on_lora_apply_mode_arg},
            {"",
             "--vae-tile-size",
             "tile size for vae tiling, format [X]x[Y] (default: 32x32)",
             on_tile_size_arg},
            {"",
             "--vae-relative-tile-size",
             "relative tile size for vae tiling, format [X]x[Y], in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides --vae-tile-size)",
             on_relative_tile_size_arg},
        };

        return options;
    }

    bool process_and_check(SDMode mode) {
        if (mode != UPSCALE && model_path.length() == 0 && diffusion_model_path.length() == 0) {
            fprintf(stderr, "error: the following arguments are required: model_path/diffusion_model\n");
            return false;
        }

        if (mode == UPSCALE) {
            if (esrgan_path.length() == 0) {
                fprintf(stderr, "error: upscale mode needs an upscaler model (--upscale-model)\n");
                return false;
            }
        }

        if (n_threads <= 0) {
            n_threads = sd_get_num_physical_cores();
        }

        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDContextParams {\n"
            << "  n_threads: " << n_threads << ",\n"
            << "  model_path: \"" << model_path << "\",\n"
            << "  clip_l_path: \"" << clip_l_path << "\",\n"
            << "  clip_g_path: \"" << clip_g_path << "\",\n"
            << "  clip_vision_path: \"" << clip_vision_path << "\",\n"
            << "  t5xxl_path: \"" << t5xxl_path << "\",\n"
            << "  llm_path: \"" << llm_path << "\",\n"
            << "  llm_vision_path: \"" << llm_vision_path << "\",\n"
            << "  diffusion_model_path: \"" << diffusion_model_path << "\",\n"
            << "  high_noise_diffusion_model_path: \"" << high_noise_diffusion_model_path << "\",\n"
            << "  vae_path: \"" << vae_path << "\",\n"
            << "  taesd_path: \"" << taesd_path << "\",\n"
            << "  esrgan_path: \"" << esrgan_path << "\",\n"
            << "  control_net_path: \"" << control_net_path << "\",\n"
            << "  embedding_dir: \"" << embedding_dir << "\",\n"
            << "  wtype: " << sd_type_name(wtype) << ",\n"
            << "  tensor_type_rules: \"" << tensor_type_rules << "\",\n"
            << "  lora_model_dir: \"" << lora_model_dir << "\",\n"
            << "  photo_maker_path: \"" << photo_maker_path << "\",\n"
            << "  rng_type: " << sd_rng_type_name(rng_type) << ",\n"
            << "  sampler_rng_type: " << sd_rng_type_name(sampler_rng_type) << ",\n"
            << "  flow_shift: " << (std::isinf(flow_shift) ? "INF" : std::to_string(flow_shift)) << "\n"
            << "  offload_params_to_cpu: " << (offload_params_to_cpu ? "true" : "false") << ",\n"
            << "  control_net_cpu: " << (control_net_cpu ? "true" : "false") << ",\n"
            << "  clip_on_cpu: " << (clip_on_cpu ? "true" : "false") << ",\n"
            << "  vae_on_cpu: " << (vae_on_cpu ? "true" : "false") << ",\n"
            << "  diffusion_flash_attn: " << (diffusion_flash_attn ? "true" : "false") << ",\n"
            << "  diffusion_conv_direct: " << (diffusion_conv_direct ? "true" : "false") << ",\n"
            << "  vae_conv_direct: " << (vae_conv_direct ? "true" : "false") << ",\n"
            << "  chroma_use_dit_mask: " << (chroma_use_dit_mask ? "true" : "false") << ",\n"
            << "  chroma_use_t5_mask: " << (chroma_use_t5_mask ? "true" : "false") << ",\n"
            << "  chroma_t5_mask_pad: " << chroma_t5_mask_pad << ",\n"
            << "  prediction: " << sd_prediction_name(prediction) << ",\n"
            << "  lora_apply_mode: " << sd_lora_apply_mode_name(lora_apply_mode) << ",\n"
            << "  vae_tiling_params: { "
            << vae_tiling_params.enabled << ", "
            << vae_tiling_params.tile_size_x << ", "
            << vae_tiling_params.tile_size_y << ", "
            << vae_tiling_params.target_overlap << ", "
            << vae_tiling_params.rel_size_x << ", "
            << vae_tiling_params.rel_size_y << " },\n"
            << "  force_sdxl_vae_conv_scale: " << (force_sdxl_vae_conv_scale ? "true" : "false") << "\n"
            << "}";
        return oss.str();
    }

    sd_ctx_params_t to_sd_ctx_params_t(bool vae_decode_only, bool free_params_immediately, bool taesd_preview) {
        sd_ctx_params_t sd_ctx_params = {
            model_path.c_str(),
            clip_l_path.c_str(),
            clip_g_path.c_str(),
            clip_vision_path.c_str(),
            t5xxl_path.c_str(),
            llm_path.c_str(),
            llm_vision_path.c_str(),
            diffusion_model_path.c_str(),
            high_noise_diffusion_model_path.c_str(),
            vae_path.c_str(),
            taesd_path.c_str(),
            control_net_path.c_str(),
            lora_model_dir.c_str(),
            embedding_dir.c_str(),
            photo_maker_path.c_str(),
            tensor_type_rules.c_str(),
            vae_decode_only,
            free_params_immediately,
            n_threads,
            wtype,
            rng_type,
            sampler_rng_type,
            prediction,
            lora_apply_mode,
            offload_params_to_cpu,
            clip_on_cpu,
            control_net_cpu,
            vae_on_cpu,
            diffusion_flash_attn,
            taesd_preview,
            diffusion_conv_direct,
            vae_conv_direct,
            force_sdxl_vae_conv_scale,
            chroma_use_dit_mask,
            chroma_use_t5_mask,
            chroma_t5_mask_pad,
            flow_shift,
        };
        return sd_ctx_params;
    }
};

template <typename T>
static std::string vec_to_string(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        oss << v[i];
        if (i + 1 < v.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

static std::string vec_str_to_string(const std::vector<std::string>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        oss << "\"" << v[i] << "\"";
        if (i + 1 < v.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

struct SDGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int clip_skip   = -1;  // <= 0 represents unspecified
    int width       = 512;
    int height      = 512;
    int batch_count = 1;
    std::string init_image_path;
    std::string end_image_path;
    std::string mask_image_path;
    std::string control_image_path;
    std::vector<std::string> ref_image_paths;
    std::string control_video_path;
    bool auto_resize_ref_image = true;
    bool increase_ref_index    = false;

    std::vector<int> skip_layers = {7, 8, 9};
    sd_sample_params_t sample_params;

    std::vector<int> high_noise_skip_layers = {7, 8, 9};
    sd_sample_params_t high_noise_sample_params;

    std::string easycache_option;
    sd_easycache_params_t easycache_params;

    float moe_boundary  = 0.875f;
    int video_frames    = 1;
    int fps             = 16;
    float vace_strength = 1.f;

    float strength         = 0.75f;
    float control_strength = 0.9f;

    int64_t seed = 42;

    // Photo Maker
    std::string pm_id_images_dir;
    std::string pm_id_embed_path;
    float pm_style_strength = 20.f;

    int upscale_repeats = 1;

    SDGenerationParams() {
        sd_sample_params_init(&sample_params);
        sd_sample_params_init(&high_noise_sample_params);
    }

    ArgOptions get_options() {
        ArgOptions options;
        options.string_options = {
            {"-p",
             "--prompt",
             "the prompt to render",
             &prompt},
            {"-n",
             "--negative-prompt",
             "the negative prompt (default: \"\")",
             &negative_prompt},
            {"-i",
             "--init-img",
             "path to the init image",
             &init_image_path},
            {"",
             "--end-img",
             "path to the end image, required by flf2v",
             &end_image_path},
            {"",
             "--mask",
             "path to the mask image",
             &mask_image_path},
            {"",
             "--control-image",
             "path to control image, control net",
             &control_image_path},
            {"",
             "--control-video",
             "path to control video frames, It must be a directory path. The video frames inside should be stored as images in "
             "lexicographical (character) order. For example, if the control video path is `frames`, the directory contain images "
             "such as 00.png, 01.png, ... etc.",
             &control_video_path},
            {"",
             "--pm-id-images-dir",
             "path to PHOTOMAKER input id images dir",
             &pm_id_images_dir},
            {"",
             "--pm-id-embed-path",
             "path to PHOTOMAKER v2 id embed",
             &pm_id_embed_path},
        };

        options.int_options = {
            {"-H",
             "--height",
             "image height, in pixel space (default: 512)",
             &height},
            {"-W",
             "--width",
             "image width, in pixel space (default: 512)",
             &width},
            {"",
             "--steps",
             "number of sample steps (default: 20)",
             &sample_params.sample_steps},
            {"",
             "--high-noise-steps",
             "(high noise) number of sample steps (default: -1 = auto)",
             &high_noise_sample_params.sample_steps},
            {"",
             "--clip-skip",
             "ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1). "
             "<= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x",
             &clip_skip},
            {"-b",
             "--batch-count",
             "batch count",
             &batch_count},
            {"",
             "--video-frames",
             "video frames (default: 1)",
             &video_frames},
            {"",
             "--fps",
             "fps (default: 24)",
             &fps},
            {"",
             "--timestep-shift",
             "shift timestep for NitroFusion models (default: 0). "
             "recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant",
             &sample_params.shifted_timestep},
            {"",
             "--upscale-repeats",
             "Run the ESRGAN upscaler this many times (default: 1)",
             &upscale_repeats},
        };

        options.float_options = {
            {"",
             "--cfg-scale",
             "unconditional guidance scale: (default: 7.0)",
             &sample_params.guidance.txt_cfg},
            {"",
             "--img-cfg-scale",
             "image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)",
             &sample_params.guidance.img_cfg},
            {"",
             "--guidance",
             "distilled guidance scale for models with guidance input (default: 3.5)",
             &sample_params.guidance.distilled_guidance},
            {"",
             "--slg-scale",
             "skip layer guidance (SLG) scale, only for DiT models: (default: 0). 0 means disabled, a value of 2.5 is nice for sd3.5 medium",
             &sample_params.guidance.slg.scale},
            {"",
             "--skip-layer-start",
             "SLG enabling point (default: 0.01)",
             &sample_params.guidance.slg.layer_start},
            {"",
             "--skip-layer-end",
             "SLG disabling point (default: 0.2)",
             &sample_params.guidance.slg.layer_end},
            {"",
             "--eta",
             "eta in DDIM, only for DDIM and TCD (default: 0)",
             &sample_params.eta},
            {"",
             "--high-noise-cfg-scale",
             "(high noise) unconditional guidance scale: (default: 7.0)",
             &high_noise_sample_params.guidance.txt_cfg},
            {"",
             "--high-noise-img-cfg-scale",
             "(high noise) image guidance scale for inpaint or instruct-pix2pix models (default: same as --cfg-scale)",
             &high_noise_sample_params.guidance.img_cfg},
            {"",
             "--high-noise-guidance",
             "(high noise) distilled guidance scale for models with guidance input (default: 3.5)",
             &high_noise_sample_params.guidance.distilled_guidance},
            {"",
             "--high-noise-slg-scale",
             "(high noise) skip layer guidance (SLG) scale, only for DiT models: (default: 0)",
             &high_noise_sample_params.guidance.slg.scale},
            {"",
             "--high-noise-skip-layer-start",
             "(high noise) SLG enabling point (default: 0.01)",
             &high_noise_sample_params.guidance.slg.layer_start},
            {"",
             "--high-noise-skip-layer-end",
             "(high noise) SLG disabling point (default: 0.2)",
             &high_noise_sample_params.guidance.slg.layer_end},
            {"",
             "--high-noise-eta",
             "(high noise) eta in DDIM, only for DDIM and TCD (default: 0)",
             &high_noise_sample_params.eta},
            {"",
             "--strength",
             "strength for noising/unnoising (default: 0.75)",
             &strength},
            {"",
             "--pm-style-strength",
             "",
             &pm_style_strength},
            {"",
             "--control-strength",
             "strength to apply Control Net (default: 0.9). 1.0 corresponds to full destruction of information in init image",
             &control_strength},
            {"",
             "--moe-boundary",
             "timestep boundary for Wan2.2 MoE model. (default: 0.875). Only enabled if `--high-noise-steps` is set to -1",
             &moe_boundary},
            {"",
             "--vace-strength",
             "wan vace strength",
             &vace_strength},
        };

        options.bool_options = {
            {"",
             "--increase-ref-index",
             "automatically increase the indices of references images based on the order they are listed (starting with 1).",
             true,
             &increase_ref_index},
            {"",
             "--disable-auto-resize-ref-image",
             "disable auto resize of ref images",
             false,
             &auto_resize_ref_image},
        };

        auto on_seed_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            seed = std::stoll(argv[index]);
            return 1;
        };

        auto on_sample_method_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg             = argv[index];
            sample_params.sample_method = str_to_sample_method(arg);
            if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
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
            const char* arg                        = argv[index];
            high_noise_sample_params.sample_method = str_to_sample_method(arg);
            if (high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
                fprintf(stderr, "error: invalid high noise sample method %s\n",
                        arg);
                return -1;
            }
            return 1;
        };

        auto on_scheduler_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg         = argv[index];
            sample_params.scheduler = str_to_scheduler(arg);
            if (sample_params.scheduler == SCHEDULER_COUNT) {
                fprintf(stderr, "error: invalid scheduler %s\n",
                        arg);
                return -1;
            }
            return 1;
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
                } catch (const std::invalid_argument&) {
                    return -1;
                }
            }
            skip_layers = layers;
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
                } catch (const std::invalid_argument&) {
                    return -1;
                }
            }
            high_noise_skip_layers = layers;
            return 1;
        };

        auto on_ref_image_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            ref_image_paths.push_back(argv[index]);
            return 1;
        };

        auto on_easycache_arg = [&](int argc, const char** argv, int index) {
            const std::string default_values = "0.2,0.15,0.95";
            auto looks_like_value            = [](const std::string& token) {
                if (token.empty()) {
                    return false;
                }
                if (token[0] != '-') {
                    return true;
                }
                if (token.size() == 1) {
                    return false;
                }
                unsigned char next = static_cast<unsigned char>(token[1]);
                return std::isdigit(next) || token[1] == '.';
            };

            std::string option_value;
            int consumed = 0;
            if (index + 1 < argc) {
                std::string next_arg = argv[index + 1];
                if (looks_like_value(next_arg)) {
                    option_value = argv_to_utf8(index + 1, argv);
                    consumed     = 1;
                }
            }
            if (option_value.empty()) {
                option_value = default_values;
            }
            easycache_option = option_value;
            return consumed;
        };

        options.manual_options = {
            {"-s",
             "--seed",
             "RNG seed (default: 42, use random seed for < 0)",
             on_seed_arg},
            {"",
             "--sampling-method",
             "sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd] "
             "(default: euler for Flux/SD3/Wan, euler_a otherwise)",
             on_sample_method_arg},
            {"",
             "--high-noise-sampling-method",
             "(high noise) sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd]"
             " default: euler for Flux/SD3/Wan, euler_a otherwise",
             on_high_noise_sample_method_arg},
            {"",
             "--scheduler",
             "denoiser sigma scheduler, one of [discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple, lcm], default: discrete",
             on_scheduler_arg},
            {"",
             "--skip-layers",
             "layers to skip for SLG steps (default: [7,8,9])",
             on_skip_layers_arg},
            {"",
             "--high-noise-skip-layers",
             "(high noise) layers to skip for SLG steps (default: [7,8,9])",
             on_high_noise_skip_layers_arg},
            {"-r",
             "--ref-image",
             "reference image for Flux Kontext models (can be used multiple times)",
             on_ref_image_arg},
            {"",
             "--easycache",
             "enable EasyCache for DiT models with optional \"threshold,start_percent,end_percent\" (default: 0.2,0.15,0.95)",
             on_easycache_arg},

        };

        return options;
    }

    bool process_and_check(SDMode mode) {
        if (width <= 0) {
            fprintf(stderr, "error: the width must be greater than 0\n");
            return false;
        }

        if (height <= 0) {
            fprintf(stderr, "error: the height must be greater than 0\n");
            return false;
        }

        if (sample_params.sample_steps <= 0) {
            fprintf(stderr, "error: the sample_steps must be greater than 0\n");
            return false;
        }

        if (high_noise_sample_params.sample_steps <= 0) {
            high_noise_sample_params.sample_steps = -1;
        }

        if (strength < 0.f || strength > 1.f) {
            fprintf(stderr, "error: can only work with strength in [0.0, 1.0]\n");
            return false;
        }

        if (!easycache_option.empty()) {
            float values[3] = {0.0f, 0.0f, 0.0f};
            std::stringstream ss(easycache_option);
            std::string token;
            int idx = 0;
            while (std::getline(ss, token, ',')) {
                auto trim = [](std::string& s) {
                    const char* whitespace = " \t\r\n";
                    auto start             = s.find_first_not_of(whitespace);
                    if (start == std::string::npos) {
                        s.clear();
                        return;
                    }
                    auto end = s.find_last_not_of(whitespace);
                    s        = s.substr(start, end - start + 1);
                };
                trim(token);
                if (token.empty()) {
                    fprintf(stderr, "error: invalid easycache option '%s'\n", easycache_option.c_str());
                    return false;
                }
                if (idx >= 3) {
                    fprintf(stderr, "error: easycache expects exactly 3 comma-separated values (threshold,start,end)\n");
                    return false;
                }
                try {
                    values[idx] = std::stof(token);
                } catch (const std::exception&) {
                    fprintf(stderr, "error: invalid easycache value '%s'\n", token.c_str());
                    return false;
                }
                idx++;
            }
            if (idx != 3) {
                fprintf(stderr, "error: easycache expects exactly 3 comma-separated values (threshold,start,end)\n");
                return false;
            }
            if (values[0] < 0.0f) {
                fprintf(stderr, "error: easycache threshold must be non-negative\n");
                return false;
            }
            if (values[1] < 0.0f || values[1] >= 1.0f || values[2] <= 0.0f || values[2] > 1.0f || values[1] >= values[2]) {
                fprintf(stderr, "error: easycache start/end percents must satisfy 0.0 <= start < end <= 1.0\n");
                return false;
            }
            easycache_params.enabled         = true;
            easycache_params.reuse_threshold = values[0];
            easycache_params.start_percent   = values[1];
            easycache_params.end_percent     = values[2];
        } else {
            easycache_params.enabled = false;
        }

        sample_params.guidance.slg.layers                 = skip_layers.data();
        sample_params.guidance.slg.layer_count            = skip_layers.size();
        high_noise_sample_params.guidance.slg.layers      = high_noise_skip_layers.data();
        high_noise_sample_params.guidance.slg.layer_count = high_noise_skip_layers.size();

        if (mode == VID_GEN && video_frames <= 0) {
            return false;
        }

        if (mode == VID_GEN && fps <= 0) {
            return false;
        }

        if (sample_params.shifted_timestep < 0 || sample_params.shifted_timestep > 1000) {
            return false;
        }

        if (upscale_repeats < 1) {
            return false;
        }

        if (mode == UPSCALE) {
            if (init_image_path.length() == 0) {
                fprintf(stderr, "error: upscale mode needs an init image (--init-img)\n");
                return false;
            }
        }

        if (seed < 0) {
            srand((int)time(nullptr));
            seed = rand();
        }

        return true;
    }

    std::string to_string() const {
        char* sample_params_str            = sd_sample_params_to_str(&sample_params);
        char* high_noise_sample_params_str = sd_sample_params_to_str(&high_noise_sample_params);
        std::ostringstream oss;
        oss << "SDGenerationParams {\n"
            << "  prompt: \"" << prompt << "\",\n"
            << "  negative_prompt: \"" << negative_prompt << "\",\n"
            << "  clip_skip: " << clip_skip << ",\n"
            << "  width: " << width << ",\n"
            << "  height: " << height << ",\n"
            << "  batch_count: " << batch_count << ",\n"
            << "  init_image_path: \"" << init_image_path << "\",\n"
            << "  end_image_path: \"" << end_image_path << "\",\n"
            << "  mask_image_path: \"" << mask_image_path << "\",\n"
            << "  control_image_path: \"" << control_image_path << "\",\n"
            << "  ref_image_paths: " << vec_str_to_string(ref_image_paths) << ",\n"
            << "  control_video_path: \"" << control_video_path << "\",\n"
            << "  auto_resize_ref_image: " << (auto_resize_ref_image ? "true" : "false") << ",\n"
            << "  increase_ref_index: " << (increase_ref_index ? "true" : "false") << ",\n"
            << "  pm_id_images_dir: \"" << pm_id_images_dir << "\",\n"
            << "  pm_id_embed_path: \"" << pm_id_embed_path << "\",\n"
            << "  pm_style_strength: " << pm_style_strength << ",\n"
            << "  skip_layers: " << vec_to_string(skip_layers) << ",\n"
            << "  sample_params: " << sample_params_str << ",\n"
            << "  high_noise_skip_layers: " << vec_to_string(high_noise_skip_layers) << ",\n"
            << "  high_noise_sample_params: " << high_noise_sample_params_str << ",\n"
            << "  easycache_option: \"" << easycache_option << "\",\n"
            << "  easycache: "
            << (easycache_params.enabled ? "enabled" : "disabled")
            << " (threshold=" << easycache_params.reuse_threshold
            << ", start=" << easycache_params.start_percent
            << ", end=" << easycache_params.end_percent << "),\n"
            << "  moe_boundary: " << moe_boundary << ",\n"
            << "  video_frames: " << video_frames << ",\n"
            << "  fps: " << fps << ",\n"
            << "  vace_strength: " << vace_strength << ",\n"
            << "  strength: " << strength << ",\n"
            << "  control_strength: " << control_strength << ",\n"
            << "  seed: " << seed << ",\n"
            << "  upscale_repeats: " << upscale_repeats << ",\n"
            << "}";
        free(sample_params_str);
        free(high_noise_sample_params_str);
        return oss.str();
    }
};

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "CLI Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nGeneration Options:\n";
    options_list[2].print();
}

void parse_args(int argc, const char** argv, SDCliParams& cli_params, SDContextParams& ctx_params, SDGenerationParams& gen_params) {
    std::vector<ArgOptions> options_vec = {cli_params.get_options(), ctx_params.get_options(), gen_params.get_options()};

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        exit(cli_params.normal_exit ? 0 : 1);
    }

    if (!cli_params.process_and_check() || !ctx_params.process_and_check(cli_params.mode) || !gen_params.process_and_check(cli_params.mode)) {
        print_usage(argc, argv, options_vec);
        exit(1);
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

std::string get_image_params(const SDCliParams& cli_params, const SDContextParams& ctx_params, const SDGenerationParams& gen_params, int64_t seed) {
    std::string parameter_string = gen_params.prompt + "\n";
    if (gen_params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + gen_params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(gen_params.sample_params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(gen_params.sample_params.guidance.txt_cfg) + ", ";
    if (gen_params.sample_params.guidance.slg.scale != 0 && gen_params.skip_layers.size() != 0) {
        parameter_string += "SLG scale: " + std::to_string(gen_params.sample_params.guidance.txt_cfg) + ", ";
        parameter_string += "Skip layers: [";
        for (const auto& layer : gen_params.skip_layers) {
            parameter_string += std::to_string(layer) + ", ";
        }
        parameter_string += "], ";
        parameter_string += "Skip layer start: " + std::to_string(gen_params.sample_params.guidance.slg.layer_start) + ", ";
        parameter_string += "Skip layer end: " + std::to_string(gen_params.sample_params.guidance.slg.layer_end) + ", ";
    }
    parameter_string += "Guidance: " + std::to_string(gen_params.sample_params.guidance.distilled_guidance) + ", ";
    parameter_string += "Eta: " + std::to_string(gen_params.sample_params.eta) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(gen_params.width) + "x" + std::to_string(gen_params.height) + ", ";
    parameter_string += "Model: " + sd_basename(ctx_params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(sd_rng_type_name(ctx_params.rng_type)) + ", ";
    if (ctx_params.sampler_rng_type != RNG_TYPE_COUNT) {
        parameter_string += "Sampler RNG: " + std::string(sd_rng_type_name(ctx_params.sampler_rng_type)) + ", ";
    }
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(gen_params.sample_params.sample_method));
    if (gen_params.sample_params.scheduler != SCHEDULER_COUNT) {
        parameter_string += " " + std::string(sd_scheduler_name(gen_params.sample_params.scheduler));
    }
    parameter_string += ", ";
    for (const auto& te : {ctx_params.clip_l_path, ctx_params.clip_g_path, ctx_params.t5xxl_path, ctx_params.llm_path, ctx_params.llm_vision_path}) {
        if (!te.empty()) {
            parameter_string += "TE: " + sd_basename(te) + ", ";
        }
    }
    if (!ctx_params.diffusion_model_path.empty()) {
        parameter_string += "Unet: " + sd_basename(ctx_params.diffusion_model_path) + ", ";
    }
    if (!ctx_params.vae_path.empty()) {
        parameter_string += "VAE: " + sd_basename(ctx_params.vae_path) + ", ";
    }
    if (gen_params.clip_skip != -1) {
        parameter_string += "Clip skip: " + std::to_string(gen_params.clip_skip) + ", ";
    }
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}

/* Enables Printing the log level tag in color using ANSI escape codes */
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDCliParams* cli_params = (SDCliParams*)data;
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!cli_params->verbose && level <= SD_LOG_DEBUG)) {
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

    if (cli_params->color == true) {
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
    if (image_buffer == nullptr) {
        fprintf(stderr, "load image from '%s' failed\n", image_path);
        return nullptr;
    }
    if (c < expected_channel) {
        fprintf(stderr,
                "the number of channels for the input image must be >= %d,"
                "but got %d channels, image_path = %s\n",
                expected_channel,
                c,
                image_path);
        free(image_buffer);
        return nullptr;
    }
    if (width <= 0) {
        fprintf(stderr, "error: the width of image must be greater than 0, image_path = %s\n", image_path);
        free(image_buffer);
        return nullptr;
    }
    if (height <= 0) {
        fprintf(stderr, "error: the height of image must be greater than 0, image_path = %s\n", image_path);
        free(image_buffer);
        return nullptr;
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
            if (cropped_image_buffer == nullptr) {
                fprintf(stderr, "error: allocate memory for crop\n");
                free(image_buffer);
                return nullptr;
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
        if (resized_image_buffer == nullptr) {
            fprintf(stderr, "error: allocate memory for resize input image\n");
            free(image_buffer);
            return nullptr;
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

    std::vector<fs::directory_entry> entries;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const fs::directory_entry& a, const fs::directory_entry& b) {
                  return a.path().filename().string() < b.path().filename().string();
              });

    for (const auto& entry : entries) {
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
            if (image_buffer == nullptr) {
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

void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data) {
    (void)step;
    (void)is_noisy;
    SDCliParams* cli_params = (SDCliParams*)data;
    // is_noisy is set to true if the preview corresponds to noisy latents, false if it's denoised latents
    // unused in this app, it will either be always noisy or always denoised here
    if (frame_count == 1) {
        stbi_write_png(cli_params->preview_path.c_str(), image->width, image->height, image->channel, image->data, 0);
    } else {
        create_mjpg_avi_from_sd_images(cli_params->preview_path.c_str(), image, frame_count, cli_params->preview_fps);
    }
}

int main(int argc, const char* argv[]) {
    SDCliParams cli_params;
    SDContextParams ctx_params;
    SDGenerationParams gen_params;

    parse_args(argc, argv, cli_params, ctx_params, gen_params);
    if (gen_params.video_frames > 4) {
        size_t last_dot_pos   = cli_params.preview_path.find_last_of(".");
        std::string base_path = cli_params.preview_path;
        std::string file_ext  = "";
        if (last_dot_pos != std::string::npos) {  // filename has extension
            base_path = cli_params.preview_path.substr(0, last_dot_pos);
            file_ext  = cli_params.preview_path.substr(last_dot_pos);
            std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);
        }
        if (file_ext == ".png") {
            cli_params.preview_path = base_path + ".avi";
        }
    }
    cli_params.preview_fps = gen_params.fps;
    if (cli_params.preview_method == PREVIEW_PROJ)
        cli_params.preview_fps /= 4;

    sd_set_log_callback(sd_log_cb, (void*)&cli_params);
    sd_set_preview_callback(step_callback,
                            cli_params.preview_method,
                            cli_params.preview_interval,
                            !cli_params.preview_noisy,
                            cli_params.preview_noisy,
                            (void*)&cli_params);

    if (cli_params.verbose) {
        printf("%s", sd_get_system_info());
        printf("%s\n", cli_params.to_string().c_str());
        printf("%s\n", ctx_params.to_string().c_str());
        printf("%s\n", gen_params.to_string().c_str());
    }

    if (cli_params.mode == CONVERT) {
        bool success = convert(ctx_params.model_path.c_str(),
                               ctx_params.vae_path.c_str(),
                               cli_params.output_path.c_str(),
                               ctx_params.wtype,
                               ctx_params.tensor_type_rules.c_str());
        if (!success) {
            fprintf(stderr,
                    "convert '%s'/'%s' to '%s' failed\n",
                    ctx_params.model_path.c_str(),
                    ctx_params.vae_path.c_str(),
                    cli_params.output_path.c_str());
            return 1;
        } else {
            printf("convert '%s'/'%s' to '%s' success\n",
                   ctx_params.model_path.c_str(),
                   ctx_params.vae_path.c_str(),
                   cli_params.output_path.c_str());
            return 0;
        }
    }

    bool vae_decode_only     = true;
    sd_image_t init_image    = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
    sd_image_t end_image     = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
    sd_image_t control_image = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
    sd_image_t mask_image    = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 1, nullptr};
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
            image.data = nullptr;
        }
        ref_images.clear();
        for (auto image : pmid_images) {
            free(image.data);
            image.data = nullptr;
        }
        pmid_images.clear();
        for (auto image : control_frames) {
            free(image.data);
            image.data = nullptr;
        }
        control_frames.clear();
    };

    if (gen_params.init_image_path.size() > 0) {
        vae_decode_only = false;

        int width       = 0;
        int height      = 0;
        init_image.data = load_image(gen_params.init_image_path.c_str(), width, height, gen_params.width, gen_params.height);
        if (init_image.data == nullptr) {
            fprintf(stderr, "load image from '%s' failed\n", gen_params.init_image_path.c_str());
            release_all_resources();
            return 1;
        }
    }

    if (gen_params.end_image_path.size() > 0) {
        vae_decode_only = false;

        int width      = 0;
        int height     = 0;
        end_image.data = load_image(gen_params.end_image_path.c_str(), width, height, gen_params.width, gen_params.height);
        if (end_image.data == nullptr) {
            fprintf(stderr, "load image from '%s' failed\n", gen_params.end_image_path.c_str());
            release_all_resources();
            return 1;
        }
    }

    if (gen_params.mask_image_path.size() > 0) {
        int c           = 0;
        int width       = 0;
        int height      = 0;
        mask_image.data = load_image(gen_params.mask_image_path.c_str(), width, height, gen_params.width, gen_params.height, 1);
        if (mask_image.data == nullptr) {
            fprintf(stderr, "load image from '%s' failed\n", gen_params.mask_image_path.c_str());
            release_all_resources();
            return 1;
        }
    } else {
        mask_image.data = (uint8_t*)malloc(gen_params.width * gen_params.height);
        memset(mask_image.data, 255, gen_params.width * gen_params.height);
        if (mask_image.data == nullptr) {
            fprintf(stderr, "malloc mask image failed\n");
            release_all_resources();
            return 1;
        }
    }

    if (gen_params.control_image_path.size() > 0) {
        int width          = 0;
        int height         = 0;
        control_image.data = load_image(gen_params.control_image_path.c_str(), width, height, gen_params.width, gen_params.height);
        if (control_image.data == nullptr) {
            fprintf(stderr, "load image from '%s' failed\n", gen_params.control_image_path.c_str());
            release_all_resources();
            return 1;
        }
        if (cli_params.canny_preprocess) {  // apply preprocessor
            preprocess_canny(control_image,
                             0.08f,
                             0.08f,
                             0.8f,
                             1.0f,
                             false);
        }
    }

    if (gen_params.ref_image_paths.size() > 0) {
        vae_decode_only = false;
        for (auto& path : gen_params.ref_image_paths) {
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = load_image(path.c_str(), width, height);
            if (image_buffer == nullptr) {
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

    if (!gen_params.control_video_path.empty()) {
        if (!load_images_from_dir(gen_params.control_video_path,
                                  control_frames,
                                  gen_params.width,
                                  gen_params.height,
                                  gen_params.video_frames,
                                  cli_params.verbose)) {
            release_all_resources();
            return 1;
        }
    }

    if (!gen_params.pm_id_images_dir.empty()) {
        if (!load_images_from_dir(gen_params.pm_id_images_dir,
                                  pmid_images,
                                  0,
                                  0,
                                  0,
                                  cli_params.verbose)) {
            release_all_resources();
            return 1;
        }
    }

    if (cli_params.mode == VID_GEN) {
        vae_decode_only = false;
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(vae_decode_only, true, cli_params.taesd_preview);

    sd_image_t* results = nullptr;
    int num_results     = 0;

    if (cli_params.mode == UPSCALE) {
        num_results = 1;
        results     = (sd_image_t*)calloc(num_results, sizeof(sd_image_t));
        if (results == nullptr) {
            printf("failed to allocate results array\n");
            release_all_resources();
            return 1;
        }

        results[0]      = init_image;
        init_image.data = nullptr;
    } else {
        sd_ctx_t* sd_ctx = new_sd_ctx(&sd_ctx_params);

        if (sd_ctx == nullptr) {
            printf("new_sd_ctx_t failed\n");
            release_all_resources();
            return 1;
        }

        if (gen_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
        }

        if (gen_params.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
        }

        if (gen_params.sample_params.scheduler == SCHEDULER_COUNT) {
            gen_params.sample_params.scheduler = sd_get_default_scheduler(sd_ctx);
        }

        if (cli_params.mode == IMG_GEN) {
            sd_img_gen_params_t img_gen_params = {
                gen_params.prompt.c_str(),
                gen_params.negative_prompt.c_str(),
                gen_params.clip_skip,
                init_image,
                ref_images.data(),
                (int)ref_images.size(),
                gen_params.auto_resize_ref_image,
                gen_params.increase_ref_index,
                mask_image,
                gen_params.width,
                gen_params.height,
                gen_params.sample_params,
                gen_params.strength,
                gen_params.seed,
                gen_params.batch_count,
                control_image,
                gen_params.control_strength,
                {
                    pmid_images.data(),
                    (int)pmid_images.size(),
                    gen_params.pm_id_embed_path.c_str(),
                    gen_params.pm_style_strength,
                },  // pm_params
                ctx_params.vae_tiling_params,
                gen_params.easycache_params,
            };

            results     = generate_image(sd_ctx, &img_gen_params);
            num_results = gen_params.batch_count;
        } else if (cli_params.mode == VID_GEN) {
            sd_vid_gen_params_t vid_gen_params = {
                gen_params.prompt.c_str(),
                gen_params.negative_prompt.c_str(),
                gen_params.clip_skip,
                init_image,
                end_image,
                control_frames.data(),
                (int)control_frames.size(),
                gen_params.width,
                gen_params.height,
                gen_params.sample_params,
                gen_params.high_noise_sample_params,
                gen_params.moe_boundary,
                gen_params.strength,
                gen_params.seed,
                gen_params.video_frames,
                gen_params.vace_strength,
                gen_params.easycache_params,
            };

            results = generate_video(sd_ctx, &vid_gen_params, &num_results);
        }

        if (results == nullptr) {
            printf("generate failed\n");
            free_sd_ctx(sd_ctx);
            return 1;
        }

        free_sd_ctx(sd_ctx);
    }

    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (ctx_params.esrgan_path.size() > 0 && gen_params.upscale_repeats > 0) {
        upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(ctx_params.esrgan_path.c_str(),
                                                        ctx_params.offload_params_to_cpu,
                                                        ctx_params.diffusion_conv_direct,
                                                        ctx_params.n_threads);

        if (upscaler_ctx == nullptr) {
            printf("new_upscaler_ctx failed\n");
        } else {
            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                sd_image_t current_image = results[i];
                for (int u = 0; u < gen_params.upscale_repeats; ++u) {
                    sd_image_t upscaled_image = upscale(upscaler_ctx, current_image, upscale_factor);
                    if (upscaled_image.data == nullptr) {
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
        const fs::path out_path = cli_params.output_path;
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
    size_t last_dot_pos   = cli_params.output_path.find_last_of(".");
    size_t last_slash_pos = std::min(cli_params.output_path.find_last_of("/"),
                                     cli_params.output_path.find_last_of("\\"));
    if (last_dot_pos != std::string::npos && (last_slash_pos == std::string::npos || last_dot_pos > last_slash_pos)) {  // filename has extension
        base_path = cli_params.output_path.substr(0, last_dot_pos);
        file_ext = file_ext_lower = cli_params.output_path.substr(last_dot_pos);
        std::transform(file_ext.begin(), file_ext.end(), file_ext_lower.begin(), ::tolower);
        is_jpg = (file_ext_lower == ".jpg" || file_ext_lower == ".jpeg" || file_ext_lower == ".jpe");
    } else {
        base_path = cli_params.output_path;
        file_ext = file_ext_lower = "";
        is_jpg                    = false;
    }

    if (cli_params.mode == VID_GEN && num_results > 1) {
        std::string vid_output_path = cli_params.output_path;
        if (file_ext_lower == ".png") {
            vid_output_path = base_path + ".avi";
        }
        create_mjpg_avi_from_sd_images(vid_output_path.c_str(), results, num_results, gen_params.fps);
        printf("save result MJPG AVI video to '%s'\n", vid_output_path.c_str());
    } else {
        // appending ".png" to absent or unknown extension
        if (!is_jpg && file_ext_lower != ".png") {
            base_path += file_ext;
            file_ext = ".png";
        }
        for (int i = 0; i < num_results; i++) {
            if (results[i].data == nullptr) {
                continue;
            }
            int write_ok;
            std::string final_image_path = i > 0 ? base_path + "_" + std::to_string(i + 1) + file_ext : base_path + file_ext;
            if (is_jpg) {
                write_ok = stbi_write_jpg(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                                          results[i].data, 90, get_image_params(cli_params, ctx_params, gen_params, gen_params.seed + i).c_str());
                printf("save result JPEG image to '%s' (%s)\n", final_image_path.c_str(), write_ok == 0 ? "failure" : "success");
            } else {
                write_ok = stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                                          results[i].data, 0, get_image_params(cli_params, ctx_params, gen_params, gen_params.seed + i).c_str());
                printf("save result PNG image to '%s' (%s)\n", final_image_path.c_str(), write_ok == 0 ? "failure" : "success");
            }
        }
    }

    for (int i = 0; i < num_results; i++) {
        free(results[i].data);
        results[i].data = nullptr;
    }
    free(results);

    release_all_resources();

    return 0;
}
