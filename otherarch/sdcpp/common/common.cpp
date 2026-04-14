#include "common.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <json.hpp>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif  // _WIN32

#include "log.h"
#include "resource_owners.hpp"

using json   = nlohmann::json;
namespace fs = std::filesystem;

const char* const modes_str[] = {
    "img_gen",
    "vid_gen",
    "convert",
    "upscale",
    "metadata",
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
    (void)argv;
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

static bool is_absolute_path(const std::string& p) {
#ifdef _WIN32
    return p.size() > 1 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':';
#else
    return !p.empty() && p[0] == '/';
#endif
}

std::string ArgOptions::wrap_text(const std::string& text, size_t width, size_t indent) {
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

void ArgOptions::print() const {
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
        std::string wrapped_desc = wrap_text(e.desc, desc_width, indent);
        std::cout << "  " << std::left << std::setw(static_cast<int>(max_name_width) + 4)
                  << e.names << wrapped_desc << "\n";
    }
}

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
            LOG_ERROR("error: invalid parameter for argument: %s", arg.c_str());
            return false;
        }
        if (!found_arg) {
            LOG_ERROR("error: unknown argument: %s", arg.c_str());
            return false;
        }
    }

    return true;
}

ArgOptions SDContextParams::get_options() {
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
         "--tae",
         "alias of --taesd",
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

    options.float_options = {};

    options.bool_options = {
        {"",
         "--force-sdxl-vae-conv-scale",
         "force use of conv scale on sdxl vae",
         true, &force_sdxl_vae_conv_scale},
        {"",
         "--offload-to-cpu",
         "place the weights in RAM to save VRAM, and automatically load them into VRAM when needed",
         true, &offload_params_to_cpu},
        {"",
         "--mmap",
         "whether to memory-map model",
         true, &enable_mmap},
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
         "--fa",
         "use flash attention",
         true, &flash_attn},
        {"",
         "--diffusion-fa",
         "use flash attention in the diffusion model only",
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
         "--circular",
         "enable circular padding for convolutions",
         true, &circular},
        {"",
         "--circularx",
         "enable circular RoPE wrapping on x-axis (width) only",
         true, &circular_x},
        {"",
         "--circulary",
         "enable circular RoPE wrapping on y-axis (height) only",
         true, &circular_y},
        {"",
         "--chroma-disable-dit-mask",
         "disable dit mask for chroma",
         false, &chroma_use_dit_mask},
        {"",
         "--qwen-image-zero-cond-t",
         "enable zero_cond_t for qwen image",
         true, &qwen_image_zero_cond_t},
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
            LOG_ERROR("error: invalid weight format %s",
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
            LOG_ERROR("error: invalid rng type %s",
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
            LOG_ERROR("error: invalid sampler rng type %s",
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
            LOG_ERROR("error: invalid prediction type %s",
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
            LOG_ERROR("error: invalid lora apply model %s",
                      arg);
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
    };

    return options;
}

void SDContextParams::build_embedding_map() {
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt"};

    if (!fs::exists(embedding_dir) || !fs::is_directory(embedding_dir)) {
        return;
    }

    for (auto& p : fs::directory_iterator(embedding_dir)) {
        if (!p.is_regular_file())
            continue;

        auto path       = p.path();
        std::string ext = path.extension().string();

        bool valid = false;
        for (auto& e : valid_ext) {
            if (ext == e) {
                valid = true;
                break;
            }
        }
        if (!valid)
            continue;

        std::string key   = path.stem().string();
        std::string value = path.string();

        embedding_map[key] = value;
    }
}

bool SDContextParams::process_and_check(SDMode mode) {
    if (mode != UPSCALE && mode != METADATA && model_path.length() == 0 && diffusion_model_path.length() == 0) {
        LOG_ERROR("error: the following arguments are required: model_path/diffusion_model\n");
        return false;
    }

    if (mode == UPSCALE) {
        if (esrgan_path.length() == 0) {
            LOG_ERROR("error: upscale mode needs an upscaler model (--upscale-model)\n");
            return false;
        }
    }

    if (n_threads <= 0) {
        n_threads = sd_get_num_physical_cores();
    }

    build_embedding_map();

    return true;
}

std::string SDContextParams::to_string() const {
    std::ostringstream emb_ss;
    emb_ss << "{\n";
    for (auto it = embedding_map.begin(); it != embedding_map.end(); ++it) {
        emb_ss << "    \"" << it->first << "\": \"" << it->second << "\"";
        if (std::next(it) != embedding_map.end()) {
            emb_ss << ",";
        }
        emb_ss << "\n";
    }
    emb_ss << "  }";

    std::string embeddings_str = emb_ss.str();
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
        << "  embeddings: " << embeddings_str << "\n"
        << "  wtype: " << sd_type_name(wtype) << ",\n"
        << "  tensor_type_rules: \"" << tensor_type_rules << "\",\n"
        << "  lora_model_dir: \"" << lora_model_dir << "\",\n"
        << "  photo_maker_path: \"" << photo_maker_path << "\",\n"
        << "  rng_type: " << sd_rng_type_name(rng_type) << ",\n"
        << "  sampler_rng_type: " << sd_rng_type_name(sampler_rng_type) << ",\n"
        << "  offload_params_to_cpu: " << (offload_params_to_cpu ? "true" : "false") << ",\n"
        << "  enable_mmap: " << (enable_mmap ? "true" : "false") << ",\n"
        << "  control_net_cpu: " << (control_net_cpu ? "true" : "false") << ",\n"
        << "  clip_on_cpu: " << (clip_on_cpu ? "true" : "false") << ",\n"
        << "  vae_on_cpu: " << (vae_on_cpu ? "true" : "false") << ",\n"
        << "  flash_attn: " << (flash_attn ? "true" : "false") << ",\n"
        << "  diffusion_flash_attn: " << (diffusion_flash_attn ? "true" : "false") << ",\n"
        << "  diffusion_conv_direct: " << (diffusion_conv_direct ? "true" : "false") << ",\n"
        << "  vae_conv_direct: " << (vae_conv_direct ? "true" : "false") << ",\n"
        << "  circular: " << (circular ? "true" : "false") << ",\n"
        << "  circular_x: " << (circular_x ? "true" : "false") << ",\n"
        << "  circular_y: " << (circular_y ? "true" : "false") << ",\n"
        << "  chroma_use_dit_mask: " << (chroma_use_dit_mask ? "true" : "false") << ",\n"
        << "  qwen_image_zero_cond_t: " << (qwen_image_zero_cond_t ? "true" : "false") << ",\n"
        << "  chroma_use_t5_mask: " << (chroma_use_t5_mask ? "true" : "false") << ",\n"
        << "  chroma_t5_mask_pad: " << chroma_t5_mask_pad << ",\n"
        << "  prediction: " << sd_prediction_name(prediction) << ",\n"
        << "  lora_apply_mode: " << sd_lora_apply_mode_name(lora_apply_mode) << ",\n"
        << "  force_sdxl_vae_conv_scale: " << (force_sdxl_vae_conv_scale ? "true" : "false") << "\n"
        << "}";
    return oss.str();
}

sd_ctx_params_t SDContextParams::to_sd_ctx_params_t(bool vae_decode_only, bool free_params_immediately, bool taesd_preview) {
    embedding_vec.clear();
    embedding_vec.reserve(embedding_map.size());
    for (const auto& kv : embedding_map) {
        sd_embedding_t item;
        item.name = kv.first.c_str();
        item.path = kv.second.c_str();
        embedding_vec.emplace_back(item);
    }

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
        embedding_vec.data(),
        static_cast<uint32_t>(embedding_vec.size()),
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
        enable_mmap,
        clip_on_cpu,
        control_net_cpu,
        vae_on_cpu,
        flash_attn,
        diffusion_flash_attn,
        taesd_preview,
        diffusion_conv_direct,
        vae_conv_direct,
        circular || circular_x,
        circular || circular_y,
        force_sdxl_vae_conv_scale,
        chroma_use_dit_mask,
        chroma_use_t5_mask,
        chroma_t5_mask_pad,
        qwen_image_zero_cond_t,
    };
    return sd_ctx_params;
}

SDGenerationParams::SDGenerationParams() {
    sd_sample_params_init(&sample_params);
    sd_sample_params_init(&high_noise_sample_params);
}

ArgOptions SDGenerationParams::get_options() {
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
        {"",
         "--upscale-tile-size",
         "tile size for ESRGAN upscaling (default: 128)",
         &upscale_tile_size},
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
         "noise multiplier (default: 0 for ddim_trailing, tcd, res_multistep and res_2s; 1 for euler_a and dpm++2s_a)",
         &sample_params.eta},
        {"",
         "--flow-shift",
         "shift value for Flow models like SD3.x or WAN (default: auto)",
         &sample_params.flow_shift},
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
         "(high noise) noise multiplier (default: 0 for ddim_trailing, tcd, res_multistep and res_2s; 1 for euler_a and dpm++2s_a)",
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
        {"",
         "--vae-tile-overlap",
         "tile overlap for vae tiling, in fraction of tile size (default: 0.5)",
         &vae_tiling_params.target_overlap},
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
        {"",
         "--disable-image-metadata",
         "do not embed generation metadata on image files",
         false,
         &embed_image_metadata},
        {"",
         "--vae-tiling",
         "process vae in tiles to reduce memory usage",
         true,
         &vae_tiling_params.enabled},
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
            LOG_ERROR("error: invalid sample method %s",
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
            LOG_ERROR("error: invalid high noise sample method %s",
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
            LOG_ERROR("error: invalid scheduler %s",
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

    auto on_sigmas_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        std::string sigmas_str = argv[index];
        if (!sigmas_str.empty() && sigmas_str.front() == '[') {
            sigmas_str.erase(0, 1);
        }
        if (!sigmas_str.empty() && sigmas_str.back() == ']') {
            sigmas_str.pop_back();
        }

        std::stringstream ss(sigmas_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
            item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!item.empty()) {
                try {
                    custom_sigmas.push_back(std::stof(item));
                } catch (const std::invalid_argument&) {
                    LOG_ERROR("error: invalid float value '%s' in --sigmas", item.c_str());
                    return -1;
                } catch (const std::out_of_range&) {
                    LOG_ERROR("error: float value '%s' out of range in --sigmas", item.c_str());
                    return -1;
                }
            }
        }

        if (custom_sigmas.empty() && !sigmas_str.empty()) {
            LOG_ERROR("error: could not parse any sigma values from '%s'", argv[index]);
            return -1;
        }
        return 1;
    };

    auto on_ref_image_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        ref_image_paths.push_back(argv[index]);
        return 1;
    };

    auto on_cache_mode_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        cache_mode = argv_to_utf8(index, argv);
        if (cache_mode != "easycache" && cache_mode != "ucache" &&
            cache_mode != "dbcache" && cache_mode != "taylorseer" && cache_mode != "cache-dit" && cache_mode != "spectrum") {
            fprintf(stderr, "error: invalid cache mode '%s', must be 'easycache', 'ucache', 'dbcache', 'taylorseer', 'cache-dit', or 'spectrum'\n", cache_mode.c_str());
            return -1;
        }
        return 1;
    };

    auto on_cache_option_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        cache_option = argv_to_utf8(index, argv);
        return 1;
    };

    auto on_scm_mask_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        scm_mask = argv_to_utf8(index, argv);
        return 1;
    };

    auto on_scm_policy_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        std::string policy = argv_to_utf8(index, argv);
        if (policy == "dynamic") {
            scm_policy_dynamic = true;
        } else if (policy == "static") {
            scm_policy_dynamic = false;
        } else {
            fprintf(stderr, "error: invalid scm policy '%s', must be 'dynamic' or 'static'\n", policy.c_str());
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
        {"-s",
         "--seed",
         "RNG seed (default: 42, use random seed for < 0)",
         on_seed_arg},
        {"",
         "--sampling-method",
         "sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd, res_multistep, res_2s] "
         "(default: euler for Flux/SD3/Wan, euler_a otherwise)",
         on_sample_method_arg},
        {"",
         "--high-noise-sampling-method",
         "(high noise) sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd, res_multistep, res_2s]"
         " default: euler for Flux/SD3/Wan, euler_a otherwise",
         on_high_noise_sample_method_arg},
        {"",
         "--scheduler",
         "denoiser sigma scheduler, one of [discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple, kl_optimal, lcm, bong_tangent], default: discrete",
         on_scheduler_arg},
        {"",
         "--sigmas",
         "custom sigma values for the sampler, comma-separated (e.g., \"14.61,7.8,3.5,0.0\").",
         on_sigmas_arg},
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
         "--cache-mode",
         "caching method: 'easycache' (DiT), 'ucache' (UNET), 'dbcache'/'taylorseer'/'cache-dit' (DiT block-level), 'spectrum' (UNET/DiT Chebyshev+Taylor forecasting)",
         on_cache_mode_arg},
        {"",
         "--cache-option",
         "named cache params (key=value format, comma-separated). easycache/ucache: threshold=,start=,end=,decay=,relative=,reset=; dbcache/taylorseer/cache-dit: Fn=,Bn=,threshold=,warmup=; spectrum: w=,m=,lam=,window=,flex=,warmup=,stop=. Examples: \"threshold=0.25\" or \"threshold=1.5,reset=0\"",
         on_cache_option_arg},
        {"",
         "--scm-mask",
         "SCM steps mask for cache-dit: comma-separated 0/1 (e.g., \"1,1,1,0,0,1,0,0,1,0\") - 1=compute, 0=can cache",
         on_scm_mask_arg},
        {"",
         "--scm-policy",
         "SCM policy: 'dynamic' (default) or 'static'",
         on_scm_policy_arg},
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

bool SDGenerationParams::from_json_str(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (...) {
        LOG_ERROR("json parse failed %s", json_str.c_str());
        return false;
    }

    auto load_if_exists = [&](const char* key, auto& out) {
        if (j.contains(key)) {
            using T = std::decay_t<decltype(out)>;
            if constexpr (std::is_same_v<T, std::string>) {
                if (j[key].is_string())
                    out = j[key];
            } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
                if (j[key].is_number_integer())
                    out = j[key];
            } else if constexpr (std::is_same_v<T, float>) {
                if (j[key].is_number())
                    out = j[key];
            } else if constexpr (std::is_same_v<T, bool>) {
                if (j[key].is_boolean())
                    out = j[key];
            } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                if (j[key].is_array())
                    out = j[key].get<std::vector<int>>();
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                if (j[key].is_array())
                    out = j[key].get<std::vector<std::string>>();
            }
        }
    };

    load_if_exists("prompt", prompt);
    load_if_exists("negative_prompt", negative_prompt);
    load_if_exists("cache_mode", cache_mode);
    load_if_exists("cache_option", cache_option);
    load_if_exists("scm_mask", scm_mask);

    load_if_exists("clip_skip", clip_skip);
    load_if_exists("width", width);
    load_if_exists("height", height);
    load_if_exists("batch_count", batch_count);
    load_if_exists("video_frames", video_frames);
    load_if_exists("fps", fps);
    load_if_exists("upscale_repeats", upscale_repeats);
    load_if_exists("seed", seed);

    load_if_exists("strength", strength);
    load_if_exists("control_strength", control_strength);
    load_if_exists("pm_style_strength", pm_style_strength);
    load_if_exists("moe_boundary", moe_boundary);
    load_if_exists("vace_strength", vace_strength);

    load_if_exists("auto_resize_ref_image", auto_resize_ref_image);
    load_if_exists("increase_ref_index", increase_ref_index);
    load_if_exists("embed_image_metadata", embed_image_metadata);

    load_if_exists("skip_layers", skip_layers);
    load_if_exists("high_noise_skip_layers", high_noise_skip_layers);

    load_if_exists("steps", sample_params.sample_steps);
    load_if_exists("high_noise_steps", high_noise_sample_params.sample_steps);
    load_if_exists("cfg_scale", sample_params.guidance.txt_cfg);
    load_if_exists("img_cfg_scale", sample_params.guidance.img_cfg);
    load_if_exists("guidance", sample_params.guidance.distilled_guidance);
    load_if_exists("flow_shift", sample_params.flow_shift);

    auto load_sampler_if_exists = [&](const char* key, enum sample_method_t& out) {
        if (j.contains(key) && j[key].is_string()) {
            enum sample_method_t tmp = str_to_sample_method(j[key].get<std::string>().c_str());
            if (tmp != SAMPLE_METHOD_COUNT) {
                out = tmp;
            }
        }
    };
    load_sampler_if_exists("sample_method", sample_params.sample_method);
    load_sampler_if_exists("high_noise_sample_method", high_noise_sample_params.sample_method);

    if (j.contains("scheduler") && j["scheduler"].is_string()) {
        enum scheduler_t tmp = str_to_scheduler(j["scheduler"].get<std::string>().c_str());
        if (tmp != SCHEDULER_COUNT) {
            sample_params.scheduler = tmp;
        }
    }

    return true;
}

void SDGenerationParams::extract_and_remove_lora(const std::string& lora_model_dir) {
    if (lora_model_dir.empty()) {
        return;
    }
    static const std::regex re(R"(<lora:([^:>]+):([^>]+)>)");
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt"};
    std::smatch m;

    std::string tmp = prompt;

    while (std::regex_search(tmp, m, re)) {
        std::string raw_path      = m[1].str();
        const std::string raw_mul = m[2].str();

        float mul = 0.f;
        try {
            mul = std::stof(raw_mul);
        } catch (...) {
            tmp    = m.suffix().str();
            prompt = std::regex_replace(prompt, re, "", std::regex_constants::format_first_only);
            continue;
        }

        bool is_high_noise              = false;
        static const std::string prefix = "|high_noise|";
        if (raw_path.rfind(prefix, 0) == 0) {
            raw_path.erase(0, prefix.size());
            is_high_noise = true;
        }

        fs::path final_path;
        if (is_absolute_path(raw_path)) {
            final_path = raw_path;
        } else {
            final_path = fs::path(lora_model_dir) / raw_path;
        }
        if (!fs::exists(final_path)) {
            bool found = false;
            for (const auto& ext : valid_ext) {
                fs::path try_path = final_path;
                try_path += ext;
                if (fs::exists(try_path)) {
                    final_path = try_path;
                    found      = true;
                    break;
                }
            }
            if (!found) {
                LOG_WARN("can not found lora %s", final_path.lexically_normal().string().c_str());
                tmp    = m.suffix().str();
                prompt = std::regex_replace(prompt, re, "", std::regex_constants::format_first_only);
                continue;
            }
        }

        const std::string key = final_path.lexically_normal().string();

        if (is_high_noise)
            high_noise_lora_map[key] += mul;
        else
            lora_map[key] += mul;

        prompt = std::regex_replace(prompt, re, "", std::regex_constants::format_first_only);

        tmp = m.suffix().str();
    }

    for (const auto& kv : lora_map) {
        sd_lora_t item;
        item.is_high_noise = false;
        item.path          = kv.first.c_str();
        item.multiplier    = kv.second;
        lora_vec.emplace_back(item);
    }

    for (const auto& kv : high_noise_lora_map) {
        sd_lora_t item;
        item.is_high_noise = true;
        item.path          = kv.first.c_str();
        item.multiplier    = kv.second;
        lora_vec.emplace_back(item);
    }
}

bool SDGenerationParams::width_and_height_are_set() const {
    return width > 0 && height > 0;
}

void SDGenerationParams::set_width_and_height_if_unset(int w, int h) {
    if (!width_and_height_are_set()) {
        LOG_INFO("set width x height to %d x %d", w, h);
        width  = w;
        height = h;
    }
}

int SDGenerationParams::get_resolved_width() const {
    return (width > 0) ? width : 512;
}

int SDGenerationParams::get_resolved_height() const {
    return (height > 0) ? height : 512;
}

bool SDGenerationParams::process_and_check(SDMode mode, const std::string& lora_model_dir) {
    prompt_with_lora = prompt;

    if (sample_params.sample_steps <= 0) {
        LOG_ERROR("error: the sample_steps must be greater than 0\n");
        return false;
    }

    if (high_noise_sample_params.sample_steps <= 0) {
        high_noise_sample_params.sample_steps = -1;
    }

    if (strength < 0.f || strength > 1.f) {
        LOG_ERROR("error: can only work with strength in [0.0, 1.0]\n");
        return false;
    }

    sd_cache_params_init(&cache_params);

    auto parse_named_params = [&](const std::string& opt_str) -> bool {
        std::stringstream ss(opt_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            size_t eq_pos = token.find('=');
            if (eq_pos == std::string::npos) {
                LOG_ERROR("error: cache option '%s' missing '=' separator", token.c_str());
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
                    LOG_ERROR("error: unknown cache parameter '%s'", key.c_str());
                    return false;
                }
            } catch (const std::exception&) {
                LOG_ERROR("error: invalid value '%s' for parameter '%s'", val.c_str(), key.c_str());
                return false;
            }
        }
        return true;
    };

    if (!cache_mode.empty()) {
        if (cache_mode == "easycache") {
            cache_params.mode = SD_CACHE_EASYCACHE;
        } else if (cache_mode == "ucache") {
            cache_params.mode = SD_CACHE_UCACHE;
        } else if (cache_mode == "dbcache") {
            cache_params.mode = SD_CACHE_DBCACHE;
        } else if (cache_mode == "taylorseer") {
            cache_params.mode = SD_CACHE_TAYLORSEER;
        } else if (cache_mode == "cache-dit") {
            cache_params.mode = SD_CACHE_CACHE_DIT;
        } else if (cache_mode == "spectrum") {
            cache_params.mode = SD_CACHE_SPECTRUM;
        }

        if (!cache_option.empty()) {
            if (!parse_named_params(cache_option)) {
                return false;
            }
        }

        if (cache_mode == "easycache" || cache_mode == "ucache") {
            if (cache_params.reuse_threshold < 0.0f) {
                LOG_ERROR("error: cache threshold must be non-negative");
                return false;
            }
            if (cache_params.start_percent < 0.0f || cache_params.start_percent >= 1.0f ||
                cache_params.end_percent <= 0.0f || cache_params.end_percent > 1.0f ||
                cache_params.start_percent >= cache_params.end_percent) {
                LOG_ERROR("error: cache start/end percents must satisfy 0.0 <= start < end <= 1.0");
                return false;
            }
        }
    }

    if (cache_params.mode == SD_CACHE_DBCACHE ||
        cache_params.mode == SD_CACHE_TAYLORSEER ||
        cache_params.mode == SD_CACHE_CACHE_DIT) {
        if (!scm_mask.empty()) {
            cache_params.scm_mask = scm_mask.c_str();
        }
        cache_params.scm_policy_dynamic = scm_policy_dynamic;
    }

    sample_params.guidance.slg.layers                 = skip_layers.data();
    sample_params.guidance.slg.layer_count            = skip_layers.size();
    sample_params.custom_sigmas                       = custom_sigmas.data();
    sample_params.custom_sigmas_count                 = static_cast<int>(custom_sigmas.size());
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

    if (upscale_tile_size < 1) {
        return false;
    }

    if (mode == UPSCALE) {
        if (init_image_path.length() == 0) {
            LOG_ERROR("error: upscale mode needs an init image (--init-img)\n");
            return false;
        }
    }

    if (seed < 0) {
        srand((int)time(nullptr));
        seed = rand();
    }

    extract_and_remove_lora(lora_model_dir);

    return true;
}

std::string SDGenerationParams::to_string() const {
    FreeUniquePtr<char> sample_params_str(sd_sample_params_to_str(&sample_params));
    FreeUniquePtr<char> high_noise_sample_params_str(sd_sample_params_to_str(&high_noise_sample_params));

    std::ostringstream lora_ss;
    lora_ss << "{\n";
    for (auto it = lora_map.begin(); it != lora_map.end(); ++it) {
        lora_ss << "    \"" << it->first << "\": \"" << it->second << "\"";
        if (std::next(it) != lora_map.end()) {
            lora_ss << ",";
        }
        lora_ss << "\n";
    }
    lora_ss << "  }";
    std::string loras_str = lora_ss.str();

    lora_ss = std::ostringstream();
    ;
    lora_ss << "{\n";
    for (auto it = high_noise_lora_map.begin(); it != high_noise_lora_map.end(); ++it) {
        lora_ss << "    \"" << it->first << "\": \"" << it->second << "\"";
        if (std::next(it) != high_noise_lora_map.end()) {
            lora_ss << ",";
        }
        lora_ss << "\n";
    }
    lora_ss << "  }";
    std::string high_noise_loras_str = lora_ss.str();

    std::ostringstream oss;
    oss << "SDGenerationParams {\n"
        << "  loras: \"" << loras_str << "\",\n"
        << "  high_noise_loras: \"" << high_noise_loras_str << "\",\n"
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
        << "  sample_params: " << SAFE_STR(sample_params_str.get()) << ",\n"
        << "  high_noise_skip_layers: " << vec_to_string(high_noise_skip_layers) << ",\n"
        << "  high_noise_sample_params: " << SAFE_STR(high_noise_sample_params_str.get()) << ",\n"
        << "  custom_sigmas: " << vec_to_string(custom_sigmas) << ",\n"
        << "  cache_mode: \"" << cache_mode << "\",\n"
        << "  cache_option: \"" << cache_option << "\",\n"
        << "  cache: "
        << (cache_params.mode != SD_CACHE_DISABLED ? "enabled" : "disabled")
        << " (threshold=" << cache_params.reuse_threshold
        << ", start=" << cache_params.start_percent
        << ", end=" << cache_params.end_percent << "),\n"
        << "  moe_boundary: " << moe_boundary << ",\n"
        << "  video_frames: " << video_frames << ",\n"
        << "  fps: " << fps << ",\n"
        << "  vace_strength: " << vace_strength << ",\n"
        << "  strength: " << strength << ",\n"
        << "  control_strength: " << control_strength << ",\n"
        << "  seed: " << seed << ",\n"
        << "  upscale_repeats: " << upscale_repeats << ",\n"
        << "  upscale_tile_size: " << upscale_tile_size << ",\n"
        << "  vae_tiling_params: { "
        << vae_tiling_params.enabled << ", "
        << vae_tiling_params.tile_size_x << ", "
        << vae_tiling_params.tile_size_y << ", "
        << vae_tiling_params.target_overlap << ", "
        << vae_tiling_params.rel_size_x << ", "
        << vae_tiling_params.rel_size_y << " },\n"
        << "}";
    return oss.str();
}

std::string version_string() {
    return std::string("stable-diffusion.cpp version ") + sd_version() + ", commit " + sd_commit();
}

std::string get_image_params(const SDContextParams& ctx_params, const SDGenerationParams& gen_params, int64_t seed) {
    std::string parameter_string;
    if (gen_params.prompt_with_lora.size() != 0) {
        parameter_string += gen_params.prompt_with_lora + "\n";
    } else {
        parameter_string += gen_params.prompt + "\n";
    }
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
    parameter_string += "Size: " + std::to_string(gen_params.get_resolved_width()) + "x" + std::to_string(gen_params.get_resolved_height()) + ", ";
    parameter_string += "Model: " + sd_basename(ctx_params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(sd_rng_type_name(ctx_params.rng_type)) + ", ";
    if (ctx_params.sampler_rng_type != RNG_TYPE_COUNT) {
        parameter_string += "Sampler RNG: " + std::string(sd_rng_type_name(ctx_params.sampler_rng_type)) + ", ";
    }
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(gen_params.sample_params.sample_method));
    if (!gen_params.custom_sigmas.empty()) {
        parameter_string += ", Custom Sigmas: [";
        for (size_t i = 0; i < gen_params.custom_sigmas.size(); ++i) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << gen_params.custom_sigmas[i];
            parameter_string += oss.str() + (i == gen_params.custom_sigmas.size() - 1 ? "" : ", ");
        }
        parameter_string += "]";
    } else if (gen_params.sample_params.scheduler != SCHEDULER_COUNT) {  // Only show schedule if not using custom sigmas
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
