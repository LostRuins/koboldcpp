#include "common.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
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
#include "media_io.h"
#include "resource_owners.hpp"

using json   = nlohmann::json;
namespace fs = std::filesystem;

const char* const modes_str[] = {
    "img_gen",
    "adetailer",
    "vid_gen",
    "convert",
    "upscale",
    "metadata",
    "vid_upscale",
};

static void prepend_backend_assignment(std::string& spec, const char* assignment) {
    spec = spec.empty() ? assignment : std::string(assignment) + "," + spec;
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

std::string argv_to_utf8(int index, const char** argv) {
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
std::string argv_to_utf8(int index, const char** argv) {
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
    size_t pos      = 0;
    size_t line_len = 0;

    while (pos < text.size()) {
        if (text[pos] == '\n') {
            oss << '\n'
                << std::string(indent, ' ');
            line_len = 0;
            ++pos;
            continue;
        }

        if (std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
            continue;
        }

        size_t word_start = pos;
        while (pos < text.size() &&
               text[pos] != '\n' &&
               !std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }

        std::string word = text.substr(word_start, pos - word_start);
        while (!word.empty()) {
            size_t separator_len = line_len == 0 ? 0 : 1;
            if (line_len + separator_len + word.size() <= width) {
                if (separator_len > 0) {
                    oss << ' ';
                    ++line_len;
                }
                oss << word;
                line_len += word.size();
                word.clear();
                continue;
            }

            if (line_len > 0) {
                oss << '\n'
                    << std::string(indent, ' ');
                line_len = 0;
                continue;
            }

            size_t chunk_len = std::min(width, word.size());
            oss << word.substr(0, chunk_len);
            line_len = chunk_len;
            word.erase(0, chunk_len);
            if (!word.empty()) {
                oss << '\n'
                    << std::string(indent, ' ');
                line_len = 0;
            }
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

void add_log_options(ArgOptions& options, sd_log_level_t& level) {
    options.manual_options.push_back({"", "--log-level",
                                      "minimum log level, one of [debug, verbose, info, warn, error] (default: info)",
                                      [&level](int argc, const char** argv, int index) {
                                          if (++index >= argc) {
                                              return -1;
                                          }
                                          if (!parse_log_level(argv[index], level)) {
                                              LOG_ERROR("invalid log level %s, must be one of [debug, verbose, info, warn, error]", argv[index]);
                                              return -1;
                                          }
                                          return 1;
                                      }});
    options.manual_options.push_back({"-v", "--verbose", "equivalent to --log-level verbose",
                                      [&level](int, const char**, int) {
                                          level = SD_LOG_VERBOSE;
                                          return 0;
                                      }});
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

    bool valid = false;
    for (int i = 1; i < argc; i++) {
        arg            = argv[i];
        bool found_arg = false;

        for (auto& options : options_list) {
            if (match_and_apply(options.string_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    if (option.concat && !option.target.get().empty()) {
                        if (option.concat > 0 && option.concat <= 0xff) {
                            option.target.set(option.target.get() + static_cast<char>(option.concat));
                        }
                        option.target.set(option.target.get() + argv_to_utf8(i, argv));
                    } else {
                        option.target.set(argv_to_utf8(i, argv));
                    }
                    found_arg = true;
                }))
                break;

            if (match_and_apply(options.int_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    try {
                        option.target.set(std::stoi(argv[i]));
                    } catch (const std::invalid_argument&) {
                        invalid_arg = true;
                    }
                    found_arg = true;
                }))
                break;

            if (match_and_apply(options.float_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    try {
                        option.target.set(std::stof(argv[i]));
                    } catch (const std::invalid_argument&) {
                        invalid_arg = true;
                    }
                    found_arg = true;
                }))
                break;

            if (match_and_apply(options.bool_options, [&](auto& option) {
                    option.target.set(option.keep_true);
                    found_arg = true;
                }))
                break;

            if (match_and_apply(options.manual_options, [&](auto& option) {
                    int ret = option.cb(argc, argv, i, valid);
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
            if (!valid) {
                LOG_ERROR("error: invalid parameter for argument \"%s\": \"%s\"",
                          arg.c_str(), (i >= argc) ? "" : argv[i]);
            }
            return false;
        }
        if (!found_arg) {
            LOG_ERROR("error: unknown argument: %s", arg.c_str());
            return false;
        }
    }

    return true;
}

static int parse_scale_override(int argc, const char** argv, int index, std::optional<float>& scale) {
    if (++index >= argc) {
        return -1;
    }
    try {
        size_t end              = 0;
        const std::string value = argv[index];
        float parsed            = std::stof(value, &end);
        if (end != value.size() || !std::isfinite(parsed) || parsed < 0.f ||
            (parsed > 0.f && !std::isfinite(1.f / parsed))) {
            return -1;
        }
        scale = parsed;
    } catch (const std::exception&) {
        return -1;
    }
    return 1;
}

static int parse_on_off_arg(int argc, const char** argv, int index, const char* option, std::optional<bool>& value) {
    if (++index >= argc) {
        LOG_ERROR("%s requires 'on' or 'off'", option);
        return -1;
    }
    const std::string arg = argv[index];
    if (arg == "on") {
        value = true;
    } else if (arg == "off") {
        value = false;
    } else {
        LOG_ERROR("invalid %s value '%s'; expected 'on' or 'off'", option, argv[index]);
        return -1;
    }
    return 1;
}

ArgOptions SDContextParams::get_options() {
    ArgOptions options;
    options.string_options = {
        {"-m",
         "--model",
         "path to full model",
         0,
         &model_path},
        {"",
         "--clip_l",
         "path to the clip-l text encoder",
         0,
         &clip_l_path},
        {"", "--clip_g",
         "path to the clip-g text encoder",
         0,
         &clip_g_path},
        {"",
         "--clip_vision",
         "path to the clip-vision encoder",
         0,
         &clip_vision_path},
        {"",
         "--t5xxl",
         "path to the t5xxl text encoder",
         0,
         &t5xxl_path},
        {"",
         "--llm",
         "path to the llm text encoder. For example: (qwenvl2.5 for qwen-image, mistral-small3.2 for flux2, ...)",
         0,
         &llm_path},
        {"",
         "--tokenizer",
         "tokenizer.json path, or comma-separated main=FILE,clip-l=FILE,clip-g=FILE assignments; required for PiD and Lens",
         (int)',',
         &tokenizer},
        {"",
         "--llm_vision",
         "path to the llm vit",
         0,
         &llm_vision_path},
        {"",
         "--qwen2vl",
         "alias of --llm. Deprecated.",
         0,
         &llm_path},
        {"",
         "--qwen2vl_vision",
         "alias of --llm_vision. Deprecated.",
         0,
         &llm_vision_path},
        {"",
         "--diffusion-model",
         "path to the standalone diffusion model",
         0,
         &diffusion_model_path},
        {"",
         "--high-noise-diffusion-model",
         "path to the standalone high noise diffusion model",
         0,
         &high_noise_diffusion_model_path},
        {"",
         "--uncond-diffusion-model",
         "path to the standalone unconditional diffusion model, currently used by Ideogram4 CFG",
         0,
         &uncond_diffusion_model_path},
        {"",
         "--embeddings-connectors",
         "path to embeddings connectors (LTXAV or P-H3-Image)",
         0,
         &embeddings_connectors_path},
        {"",
         "--vae",
         "path to standalone vae model",
         0,
         &vae_path},
        {"",
         "--vae-format",
         "VAE latent format override: auto, flux, sd3, flux2, or wan (default: auto)",
         0,
         &vae_format},
        {"",
         "--audio-vae",
         "path to standalone LTX audio vae model",
         0,
         &audio_vae_path},
        {"",
         "--audio-encoder",
         "path to wav2vec2 audio encoder model (Wan2.2 S2V)",
         0,
         &audio_encoder_path},
        {"",
         "--taesd",
         "path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)",
         0,
         &taesd_path},
        {"",
         "--tae",
         "alias of --taesd",
         0,
         &taesd_path},
        {"",
         "--control-net",
         "path to control net model",
         0,
         &control_net_path},
        {"",
         "--ip-adapter",
         "path to IP-Adapter model (requires --clip_vision)",
         0,
         &ip_adapter_path},
        {"",
         "--motion-module",
         "path to AnimateDiff motion module (SD 1.5); enables video generation on --video-frames > 1",
         0,
         &motion_module_path},
        {"",
         "--embd-dir",
         "embeddings directory",
         0,
         &embedding_dir},
        {"",
         "--lora-model-dir",
         "lora model directory",
         0,
         &lora_model_dir},
        {"",
         "--hires-upscalers-dir",
         "highres fix upscaler model directory",
         0,
         &hires_upscalers_dir},
        {"",
         "--tensor-type-rules",
         "weight type per tensor pattern (example: \"^vae\\.=f16,model\\.=q8_0\")",
         (int)',',
         &tensor_type_rules},
        {"",
         "--model-args",
         "extra model args, key=value list. Supports chroma_use_dit_mask, chroma_use_t5_mask, "
         "chroma_t5_mask_pad, qwen_image_zero_cond_t, qwen_image_2_1_prefix_cache, "
         "qwen_image_2_1_prefix_cache_type (auto or a type name from --type)",
         (int)',',
         &model_args},
        {"",
         "--photo-maker",
         "path to PHOTOMAKER model",
         0,
         &photo_maker_path},
        {"",
         "--pulid-weights",
         "path to PuLID Flux weights",
         0,
         &pulid_weights_path},
        {"",
         "--upscale-model",
         "path to esrgan model.",
         0,
         &esrgan_path},
        {"",
         "--backend",
         "runtime backend assignment, e.g. cpu or clip=cpu,vae=cuda0,diffusion=vulkan0",
         (int)',',
         &backend},
        {"",
         "--params-backend",
         "parameter backend assignment, e.g. disk, cpu, or diffusion=disk,clip=cpu",
         (int)',',
         &params_backend},
        {"",
         "--split-mode",
         "weight distribution for modules assigned multiple devices (--backend \"diffusion=cuda0&cuda1\"): "
         "layer (whole transformer blocks per device, default) or row (matmul rows split across devices, CUDA only). "
         "Accepts a single mode or per-module assignments, e.g. row or diffusion=row,te=layer",
         (int)',',
         &split_mode},
        {"",
         "--rpc-servers",
         "comma-separated list of RPC servers to connect to for offloading, in the format host:port, e.g. localhost:50052,192.168.1.3:50052",
         (int)',',
         &rpc_servers},
        {"",
         "--max-vram",
         "optional per-device budget in GiB for managed weights and runner buffers during automatic graph-cut execution. Accepts a single value or assignments by backend/device, e.g. 6 or cuda0=6,vulkan0=4. 0 uses live free VRAM without an explicit budget; a negative value reserves that much free VRAM",
         0,
         &max_vram},
    };

    options.int_options = {
        {"-t",
         "--threads",
         "positive number of computation threads; unset uses the library CPU core selection",
         &n_threads},
        {"",
         "--conditioning-cache-size",
         "maximum cached conditioning results; unset uses the library default, 0 disables caching",
         &conditioning_cache_size},
    };

    options.bool_options = {
        {"",
         "--list-devices",
         "list available ggml backend devices, including --rpc-servers (one 'name<TAB>description' per line), and exit; "
         "the names are the device names accepted by --backend and --params-backend",
         true, &list_devices},
        {"",
         "--disable-prefetch",
         "disable asynchronous next-segment weight prefetch (defaults to false)",
         true, &disable_prefetch},
        {"",
         "--disable-segmented-compute",
         "force monolithic graph execution even when automatic graph cutting is needed (defaults to false)",
         true, &disable_segmented_compute},
        {"",
         "--eager-load",
         "load all params into the params backend at model-load time instead of lazily on first use (defaults to false)",
         true, &eager_load},
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
         "deprecated; use --backend controlnet=cpu",
         true, &control_net_cpu},
        {"",
         "--clip-on-cpu",
         "deprecated; use --backend te=cpu",
         true, &clip_on_cpu},
        {"",
         "--vae-on-cpu",
         "deprecated; use --backend vae=cpu",
         true, &vae_on_cpu},
        {"",
         "--sage-attn",
         "use native CUDA SageAttention in the diffusion model, with flash/default attention fallback",
         true, &sage_attn},
        {"",
         "--taesd-preview-only",
         "prevents usage of taesd for decoding the final image; the VAE keeps encoding / decoding and taesd is only used for step previews",
         true, &taesd_preview_only},
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
        rng_type        = arg;
        return 1;
    };

    auto on_sampler_rng_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg  = argv[index];
        sampler_rng_type = arg;
        return 1;
    };

    auto on_prediction_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg = argv[index];
        prediction      = arg;
        return 1;
    };

    auto on_lora_apply_mode_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg = argv[index];
        lora_apply_mode = arg;
        return 1;
    };

    options.manual_options = {
        {"",
         "--linear-scale",
         "linear input scale override (float, default: 0 = model default, 1 = no scaling)",
         [this](int argc, const char** argv, int index) {
             return parse_scale_override(argc, argv, index, linear_scale);
         }},
        {"",
         "--attn-scale",
         "flash-attention K/V scale override (float, default: 0 = model default, 1 = no scaling); applies when flash attention is selected",
         [this](int argc, const char** argv, int index) {
             return parse_scale_override(argc, argv, index, attn_scale);
         }},
        {"",
         "--diffusion-conv-direct",
         "deprecated; use --conv-direct auto,diffusion=on,controlnet=on",
         [this](int, const char**, int) {
             LOG_WARN("--diffusion-conv-direct is deprecated; use --conv-direct auto,diffusion=on,controlnet=on instead.");
             diffusion_conv_direct = true;
             return 0;
         }},
        {"",
         "--vae-conv-direct",
         "deprecated; use --conv-direct auto,vae=on",
         [this](int, const char**, int) {
             LOG_WARN("--vae-conv-direct is deprecated; use --conv-direct auto,vae=on instead.");
             vae_conv_direct = true;
             return 0;
         }},
        {"",
         "--conv-direct",
         "on|off|auto or diffusion/vae/controlnet assignments (default: auto), e.g. auto,vae=off. Overrides legacy convolution flags",
         [this](int argc, const char** argv, int index) {
             if (++index >= argc || argv[index][0] == '\0') {
                 LOG_ERROR("--conv-direct requires on, off, auto, or module assignments");
                 return -1;
             }
             conv_direct_mode = argv[index];
             return 1;
         }},
        {"",
         "--flash-attn",
         "on|off|auto or module assignments (default: auto), e.g. auto,te=off,vae=on. Last repeated option wins",
         [this](int argc, const char** argv, int index) {
             if (++index >= argc) {
                 LOG_ERROR("--flash-attn requires on, off, or auto");
                 return -1;
             }
             const std::string value = argv[index];
             if (value.empty()) {
                 LOG_ERROR("--flash-attn requires a nonempty value");
                 return -1;
             }
             flash_attn_mode = value;
             return 1;
         }},
        {"",
         "--fa",
         "deprecated; use --flash-attn on",
         [this](int, const char**, int) {
             LOG_WARN("--fa is deprecated; use --flash-attn on instead.");
             flash_attn_mode = "on";
             return 0;
         }},
        {"",
         "--diffusion-fa",
         "deprecated; use --flash-attn off,diffusion=on",
         [this](int, const char**, int) {
             LOG_WARN("--diffusion-fa is deprecated; use --flash-attn off,diffusion=on instead.");
             flash_attn_mode = "off,diffusion=on";
             return 0;
         }},
        {"",
         "--auto-fit",
         "on|off (default: on). Preserve --backend (otherwise select one GPU) and place weights on the compute GPU, "
         "RAM, another GPU, or disk in that order, according to available memory (--max-vram limits GPU budgets). "
         "Disabled by explicit --params-backend; uses automatic graph segmentation when needed",
         [this](int argc, const char** argv, int index) {
             return parse_on_off_arg(argc, argv, index, "--auto-fit", auto_fit);
         }},
        {"",
         "--batched-cfg",
         "on|off (default: on). Run the conditional and unconditional CFG branches in one batched UNet forward when supported",
         [this](int argc, const char** argv, int index) {
             return parse_on_off_arg(argc, argv, index, "--batched-cfg", batched_cfg);
         }},
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
         "prediction type override, one of [eps, v, edm_v, sd3_flow, flux_flow, sefi_flow, minit2i_flow, sensenova_u1_flow, flux3_flow, h3_av_flow, ming_image_flow]",
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

bool SDContextParams::print_devices() const {
    if (!sd_add_rpc_servers(rpc_servers ? rpc_servers->c_str() : nullptr)) {
        return false;
    }
    size_t device_list_size = sd_list_devices(nullptr, 0);
    std::vector<char> devices(device_list_size + 1);
    sd_list_devices(devices.data(), devices.size());
    fputs(devices.data(), stdout);
    return true;
}

void SDContextParams::build_embedding_map() {
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt", ".ckpt"};

    const fs::path embedding_dir_path = fs::u8path(embedding_dir);
    if (!fs::exists(embedding_dir_path) || !fs::is_directory(embedding_dir_path)) {
        return;
    }

    for (auto& p : fs::directory_iterator(embedding_dir_path)) {
        if (!p.is_regular_file())
            continue;

        auto path       = p.path();
        std::string ext = path.extension().u8string();

        bool valid = false;
        for (auto& e : valid_ext) {
            if (ext == e) {
                valid = true;
                break;
            }
        }
        if (!valid)
            continue;

        std::string key   = path.stem().u8string();
        std::string value = path.u8string();

        embedding_map[key] = value;
    }
}

bool SDContextParams::resolve(SDMode mode) {
    build_embedding_map();

    return true;
}

bool SDContextParams::validate(SDMode mode) {
    if (mode == CONVERT) {
        const bool has_convert_input = model_path.length() != 0 ||
                                       clip_l_path.length() != 0 ||
                                       clip_g_path.length() != 0 ||
                                       t5xxl_path.length() != 0 ||
                                       diffusion_model_path.length() != 0 ||
                                       vae_path.length() != 0;
        if (!has_convert_input) {
            LOG_ERROR("error: convert mode needs at least one model input path\n");
            return false;
        }
    } else if (mode != UPSCALE && mode != METADATA && model_path.length() == 0 && diffusion_model_path.length() == 0) {
        LOG_ERROR("error: the following arguments are required: model_path/diffusion_model\n");
        return false;
    }

    if (mode == UPSCALE) {
        if (esrgan_path.length() == 0) {
            LOG_ERROR("error: upscale mode needs an upscaler model (--upscale-model)\n");
            return false;
        }
    }

    if (!to_context_opt()) {
        LOG_ERROR("%s", sd_get_last_error());
        return false;
    }

    return true;
}

bool SDContextParams::resolve_and_validate(SDMode mode) {
    if (!resolve(mode)) {
        return false;
    }
    if (!validate(mode)) {
        return false;
    }
    return true;
}

void SDContextParams::prepare_backend_assignments() {
    effective_backend        = backend.value_or("");
    effective_params_backend = params_backend.value_or("");

    if (offload_params_to_cpu) {
        prepend_backend_assignment(effective_params_backend, "*=cpu");
    }

    if (clip_on_cpu) {
        prepend_backend_assignment(effective_backend, "te=cpu");
    }
    if (vae_on_cpu) {
        prepend_backend_assignment(effective_backend, "vae=cpu");
    }
    if (control_net_cpu) {
        prepend_backend_assignment(effective_backend, "controlnet=cpu");
    }
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
        << "  conditioning_cache_size: " << conditioning_cache_size << ",\n"
        << "  model_path: \"" << model_path << "\",\n"
        << "  clip_l_path: \"" << clip_l_path << "\",\n"
        << "  clip_g_path: \"" << clip_g_path << "\",\n"
        << "  clip_vision_path: \"" << clip_vision_path << "\",\n"
        << "  t5xxl_path: \"" << t5xxl_path << "\",\n"
        << "  llm_path: \"" << llm_path << "\",\n"
        << "  llm_vision_path: \"" << llm_vision_path << "\",\n"
        << "  tokenizer: \"" << tokenizer << "\",\n"
        << "  diffusion_model_path: \"" << diffusion_model_path << "\",\n"
        << "  high_noise_diffusion_model_path: \"" << high_noise_diffusion_model_path << "\",\n"
        << "  uncond_diffusion_model_path: \"" << uncond_diffusion_model_path << "\",\n"
        << "  embeddings_connectors_path: \"" << embeddings_connectors_path << "\",\n"
        << "  vae_path: \"" << vae_path << "\",\n"
        << "  vae_format: \"" << vae_format << "\",\n"
        << "  audio_vae_path: \"" << audio_vae_path << "\",\n"
        << "  audio_encoder_path: \"" << audio_encoder_path << "\",\n"
        << "  taesd_path: \"" << taesd_path << "\",\n"
        << "  esrgan_path: \"" << esrgan_path << "\",\n"
        << "  control_net_path: \"" << control_net_path << "\",\n"
        << "  embedding_dir: \"" << embedding_dir << "\",\n"
        << "  embeddings: " << embeddings_str << "\n"
        << "  wtype: " << sd_type_name(wtype) << ",\n"
        << "  tensor_type_rules: \"" << tensor_type_rules << "\",\n"
        << "  lora_model_dir: \"" << lora_model_dir << "\",\n"
        << "  hires_upscalers_dir: \"" << hires_upscalers_dir << "\",\n"
        << "  photo_maker_path: \"" << photo_maker_path << "\",\n"
        << "  rng_type: " << rng_type << ",\n"
        << "  sampler_rng_type: " << sampler_rng_type << ",\n"
        << "  offload_params_to_cpu: " << (offload_params_to_cpu ? "true" : "false") << ",\n"
        << "  max_vram: \"" << max_vram << "\",\n"
        << "  disable_prefetch: " << disable_prefetch << ",\n"
        << "  disable_segmented_compute: " << disable_segmented_compute << ",\n"
        << "  batched_cfg: " << batched_cfg << ",\n"
        << "  eager_load: " << eager_load << ",\n"
        << "  backend: \"" << backend << "\",\n"
        << "  params_backend: \"" << params_backend << "\",\n"
        << "  split_mode: \"" << split_mode << "\",\n"
        << "  model_args: \"" << model_args << "\",\n"
        << "  auto_fit: " << auto_fit << ",\n"
        << "  enable_mmap: " << enable_mmap << ",\n"
        << "  control_net_cpu: " << (control_net_cpu ? "true" : "false") << ",\n"
        << "  clip_on_cpu: " << (clip_on_cpu ? "true" : "false") << ",\n"
        << "  vae_on_cpu: " << (vae_on_cpu ? "true" : "false") << ",\n"
        << "  flash_attn_mode: " << flash_attn_mode << ",\n"
        << "  conv_direct_mode: " << conv_direct_mode << ",\n"
        << "  sage_attn: " << sage_attn << ",\n"
        << "  linear_scale: " << linear_scale << ",\n"
        << "  attn_scale: " << attn_scale << ",\n"
        << "  diffusion_conv_direct: " << diffusion_conv_direct << ",\n"
        << "  vae_conv_direct: " << vae_conv_direct << ",\n"
        << "  taesd_preview_only: " << taesd_preview_only << ",\n"
        << "  prediction: " << prediction << ",\n"
        << "  lora_apply_mode: " << lora_apply_mode << ",\n"
        << "  force_sdxl_vae_conv_scale: " << force_sdxl_vae_conv_scale << "\n"
        << "}";
    return oss.str();
}

SDOptPtr SDContextParams::to_context_opt() {
    prepare_backend_assignments();
    embedding_vec.clear();
    embedding_vec.reserve(embedding_map.size());
    for (const auto& kv : embedding_map) {
        sd_embedding_t item;
        item.name = kv.first.c_str();
        item.path = kv.second.c_str();
        embedding_vec.emplace_back(item);
    }

    SDOptBuilder opt(sd_ctx_opt_create());
    opt.set_string("model_path", model_path);
    opt.set_string("clip_l_path", clip_l_path);
    opt.set_string("clip_g_path", clip_g_path);
    opt.set_string("clip_vision_path", clip_vision_path);
    opt.set_string("t5xxl_path", t5xxl_path);
    opt.set_string("llm_path", llm_path);
    opt.set_string("llm_vision_path", llm_vision_path);
    opt.set_string("tokenizer", tokenizer);
    opt.set_string("diffusion_model_path", diffusion_model_path);
    opt.set_string("high_noise_diffusion_model_path", high_noise_diffusion_model_path);
    opt.set_string("uncond_diffusion_model_path", uncond_diffusion_model_path);
    opt.set_string("embeddings_connectors_path", embeddings_connectors_path);
    opt.set_string("vae_path", vae_path);
    opt.set_string("audio_vae_path", audio_vae_path);
    opt.set_string("audio_encoder_path", audio_encoder_path);
    opt.set_string("taesd_path", taesd_path);
    opt.set_string("control_net_path", control_net_path);
    opt.set_string("ip_adapter_path", ip_adapter_path);
    opt.set_string("motion_module_path", motion_module_path);
    opt.set_string("photo_maker_path", photo_maker_path);
    opt.set_string("pulid_weights_path", pulid_weights_path);
    opt.set_string("tensor_type_rules", tensor_type_rules);
    opt.set_int("n_threads", n_threads);
    opt.set_int("conditioning_cache_size", conditioning_cache_size);
    if (wtype != SD_TYPE_COUNT)
        opt.set_string("wtype", sd_type_name(wtype));
    opt.set_string("rng_type", rng_type);
    opt.set_string("sampler_rng_type", sampler_rng_type);
    opt.set_string("prediction", prediction);
    opt.set_string("lora_apply_mode", lora_apply_mode);
    opt.set_bool("enable_mmap", enable_mmap);
    opt.set_string("flash_attn_mode", flash_attn_mode);
    opt.set_string("conv_direct_mode", conv_direct_mode);
    opt.set_bool("sage_attn", sage_attn);
    opt.set_float("linear_scale", linear_scale);
    opt.set_float("attn_scale", attn_scale);
    opt.set_bool("tae_preview_only", taesd_preview_only);
    opt.set_bool("diffusion_conv_direct", diffusion_conv_direct);
    opt.set_bool("vae_conv_direct", vae_conv_direct);
    opt.set_bool("force_sdxl_vae_conv_scale", force_sdxl_vae_conv_scale);
    opt.set_string("vae_format", vae_format);
    opt.set_string("max_vram", max_vram);
    opt.set_bool("disable_prefetch", disable_prefetch);
    opt.set_bool("disable_segmented_compute", disable_segmented_compute);
    opt.set_bool("batched_cfg", batched_cfg);
    opt.set_bool("eager_load", eager_load);
    if (backend || !effective_backend.empty())
        opt.set_string("backend", effective_backend);
    if (params_backend || !effective_params_backend.empty())
        opt.set_string("params_backend", effective_params_backend);
    opt.set_string("split_mode", split_mode);
    opt.set_bool("auto_fit", auto_fit);
    opt.set_string("rpc_servers", rpc_servers);
    opt.set_string("model_args", model_args);
    opt.set_embeddings("embeddings", embedding_vec.data(), embedding_vec.size());
    return opt.finish();
}

SDGenerationParams::SDGenerationParams() {
}

ArgOptions SDGenerationParams::get_options() {
    ArgOptions options;
    options.string_options = {
        {"-p",
         "--prompt",
         "the prompt to render",
         0,
         &prompt},
        {"-n",
         "--negative-prompt",
         "the negative prompt (default: \"\")",
         0,
         &negative_prompt},
        {"",
         "--ad-model",
         "path to a converted YOLOv8 detection model for ADetailer",
         0,
         &ad_model_path},
        {"",
         "--ad-prompt",
         "ADetailer prompt; empty inherits the main prompt, supports [PROMPT], [SEP], and [SKIP]",
         0,
         &ad_prompt},
        {"",
         "--ad-negative-prompt",
         "ADetailer negative prompt; empty inherits the main negative prompt, supports [PROMPT] and [SEP]",
         0,
         &ad_negative_prompt},
        {"",
         "--extra-ad-args",
         "extra ADetailer args, key=value list. Supports input_size, confidence, nms, max_detections, mask_k_largest, mask_min_ratio, mask_max_ratio, dilate_erode, x_offset, y_offset, mask_mode, merge_masks, invert_mask, mask_blur, inpaint_padding, inpaint_width, inpaint_height, denoising_strength, steps, cfg_scale, sample_method, scheduler, sort_by",
         (int)',',
         &extra_ad_args},
        {"-i",
         "--init-img",
         "path to the init image, or a single input image for vid_upscale",
         0,
         &init_image_path},
        {"",
         "--end-img",
         "path to the end image, required by flf2v",
         0,
         &end_image_path},
        {"",
         "--mask",
         "path to the mask image",
         0,
         &mask_image_path},
        {"",
         "--control-image",
         "path to control image, control net",
         0,
         &control_image_path},
        {"",
         "--ip-adapter-image",
         "path to the IP-Adapter reference image",
         0,
         &ip_adapter_image_path},
        {"",
         "--control-video",
         "path to control video frames, It must be a directory path. The video frames inside should be stored as images in "
         "lexicographical (character) order. For example, if the control video path is `frames`, the directory contain images "
         "such as 00.png, 01.png, ... etc.",
         0,
         &control_video_path},
        {"",
         "--pm-id-images-dir",
         "path to PHOTOMAKER input id images dir",
         0,
         &pm_id_images_dir},
        {"",
         "--pm-id-embed-path",
         "path to PHOTOMAKER v2 id embed",
         0,
         &pm_id_embed_path},
        {"",
         "--pulid-id-embedding",
         "path to PuLID id embedding",
         0,
         &pulid_id_embedding_path},
        {"",
         "--hires-upscaler",
         "highres fix upscaler, Lanczos, Nearest, Latent, Latent (nearest), Latent (nearest-exact), "
         "Latent (antialiased), Latent (bicubic), Latent (bicubic antialiased), or a model name "
         "under --hires-upscalers-dir (default: Latent)",
         0,
         &hires_upscaler},
        {"",
         "--control-type",
         "kind of the control image, one of [openpose, depth, scribble, canny, normal, segment, tile, repaint]. "
         "Required by an SDXL Control Net Union model (tile and repaint only exist in its ProMax variant), ignored by any other Control Net. "
         "Also accepted: hed, softedge, pidi, ted (scribble); lineart, anime_lineart, mlsd (canny); seg (segment); inpaint, outpaint (repaint)",
         0,
         &control_type},
        {"",
         "--extra-sample-args",
         "extra sampler/scheduler/guidance args, key=value list. CFG supports guidance_schedule; APG supports apg_eta, apg_momentum, apg_norm_threshold, apg_norm_threshold_smoothing; SLG supports slg_uncond; lcm supports noise_clip_std, noise_scale_start, noise_scale_end; flux supports base_shift, max_shift; ltx2 supports max_shift, base_shift, stretch, terminal; euler_ge supports gamma; beta scheduler supports alpha, beta; logit_normal supports mu, std, logsnr_min, logsnr_max, resolution_aware; llada_image supports uniform; lms supports lms_max_order, lms_shift, lms_divisions; noise-injecting samplers support noise_sampler with value iid (default except for dpm++2m_sde_bt) or brownian_tree; brownian_tree_rng supports cpu (default), cuda, std_default or sampler_rng",
         (int)',',
         &extra_sample_args},
        {"",
         "--extra-tiling-args",
         "extra VAE tiling args, key=value list. Supported video VAEs accept temporal_tile_frames/temporal_tile_size (default: 4), temporal_tile_overlap (default: 1); Wan2.2 TI2V 5B accepts temporal_decode_frames (default: 1)",
         (int)',',
         &extra_tiling_args},
        {"",
         "--ref-image-args",
         "Key-value list to set up the way the reference images are processed (empty = auto-detect from model weigths)",
         (int)',',
         &ref_image_args},
        {"", "--image-preprocess",
         "Image preprocessing rule: target=init|end|mask|control|ref|ip-adapter|id|control-frame,index=N,mode=auto|none|stretch|crop|crop-resize|fit-pad,filter=auto|nearest|nearest-exact|bilinear|bicubic|lanczos,antialias=auto|true|false,width=W,height=H,anchor=center|top|bottom|left|right,pad_color=#RRGGBB[AA],canny=true|false. Repeat for multiple rules.",
         (int)';', &image_preprocess},
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
         "--qwen-image-layers",
         "number of Qwen Image Layered layers; latent/output count is layers + 1 (default: 3)",
         &qwen_image_layers},
        {"",
         "--video-frames",
         "video frames; unset uses the library default",
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
        {"",
         "--hires-width",
         "highres fix target width, 0 to use --hires-scale (default: 0)",
         &hires_width},
        {"",
         "--hires-height",
         "highres fix target height, 0 to use --hires-scale (default: 0)",
         &hires_height},
        {"",
         "--hires-steps",
         "highres fix second pass sample steps, 0 to reuse --steps (default: 0)",
         &hires_steps},
        {"",
         "--hires-upscale-tile-size",
         "highres fix upscaler tile size, reserved for model-backed upscalers (default: 128)",
         &hires_upscale_tile_size},
    };

    options.float_options = {
        {"",
         "--cfg-scale",
         "unconditional guidance scale: (default: 7.0)",
         &sample_params.guidance.txt_cfg},
        {"",
         "--img-cfg-scale",
         "image guidance scale for inpaint or image edit models: (default: same as --cfg-scale)",
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
         "noise multiplier (default: 0 for ddim_trailing, tcd, res_multistep and res_2s; 1 for euler_a, er_sde, dpm++2s_a, dpm++2m_sde and dpm++2m_sde_bt)",
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
         "(high noise) image guidance scale for inpaint or image edit models (default: same as --cfg-scale)",
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
         "(high noise) noise multiplier (default: 0 for ddim_trailing, tcd, res_multistep and res_2s; 1 for euler_a, er_sde, dpm++2s_a, dpm++2m_sde and dpm++2m_sde_bt)",
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
         "--pulid-id-weight",
         "strength of PuLID identity injection",
         &pulid_id_weight},
        {"",
         "--control-strength",
         "strength to apply Control Net (default: 0.9). 1.0 corresponds to full destruction of information in init image",
         &control_strength},
        {"",
         "--ip-adapter-strength",
         "strength to apply IP-Adapter (default: 1.0)",
         &ip_adapter_strength},
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
        {"",
         "--hires-scale",
         "highres fix scale when target size is not set (default: 2.0)",
         &hires_scale},
        {"",
         "--hires-denoising-strength",
         "highres fix second pass denoising strength (default: 0.7)",
         &hires_denoising_strength},
    };

    options.bool_options = {
        {"",
         "--increase-ref-index",
         "automatically increase the indices of references images based on the order they are listed (starting with 1).",
         true,
         &increase_ref_index},
        {"",
         "--circular",
         "enable circular padding on both axes for tileable output",
         true, &circular},
        {"",
         "--circularx",
         "enable circular padding on x-axis (width) only",
         true, &circular_x},
        {"",
         "--circulary",
         "enable circular padding on y-axis (height) only",
         true, &circular_y},
        {"",
         "--disable-image-metadata",
         "do not embed generation metadata on image files",
         false,
         &embed_image_metadata},
        {"",
         "--vae-tiling",
         "process vae encode and decode in spatial tiles to reduce memory usage (default: 256x256 image pixels)",
         true,
         &vae_tiling_params.enabled},
        {"",
         "--temporal-tiling",
         "enable temporal tiling for supported video VAE decode",
         true,
         &vae_tiling_params.temporal_tiling},
        {"",
         "--hires",
         "enable highres fix",
         true,
         &hires_enabled},
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
        sample_params.sample_method = arg;
        return 1;
    };

    auto on_high_noise_sample_method_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg                        = argv[index];
        high_noise_sample_params.sample_method = arg;
        return 1;
    };

    auto on_scheduler_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg         = argv[index];
        sample_params.scheduler = arg;
        return 1;
    };

    auto on_skip_layers_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        std::string layers_str = argv[index];
        if (layers_str.size() < 2 || layers_str.front() != '[' || layers_str.back() != ']') {
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
                size_t used     = 0;
                const int layer = std::stoi(token, &used);
                if (used != token.size())
                    return -1;
                layers.push_back(layer);
            } catch (const std::exception&) {
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
        if (layers_str.size() < 2 || layers_str.front() != '[' || layers_str.back() != ']') {
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
                size_t used     = 0;
                const int layer = std::stoi(token, &used);
                if (used != token.size())
                    return -1;
                layers.push_back(layer);
            } catch (const std::exception&) {
                return -1;
            }
        }
        high_noise_skip_layers = layers;
        return 1;
    };

    auto parse_sigmas_arg = [&](const char* value, std::optional<std::vector<float>>* target, const char* option_name) {
        if (target == nullptr || value == nullptr) {
            return -1;
        }
        std::string sigmas_str = value;
        if (sigmas_str == "qwen-image-2.1-turbo") {
            sigmas_str = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0";
        } else if (sigmas_str == "p-h3-image") {
            sigmas_str = "1.0,0.984753,0.967905,0.951688,0.925372,0.881923,0.810083,0.528718,0.0";
        } else if (sigmas_str == "minimax-h3-image") {
            sigmas_str = "1.0,0.9675236940383911,0.8875485062599182,0.75,0.5327739119529724,0.23200887441635132,0.0";
        } else if (sigmas_str == "hyperflow-h3-flash-4") {
            sigmas_str = "1.0,0.962575853,0.755781889,0.312497854,0.0";
        } else if (sigmas_str == "hyperflow-h3-flash-6") {
            sigmas_str = "1.0,0.982655942,0.914642572,0.674755692,0.385223746,0.159055337,0.0";
        } else if (sigmas_str == "hyperflow-h3-flash-8") {
            sigmas_str = "1.0,0.993909776,0.9842875,0.966063678,0.923076928,0.834942341,0.696852028,0.468753338,0.0";
        }
        if (!sigmas_str.empty() && sigmas_str.front() == '[') {
            sigmas_str.erase(0, 1);
        }
        if (!sigmas_str.empty() && sigmas_str.back() == ']') {
            sigmas_str.pop_back();
        }

        std::vector<float> parsed;
        std::stringstream ss(sigmas_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
            item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!item.empty()) {
                try {
                    parsed.push_back(std::stof(item));
                } catch (const std::invalid_argument&) {
                    LOG_ERROR("error: invalid float value '%s' in %s", item.c_str(), option_name);
                    return -1;
                } catch (const std::out_of_range&) {
                    LOG_ERROR("error: float value '%s' out of range in %s", item.c_str(), option_name);
                    return -1;
                }
            }
        }

        if (parsed.empty() && !sigmas_str.empty()) {
            LOG_ERROR("error: could not parse any sigma values from '%s'", value);
            return -1;
        }
        *target = std::move(parsed);
        return 1;
    };

    auto on_sigmas_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        return parse_sigmas_arg(argv[index], &custom_sigmas, "--sigmas");
    };

    auto on_hires_sigmas_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        return parse_sigmas_arg(argv[index], &hires_custom_sigmas, "--hires-sigmas");
    };

    auto on_ref_image_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        ref_image_paths.push_back(argv_to_utf8(index, argv));
        return 1;
    };

    auto on_ref_video_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        ref_video_paths.push_back(argv_to_utf8(index, argv));
        return 1;
    };

    auto on_ref_video_audio_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        ref_video_audio_paths.push_back(argv_to_utf8(index, argv));
        return 1;
    };

    auto on_ref_audio_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        ref_audio_paths.push_back(argv_to_utf8(index, argv));
        return 1;
    };

    auto on_audio_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        ref_audio_paths.push_back(argv_to_utf8(index, argv));
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
                std::string tile_w_str        = tile_size_str.substr(0, x_pos);
                std::string tile_h_str        = tile_size_str.substr(x_pos + 1);
                vae_tiling_params.tile_size_w = std::stoi(tile_w_str);
                vae_tiling_params.tile_size_h = std::stoi(tile_h_str);
            } else {
                vae_tiling_params.tile_size_w = vae_tiling_params.tile_size_h = std::stoi(tile_size_str);
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
                std::string rel_w_str        = rel_size_str.substr(0, x_pos);
                std::string rel_h_str        = rel_size_str.substr(x_pos + 1);
                vae_tiling_params.rel_size_w = std::stof(rel_w_str);
                vae_tiling_params.rel_size_h = std::stof(rel_h_str);
            } else {
                vae_tiling_params.rel_size_w = vae_tiling_params.rel_size_h = std::stof(rel_size_str);
            }
        } catch (const std::invalid_argument&) {
            return -1;
        } catch (const std::out_of_range&) {
            return -1;
        }
        return 1;
    };

    auto on_prompt_file_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const std::string arg = argv_to_utf8(index, argv);
        std::ifstream f(fs::u8path(arg), std::ios::binary);
        try {
            prompt = std::string(std::istreambuf_iterator<char>{f}, {});
        } catch (const std::ios_base::failure&) {
            f.setstate(std::ios_base::failbit);
        }
        if (f.fail()) {
            LOG_ERROR("error: failed to read prompt file '%s'\n", arg.c_str());
            return -1;
        }
        return 1;
    };

    auto on_negative_prompt_file_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const std::string arg = argv_to_utf8(index, argv);
        std::ifstream f(fs::u8path(arg), std::ios::binary);
        try {
            negative_prompt = std::string(std::istreambuf_iterator<char>{f}, {});
        } catch (const std::ios_base::failure&) {
            f.setstate(std::ios_base::failbit);
        }
        if (f.fail()) {
            LOG_ERROR("error: failed to read negative prompt file '%s'\n", arg.c_str());
            return -1;
        }
        return 1;
    };

    std::string sample_methods;
    for (const char* name : example_options::sample_methods) {
        if (!sample_methods.empty())
            sample_methods += ", ";
        sample_methods += name;
    }
    std::string schedulers;
    for (const char* name : example_options::schedulers) {
        if (!schedulers.empty())
            schedulers += ", ";
        schedulers += name;
    }

    options.manual_options = {
        {"-s",
         "--seed",
         "RNG seed; unset or a negative value requests a random seed",
         on_seed_arg},
        {"",
         "--sampling-method",
         "sampling method, one of [" + sample_methods + "], "
                                                        "default: euler for Flux/SD3/Wan, euler_a otherwise",
         on_sample_method_arg},
        {"",
         "--high-noise-sampling-method",
         "(high noise) sampling method, one of [" + sample_methods + "], "
                                                                     "default: euler for Flux/SD3/Wan, euler_a otherwise",
         on_high_noise_sample_method_arg},
        {"",
         "--scheduler",
         "denoiser sigma scheduler, one of [" + schedulers + "], "
                                                             "alias: normal=discrete, default: model-specific",
         on_scheduler_arg},
        {"",
         "--sigmas",
         "custom sigma values for the sampler, comma-separated (e.g., \"14.61,7.8,3.5,0.0\") or preset: qwen-image-2.1-turbo, p-h3-image, minimax-h3-image, hyperflow-h3-flash-4, hyperflow-h3-flash-6, hyperflow-h3-flash-8.",
         on_sigmas_arg},
        {"",
         "--hires-sigmas",
         "custom sigma values for the highres fix second pass, comma-separated (e.g., \"0.85,0.725,0.421875,0.0\") or preset: qwen-image-2.1-turbo, p-h3-image, minimax-h3-image, hyperflow-h3-flash-4, hyperflow-h3-flash-6, hyperflow-h3-flash-8.",
         on_hires_sigmas_arg},
        {"",
         "--skip-layers",
         "layers to skip for SLG steps; unset uses the library default",
         on_skip_layers_arg},
        {"",
         "--high-noise-skip-layers",
         "(high noise) layers to skip for SLG steps; unset uses the library default",
         on_high_noise_skip_layers_arg},
        {"-r",
         "--ref-image",
         "reference image for Flux Kontext or MiniMax-H3 Ref2VA (can be used multiple times)",
         on_ref_image_arg},
        {"",
         "--ref-video",
         "MiniMax-H3 Ref2VA reference video frame directory at 24 fps (can be used multiple times)",
         on_ref_video_arg},
        {"",
         "--ref-video-audio",
         "WAV soundtrack paired by index with --ref-video (can be used multiple times)",
         on_ref_video_audio_arg},
        {"",
         "--ref-audio",
         "standalone WAV reference for MiniMax-H3 Ref2VA (can be used multiple times)",
         on_ref_audio_arg},
        {"",
         "--audio",
         "driving audio track (Wan2.2 S2V; can be used once)",
         on_audio_arg},
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
         "tile size for vae encode and decode in image pixels, format [W]x[H] or [S] (default: 256x256; requires --vae-tiling)",
         on_tile_size_arg},
        {"",
         "--vae-relative-tile-size",
         "relative tile size for vae encode and decode, format [W]x[H] or [S]: <=1 is a dimension fraction, >1 a target tile count (overrides --vae-tile-size; requires --vae-tiling)",
         on_relative_tile_size_arg},
        {"",
         "--prompt-file",
         "path to the file containing the prompt to render",
         on_prompt_file_arg},
        {"",
         "--negative-prompt-file",
         "path to the file containing the negative prompt",
         on_negative_prompt_file_arg},

    };

    return options;
}

static const std::string k_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static bool is_base64(unsigned char c) {
    return std::isalnum(c) || c == '+' || c == '/';
}

static std::vector<uint8_t> decode_base64_bytes(const std::string& encoded_string) {
    int in_len = static_cast<int>(encoded_string.size());
    int i      = 0;
    int j      = 0;
    int in_    = 0;
    uint8_t char_array_4[4];
    uint8_t char_array_3[3];
    std::vector<uint8_t> ret;

    while (in_len-- && encoded_string[in_] != '=' && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = static_cast<uint8_t>(k_base64_chars.find(char_array_4[i]));
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++) {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = static_cast<uint8_t>(k_base64_chars.find(char_array_4[j]));
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

bool decode_base64_image(const std::string& encoded_input,
                         int target_channels,
                         int expected_width,
                         int expected_height,
                         SDImageOwner& out_image) {
    std::string encoded = encoded_input;
    auto comma_pos      = encoded.find(',');
    if (comma_pos != std::string::npos) {
        encoded = encoded.substr(comma_pos + 1);
    }

    std::vector<uint8_t> image_bytes = decode_base64_bytes(encoded);
    if (image_bytes.empty()) {
        return false;
    }

    int decoded_width    = 0;
    int decoded_height   = 0;
    int resolved_channel = target_channels;
    uint8_t* raw_data    = load_image_from_memory(reinterpret_cast<const char*>(image_bytes.data()),
                                                  static_cast<int>(image_bytes.size()),
                                                  decoded_width,
                                                  decoded_height,
                                                  resolved_channel,
                                                  expected_width,
                                                  expected_height,
                                                  target_channels);
    if (raw_data == nullptr) {
        return false;
    }

    out_image.reset({(uint32_t)decoded_width, (uint32_t)decoded_height, (uint32_t)resolved_channel, raw_data});
    return true;
}

static bool parse_image_json_field(const json& parent,
                                   const char* key,
                                   int channels,
                                   SDImageOwner& out_image) {
    if (!parent.contains(key)) {
        return true;
    }
    if (parent.at(key).is_null()) {
        out_image.reset({0, 0, (uint32_t)channels, nullptr});
        return true;
    }
    if (!parent.at(key).is_string()) {
        return false;
    }
    return decode_base64_image(parent.at(key).get<std::string>(), channels, 0, 0, out_image);
}

/** Parses a JSON array of base64 image strings into owned images. */
static bool parse_image_array_json_field(const json& parent,
                                         const char* key,
                                         int channels,
                                         std::vector<SDImageOwner>& out_images) {
    if (!parent.contains(key)) {
        return true;
    }
    if (parent.at(key).is_null()) {
        out_images.clear();
        return true;
    }
    if (!parent.at(key).is_array()) {
        return false;
    }

    out_images.clear();
    for (const auto& item : parent.at(key)) {
        if (!item.is_string()) {
            return false;
        }
        SDImageOwner image;
        if (!decode_base64_image(item.get<std::string>(), channels, 0, 0, image)) {
            return false;
        }
        out_images.push_back(std::move(image));
    }
    return true;
}

/** Parses a JSON array of base64 WAV strings into owned audio clips. */
static bool parse_audio_array_json_field(const json& parent,
                                         const char* key,
                                         std::vector<SDAudioOwner>& out_audios) {
    if (!parent.contains(key)) {
        return true;
    }
    if (parent.at(key).is_null()) {
        out_audios.clear();
        return true;
    }
    if (!parent.at(key).is_array()) {
        return false;
    }

    out_audios.clear();
    for (const auto& item : parent.at(key)) {
        if (!item.is_string()) {
            return false;
        }
        std::string encoded = item.get<std::string>();
        auto comma_pos      = encoded.find(',');
        if (comma_pos != std::string::npos) {
            encoded = encoded.substr(comma_pos + 1);
        }
        std::vector<uint8_t> audio_bytes = decode_base64_bytes(encoded);
        if (audio_bytes.empty()) {
            return false;
        }
        std::vector<float> samples;
        uint32_t sample_rate = 0;
        uint32_t channels    = 0;
        if (!load_wav_from_memory(audio_bytes.data(), audio_bytes.size(), samples, sample_rate, channels)) {
            return false;
        }
        SDAudioOwner audio;
        audio.reset(std::move(samples), sample_rate, channels);
        out_audios.push_back(std::move(audio));
    }
    return true;
}

/** Parses a JSON array of per-video base64 frame arrays into owned videos. */
static bool parse_video_array_json_field(const json& parent,
                                         const char* key,
                                         std::vector<std::vector<SDImageOwner>>& out_videos) {
    if (!parent.contains(key)) {
        return true;
    }
    if (parent.at(key).is_null()) {
        out_videos.clear();
        return true;
    }
    if (!parent.at(key).is_array()) {
        return false;
    }

    out_videos.clear();
    for (const auto& video : parent.at(key)) {
        if (!video.is_array() || video.empty()) {
            return false;
        }
        std::vector<SDImageOwner> frames;
        for (const auto& frame : video) {
            if (!frame.is_string()) {
                return false;
            }
            SDImageOwner image;
            if (!decode_base64_image(frame.get<std::string>(), 0, 0, 0, image)) {
                return false;
            }
            frames.push_back(std::move(image));
        }
        out_videos.push_back(std::move(frames));
    }
    return true;
}

static bool parse_lora_json_field(const json& parent,
                                  const std::function<std::string(const std::string&)>& lora_path_resolver,
                                  std::map<std::string, float>& lora_map,
                                  std::map<std::string, float>& high_noise_lora_map) {
    if (!parent.contains("lora")) {
        return true;
    }
    if (!parent.at("lora").is_array()) {
        return false;
    }

    lora_map.clear();
    high_noise_lora_map.clear();
    for (const auto& item : parent.at("lora")) {
        if (!item.is_object()) {
            return false;
        }

        std::string path = item.value("path", "");
        if (path.empty()) {
            return false;
        }

        std::string resolved_path = lora_path_resolver ? lora_path_resolver(path) : path;
        if (resolved_path.empty()) {
            return false;
        }

        const float multiplier   = item.value("multiplier", 1.0f);
        const bool is_high_noise = item.value("is_high_noise", false);
        if (is_high_noise) {
            high_noise_lora_map[resolved_path] += multiplier;
        } else {
            lora_map[resolved_path] += multiplier;
        }
    }

    return true;
}

static bool resolve_model_file_from_dir(const std::string& model_name,
                                        const std::string& model_dir,
                                        const std::vector<std::string>& valid_ext,
                                        const char* label,
                                        std::string& resolved_path) {
    if (model_dir.empty()) {
        LOG_ERROR("%s directory is empty", label);
        return false;
    }
    auto ends_with_valid_ext = [&]() {
        for (const auto& ext : valid_ext) {
            if (model_name.size() < ext.size()) {
                continue;
            }
            auto suffix = model_name.substr(model_name.size() - ext.size());
            std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            std::string lower_ext = ext;
            std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (suffix == lower_ext) {
                return true;
            }
        }
        return false;
    };

    if (model_name.empty() ||
        model_name.find('/') != std::string::npos ||
        model_name.find('\\') != std::string::npos ||
        fs::u8path(model_name).has_root_path() ||
        ends_with_valid_ext()) {
        LOG_ERROR("%s must be a model name without path or extension: %s", label, model_name.c_str());
        return false;
    }

    fs::path model_dir_path = fs::u8path(model_dir);
    for (const auto& ext : valid_ext) {
        fs::path try_path = model_dir_path / fs::u8path(model_name + ext);
        if (fs::exists(try_path) && fs::is_regular_file(try_path)) {
            resolved_path = try_path.lexically_normal().u8string();
            return true;
        }
    }

    LOG_ERROR("can not find %s %s in %s", label, model_name.c_str(), model_dir_path.lexically_normal().u8string().c_str());
    return false;
}

bool SDGenerationParams::parse_image_preprocess_json(const std::string& json_str) {
    const auto value = json::parse(json_str, nullptr, false);
    std::string rules;
    if (value.is_string()) {
        rules = value.get<std::string>();
    } else if (value.is_array()) {
        for (const auto& item : value) {
            if (!item.is_string()) {
                LOG_ERROR("image_preprocess must contain rule strings");
                return false;
            }
            if (!rules.empty())
                rules += ";";
            rules += item.get<std::string>();
        }
    } else {
        LOG_ERROR("image_preprocess must be a string or array of strings");
        return false;
    }
    image_preprocess = std::move(rules);
    return true;
}

bool SDGenerationParams::parse_prompt_images_json(const std::string& json_str) {
    auto images = json::parse(json_str, nullptr, false);
    if (!images.is_object()) {
        return false;
    }
    std::map<std::string, SDImageOwner> decoded;
    for (auto it = images.begin(); it != images.end(); ++it) {
        if (it.key().empty() || it.key().find('>') != std::string::npos || !it.value().is_string()) {
            return false;
        }
        SDImageOwner image;
        if (!decode_base64_image(it.value().get<std::string>(), 3, 0, 0, image)) {
            return false;
        }
        decoded.emplace(it.key(), std::move(image));
    }
    prompt_images = std::move(decoded);
    return true;
}

template <typename Field>
static void load_json_input(const json& object, const char* key, Field& out) {
    if (!object.contains(key))
        return;
    const auto& value = object.at(key);
    if (value.is_null()) {
        out = Field{};
        return;
    }
    using T    = typename InputType<Field>::type;
    bool valid = false;
    if constexpr (std::is_same_v<T, std::string>)
        valid = value.is_string();
    else if constexpr (std::is_same_v<T, bool>)
        valid = value.is_boolean();
    else if constexpr (std::is_integral_v<T>) {
        valid = value.is_number_integer();
        if (valid) {
            if (value.is_number_unsigned())
                valid = value.get<uint64_t>() <= static_cast<uint64_t>(std::numeric_limits<T>::max());
            else {
                const auto integer = value.get<int64_t>();
                valid              = integer >= std::numeric_limits<T>::min() && integer <= std::numeric_limits<T>::max();
            }
        }
    } else if constexpr (std::is_floating_point_v<T>)
        valid = value.is_number();
    else
        valid = value.is_array();
    if (!valid)
        throw std::invalid_argument(std::string(key) + ": invalid value type or range");
    out = value.get<T>();
}

/** Populates generation params from a JSON payload (server APIs); returns false on any malformed field. */
bool SDGenerationParams::from_json_str(
    const std::string& json_str,
    const std::function<std::string(const std::string&)>& lora_path_resolver) try {
    json j;
    try {
        j = json::parse(json_str);
    } catch (...) {
        LOG_ERROR("json parse failed %s", json_str.c_str());
        return false;
    }

    if (j.contains("image_preprocess") && !parse_image_preprocess_json(j["image_preprocess"].dump()))
        return false;
    if (j.contains("prompt_images") && !parse_prompt_images_json(j["prompt_images"].dump()))
        return false;

    auto load_if_exists = [&](const char* key, auto& out) { load_json_input(j, key, out); };

    load_if_exists("prompt", prompt);
    load_if_exists("negative_prompt", negative_prompt);
    load_if_exists("ad_model", ad_model_path);
    load_if_exists("ad_prompt", ad_prompt);
    load_if_exists("ad_negative_prompt", ad_negative_prompt);
    load_if_exists("extra_ad_args", extra_ad_args);
    load_if_exists("cache_mode", cache_mode);
    load_if_exists("cache_option", cache_option);
    load_if_exists("scm_mask", scm_mask);
    load_if_exists("ref_image_args", ref_image_args);

    load_if_exists("clip_skip", clip_skip);
    load_if_exists("width", width);
    load_if_exists("height", height);
    load_if_exists("batch_count", batch_count);
    load_if_exists("qwen_image_layers", qwen_image_layers);
    load_if_exists("video_frames", video_frames);
    load_if_exists("fps", fps);
    load_if_exists("upscale_repeats", upscale_repeats);
    load_if_exists("seed", seed);

    load_if_exists("strength", strength);
    load_if_exists("control_strength", control_strength);
    load_if_exists("control_type", control_type);
    load_if_exists("ip_adapter_strength", ip_adapter_strength);
    load_if_exists("moe_boundary", moe_boundary);
    load_if_exists("vace_strength", vace_strength);

    load_if_exists("increase_ref_index", increase_ref_index);
    load_if_exists("embed_image_metadata", embed_image_metadata);

    if (j.contains("hires") && j["hires"].is_object()) {
        const json& hires_json = j["hires"];
        load_json_input(hires_json, "enabled", hires_enabled);
        load_json_input(hires_json, "upscaler", hires_upscaler);
        load_json_input(hires_json, "scale", hires_scale);
        load_json_input(hires_json, "target_width", hires_width);
        load_json_input(hires_json, "target_height", hires_height);
        load_json_input(hires_json, "steps", hires_steps);
        load_json_input(hires_json, "denoising_strength", hires_denoising_strength);
        load_json_input(hires_json, "custom_sigmas", hires_custom_sigmas);
        load_json_input(hires_json, "upscale_tile_size", hires_upscale_tile_size);
    }

    auto parse_sample_params_json = [&](const json& sample_json,
                                        SDSampleParams& target_params,
                                        std::string& target_extra_sample_args,
                                        std::optional<std::vector<int>>& target_skip_layers,
                                        std::optional<std::vector<float>>* target_custom_sigmas) {
        load_json_input(sample_json, "sample_steps", target_params.sample_steps);
        load_json_input(sample_json, "eta", target_params.eta);
        load_json_input(sample_json, "shifted_timestep", target_params.shifted_timestep);
        load_json_input(sample_json, "flow_shift", target_params.flow_shift);
        load_json_input(sample_json, "extra_sample_args", target_extra_sample_args);
        if (target_custom_sigmas)
            load_json_input(sample_json, "custom_sigmas", *target_custom_sigmas);
        load_json_input(sample_json, "sample_method", target_params.sample_method);
        load_json_input(sample_json, "scheduler", target_params.scheduler);
        if (sample_json.contains("guidance") && sample_json["guidance"].is_object()) {
            const json& guidance_json = sample_json["guidance"];
            load_json_input(guidance_json, "txt_cfg", target_params.guidance.txt_cfg);
            load_json_input(guidance_json, "img_cfg", target_params.guidance.img_cfg);
            load_json_input(guidance_json, "distilled_guidance", target_params.guidance.distilled_guidance);
            if (guidance_json.contains("slg") && guidance_json["slg"].is_object()) {
                const json& slg_json = guidance_json["slg"];
                load_json_input(slg_json, "layers", target_skip_layers);
                load_json_input(slg_json, "layer_start", target_params.guidance.slg.layer_start);
                load_json_input(slg_json, "layer_end", target_params.guidance.slg.layer_end);
                load_json_input(slg_json, "scale", target_params.guidance.slg.scale);
            }
        }
    };

    if (j.contains("sample_params") && j["sample_params"].is_object()) {
        parse_sample_params_json(j["sample_params"], sample_params, extra_sample_args, skip_layers, &custom_sigmas);
    }
    if (j.contains("high_noise_sample_params") && j["high_noise_sample_params"].is_object()) {
        parse_sample_params_json(j["high_noise_sample_params"],
                                 high_noise_sample_params,
                                 high_noise_extra_sample_args,
                                 high_noise_skip_layers,
                                 nullptr);
    }

    if (j.contains("vae_tiling_params") && j["vae_tiling_params"].is_object()) {
        const json& tiling_json = j["vae_tiling_params"];
        load_json_input(tiling_json, "enabled", vae_tiling_params.enabled);
        load_json_input(tiling_json, "temporal_tiling", vae_tiling_params.temporal_tiling);
        load_json_input(tiling_json, "tile_size_w", vae_tiling_params.tile_size_w);
        load_json_input(tiling_json, "tile_size_h", vae_tiling_params.tile_size_h);
        load_json_input(tiling_json, "target_overlap", vae_tiling_params.target_overlap);
        load_json_input(tiling_json, "rel_size_w", vae_tiling_params.rel_size_w);
        load_json_input(tiling_json, "rel_size_h", vae_tiling_params.rel_size_h);
        load_json_input(tiling_json, "extra_tiling_args", extra_tiling_args);
    }

    if (!parse_lora_json_field(j, lora_path_resolver, lora_map, high_noise_lora_map)) {
        LOG_ERROR("invalid lora");
        return false;
    }
    auto load_image = [&](const char* key, int channels, SDImageOwner& image) {
        if (!parse_image_json_field(j, key, channels, image)) {
            LOG_ERROR("invalid %s", key);
            return false;
        }
        return true;
    };
    if (!load_image("init_image", 0, init_image) ||
        !load_image("end_image", 3, end_image) ||
        !load_image("mask_image", 1, mask_image) ||
        !load_image("control_image", 3, control_image) ||
        !load_image("ip_adapter_image", 3, ip_adapter_image)) {
        return false;
    }
    if (!parse_image_array_json_field(j, "ref_images", 0, ref_images) ||
        !parse_image_array_json_field(j, "control_frames", 3, control_frames) ||
        !parse_audio_array_json_field(j, "ref_audios", ref_audios) ||
        !parse_audio_array_json_field(j, "ref_video_audios", ref_video_audios) ||
        !parse_video_array_json_field(j, "ref_videos", ref_videos)) {
        LOG_ERROR("invalid input image/audio/video array");
        return false;
    }

    return true;
} catch (const std::exception& error) {
    LOG_ERROR("invalid generation parameters: %s", error.what());
    return false;
}

bool SDGenerationParams::extract_and_remove_lora(const std::string& lora_model_dir) {
    if (lora_model_dir.empty()) {
        return true;
    }
    static const std::regex re(R"(<lora:([^:>]+):([^>]+)>)");
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt", ".ckpt"};
    std::smatch m;

    bool missing    = false;
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
        try {
            if (is_absolute_path(raw_path)) {
                final_path = fs::u8path(raw_path);
            } else {
                final_path = fs::u8path(lora_model_dir) / fs::u8path(raw_path);
            }
        } catch (const std::exception&) {  // not UTF-8, e.g. a prompt file in another encoding
            LOG_WARN("invalid lora path %s", raw_path.c_str());
            tmp     = m.suffix().str();
            prompt  = std::regex_replace(prompt, re, "", std::regex_constants::format_first_only);
            missing = true;
            continue;
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
                LOG_WARN("can not found lora %s", final_path.lexically_normal().u8string().c_str());
                tmp     = m.suffix().str();
                prompt  = std::regex_replace(prompt, re, "", std::regex_constants::format_first_only);
                missing = true;
                continue;
            }
        }

        const std::string key = final_path.lexically_normal().u8string();

        if (is_high_noise)
            high_noise_lora_map[key] += mul;
        else
            lora_map[key] += mul;

        prompt = std::regex_replace(prompt, re, "", std::regex_constants::format_first_only);

        tmp = m.suffix().str();
    }

    return !missing;
}

bool SDGenerationParams::width_and_height_are_set() const {
    return width.has_value() && height.has_value();
}

void SDGenerationParams::set_width_and_height_if_unset(int w, int h) {
    if (!width)
        width = w;
    if (!height)
        height = h;
}

int SDGenerationParams::get_requested_width() const {
    return width.value_or(0);
}

int SDGenerationParams::get_requested_height() const {
    return height.value_or(0);
}

bool SDGenerationParams::initialize_cache_params() {
    cache_params = SDCacheParams{};

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
        if (cache_mode == "disabled") {
            cache_params.mode = "disabled";
        } else if (cache_mode == "easycache") {
            cache_params.mode = "easycache";
        } else if (cache_mode == "ucache") {
            cache_params.mode = "ucache";
        } else if (cache_mode == "dbcache") {
            cache_params.mode = "dbcache";
        } else if (cache_mode == "taylorseer") {
            cache_params.mode = "taylorseer";
        } else if (cache_mode == "cache-dit") {
            cache_params.mode = "cache_dit";
        } else if (cache_mode == "spectrum") {
            cache_params.mode = "spectrum";
        } else {
            LOG_ERROR("error: invalid cache mode '%s'", cache_mode.c_str());
            return false;
        }
    }

    if (!cache_option.empty() && !parse_named_params(cache_option)) {
        return false;
    }

    if (cache_params.mode == "dbcache" ||
        cache_params.mode == "taylorseer" ||
        cache_params.mode == "cache_dit") {
        cache_params.scm_policy_dynamic = scm_policy_dynamic;
    }

    return true;
}

bool SDGenerationParams::resolve(const std::string& lora_model_dir, const std::string& hires_upscalers_dir, bool strict, bool validate_missing_loras) {
    vae_tiling_params.extra_tiling_args = extra_tiling_args.empty() ? nullptr : extra_tiling_args.c_str();

    if (!initialize_cache_params()) {
        return false;
    }

    if (strict && ((batch_count && (*batch_count < 1 || *batch_count > 8)) ||
                   (sample_params.sample_steps && (*sample_params.sample_steps < 1 || *sample_params.sample_steps > 100)))) {
        LOG_ERROR("server requests require batch_count in [1, 8] and sample_steps in [1, 100]");
        return false;
    }

    hires_upscaler_model_path.clear();
    resolved_hires_upscaler.clear();
    if (hires_upscaler) {
        static const std::map<std::string, std::string> names = {
            {"None", "none"}, {"Latent", "latent"}, {"Latent (nearest)", "latent_nearest"}, {"Latent (nearest-exact)", "latent_nearest_exact"}, {"Latent (antialiased)", "latent_antialiased"}, {"Latent (bicubic)", "latent_bicubic"}, {"Latent (bicubic antialiased)", "latent_bicubic_antialiased"}, {"Lanczos", "lanczos"}, {"Nearest", "nearest"}, {"Model", "model"}};
        const auto it = names.find(*hires_upscaler);
        if (it != names.end()) {
            resolved_hires_upscaler = it->second;
        } else if (std::any_of(names.begin(), names.end(), [&](const auto& name) { return name.second == *hires_upscaler; })) {
            resolved_hires_upscaler = *hires_upscaler;
        } else {
            static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt", ".pth"};
            if (hires_enabled.value_or(false) &&
                !resolve_model_file_from_dir(*hires_upscaler, hires_upscalers_dir, valid_ext, "hires upscaler", hires_upscaler_model_path))
                return false;
            resolved_hires_upscaler = "model";
        }
        if (resolved_hires_upscaler == "none")
            hires_enabled = false;
    }

    prompt_with_lora = prompt;
    if (!lora_model_dir.empty()) {
        if (!extract_and_remove_lora(lora_model_dir) && validate_missing_loras) {
            return false;
        }
    }
    return true;
}

/** Validates params for a generation mode; enforces mode-specific input rules. */
bool SDGenerationParams::validate(SDMode mode) {
    if (!image_preprocess.empty() && mode != IMG_GEN && mode != VID_GEN) {
        LOG_ERROR("--image-preprocess requires img_gen or vid_gen mode");
        return false;
    }
    if (ref_video_audio_paths.size() + ref_video_audios.size() > ref_video_paths.size() + ref_videos.size()) {
        LOG_ERROR("error: each --ref-video-audio needs a corresponding --ref-video");
        return false;
    }

    if (mode != VID_GEN && (!ref_video_paths.empty() || !ref_video_audio_paths.empty() || !ref_audio_paths.empty() ||
                            !ref_videos.empty() || !ref_video_audios.empty() || !ref_audios.empty())) {
        LOG_ERROR("error: reference video and audio inputs require vid_gen mode");
        return false;
    }

    if (upscale_repeats < 1) {
        return false;
    }

    if (mode == UPSCALE || mode == ADETAILER) {
        if (init_image_path.length() == 0) {
            LOG_ERROR("error: %s mode needs an init image (--init-img)\n",
                      mode == UPSCALE ? "upscale" : "adetailer");
            return false;
        }
    }

    if (mode == ADETAILER && ad_model_path.empty()) {
        LOG_ERROR("error: adetailer mode needs a detector model (--ad-model)\n");
        return false;
    }

    if (mode == IMG_GEN || mode == ADETAILER || mode == VID_GEN) {
        auto opt = mode == VID_GEN ? to_video_opt() : to_image_opt(mode == ADETAILER);
        if (!opt) {
            LOG_ERROR("%s", sd_get_last_error());
            return false;
        }
    }
    return true;
}

bool SDGenerationParams::resolve_and_validate(SDMode mode,
                                              const std::string& lora_model_dir,
                                              const std::string& hires_upscalers_dir,
                                              bool strict) {
    if (!resolve(lora_model_dir, hires_upscalers_dir, strict, mode == CONVERT)) {
        return false;
    }
    if (!validate(mode)) {
        return false;
    }
    return true;
}

static void set_sample_options(SDOptBuilder& opt, const std::string& prefix, const SDSampleParams& params, const std::optional<std::vector<int>>& layers, const std::optional<std::vector<float>>& sigmas, const std::string& extra_args) {
    if (!extra_args.empty())
        opt.set_string((prefix + "args").c_str(), extra_args);
    opt.set_float((prefix + "eta").c_str(), params.eta);
    opt.set_float((prefix + "guidance.distilled_guidance").c_str(), params.guidance.distilled_guidance);
    opt.set_float((prefix + "guidance.img_cfg").c_str(), params.guidance.img_cfg);
    opt.set_float((prefix + "guidance.slg.layer_end").c_str(), params.guidance.slg.layer_end);
    opt.set_float((prefix + "guidance.slg.layer_start").c_str(), params.guidance.slg.layer_start);
    if (layers)
        opt.set_ints((prefix + "guidance.slg.layers").c_str(), layers->data(), layers->size());
    opt.set_float((prefix + "guidance.slg.scale").c_str(), params.guidance.slg.scale);
    opt.set_float((prefix + "guidance.txt_cfg").c_str(), params.guidance.txt_cfg);
    opt.set_string((prefix + "method").c_str(), params.sample_method);
    if (prefix == "sample.") {
        opt.set_string("sample.scheduler", params.scheduler);
        opt.set_float("sample.flow_shift", params.flow_shift);
        opt.set_int("sample.shifted_timestep", params.shifted_timestep);
        if (sigmas)
            opt.set_floats((prefix + "sigmas").c_str(), sigmas->data(), sigmas->size());
    }
    opt.set_int((prefix + "steps").c_str(), params.sample_steps);
}

static void set_cache_options(SDOptBuilder& opt, const SDCacheParams& params) {
    opt.set_int("cache.bn_compute_blocks", params.Bn_compute_blocks);
    opt.set_float("cache.end_percent", params.end_percent);
    opt.set_float("cache.error_decay_rate", params.error_decay_rate);
    opt.set_int("cache.fn_compute_blocks", params.Fn_compute_blocks);
    opt.set_int("cache.max_cached_steps", params.max_cached_steps);
    opt.set_int("cache.max_continuous_cached_steps", params.max_continuous_cached_steps);
    opt.set_int("cache.max_warmup_steps", params.max_warmup_steps);
    opt.set_string("cache.mode", params.mode);
    opt.set_bool("cache.reset_error_on_compute", params.reset_error_on_compute);
    opt.set_float("cache.residual_diff_threshold", params.residual_diff_threshold);
    opt.set_float("cache.reuse_threshold", params.reuse_threshold);
    opt.set_string("cache.scm_mask", params.scm_mask);
    opt.set_bool("cache.scm_policy_dynamic", params.scm_policy_dynamic);
    opt.set_float("cache.spectrum_flex_window", params.spectrum_flex_window);
    opt.set_float("cache.spectrum_lam", params.spectrum_lam);
    opt.set_int("cache.spectrum_m", params.spectrum_m);
    opt.set_float("cache.spectrum_stop_percent", params.spectrum_stop_percent);
    opt.set_float("cache.spectrum_w", params.spectrum_w);
    opt.set_int("cache.spectrum_warmup_steps", params.spectrum_warmup_steps);
    opt.set_int("cache.spectrum_window_size", params.spectrum_window_size);
    opt.set_float("cache.start_percent", params.start_percent);
    opt.set_int("cache.taylorseer_n_derivatives", params.taylorseer_n_derivatives);
    opt.set_int("cache.taylorseer_skip_interval", params.taylorseer_skip_interval);
    opt.set_bool("cache.use_relative_threshold", params.use_relative_threshold);
}

static void set_tiling_options(SDOptBuilder& opt, const SDTilingParams& params) {
    opt.set_bool("vae_tiling.enabled", params.enabled);
    opt.set_string("vae_tiling.extra_tiling_args", params.extra_tiling_args);
    opt.set_float("vae_tiling.rel_size_h", params.rel_size_h);
    opt.set_float("vae_tiling.rel_size_w", params.rel_size_w);
    opt.set_float("vae_tiling.target_overlap", params.target_overlap);
    opt.set_bool("vae_tiling.temporal_tiling", params.temporal_tiling);
    opt.set_int("vae_tiling.tile_size_h", params.tile_size_h);
    opt.set_int("vae_tiling.tile_size_w", params.tile_size_w);
}

SDOptPtr SDGenerationParams::to_image_opt(bool adetail) {
    lora_vec.clear();
    lora_vec.reserve(lora_map.size() + high_noise_lora_map.size());
    for (const auto& kv : lora_map) {
        lora_vec.push_back({false, kv.second, kv.first.c_str()});
    }
    for (const auto& kv : high_noise_lora_map) {
        lora_vec.push_back({true, kv.second, kv.first.c_str()});
    }

    ref_image_views.clear();
    ref_image_views.reserve(ref_images.size());
    for (auto& ref_image : ref_images) {
        ref_image_views.push_back(ref_image.get());
    }

    pm_id_image_views.clear();
    pm_id_image_views.reserve(pm_id_images.size());
    for (auto& image : pm_id_images) {
        pm_id_image_views.push_back(image.get());
    }

    vae_tiling_params.extra_tiling_args = extra_tiling_args.empty() ? nullptr : extra_tiling_args.c_str();
    cache_params.scm_mask               = scm_mask.empty() ? nullptr : scm_mask.c_str();

    std::string effective_ref_image_args = ref_image_args;
    if (!adetail && increase_ref_index) {
        if (!effective_ref_image_args.empty())
            effective_ref_image_args += ",";
        effective_ref_image_args += "ref_index_mode=increase";
    }

    SDOptBuilder opt(adetail ? sd_adetail_opt_create() : sd_img_gen_opt_create());
    opt.set_string("prompt", prompt);
    opt.set_string("negative_prompt", negative_prompt);
    opt.set_int("clip_skip", clip_skip);
    if (!adetail)
        opt.set_image("init_image", init_image.get());
    if (!adetail)
        opt.set_string("ref_image_args", effective_ref_image_args);
    if (!adetail)
        opt.set_image("mask_image", mask_image.get());
    opt.set_int("width", width);
    opt.set_int("height", height);
    opt.set_float("strength", strength);
    opt.set_int("seed", seed);
    if (!adetail)
        opt.set_int("batch_count", batch_count);
    if (!adetail)
        opt.set_int("qwen_image_layers", qwen_image_layers);
    if (!adetail)
        opt.set_image("control_image", control_image.get());
    if (!adetail)
        opt.set_float("control_strength", control_strength);
    if (!adetail)
        opt.set_string("control_type", control_type);
    if (!adetail)
        opt.set_image("ip_adapter_image", ip_adapter_image.get());
    if (!adetail)
        opt.set_float("ip_adapter_strength", ip_adapter_strength);
    if (!adetail)
        opt.set_bool("hires.enabled", hires_enabled);
    if (!adetail)
        opt.set_string("hires.upscaler", resolved_hires_upscaler.empty() ? hires_upscaler : std::optional<std::string>(resolved_hires_upscaler));
    if (!adetail)
        opt.set_string("hires.model_path", hires_upscaler_model_path);
    if (!adetail)
        opt.set_float("hires.scale", hires_scale);
    if (!adetail)
        opt.set_int("hires.target_width", hires_width);
    if (!adetail)
        opt.set_int("hires.target_height", hires_height);
    if (!adetail)
        opt.set_int("hires.steps", hires_steps);
    if (!adetail)
        opt.set_float("hires.denoising_strength", hires_denoising_strength);
    if (!adetail)
        opt.set_int("hires.upscale_tile_size", hires_upscale_tile_size);
    if (circular)
        opt.set_bool("circular_x", true);
    else
        opt.set_bool("circular_x", circular_x);
    if (circular)
        opt.set_bool("circular_y", true);
    else
        opt.set_bool("circular_y", circular_y);
    opt.set_loras("loras", lora_vec.data(), lora_vec.size());
    if (!adetail)
        opt.set_images("ref_images", ref_image_views.data(), ref_image_views.size());
    if (!adetail)
        opt.set_images("photomaker.id_images", pm_id_image_views.data(), pm_id_image_views.size());
    if (!adetail)
        opt.set_string("photomaker.id_embed_path", pm_id_embed_path);
    if (!adetail)
        opt.set_float("photomaker.style_strength", pm_style_strength);
    if (!adetail)
        opt.set_string("pulid.id_embedding_path", pulid_id_embedding_path);
    if (!adetail)
        opt.set_float("pulid.id_weight", pulid_id_weight);
    if (!adetail)
        opt.set_floats("hires.sigmas", hires_custom_sigmas);
    if (!adetail)
        opt.set_string("image_preprocess.rules", image_preprocess);
    set_sample_options(opt, "sample.", sample_params, skip_layers, custom_sigmas, extra_sample_args);
    set_cache_options(opt, cache_params);
    set_tiling_options(opt, vae_tiling_params);
    if (adetail) {
        if (!ad_prompt.empty())
            opt.set_string("adetail.prompt", ad_prompt);
        if (!ad_negative_prompt.empty())
            opt.set_string("adetail.negative_prompt", ad_negative_prompt);
        opt.set_string("adetail.args", extra_ad_args);
    }
    return opt.finish();
}

SDOptPtr SDGenerationParams::to_video_opt() {
    lora_vec.clear();
    lora_vec.reserve(lora_map.size() + high_noise_lora_map.size());
    for (const auto& kv : lora_map) {
        lora_vec.push_back({false, kv.second, kv.first.c_str()});
    }
    for (const auto& kv : high_noise_lora_map) {
        lora_vec.push_back({true, kv.second, kv.first.c_str()});
    }

    control_frame_views.clear();
    control_frame_views.reserve(control_frames.size());
    for (auto& frame : control_frames) {
        control_frame_views.push_back(frame.get());
    }

    ref_image_views.clear();
    ref_image_views.reserve(ref_images.size());
    for (auto& image : ref_images) {
        ref_image_views.push_back(image.get());
    }

    ref_video_frame_views.clear();
    ref_video_frame_views.resize(ref_videos.size());
    ref_video_views.clear();
    ref_video_views.reserve(ref_videos.size());
    for (size_t i = 0; i < ref_videos.size(); ++i) {
        auto& frame_views = ref_video_frame_views[i];
        frame_views.reserve(ref_videos[i].size());
        for (auto& frame : ref_videos[i]) {
            frame_views.push_back(frame.get());
        }
        SDAudio audio = i < ref_video_audios.size() ? ref_video_audios[i].get() : SDAudio{};
        ref_video_views.push_back({frame_views.empty() ? nullptr : frame_views.data(),
                                   static_cast<int>(frame_views.size()),
                                   24,
                                   audio});
    }

    ref_audio_views.clear();
    ref_audio_views.reserve(ref_audios.size());
    for (auto& audio : ref_audios) {
        ref_audio_views.push_back(audio.get());
    }

    vae_tiling_params.extra_tiling_args = extra_tiling_args.empty() ? nullptr : extra_tiling_args.c_str();
    cache_params.scm_mask               = scm_mask.empty() ? nullptr : scm_mask.c_str();

    SDOptBuilder opt(sd_vid_gen_opt_create());
    opt.set_string("prompt", prompt);
    opt.set_string("negative_prompt", negative_prompt);
    opt.set_int("clip_skip", clip_skip);
    opt.set_image("init_image", init_image.get());
    opt.set_image("end_image", end_image.get());
    opt.set_int("width", width);
    opt.set_int("height", height);
    opt.set_float("moe_boundary", moe_boundary);
    opt.set_float("strength", strength);
    opt.set_int("seed", seed);
    opt.set_int("video_frames", video_frames);
    opt.set_int("fps", fps);
    opt.set_float("vace_strength", vace_strength);
    opt.set_bool("hires.enabled", hires_enabled);
    opt.set_string("hires.upscaler", resolved_hires_upscaler.empty() ? hires_upscaler : std::optional<std::string>(resolved_hires_upscaler));
    opt.set_string("hires.model_path", hires_upscaler_model_path);
    opt.set_float("hires.scale", hires_scale);
    opt.set_int("hires.target_width", hires_width);
    opt.set_int("hires.target_height", hires_height);
    opt.set_int("hires.steps", hires_steps);
    opt.set_float("hires.denoising_strength", hires_denoising_strength);
    opt.set_int("hires.upscale_tile_size", hires_upscale_tile_size);
    if (circular)
        opt.set_bool("circular_x", true);
    else
        opt.set_bool("circular_x", circular_x);
    if (circular)
        opt.set_bool("circular_y", true);
    else
        opt.set_bool("circular_y", circular_y);
    opt.set_string("ref_image_args", ref_image_args);
    opt.set_loras("loras", lora_vec.data(), lora_vec.size());
    opt.set_images("ref_images", ref_image_views.data(), ref_image_views.size());
    opt.set_videos("ref_videos", ref_video_views.data(), ref_video_views.size());
    opt.set_audios("ref_audios", ref_audio_views.data(), ref_audio_views.size());
    opt.set_frames("control_frames", control_frame_views.data(), control_frame_views.size());
    opt.set_floats("hires.sigmas", hires_custom_sigmas);
    opt.set_string("image_preprocess.rules", image_preprocess);
    opt.set_string("sample.scheduler", high_noise_sample_params.scheduler);
    opt.set_float("sample.flow_shift", high_noise_sample_params.flow_shift);
    opt.set_int("sample.shifted_timestep", high_noise_sample_params.shifted_timestep);
    set_sample_options(opt, "sample.", sample_params, skip_layers, custom_sigmas, extra_sample_args);
    set_sample_options(opt, "high_noise_sample.", high_noise_sample_params, high_noise_skip_layers, {}, high_noise_extra_sample_args);
    set_cache_options(opt, cache_params);
    set_tiling_options(opt, vae_tiling_params);
    return opt.finish();
}

std::string SDGenerationParams::to_string() const {
    const std::string sample_params_str            = sample_params.sample_method.value_or("unset") + ", " + sample_params.scheduler.value_or("unset") + ", steps=" + (sample_params.sample_steps ? std::to_string(*sample_params.sample_steps) : "unset");
    const std::string high_noise_sample_params_str = high_noise_sample_params.sample_method.value_or("unset") + ", " + high_noise_sample_params.scheduler.value_or("unset") + ", steps=" + (high_noise_sample_params.sample_steps ? std::to_string(*high_noise_sample_params.sample_steps) : "unset");

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
        << "  ad_model_path: \"" << ad_model_path << "\",\n"
        << "  ad_prompt: \"" << ad_prompt << "\",\n"
        << "  ad_negative_prompt: \"" << ad_negative_prompt << "\",\n"
        << "  extra_ad_args: \"" << extra_ad_args << "\",\n"
        << "  clip_skip: " << clip_skip << ",\n"
        << "  width: " << width << ",\n"
        << "  height: " << height << ",\n"
        << "  batch_count: " << batch_count << ",\n"
        << "  qwen_image_layers: " << qwen_image_layers << ",\n"
        << "  init_image_path: \"" << init_image_path << "\",\n"
        << "  end_image_path: \"" << end_image_path << "\",\n"
        << "  mask_image_path: \"" << mask_image_path << "\",\n"
        << "  control_image_path: \"" << control_image_path << "\",\n"
        << "  ref_image_paths: " << vec_str_to_string(ref_image_paths) << ",\n"
        << "  ref_video_paths: " << vec_str_to_string(ref_video_paths) << ",\n"
        << "  ref_video_audio_paths: " << vec_str_to_string(ref_video_audio_paths) << ",\n"
        << "  ref_audio_paths: " << vec_str_to_string(ref_audio_paths) << ",\n"
        << "  control_video_path: \"" << control_video_path << "\",\n"
        << "  image_preprocess: " << image_preprocess << ",\n"
        << "  ref_image_args: " << ref_image_args << ",\n"
        << "  increase_ref_index: " << (increase_ref_index ? "true" : "false") << ",\n"
        << "  pm_id_images_dir: \"" << pm_id_images_dir << "\",\n"
        << "  pm_id_embed_path: \"" << pm_id_embed_path << "\",\n"
        << "  pm_style_strength: " << pm_style_strength << ",\n"
        << "  skip_layers: " << (skip_layers ? vec_to_string(*skip_layers) : "unset") << ",\n"
        << "  sample_params: " << SAFE_STR(sample_params_str.c_str()) << ",\n"
        << "  high_noise_skip_layers: " << (high_noise_skip_layers ? vec_to_string(*high_noise_skip_layers) : "unset") << ",\n"
        << "  high_noise_sample_params: " << SAFE_STR(high_noise_sample_params_str.c_str()) << ",\n"
        << "  custom_sigmas: " << (custom_sigmas ? vec_to_string(*custom_sigmas) : "unset") << ",\n"
        << "  cache_mode: \"" << cache_mode << "\",\n"
        << "  cache_option: \"" << cache_option << "\",\n"
        << "  cache: "
        << cache_params.mode
        << " (threshold=" << cache_params.reuse_threshold
        << ", start=" << cache_params.start_percent
        << ", end=" << cache_params.end_percent << "),\n"
        << "  moe_boundary: " << moe_boundary << ",\n"
        << "  video_frames: " << video_frames << ",\n"
        << "  fps: " << fps << ",\n"
        << "  vace_strength: " << vace_strength << ",\n"
        << "  strength: " << strength << ",\n"
        << "  control_strength: " << control_strength << ",\n"
        << "  control_type: \"" << control_type << "\",\n"
        << "  seed: " << seed << ",\n"
        << "  upscale_repeats: " << upscale_repeats << ",\n"
        << "  upscale_tile_size: " << upscale_tile_size << ",\n"
        << "  hires: { enabled: " << hires_enabled
        << ", upscaler: \"" << hires_upscaler << "\""
        << ", model_path: \"" << hires_upscaler_model_path << "\""
        << ", scale: " << hires_scale
        << ", target_width: " << hires_width
        << ", target_height: " << hires_height
        << ", steps: " << hires_steps
        << ", denoising_strength: " << hires_denoising_strength
        << ", custom_sigmas: " << (hires_custom_sigmas ? vec_to_string(*hires_custom_sigmas) : "unset")
        << ", upscale_tile_size: " << hires_upscale_tile_size << " },\n"
        << "  vae_tiling_params: { "
        << vae_tiling_params.enabled << ", "
        << vae_tiling_params.temporal_tiling << ", "
        << vae_tiling_params.tile_size_w << ", "
        << vae_tiling_params.tile_size_h << ", "
        << vae_tiling_params.target_overlap << ", "
        << vae_tiling_params.rel_size_w << ", "
        << vae_tiling_params.rel_size_h << ", "
        << "\"" << extra_tiling_args << "\" },\n"
        << "}";
    return oss.str();
}

std::string version_string() {
    return std::string("stable-diffusion.cpp version ") + sd_version() + ", commit " + sd_commit();
}
