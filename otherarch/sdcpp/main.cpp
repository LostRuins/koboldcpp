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

#include "common/common.hpp"

#include "avi_writer.h"

const char* previews_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

std::regex format_specifier_regex("(?:[^%]|^)(?:%%)*(%\\d{0,3}d)");

struct SDCliParams {
    SDMode mode             = IMG_GEN;
    std::string output_path = "output.png";
    int output_begin_idx    = -1;

    bool verbose          = false;
    bool canny_preprocess = false;
    bool convert_name     = false;

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
             "path to write result image to. you can use printf-style %d format specifiers for image sequences (default: ./output.png) (eg. output_%03d.png)",
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
            {"",
             "--output-begin-idx",
             "starting index for output image sequence, must be non-negative (default 0 if specified %d in output path, 1 otherwise)",
             &output_begin_idx},
        };

        options.bool_options = {
            {"",
             "--canny",
             "apply canny preprocessor (edge detection)",
             true, &canny_preprocess},
            {"",
             "--convert-name",
             "convert tensor name (for convert mode)",
             true, &convert_name},
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
                    LOG_ERROR("error: invalid mode %s, must be one of [%s]\n",
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
            int preview_found   = -1;
            for (int m = 0; m < PREVIEW_COUNT; m++) {
                if (!strcmp(preview, previews_str[m])) {
                    preview_found = m;
                }
            }
            if (preview_found == -1) {
                LOG_ERROR("error: preview method %s", preview);
                return -1;
            }
            preview_method = (preview_t)preview_found;
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
            LOG_ERROR("error: the following arguments are required: output_path");
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
            << "  convert_name: " << (convert_name ? "true" : "false") << ",\n"
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

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
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

    if (!cli_params.process_and_check() ||
        !ctx_params.process_and_check(cli_params.mode) ||
        !gen_params.process_and_check(cli_params.mode, ctx_params.lora_model_dir)) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

std::string get_image_params(const SDCliParams& cli_params, const SDContextParams& ctx_params, const SDGenerationParams& gen_params, int64_t seed) {
    std::string parameter_string = gen_params.prompt_with_lora + "\n";
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

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDCliParams* cli_params = (SDCliParams*)data;
    log_print(level, log, cli_params->verbose, cli_params->color);
}

bool load_images_from_dir(const std::string dir,
                          std::vector<sd_image_t>& images,
                          int expected_width  = 0,
                          int expected_height = 0,
                          int max_image_num   = 0,
                          bool verbose        = false) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        LOG_ERROR("'%s' is not a valid directory\n", dir.c_str());
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
            LOG_DEBUG("load image %zu from '%s'", images.size(), path.c_str());
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height, expected_width, expected_height);
            if (image_buffer == nullptr) {
                LOG_ERROR("load image from '%s' failed", path.c_str());
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

std::string format_frame_idx(std::string pattern, int frame_idx) {
    std::smatch match;
    std::string result = pattern;
    while (std::regex_search(result, match, format_specifier_regex)) {
        std::string specifier = match.str(1);
        char buffer[32];
        snprintf(buffer, sizeof(buffer), specifier.c_str(), frame_idx);
        result.replace(match.position(1), match.length(1), buffer);
    }

    // Then replace all '%%' with '%'
    size_t pos = 0;
    while ((pos = result.find("%%", pos)) != std::string::npos) {
        result.replace(pos, 2, "%");
        pos += 1;
    }
    return result;
}

bool save_results(const SDCliParams& cli_params,
                  const SDContextParams& ctx_params,
                  const SDGenerationParams& gen_params,
                  sd_image_t* results,
                  int num_results) {
    if (results == nullptr || num_results <= 0) {
        return false;
    }

    namespace fs      = std::filesystem;
    fs::path out_path = cli_params.output_path;

    if (!out_path.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
        if (ec) {
            LOG_ERROR("failed to create directory '%s': %s",
                      out_path.parent_path().string().c_str(), ec.message().c_str());
            return false;
        }
    }

    fs::path base_path = out_path;
    fs::path ext       = out_path.has_extension() ? out_path.extension() : fs::path{};
    if (!ext.empty())
        base_path.replace_extension();

    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    bool is_jpg = (ext_lower == ".jpg" || ext_lower == ".jpeg" || ext_lower == ".jpe");

    int output_begin_idx = cli_params.output_begin_idx;
    if (output_begin_idx < 0) {
        output_begin_idx = 0;
    }

    auto write_image = [&](const fs::path& path, int idx) {
        const sd_image_t& img = results[idx];
        if (!img.data)
            return;

        std::string params = get_image_params(cli_params, ctx_params, gen_params, gen_params.seed + idx);
        int ok             = 0;
        if (is_jpg) {
            ok = stbi_write_jpg(path.string().c_str(), img.width, img.height, img.channel, img.data, 90, params.c_str());
        } else {
            ok = stbi_write_png(path.string().c_str(), img.width, img.height, img.channel, img.data, 0, params.c_str());
        }
        LOG_INFO("save result image %d to '%s' (%s)", idx, path.string().c_str(), ok ? "success" : "failure");
    };

    if (std::regex_search(cli_params.output_path, format_specifier_regex)) {
        if (!is_jpg && ext_lower != ".png")
            ext = ".png";
        fs::path pattern = base_path;
        pattern += ext;

        for (int i = 0; i < num_results; ++i) {
            fs::path img_path = format_frame_idx(pattern.string(), output_begin_idx + i);
            write_image(img_path, i);
        }
        return true;
    }

    if (cli_params.mode == VID_GEN && num_results > 1) {
        if (ext_lower != ".avi")
            ext = ".avi";
        fs::path video_path = base_path;
        video_path += ext;
        create_mjpg_avi_from_sd_images(video_path.string().c_str(), results, num_results, gen_params.fps);
        LOG_INFO("save result MJPG AVI video to '%s'", video_path.string().c_str());
        return true;
    }

    if (!is_jpg && ext_lower != ".png")
        ext = ".png";

    for (int i = 0; i < num_results; ++i) {
        fs::path img_path = base_path;
        if (num_results > 1) {
            img_path += "_" + std::to_string(output_begin_idx + i);
        }
        img_path += ext;
        write_image(img_path, i);
    }

    return true;
}

int main(int argc, const char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

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
    log_verbose = cli_params.verbose;
    log_color   = cli_params.color;
    sd_set_preview_callback(step_callback,
                            cli_params.preview_method,
                            cli_params.preview_interval,
                            !cli_params.preview_noisy,
                            cli_params.preview_noisy,
                            (void*)&cli_params);

    LOG_DEBUG("version: %s", version_string().c_str());
    LOG_DEBUG("%s", sd_get_system_info());
    LOG_DEBUG("%s", cli_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", gen_params.to_string().c_str());

    if (cli_params.mode == CONVERT) {
        bool success = convert(ctx_params.model_path.c_str(),
                               ctx_params.vae_path.c_str(),
                               cli_params.output_path.c_str(),
                               ctx_params.wtype,
                               ctx_params.tensor_type_rules.c_str(),
                               cli_params.convert_name);
        if (!success) {
            LOG_ERROR("convert '%s'/'%s' to '%s' failed",
                      ctx_params.model_path.c_str(),
                      ctx_params.vae_path.c_str(),
                      cli_params.output_path.c_str());
            return 1;
        } else {
            LOG_INFO("convert '%s'/'%s' to '%s' success",
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
        init_image.data = load_image_from_file(gen_params.init_image_path.c_str(), width, height, gen_params.width, gen_params.height);
        if (init_image.data == nullptr) {
            LOG_ERROR("load image from '%s' failed", gen_params.init_image_path.c_str());
            release_all_resources();
            return 1;
        }
    }

    if (gen_params.end_image_path.size() > 0) {
        vae_decode_only = false;

        int width      = 0;
        int height     = 0;
        end_image.data = load_image_from_file(gen_params.end_image_path.c_str(), width, height, gen_params.width, gen_params.height);
        if (end_image.data == nullptr) {
            LOG_ERROR("load image from '%s' failed", gen_params.end_image_path.c_str());
            release_all_resources();
            return 1;
        }
    }

    if (gen_params.mask_image_path.size() > 0) {
        int c           = 0;
        int width       = 0;
        int height      = 0;
        mask_image.data = load_image_from_file(gen_params.mask_image_path.c_str(), width, height, gen_params.width, gen_params.height, 1);
        if (mask_image.data == nullptr) {
            LOG_ERROR("load image from '%s' failed", gen_params.mask_image_path.c_str());
            release_all_resources();
            return 1;
        }
    } else {
        mask_image.data = (uint8_t*)malloc(gen_params.width * gen_params.height);
        memset(mask_image.data, 255, gen_params.width * gen_params.height);
        if (mask_image.data == nullptr) {
            LOG_ERROR("malloc mask image failed");
            release_all_resources();
            return 1;
        }
    }

    if (gen_params.control_image_path.size() > 0) {
        int width          = 0;
        int height         = 0;
        control_image.data = load_image_from_file(gen_params.control_image_path.c_str(), width, height, gen_params.width, gen_params.height);
        if (control_image.data == nullptr) {
            LOG_ERROR("load image from '%s' failed", gen_params.control_image_path.c_str());
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
            uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height);
            if (image_buffer == nullptr) {
                LOG_ERROR("load image from '%s' failed", path.c_str());
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
            LOG_INFO("failed to allocate results array");
            release_all_resources();
            return 1;
        }

        results[0]      = init_image;
        init_image.data = nullptr;
    } else {
        sd_ctx_t* sd_ctx = new_sd_ctx(&sd_ctx_params);

        if (sd_ctx == nullptr) {
            LOG_INFO("new_sd_ctx_t failed");
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
            gen_params.sample_params.scheduler = sd_get_default_scheduler(sd_ctx, gen_params.sample_params.sample_method);
        }

        if (cli_params.mode == IMG_GEN) {
            sd_img_gen_params_t img_gen_params = {
                gen_params.lora_vec.data(),
                static_cast<uint32_t>(gen_params.lora_vec.size()),
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
                gen_params.cache_params,
            };

            results     = generate_image(sd_ctx, &img_gen_params);
            num_results = gen_params.batch_count;
        } else if (cli_params.mode == VID_GEN) {
            sd_vid_gen_params_t vid_gen_params = {
                gen_params.lora_vec.data(),
                static_cast<uint32_t>(gen_params.lora_vec.size()),
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
                ctx_params.vae_tiling_params,
                gen_params.cache_params,
            };

            results = generate_video(sd_ctx, &vid_gen_params, &num_results);
        }

        if (results == nullptr) {
            LOG_ERROR("generate failed");
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
                                                        ctx_params.n_threads,
                                                        gen_params.upscale_tile_size);

        if (upscaler_ctx == nullptr) {
            LOG_ERROR("new_upscaler_ctx failed");
        } else {
            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                sd_image_t current_image = results[i];
                for (int u = 0; u < gen_params.upscale_repeats; ++u) {
                    sd_image_t upscaled_image = upscale(upscaler_ctx, current_image, upscale_factor);
                    if (upscaled_image.data == nullptr) {
                        LOG_ERROR("upscale failed");
                        break;
                    }
                    free(current_image.data);
                    current_image = upscaled_image;
                }
                results[i] = current_image;  // Set the final upscaled image as the result
            }
        }
    }

    if (!save_results(cli_params, ctx_params, gen_params, results, num_results)) {
        return 1;
    }

    for (int i = 0; i < num_results; i++) {
        free(results[i].data);
        results[i].data = nullptr;
    }
    free(results);

    release_all_resources();

    return 0;
}
