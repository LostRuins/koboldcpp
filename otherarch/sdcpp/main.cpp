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

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "image_metadata.h"

namespace fs = std::filesystem;

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
    std::string image_path;
    std::string metadata_format = "text";

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
    bool metadata_raw        = false;
    bool metadata_brief      = false;
    bool metadata_all        = false;

    bool normal_exit = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-o",
             "--output",
             "path to write result image to. you can use printf-style %d format specifiers for image sequences (default: ./output.png) (eg. output_%03d.png). Single-file video outputs support .avi, .webm, and animated .webp",
             &output_path},
            {"",
             "--image",
             "path to the image to inspect (for metadata mode)",
             &image_path},
            {"",
             "--metadata-format",
             "metadata output format, one of [text, json] (default: text)",
             &metadata_format},
            {"",
             "--preview-path",
             "path to write preview image to (default: ./preview.png). Multi-frame previews support .avi, .webm, and animated .webp",
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
            {"",
             "--metadata-raw",
             "include raw hex previews for unparsed metadata payloads",
             true, &metadata_raw},
            {"",
             "--metadata-brief",
             "truncate long metadata text values in text output",
             true, &metadata_brief},
            {"",
             "--metadata-all",
             "include structural/container entries such as IHDR, IDAT, and non-metadata JPEG segments",
             true, &metadata_all},

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
             "run mode, one of [img_gen, vid_gen, upscale, convert, metadata], default: img_gen",
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

    bool resolve() {
        if (mode == CONVERT) {
            if (output_path == "output.png") {
                output_path = "output.gguf";
            }
        }
        return true;
    }

    bool validate() {
        if (mode != METADATA) {
            if (output_path.length() == 0) {
                LOG_ERROR("error: the following arguments are required: output_path");
                return false;
            }
        } else {
            if (image_path.empty()) {
                LOG_ERROR("error: metadata mode needs an image path (--image)");
                return false;
            }
            if (metadata_format != "text" && metadata_format != "json") {
                LOG_ERROR("error: invalid metadata format %s, must be one of [text, json]",
                          metadata_format.c_str());
                return false;
            }
        }
        return true;
    }

    bool resolve_and_validate() {
        if (!resolve()) {
            return false;
        }
        if (!validate()) {
            return false;
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDCliParams {\n"
            << "  mode: " << modes_str[mode] << ",\n"
            << "  output_path: \"" << output_path << "\",\n"
            << "  image_path: \"" << image_path << "\",\n"
            << "  metadata_format: \"" << metadata_format << "\",\n"
            << "  verbose: " << (verbose ? "true" : "false") << ",\n"
            << "  color: " << (color ? "true" : "false") << ",\n"
            << "  canny_preprocess: " << (canny_preprocess ? "true" : "false") << ",\n"
            << "  convert_name: " << (convert_name ? "true" : "false") << ",\n"
            << "  preview_method: " << previews_str[preview_method] << ",\n"
            << "  preview_interval: " << preview_interval << ",\n"
            << "  preview_path: \"" << preview_path << "\",\n"
            << "  preview_fps: " << preview_fps << ",\n"
            << "  taesd_preview: " << (taesd_preview ? "true" : "false") << ",\n"
            << "  preview_noisy: " << (preview_noisy ? "true" : "false") << ",\n"
            << "  metadata_raw: " << (metadata_raw ? "true" : "false") << ",\n"
            << "  metadata_brief: " << (metadata_brief ? "true" : "false") << ",\n"
            << "  metadata_all: " << (metadata_all ? "true" : "false") << "\n"
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

    bool valid = cli_params.resolve_and_validate();
    if (valid && cli_params.mode != METADATA) {
        valid = ctx_params.resolve_and_validate(cli_params.mode) &&
                gen_params.resolve_and_validate(cli_params.mode,
                                                ctx_params.lora_model_dir,
                                                ctx_params.hires_upscalers_dir);
    }

    if (!valid) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDCliParams* cli_params = (SDCliParams*)data;
    log_print(level, log, cli_params->verbose, cli_params->color);
}

bool load_images_from_dir(const std::string dir,
                          std::vector<SDImageOwner>& images,
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

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            LOG_DEBUG("load image %zu from '%s'", images.size(), path.c_str());
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height, expected_width, expected_height);
            if (image_buffer == nullptr) {
                LOG_ERROR("load image from '%s' failed", path.c_str());
                return false;
            }

            images.emplace_back(sd_image_t{(uint32_t)width,
                                           (uint32_t)height,
                                           3,
                                           image_buffer});

            if (max_image_num > 0 && static_cast<int>(images.size()) >= max_image_num) {
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
        if (!write_image_to_file(cli_params->preview_path,
                                 image->data,
                                 image->width,
                                 image->height,
                                 image->channel)) {
            LOG_ERROR("save preview image to '%s' failed", cli_params->preview_path.c_str());
        }
    } else {
        if (create_video_from_sd_images(cli_params->preview_path.c_str(), image, frame_count, cli_params->preview_fps) != 0) {
            LOG_ERROR("save preview video to '%s' failed", cli_params->preview_path.c_str());
        }
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

    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    const EncodedImageFormat output_format = encoded_image_format_from_path(out_path.string());
    if (!ext.empty()) {
        if (output_format == EncodedImageFormat::JPEG ||
            output_format == EncodedImageFormat::PNG ||
            output_format == EncodedImageFormat::WEBP ||
            ext_lower == ".avi" ||
            ext_lower == ".webm") {
            base_path.replace_extension();
        }
    }

    int output_begin_idx = cli_params.output_begin_idx;
    if (output_begin_idx < 0) {
        output_begin_idx = 0;
    }

    auto write_image = [&](const fs::path& path, int idx) {
        const sd_image_t& img = results[idx];
        if (!img.data)
            return false;

        const int64_t metadata_seed = cli_params.mode == VID_GEN ? gen_params.seed : gen_params.seed + idx;
        std::string params          = gen_params.embed_image_metadata
                                          ? get_image_params(ctx_params, gen_params, metadata_seed, cli_params.mode)
                                          : "";
        const bool ok               = write_image_to_file(path.string(), img.data, img.width, img.height, img.channel, params, 90);
        LOG_INFO("save result image %d to '%s' (%s)", idx, path.string().c_str(), ok ? "success" : "failure");
        return ok;
    };

    int sucessful_reults = 0;

    if (std::regex_search(cli_params.output_path, format_specifier_regex)) {
        if (output_format == EncodedImageFormat::UNKNOWN)
            ext = ".png";
        fs::path pattern = base_path;
        pattern += ext;

        for (int i = 0; i < num_results; ++i) {
            fs::path img_path = format_frame_idx(pattern.string(), output_begin_idx + i);
            if (write_image(img_path, i)) {
                sucessful_reults++;
            }
        }
        LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
        return sucessful_reults != 0;
    }

    if (cli_params.mode == VID_GEN && num_results > 1) {
        if (ext_lower != ".avi" && ext_lower != ".webp" && ext_lower != ".webm")
            ext = ".avi";
        fs::path video_path = base_path;
        video_path += ext;
        if (create_video_from_sd_images(video_path.string().c_str(), results, num_results, gen_params.fps) == 0) {
            LOG_INFO("save result video to '%s'", video_path.string().c_str());
            return true;
        } else {
            LOG_ERROR("Failed to save result video to '%s'", video_path.string().c_str());
            return false;
        }
    }

    if (output_format == EncodedImageFormat::UNKNOWN)
        ext = ".png";

    for (int i = 0; i < num_results; ++i) {
        fs::path img_path = base_path;
        if (num_results > 1) {
            img_path += "_" + std::to_string(output_begin_idx + i);
        }
        img_path += ext;
        if (write_image(img_path, i)) {
            sucessful_reults++;
        }
    }
    LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
    return sucessful_reults != 0;
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
    sd_set_log_callback(sd_log_cb, (void*)&cli_params);
    log_verbose = cli_params.verbose;
    log_color   = cli_params.color;

    if (cli_params.mode == METADATA) {
        MetadataReadOptions options;
        options.output_format      = cli_params.metadata_format == "json"
                                         ? MetadataOutputFormat::JSON
                                         : MetadataOutputFormat::TEXT;
        options.include_raw        = cli_params.metadata_raw;
        options.brief              = cli_params.metadata_brief;
        options.include_structural = cli_params.metadata_all;

        std::string error;
        if (!print_image_metadata(cli_params.image_path, options, std::cout, error)) {
            LOG_ERROR("%s", error.c_str());
            return 1;
        }
        return 0;
    }

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

    bool vae_decode_only = true;

    auto load_image_and_update_size = [&](const std::string& path,
                                          SDImageOwner& image,
                                          bool resize_image    = true,
                                          int expected_channel = 3) -> bool {
        int expected_width  = 0;
        int expected_height = 0;
        if (resize_image && gen_params.width_and_height_are_set()) {
            expected_width  = gen_params.width;
            expected_height = gen_params.height;
        }

        if (!load_sd_image_from_file(image.put(), path.c_str(), expected_width, expected_height, expected_channel)) {
            LOG_ERROR("load image from '%s' failed", path.c_str());
            return false;
        }

        gen_params.set_width_and_height_if_unset(image.get().width, image.get().height);
        return true;
    };

    if (gen_params.init_image_path.size() > 0) {
        vae_decode_only = false;
        if (!load_image_and_update_size(gen_params.init_image_path, gen_params.init_image)) {
            return 1;
        }
    }

    if (gen_params.end_image_path.size() > 0) {
        vae_decode_only = false;
        if (!load_image_and_update_size(gen_params.end_image_path, gen_params.end_image)) {
            return 1;
        }
    }

    if (gen_params.ref_image_paths.size() > 0) {
        vae_decode_only = false;
        gen_params.ref_images.clear();
        for (auto& path : gen_params.ref_image_paths) {
            SDImageOwner ref_image({0, 0, 3, nullptr});
            if (!load_image_and_update_size(path, ref_image, false)) {
                return 1;
            }
            gen_params.ref_images.push_back(std::move(ref_image));
        }
    }

    if (gen_params.mask_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.mask_image.put(),
                                     gen_params.mask_image_path.c_str(),
                                     gen_params.get_resolved_width(),
                                     gen_params.get_resolved_height(),
                                     1)) {
            LOG_ERROR("load image from '%s' failed", gen_params.mask_image_path.c_str());
            return 1;
        }
    } else {
        sd_image_t generated_mask = {0, 0, 1, nullptr};
        generated_mask.data       = (uint8_t*)malloc(gen_params.get_resolved_width() * gen_params.get_resolved_height());
        if (generated_mask.data == nullptr) {
            LOG_ERROR("malloc mask image failed");
            return 1;
        }
        generated_mask.width  = gen_params.get_resolved_width();
        generated_mask.height = gen_params.get_resolved_height();
        memset(generated_mask.data, 255, gen_params.get_resolved_width() * gen_params.get_resolved_height());
        gen_params.mask_image.reset(generated_mask);
    }

    if (gen_params.control_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.control_image.put(),
                                     gen_params.control_image_path.c_str(),
                                     gen_params.get_resolved_width(),
                                     gen_params.get_resolved_height())) {
            LOG_ERROR("load image from '%s' failed", gen_params.control_image_path.c_str());
            return 1;
        }
        if (cli_params.canny_preprocess) {  // apply preprocessor
            preprocess_canny(gen_params.control_image.get(),
                             0.08f,
                             0.08f,
                             0.8f,
                             1.0f,
                             false);
        }
    }

    if (!gen_params.control_video_path.empty()) {
        gen_params.control_frames.clear();
        if (!load_images_from_dir(gen_params.control_video_path,
                                  gen_params.control_frames,
                                  gen_params.get_resolved_width(),
                                  gen_params.get_resolved_height(),
                                  gen_params.video_frames,
                                  cli_params.verbose)) {
            return 1;
        }
    }

    if (!gen_params.pm_id_images_dir.empty()) {
        gen_params.pm_id_images.clear();
        if (!load_images_from_dir(gen_params.pm_id_images_dir,
                                  gen_params.pm_id_images,
                                  0,
                                  0,
                                  0,
                                  cli_params.verbose)) {
            return 1;
        }
    }

    if (cli_params.mode == VID_GEN) {
        vae_decode_only = false;
    }

    if (gen_params.hires_enabled &&
        (gen_params.resolved_hires_upscaler == SD_HIRES_UPSCALER_MODEL ||
         gen_params.resolved_hires_upscaler == SD_HIRES_UPSCALER_LANCZOS ||
         gen_params.resolved_hires_upscaler == SD_HIRES_UPSCALER_NEAREST)) {
        vae_decode_only = false;
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(vae_decode_only, true, cli_params.taesd_preview);

    SDImageVec results;
    int num_results = 0;

    if (cli_params.mode == UPSCALE) {
        num_results = 1;
        results.push_back(gen_params.init_image.release());
    } else {
        SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));

        if (sd_ctx == nullptr) {
            LOG_INFO("new_sd_ctx_t failed");
            return 1;
        }

        if (gen_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }

        if (gen_params.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }

        if (gen_params.sample_params.scheduler == SCHEDULER_COUNT) {
            gen_params.sample_params.scheduler = sd_get_default_scheduler(sd_ctx.get(), gen_params.sample_params.sample_method);
        }

        if (cli_params.mode == IMG_GEN) {
            sd_img_gen_params_t img_gen_params = gen_params.to_sd_img_gen_params_t();

            num_results = gen_params.batch_count;
            results.adopt(generate_image(sd_ctx.get(), &img_gen_params), num_results);
        } else if (cli_params.mode == VID_GEN) {
            sd_vid_gen_params_t vid_gen_params = gen_params.to_sd_vid_gen_params_t();
            sd_image_t* generated_video        = generate_video(sd_ctx.get(), &vid_gen_params, &num_results);
            results.adopt(generated_video, num_results);
        }

        if (!results) {
            LOG_ERROR("generate failed");
            return 1;
        }
    }

    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (ctx_params.esrgan_path.size() > 0 && gen_params.upscale_repeats > 0) {
        UpscalerCtxPtr upscaler_ctx(new_upscaler_ctx(ctx_params.esrgan_path.c_str(),
                                                     ctx_params.offload_params_to_cpu,
                                                     ctx_params.diffusion_conv_direct,
                                                     ctx_params.n_threads,
                                                     gen_params.upscale_tile_size));

        if (upscaler_ctx == nullptr) {
            LOG_ERROR("new_upscaler_ctx failed");
        } else {
            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                SDImageOwner current_image(results[i]);
                results[i] = {0, 0, 0, nullptr};
                for (int u = 0; u < gen_params.upscale_repeats; ++u) {
                    SDImageOwner upscaled_image(upscale(upscaler_ctx.get(), current_image.get(), upscale_factor));
                    if (upscaled_image.get().data == nullptr) {
                        LOG_ERROR("upscale failed");
                        break;
                    }
                    current_image = std::move(upscaled_image);
                }
                results[i] = current_image.release();  // Set the final upscaled image as the result
            }
        }
    }

    if (!save_results(cli_params, ctx_params, gen_params, results.data(), num_results)) {
        return 1;
    }

    return 0;
}
