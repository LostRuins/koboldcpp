#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <locale>

#include <nlohmann/json.hpp>
#include <inttypes.h>
#include <cinttypes>
#include <algorithm>
#include <filesystem>
#include <set>
#include <tuple>
#include <type_traits>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <optional>

#include "otherarch/utils.h"
#include "otherarch/webp_image.h"
#include "model_adapter.h"
#include "model_file_probe.h"

#include "stable-diffusion.h"
#include "kcpp_media.h"
#include "src/kcpp_sd_extensions.h"
#include "src/conditioning/prompt_input.h"
#include "sdcpp_gguf_tokenizer.h"
#include "src/core/util.h"
#include "src/core/option_types.h"
#include "src/model.h"
#include "src/model_definition.h"
#include "src/model_loader.h"
#include "src/model_io/gguf_io.h"
#include "src/model_io/safetensors_io.h"
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
#include "webp/encode.h"
#include "webp/mux.h"

// Encodes an array of RGB, RGBA or grayscale frames into an animated WebP.
// The returned buffer uses malloc so callers can continue to release it with
// free(), matching the old GIF helper's ownership contract.
static int create_animated_webp_buf(const KcppImages& images,
                                     int fps,
                                     uint8_t** out_data,
                                     size_t* out_len) {
    if (out_data == nullptr || out_len == nullptr) {
        return -1;
    }
    *out_data = nullptr;
    *out_len  = 0;

    if (images.empty() || fps <= 0 ||
        !images.at(0).valid() ||
        images.at(0).w > WEBP_MAX_DIMENSION || images.at(0).h > WEBP_MAX_DIMENSION) {
        fprintf(stderr, "Invalid animated WebP input.\n");
        return -1;
    }

    const int width  = static_cast<int>(images.at(0).w);
    const int height = static_cast<int>(images.at(0).h);

    WebPAnimEncoderOptions anim_options;
    WebPConfig config;
    if (!WebPAnimEncoderOptionsInit(&anim_options) ||
        !WebPConfigPreset(&config, WEBP_PRESET_PICTURE, 80.0f)) {
        fprintf(stderr, "Failed to initialize the animated WebP encoder.\n");
        return -1;
    }

    // Avoid exhaustively encoding every compression/disposal candidate. Mixed
    // mode still uses heuristics to choose lossless coding when appropriate.
    anim_options.minimize_size = 0;
    anim_options.allow_mixed   = 1;
    config.method              = 3;
    config.thread_level        = 1;
    if (!WebPValidateConfig(&config)) {
        fprintf(stderr, "Invalid animated WebP encoder configuration.\n");
        return -1;
    }

    WebPAnimEncoder* encoder = WebPAnimEncoderNew(width, height, &anim_options);
    if (encoder == nullptr) {
        fprintf(stderr, "Failed to create the animated WebP encoder.\n");
        return -1;
    }

    int last_timestamp_ms = -1;
    for (int i = 0; i < (int)images.size(); ++i) {
        const KcppImage& image = images.at(i);
        if (!image.valid() || static_cast<int>(image.w) != width ||
            static_cast<int>(image.h) != height ||
            (image.c != 1 && image.c != 3 && image.c != 4)) {
            fprintf(stderr, "Animated WebP frame %d has invalid dimensions or channels.\n", i);
            WebPAnimEncoderDelete(encoder);
            return -1;
        }

        WebPPicture picture;
        if (!WebPPictureInit(&picture)) {
            fprintf(stderr, "Failed to initialize animated WebP frame %d.\n", i);
            WebPAnimEncoderDelete(encoder);
            return -1;
        }
        picture.use_argb = 1;
        picture.width    = width;
        picture.height   = height;

        int imported = 0;
        std::vector<uint8_t> rgb_buffer;
        if (image.c == 1) {
            const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
            rgb_buffer.resize(pixel_count * 3);
            for (size_t p = 0; p < pixel_count; ++p) {
                rgb_buffer[p * 3 + 0] = image.px->data()[p];
                rgb_buffer[p * 3 + 1] = image.px->data()[p];
                rgb_buffer[p * 3 + 2] = image.px->data()[p];
            }
            imported = WebPPictureImportRGB(&picture, rgb_buffer.data(), width * 3);
        } else if (image.c == 4) {
            imported = WebPPictureImportRGBA(&picture, image.px->data(), width * 4);
        } else {
            imported = WebPPictureImportRGB(&picture, image.px->data(), width * 3);
        }

        if (!imported) {
            fprintf(stderr, "Failed to import animated WebP frame %d.\n", i);
            WebPPictureFree(&picture);
            WebPAnimEncoderDelete(encoder);
            return -1;
        }

        int timestamp_ms = static_cast<int>(
            (static_cast<int64_t>(i) * 1000 + fps / 2) / fps);
        if (timestamp_ms <= last_timestamp_ms) {
            timestamp_ms = last_timestamp_ms + 1;
        }
        if (!WebPAnimEncoderAdd(encoder, &picture, timestamp_ms, &config)) {
            const char* error = WebPAnimEncoderGetError(encoder);
            fprintf(stderr, "Failed to encode animated WebP frame %d: %s\n",
                    i, error != nullptr ? error : "unknown error");
            WebPPictureFree(&picture);
            WebPAnimEncoderDelete(encoder);
            return -1;
        }
        WebPPictureFree(&picture);
        last_timestamp_ms = timestamp_ms;
    }

    int final_timestamp_ms = static_cast<int>(
        (static_cast<int64_t>(images.size()) * 1000 + fps / 2) / fps);
    if (final_timestamp_ms <= last_timestamp_ms) {
        final_timestamp_ms = last_timestamp_ms + 1;
    }
    if (!WebPAnimEncoderAdd(encoder, nullptr, final_timestamp_ms, nullptr)) {
        const char* error = WebPAnimEncoderGetError(encoder);
        fprintf(stderr, "Failed to finalize animated WebP frames: %s\n",
                error != nullptr ? error : "unknown error");
        WebPAnimEncoderDelete(encoder);
        return -1;
    }

    WebPData webp_data;
    WebPDataInit(&webp_data);
    if (!WebPAnimEncoderAssemble(encoder, &webp_data) ||
        webp_data.bytes == nullptr || webp_data.size == 0) {
        const char* error = WebPAnimEncoderGetError(encoder);
        fprintf(stderr, "Failed to assemble animated WebP: %s\n",
                error != nullptr ? error : "unknown error");
        WebPDataClear(&webp_data);
        WebPAnimEncoderDelete(encoder);
        return -1;
    }

    uint8_t* owned_data = static_cast<uint8_t*>(malloc(webp_data.size));
    if (owned_data == nullptr) {
        fprintf(stderr, "Failed to allocate animated WebP output buffer.\n");
        WebPDataClear(&webp_data);
        WebPAnimEncoderDelete(encoder);
        return -1;
    }
    memcpy(owned_data, webp_data.bytes, webp_data.size);
    *out_data = owned_data;
    *out_len  = webp_data.size;

    WebPDataClear(&webp_data);
    WebPAnimEncoderDelete(encoder);
    return 0;
}

// early callback log initialization
namespace {
    const bool _ = [] {
        // activate kcpp logging backend, and prevent sd.cpp setting ggml logging
        // callback (will keep using the default)
        // the debug flag will be reset on the first image generation call
        set_sd_log_level(1);
        return true;
    }();
}

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

};

// tracks type errors and unknown keys in the json APIs
struct SdErrors {
    std::vector<std::string> problems;
    std::set<std::string> accepted_keys;

    bool empty() const { return problems.empty(); }

    void add(const std::string& problem) { problems.push_back(problem); }

    std::string message() const {
        std::string msg;
        for (size_t i = 0; i < problems.size(); i++) {
            if (i != 0) msg += "; ";
            msg += problems[i];
        }
        return msg;
    }
};

// extract a JSON integer into a signed or unsigned target with a range check
template <typename T>
static bool sdjson_int_value(const nlohmann::json& value, T& out)
{
    // nlohmann::json stores integers that fit in int64 as int64_t and larger
    // values as double: a float is a type mismatch for signed targets, and for
    // unsigned targets only an exactly integral value in range is accepted.
    static_assert(std::is_integral_v<T>, "sdjson_int_value: integral type required");
    if (value.is_number_integer()) {
        const uint64_t u = value.is_number_unsigned() ? value.get<uint64_t>()
                                                       : static_cast<uint64_t>(value.get<int64_t>());
        if constexpr (std::is_signed_v<T>) {
            // values above INT64_MAX cast to a negative int64 and are
            // rejected by the range check
            if (static_cast<int64_t>(u) < std::numeric_limits<T>::min()
                || static_cast<int64_t>(u) > std::numeric_limits<T>::max()) {
                return false;
            }
            out = static_cast<T>(static_cast<int64_t>(u));
        } else {
            if (u > static_cast<uint64_t>(std::numeric_limits<T>::max())) return false;
            out = static_cast<T>(u);
        }
        return true;
    }
    if constexpr (std::is_unsigned_v<T>) {
        if (value.is_number_float()) {
            const double d = value.get<double>();
            if (d < 0.0 || d >= 18446744073709551616.0) return false; // 2^64
            if (d != std::floor(d)) return false;
            if (d > static_cast<double>(std::numeric_limits<T>::max())) return false;
            out = static_cast<T>(static_cast<uint64_t>(d));
            return true;
        }
    }
    return false;
}

// strict typed getter for one payload key:
//  - absent key or null value: `out` keeps its default, returns false
//  - present with the exact JSON type: copied into `out`, the key is recorded
//    as accepted, returns true
//  - present with a different type, or an integer out of range: a problem is
//    recorded and false is returned
//  - a present empty string decodes as absent
template <typename T>
static bool sdjson_get(T& out, const char* key, const nlohmann::json& payload, SdErrors& errors)
{
    if (key == nullptr) return false;
    // copy the key to a non-dependent type so payload.find() resolves to a
    // concrete overload (GCC rejects member access through the dependent
    // iterator of the const char* overload)
    const std::string k(key);
    const auto it = payload.find(k);
    if (it == payload.end()) {
        return false;
    }
    // a present key is recorded as accepted even when its value decodes as
    // absent (null or empty), so the unknown-key warning stays a drift
    // detector for keys the interface does not decode
    errors.accepted_keys.insert(k);
    if (it->is_null()) {
        return false;
    }
    const nlohmann::json& v = *it;
    if constexpr (std::is_same_v<T, bool>) {
        if (!v.is_boolean()) {
            errors.add("key '" + k + "': expected a boolean");
            return false;
        }
        out = v.get<bool>();
    } else if constexpr (std::is_integral_v<T>) {
        if (!sdjson_int_value(v, out)) {
            errors.add("key '" + k + "': expected an integer in the range of the target type");
            return false;
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        if (!v.is_number()) {
            errors.add("key '" + k + "': expected a number");
            return false;
        }
        out = static_cast<T>(v.get<double>());
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!v.is_string()) {
            errors.add("key '" + k + "': expected a string");
            return false;
        }
        const std::string s = v.get<std::string>();
        if (s.empty()) {
            return false; // empty string decodes as absent
        }
        out = std::move(s);
    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
        if (!v.is_array()) {
            errors.add("key '" + k + "': expected an array of strings");
            return false;
        }
        for (size_t i = 0; i < v.size(); ++i) {
            const auto& entry = v[i];
            if (!entry.is_string()) {
                errors.add("key '" + k + "[" + std::to_string(i) + "]: expected a string");
                continue;
            }
            out.push_back(entry.get<std::string>());
        }
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
        if (!v.is_array()) {
            errors.add("key '" + k + "': expected an array of numbers");
            return false;
        }
        for (size_t i = 0; i < v.size(); ++i) {
            const auto& entry = v[i];
            if (!entry.is_number()) {
                errors.add("key '" + k + "[" + std::to_string(i) + "]: expected a number");
                continue;
            }
            out.push_back(entry.get<float>());
        }
    } else {
        static_assert(sizeof(T) == 0, "sdjson_get: unsupported type");
    }
    return true;
}

// decode a "loras" payload key: an array of {path, multiplier, is_high_noise?}
// objects, ignoring a null or empty key
static void sdjson_get_loras(const char* key, const nlohmann::json& payload,
                             SdErrors& errors, LoraMap & lora_map)
{
    const std::string k(key);
    const auto it = payload.find(k);
    if (it == payload.end()) return;
    errors.accepted_keys.insert(k);
    if (it->is_null()) return; // null decodes as absent
    if (!it->is_array()) {
        errors.add("key '" + k + "': expected an array of {path, multiplier} objects");
        return;
    }
    const auto& arr = *it;
    for (size_t i = 0; i < arr.size(); ++i) {
        const auto& entry = arr[i];
        if (!entry.is_object()
            || !entry.contains("path") || !entry["path"].is_string()
            || entry["path"].get<std::string>().empty()
            || !entry.contains("multiplier") || !entry["multiplier"].is_number()
            || (entry.contains("is_high_noise") && !entry["is_high_noise"].is_boolean())) {
            errors.add("key '" + k + "[" + std::to_string(i) + "]: expected an object with a non-empty string 'path', a numeric 'multiplier' and an optional boolean 'is_high_noise'");
            continue;
        }
        lora_map.add_lora(entry["path"].get<std::string>(), entry["multiplier"].get<float>());
    }
}

// parse a frontier payload without exceptions; a failed parse or a
// non-object top level is recorded as a problem and yields an invalid json
static nlohmann::json sdjson_parse_payload(const char* data, SdErrors& errors)
{
    if (data == nullptr) {
        errors.add("payload is null");
        return nlohmann::json();
    }
    nlohmann::json payload;
    try {
        payload = nlohmann::json::parse(data);
    } catch (const nlohmann::json::exception& e) {
        errors.add(std::string("invalid JSON: ") + e.what());
        return nlohmann::json();
    }
    if (!payload.is_object()) {
        errors.add("payload is not a JSON object");
        return nlohmann::json();
    }
    return payload;
}

// a key the interface does not decode is a debug-level warning, not an error
// (both ends ship together; the warning is the drift detector). The "sd"
// namespace is reserved (schema-valid, decoded in a later phase) and never
// warns; the "kcpp" scope is recursed.
static void sdjson_warn_unknown_keys(const nlohmann::json& payload, SdErrors& errors)
{
    if (!payload.is_object()) return;
    for (auto it = payload.cbegin(); it != payload.cend(); ++it) {
        const std::string& key = it.key();
        if (key == "sd") continue;
        if (key == "kcpp") {
            if (it->is_object()) sdjson_warn_unknown_keys(*it, errors);
            continue;
        }
        if (!errors.accepted_keys.count(key)) {
            printf("KCPP SD: unknown key '%s' in the params payload\n", key.c_str());
        }
    }
}

// strict decode of one payload image key: a base64 string into a KcppImage
// (absent/null key or empty string = unset); a type mismatch or a failed
// image decode is an error
static bool sdjson_get_image(KcppImage& out, const char* key, const nlohmann::json& payload, SdErrors& errors)
{
    const std::string k(key);
    const auto it = payload.find(k);
    if (it == payload.end()) {
        return false;
    }
    errors.accepted_keys.insert(k);
    if (it->is_null()) {
        return false;
    }
    if (!it->is_string()) {
        errors.add("key '" + k + "': expected a base64 image string");
        return false;
    }
    const std::string s = it->get<std::string>();
    if (s.empty()) {
        return false; // empty string decodes as absent
    }
    out = KcppImage::from_b64(s);
    if (!out.valid()) {
        errors.add("key '" + k + "': failed to decode the image");
    }
    return out.valid();
}

// strict decode of one payload image-array key: a list of base64 strings into
// a KcppImages (absent/null key or empty array = unset); a malformed element
// is an error (positions must stay intact for <image:N> tags)
static bool sdjson_get_images(KcppImages& out, const char* key, const nlohmann::json& payload, SdErrors& errors)
{
    const std::string k(key);
    const auto it = payload.find(k);
    if (it == payload.end()) {
        return false;
    }
    errors.accepted_keys.insert(k);
    if (it->is_null()) {
        return false;
    }
    if (!it->is_array()) {
        errors.add("key '" + k + "': expected an array of base64 image strings");
        return false;
    }
    const auto& arr = *it;
    out.clear();
    for (size_t i = 0; i < arr.size(); ++i) {
        const auto& e = arr[i];
        if (!e.is_string() || e.get<std::string>().empty()) {
            errors.add("key '" + k + "[" + std::to_string(i) + "]: expected a non-empty base64 image string");
            continue;
        }
        KcppImage img = KcppImage::from_b64(e.get<std::string>());
        if (!img.valid()) {
            errors.add("key '" + k + "[" + std::to_string(i) + "]: failed to decode the image");
            continue;
        }
        out.push_back(img);
    }
    return true;
}


// local intermediate representation of the model loading parameters.
struct SDLoadParams {
    // model file slots
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string audio_vae_path;
    std::string taesd_path;
    std::string uncond_diffusion_model_path;
    std::string clip_vision_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string embeddings_connectors_path;
    std::string photo_maker_path;
    std::string tokenizer;

    // device / runtime
    std::string backend;
    std::string params_backend;
    std::string split_mode;
    std::string conv_direct_mode;
    std::string flash_attn_mode;
    std::string max_vram;
    std::string tensor_type_rules;
    int n_threads = -1;
    sd_type_t wtype = SD_TYPE_COUNT;
    bool enable_mmap = false;
    bool auto_fit = false;
    lora_apply_mode_t lora_apply_mode = LORA_APPLY_AT_RUNTIME;
};

struct model_info: public sd_model_info {
    /* from base class:
        int version;
        int vae_scale_factor;
        int spatial_multiple;
    */

    bool is_flux1;
    bool is_flux2;
    bool is_kontext;
    bool is_qwenimg;
    bool is_sd1;
    bool is_sd2;
    bool is_sdxs;
    bool is_sdxl;
    bool is_wan;
    bool is_zimage;
    bool is_ltx;
    bool is_minimaxh3;
    bool is_boogu;
    bool supports_ref_image;
    bool is_video_model;
};

struct KcppLoadParams {
    // tracks loading state that does not have a direct sd_opt_t mapping

    model_info info {};

    // LoRA
    LoraMap lora_map;
    bool lora_dynamic = false;
    bool lora_cache = false;
    int kcpp_lora_apply_mode = 0;

    int quant = 0;              // decoded into wtype
    bool flash_attention = false;
    bool taesd = false;
    int tiled_vae_threshold = 0;
    std::string upscaler_filename;
    int img_hard_limit = 0;
    int img_soft_limit = 0;
    bool has_llm_vision = false;
    bool photomaker_enabled = false;
};

// local intermediate representation of a generation request: decoded from
// sd_generation_inputs at the top of sdtype_generate, processed in place, then
// settled into the sd_img_gen_opt_t / sd_vid_gen_opt_t option set before the call.
struct SDGenParams {
    // prompts
    std::string prompt;
    std::string negative_prompt;

    // sampling
    float cfg_scale          = 7.0f;
    float distilled_guidance = -1.0f;
    int sample_steps         = 20;
    int shifted_timestep     = 0;
    float flow_shift         = -1.0f;
    float eta                = -1.0f;
    int64_t seed             = 42;
    int clip_skip            = -1;  // <= 0 represents unspecified
    sample_method_t sample_method = sample_method_t::SAMPLE_METHOD_COUNT;
    scheduler_t scheduler         = scheduler_t::SCHEDULER_COUNT;
    std::vector<float> custom_sigmas;
    std::string extra_sample_args;
    std::string ref_image_args;

    // size / img2img
    int width  = 512;
    int height = 512;
    float strength = 0.75f;

    // inference cache
    std::string cache_mode;
    std::string cache_options;

    // circular sampling
    bool circular_x = false;
    bool circular_y = false;

    // LoRAs merged from the model-load list and the request (dynamic mode);
    // loras[i].path points into lora_map, which must outlive loras
    LoraMap lora_map;
    std::vector<sd_lora_t> loras;

    bool upscale = false;
    bool remove_limits = false;

    // state tracking
    bool force_image_edit = false;
    int req_width  = 512;
    int req_height = 512;

    // request kind and routing, settled by kcpp_resolve_generation_parameters
    // before the upstream sd_*_gen_params structs are built
    bool is_vidgen      = false; // generated through the video path
    bool is_img2img     = false; // an input image is present
    bool is_passthrough = false; // one step at strength 0: return the input as-is

    // VAE tiling decision (image area vs. the load-time threshold)
    bool dotile       = false;
    int vae_tile_size = -1;

    // media: images decode at natural size (3 channels) and are fitted in
    // place during processing; a null image means "not provided"
    KcppImage img_init_image;
    KcppImage img_mask;
    bool img_flip_mask = false;
    KcppImages img_extra_images;

    // video fields
    KcppImage vid_start_frame;
    KcppImage vid_end_frame;
    std::vector<std::string> vid_ref_audios;
    int vid_req_frames  = 0;
    int vid_output_type = 0;
    int vid_fps         = 0;

    // media lists derived during processing
    KcppImages reference_imgs;
    KcppImages photomaker_imgs;
    KcppAudios ref_audios;
};

struct SDParams {
    SDLoadParams load_params;
    KcppLoadParams kcpp_load_params;

    // owned sd_* resources of the loaded model, released here
    sd_ctx_t* sd_ctx = nullptr;
    sd_upscaler_t* upscaler_ctx = nullptr;

    ~SDParams() {
        // only runs on model reload: the sd_params global is a raw pointer,
        // so the model is intentionally leaked at process exit
        if (sd_ctx != nullptr) {
            sd_ctx_free(sd_ctx);
            sd_ctx = nullptr;
        }
        if (upscaler_ctx != nullptr) {
            sd_upscaler_free(upscaler_ctx);
            upscaler_ctx = nullptr;
        }
    }
};

//shared
int total_img_gens = 0;

//global static vars for SD
static std::mutex g_sd_params_mux;
// raw pointer on purpose: a static unique_ptr would destroy the model during
// process exit, when sd_ctx_free()/sd_upscaler_free() may touch global state
// that is already gone (crashes, particularly on Windows). Leaking the model
// at exit is harmless.
static SDParams* sd_params = nullptr;
static int sddebugmode = 0;
static std::string sd_executable_path;
const int max_extra_images = 4;

static bool sd_is_quiet = false;

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
    bool aborted = false;
    gendata_st gendata;
} geninfo;

// this mutex, and thread_local return buffer declarations, are intended as
// safety nets: although the generation can concur with preview and cancellation
// functions, generations can't be called concurrently (sd_ctx_t does not
// support it)
static std::mutex g_sd_op_mutex;
using TryLocker = std::unique_lock<std::mutex>;

static thread_local struct {
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

static std::string sd_get_u8path(const std::string& file_path)
{
    return std::filesystem::u8path(file_path).string();
}

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

struct KcppTokenizerResources {
    std::string explicit_umt5_json;
    std::string extracted_umt5_json;
    std::string extracted_gemma_merges;
    std::string extracted_gemma_vocab;

    void reset() {
        explicit_umt5_json.clear();
        extracted_umt5_json.clear();
        extracted_gemma_merges.clear();
        extracted_gemma_vocab.clear();
    }
};

static KcppTokenizerResources kcpp_tokenizer_resources;

static std::string load_embd_file(std::string& cache, const char* filename)
{
    if (cache.empty()) {
        std::string filepath = sd_executable_path + filename;
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
    if (!kcpp_tokenizer_resources.extracted_gemma_merges.empty()) {
        return kcpp_tokenizer_resources.extracted_gemma_merges;
    }
    return load_embd_file(cache, "embd_res/gemma_merges_utf8_c_str.embd");
}
std::string load_gemma_vocab_json()
{
    static std::string cache;
    if (!kcpp_tokenizer_resources.extracted_gemma_vocab.empty()) {
        return kcpp_tokenizer_resources.extracted_gemma_vocab;
    }
    return load_embd_file(cache, "embd_res/gemma_vocab_json.embd");
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
    if (!kcpp_tokenizer_resources.explicit_umt5_json.empty()) {
        return kcpp_tokenizer_resources.explicit_umt5_json;
    }
    if (!kcpp_tokenizer_resources.extracted_umt5_json.empty()) {
        return kcpp_tokenizer_resources.extracted_umt5_json;
    }
    return load_embd_file(cache, "embd_res/umt5_tokenizer_json.embd");
}
static void step_callback(int step, int frame_count, const sd_tensor_t* const* frames, bool is_noisy, void* data);

// 0 disable, 1 initial, 2 denoised
// the first noisy call is used to detect the inference phase
static void set_preview_images(int enable) {
    bool denoised = (enable == 2);
    bool noisy = (enable == 1);
    sd_set_preview_callback(step_callback, enable == 0 ? "none" : "proj", 1, denoised, noisy, nullptr);
}

static inline double get_time_delta(const std::chrono::steady_clock::time_point& start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

static void generation_progress_callback(const sd_generation_progress_t* progress, void* data)
{
    (void) data;
    if (progress == nullptr || progress->stage == nullptr) {
        return;
    }
    const std::string stage = progress->stage;
    const int64_t completed = progress->completed;
    const int64_t total = progress->total;
    try {
        std::lock_guard<std::mutex> lock(geninfo.mux);
        if (geninfo.aborted) {
            return;
        }
        if (stage == "sampling") {
            /* the effective total can change (img2img, custom schedules, hires passes) */
            if (total > 0) {
                geninfo.steps = (int) total;
            }
            if (completed != geninfo.gendata.step) {
                geninfo.gendata.preview = "";
            }
            geninfo.gendata.step = (int) completed;
            if (geninfo.gendata.status != 2) {
                geninfo.gendata.status = 2;
                if (!geninfo.preview_requested) {
                    /* previews are only generated on demand */
                    set_preview_images(0);
                }
            }
            if (geninfo.preview_requested && !geninfo.preview_enabled) {
                geninfo.preview_enabled = true;
                set_preview_images(2);
            }
        } else if (stage == "vae_decode") {
            geninfo.gendata.status = 3;
        }
    } catch (...) {
        /* generation progress callbacks must not throw */
    }
}

static void progress_callback(int step, int steps, float time, void* data)
{
    step = step < 0 ? -step : step;
    (void) data;
    const char* phase = "Encoding";
    {
        std::lock_guard<std::mutex> lock(geninfo.mux);
        if (geninfo.aborted) {
            return;
        }
        /* phases and step counts are driven by generation_progress_callback */
        if (geninfo.gendata.status == 2) {
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

static model_info get_model_info(sd_ctx_t* ctx)
{
    auto base = sd_get_model_info(ctx);
    model_info res = {};
    *static_cast<sd_model_info*>(&res) = base;

    SDVersion loadedsdver = (SDVersion)res.version;
    res.is_wan = (loadedsdver == SDVersion::VERSION_WAN2 || loadedsdver == SDVersion::VERSION_WAN2_2_I2V || loadedsdver == SDVersion::VERSION_WAN2_2_TI2V);
    res.is_qwenimg = sd_version_is_qwen_image(loadedsdver);
    res.is_kontext = (loadedsdver == SDVersion::VERSION_FLUX);
    res.is_flux2 = (loadedsdver == SDVersion::VERSION_FLUX2 || loadedsdver == SDVersion::VERSION_FLUX2_KLEIN);
    res.is_flux1 = (loadedsdver == SDVersion::VERSION_FLUX);
    res.is_zimage = (loadedsdver == SDVersion::VERSION_Z_IMAGE);
    res.is_sdxs = (loadedsdver == SDVersion::VERSION_SDXS_512_DS || loadedsdver == SDVersion::VERSION_SDXS_09);
    res.is_sd1 = (loadedsdver == SDVersion::VERSION_SD1);
    res.is_sd2 = (loadedsdver == SDVersion::VERSION_SD2);
    res.is_sdxl = sd_version_is_sdxl(loadedsdver);
    res.is_ltx = sd_version_is_ltxav(loadedsdver);
    res.is_minimaxh3 = sd_version_is_minimax_h3(loadedsdver);
    res.is_boogu = sd_version_is_boogu_image(loadedsdver);

    const auto* model_def = find_model_definition(loadedsdver);

    if (!model_def
        || loadedsdver == VERSION_CHROMA_RADIANCE
        || model_def->traits.is_inpaint
        || model_def->traits.is_control)
        res.supports_ref_image = false;
    else if (model_def->traits.is_unet_edit || loadedsdver == VERSION_PID)
        res.supports_ref_image = true;
    else
        res.supports_ref_image = (std::string(model_def->ref_image_preset) != "default");

    res.is_video_model = model_def && model_def->traits.supports_video_generation;

    return res;
}

// image prompt spec:
// tags are <image:N> or <image:+N>, with N indexing the extra images array
// the '+' means the image is processed by the VLM _and_ provided in ref_images;
// images mentioned only by tags without '+' are removed from ref_images
// indexing is adaptive: zero-based if any tag uses 0, otherwise 1-based.

struct image_tag_refs {
    std::set<int> mentioned;
    std::set<int> hidden;  // referenced, and all occurrences without '+'
    bool zero_based = false;
    bool has_tags = false;
};

static bool parse_image_tag_source(const std::string& source, int& idx_out, bool& plus_out)
{
    size_t b = source.find_first_not_of(" \t");
    size_t e = source.find_last_not_of(" \t");
    if (b == std::string::npos) {
        return false;
    }
    size_t pos = (source[b] == '+') ? b + 1 : b;
    if (pos == e + 1) {
        return false;
    }
    int idx = 0;
    for (; pos <= e; ++pos) {
        char c = source[pos];
        if (c < '0' || c > '9') {
            return false;
        }
        idx = idx * 10 + (c - '0');
        if (idx > 1000000) {
            return false;
        }
    }
    idx_out = idx;
    plus_out = (source[b] == '+');
    return true;
}

// Collects the <image:N>/<image:+N> tags from both prompts into a single pool,
// and extract the set of hidden images and the zero-based indicator
static bool collect_image_tags(const std::string& prompt, const std::string& negative_prompt,
                               size_t image_count, image_tag_refs& refs, std::string& err)
{
    refs = {};
    std::vector<std::tuple<std::string, int, bool>> raw; // (source, idx, plus)
    for (const std::string& text : {prompt, negative_prompt}) {
        std::vector<sd::PromptPart> parts;
        try {
            parts = sd::parse_prompt_input(text);
        } catch (const std::exception& e) {
            err = std::string("invalid <image:...> prompt tag: ") + e.what();
            return false;
        }
        for (const auto& part : parts) {
            if (part.type != sd::PromptPart::Type::IMAGE) {
                continue;
            }
            int idx;
            bool plus;
            if (!parse_image_tag_source(part.content, idx, plus)) {
                err = "unsupported <image:" + part.content + "> tag, only <image:N> and <image:+N> are supported";
                return false;
            }
            raw.emplace_back(part.content, idx, plus);
        }
    }
    if (raw.empty()) {
        return true;
    }
    refs.has_tags = true;
    for (const auto& [source, idx, plus] : raw) {
        refs.zero_based = refs.zero_based || (idx == 0);
    }
    std::set<int> kept;
    for (const auto& [source, idx, plus] : raw) {
        int real = refs.zero_based ? idx : idx - 1;
        if (real < 0 || real >= (int)image_count) {
            err = "image tag <image:" + source + "> out of range, this request has "
                + std::to_string(image_count) + " reference image(s)";
            return false;
        }
        refs.mentioned.insert(real);
        if (plus) {
            kept.insert(real);
        }
    }
    for (int i : refs.mentioned) {
        if (!kept.count(i)) {
            refs.hidden.insert(i);
        }
    }
    return true;
}

// replaces each <image:N> tag with " imageK " (K the 1-based index of the
// reference image), dropping the tag syntax but keeping the reference in text
// this at least has a fair chance to be understood by an edit model
static std::string strip_image_tags(const std::string& text, bool zero_based)
{
    // the caller has already validated the text with collect_image_tags
    try {
        std::string result;
        for (const auto& part : sd::parse_prompt_input(text)) {
            if (part.type != sd::PromptPart::Type::IMAGE) {
                // The parser removes one escape from literal tags. Put it back
                // so the rewritten prompt can be parsed again.
                std::string literal = part.content;
                for (size_t pos = literal.find("<image:"); pos != std::string::npos;
                     pos = literal.find("<image:", pos + 8)) {
                    literal.insert(pos, 1, '\\');
                }
                result += literal;
                continue;
            }
            int idx;
            bool plus;
            if (!parse_image_tag_source(part.content, idx, plus)) {
                result += " " + part.content + " ";
                continue;
            }
            result += " image" + std::to_string(idx + (zero_based ? 1 : 0)) + " ";
        }
        return result;
    } catch (...) {
        return text;
    }
}

// --- sdcpp media boundary -------------------------------------------------
// Kcpp media cross the sdcpp boundary as tensor handles. The library reads
// the tensor buffers as interleaved sd::Image/sd::Audio — exactly our own
// storage layout — so the raw bytes are passed through unchanged.
// sd_tensor_create copies the storage, and a successful setter retains the
// tensor storage, so the handles may be released right after the setter.

// a KcppImage as a U8 [C,W,H] tensor handle (interleaved HWC buffer);
// nullptr for a null image
static std::shared_ptr<sd_tensor_t> image_to_tensor(const KcppImage& image)
{
    if (!image.valid()) {
        return nullptr;
    }
    const int64_t shape[3] = { image.c, image.w, image.h };
    return std::shared_ptr<sd_tensor_t>(
        sd_tensor_create(SD_TENSOR_U8, 3, shape, image.px->data(), image.px->size()),
        sd_tensor_free);
}

// a KcppAudio as an interleaved F32 [C,N] tensor handle; nullptr for a null
// audio
static std::shared_ptr<sd_tensor_t> audio_to_tensor(const KcppAudio& audio)
{
    if (!audio.valid()) {
        return nullptr;
    }
    const int64_t shape[2] = { audio.channels, (int)(audio.samples->size() / audio.channels) };
    return std::shared_ptr<sd_tensor_t>(
        sd_tensor_create(SD_TENSOR_F32, 2, shape, audio.samples->data(), audio.samples->size() * sizeof(float)),
        sd_tensor_free);
}

// set a KcppImage on a tensor option key; the handle is released after the
// setter (the option set retains the tensor storage)
static void set_image_opt(sd_opt_t* opt, const char* key, const KcppImage& image)
{
    auto t = image_to_tensor(image);
    if (t) {
        sd_opt_set_tensor(opt, key, t.get());
    }
}

// set a KcppImages list on a tensor option key (null images are skipped)
static void set_images_opt(sd_opt_t* opt, const char* key, const KcppImages& images)
{
    if (images.empty()) {
        return;
    }
    std::vector<std::shared_ptr<sd_tensor_t>> tensors;
    tensors.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        tensors.push_back(image_to_tensor(images[i]));
    }
    std::vector<sd_tensor_t*> handles;
    handles.reserve(tensors.size());
    for (auto& t : tensors) {
        if (t) {
            handles.push_back(t.get());
        }
    }
    if (!handles.empty()) {
        sd_opt_set_tensors(opt, key, handles.data(), handles.size());
    }
}

// set a KcppAudios list on an audio option key (null audios are skipped)
static void set_audios_opt(sd_opt_t* opt, const char* key, const KcppAudios& audios)
{
    if (audios.empty()) {
        return;
    }
    std::vector<std::shared_ptr<sd_tensor_t>> tensors;
    tensors.reserve(audios.size());
    for (size_t i = 0; i < audios.size(); ++i) {
        tensors.push_back(audio_to_tensor(audios[i]));
    }
    std::vector<sd_audio_t> descriptors;
    descriptors.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i]) {
            descriptors.push_back(sd_audio_t{audios[i].sample_rate, tensors[i].get()});
        }
    }
    if (!descriptors.empty()) {
        sd_opt_set_audios(opt, key, descriptors.data(), descriptors.size());
    }
}

struct PromptImageRef {
    // per-request pool of reference images (original request order) served by
    // prompt_image_from_refs for <image:N> prompt tags
    std::vector<KcppImage> image_pool;
    // lazily created tensor views of the pool: the sdcpp callback copies each
    // served tensor before the next callback, so a pooled handle stays valid
    std::vector<std::shared_ptr<sd_tensor_t>> image_views;
    bool zero_based = false;

    void resize(size_t n) {
        image_pool.resize(n);
        image_views.assign(n, nullptr);
    }
    void set(size_t i, const KcppImage& image) {
        image_pool[i] = image;
        image_views[i].reset(); // invalidate the pooled tensor
    }
};

// callback registered per-generation: resolves
// <image:N> / <image:+N> prompt tags to the reference image pool
static bool prompt_image_from_refs(const char* source, void* data, const sd_tensor_t** image_out)
{
    PromptImageRef *rdata = reinterpret_cast<PromptImageRef*>(data);
    if (rdata == nullptr)
        return false;
    int idx;
    bool plus;
    if (!source || !parse_image_tag_source(source, idx, plus)) {
        return false;
    }
    idx = rdata->zero_based ? idx : idx - 1;
    if (idx < 0 || idx >= (int)rdata->image_pool.size()) {
        return false;
    }
    const KcppImage& image = rdata->image_pool[idx];
    if (!image.valid()) {
        return false;
    }
    if (!rdata->image_views[idx]) {
        auto t = image_to_tensor(image);
        if (!t) {
            return false;
        }
        rdata->image_views[idx] = t;
    }
    *image_out = rdata->image_views[idx].get();
    return true;
}

// this is a class only to access a protected member
class Kcpp_ModelLoader: public ModelLoader
{
public:
    bool has_diffusion_model_tensors()
    {
        for (auto& [name, tensor_storage] : tensor_storage_map) {
            if (tensor_storage.name.find("model.diffusion_model.") != std::string::npos) {
                return true;
            }
        }
        return false;
    }
};

static std::string toLowerCase(const std::string& str) {
    std::string result;
    std::locale loc;
    for (char ch : str) {
        result += std::tolower(ch, loc); // Use locale-aware tolower
    }
    return result;
}

// kcpp: content-based classification for the "extra model file" fields
// (clip_1 / clip_2 / t5 / llm / vision / ...). These fields are generic: they
// may hold a text encoder, an mmproj, a tokenizer JSON, etc. depending on the
// model family, so misplaced files are detected by content (kcpp_file_probe)
// rather than by filename or extension alone.

struct kcpp_st_class
{
    // safetensors-only classification: tensor counts by family prefix
    int text_t5   = 0;   // T5 / UMT5 encoder
    int text_clip = 0;   // CLIP-L / CLIP-G / SigLIP text tower
    int vision    = 0;   // CLIP / SigLIP / Qwen vision tower
    int llm       = 0;   // causal LLM (or text side of a combined VLM file)
    int diffusion = 0;   // UNet / DiT
    int vae       = 0;
    const char* label = nullptr; // dominant st bucket, for logging; null if unclear

    bool is_vision_only() const {
        return vision >= 3 && text_t5 == 0 && text_clip == 0 && llm == 0 && diffusion == 0 && vae == 0;
    }
};

struct kcpp_path_class
{
    bool valid              = false;  // the probe managed to open the file
    bool is_safetensors     = false;
    bool is_json            = false;
    bool is_mmproj          = false;
    bool is_mmproj_filename = false;  // name heuristic: "mmproj" in the filename (fallback only)
    bool is_connector       = false;  // connector file; currently only MiniMax P-H3 text connector
    bool is_text_encoder    = false;
    kcpp_st_class st;                 // safetensors only
};

// count tensors whose name starts with any of the prefixes
static int st_count_prefix(const std::vector<kcpp_probe_tensor>& ts,
                           std::initializer_list<const char*> prefixes)
{
    int n = 0;
    for (const auto& t : ts)
    {
        for (const char* p : prefixes)
        {
            if (t.name.rfind(p, 0) == 0) { n++; break; }
        }
    }
    return n;
}

// true only when *every* tensor of the file is a connector tensor: the
// standalone connector file is exactly these tensors, while a combined dump
// (diffusion / VAE / LLM + connector) must not be routed as a connector
static bool kcpp_tensors_standalone_connector(const std::vector<kcpp_probe_tensor>& ts)
{
    if (ts.empty()) return false;

    // P-H3 flavour 1: the standalone connector file (p-h3-text-connector.safetensors
    // and external GGUF conversions of it) stores the tensors under bare names;
    // the loader prepends "embeddings_connectors." (diffusion_engine.cpp) and
    // name_conversion.cpp maps that onto the model's names. Require the exact
    // tensor set (plus the proj shape, matching PH3ImageTextConnector's fixed
    // input/output dims) so unrelated files never match
    static const std::set<std::string> bare_names = {
        "input_mean", "input_std", "output_mean", "output_std",
        "proj.weight", "sink_embedding",
    };
    if (ts.size() == bare_names.size()) {
        bool all_bare = true;
        for (const auto& t : ts) {
            if (!bare_names.count(t.name)) { all_bare = false; break; }
        }
        if (all_bare) {
            for (const auto& t : ts) {
                if (t.name == "proj.weight") {
                    // shape order differs between safetensors (torch order) and
                    // gguf (ne order); accept either
                    if (t.shape.size() == 2
                     && ((t.shape[0] == 5120 && t.shape[1] == 12800)
                      || (t.shape[0] == 12800 && t.shape[1] == 5120)))
                        return true;
                    break;
                }
            }
        }
    }

    // P-H3 flavour 2: combined checkpoints keep the connector tensors under a
    // prefixed name; these prefixes are unique to the MiniMax-H3 family
    // (see src/name_conversion.cpp), so a file whose tensors all carry one of
    // them must be a connector dump
    for (const auto& t : ts) {
        if (!(t.name.rfind("text_encoders.ph3_image_connector.", 0) == 0
           || t.name.rfind("text_encoders.llm.text_connector.", 0) == 0
           || t.name.rfind("embeddings_connectors.", 0) == 0))
            return false;
    }
    return true;
}

static void kcpp_classify_safetensors(const kcpp_file_probe& probe, kcpp_path_class& c)
{
    const std::vector<kcpp_probe_tensor>& ts = probe.tensors();
    if (ts.empty()) return;

    // the connector tensors (e.g. P-H3) match no bucket below; classify it first
    if (kcpp_tensors_standalone_connector(ts))
    {
        c.is_connector = true;
        c.st.label = "text connector";
        return;
    }

    c.st.text_t5   = st_count_prefix(ts, {"encoder.block.", "encoder.embed_tokens.",
                                          "encoder.final_layer_norm.", "shared.weight"});
    c.st.text_clip = st_count_prefix(ts, {"text_model.", "text_projection"});
    c.st.vision    = st_count_prefix(ts, {"vision_model.", "vision_tower.", "model.visual.",
                                          "visual.", "multi_modal_projector."});
    c.st.llm       = st_count_prefix(ts, {"model.layers.", "model.embed_tokens.",
                                          "model.language_model.", "language_model.",
                                          "layers.", "embed_tokens.", "lm_head."});
    c.st.diffusion = st_count_prefix(ts, {"model.diffusion_model.", "input_blocks.",
                                          "output_blocks.", "transformer_blocks.",
                                          "down_blocks.", "up_blocks.", "mid_block."});
    c.st.vae       = st_count_prefix(ts, {"encoder.conv_in.", "decoder.conv_in.",
                                          "quant_conv.", "post_quant_conv.",
                                          "first_stage_model."});

    int best = c.st.text_t5;
    const char* label = "T5 text encoder";
    if (c.st.text_clip  > best) { best = c.st.text_clip;  label = "CLIP text encoder"; }
    if (c.st.vision     > best) { best = c.st.vision;     label = "vision encoder"; }
    if (c.st.llm        > best) { best = c.st.llm;        label = "LLM"; }
    if (c.st.diffusion  > best) { best = c.st.diffusion;  label = "diffusion model"; }
    if (c.st.vae        > best) { best = c.st.vae;        label = "VAE"; }
    if (best < 3) return;  // too few hits to claim anything
    // a combined VLM dump (e.g. convert_qwen3_vl.py output) has both towers
    if (c.st.llm >= 3 && c.st.vision >= 3 && best == std::max(c.st.llm, c.st.vision)) {
        label = "LLM + vision (combined VLM file)";
    }
    c.st.label = label;
}

static std::map<std::string, kcpp_path_class> kcpp_path_class_cache;

static kcpp_path_class kcpp_classify_path(const std::string& path)
{
    if (path.empty()) return {};

    std::string key = path;
    {
        auto it = kcpp_path_class_cache.find(key);
        if (it != kcpp_path_class_cache.end()) return it->second;
    }

    kcpp_path_class c;
    kcpp_file_probe probe;
    c.valid = probe.open(path);
    bool identified = false;
    if (c.valid) {
        auto format = probe.format();
        if (kcpp_file_format::JSON == format) {
            c.is_json = true;
            identified = true;
        } else if (kcpp_file_format::GGUF == format) {
            const std::string type = probe.metadata_get("general.type");
            c.is_mmproj = type == "mmproj"
                || type == "clip-vision" // Mistral/pixtral-era files "just in case"
                || probe.metadata_has_prefix("clip.vision.")
                || probe.metadata_get("clip.has_vision_encoder") == "1"
                || !probe.metadata_get("clip.projector_type").empty();
            // connector GGUFs come from external converters with no metadata
            // convention; the tensor names are the reliable signal
            c.is_connector = !c.is_mmproj && kcpp_tensors_standalone_connector(probe.tensors());
            identified = c.is_mmproj || c.is_connector;
            if (!identified) {
                c.is_text_encoder = probe.tensor_exists("token_embd.weight")
                    || probe.tensor_exists("embed_tokens.weight", false);
                identified = c.is_text_encoder;
            }
        } else if (kcpp_file_format::SAFETENSORS == format || kcpp_file_format::SAFETENSORS_INDEX == format) {
            c.is_safetensors = true;
            kcpp_classify_safetensors(probe, c);
            identified = c.st.label != nullptr;
            c.is_text_encoder = std::max({c.st.llm, c.st.text_clip, c.st.text_t5}) >= 3;
        }
        if (!identified && !probe.error().empty())
        {
            printf("\nKCPP: could not identify %s (%s)\n", path.c_str(), probe.error().c_str());
        }
    }
    // the name heuristic is tracked separately: should only apply if
    // metadata couldn't identify it
    if (!identified) {
        const auto basename = std::filesystem::u8path(path).filename().u8string();
        c.is_mmproj_filename = toLowerCase(basename).find("mmproj") != std::string::npos;
    }

    kcpp_path_class_cache[key] = c;
    return c;
}

static bool kcpp_is_confirmed_vision_encoder(const kcpp_path_class& c, bool for_llm = false)
{
    // The LLM vision slot prepends its own visual namespace. Safetensors
    // with visual.* or model.visual.* keys need the ordinary LLM slot.
    return c.is_mmproj || (!for_llm && c.st.is_vision_only());
}

static bool kcpp_is_vision_encoder(const kcpp_path_class& c, bool for_llm = false)
{
    return kcpp_is_confirmed_vision_encoder(c, for_llm) || c.is_mmproj_filename;
}

// read the model version from the file metadata using a temporary loader and
// resolve model / diffusion_model and text encoder / llm / proj / vision paths
static void kcpp_set_model_paths(SDLoadParams& lp, KcppLoadParams& klp)
{
    kcpp_tokenizer_resources.reset();

    bool model_is_gguf = false;
    if(lp.t5xxl_path!="" || lp.clip_l_path!="" || lp.clip_g_path!="")
    {
        model_is_gguf = is_gguf_file(lp.model_path);
        //extra check - make sure there is no diffusion model prefix already inside!
        if(model_is_gguf && !gguf_tensor_exists(lp.model_path,"model.diffusion_model.",false))
        {
            printf("\nSwap to Diffusion Model Path:%s",lp.model_path.c_str());
            lp.diffusion_model_path = lp.model_path;
            lp.model_path = "";
        }
    }

    std::string& kcpp_tokenizer_path = lp.tokenizer;
    std::string& kcpp_taesd_path     = lp.taesd_path;

    SDVersion tempver = VERSION_COUNT;

    {
        // use a temporary loader to detect the model version, then set
        // model_path and diffusion_model_path to the correct places to be
        // later loaded in the same way by the sdcpp loader
        Kcpp_ModelLoader model_loader;

        if (!lp.model_path.empty()) {
            printf("loading model from '%s'", lp.model_path.c_str());
            if (!model_loader.init_from_file(lp.model_path)) {
                printf("init model loader from file failed: '%s'", lp.model_path.c_str());
            }
        }

        if (!lp.diffusion_model_path.empty()) {
            printf("loading diffusion model from '%s'", lp.diffusion_model_path.c_str());
            if (!model_loader.init_from_file(lp.diffusion_model_path, "model.diffusion_model.")) {
                printf("loading diffusion model from '%s' failed", lp.diffusion_model_path.c_str());
            }
        }

        tempver = model_loader.get_sd_version();

        bool fallback_prefixed = false;

        // kcpp fallback to separate diffusion model passed as model
        if (!lp.model_path.empty() &&
            lp.diffusion_model_path.empty() &&
            (!lp.t5xxl_path.empty()||!lp.clip_l_path.empty()))
        {
            bool model_is_safetensors = !model_is_gguf && is_safetensors_file(lp.model_path);
            if(model_is_safetensors && !model_loader.has_diffusion_model_tensors())
            {
                printf("SD Diffusion Model tensors missing! Fallback trying alternative tensor names...\n");
                if (!model_loader.init_from_file(lp.model_path, "model.diffusion_model.")) {
                    printf("loading diffusion model from '%s' failed", lp.model_path.c_str());
                }
                fallback_prefixed = true;
                // Keep both paths so VAE or conditioner tensors that must remain at their original names still work.
                lp.diffusion_model_path = lp.model_path;
                tempver = model_loader.get_sd_version();
            }
        }

        if (tempver == VERSION_ANIMA && !fallback_prefixed &&
            !lp.model_path.empty() &&
            lp.diffusion_model_path.empty() &&
            !model_loader.has_diffusion_model_tensors()
           )
        {
            printf("Anima: SD Diffusion Model tensors missing! Fallback trying alternative tensor names...\n");
            if (!model_loader.init_from_file(lp.model_path, "model.diffusion_model.")) {
                printf("loading diffusion model from '%s' failed", lp.model_path.c_str());
            }
            lp.diffusion_model_path = lp.model_path;
            tempver = model_loader.get_sd_version();
        }

    }

    auto sibling_tokenizer_json = [&](const std::string& model_path) -> std::string {
        if (model_path.empty()) {
            return {};
        }
        std::error_code error;
        const auto candidate = std::filesystem::u8path(model_path).parent_path() / "tokenizer.json";
        if (std::filesystem::is_regular_file(candidate, error)) {
            return candidate.string();
        }
        return {};
    };

    bool iswan = sd_version_is_wan(tempver);
    bool is_wan21 = sd_version_is_wan(tempver) && tempver != VERSION_WAN2_2_TI2V;
    bool is_qwenimg = sd_version_is_qwen_image(tempver);
    bool is_zimg = sd_version_is_z_image(tempver);
    bool is_zimg_l2p = sd_version_is_z_image_l2p(tempver);
    bool is_iris = (tempver == VERSION_IRIS);
    bool is_flux2 = sd_version_is_flux2(tempver);
    bool is_flux3 = sd_version_is_flux3(tempver);
    bool is_mosi = (tempver == VERSION_MOSS_IMAGE);
    bool is_ovis =  (tempver==VERSION_OVIS_IMAGE);
    bool is_anima = sd_version_is_anima(tempver);
    bool is_ernie = sd_version_is_ernie_image(tempver);
    bool is_longcat = sd_version_is_longcat(tempver);
    bool is_lens = sd_version_is_lens(tempver);
    bool is_ming = (tempver == VERSION_MING_IMAGE);
    bool is_pid = sd_version_is_pid(tempver);
    bool is_llada = sd_version_is_llada_image(tempver);
    bool is_ltx = sd_version_is_ltxav(tempver);
    bool is_ideogram = sd_version_is_ideogram4(tempver);
    bool is_boogu = sd_version_is_boogu_image(tempver);
    bool is_krea2 = sd_version_is_krea2(tempver);
    bool is_sefi = sd_version_is_sefi_image(tempver);
    bool is_mageflow = sd_version_is_mage_flow(tempver);
    bool is_minimaxh3 = sd_version_is_minimax_h3(tempver);
    bool is_fibo2 = (tempver == VERSION_FIBO2);
    bool is_lingbot = sd_version_is_lingbot_video(tempver);
    bool conditioner_is_llm = (is_qwenimg || is_zimg || is_flux2 || is_flux3 || is_mosi || is_ovis || is_anima || is_ernie || is_longcat || is_lens || is_llada || is_ltx || is_ideogram || is_boogu || is_krea2 || is_sefi || is_mageflow || is_minimaxh3 || is_pid || is_ming || is_zimg_l2p || is_iris || is_fibo2);
    const bool legacy_clip2_llm_vision = is_qwenimg || is_longcat || is_boogu || is_fibo2;
    // LLM text encoders with a vision tower (Qwen-VL family); these models
    // can take a mmproj file (see the llm.visual handling in src/name_conversion.cpp)
    bool has_llm_vision = (
        // Qwen 2.5-VL
        is_qwenimg ||  // Qwen Image 1.0/2.0 (2.1 uses Qwen 3-VL)
        is_longcat ||
        // Qwen 3-VL
        is_fibo2 || is_flux3 || is_lingbot || is_ideogram || is_boogu ||
        is_sefi || is_krea2 || is_minimaxh3 || is_mageflow || is_iris ||
        // Qwen 3.5
        is_mosi
    );
    klp.has_llm_vision = has_llm_vision;
    std::string* vision_path = has_llm_vision ? &lp.llm_vision_path
                            : (iswan || is_ltx) ? &lp.clip_vision_path : nullptr;
    bool has_tokenizer = (is_lens || is_pid || is_ming || is_mosi || is_llada);

    std::string kcpp_main_tokenizer;

    // content-based pre-pass over the three "extra model file" slots
    // (t5/llm, sdclip1, sdclip2). A file that is a tokenizer JSON or an
    // mmproj by content is moved out of its slot here, so the heuristics
    // below only have to deal with text encoders. Two files of the same
    // kind is an error: the first one (in slot order) wins and the second
    // is left where it is.
    {
        std::string* slots[3] = {&lp.t5xxl_path, &lp.clip_l_path, &lp.clip_g_path};

        // Content-confirmed vision encoders outrank the filename fallback,
        // regardless of which generic input slot they came from.
        if (vision_path != nullptr && vision_path->empty())
        {
            std::string* filename_guess = nullptr;
            std::string* vision_source = nullptr;
            for (auto slot: slots)
            {
                if (slot->empty()) continue;
                const kcpp_path_class c = kcpp_classify_path(*slot);
                if (kcpp_is_confirmed_vision_encoder(c, has_llm_vision))
                {
                    vision_source = slot;
                    break;
                }
                if (filename_guess == nullptr && c.is_mmproj_filename)
                {
                    filename_guess = slot;
                }
            }
            if (vision_source == nullptr) vision_source = filename_guess;
            if (vision_source != nullptr)
            {
                printf("\nKCPP: routing %s to the vision encoder slot\n", vision_source->c_str());
                std::swap(*vision_path, *vision_source);
            }
        }

        for (auto slot: slots)
        {
            if (slot->empty()) continue;
            const kcpp_path_class c = kcpp_classify_path(*slot);
            if (!c.valid) continue;

            if (c.is_json)
            {
                // a second tokenizer is an error: the first one wins
                if (!kcpp_main_tokenizer.empty() || !kcpp_tokenizer_resources.explicit_umt5_json.empty()) continue;
                if (iswan)
                {
                    try
                    {
                        kcpp_tokenizer_resources.explicit_umt5_json = read_str_from_disk(*slot);
                    }
                    catch (const std::exception& exception)
                    {
                        printf("\nKCPP: could not read UMT5 tokenizer JSON from %s: %s\n",
                               slot->c_str(), exception.what());
                        continue;
                    }
                }
                else
                {
                    kcpp_main_tokenizer = *slot;
                }
                printf("\nKCPP: using tokenizer JSON found in %s\n", slot->c_str());
                *slot = "";
            }
            else if (c.is_connector)
            {
                // the P-H3 text connector belongs in the embeddings-connectors
                // field; route it out before the llm swaps below can grab it
                if (lp.embeddings_connectors_path.empty())
                {
                    printf("\nKCPP: routing P-H3 text connector found in %s to the embeddings-connectors field\n", slot->c_str());
                    std::swap(lp.embeddings_connectors_path, *slot);
                }
                else
                {
                    printf("\nKCPP: P-H3 text connector found in %s, but the embeddings-connectors field is already in use; leaving it as is\n", slot->c_str());
                }
            }
            else if (c.is_safetensors && c.st.label)
            {
                // Keep combined VLM safetensors intact so sd.cpp can split them.
                printf("\nKCPP: %s looks like a %s (safetensors)\n", slot->c_str(), c.st.label);
            }
        }
    }

    //kcpp qol fallback: if a llm was loaded as t5 by mistake
    if(conditioner_is_llm && !lp.t5xxl_path.empty())
    {
        if(lp.clip_l_path.empty())
        {
            std::swap(lp.clip_l_path, lp.t5xxl_path);
        }
        else if(lp.clip_g_path.empty())
        {
            //very tricky case. see if we can tell if clipl is an mmproj, if so move to right place
            if(kcpp_is_vision_encoder(kcpp_classify_path(lp.clip_l_path), has_llm_vision) || is_ltx)
            {
                // g = l, l = t, t = ""
                std::swap(lp.clip_g_path, lp.clip_l_path);
                std::swap(lp.clip_l_path, lp.t5xxl_path);
            }
        }
    }

    //settle clip-l replacements
    if (!lp.clip_l_path.empty())
    {
        if(conditioner_is_llm && lp.llm_path.empty())
        {
            if (kcpp_is_vision_encoder(kcpp_classify_path(lp.clip_l_path), has_llm_vision))
            {
                // an mmproj in a text-encoder field: route it to the vision
                // field when the model has one and it is free
                if (has_llm_vision && lp.llm_vision_path.empty())
                {
                    std::swap(lp.clip_l_path, lp.llm_vision_path);
                }
                else
                {
                    printf("\nKCPP: clip_1 holds a vision encoder; leaving clip_1 as is\n");
                }
            }
            else
            {
                std::swap(lp.llm_path, lp.clip_l_path);
            }
        }
        else if(iswan)
        {
            if(lp.t5xxl_path.empty())
            {
                if (!kcpp_is_vision_encoder(kcpp_classify_path(lp.clip_l_path)))
                    std::swap(lp.t5xxl_path, lp.clip_l_path);
            } else if (lp.clip_vision_path.empty()) {
                if (!kcpp_is_vision_encoder(kcpp_classify_path(lp.clip_l_path)))
                    std::swap(lp.clip_vision_path, lp.clip_l_path);
            }
        }
    }

    //settle clip-g replacements
    if (!lp.clip_g_path.empty())
    {
        const kcpp_path_class c = kcpp_classify_path(lp.clip_g_path);
        if (kcpp_is_vision_encoder(c, has_llm_vision))
        {
            if (vision_path != nullptr && vision_path->empty())
            {
                std::swap(*vision_path, lp.clip_g_path);
            }
        } else if (iswan && lp.clip_vision_path.empty()) {
            // Older Wan CLIP vision checkpoints do not carry mmproj metadata.
            std::swap(lp.clip_vision_path, lp.clip_g_path);
        } else if (conditioner_is_llm && lp.llm_path.empty()) {
            // The slot can hold a text encoder whose format the probe cannot classify.
            std::swap(lp.llm_path, lp.clip_g_path);
        } else if (legacy_clip2_llm_vision && lp.llm_vision_path.empty() &&
                   !c.is_text_encoder && !c.st.is_vision_only()) {
            std::swap(lp.llm_vision_path, lp.clip_g_path);
        } else {
            if(is_ltx)
            {
                std::swap(lp.embeddings_connectors_path, lp.clip_g_path);
            }
            else if(is_ideogram)
            {
                std::swap(lp.uncond_diffusion_model_path, lp.clip_g_path);
            }
            else if(sd_version_is_minimax_h3_image(tempver) && lp.embeddings_connectors_path.empty())
            {
                // the P-H3 text connector goes to the embeddings connectors field
                std::swap(lp.embeddings_connectors_path, lp.clip_g_path);
            }
        }
    }

    //settle possible inversions for mmproj
    if(!lp.llm_vision_path.empty() && !lp.llm_path.empty())
    {
        if(!kcpp_is_vision_encoder(kcpp_classify_path(lp.llm_vision_path), true) &&
        kcpp_is_vision_encoder(kcpp_classify_path(lp.llm_path), true))
        {
            std::swap(lp.llm_path, lp.llm_vision_path);
        }
    }

    // Safetensors encoders have no embedded tokenizer metadata, so Wan/LTX may
    // use a tokenizer.json placed directly beside the selected text encoder.
    // Never probe beside GGUF files; their tokenizer comes from GGUF metadata.
    auto is_safetensors_source = [](const std::string& path) {
        return !path.empty() && kcpp_classify_path(path).is_safetensors;
    };
    if (iswan && kcpp_tokenizer_resources.explicit_umt5_json.empty()) {
        const std::string tokenizer_source = !lp.t5xxl_path.empty() ? lp.t5xxl_path : lp.model_path;
        const std::string tokenizer_path =
            is_safetensors_source(tokenizer_source)
                ? sibling_tokenizer_json(tokenizer_source)
                : std::string();
        if (!tokenizer_path.empty()) {
            try {
                kcpp_tokenizer_resources.explicit_umt5_json = read_str_from_disk(tokenizer_path);
                printf("\nKCPP: using adjacent UMT5 tokenizer JSON from %s\n", tokenizer_path.c_str());
            } catch (const std::exception& exception) {
                printf("\nKCPP: could not read adjacent UMT5 tokenizer JSON from %s: %s\n",
                       tokenizer_path.c_str(),
                       exception.what());
            }
        }
    }
    if (is_ltx && kcpp_main_tokenizer.empty()) {
        const std::string tokenizer_source = !lp.llm_path.empty() ? lp.llm_path : lp.model_path;
        if (is_safetensors_source(tokenizer_source)) {
            kcpp_main_tokenizer = sibling_tokenizer_json(tokenizer_source);
        }
        if (!kcpp_main_tokenizer.empty()) {
            printf("\nKCPP: using adjacent Gemma tokenizer JSON from %s\n", kcpp_main_tokenizer.c_str());
        }
    }

    //settle tae: build the TAE path for the detected model version
    if(klp.taesd)
    {
        const char* tae_name = nullptr;
        if(sd_version_is_sd1(tempver) || sd_version_is_sd2(tempver))
        {
            tae_name = "taesd.embd";
        }
        else if(sd_version_is_sdxl(tempver))
        {
            tae_name = "taesd_xl.embd";
        }
        else if(sd_version_uses_flux_vae(tempver))
        {
            tae_name = "taesd_f.embd";
        }
        else if(sd_version_is_sd3(tempver))
        {
            tae_name = "taesd_3.embd";
        }
        else if(sd_version_uses_flux2_vae(tempver))
        {
            tae_name = "taesd_f2.embd";
        }
        else if(tempver == VERSION_QWEN_IMAGE_2_1)
        {
            tae_name = "taesd_qi21.embd";
        }
        else if(sd_version_uses_wan_vae(tempver))
        {
            tae_name = "taesd_w21.embd";
        }

        if(tae_name)
        {
            kcpp_taesd_path = sd_executable_path + "embd_res/" + tae_name;
            if(!file_exists(kcpp_taesd_path))
            {
                printf("\nCannot use TAESD: \"%s\" not found. TAESD Disabled!\n", kcpp_taesd_path.c_str());
                kcpp_taesd_path = "";
            }
        }
        else
        {
            printf("\nCannot use TAESD: Unknown model version %d. TAESD Disabled!\n", tempver);
        }
        if(kcpp_taesd_path != "")
        {
            printf("With TAE SD VAE: %s\n", kcpp_taesd_path.c_str());
        }
    }
    if (!lp.photo_maker_path.empty() && tempver != VERSION_SDXL) {
        printf("\nWARNING: PhotoMaker is only compatible with SDXL models. PhotoMaker will be disabled!\n");
        lp.photo_maker_path = "";
        klp.photomaker_enabled = false;
    }

    //for models with TAE suppport, if vae is set, tae is off, and it looks like a tae, swap to tae
    if(kcpp_taesd_path=="" && !lp.vae_path.empty() && toLowerCase(lp.vae_path).rfind("tae")!=std::string::npos)
    {
        try {
            const uintmax_t tae_size_limit = 64 * 1024 * 1024; //if its less than 64mb, it might be a TAE
            // Get the file size in bytes cross-platform
            uintmax_t size = std::filesystem::file_size(lp.vae_path);
            if (size > 0 && size < tae_size_limit) {
                printf("\nVAE appears to be a TAE, loading as TAE instead!\n");
                kcpp_taesd_path = lp.vae_path;
                lp.vae_path = "";
            }
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::printf("Error accessing file: %s\n", e.what());
        }
    }

    const auto resource_missing = [&](const char* name) {
        return !file_exists(sd_executable_path + "embd_res/" + name);
    };

    // Reconstruct exactly what the existing tokenizer classes need from the
    // selected text encoder's GGUF metadata, entirely in memory.
    if (iswan && kcpp_tokenizer_resources.explicit_umt5_json.empty()) {
        const std::string tokenizer_gguf = !lp.t5xxl_path.empty() ? lp.t5xxl_path : lp.model_path;
        if (!tokenizer_gguf.empty() && is_gguf_file(tokenizer_gguf)) {
            std::string tokenizer_error;
            if (kcpp_sd::gguf_tokenizer_extractor::to_unigram_json(
                    tokenizer_gguf,
                    kcpp_tokenizer_resources.extracted_umt5_json,
                    tokenizer_error)) {
                printf("\nKCPP: using UMT5 tokenizer metadata from %s\n", tokenizer_gguf.c_str());
            } else {
                printf("\nKCPP: could not use UMT5 tokenizer metadata from %s: %s\n",
                       tokenizer_gguf.c_str(),
                       tokenizer_error.c_str());
            }
        }
    }

    if (is_ltx && kcpp_main_tokenizer.empty()) {
        const std::string tokenizer_gguf = !lp.llm_path.empty() ? lp.llm_path : lp.model_path;
        if (!tokenizer_gguf.empty() && is_gguf_file(tokenizer_gguf)) {
            std::string tokenizer_error;
            if (kcpp_sd::gguf_tokenizer_extractor::to_vocab_and_merges(
                    tokenizer_gguf,
                    kcpp_tokenizer_resources.extracted_gemma_vocab,
                    kcpp_tokenizer_resources.extracted_gemma_merges,
                    tokenizer_error)) {
                printf("\nKCPP: using Gemma tokenizer metadata from %s\n", tokenizer_gguf.c_str());
            } else {
                printf("\nKCPP: could not use Gemma tokenizer metadata from %s: %s\n",
                       tokenizer_gguf.c_str(),
                       tokenizer_error.c_str());
            }
        }
    }

    if (!kcpp_main_tokenizer.empty()) {
        // assemble the tokenizer config
        if (!file_exists(kcpp_main_tokenizer)) {
            printf("\nKCPP: tokenizer not found: %s\n", kcpp_main_tokenizer.c_str());
        }
        if (!kcpp_tokenizer_path.empty()) {
            kcpp_tokenizer_path += ",";
        }
        kcpp_tokenizer_path += "main=";
        kcpp_tokenizer_path += kcpp_main_tokenizer;
    } else if (kcpp_tokenizer_path.empty()) {
        // Explicit tokenizer JSON wins. Models with built-in tokenizer assets
        // keep using them; GGUF metadata is only the final in-memory fallback.
        const bool uses_baked_qwen = conditioner_is_llm && !has_tokenizer && !is_flux2 && !is_ernie && !is_ltx;
        const bool uses_baked_mistral = is_flux2 || is_ernie;
        const bool baked_missing =
            (uses_baked_qwen && resource_missing("qwen2_merges_utf8_c_str.embd")) ||
            (uses_baked_mistral &&
             (resource_missing("mistral2_merges_utf8_c_str.embd") || resource_missing("mistral2_vocab_json.embd")));

        if (has_tokenizer || baked_missing) {
            const std::string tokenizer_gguf = !lp.llm_path.empty() ? lp.llm_path : lp.model_path;
            if (!tokenizer_gguf.empty() && is_gguf_file(tokenizer_gguf)) {
                std::string tokenizer_error;
                const auto profile = is_mosi ? kcpp_sd::tokenizer_profile::MOSS_IMAGE
                                             : kcpp_sd::tokenizer_profile::DEFAULT;
                if (kcpp_sd::gguf_tokenizer_extractor::to_hf_json(tokenizer_gguf,
                                                                  kcpp_tokenizer_path,
                                                                  tokenizer_error,
                                                                  profile)) {
                    printf("\nKCPP: using tokenizer metadata from %s\n", tokenizer_gguf.c_str());
                } else {
                    printf("\nKCPP: could not use tokenizer metadata from %s: %s\n",
                           tokenizer_gguf.c_str(),
                           tokenizer_error.c_str());
                }
            }
        }
    }

    // patch hidream to fix broken images on vulkan
    // https://github.com/leejet/stable-diffusion.cpp/issues/1496
    if(tempver == VERSION_HIDREAM_O1 && lp.tensor_type_rules.empty())
    {
        lp.tensor_type_rules = "^model.language_model.layers.[0-9]+.mlp.down_proj.weight=bf16";
    }

    lp.flash_attn_mode = "off";
    if(klp.flash_attention) {
        if(sd_version_is_qwen_image(tempver)) {
            // edit 14feb, breaks qwen image edit (moved from diffusion_engine)
            lp.flash_attn_mode = "on,te=off";
        } else {
            lp.flash_attn_mode = "on";
        }
    }

    //debug print
    // printf("\n\nclip_g: %s\nclip_l: %s\nclip_vision: %s\nllm: %s\nllm_vision: %s\nt5xxl: %s\ntaesd: %s\n",
    // lp.clip_g_path, lp.clip_l_path, lp.clip_vision_path,
    // lp.llm_path, lp.llm_vision_path, lp.t5xxl_path,
    // lp.taesd_path);
}

// helper to unify the loading table
struct LoadParams: public SDLoadParams, public KcppLoadParams {};

static void sd_track_key(std::set<std::string>& present, const char* key)
{
    if (key != nullptr) {
        present.insert(key);
    }
}

static bool sd_check_key(const std::set<std::string>& present, const char* key)
{
    return key != nullptr && present.count(key) != 0;
}

// field-row binding for the sd_load_model payload: input JSON key, output
// sd.cpp key, and the IR member. The decode
// loop walks the rows with one strict sdjson_get each and records the tracking
// key of the fields present in the request; the settle loop forwards the rows
// with an sd_opt key to the context options, only for present fields.
// Converted fields (quant, flash_attention) resolve to their upstream key in
// the QoL/routing step and are settled there (settle=false). Executable_path
// and loras stay explicit.
struct LoadStr  { const char* in; const char* out; std::string LoadParams::*  p; };
struct LoadI32  { const char* in; const char* out; int         LoadParams::*  p; bool settle = true; };
struct LoadBool { const char* in; const char* out; bool        LoadParams::*  p; bool settle = true; };

static const LoadStr load_str[] = {
    // in key               out key (sd_opt)              IR member
    {"model_filename",     "model_path",                  &LoadParams::model_path},
    {nullptr,              "diffusion_model_path",        &LoadParams::diffusion_model_path},
    {nullptr,              "clip_vision_path",            &LoadParams::clip_vision_path},
    {nullptr,              "llm_path",                    &LoadParams::llm_path},
    {nullptr,              "llm_vision_path",             &LoadParams::llm_vision_path},
    {nullptr,              "tokenizer",                   &LoadParams::tokenizer},
    {nullptr,              "embeddings_connectors_path",  &LoadParams::embeddings_connectors_path},
    {nullptr,              "uncond_diffusion_model_path", &LoadParams::uncond_diffusion_model_path},
    {"vae_filename",       "vae_path",                    &LoadParams::vae_path},
    {"audio_vae_filename", "audio_vae_path",              &LoadParams::audio_vae_path},
    {"llm_filename",       "t5xxl_path",                  &LoadParams::t5xxl_path},
    {"clip1_filename",     "clip_l_path",                 &LoadParams::clip_l_path},
    {"clip2_filename",     "clip_g_path",                 &LoadParams::clip_g_path},
    {"photomaker_filename","photo_maker_path",            &LoadParams::photo_maker_path},
    {"backend",            "backend",                     &LoadParams::backend},
    {"params_backend",     "params_backend",              &LoadParams::params_backend},
    {"split_mode",         "split_mode",                  &LoadParams::split_mode},
    {"conv_direct_mode",   "conv_direct_mode",            &LoadParams::conv_direct_mode},
    {"max_vram",           "max_vram",                    &LoadParams::max_vram},
    {"upscaler_filename",  nullptr,                       &LoadParams::upscaler_filename},
    {nullptr,              "tensor_type_rules",           &LoadParams::tensor_type_rules},
};
static const LoadI32 load_i32[] = {
    {"threads",            "n_threads",         &LoadParams::n_threads},
    {"lora_apply_mode",    nullptr,             &LoadParams::kcpp_lora_apply_mode},
    {"quant",              "wtype",             &LoadParams::quant, false},
    {"tiled_vae_threshold",nullptr,             &LoadParams::tiled_vae_threshold},
    {"img_hard_limit",     nullptr,             &LoadParams::img_hard_limit},
    {"img_soft_limit",     nullptr,             &LoadParams::img_soft_limit},
};
static const LoadBool load_bool[] = {
    {"use_mmap",           "enable_mmap",       &LoadParams::enable_mmap},
    {"auto_fit",           "auto_fit",          &LoadParams::auto_fit},
    {"flash_attention",    "flash_attn_mode",   &LoadParams::flash_attention, false},
    {"taesd",              nullptr,             &LoadParams::taesd},
};

// settle the load parameters into a context option set and create the
// context; only members present in the request are set (absent = library
// default). The kcpp policy (rng, eager load, model args, lora apply mode,
// flash attention) is always set.
static sd_ctx_t* create_sd_context(const LoadParams& lp, const std::set<std::string>& present)
{
    auto* opt = sd_ctx_opt_create();
    if (opt == nullptr) {
        return nullptr;
    }
    sd_opt_set_string(opt, "rng_type", sd_rng_type_name(CUDA_RNG));
    sd_opt_set_string(opt, "model_args", "chroma_use_dit_mask=true");
    sd_opt_set_bool(opt, "eager_load", true);
    sd_opt_set_string(opt, "lora_apply_mode", sd_lora_apply_mode_name(lp.lora_apply_mode));
    sd_opt_set_string(opt, "flash_attn_mode", lp.flash_attn_mode.c_str());
    for (const auto& f : load_str) {
        if (f.out != nullptr && !(lp.*f.p).empty()) {
            sd_opt_set_string(opt, f.out, (lp.*f.p).c_str());
        }
    }
    // set by the QoL/routing step, not by the payload decode
    if (!lp.taesd_path.empty()) {
        sd_opt_set_string(opt, "taesd_path", lp.taesd_path.c_str());
    }
    for (const auto& f : load_i32) {
        if (f.settle && sd_check_key(present, f.out)) {
            sd_opt_set_int(opt, f.out, lp.*f.p);
        }
    }
    for (const auto& f : load_bool) {
        if (f.settle && sd_check_key(present, f.out)) {
            sd_opt_set_bool(opt, f.out, lp.*f.p);
        }
    }
    // resolved from quant in the QoL/routing step
    if (lp.wtype != SD_TYPE_COUNT) {
        sd_opt_set_string(opt, "wtype", sd_type_name(lp.wtype));
    }
    sd_ctx_t* ctx = nullptr;
    sd_ctx_create(opt, &ctx);
    sd_opt_free(opt); // the context retains its own configuration
    return ctx;
}

static void kcpp_resolve_load_parameters(SDLoadParams& lp, KcppLoadParams& klp)
{
    klp.tiled_vae_threshold = (klp.tiled_vae_threshold > 8192 ? 8192 : klp.tiled_vae_threshold);
    klp.tiled_vae_threshold = (klp.tiled_vae_threshold <= 0 ? 8192 : klp.tiled_vae_threshold); //if negative dont tile
    if (klp.quant > 0) {
        lp.wtype = (klp.quant==1?SD_TYPE_Q8_0:SD_TYPE_Q4_0);
    }

    printf("\nImageGen Init - Load Model: %s\n", lp.model_path.c_str());

    { // LoRA parameters and initial list
        lp.lora_apply_mode = LORA_APPLY_AT_RUNTIME;
        klp.lora_dynamic = false;

        if(klp.kcpp_lora_apply_mode >= 1 && klp.kcpp_lora_apply_mode <= 2) {
            lp.lora_apply_mode = (lora_apply_mode_t)klp.kcpp_lora_apply_mode;
        }
        else {
            // bit 3: LoRAs can be changed dynamically
            // bit 4: cache the initial LoRA list in VRAM
            klp.lora_dynamic = !!(klp.kcpp_lora_apply_mode & (1<<3));
            klp.lora_cache   = klp.lora_dynamic && !!(klp.kcpp_lora_apply_mode & (1<<4));
        }
        // TODO: LoRA caching produces errors after a preloaded LoRA is later removed on a request
        klp.lora_cache = false;

        if(klp.lora_map.items.size() > 0)
        {
            const char* lora_apply_mode_name = lp.lora_apply_mode == LORA_APPLY_IMMEDIATELY ? "immediately"
                : lp.lora_apply_mode == LORA_APPLY_AT_RUNTIME ? "at runtime"
                : "auto";
            const char * lora_dynamic_name = klp.lora_dynamic ? ", dynamic" : "";
            const char * lora_cache_name = klp.lora_cache ? ", with caching" : "";
            printf("With LoRAs in apply mode %s%s%s:\n", lora_apply_mode_name, lora_dynamic_name, lora_cache_name);
            for(auto lora: klp.lora_map.items)
            {
                printf("  %s at %f power\n", lora.first.c_str(), lora.second);
            }
        }
    }

    if(klp.taesd)
    {
        printf("With Tiny Autoencoder (TAE)\n");
        if (klp.tiled_vae_threshold < 8192) {
            printf("  disabling VAE tiling for TAESD\n");
            klp.tiled_vae_threshold = 8192;
        }
    }
    else if(lp.vae_path!="")
    {
        printf("With Custom VAE: %s\n",lp.vae_path.c_str());
    }
    if(lp.audio_vae_path!="")
    {
        printf("With Audio VAE: %s\n",lp.audio_vae_path.c_str());
    }
    if(lp.t5xxl_path!="")
    {
        printf("With Custom Text Encoder LLM Model: %s\n",lp.t5xxl_path.c_str());
    }
    if(lp.clip_l_path!="")
    {
        printf("With Custom Clip-1 Model: %s\n",lp.clip_l_path.c_str());
    }
    if(lp.clip_g_path!="")
    {
        printf("With Custom Clip-2 Model: %s\n",lp.clip_g_path.c_str());
    }
    if(lp.photo_maker_path!="")
    {
        printf("With PhotoMaker Model: %s\n",lp.photo_maker_path.c_str());
        klp.photomaker_enabled = true;
    }
    if(klp.upscaler_filename!="")
    {
        printf("With Upscaler Model: %s\n",klp.upscaler_filename.c_str());
    }
    if(klp.flash_attention)
    {
        printf("Flash Attention is enabled\n");
    }
    if(!lp.conv_direct_mode.empty())
    {
        printf("Conv2D Direct mode: \"%s\"\n", lp.conv_direct_mode.c_str());
    }
    if(lp.backend != "")
    {
        printf("Backend assignment: \"%s\"\n", lp.backend.c_str());
    }
    if (lp.enable_mmap && lp.params_backend == "CPU") {
        printf("Offloading weights to system RAM with mmap\n");
        if (!klp.lora_dynamic && klp.lora_map.items.size() > 0) {
            printf("Note: static LoRAs can reduce mmap memory savings!\n");
        }
    } else if (lp.params_backend == "CPU") {
        printf("Offloading weights to system RAM\n");
    } else if (lp.enable_mmap) {
        printf("Using mmap for I/O\n");
    }
    if(lp.auto_fit) {
        printf("Using auto-fit");
    }
    if(lp.params_backend != "" && lp.params_backend != "CPU") {
        printf("Parameters backend assignment: \"%s\"\n", lp.params_backend.c_str());
    }
    if(lp.split_mode != "") {
        printf("Using split mode: \"%s\"\n", lp.split_mode.c_str());
    }
    if(!lp.max_vram.empty()) {
        printf("Using max VRAM = %s GB\n", lp.max_vram.c_str());
        if (lp.params_backend == "") {
            printf("Note: a VRAM limit may not be effective without offloading!\n");
        }
    }
    if(lp.wtype != SD_TYPE_COUNT)
    {
        printf("Diffusion Model quantized to %s\n", sd_type_name(lp.wtype));
        printf("Note: Loading a pre-quantized model is always faster than using compress weights!\n");
    }

    kcpp_set_model_paths(lp, klp);
}

bool sdtype_load_model(const sd_load_model_inputs inputs) {

    TryLocker no_concurrency{g_sd_op_mutex, std::try_to_lock};
    if (!no_concurrency.owns_lock()) {
        // shouldn't happen
        printf("\nError: ImageGen Init - another operation is in progress\n");
        return false;
    }

    // globals
    sddebugmode = inputs.debugmode;
    set_sd_log_level(sddebugmode);
    sd_is_quiet = inputs.quiet;
    set_sd_quiet(sd_is_quiet);

    auto new_params = std::make_unique<SDParams>();
    SDLoadParams& lp = new_params->load_params;
    KcppLoadParams& klp = new_params->kcpp_load_params;

    // tracking keys of the fields present in the request payload: the settle
    // forwards only these (absent = library default)
    std::set<std::string> present;
    {
        // assigned to the global before any QoL probe
        std::string executable_path;
        SdErrors errors;
        LoadParams lptmp;
        nlohmann::json payload = sdjson_parse_payload(inputs.params_json, errors);
        if (payload.is_object()) {
            sdjson_get(executable_path, "executable_path", payload, errors);

            // one strict getter per field row
            for (const auto& f : load_str)
                if (sdjson_get(lptmp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : load_i32)
                if (sdjson_get(lptmp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : load_bool)
                if (sdjson_get(lptmp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);

            sdjson_get_loras("loras", payload, errors, lptmp.lora_map);

            if (sddebugmode) {
                sdjson_warn_unknown_keys(payload, errors);
            }
        }
        if (!errors.empty()) {
            printf("\nError: KCPP SD load: %s\n", errors.message().c_str());
            return false;
        }
        sd_executable_path = sd_get_u8path(executable_path);
        lp = static_cast<SDLoadParams&>(lptmp);
        klp = static_cast<KcppLoadParams&>(lptmp);
    }

    kcpp_resolve_load_parameters(lp, klp);

    {
        // release the previous model (if any) to free resources, before building
        // the new one; we can't risk a failure to load the new one due to low VRAM
        std::unique_ptr<SDParams> old_params;
        {
            std::lock_guard<std::mutex> lock(g_sd_params_mux);
            // take ownership so the old model is destroyed when the
            // temporary goes out of scope
            old_params.reset(sd_params);
            sd_params = nullptr;
        }
    }

    {
        // the full load aggregate, from the two stored halves
        LoadParams load_agg{lp, klp};
        new_params->sd_ctx = create_sd_context(load_agg, present);
    }

    if (new_params->sd_ctx == nullptr) {
        printf("\nError: Image generation failed to setup!\nMake sure you have ALL files required (e.g. VAE, T5, Clip...) or baked into the model!\n");
        return false;
    }

    const auto& info = klp.info = get_model_info(new_params->sd_ctx);

    if (info.is_video_model)
    {
        printf("\nSetting to Video Generation Mode!\n");
    }

    // preload the LoRAs with the initial multipliers
    std::vector<sd_lora_t> lora_specs = klp.lora_map.get_lora_specs(klp.lora_dynamic && klp.lora_cache);
    if(lora_specs.size()>0)
    {
        printf("  applying %zu LoRAs...\n", lora_specs.size());
        set_lora_cache(new_params->sd_ctx, klp.lora_cache);
        apply_loras(new_params->sd_ctx, lora_specs);
        set_lora_cache(new_params->sd_ctx, false);
    }

    //load upscaler if provided
    if (klp.upscaler_filename!="") {
        const int upscale_tile_size = 128;
        auto* opt = sd_upscaler_opt_create();
        sd_opt_set_string(opt, "model_path", klp.upscaler_filename.c_str());
        sd_opt_set_bool(opt, "conv_direct", lp.conv_direct_mode == "on");
        if (lp.n_threads > 0) {
            sd_opt_set_int(opt, "n_threads", lp.n_threads);
        }
        sd_opt_set_int(opt, "tile_size", upscale_tile_size);
        if (!lp.backend.empty()) {
            sd_opt_set_string(opt, "backend", lp.backend.c_str());
        }
        if (!lp.params_backend.empty()) {
            sd_opt_set_string(opt, "params_backend", lp.params_backend.c_str());
        }
        sd_upscaler_t* upscaler = nullptr;
        sd_upscaler_create(opt, &upscaler);
        sd_opt_free(opt); // the upscaler retains its own configuration
        new_params->upscaler_ctx = upscaler;

        if (new_params->upscaler_ctx == nullptr) {
             printf("\nError: KCPP failed to load upscaler!\n");
        } else {
            printf("\nUpscaler has been loaded.\n");
        }
    }

    set_preview_images(0);

    sd_set_progress_callback(progress_callback, nullptr);
    sd_set_generation_progress_callback(new_params->sd_ctx, generation_progress_callback, nullptr);

    {
        std::lock_guard<std::mutex> lock(g_sd_params_mux);
        // publish the new model (the previous one, if any, was released above)
        sd_params = new_params.release();
    }

    return true;
}

static inline int rounddown_to(int n, int fac) {
    return n - n % fac;
}

static inline int roundup_to(int n, int fac) {
    return ((n + fac - 1) / fac) * fac;
}

// round a float/double avoiding too many decimal places
// (like 6/10 -> 0.6000000238418579)
static inline double round_for_info(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", v);
    return strtod(buf, nullptr);
}

// the library's generation-info string, parsed without exceptions; a failed
// parse is a library bug, not a request problem, so it degrades to a null
// entry (which the info builders below skip) rather than an SdErrors problem
static nlohmann::json parse_generation_info(const char* data) {
    if (data == nullptr)
        return nullptr;
    try {
        return nlohmann::json::parse(data);
    } catch (const nlohmann::json::exception& e) {
        printf("KCPP SD: invalid generation info from sdcpp: %s\n", e.what());
        return nullptr;
    }
}

// accessors for the library's generation-info JSON (the resolved effective
// parameters; key presence replaces the old sentinel/range gates)
// keys are dotted paths (the same convention the input requests use); a
// missing key or a type mismatch yields nullopt, so unavailable items can be
// omitted
static const nlohmann::json* info_path(const nlohmann::json& info, const std::string& path) {
    const nlohmann::json* cur = &info;
    size_t start = 0;
    while (true) {
        const size_t dot = path.find('.', start);
        const std::string key = path.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (!cur->is_object())
            return nullptr;
        auto it = cur->find(key);
        if (it == cur->end())
            return nullptr;
        cur = &*it;
        if (dot == std::string::npos)
            return cur;
        start = dot + 1;
    }
}

static std::optional<std::string> info_str(const nlohmann::json& info, const std::string& key) {
    if (const auto* v = info_path(info, key); v != nullptr && v->is_string())
        return v->get<std::string>();
    return std::nullopt;
}

static std::optional<int> info_int(const nlohmann::json& info, const std::string& key) {
    if (const auto* v = info_path(info, key); v != nullptr && v->is_number_integer())
        return v->get<int>();
    return std::nullopt;
}

static std::optional<double> info_double(const nlohmann::json& info, const std::string& key, bool round = true) {
    if (const auto* v = info_path(info, key); v != nullptr && v->is_number()) {
        auto res = v->get<double>();
        return round ? round_for_info(res) : res;
    }
    return std::nullopt;
}

// for an int64 (e.g. seed), accept a numeric string, too
static std::optional<int64_t> info_int64(const nlohmann::json& info, const std::string& key) {
    if (const auto* v = info_path(info, key); v != nullptr) {
        if (v->is_number_integer())
            return v->get<int64_t>();
        if (v->is_string())
            return std::strtoll(v->get<std::string>().c_str(), nullptr, 10);
    }
    return std::nullopt;
}

// the loaded model filenames are reported under "models", keyed by role
static std::string info_model_name(const nlohmann::json& info, const char* key) {
    auto models = info.find("models");
    if (models == info.end() || !models->is_object())
        return "";
    auto it = models->find(key);
    if (it == models->end() || !it->is_string() || it->get_ref<const std::string&>().empty())
        return "";
    return it->get<std::string>();
}

// the <lora:name:multiplier> line, reconstructed from the resolved loras list
static std::string get_lora_meta_from_info(const nlohmann::json& info) {
    std::stringstream lora_meta;
    lora_meta << std::setprecision(6);
    auto loras = info.find("loras");
    if (loras != info.end() && loras->is_array()) {
        for (const auto& lora : *loras) {
            auto path = lora.find("path");
            auto mult = lora.find("multiplier");
            if (path == lora.end() || !path->is_string() || mult == lora.end() || !mult->is_number())
                continue;
            const float multiplier = mult->get<float>();
            if (multiplier == 0.f)
                continue;
            lora_meta << "<lora:" << std::filesystem::path(path->get<std::string>()).stem().string()
                      << ":" << multiplier << ">";
        }
    }
    return lora_meta.str();
}

static std::string get_image_info_text(const nlohmann::json& info, int seed_offset) {
    std::stringstream ss;
    ss << std::setprecision(3)
        << "Prompt: " << info_str(info, "prompt").value_or("") + get_lora_meta_from_info(info);
    if (auto v = info_str(info, "negative_prompt"); v.has_value())
        ss << " | NegativePrompt: " << *v;
    if (auto v = info_int(info, "sample_params.sample_steps"); v.has_value())
        ss << " | Steps: " << *v;
    if (auto v = info_double(info, "sample_params.guidance.txt_cfg"); v.has_value())
        ss << " | CFGScale: " << *v;
    if (auto v = info_double(info, "sample_params.guidance.distilled_guidance"); v.has_value())
        ss << " | Guidance: " << *v;
    if (auto v = info_int64(info, "seed"); v.has_value())
        ss << " | Seed: " << (*v + seed_offset);
    if (auto w = info_int(info, "width"); w.has_value()) {
        if (auto h = info_int(info, "height"); h.has_value()) {
            ss << " | Size: " << *w << "x" << *h;
        }
    }
    if (auto sampler = info_str(info, "sample_params.sample_method"); sampler.has_value()) {
        ss << " | Sampler: " << *sampler;
        if (auto scheduler = info_str(info, "sample_params.scheduler"); scheduler.has_value())
            ss << " " << *scheduler;
    }
    if (auto v = info_double(info, "sample_params.eta"); v.has_value())
        ss << " | Eta: " << *v;
    if (auto v = info_int(info, "sample_params.shifted_timestep"); v.has_value())
        ss << " | Timestep Shift: " << *v;
    if (auto v = info_double(info, "sample_params.flow_shift"); v.has_value())
        ss << " | Flow Shift: " << *v;
    if (auto v = info_int(info, "clip_skip"); v.has_value())
        ss << " | Clip skip: " << *v;
    std::string model = info_model_name(info, "model");
    if (model.empty())
        model = info_model_name(info, "diffusion_model");
    if (!model.empty())
        ss << " | Model: " << model;
    if (auto vae = info_model_name(info, "vae"); !vae.empty())
        ss << " | VAE: " << vae;
    if (auto tae = info_model_name(info, "taesd"); !tae.empty())
        ss << " | TAE: " << tae;
    ss << " | Version: KoboldCpp";
    return ss.str();
}

// the A1111-style "info" object for the generation result, built from the
// library's per-item generation info (in a batch, the first valid item wins)
static nlohmann::json build_result_jsoninfo(const std::vector<nlohmann::json>& jsonresults) {
    nlohmann::json jsoninfo = nlohmann::json::object();
    for (const auto& geninfo : jsonresults) {
        if (geninfo.is_null())
            continue;
        jsoninfo["prompt"] = info_str(geninfo, "prompt").value_or("") + get_lora_meta_from_info(geninfo);
        if (auto v = info_str(geninfo, "negative_prompt"); v.has_value() && !v->empty())
            jsoninfo["negative_prompt"] = *v;
        if (auto v = info_int64(geninfo, "seed"); v.has_value())
            jsoninfo["seed"] = *v;
        if (auto v = info_int(geninfo, "width"); v.has_value())
            jsoninfo["width"] = *v;
        if (auto v = info_int(geninfo, "height"); v.has_value())
            jsoninfo["height"] = *v;
        if (auto v = info_int(geninfo, "sample_params.sample_steps"); v.has_value())
            jsoninfo["steps"] = *v;
        if (auto v = info_double(geninfo, "sample_params.guidance.txt_cfg"); v.has_value())
            jsoninfo["cfg_scale"] = *v;
        if (auto v = info_double(geninfo, "sample_params.eta"); v.has_value())
            jsoninfo["eta"] = *v;
        if (auto v = info_int(geninfo, "clip_skip"); v.has_value())
            jsoninfo["clip_skip"] = *v;
        if (auto v = info_double(geninfo, "strength"); v.has_value())
            jsoninfo["denoising_strength"] = *v;
        if (auto v = info_str(geninfo, "sample_params.sample_method"); v.has_value())
            jsoninfo["sampler_name"] = *v;
        {
            nlohmann::json extra = nlohmann::json::object();
            if (auto v = info_str(geninfo, "sample_params.scheduler"); v.has_value())
                extra["Schedule type"] = *v;
            if (!extra.empty())
                jsoninfo["extra_generation_params"] = extra;
        }
        std::string model = info_model_name(geninfo, "model");
        if (model.empty())
            model = info_model_name(geninfo, "diffusion_model");
        if (!model.empty())
            jsoninfo["sd_model_name"] = model;
        std::string vae = info_model_name(geninfo, "vae");
        if (vae.empty())
            vae = info_model_name(geninfo, "taesd");
        if (!vae.empty())
            jsoninfo["sd_vae_name"] = vae;
        jsoninfo["lora_meta"] = get_lora_meta_from_info(geninfo);
        jsoninfo["version"] = "KoboldCpp";
        break;
    }
    return jsoninfo;
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

// settle the parsed cache options onto the generation option set. The
// cache_mode value is the sdcpp mode name (the kcpp "cache-dit" spelling is
// converted on decode); the named-parameter wire format is the sdcpp one.
// Only options present in the request are set (absent = library default).
static void parse_cache_options(sd_opt_t* opt, const std::string& cache_mode,
    const std::string& cache_options) {

    // the sdcpp mode names (cache.mode option values)
    static const char* cache_modes[] = {
        "disabled", "easycache", "ucache", "dbcache", "taylorseer", "cache_dit", "spectrum",
    };
    bool mode_known = false;
    for (const char* m : cache_modes) {
        if (cache_mode == m) {
            mode_known = true;
            break;
        }
    }
    if (!mode_known) {
        if (!cache_mode.empty()) {
            printf("warning: unknown cache mode '%s'", cache_mode.c_str());
        }
        return;
    }
    if (cache_mode == "disabled")
        return;

    sd_opt_set_string(opt, "cache.mode", cache_mode.c_str());

    if (cache_options == "")
        return;

    auto set_float = [&](const char* key, const std::string& val) -> bool {
        try {
            sd_opt_set_float(opt, key, std::stof(val));
            return true;
        } catch (const std::exception&) {
            printf("error: invalid value '%s' for parameter '%s'", val.c_str(), key);
            return false;
        }
    };
    auto set_int = [&](const char* key, const std::string& val) -> bool {
        try {
            sd_opt_set_int(opt, key, std::stoi(val));
            return true;
        } catch (const std::exception&) {
            printf("error: invalid value '%s' for parameter '%s'", val.c_str(), key);
            return false;
        }
    };
    auto set_flag = [&](const char* key, const std::string& val) -> bool {
        try {
            sd_opt_set_bool(opt, key, std::stof(val) != 0.0f);
            return true;
        } catch (const std::exception&) {
            printf("error: invalid value '%s' for parameter '%s'", val.c_str(), key);
            return false;
        }
    };

    std::stringstream ss(cache_options);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t eq_pos = token.find('=');
        if (eq_pos == std::string::npos) {
            printf("error: cache option '%s' missing '=' separator", token.c_str());
            return;
        }
        std::string key = token.substr(0, eq_pos);
        std::string val = token.substr(eq_pos + 1);
        if (key == "threshold") {
            if (cache_mode == "easycache" || cache_mode == "ucache") {
                if (!set_float("cache.reuse_threshold", val)) return;
            } else {
                if (!set_float("cache.residual_diff_threshold", val)) return;
            }
        } else if (key == "start") {
            if (!set_float("cache.start_percent", val)) return;
        } else if (key == "end") {
            if (!set_float("cache.end_percent", val)) return;
        } else if (key == "decay") {
            if (!set_float("cache.error_decay_rate", val)) return;
        } else if (key == "relative") {
            if (!set_flag("cache.use_relative_threshold", val)) return;
        } else if (key == "reset") {
            if (!set_flag("cache.reset_error_on_compute", val)) return;
        } else if (key == "Fn" || key == "fn") {
            if (!set_int("cache.fn_compute_blocks", val)) return;
        } else if (key == "Bn" || key == "bn") {
            if (!set_int("cache.bn_compute_blocks", val)) return;
        } else if (key == "warmup") {
            if (cache_mode == "spectrum") {
                if (!set_int("cache.spectrum_warmup_steps", val)) return;
            } else {
                if (!set_int("cache.max_warmup_steps", val)) return;
            }
        } else if (key == "w") {
            if (!set_float("cache.spectrum_w", val)) return;
        } else if (key == "m") {
            if (!set_int("cache.spectrum_m", val)) return;
        } else if (key == "lam") {
            if (!set_float("cache.spectrum_lam", val)) return;
        } else if (key == "window") {
            if (!set_int("cache.spectrum_window_size", val)) return;
        } else if (key == "flex") {
            if (!set_float("cache.spectrum_flex_window", val)) return;
        } else if (key == "stop") {
            if (!set_float("cache.spectrum_stop_percent", val)) return;
        } else {
            printf("error: unknown cache parameter '%s'", key.c_str());
            return;
        }
    }
}

static std::string raw_image_to_png_base64(const KcppImage& img, std::string parameters = "") {
    std::string result;
    if (!img.valid()) {
        return result;
    }
    int out_data_len = 0;
    unsigned char * png = stbi_write_png_to_mem(img.px->data(), 0, img.w, img.h, img.c, &out_data_len, parameters != "" ? parameters.c_str() : nullptr);
    if (png != NULL) {
        result = kcpp_base64_encode(png,out_data_len);
        free(png);
    }
    return result;
}

static std::string upscale_image_to_png_base64(sd_upscaler_t* upscaler_ctx, const KcppImage& input_image, const std::string& meta_image_info = "")
{
    std::string gen_data;
    sd_result_t* result = nullptr;
    bool ok = false;
    {
        auto* opt = sd_upscale_opt_create();
        auto t = image_to_tensor(input_image);
        if (opt && t) {
            ok = (sd_opt_set_tensor(opt, "image", t.get()) == SD_STATUS_OK);
            if (ok) {
                ok = (sd_upscale(upscaler_ctx, opt, &result) == SD_STATUS_OK) && (result != nullptr);
            }
        }
        sd_opt_free(opt); // the option set retained the tensor storage
        // t is released here
    }
    if (ok) {
        gen_data = raw_image_to_png_base64(KcppImage::from_tensor(sd_result_get_image(result, 0)), meta_image_info);
        sd_result_free(result);
    } else {
        printf("Upscaling failed!\n");
        gen_data = raw_image_to_png_base64(input_image, meta_image_info);
    }
    return gen_data;
}

void sdtype_abort_generation() {
    {
        std::lock_guard<std::mutex> lock(geninfo.mux);
        geninfo.aborted = true;
        geninfo.gendata.status = 0;
        geninfo.gendata.preview = "";
        geninfo.preview_requested = false;
        geninfo.preview_enabled = false;
    }
    {
        std::lock_guard<std::mutex> lock(g_sd_params_mux);
        // safe to call while holding the mutex: it's a signal-safe flag store
        sd_cancel_generation(sd_params ? sd_params->sd_ctx : nullptr, SD_CANCEL_ALL);
    }
}

// field-row binding for the sd_generate payload: one row per wire
// parameter, binding the JSON key (in), the upstream sd_opt key (out;
// nullptr = kcpp-only, never forwarded) and the IR member. The
// decode loop walks the rows with one strict sdjson_get each. Explicit:
// sample_method/scheduler (string -> enum), images (base64 decode), loras
// the kcpp wire format spells the DIT cache mode "cache-dit"; the sdcpp
// option value is "cache_dit"
static std::string dec_cache_mode(const std::string& s) {
    return (s == "cache-dit") ? "cache_dit" : s;
}

// and custom_sigmas.
struct GenStr   { const char* in; const char* out; std::string              SDGenParams::* p; std::string (*dec)(const std::string&) = nullptr; };
struct GenF32   { const char* in; const char* out; float                    SDGenParams::* p; };
struct GenI32   { const char* in; const char* out; int                      SDGenParams::* p; };
struct GenI64   { const char* in; const char* out; int64_t                  SDGenParams::* p; };
struct GenBool  { const char* in; const char* out; bool                     SDGenParams::* p; };
struct GenVecStr{ const char* in; const char* out; std::vector<std::string> SDGenParams::* p; };
struct GenImage { const char* in; const char* out; KcppImage                SDGenParams::* p; };
struct GenImages{ const char* in; const char* out; KcppImages               SDGenParams::* p; };

static const GenStr gen_str[] = {
    // in key             out key (sd_opt)                      IR member
    {"prompt",           "prompt",                              &SDGenParams::prompt},
    {"negative_prompt",  "negative_prompt",                     &SDGenParams::negative_prompt},
    {"ref_image_args",   "ref_image_args",                      &SDGenParams::ref_image_args},
    {"extra_sample_args","sample.args",                         &SDGenParams::extra_sample_args},
    {"cache_mode",       "cache.mode",                          &SDGenParams::cache_mode, dec_cache_mode},
    {"cache_options",    nullptr,                               &SDGenParams::cache_options},
};
static const GenF32 gen_f32[] = {
    {"cfg_scale",         "sample.guidance.txt_cfg",             &SDGenParams::cfg_scale},
    {"distilled_guidance","sample.guidance.distilled_guidance",  &SDGenParams::distilled_guidance},
    {"flow_shift",        "sample.flow_shift",                   &SDGenParams::flow_shift},
    {"eta",               "sample.eta",                          &SDGenParams::eta},
    {"denoising_strength","strength",                            &SDGenParams::strength},
};
static const GenI32 gen_i32[] = {
    {"sample_steps",      "sample.steps",                        &SDGenParams::sample_steps},
    {"width",             "width",                               &SDGenParams::width},
    {"height",            "height",                              &SDGenParams::height},
    {"shifted_timestep",  "sample.shifted_timestep",             &SDGenParams::shifted_timestep},
    {"clip_skip",         "clip_skip",                           &SDGenParams::clip_skip},
    {"vid_req_frames",    nullptr,                               &SDGenParams::vid_req_frames},
    {"video_output_type",nullptr,                               &SDGenParams::vid_output_type},
    {"vid_fps",           nullptr,                               &SDGenParams::vid_fps},
};
static const GenI64 gen_i64[] = {
    {"seed",              "seed",                                &SDGenParams::seed},
};
static const GenBool gen_bool[] = {
    {"circular_x",        "circular_x",                          &SDGenParams::circular_x},
    {"circular_y",        "circular_y",                          &SDGenParams::circular_y},
    {"flip_mask",         nullptr,                               &SDGenParams::img_flip_mask},
    {"upscale",           nullptr,                               &SDGenParams::upscale},
    {"remove_limits",     nullptr,                               &SDGenParams::remove_limits},
};
static const GenVecStr gen_vec_str[] = {
    {"ref_audios",        nullptr,                               &SDGenParams::vid_ref_audios},
};
static const GenImage gen_image[] = {
    {"init_images",      nullptr,                               &SDGenParams::img_init_image},
    {"mask",             nullptr,                               &SDGenParams::img_mask},
    {"video_start_frame",nullptr,                               &SDGenParams::vid_start_frame},
    {"video_end_frame",  nullptr,                               &SDGenParams::vid_end_frame},
};
static const GenImages gen_images[] = {
    {"extra_images",     nullptr,                               &SDGenParams::img_extra_images},
};

// tracking keys of the rows rewritten/settled outside the table loops
// (must match the `out` key of the row)
static constexpr char TRACK_REF_IMAGE_ARGS[] = "ref_image_args";
static constexpr char TRACK_CFG_SCALE[]      = "sample.guidance.txt_cfg";
static constexpr char TRACK_SAMPLE_STEPS[]   = "sample.steps";
static constexpr char TRACK_WIDTH[]          = "width";
static constexpr char TRACK_HEIGHT[]         = "height";

// settle the shared table fields onto a generation option set: only members
// present in the request are set (absent = library default); rows without an
// sd_opt key (kcpp-only) are skipped
static void settle_gen_tables(sd_opt_t* opt, const SDGenParams& gp, const std::set<std::string>& present)
{
    for (const auto& f : gen_str) {
        if (sd_check_key(present, f.out)) {
            sd_opt_set_string(opt, f.out, (gp.*f.p).c_str());
        }
    }
    for (const auto& f : gen_f32) {
        if (sd_check_key(present, f.out)) {
            sd_opt_set_float(opt, f.out, gp.*f.p);
        }
    }
    for (const auto& f : gen_i32) {
        if (sd_check_key(present, f.out)) {
            sd_opt_set_int(opt, f.out, gp.*f.p);
        }
    }
    for (const auto& f : gen_i64) {
        if (sd_check_key(present, f.out)) {
            sd_opt_set_int(opt, f.out, gp.*f.p);
        }
    }
    for (const auto& f : gen_bool) {
        if (sd_check_key(present, f.out)) {
            sd_opt_set_bool(opt, f.out, gp.*f.p);
        }
    }
}

// resolve the decoded generation parameters in place: apply the model-aware
// clamps, fix the resolution and tiling, decode/route/fit the media, and settle
// the request kind (image vs video, img2img, passthrough). This is the generate
// counterpart of kcpp_resolve_load_parameters. The sd_*_gen_params defaults are
// deliberately not resolved here: they stay in the settle step, next to the
// upstream structs (the API transition tracks them as a set of defined keys).
// Returns an empty string on success, otherwise the user-visible error.
static std::string kcpp_resolve_generation_parameters(SDGenParams& gp, const KcppLoadParams& klp,
                                                      PromptImageRef& prompt_image_ref,
                                                      std::set<std::string>& present)
{
    // ref_image_args: auto-resize prefix, forced edit mode when a preset is given
    if (gp.ref_image_args.find("preset") != std::string::npos) {
        gp.force_image_edit = true;
        if(!sd_is_quiet && sddebugmode==1) {
            printf("ref_image_args=\"%s\", forcing edit mode", gp.ref_image_args.c_str());
        }
    }
    gp.ref_image_args = "resize_before_vae=on" + ((gp.ref_image_args.empty() ? "" : ",") + gp.ref_image_args);
    // always rewritten: settle it even when the request gave no value
    present.insert(TRACK_REF_IMAGE_ARGS);

    gp.loras = gp.lora_map.get_lora_specs();

    gp.req_width = gp.width;
    gp.req_height = gp.height;

    const auto& info = klp.info;
    gp.is_vidgen = info.is_video_model;
    const bool supports_reference_images = (gp.is_vidgen || info.supports_ref_image || info.is_kontext || klp.photomaker_enabled);
    const bool remove_limits = gp.remove_limits;

    // <image:N>/<image:+N> prompt tags reference gp.img_extra_images by position
    // and route the images to the VLM conditioner; images will be removed from
    // the ref_images parameter unless a reference has a '+' tag
    image_tag_refs image_tags;
    bool vision_tags_active = false;
    {
        size_t ref_count = std::min(gp.img_extra_images.size(), (size_t)max_extra_images);
        std::string tag_err;
        if (!collect_image_tags(gp.prompt, gp.negative_prompt, ref_count, image_tags, tag_err)) {
            return "KCPP SD: " + tag_err;
        }
        prompt_image_ref.zero_based = image_tags.zero_based;
        if (image_tags.has_tags) {
            if (klp.has_llm_vision) {
                vision_tags_active = true;
            } else {
                // the conditioner can't see images: replace the tags with plain
                // references (upstream would abort the generation)
                if (!sd_is_quiet && sddebugmode == 1) {
                    printf("\nKCPP SD: replacing <image:...> tags with plain references, the model has no vision encoder\n");
                }
                gp.prompt = strip_image_tags(gp.prompt, image_tags.zero_based);
                gp.negative_prompt = strip_image_tags(gp.negative_prompt, image_tags.zero_based);
            }
        }
    }

    gp.is_img2img = gp.img_init_image.valid();

    if (!remove_limits && info.is_flux1)
    {
        if (gp.cfg_scale != 1.0f) {
            // clamp cfg scale
            if (!sd_is_quiet && sddebugmode) {
                printf("Flux: clamping CFG Scale to 1\n");
            }
            gp.cfg_scale = 1.0f;
            present.insert(TRACK_CFG_SCALE);
        }
    }

    if(!remove_limits && info.is_zimage)
    {
        if(gp.cfg_scale > 4.0f)
        {
            if (!sd_is_quiet && sddebugmode) {
                printf("Z-Image: clamping CFG Scale to 4.0 to preserve quality\n");
            }
            gp.cfg_scale = 4.0f;
            present.insert(TRACK_CFG_SCALE);
        }
    }

    if(info.is_sdxs)
    {
        if(gp.cfg_scale > 1.0f || gp.sample_steps > 1)
        {
            if (!sd_is_quiet && sddebugmode) {
                printf("SDXS: clamping steps and cfg to 1\n");
            }
            gp.cfg_scale = 1.0f;
            gp.sample_steps = 1;
            present.insert(TRACK_CFG_SCALE);
            present.insert(TRACK_SAMPLE_STEPS);
        }
        // A custom sigma list determines the real sampler iteration count and
        // would otherwise bypass the SDXS one-step restriction above. A
        // two-value list already represents one step and remains valid.
        if (gp.custom_sigmas.size() > 2) {
            if (!sd_is_quiet && sddebugmode) {
                printf("SDXS: ignoring custom sigma schedule with more than 1 step\n");
            }
            gp.custom_sigmas.clear();
        }
    }

    bool redirect_single_reference_to_img2img = (
            // we are not doing img2img already
            !gp.is_img2img
            // a single extra image is provided, and mask is NOT provided
            && gp.img_extra_images.size()==1 && !gp.img_mask.valid()
            // the model does not support references natively, or through an edit mode/LoRa
            && !supports_reference_images && !gp.force_image_edit
            // the model can't handle image tags, or the user didn't ask for them
            && !vision_tags_active);

    if(redirect_single_reference_to_img2img)
    {
        gp.is_img2img = true;
        gp.img_init_image = gp.img_extra_images[0];
        gp.img_extra_images.clear();
        if (!sd_is_quiet && sddebugmode == 1) {
            printf("Switching reference image to img2img\n");
        }
    }

    if (gp.is_vidgen && !gp.vid_start_frame.valid() && gp.is_img2img && !gp.img_mask.valid())
    {
        // no explicit video start frame: the img2img input becomes the video start frame
        std::swap(gp.vid_start_frame, gp.img_init_image);
    }

    // limit by image side
    int img_hard_limit = 8192; // "large enough", just to simplify the code
    if (klp.img_hard_limit > 0) {
        img_hard_limit = std::max(std::min(klp.img_hard_limit, img_hard_limit), img_side_min);
    }

    // limit by image area: avoid crashes due to bugs/limitations on certain models
    // a single side can be larger, but width*height are limited by img_soft_limit²
    int img_soft_limit;
    int hard_megapixel_res_limit = 2048; // hard area limit, no matter the config
    if (klp.img_soft_limit <= 0) {
        // default limit is model dependent: ~0.66 megapixel for SD1.5/SD2, 1 megapixel for most models
        img_soft_limit = (info.is_sd1 || info.is_sd2)?832:1024;
    } else {
        // force img_side_min <= limit <= hard_megapixel_res_limit
        img_soft_limit = std::max(std::min(klp.img_soft_limit, hard_megapixel_res_limit), img_side_min);
    }

    // unet is limited to multiples of 64; dit models vary
    int spatial_multiple = info.spatial_multiple;

    sd_fix_resolution(gp.width, gp.height, img_hard_limit, img_soft_limit, spatial_multiple);
    if (gp.req_width != gp.width || gp.req_height != gp.height) {
        printf("\nKCPP SD: Requested dimensions %dx%d changed to %dx%d\n",
            gp.req_width, gp.req_height, gp.width, gp.height);
        present.insert(TRACK_WIDTH);
        present.insert(TRACK_HEIGHT);
    }

    // trigger tiling by image area, the memory used for the VAE buffer is 6656 bytes per image pixel, default 768x768
    gp.dotile = (gp.width*gp.height > klp.tiled_vae_threshold*klp.tiled_vae_threshold);

    gp.vae_tile_size = -1;
    if (gp.dotile) {
        // bump tile size to 1/4 of the configured maximum non-tiled image size
        // (so 1/4 of the memory that would be used for that image)
        const int minimum_tile_size = 256; // from mainline sdcpp
        int new_vae_tile_size = klp.tiled_vae_threshold;
        new_vae_tile_size = new_vae_tile_size / 2;
        new_vae_tile_size = roundup_to(new_vae_tile_size, 64);
        if (new_vae_tile_size > minimum_tile_size) {
            gp.vae_tile_size = new_vae_tile_size;
        }
    }

    //for img2img
    std::vector<KcppImage> legacy_video_imgs;
    int img2imgW = gp.width; //for img2img input
    int img2imgH = gp.height;

    // fit the img2img input (and mask) to the generation size; the mask is
    // completed to a full 255 ("keep everything") mask when not provided
    if (gp.is_img2img)
    {
        // the init image may have been swapped into the video start frame
        if (gp.img_init_image.valid()) {
            gp.img_init_image = gp.img_init_image.resize(img2imgW, img2imgH, 3);
            if (!gp.img_init_image.valid()) {
                return "KCPP SD: load image from memory failed!";
            }
        }
        if (gp.img_mask.valid()) {
            gp.img_mask = gp.img_mask.resize(img2imgW, img2imgH, 1);
            if (!gp.img_mask.valid()) {
                return "KCPP SD: load mask from memory failed!";
            }
            if (gp.img_flip_mask) {
                gp.img_mask = gp.img_mask.flipped_mask();
            }
        } else {
            gp.img_mask = KcppImage::solid(img2imgW, img2imgH, 1, 255);
        }
    }

    if(!gp.img_extra_images.empty())
    {
        if (vision_tags_active) {
            prompt_image_ref.resize(std::min(gp.img_extra_images.size(), (size_t)max_extra_images));
        }
        for(size_t i=0;i<gp.img_extra_images.size() && i<(size_t)max_extra_images;++i)
        {
            const KcppImage& src = gp.img_extra_images.at(i);
            const bool duplicates_video_frame = gp.is_vidgen && info.is_minimaxh3
                && (src.same_as(gp.vid_start_frame) || src.same_as(gp.vid_end_frame));
            KcppImage image;
            if(supports_reference_images||gp.force_image_edit||vision_tags_active)
            {
                if(gp.is_vidgen)
                {
                    if (!duplicates_video_frame || (vision_tags_active && image_tags.mentioned.count(i)))
                    {
                        // the minimax H3 keeps its references at natural size
                        image = info.is_minimaxh3 ? src : src.resize(img2imgW, img2imgH, 3);
                    }
                }
                else if(info.supports_ref_image||gp.force_image_edit)
                {
                    image = src;
                    //kcpp fix: qwen image can stack overflow and crash when ref images exceed
                    // a total res of 512x512 = 262144, so we downscale if that's the case
                    // kcpp edit 2mar2026: this seems to be better now, so limit to 1024x1024 instead
                    int res_lim_crash = 1024 * 1024;
                    if ((int)src.w * (int)src.h > res_lim_crash)
                    {
                        float factor = sqrtf((float)res_lim_crash / ((float)src.w * (float)src.h));
                        uint32_t tgtx = (uint32_t)(src.w * factor);
                        uint32_t tgty = (uint32_t)(src.h * factor);
                        if (!sd_is_quiet && sddebugmode == 1)
                        {
                            printf("\nResized RefImg %ux%u to %ux%u", src.w, src.h, tgtx, tgty);
                        }
                        image = src.resize(tgtx, tgty, 3);
                    }
                }
                else if (info.is_kontext || klp.photomaker_enabled || vision_tags_active)
                {
                    image = src;
                }
            }
            if(!image.valid())
            {
                continue;
            }
            if (vision_tags_active) {
                prompt_image_ref.set(i, image);
                if (image_tags.hidden.count(i)) {
                    continue; // used only as VLM context, kept out of the ref list
                }
            }
            if (duplicates_video_frame || !(supports_reference_images||gp.force_image_edit)) {
                continue; // Keep frame deduplication in the diffusion reference list.
            }
            // version selection for video references happens here, once: only the
            // minimax H3 (Ref2VA) takes them as references, every other video model
            // takes its extra images as start/end frames (legacy_video_imgs). The
            // video settle forwards the IR as-is, without repeating the version
            // check, so this branch must stay exclusive.
            if (gp.is_vidgen) {
                if (info.is_minimaxh3) {
                    gp.reference_imgs.push_back(image);
                } else {
                    legacy_video_imgs.push_back(image);
                }
            }
            else if (info.is_kontext) {
                gp.reference_imgs.push_back(image);
            }
            else if (klp.photomaker_enabled) {
                gp.photomaker_imgs.push_back(image);
            }
            else {
                gp.reference_imgs.push_back(image);
            }
        }

        //ensure prompt has img keyword, otherwise append it
        if(klp.photomaker_enabled)
        {
            if (gp.prompt.find("img") == std::string::npos) {
                gp.prompt += " img";
            } else if (gp.prompt.rfind("img", 0) == 0) {
                // "img" found at the start of the string (position 0), which is not allowed. Add some text before it
                gp.prompt = "person " + gp.prompt;
            }
        }

        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nImageGen References: RefImg=%zu Wan=%zu Photomaker=%zu\n",gp.reference_imgs.size(),legacy_video_imgs.size(),gp.photomaker_imgs.size());
        }
    }

    if (gp.is_vidgen)
    {
        // video start/end frames are fitted to the generation size
        if (gp.vid_start_frame.valid()) {
            gp.vid_start_frame = gp.vid_start_frame.resize(img2imgW, img2imgH, 3);
        }
        if (gp.vid_end_frame.valid()) {
            gp.vid_end_frame = gp.vid_end_frame.resize(img2imgW, img2imgH, 3);
        }
    }

    // reference audio is a video-model feature; the H3 image variant ignores it
    if(info.is_minimaxh3 && gp.is_vidgen && gp.vid_ref_audios.size()>0)
    {
        for(size_t i=0;i<gp.vid_ref_audios.size();++i)
        {
            KcppAudio ref_audio = KcppAudio::from_b64(gp.vid_ref_audios[i]);
            if (!ref_audio.valid()) {
                return "KCPP SD: load reference audio from base64 failed!";
            }
            gp.ref_audios.push_back(ref_audio);
        }
    }

    const bool has_minimax_keyframes = info.is_minimaxh3 && (gp.vid_start_frame.valid() || gp.vid_end_frame.valid() || legacy_video_imgs.size() > 0);
    const bool has_minimax_references = info.is_minimaxh3 && (gp.reference_imgs.size() > 0 || gp.ref_audios.size() > 0);
    if(has_minimax_keyframes && has_minimax_references)
    {
        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nMiniMax-H3: keyframes and Ref2VA references cannot be mixed; using reference media and ignoring keyframes.\n");
        }
        gp.vid_start_frame = KcppImage{};
        gp.vid_end_frame = KcppImage{};
        legacy_video_imgs.clear();
    }

    // the non-minimax video references become the start/end frames when no
    // explicit frame was given
    if (legacy_video_imgs.size() >= 2) {
        if(!gp.vid_start_frame.valid()) {
            gp.vid_start_frame = legacy_video_imgs[0];
        }
        if(!gp.vid_end_frame.valid()) {
            gp.vid_end_frame = legacy_video_imgs[1];
        }
    } else if (legacy_video_imgs.size() == 1) {
        if(!gp.vid_start_frame.valid() && !gp.vid_end_frame.valid()) {
            gp.vid_start_frame = legacy_video_imgs[0];
        }
    }

    // a passthrough request returns the input image instead of generating
    gp.is_passthrough = (gp.sample_steps<=1 && gp.strength<=0 && gp.is_img2img && gp.vid_req_frames<=1 && gp.img_extra_images.size()==0);
    if (gp.is_passthrough && !gp.vid_start_frame.valid() && !gp.img_init_image.valid()) {
        return "KCPP SD: load image from memory failed!";
    }

    return {};
}

sd_generation_outputs sdtype_generate(const sd_generation_inputs inputs)
{
    TryLocker no_concurrency{g_sd_op_mutex, std::try_to_lock};
    if (!no_concurrency.owns_lock()) {
        return sd_generation.error("Error: Generate: another operation is in progress");
    }

    struct CleanupInfoOnExit {
        ~CleanupInfoOnExit() {
            {
                std::lock_guard<std::mutex> lock(geninfo.mux);
                geninfo.gendata.status = 0;
                geninfo.gendata.preview = "";
                geninfo.preview_requested = false;
                geninfo.preview_enabled = false;
                geninfo.aborted = false;
            }
            set_preview_images(0);
        }
    } cleanup_info_on_exit;

    if(!sd_params)
    {
        return sd_generation.error("Warning: KCPP image generation not initialized!");
    }

    auto& klp = sd_params->kcpp_load_params;

    // decode the request into a local parameter representation
    SDGenParams gp;
    gp.lora_map = klp.lora_map;

    // tracking keys of the fields present in the request payload: the settle
    // forwards only these (absent = library default)
    std::set<std::string> present;
    {
        SdErrors errors;
        nlohmann::json payload = sdjson_parse_payload(inputs.params_json, errors);
        if (payload.is_object()) {
            // one strict getter per field row
            for (const auto& f : gen_str) {
                std::string s;
                if (sdjson_get(s, f.in, payload, errors)) {
                    gp.*f.p = f.dec ? f.dec(s) : s;
                    sd_track_key(present, f.out);
                }
            }
            for (const auto& f : gen_f32)
                if (sdjson_get(gp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : gen_i32)
                if (sdjson_get(gp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : gen_i64)
                if (sdjson_get(gp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : gen_bool)
                if (sdjson_get(gp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : gen_vec_str)
                if (sdjson_get(gp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : gen_image)
                if (sdjson_get_image(gp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);
            for (const auto& f : gen_images)
                if (sdjson_get_images(gp.*f.p, f.in, payload, errors)) sd_track_key(present, f.out);

            // sample_method / scheduler: JSON string -> enum on decode
            std::string sample_method, scheduler;
            sdjson_get(sample_method, "sample_method", payload, errors);
            sdjson_get(scheduler, "scheduler", payload, errors);
            gp.sample_method = str_to_sample_method(sample_method.c_str());
            gp.scheduler = str_to_scheduler(scheduler.c_str());

            if (klp.lora_dynamic) {
                sdjson_get_loras("loras", payload, errors, gp.lora_map);
            } else {
                errors.accepted_keys.insert("loras");
            }

            sdjson_get(gp.custom_sigmas, "custom_sigmas", payload, errors);

            if (sddebugmode) {
                sdjson_warn_unknown_keys(payload, errors);
            }
        }
        if (!errors.empty()) {
            printf("\nError: KCPP SD generate: %s\n", errors.message().c_str());
            return sd_generation.error(errors.message().c_str());
        }
    }

    const auto& info = klp.info;

    {
        std::lock_guard<std::mutex> lock(geninfo.mux);
        geninfo.start_time = std::chrono::steady_clock::now();
        geninfo.steps = gp.sample_steps;
        geninfo.preview_requested = false;
        geninfo.preview_enabled = false;
        geninfo.aborted = false;
        geninfo.gendata = {};
        geninfo.gendata.status = 1;
        set_preview_images(1);
    }

    sd_ctx_t* sd_ctx = sd_params->sd_ctx; // protected by no_concurrency mutex

    PromptImageRef prompt_image_ref;
    std::string resolve_error = kcpp_resolve_generation_parameters(gp, klp, prompt_image_ref, present);
    if (!resolve_error.empty()) {
        return sd_generation.error(resolve_error.c_str());
    }
    // strength applies to image-to-image and to every video request; a txt2img
    // request leaves it at the library default (as the per-kind settle gates did)
    if (!gp.is_img2img && !gp.is_vidgen)
        present.erase("strength");

    struct CleanupImageCallbackOnExit {
        sd_ctx_t* sd_ctx;
        ~CleanupImageCallbackOnExit() {
            sd_set_prompt_image_callback(sd_ctx, nullptr, nullptr);
        }
    } cleanup_image_callback_onexit{sd_ctx};
    sd_set_prompt_image_callback(sd_ctx, prompt_image_from_refs, &prompt_image_ref);

    if(!gp.is_passthrough && !sd_is_quiet && sddebugmode==1) {
        if (gp.loras.size() > 0) {
            printf("Applying LoRAs:\n");
            for(size_t i=0;i<gp.loras.size();++i)
            {
                printf("  %s @ %.3f\n", gp.loras[i].path, gp.loras[i].multiplier);
            }
        }
    }

    KcppAudio generated_audio;
    KcppImages result_images;
    int generated_num_results = -1; // -1 failure, 0 passthrough
    std::vector<nlohmann::json> jsonresults; // the library's per-result generation info

    // the timestamped start line lives in each branch: it names what is about to
    // run, and passthrough gets one too, since the upscaler may still run
    if (gp.is_passthrough)
    {
        // no generation at all: the output block below returns the input image
        // (upscaled when requested) as-is
        generated_num_results = 0;
        printf("\n[%s] No generation triggered, passthrough mode.\n", get_timestamp_str().c_str());
        fflush(stdout);
    }
    else if(gp.is_vidgen)
    {
        std::string ts = get_timestamp_str();
        if(!sd_is_quiet)
        {
            printf("\n[%s] Generating Video (%d steps)\n",ts.c_str(),gp.sample_steps);
        }else{
            printf("\n[%s] Generating (%d st.)\n",ts.c_str(),gp.sample_steps);
        }

        fflush(stdout);

        // settle the resolved parameters into the video generation options;
        // only request-present fields are set (absent = library default)
        sd_result_t* vid_result = nullptr;
        {
            auto* opt = sd_vid_gen_opt_create();
            settle_gen_tables(opt, gp, present);
            if (gp.vid_req_frames > 0) {
                sd_opt_set_int(opt, "video_frames", gp.vid_req_frames);
            }
            if (gp.vid_fps > 0) {
                sd_opt_set_int(opt, "fps", gp.vid_fps);
            }
            if (!gp.custom_sigmas.empty()) {
                sd_opt_set_floats(opt, "sample.sigmas", gp.custom_sigmas.data(), gp.custom_sigmas.size());
            }
            if (gp.sample_method != sample_method_t::SAMPLE_METHOD_COUNT) {
                sd_opt_set_string(opt, "sample.method", sd_sample_method_name(gp.sample_method));
            }
            if (gp.scheduler != scheduler_t::SCHEDULER_COUNT) {
                sd_opt_set_string(opt, "sample.scheduler", sd_scheduler_name(gp.scheduler));
            }
            sd_opt_set_bool(opt, "vae_tiling.enabled", gp.dotile);
            if (gp.vae_tile_size > 0) {
                sd_opt_set_int(opt, "vae_tiling.tile_size_w", gp.vae_tile_size);
                sd_opt_set_int(opt, "vae_tiling.tile_size_h", gp.vae_tile_size);
            }
            if (gp.dotile) {
                sd_opt_set_bool(opt, "vae_tiling.temporal_tiling", true);
            }
            parse_cache_options(opt, gp.cache_mode, gp.cache_options);
            // the library tracks previously applied LoRAs and applies the
            // difference at each gen: an empty list unapplies all
            sd_opt_set_loras(opt, "loras", gp.loras.data(), gp.loras.size());
            set_images_opt(opt, "ref_images", gp.reference_imgs);
            set_audios_opt(opt, "ref_audios", gp.ref_audios);
            set_image_opt(opt, "init_image", gp.vid_start_frame);
            set_image_opt(opt, "end_image", gp.vid_end_frame);

            sd_status_t status = sd_generate_video(sd_ctx, opt, &vid_result);
            sd_opt_free(opt); // the option set is consumed (or rejected) by now
            if (status != SD_STATUS_OK || vid_result == nullptr) {
                if (vid_result != nullptr) {
                    sd_result_free(vid_result);
                    vid_result = nullptr;
                }
            }
        }

        if(!sd_is_quiet && sddebugmode==1)
        {
            std::stringstream ss;
            ss  << "\nVID PROMPT:" << gp.prompt
                << "\nNPROMPT:"   << gp.negative_prompt
                << "\nCLPSKP:"   << gp.clip_skip
                << "\nSIZE:"     << gp.width << "x" << gp.height
                << "\nSTEP:"     << gp.sample_steps
                << "\nSEED:"     << gp.seed
                << "\nSTRENGTH:" << gp.strength
                << "\nFRAMES:"   << gp.vid_req_frames
                << "\nINIT_IMGS:" << ((gp.vid_start_frame.valid() ? 1 : 0) + (gp.vid_end_frame.valid() ? 1 : 0))
                << "\nREF_IMGS:" << gp.reference_imgs.size()
                << "\nREF_AUDIOS:" << gp.ref_audios.size()
                << "\n\n";
            printf("%s", ss.str().c_str());
        }

        fflush(stdout);

        generated_audio = KcppAudio{};
        int output_fps = gp.vid_fps;
        if (vid_result != nullptr) {
            const sd_video_t* vid = sd_result_get_video(vid_result);
            if (vid != nullptr && vid->data != nullptr) {
                generated_num_results = (int) sd_tensor_dim(vid->data, 3);
                for (int i = 0; i < generated_num_results; ++i) {
                    result_images.push_back(KcppImage::from_tensor(vid->data, i));
                }
                output_fps = vid->fps;
                generated_audio = (vid->audio.data != nullptr) ? KcppAudio::from_audio(&vid->audio) : KcppAudio{};
            }
            jsonresults.push_back(parse_generation_info(sd_result_get_generation_info(vid_result, 0)));
            sd_result_free(vid_result);
            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("\nRequested Vid Frames: %d, Generated Vid Frames: %d\n",gp.vid_req_frames, generated_num_results);
                if (output_fps != gp.vid_fps) {
                    printf("\nRequested FPS: %d, Generated FPS: %d\n", gp.vid_fps, output_fps);
                }
            }
            gp.vid_fps = output_fps;
        }
    }
    else
    {
        std::string ts = get_timestamp_str();
        if(!sd_is_quiet)
        {
            printf("\n[%s] Generating Image (%d steps)\n",ts.c_str(),gp.sample_steps);
        }else{
            printf("\n[%s] Generating (%d st.)\n",ts.c_str(),gp.sample_steps);
        }

        fflush(stdout);

        // settle the resolved parameters into the image generation options;
        // only request-present fields are set (absent = library default)
        sd_result_t* img_result = nullptr;
        {
            auto* opt = sd_img_gen_opt_create();
            settle_gen_tables(opt, gp, present);
            if (gp.is_img2img) {
                // the input and mask were fitted (and the mask completed/flipped)
                // while resolving the parameters
                set_image_opt(opt, "init_image", gp.img_init_image);
                set_image_opt(opt, "mask_image", gp.img_mask);
            }
            if (!gp.custom_sigmas.empty()) {
                sd_opt_set_floats(opt, "sample.sigmas", gp.custom_sigmas.data(), gp.custom_sigmas.size());
            }
            if (gp.sample_method != sample_method_t::SAMPLE_METHOD_COUNT) {
                sd_opt_set_string(opt, "sample.method", sd_sample_method_name(gp.sample_method));
            }
            if (gp.scheduler != scheduler_t::SCHEDULER_COUNT) {
                sd_opt_set_string(opt, "sample.scheduler", sd_scheduler_name(gp.scheduler));
            }
            sd_opt_set_bool(opt, "vae_tiling.enabled", gp.dotile);
            if (gp.vae_tile_size > 0) {
                sd_opt_set_int(opt, "vae_tiling.tile_size_w", gp.vae_tile_size);
                sd_opt_set_int(opt, "vae_tiling.tile_size_h", gp.vae_tile_size);
            }
            if (gp.dotile) {
                sd_opt_set_bool(opt, "vae_tiling.temporal_tiling", true);
            }
            parse_cache_options(opt, gp.cache_mode, gp.cache_options);
            // the library tracks previously applied LoRAs and applies the
            // difference at each gen: an empty list unapplies all
            sd_opt_set_loras(opt, "loras", gp.loras.data(), gp.loras.size());
            set_images_opt(opt, "ref_images", gp.reference_imgs);
            set_images_opt(opt, "photomaker.id_images", gp.photomaker_imgs);

            sd_status_t status = sd_generate_image(sd_ctx, opt, &img_result);
            sd_opt_free(opt); // the option set is consumed (or rejected) by now
            if (status != SD_STATUS_OK || img_result == nullptr) {
                if (img_result != nullptr) {
                    sd_result_free(img_result);
                    img_result = nullptr;
                }
            }
        }

        // txt2img and img2img differ only by what the settle put in
        // init_image/mask, so both use the same call
        if (img_result != nullptr) {
            generated_num_results = (int) sd_result_get_image_count(img_result);
            for (int i = 0; i < generated_num_results; ++i) {
                result_images.push_back(KcppImage::from_tensor(sd_result_get_image(img_result, i)));
                jsonresults.push_back(parse_generation_info(sd_result_get_generation_info(img_result, i)));
            }
            sd_result_free(img_result);
        }
    }

    if (generated_num_results < 0) {
        if (info.is_qwenimg && gp.reference_imgs.size() > 0) {
            printf("\nKCPP SD: If using Qwen Image 2.1, editing requires Qwen3-VL vision mmproj!\n");
        }
        return sd_generation.error("KCPP SD generate failed!");
    }

    bool isanim = (gp.vid_req_frames>1 && generated_num_results>1 && gp.is_vidgen);
    nlohmann::json jsoninfo = nlohmann::json::object();
    std::string gen_data;
    std::string gen_data2;
    std::string final_frame_data;

    auto upscaler_ctx = gp.upscale ? sd_params->upscaler_ctx : nullptr;

    if (gp.is_passthrough)
    {
        // passthrough ran no generation: don't report the parameters as if it did,
        // only echo the prompt, which may describe the imported image
        if (!gp.prompt.empty())
            jsoninfo["prompt"] = gp.prompt;

        // the start frame (the swapped-in img2img input) if present, else the input
        const KcppImage& pimg = gp.vid_start_frame.valid() ? gp.vid_start_frame : gp.img_init_image;
        if(upscaler_ctx != nullptr)
        {
            printf("Upscaling original image (passthrough)...\n");
            gen_data = upscale_image_to_png_base64(upscaler_ctx, pimg);
        }
        else {
            gen_data = raw_image_to_png_base64(pimg);
        }
    }
    else if (isanim)
    {
        jsoninfo = build_result_jsoninfo(jsonresults);
        //if multiframe, make a video
        if (!result_images.empty() && result_images.at(0).valid())
        {
            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("\nSaving video buffer, VIDEO_OUTPUT_TYPE=%d...",gp.vid_output_type);
            }
            uint8_t * out_data = nullptr;
            uint8_t * out_data2 = nullptr;
            size_t out_len = 0;
            size_t out_len2 = 0;
            int status = 0;
            int status2 = 0;

            const bool output_gif  = gp.vid_output_type == 0 || gp.vid_output_type == 2;
            const bool output_webp = gp.vid_output_type == 3 || gp.vid_output_type == 4;
            const bool output_avi  = gp.vid_output_type == 1 || gp.vid_output_type == 2 || gp.vid_output_type == 4;

            if(output_gif)
            {
                status = create_gif_buf_msf(result_images, gp.vid_fps, &out_data,&out_len);
            }
            else if(output_webp)
            {
                status = create_animated_webp_buf(result_images, gp.vid_fps, &out_data,&out_len);
            }
            if(output_avi)
            {
                status2 = create_mjpg_avi_membuf(result_images, gp.vid_fps, 40, &out_data2,&out_len2, &generated_audio);
            }

            if(generated_num_results>1)
            {
                final_frame_data = raw_image_to_png_base64(result_images.at(generated_num_results-1));
            }

            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("Video Output Sizes: GIF=%zu WEBP=%zu AVI=%zu\n",
                       output_gif ? out_len : 0,
                       output_webp ? out_len : 0,
                       out_len2);
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
    }
    else
    {
        jsoninfo = build_result_jsoninfo(jsonresults);
        for (int i = 0; i < generated_num_results; i++)
        {
            const KcppImage& result_image = result_images.at(i);
            if (!result_image.valid()) {
                continue;
            }
            std::string meta_image_info = get_image_info_text(jsonresults[i], i);
            if(upscaler_ctx != nullptr && gp.upscale)
            {
                printf("Upscaling output image...\n");
                gen_data = upscale_image_to_png_base64(upscaler_ctx, result_image, meta_image_info);
            } else {
                gen_data = raw_image_to_png_base64(result_image, meta_image_info);
            }
            jsoninfo["infotexts"][i] = meta_image_info;
            jsoninfo["all_seeds"][i] = jsoninfo.value("seed", gp.seed) + i;
            jsoninfo["all_prompts"][i] = jsoninfo["prompt"];
            jsoninfo["all_negative_prompts"][i] = jsoninfo.value("negative_prompt", "");
        }
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

static void step_callback(int step, int frame_count, const sd_tensor_t* const* frames, bool is_noisy, void* data)
{
    step = step < 0 ? -step : step;
    std::lock_guard<std::mutex> lock(geninfo.mux);
    if (geninfo.aborted) {
        return;
    }
    double step_time = get_time_delta(geninfo.start_time);

    bool should_encode_preview = false;
    {
        if (geninfo.gendata.status <= 1) {
            geninfo.gendata.status = 2;
            if (geninfo.preview_requested && !geninfo.preview_enabled) {
                geninfo.preview_enabled = true;
                set_preview_images(2);
            } else {
                set_preview_images(0);
            }
        }
        geninfo.gendata.step = is_noisy ? 0 : step;
        geninfo.gendata.step_time = step_time;
        should_encode_preview = !is_noisy && geninfo.preview_enabled;
    }

    if (!should_encode_preview) {
        return;
    }

    std::string preview;
    if (frames != nullptr && frame_count > 0) {
        constexpr uint32_t preview_max_dimension = 128;
        KcppImages preview_images;
        std::vector<std::vector<uint8_t>> resized_image_data(static_cast<size_t>(frame_count));
        bool resize_ok = true;

        for (int i = 0; i < frame_count; ++i) {
            // copy the frame out of the borrowed tensor; the preview is
            // encoded from the copy (the tensor is valid only in this call)
            KcppImage source = KcppImage::from_tensor(frames[i]);
            if (!source.valid()) {
                resize_ok = false;
                break;
            }
            if (source.w <= preview_max_dimension && source.h <= preview_max_dimension) {
                preview_images.push_back(source);
                continue;
            }

            uint32_t resized_width;
            uint32_t resized_height;
            if (source.w >= source.h) {
                resized_width = preview_max_dimension;
                resized_height = std::max(1u, static_cast<uint32_t>(
                    static_cast<uint64_t>(source.h) * preview_max_dimension / source.w));
            } else {
                resized_height = preview_max_dimension;
                resized_width = std::max(1u, static_cast<uint32_t>(
                    static_cast<uint64_t>(source.w) * preview_max_dimension / source.h));
            }

            auto& pixels = resized_image_data[static_cast<size_t>(i)];
            pixels.resize(static_cast<size_t>(resized_width) * resized_height * source.c);
            if (!stbir_resize_uint8(source.px->data(), source.w, source.h, 0,
                                    pixels.data(), resized_width, resized_height, 0, source.c)) {
                resize_ok = false;
                break;
            }
            auto v = std::make_shared<std::vector<uint8_t>>(std::move(pixels));
            preview_images.push_back(KcppImage{resized_width, resized_height, source.c, std::move(v)});
        }

        if (resize_ok) {
            if (frame_count == 1) {
                preview = raw_image_to_png_base64(preview_images.at(0));
            } else {
                uint8_t * out_data = nullptr;
                size_t out_len = 0;
                if (create_gif_buf_msf(preview_images, 16, &out_data,&out_len) == 0 && out_data && out_len > 0) {
                    preview = kcpp_base64_encode(out_data, out_len);
                }
                if (out_data) {
                    free(out_data);
                }
            }
        }
    }

    {
        geninfo.gendata.step = step;
        geninfo.gendata.step_time = step_time;
        geninfo.gendata.preview = preview;
    }
}

void sdtype_request_ongoing_generation_preview()
{
    std::lock_guard<std::mutex> lock(geninfo.mux);
    if (!geninfo.aborted && geninfo.gendata.status != 0 && !geninfo.preview_requested) {
        geninfo.preview_requested = true;
    }
}

sd_generation_outputs sdtype_upscale(const sd_upscale_inputs inputs)
{
    TryLocker no_concurrency{g_sd_op_mutex, std::try_to_lock};
    if (!no_concurrency.owns_lock()) {
        return sd_generation.error("Error: Upscale: another operation is in progress");
    }
    auto upscaler_ctx = sd_params ? sd_params->upscaler_ctx : nullptr;

    sd_generation.reset();

    if(!upscaler_ctx)
    {
        return sd_generation.error("Warning: KCPP image upscaling not initialized!");
    }

    struct {
        std::string rawb64;
        int upscaling_resize = 2;
    } params;
    {
        SdErrors errors;
        nlohmann::json payload = sdjson_parse_payload(inputs.params_json, errors);
        if (payload.is_object()) {
            sdjson_get(params.rawb64, "init_images", payload, errors);
            sdjson_get(params.upscaling_resize, "upscaling_resize", payload, errors);
            if (sddebugmode) {
                sdjson_warn_unknown_keys(payload, errors);
            }
        }
        if (!errors.empty()) {
            const std::string msg = "KCPP SD upscale: " + errors.message();
            return sd_generation.error(msg.c_str());
        }
    }

    KcppImage source_img = KcppImage::from_b64(params.rawb64);
    std::string result;
    if(source_img.valid())
    {
        result = upscale_image_to_png_base64(upscaler_ctx, source_img);
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

    static thread_local std::string recent_info = j.dump();
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
    j["steps"] = steps;
    j["elapsed_time"] = elapsed_time;
    j["step"] = gendata.step;
    j["step_time"] = gendata.step_time;
    if (gendata.status == 1)
        j["status"] = "conditioning";
    else if (gendata.status == 2)
        j["status"] = "diffusing";
    else if (gendata.status == 3)
        j["status"] = "decoding";
    else if (gendata.status == 0)
        j["status"] = "idle";
    else
        j["status"] = "UNKNOWN";
    j["preview"] = gendata.preview;

    static thread_local std::string recent_info;
    recent_info = j.dump();
    sd_info_outputs output;
    output.status = 0;
    output.data = recent_info.c_str();
    return output;
}

