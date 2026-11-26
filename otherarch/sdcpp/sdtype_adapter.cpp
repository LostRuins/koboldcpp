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

#include "otherarch/utils.h"
#include "otherarch/webp_image.h"
#include "model_adapter.h"
#include "model_file_probe.h"

#include "stable-diffusion.h"
#include "src/kcpp_sd_extensions.h"
#include "src/conditioning/prompt_input.h"
#include "sdcpp_gguf_tokenizer.h"
#include "src/core/util.h"
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
static int create_animated_webp_buf_from_sd_images(sd_image_t* images,
                                                    int num_images,
                                                    int fps,
                                                    uint8_t** out_data,
                                                    size_t* out_len) {
    if (out_data == nullptr || out_len == nullptr) {
        return -1;
    }
    *out_data = nullptr;
    *out_len  = 0;

    if (images == nullptr || num_images <= 0 || fps <= 0 ||
        images[0].data == nullptr || images[0].width == 0 || images[0].height == 0 ||
        images[0].width > WEBP_MAX_DIMENSION || images[0].height > WEBP_MAX_DIMENSION) {
        fprintf(stderr, "Invalid animated WebP input.\n");
        return -1;
    }

    const int width  = static_cast<int>(images[0].width);
    const int height = static_cast<int>(images[0].height);

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
    for (int i = 0; i < num_images; ++i) {
        const sd_image_t& image = images[i];
        if (image.data == nullptr || static_cast<int>(image.width) != width ||
            static_cast<int>(image.height) != height ||
            (image.channel != 1 && image.channel != 3 && image.channel != 4)) {
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
        if (image.channel == 1) {
            const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
            rgb_buffer.resize(pixel_count * 3);
            for (size_t p = 0; p < pixel_count; ++p) {
                rgb_buffer[p * 3 + 0] = image.data[p];
                rgb_buffer[p * 3 + 1] = image.data[p];
                rgb_buffer[p * 3 + 2] = image.data[p];
            }
            imported = WebPPictureImportRGB(&picture, rgb_buffer.data(), width * 3);
        } else if (image.channel == 4) {
            imported = WebPPictureImportRGBA(&picture, image.data, width * 4);
        } else {
            imported = WebPPictureImportRGB(&picture, image.data, width * 3);
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
        (static_cast<int64_t>(num_images) * 1000 + fps / 2) / fps);
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

    std::string get_lora_meta() {
        std::stringstream lora_meta;
        lora_meta << std::setprecision(6);
        for (const auto & lora: items) {
            float multiplier = check_small_mult(lora.second);
            if (multiplier != 0.f) {
                std::string lora_name = std::filesystem::path(lora.first).stem().string();
                lora_meta << "<lora:" << lora_name << ":" << multiplier << ">";
            }
        }
        return lora_meta.str();
    }

};


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
    int n_threads = 0;    // if -1 use physical cores
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
    // tracks loading state that does not have a direct sd_ctx_params_t mapping

    model_info info {};

    // LoRA
    LoraMap lora_map;
    bool lora_dynamic = false;
    bool lora_cache = false;

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
// settled into sd_img_gen_params_t / sd_vid_gen_params_t before the call.
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
    std::string lora_meta;

    bool upscale = false;

    // state tracking
    bool force_image_edit = false;
    int req_width  = 512;
    int req_height = 512;

    struct {
        std::string init_image;
        std::string mask;
        bool flip_mask = false;
        std::vector<std::string> extra_images;
    } img;

    struct {
        std::string start_frame;
        std::string end_frame;
        std::vector<std::string> ref_audios;
        int req_frames   = 0;
        int output_type  = 0;
        int fps          = 0;
    } vid;
};

struct SDParams {
    SDLoadParams load_params;
    KcppLoadParams kcpp_load_params;

    // owned sd_* resources of the loaded model, released here
    sd_ctx_t* sd_ctx = nullptr;
    upscaler_ctx_t* upscaler_ctx = nullptr;

    ~SDParams() {
        // only runs on model reload: the sd_params global is a raw pointer,
        // so the model is intentionally leaked at process exit
        if (sd_ctx != nullptr) {
            free_sd_ctx(sd_ctx);
            sd_ctx = nullptr;
        }
        if (upscaler_ctx != nullptr) {
            free_upscaler_ctx(upscaler_ctx);
            upscaler_ctx = nullptr;
        }
    }
};

//shared
int total_img_gens = 0;

//global static vars for SD
static std::mutex g_sd_params_mux;
// raw pointer on purpose: a static unique_ptr would destroy the model during
// process exit, when free_sd_ctx()/free_upscaler_ctx() may touch global state
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
static void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data);

// 0 disable, 1 initial, 2 denoised
// the first noisy call is used to detect the inference phase
static void set_preview_images(int enable) {
    bool denoised = (enable == 2);
    bool noisy = (enable == 1);
    sd_set_preview_callback(step_callback, PREVIEW_PROJ, 1, denoised, noisy, nullptr);
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

struct PromptImageRef {
    // per-request pool of reference images (original request order) served by
    // prompt_image_from_refs for <image:N> prompt tags
    std::vector<sd_image_t> image_pool;
    bool zero_based = false;
};

// callback registered per-generation: resolves
// <image:N> / <image:+N> prompt tags to the reference image pool
static bool prompt_image_from_refs(const char* source, void* data, sd_image_t* image_out)
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
    const sd_image_t& image = rdata->image_pool[idx];
    if (!image.data) {
        return false;
    }
    *image_out = image;
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
    sd_executable_path = sd_get_u8path(SAFE_STR(inputs.executable_path));

    auto new_params = std::make_unique<SDParams>();
    SDLoadParams& lp = new_params->load_params;
    KcppLoadParams& klp = new_params->kcpp_load_params;

    lp.model_path = SAFE_STR(inputs.model_filename);
    lp.vae_path = SAFE_STR(inputs.vae_filename);
    lp.audio_vae_path = SAFE_STR(inputs.audio_vae_filename);
    lp.t5xxl_path = SAFE_STR(inputs.llm_filename);
    lp.clip_l_path = SAFE_STR(inputs.clip1_filename);
    lp.clip_g_path = SAFE_STR(inputs.clip2_filename);
    lp.photo_maker_path = SAFE_STR(inputs.photomaker_filename);
    klp.upscaler_filename = SAFE_STR(inputs.upscaler_filename);
    klp.tiled_vae_threshold = inputs.tiled_vae_threshold;
    klp.tiled_vae_threshold = (klp.tiled_vae_threshold > 8192 ? 8192 : klp.tiled_vae_threshold);
    klp.tiled_vae_threshold = (klp.tiled_vae_threshold <= 0 ? 8192 : klp.tiled_vae_threshold); //if negative dont tile
    klp.img_hard_limit = inputs.img_hard_limit;
    klp.img_soft_limit = inputs.img_soft_limit;

    lp.backend = SAFE_STR(inputs.backend);
    lp.params_backend = SAFE_STR(inputs.params_backend);
    lp.split_mode = SAFE_STR(inputs.split_mode);
    lp.conv_direct_mode = SAFE_STR(inputs.conv_direct_mode);
    lp.n_threads = inputs.threads; //if -1 use physical cores
    lp.enable_mmap = inputs.use_mmap;
    lp.auto_fit = inputs.auto_fit;
    klp.quant = inputs.quant;
    if (klp.quant > 0) {
        lp.wtype = (klp.quant==1?SD_TYPE_Q8_0:SD_TYPE_Q4_0);
    }
    klp.flash_attention = inputs.flash_attention;
    klp.taesd = inputs.taesd;
    lp.max_vram = SAFE_STR(inputs.max_vram);

    for(int i=0;i<inputs.lora_len;++i) {
        klp.lora_map.add_lora(inputs.lora_filenames[i], inputs.lora_multipliers[i]);
    }

    printf("\nImageGen Init - Load Model: %s\n", lp.model_path.c_str());

    { // LoRA parameters and initial list
        lp.lora_apply_mode = LORA_APPLY_AT_RUNTIME;
        klp.lora_dynamic = false;

        int lora_apply_mode = inputs.lora_apply_mode;
        if(lora_apply_mode >= 1 && lora_apply_mode <= 2) {
            lp.lora_apply_mode = (lora_apply_mode_t)lora_apply_mode;
        }
        else {
            // bit 3: LoRAs can be changed dynamically
            // bit 4: cache the initial LoRA list in VRAM
            klp.lora_dynamic = !!(lora_apply_mode & (1<<3));
            klp.lora_cache   = klp.lora_dynamic && !!(lora_apply_mode & (1<<4));
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

    sd_ctx_params_t params = {};
    sd_ctx_params_init(&params);

    params.rng_type = CUDA_RNG;
    params.model_args = "chroma_use_dit_mask=true";
    params.eager_load = true; //klp should preload everything

    params.n_threads = lp.n_threads;
    params.wtype = lp.wtype;
    params.conv_direct_mode = lp.conv_direct_mode.c_str();
    params.max_vram = lp.max_vram.c_str();
    params.enable_mmap = lp.enable_mmap;
    params.backend = lp.backend.c_str();
    params.params_backend = lp.params_backend.c_str();
    params.split_mode = lp.split_mode.c_str();
    params.auto_fit = lp.auto_fit;
    params.lora_apply_mode = lp.lora_apply_mode;
    params.model_path = lp.model_path.c_str();
    params.clip_l_path = lp.clip_l_path.c_str();
    params.clip_g_path = lp.clip_g_path.c_str();
    params.t5xxl_path = lp.t5xxl_path.c_str();
    params.diffusion_model_path = lp.diffusion_model_path.c_str();
    params.vae_path = lp.vae_path.c_str();
    params.audio_vae_path = lp.audio_vae_path.c_str();
    params.taesd_path = lp.taesd_path.c_str();
    params.photo_maker_path = lp.photo_maker_path.c_str();
    params.tokenizer = lp.tokenizer.c_str();
    params.clip_vision_path = lp.clip_vision_path.c_str();
    params.llm_path = lp.llm_path.c_str();
    params.llm_vision_path = lp.llm_vision_path.c_str();
    params.embeddings_connectors_path = lp.embeddings_connectors_path.c_str();
    params.uncond_diffusion_model_path = lp.uncond_diffusion_model_path.c_str();
    params.tensor_type_rules = lp.tensor_type_rules.c_str();
    params.flash_attn_mode = lp.flash_attn_mode.c_str();

    if(sddebugmode==1)
    {
        const char* tokenizer = params.tokenizer;
        if (tokenizer != nullptr && *tokenizer == '{') {
            params.tokenizer = "<in-memory tokenizer JSON>";
        }
        char* buf = sd_ctx_params_to_str(&params);
        params.tokenizer = tokenizer;
        if(buf)
        {
            printf("\n%s\n", buf);
            free(buf);
        }
    }

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

    new_params->sd_ctx = new_sd_ctx(&params);

    if (new_params->sd_ctx == NULL) {
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
        new_params->upscaler_ctx = new_upscaler_ctx(klp.upscaler_filename.c_str(),
                                        lp.conv_direct_mode == "on",
                                        params.n_threads,
                                        upscale_tile_size,
                                        params.backend,
                                        params.params_backend);

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

static std::string friendly_model_name(std::filesystem::path model_path) {
    return model_path.filename().string();
}

static std::string get_scheduler_name(scheduler_t scheduler, bool as_sampler_suffix = false)
{
    if (scheduler == scheduler_t::SCHEDULER_COUNT) {
        return as_sampler_suffix ? "" : "default";
    } else {
        std::string prefix = as_sampler_suffix ? " " : "";
        return prefix + sd_scheduler_name(scheduler);
    }
}

static std::string get_image_params(const sd_img_gen_params_t & params, const std::string& lora_meta, int seed_offset) {
    std::string model = sd_params->load_params.model_path;
    if (model.empty())
        model = sd_params->load_params.diffusion_model_path;
    model = friendly_model_name(model);
    std::stringstream ss;
    ss << std::setprecision(3)
        <<    "Prompt: " << params.prompt << lora_meta
        << " | NegativePrompt: " << params.negative_prompt
        << " | Steps: " << params.sample_params.sample_steps
        << " | CFGScale: " << params.sample_params.guidance.txt_cfg
        << " | Guidance: " << params.sample_params.guidance.distilled_guidance
        << " | Seed: " << (params.seed + seed_offset)
        << " | Size: " << params.width << "x" << params.height
        << " | Sampler: " << sd_sample_method_name(params.sample_params.sample_method)
        << get_scheduler_name(params.sample_params.scheduler, true);
    if (params.sample_params.eta != -1.0f)
        ss << "| Eta: " << params.sample_params.eta;
    if (params.sample_params.shifted_timestep != 0)
        ss << "| Timestep Shift: " << params.sample_params.shifted_timestep;
    if (params.sample_params.flow_shift > 0.f && params.sample_params.flow_shift != INFINITY)
        ss << "| Flow Shift: " << params.sample_params.flow_shift;
    ss  << " | Clip skip: " << params.clip_skip
        << " | Model: " << model;
    std::string vae = sd_params->load_params.vae_path;
    if (!vae.empty())
        ss << " | VAE: " << friendly_model_name(vae);
    std::string tae = sd_params->load_params.taesd_path;
    if (!tae.empty())
        ss << " | TAE: " << friendly_model_name(tae);
    ss << " | Version: KoboldCpp";
    return ss.str();
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

static enum sample_method_t sampler_from_name(const std::string& sampler)
{
    // all lowercase
    enum sample_method_t result = str_to_sample_method(sampler.c_str());
    if (result != sample_method_t::SAMPLE_METHOD_COUNT) {
        return result;
    } else {
        return sample_method_t::SAMPLE_METHOD_COUNT;
    }
}

uint8_t* resize_image(uint8_t * image_buffer, int& width, int& height, int expected_width = 0, int expected_height = 0, int expected_channel = 3)
{
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

uint8_t* load_image_from_b64(const std::string & b64str, int& width, int& height, int expected_width = 0, int expected_height = 0, int expected_channel = 3)
{
    std::vector<uint8_t> decoded_buf = kcpp_base64_decode(b64str);
    int c = 0;
    uint8_t* image_buffer = (uint8_t*)stbi_load_from_memory(decoded_buf.data(), decoded_buf.size(), &width, &height, &c, expected_channel);
    if (image_buffer == nullptr) {
        image_buffer = kcpp_decode_webp_from_memory(
            decoded_buf.data(), decoded_buf.size(), &width, &height, &c, expected_channel);
    }

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
    image_buffer = resize_image(image_buffer,width,height,expected_width,expected_height,expected_channel);
    return image_buffer;
}

static enum scheduler_t scheduler_from_name(const char * scheduler)
{
    if (scheduler) {
        enum scheduler_t result = str_to_scheduler(scheduler);
        if (result != scheduler_t::SCHEDULER_COUNT)
        {
            return result;
        }
    }
    return scheduler_t::SCHEDULER_COUNT;
}

static void parse_cache_options(sd_cache_params_t & params, const std::string& cache_mode,
    const std::string& cache_options) {

    sd_cache_params_init(&params);
    if (cache_mode == "easycache") {
        params.mode = SD_CACHE_EASYCACHE;
    } else if (cache_mode == "ucache") {
        params.mode = SD_CACHE_UCACHE;
    } else if (cache_mode == "dbcache") {
        params.mode  = SD_CACHE_DBCACHE;
    } else if (cache_mode == "taylorseer") {
        params.mode  = SD_CACHE_TAYLORSEER;
    } else if (cache_mode == "cache-dit") {
        params.mode  = SD_CACHE_CACHE_DIT;
    } else if (cache_mode == "spectrum") {
        params.mode  = SD_CACHE_SPECTRUM;
    } else if (cache_mode != "" && cache_mode != "disabled") {
        printf("warning: unknown cache mode '%s'", cache_mode.c_str());
    }

    if (params.mode == SD_CACHE_DISABLED)
        return;

    if (cache_options == "")
        return;

    sd_cache_params_t cache_params = params;

    // from examples/common/common.hpp
    auto parse_named_params = [&](const std::string& opt_str) -> bool {
        std::stringstream ss(opt_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            size_t eq_pos = token.find('=');
            if (eq_pos == std::string::npos) {
                printf("error: cache option '%s' missing '=' separator", token.c_str());
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
                    printf("error: unknown cache parameter '%s'", key.c_str());
                    return false;
                }
            } catch (const std::exception&) {
                printf("error: invalid value '%s' for parameter '%s'", val.c_str(), key.c_str());
                return false;
            }
        }

        switch (cache_params.mode) {
            case SD_CACHE_EASYCACHE:
            case SD_CACHE_UCACHE:
                if (cache_params.reuse_threshold < 0.0f) {
                    printf("error: cache threshold must be non-negative");
                    return false;
                }
                if (cache_params.start_percent < 0.0f || cache_params.start_percent >= 1.0f ||
                    cache_params.end_percent <= 0.0f || cache_params.end_percent > 1.0f ||
                    cache_params.start_percent >= cache_params.end_percent) {
                    printf("error: cache start/end percents must satisfy 0.0 <= start < end <= 1.0");
                    return false;
                }
                break;
            default: ;
        }
        return true;
    };

    if (parse_named_params(cache_options)) {
        params = cache_params;
    }
}

static std::string raw_image_to_png_base64(const sd_image_t& img, std::string parameters = "") {
    std::string result;
    int out_data_len = 0;
    unsigned char * png = stbi_write_png_to_mem(img.data, 0, img.width, img.height, img.channel, &out_data_len, parameters != "" ? parameters.c_str() : nullptr);
    if (png != NULL) {
        result = kcpp_base64_encode(png,out_data_len);
        free(png);
    }
    return result;
}

static sd_audio_t load_audio_from_b64(const std::string& b64audio) {
    sd_audio_t audio = {0, 0, 0, nullptr};
    if (b64audio.empty()) {
        return audio;
    }

    std::vector<uint8_t> audio_data = kcpp_base64_decode(b64audio);
    std::vector<float> decoded_samples;
    int sample_rate = 0;
    int channels = 0;
    if (!kcpp_decode_audio_file_from_buf(audio_data.data(), audio_data.size(), sample_rate, channels, decoded_samples)) {
        printf("KCPP SD: failed to decode input audio\n");
        return audio;
    }

    uint64_t sample_count = static_cast<uint64_t>(decoded_samples.size() / static_cast<size_t>(channels));
    size_t float_count = decoded_samples.size();
    float* samples = (float*)malloc(float_count * sizeof(float));
    if (samples == nullptr || sample_count == 0) {
        free(samples);
        return audio;
    }

    std::memcpy(samples, decoded_samples.data(), float_count * sizeof(float));
    audio.sample_rate = static_cast<uint32_t>(sample_rate);
    audio.channels = static_cast<uint32_t>(channels);
    audio.sample_count = sample_count;
    audio.data = samples;
    return audio;
}

static std::string upscale_image_to_png_base64(upscaler_ctx_t* upscaler_ctx, const sd_image_t& input_image, int upscale_factor = 2, const std::string& meta_image_info = "")
{
    std::string gen_data;
    sd_image_generation_result_t* upscaled = nullptr;
    if (upscale(upscaler_ctx, input_image, upscale_factor, &upscaled)) {
        gen_data = raw_image_to_png_base64(upscaled->items[0].image, meta_image_info);
        free_sd_image_generation_result(upscaled);
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
    gp.prompt = SAFE_STR(inputs.prompt);
    gp.negative_prompt = SAFE_STR(inputs.negative_prompt);
    gp.cfg_scale = inputs.cfg_scale;
    gp.distilled_guidance = inputs.distilled_guidance;
    gp.sample_steps = inputs.sample_steps;
    gp.shifted_timestep = inputs.shifted_timestep;
    gp.flow_shift = inputs.flow_shift;
    gp.extra_sample_args = SAFE_STR(inputs.extra_sample_args);
    gp.eta = inputs.eta;
    gp.seed = inputs.seed;
    gp.width = gp.req_width = inputs.width;
    gp.height = gp.req_height = inputs.height;
    gp.strength = inputs.denoising_strength;
    gp.clip_skip = inputs.clip_skip;
    gp.sample_method = sampler_from_name(inputs.sample_method);
    gp.scheduler = scheduler_from_name(inputs.scheduler);
    gp.cache_mode = SAFE_STR(inputs.cache_mode);
    gp.cache_options = SAFE_STR(inputs.cache_options);
    gp.circular_x = inputs.circular_x;
    gp.circular_y = inputs.circular_y;
    if (inputs.custom_sigmas && inputs.custom_sigmas_count > 0) {
        gp.custom_sigmas.assign(inputs.custom_sigmas, inputs.custom_sigmas + inputs.custom_sigmas_count);
    }
    gp.lora_map = klp.lora_map;
    if (klp.lora_dynamic) {
        for (int i = 0; i < inputs.lora_len; i++) {
            gp.lora_map.add_lora(inputs.lora_filenames[i], inputs.lora_multipliers[i]);
        }
    }
    gp.loras = gp.lora_map.get_lora_specs();
    gp.lora_meta = gp.lora_map.get_lora_meta();
    gp.upscale = inputs.upscale;

    gp.ref_image_args = "resize_before_vae=on"; // auto resize ref image
    if (inputs.ref_image_args && *inputs.ref_image_args) {
        gp.ref_image_args += ",";
        gp.ref_image_args += inputs.ref_image_args;
        if (gp.ref_image_args.find("preset") != std::string::npos) {
            gp.force_image_edit = true;
            if(!sd_is_quiet && sddebugmode==1) {
                printf("ref_image_args=\"%s\", forcing edit mode", inputs.ref_image_args);
            }
        }
    }

    gp.img.init_image = SAFE_STR(inputs.init_images);
    gp.img.mask = SAFE_STR(inputs.mask);
    gp.img.flip_mask = inputs.flip_mask;
    for(int i=0;i<inputs.extra_images_len;++i)
    {
        gp.img.extra_images.push_back(std::string(inputs.extra_images[i]));
    }

    gp.vid.req_frames = inputs.vid_req_frames;
    gp.vid.output_type = inputs.video_output_type;
    gp.vid.fps = inputs.vid_fps;
    gp.vid.start_frame = SAFE_STR(inputs.video_start_frame);
    gp.vid.end_frame = SAFE_STR(inputs.video_end_frame);
    for(int i=0;i<inputs.ref_audios_len;++i)
    {
        gp.vid.ref_audios.push_back(std::string(inputs.ref_audios[i]));
    }

    const auto& info = klp.info;
    const bool supports_reference_images = (info.is_video_model || info.supports_ref_image || info.is_kontext || klp.photomaker_enabled);
    const bool remove_limits = inputs.remove_limits;

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

    // <image:N>/<image:+N> prompt tags reference gp.img.extra_images by position
    // and route the images to the VLM conditioner; images will be removed from
    // the ref_images parameter unless a reference has a '+' tag
    image_tag_refs image_tags;
    bool vision_tags_active = false;
    {
        size_t ref_count = std::min(gp.img.extra_images.size(), (size_t)max_extra_images);
        std::string tag_err;
        if (!collect_image_tags(gp.prompt, gp.negative_prompt, ref_count, image_tags, tag_err)) {
            return sd_generation.error(("KCPP SD: " + tag_err).c_str());
        }
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

    sd_ctx_t* sd_ctx = sd_params->sd_ctx; // protected by no_concurrency mutex

    PromptImageRef prompt_image_ref;
    struct CleanupImageCallbackOnExit {
        sd_ctx_t* sd_ctx;
        ~CleanupImageCallbackOnExit() {
            sd_set_prompt_image_callback(sd_ctx, nullptr, nullptr);
        }
    } cleanup_image_callback_onexit{sd_ctx};
    sd_set_prompt_image_callback(sd_ctx, prompt_image_from_refs, &prompt_image_ref);
    prompt_image_ref.zero_based = image_tags.zero_based;

    if (gp.sample_method == sample_method_t::SAMPLE_METHOD_COUNT) {
        gp.sample_method = sd_get_default_sample_method(sd_ctx);
    }

    bool is_img2img = gp.img.init_image != "";

    if (!remove_limits && info.is_flux1)
    {
        if (gp.cfg_scale != 1.0f) {
            // clamp cfg scale
            if (!sd_is_quiet && sddebugmode) {
                printf("Flux: clamping CFG Scale to 1\n");
            }
            gp.cfg_scale = 1.0f;
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
            !is_img2img
            // a single extra image is provided, and mask is NOT provided
            && gp.img.extra_images.size()==1 && gp.img.mask==""
            // the model does not support references natively, or through an edit mode/LoRa
            && !supports_reference_images && !gp.force_image_edit
            // the model can't handle image tags, or the user didn't ask for them
            && !vision_tags_active);

    if(redirect_single_reference_to_img2img)
    {
        is_img2img = true;
        gp.img.init_image = gp.img.extra_images[0];
        gp.img.extra_images.clear();
        if (!sd_is_quiet && sddebugmode == 1) {
            printf("Switching reference image to img2img\n");
        }
    }

    if (info.is_video_model && gp.vid.start_frame.empty() && is_img2img && gp.img.mask=="")
    {
        // no explicit video start frame: the img2img input becomes the video start frame
        std::swap(gp.vid.start_frame, gp.img.init_image);
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
    }

    // trigger tiling by image area, the memory used for the VAE buffer is 6656 bytes per image pixel, default 768x768
    bool dotile = (gp.width*gp.height > klp.tiled_vae_threshold*klp.tiled_vae_threshold);

    int vae_tile_size = -1;
    if (dotile) {
        // bump tile size to 1/4 of the configured maximum non-tiled image size
        // (so 1/4 of the memory that would be used for that image)
        const int minimum_tile_size = 256; // from mainline sdcpp
        int new_vae_tile_size = klp.tiled_vae_threshold;
        new_vae_tile_size = new_vae_tile_size / 2;
        new_vae_tile_size = roundup_to(new_vae_tile_size, 64);
        if (new_vae_tile_size > minimum_tile_size) {
            vae_tile_size = new_vae_tile_size;
        }
    }

    //for img2img
    sd_image_t input_image = {0,0,0,nullptr};
    sd_image_t video_start_image = {0,0,0,nullptr};
    sd_image_t video_end_image = {0,0,0,nullptr};
    std::vector<sd_image_t> reference_imgs;
    std::vector<sd_image_t> legacy_video_imgs;
    std::vector<sd_image_t> photomaker_imgs;
    std::vector<sd_audio_t> reference_audios;
    struct ReferenceAudioCleanup {
        std::vector<sd_audio_t>& audios;
        ~ReferenceAudioCleanup() {
            for (auto& audio : audios) {
                if (audio.data) {
                    free(audio.data);
                    audio.data = nullptr;
                }
            }
        }
    } reference_audio_cleanup { reference_audios };

    int nx, ny, nc;
    int img2imgW = gp.width; //for img2img input
    int img2imgH = gp.height;
    int img2imgC = 3; // Assuming RGB image

    std::string ts = get_timestamp_str();
    if(!sd_is_quiet)
    {
        printf("\n[%s] Generating Image (%d steps)\n",ts.c_str(),gp.sample_steps);
    }else{
        printf("\n[%s] Generating (%d st.)\n",ts.c_str(),gp.sample_steps);
    }

    fflush(stdout);

    struct ImageBuffer {
        uint8_t * input = nullptr;
        uint8_t * mask = nullptr;
        std::vector<uint8_t *> extra;
        ~ImageBuffer() {
            stbi_image_free(input);
            stbi_image_free(mask);
            for(auto buf : extra) {
                stbi_image_free(buf);
            }
        }
    } image_buffer;

    auto load_extra_image = [&](const std::string& b64, bool resize_to_generation) -> sd_image_t {
        sd_image_t image = {0,0,0,nullptr};
        if (b64.empty()) {
            return image;
        }
        int nx2, ny2, nc2;
        uint8_t * loaded = resize_to_generation
            ? load_image_from_b64(b64,nx2,ny2,img2imgW,img2imgH,3)
            : load_image_from_b64(b64,nx2,ny2);
        if(loaded)
        {
            image_buffer.extra.push_back(loaded);
            image.width = nx2;
            image.height = ny2;
            image.channel = 3;
            image.data = loaded;
        }
        return image;
    };

    if (info.is_video_model)
    {
        video_start_image = load_extra_image(gp.vid.start_frame, true);
        video_end_image = load_extra_image(gp.vid.end_frame, true);
    }

    if(gp.img.extra_images.size()>0)
    {
        if (vision_tags_active) {
            prompt_image_ref.image_pool.resize(std::min(gp.img.extra_images.size(), (size_t)max_extra_images));
        }
        for(int i=0;i<gp.img.extra_images.size() && i<max_extra_images;++i)
        {
            const bool duplicates_video_frame = info.is_video_model && info.is_minimaxh3
                && (gp.img.extra_images[i] == gp.vid.start_frame || gp.img.extra_images[i] == gp.vid.end_frame);
            sd_image_t image = {0,0,0,nullptr};
            if(supports_reference_images||gp.force_image_edit||vision_tags_active)
            {
                int nx2, ny2, nc2;
                int desiredchannels = 3;
                if(info.is_video_model)
                {
                    if (!duplicates_video_frame || (vision_tags_active && image_tags.mentioned.count(i)))
                    {
                        image = load_extra_image(gp.img.extra_images[i], !info.is_minimaxh3);
                    }
                }
                else if(info.supports_ref_image||gp.force_image_edit)
                {
                    uint8_t * loaded = load_image_from_b64(gp.img.extra_images[i],nx2,ny2);
                    if(loaded)
                    {
                        //kcpp fix: qwen image can stack overflow and crash when ref images exceed
                        // a total res of 512x512 = 262144, so we downscale if that's the case
                        // kcpp edit 2mar2026: this seems to be better now, so limit to 1024x1024 instead
                        int tgtx = nx2;
                        int tgty = ny2;
                        int res_lim_crash = 1024 * 1024;
                        if (nx2 * ny2 > res_lim_crash)
                        {
                            float factor = sqrtf((float)res_lim_crash / ((float)nx2 * (float)ny2));
                            tgtx = (int)(nx2 * factor);
                            tgty = (int)(ny2 * factor);
                            if (!sd_is_quiet && sddebugmode == 1)
                            {
                                printf("\nResized RefImg %dx%d to %dx%d", nx2, ny2, tgtx, tgty);
                            }
                            loaded = resize_image(loaded, nx2, ny2, tgtx, tgty);
                        }
                        if(loaded)
                        {
                            image_buffer.extra.push_back(loaded);
                            image.width = nx2;
                            image.height = ny2;
                            image.channel = desiredchannels;
                            image.data = loaded;
                        }
                    }
                }
                else if (info.is_kontext || klp.photomaker_enabled || vision_tags_active)
                {
                    uint8_t * loaded = load_image_from_b64(gp.img.extra_images[i],nx2,ny2);
                    if(loaded)
                    {
                        image_buffer.extra.push_back(loaded);
                        image.width = nx2;
                        image.height = ny2;
                        image.channel = desiredchannels;
                        image.data = loaded;
                    }
                }
            }
            if(!image.data)
            {
                continue;
            }
            if (vision_tags_active) {
                prompt_image_ref.image_pool[i] = image;
                if (image_tags.hidden.count(i)) {
                    continue; // used only as VLM context, kept out of the ref list
                }
            }
            if (duplicates_video_frame || !(supports_reference_images||gp.force_image_edit)) {
                continue; // Keep frame deduplication in the diffusion reference list.
            }
            if (info.is_video_model) {
                if (info.is_minimaxh3) {
                    reference_imgs.push_back(image);
                } else {
                    legacy_video_imgs.push_back(image);
                }
            }
            else if (info.is_kontext) {
                reference_imgs.push_back(image);
            }
            else if (klp.photomaker_enabled) {
                photomaker_imgs.push_back(image);
            }
            else {
                reference_imgs.push_back(image);
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
            printf("\nImageGen References: RefImg=%zu Wan=%zu Photomaker=%zu\n",reference_imgs.size(),legacy_video_imgs.size(),photomaker_imgs.size());
        }
    }

    // reference audio is a video-model feature; the H3 image variant ignores it
    if(info.is_minimaxh3 && info.is_video_model && gp.vid.ref_audios.size()>0)
    {
        for(int i=0;i<gp.vid.ref_audios.size();++i)
        {
            sd_audio_t ref_audio = load_audio_from_b64(gp.vid.ref_audios[i]);
            if (ref_audio.data == nullptr) {
                return sd_generation.error("KCPP SD: load reference audio from base64 failed!");
            }
            reference_audios.push_back(ref_audio);
        }
    }

    const bool has_minimax_keyframes = info.is_minimaxh3 && (video_start_image.data || video_end_image.data || legacy_video_imgs.size() > 0);
    const bool has_minimax_references = info.is_minimaxh3 && (reference_imgs.size() > 0 || reference_audios.size() > 0);
    if(has_minimax_keyframes && has_minimax_references)
    {
        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nMiniMax-H3: keyframes and Ref2VA references cannot be mixed; using reference media and ignoring keyframes.\n");
        }
        video_start_image = {0,0,0,nullptr};
        video_end_image = {0,0,0,nullptr};
        legacy_video_imgs.clear();
    }

    // settle the local parameters into the upstream image generation params
    sd_img_gen_params_t params = {};
    sd_img_gen_params_init (&params);
    params.batch_count = 1;
    params.ref_image_args = gp.ref_image_args.c_str();
    params.prompt = gp.prompt.c_str();
    params.negative_prompt = gp.negative_prompt.c_str();
    params.clip_skip = gp.clip_skip;
    params.sample_params.guidance.txt_cfg = gp.cfg_scale;
    // params.sample_params.guidance.img_cfg = gp.cfg_scale; //removed, breaks qwen img edit and more
    if (gp.distilled_guidance >= 0.f) {
        params.sample_params.guidance.distilled_guidance = gp.distilled_guidance;
    }
    params.width = gp.width;
    params.height = gp.height;
    params.sample_params.sample_method = gp.sample_method;
    params.sample_params.scheduler = gp.scheduler;
    params.sample_params.sample_steps = gp.sample_steps;
    if (!gp.custom_sigmas.empty()) {
        params.sample_params.custom_sigmas = gp.custom_sigmas.data();
        params.sample_params.custom_sigmas_count = (int)gp.custom_sigmas.size();
    }
    params.sample_params.shifted_timestep = gp.shifted_timestep;
    if (gp.eta >= 0.f && gp.eta <= 1.f) {
        params.sample_params.eta = gp.eta;
    }
    if (gp.flow_shift > 0.f && gp.flow_shift != INFINITY) {
        params.sample_params.flow_shift = gp.flow_shift;
    }
    params.sample_params.extra_sample_args = gp.extra_sample_args.c_str();
    params.seed = gp.seed;
    params.strength = gp.strength;
    params.vae_tiling_params.enabled = dotile;
    if (vae_tile_size > 0) {
        params.vae_tiling_params.tile_size_w = vae_tile_size;
        params.vae_tiling_params.tile_size_h = vae_tile_size;
    }
    if(dotile)
    {
        params.vae_tiling_params.temporal_tiling = true;
    }
    parse_cache_options(params.cache, gp.cache_mode, gp.cache_options);
    params.circular_x = gp.circular_x;
    params.circular_y = gp.circular_y;

    if(!sd_is_quiet && sddebugmode==1) {
        if (gp.loras.size() > 0) {
            printf("Applying LoRAs:\n");
            for(size_t i=0;i<gp.loras.size();++i)
            {
                printf("  %s @ %.3f\n", gp.loras[i].path, gp.loras[i].multiplier);
            }
        }
    }

    // note sdcpp tracks previously applied LoRAs and weights,
    // and apply/unapply the differences at each gen
    params.loras = gp.loras.data();
    params.lora_count = gp.loras.size();

    params.ref_images = reference_imgs.data();
    params.ref_images_count = reference_imgs.size();
    params.pm_params.id_images = photomaker_imgs.data();
    params.pm_params.id_images_count = photomaker_imgs.size();

     //special case, is img2img and denoise strength is 0 and steps is 1, do a passthru
    bool is_passthrough = (gp.sample_steps<=1 && gp.strength<=0 && is_img2img && gp.vid.req_frames<=1 && gp.img.extra_images.size()==0);
    sd_audio_t* generated_audio = nullptr;

    sd_image_generation_result_t* img_result = nullptr;
    sd_video_generation_result_t* vid_result = nullptr;
    int generated_num_results = 0;

    if(info.is_video_model)
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
        vid_gen_params.video_frames = gp.vid.req_frames;
        vid_gen_params.fps = gp.vid.fps;
        vid_gen_params.vae_tiling_params = params.vae_tiling_params;
        vid_gen_params.loras = params.loras;
        vid_gen_params.lora_count = params.lora_count;
        if (info.is_minimaxh3 && reference_imgs.size() > 0) {
            vid_gen_params.ref_images = reference_imgs.data();
            vid_gen_params.ref_images_count = (int)reference_imgs.size();
        }
        if (info.is_minimaxh3 && info.is_video_model && reference_audios.size() > 0) {
            vid_gen_params.ref_audios = reference_audios.data();
            vid_gen_params.ref_audios_count = (int)reference_audios.size();
        }
        if (legacy_video_imgs.size() >= 2) {
            if(!video_start_image.data) {
                video_start_image = legacy_video_imgs[0];
            }
            if(!video_end_image.data) {
                video_end_image = legacy_video_imgs[1];
            }
        } else if (legacy_video_imgs.size() == 1) {
            if(!video_start_image.data && !video_end_image.data) {
                video_start_image = legacy_video_imgs[0];
            }
        }
        if (video_start_image.data) {
            vid_gen_params.init_image = video_start_image;
        }
        if (video_end_image.data) {
            vid_gen_params.end_image = video_end_image;
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
            << "\nINIT_IMGS:" << ((vid_gen_params.init_image.data ? 1 : 0) + (vid_gen_params.end_image.data ? 1 : 0))
            << "\nREF_IMGS:" << vid_gen_params.ref_images_count
            << "\nREF_AUDIOS:" << vid_gen_params.ref_audios_count
            << "\n\n";
            printf("%s", ss.str().c_str());
        }

        fflush(stdout);

        vid_result = nullptr;
        generated_audio = nullptr;
        int output_fps = gp.vid.fps;
        if (is_passthrough) {
            // passthrough: skip generation and return the start frame (the img2img
            // input, swapped in above) as-is. input_image feeds the passthrough block
            // below; generated_num_results stays 0 so the result loop is skipped.
            if (video_start_image.data) {
                input_image = video_start_image;
            } else if (gp.img.init_image != "") {
                // a mask prevented the init->start-frame swap; load the input directly
                image_buffer.input = load_image_from_b64(gp.img.init_image, nx, ny, img2imgW, img2imgH, 3);
                input_image.width = img2imgW;
                input_image.height = img2imgH;
                input_image.channel = 3;
                input_image.data = image_buffer.input;
            }
            if (!input_image.data) {
                return sd_generation.error("KCPP SD: load image from memory failed!");
            }
        } else if (!generate_video(sd_ctx, &vid_gen_params, &vid_result)) {
            vid_result = nullptr;
        }
        if (vid_result) {
            generated_num_results = vid_result->frame_count;
            generated_audio = vid_result->audio;
            output_fps = vid_result->fps;
        }
        if(!sd_is_quiet && sddebugmode==1)
        {
            printf("\nRequested Vid Frames: %d, Generated Vid Frames: %d\n",gp.vid.req_frames, generated_num_results);
            if (output_fps != gp.vid.fps) {
                printf("\nRequested FPS: %d, Generated FPS: %d\n", gp.vid.fps, output_fps);
            }
        }
        gp.vid.fps = output_fps;
    }
    else if (!is_img2img)
    {
        if(!sd_is_quiet && sddebugmode==1)
        {
            char* buf = sd_img_gen_params_to_str(&params);
            if(buf)
            {
                printf("\n%s\n", buf);
                free(buf);
            }
        }

        fflush(stdout);

        img_result = nullptr;
        if (!generate_image(sd_ctx, &params, &img_result)) {
            img_result = nullptr;
        }
        generated_num_results = img_result ? img_result->count : 0;

    } else {

        image_buffer.input = load_image_from_b64(gp.img.init_image,nx,ny,img2imgW,img2imgH,3);

        if (!image_buffer.input) {
            return sd_generation.error("KCPP SD: load image from memory failed!");
        }

        if(gp.img.mask!="")
        {
            int nx2, ny2, nc2;
            image_buffer.mask = load_image_from_b64(gp.img.mask,nx2,ny2,img2imgW,img2imgH,1);

            if (!image_buffer.mask) {
                return sd_generation.error("KCPP SD: load mask from memory failed!");
            }

            if(gp.img.flip_mask)
            {
                int bufsiz = nx2 * ny2 * 1; //1 channel
                for (int i = 0; i < bufsiz; ++i) {
                    image_buffer.mask[i] = 255 - image_buffer.mask[i];
                }
            }
        }

        input_image.width = img2imgW;
        input_image.height = img2imgH;
        input_image.channel = img2imgC;
        input_image.data = image_buffer.input;

        uint8_t* mask_image_buffer    = NULL;
        std::vector<uint8_t> default_mask_image_vec(img2imgW * img2imgH, 255);
        if (image_buffer.mask) {
            mask_image_buffer = image_buffer.mask;
        } else {
            mask_image_buffer = default_mask_image_vec.data();
        }
        sd_image_t mask_image = { (uint32_t) img2imgW, (uint32_t) img2imgH, 1, mask_image_buffer };

        params.init_image = input_image;
        params.mask_image = mask_image;

        if(!sd_is_quiet && sddebugmode==1)
        {
            char* buf = sd_img_gen_params_to_str(&params);
            if(buf)
            {
                printf("\n%s\n", buf);
                free(buf);
            }
        }

        fflush(stdout);

        if (is_passthrough) {
            printf("No generation triggered, passthrough mode.\n");
        } else {
            img_result = nullptr;
            if (!generate_image(sd_ctx, &params, &img_result)) {
                img_result = nullptr;
            }
            generated_num_results = img_result ? img_result->count : 0;
        }
    }

    if (!is_passthrough && (info.is_video_model ? vid_result == nullptr : img_result == nullptr)) {
        if (info.is_qwenimg && params.ref_images_count > 0) {
            printf("\nKCPP SD: If using Qwen Image 2.1, editing requires Qwen3-VL vision mmproj!\n");
        }
        return sd_generation.error("KCPP SD generate failed!");
    }

    bool isanim = (gp.vid.req_frames>1 && generated_num_results>1 && info.is_video_model);
    nlohmann::json jsoninfo = nlohmann::json::object();
    if (!isanim) {
        jsoninfo["prompt"] = params.prompt + gp.lora_meta;
        if (*params.negative_prompt)
            jsoninfo["negative_prompt"] = params.negative_prompt;
        jsoninfo["seed"] = params.seed;
        jsoninfo["cfg_scale"] = round_for_info(params.sample_params.guidance.txt_cfg);
        jsoninfo["width"] = params.width;
        jsoninfo["height"] = params.height;
        jsoninfo["steps"] = params.sample_params.sample_steps;
        jsoninfo["sampler_name"] = sd_sample_method_name(params.sample_params.sample_method);
        if (params.clip_skip > 0)
            jsoninfo["clip_skip"] = params.clip_skip;
        jsoninfo["extra_generation_params"] = nlohmann::json::object();
        if (params.sample_params.scheduler != scheduler_t::SCHEDULER_COUNT)
            jsoninfo["extra_generation_params"]["Schedule type"] = get_scheduler_name(params.sample_params.scheduler);
        if (params.sample_params.eta >= 0 && params.sample_params.eta <= 1)
            jsoninfo["eta"] = round_for_info(params.sample_params.eta);
        if (is_img2img)
            jsoninfo["denoising_strength"] = round_for_info(params.strength);
        if (sd_params->load_params.model_path.empty())
            jsoninfo["sd_model_name"] = friendly_model_name(sd_params->load_params.diffusion_model_path);
        else
            jsoninfo["sd_model_name"] = friendly_model_name(sd_params->load_params.model_path);
        std::string vae = sd_params->load_params.vae_path;
        if (vae.empty())
            // TODO: check if auto1111 uses a specific field for tae
            vae = sd_params->load_params.taesd_path;
        if (!vae.empty())
            jsoninfo["sd_vae_name"] = friendly_model_name(vae);
        jsoninfo["infotexts"] = nlohmann::json::array();
        jsoninfo["all_prompts"] = nlohmann::json::array();
        jsoninfo["all_negative_prompts"] = nlohmann::json::array();
        jsoninfo["all_seeds"] = nlohmann::json::array();
        jsoninfo["lora_meta"] = gp.lora_meta;
        jsoninfo["version"] = "KoboldCpp";
    }
    sd_image_t* upscaled_image = nullptr;
    std::string gen_data;
    std::string gen_data2;
    std::string final_frame_data;

    auto upscaler_ctx = gp.upscale ? sd_params->upscaler_ctx : nullptr;

    if (is_passthrough)
    {
        if(upscaler_ctx != nullptr)
        {
            printf("Upscaling original image (passthrough)...\n");
            gen_data = upscale_image_to_png_base64(upscaler_ctx, input_image, 2);
        }
        else {
            gen_data = raw_image_to_png_base64(input_image);
        }
    }
    else if (isanim)
    {
        //if multiframe, make a video
        if (generated_num_results > 0 && vid_result && vid_result->frames && vid_result->frames[0].data)
        {
            if(!sd_is_quiet && sddebugmode==1)
            {
                printf("\nSaving video buffer, VIDEO_OUTPUT_TYPE=%d...",gp.vid.output_type);
            }
            uint8_t * out_data = nullptr;
            uint8_t * out_data2 = nullptr;
            size_t out_len = 0;
            size_t out_len2 = 0;
            int status = 0;
            int status2 = 0;

            const bool output_gif  = gp.vid.output_type == 0 || gp.vid.output_type == 2;
            const bool output_webp = gp.vid.output_type == 3 || gp.vid.output_type == 4;
            const bool output_avi  = gp.vid.output_type == 1 || gp.vid.output_type == 2 || gp.vid.output_type == 4;

            if(output_gif)
            {
                status = create_gif_buf_from_sd_images_msf(vid_result->frames, generated_num_results, gp.vid.fps, &out_data,&out_len);
            }
            else if(output_webp)
            {
                status = create_animated_webp_buf_from_sd_images(vid_result->frames, generated_num_results, gp.vid.fps, &out_data,&out_len);
            }
            if(output_avi)
            {
                status2 = create_mjpg_avi_membuf_from_sd_images(vid_result->frames, generated_num_results, gp.vid.fps, 40, &out_data2,&out_len2, generated_audio);
            }

            if(generated_num_results>1)
            {
                sd_image_t *final_frame_image = &vid_result->frames[generated_num_results-1];
                final_frame_data = raw_image_to_png_base64(*final_frame_image);
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
        free_sd_video_generation_result(vid_result);
        vid_result = nullptr;
    }
    else
    {
        for (int i = 0; i < generated_num_results; i++)
        {
            sd_image_t& result_image = info.is_video_model ? vid_result->frames[i] : img_result->items[i].image;
            if (result_image.data == NULL) {
                continue;
            }
            std::string meta_image_info = get_image_params(params, gp.lora_meta, i);
            if(upscaler_ctx != nullptr && gp.upscale)
            {
                printf("Upscaling output image...\n");
                gen_data = upscale_image_to_png_base64(upscaler_ctx, result_image, 2, meta_image_info);
            } else {
                gen_data = raw_image_to_png_base64(result_image, meta_image_info);
            }
            jsoninfo["infotexts"][i] = meta_image_info;
            jsoninfo["all_seeds"][i] = params.seed + i;
            jsoninfo["all_prompts"][i] = params.prompt;
            jsoninfo["all_negative_prompts"][i] = params.negative_prompt;
        }
        if (info.is_video_model) {
            free_sd_video_generation_result(vid_result);
            vid_result = nullptr;
        } else {
            free_sd_image_generation_result(img_result);
            img_result = nullptr;
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

static void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data)
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
    if (image != nullptr && frame_count > 0) {
        constexpr uint32_t preview_max_dimension = 128;
        std::vector<sd_image_t> resized_images(image, image + frame_count);
        std::vector<std::vector<uint8_t>> resized_image_data(static_cast<size_t>(frame_count));
        bool resize_ok = true;

        for (int i = 0; i < frame_count; ++i) {
            const sd_image_t& source = image[i];
            if (source.width <= preview_max_dimension && source.height <= preview_max_dimension) {
                continue;
            }

            uint32_t resized_width;
            uint32_t resized_height;
            if (source.width >= source.height) {
                resized_width = preview_max_dimension;
                resized_height = std::max(1u, static_cast<uint32_t>(
                    static_cast<uint64_t>(source.height) * preview_max_dimension / source.width));
            } else {
                resized_height = preview_max_dimension;
                resized_width = std::max(1u, static_cast<uint32_t>(
                    static_cast<uint64_t>(source.width) * preview_max_dimension / source.height));
            }

            auto& pixels = resized_image_data[static_cast<size_t>(i)];
            pixels.resize(static_cast<size_t>(resized_width) * resized_height * source.channel);
            if (source.data == nullptr || source.channel == 0 ||
                !stbir_resize_uint8(source.data, source.width, source.height, 0,
                                    pixels.data(), resized_width, resized_height, 0, source.channel)) {
                resize_ok = false;
                break;
            }
            resized_images[static_cast<size_t>(i)] = {
                resized_width, resized_height, source.channel, pixels.data()
            };
        }

        sd_image_t* preview_images = resize_ok ? resized_images.data() : nullptr;
        if (frame_count == 1) {
            if (preview_images != nullptr) {
                preview = raw_image_to_png_base64(*preview_images);
            }
        } else if (preview_images != nullptr) {
            uint8_t * out_data = nullptr;
            size_t out_len = 0;
            if (create_gif_buf_from_sd_images_msf(preview_images, frame_count, 16, &out_data,&out_len) == 0 && out_data && out_len > 0) {
                preview = kcpp_base64_encode(out_data, out_len);
            }
            if (out_data) {
                free(out_data);
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

    std::string rawb64 = inputs.init_images;
    int nx, ny;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> upscale_src_buffer(
        load_image_from_b64(rawb64,nx,ny), &stbi_image_free);
    sd_image_t source_img;
    source_img.data = nullptr;
    std::string result;
    if(upscale_src_buffer)
    {
        source_img.width = nx;
        source_img.height = ny;
        source_img.channel = 3;
        source_img.data = upscale_src_buffer.get();

        result = upscale_image_to_png_base64(upscaler_ctx, source_img, inputs.upscaling_resize);
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

