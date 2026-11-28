#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles

typedef struct sd_tensor_t sd_tensor_t;
typedef struct sd_ctx_t sd_ctx_t;
typedef struct sd_result sd_result_t;
typedef struct sd_opt sd_opt_t;
typedef struct sd_upscaler sd_upscaler_t;
typedef struct sd_vid_upscaler sd_vid_upscaler_t;
typedef struct sd_adetailer sd_adetailer_t;

// Status and diagnostics

typedef enum {
    SD_STATUS_OK                = 0,
    SD_STATUS_INVALID_ARGUMENT  = 1,
    SD_STATUS_UNKNOWN_KEY       = 2,
    SD_STATUS_TYPE_MISMATCH     = 3,
    SD_STATUS_OUT_OF_RANGE      = 4,
    SD_STATUS_UNSUPPORTED       = 5,
    SD_STATUS_OUT_OF_MEMORY     = 6,
    SD_STATUS_LOAD_FAILED       = 7,
    SD_STATUS_GENERATION_FAILED = 8,
    SD_STATUS_BUSY              = 9,
    SD_STATUS_CANCELLED         = 10,
} sd_status_t;

// Thread-local diagnostic, cleared by successful status/accessor calls.
// Valid until the next API call; free functions and tensor retain preserve it.
SD_API const char* sd_get_last_error(void);

// Tensors and media

typedef int32_t sd_tensor_type_t;
#define SD_TENSOR_U8 1
#define SD_TENSOR_I32 2
#define SD_TENSOR_I64 3
#define SD_TENSOR_F32 4
#define SD_TENSOR_F64 5

typedef struct {
    uint32_t sample_rate;
    const sd_tensor_t* data;
} sd_audio_t;

typedef struct {
    const sd_tensor_t* data;
    int fps;
    sd_audio_t audio;
} sd_video_t;

// Contiguous dimension first. Images: U8 [C,W,H], audio: F32 [C,N], video: U8 [C,W,H,T].
// create copies data; data_size must exactly match type and shape. Rank <= 8.
// Rank zero is a scalar; zero dimensions represent empty tensors.
SD_API sd_tensor_t* sd_tensor_create(
    sd_tensor_type_t type,
    size_t rank,
    const int64_t* shape,
    const void* data,
    size_t data_size);

// retain returns the same immutable handle with one additional ownership reference.
SD_API sd_tensor_t* sd_tensor_retain(const sd_tensor_t* tensor);
SD_API void sd_tensor_free(sd_tensor_t* tensor);
SD_API sd_tensor_type_t sd_tensor_type(const sd_tensor_t* tensor);
SD_API size_t sd_tensor_rank(const sd_tensor_t* tensor);
SD_API int64_t sd_tensor_dim(const sd_tensor_t* tensor, size_t axis);
SD_API size_t sd_tensor_byte_size(const sd_tensor_t* tensor);
SD_API const void* sd_tensor_data(const sd_tensor_t* tensor);

// Options

typedef struct {
    const char* name;
    const char* path;
} sd_embedding_t;

typedef struct {
    bool is_high_noise;
    float multiplier;
    const char* path;
} sd_lora_t;

typedef sd_opt_t sd_ctx_opt_t;
typedef sd_opt_t sd_img_gen_opt_t;
typedef sd_opt_t sd_vid_gen_opt_t;
typedef sd_opt_t sd_upscaler_opt_t;
typedef sd_opt_t sd_upscale_opt_t;
typedef sd_opt_t sd_vid_upscaler_opt_t;
typedef sd_opt_t sd_vid_upscale_opt_t;
typedef sd_opt_t sd_adetailer_opt_t;
typedef sd_opt_t sd_adetail_opt_t;

// Parameter keys and types are documented in docs/guides/c_api.md.
// Setters replace values atomically; a failed setter preserves the previous value.
// Strings, arrays and media descriptors are copied. Tensor storage is retained;
// callers may free their tensor handles after a successful setter.
// Do not mutate/free params concurrently with an operation using them.
// The aliases share one C type; constructors bind their purpose at runtime.
SD_API sd_ctx_opt_t* sd_ctx_opt_create(void);
SD_API sd_img_gen_opt_t* sd_img_gen_opt_create(void);
SD_API sd_vid_gen_opt_t* sd_vid_gen_opt_create(void);
SD_API sd_upscaler_opt_t* sd_upscaler_opt_create(void);
SD_API sd_upscale_opt_t* sd_upscale_opt_create(void);
SD_API sd_vid_upscaler_opt_t* sd_vid_upscaler_opt_create(void);
SD_API sd_vid_upscale_opt_t* sd_vid_upscale_opt_create(void);
SD_API sd_adetailer_opt_t* sd_adetailer_opt_create(void);
SD_API sd_adetail_opt_t* sd_adetail_opt_create(void);
SD_API void sd_opt_free(sd_opt_t* params);

SD_API sd_status_t sd_opt_set_bool(sd_opt_t* params, const char* key, bool value);
SD_API sd_status_t sd_opt_set_int(sd_opt_t* params, const char* key, int64_t value);
SD_API sd_status_t sd_opt_set_float(sd_opt_t* params, const char* key, double value);
SD_API sd_status_t sd_opt_set_string(sd_opt_t* params, const char* key, const char* value);
SD_API sd_status_t sd_opt_set_tensor(sd_opt_t* params, const char* key, const sd_tensor_t* tensor);
SD_API sd_status_t sd_opt_set_tensors(
    sd_opt_t* params,
    const char* key,
    const sd_tensor_t* const* tensors,
    size_t count);
SD_API sd_status_t sd_opt_set_floats(sd_opt_t* params, const char* key, const float* values, size_t count);
SD_API sd_status_t sd_opt_set_ints(sd_opt_t* params, const char* key, const int32_t* values, size_t count);
SD_API sd_status_t sd_opt_set_loras(sd_opt_t* params, const char* key, const sd_lora_t* values, size_t count);
SD_API sd_status_t sd_opt_set_embeddings(
    sd_opt_t* params,
    const char* key,
    const sd_embedding_t* values,
    size_t count);
SD_API sd_status_t sd_opt_set_audios(sd_opt_t* params, const char* key, const sd_audio_t* values, size_t count);
SD_API sd_status_t sd_opt_set_videos(sd_opt_t* params, const char* key, const sd_video_t* values, size_t count);

// Setters/unset reject keys for another purpose. Unsetting an applicable known
// absent key succeeds; unknown keys fail. NULL is not a string value.
SD_API sd_status_t sd_opt_unset(sd_opt_t* params, const char* key);

// Context and generation

// Output pointers are required and set to NULL on failure. Context creation retains
// its own configuration; params may be freed immediately after the call returns.
SD_API sd_status_t sd_ctx_create(const sd_ctx_opt_t* params, sd_ctx_t** ctx_out);
SD_API void sd_ctx_free(sd_ctx_t* ctx);

SD_API sd_status_t sd_generate_image(sd_ctx_t* ctx, const sd_img_gen_opt_t* params, sd_result_t** result_out);
SD_API sd_status_t sd_generate_video(sd_ctx_t* ctx, const sd_vid_gen_opt_t* params, sd_result_t** result_out);

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx);

// ControlNet hot-swap APIs are not safe to call while generation is in flight.
SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path);
SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx);

// Requires a loaded context; returns a static string owned by the library, or "Unknown".
SD_API const char* sd_get_model_version_name(const sd_ctx_t* sd_ctx);

enum sd_cancel_mode_t {
    // Stop the current generation as soon as possible.
    SD_CANCEL_ALL,
    // Finish the current image sample, then skip additional batch latents and return completed images.
    SD_CANCEL_NEW_LATENTS,
    // Clear a pending cancellation request.
    SD_CANCEL_RESET
};

SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode);

// Results

typedef enum {
    SD_RESULT_INVALID = 0,
    SD_RESULT_IMAGE   = 1,
    SD_RESULT_VIDEO   = 2,
} sd_result_kind_t;

// Results own all outputs independently of ctx and params. Views are read-only and
// borrowed until sd_result_free(). Retain a tensor to keep its data independently.
// Media descriptors contain metadata and borrowed immutable tensor handles.
SD_API sd_result_kind_t sd_result_get_kind(const sd_result_t* result);
SD_API size_t sd_result_get_image_count(const sd_result_t* result);
SD_API const sd_tensor_t* sd_result_get_image(const sd_result_t* result, size_t index);

// Materializes contiguous U8 [C,W,H,T] storage on first access.
SD_API const sd_video_t* sd_result_get_video(const sd_result_t* result);

// Image metadata is per image; video metadata is shared by all frame indices.
SD_API const char* sd_result_get_generation_info(const sd_result_t* result, size_t index);
SD_API const sd_audio_t* sd_result_get_audio(const sd_result_t* result);
SD_API int sd_result_get_fps(const sd_result_t* result);

SD_API void sd_result_free(sd_result_t* result);

// Callbacks and progress

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_VERBOSE,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);

typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);

typedef struct {
    const char* stage;
    const char* detail;
    const char* unit;
    int sample_pass;
    int64_t completed;
    int64_t total;
} sd_generation_progress_t;
typedef void (*sd_generation_progress_cb_t)(const sd_generation_progress_t* progress, void* data);

// Context-local generation events, independent of preview and the legacy progress callback.
// Strings and the event are valid only during the synchronous callback. Callbacks must not throw.
// total == 0 means unknown; sample_pass counts from 1 since callback registration.
// Register/clear only while the context is idle. Passing NULL clears the callback.
SD_API void sd_set_generation_progress_callback(sd_ctx_t* ctx, sd_generation_progress_cb_t cb, void* data);

typedef struct {
    int sample_pass;
    int total_steps;
} sd_preview_info_t;

// Frames are borrowed during the callback; retain a tensor to keep its snapshot.
typedef void (*sd_preview_cb_t)(int step, int frame_count, const sd_tensor_t* const* frames, bool is_noisy, void* data);

// In each sampling pass, a positive interval previews every Nth denoiser step, while a
// negative interval previews only completed logical step -interval. Zero previews the final
// completed step of the first sampling pass (base-resolution or high-noise).
// Mode is "none", "proj", "tae" or "vae"; invalid input preserves registration.
SD_API sd_status_t sd_set_preview_callback(
    sd_preview_cb_t cb,
    const char* mode,
    int interval,
    bool denoised,
    bool noisy,
    void* data);

// Query from a preview callback. Passes are numbered from 1 since the last
// sd_set_preview_callback call; total_steps is the actual count in that pass.
// Both fields are zero before sampling starts.
SD_API sd_preview_info_t sd_get_preview_info();

// Resolves <image:source> markers. The returned U8 [C,W,H] tensor is borrowed
// and copied before the next callback. Register/clear while ctx is idle.
typedef bool (*sd_prompt_image_cb_t)(const char* source, void* data, const sd_tensor_t** image_out);
SD_API void sd_set_prompt_image_callback(sd_ctx_t* ctx, sd_prompt_image_cb_t cb, void* data);

struct ggml_tensor;

typedef bool (*sd_graph_eval_callback_t)(struct ggml_tensor* t, bool ask, void* user_data);
SD_API void sd_set_backend_eval_callback(sd_graph_eval_callback_t cb, void* data);

// Image upscaling

SD_API sd_status_t sd_upscaler_create(const sd_upscaler_opt_t* opt, sd_upscaler_t** out);
SD_API void sd_upscaler_free(sd_upscaler_t* upscaler);
SD_API sd_status_t sd_upscale(sd_upscaler_t* upscaler, const sd_upscale_opt_t* opt, sd_result_t** out);
SD_API int sd_upscaler_get_scale(const sd_upscaler_t* upscaler);

// Reads model metadata only; returns 0 if it is not a recognized RGB upscaler.
SD_API int sd_upscaler_probe_scale(const char* model_path);

// Video upscaling

SD_API sd_status_t sd_vid_upscaler_create(const sd_vid_upscaler_opt_t* opt, sd_vid_upscaler_t** out);
SD_API void sd_vid_upscaler_free(sd_vid_upscaler_t* upscaler);
SD_API sd_status_t sd_upscale_video(
    sd_vid_upscaler_t* upscaler,
    const sd_vid_upscale_opt_t* opt,
    sd_result_t** out);

// ADetailer

SD_API sd_status_t sd_adetailer_create(const sd_adetailer_opt_t* opt, sd_adetailer_t** out);
SD_API void sd_adetailer_free(sd_adetailer_t* adetailer);
SD_API sd_status_t sd_adetail(
    sd_adetailer_t* adetailer,
    sd_ctx_t* ctx,
    const sd_adetail_opt_t* opt,
    sd_result_t** out);

// Image preprocessing

SD_API sd_status_t sd_preprocess_canny(
    const sd_tensor_t* image,
    float high_threshold,
    float low_threshold,
    float weak,
    float strong,
    bool inverse,
    sd_tensor_t** image_out);

// Weight types and model conversion

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // SD_TYPE_Q4_0_4_8 = 32,
    // SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0 = 34,
    SD_TYPE_TQ2_0 = 35,
    // SD_TYPE_IQ4_NL_4_4 = 36,
    // SD_TYPE_IQ4_NL_4_8 = 37,
    // SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4   = 39,  // MXFP4 (1 block)
    SD_TYPE_NVFP4   = 40,  // NVFP4 (4 blocks, E4M3 scale)
    SD_TYPE_Q1_0    = 41,
    SD_TYPE_Q2_0    = 42,
    SD_TYPE_F8_E4M3 = 43,
    SD_TYPE_F8_E5M2 = 44,
    SD_TYPE_COUNT   = 45,
};

SD_API const char* sd_type_name(enum sd_type_t type);
SD_API enum sd_type_t str_to_sd_type(const char* str);

SD_API bool convert(
    const char* input_path,
    const char* vae_path,
    const char* output_path,
    enum sd_type_t output_type,
    const char* tensor_type_rules,
    bool convert_name);

SD_API bool convert_with_components(
    const char* model_path,
    const char* clip_l_path,
    const char* clip_g_path,
    const char* t5xxl_path,
    const char* diffusion_model_path,
    const char* vae_path,
    const char* output_path,
    enum sd_type_t output_type,
    const char* tensor_type_rules,
    bool convert_name,
    int n_threads,
    const sd_lora_t* loras,
    int lora_count);

// Importance matrix

SD_API bool load_imatrix(const char* imatrix_path);
SD_API void save_imatrix(const char* imatrix_path);
SD_API void enable_imatrix_collection(void);
SD_API void disable_imatrix_collection(void);

// Devices and system information

// List available ggml backend devices, one `name<TAB>description` per line.
// The names are the device names accepted by the --backend / --params-backend
// assignment specs. Returns the number of bytes required, excluding the null
// terminator. Passing nullptr or buffer_size 0 only queries the required size.
SD_API size_t sd_list_devices(char* buffer, size_t buffer_size);

// Register comma-separated RPC endpoints (host:port) before listing devices.
// Repeated endpoints are safe; nullptr or an empty string is a no-op.
// Returns false if RPC is unavailable or an endpoint has no usable devices.
SD_API bool sd_add_rpc_servers(const char* servers);

SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();

// Build information

SD_API const char* sd_commit(void);
SD_API const char* sd_version(void);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
