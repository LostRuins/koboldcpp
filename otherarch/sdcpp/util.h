#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "stable-diffusion.h"
#include "tensor.hpp"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

bool ends_with(const std::string& str, const std::string& ending);
bool starts_with(const std::string& str, const std::string& start);
bool contains(const std::string& str, const std::string& substr);

std::string sd_format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

int round_up_to(int value, int base);

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);

std::u32string utf8_to_utf32(const std::string& utf8_str);
std::string utf32_to_utf8(const std::u32string& utf32_str);
std::u32string unicode_value_to_utf32(int unicode_value);
// std::string sd_basename(const std::string& path);

std::string sd_get_u8path(const std::string& file_path);

sd_image_t tensor_to_sd_image(const sd::Tensor<float>& tensor, int frame_index = 0);


sd::Tensor<float> sd_image_to_tensor(sd_image_t image,
                                     int target_width  = -1,
                                     int target_height = -1,
                                     bool scale        = true);

sd::Tensor<float> clip_preprocess(const sd::Tensor<float>& image, int target_width, int target_height);

class MmapWrapper {
public:
    static std::unique_ptr<MmapWrapper> create(const std::string& filename, bool writable = false);

    virtual ~MmapWrapper() = default;

    MmapWrapper(const MmapWrapper&)            = delete;
    MmapWrapper& operator=(const MmapWrapper&) = delete;
    MmapWrapper(MmapWrapper&&)                 = delete;
    MmapWrapper& operator=(MmapWrapper&&)      = delete;

    const uint8_t* data() const { return static_cast<uint8_t*>(data_); }
    uint8_t* writable_data() { return static_cast<uint8_t*>(data_); }
    size_t size() const { return size_; }
    bool copy_data(void* buf, size_t n, size_t offset) const;

protected:
    MmapWrapper(void* data, size_t size)
        : data_(data), size_(size) {}
    void* data_  = nullptr;
    size_t size_ = 0;
};

std::string path_join(const std::string& p1, const std::string& p2);
std::vector<std::string> split_string(const std::string& str, char delimiter);
void pretty_progress(int step, int steps, float time);
void pretty_bytes_progress(int step, int steps, uint64_t bytes_processed, float elapsed_seconds);

void log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...);

std::string trim(const std::string& s);

std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text);

sd_progress_cb_t sd_get_progress_callback();
void* sd_get_progress_callback_data();

sd_preview_cb_t sd_get_preview_callback();
void* sd_get_preview_callback_data();
preview_t sd_get_preview_mode();
int sd_get_preview_interval();
bool sd_should_preview_denoised();
bool sd_should_preview_noisy();

// test if the backend is a specific one, e.g. "CUDA", "ROCm", "Vulkan" etc.
bool sd_backend_is(ggml_backend_t backend, const std::string& name);

void log_message(const char* format, ...);
#define LOG_DEBUG(...)  log_message(__VA_ARGS__)
#define LOG_INFO(...)  log_message(__VA_ARGS__)
#define LOG_WARN(...)  log_message(__VA_ARGS__)
#define LOG_ERROR(...)  log_message(__VA_ARGS__)
// #define LOG_DEBUG(format, ...) log_printf(SD_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
// #define LOG_INFO(format, ...) log_printf(SD_LOG_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
// #define LOG_WARN(format, ...) log_printf(SD_LOG_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
// #define LOG_ERROR(format, ...) log_printf(SD_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif  // __UTIL_H__
