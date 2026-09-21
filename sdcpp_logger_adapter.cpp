#include "sdcpp_logger_adapter.h"

#include <atomic>

#include "otherarch/sdcpp/src/core/util.h"

static std::atomic<ggml_log_callback> previous_ggml_log_callback{nullptr};
static std::atomic<void*> previous_ggml_log_user_data{nullptr};
static std::atomic<bool> preserve_ggml_logger{false};

void kcpp_sd_preserve_ggml_logger() {
    ggml_log_callback current_callback = nullptr;
    void* current_user_data             = nullptr;
    ggml_log_get(&current_callback, &current_user_data);

    if (current_callback != kcpp_sd_ggml_log_callback) {
        previous_ggml_log_callback.store(current_callback, std::memory_order_relaxed);
        previous_ggml_log_user_data.store(current_user_data, std::memory_order_relaxed);
    }
    preserve_ggml_logger.store(true, std::memory_order_release);
}

void kcpp_sd_ggml_log_callback(ggml_log_level level, const char* text, void*) {
    if (preserve_ggml_logger.load(std::memory_order_acquire)) {
        ggml_log_callback callback = previous_ggml_log_callback.load(std::memory_order_relaxed);
        if (callback != nullptr && callback != kcpp_sd_ggml_log_callback) {
            callback(level, text, previous_ggml_log_user_data.load(std::memory_order_relaxed));
        }
        return;
    }

    sd_ggml_log_callback(level, text, nullptr);
}
