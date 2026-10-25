#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

struct ggml_context;

namespace engine::debug {

struct LoggingConfig {
    bool enabled = false;
    std::optional<std::string> file_path = std::nullopt;
};

enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error,
};

std::string_view to_string(LogLevel level) noexcept;

void configure_logging(const LoggingConfig & config);
void reset_logging();
bool log_enabled();
void log_message(std::string_view line);
void log_message(LogLevel level, std::string_view category, std::string_view message);
bool trace_log_enabled();
bool timing_log_enabled();
std::string format_dims(const std::vector<int64_t> & dims);
std::vector<size_t> sample_point_indices(size_t count, size_t target = 40);
std::string sample_points_f32(const std::vector<float> & values);
std::string sample_points_i32(const std::vector<int32_t> & values);
void trace_log_f32(const std::string & name, const std::vector<int64_t> & dims, const std::vector<float> & values);
void trace_log_i32(const std::string & name, const std::vector<int64_t> & dims, const std::vector<int32_t> & values);
void trace_log_scalar(const std::string & name, std::string_view value);
void trace_log_scalar(const std::string & name, double value);
void trace_log_scalar(const std::string & name, int64_t value);
void trace_log_scalar(const std::string & name, uint64_t value);
void trace_log_scalar(const std::string & name, bool value);
void timing_log_scalar(const std::string & name, std::string_view value);
void timing_log_scalar(const std::string & name, double value);
void timing_log_scalar(const std::string & name, int64_t value);
void timing_log_scalar(const std::string & name, uint64_t value);
void timing_log_scalar(const std::string & name, bool value);
// Logs `<name>.ctx_reserved_mb` (the ggml_init mem_size, a host allocation
// made in full up front) and `<name>.ctx_used_mb` (what the tensor headers in
// it actually take). For a no_alloc context the gap is memory the process
// charges but never touches; call it once the graph is built.
void timing_log_context_reservation(const std::string & name, const ggml_context * ctx);

template <typename Integer, std::enable_if_t<std::is_integral_v<Integer> && !std::is_same_v<std::decay_t<Integer>, bool>, int> = 0>
inline void trace_log_scalar(const std::string & name, Integer value) {
    if constexpr (std::is_signed_v<Integer>) {
        trace_log_scalar(name, static_cast<int64_t>(value));
    } else {
        trace_log_scalar(name, static_cast<uint64_t>(value));
    }
}

template <typename Integer, std::enable_if_t<std::is_integral_v<Integer> && !std::is_same_v<std::decay_t<Integer>, bool>, int> = 0>
inline void timing_log_scalar(const std::string & name, Integer value) {
    if constexpr (std::is_signed_v<Integer>) {
        timing_log_scalar(name, static_cast<int64_t>(value));
    } else {
        timing_log_scalar(name, static_cast<uint64_t>(value));
    }
}

}  // namespace engine::debug
