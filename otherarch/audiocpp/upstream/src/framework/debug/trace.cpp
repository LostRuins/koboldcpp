#include "engine/framework/debug/trace.h"

#include <ggml.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace engine::debug {

namespace {

struct LoggerState {
    std::optional<std::string> file_path;
    std::unique_ptr<std::ofstream> file_stream;
};

LoggerState & logger_state() {
    static LoggerState state;
    return state;
}

std::mutex & logger_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::atomic_bool & logger_enabled_flag() {
    static std::atomic_bool enabled{false};
    return enabled;
}

std::ostream & logger_output_locked(LoggerState & state) {
    if (!state.file_path.has_value() || state.file_path->empty()) {
        return std::cout;
    }
    if (!state.file_stream || !state.file_stream->is_open()) {
        state.file_stream = std::make_unique<std::ofstream>(*state.file_path, std::ios::app);
        if (!state.file_stream->is_open()) {
            throw std::runtime_error("Failed to open log file: " + *state.file_path);
        }
    }
    return *state.file_stream;
}

std::string timestamp_seconds_local() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif
    std::ostringstream out;
    out << std::put_time(&local_tm, "%Y%m%d-%H%M%S");
    return out.str();
}

std::string format_scalar_double(double value) {
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
    if (result.ec == std::errc()) {
        return std::string(buffer, result.ptr);
    }
    std::ostringstream out;
    out.precision(17);
    out << value;
    return out.str();
}

}  // namespace

std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
    case LogLevel::Debug:
        return "debug";
    case LogLevel::Info:
        return "info";
    case LogLevel::Warning:
        return "warning";
    case LogLevel::Error:
        return "error";
    }
    return "unknown";
}

void configure_logging(const LoggingConfig & config) {
    std::lock_guard<std::mutex> lock(logger_mutex());
    auto & state = logger_state();
    logger_enabled_flag().store(false, std::memory_order_release);
    if (state.file_path != config.file_path) {
        state.file_stream.reset();
    }
    state.file_path = config.file_path;
    if (config.enabled && state.file_path.has_value() && !state.file_path->empty()) {
        (void)logger_output_locked(state);
    }
    logger_enabled_flag().store(config.enabled, std::memory_order_release);
}

void reset_logging() {
    configure_logging(LoggingConfig{});
}

bool log_enabled() {
    return logger_enabled_flag().load(std::memory_order_acquire);
}

void log_message(std::string_view line) {
    if (!log_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(logger_mutex());
    if (!log_enabled()) {
        return;
    }
    auto & state = logger_state();
    auto & output = logger_output_locked(state);
    output << line << "\n";
    output.flush();
}

void log_message(LogLevel level, std::string_view category, std::string_view message) {
    if (!log_enabled()) {
        return;
    }
    std::ostringstream out;
    out << "[" << to_string(level) << "]";
    if (!category.empty()) {
        out << "[" << category << "]";
    }
    out << " " << message;
    log_message(out.str());
}

bool trace_log_enabled() {
    return log_enabled();
}

bool timing_log_enabled() {
    return log_enabled();
}

std::string format_dims(const std::vector<int64_t> & dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<size_t> sample_point_indices(size_t count, size_t target) {
    if (count == 0) {
        return {};
    }
    if (count <= target) {
        std::vector<size_t> all(count);
        for (size_t i = 0; i < count; ++i) {
            all[i] = i;
        }
        return all;
    }

    const size_t first_count = 14;
    const size_t middle_count = 12;
    const size_t last_count = 14;
    const size_t first_end = count / 3;
    const size_t middle_end = (count * 2) / 3;

    auto append_range = [](std::vector<size_t> & dst, size_t begin, size_t end, size_t samples) {
        if (begin >= end || samples == 0) {
            return;
        }
        const size_t span = end - begin;
        if (span <= samples) {
            for (size_t i = begin; i < end; ++i) {
                if (dst.empty() || dst.back() != i) {
                    dst.push_back(i);
                }
            }
            return;
        }
        for (size_t i = 0; i < samples; ++i) {
            const double pos = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
            const size_t offset = static_cast<size_t>(pos * static_cast<double>(span - 1));
            const size_t index = begin + offset;
            if (dst.empty() || dst.back() != index) {
                dst.push_back(index);
            }
        }
    };

    std::vector<size_t> points;
    points.reserve(target);
    append_range(points, 0, first_end, first_count);
    append_range(points, first_end, middle_end, middle_count);
    append_range(points, middle_end, count, last_count);

    if (points.empty() || points.front() != 0) {
        points.insert(points.begin(), 0);
    }
    if (points.back() != count - 1) {
        points.push_back(count - 1);
    }
    points.erase(std::unique(points.begin(), points.end()), points.end());
    return points;
}

std::string sample_points_f32(const std::vector<float> & values) {
    if (values.empty()) {
        return "[]";
    }
    const std::vector<size_t> points = sample_point_indices(values.size());
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        const size_t index = points[i];
        oss << index << ":" << values[index];
    }
    oss << "]";
    return oss.str();
}

std::string sample_points_i32(const std::vector<int32_t> & values) {
    if (values.empty()) {
        return "[]";
    }
    const std::vector<size_t> points = sample_point_indices(values.size());
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        const size_t index = points[i];
        oss << index << ":" << values[index];
    }
    oss << "]";
    return oss.str();
}

void trace_log_f32(const std::string & name, const std::vector<int64_t> & dims, const std::vector<float> & values) {
    if (!trace_log_enabled()) {
        return;
    }
    std::ostringstream output;
    output << "[TRACE ts=" << timestamp_seconds_local() << "] " << name
        << " shape=" << format_dims(dims)
        << " size=" << values.size()
        << " samples=" << sample_points_f32(values);
    log_message(output.str());
}

void trace_log_i32(const std::string & name, const std::vector<int64_t> & dims, const std::vector<int32_t> & values) {
    if (!trace_log_enabled()) {
        return;
    }
    std::ostringstream output;
    output << "[TRACE ts=" << timestamp_seconds_local() << "] " << name
        << " shape=" << format_dims(dims)
        << " size=" << values.size()
        << " samples=" << sample_points_i32(values);
    log_message(output.str());
}

void trace_log_scalar(const std::string & name, std::string_view value) {
    if (!trace_log_enabled()) {
        return;
    }
    std::ostringstream output;
    output << "[TRACE ts=" << timestamp_seconds_local() << "] " << name << " " << value;
    log_message(output.str());
}

void trace_log_scalar(const std::string & name, double value) {
    if (!trace_log_enabled()) {
        return;
    }
    trace_log_scalar(name, format_scalar_double(value));
}

void trace_log_scalar(const std::string & name, int64_t value) {
    if (!trace_log_enabled()) {
        return;
    }
    trace_log_scalar(name, std::to_string(value));
}

void trace_log_scalar(const std::string & name, uint64_t value) {
    if (!trace_log_enabled()) {
        return;
    }
    trace_log_scalar(name, std::to_string(value));
}

void trace_log_scalar(const std::string & name, bool value) {
    if (!trace_log_enabled()) {
        return;
    }
    trace_log_scalar(name, std::string_view(value ? "1" : "0"));
}

void timing_log_scalar(const std::string & name, std::string_view value) {
    if (!timing_log_enabled()) {
        return;
    }
    std::ostringstream output;
    output << "[TIMING ts=" << timestamp_seconds_local() << "] " << name << " " << value;
    log_message(output.str());
}

void timing_log_scalar(const std::string & name, double value) {
    if (!timing_log_enabled()) {
        return;
    }
    timing_log_scalar(name, format_scalar_double(value));
}

void timing_log_scalar(const std::string & name, int64_t value) {
    if (!timing_log_enabled()) {
        return;
    }
    timing_log_scalar(name, std::to_string(value));
}

void timing_log_scalar(const std::string & name, uint64_t value) {
    if (!timing_log_enabled()) {
        return;
    }
    timing_log_scalar(name, std::to_string(value));
}

void timing_log_context_reservation(const std::string & name, const ggml_context * ctx) {
    if (!timing_log_enabled() || ctx == nullptr) {
        return;
    }
    constexpr double kMiB = 1024.0 * 1024.0;
    timing_log_scalar(name + ".ctx_reserved_mb", static_cast<double>(ggml_get_mem_size(ctx)) / kMiB);
    timing_log_scalar(name + ".ctx_used_mb", static_cast<double>(ggml_used_mem(ctx)) / kMiB);
}

void timing_log_scalar(const std::string & name, bool value) {
    if (!timing_log_enabled()) {
        return;
    }
    timing_log_scalar(name, std::string_view(value ? "1" : "0"));
}

}  // namespace engine::debug
