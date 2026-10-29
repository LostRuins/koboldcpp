#pragma once

#include "engine/framework/debug/trace.h"

#include <chrono>
#include <string_view>
#include <utility>

namespace engine::debug {

inline double elapsed_ms(
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point ended) {
    return std::chrono::duration<double, std::milli>(ended - started).count();
}

inline double elapsed_ms(std::chrono::steady_clock::time_point started) {
    return elapsed_ms(started, std::chrono::steady_clock::now());
}

template <typename Fn>
double measure_ms(Fn && fn) {
    const auto started = std::chrono::steady_clock::now();
    std::forward<Fn>(fn)();
    return elapsed_ms(started);
}

class ScopeTimer {
public:
    ScopeTimer(
        LogLevel level,
        std::string_view category,
        std::string_view name);
    ~ScopeTimer();

    ScopeTimer(const ScopeTimer &) = delete;
    ScopeTimer & operator=(const ScopeTimer &) = delete;

private:
    bool active_ = false;
    LogLevel level_ = LogLevel::Info;
    std::string category_;
    std::string name_;
    std::chrono::steady_clock::time_point started_;
};

}  // namespace engine::debug
