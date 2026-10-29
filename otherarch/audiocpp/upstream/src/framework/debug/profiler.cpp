#include "engine/framework/debug/profiler.h"

#include <string>

namespace engine::debug {

ScopeTimer::ScopeTimer(
    LogLevel level,
    std::string_view category,
    std::string_view name)
    : active_(log_enabled()),
      level_(level),
      category_(category),
      name_(name),
      started_(active_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}

ScopeTimer::~ScopeTimer() {
    if (!active_) {
        return;
    }
    const auto ended = std::chrono::steady_clock::now();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(ended - started_).count();
    log_message(level_, category_, name_ + " took " + std::to_string(micros) + " us");
}

}  // namespace engine::debug
