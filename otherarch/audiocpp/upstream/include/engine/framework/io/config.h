#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::io {

using ConfigMap = std::unordered_map<std::string, std::string>;

ConfigMap parse_flat_json_object(const std::string & text);
ConfigMap parse_key_value_text(const std::string & text, char separator = '=');

ConfigMap load_config_map(const std::filesystem::path & path);
std::vector<std::string> discover_config_keys(const ConfigMap & config);

inline void require_positive(int64_t value, std::string_view label) {
    if (value <= 0) {
        throw std::runtime_error(std::string(label) + " must be positive");
    }
}

inline void require_positive(int value, std::string_view label) {
    require_positive(static_cast<int64_t>(value), label);
}

inline void require_positive(float value, std::string_view label) {
    if (!(value > 0.0F)) {
        throw std::runtime_error(std::string(label) + " must be positive");
    }
}

inline void require_nonnegative(float value, std::string_view label) {
    if (!(value >= 0.0F)) {
        throw std::runtime_error(std::string(label) + " must be non-negative");
    }
}

inline void require_divisible(int64_t value, int64_t divisor, std::string_view label) {
    if (divisor <= 0 || value % divisor != 0) {
        throw std::runtime_error(std::string(label) + " must be divisible");
    }
}

inline void require_all_positive(const std::vector<int64_t> & values, std::string_view label) {
    for (const int64_t value : values) {
        require_positive(value, label);
    }
}

}  // namespace engine::io
