#include "engine/framework/io/config.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace engine::io {
namespace {

std::string strip_quotes(std::string value) {
    value = trim_ascii_whitespace(std::move(value));
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') || (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

ConfigMap parse_lines(const std::string & text, char separator) {
    ConfigMap config;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        line = trim_ascii_whitespace(std::move(line));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto pos = line.find(separator);
        if (pos == std::string::npos) {
            continue;
        }
        auto key = trim_ascii_whitespace(line.substr(0, pos));
        auto value = strip_quotes(line.substr(pos + 1));
        if (!key.empty()) {
            config[std::move(key)] = std::move(value);
        }
    }
    return config;
}

}  // namespace

ConfigMap parse_flat_json_object(const std::string & text) {
    const auto root = json::parse(text);
    const auto & object = root.as_object();
    ConfigMap config;
    config.reserve(object.size());
    for (const auto & [key, value] : object) {
        if (value.is_string()) {
            config[key] = value.as_string();
            continue;
        }
        if (value.is_bool()) {
            config[key] = value.as_bool() ? "true" : "false";
            continue;
        }
        if (value.is_null()) {
            config[key] = "null";
            continue;
        }
        if (value.is_number()) {
            const double number = value.as_number();
            if (std::isfinite(number) && std::floor(number) == number) {
                config[key] = std::to_string(static_cast<long long>(number));
            } else {
                std::ostringstream stream;
                stream << number;
                config[key] = stream.str();
            }
            continue;
        }
        throw std::runtime_error("flat json config values must be scalar");
    }
    return config;
}

ConfigMap parse_key_value_text(const std::string & text, char separator) {
    return parse_lines(text, separator);
}

ConfigMap load_config_map(const std::filesystem::path & path) {
    const auto text = read_text_file(path);
    const auto ext = path.extension().string();
    if (ext == ".json") {
        return parse_flat_json_object(text);
    }
    if (ext == ".yaml" || ext == ".yml") {
        // kcpp: The selective integration intentionally does not vendor
        // Audio.cpp's libyaml dependency. First-wave model packages use JSON.
        throw std::runtime_error("YAML configuration is not enabled in the KoboldCpp Audio.cpp integration: " +
                                 path.string());
    }
    return parse_key_value_text(text);
}

std::vector<std::string> discover_config_keys(const ConfigMap & config) {
    std::vector<std::string> keys;
    keys.reserve(config.size());
    for (const auto & [key, _] : config) {
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

}  // namespace engine::io
