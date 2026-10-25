#include "engine/framework/io/filesystem.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace engine::io {

bool is_existing_directory(const std::filesystem::path & path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

bool is_existing_file(const std::filesystem::path & path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

std::filesystem::path require_directory(const std::filesystem::path & path, std::string_view role) {
    if (!is_existing_directory(path)) {
        throw std::runtime_error("missing " + std::string(role) + " directory: " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path require_file(const std::filesystem::path & path, std::string_view role) {
    if (!is_existing_file(path)) {
        throw std::runtime_error("missing " + std::string(role) + " file: " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::optional<std::filesystem::path> find_first_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates) {
    for (const auto & candidate : relative_candidates) {
        const auto path = root / candidate;
        if (is_existing_file(path) || is_existing_directory(path)) {
            return std::filesystem::weakly_canonical(path);
        }
    }
    return std::nullopt;
}

std::vector<std::filesystem::path> collect_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates) {
    std::vector<std::filesystem::path> results;
    results.reserve(relative_candidates.size());
    for (const auto & candidate : relative_candidates) {
        const auto path = root / candidate;
        if (is_existing_file(path) || is_existing_directory(path)) {
            results.push_back(std::filesystem::weakly_canonical(path));
        }
    }
    return results;
}

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open text file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace engine::io
