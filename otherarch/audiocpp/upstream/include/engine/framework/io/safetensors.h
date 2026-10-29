#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::io {

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_begin = 0;
    size_t data_end = 0;
};

struct SafeTensorIndex {
    std::filesystem::path source_path;
    std::unordered_map<std::string, std::string> metadata;
    std::unordered_map<std::string, SafeTensorInfo> tensors;
    size_t header_bytes = 0;
};

struct SafeTensorWriteEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<unsigned char> data;
};

SafeTensorIndex load_safetensors_index(const std::filesystem::path & path);
// The bytes of a safetensors file; metadata becomes the string map `__metadata__`.
std::vector<unsigned char> encode_safetensors(
    const std::vector<SafeTensorWriteEntry> & entries,
    const std::vector<std::pair<std::string, std::string>> & metadata = {});
void write_safetensors_file(
    const std::filesystem::path & path,
    const std::vector<SafeTensorWriteEntry> & entries);

}  // namespace engine::io
