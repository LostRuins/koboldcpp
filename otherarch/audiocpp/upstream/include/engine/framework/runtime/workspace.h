#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::runtime {

class RuntimeWorkspace {
public:
    std::vector<float> & floats(const std::string & key, size_t size);
    std::vector<int32_t> & ints(const std::string & key, size_t size);
    std::vector<std::byte> & bytes(const std::string & key, size_t size);
    void clear();

private:
    std::unordered_map<std::string, std::vector<float>> float_buffers_;
    std::unordered_map<std::string, std::vector<int32_t>> int_buffers_;
    std::unordered_map<std::string, std::vector<std::byte>> byte_buffers_;
};

}  // namespace engine::runtime
