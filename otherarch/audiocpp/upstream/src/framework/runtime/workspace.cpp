#include "engine/framework/runtime/workspace.h"

namespace engine::runtime {

std::vector<float> & RuntimeWorkspace::floats(const std::string & key, size_t size) {
    auto & buffer = float_buffers_[key];
    buffer.resize(size);
    return buffer;
}

std::vector<int32_t> & RuntimeWorkspace::ints(const std::string & key, size_t size) {
    auto & buffer = int_buffers_[key];
    buffer.resize(size);
    return buffer;
}

std::vector<std::byte> & RuntimeWorkspace::bytes(const std::string & key, size_t size) {
    auto & buffer = byte_buffers_[key];
    buffer.resize(size);
    return buffer;
}

void RuntimeWorkspace::clear() {
    float_buffers_.clear();
    int_buffers_.clear();
    byte_buffers_.clear();
}

}  // namespace engine::runtime
