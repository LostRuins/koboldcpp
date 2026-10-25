#include "components/component_weights.h"

#include <random>

namespace engine::models::chatterbox::components {

uint64_t choose_seed(uint64_t seed) {
    if (seed != 0) {
        return seed;
    }
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32U) ^ static_cast<uint64_t>(rd());
}

std::vector<float> read_f32_tensor(
    const engine::assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    return source.require_f32(name, expected_shape);
}

}  // namespace engine::models::chatterbox::components
