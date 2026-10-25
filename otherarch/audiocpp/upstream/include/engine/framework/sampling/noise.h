#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::sampling {

std::vector<float> generate_normal_noise(std::size_t count, uint32_t seed, float scale = 1.0F);
void clamp_noise(std::vector<float> & noise, float min_value, float max_value);

}  // namespace engine::sampling
