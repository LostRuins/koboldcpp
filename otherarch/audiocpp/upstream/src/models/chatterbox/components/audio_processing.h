#pragma once

#include "component_weights.h"

namespace engine::models::chatterbox::components {

std::vector<float> resample_component_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate);

std::vector<float> resample_component_torchaudio_hann_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate);

}  // namespace engine::models::chatterbox::components
