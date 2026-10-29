#pragma once

#include "engine/framework/decoders/tdt_decoder_algorithm.h"
#include "engine/framework/decoders/tdt_decoder_core.h"
#include "engine/framework/decoders/tdt_types.h"

#include <cstdint>
#include <vector>

namespace engine::decoders {

TdtDecodeResult run_tdt_decoder(
    TdtDecoderAlgorithm algorithm,
    TdtDecoderCore & core,
    const std::vector<float> & encoder_projected,
    int64_t frames,
    int64_t hidden_size,
    int32_t blank_id,
    const std::vector<int32_t> & durations,
    int64_t max_symbols_per_step);

}  // namespace engine::decoders
