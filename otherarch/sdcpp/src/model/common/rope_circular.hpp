#ifndef __SD_MODEL_COMMON_ROPE_CIRCULAR_HPP__
#define __SD_MODEL_COMMON_ROPE_CIRCULAR_HPP__

#include "model/common/rope.hpp"

namespace Rope {
    __STATIC_INLINE__ void apply_circular(Embedding& embedding, bool circular_x, bool circular_y) {
        if (!circular_x && !circular_y) {
            return;
        }

        GGML_ASSERT(embedding.batch_size > 0);
        GGML_ASSERT(embedding.ids.size() % embedding.batch_size == 0);
        size_t pos_len  = embedding.ids.size() / embedding.batch_size;
        size_t half_dim = embedding.frequencies.size();
        GGML_ASSERT(embedding.positions.token_count == pos_len);
        GGML_ASSERT(embedding.values.size() == embedding.ids.size() * half_dim * 4);

        constexpr float TWO_PI = 6.28318530717958647692f;
        for (const auto& region : embedding.positions.images) {
            GGML_ASSERT(region.begin <= pos_len && region.count <= pos_len - region.begin);
            for (size_t j = 0; j < half_dim; ++j) {
                const auto& frequency = embedding.frequencies[j];
                float period          = 0.f;
                if (circular_y && frequency.axis == static_cast<size_t>(region.height_axis)) {
                    period = region.height_period;
                } else if (circular_x && frequency.axis == static_cast<size_t>(region.width_axis)) {
                    period = region.width_period;
                }
                if (period <= 0) {
                    continue;
                }

                // Quantize to periodic harmonics while preserving the original coordinate offsets.
                float rounded = std::round(frequency.omega * period / TWO_PI);
                for (int b = 0; b < embedding.batch_size; ++b) {
                    size_t begin = b * pos_len + region.begin;
                    for (size_t i = begin; i < begin + region.count; ++i) {
                        GGML_ASSERT(frequency.axis < embedding.ids[i].size());
                        float angle   = embedding.ids[i][frequency.axis] * TWO_PI * rounded / period;
                        float cos_val = std::cos(angle);
                        float sin_val = std::sin(angle);
                        if (embedding.layout == EmbedNDLayout::ErnieImage) {
                            size_t cos_offset                = (i * half_dim + j) * 2;
                            size_t sin_offset                = embedding.ids.size() * half_dim * 2 + cos_offset;
                            embedding.values[cos_offset]     = cos_val;
                            embedding.values[cos_offset + 1] = cos_val;
                            embedding.values[sin_offset]     = sin_val;
                            embedding.values[sin_offset + 1] = sin_val;
                        } else {
                            size_t offset                = (i * half_dim + j) * 4;
                            embedding.values[offset]     = cos_val;
                            embedding.values[offset + 1] = -sin_val;
                            embedding.values[offset + 2] = sin_val;
                            embedding.values[offset + 3] = cos_val;
                        }
                    }
                }
            }
        }
    }

}  // namespace Rope

#endif  // __SD_MODEL_COMMON_ROPE_CIRCULAR_HPP__
