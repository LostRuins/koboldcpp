#include "runtime/tiling.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/util.h"
#include "ggml.h"

struct TileSpan {
    int offset;
    int overlap_before;
    int overlap_after;
};

static std::vector<TileSpan> sd_tiling_plan_axis(int dimension,
                                                 int tile_size,
                                                 float target_overlap,
                                                 bool circular) {
    if (tile_size >= dimension) {
        return {{0, 0, 0}};
    }

    const int extent = circular ? dimension : dimension - tile_size;
    int intervals    = extent;
    if (tile_size > 1) {
        const float target = extent / ((1.0f - target_overlap) * tile_size);
        // Adjacent tiles must overlap, but three tiles must not cover the same point.
        const int min_intervals = 1 + (extent - 1) / (tile_size - 1);
        const int max_intervals = std::max(1, static_cast<int>(2LL * extent / tile_size));
        const int lower         = std::clamp(static_cast<int>(std::floor(target)), min_intervals, max_intervals);
        const int upper         = std::clamp(static_cast<int>(std::ceil(target)), min_intervals, max_intervals);
        auto error              = [&](int count) {
            return std::abs(1.0f / count - 1.0f / target);
        };
        intervals = error(upper) < error(lower) ? upper : lower;
    }

    const int num_tiles = circular ? intervals : intervals + 1;
    auto position       = [&](int index) -> int {
        return static_cast<int>(int64_t(index) * extent / intervals);
    };
    std::vector<TileSpan> tiles;
    tiles.reserve(num_tiles);
    for (int i = 0; i < num_tiles; ++i) {
        const int offset = position(i);
        // Keep wrapped neighbors in the same unwrapped coordinate space.
        const int previous = i > 0 ? position(i - 1) : position(num_tiles - 1) - dimension;
        const int next     = i + 1 < num_tiles ? position(i + 1) : dimension;
        const int before   = i > 0 || circular ? previous + tile_size - offset : 0;
        const int after    = i + 1 < num_tiles || circular ? offset + tile_size - next : 0;
        tiles.push_back({offset, before, after});
    }
    return tiles;
}

static int64_t sd_tensor_plane_size(const sd::Tensor<float>& tensor) {
    GGML_ASSERT(tensor.dim() >= 2);
    return tensor.shape()[0] * tensor.shape()[1];
}

static sd::Tensor<float> sd_tensor_split_2d(const sd::Tensor<float>& input, int width, int height, int x, int y) {
    GGML_ASSERT(input.dim() >= 4);
    std::vector<int64_t> output_shape = input.shape();
    output_shape[0]                   = width;
    output_shape[1]                   = height;
    sd::Tensor<float> output(std::move(output_shape));
    int64_t input_width  = input.shape()[0];
    int64_t input_height = input.shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(output);
    int64_t plane_count  = input.numel() / input_plane;
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            int64_t src_xy = (ix + x) % input_width + input_width * ((iy + y) % input_height);
            int64_t dst_xy = ix + width * iy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                output[plane * output_plane + dst_xy] = input[plane * input_plane + src_xy];
            }
        }
    }
    return output;
}

static void sd_tensor_merge_2d(const sd::Tensor<float>& input,
                               sd::Tensor<float>* output,
                               int x,
                               int y,
                               int overlap_left,
                               int overlap_right,
                               int overlap_top,
                               int overlap_bottom) {
    GGML_ASSERT(output != nullptr);

    int64_t in_width    = input.shape()[0];
    int64_t in_height   = input.shape()[1];
    int64_t out_width   = output->shape()[0];
    int64_t out_height  = output->shape()[1];
    int64_t in_size     = sd_tensor_plane_size(input);
    int64_t out_size    = sd_tensor_plane_size(*output);
    int64_t plane_count = input.numel() / in_size;

    GGML_ASSERT(output->numel() == plane_count * out_size);
    GGML_ASSERT(x >= 0 && y >= 0);
    GGML_ASSERT(x < out_width && in_width <= out_width);
    GGML_ASSERT(y < out_height && in_height <= out_height);
    GGML_ASSERT(overlap_left >= 0 && overlap_right >= 0);
    GGML_ASSERT(overlap_top >= 0 && overlap_bottom >= 0);

    auto smootherstep_f32 = [](const float x) -> float {
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };
    for (int64_t plane = 0; plane < plane_count; ++plane) {
        for (int iy = 0; iy < in_height; ++iy) {
            float y_f = 1.0f;
            if (iy < overlap_top) {
                y_f = static_cast<float>(iy) / overlap_top;
            }
            if (iy >= in_height - overlap_bottom) {
                y_f = static_cast<float>(in_height - iy) / overlap_bottom;
            }
            const float y_weight = smootherstep_f32(std::clamp(y_f, 0.0f, 1.0f));
            for (int ix = 0; ix < in_width; ++ix) {
                float x_f = 1.0f;
                if (ix < overlap_left) {
                    x_f = static_cast<float>(ix) / overlap_left;
                }
                if (ix >= in_width - overlap_right) {
                    x_f = static_cast<float>(in_width - ix) / overlap_right;
                }
                float x_weight = smootherstep_f32(std::clamp(x_f, 0.0f, 1.0f));
                (*output)[plane * out_size + out_width * ((y + iy) % out_height) + ((x + ix) % out_width)] += x_weight * y_weight * input[plane * in_size + in_width * iy + ix];
            }
        }
    }
}

sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                   int output_width,
                                   int output_height,
                                   int scale,
                                   int p_tile_size_w,
                                   int p_tile_size_h,
                                   float tile_overlap_factor,
                                   bool circular_x,
                                   bool circular_y,
                                   const TileProcessCallback& on_processing,
                                   bool silent) {
    sd::Tensor<float> output;
    int input_width  = static_cast<int>(input.shape()[0]);
    int input_height = static_cast<int>(input.shape()[1]);

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    bool decode      = output_width > input_width;  // scale up
    int small_width  = decode ? input_width : output_width;
    int small_height = decode ? input_height : output_height;
    int scale_in     = decode ? 1 : scale;
    int scale_out    = decode ? scale : 1;

    const auto tiles_x = sd_tiling_plan_axis(small_width, p_tile_size_w, tile_overlap_factor, circular_x);
    const auto tiles_y = sd_tiling_plan_axis(small_height, p_tile_size_h, tile_overlap_factor, circular_y);

    int tile_width         = std::min(p_tile_size_w, small_width);
    int tile_height        = std::min(p_tile_size_h, small_height);
    int input_tile_width   = tile_width * scale_in;
    int input_tile_height  = tile_height * scale_in;
    int output_tile_width  = tile_width * scale_out;
    int output_tile_height = tile_height * scale_out;

    const int num_tiles = static_cast<int>(tiles_x.size() * tiles_y.size());
    int tile_count      = 0;
    if (!silent) {
        LOG_VERBOSE("num tiles : %d, %d ", static_cast<int>(tiles_x.size()), static_cast<int>(tiles_y.size()));
        LOG_VERBOSE("processing %i tiles", num_tiles);
        pretty_progress(0, num_tiles, 0.0f);
    }
    for (const auto& y : tiles_y) {
        for (const auto& x : tiles_x) {
            int64_t t1       = ggml_time_ms();
            auto input_tile  = sd_tensor_split_2d(input, input_tile_width, input_tile_height, x.offset * scale_in, y.offset * scale_in);
            auto output_tile = on_processing(input_tile);
            if (output_tile.empty()) {
                return {};
            }
            GGML_ASSERT(output_tile.shape()[0] == output_tile_width && output_tile.shape()[1] == output_tile_height);
            if (output.empty()) {
                std::vector<int64_t> output_shape = output_tile.shape();
                output_shape[0]                   = output_width;
                output_shape[1]                   = output_height;
                output                            = sd::Tensor<float>::zeros(std::move(output_shape));
            }
            sd_tensor_merge_2d(output_tile, &output,
                               x.offset * scale_out, y.offset * scale_out,
                               x.overlap_before * scale_out, x.overlap_after * scale_out,
                               y.overlap_before * scale_out, y.overlap_after * scale_out);
            ++tile_count;
            if (!silent) {
                pretty_progress(tile_count, num_tiles, (ggml_time_ms() - t1) / 1000.0f);
            }
        }
    }
    return output;
}
