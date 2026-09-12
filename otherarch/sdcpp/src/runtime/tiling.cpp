#include "runtime/tiling.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/util.h"
#include "ggml.h"

static void sd_tiling_calc_tiles(int& num_tiles_dim,
                                 float& tile_overlap_factor_dim,
                                 int small_dim,
                                 int tile_size,
                                 const float tile_overlap_factor,
                                 bool circular) {
    int tile_overlap     = static_cast<int>(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    if (circular) {
        // circular means the last and first tile are overlapping (wraping around)
        num_tiles_dim = small_dim / non_tile_overlap;

        if (num_tiles_dim < 1) {
            num_tiles_dim = 1;
        }

        tile_overlap_factor_dim = (tile_size - small_dim / num_tiles_dim) / (float)tile_size;

        // if single tile and tile_overlap_factor is not 0, add one to ensure we have at least two overlapping tiles
        if (num_tiles_dim == 1 && tile_overlap_factor_dim > 0) {
            num_tiles_dim++;
            tile_overlap_factor_dim = 0.5;
        }

        return;
    }
    // else, non-circular means the last and first tile are not overlapping

    num_tiles_dim     = (small_dim - tile_overlap) / non_tile_overlap;
    int overshoot_dim = ((num_tiles_dim + 1) * non_tile_overlap + tile_overlap) % small_dim;

    if ((overshoot_dim != non_tile_overlap) && (overshoot_dim <= num_tiles_dim * (tile_size / 2 - tile_overlap))) {
        // if tiles don't fit perfectly using the desired overlap
        // and there is enough room to squeeze an extra tile without overlap becoming >0.5
        num_tiles_dim++;
    }

    tile_overlap_factor_dim = (float)(tile_size * num_tiles_dim - small_dim) / (float)(tile_size * (num_tiles_dim - 1));
    if (num_tiles_dim <= 2) {
        if (small_dim <= tile_size) {
            num_tiles_dim           = 1;
            tile_overlap_factor_dim = 0;
        } else {
            num_tiles_dim           = 2;
            tile_overlap_factor_dim = (2 * tile_size - small_dim) / (float)tile_size;
        }
    }
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
                               int overlap_x,
                               int overlap_y,
                               bool circular_x,
                               bool circular_y,
                               int x_skip,
                               int y_skip) {
    GGML_ASSERT(output != nullptr);
    int64_t width        = input.shape()[0];
    int64_t height       = input.shape()[1];
    int64_t img_width    = output->shape()[0];
    int64_t img_height   = output->shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(*output);
    int64_t plane_count  = input.numel() / input_plane;
    GGML_ASSERT(output->numel() / output_plane == plane_count);

    // unclamped -> expects x in the range [0-1]
    auto smootherstep_f32 = [](const float x) -> float {
        GGML_ASSERT(x >= 0.f && x <= 1.f);
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };

    for (int iy = y_skip; iy < height; iy++) {
        for (int ix = x_skip; ix < width; ix++) {
            int64_t src_xy = ix + width * iy;
            int64_t ox     = (x + ix) % img_width;
            int64_t oy     = (y + iy) % img_height;
            int64_t dst_xy = ox + img_width * oy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                float new_value = input[plane * input_plane + src_xy];
                if (overlap_x > 0 || overlap_y > 0) {
                    float old_value   = (*output)[plane * output_plane + dst_xy];
                    const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0)) ? (ix - x_skip) / float(overlap_x) : 1.f;
                    const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width))) ? (width - ix) / float(overlap_x) : 1.f;
                    const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0)) ? (iy - y_skip) / float(overlap_y) : 1.f;
                    const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height))) ? (height - iy) / float(overlap_y) : 1.f;
                    const float x_f   = std::min(std::min(x_f_0, x_f_1), 1.f);
                    const float y_f   = std::min(std::min(y_f_0, y_f_1), 1.f);
                    (*output)[plane * output_plane + dst_xy] =
                        old_value + new_value * smootherstep_f32(y_f) * smootherstep_f32(x_f);
                } else {
                    (*output)[plane * output_plane + dst_xy] = new_value;
                }
            }
        }
    }
}

sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                   int output_width,
                                   int output_height,
                                   int scale,
                                   int p_tile_size_x,
                                   int p_tile_size_y,
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

    int small_width  = output_width;
    int small_height = output_height;
    bool decode      = output_width > input_width;
    if (decode) {
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor, circular_x);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor, circular_y);

    int tile_overlap_x     = static_cast<int32_t>(p_tile_size_x * tile_overlap_factor_x);
    int non_tile_overlap_x = p_tile_size_x - tile_overlap_x;
    int tile_overlap_y     = static_cast<int32_t>(p_tile_size_y * tile_overlap_factor_y);
    int non_tile_overlap_y = p_tile_size_y - tile_overlap_y;
    int tile_size_x        = p_tile_size_x < small_width ? p_tile_size_x : small_width;
    int tile_size_y        = p_tile_size_y < small_height ? p_tile_size_y : small_height;
    int input_tile_size_x  = tile_size_x;
    int input_tile_size_y  = tile_size_y;
    int output_tile_size_x = tile_size_x;
    int output_tile_size_y = tile_size_y;
    if (decode) {
        output_tile_size_x *= scale;
        output_tile_size_y *= scale;
    } else {
        input_tile_size_x *= scale;
        input_tile_size_y *= scale;
    }

    int num_tiles   = num_tiles_x * num_tiles_y;
    int tile_count  = 1;
    bool last_y     = false;
    bool last_x     = false;
    float last_time = 0.0f;
    if (!silent) {
        LOG_VERBOSE("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
        LOG_VERBOSE("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);
        LOG_VERBOSE("processing %i tiles", num_tiles);
        pretty_progress(0, num_tiles, 0.0f);
    }
    for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
        int dy = 0;
        if (!circular_y && y + tile_size_y >= small_height) {
            int original_y = y;
            y              = small_height - tile_size_y;
            dy             = original_y - y;
            if (decode) {
                dy *= scale;
            }
            last_y = true;
        }
        for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
            int dx = 0;
            if (!circular_x && x + tile_size_x >= small_width) {
                int original_x = x;
                x              = small_width - tile_size_x;
                dx             = original_x - x;
                if (decode) {
                    dx *= scale;
                }
                last_x = true;
            }

            int x_in  = decode ? x : scale * x;
            int y_in  = decode ? y : scale * y;
            int x_out = decode ? x * scale : x;
            int y_out = decode ? y * scale : y;

            int overlap_x_out = decode ? tile_overlap_x * scale : tile_overlap_x;
            int overlap_y_out = decode ? tile_overlap_y * scale : tile_overlap_y;

            int64_t t1       = ggml_time_ms();
            auto input_tile  = sd_tensor_split_2d(input, input_tile_size_x, input_tile_size_y, x_in, y_in);
            auto output_tile = on_processing(input_tile);
            if (output_tile.empty()) {
                return {};
            }
            GGML_ASSERT(output_tile.shape()[0] == output_tile_size_x && output_tile.shape()[1] == output_tile_size_y);
            if (output.empty()) {
                std::vector<int64_t> output_shape = output_tile.shape();
                output_shape[0]                   = output_width;
                output_shape[1]                   = output_height;
                output                            = sd::Tensor<float>::zeros(std::move(output_shape));
            }
            sd_tensor_merge_2d(output_tile, &output, x_out, y_out, overlap_x_out, overlap_y_out, circular_x, circular_y, dx, dy);

            if (!silent) {
                int64_t t2 = ggml_time_ms();
                last_time  = (t2 - t1) / 1000.0f;
                pretty_progress(tile_count, num_tiles, last_time);
            }
            tile_count++;
        }
        last_x = false;
    }
    if (!silent && tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    if (output.empty()) {
        return {};
    }
    return output;
}
