#ifndef __SD_RUNTIME_TILING_H__
#define __SD_RUNTIME_TILING_H__

#include <functional>

#include "core/tensor.hpp"

using TileProcessCallback = std::function<sd::Tensor<float>(const sd::Tensor<float>&)>;

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
                                   bool silent = false);

#endif  // __SD_RUNTIME_TILING_H__
