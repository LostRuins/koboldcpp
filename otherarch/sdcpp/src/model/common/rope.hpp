#ifndef __SD_MODEL_COMMON_ROPE_HPP__
#define __SD_MODEL_COMMON_ROPE_HPP__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <utility>
#include <vector>
#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "core/util.h"

namespace Rope {
    enum class EmbedNDLayout {
        Matrix,
        ErnieImage,
    };

    struct SpatialRegion {
        size_t begin;
        size_t count;
        float height_period;
        float width_period;
        int height_axis = 1;
        int width_axis  = 2;
    };

    struct PositionLayout {
        // Token ranges are relative to one batch item.
        std::vector<SpatialRegion> images;
        size_t token_count = 0;

        void append_tokens(size_t count) {
            token_count += count;
        }

        void append_image(int height, int width, int frames = 1, float height_step = 1.f, float width_step = 1.f) {
            size_t count = static_cast<size_t>(height) * width * frames;
            images.push_back({token_count, count, height * height_step, width * width_step});
            append_tokens(count);
        }
    };

    struct Frequency {
        size_t axis;
        float omega;
    };

    struct Embedding {
        std::vector<float> values;
        std::vector<std::vector<float>> ids;
        PositionLayout positions;
        std::vector<Frequency> frequencies;
        EmbedNDLayout layout = EmbedNDLayout::Matrix;
        int batch_size       = 1;
    };

    enum class RefIndexMode {
        FIXED,
        INCREASE,
        DECREASE,
    };

    template <class T>
    __STATIC_INLINE__ std::vector<T> linspace(T start, T end, int num) {
        std::vector<T> result(num);
        if (num == 1) {
            result[0] = start;
            return result;
        }
        T step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) {
            result[i] = start + i * step;
        }
        return result;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
        size_t rows = mat.size();
        size_t cols = mat[0].size();
        std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    __STATIC_INLINE__ std::vector<float> flatten(const std::vector<std::vector<float>>& vec) {
        std::vector<float> flat_vec;
        for (const auto& sub_vec : vec) {
            flat_vec.insert(flat_vec.end(), sub_vec.begin(), sub_vec.end());
        }
        return flat_vec;
    }

    __STATIC_INLINE__ std::vector<float> rope_frequencies(int dim, float theta) {
        assert(dim % 2 == 0);
        int half_dim             = dim / 2;
        std::vector<float> scale = linspace(0.f, (dim * 1.f - 2) / dim, half_dim);
        std::vector<float> omega(half_dim);
        for (int i = 0; i < half_dim; ++i) {
            omega[i] = 1.0f / ::powf(1.f * theta, scale[i]);
        }
        return omega;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> rope(const std::vector<float>& pos,
                                                           const std::vector<float>& omega) {
        int half_dim    = static_cast<int>(omega.size());
        size_t pos_size = pos.size();
        std::vector<std::vector<float>> out(pos_size, std::vector<float>(half_dim));
        for (size_t i = 0; i < pos_size; ++i) {
            for (size_t j = 0; j < half_dim; ++j) {
                float angle = pos[i] * omega[j];

                out[i][j] = angle;
            }
        }

        std::vector<std::vector<float>> result(pos_size, std::vector<float>(half_dim * 4));
        for (int i = 0; i < pos_size; ++i) {
            for (int j = 0; j < half_dim; ++j) {
                result[i][4 * j]     = std::cos(out[i][j]);
                result[i][4 * j + 1] = -std::sin(out[i][j]);
                result[i][4 * j + 2] = std::sin(out[i][j]);
                result[i][4 * j + 3] = std::cos(out[i][j]);
            }
        }

        return result;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> rope(const std::vector<float>& pos,
                                                           int dim,
                                                           float theta) {
        return rope(pos, rope_frequencies(dim, theta));
    }

    // Generate IDs for image patches and text
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_flux_txt_ids(int bs, int context_len, int axes_dim_num, std::set<int> arange_dims) {
        auto txt_ids = std::vector<std::vector<float>>(bs * context_len, std::vector<float>(axes_dim_num, 0.0f));
        for (int dim = 0; dim < axes_dim_num; dim++) {
            if (arange_dims.find(dim) != arange_dims.end()) {
                for (int i = 0; i < bs * context_len; i++) {
                    txt_ids[i][dim] = 1.f * (i % context_len);
                }
            }
        }
        return txt_ids;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_longcat_txt_ids(int bs, int context_len, int axes_dim_num) {
        auto txt_ids = std::vector<std::vector<float>>(bs * context_len, std::vector<float>(axes_dim_num, 0.0f));
        for (int i = 0; i < bs * context_len; i++) {
            float token_index = static_cast<float>(i % context_len);
            txt_ids[i][1]     = token_index;
            txt_ids[i][2]     = token_index;
        }
        return txt_ids;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_flux_img_ids(int h,
                                                                       int w,
                                                                       int patch_size,
                                                                       int bs,
                                                                       int axes_dim_num,
                                                                       int index              = 0,
                                                                       int h_offset           = 0,
                                                                       int w_offset           = 0,
                                                                       bool scale_rope        = false,
                                                                       PositionLayout* layout = nullptr) {
        int h_len = (h + (patch_size / 2)) / patch_size;
        int w_len = (w + (patch_size / 2)) / patch_size;
        if (layout) {
            layout->append_image(h_len, w_len);
        }
        std::vector<std::vector<float>> img_ids(h_len * w_len, std::vector<float>(axes_dim_num, 0.0));

        int h_start = h_offset;
        int w_start = w_offset;

        if (scale_rope) {
            h_start -= h_len / 2;
            w_start -= w_len / 2;
        }

        std::vector<float> row_ids = linspace<float>(1.f * h_start, 1.f * h_start + h_len - 1, h_len);
        std::vector<float> col_ids = linspace<float>(1.f * w_start, 1.f * w_start + w_len - 1, w_len);
        for (int i = 0; i < h_len; ++i) {
            for (int j = 0; j < w_len; ++j) {
                img_ids[i * w_len + j][0] = 1.f * index;
                img_ids[i * w_len + j][1] = row_ids[i];
                img_ids[i * w_len + j][2] = col_ids[j];
            }
        }

        std::vector<std::vector<float>> img_ids_repeated(bs * img_ids.size(), std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < img_ids.size(); ++j) {
                img_ids_repeated[i * img_ids.size() + j] = img_ids[j];
            }
        }
        return img_ids_repeated;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> concat_ids(const std::vector<std::vector<float>>& a,
                                                                 const std::vector<std::vector<float>>& b,
                                                                 int bs) {
        size_t a_len = a.size() / bs;
        size_t b_len = b.size() / bs;
        std::vector<std::vector<float>> ids(a.size() + b.size(), std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < a_len; ++j) {
                ids[i * (a_len + b_len) + j] = a[i * a_len + j];
            }
            for (int j = 0; j < b_len; ++j) {
                ids[i * (a_len + b_len) + a_len + j] = b[i * b_len + j];
            }
        }
        return ids;
    }

    __STATIC_INLINE__ std::vector<float> embed_nd(const std::vector<std::vector<float>>& ids,
                                                  int bs,
                                                  const std::vector<float>& axis_thetas,
                                                  const std::vector<int>& axes_dim,
                                                  EmbedNDLayout layout                = EmbedNDLayout::Matrix,
                                                  std::vector<Frequency>* frequencies = nullptr) {
        std::vector<std::vector<float>> trans_ids = transpose(ids);
        size_t pos_len                            = ids.size() / bs;
        size_t num_axes                           = axes_dim.size();
        // for (int i = 0; i < pos_len; i++) {
        //     std::cout << trans_ids[0][i] << " " << trans_ids[1][i] << " " << trans_ids[2][i] << std::endl;
        // }

        int emb_dim = 0;
        for (int d : axes_dim)
            emb_dim += d / 2;

        if (frequencies) {
            frequencies->clear();
            frequencies->reserve(emb_dim);
        }
        std::vector<std::vector<float>> emb(bs * pos_len, std::vector<float>(emb_dim * 2 * 2, 0.0));
        size_t offset = 0;
        for (size_t i = 0; i < num_axes; ++i) {
            float axis_theta = 10000.0f;
            if (!axis_thetas.empty()) {
                axis_theta = axis_thetas[std::min(i, axis_thetas.size() - 1)];
            }
            auto omega = rope_frequencies(axes_dim[i], axis_theta);
            if (frequencies) {
                for (float frequency : omega) {
                    frequencies->push_back({i, frequency});
                }
            }
            std::vector<std::vector<float>> rope_emb =
                rope(trans_ids[i], omega);  // [bs*pos_len, axes_dim[i]/2 * 2 * 2]
            for (int b = 0; b < bs; ++b) {
                for (int j = 0; j < pos_len; ++j) {
                    for (int k = 0; k < rope_emb[0].size(); ++k) {
                        emb[b * pos_len + j][offset + k] = rope_emb[j][k];
                    }
                }
            }
            offset += rope_emb[0].size();
        }

        if (layout == EmbedNDLayout::ErnieImage) {
            int head_dim = emb_dim * 2;
            std::vector<float> ernie_emb(bs * pos_len * head_dim * 2, 0.0f);
            for (size_t pos_idx = 0; pos_idx < bs * pos_len; ++pos_idx) {
                for (int i = 0; i < emb_dim; ++i) {
                    float cos_val             = emb[pos_idx][4 * i];
                    float sin_val             = emb[pos_idx][4 * i + 2];
                    size_t cos_offset         = pos_idx * head_dim + 2 * i;
                    size_t sin_offset         = bs * pos_len * head_dim + cos_offset;
                    ernie_emb[cos_offset]     = cos_val;
                    ernie_emb[cos_offset + 1] = cos_val;
                    ernie_emb[sin_offset]     = sin_val;
                    ernie_emb[sin_offset + 1] = sin_val;
                }
            }
            return ernie_emb;
        }

        return flatten(emb);
    }

    __STATIC_INLINE__ std::vector<float> embed_nd(const std::vector<std::vector<float>>& ids,
                                                  int bs,
                                                  float theta,
                                                  const std::vector<int>& axes_dim,
                                                  EmbedNDLayout layout                = EmbedNDLayout::Matrix,
                                                  std::vector<Frequency>* frequencies = nullptr) {
        std::vector<float> axis_thetas(axes_dim.size(), theta);
        return embed_nd(ids, bs, axis_thetas, axes_dim, layout, frequencies);
    }

    __STATIC_INLINE__ std::vector<float> embed_interleaved_mrope(const std::vector<std::vector<float>>& ids,
                                                                 int bs,
                                                                 float theta,
                                                                 int head_dim,
                                                                 const std::vector<int>& mrope_section,
                                                                 std::vector<Frequency>* frequencies = nullptr) {
        GGML_ASSERT(bs > 0);
        GGML_ASSERT(head_dim % 2 == 0);
        GGML_ASSERT(mrope_section.size() >= 3);

        std::vector<std::vector<float>> trans_ids = transpose(ids);
        size_t pos_len                            = ids.size() / bs;
        int half_dim                              = head_dim / 2;

        auto omega = rope_frequencies(head_dim, theta);
        if (frequencies) {
            frequencies->clear();
            for (float frequency : omega) {
                frequencies->push_back({0, frequency});
            }
        }
        std::vector<std::vector<std::vector<float>>> axis_embs;
        axis_embs.reserve(3);
        for (int axis = 0; axis < 3; ++axis) {
            axis_embs.push_back(rope(trans_ids[axis], omega));
        }

        std::vector<std::vector<float>> emb = axis_embs[0];
        for (int axis = 1; axis < 3; ++axis) {
            int length = std::min<int>(mrope_section[axis] * 3, half_dim);
            for (int freq_idx = axis; freq_idx < length; freq_idx += 3) {
                if (frequencies) {
                    (*frequencies)[freq_idx].axis = axis;
                }
                for (size_t pos_idx = 0; pos_idx < bs * pos_len; ++pos_idx) {
                    for (int k = 0; k < 4; ++k) {
                        emb[pos_idx][4 * freq_idx + k] = axis_embs[axis][pos_idx][4 * freq_idx + k];
                    }
                }
            }
        }

        return flatten(emb);
    }

    __STATIC_INLINE__ Embedding embed_2d_interleaved(int height,
                                                     int width,
                                                     int dim,
                                                     float theta    = 10000.f,
                                                     float scale    = 16.f,
                                                     int ref_grid_h = 0,
                                                     int ref_grid_w = 0) {
        assert(dim % 4 == 0);
        int half_dim      = dim / 2;
        int dim_axis      = dim / 2;
        int axis_half_dim = dim_axis / 2;

        float h_ntk = 1.f;
        float w_ntk = 1.f;
        if (ref_grid_h > 0 && ref_grid_w > 0 && dim_axis > 2) {
            float power = static_cast<float>(dim_axis) / static_cast<float>(dim_axis - 2);
            h_ntk       = std::pow(static_cast<float>(height) / static_cast<float>(ref_grid_h), power);
            w_ntk       = std::pow(static_cast<float>(width) / static_cast<float>(ref_grid_w), power);
        }

        Embedding result;
        result.positions.append_image(height, width, 1,
                                      height > 1 ? scale / (height - 1) : 1.f,
                                      width > 1 ? scale / (width - 1) : 1.f);
        std::vector<float> x_pos;
        std::vector<float> y_pos;
        x_pos.reserve(static_cast<size_t>(height) * width);
        y_pos.reserve(static_cast<size_t>(height) * width);
        for (int iy = 0; iy < height; ++iy) {
            float y = height == 1 ? 0.f : scale * static_cast<float>(iy) / static_cast<float>(height - 1);
            for (int ix = 0; ix < width; ++ix) {
                float x = width == 1 ? 0.f : scale * static_cast<float>(ix) / static_cast<float>(width - 1);
                result.ids.push_back({0.f, y, x});
                x_pos.push_back(x);
                y_pos.push_back(y);
            }
        }

        auto x_freq = rope_frequencies(dim_axis, theta * w_ntk);
        auto y_freq = rope_frequencies(dim_axis, theta * h_ntk);
        auto x_emb  = rope(x_pos, x_freq);
        auto y_emb  = rope(y_pos, y_freq);
        for (int i = 0; i < axis_half_dim; ++i) {
            result.frequencies.push_back({2, x_freq[i]});
            result.frequencies.push_back({1, y_freq[i]});
        }

        std::vector<float> out(static_cast<size_t>(height) * width * half_dim * 4);
        for (int pos = 0; pos < height * width; ++pos) {
            for (int i = 0; i < axis_half_dim; ++i) {
                int jx        = 2 * i;
                int jy        = 2 * i + 1;
                size_t base_x = static_cast<size_t>(pos) * half_dim * 4 + static_cast<size_t>(jx) * 4;
                size_t base_y = static_cast<size_t>(pos) * half_dim * 4 + static_cast<size_t>(jy) * 4;
                size_t axis   = static_cast<size_t>(i) * 4;
                for (int k = 0; k < 4; ++k) {
                    out[base_x + k] = x_emb[pos][axis + k];
                    out[base_y + k] = y_emb[pos][axis + k];
                }
            }
        }
        result.values = std::move(out);
        return result;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_refs_ids(int patch_size,
                                                                   int bs,
                                                                   int axes_dim_num,
                                                                   int start_index,
                                                                   const std::vector<ggml_tensor*>& ref_latents,
                                                                   RefIndexMode ref_index_mode,
                                                                   float ref_index_scale,
                                                                   bool scale_rope,
                                                                   int base_offset        = 0,
                                                                   PositionLayout* layout = nullptr) {
        std::vector<std::vector<float>> ids;
        int curr_h_offset = 0;
        int curr_w_offset = 0;
        int index         = start_index;
        for (ggml_tensor* ref : ref_latents) {
            int h_offset = 0;
            int w_offset = 0;
            if (ref_index_mode == RefIndexMode::FIXED) {
                if (ref->ne[1] + curr_h_offset > ref->ne[0] + curr_w_offset) {
                    w_offset = curr_w_offset;
                } else {
                    h_offset = curr_h_offset;
                }
                scale_rope = false;
            } else if (ref_index_mode == RefIndexMode::DECREASE) {
                index--;
            }

            auto ref_ids = gen_flux_img_ids(static_cast<int>(ref->ne[1]),
                                            static_cast<int>(ref->ne[0]),
                                            patch_size,
                                            bs,
                                            axes_dim_num,
                                            static_cast<int>(index * ref_index_scale),
                                            h_offset + base_offset,
                                            w_offset + base_offset,
                                            scale_rope,
                                            layout);
            ids          = concat_ids(ids, ref_ids, bs);

            if (ref_index_mode == RefIndexMode::INCREASE) {
                index++;
            }

            curr_h_offset = std::max(curr_h_offset, static_cast<int>(ref->ne[1]) + h_offset);
            curr_w_offset = std::max(curr_w_offset, static_cast<int>(ref->ne[0]) + w_offset);
        }
        return ids;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_flux_ids(int h,
                                                                   int w,
                                                                   int patch_size,
                                                                   int bs,
                                                                   int axes_dim_num,
                                                                   int context_len,
                                                                   std::set<int> txt_arange_dims,
                                                                   const std::vector<ggml_tensor*>& ref_latents,
                                                                   RefIndexMode ref_index_mode,
                                                                   float ref_index_scale,
                                                                   bool is_longcat,
                                                                   PositionLayout* layout = nullptr) {
        if (layout) {
            layout->append_tokens(context_len);
        }
        int x_index = is_longcat ? 1 : 0;

        auto txt_ids = is_longcat ? gen_longcat_txt_ids(bs, context_len, axes_dim_num) : gen_flux_txt_ids(bs, context_len, axes_dim_num, txt_arange_dims);
        int offset   = is_longcat ? context_len : 0;
        auto img_ids = gen_flux_img_ids(h, w, patch_size, bs, axes_dim_num, x_index, offset, offset, false, layout);

        auto ids = concat_ids(txt_ids, img_ids, bs);
        if (ref_latents.size() > 0) {
            auto refs_ids = gen_refs_ids(patch_size, bs, axes_dim_num, x_index + 1, ref_latents, ref_index_mode, ref_index_scale, false, offset, layout);
            ids           = concat_ids(ids, refs_ids, bs);
        }
        return ids;
    }

    // Generate flux positional embeddings
    __STATIC_INLINE__ Embedding gen_flux_pe(int h,
                                            int w,
                                            int patch_size,
                                            int bs,
                                            int context_len,
                                            std::set<int> txt_arange_dims,
                                            const std::vector<ggml_tensor*>& ref_latents,
                                            RefIndexMode ref_index_mode,
                                            float ref_index_scale,
                                            int theta,
                                            const std::vector<int>& axes_dim,
                                            bool is_longcat) {
        Embedding result;
        result.batch_size = bs;
        result.ids        = gen_flux_ids(h,
                                         w,
                                         patch_size,
                                         bs,
                                         static_cast<int>(axes_dim.size()),
                                         context_len,
                                         txt_arange_dims,
                                         ref_latents,
                                         ref_index_mode,
                                         ref_index_scale,
                                         is_longcat, &result.positions);
        result.values     = embed_nd(result.ids, bs, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_vid_ids(int t,
                                                                  int h,
                                                                  int w,
                                                                  int pt,
                                                                  int ph,
                                                                  int pw,
                                                                  int bs,
                                                                  int t_offset           = 0,
                                                                  int h_offset           = 0,
                                                                  int w_offset           = 0,
                                                                  bool scale_rope        = false,
                                                                  PositionLayout* layout = nullptr) {
        int t_len = (t + (pt / 2)) / pt;
        int h_len = (h + (ph / 2)) / ph;
        int w_len = (w + (pw / 2)) / pw;

        if (layout) {
            layout->append_image(h_len, w_len, t_len);
        }
        std::vector<std::vector<float>> vid_ids(t_len * h_len * w_len, std::vector<float>(3, 0.0));

        if (scale_rope) {
            h_offset -= h_len / 2;
            w_offset -= w_len / 2;
        }

        std::vector<float> t_ids = linspace<float>(1.f * t_offset, 1.f * t_len - 1 + t_offset, t_len);
        std::vector<float> h_ids = linspace<float>(1.f * h_offset, 1.f * h_len - 1 + h_offset, h_len);
        std::vector<float> w_ids = linspace<float>(1.f * w_offset, 1.f * w_len - 1 + w_offset, w_len);

        for (int i = 0; i < t_len; ++i) {
            for (int j = 0; j < h_len; ++j) {
                for (int k = 0; k < w_len; ++k) {
                    int idx         = i * h_len * w_len + j * w_len + k;
                    vid_ids[idx][0] = t_ids[i];
                    vid_ids[idx][1] = h_ids[j];
                    vid_ids[idx][2] = w_ids[k];
                }
            }
        }

        std::vector<std::vector<float>> vid_ids_repeated(bs * vid_ids.size(), std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < vid_ids.size(); ++j) {
                vid_ids_repeated[i * vid_ids.size() + j] = vid_ids[j];
            }
        }
        return vid_ids_repeated;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_hunyuan_video_ids(int t,
                                                                            int h,
                                                                            int w,
                                                                            int patch_t,
                                                                            int patch_h,
                                                                            int patch_w,
                                                                            int bs,
                                                                            int context_len) {
        std::vector<std::vector<float>> txt_ids(bs * context_len, std::vector<float>(3, 0.0f));
        auto img_ids = gen_vid_ids(t, h, w, patch_t, patch_h, patch_w, bs);
        return concat_ids(txt_ids, img_ids, bs);
    }

    __STATIC_INLINE__ std::vector<float> gen_hunyuan_video_pe(int t,
                                                              int h,
                                                              int w,
                                                              int patch_t,
                                                              int patch_h,
                                                              int patch_w,
                                                              int bs,
                                                              int context_len,
                                                              float theta,
                                                              const std::vector<int>& axes_dim) {
        auto ids = gen_hunyuan_video_ids(t, h, w, patch_t, patch_h, patch_w, bs, context_len);
        return embed_nd(ids, bs, theta, axes_dim);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_qwen_image_ids(int t,
                                                                         int h,
                                                                         int w,
                                                                         int patch_size,
                                                                         int bs,
                                                                         int context_len,
                                                                         const std::vector<ggml_tensor*>& ref_latents,
                                                                         RefIndexMode ref_index_mode,
                                                                         PositionLayout* layout = nullptr) {
        if (layout) {
            layout->append_tokens(context_len);
        }
        int h_len        = (h + (patch_size / 2)) / patch_size;
        int w_len        = (w + (patch_size / 2)) / patch_size;
        int txt_id_start = std::max(h_len, w_len) / 2;
        auto txt_ids     = linspace<float>(1.f * txt_id_start, 1.f * txt_id_start + context_len - 1, context_len);
        std::vector<std::vector<float>> txt_ids_repeated(bs * context_len, std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < txt_ids.size(); ++j) {
                txt_ids_repeated[i * txt_ids.size() + j] = {txt_ids[j], txt_ids[j], txt_ids[j]};
            }
        }
        int axes_dim_num = 3;
        auto img_ids     = gen_vid_ids(t, h, w, 1, patch_size, patch_size, bs, 0, 0, 0, true, layout);
        auto ids         = concat_ids(txt_ids_repeated, img_ids, bs);
        if (ref_latents.size() > 0) {
            int ref_start_index = ref_index_mode == RefIndexMode::DECREASE ? 0 : 1;
            auto refs_ids       = gen_refs_ids(patch_size, bs, axes_dim_num, ref_start_index, ref_latents, ref_index_mode, 1.f, true, 0, layout);
            ids                 = concat_ids(ids, refs_ids, bs);
        }
        return ids;
    }

    // Generate qwen_image positional embeddings
    __STATIC_INLINE__ Embedding gen_qwen_image_pe(int t,
                                                  int h,
                                                  int w,
                                                  int patch_size,
                                                  int bs,
                                                  int context_len,
                                                  const std::vector<ggml_tensor*>& ref_latents,
                                                  RefIndexMode ref_index_mode,
                                                  int theta,
                                                  const std::vector<int>& axes_dim) {
        Embedding result;
        result.batch_size = bs;
        result.ids        = gen_qwen_image_ids(t, h, w, patch_size, bs, context_len, ref_latents, ref_index_mode, &result.positions);
        result.values     = embed_nd(result.ids, bs, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    __STATIC_INLINE__ Embedding gen_mage_flow_pe(int h,
                                                 int w,
                                                 int bs,
                                                 int context_len,
                                                 const std::vector<ggml_tensor*>& ref_latents,
                                                 int theta,
                                                 const std::vector<int>& axes_dim) {
        Embedding result;
        result.batch_size = bs;
        result.positions.append_tokens(context_len);
        const int axes_dim_num = static_cast<int>(axes_dim.size());
        auto make_image_ids    = [=, &result](int image_h, int image_w, int image_index) {
            std::vector<std::vector<float>> image_ids(static_cast<size_t>(bs) * image_h * image_w,
                                                         std::vector<float>(axes_dim_num, 0.f));
            result.positions.append_image(image_h, image_w);
            int h_start = -(image_h - image_h / 2);
            int w_start = -(image_w - image_w / 2);
            for (int b = 0; b < bs; ++b) {
                for (int y = 0; y < image_h; ++y) {
                    for (int x = 0; x < image_w; ++x) {
                        auto& id = image_ids[static_cast<size_t>(b) * image_h * image_w + y * image_w + x];
                        id[0]    = static_cast<float>(image_index);
                        id[1]    = static_cast<float>(h_start + y);
                        id[2]    = static_cast<float>(w_start + x);
                    }
                }
            }
            return image_ids;
        };
        auto ids     = gen_flux_txt_ids(bs, context_len, axes_dim_num, {});
        auto img_ids = make_image_ids(h, w, 0);
        ids          = concat_ids(ids, img_ids, bs);
        for (size_t i = 0; i < ref_latents.size(); ++i) {
            auto ref_ids = make_image_ids(static_cast<int>(ref_latents[i]->ne[1]),
                                          static_cast<int>(ref_latents[i]->ne[0]),
                                          static_cast<int>(i + 1));
            ids          = concat_ids(ids, ref_ids, bs);
        }
        result.ids    = std::move(ids);
        result.values = embed_nd(result.ids, bs, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_lens_ids(int h,
                                                                   int w,
                                                                   int bs,
                                                                   int context_len,
                                                                   bool scale_rope        = true,
                                                                   PositionLayout* layout = nullptr) {
        auto img_ids_repeated = gen_flux_img_ids(h, w, 1, bs, 3, 0, 0, 0, scale_rope, layout);

        int txt_id_start = scale_rope ? std::max(h / 2, w / 2) : 0;
        auto txt_ids     = linspace<float>(1.f * txt_id_start, 1.f * context_len + txt_id_start, context_len);
        std::vector<std::vector<float>> txt_ids_repeated(bs * context_len, std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < txt_ids.size(); ++j) {
                txt_ids_repeated[i * txt_ids.size() + j] = {txt_ids[j], txt_ids[j], txt_ids[j]};
            }
        }

        if (layout) {
            layout->append_tokens(context_len);
        }
        return concat_ids(img_ids_repeated, txt_ids_repeated, bs);
    }

    __STATIC_INLINE__ Embedding gen_lens_pe(int h,
                                            int w,
                                            int bs,
                                            int context_len,
                                            int theta,
                                            const std::vector<int>& axes_dim) {
        Embedding result;
        result.batch_size = bs;
        result.ids        = gen_lens_ids(h, w, bs, context_len, true, &result.positions);
        result.values     = embed_nd(result.ids, bs, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_ernie_image_ids(int h,
                                                                          int w,
                                                                          int patch_size,
                                                                          int bs,
                                                                          int context_len,
                                                                          PositionLayout* layout = nullptr) {
        int h_len = h / patch_size;
        int w_len = w / patch_size;

        if (layout) {
            layout->append_image(h_len, w_len);
        }
        std::vector<std::vector<float>> img_ids(h_len * w_len, std::vector<float>(3, 0.0f));
        std::vector<float> h_ids = linspace<float>(0.f, static_cast<float>(h_len - 1), h_len);
        std::vector<float> w_ids = linspace<float>(0.f, static_cast<float>(w_len - 1), w_len);
        for (int i = 0; i < h_len; ++i) {
            for (int j = 0; j < w_len; ++j) {
                img_ids[i * w_len + j][0] = static_cast<float>(context_len);
                img_ids[i * w_len + j][1] = h_ids[i];
                img_ids[i * w_len + j][2] = w_ids[j];
            }
        }

        std::vector<std::vector<float>> img_ids_repeated(bs * img_ids.size(), std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < static_cast<int>(img_ids.size()); ++j) {
                img_ids_repeated[i * img_ids.size() + j] = img_ids[j];
            }
        }

        std::vector<std::vector<float>> txt_ids(bs * context_len, std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < context_len; ++j) {
                txt_ids[i * context_len + j][0] = static_cast<float>(j);
            }
        }

        if (layout) {
            layout->append_tokens(context_len);
        }
        return concat_ids(img_ids_repeated, txt_ids, bs);
    }

    __STATIC_INLINE__ Embedding gen_ernie_image_pe(int h,
                                                   int w,
                                                   int patch_size,
                                                   int bs,
                                                   int context_len,
                                                   int theta,
                                                   const std::vector<int>& axes_dim) {
        Embedding result;
        result.batch_size = bs;
        result.layout     = EmbedNDLayout::ErnieImage;
        result.ids        = gen_ernie_image_ids(h, w, patch_size, bs, context_len, &result.positions);
        result.values     = embed_nd(result.ids, bs, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    // Generate wan positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_wan_pe(int t,
                                                    int h,
                                                    int w,
                                                    int pt,
                                                    int ph,
                                                    int pw,
                                                    int bs,
                                                    int theta,
                                                    const std::vector<int>& axes_dim,
                                                    int t_offset = 0) {
        std::vector<std::vector<float>> ids = gen_vid_ids(t, h, w, pt, ph, pw, bs, t_offset);
        return embed_nd(ids, bs, static_cast<float>(theta), axes_dim);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_lingbot_video_ids(int t,
                                                                            int h,
                                                                            int w,
                                                                            int pt,
                                                                            int ph,
                                                                            int pw,
                                                                            int bs,
                                                                            int context_len) {
        auto vid_ids_repeated = gen_vid_ids(t, h, w, pt, ph, pw, bs, context_len + 1);

        std::vector<std::vector<float>> txt_ids(bs * context_len, std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < context_len; ++j) {
                txt_ids[i * context_len + j][0] = static_cast<float>(j + 1);
            }
        }

        return concat_ids(vid_ids_repeated, txt_ids, bs);
    }

    __STATIC_INLINE__ std::vector<float> gen_lingbot_video_pe(int t,
                                                              int h,
                                                              int w,
                                                              int pt,
                                                              int ph,
                                                              int pw,
                                                              int bs,
                                                              int context_len,
                                                              int theta,
                                                              const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_lingbot_video_ids(t, h, w, pt, ph, pw, bs, context_len);
        return embed_nd(ids, bs, static_cast<float>(theta), axes_dim);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_qwen2vl_ids(int grid_h,
                                                                      int grid_w,
                                                                      int merge_size,
                                                                      const std::vector<int>& window_index) {
        std::vector<std::vector<float>> ids(grid_h * grid_w, std::vector<float>(2, 0.0));
        int index = 0;
        for (int ih = 0; ih < grid_h; ih += merge_size) {
            for (int iw = 0; iw < grid_w; iw += merge_size) {
                for (int iy = 0; iy < merge_size; iy++) {
                    for (int ix = 0; ix < merge_size; ix++) {
                        int inverse_index = window_index[index / (merge_size * merge_size)];
                        int i             = inverse_index * (merge_size * merge_size) + index % (merge_size * merge_size);

                        GGML_ASSERT(i < grid_h * grid_w);

                        ids[i][0] = static_cast<float>(ih + iy);
                        ids[i][1] = static_cast<float>(iw + ix);
                        index++;
                    }
                }
            }
        }
        return ids;
    }

    // Generate qwen2vl positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_qwen2vl_pe(int grid_h,
                                                        int grid_w,
                                                        int merge_size,
                                                        const std::vector<int>& window_index,
                                                        int theta,
                                                        const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_qwen2vl_ids(grid_h, grid_w, merge_size, window_index);
        return embed_nd(ids, 1, static_cast<float>(theta), axes_dim);
    }

    __STATIC_INLINE__ int bound_mod(int a, int m) {
        return (m - (a % m)) % m;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_z_image_ids(int h,
                                                                      int w,
                                                                      int patch_size,
                                                                      int bs,
                                                                      int context_len,
                                                                      int seq_multi_of,
                                                                      const std::vector<ggml_tensor*>& ref_latents,
                                                                      RefIndexMode ref_index_mode,
                                                                      PositionLayout* layout = nullptr) {
        SD_UNUSED(ref_index_mode);
        int padded_context_len = context_len + bound_mod(context_len, seq_multi_of);
        auto txt_ids           = std::vector<std::vector<float>>(bs * padded_context_len, std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs * padded_context_len; i++) {
            txt_ids[i][0] = (i % padded_context_len) + 1.f;
        }

        if (layout) {
            layout->append_tokens(padded_context_len);
        }
        int axes_dim_num = 3;
        int index        = padded_context_len + 1;
        auto img_ids     = gen_flux_img_ids(h, w, patch_size, bs, axes_dim_num, index, 0, 0, false, layout);

        int img_pad_len = bound_mod(static_cast<int>(img_ids.size() / bs), seq_multi_of);
        if (layout) {
            layout->append_tokens(img_pad_len);
        }
        if (img_pad_len > 0) {
            std::vector<std::vector<float>> img_pad_ids(bs * img_pad_len, std::vector<float>(3, 0.f));
            img_ids = concat_ids(img_ids, img_pad_ids, bs);
        }

        auto ids = concat_ids(txt_ids, img_ids, bs);

        // ignore ref_latents for now
        return ids;
    }

    // LLaDA-Image shares Lumina2/z_image's axes layout, but assigns position (0,0,0) to the
    // padding slots of the caption stream instead of continuing the caption ramp through them.
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_llada_image_ids(int h,
                                                                          int w,
                                                                          int patch_size,
                                                                          int bs,
                                                                          int context_len,
                                                                          int seq_multi_of,
                                                                          PositionLayout* layout = nullptr) {
        int context_pad_len    = bound_mod(context_len, seq_multi_of);
        int padded_context_len = context_len + context_pad_len;
        auto txt_ids           = std::vector<std::vector<float>>(bs * padded_context_len, std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs * padded_context_len; i++) {
            int pos = i % padded_context_len;
            if (pos < context_len) {
                txt_ids[i][0] = pos + 1.f;
            }
        }

        if (layout) {
            layout->append_tokens(padded_context_len);
        }
        int axes_dim_num = 3;
        int index        = padded_context_len + 1;
        auto img_ids     = gen_flux_img_ids(h, w, patch_size, bs, axes_dim_num, index, 0, 0, false, layout);

        int img_pad_len = bound_mod(static_cast<int>(img_ids.size() / bs), seq_multi_of);
        if (layout) {
            layout->append_tokens(img_pad_len);
        }
        if (img_pad_len > 0) {
            std::vector<std::vector<float>> img_pad_ids(bs * img_pad_len, std::vector<float>(3, 0.f));
            img_ids = concat_ids(img_ids, img_pad_ids, bs);
        }

        return concat_ids(txt_ids, img_ids, bs);
    }

    // LLaDA-Image editing packs two caption copies (clean and noisy), the source and target
    // latents anchored at their own caption's end position, and the SigVQ stream after both.
    // Padding slots keep position (0,0,0), as in the text-only layout.
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_llada_image_edit_ids(int h,
                                                                               int w,
                                                                               int patch_size,
                                                                               int context_len,
                                                                               int sigvq_len,
                                                                               int seq_multi_of,
                                                                               PositionLayout* layout = nullptr) {
        const int context_pad    = bound_mod(context_len, seq_multi_of);
        const int padded_context = context_len + context_pad;
        const int h_len          = (h + (patch_size / 2)) / patch_size;
        const int w_len          = (w + (patch_size / 2)) / patch_size;
        const int image_len      = h_len * w_len;
        const int image_pad      = bound_mod(image_len, seq_multi_of);
        const int padded_image   = image_len + image_pad;
        const int sigvq_pad      = bound_mod(sigvq_len, seq_multi_of);

        std::vector<std::vector<float>> cap_ids;
        std::vector<int> cap_end_positions;
        int cursor = 1;
        for (int copy = 0; copy < 2; ++copy) {
            for (int i = 0; i < padded_context; ++i) {
                std::vector<float> id(3, 0.f);
                if (i < context_len) {
                    id[0] = static_cast<float>(cursor + i);
                }
                cap_ids.push_back(id);
            }
            cursor += context_len;
            cap_end_positions.push_back(cursor);
            cursor += 2;
        }

        if (layout) {
            layout->append_tokens(cap_ids.size());
        }
        std::vector<std::vector<float>> img_ids;
        for (int copy = 0; copy < 2; ++copy) {
            auto ids = gen_flux_img_ids(h, w, patch_size, 1, 3, cap_end_positions[copy], 0, 0, false, layout);
            img_ids.insert(img_ids.end(), ids.begin(), ids.end());
            img_ids.insert(img_ids.end(), image_pad, std::vector<float>(3, 0.f));
            if (layout) {
                layout->append_tokens(image_pad);
            }
        }

        const int sigvq_start = static_cast<int>(cap_ids.size() + img_ids.size()) + 1;
        std::vector<std::vector<float>> sigvq_ids;
        for (int i = 0; i < sigvq_len + sigvq_pad; ++i) {
            std::vector<float> id(3, 0.f);
            if (i < sigvq_len) {
                id[0] = static_cast<float>(sigvq_start + i);
            }
            sigvq_ids.push_back(id);
        }

        std::vector<std::vector<float>> ids;
        ids.reserve(cap_ids.size() + img_ids.size() + sigvq_ids.size());
        ids.insert(ids.end(), cap_ids.begin(), cap_ids.end());
        ids.insert(ids.end(), img_ids.begin(), img_ids.end());
        ids.insert(ids.end(), sigvq_ids.begin(), sigvq_ids.end());
        if (layout) {
            layout->append_tokens(sigvq_ids.size());
        }
        SD_UNUSED(padded_image);
        return ids;
    }

    __STATIC_INLINE__ Embedding gen_llada_image_edit_pe(int h,
                                                        int w,
                                                        int patch_size,
                                                        int context_len,
                                                        int sigvq_len,
                                                        int seq_multi_of,
                                                        int theta,
                                                        const std::vector<int>& axes_dim) {
        Embedding result;
        result.batch_size = 1;
        result.ids        = gen_llada_image_edit_ids(h, w, patch_size, context_len, sigvq_len, seq_multi_of, &result.positions);
        result.values     = embed_nd(result.ids, 1, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    __STATIC_INLINE__ Embedding gen_llada_image_pe(int h,
                                                   int w,
                                                   int patch_size,
                                                   int bs,
                                                   int context_len,
                                                   int seq_multi_of,
                                                   int theta,
                                                   const std::vector<int>& axes_dim) {
        Embedding result;
        result.batch_size = bs;
        result.ids        = gen_llada_image_ids(h, w, patch_size, bs, context_len, seq_multi_of, &result.positions);
        result.values     = embed_nd(result.ids, bs, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    // Generate z_image positional embeddings
    __STATIC_INLINE__ Embedding gen_z_image_pe(int h,
                                               int w,
                                               int patch_size,
                                               int bs,
                                               int context_len,
                                               int seq_multi_of,
                                               const std::vector<ggml_tensor*>& ref_latents,
                                               RefIndexMode ref_index_mode,
                                               int theta,
                                               const std::vector<int>& axes_dim) {
        Embedding result;
        result.batch_size = bs;
        result.ids        = gen_z_image_ids(h, w, patch_size, bs, context_len, seq_multi_of, ref_latents, ref_index_mode, &result.positions);
        result.values     = embed_nd(result.ids, bs, static_cast<float>(theta), axes_dim, result.layout, &result.frequencies);
        return result;
    }

    __STATIC_INLINE__ ggml_tensor* apply_rope(ggml_context* ctx,
                                              ggml_tensor* x,
                                              ggml_tensor* pe,
                                              bool rope_interleaved = true) {
        // x: [N, L, n_head, d_head]
        // pe: [L, d_head/2, 2, 2], [[cos, -sin], [sin, cos]]
        int64_t d_head = x->ne[0];
        int64_t n_head = x->ne[1];
        int64_t L      = x->ne[2];
        int64_t N      = x->ne[3];
        x              = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, n_head, L, d_head]
        if (rope_interleaved) {
            x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);  // [N * n_head, L, d_head/2, 2]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));       // [2, N * n_head, L, d_head/2]
        } else {
            x = ggml_reshape_4d(ctx, x, d_head / 2, 2, L, n_head * N);       // [N * n_head, L, 2, d_head/2]
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 3, 1));  // [2, N * n_head, L, d_head/2]
        }

        int64_t offset = x->nb[2] * x->ne[2];
        auto x_0       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);  // [N * n_head, L, d_head/2]
        auto x_1       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);  // [N * n_head, L, d_head/2]
        x_0            = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);                    // [N * n_head, L, d_head/2, 1]
        x_1            = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);                    // [N * n_head, L, d_head/2, 1]
        auto temp_x    = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
        x_0            = ggml_repeat(ctx, x_0, temp_x);  // [N * n_head, L, d_head/2, 2]
        x_1            = ggml_repeat(ctx, x_1, temp_x);  // [N * n_head, L, d_head/2, 2]

        pe        = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));  // [2, L, d_head/2, 2]
        offset    = pe->nb[2] * pe->ne[2];
        auto pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);  // [L, d_head/2, 2]
        auto pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);  // [L, d_head/2, 2]

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));  // [N * n_head, L, d_head/2, 2]
        if (!rope_interleaved) {
            x_out = ggml_cont(ctx, ggml_permute(ctx, x_out, 1, 0, 2, 3));  // [N * n_head, L, x, d_head/2]
        }
        x_out = ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);  // [N*n_head, L, d_head]
        return x_out;
    }

    __STATIC_INLINE__ ggml_tensor* attention(GGMLRunnerContext* ctx,
                                             ggml_tensor* q,
                                             ggml_tensor* k,
                                             ggml_tensor* v,
                                             ggml_tensor* pe,
                                             ggml_tensor* mask,
                                             float kv_scale        = 1.0f,
                                             bool rope_interleaved = true) {
        // q,k,v: [N, L, n_head, d_head]
        // pe: [L, d_head/2, 2, 2]
        // return: [N, L, n_head*d_head]
        int64_t n_head = q->ne[1];

        q = apply_rope(ctx->ggml_ctx, q, pe, rope_interleaved);  // [N*n_head, L, d_head]
        k = apply_rope(ctx->ggml_ctx, k, pe, rope_interleaved);  // [N*n_head, L, d_head]

        auto x = ggml_ext_attention_ext(ctx, q, k, v, n_head, mask, true, ctx->flash_attn_enabled, kv_scale);  // [N, L, n_head*d_head]
        return x;
    }
};  // namespace Rope

#endif  // __SD_MODEL_COMMON_ROPE_HPP__
