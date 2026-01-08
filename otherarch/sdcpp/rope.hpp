#ifndef __ROPE_HPP__
#define __ROPE_HPP__

#include <algorithm>
#include <cmath>
#include <vector>
#include "ggml_extend.hpp"

namespace Rope {
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

    __STATIC_INLINE__ std::vector<std::vector<float>> rope(const std::vector<float>& pos,
                                                           int dim,
                                                           int theta,
                                                           const std::vector<int>& axis_wrap_dims = {}) {
        assert(dim % 2 == 0);
        int half_dim = dim / 2;

        std::vector<float> scale = linspace(0.f, (dim * 1.f - 2) / dim, half_dim);

        std::vector<float> omega(half_dim);
        for (int i = 0; i < half_dim; ++i) {
            omega[i] = 1.0f / ::powf(1.f * theta, scale[i]);
        }

        size_t pos_size = pos.size();
        std::vector<std::vector<float>> out(pos_size, std::vector<float>(half_dim));
        for (size_t i = 0; i < pos_size; ++i) {
            for (size_t j = 0; j < half_dim; ++j) {
                float angle = pos[i] * omega[j];
                if (!axis_wrap_dims.empty()) {
                    size_t wrap_size = axis_wrap_dims.size();
                    // mod batch size since we only store this for one item in the batch
                    size_t wrap_idx = wrap_size > 0 ? (i % wrap_size) : 0;
                    int wrap_dim    = axis_wrap_dims[wrap_idx];
                    if (wrap_dim > 0) {
                        constexpr float TWO_PI = 6.28318530717958647692f;
                        float cycles           = omega[j] * wrap_dim / TWO_PI;
                        // closest periodic harmonic, necessary to ensure things neatly tile
                        // without this round, things don't tile at the boundaries and you end up
                        // with the model knowing what is "center"
                        float rounded = std::round(cycles);
                        angle         = pos[i] * TWO_PI * rounded / wrap_dim;
                    }
                }

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

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_flux_img_ids(int h,
                                                                       int w,
                                                                       int patch_size,
                                                                       int bs,
                                                                       int axes_dim_num,
                                                                       int index       = 0,
                                                                       int h_offset    = 0,
                                                                       int w_offset    = 0,
                                                                       bool scale_rope = false) {
        int h_len = (h + (patch_size / 2)) / patch_size;
        int w_len = (w + (patch_size / 2)) / patch_size;

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
                                                  int theta,
                                                  const std::vector<int>& axes_dim,
                                                  const std::vector<std::vector<int>>& wrap_dims = {}) {
        std::vector<std::vector<float>> trans_ids = transpose(ids);
        size_t pos_len                            = ids.size() / bs;
        size_t num_axes                           = axes_dim.size();
        // for (int i = 0; i < pos_len; i++) {
        //     std::cout << trans_ids[0][i] << " " << trans_ids[1][i] << " " << trans_ids[2][i] << std::endl;
        // }

        int emb_dim = 0;
        for (int d : axes_dim)
            emb_dim += d / 2;

        std::vector<std::vector<float>> emb(bs * pos_len, std::vector<float>(emb_dim * 2 * 2, 0.0));
        size_t offset = 0;
        for (size_t i = 0; i < num_axes; ++i) {
            std::vector<int> axis_wrap_dims;
            if (!wrap_dims.empty() && i < (int)wrap_dims.size()) {
                axis_wrap_dims = wrap_dims[i];
            }
            std::vector<std::vector<float>> rope_emb =
                rope(trans_ids[i], axes_dim[i], theta, axis_wrap_dims);  // [bs*pos_len, axes_dim[i]/2 * 2 * 2]
            for (int b = 0; b < bs; ++b) {
                for (int j = 0; j < pos_len; ++j) {
                    for (int k = 0; k < rope_emb[0].size(); ++k) {
                        emb[b * pos_len + j][offset + k] = rope_emb[j][k];
                    }
                }
            }
            offset += rope_emb[0].size();
        }

        return flatten(emb);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_refs_ids(int patch_size,
                                                                   int bs,
                                                                   int axes_dim_num,
                                                                   const std::vector<ggml_tensor*>& ref_latents,
                                                                   bool increase_ref_index,
                                                                   float ref_index_scale,
                                                                   bool scale_rope) {
        std::vector<std::vector<float>> ids;
        int curr_h_offset = 0;
        int curr_w_offset = 0;
        int index         = 1;
        for (ggml_tensor* ref : ref_latents) {
            int h_offset = 0;
            int w_offset = 0;
            if (!increase_ref_index) {
                if (ref->ne[1] + curr_h_offset > ref->ne[0] + curr_w_offset) {
                    w_offset = curr_w_offset;
                } else {
                    h_offset = curr_h_offset;
                }
                scale_rope = false;
            }

            auto ref_ids = gen_flux_img_ids(static_cast<int>(ref->ne[1]),
                                            static_cast<int>(ref->ne[0]),
                                            patch_size,
                                            bs,
                                            axes_dim_num,
                                            static_cast<int>(index * ref_index_scale),
                                            h_offset,
                                            w_offset,
                                            scale_rope);
            ids          = concat_ids(ids, ref_ids, bs);

            if (increase_ref_index) {
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
                                                                   bool increase_ref_index,
                                                                   float ref_index_scale) {
        auto txt_ids = gen_flux_txt_ids(bs, context_len, axes_dim_num, txt_arange_dims);
        auto img_ids = gen_flux_img_ids(h, w, patch_size, bs, axes_dim_num);

        auto ids = concat_ids(txt_ids, img_ids, bs);
        if (ref_latents.size() > 0) {
            auto refs_ids = gen_refs_ids(patch_size, bs, axes_dim_num, ref_latents, increase_ref_index, ref_index_scale, false);
            ids           = concat_ids(ids, refs_ids, bs);
        }
        return ids;
    }

    // Generate flux positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_flux_pe(int h,
                                                     int w,
                                                     int patch_size,
                                                     int bs,
                                                     int context_len,
                                                     std::set<int> txt_arange_dims,
                                                     const std::vector<ggml_tensor*>& ref_latents,
                                                     bool increase_ref_index,
                                                     float ref_index_scale,
                                                     int theta,
                                                     bool circular_h,
                                                     bool circular_w,
                                                     const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_flux_ids(h,
                                                           w,
                                                           patch_size,
                                                           bs,
                                                           static_cast<int>(axes_dim.size()),
                                                           context_len,
                                                           txt_arange_dims,
                                                           ref_latents,
                                                           increase_ref_index,
                                                           ref_index_scale);
        std::vector<std::vector<int>> wrap_dims;
        if ((circular_h || circular_w) && bs > 0 && axes_dim.size() >= 3) {
            int h_len = (h + (patch_size / 2)) / patch_size;
            int w_len = (w + (patch_size / 2)) / patch_size;
            if (h_len > 0 && w_len > 0) {
                size_t pos_len = ids.size() / bs;
                wrap_dims.assign(axes_dim.size(), std::vector<int>(pos_len, 0));
                size_t cursor           = context_len;  // text first
                const size_t img_tokens = static_cast<size_t>(h_len) * static_cast<size_t>(w_len);
                for (size_t token_i = 0; token_i < img_tokens; ++token_i) {
                    if (circular_h) {
                        wrap_dims[1][cursor + token_i] = h_len;
                    }
                    if (circular_w) {
                        wrap_dims[2][cursor + token_i] = w_len;
                    }
                }
                cursor += img_tokens;
                // reference latents
                for (ggml_tensor* ref : ref_latents) {
                    if (ref == nullptr) {
                        continue;
                    }
                    int ref_h         = static_cast<int>(ref->ne[1]);
                    int ref_w         = static_cast<int>(ref->ne[0]);
                    int ref_h_l       = (ref_h + (patch_size / 2)) / patch_size;
                    int ref_w_l       = (ref_w + (patch_size / 2)) / patch_size;
                    size_t ref_tokens = static_cast<size_t>(ref_h_l) * static_cast<size_t>(ref_w_l);
                    for (size_t token_i = 0; token_i < ref_tokens; ++token_i) {
                        if (circular_h) {
                            wrap_dims[1][cursor + token_i] = ref_h_l;
                        }
                        if (circular_w) {
                            wrap_dims[2][cursor + token_i] = ref_w_l;
                        }
                    }
                    cursor += ref_tokens;
                }
            }
        }
        return embed_nd(ids, bs, theta, axes_dim, wrap_dims);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_qwen_image_ids(int h,
                                                                         int w,
                                                                         int patch_size,
                                                                         int bs,
                                                                         int context_len,
                                                                         const std::vector<ggml_tensor*>& ref_latents,
                                                                         bool increase_ref_index) {
        int h_len        = (h + (patch_size / 2)) / patch_size;
        int w_len        = (w + (patch_size / 2)) / patch_size;
        int txt_id_start = std::max(h_len, w_len);
        auto txt_ids     = linspace<float>(1.f * txt_id_start, 1.f * context_len + txt_id_start, context_len);
        std::vector<std::vector<float>> txt_ids_repeated(bs * context_len, std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < txt_ids.size(); ++j) {
                txt_ids_repeated[i * txt_ids.size() + j] = {txt_ids[j], txt_ids[j], txt_ids[j]};
            }
        }
        int axes_dim_num = 3;
        auto img_ids     = gen_flux_img_ids(h, w, patch_size, bs, axes_dim_num, 0, 0, 0, true);
        auto ids         = concat_ids(txt_ids_repeated, img_ids, bs);
        if (ref_latents.size() > 0) {
            auto refs_ids = gen_refs_ids(patch_size, bs, axes_dim_num, ref_latents, increase_ref_index, 1.f, true);
            ids           = concat_ids(ids, refs_ids, bs);
        }
        return ids;
    }

    // Generate qwen_image positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_qwen_image_pe(int h,
                                                           int w,
                                                           int patch_size,
                                                           int bs,
                                                           int context_len,
                                                           const std::vector<ggml_tensor*>& ref_latents,
                                                           bool increase_ref_index,
                                                           int theta,
                                                           bool circular_h,
                                                           bool circular_w,
                                                           const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_qwen_image_ids(h, w, patch_size, bs, context_len, ref_latents, increase_ref_index);
        std::vector<std::vector<int>> wrap_dims;
        // This logic simply stores the (pad and patch_adjusted) sizes of images so we can make sure rope correctly tiles
        if ((circular_h || circular_w) && bs > 0 && axes_dim.size() >= 3) {
            int pad_h = (patch_size - (h % patch_size)) % patch_size;
            int pad_w = (patch_size - (w % patch_size)) % patch_size;
            int h_len = (h + pad_h) / patch_size;
            int w_len = (w + pad_w) / patch_size;
            if (h_len > 0 && w_len > 0) {
                const size_t total_tokens = ids.size();
                // Track per-token wrap lengths for the row/column axes so only spatial tokens become periodic.
                wrap_dims.assign(axes_dim.size(), std::vector<int>(total_tokens / bs, 0));
                size_t cursor           = context_len;  // ignore text tokens
                const size_t img_tokens = static_cast<size_t>(h_len) * static_cast<size_t>(w_len);
                for (size_t token_i = 0; token_i < img_tokens; ++token_i) {
                    if (circular_h) {
                        wrap_dims[1][cursor + token_i] = h_len;
                    }
                    if (circular_w) {
                        wrap_dims[2][cursor + token_i] = w_len;
                    }
                }
                cursor += img_tokens;
                // For each reference image, store wrap sizes as well
                for (ggml_tensor* ref : ref_latents) {
                    if (ref == nullptr) {
                        continue;
                    }
                    int ref_h           = static_cast<int>(ref->ne[1]);
                    int ref_w           = static_cast<int>(ref->ne[0]);
                    int ref_pad_h       = (patch_size - (ref_h % patch_size)) % patch_size;
                    int ref_pad_w       = (patch_size - (ref_w % patch_size)) % patch_size;
                    int ref_h_len       = (ref_h + ref_pad_h) / patch_size;
                    int ref_w_len       = (ref_w + ref_pad_w) / patch_size;
                    size_t ref_n_tokens = static_cast<size_t>(ref_h_len) * static_cast<size_t>(ref_w_len);
                    for (size_t token_i = 0; token_i < ref_n_tokens; ++token_i) {
                        if (circular_h) {
                            wrap_dims[1][cursor + token_i] = ref_h_len;
                        }
                        if (circular_w) {
                            wrap_dims[2][cursor + token_i] = ref_w_len;
                        }
                    }
                    cursor += ref_n_tokens;
                }
            }
        }
        return embed_nd(ids, bs, theta, axes_dim, wrap_dims);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_vid_ids(int t,
                                                                  int h,
                                                                  int w,
                                                                  int pt,
                                                                  int ph,
                                                                  int pw,
                                                                  int bs,
                                                                  int t_offset = 0,
                                                                  int h_offset = 0,
                                                                  int w_offset = 0) {
        int t_len = (t + (pt / 2)) / pt;
        int h_len = (h + (ph / 2)) / ph;
        int w_len = (w + (pw / 2)) / pw;

        std::vector<std::vector<float>> vid_ids(t_len * h_len * w_len, std::vector<float>(3, 0.0));

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

    // Generate wan positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_wan_pe(int t,
                                                    int h,
                                                    int w,
                                                    int pt,
                                                    int ph,
                                                    int pw,
                                                    int bs,
                                                    int theta,
                                                    const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_vid_ids(t, h, w, pt, ph, pw, bs);
        return embed_nd(ids, bs, theta, axes_dim);
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
        return embed_nd(ids, 1, theta, axes_dim);
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
                                                                      bool increase_ref_index) {
        int padded_context_len = context_len + bound_mod(context_len, seq_multi_of);
        auto txt_ids           = std::vector<std::vector<float>>(bs * padded_context_len, std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs * padded_context_len; i++) {
            txt_ids[i][0] = (i % padded_context_len) + 1.f;
        }

        int axes_dim_num = 3;
        int index        = padded_context_len + 1;
        auto img_ids     = gen_flux_img_ids(h, w, patch_size, bs, axes_dim_num, index);

        int img_pad_len = bound_mod(static_cast<int>(img_ids.size() / bs), seq_multi_of);
        if (img_pad_len > 0) {
            std::vector<std::vector<float>> img_pad_ids(bs * img_pad_len, std::vector<float>(3, 0.f));
            img_ids = concat_ids(img_ids, img_pad_ids, bs);
        }

        auto ids = concat_ids(txt_ids, img_ids, bs);

        // ignore ref_latents for now
        return ids;
    }

    // Generate z_image positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_z_image_pe(int h,
                                                        int w,
                                                        int patch_size,
                                                        int bs,
                                                        int context_len,
                                                        int seq_multi_of,
                                                        const std::vector<ggml_tensor*>& ref_latents,
                                                        bool increase_ref_index,
                                                        int theta,
                                                        bool circular_h,
                                                        bool circular_w,
                                                        const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_z_image_ids(h, w, patch_size, bs, context_len, seq_multi_of, ref_latents, increase_ref_index);
        std::vector<std::vector<int>> wrap_dims;
        if ((circular_h || circular_w) && bs > 0 && axes_dim.size() >= 3) {
            int pad_h = (patch_size - (h % patch_size)) % patch_size;
            int pad_w = (patch_size - (w % patch_size)) % patch_size;
            int h_len = (h + pad_h) / patch_size;
            int w_len = (w + pad_w) / patch_size;
            if (h_len > 0 && w_len > 0) {
                size_t pos_len = ids.size() / bs;
                wrap_dims.assign(axes_dim.size(), std::vector<int>(pos_len, 0));
                size_t cursor     = context_len + bound_mod(context_len, seq_multi_of);  // skip text (and its padding)
                size_t img_tokens = static_cast<size_t>(h_len) * static_cast<size_t>(w_len);
                for (size_t token_i = 0; token_i < img_tokens; ++token_i) {
                    if (circular_h) {
                        wrap_dims[1][cursor + token_i] = h_len;
                    }
                    if (circular_w) {
                        wrap_dims[2][cursor + token_i] = w_len;
                    }
                }
            }
        }

        return embed_nd(ids, bs, theta, axes_dim, wrap_dims);
    }

    __STATIC_INLINE__ struct ggml_tensor* apply_rope(struct ggml_context* ctx,
                                                     struct ggml_tensor* x,
                                                     struct ggml_tensor* pe,
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

    __STATIC_INLINE__ struct ggml_tensor* attention(GGMLRunnerContext* ctx,
                                                    struct ggml_tensor* q,
                                                    struct ggml_tensor* k,
                                                    struct ggml_tensor* v,
                                                    struct ggml_tensor* pe,
                                                    struct ggml_tensor* mask,
                                                    float kv_scale        = 1.0f,
                                                    bool rope_interleaved = true) {
        // q,k,v: [N, L, n_head, d_head]
        // pe: [L, d_head/2, 2, 2]
        // return: [N, L, n_head*d_head]
        q = apply_rope(ctx->ggml_ctx, q, pe, rope_interleaved);  // [N*n_head, L, d_head]
        k = apply_rope(ctx->ggml_ctx, k, pe, rope_interleaved);  // [N*n_head, L, d_head]

        auto x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, v->ne[1], mask, false, true, ctx->flash_attn_enabled, kv_scale);  // [N, L, n_head*d_head]
        return x;
    }
};  // namespace Rope

#endif  // __ROPE_HPP__
