#ifndef __ROPE_HPP__
#define __ROPE_HPP__

#include <vector>
#include "ggml_extend.hpp"

struct Rope {
    template <class T>
    static std::vector<T> linspace(T start, T end, int num) {
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

    static std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
        int rows = mat.size();
        int cols = mat[0].size();
        std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    static std::vector<float> flatten(const std::vector<std::vector<float>>& vec) {
        std::vector<float> flat_vec;
        for (const auto& sub_vec : vec) {
            flat_vec.insert(flat_vec.end(), sub_vec.begin(), sub_vec.end());
        }
        return flat_vec;
    }

    static std::vector<std::vector<float>> rope(const std::vector<float>& pos, int dim, int theta) {
        assert(dim % 2 == 0);
        int half_dim = dim / 2;

        std::vector<float> scale = linspace(0.f, (dim * 1.f - 2) / dim, half_dim);

        std::vector<float> omega(half_dim);
        for (int i = 0; i < half_dim; ++i) {
            omega[i] = 1.0 / std::pow(theta, scale[i]);
        }

        int pos_size = pos.size();
        std::vector<std::vector<float>> out(pos_size, std::vector<float>(half_dim));
        for (int i = 0; i < pos_size; ++i) {
            for (int j = 0; j < half_dim; ++j) {
                out[i][j] = pos[i] * omega[j];
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
    static std::vector<std::vector<float>> gen_txt_ids(int bs, int context_len) {
        return std::vector<std::vector<float>>(bs * context_len, std::vector<float>(3, 0.0));
    }

    static std::vector<std::vector<float>> gen_img_ids(int h, int w, int patch_size, int bs, int index = 0, int h_offset = 0, int w_offset = 0) {
        int h_len = (h + (patch_size / 2)) / patch_size;
        int w_len = (w + (patch_size / 2)) / patch_size;

        std::vector<std::vector<float>> img_ids(h_len * w_len, std::vector<float>(3, 0.0));

        std::vector<float> row_ids = linspace<float>(h_offset, h_len - 1 + h_offset, h_len);
        std::vector<float> col_ids = linspace<float>(w_offset, w_len - 1 + w_offset, w_len);

        for (int i = 0; i < h_len; ++i) {
            for (int j = 0; j < w_len; ++j) {
                img_ids[i * w_len + j][0] = index;
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

    static std::vector<std::vector<float>> concat_ids(const std::vector<std::vector<float>>& a,
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

    static std::vector<float> embed_nd(const std::vector<std::vector<float>>& ids,
                                       int bs,
                                       int theta,
                                       const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> trans_ids = transpose(ids);
        size_t pos_len                            = ids.size() / bs;
        int num_axes                              = axes_dim.size();
        // for (int i = 0; i < pos_len; i++) {
        //     std::cout << trans_ids[0][i] << " " << trans_ids[1][i] << " " << trans_ids[2][i] << std::endl;
        // }

        int emb_dim = 0;
        for (int d : axes_dim)
            emb_dim += d / 2;

        std::vector<std::vector<float>> emb(bs * pos_len, std::vector<float>(emb_dim * 2 * 2, 0.0));
        int offset = 0;
        for (int i = 0; i < num_axes; ++i) {
            std::vector<std::vector<float>> rope_emb = rope(trans_ids[i], axes_dim[i], theta);  // [bs*pos_len, axes_dim[i]/2 * 2 * 2]
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

    static std::vector<std::vector<float>> gen_flux_ids(int h,
                                                        int w,
                                                        int patch_size,
                                                        int bs,
                                                        int context_len,
                                                        std::vector<ggml_tensor*> ref_latents,
                                                        bool increase_ref_index) {
        auto txt_ids = gen_txt_ids(bs, context_len);
        auto img_ids = gen_img_ids(h, w, patch_size, bs);

        auto ids               = concat_ids(txt_ids, img_ids, bs);
        uint64_t curr_h_offset = 0;
        uint64_t curr_w_offset = 0;
        int index              = 1;
        for (ggml_tensor* ref : ref_latents) {
            uint64_t h_offset = 0;
            uint64_t w_offset = 0;
            if (!increase_ref_index) {
                if (ref->ne[1] + curr_h_offset > ref->ne[0] + curr_w_offset) {
                    w_offset = curr_w_offset;
                } else {
                    h_offset = curr_h_offset;
                }
            }

            auto ref_ids = gen_img_ids(ref->ne[1], ref->ne[0], patch_size, bs, index, h_offset, w_offset);
            ids          = concat_ids(ids, ref_ids, bs);

            if (increase_ref_index) {
                index++;
            }

            curr_h_offset = std::max(curr_h_offset, ref->ne[1] + h_offset);
            curr_w_offset = std::max(curr_w_offset, ref->ne[0] + w_offset);
        }
        return ids;
    }

    // Generate flux positional embeddings
    static std::vector<float> gen_flux_pe(int h,
                                          int w,
                                          int patch_size,
                                          int bs,
                                          int context_len,
                                          std::vector<ggml_tensor*> ref_latents,
                                          bool increase_ref_index,
                                          int theta,
                                          const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_flux_ids(h, w, patch_size, bs, context_len, ref_latents, increase_ref_index);
        return embed_nd(ids, bs, theta, axes_dim);
    }

    static std::vector<std::vector<float>> gen_vid_ids(int t,
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

        std::vector<float> t_ids = linspace<float>(t_offset, t_len - 1 + t_offset, t_len);
        std::vector<float> h_ids = linspace<float>(h_offset, h_len - 1 + h_offset, h_len);
        std::vector<float> w_ids = linspace<float>(w_offset, w_len - 1 + w_offset, w_len);

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
    static std::vector<float> gen_wan_pe(int t,
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
};  // struct Rope

#endif  // __ROPE_HPP__
