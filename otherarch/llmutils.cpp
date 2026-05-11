
#include "llmutils.h"

void kcpp_embd_batch::init_kcpp_batch(int32_t n_tokens,
                                      int32_t npast,
                                      bool    use_mrope,
                                      bool    return_all_logits,
                                      bool    mrope_is_image,
                                      int     img_nx,
                                      int     img_ny) {
    const int          n_pos_per_embd = use_mrope ? 4 : 1;
    const llama_seq_id seq_id         = 0;

    if (use_mrope && mrope_is_image) {
        GGML_ASSERT(img_nx > 0 && img_ny > 0);
        GGML_ASSERT(img_nx * img_ny == n_tokens);
    }

    pos.resize(n_tokens * n_pos_per_embd);
    std::fill(pos.begin(), pos.end(), 0);

    n_seq_id.resize(n_tokens);
    seq_ids.resize(n_tokens + 1);
    logits.resize(n_tokens);
    seq_id_0.resize(1);

    seq_id_0[0]       = seq_id;
    seq_ids[n_tokens] = nullptr;

    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_ids.data();
    batch.logits   = logits.data();

    for (int i = 0; i < n_tokens; ++i) {
        n_seq_id[i] = 1;
        seq_ids[i]  = seq_id_0.data();
        logits[i]   = return_all_logits;
    }

    // ---- position encoding ----
    if (!use_mrope) {
        for (int i = 0; i < n_tokens; ++i) {
            pos[i] = npast + i;
        }
    } else if (!mrope_is_image) {
        // 1D M-RoPE (audio / embedding stream)
        for (int i = 0; i < n_tokens; ++i) {
            pos[i + 0 * n_tokens] = npast + i;
            pos[i + 1 * n_tokens] = npast + i;
            pos[i + 2 * n_tokens] = npast + i;
            pos[i + 3 * n_tokens] = 0;
        }
    } else {
        // 2D image M-RoPE
        int idx = 0;
        for (int y = 0; y < img_ny; ++y) {
            for (int x = 0; x < img_nx; ++x) {
                pos[idx + 0 * n_tokens] = npast;
                pos[idx + 1 * n_tokens] = npast + y;
                pos[idx + 2 * n_tokens] = npast + x;
                pos[idx + 3 * n_tokens] = 0;
                ++idx;
            }
        }
    }

    // Always request logits for last token
    logits[n_tokens - 1] = true;
}

//for embeddings
kcpp_embd_batch::kcpp_embd_batch(float * embd,
                                 int32_t n_tokens,
                                 int32_t npast,
                                 bool    use_mrope,
                                 bool    mrope_is_image,
                                 int     img_nx,
                                 int     img_ny) {
    batch = {
        /* n_tokens = */ n_tokens,
        /* tokens   = */ nullptr,
        /* embd     = */ embd,
        /* pos      = */ nullptr,
        /* n_seq_id = */ nullptr,
        /* seq_id   = */ nullptr,
        /* logits   = */ nullptr,
    };

    init_kcpp_batch(n_tokens, npast, use_mrope,
                    /*return_all_logits=*/false, mrope_is_image, img_nx, img_ny);
}

// for tokens
kcpp_embd_batch::kcpp_embd_batch(std::vector<llama_token> & tokens,
                                 int32_t                    npast,
                                 bool                       use_mrope,
                                 bool                       return_all_logits,
                                 bool                       mrope_is_image,
                                 int                        img_nx,
                                 int                        img_ny) {
    batch = {
        /* n_tokens = */ (int32_t) tokens.size(),
        /* tokens   = */ tokens.data(),
        /* embd     = */ nullptr,
        /* pos      = */ nullptr,
        /* n_seq_id = */ nullptr,
        /* seq_id   = */ nullptr,
        /* logits   = */ nullptr,
    };

    init_kcpp_batch(batch.n_tokens, npast, use_mrope, return_all_logits, mrope_is_image, img_nx, img_ny);
}

llama_batch kcpp_embd_batch::get_view(int offset, int n_tokens, int n_embd_mmproj) {
    GGML_ASSERT(offset >= 0);
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(offset + n_tokens <= batch.n_tokens);

    const int total_tokens = batch.n_tokens;
    llama_pos * pos_ptr = nullptr;

    // Detect M-RoPE vs normal RoPE
    const bool is_mrope = (pos.size() > (size_t)total_tokens);

    pos_view.clear();

    if (is_mrope) {
        const int n_pos_per_embd = pos.size() / total_tokens;
        GGML_ASSERT(n_pos_per_embd == 4);

        // Layout:
        // src: [dim0_all_tokens][dim1_all_tokens][dim2_all_tokens][dim3_all_tokens]
        // dst: same layout, but only [offset : offset + n_tokens]
        pos_view.reserve(n_tokens * n_pos_per_embd);

        for (int dim = 0; dim < n_pos_per_embd; ++dim) {
            const llama_pos * src =
                pos.data() + dim * total_tokens + offset;

            pos_view.insert(
                pos_view.end(),
                src,
                src + n_tokens
            );
        }

        pos_ptr = pos_view.data();
    }
    else {
        // Normal RoPE: contiguous slice
        pos_ptr = pos.data() + offset;
    }

    return {
        /* n_tokens = */ n_tokens,
        /* tokens   = */ nullptr,
        /* embd     = */ batch.embd ? batch.embd + offset*n_embd_mmproj : nullptr,
        /* pos      = */ pos_ptr,
        /* n_seq_id = */ batch.n_seq_id + offset,
        /* seq_id   = */ batch.seq_id   + offset,
        /* logits   = */ batch.logits   + offset,
    };
}