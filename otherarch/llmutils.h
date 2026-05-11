#pragma once

#include <string>
#include <map>
#include <vector>
#include <random>
#include <thread>
#include "llama.h"

//duplcated and modified from llava_embd_batch
struct kcpp_embd_batch {
    std::vector<llama_pos>    pos;
    std::vector<llama_pos>    pos_view;
    std::vector<int32_t>      n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id*> seq_ids;
    std::vector<int8_t>       logits;
    llama_batch batch;

    llama_batch get_view(int offset, int n_tokens, int n_embd_mmproj);

    // Embedding constructor
    kcpp_embd_batch(
        float * embd,
        int32_t n_tokens,
        int32_t npast,
        bool use_mrope,
        bool mrope_is_image = false,
        int img_nx = 0,
        int img_ny = 0
    );

    // Token constructor
    kcpp_embd_batch(
        std::vector<llama_token> & tokens,
        int32_t npast,
        bool use_mrope,
        bool return_all_logits,
        bool mrope_is_image = false,
        int img_nx = 0,
        int img_ny = 0
    );

private:
    void init_kcpp_batch(
        int32_t n_tokens,
        int32_t npast,
        bool use_mrope,
        bool return_all_logits,
        bool mrope_is_image,
        int img_nx,
        int img_ny
    );
};