// Various helper functions and utilities

#pragma once

#include <string>
#include <map>
#include <vector>
#include <random>
#include <thread>
#include "ggml_v3.h"
#include "llama.h"

//
// CLI argument parsing
//


//
// Vocab utils
//

struct gpt_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    void add_special_token(const std::string & token);
};

void utreplace(std::string & str, const std::string & needle, const std::string & replacement);

// poor-man's JSON parsing
std::map<std::string, int32_t> json_parse(const std::string & fname);

std::string convert_to_utf8(const std::wstring & input);

std::wstring convert_to_wstring(const std::string & input);

void gpt_split_words(std::string str, std::vector<std::string>& words);

// split text into tokens
//
// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//
std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text);

bool should_transpose_layer(std::string name);
void kcpp_graph_compute_helper(ggml_v3_cgraph * graph, int n_threads);

std::vector<uint8_t> kcpp_base64_decode(const std::string & encoded_string);
std::string kcpp_base64_encode(const unsigned char* data, unsigned int data_length);
std::string kcpp_base64_encode(const std::string &data);

std::string get_timestamp_str();
std::vector<float> resample_wav(const std::vector<float>& input, uint32_t input_rate, uint32_t output_rate);

int32_t kcpp_quick_sample(float * logits, const int n_logits, const std::vector<int32_t> & last_n_tokens, float rep_pen, float top_p, int top_k, float temp, std::mt19937 & rng);

std::vector<std::string> split_string(const std::string& input, const std::string& separator);

struct kcpp_embd_batch { //duplcated from llava_embd_batch
    std::vector<int32_t> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<int32_t> seq_id_0;
    std::vector<int32_t *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;
    kcpp_embd_batch(float * embd, int32_t n_tokens, int32_t npast, bool use_mrope);
    kcpp_embd_batch(std::vector<llama_token> & tokens, int32_t npast, bool use_mrope, bool return_all_logits);
};