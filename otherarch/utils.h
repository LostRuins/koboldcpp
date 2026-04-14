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
std::vector<std::vector<int>> split_big_vector(const std::vector<int>& big_arr, size_t chunk_size);
std::vector<std::vector<int>> split_big_vector_in_two(const std::vector<int>& big_arr, size_t chunk_size);

std::vector<float> resample_wav(int num_channels,const std::vector<float>& input, uint32_t input_rate, uint32_t output_rate);
std::vector<float> mix_planar_stereo_to_mono(const float* audio, int T_audio);

int32_t kcpp_quick_sample(float * logits, const int n_logits, const std::vector<int32_t> & last_n_tokens, float rep_pen, float top_p, int top_k, float temp, std::mt19937 & rng);

std::vector<std::string> split_string(const std::string& input, const std::string& separator);
bool kcpp_decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono);
bool kcpp_decode_audio_to_f32_stereo_48k(const uint8_t * data, size_t data_size, std::vector<float> & pcm, int & T_audio);

std::vector<ggml_backend_dev_t> kcpp_parse_device_list(const std::string & value);

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

#pragma pack(push, 1)
struct wav16_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct wav_ulaw_header {
    char     riff[4]        = {'R','I','F','F'};
    uint32_t chunk_size;              // 36 + data_size + 2 (for cbSize)
    char     wave[4]        = {'W','A','V','E'};
    char     fmt[4]         = {'f','m','t',' '};
    uint32_t fmt_chunk_size = 18;     // 18 for non-PCM
    uint16_t audio_format   = 7;      // 7 = μ-law
    uint16_t num_channels   = 1;      // mono
    uint32_t sample_rate;
    uint32_t byte_rate;               // sample_rate * channels * 1
    uint16_t block_align;             // channels * 1
    uint16_t bits_per_sample = 8;     // 8-bit μ-law
    uint16_t cbSize = 0;              // required for non-PCM
    char     data[4]        = {'d','a','t','a'};
    uint32_t data_size;
};
#pragma pack(pop)

std::string save_ulaw_wav8_base64(const std::vector<float> &data, int sample_rate);
std::string save_wav16_base64(const std::vector<float> &data, int sample_rate);
std::string save_stereo_wav16_base64(const std::vector<float> & raw_audio, int T_audio, int sample_rate);
std::string save_stereo_mp3_base64(const std::vector<float> & raw_audio,int T_audio,int sample_rate);