#pragma once

#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <iostream>
#include <iterator>
#include <queue>
#include <string>
#include <math.h>
#include <vector>

#include "expose.h"
#include "src/llama-arch.h"

enum FileFormat
{
    BADFORMAT=0, //unknown, uninit, or failed to load
    GGML=1, // 1=(original llama ggml, alpaca, GPT4ALL, GPTJ header)
    GGHF=2, // 2=(llama ggmf)
    GGJT=3, // 3=(llama ggjt)
    GGJT_2=4, //newer llama format unshuffled
    GGJT_3=5, //using 16bit scalar

    GGUF_GENERIC=6, //GGUF (llama newest ver)

    GPTJ_1=100, //the very first super old GPTJ format
    GPTJ_2=101, //pygmalion, uses old ggml lib
    GPTJ_3=102, //uses new ggml lib
    GPTJ_4=103, //unshuffled
    GPTJ_5=104, //using 16bit scalar

    GPT2_1=200,
    GPT2_2=201,
    GPT2_3=202, //unshuffled
    GPT2_4=203, //using 16bit scalar

    RWKV_1=300,
    RWKV_2=301,

    NEOX_1=400,
    NEOX_2=401,
    NEOX_3=402, //redpajama
    NEOX_4=403, //unshuffled
    NEOX_5=404, //unshuffled redpajama
    NEOX_6=405, //using 16bit scalar
    NEOX_7=406, //using 16bit scalar redpajama

    MPT_1=500, //first supported mpt version

};

struct FileFormatExtraMeta
{
    int n_ctx_train = 2048;
    int fileversion = 0;
    llm_arch model_architecture = llm_arch::LLM_ARCH_UNKNOWN;
    int n_expert_count = 0;
    std::string model_architecture_str = "";
    bool explicitly_no_bos = false; //only true if key exists AND is false
};

struct TopPicksData
{
    std::string selected_token;
    int32_t selected_tokenid;
    float selected_logprob;
    float selected_probability;
    std::vector<std::string> tokens;
    std::vector<int> tokenid;
    std::vector<float> logprobs;
    std::vector<float> p;
};

enum ModelLoadResult
{
    FAIL = 0,
    SUCCESS = 1,
    RETRY_LOAD = 2, //used if it's suspected that the model is an older format
};

ModelLoadResult gpttype_load_model(const load_model_inputs inputs, FileFormat in_file_format, FileFormatExtraMeta file_format_meta);
generation_outputs gpttype_generate(const generation_inputs inputs);
bool gpttype_generate_abort();
std::string gpttype_get_chat_template();
bool gpttype_batch_generate_enabled();
int gpttype_batch_generate_submit(const generation_inputs inputs);
bool gpttype_batch_generate_has_finished(int request_id);
int gpttype_batch_generate_stream_count(int request_id);
const char * gpttype_batch_generate_new_token(int request_id, int idx);
const char * gpttype_batch_generate_pending_output(int request_id);
generation_outputs gpttype_batch_generate_result(int request_id);
bool gpttype_batch_generate_abort(int request_id);
void gpttype_batch_generate_release(int request_id);

const std::string & gpttype_get_pending_output();
std::vector<int> gpttype_get_token_arr(const std::string & input, bool addbos);
std::string gpttype_detokenize(const std::vector<int> & input, bool render_special);
const std::vector<TopPicksData> gpttype_get_top_picks_data();

bool sdtype_load_model(const sd_load_model_inputs inputs);
sd_generation_outputs sdtype_generate(const sd_generation_inputs inputs);
sd_generation_outputs sdtype_upscale(const sd_upscale_inputs inputs);
sd_info_outputs sdtype_get_info();

bool whispertype_load_model(const whisper_load_model_inputs inputs);
whisper_generation_outputs whispertype_generate(const whisper_generation_inputs inputs);

bool ttstype_load_model(const tts_load_model_inputs inputs);
tts_generation_outputs ttstype_generate(const tts_generation_inputs inputs);

bool embeddingstype_load_model(const embeddings_load_model_inputs inputs);
embeddings_generation_outputs embeddingstype_generate(const embeddings_generation_inputs inputs);

bool musictype_load_model(const music_load_model_inputs inputs);
music_generation_outputs musictype_generate(const music_generation_inputs inputs);

void timer_start();
double timer_check();
void print_tok_vec(std::vector<int> &embd);
void print_tok_vec(std::vector<float> &embd);
void print_vec(std::vector<std::string> &embd);
std::vector<int> LongestCommonSubseq(const std::vector<int> x, const std::vector<int> y);
bool ArrStartWith(const std::vector<int> targetArray, const std::vector<int> searchSeq);
int ArrFindIndexOf(const std::vector<int> targetArray, const std::vector<int> searchSeq);

FileFormat check_file_format(const std::string & fname, FileFormatExtraMeta * fileformatmeta);
void ContextFastForward(std::vector<int> &current_context_tokens, std::vector<int> &embd_inp, int &n_past, std::vector<int> &last_n_tokens, const int nctx, std::vector<int> &smartcontext, const bool useSmartContext, const bool requireFullSubset, const int minimum_to_proceed, const int minimum_input_to_keep);
bool gguf_tensor_exists(const std::string & filename, std::string tensor_name, bool exactmatch);
std::string gguf_get_model_arch(const std::string & filename);

size_t gpttype_calc_new_state_kv();
size_t gpttype_calc_new_state_tokencount();
size_t gpttype_calc_old_state_kv(int slot);
size_t gpttype_calc_old_state_tokencount(int slot);
size_t gpttype_save_state_kv(int slot);
bool gpttype_load_state_kv(int slot);
bool gpttype_clear_state_kv(bool shrink);
int get_oldest_slot(int excludeSlotId);
void touch_slot(int slot);
int get_identical_existing_slot();
