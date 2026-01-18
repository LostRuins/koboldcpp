#ifndef tts_h
#define tts_h

#include "parler_model.h"
#include "kokoro_model.h"
#include "dia_model.h"
#include "orpheus_model.h"
#include <thread>
#include <fstream>
#include <array>

struct tts_runner * parler_tts_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only);
struct tts_runner * kokoro_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only);
struct tts_runner * dia_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only);
struct tts_runner * orpheus_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only);
struct tts_runner * runner_from_file(const std::string & fname, int n_threads, generation_configuration * config, bool cpu_only = true);
int generate(tts_runner * runner, std::string sentence, struct tts_response * response, generation_configuration * config);
void update_conditional_prompt(tts_runner * runner, const std::string file_path, const std::string prompt, bool cpu_only = true);
std::vector<std::string> list_voices(tts_runner * runner);

struct quantization_params {
    quantization_params(uint32_t n_threads, enum ggml_type quantize_type): n_threads(n_threads), quantize_type(quantize_type) {};
    uint32_t n_threads;
    enum ggml_type quantize_type; // quantization type
    bool quantize_output_heads = false;
    bool quantize_text_embeddings = false;
    bool quantize_cross_attn_kv = false;
    bool convert_dac_to_f16 = false;
    bool convert_non_quantizable_to_f16 = false;
};

void quantize_gguf(const std::string & ifile, const std::string & ofile, struct quantization_params * params);

#endif
