#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_tokenizer_decoder.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace qwen3_tts {

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate
    int32_t max_audio_tokens = 4096;

    // Temperature for sampling (0 = greedy)
    float temperature = 0.9f;

    // Top-p sampling
    float top_p = 1.0f;

    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;

    // Number of threads
    int32_t n_threads = 4;

    // Print progress during generation
    bool print_progress = false;

    // Print timing information
    bool print_timing = true;

    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;

    // Sample rate
    int32_t sample_rate = 24000;

    // Success flag
    bool success = false;

    // Error message if failed
    std::string error_msg;

    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_generate_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;

};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();

    void set_seed(int seed);

    // Load all models from directory
    // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
    bool load_models(const std::string & model_dir);
    bool load_models(const std::string & model,const std::string & tokenizer);

    // Generate speech from text
    // text: input text to synthesize
    // params: generation parameters
    tts_result synthesize(const std::string & text, const std::string & instruction, const int speaker_id,
                          const tts_params & params = tts_params());

    // Generate speech with voice cloning
    // text: input text to synthesize
    // reference_audio: path to reference audio file (WAV, 24kHz)
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params & params = tts_params());

    // Generate speech with voice cloning from samples
    // text: input text to synthesize
    // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
    // n_ref_samples: number of reference samples
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params & params = tts_params(), std::size_t reuse_hash_val=0);

    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);

    bool load_speaker_enc(); //return false if failed to load encoder

    // Get error message
    const std::string & get_error() const { return error_msg_; }

    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }

private:
    tts_result synthesize_internal(const std::string & text, const std::string & instruction, const int speakerid,
                                   const float * speaker_embedding,
                                   const tts_params & params,
                                   tts_result & result);

    TextTokenizer tokenizer_;
    TTSTransformer transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioTokenizerDecoder audio_decoder_;

    bool models_loaded_ = false;
    bool encoder_loaded_ = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;
    tts_progress_callback_t progress_callback_;
};

// Utility: Load audio file (WAV format)
bool load_audio_file(const std::string & path, std::vector<float> & samples,
                     int & sample_rate);

// Utility: Save audio file (WAV format)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate);

} // namespace qwen3_tts
