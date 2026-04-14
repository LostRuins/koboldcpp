#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3_tts {

// Speaker encoder configuration (ECAPA-TDNN)
// Mel parameters MUST match extract_speaker_embedding() in modeling_qwen3_tts.py
struct speaker_encoder_config {
    int32_t sample_rate = 24000;
    int32_t n_mels = 128;
    int32_t n_fft = 1024;
    int32_t hop_length = 256;
    int32_t win_length = 1024;
    int32_t embedding_dim = 1024;
    int32_t hidden_dim = 512;
    int32_t n_res2net_blocks = 3;
    int32_t res2net_scale = 8;
    float f_min = 0.0f;
    float f_max = 12000.0f;
};

// Res2Net block weights
struct res2net_block {
    // TDNN1: 1x1 conv (512 -> 512)
    struct ggml_tensor * tdnn1_w = nullptr;
    struct ggml_tensor * tdnn1_b = nullptr;
    
    // Res2Net branches: 7 conv layers (kernel=3, 64 -> 64)
    struct ggml_tensor * res2net_w[7] = {nullptr};
    struct ggml_tensor * res2net_b[7] = {nullptr};
    
    // TDNN2: 1x1 conv (512 -> 512)
    struct ggml_tensor * tdnn2_w = nullptr;
    struct ggml_tensor * tdnn2_b = nullptr;
    
    // SE (Squeeze-Excitation)
    struct ggml_tensor * se_conv1_w = nullptr;
    struct ggml_tensor * se_conv1_b = nullptr;
    struct ggml_tensor * se_conv2_w = nullptr;
    struct ggml_tensor * se_conv2_b = nullptr;
};

// Speaker encoder model weights
struct speaker_encoder_model {
    speaker_encoder_config config;
    
    // Initial conv: (5, 128, 512) - kernel 5, in 128 (mel), out 512
    struct ggml_tensor * conv0_w = nullptr;
    struct ggml_tensor * conv0_b = nullptr;
    
    // Res2Net blocks (3 blocks)
    res2net_block blocks[3];
    
    // MFA (Multi-Frame Aggregation): 1x1 conv (1536 -> 1536)
    struct ggml_tensor * mfa_w = nullptr;
    struct ggml_tensor * mfa_b = nullptr;
    
    // ASP (Attentive Statistics Pooling)
    struct ggml_tensor * asp_conv_w = nullptr;
    struct ggml_tensor * asp_conv_b = nullptr;
    struct ggml_tensor * asp_tdnn_w = nullptr;
    struct ggml_tensor * asp_tdnn_b = nullptr;
    
    // Final FC: (3072 -> 1024)
    struct ggml_tensor * fc_w = nullptr;
    struct ggml_tensor * fc_b = nullptr;
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Compute state for speaker encoder
struct speaker_encoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

// Speaker encoder class (ECAPA-TDNN)
// Extracts speaker embedding from audio waveform
class AudioTokenizerEncoder {
public:
    AudioTokenizerEncoder();
    ~AudioTokenizerEncoder();
    
    // Load model from GGUF file (main TTS model, not tokenizer)
    bool load_model(const std::string & model_path);
    
    // Encode audio samples to speaker embedding
    // samples: audio samples normalized to [-1, 1], 24kHz
    // n_samples: number of samples
    // embedding: output speaker embedding [1024]
    bool encode(const float * samples, int32_t n_samples,
                std::vector<float> & embedding);
    
    // Legacy interface for compatibility (not used for speaker encoding)
    bool encode(const float * samples, int32_t n_samples,
                std::vector<int32_t> & codes, int32_t & n_frames) {
        (void)samples; (void)n_samples; (void)codes; (void)n_frames;
        error_msg_ = "Use encode(samples, n_samples, embedding) for speaker encoding";
        return false;
    }
    
    // Legacy interface (not used)
    bool get_embeddings(const int32_t * codes, int32_t n_frames,
                        std::vector<float> & embeddings) {
        (void)codes; (void)n_frames; (void)embeddings;
        error_msg_ = "Use encode() for speaker embedding extraction";
        return false;
    }
    
    const speaker_encoder_config & get_config() const { return model_.config; }
    
    const std::string & get_error() const { return error_msg_; }
    
private:
    // Compute mel spectrogram from waveform
    bool compute_mel_spectrogram(const float * samples, int32_t n_samples,
                                  std::vector<float> & mel, int32_t & n_frames);
    
    // Build computation graph
    struct ggml_cgraph * build_graph(int32_t n_frames);
    
    speaker_encoder_model model_;
    speaker_encoder_state state_;
    std::string error_msg_;
};

// Free model resources
void free_speaker_encoder_model(speaker_encoder_model & model);

// Backward compatibility alias
using audio_encoder_config = speaker_encoder_config;
using audio_encoder_model = speaker_encoder_model;
inline void free_audio_encoder_model(audio_encoder_model & model) {
    free_speaker_encoder_model(model);
}

} // namespace qwen3_tts
