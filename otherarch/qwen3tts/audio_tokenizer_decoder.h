#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3_tts {

// Audio tokenizer decoder (vocoder) configuration
struct audio_decoder_config {
    int32_t sample_rate = 24000;
    int32_t n_codebooks = 16;           // Total codebooks (1 first + 15 rest)
    int32_t codebook_size = 2048;       // Entries per codebook
    int32_t codebook_dim = 256;         // Embedding dimension per codebook
    int32_t latent_dim = 1024;          // Latent dimension after VQ
    int32_t hidden_dim = 512;           // Pre-transformer hidden dimension
    int32_t n_pre_tfm_layers = 8;       // Pre-transformer layers
    int32_t n_heads = 16;               // Attention heads in pre-transformer
    int32_t ffn_dim = 1024;             // FFN intermediate dimension
    int32_t decoder_dim = 1536;         // Initial decoder dimension
    int32_t upsample_rates[4] = {8, 5, 4, 3};  // Total: 480x upsampling
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

// Pre-transformer layer weights
struct pre_tfm_layer {
    // Attention
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_output_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;  // layer_scale for attention
    
    // FFN (SwiGLU)
    struct ggml_tensor * ffn_norm_w = nullptr;
    struct ggml_tensor * ffn_gate_w = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_scale = nullptr;   // layer_scale for FFN
};

// Residual block weights (Snake + Conv + Snake + Conv)
struct residual_block {
    int dilation = 1;  // Dilation for conv1: [1, 3, 9] for res[0], res[1], res[2]
    struct ggml_tensor * act1_alpha = nullptr;
    struct ggml_tensor * act1_beta = nullptr;
    struct ggml_tensor * conv1_w = nullptr;
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * act2_alpha = nullptr;
    struct ggml_tensor * act2_beta = nullptr;
    struct ggml_tensor * conv2_w = nullptr;
    struct ggml_tensor * conv2_b = nullptr;
};

// Decoder block weights (Snake + ConvTranspose + Residual blocks)
struct decoder_block {
    // Snake activation before conv transpose
    struct ggml_tensor * snake_alpha = nullptr;
    struct ggml_tensor * snake_beta = nullptr;
    
    // Transposed convolution for upsampling
    struct ggml_tensor * conv_t_w = nullptr;
    struct ggml_tensor * conv_t_b = nullptr;
    
    // Residual blocks (3 per decoder block)
    residual_block res[3];
};

// Upsample block weights (ConvNeXt-style)
struct upsample_block {
    struct ggml_tensor * conv_w = nullptr;
    struct ggml_tensor * conv_b = nullptr;
    struct ggml_tensor * dwconv_w = nullptr;
    struct ggml_tensor * dwconv_b = nullptr;
    struct ggml_tensor * norm_w = nullptr;
    struct ggml_tensor * norm_b = nullptr;
    struct ggml_tensor * pwconv1_w = nullptr;
    struct ggml_tensor * pwconv1_b = nullptr;
    struct ggml_tensor * pwconv2_w = nullptr;
    struct ggml_tensor * pwconv2_b = nullptr;
    struct ggml_tensor * gamma = nullptr;
};

// Audio tokenizer decoder model weights
struct audio_decoder_model {
    audio_decoder_config config;
    
    // VQ codebooks
    // vq_first: 1 codebook for first code
    struct ggml_tensor * vq_first_input_proj = nullptr;   // [1, 512, 256]
    struct ggml_tensor * vq_first_output_proj = nullptr;  // [1, 256, 512]
    struct ggml_tensor * vq_first_codebook = nullptr;     // [256, 2048] embedding_sum
    struct ggml_tensor * vq_first_usage = nullptr;        // [2048] cluster_usage
    
    // vq_rest: 15 codebooks for remaining codes
    struct ggml_tensor * vq_rest_input_proj = nullptr;    // [1, 512, 256]
    struct ggml_tensor * vq_rest_output_proj = nullptr;   // [1, 256, 512]
    struct ggml_tensor * vq_rest_codebook[15] = {nullptr}; // [256, 2048] embedding_sum each
    struct ggml_tensor * vq_rest_usage[15] = {nullptr};   // [2048] cluster_usage each
    
    // Upsample blocks (2 ConvNeXt-style blocks)
    upsample_block upsample[2];
    
    // Pre-transformer
    struct ggml_tensor * pre_tfm_input_proj_w = nullptr;  // [1024, 512]
    struct ggml_tensor * pre_tfm_input_proj_b = nullptr;
    pre_tfm_layer pre_tfm_layers[8];
    struct ggml_tensor * pre_tfm_norm_w = nullptr;        // Final RMSNorm
    struct ggml_tensor * pre_tfm_output_proj_w = nullptr; // [512, 1024]
    struct ggml_tensor * pre_tfm_output_proj_b = nullptr;
    
    // Pre-conv: [3, 512, 1024]
    struct ggml_tensor * pre_conv_w = nullptr;
    struct ggml_tensor * pre_conv_b = nullptr;
    
    // Decoder blocks
    // Block 0: Initial conv [7, 1024, 1536]
    struct ggml_tensor * dec0_conv_w = nullptr;
    struct ggml_tensor * dec0_conv_b = nullptr;
    
    // Blocks 1-4: Snake + ConvTranspose + 3 residual blocks
    decoder_block dec_blocks[4];
    
    // Block 5: Final snake activation
    struct ggml_tensor * dec5_snake_alpha = nullptr;
    struct ggml_tensor * dec5_snake_beta = nullptr;
    
    // Block 6: Output conv [7, 96, 1]
    struct ggml_tensor * dec6_conv_w = nullptr;
    struct ggml_tensor * dec6_conv_b = nullptr;
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Compute state for decoder
struct audio_decoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

// Audio tokenizer decoder (vocoder) class
// Decodes discrete audio codes to waveform
class AudioTokenizerDecoder {
public:
    AudioTokenizerDecoder();
    ~AudioTokenizerDecoder();
    
    // Load model from GGUF file (tokenizer model)
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Decode audio codes to waveform
    // codes: audio codes [n_frames, n_codebooks] as int32_t (row-major)
    // n_frames: number of frames
    // Returns: audio samples normalized to [-1, 1] at 24kHz
    bool decode(const int32_t * codes, int32_t n_frames,
                std::vector<float> & samples);
    
    const audio_decoder_config & get_config() const { return model_.config; }
    
    const std::string & get_error() const { return error_msg_; }
    
private:
    // Build computation graph for decoding
    struct ggml_cgraph * build_graph(int32_t n_frames);
    
    // Apply Snake activation: x + (1/alpha) * sin^2(alpha * x)
    struct ggml_tensor * apply_snake(struct ggml_context * ctx,
                                      struct ggml_tensor * x,
                                      struct ggml_tensor * alpha,
                                      struct ggml_tensor * beta);
    
    // Apply RMSNorm
    struct ggml_tensor * apply_rms_norm(struct ggml_context * ctx,
                                         struct ggml_tensor * x,
                                         struct ggml_tensor * w,
                                         float eps);
    
    // Apply pre-transformer layer
    struct ggml_tensor * apply_pre_tfm_layer(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const pre_tfm_layer & layer,
                                              int32_t n_frames,
                                              struct ggml_tensor * positions);
    
    // Apply upsample block (ConvNeXt-style)
    struct ggml_tensor * apply_upsample_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const upsample_block & block,
                                               int block_idx);
    
    // Apply residual block
    struct ggml_tensor * apply_residual_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const residual_block & block);
    
    // Apply decoder block (Snake + ConvTranspose + Residuals)
    struct ggml_tensor * apply_decoder_block(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const decoder_block & block,
                                              int upsample_rate,
                                              int block_idx);
    
    void normalize_codebooks();
    
    audio_decoder_model model_;
    audio_decoder_state state_;
    std::string error_msg_;
    
    // Temporary storage for codes input
    std::vector<int32_t> codes_buf_;
};

// Free model resources
void free_audio_decoder_model(audio_decoder_model & model);

} // namespace qwen3_tts
