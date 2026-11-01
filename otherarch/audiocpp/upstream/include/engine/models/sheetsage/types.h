#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstddef>
#include <cstdint>

namespace engine::models::sheetsage {

struct SheetSage2DecoderConfig {
    int64_t vocab_size = 31678;
    int64_t hidden_size = 512;
    int64_t encoder_hidden_size = 1024;
    int64_t intermediate_size = 2048;
    int64_t decoder_layers = 6;
    int64_t num_attention_heads = 8;
    int64_t max_position_embeddings = 5120;
    int64_t pad_token_id = 1;
    float layer_norm_eps = 1.0e-5F;
    int64_t encoder_layers = 24;
    int64_t encoder_attention_heads = 16;
    int64_t encoder_intermediate_size = 4096;
    int64_t mel_bins = 128;
    int64_t sampling_rate = 24000;
    int64_t n_fft = 2048;
    int64_t win_length = 2048;
    int64_t hop_length = 240;
    int64_t input_audio_length_samples = 7200000;
    int64_t convnext_kernel_size = 7;
    int64_t conformer_conv_kernel_size = 31;
    float encoder_layer_norm_eps = 1.0e-5F;
    float subsampling_layer_norm_eps = 1.0e-6F;
    float rotary_embedding_base = 10000.0F;
};

struct SheetSage2DecoderRuntimeOptions {
    size_t graph_arena_bytes = 1536ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    int64_t attention_tile_rows = 0;
};

}  // namespace engine::models::sheetsage
