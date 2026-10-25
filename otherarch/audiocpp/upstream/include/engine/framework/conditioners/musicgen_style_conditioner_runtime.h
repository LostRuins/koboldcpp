#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/wav2vec2_encoder.h"

#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::conditioners {

struct MusicGenStyleConfig {
    modules::Wav2Vec2EncoderConfig mert;
    int64_t hidden_size = 512;
    int64_t mert_hidden_size = 768;
    int64_t output_dim = 1536;
    int64_t transformer_layers = 8;
    int64_t transformer_heads = 8;
    int64_t transformer_intermediate_size = 2048;
    int64_t codebook_size = 1024;
    int64_t eval_codebooks = 1;
    int64_t downsample_factor = 15;
    float layer_norm_eps = 1.0e-5F;
    float batch_norm_eps = 1.0e-5F;

    static MusicGenStyleConfig mert_default();
};

struct MusicGenStyleRuntimeOptions {
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    size_t style_weight_context_bytes = 512ull * 1024ull * 1024ull;
    size_t graph_arena_bytes = 512ull * 1024ull * 1024ull;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
};

struct MusicGenStyleEmbedding {
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t features = 0;
    std::vector<float> values;
};

class MusicGenStyleConditionerRuntime {
public:
    MusicGenStyleConditionerRuntime(
        std::shared_ptr<const assets::TensorSource> mert_source,
        std::shared_ptr<const assets::TensorSource> style_source,
        core::ExecutionContext & execution,
        MusicGenStyleConfig config = MusicGenStyleConfig::mert_default(),
        MusicGenStyleRuntimeOptions options = {});
    ~MusicGenStyleConditionerRuntime();

    MusicGenStyleConditionerRuntime(const MusicGenStyleConditionerRuntime &) = delete;
    MusicGenStyleConditionerRuntime & operator=(const MusicGenStyleConditionerRuntime &) = delete;

    MusicGenStyleEmbedding encode_audio(
        const std::vector<float> & waveform,
        int64_t batch,
        int64_t samples,
        size_t threads = 0);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::conditioners
