#pragma once

#include "engine/models/chatterbox/components.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace engine::models::chatterbox::components {

struct S3TokenizerLogMelOutputs {
    std::vector<float> log_mel;
    int64_t n_mels = 0;
    int64_t frames = 0;
};

struct S3PromptMelOutputs {
    std::vector<float> mel;
    int64_t n_mels = 0;
    int64_t frames = 0;
};

struct CampplusFbankOutputs {
    std::vector<float> features;
    int64_t frames = 0;
    int64_t dims = 0;
};

uint64_t choose_seed(uint64_t seed);

S3TokenizerLogMelOutputs compute_s3tokenizer_log_mel(const runtime::AudioBuffer & audio);
S3PromptMelOutputs compute_s3_prompt_mel(const runtime::AudioBuffer & audio);
CampplusFbankOutputs compute_campplus_fbank(const runtime::AudioBuffer & audio);

struct S3TokenizerV2Weights {
    struct Conv1dWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
    };

    struct LayerNormWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
    };

    struct LinearWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };

    struct BlockWeights {
        LayerNormWeights attn_ln;
        LinearWeights attn_qkv_packed;
        LinearWeights attn_out;
        engine::core::TensorValue fsmn_weight_tensor;
        LayerNormWeights mlp_ln;
        LinearWeights mlp_fc1;
        LinearWeights mlp_fc2;
    };

    Conv1dWeights conv1;
    Conv1dWeights conv2;
    std::vector<BlockWeights> blocks;
    LinearWeights quantizer_project_down;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

std::vector<float> read_f32_tensor(
    const engine::assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape);

}  // namespace engine::models::chatterbox::components
