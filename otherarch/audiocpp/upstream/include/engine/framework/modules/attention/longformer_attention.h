#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"

#include <optional>
#include <vector>

namespace engine::modules {

struct LongformerAttentionDebugTensors {
    core::TensorValue input;
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
    core::TensorValue p;
};

struct LongformerAttentionExecutionState {
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    int64_t padded_time = 0;
    int64_t pos_frames = 0;
    int64_t left_context = 0;
    int64_t right_context = 0;
    int64_t hidden_size = 0;
    bool capture_debug = false;

    core::TensorValue q_weight;
    core::TensorValue k_weight;
    core::TensorValue v_weight;
    core::TensorValue pos_weight;
    core::TensorValue out_weight;
    core::TensorValue pos_bias_u_tensor;
    core::TensorValue pos_bias_v_tensor;

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> p;
    std::vector<float> q_with_bias_u;
    std::vector<float> q_with_bias_v;
    std::vector<float> diagonal_matrix_ac;
    std::vector<float> diagonal_matrix_bd;
    std::vector<float> d_mask;
    std::vector<float> scores;
    std::vector<float> attn;
    std::vector<float> out;
};

class LongformerRelativeSelfAttentionModule {
public:
    explicit LongformerRelativeSelfAttentionModule(RelativeAttentionConfig config);

    const RelativeAttentionConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & pos_emb,
        const RelativeAttentionWeights & weights,
        const core::TensorValue & keep_mask,
        int64_t layer_index,
        LongformerAttentionExecutionState * exec_state,
        LongformerAttentionDebugTensors * debug = nullptr,
        const std::optional<core::TensorValue> & projected_pos_emb = std::nullopt) const;

private:
    RelativeAttentionConfig config_;
};

class LongformerRelativeSelfAttentionGpuModule {
public:
    explicit LongformerRelativeSelfAttentionGpuModule(RelativeAttentionConfig config);

    const RelativeAttentionConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & pos_emb,
        const RelativeAttentionWeights & weights,
        const core::TensorValue & keep_mask,
        int64_t layer_index,
        LongformerAttentionExecutionState * exec_state,
        LongformerAttentionDebugTensors * debug = nullptr,
        const std::optional<core::TensorValue> & projected_pos_emb = std::nullopt) const;

private:
    RelativeAttentionConfig config_;
};

}  // namespace engine::modules
