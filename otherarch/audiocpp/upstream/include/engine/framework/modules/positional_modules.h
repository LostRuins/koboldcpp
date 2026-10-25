#pragma once

#include "engine/framework/core/module.h"

#include <cstdint>
#include <optional>

namespace engine::modules {

struct RoPEConfig {
    int64_t dimensions = 0;
    int mode = GGML_ROPE_TYPE_NORMAL;
    float theta = 10000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
    std::optional<ggml_type> intermediate_round_type = std::nullopt;
};

class RoPEModule {
public:
    explicit RoPEModule(RoPEConfig config);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const core::TensorValue * frequency_factors = nullptr) const;

private:
    RoPEConfig config_;
};

struct SplitRoPEConfig {
    int64_t dimensions = 0;
};

class SplitRoPEModule {
public:
    explicit SplitRoPEModule(SplitRoPEConfig config);

    const SplitRoPEConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & cos,
        const core::TensorValue & sin) const;

private:
    SplitRoPEConfig config_;
};

struct SplitRoPEAttentionConfig {
    int64_t heads = 0;
    int64_t head_dim = 0;
};

class SplitRoPEAttentionModule {
public:
    explicit SplitRoPEAttentionModule(SplitRoPEAttentionConfig config);

    const SplitRoPEAttentionConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & cos,
        const core::TensorValue & sin) const;

private:
    SplitRoPEAttentionConfig config_;
};

}  // namespace engine::modules
