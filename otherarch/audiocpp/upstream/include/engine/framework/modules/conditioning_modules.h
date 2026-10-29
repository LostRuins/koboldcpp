#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>
#include <vector>

namespace engine::modules {

struct LayerScaleWeights {
    core::TensorValue scale;
};

class LayerScaleModule {
public:
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const LayerScaleWeights & weights) const;
};

struct AdaLNModulationWeights {
    LinearWeights projection;
};

struct AdaLNResidualMLPWeights {
    NormWeights norm;
    AdaLNModulationWeights modulation;
    LinearWeights fc1;
    LinearWeights fc2;
};

struct AdaLNResidualMLPConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t conditioning_size = 0;
    float eps = 1e-6f;
    bool use_bias = true;
};

class AdaLNResidualMLPModule {
public:
    explicit AdaLNResidualMLPModule(AdaLNResidualMLPConfig config);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & conditioning,
        const AdaLNResidualMLPWeights & weights) const;

private:
    AdaLNResidualMLPConfig config_;
};

struct FinalAdaLNProjectionWeights {
    AdaLNModulationWeights modulation;
    LinearWeights projection;
};

class FinalAdaLNProjectionModule {
public:
    explicit FinalAdaLNProjectionModule(AdaLNResidualMLPConfig config);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & conditioning,
        const FinalAdaLNProjectionWeights & weights) const;

private:
    AdaLNResidualMLPConfig config_;
};

struct ConditionedFlowMLPConfig {
    int64_t input_size = 0;
    int64_t hidden_size = 0;
    int64_t condition_size = 0;
    int64_t output_size = 0;
    int64_t layers = 0;
    float eps = 1e-6f;
    bool use_bias = true;
};

struct TimestepEmbeddingWeights {
    core::TensorValue freqs;
    LinearWeights fc1;
    LinearWeights fc2;
    std::optional<core::TensorValue> rms_weight;
};

struct TimestepEmbeddingConfig {
    int64_t frequency_embedding_size = 0;
    int64_t hidden_size = 0;
    float rms_eps = 1e-5f;
};

class TimestepEmbeddingModule {
public:
    explicit TimestepEmbeddingModule(TimestepEmbeddingConfig config);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & timestep,
        const TimestepEmbeddingWeights & weights) const;

private:
    TimestepEmbeddingConfig config_;
};

struct ConditionedFlowMLPWeights {
    LinearWeights input_projection;
    LinearWeights condition_projection;
    std::vector<AdaLNResidualMLPWeights> residual_layers;
    FinalAdaLNProjectionWeights output_projection;
};

class ConditionedFlowMLPModule {
public:
    explicit ConditionedFlowMLPModule(ConditionedFlowMLPConfig config);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & condition,
        const core::TensorValue & input,
        const ConditionedFlowMLPWeights & weights) const;

private:
    ConditionedFlowMLPConfig config_;
};

struct TimedConditionedFlowMLPWeights {
    TimestepEmbeddingWeights start_time_embedding;
    TimestepEmbeddingWeights end_time_embedding;
    LinearWeights input_projection;
    LinearWeights condition_projection;
    std::vector<AdaLNResidualMLPWeights> residual_layers;
    FinalAdaLNProjectionWeights output_projection;
};

class TimedConditionedFlowMLPModule {
public:
    explicit TimedConditionedFlowMLPModule(ConditionedFlowMLPConfig config);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & condition,
        const core::TensorValue & start_time,
        const core::TensorValue & end_time,
        const core::TensorValue & input,
        const TimedConditionedFlowMLPWeights & weights) const;

private:
    ConditionedFlowMLPConfig config_;
};

struct SpeakerConditioningConfig {
    int64_t hidden_size = 0;
    int64_t speaker_dim = 0;
    bool use_bias = true;
};

struct SpeakerConditioningWeights {
    core::TensorValue proj_weight;
    std::optional<core::TensorValue> proj_bias;
};

class SpeakerConditioningModule {
public:
    explicit SpeakerConditioningModule(SpeakerConditioningConfig config);

    const SpeakerConditioningConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & speaker,
        const SpeakerConditioningWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    SpeakerConditioningConfig config_;
};

struct FiLMConfig {
    int64_t hidden_size = 0;
    int64_t conditioning_dim = 0;
    bool use_bias = true;
};

struct FiLMWeights {
    core::TensorValue gamma_weight;
    std::optional<core::TensorValue> gamma_bias;
    core::TensorValue beta_weight;
    std::optional<core::TensorValue> beta_bias;
};

class FiLMModule {
public:
    explicit FiLMModule(FiLMConfig config);

    const FiLMConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & conditioning,
        const FiLMWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    FiLMConfig config_;
};

enum class AdaptiveLayerNormMode {
    Affine,
    RmsAffine,
    Scale,
};

struct AdaptiveLayerNormConfig {
    int64_t hidden_size = 0;
    float eps = 1e-5f;
    AdaptiveLayerNormMode mode = AdaptiveLayerNormMode::RmsAffine;
    int64_t shift_index = -1;
    int64_t scale_index = 0;
};

struct AdaptiveLayerNormWeights {
    core::TensorValue table;
};

class AdaptiveLayerNormModule {
public:
    explicit AdaptiveLayerNormModule(AdaptiveLayerNormConfig config);

    const AdaptiveLayerNormConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & conditioning,
        const AdaptiveLayerNormWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    AdaptiveLayerNormConfig config_;
};

struct PriorEncoderBlockConfig {
    int64_t hidden_size = 0;
    int64_t latent_size = 0;
    float eps = 1e-5f;
    bool use_bias = true;
};

struct PriorEncoderBlockWeights {
    core::TensorValue proj_weight;
    std::optional<core::TensorValue> proj_bias;
    NormWeights norm;
};

class PriorEncoderBlockModule {
public:
    explicit PriorEncoderBlockModule(PriorEncoderBlockConfig config);

    const PriorEncoderBlockConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const PriorEncoderBlockWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    PriorEncoderBlockConfig config_;
};

struct SqueezeExcite1dConfig {
    int64_t channels = 0;
    int64_t hidden_channels = 0;
    bool use_bias = true;
};

struct SqueezeExcite1dWeights {
    Conv1dWeights fc1;
    Conv1dWeights fc2;
};

class SqueezeExcite1dModule {
public:
    explicit SqueezeExcite1dModule(SqueezeExcite1dConfig config);

    const SqueezeExcite1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const SqueezeExcite1dWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    SqueezeExcite1dConfig config_;
};

}  // namespace engine::modules
