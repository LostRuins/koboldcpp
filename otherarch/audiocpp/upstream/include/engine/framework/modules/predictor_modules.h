#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conv_modules.h"

namespace engine::modules {

struct DurationPredictorConfig {
    int64_t hidden_size = 0;
    int64_t channels = 0;
    bool use_bias = true;
};

struct DurationPredictorWeights {
    Conv1dWeights conv1;
    Conv1dWeights conv2;
    Conv1dWeights proj;
};

class DurationPredictorModule {
public:
    explicit DurationPredictorModule(DurationPredictorConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const DurationPredictorWeights & weights) const;

private:
    DurationPredictorConfig config_;
};

class VariancePredictorModule {
public:
    explicit VariancePredictorModule(DurationPredictorConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const DurationPredictorWeights & weights) const;

private:
    DurationPredictorConfig config_;
};

struct GlobalStyleTokenBlockConfig {
    int64_t style_dim = 0;
    int64_t token_count = 0;
    int64_t num_heads = 0;
    bool use_bias = true;
};

struct GlobalStyleTokenBlockWeights {
    core::TensorValue tokens;
    AttentionWeights attention;
};

class GlobalStyleTokenBlockModule {
public:
    explicit GlobalStyleTokenBlockModule(GlobalStyleTokenBlockConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & query,
        const GlobalStyleTokenBlockWeights & weights) const;

private:
    GlobalStyleTokenBlockConfig config_;
};

struct PosteriorEncoderBlockConfig {
    int64_t channels_in = 0;
    int64_t latent_size = 0;
    bool use_bias = true;
};

struct PosteriorEncoderBlockWeights {
    Conv1dWeights conv1;
    Conv1dWeights conv2;
};

class PosteriorEncoderBlockModule {
public:
    explicit PosteriorEncoderBlockModule(PosteriorEncoderBlockConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const PosteriorEncoderBlockWeights & weights) const;

private:
    PosteriorEncoderBlockConfig config_;
};

struct FlowBlockConfig {
    int64_t channels = 0;
    bool use_bias = true;
};

struct FlowBlockWeights {
    Conv1dWeights scale;
    Conv1dWeights shift;
};

class FlowBlockModule {
public:
    explicit FlowBlockModule(FlowBlockConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const FlowBlockWeights & weights) const;

private:
    FlowBlockConfig config_;
};

}  // namespace engine::modules
