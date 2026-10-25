#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"

#include <optional>

namespace engine::modules {

enum class FeedForwardActivation {
    Gelu,
    Relu,
};

struct FeedForwardConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    bool use_bias = true;
    GeluApproximation gelu_approximation = GeluApproximation::ExactErf;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    FeedForwardActivation activation = FeedForwardActivation::Gelu;
};

struct FeedForwardWeights {
    core::TensorValue fc1_weight;
    std::optional<core::TensorValue> fc1_bias;
    core::TensorValue fc2_weight;
    std::optional<core::TensorValue> fc2_bias;
};

enum class GatedFeedForwardActivation {
    Gelu,
    Silu,
};

struct GatedFeedForwardConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    bool use_bias = false;
    GatedFeedForwardActivation activation = GatedFeedForwardActivation::Gelu;
    GeluApproximation gelu_approximation = GeluApproximation::Tanh;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    bool use_weight_type_projection_precision = false;
    bool use_fast_cuda_projection = false;
};

struct GatedFeedForwardWeights {
    LinearWeights gate_proj;
    LinearWeights up_proj;
    LinearWeights down_proj;
};

struct ConvFeedForwardConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t kernel_size = 0;
    bool causal = false;
    bool use_bias = false;
    GeluApproximation gelu_approximation = GeluApproximation::Tanh;
};

struct ConvFeedForwardWeights {
    Conv1dWeights proj;
    Conv1dWeights out;
};

struct GatedConvFeedForwardConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t kernel_size = 0;
    bool causal = false;
    bool explicit_symmetric_padding = false;
    bool use_bias = false;
    GatedFeedForwardActivation activation = GatedFeedForwardActivation::Silu;
};

struct GatedConvFeedForwardWeights {
    Conv1dWeights gate_proj;
    Conv1dWeights up_proj;
    Conv1dWeights down_proj;
};

class FeedForwardModule {
public:
    explicit FeedForwardModule(FeedForwardConfig config);

    const FeedForwardConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const FeedForwardWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    FeedForwardConfig config_;
};

class FeedForwardGeluModule {
public:
    explicit FeedForwardGeluModule(FeedForwardConfig config);

    const FeedForwardConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const FeedForwardWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    FeedForwardConfig config_;
};

class GatedFeedForwardModule {
public:
    explicit GatedFeedForwardModule(GatedFeedForwardConfig config);

    const GatedFeedForwardConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const GatedFeedForwardWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    GatedFeedForwardConfig config_;
};

class ConvFeedForwardModule {
public:
    explicit ConvFeedForwardModule(ConvFeedForwardConfig config);

    const ConvFeedForwardConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const ConvFeedForwardWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    ConvFeedForwardConfig config_;
};

class GatedConvFeedForwardModule {
public:
    explicit GatedConvFeedForwardModule(GatedConvFeedForwardConfig config);

    const GatedConvFeedForwardConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const GatedConvFeedForwardWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    GatedConvFeedForwardConfig config_;
};

}  // namespace engine::modules
