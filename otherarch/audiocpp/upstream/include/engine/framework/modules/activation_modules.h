#pragma once

#include "engine/framework/core/module.h"

#include <optional>

namespace engine::modules {

enum class GeluApproximation {
    ExactErf,
    Tanh,
    Quick,
};

struct GeluConfig {
    GeluApproximation approximation = GeluApproximation::ExactErf;
};

class ReluModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

struct LeakyReluConfig {
    float negative_slope = 0.01F;
};

struct PReluConfig {
    size_t channel_axis = 1;
};

class PReluModule {
public:
    explicit PReluModule(PReluConfig config = {});
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & slope) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    PReluConfig config_;
};

class LeakyReluModule {
public:
    explicit LeakyReluModule(LeakyReluConfig config = {});

    const LeakyReluConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    LeakyReluConfig config_;
};

class SigmoidModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class TanhModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SqrtModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class Bf16RoundModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class GeluModule {
public:
    explicit GeluModule(GeluConfig config = {});

    const GeluConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    GeluConfig config_;
};

class GegluModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SiluModule {
public:
    SiluModule() = default;
    explicit SiluModule(bool preserve_input_type) : preserve_input_type_(preserve_input_type) {}
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
private:
    bool preserve_input_type_ = false;
};

class SeluModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SwooshLModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SwooshRModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class EluModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SoftmaxModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

struct GLUConfig {
    bool contiguous_gate = false;
};

class GLUModule {
public:
    GLUModule() = default;
    explicit GLUModule(GLUConfig config);
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    GLUConfig config_;
};

struct Snake1dConfig {
    int64_t hidden_size = 0;
};

struct Snake1dWeights {
    core::TensorValue alpha;
    std::optional<core::TensorValue> inverse_alpha;
};

class Snake1dModule {
public:
    explicit Snake1dModule(Snake1dConfig config);

    const Snake1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Snake1dWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    Snake1dConfig config_;
};

struct SnakeBeta1dConfig {
    int64_t hidden_size = 0;
    bool logscale = true;
};

struct SnakeBeta1dWeights {
    core::TensorValue alpha;
    core::TensorValue beta;
};

class SnakeBeta1dModule {
public:
    explicit SnakeBeta1dModule(SnakeBeta1dConfig config);

    const SnakeBeta1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const SnakeBeta1dWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    SnakeBeta1dConfig config_;
};

enum class AliasFreeActivationKind {
    SnakeBeta,
};

struct AliasFreeActivationConfig {
    int64_t channels = 0;
    int64_t kernel_size = 0;
    int64_t upsample_ratio = 2;
    AliasFreeActivationKind kind = AliasFreeActivationKind::SnakeBeta;
};

struct AliasFreeActivationWeights {
    core::TensorValue alpha;
    core::TensorValue inv_beta;
    core::TensorValue up_filter_even;
    core::TensorValue up_filter_odd;
    core::TensorValue down_filter;
};

class AliasFreeActivationModule {
public:
    explicit AliasFreeActivationModule(AliasFreeActivationConfig config);

    const AliasFreeActivationConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const AliasFreeActivationWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    AliasFreeActivationConfig config_;
};

}  // namespace engine::modules
