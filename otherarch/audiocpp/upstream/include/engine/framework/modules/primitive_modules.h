#pragma once

#include "engine/framework/core/module.h"

#include <limits>

namespace engine::modules {

class MatMulModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & lhs,
        const core::TensorValue & rhs) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class AddModule {
public:
    AddModule() = default;
    explicit AddModule(bool preserve_input_type, bool allow_broadcast = false)
        : preserve_input_type_(preserve_input_type), allow_broadcast_(allow_broadcast) {}
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & lhs,
        const core::TensorValue & rhs) const;

    static const core::ModuleSchema & static_schema() noexcept;
private:
    bool preserve_input_type_ = false;
    bool allow_broadcast_ = false;
};

class MulModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & lhs,
        const core::TensorValue & rhs) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

struct ScaleBiasConfig {
    float scale = 1.0f;
    float bias = 0.0f;
};

class ScaleBiasModule {
public:
    explicit ScaleBiasModule(ScaleBiasConfig config);
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ScaleBiasConfig config_;
};

struct WeightedStatisticsConfig {
    int axis = -1;
    float variance_min = 0.0f;
    float variance_max = std::numeric_limits<float>::infinity();
};

struct WeightedStatisticsOutputs {
    core::TensorValue mean;
    core::TensorValue stddev;
};

class WeightedStatisticsModule {
public:
    explicit WeightedStatisticsModule(WeightedStatisticsConfig config);
    const core::ModuleSchema & schema() const noexcept;
    WeightedStatisticsOutputs build(core::ModuleBuildContext & ctx, const core::TensorValue & input,
        const core::TensorValue & normalized_weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    WeightedStatisticsConfig config_;
};

class RopeInterleavedPairsModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & even,
        const core::TensorValue & odd,
        const core::TensorValue & cos,
        const core::TensorValue & sin) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class TimeMask4dModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & mask) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class ResidualAddModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & residual) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class ScaledBypassModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & original,
        const core::TensorValue & transformed,
        const core::TensorValue & channel_scale) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

struct ReduceConfig {
    int axis = -1;
};

class ReduceMeanModule {
public:
    explicit ReduceMeanModule(ReduceConfig config);

    const ReduceConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ReduceConfig config_;
};

class ReduceSumModule {
public:
    explicit ReduceSumModule(ReduceConfig config);

    const ReduceConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ReduceConfig config_;
};

}  // namespace engine::modules
