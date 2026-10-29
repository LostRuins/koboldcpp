#pragma once

#include "engine/framework/core/module.h"

#include <array>
#include <cstdint>

namespace engine::modules {

struct ReshapeConfig {
    core::TensorShape output_shape;
};

class ReshapeModule {
public:
    explicit ReshapeModule(ReshapeConfig config);

    const ReshapeConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ReshapeConfig config_;
};

struct TransposeConfig {
    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    size_t rank = 0;
};

class TransposeModule {
public:
    explicit TransposeModule(TransposeConfig config);

    const TransposeConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    TransposeConfig config_;
};

struct SliceConfig {
    int axis = 0;
    int64_t start = 0;
    int64_t length = 0;
};

class SliceModule {
public:
    explicit SliceModule(SliceConfig config);

    const SliceConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    SliceConfig config_;
};

struct ConcatConfig {
    int axis = 0;
    bool cuda_contiguous_4d_lowering = false;
};

class ConcatModule {
public:
    explicit ConcatModule(ConcatConfig config);

    const ConcatConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & lhs,
        const core::TensorValue & rhs) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ConcatConfig config_;
};

struct RepeatConfig {
    core::TensorShape output_shape;
};

class RepeatModule {
public:
    explicit RepeatModule(RepeatConfig config);

    const RepeatConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    RepeatConfig config_;
};

struct PaddingConfig {
    int64_t target_frames = 0;
};

class PaddingModule {
public:
    explicit PaddingModule(PaddingConfig config);

    const PaddingConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    PaddingConfig config_;
};

struct ReflectPad1dConfig {
    int64_t left = 0;
    int64_t right = 0;
};

class ReflectPad1dModule {
public:
    explicit ReflectPad1dModule(ReflectPad1dConfig config);

    const ReflectPad1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ReflectPad1dConfig config_;
};

struct Pad2dConfig {
    int64_t left = 0;
    int64_t right = 0;
    int64_t top = 0;
    int64_t bottom = 0;
};

class Pad2dModule {
public:
    explicit Pad2dModule(Pad2dConfig config);

    const Pad2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    Pad2dConfig config_;
};

enum class Interpolate1dMode {
    Nearest,
    Linear,
};

struct Interpolate1dConfig {
    int64_t output_frames = 0;
    Interpolate1dMode mode = Interpolate1dMode::Nearest;
};

class Interpolate1dModule {
public:
    explicit Interpolate1dModule(Interpolate1dConfig config);

    const Interpolate1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    Interpolate1dConfig config_;
};

struct NearestUpsample2dConfig {
    int64_t output_height = 0;
    int64_t output_width = 0;
};

class NearestUpsample2dModule {
public:
    explicit NearestUpsample2dModule(NearestUpsample2dConfig config);

    const NearestUpsample2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    NearestUpsample2dConfig config_;
};

class MaskingModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & mask) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

}  // namespace engine::modules
