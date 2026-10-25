#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace engine::core {

enum class BackendType {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
    Metal,
    BestAvailable,
};

// CUDA and HIP share ggml's ggml-cuda implementation for a small set of
// explicitly verified operators. This is intentionally not a generic GPU test.
constexpr bool uses_ggml_cuda_or_hip_backend(BackendType type) noexcept {
    return type == BackendType::Cuda || type == BackendType::Hip;
}

constexpr size_t kMaxTensorRank = 4;

struct TensorShape {
    std::array<int64_t, kMaxTensorRank> dims = {1, 1, 1, 1};
    size_t rank = 0;

    static TensorShape from_dims(std::initializer_list<int64_t> dims_init);

    int64_t at(size_t index) const;
    int64_t last_dim() const;
    int64_t prefix_elements() const;
    int64_t num_elements() const;
    TensorShape with_last_dim(int64_t value) const;
    std::string to_string() const;
};

struct TensorValue {
    ggml_tensor * tensor = nullptr;
    TensorShape shape = {};
    ggml_type type = GGML_TYPE_F32;

    bool valid() const noexcept;
};

struct ModuleBuildContext {
    ggml_context * ggml = nullptr;
    const char * module_instance_name = nullptr;
    BackendType backend_type = BackendType::Cpu;
    // Preserve logical shapes while omitting physical identity views. Opt-in:
    // removing those views can enable backend fusion and change rounding.
    bool elide_identity_reshapes = false;
};

enum class PortKind {
    Activation,
    Parameter,
};

struct ModulePortSpec {
    const char * name;
    PortKind kind;
    bool optional;
};

struct ModuleSchema {
    const char * type_name;
    const char * category;
    const ModulePortSpec * inputs;
    size_t input_count;
    const ModulePortSpec * outputs;
    size_t output_count;
    const char * description;
};

std::array<int64_t, kMaxTensorRank> to_ggml_dims(const TensorShape & shape);
int logical_axis_to_ggml_axis(size_t rank, int logical_axis);

TensorValue make_tensor(ModuleBuildContext & ctx, ggml_type type, const TensorShape & shape);
TensorValue wrap_tensor(ggml_tensor * tensor, const TensorShape & shape, ggml_type type = GGML_TYPE_F32);
TensorValue reshape_tensor(ModuleBuildContext & ctx, const TensorValue & value, const TensorShape & new_shape);
bool has_backend_addressable_layout(const ggml_tensor * tensor) noexcept;
TensorValue ensure_backend_addressable_layout(ModuleBuildContext & ctx, const TensorValue & value);

void validate_shape(const TensorValue & value, const TensorShape & expected, const char * name);
void validate_last_dim(const TensorValue & value, int64_t expected, const char * name);
void validate_rank_between(const TensorValue & value, size_t min_rank, size_t max_rank, const char * name);

}  // namespace engine::core
