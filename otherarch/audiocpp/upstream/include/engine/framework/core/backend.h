#pragma once

#include "engine/framework/core/module.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace engine::core {

class ExecutionContext;

struct BackendConfig {
    BackendType type = BackendType::Cpu;
    int device = 0;
    int threads = 1;
};

struct BackendDeviceInfo {
    std::string backend;    // ggml registry name, e.g. "CUDA", "ROCm", "Vulkan", "CPU"
    int index = 0;          // device index within the owning registry (the value --device takes)
    std::string name;       // human-readable device name
    std::string type;       // CPU, GPU, IGPU, ACCEL, or META
};

// Enumerates every device of every loaded ggml backend registry, in registry order.
std::vector<BackendDeviceInfo> list_backend_devices();
void print_backend_devices(std::ostream & out);

// Load every registered ggml backend registry (idempotent). Needed before
// query_backend_memory() can report GPU memory on a process that has not yet
// initialized a backend.
void ensure_backends_loaded();

struct BackendMemorySnapshot {
    bool available = false;
    int64_t total_bytes = 0;
    int64_t used_bytes = 0;
    int64_t free_bytes = 0;
};

ggml_backend_t init_backend(const BackendConfig & config);
void set_backend_threads(ggml_backend_t backend, int threads);
// The CPU backend's extra weight buffer types, in ggml's order of preference
// (AMX and KleidiAI when built, then CPU_REPACK). Their weights are laid out
// for their own matmul kernels. Empty for other backends.
std::vector<ggml_backend_buffer_type_t> cpu_extra_buffer_types(ggml_backend_t backend);
// Whether `backend` runs ggml_mul_mat(w, F32 x) for a 2-D `type` weight w of
// [ne0, ne1] (ggml order) held in `buffer_type`; llama.cpp's
// weight_buft_supported probe.
bool buffer_type_supports_matmul_weight(
    ggml_backend_t backend, ggml_backend_buffer_type_t buffer_type, ggml_type type, int64_t ne0, int64_t ne1);
BackendType backend_type(ggml_backend_t backend);
bool is_host_backend(ggml_backend_t backend);
bool uses_host_graph_plan(BackendType type);
bool uses_host_graph_plan(ggml_backend_t backend);
bool requested_backend_uses_host_graph_plan(const BackendConfig & config);
// Drop the CUDA/HIP context's cached (idle) pool memory back to the driver.
// No-op on other backends. For use on allocation-failure paths before a retry.
void trim_backend_pools(ggml_backend_t backend);

// Sets the CUDA stream-creation priority for this backend instance (lower =
// higher priority, 0 = default). No-op on non-CUDA backends and on builds
// whose CUDA backend does not export the hook.
void set_backend_stream_priority(ggml_backend_t backend, int priority);
void * backend_cuda_stream(ggml_backend_t backend);
// evict_cuda_graph_cache=false (the default) is the historical no-op;
// true drops the backend's cached compiled-graph state (CUDA/HIP graph
// cache) for this cgraph at destruction — opt in per family.
void release_backend_graph_resources(ggml_backend_t backend, ggml_cgraph * graph, bool evict_cuda_graph_cache = false);
void release_backend_graph_resources(BackendType backend_type, ggml_backend_t backend, ggml_cgraph * graph, bool evict_cuda_graph_cache = false);
void validate_backend_graph_supported(ggml_backend_t backend, ggml_cgraph * graph, const char * label);
// validate_backend_graph_supported for a graph whose weights may be held in
// CPU extra buffers: first throws for a node that views such a weight, or a
// matmul of one with a non-contiguous input.
void validate_backend_graph_with_cpu_extra_buffers(ggml_backend_t backend, ggml_cgraph * graph, const char * label);
BackendMemorySnapshot query_backend_memory(ggml_backend_t backend, int device_hint);
BackendMemorySnapshot query_backend_memory(const BackendConfig & config);
ggml_backend_graph_plan_t create_backend_graph_plan_if_host(ggml_backend_t backend, ggml_cgraph * graph);
void free_backend_graph_plan(ggml_backend_t backend, ggml_backend_graph_plan_t & plan);
ggml_status compute_backend_graph(
    ggml_backend_t backend,
    ggml_cgraph * graph,
    ggml_backend_graph_plan_t plan = nullptr,
    const char * label = nullptr);

struct HostGraphPlan {
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_t backend = nullptr;

    ~HostGraphPlan() { reset(); }

    bool active() const noexcept { return plan != nullptr; }
    void reset() {
        if (plan != nullptr && backend != nullptr) {
            ggml_backend_graph_plan_free(backend, plan);
        }
        plan = nullptr;
        backend = nullptr;
    }
};

void prepare_host_graph_plan(const ExecutionContext & execution_context, ggml_cgraph * graph, HostGraphPlan & plan);
ggml_status compute_graph(
    const ExecutionContext & execution_context,
    ggml_cgraph * graph,
    HostGraphPlan & plan,
    const char * label = nullptr);

void write_tensor_f32(const TensorValue & tensor, const float * values, size_t count);
void write_tensor_f32_slice(const TensorValue & tensor, size_t element_offset, const float * values, size_t count);
void write_tensor_f32(const TensorValue & tensor, const std::vector<float> & values);
void write_tensor_f16(const TensorValue & tensor, const float * values, size_t count);
void write_tensor_f16(const TensorValue & tensor, const std::vector<float> & values);
void write_tensor_bf16(const TensorValue & tensor, const float * values, size_t count);
void write_tensor_bf16(const TensorValue & tensor, const std::vector<float> & values);
void write_tensor_float(const TensorValue & tensor, const float * values, size_t count);
void write_tensor_float(const TensorValue & tensor, const std::vector<float> & values);
void write_tensor_bytes(const TensorValue & tensor, const std::vector<std::byte> & bytes);
void round_f32_to_bf16_in_place(float * values, size_t count);
void round_f32_to_bf16_in_place(std::vector<float> & values);
void write_tensor_i32(const TensorValue & tensor, const int32_t * values, size_t count);
void write_tensor_i32(const TensorValue & tensor, const std::vector<int32_t> & values);
void read_tensor_f32_into(const ggml_tensor * tensor, std::vector<float> & values);
std::vector<float> read_tensor_f32(const ggml_tensor * tensor);
void read_tensor_f16_into(const ggml_tensor * tensor, std::vector<float> & values);
std::vector<float> read_tensor_f16(const ggml_tensor * tensor);
void read_tensor_bf16_into(const ggml_tensor * tensor, std::vector<float> & values);
std::vector<float> read_tensor_bf16(const ggml_tensor * tensor);
void read_tensor_float_into(const ggml_tensor * tensor, std::vector<float> & values);
std::vector<float> read_tensor_float(const ggml_tensor * tensor);
void read_tensor_bytes_into(const ggml_tensor * tensor, std::vector<std::byte> & bytes);
std::vector<std::byte> read_tensor_bytes(const ggml_tensor * tensor);
void read_tensor_i32_into(const ggml_tensor * tensor, std::vector<int32_t> & values);
std::vector<int32_t> read_tensor_i32(const ggml_tensor * tensor);

}  // namespace engine::core
