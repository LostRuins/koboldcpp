#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

namespace engine::core {

void ensure_backends_loaded() {
    if (ggml_backend_reg_count() == 0) {
        ggml_backend_load_all();
    }
}

namespace {

// A backend is identified by the name of the ggml registry that owns it. The device type
// (GPU/IGPU/ACCEL) deliberately plays no part in that: Metal reports GPU rather than ACCEL,
// Vulkan reports IGPU on integrated GPUs, and those values are free to change upstream.
//
// The CUDA registry name depends on how ggml was built - GGML_CUDA_NAME is "MUSA" under
// MUSA - so every alias has to be accepted. HIP builds share the ggml CUDA backend but
// register as "ROCm" and get their own BackendType::Hip. The names are spelled out here
// instead of pulled from ggml-cuda.h/ggml-vulkan.h because those headers resolve against the
// backend's own build flags, which are not visible from this translation unit.
struct BackendRegNames {
    BackendType  type;
    const char * names[3];  // unused entries are nullptr
};

constexpr BackendRegNames k_backend_reg_names[] = {
    {BackendType::Cuda,   {"CUDA", "MUSA", nullptr}},
    {BackendType::Hip,    {"ROCm", nullptr, nullptr}},
    {BackendType::Vulkan, {"Vulkan", nullptr, nullptr}},
    {BackendType::Metal,  {"MTL", nullptr, nullptr}},
};

bool reg_name_matches(BackendType type, const char * reg_name) {
    if (reg_name == nullptr) {
        return false;
    }
    for (const BackendRegNames & entry : k_backend_reg_names) {
        if (entry.type != type) {
            continue;
        }
        for (const char * name : entry.names) {
            if (name != nullptr && std::strcmp(reg_name, name) == 0) {
                return true;
            }
        }
        return false;
    }
    return false;
}

bool backend_handle_matches(ggml_backend_t backend, BackendType type) {
    if (backend == nullptr) return false;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) return false;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    return reg != nullptr && reg_name_matches(type, ggml_backend_reg_name(reg));
}

bool is_cuda_backend_handle(ggml_backend_t backend) {
    return backend_handle_matches(backend, BackendType::Cuda);
}

bool is_hip_backend_handle(ggml_backend_t backend) {
    return backend_handle_matches(backend, BackendType::Hip);
}

bool is_vulkan_backend_handle(ggml_backend_t backend) {
    return backend_handle_matches(backend, BackendType::Vulkan);
}

bool is_metal_backend_handle(ggml_backend_t backend) {
    return backend_handle_matches(backend, BackendType::Metal);
}

ggml_backend_reg_t find_reg_by_backend_type(BackendType type) {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (reg != nullptr && reg_name_matches(type, ggml_backend_reg_name(reg))) {
            return reg;
        }
    }
    return nullptr;
}

// Device indices are relative to the owning registry, matching how ggml itself numbers the
// devices of a backend.
ggml_backend_dev_t find_device_by_backend_type(BackendType type, int device_index) {
    if (device_index < 0) {
        return nullptr;
    }
    ggml_backend_reg_t reg = find_reg_by_backend_type(type);
    if (reg == nullptr) {
        return nullptr;
    }
    if (static_cast<size_t>(device_index) >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_reg_dev_get(reg, static_cast<size_t>(device_index));
}

const char * backend_dev_type_label(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "UNKNOWN";
}

std::string describe_available_devices() {
    std::string description;
    for (const auto & device : list_backend_devices()) {
        if (!description.empty()) {
            description += ", ";
        }
        description += device.backend + ":" + std::to_string(device.index);
        if (!device.name.empty()) {
            description += " \"" + device.name + "\"";
        }
        description += " [" + device.type + "]";
    }
    return description.empty() ? "none" : description;
}

std::string describe_missing_device(BackendType type, const char * label, int device_index) {
    ggml_backend_reg_t reg = find_reg_by_backend_type(type);
    std::string message = std::string(label) + " backend requested but ";
    if (reg == nullptr) {
        message += "it is not registered in this build";
    } else {
        message += "registry '" + std::string(ggml_backend_reg_name(reg)) +
            "' has no device " + std::to_string(device_index);
    }
    return message + " (available: " + describe_available_devices() + ")";
}

ggml_backend_t init_device_backend(BackendType type, const char * label, const BackendConfig & config) {
    if (config.device < 0) {
        throw std::runtime_error(
            std::string(label) + " backend requested with negative device index");
    }
    ggml_backend_dev_t device = find_device_by_backend_type(type, config.device);
    if (device == nullptr) {
        throw std::runtime_error(describe_missing_device(type, label, config.device));
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        throw std::runtime_error(
            "Failed to initialize " + std::string(label) + " backend on device " +
            std::to_string(config.device));
    }
    return backend;
}


#ifndef NDEBUG
bool backend_graph_validation_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("ENGINE_VALIDATE_BACKEND_GRAPH");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
#endif

}  // namespace

std::vector<BackendDeviceInfo> list_backend_devices() {
    ensure_backends_loaded();
    std::vector<BackendDeviceInfo> devices;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (reg == nullptr) {
            continue;
        }
        const char * reg_name = ggml_backend_reg_name(reg);
        for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
            if (dev == nullptr) {
                continue;
            }
            BackendDeviceInfo info;
            info.backend = reg_name != nullptr ? reg_name : "<unnamed>";
            info.index = static_cast<int>(j);
            // Prefer the description: ggml's device name is a generic handle
            // ("CUDA0", "ROCm0"), while the description carries the hardware
            // name ("NVIDIA GeForce RTX 2080 Ti") that tells identical cards apart.
            const char * dev_description = ggml_backend_dev_description(dev);
            const char * dev_name = ggml_backend_dev_name(dev);
            if (dev_description != nullptr && dev_description[0] != '\0') {
                info.name = dev_description;
            } else if (dev_name != nullptr) {
                info.name = dev_name;
            }
            info.type = backend_dev_type_label(ggml_backend_dev_type(dev));
            devices.push_back(std::move(info));
        }
    }
    return devices;
}

void print_backend_devices(std::ostream & out) {
    const auto devices = list_backend_devices();
    out << "available_devices=" << devices.size() << "\n";
    for (const auto & device : devices) {
        out << device.backend << ":" << device.index;
        if (!device.name.empty()) {
            out << " \"" << device.name << "\"";
        }
        out << " [" << device.type << "]\n";
    }
    out << "select with: --backend <cuda|hip|vulkan|metal|cpu> --device <index>\n";
}

ggml_backend_t init_backend(const BackendConfig & config) {
    ensure_backends_loaded();
    switch (config.type) {
        case BackendType::Cpu: {
            ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error("Failed to initialize CPU backend");
            }
            return backend;
        }
        case BackendType::Hip:
            return init_device_backend(BackendType::Hip, "HIP", config);
        case BackendType::Cuda:
            return init_device_backend(BackendType::Cuda, "CUDA", config);
        case BackendType::Vulkan:
            return init_device_backend(BackendType::Vulkan, "Vulkan", config);
        case BackendType::Metal:
            return init_device_backend(BackendType::Metal, "Metal", config);
        case BackendType::BestAvailable: {
            ggml_backend_t backend = ggml_backend_init_best();
            if (backend == nullptr) {
                throw std::runtime_error("Failed to initialize best backend");
            }
            return backend;
        }
        default:
            throw std::runtime_error("Unsupported backend type");
    }
}

void set_backend_threads(ggml_backend_t backend, int threads) {
    if (backend == nullptr) {
        return;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }
    // Use generic proc-address lookup for thread-setting (works with GGML_BACKEND_DL)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    void * fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn != nullptr) {
        ((ggml_backend_set_n_threads_t)fn)(backend, threads);
    }
}

bool is_host_backend(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    return device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

std::vector<ggml_backend_buffer_type_t> cpu_extra_buffer_types(ggml_backend_t backend) {
    std::vector<ggml_backend_buffer_type_t> out;
    if (!is_host_backend(backend)) {
        return out;
    }
    // Through the proc address, like the thread count (works with GGML_BACKEND_DL)
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    auto fn = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_dev_get_extra_bufts");
    if (fn == nullptr) {
        return out;
    }
    for (ggml_backend_buffer_type_t * buffer_type = fn(device); buffer_type != nullptr && *buffer_type != nullptr; ++buffer_type) {
        out.push_back(*buffer_type);
    }
    return out;
}

bool buffer_type_supports_matmul_weight(
    ggml_backend_t backend, ggml_backend_buffer_type_t buffer_type, ggml_type type, int64_t ne0, int64_t ne1) {
    ggml_backend_dev_t device = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    if (device == nullptr || buffer_type == nullptr || ne0 <= 0 || ne1 <= 0 || ne0 % ggml_blck_size(type) != 0) {
        return false;
    }
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize the weight buffer probe context");
    }
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), type, ne0, ne1);
    ggml_tensor * op = ggml_mul_mat(ctx.get(), weight, ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, 512));
    // supports_op looks at the buffer type of the weight's buffer, so an
    // empty buffer of the candidate type stands in for the real one.
    weight->buffer = ggml_backend_buft_alloc_buffer(buffer_type, 0);
    const bool supported = ggml_backend_dev_supports_op(device, op);
    ggml_backend_buffer_free(weight->buffer);
    weight->buffer = nullptr;
    return supported;
}

BackendType backend_type(ggml_backend_t backend) {
    if (is_host_backend(backend)) {
        return BackendType::Cpu;
    }
    if (is_hip_backend_handle(backend)) {
        return BackendType::Hip;
    }
    if (is_cuda_backend_handle(backend)) {
        return BackendType::Cuda;
    }
    if (is_vulkan_backend_handle(backend)) {
        return BackendType::Vulkan;
    }
    if (is_metal_backend_handle(backend)) {
        return BackendType::Metal;
    }
    return BackendType::BestAvailable;
}

bool uses_host_graph_plan(BackendType type) {
    return type == BackendType::Cpu;
}

bool uses_host_graph_plan(ggml_backend_t backend) {
    return is_host_backend(backend);
}

bool requested_backend_uses_host_graph_plan(const BackendConfig & config) {
    return uses_host_graph_plan(config.type);
}

static void cuda_clear_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    if (backend == nullptr || graph == nullptr) return;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) return;
    auto fn = (void (*)(ggml_backend_t, const ggml_cgraph *))
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),
            "ggml_backend_cuda_clear_graph");
    if (fn != nullptr) fn(backend, graph);
}

static void cuda_trim_pools(ggml_backend_t backend) {
    if (backend == nullptr) return;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) return;
    auto fn = (void (*)(ggml_backend_t))
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),
            "ggml_backend_cuda_trim_pools");
    if (fn != nullptr) fn(backend);
}

void trim_backend_pools(ggml_backend_t backend) {
    if (is_cuda_backend_handle(backend) || is_hip_backend_handle(backend)) cuda_trim_pools(backend);
}

void set_backend_stream_priority(ggml_backend_t backend, int priority) {
    if (backend == nullptr) return;
    if (!is_cuda_backend_handle(backend) && !is_hip_backend_handle(backend)) return;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) return;
    auto fn = (void (*)(ggml_backend_t, int))
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),
            "ggml_backend_cuda_set_stream_priority");
    if (fn != nullptr) fn(backend, priority);
}

void * backend_cuda_stream(ggml_backend_t backend) {
    if (backend == nullptr) return nullptr;
    if (!is_cuda_backend_handle(backend) && !is_hip_backend_handle(backend)) return nullptr;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) {
        throw std::runtime_error("CUDA backend stream lookup failed: backend has no device");
    }
    auto fn = (void * (*)(ggml_backend_t))
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),
            "ggml_backend_cuda_get_stream");
    if (fn == nullptr) {
        throw std::runtime_error("CUDA backend stream lookup failed: backend does not export ggml_backend_cuda_get_stream");
    }
    return fn(backend);
}

// evict_cuda_graph_cache defaults to false, preserving historical behavior
// for existing call sites: before the CUDA backend exported
// ggml_backend_cuda_clear_graph the lookup resolved nothing, and families
// that rebuild same-shape graphs between requests inherit a warm CUDA-graph
// cache from that. Families that prefer bounded memory over the warm
// carry-over opt in with true.
void release_backend_graph_resources(ggml_backend_t backend, ggml_cgraph * graph, bool evict_cuda_graph_cache) {
    if (!evict_cuda_graph_cache) return;  // existing callsite/behavior unchanged
    if (is_cuda_backend_handle(backend) || is_hip_backend_handle(backend)) cuda_clear_graph(backend, graph);
}

void release_backend_graph_resources(BackendType backend_type, ggml_backend_t backend, ggml_cgraph * graph, bool evict_cuda_graph_cache) {
    if (!evict_cuda_graph_cache) return;  // existing callsite/behavior unchanged
    if (backend_type == BackendType::Cuda || backend_type == BackendType::Hip) cuda_clear_graph(backend, graph);
}

void validate_backend_graph_supported(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    if (backend == nullptr || graph == nullptr) {
        throw std::runtime_error("Cannot validate backend graph support for null backend or graph");
    }
    const int nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node != nullptr && !ggml_backend_supports_op(backend, node)) {
            const char * graph_label = label != nullptr ? label : "graph";
            std::string op_name = ggml_op_name(node->op);
            if (node->op == GGML_OP_UNARY) {
                op_name += "/";
                op_name += ggml_unary_op_name(ggml_get_unary_op(node));
            }
            throw std::runtime_error(
                std::string(graph_label) +
                " contains unsupported backend op '" +
                op_name +
                "' at node " +
                std::to_string(i) +
                " tensor '" +
                (node->name[0] != '\0' ? node->name : "<unnamed>") +
                "'");
        }
    }
}

// Whether `tensor` is held in a buffer of one of `buffer_types`.
static bool held_in(const ggml_tensor * tensor, const std::vector<ggml_backend_buffer_type_t> & buffer_types) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }
    const ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(tensor->buffer);
    return std::find(buffer_types.begin(), buffer_types.end(), buffer_type) != buffer_types.end();
}

static std::string tensor_name_of(const ggml_tensor * tensor) {
    return tensor->name[0] != '\0' ? tensor->name : "<unnamed>";
}

void validate_backend_graph_with_cpu_extra_buffers(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    if (backend == nullptr || graph == nullptr) {
        throw std::runtime_error("Cannot validate backend graph support for null backend or graph");
    }
    // A weight in a CPU extra buffer is laid out for that buffer type's own
    // matmul kernels, which read it whole. The CPU backend accepts views, and
    // a matmul on a view before the view has a buffer, so a view of such a
    // weight would read misplaced rows without failing. CPU_REPACK's kernels
    // also read the input's rows as if each followed the last, whatever its
    // strides, and its supports_op does not check them. These checks come
    // before validate_backend_graph_supported, so they name the weight even
    // where the buffer type's own supports_op (AMX's) turns such a matmul
    // down.
    const auto extra_buffer_types = cpu_extra_buffer_types(backend);
    const int nodes = extra_buffer_types.empty() ? 0 : ggml_graph_n_nodes(graph);
    for (int i = 0; i < nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            continue;
        }
        const auto misuse = [&](const ggml_tensor * weight, const std::string & what) {
            throw std::runtime_error(
                std::string(label != nullptr ? label : "graph") + " " + what + " weight '" + tensor_name_of(weight) + "', held in a " +
                ggml_backend_buffer_name(weight->buffer) + " buffer, at node " + std::to_string(i) + " tensor '" + tensor_name_of(node) +
                "'");
        };
        // The node itself, or a source that is a view but no node (ggml_view_tensor).
        for (int src = -1; src < GGML_MAX_SRC; ++src) {
            const ggml_tensor * tensor = src < 0 ? node : node->src[src];
            if (tensor != nullptr && held_in(tensor->view_src, extra_buffer_types)) {
                misuse(tensor->view_src, "views");
            }
        }
        if (node->op == GGML_OP_MUL_MAT && held_in(node->src[0], extra_buffer_types) && !ggml_is_contiguous(node->src[1])) {
            misuse(node->src[0], "multiplies a non-contiguous input by");
        }
    }
    validate_backend_graph_supported(backend, graph, label);
}

BackendMemorySnapshot query_backend_memory(ggml_backend_t backend, int device_hint) {
    BackendMemorySnapshot snapshot;
    if (backend == nullptr) {
        return snapshot;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device != nullptr) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        if (total_bytes > 0 && free_bytes <= total_bytes) {
            snapshot.available = true;
            snapshot.total_bytes = static_cast<int64_t>(total_bytes);
            snapshot.free_bytes = static_cast<int64_t>(free_bytes);
            snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
        }
    }
    (void)device_hint;
    return snapshot;
}

BackendMemorySnapshot query_backend_memory(const BackendConfig & config) {
    BackendMemorySnapshot snapshot;
    switch (config.type) {
        case BackendType::Cpu:
            return snapshot;
        case BackendType::BestAvailable:
            return snapshot;
        default:
            break;
    }
    ggml_backend_dev_t device = find_device_by_backend_type(config.type, config.device);
    if (device != nullptr) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        if (total_bytes > 0 && free_bytes <= total_bytes) {
            snapshot.available = true;
            snapshot.total_bytes = static_cast<int64_t>(total_bytes);
            snapshot.free_bytes = static_cast<int64_t>(free_bytes);
            snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
        }
    }
    return snapshot;
}

ggml_backend_graph_plan_t create_backend_graph_plan_if_host(ggml_backend_t backend, ggml_cgraph * graph) {
    if (backend == nullptr || graph == nullptr || !is_host_backend(backend)) {
        return nullptr;
    }
    return ggml_backend_graph_plan_create(backend, graph);
}

void free_backend_graph_plan(ggml_backend_t backend, ggml_backend_graph_plan_t & plan) {
    if (plan != nullptr) {
        ggml_backend_graph_plan_free(backend, plan);
        plan = nullptr;
    }
}

ggml_status compute_backend_graph(
    ggml_backend_t backend,
    ggml_cgraph * graph,
    ggml_backend_graph_plan_t plan,
    const char * label) {
    if (backend == nullptr || graph == nullptr) {
        return GGML_STATUS_FAILED;
    }
#ifndef NDEBUG
    if (plan == nullptr && backend_graph_validation_enabled()) {
        validate_backend_graph_supported(backend, graph, label);
    } else {
        (void)label;
    }
#else
    (void)label;
#endif
    return plan != nullptr
        ? ggml_backend_graph_plan_compute(backend, plan)
        : ggml_backend_graph_compute(backend, graph);
}

void prepare_host_graph_plan(const ExecutionContext & execution_context, ggml_cgraph * graph, HostGraphPlan & plan) {
    plan.reset();
    if (!execution_context.uses_host_graph_plan()) {
        return;
    }
    ggml_backend_t backend = execution_context.backend();
    if (backend == nullptr) {
        return;
    }
    ggml_backend_graph_plan_t new_plan = ggml_backend_graph_plan_create(backend, graph);
    if (new_plan != nullptr) {
        plan.plan = new_plan;
        plan.backend = backend;
    }
}

ggml_status compute_graph(
    const ExecutionContext & execution_context,
    ggml_cgraph * graph,
    HostGraphPlan & plan,
    const char * label) {
    if (plan.active()) {
        return ggml_backend_graph_plan_compute(plan.backend, plan.plan);
    }
    return compute_backend_graph(execution_context.backend(), graph, nullptr, label);
}

void write_tensor_f32(const TensorValue & tensor, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_F32) {
        throw std::runtime_error("write_tensor_f32 requires GGML_TYPE_F32 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error(
            "write_tensor_f32 value count does not match tensor shape for tensor '" +
            std::string(tensor.tensor != nullptr ? tensor.tensor->name : "<null>") +
            "': expected " + std::to_string(tensor.shape.num_elements()) +
            ", got " + std::to_string(count));
    }
    ggml_backend_tensor_set(tensor.tensor, values, 0, count * sizeof(float));
}

void write_tensor_f32_slice(const TensorValue & tensor, size_t element_offset, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_F32) {
        throw std::runtime_error("write_tensor_f32_slice requires GGML_TYPE_F32 tensor");
    }
    const size_t total = static_cast<size_t>(tensor.shape.num_elements());
    if (element_offset > total || count > total - element_offset) {
        throw std::runtime_error("write_tensor_f32_slice range exceeds tensor shape");
    }
    ggml_backend_tensor_set(
        tensor.tensor,
        values,
        element_offset * sizeof(float),
        count * sizeof(float));
}

void write_tensor_f32(const TensorValue & tensor, const std::vector<float> & values) {
    write_tensor_f32(tensor, values.data(), values.size());
}

void write_tensor_f16(const TensorValue & tensor, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_F16) {
        throw std::runtime_error("write_tensor_f16 requires GGML_TYPE_F16 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error(
            "write_tensor_f16 value count does not match tensor shape for tensor '" +
            std::string(tensor.tensor != nullptr ? tensor.tensor->name : "<null>") +
            "': expected " + std::to_string(tensor.shape.num_elements()) +
            ", got " + std::to_string(count));
    }
    std::vector<ggml_fp16_t> fp16_values(count);
    ggml_fp32_to_fp16_row(values, fp16_values.data(), static_cast<int64_t>(count));
    ggml_backend_tensor_set(tensor.tensor, fp16_values.data(), 0, count * sizeof(ggml_fp16_t));
}

void write_tensor_f16(const TensorValue & tensor, const std::vector<float> & values) {
    write_tensor_f16(tensor, values.data(), values.size());
}

void write_tensor_bf16(const TensorValue & tensor, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_BF16) {
        throw std::runtime_error("write_tensor_bf16 requires GGML_TYPE_BF16 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error(
            "write_tensor_bf16 value count does not match tensor shape for tensor '" +
            std::string(tensor.tensor != nullptr ? tensor.tensor->name : "<null>") +
            "': expected " + std::to_string(tensor.shape.num_elements()) +
            ", got " + std::to_string(count));
    }
    std::vector<ggml_bf16_t> bf16_values(count);
    ggml_fp32_to_bf16_row(values, bf16_values.data(), static_cast<int64_t>(count));
    ggml_backend_tensor_set(tensor.tensor, bf16_values.data(), 0, count * sizeof(ggml_bf16_t));
}

void write_tensor_bf16(const TensorValue & tensor, const std::vector<float> & values) {
    write_tensor_bf16(tensor, values.data(), values.size());
}

void write_tensor_float(const TensorValue & tensor, const float * values, size_t count) {
    switch (tensor.type) {
        case GGML_TYPE_F32:
            write_tensor_f32(tensor, values, count);
            return;
        case GGML_TYPE_F16:
            write_tensor_f16(tensor, values, count);
            return;
        case GGML_TYPE_BF16:
            write_tensor_bf16(tensor, values, count);
            return;
        default:
            throw std::runtime_error("write_tensor_float supports only f32/f16/bf16 tensors");
    }
}

void write_tensor_float(const TensorValue & tensor, const std::vector<float> & values) {
    write_tensor_float(tensor, values.data(), values.size());
}

void write_tensor_bytes(const TensorValue & tensor, const std::vector<std::byte> & bytes) {
    if (tensor.tensor == nullptr) {
        throw std::runtime_error("write_tensor_bytes requires non-null tensor");
    }
    const size_t expected = static_cast<size_t>(ggml_nbytes(tensor.tensor));
    if (bytes.size() != expected) {
        throw std::runtime_error(
            "write_tensor_bytes byte count does not match tensor '" +
            std::string(tensor.tensor->name) +
            "': expected " + std::to_string(expected) +
            ", got " + std::to_string(bytes.size()));
    }
    ggml_backend_tensor_set(tensor.tensor, bytes.data(), 0, bytes.size());
}

void round_f32_to_bf16_in_place(float * values, size_t count) {
    if (count == 0) {
        return;
    }
    std::vector<ggml_bf16_t> bf16_values(count);
    ggml_fp32_to_bf16_row(values, bf16_values.data(), static_cast<int64_t>(count));
    ggml_bf16_to_fp32_row(bf16_values.data(), values, static_cast<int64_t>(count));
}

void round_f32_to_bf16_in_place(std::vector<float> & values) {
    round_f32_to_bf16_in_place(values.data(), values.size());
}

void write_tensor_i32(const TensorValue & tensor, const int32_t * values, size_t count) {
    if (tensor.type != GGML_TYPE_I32) {
        throw std::runtime_error("write_tensor_i32 requires GGML_TYPE_I32 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error("write_tensor_i32 value count does not match tensor shape");
    }
    ggml_backend_tensor_set(tensor.tensor, values, 0, count * sizeof(int32_t));
}

void write_tensor_i32(const TensorValue & tensor, const std::vector<int32_t> & values) {
    write_tensor_i32(tensor, values.data(), values.size());
}

template <typename T>
void read_tensor_typed_into(const ggml_tensor * tensor, ggml_type expected_type, std::vector<T> & values) {
    if (tensor->type != expected_type) {
        throw std::runtime_error("read_tensor_typed type mismatch");
    }
    const size_t element_count = static_cast<size_t>(ggml_nelements(tensor));
    values.resize(element_count);
    const size_t byte_count = static_cast<size_t>(ggml_nbytes(tensor));
    if (ggml_is_contiguous(tensor) && tensor->nb[0] == sizeof(T)) {
        ggml_backend_tensor_get(tensor, values.data(), 0, byte_count);
        return;
    }

    std::vector<uint8_t> raw(byte_count);
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    size_t out_index = 0;
    const char * base = reinterpret_cast<const char *>(raw.data());
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const char * ptr = base
                        + i0 * tensor->nb[0]
                        + i1 * tensor->nb[1]
                        + i2 * tensor->nb[2]
                        + i3 * tensor->nb[3];
                    std::memcpy(&values[out_index++], ptr, sizeof(T));
                }
            }
        }
    }
}

template <typename T>
std::vector<T> read_tensor_typed(const ggml_tensor * tensor, ggml_type expected_type) {
    std::vector<T> values;
    read_tensor_typed_into(tensor, expected_type, values);
    return values;
}

void read_tensor_f32_into(const ggml_tensor * tensor, std::vector<float> & values) {
    read_tensor_typed_into<float>(tensor, GGML_TYPE_F32, values);
}
std::vector<float> read_tensor_f32(const ggml_tensor * tensor) {
    return read_tensor_typed<float>(tensor, GGML_TYPE_F32);
}

void read_tensor_f16_into(const ggml_tensor * tensor, std::vector<float> & values) {
    const auto fp16_values = read_tensor_typed<ggml_fp16_t>(tensor, GGML_TYPE_F16);
    values.resize(fp16_values.size());
    ggml_fp16_to_fp32_row(fp16_values.data(), values.data(), static_cast<int64_t>(values.size()));
}

std::vector<float> read_tensor_f16(const ggml_tensor * tensor) {
    std::vector<float> values;
    read_tensor_f16_into(tensor, values);
    return values;
}

void read_tensor_bf16_into(const ggml_tensor * tensor, std::vector<float> & values) {
    const auto bf16_values = read_tensor_typed<ggml_bf16_t>(tensor, GGML_TYPE_BF16);
    values.resize(bf16_values.size());
    ggml_bf16_to_fp32_row(bf16_values.data(), values.data(), static_cast<int64_t>(values.size()));
}

std::vector<float> read_tensor_bf16(const ggml_tensor * tensor) {
    std::vector<float> values;
    read_tensor_bf16_into(tensor, values);
    return values;
}

void read_tensor_float_into(const ggml_tensor * tensor, std::vector<float> & values) {
    switch (tensor->type) {
        case GGML_TYPE_F32:
            read_tensor_f32_into(tensor, values);
            return;
        case GGML_TYPE_F16:
            read_tensor_f16_into(tensor, values);
            return;
        case GGML_TYPE_BF16:
            read_tensor_bf16_into(tensor, values);
            return;
        default:
            throw std::runtime_error("read_tensor_float supports only f32/f16/bf16 tensors");
    }
}

std::vector<float> read_tensor_float(const ggml_tensor * tensor) {
    std::vector<float> values;
    read_tensor_float_into(tensor, values);
    return values;
}

void read_tensor_bytes_into(const ggml_tensor * tensor, std::vector<std::byte> & bytes) {
    if (tensor == nullptr) {
        throw std::runtime_error("read_tensor_bytes requires non-null tensor");
    }
    const size_t byte_count = static_cast<size_t>(ggml_nbytes(tensor));
    bytes.resize(byte_count);
    ggml_backend_tensor_get(tensor, bytes.data(), 0, byte_count);
}

std::vector<std::byte> read_tensor_bytes(const ggml_tensor * tensor) {
    std::vector<std::byte> bytes;
    read_tensor_bytes_into(tensor, bytes);
    return bytes;
}

void read_tensor_i32_into(const ggml_tensor * tensor, std::vector<int32_t> & values) {
    read_tensor_typed_into<int32_t>(tensor, GGML_TYPE_I32, values);
}
std::vector<int32_t> read_tensor_i32(const ggml_tensor * tensor) {
    return read_tensor_typed<int32_t>(tensor, GGML_TYPE_I32);
}

}  // namespace engine::core
