#ifndef __GGML_EXTEND_BACKEND_HPP__
#define __GGML_EXTEND_BACKEND_HPP__

#include <cstring>
#include <mutex>

#include "ggml-backend.h"
#include "ggml.h"

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

inline void ggml_backend_load_all_once() {
    // If the registry already has devices and the CPU backend is present,
    // assume either static registration or explicit host-side preloading has
    // completed and avoid rescanning the default paths.
    if (ggml_backend_dev_count() > 0 && ggml_backend_reg_by_name("CPU") != nullptr) {
        return;
    }
    // In dynamic-backend mode the backend modules are discovered at runtime,
    // so we must load them before asking for the CPU backend or its proc table.
    // If the host preloaded only a subset of backends, allow one default-path
    // scan so missing modules can still be discovered.
    static std::once_flag once;
    std::call_once(once, []() {
        if (ggml_backend_dev_count() > 0 && ggml_backend_reg_by_name("CPU") != nullptr) {
            return;
        }
        ggml_backend_load_all();
    });
}

// Do not gate this branch on GGML_CPU or GGML_CPU_ALL_VARIANTS:
// those are CMake options used to configure ggml itself, but they are not
// exported as PUBLIC compile definitions to stable-diffusion in backend-DL mode.
// In practice, this target can reliably see GGML_BACKEND_DL, but not whether
// the CPU backend was compiled as a loadable module. We therefore use runtime
// backend discovery instead of compile-time assumptions.

__STATIC_INLINE__ ggml_backend_reg_t ggml_backend_cpu_reg() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (reg != nullptr) {
        return reg;
    }

    ggml_backend_load_all_once();
    return ggml_backend_reg_by_name("CPU");
}

__STATIC_INLINE__ ggml_backend_reg_t ggml_backend_reg_from_backend(ggml_backend_t backend) {
    if (backend != nullptr) {
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (device != nullptr) {
            return ggml_backend_dev_backend_reg(device);
        }
    }

    return ggml_backend_cpu_reg();
}

__STATIC_INLINE__ ggml_backend_t ggml_backend_cpu_init() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend != nullptr) {
        return backend;
    }

    ggml_backend_load_all_once();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

__STATIC_INLINE__ bool ggml_backend_is_cpu(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device != nullptr) {
        return ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
    }

    const char* backend_name = ggml_backend_name(backend);
    return backend_name != nullptr && std::strcmp(backend_name, "CPU") == 0;
}

__STATIC_INLINE__ void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_reg_from_backend(backend_cpu);
    if (reg == nullptr) {
        return;
    }

    auto fn = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (fn != nullptr) {
        fn(backend_cpu, n_threads);
    }
}

using __ggml_backend_cpu_set_threadpool_t = void (*)(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);

__STATIC_INLINE__ void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    ggml_backend_reg_t reg = ggml_backend_reg_from_backend(backend_cpu);
    if (reg == nullptr) {
        return;
    }

    auto fn = reinterpret_cast<__ggml_backend_cpu_set_threadpool_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
    if (fn != nullptr) {
        fn(backend_cpu, threadpool);
    }
}

__STATIC_INLINE__ void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void* abort_callback_data) {
    ggml_backend_reg_t reg = ggml_backend_reg_from_backend(backend_cpu);
    if (reg == nullptr) {
        return;
    }

    auto fn = reinterpret_cast<ggml_backend_set_abort_callback_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback"));
    if (fn != nullptr) {
        fn(backend_cpu, abort_callback, abort_callback_data);
    }
}

__STATIC_INLINE__ ggml_backend_buffer_t ggml_backend_tensor_buffer(const struct ggml_tensor* tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }

    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

__STATIC_INLINE__ bool ggml_backend_tensor_is_host_accessible(const struct ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_tensor_buffer(tensor);
    return buffer == nullptr || ggml_backend_buffer_is_host(buffer);
}

__STATIC_INLINE__ size_t ggml_backend_tensor_offset(const struct ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return (size_t)(i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
}

template <typename T>
__STATIC_INLINE__ void ggml_backend_tensor_write_scalar(const struct ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3, T value) {
    const size_t offset = ggml_backend_tensor_offset(tensor, i0, i1, i2, i3);

    if (ggml_backend_tensor_is_host_accessible(tensor)) {
        auto* dst = reinterpret_cast<T*>(reinterpret_cast<char*>(tensor->data) + offset);
        *dst      = value;
        return;
    }

    ggml_backend_tensor_set(const_cast<struct ggml_tensor*>(tensor), &value, offset, sizeof(T));
}

__STATIC_INLINE__ void ggml_set_f32_nd(const struct ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float value) {
    switch (tensor->type) {
        case GGML_TYPE_I8:
            ggml_backend_tensor_write_scalar(tensor, i0, i1, i2, i3, static_cast<int8_t>(value));
            break;
        case GGML_TYPE_I16:
            ggml_backend_tensor_write_scalar(tensor, i0, i1, i2, i3, static_cast<int16_t>(value));
            break;
        case GGML_TYPE_I32:
            ggml_backend_tensor_write_scalar(tensor, i0, i1, i2, i3, static_cast<int32_t>(value));
            break;
        case GGML_TYPE_F16:
            ggml_backend_tensor_write_scalar(tensor, i0, i1, i2, i3, ggml_fp32_to_fp16(value));
            break;
        case GGML_TYPE_BF16:
            ggml_backend_tensor_write_scalar(tensor, i0, i1, i2, i3, ggml_fp32_to_bf16(value));
            break;
        case GGML_TYPE_F32:
            ggml_backend_tensor_write_scalar(tensor, i0, i1, i2, i3, value);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

__STATIC_INLINE__ void ggml_set_f32_1d(const struct ggml_tensor* tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = {0, 0, 0, 0};
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }

    switch (tensor->type) {
        case GGML_TYPE_I8:
            ggml_backend_tensor_write_scalar(tensor, i, 0, 0, 0, static_cast<int8_t>(value));
            break;
        case GGML_TYPE_I16:
            ggml_backend_tensor_write_scalar(tensor, i, 0, 0, 0, static_cast<int16_t>(value));
            break;
        case GGML_TYPE_I32:
            ggml_backend_tensor_write_scalar(tensor, i, 0, 0, 0, static_cast<int32_t>(value));
            break;
        case GGML_TYPE_F16:
            ggml_backend_tensor_write_scalar(tensor, i, 0, 0, 0, ggml_fp32_to_fp16(value));
            break;
        case GGML_TYPE_BF16:
            ggml_backend_tensor_write_scalar(tensor, i, 0, 0, 0, ggml_fp32_to_bf16(value));
            break;
        case GGML_TYPE_F32:
            ggml_backend_tensor_write_scalar(tensor, i, 0, 0, 0, value);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

__STATIC_INLINE__ enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context* ctx, struct ggml_cgraph* cgraph, int n_threads) {
    (void)ctx;

    // The legacy ggml_graph_compute_with_ctx() symbol lives in ggml-cpu, but
    // the backend proc table does not expose it in GGML_BACKEND_DL mode.
    // Recreate the old behavior by initializing the CPU backend explicitly and
    // executing the graph through the generic backend API.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return GGML_STATUS_ALLOC_FAILED;
    }

    ggml_backend_cpu_set_n_threads(backend, n_threads);

    const enum ggml_status status = ggml_backend_graph_compute(backend, cgraph);
    ggml_backend_free(backend);

    return status;
}

__STATIC_INLINE__ ggml_tensor* ggml_set_f32(struct ggml_tensor* tensor, float value) {
    GGML_ASSERT(tensor != nullptr);

    if (ggml_backend_tensor_is_host_accessible(tensor) && ggml_is_contiguous(tensor)) {
        const int64_t nelements = ggml_nelements(tensor);

        switch (tensor->type) {
            case GGML_TYPE_I8: {
                auto* data     = reinterpret_cast<int8_t*>(tensor->data);
                const int8_t v = static_cast<int8_t>(value);
                for (int64_t i = 0; i < nelements; ++i) {
                    data[i] = v;
                }
            } break;
            case GGML_TYPE_I16: {
                auto* data      = reinterpret_cast<int16_t*>(tensor->data);
                const int16_t v = static_cast<int16_t>(value);
                for (int64_t i = 0; i < nelements; ++i) {
                    data[i] = v;
                }
            } break;
            case GGML_TYPE_I32: {
                auto* data      = reinterpret_cast<int32_t*>(tensor->data);
                const int32_t v = static_cast<int32_t>(value);
                for (int64_t i = 0; i < nelements; ++i) {
                    data[i] = v;
                }
            } break;
            case GGML_TYPE_F16: {
                auto* data          = reinterpret_cast<ggml_fp16_t*>(tensor->data);
                const ggml_fp16_t v = ggml_fp32_to_fp16(value);
                for (int64_t i = 0; i < nelements; ++i) {
                    data[i] = v;
                }
            } break;
            case GGML_TYPE_BF16: {
                auto* data          = reinterpret_cast<ggml_bf16_t*>(tensor->data);
                const ggml_bf16_t v = ggml_fp32_to_bf16(value);
                for (int64_t i = 0; i < nelements; ++i) {
                    data[i] = v;
                }
            } break;
            case GGML_TYPE_F32: {
                auto* data = reinterpret_cast<float*>(tensor->data);
                for (int64_t i = 0; i < nelements; ++i) {
                    data[i] = value;
                }
            } break;
            default:
                GGML_ABORT("fatal error");
        }

        return tensor;
    }

    const int64_t nelements = ggml_nelements(tensor);
    for (int64_t i = 0; i < nelements; ++i) {
        ggml_set_f32_1d(tensor, static_cast<int>(i), value);
    }

    return tensor;
}

#endif
