#include "kcpp_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#  include "ggml_v2-cuda.h"
#  include "ggml_v3-cuda.h"
#endif

#if defined(GGML_USE_HIP)
// for rocblas_initialize()
#  include "rocblas/rocblas.h"
#endif

#if defined(GGML_USE_METAL)
#  include "ggml-metal.h"
#endif

static std::string to_lowercase(const char* ptr) {
    std::string s = (ptr == nullptr) ? "" : ptr;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

static bool has_any_prefix(const std::string& str, const std::string& prefixes, char delimiter) {
    size_t start = 0;
    size_t end = prefixes.find(delimiter);
    while (start != std::string::npos) {
        std::string prefix = prefixes.substr(start, end - start);
        if (str.rfind(prefix, 0) == 0) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
        end = prefixes.find(delimiter, start);
    }
    return false;
}

static ggml_backend_dev_t get_ggml_main_device(void)
{
    // similar to ggml_backend_init_best
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return dev;
}

ggml_backend_dev_t kcpp_backend_get_device(int index)
{
    if (index < 0) {
        if (index == -1) {
            return get_ggml_main_device();
        } else {
           return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    } else {
        return ggml_backend_dev_get((size_t)index);
    }
}

// this is similar to sd_backend_is, except:
// - if no backend is provided, checks the first ggml device (should be equivalent to a compile-time check)
// - tests a |-separated list of device names
int kcpp_backend_check(const char* name, ggml_backend_t backend)
{
    std::string loname = to_lowercase(name);
    if (loname == "") {
        return false;
    }

    ggml_backend_dev_t dev = nullptr;
    if (backend == nullptr) {
        // note we are assuming there is only one gpu backend type

        const char* KCPP_BACKEND_DYNAMIC = std::getenv("KCPP_BACKEND_DYNAMIC");
        if (KCPP_BACKEND_DYNAMIC == nullptr || std::string(KCPP_BACKEND_DYNAMIC) == "0") {

            #if defined(GGML_USE_VULKAN)
            const char * devname = "vulkan";

            #elif defined(GGML_USE_METAL)
            const char * devname = "mtl";

            #elif defined(GGML_USE_HIP)
            const char * devname = "rocm";

            #elif defined(GGML_USE_CUDA)
            const char * devname = "cuda";

            #elif defined(GGML_USE_SYCL)
            const char * devname = "sycl";

            #elif defined(GGML_USE_BLAS)
            const char * devname = "blas";

            #else
            const char * devname = nullptr;
            #endif

            if (devname != nullptr) {
                return has_any_prefix(devname, name, '|');
            }
        }

        dev = get_ggml_main_device();
    } else {
        dev = ggml_backend_get_device(backend);
    }
    if (!dev) {
        return false;
    }
    std::string lo_dev_name = to_lowercase(ggml_backend_dev_name(dev));
    return has_any_prefix(lo_dev_name, name, '|');
}

bool kcpp_backend_metal_supports_family(ggml_backend_t backend, int family)
{
    #if defined(GGML_USE_METAL)
    return ggml_backend_metal_supports_family(backend, family);
    #else
    return false;
    #endif
}

void kcpp_backend_cuda_ggmlv2_transform_tensor(ggml_v2_tensor * tensor)
{
    #if defined(GGML_USE_CUDA)
    ggml_v2_cuda_transform_tensor(tensor);
    #endif
}

void kcpp_backend_cuda_ggmlv3_set_main_device(int device)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_set_main_device(device);
    #endif
}

void kcpp_backend_cuda_ggmlv3_set_tensor_split(const float * tensor_split)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_set_tensor_split(tensor_split);
    #endif
}

void kcpp_backend_cuda_ggmlv3_transform_tensor(void * data, struct ggml_v3_tensor * tensor)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_transform_tensor(tensor->data, tensor);
    #endif
}

void kcpp_backend_cuda_ggmlv3_free_data(struct ggml_v3_tensor * tensor)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_free_data(tensor);
    #endif
}

void kcpp_backend_cuda_ggmlv3_free_scratch(void)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_free_scratch();
    #endif
}

void kcpp_backend_cuda_ggmlv3_assign_buffers(struct ggml_v3_tensor * tensor)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_assign_buffers(tensor);
    #endif
}

void kcpp_backend_cuda_ggmlv3_assign_buffers_force_inplace(struct ggml_v3_tensor * tensor)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_assign_buffers_force_inplace(tensor);
    #endif
}

void kcpp_backend_cuda_ggmlv3_assign_buffers_no_scratch(struct ggml_v3_tensor * tensor)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_assign_buffers_no_scratch(tensor);
    #endif
}

void kcpp_backend_cuda_ggmlv3_set_mul_mat_q(bool mul_mat_q)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_set_mul_mat_q(mul_mat_q);
    #endif
}

void kcpp_backend_cuda_ggmlv3_set_scratch_size(size_t scratch_size)
{
    #if defined(GGML_USE_CUDA)
    ggml_v3_cuda_set_scratch_size(scratch_size);
    #endif
}

void kcpp_backend_cuda_set_mul_mat_q(int use_mmq)
{
    #if defined(GGML_USE_CUDA)
    ggml_cuda_set_mul_mat_q(use_mmq);
    #endif
}

void kcpp_backend_hip_initialize()
{
    #if defined(GGML_USE_HIP)
    rocblas_initialize();
    #endif
}

