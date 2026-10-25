#include "engine/framework/core/attention_fallback.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "ggml.h"
#include "ggml-backend.h"

// ggml-hip publicly defines GGML_USE_CUDA for its consumers (hipified CUDA
// sources), so GGML_USE_CUDA alone does not imply a real CUDA toolchain with
// the driver library linked. ENGINE_GGML_HIP_BACKEND marks the HIP case.
#if defined(GGML_USE_CUDA) && !defined(ENGINE_GGML_HIP_BACKEND)
#define AUDIOCPP_CUDA_DRIVER_PROBE 1
// CUDA driver API, declared manually so this translation unit needs neither
// the CUDA headers on its include path nor any CMake changes. The driver
// library is already linked transitively through ggml-cuda. Attribute ids
// CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=75 / MINOR=76 are stable ABI.
extern "C" {
typedef int kCcProbeCuDevice;
typedef int kCcProbeCuResult;
kCcProbeCuResult cuDeviceGet(kCcProbeCuDevice * device, int ordinal);
kCcProbeCuResult cuDeviceGetAttribute(int * value, int attrib, kCcProbeCuDevice device);
}
#endif

namespace engine::core {
namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

AttentionPreference parse_preference_value(const std::string & value, const char * option_name) {
    const std::string lowered = to_lower(value);
    if (lowered == "auto") {
        return AttentionPreference::Auto;
    }
    if (lowered == "flash" || lowered == "on" || lowered == "1") {
        return AttentionPreference::Flash;
    }
    if (lowered == "eager" || lowered == "off" || lowered == "0") {
        return AttentionPreference::Eager;
    }
    throw std::runtime_error(
        std::string(option_name) + " must be 'auto', 'flash', or 'eager' (got '" + value + "')");
}

// Auto-resolution for the CUDA flash-attention path.
//
// ggml_backend_supports_op() cannot be used here: on Volta it returns true
// (the MMA kernel is "selected") yet large prefill shapes die at launch with
// "flash_attn_ext_f16 has no device code compatible with CUDA arch 700".
// Instead, gate on compute capability, mirroring the kernel guards in
// ggml-cuda: flash below 700 (only generic TILE/VEC kernels exist) and at or
// above 800 (MMA fully instantiated); eager on 700-800, where large shapes
// select the MMA kernel with no usable device code. Unknown backends and
// query failures fail OPEN to preserve current behavior.
bool cuda_device_wants_eager(ggml_backend_t backend) {
#ifdef AUDIOCPP_CUDA_DRIVER_PROBE
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) {
        return false;
    }
    if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const char * name = ggml_backend_dev_name(device);
    if (name == nullptr || std::strncmp(name, "CUDA", 4) != 0) {
        return false;  // HIP / Vulkan / Metal / CPU: unchanged behavior.
    }
    char * end = nullptr;
    const long ordinal = std::strtol(name + 4, &end, 10);
    if (end == name + 4 || ordinal < 0) {
        return false;
    }
    kCcProbeCuDevice cu_device = -1;
    if (cuDeviceGet(&cu_device, static_cast<int>(ordinal)) != 0) {
        return false;
    }
    int major = 0;
    int minor = 0;
    if (cuDeviceGetAttribute(&major, 75 /* COMPUTE_CAPABILITY_MAJOR */, cu_device) != 0) {
        return false;
    }
    if (cuDeviceGetAttribute(&minor, 76 /* COMPUTE_CAPABILITY_MINOR */, cu_device) != 0) {
        return false;
    }
    const int cc = major * 100 + minor * 10;
    return cc >= 700 && cc < 800;
#else
    (void) backend;
    return false;
#endif  // AUDIOCPP_CUDA_DRIVER_PROBE
}

}  // namespace

AttentionPreference parse_attention_preference(const std::string & value, const char * option_name) {
    return parse_preference_value(value, option_name != nullptr ? option_name : "attention");
}

bool resolve_flash_attention(ggml_backend_t backend, int64_t head_dim, AttentionPreference preference) {
    (void) head_dim;
    switch (preference) {
        case AttentionPreference::Flash:
            return true;
        case AttentionPreference::Eager:
            return false;
        case AttentionPreference::Auto:
            break;
    }
    return !cuda_device_wants_eager(backend);
}

// The device description is the only vendor signal ggml-backend exposes for
// Vulkan ("Intel(R) Graphics (BMG G31)", "Intel(R) Arc(tm) A770 Graphics", ...).
// Any null handle, non-Vulkan device, or missing description answers false.
bool vulkan_device_is_intel(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) {
        return false;
    }
    if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const char * name = ggml_backend_dev_name(device);
    if (name == nullptr || std::strncmp(name, "Vulkan", 6) != 0) {
        return false;
    }
    const char * description = ggml_backend_dev_description(device);
    if (description == nullptr) {
        return false;
    }
    return to_lower(description).find("intel") != std::string::npos;
}

}  // namespace engine::core
