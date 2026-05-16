#include "ggml_extend_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util.h"

static std::string trim_copy(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static std::vector<std::string> split_copy(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(value);
    while (std::getline(stream, part, delimiter)) {
        parts.push_back(part);
    }
    return parts;
}

static bool is_default_backend_token(const std::string& name) {
    const std::string lower = lower_copy(trim_copy(name));
    return lower.empty() || lower == "default" || lower == "auto";
}

static bool parse_backend_module(const std::string& raw_name, SDBackendModule* module) {
    std::string name = lower_copy(trim_copy(raw_name));
    name.erase(std::remove(name.begin(), name.end(), '-'), name.end());
    name.erase(std::remove(name.begin(), name.end(), '_'), name.end());

    if (name == "diffusion" || name == "model" || name == "unet" || name == "dit") {
        *module = SDBackendModule::DIFFUSION;
        return true;
    }
    if (name == "te" || name == "clip" || name == "text" || name == "textencoder" || name == "textencoders" || name == "conditioner" || name == "cond" || name == "llm" || name == "t5" || name == "t5xxl") {
        *module = SDBackendModule::TE;
        return true;
    }
    if (name == "clipvision" || name == "vision") {
        *module = SDBackendModule::CLIP_VISION;
        return true;
    }
    if (name == "vae" || name == "firststage" || name == "autoencoder" || name == "tae") {
        *module = SDBackendModule::VAE;
        return true;
    }
    if (name == "controlnet" || name == "control") {
        *module = SDBackendModule::CONTROL_NET;
        return true;
    }
    if (name == "photomaker" || name == "photomakerid" || name == "pmid" || name == "photo") {
        *module = SDBackendModule::PHOTOMAKER;
        return true;
    }
    if (name == "upscaler" || name == "esrgan" || name == "hires") {
        *module = SDBackendModule::UPSCALER;
        return true;
    }
    return false;
}

static std::string module_assignment_name(const SDBackendAssignment& assignment, SDBackendModule module) {
    auto it = assignment.module_names.find(module);
    if (it != assignment.module_names.end()) {
        return it->second;
    }
    return assignment.default_name;
}

static std::string backend_cache_key(ggml_backend_t backend) {
    if (backend == nullptr) {
        return "";
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev != nullptr) {
        return lower_copy(ggml_backend_dev_name(dev));
    }
    const char* backend_name = ggml_backend_name(backend);
    return backend_name != nullptr ? lower_copy(backend_name) : "";
}

static std::string resolve_first_device_by_type(enum ggml_backend_dev_type type) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (dev == nullptr) {
        return "";
    }
    return ggml_backend_dev_name(dev);
}

static ggml_backend_buffer_t ggml_backend_tensor_buffer(const struct ggml_tensor* tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }

    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

static bool ggml_backend_tensor_is_host_accessible(const struct ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_tensor_buffer(tensor);
    return buffer == nullptr || ggml_backend_buffer_is_host(buffer);
}

static size_t ggml_backend_tensor_offset(const struct ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return static_cast<size_t>(i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
}

template <typename T>
static void ggml_backend_tensor_write_scalar(const struct ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3, T value) {
    const size_t offset = ggml_backend_tensor_offset(tensor, i0, i1, i2, i3);

    if (ggml_backend_tensor_is_host_accessible(tensor)) {
        auto* dst = reinterpret_cast<T*>(reinterpret_cast<char*>(tensor->data) + offset);
        *dst      = value;
        return;
    }

    ggml_backend_tensor_set(const_cast<struct ggml_tensor*>(tensor), &value, offset, sizeof(T));
}

static void ggml_set_f32_nd(const struct ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float value) {
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

void ggml_ext_im_set_f32_1d(const struct ggml_tensor* tensor, int i, float value) {
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

static void ggml_backend_load_all_once() {
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

bool sd_backend_is(ggml_backend_t backend, const std::string& name) {
    if (!backend) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return false;
    }
    std::string dev_name = ggml_backend_dev_name(dev);
    return lower_copy(dev_name).find(lower_copy(name)) != std::string::npos;
}

static std::string get_default_backend_name() {
    ggml_backend_load_all_once();
    // should pick the same backend preference as ggml_backend_init_best
    std::string name = resolve_first_device_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!name.empty()) {
        return name;
    }
    name = resolve_first_device_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!name.empty()) {
        return name;
    }
    return resolve_first_device_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

static std::string sd_resolve_backend_name(const std::string& name) {
    ggml_backend_load_all_once();
    std::string requested = trim_copy(name);
    std::string lower     = lower_copy(requested);

    if (is_default_backend_token(lower)) {
        return get_default_backend_name();
    }
    if (lower == "gpu") {
        std::string result = resolve_first_device_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!result.empty()) {
            return result;
        }
        return resolve_first_device_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }

    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string dev_name   = ggml_backend_dev_name(dev);
        if (lower_copy(dev_name) == lower) {
            return dev_name;
        }
    }

    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string dev_name   = ggml_backend_dev_name(dev);
        std::string dev_lower  = lower_copy(dev_name);
        if (dev_lower.rfind(lower, 0) == 0) {
            return dev_name;
        }
    }

    return "";
}

static bool backend_name_exists(const std::string& name) {
    return !sd_resolve_backend_name(name).empty();
}

static ggml_backend_t init_named_backend(const std::string& name) {
    ggml_backend_load_all_once();
    LOG_DEBUG("Initializing backend: %s", name.c_str());
    if (trim_copy(name).empty()) {
        return ggml_backend_init_best();
    }

    std::string resolved = sd_resolve_backend_name(name);
    if (resolved.empty()) {
        return nullptr;
    }
    return ggml_backend_init_by_name(resolved.c_str(), nullptr);
}

static ggml_backend_t sd_get_default_backend() {
    ggml_backend_load_all_once();
    static std::once_flag once;
    std::call_once(once, []() {
        size_t dev_count = ggml_backend_dev_count();
        if (dev_count == 0) {
            LOG_ERROR("No devices found!");
        } else {
            LOG_DEBUG("Found %zu backend devices:", dev_count);
            for (size_t i = 0; i < dev_count; ++i) {
                auto dev = ggml_backend_dev_get(i);
                LOG_DEBUG("#%zu: %s", i, ggml_backend_dev_name(dev));
            }
        }
    });

    ggml_backend_t backend   = nullptr;
    const char* SD_VK_DEVICE = getenv("SD_VK_DEVICE");
    if (SD_VK_DEVICE != nullptr) {
        std::string sd_vk_device_str = SD_VK_DEVICE;
        try {
            unsigned long long device  = std::stoull(sd_vk_device_str);
            std::string vk_device_name = "Vulkan" + std::to_string(device);
            if (backend_name_exists(vk_device_name)) {
                LOG_INFO("Selecting %s as main device by env var SD_VK_DEVICE", vk_device_name.c_str());
                backend = init_named_backend(vk_device_name);
                if (!backend) {
                    LOG_WARN("Device %s requested by SD_VK_DEVICE failed to init. Falling back to the default device.", vk_device_name.c_str());
                }
            } else {
                LOG_WARN("Device %s requested by SD_VK_DEVICE was not found. Falling back to the default device.", vk_device_name.c_str());
            }
        } catch (const std::invalid_argument&) {
            LOG_WARN("SD_VK_DEVICE environment variable is not a valid integer (%s). Falling back to the default device.", SD_VK_DEVICE);
        } catch (const std::out_of_range&) {
            LOG_WARN("SD_VK_DEVICE environment variable value is out of range for `unsigned long long` type (%s). Falling back to the default device.", SD_VK_DEVICE);
        }
    }

    if (!backend) {
        std::string dev_name = get_default_backend_name();
        backend              = init_named_backend(dev_name);
        if (!backend && !dev_name.empty()) {
            LOG_WARN("device %s failed to init", dev_name.c_str());
        }
    }

    if (!backend) {
        LOG_WARN("loading CPU backend");
        backend = ggml_backend_cpu_init();
    }

    if (ggml_backend_is_cpu(backend)) {
        LOG_DEBUG("Using CPU backend");
    }

    return backend;
}

static bool sd_parse_backend_assignment(const std::string& spec, SDBackendAssignment* assignment, std::string* error) {
    if (assignment == nullptr) {
        return false;
    }

    *assignment          = {};
    const std::string in = trim_copy(spec);
    if (in.empty()) {
        return true;
    }

    for (const std::string& raw_part : split_copy(in, ',')) {
        const std::string part = trim_copy(raw_part);
        if (part.empty()) {
            continue;
        }

        const size_t eq = part.find('=');
        if (eq == std::string::npos) {
            assignment->set_default(part);
            continue;
        }

        const std::string key   = trim_copy(part.substr(0, eq));
        const std::string value = trim_copy(part.substr(eq + 1));
        if (key.empty() || value.empty()) {
            if (error != nullptr) {
                *error = "invalid backend assignment '" + part + "'";
            }
            return false;
        }

        const std::string key_lower = lower_copy(key);
        if (key_lower == "all" || key_lower == "default" || key_lower == "*") {
            assignment->set_default(value);
            continue;
        }

        SDBackendModule module = SDBackendModule::DIFFUSION;
        if (!parse_backend_module(key, &module)) {
            if (error != nullptr) {
                *error = "unknown backend module '" + key + "'";
            }
            return false;
        }
        assignment->set_module(module, value);
    }
    return true;
}

bool SDBackendAssignment::empty() const {
    return default_name.empty() && module_names.empty();
}

std::string SDBackendAssignment::get(SDBackendModule module) const {
    return module_assignment_name(*this, module);
}

void SDBackendAssignment::set_default(const std::string& name) {
    default_name = trim_copy(name);
}

void SDBackendAssignment::set_module(SDBackendModule module, const std::string& name) {
    module_names[module] = trim_copy(name);
}

void SDBackendHandleDeleter::operator()(ggml_backend_t backend) const {
    ggml_backend_free(backend);
}

SDBackendManager::~SDBackendManager() {
    reset();
}

void SDBackendManager::reset() {
    backends_.clear();
    runtime_assignment_ = {};
    params_assignment_  = {};
}

ggml_backend_t SDBackendManager::runtime_backend(SDBackendModule module) {
    return init_cached_backend(runtime_assignment_.get(module));
}

ggml_backend_t SDBackendManager::params_backend(SDBackendModule module) {
    std::string name = params_assignment_.get(module);
    if (name.empty()) {
        return runtime_backend(module);
    }
    return init_cached_backend(name);
}

bool SDBackendManager::runtime_backend_is_cpu(SDBackendModule module) {
    return ggml_backend_is_cpu(runtime_backend(module));
}

bool SDBackendManager::params_backend_is_cpu(SDBackendModule module) {
    return ggml_backend_is_cpu(params_backend(module));
}

bool SDBackendManager::runtime_backend_supports_host_buffer(SDBackendModule module) {
    ggml_backend_t backend = runtime_backend(module);
    if (backend == nullptr) {
        return false;
    }
    if (ggml_backend_is_cpu(backend)) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.caps.buffer_from_host_ptr;
}

bool SDBackendManager::init(const char* backend_spec,
                            const char* params_backend_spec,
                            bool offload_params_to_cpu,
                            bool keep_clip_on_cpu,
                            bool keep_vae_on_cpu,
                            bool keep_control_net_on_cpu,
                            std::string* error) {
    reset();

    if (!sd_parse_backend_assignment(SAFE_STR(backend_spec), &runtime_assignment_, error)) {
        return false;
    }
    if (!sd_parse_backend_assignment(SAFE_STR(params_backend_spec), &params_assignment_, error)) {
        return false;
    }

    if (runtime_assignment_.empty()) {
        if (keep_clip_on_cpu) {
            runtime_assignment_.set_module(SDBackendModule::TE, "cpu");
        }
        if (keep_vae_on_cpu) {
            runtime_assignment_.set_module(SDBackendModule::VAE, "cpu");
        }
        if (keep_control_net_on_cpu) {
            runtime_assignment_.set_module(SDBackendModule::CONTROL_NET, "cpu");
        }
    }

    if (params_assignment_.empty() && offload_params_to_cpu) {
        params_assignment_.set_default("cpu");
    }

    return validate(error);
}

bool SDBackendManager::validate(std::string* error) const {
    auto validate_name = [&](const std::string& name) -> bool {
        if (is_default_backend_token(name)) {
            return true;
        }
        if (!sd_resolve_backend_name(name).empty()) {
            return true;
        }
        if (error != nullptr) {
            *error = "backend '" + name + "' was not found";
        }
        return false;
    };

    if (!validate_name(runtime_assignment_.default_name) ||
        !validate_name(params_assignment_.default_name)) {
        return false;
    }
    for (const auto& kv : runtime_assignment_.module_names) {
        if (!validate_name(kv.second)) {
            return false;
        }
    }
    for (const auto& kv : params_assignment_.module_names) {
        if (!validate_name(kv.second)) {
            return false;
        }
    }
    return true;
}

ggml_backend_t SDBackendManager::init_cached_backend(const std::string& name) {
    std::string resolved   = sd_resolve_backend_name(name);
    std::string key        = lower_copy(resolved);
    ggml_backend_t backend = nullptr;

    if (!key.empty()) {
        auto it = backends_.find(key);
        if (it != backends_.end()) {
            return it->second.get();
        }
    } else if (!is_default_backend_token(name)) {
        LOG_ERROR("backend '%s' was not found", name.c_str());
        return nullptr;
    }

    backend = is_default_backend_token(name) ? sd_get_default_backend() : init_named_backend(resolved);
    if (backend == nullptr) {
        LOG_ERROR("failed to initialize backend '%s'", name.c_str());
        return nullptr;
    }

    std::string actual_key = backend_cache_key(backend);
    if (actual_key.empty()) {
        actual_key = !key.empty() ? key : lower_copy(trim_copy(name));
    }

    auto it = backends_.find(actual_key);
    if (it != backends_.end()) {
        ggml_backend_free(backend);
        return it->second.get();
    }

    SDBackendHandle handle(backend);
    backends_.emplace(actual_key, std::move(handle));
    return backend;
}

const char* sd_backend_module_name(SDBackendModule module) {
    switch (module) {
        case SDBackendModule::DIFFUSION:
            return "diffusion";
        case SDBackendModule::TE:
            return "te";
        case SDBackendModule::CLIP_VISION:
            return "clip_vision";
        case SDBackendModule::VAE:
            return "vae";
        case SDBackendModule::CONTROL_NET:
            return "controlnet";
        case SDBackendModule::PHOTOMAKER:
            return "photomaker";
        case SDBackendModule::UPSCALER:
            return "upscaler";
    }
    return "unknown";
}
