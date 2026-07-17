#include "core/ggml_extend_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "core/util.h"
#include "ggml/src/ggml-impl.h"
#include "stable-diffusion.h"

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

static bool is_disk_backend_token(const std::string& name) {
    return lower_copy(trim_copy(name)) == "disk";
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
    if (name == "detector" || name == "adetailer" || name == "yolo") {
        *module = SDBackendModule::DETECTOR;
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
    const char* dev_name = ggml_backend_dev_name(dev);
    if (dev_name != nullptr && dev_name[0] != '\0') {
        return dev_name;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char* reg_name   = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
    return reg_name != nullptr ? reg_name : "";
}

static ggml_backend_dev_t resolve_first_device_by_registry_name(const std::string& name) {
    std::string lower = lower_copy(trim_copy(name));
    if (lower == "metal") {
        lower = "mtl";
    }
    if (lower.empty()) {
        return nullptr;
    }

    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg == nullptr) {
            continue;
        }
        const char* reg_name = ggml_backend_reg_name(reg);
        if (reg_name != nullptr && lower_copy(reg_name) == lower) {
            return dev;
        }
    }
    return nullptr;
}

static ggml_backend_dev_t resolve_device_by_name(const std::string& name) {
    const std::string lower = lower_copy(trim_copy(name));
    if (lower.empty()) {
        return nullptr;
    }

    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* dev_name   = ggml_backend_dev_name(dev);
        if (dev_name != nullptr && lower_copy(dev_name) == lower) {
            return dev;
        }
    }
    return nullptr;
}

static std::string backend_device_name(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return "";
    }
    const char* name = ggml_backend_dev_name(dev);
    if (name != nullptr && name[0] != '\0') {
        return name;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char* reg_name   = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
    return reg_name != nullptr ? reg_name : "";
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

bool add_rpc_devices(const std::string& servers) {
    const std::string in = trim_copy(servers);
    if (in.empty()) {
        return true;
    }
    auto rpc_servers = split_copy(in, ',');
    if (rpc_servers.empty()) {
        LOG_ERROR("invalid RPC servers specification: '%s'", servers.c_str());
        return false;
    }
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LOG_ERROR("RPC backend not found, cannot add RPC servers");
        return false;
    }
    typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char* endpoint);
    ggml_backend_rpc_add_server_t ggml_backend_rpc_add_server_fn = (ggml_backend_rpc_add_server_t)ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        LOG_ERROR("RPC backend does not have ggml_backend_rpc_add_server function, cannot add RPC servers");
        return false;
    }
    for (const auto& server : rpc_servers) {
        LOG_INFO("Adding RPC server: %s", server.c_str());
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        // no return value to check for success but should print errors from the RPC backend if it fails to add the server
        ggml_backend_register(reg);
    }
    return true;
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

std::string sd_backend_resolve_name(const std::string& name) {
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

    if (ggml_backend_dev_t dev = resolve_first_device_by_registry_name(requested)) {
        return backend_device_name(dev);
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
    return !sd_backend_resolve_name(name).empty();
}

static ggml_backend_t init_named_backend(const std::string& name) {
    ggml_backend_load_all_once();
    LOG_DEBUG("Initializing backend: %s", name.c_str());
    if (trim_copy(name).empty()) {
        return ggml_backend_init_best();
    }

    if (ggml_backend_dev_t dev = resolve_device_by_name(name)) {
        return ggml_backend_dev_init(dev, nullptr);
    }
    if (ggml_backend_dev_t dev = resolve_first_device_by_registry_name(name)) {
        return ggml_backend_dev_init(dev, nullptr);
    }

    std::string resolved = sd_backend_resolve_name(name);
    if (ggml_backend_dev_t dev = resolve_device_by_name(resolved)) {
        return ggml_backend_dev_init(dev, nullptr);
    }
    if (ggml_backend_dev_t dev = resolve_first_device_by_registry_name(resolved)) {
        return ggml_backend_dev_init(dev, nullptr);
    }
    if (resolved.empty()) {
        return nullptr;
    }
    return ggml_backend_init_by_name(resolved.c_str(), nullptr);
}

bool sd_backend_is_cpu(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    auto dev = ggml_backend_get_device(backend);
    return dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

ggml_backend_t sd_backend_cpu_init() {
    ggml_backend_load_all_once();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

bool sd_backend_cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    if (backend == nullptr) {
        return false;
    }
    auto dev = ggml_backend_get_device(backend);
    if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        auto reg                           = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn != nullptr) {
            ggml_backend_set_n_threads_fn(backend, n_threads);
            return true;
        }
    }
    return false;
}

static ggml_cgraph sd_ggml_graph_view(ggml_cgraph* cgraph0, int i0, int i1) {
    ggml_cgraph cgraph = {
        /*.size             =*/0,
        /*.n_nodes          =*/i1 - i0,
        /*.n_leafs          =*/0,
        /*.nodes            =*/cgraph0->nodes + i0,
        /*.grads            =*/nullptr,
        /*.grad_accs        =*/nullptr,
        /*.leafs            =*/nullptr,
        /*.use_counts       =*/cgraph0->use_counts,
        /*.visited_hash_set =*/cgraph0->visited_hash_set,
        /*.order            =*/cgraph0->order,
        /*.uid              =*/0,
    };
    return cgraph;
}

ggml_status sd_backend_graph_compute_with_eval_callback(ggml_backend_t backend,
                                                        ggml_cgraph* gf,
                                                        sd_graph_eval_callback_t callback_eval,
                                                        void* callback_eval_user_data) {
    if (callback_eval == nullptr) {
        return ggml_backend_graph_compute(backend, gf);
    }

    ggml_status status = GGML_STATUS_SUCCESS;
    const int n_nodes  = ggml_graph_n_nodes(gf);
    bool stopped       = false;

    for (int j0 = 0; j0 < n_nodes; ++j0) {
        ggml_tensor* t = ggml_graph_node(gf, j0);
        bool need      = callback_eval(t, true, callback_eval_user_data);
        int j1         = j0;

        while (!need && j1 < n_nodes - 1) {
            t    = ggml_graph_node(gf, ++j1);
            need = callback_eval(t, true, callback_eval_user_data);
        }

        ggml_cgraph gv = sd_ggml_graph_view(gf, j0, j1 + 1);
        status         = ggml_backend_graph_compute_async(backend, &gv);
        if (status != GGML_STATUS_SUCCESS) {
            break;
        }

        ggml_backend_synchronize(backend);

        if (need && !callback_eval(t, false, callback_eval_user_data)) {
            stopped = true;
            break;
        }

        j0 = j1;
    }

    ggml_backend_synchronize(backend);
    if (stopped && status == GGML_STATUS_SUCCESS) {
        status = GGML_STATUS_ABORTED;
    }
    return status;
}

const char* sd_get_system_info() {
    static std::string cache_info = []() -> std::string {
        ggml_backend_load_all_once();
        std::stringstream ss;
        ss << "System Info: \n";
        auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev != nullptr) {
            auto reg                          = ggml_backend_dev_backend_reg(dev);
            auto ggml_backend_get_features_fn = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
            if (ggml_backend_get_features_fn != nullptr) {
                ggml_backend_feature* feat = ggml_backend_get_features_fn(reg);
                while (feat->name && feat->value) {
                    ss << "   " << feat->name << " = " << feat->value << " | ";
                    feat++;
                }
            } else {
                LOG_WARN("unable to get CPU features");
            }
        } else {
            LOG_WARN("unable to get CPU features");
        }
        return ss.str();
    }();
    return cache_info.c_str();
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
        backend = sd_backend_cpu_init();
    }

    if (sd_backend_is_cpu(backend)) {
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
    runtime_assignment_    = {};
    params_assignment_     = {};
    split_mode_assignment_ = {};
}

static std::vector<std::string> split_device_list(const std::string& value) {
    std::vector<std::string> names;
    for (const std::string& raw : split_copy(value, '&')) {
        const std::string name = trim_copy(raw);
        if (!name.empty()) {
            names.push_back(name);
        }
    }
    return names;
}

static std::string primary_device_name(const std::string& value) {
    std::vector<std::string> names = split_device_list(value);
    return names.empty() ? std::string() : names.front();
}

ggml_backend_t SDBackendManager::runtime_backend(SDBackendModule module) {
    return init_cached_backend(primary_device_name(runtime_assignment_.get(module)));
}

std::vector<ggml_backend_t> SDBackendManager::runtime_backends(SDBackendModule module) {
    std::vector<ggml_backend_t> backends;
    for (const std::string& name : split_device_list(runtime_assignment_.get(module))) {
        ggml_backend_t backend = init_cached_backend(name);
        if (backend == nullptr) {
            LOG_ERROR("failed to initialize backend '%s' for module %s",
                      name.c_str(),
                      sd_backend_module_name(module));
            continue;
        }
        if (std::find(backends.begin(), backends.end(), backend) == backends.end()) {
            backends.push_back(backend);
        }
    }
    if (backends.empty()) {
        ggml_backend_t backend = runtime_backend(module);
        if (backend != nullptr) {
            backends.push_back(backend);
        }
    }
    return backends;
}

ggml_backend_t SDBackendManager::params_backend(SDBackendModule module) {
    std::string name = params_assignment_.get(module);
    if (name.empty()) {
        return runtime_backend(module);
    }
    if (is_disk_backend_token(name)) {
        return runtime_backend(module);
    }
    return init_cached_backend(name);
}

bool SDBackendManager::runtime_backend_is_cpu(SDBackendModule module) {
    return sd_backend_is_cpu(runtime_backend(module));
}

bool SDBackendManager::params_backend_is_cpu(SDBackendModule module) {
    return sd_backend_is_cpu(params_backend(module));
}

bool SDBackendManager::params_backend_is_disk(SDBackendModule module) const {
    return is_disk_backend_token(params_assignment_.get(module));
}

bool SDBackendManager::params_backend_follows_runtime(SDBackendModule module) const {
    return params_assignment_.get(module).empty();
}

bool SDBackendManager::runtime_backend_supports_host_buffer(SDBackendModule module) {
    ggml_backend_t backend = runtime_backend(module);
    if (backend == nullptr) {
        return false;
    }
    if (sd_backend_is_cpu(backend)) {
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
                            const char* split_mode_spec,
                            std::string* error) {
    reset();

    if (!sd_parse_backend_assignment(SAFE_STR(backend_spec), &runtime_assignment_, error)) {
        return false;
    }
    if (!sd_parse_backend_assignment(SAFE_STR(params_backend_spec), &params_assignment_, error)) {
        return false;
    }
    if (!sd_parse_backend_assignment(SAFE_STR(split_mode_spec), &split_mode_assignment_, error)) {
        return false;
    }

    return validate(error);
}

SDSplitMode SDBackendManager::split_mode(SDBackendModule module) const {
    return lower_copy(trim_copy(split_mode_assignment_.get(module))) == "row" ? SDSplitMode::ROW
                                                                              : SDSplitMode::LAYER;
}

ggml_backend_buffer_type_t SDBackendManager::split_buffer_type(ggml_backend_t backend,
                                                               const std::vector<float>& tensor_split) {
    if (backend == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }
    auto fn = (ggml_backend_split_buffer_type_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
    if (fn == nullptr) {
        return nullptr;
    }
    int main_device        = -1;
    const size_t dev_count = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < dev_count; ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            main_device = (int)i;
            break;
        }
    }
    if (main_device < 0) {
        return nullptr;
    }
    std::vector<float> padded_split(std::max<size_t>(tensor_split.size(), 64), 0.0f);
    std::copy(tensor_split.begin(), tensor_split.end(), padded_split.begin());
    return fn(main_device, padded_split.data());
}

bool SDBackendManager::validate(std::string* error) const {
    auto validate_single_runtime_name = [&](const std::string& name) -> bool {
        if (is_default_backend_token(name)) {
            return true;
        }
        if (is_disk_backend_token(name)) {
            if (error != nullptr) {
                *error = "backend 'disk' is only supported by params_backend";
            }
            return false;
        }
        if (!sd_backend_resolve_name(name).empty() || resolve_first_device_by_registry_name(name) != nullptr) {
            return true;
        }
        if (error != nullptr) {
            *error = "backend '" + name + "' was not found";
        }
        return false;
    };
    auto validate_runtime_name = [&](const std::string& name) -> bool {
        if (name.find('&') == std::string::npos) {
            return validate_single_runtime_name(name);
        }
        std::vector<std::string> names = split_device_list(name);
        if (names.empty()) {
            if (error != nullptr) {
                *error = "invalid backend device list '" + name + "'";
            }
            return false;
        }
        for (const std::string& entry : names) {
            if (is_default_backend_token(entry)) {
                if (error != nullptr) {
                    *error = "default backend token is not allowed in a device list '" + name + "'";
                }
                return false;
            }
            if (!validate_single_runtime_name(entry)) {
                return false;
            }
        }
        return true;
    };
    auto validate_params_name = [&](const std::string& name) -> bool {
        if (is_disk_backend_token(name)) {
            return true;
        }
        if (name.find('&') != std::string::npos) {
            if (error != nullptr) {
                *error = "params_backend does not accept device lists ('" + name + "')";
            }
            return false;
        }
        return validate_single_runtime_name(name);
    };
    auto validate_split_mode_name = [&](const std::string& name) -> bool {
        const std::string lower = lower_copy(trim_copy(name));
        if (lower.empty() || lower == "layer" || lower == "row") {
            return true;
        }
        if (error != nullptr) {
            *error = "invalid split mode '" + name + "' (expected layer or row)";
        }
        return false;
    };

    if (!validate_runtime_name(runtime_assignment_.default_name) ||
        !validate_params_name(params_assignment_.default_name) ||
        !validate_split_mode_name(split_mode_assignment_.default_name)) {
        return false;
    }
    for (const auto& kv : runtime_assignment_.module_names) {
        if (!validate_runtime_name(kv.second)) {
            return false;
        }
    }
    for (const auto& kv : params_assignment_.module_names) {
        if (!validate_params_name(kv.second)) {
            return false;
        }
    }
    for (const auto& kv : split_mode_assignment_.module_names) {
        if (!validate_split_mode_name(kv.second)) {
            return false;
        }
    }
    return true;
}

ggml_backend_t SDBackendManager::init_cached_backend(const std::string& name) {
    std::string resolved   = sd_backend_resolve_name(name);
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
        case SDBackendModule::DETECTOR:
            return "detector";
    }
    return "unknown";
}
