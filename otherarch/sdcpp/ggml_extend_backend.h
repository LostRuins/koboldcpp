#ifndef __SD_GGML_EXTEND_BACKEND_H__
#define __SD_GGML_EXTEND_BACKEND_H__

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

#include "ggml-backend.h"
#include "ggml.h"

enum class SDBackendModule {
    DIFFUSION,
    TE,
    CLIP_VISION,
    VAE,
    CONTROL_NET,
    PHOTOMAKER,
    UPSCALER,
};

struct SDBackendAssignment {
    std::string default_name;
    std::unordered_map<SDBackendModule, std::string> module_names;

    bool empty() const;
    std::string get(SDBackendModule module) const;
    void set_default(const std::string& name);
    void set_module(SDBackendModule module, const std::string& name);
};

struct SDBackendHandleDeleter {
    void operator()(ggml_backend_t backend) const;
};

using SDBackendHandle = std::unique_ptr<struct ggml_backend, SDBackendHandleDeleter>;

class SDBackendManager {
private:
    SDBackendAssignment runtime_assignment_;
    SDBackendAssignment params_assignment_;
    std::unordered_map<std::string, SDBackendHandle> backends_;

public:
    SDBackendManager() = default;
    ~SDBackendManager();

    SDBackendManager(const SDBackendManager&)            = delete;
    SDBackendManager& operator=(const SDBackendManager&) = delete;

    bool init(const char* backend_spec,
              const char* params_backend_spec,
              bool offload_params_to_cpu,
              bool keep_clip_on_cpu,
              bool keep_vae_on_cpu,
              bool keep_control_net_on_cpu,
              std::string* error);
    void reset();

    ggml_backend_t runtime_backend(SDBackendModule module);
    ggml_backend_t params_backend(SDBackendModule module);

    bool runtime_backend_is_cpu(SDBackendModule module);
    bool params_backend_is_cpu(SDBackendModule module);
    bool runtime_backend_supports_host_buffer(SDBackendModule module);

private:
    bool validate(std::string* error) const;
    ggml_backend_t init_cached_backend(const std::string& name);
};

bool sd_backend_is(ggml_backend_t backend, const std::string& name);
bool sd_backend_is_cpu(ggml_backend_t backend);
ggml_backend_t sd_backend_cpu_init();
bool sd_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);
const char* sd_backend_module_name(SDBackendModule module);
void ggml_ext_im_set_f32_1d(const struct ggml_tensor* tensor, int i, float value);
#endif
