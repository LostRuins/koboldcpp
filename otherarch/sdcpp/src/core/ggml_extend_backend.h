#ifndef __SD_CORE_GGML_EXTEND_BACKEND_H__
#define __SD_CORE_GGML_EXTEND_BACKEND_H__

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "stable-diffusion.h"

enum class SDBackendModule {
    DIFFUSION,
    TE,
    CLIP_VISION,
    VAE,
    CONTROL_NET,
    PHOTOMAKER,
    UPSCALER,
    DETECTOR,
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

enum class SDSplitMode {
    LAYER,
    ROW,
};

class SDBackendManager {
private:
    SDBackendAssignment runtime_assignment_;
    SDBackendAssignment params_assignment_;
    SDBackendAssignment split_mode_assignment_;
    std::unordered_map<std::string, SDBackendHandle> backends_;

public:
    SDBackendManager() = default;
    ~SDBackendManager();

    SDBackendManager(const SDBackendManager&)            = delete;
    SDBackendManager& operator=(const SDBackendManager&) = delete;

    bool init(const char* backend_spec,
              const char* params_backend_spec,
              const char* split_mode_spec,
              std::string* error);
    void reset();

    ggml_backend_t runtime_backend(SDBackendModule module);
    ggml_backend_t params_backend(SDBackendModule module);

    std::vector<ggml_backend_t> runtime_backends(SDBackendModule module);

    SDSplitMode split_mode(SDBackendModule module) const;
    ggml_backend_buffer_type_t split_buffer_type(ggml_backend_t backend,
                                                 const std::vector<float>& tensor_split);

    bool runtime_backend_is_cpu(SDBackendModule module);
    bool params_backend_is_cpu(SDBackendModule module);
    bool params_backend_is_disk(SDBackendModule module) const;
    bool params_backend_follows_runtime(SDBackendModule module) const;
    bool runtime_backend_supports_host_buffer(SDBackendModule module);

private:
    bool validate(std::string* error) const;
    ggml_backend_t init_cached_backend(const std::string& name);
};

bool sd_backend_is(ggml_backend_t backend, const std::string& name);
bool sd_backend_is_cpu(ggml_backend_t backend);
ggml_backend_t sd_backend_cpu_init();
bool sd_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);
ggml_status sd_backend_graph_compute_with_eval_callback(ggml_backend_t backend,
                                                        ggml_cgraph* gf,
                                                        sd_graph_eval_callback_t callback_eval,
                                                        void* callback_eval_user_data);
std::string sd_backend_resolve_name(const std::string& name);
const char* sd_backend_module_name(SDBackendModule module);
void ggml_ext_im_set_f32_1d(const struct ggml_tensor* tensor, int i, float value);
bool add_rpc_devices(const std::string& servers);
#endif  // __SD_CORE_GGML_EXTEND_BACKEND_H__
