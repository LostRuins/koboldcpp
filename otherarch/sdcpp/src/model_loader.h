#ifndef __MODEL_LOADER_H__
#define __MODEL_LOADER_H__

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "model.h"

TensorTypeRules parse_tensor_type_rules(const std::string& tensor_type_rules);

class MmapWrapper;

struct ModelFileData {
    std::string path;
    std::vector<TensorStorage> tensors;
    std::shared_ptr<MmapWrapper> mmapped;
    std::shared_ptr<struct ggml_backend_buffer> mmbuffer;
    bool is_zip;
};

struct MmapTensorStore {
    std::shared_ptr<MmapWrapper> mmapped;
    std::shared_ptr<struct ggml_backend_buffer> mmbuffer;
};

class ModelLoader {
protected:
    SDVersion version_ = VERSION_COUNT;
    std::vector<std::string> file_paths_;
    std::vector<ModelFileData> file_data;
    bool model_files_processed = false;
    String2TensorStorage tensor_storage_map;

    void add_tensor_storage(const TensorStorage& tensor_storage);

    bool init_from_gguf_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_torch_zip_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_torch_legacy_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_diffusers_file(const std::string& file_path, const std::string& prefix = "");

public:
    bool init_from_file(const std::string& file_path, const std::string& prefix = "");
    void convert_tensors_name();
    bool init_from_file_and_convert_name(const std::string& file_path,
                                         const std::string& prefix = "",
                                         SDVersion version         = VERSION_COUNT);
    SDVersion get_sd_version();
    std::map<ggml_type, uint32_t> get_wtype_stat();
    std::map<ggml_type, uint32_t> get_conditioner_wtype_stat();
    std::map<ggml_type, uint32_t> get_diffusion_model_wtype_stat();
    std::map<ggml_type, uint32_t> get_vae_wtype_stat();
    String2TensorStorage& get_tensor_storage_map() { return tensor_storage_map; }
    void set_wtype_override(ggml_type wtype, std::string tensor_type_rules = "");
    void process_model_files(bool enable_mmap = false, bool writable_mmap = true);
    std::vector<MmapTensorStore> mmap_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                              std::set<std::string> ignore_tensors = {},
                                              bool writable                        = true);
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads = 0, bool use_mmap = false);
    bool load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                      std::set<std::string> ignore_tensors = {},
                      int n_threads                        = 0,
                      bool use_mmap                        = false);

    std::vector<std::string> get_tensor_names() const {
        std::vector<std::string> names;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            names.push_back(name);
        }
        return names;
    }

    bool tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type);
    int64_t get_params_mem_size(ggml_backend_t backend, ggml_type type = GGML_TYPE_COUNT);
    ~ModelLoader() = default;
    bool has_diffusion_model_tensors();
};

#endif  // __MODEL_LOADER_H__
