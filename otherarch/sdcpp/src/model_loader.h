#ifndef __MODEL_LOADER_H__
#define __MODEL_LOADER_H__

#include <cstdint>
#include <filesystem>
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

bool is_unused_tensor(const std::string& name);

class ModelLoader {
public:
    using FileId       = uint64_t;
    using FileVersions = std::map<FileId, uint64_t>;
    enum class FileScope { Catalog,
                           Isolated };

private:
    struct FileStamp {
        std::string path;
        uintmax_t size = 0;
        std::filesystem::file_time_type modified;
    };

    struct FileRecord {
        FileId id         = 0;
        uint64_t revision = 0;
        std::string path;
        std::string prefix;
        FileScope scope = FileScope::Catalog;
        std::vector<FileStamp> dependencies;
        String2TensorStorage tensors;
        std::map<std::string, std::string> metadata;
    };

    std::vector<FileRecord> files_;
    uint64_t revision_        = 0;
    bool names_converted_     = false;
    ggml_type wtype_override_ = GGML_TYPE_COUNT;
    std::string tensor_type_rules_;
    std::vector<FileStamp> parsed_dependencies_;
    std::map<std::string, std::set<std::string>> parsed_tensor_names_;

    static bool read_file_stamp(const std::string& path, FileStamp& stamp);
    static bool file_unchanged(const FileStamp& stamp);
    bool parse_file(const std::string& path, const std::string& prefix);
    bool add_file_impl(const std::string& path, const std::string& prefix, FileId* id, bool force, FileScope scope);
    ModelLoader file_reader(FileId id, SDVersion version) const;
    void rebuild_catalog();
    void invalidate_file_data();

protected:
    SDVersion version_ = VERSION_COUNT;
    std::vector<std::string> file_paths_;
    std::vector<ModelFileData> file_data;
    bool model_files_processed = false;
    String2TensorStorage tensor_storage_map;
    std::map<std::string, std::string> metadata_;
    int n_threads_;

    size_t add_file_path(const std::string& file_path);
    void add_tensor_storage(const TensorStorage& tensor_storage);

    bool init_from_gguf_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_index_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_torch_zip_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_torch_legacy_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_diffusers_file(const std::string& file_path, const std::string& prefix = "");

public:
    ModelLoader();

    bool add_file(const std::string& path, const std::string& prefix = "", FileId* id = nullptr, bool force = false, FileScope scope = FileScope::Catalog);
    bool del_file(FileId id);
    uint64_t file_revision(FileId id) const;
    std::string file_path(FileId id) const;
    String2TensorStorage file_tensors(FileId id, SDVersion version) const;
    bool load_file_tensors(FileId id, SDVersion version, on_new_tensor_cb_t callback, const std::set<std::string>& names, bool use_mmap = false) const;
    bool refresh_files(bool include_isolated = true);
    bool files_changed(bool& changed, bool include_isolated = true) const;
    bool validate_sources(const std::set<std::string>* tensor_names = nullptr) const;
    uint64_t revision() const { return revision_; }
    FileVersions file_versions(const std::vector<std::string>& prefixes = {}) const;
    bool init_from_file(const std::string& file_path, const std::string& prefix = "");
    void convert_tensors_name();
    bool init_from_file_and_convert_name(const std::string& file_path,
                                         const std::string& prefix = "",
                                         SDVersion version         = VERSION_COUNT);
    SDVersion get_sd_version() const;
    std::map<ggml_type, uint32_t> get_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_conditioner_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_diffusion_model_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_vae_wtype_stat() const;
    String2TensorStorage& get_tensor_storage_map() { return tensor_storage_map; }
    const String2TensorStorage& get_tensor_storage_map() const { return tensor_storage_map; }
    const std::map<std::string, std::string>& get_metadata() const { return metadata_; }
    void set_n_threads(int n_threads);
    void set_wtype_override(ggml_type wtype, std::string tensor_type_rules = "");
    void process_model_files(bool enable_mmap = false, bool writable_mmap = true);
    std::vector<MmapTensorStore> mmap_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                              std::set<std::string> ignore_tensors = {},
                                              bool writable                        = true);
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb,
                      bool use_mmap                                    = false,
                      const std::set<std::string>* target_tensor_names = nullptr,
                      bool log_progress                                = true);
    bool load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                      std::set<std::string> ignore_tensors = {},
                      bool use_mmap                        = false);
    bool load_float_tensor(const std::string& name,
                           std::vector<float>& data,
                           int n_threads = 0,
                           bool use_mmap = false);
    bool load_tensor(const TensorStorage& tensor_storage, ggml_tensor* dst_tensor);

    std::vector<std::string> get_tensor_names() const {
        std::vector<std::string> names;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            names.push_back(name);
        }
        return names;
    }

    bool tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) const;
    int64_t get_params_mem_size(ggml_backend_t backend, ggml_type type = GGML_TYPE_COUNT) const;
    ~ModelLoader() = default;
    bool has_diffusion_model_tensors();
};

#endif  // __MODEL_LOADER_H__
