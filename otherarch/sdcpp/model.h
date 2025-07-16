#ifndef __MODEL_H__
#define __MODEL_H__

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include <nlohmann/json.hpp>
#include "zip.h"
#include "gguf.h"

#define SD_MAX_DIMS 5

enum SDVersion {
    VERSION_SD1,
    VERSION_SD1_INPAINT,
    VERSION_SD2,
    VERSION_SD2_INPAINT,
    VERSION_SDXL,
    VERSION_SDXL_INPAINT,
    VERSION_SVD,
    VERSION_SD3,
    VERSION_FLUX,
    VERSION_FLUX_FILL,
    VERSION_COUNT,
};

static inline bool sd_version_is_flux(SDVersion version) {
    if (version == VERSION_FLUX || version == VERSION_FLUX_FILL) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_sd3(SDVersion version) {
    if (version == VERSION_SD3) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_sd1(SDVersion version) {
    if (version == VERSION_SD1 || version == VERSION_SD1_INPAINT) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_sd2(SDVersion version) {
    if (version == VERSION_SD2 || version == VERSION_SD2_INPAINT) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_sdxl(SDVersion version) {
    if (version == VERSION_SDXL || version == VERSION_SDXL_INPAINT) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_inpaint(SDVersion version) {
    if (version == VERSION_SD1_INPAINT || version == VERSION_SD2_INPAINT || version == VERSION_SDXL_INPAINT || version == VERSION_FLUX_FILL) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_dit(SDVersion version) {
    if (sd_version_is_flux(version) || sd_version_is_sd3(version)) {
        return true;
    }
    return false;
}

enum PMVersion {
    PM_VERSION_1,
    PM_VERSION_2,
};

struct TensorStorage {
    std::string name;
    ggml_type type          = GGML_TYPE_F32;
    bool is_bf16            = false;
    bool is_f8_e4m3         = false;
    bool is_f8_e5m2         = false;
    int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
    int n_dims              = 0;

    size_t file_index = 0;
    int index_in_zip  = -1;  // >= means stored in a zip file
    size_t offset     = 0;   // offset in file

    TensorStorage() = default;

    TensorStorage(const std::string& name, ggml_type type, int64_t* ne, int n_dims, size_t file_index, size_t offset = 0)
        : name(name), type(type), n_dims(n_dims), file_index(file_index), offset(offset) {
        for (int i = 0; i < n_dims; i++) {
            this->ne[i] = ne[i];
        }
    }

    int64_t nelements() const {
        int64_t n = 1;
        for (int i = 0; i < SD_MAX_DIMS; i++) {
            n *= ne[i];
        }
        return n;
    }

    int64_t nbytes() const {
        return nelements() * ggml_type_size(type) / ggml_blck_size(type);
    }

    int64_t nbytes_to_read() const {
        if (is_bf16 || is_f8_e4m3 || is_f8_e5m2) {
            return nbytes() / 2;
        } else {
            return nbytes();
        }
    }

    void unsqueeze() {
        if (n_dims == 2) {
            n_dims = 4;
            ne[3]  = ne[1];
            ne[2]  = ne[0];
            ne[1]  = 1;
            ne[0]  = 1;
        }
    }

    std::vector<TensorStorage> chunk(size_t n) {
        std::vector<TensorStorage> chunks;
        size_t chunk_size = nbytes_to_read() / n;
        // printf("%d/%d\n", chunk_size, nbytes_to_read());
        reverse_ne();
        for (int i = 0; i < n; i++) {
            TensorStorage chunk_i = *this;
            chunk_i.ne[0]         = ne[0] / n;
            chunk_i.offset        = offset + i * chunk_size;
            chunk_i.reverse_ne();
            chunks.push_back(chunk_i);
        }
        reverse_ne();
        return chunks;
    }

    void reverse_ne() {
        int64_t new_ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            new_ne[i] = ne[n_dims - 1 - i];
        }
        for (int i = 0; i < n_dims; i++) {
            ne[i] = new_ne[i];
        }
    }

    std::string to_string() const {
        std::stringstream ss;
        const char* type_name = ggml_type_name(type);
        if (is_bf16) {
            type_name = "bf16";
        } else if (is_f8_e4m3) {
            type_name = "f8_e4m3";
        } else if (is_f8_e5m2) {
            type_name = "f8_e5m2";
        }
        ss << name << " | " << type_name << " | ";
        ss << n_dims << " [";
        for (int i = 0; i < SD_MAX_DIMS; i++) {
            ss << ne[i];
            if (i != SD_MAX_DIMS - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }
};

typedef std::function<bool(const TensorStorage&, ggml_tensor**)> on_new_tensor_cb_t;

class ModelLoader {
protected:
    std::vector<std::string> file_paths_;
    std::vector<TensorStorage> tensor_storages;

    bool parse_data_pkl(uint8_t* buffer,
                        size_t buffer_size,
                        zip_t* zip,
                        std::string dir,
                        size_t file_index,
                        const std::string prefix);

    bool init_from_gguf_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_ckpt_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_diffusers_file(const std::string& file_path, const std::string& prefix = "");

public:
    std::map<std::string, enum ggml_type> tensor_storages_types;

    bool init_from_file(const std::string& file_path, const std::string& prefix = "");
    bool has_diffusion_model_tensors();
    SDVersion get_sd_version();
    ggml_type get_sd_wtype();
    ggml_type get_conditioner_wtype();
    ggml_type get_diffusion_model_wtype();
    ggml_type get_vae_wtype();
    void set_wtype_override(ggml_type wtype, std::string prefix = "");
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb, ggml_backend_t backend);
    bool load_tensors(std::map<std::string, struct ggml_tensor*>& tensors,
                      ggml_backend_t backend,
                      std::set<std::string> ignore_tensors = {});

    bool save_to_gguf_file(const std::string& file_path, ggml_type type);
    bool tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type);
    int64_t get_params_mem_size(ggml_backend_t backend, ggml_type type = GGML_TYPE_COUNT);
    ~ModelLoader() = default;

    static std::string load_merges();
    static std::string load_t5_tokenizer_json();
};

#endif  // __MODEL_H__