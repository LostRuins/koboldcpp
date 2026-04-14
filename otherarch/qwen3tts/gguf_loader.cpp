#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>

#ifdef _WIN32
#    define ttsfseek _fseeki64
#else
#    define ttsfseek fseeko
#endif

namespace qwen3_tts {

namespace {
struct shared_backend_state {
    ggml_backend_t backend = nullptr;
    int32_t ref_count = 0;
};

shared_backend_state & get_shared_backend_state() {
    static shared_backend_state state;
    return state;
}
}

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg, bool allow_gpu) {
    if (error_msg) error_msg->clear();

    auto & shared = get_shared_backend_state();
    if (shared.backend) {
        shared.ref_count++;
        return shared.backend;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if(allow_gpu)
    {
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize backend (IGPU/GPU/ACCEL/CPU) for " + std::string(name);
    }

    if (backend) {
        shared.backend = backend;
        shared.ref_count = 1;
    }

    return backend;
}

void release_preferred_backend(ggml_backend_t backend) {
    if (!backend) {
        return;
    }

    auto & shared = get_shared_backend_state();
    if (shared.backend == backend) {
        shared.ref_count--;
        if (shared.ref_count <= 0) {
            ggml_backend_free(shared.backend);
            shared.backend = nullptr;
            shared.ref_count = 0;
        }
        return;
    }

    ggml_backend_free(backend);
}

bool GGUFLoader::open(const std::string & path) {
    close();  // Close any previously opened file

    file_path_ = path;

    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };

    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }

    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    bool allowgpu
) {
    ggml_backend_t backend = init_preferred_backend("TensorLoader", &error_msg, allowgpu);

    if (!backend) {
        error_msg = "Failed to initialize backend for GGUF tensor loader";
        return false;
    }

    // Allocate buffer for all tensors
    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) {
        error_msg = "Failed to allocate tensor buffer";
        release_preferred_backend(backend);
        return false;
    }

    // Open file for reading tensor data
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        release_preferred_backend(backend);
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        auto it = tensors.find(name);
        if (it == tensors.end()) {
            continue;  // Skip tensors not in our map
        }

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        read_buf.resize(nbytes);

        if (ttsfseek(f, data_offset + offset, SEEK_SET) != 0) {
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);
    release_preferred_backend(backend);

    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

} // namespace qwen3_tts
