#ifndef __SD_TENSOR_STORAGE_H__
#define __SD_TENSOR_STORAGE_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"

#define SD_MAX_DIMS 5

struct TensorStorage {
    std::string name;
    ggml_type type          = GGML_TYPE_F32;
    ggml_type expected_type = GGML_TYPE_COUNT;
    bool is_f8_e4m3         = false;
    bool is_f8_e5m2         = false;
    bool is_f64             = false;
    bool is_i64             = false;
    int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
    int n_dims              = 0;

    std::string storage_key;
    size_t file_index = 0;
    int index_in_zip  = -1;  // >= means stored in a zip file
    uint64_t offset   = 0;   // offset in file

    TensorStorage() = default;

    TensorStorage(std::string name, ggml_type type, const int64_t* ne, int n_dims, size_t file_index, size_t offset = 0)
        : name(std::move(name)), type(type), n_dims(n_dims), file_index(file_index), offset(offset) {
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
        if (is_f8_e4m3 || is_f8_e5m2) {
            return nbytes() / 2;
        } else if (is_f64 || is_i64) {
            return nbytes() * 2;
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
        uint64_t chunk_size = nbytes_to_read() / n;
        // printf("%d/%d\n", chunk_size, nbytes_to_read());
        reverse_ne();
        for (size_t i = 0; i < n; i++) {
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
        if (is_f8_e4m3) {
            type_name = "f8_e4m3";
        } else if (is_f8_e5m2) {
            type_name = "f8_e5m2";
        } else if (is_f64) {
            type_name = "f64";
        } else if (is_i64) {
            type_name = "i64";
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

struct TensorWriteInfo {
    int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
    int n_dims              = 0;
    ggml_tensor* tensor     = nullptr;
};

typedef std::function<bool(const TensorStorage&, ggml_tensor**)> on_new_tensor_cb_t;

#endif  // __SD_TENSOR_STORAGE_H__
