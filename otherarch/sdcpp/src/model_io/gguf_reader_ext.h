#ifndef __SD_MODEL_IO_GGUF_READER_EXT_H__
#define __SD_MODEL_IO_GGUF_READER_EXT_H__

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "core/util.h"
#include "ggml.h"

struct GGUFTensorInfo {
    std::string name;
    ggml_type type;
    std::vector<int64_t> shape;
    size_t offset;
};

enum class GGUFMetadataType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

class GGUFReader {
private:
    std::vector<GGUFTensorInfo> tensors_;
    bool has_wide_tensors_    = false;
    uint64_t remaining_bytes_ = 0;
    size_t data_offset_;
    size_t alignment_ = 32;  // default alignment is 32

    template <typename T>
    bool safe_read(std::ifstream& fin, T& value) {
        return safe_read(fin, reinterpret_cast<char*>(&value), sizeof(T));
    }

    bool safe_read(std::ifstream& fin, char* buffer, size_t size) {
        if (size > remaining_bytes_)
            return false;
        fin.read(buffer, size);
        if (!fin.good())
            return false;
        remaining_bytes_ -= size;
        return true;
    }

    bool safe_skip(std::ifstream& fin, uint64_t count, uint64_t element_size = 1) {
        if (count > remaining_bytes_ / element_size)
            return false;
        uint64_t size = count * element_size;
        fin.seekg(static_cast<std::streamoff>(size), std::ios::cur);
        if (!fin.good())
            return false;
        remaining_bytes_ -= size;
        return true;
    }

    bool skip_metadata_values(std::ifstream& fin, GGUFMetadataType type, uint64_t count) {
        switch (type) {
            case GGUFMetadataType::UINT8:
            case GGUFMetadataType::INT8:
            case GGUFMetadataType::BOOL:
                return safe_skip(fin, count);

            case GGUFMetadataType::UINT16:
            case GGUFMetadataType::INT16:
                return safe_skip(fin, count, 2);

            case GGUFMetadataType::UINT32:
            case GGUFMetadataType::INT32:
            case GGUFMetadataType::FLOAT32:
                return safe_skip(fin, count, 4);

            case GGUFMetadataType::UINT64:
            case GGUFMetadataType::INT64:
            case GGUFMetadataType::FLOAT64:
                return safe_skip(fin, count, 8);

            case GGUFMetadataType::STRING:
                if (count > remaining_bytes_ / sizeof(uint64_t))
                    return false;
                for (uint64_t i = 0; i < count; i++) {
                    uint64_t len = 0;
                    if (!safe_read(fin, len) || !safe_skip(fin, len))
                        return false;
                }
                return true;

            default:
                LOG_ERROR("Unknown metadata type=%u", static_cast<uint32_t>(type));
                return false;
        }
    }

    bool read_metadata(std::ifstream& fin) {
        uint64_t key_len = 0;
        if (!safe_read(fin, key_len))
            return false;

        if (key_len > 4096)
            return false;

        std::string key(key_len, '\0');
        if (!safe_read(fin, (char*)key.data(), key_len))
            return false;

        uint32_t type = 0;
        if (!safe_read(fin, type))
            return false;

        if (key == "general.alignment") {
            uint32_t align_val = 0;
            if (!safe_read(fin, align_val))
                return false;

            if (align_val != 0 && (align_val & (align_val - 1)) == 0) {
                alignment_ = align_val;
                LOG_VERBOSE("Found alignment: %zu", alignment_);
            } else {
                LOG_ERROR("Invalid alignment value %u, fallback to default %zu", align_val, alignment_);
            }
            return true;
        }

        uint64_t count = 1;
        if (type == static_cast<uint32_t>(GGUFMetadataType::ARRAY)) {
            if (!safe_read(fin, type) || !safe_read(fin, count))
                return false;
        }
        return skip_metadata_values(fin, static_cast<GGUFMetadataType>(type), count);
    }

    GGUFTensorInfo read_tensor_info(std::ifstream& fin) {
        GGUFTensorInfo info;

        uint64_t name_len;
        if (!safe_read(fin, name_len))
            throw std::runtime_error("read tensor name length failed");

        info.name.resize(name_len);
        if (!safe_read(fin, (char*)info.name.data(), name_len))
            throw std::runtime_error("read tensor name failed");

        uint32_t n_dims;
        if (!safe_read(fin, n_dims))
            throw std::runtime_error("read tensor dims failed");

        info.shape.resize(n_dims);
        for (uint32_t i = 0; i < n_dims; i++) {
            if (!safe_read(fin, info.shape[i]))
                throw std::runtime_error("read tensor shape failed");
        }

        if (n_dims > GGML_MAX_DIMS) {
            has_wide_tensors_ = true;
            for (uint32_t i = GGML_MAX_DIMS; i < n_dims; i++) {
                info.shape[GGML_MAX_DIMS - 1] *= info.shape[i];  // stack to last dim;
            }
            info.shape.resize(GGML_MAX_DIMS);
            n_dims = GGML_MAX_DIMS;
        }

        uint32_t type;
        if (!safe_read(fin, type))
            throw std::runtime_error("read tensor type failed");
        info.type = static_cast<ggml_type>(type);

        if (!safe_read(fin, info.offset))
            throw std::runtime_error("read tensor offset failed");

        return info;
    }

public:
    bool load(const std::string& file_path) {
        std::ifstream fin(sd_get_u8path(file_path), std::ios::binary | std::ios::ate);
        if (!fin) {
            LOG_ERROR("failed to open '%s'", file_path.c_str());
            return false;
        }

        std::streamoff file_size = fin.tellg();
        if (file_size < 0)
            return false;
        remaining_bytes_ = static_cast<uint64_t>(file_size);
        fin.seekg(0, std::ios::beg);
        if (!fin.good())
            return false;

        // --- Header ---
        char magic[4];
        if (!safe_read(fin, magic, 4) || strncmp(magic, "GGUF", 4) != 0) {
            LOG_ERROR("not a valid GGUF file");
            return false;
        }

        uint32_t version;
        if (!safe_read(fin, version))
            return false;

        uint64_t tensor_count, metadata_kv_count;
        if (!safe_read(fin, tensor_count))
            return false;
        if (!safe_read(fin, metadata_kv_count))
            return false;

        LOG_VERBOSE("GGUF v%u, tensor_count=%llu, metadata_kv_count=%llu",
                    version, (unsigned long long)tensor_count, (unsigned long long)metadata_kv_count);

        // --- Read Metadata ---
        for (uint64_t i = 0; i < metadata_kv_count; i++) {
            if (!read_metadata(fin)) {
                LOG_ERROR("read meta data failed");
                return false;
            }
        }

        // --- Tensor Infos ---
        tensors_.clear();
        try {
            for (uint64_t i = 0; i < tensor_count; i++) {
                tensors_.push_back(read_tensor_info(fin));
            }
        } catch (const std::runtime_error& e) {
            LOG_ERROR("%s", e.what());
            return false;
        }

        data_offset_ = static_cast<size_t>(fin.tellg());
        if ((data_offset_ % alignment_) != 0) {
            data_offset_ = ((data_offset_ + alignment_ - 1) / alignment_) * alignment_;
        }
        fin.close();
        return true;
    }

    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }

    bool has_tensors_beyond_ggml_limits() const { return has_wide_tensors_; }
    size_t data_offset() const { return data_offset_; }
};

#endif  // __SD_MODEL_IO_GGUF_READER_EXT_H__
