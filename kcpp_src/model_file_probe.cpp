#include "model_file_probe.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

namespace {

constexpr size_t kMaxSafetensorsHeader   = 1ULL << 30;  // 1 GB - a header is never bigger
constexpr size_t kMaxGgufArrayElements   = 4096;       // serialize at most this many array elements
constexpr size_t kJsonTailWindow         = 64;         // bytes inspected at the end for the JSON check

bool is_json_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string format_float32(float v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

std::string format_float64(double v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

// Scalar GGUF value (not string, not array) to string, from a raw data pointer
std::string gguf_scalar_to_string(enum gguf_type type, const void* data) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(*reinterpret_cast<const uint8_t*>(data));
        case GGUF_TYPE_INT8:    return std::to_string(*reinterpret_cast<const int8_t*>(data));
        case GGUF_TYPE_UINT16:  return std::to_string(*reinterpret_cast<const uint16_t*>(data));
        case GGUF_TYPE_INT16:   return std::to_string(*reinterpret_cast<const int16_t*>(data));
        case GGUF_TYPE_UINT32:  return std::to_string(*reinterpret_cast<const uint32_t*>(data));
        case GGUF_TYPE_INT32:   return std::to_string(*reinterpret_cast<const int32_t*>(data));
        case GGUF_TYPE_FLOAT32: return format_float32(*reinterpret_cast<const float*>(data));
        case GGUF_TYPE_BOOL:    return *reinterpret_cast<const int8_t*>(data) ? "1" : "0";
        case GGUF_TYPE_UINT64:  return std::to_string(*reinterpret_cast<const uint64_t*>(data));
        case GGUF_TYPE_INT64:   return std::to_string(*reinterpret_cast<const int64_t*>(data));
        case GGUF_TYPE_FLOAT64: return format_float64(*reinterpret_cast<const double*>(data));
        default:                return "";
    }
}

// Any GGUF KV value (scalars, strings, arrays) to string
std::string gguf_value_to_string(const struct gguf_context* ctx, int64_t key_id) {
    const auto type = gguf_get_kv_type(ctx, key_id);
    if (type != GGUF_TYPE_ARRAY) {
        if (type == GGUF_TYPE_STRING) {
            const char* s = gguf_get_val_str(ctx, key_id);
            return s ? s : "";
        }
        const void* data = gguf_get_val_data(ctx, key_id);
        return data ? gguf_scalar_to_string(type, data) : "";
    }

    const auto etype = gguf_get_arr_type(ctx, key_id);
    const size_t n   = gguf_get_arr_n(ctx, key_id);
    const size_t count = std::min(n, kMaxGgufArrayElements);
    const void* data   = (etype != GGUF_TYPE_STRING) ? gguf_get_arr_data(ctx, key_id) : nullptr;

    size_t esize = 0;
    switch (etype) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL:     esize = 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                        esize = 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: esize = 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: esize = 8; break;
        default: break;
    }

    std::string out = "[";
    for (size_t i = 0; i < count; i++) {
        if (i)
            out += ", ";
        if (etype == GGUF_TYPE_STRING) {
            const char* s = gguf_get_arr_str(ctx, key_id, i);
            out += (s ? s : "");
        } else if (data && esize) {
            out += gguf_scalar_to_string(etype, static_cast<const char*>(data) + i * esize);
        } else {
            out += "?";
        }
    }
    if (n > count)
        out += ", ...";
    out += "]";
    return out;
}



}  // namespace

// GGUF backend: header + KV + tensor table only (no weight data)
bool kcpp_file_probe::open_gguf(kcpp_file_probe& p, const std::string& path) {
    struct ggml_context* meta = nullptr;
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &meta;
    struct gguf_context* ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        return false;
    }

    p.gguf_ctx_      = ctx;
    p.gguf_meta_ctx_ = meta;
    p.format_        = kcpp_file_format::GGUF;
    p.version_       = static_cast<int>(gguf_get_version(ctx));

    const int64_t n_kv = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < n_kv; i++) {
        const char* key = gguf_get_key(ctx, i);
        if (key) {
            p.metadata_[key] = gguf_value_to_string(ctx, i);
        }
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    p.tensors_.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; i++) {
        kcpp_probe_tensor t;
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name)
            continue;
        t.name  = name;
        t.dtype = ggml_type_name(gguf_get_tensor_type(ctx, i));
        if (meta) {
            const struct ggml_tensor* dummy = ggml_get_tensor(meta, name);
            if (dummy) {
                const int nd = ggml_n_dims(dummy);
                for (int d = 0; d < nd; d++) {
                    t.shape.push_back(dummy->ne[d]);
                }
            }
        }
        p.tensors_.push_back(std::move(t));
    }
    p.tensors_complete_ = true;
    return true;
}

// Safetensors backend: header JSON only (no weight data).
// Returns false (fall through) when the file is not a safetensors file.
bool kcpp_file_probe::open_safetensors(kcpp_file_probe& p, std::ifstream& fin, uint64_t file_size) {
    uint8_t len_buf[8];
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char*>(len_buf), sizeof(len_buf));
    if (!fin) {
        return false;
    }
    uint64_t header_len = 0;
    memcpy(&header_len, len_buf, sizeof(header_len));
    if (header_len < 2 || header_len > file_size - 8 || header_len > kMaxSafetensorsHeader) {
        return false;
    }

    std::vector<char> header(header_len + 1, '\0');
    fin.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!fin) {
        return false;
    }

    nlohmann::json header_json;
    try {
        header_json = nlohmann::json::parse(header.data());
    } catch (const std::exception&) {
        return false;
    }
    if (!header_json.is_object()) {
        return false;
    }

    // A valid safetensors header has at least one tensor entry or __metadata__
    bool has_content = header_json.contains("__metadata__");
    if (!has_content) {
        for (auto it = header_json.begin(); it != header_json.end(); ++it) {
            if (it.value().is_object() && it.value().contains("dtype")) {
                has_content = true;
                break;
            }
        }
    }
    if (!has_content) {
        return false;
    }

    p.format_ = kcpp_file_format::SAFETENSORS;

    auto meta = header_json.find("__metadata__");
    if (meta != header_json.end() && meta->is_object()) {
        for (auto& [key, value] : meta->items()) {
            if (value.is_string()) {
                p.metadata_[key] = value.get<std::string>();
            }
        }
    }

    for (auto& [name, entry] : header_json.items()) {
        if (name == "__metadata__" || !entry.is_object() || !entry.contains("dtype")) {
            continue;
        }
        kcpp_probe_tensor t;
        t.name = name;
        if (entry["dtype"].is_string()) {
            t.dtype = entry["dtype"].get<std::string>();
        }
        auto shape = entry.find("shape");
        if (shape != entry.end() && shape->is_array()) {
            for (const auto& dim : *shape) {
                if (dim.is_number_integer()) {
                    t.shape.push_back(dim.get<int64_t>());
                }
            }
        }
        p.tensors_.push_back(std::move(t));
    }
    p.tensors_complete_ = true;
    return true;
}

// Cheap "looks like JSON" check: first non-whitespace byte is '{' or '[' and
// the last non-whitespace byte is the matching closer. No full parse.
bool kcpp_file_probe::looks_like_json(std::ifstream& fin, uint64_t file_size) {
    char head[8] = {0};
    fin.seekg(0, std::ios::beg);
    fin.read(head, sizeof(head));
    const auto head_len = fin.gcount();
    if (head_len <= 0) {
        return false;
    }

    const char* first = nullptr;
    // skip a UTF-8 BOM, if present (future-proofing)
    std::streamoff offset = 0;
    if (head_len >= 3 && head[0] == 0xEF && head[1] == 0xBB && head[2] == 0xBF) {
        offset = 3;
    }
    for (std::streamoff i = offset; i < head_len; i++) {
        if (!is_json_ws(head[i])) {
            first = &head[i];
            break;
        }
    }
    if (!first) {
        return false;
    }
    const char closer = *first == '{' ? '}' : (*first == '[' ? ']' : 0);
    if (!closer) {
        return false;
    }

    const size_t window = std::min<uint64_t>(file_size, kJsonTailWindow);
    std::vector<char> tail(window);
    fin.seekg(static_cast<std::streamoff>(file_size - window), std::ios::beg);
    fin.read(tail.data(), static_cast<std::streamsize>(window));
    for (std::streamoff i = fin.gcount() - 1; i >= 0; i--) {
        if (!is_json_ws(tail[i])) {
            return tail[i] == closer;
        }
    }
    return false;
}

kcpp_file_probe::~kcpp_file_probe() {
    reset();
}

kcpp_file_probe::kcpp_file_probe(kcpp_file_probe&& other) noexcept {
    *this = std::move(other);
}

kcpp_file_probe& kcpp_file_probe::operator=(kcpp_file_probe&& other) noexcept {
    if (this != &other) {
        reset();
        path_             = std::move(other.path_);
        error_            = std::move(other.error_);
        format_           = other.format_;
        version_          = other.version_;
        tensors_          = std::move(other.tensors_);
        tensors_complete_ = other.tensors_complete_;
        metadata_         = std::move(other.metadata_);
        gguf_ctx_         = other.gguf_ctx_;
        other.gguf_ctx_   = nullptr;
        gguf_meta_ctx_    = other.gguf_meta_ctx_;
        other.gguf_meta_ctx_ = nullptr;
    }
    return *this;
}

void kcpp_file_probe::reset() {
    if (gguf_ctx_) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
    if (gguf_meta_ctx_) {
        ggml_free(gguf_meta_ctx_);
        gguf_meta_ctx_ = nullptr;
    }
    path_.clear();
    error_.clear();
    format_           = kcpp_file_format::UNKNOWN;
    version_          = 0;
    tensors_.clear();
    tensors_complete_ = false;
    metadata_.clear();
}

bool kcpp_file_probe::open(const std::string& path) {
    reset();
    path_ = path;

    std::ifstream fin(std::filesystem::u8path(path), std::ios::binary);
    if (!fin) {
        error_ = "failed to open '" + path + "'";
        return false;
    }
    fin.seekg(0, std::ios::end);
    const auto file_size = fin.tellg();
    if (file_size < 0) {
        error_ = "failed to determine size of '" + path + "'";
        return false;
    }
    fin.seekg(0, std::ios::beg);

    if (static_cast<uint64_t>(file_size) < 8) {
        error_ = "file too small to identify";
        return true;  // readable, UNKNOWN
    }

    uint8_t head[8] = {0};
    fin.read(reinterpret_cast<char*>(head), sizeof(head));
    if (!fin) {
        error_ = "failed to read file header";
        return false;
    }

    if (memcmp(head, "GGUF", 4) == 0) {
        if (open_gguf(*this, path)) {
            return true;
        }
        // GGUF magic but unparseable: report as unknown, not unreadable
        error_ = "GGUF magic present, but the file could not be parsed";
        return true;
    }

    if (open_safetensors(*this, fin, static_cast<uint64_t>(file_size))) {
        return true;
    }

    if (looks_like_json(fin, static_cast<uint64_t>(file_size))) {
        format_           = kcpp_file_format::JSON;
        tensors_complete_ = true;
        return true;
    }

    if (head[0] == 'P' && head[1] == 'K' && head[2] == 0x03 && head[3] == 0x04) {
        format_ = kcpp_file_format::TORCH_ZIP;
        return true;
    }

    error_ = "unrecognized file format";
    return true;  // readable, UNKNOWN
}

std::string kcpp_file_probe::metadata_get(const std::string& key) const {
    const auto it = metadata_.find(key);
    return (it == metadata_.end()) ? std::string() : it->second;
}

bool kcpp_file_probe::tensor_exists(const std::string& name, bool exact) const {
    for (const auto& t : tensors_) {
        if (exact ? t.name == name : t.name.find(name) != std::string::npos) {
            return true;
        }
    }
    return false;
}
