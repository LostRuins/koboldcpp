#include "model_file_probe.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <utility>

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

namespace {

constexpr size_t kMaxSafetensorsHeader   = 1ULL << 30;  // 1 GB - a header is never bigger
constexpr size_t kMaxGgufArrayElements   = 4096;       // serialize at most this many array elements
bool is_json_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
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
    struct gguf_context* ctx = gguf_init_from_file(path.c_str(), {true, nullptr});
    if (!ctx) {
        return false;
    }

    p.gguf_ctx_      = ctx;
    p.format_        = kcpp_file_format::GGUF;
    p.version_       = static_cast<int>(gguf_get_version(ctx));

    p.tensors_complete_ = true;
    return true;
}

const std::vector<kcpp_probe_tensor>& kcpp_file_probe::tensors() const {
    if (!gguf_ctx_ || !tensors_.empty()) {
        return tensors_;
    }
    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx_);
    std::vector<kcpp_probe_tensor> tensors;
    tensors.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; i++) {
        const int64_t* shape = gguf_get_tensor_ne(gguf_ctx_, i);
        int dims = GGML_MAX_DIMS;
        while (dims > 1 && shape[dims - 1] == 1) {
            --dims;
        }
        tensors.push_back({gguf_get_tensor_name(gguf_ctx_, i),
                           ggml_type_name(gguf_get_tensor_type(gguf_ctx_, i)),
                           {shape, shape + dims}});
    }
    tensors_ = std::move(tensors);
    return tensors_;
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

    std::vector<char> header(header_len);
    fin.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!fin) {
        return false;
    }

    nlohmann::json header_json;
    try {
        header_json = nlohmann::json::parse(header.begin(), header.end());
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

namespace {
// Inspect only the root weight_map. Tokenizer vocabularies can be large.
struct index_probe : nlohmann::detail::json_sax_acceptor<nlohmann::json> {
    size_t depth = 0;
    bool weight_map_key = false;
    bool in_weight_map = false;
    std::map<std::string, bool> shard_paths_are_strings;
    bool* current_shard_path_is_string = nullptr;

    bool key(std::string& name) {
        if (depth == 1) {
            weight_map_key = name == "weight_map";
            if (weight_map_key) {
                // Match the loader's last-value-wins handling of duplicate keys.
                shard_paths_are_strings.clear();
                current_shard_path_is_string = nullptr;
            }
        } else if (in_weight_map && depth == 2) {
            current_shard_path_is_string = &shard_paths_are_strings.insert_or_assign(name, false).first->second;
        }
        return true;
    }
    bool string(std::string&) {
        if (in_weight_map && depth == 2) {
            *current_shard_path_is_string = true;
        }
        return true;
    }
    bool start_object(size_t) {
        if (depth == 1 && weight_map_key) {
            in_weight_map = true;
        }
        ++depth;
        return true;
    }
    bool end_object() {
        if (--depth == 1) in_weight_map = false;
        return true;
    }
    bool start_array(size_t) { ++depth; return true; }
    bool end_array() { --depth; return true; }
};
}

bool kcpp_file_probe::open_json(std::ifstream& fin) {
    fin.clear();
    fin.seekg(0, std::ios::beg);
    int first = fin.get();
    if (first == 0xEF) {
        if (fin.get() != 0xBB || fin.get() != 0xBF) {
            return false;
        }
        first = fin.get();
    }
    while (first != std::char_traits<char>::eof() && is_json_ws(static_cast<char>(first))) {
        first = fin.get();
    }
    if (first != '{' && first != '[') {
        return false;
    }

    // Validate without building a JSON object tree. Require stream EOF because
    // the parser also accepts a NUL byte as an end-of-input token.
    fin.seekg(0, std::ios::beg);
    index_probe index;
    if (!nlohmann::json::sax_parse(fin, &index) || !fin.eof()) return false;

    format_ = kcpp_file_format::JSON;
    const auto& shard_paths = index.shard_paths_are_strings;
    if (!shard_paths.empty() && std::all_of(shard_paths.begin(), shard_paths.end(),
                                           [](const auto& entry) { return entry.second; })) {
        tensors_.reserve(shard_paths.size());
        for (const auto& entry : shard_paths) {
            tensors_.push_back({entry.first, {}, {}});
        }
        format_ = kcpp_file_format::SAFETENSORS_INDEX;
    }
    tensors_complete_ = true;
    return true;
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
        gguf_ctx_         = std::exchange(other.gguf_ctx_, nullptr);
    }
    return *this;
}

void kcpp_file_probe::reset() {
    if (gguf_ctx_) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
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

    uint8_t head[8] = {0};
    const auto head_size = std::min<std::streamoff>(file_size, sizeof(head));
    fin.read(reinterpret_cast<char*>(head), head_size);
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

    if (head_size == sizeof(head) && open_safetensors(*this, fin, static_cast<uint64_t>(file_size))) {
        return true;
    }

    if (open_json(fin)) {
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
    if (gguf_ctx_) {
        const int64_t index = gguf_find_key(gguf_ctx_, key.c_str());
        return index >= 0 ? gguf_value_to_string(gguf_ctx_, index) : std::string();
    }
    const auto it = metadata_.find(key);
    return (it == metadata_.end()) ? std::string() : it->second;
}

bool kcpp_file_probe::metadata_has_prefix(const std::string& prefix) const {
    if (gguf_ctx_) {
        for (int64_t i = 0, n = gguf_get_n_kv(gguf_ctx_); i < n; ++i) {
            if (std::string_view(gguf_get_key(gguf_ctx_, i)).rfind(prefix, 0) == 0) {
                return true;
            }
        }
        return false;
    }
    const auto it = metadata_.lower_bound(prefix);
    return it != metadata_.end() && it->first.rfind(prefix, 0) == 0;
}

bool kcpp_file_probe::tensor_exists(const std::string& name, bool exact) const {
    if (gguf_ctx_) {
        if (exact) {
            return gguf_find_tensor(gguf_ctx_, name.c_str()) >= 0;
        }
        for (int64_t i = 0, n = gguf_get_n_tensors(gguf_ctx_); i < n; ++i) {
            if (std::string_view(gguf_get_tensor_name(gguf_ctx_, i)).find(name) != std::string_view::npos) {
                return true;
            }
        }
        return false;
    }
    for (const auto& t : tensors_) {
        if (exact ? t.name == name : t.name.find(name) != std::string::npos) {
            return true;
        }
    }
    return false;
}
