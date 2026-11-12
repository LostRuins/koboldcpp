#ifndef KCPP_MODEL_FILE_PROBE_H
#define KCPP_MODEL_FILE_PROBE_H

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;

// Read-only probe for model-related files (weights, tokenizers, ...).
//
// open() reads just enough of the file to classify it and expose its tensors
// and metadata; it never touches weight data.
//
//   format() == GGUF        : full tensor list + metadata (values serialized to strings)
//   format() == SAFETENSORS : full tensor list + __metadata__ (string values only)
//   format() == JSON        : classification only (tokenizer files, ...)
//   format() == TORCH_ZIP   : classification only
//   format() == UNKNOWN     : file is readable, but not one of the above
//                             (e.g. torch legacy pickle - a full parse is too
//                             expensive for a probe)
//
// open() returns false only when the file cannot be read at all (see error());
// a readable-but-unrecognized file yields format() == UNKNOWN and a true return.
// Not thread-safe: do not share an instance between threads.

enum class kcpp_file_format : int {
    UNKNOWN = 0,
    GGUF,
    SAFETENSORS,
    JSON,
    TORCH_ZIP,
};

struct kcpp_probe_tensor {
    std::string name;
    std::string dtype;  // e.g. "F16", "Q4_K" (GGUF) / "F16", "BF16" (safetensors)
    std::vector<int64_t> shape;
};

class kcpp_file_probe {
public:
    kcpp_file_probe() = default;
    ~kcpp_file_probe();
    kcpp_file_probe(const kcpp_file_probe&)            = delete;
    kcpp_file_probe& operator=(const kcpp_file_probe&) = delete;
    kcpp_file_probe(kcpp_file_probe&& other) noexcept;
    kcpp_file_probe& operator=(kcpp_file_probe&& other) noexcept;

    // Classify the file and fill tensors()/metadata(). See the class comment
    // for the open()/format() contract.
    bool open(const std::string& path);

    kcpp_file_format format() const { return format_; }
    int version() const { return version_; }  // e.g. GGUF version; 0 if n/a
    const std::string& path() const { return path_; }
    const std::string& error() const { return error_; }

    const std::vector<kcpp_probe_tensor>& tensors() const { return tensors_; }
    // false when the format is known but the tensor list was not (or could
    // not) be extracted, e.g. TORCH_ZIP
    bool tensors_complete() const { return tensors_complete_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    // "" when the key is absent
    std::string metadata_get(const std::string& key) const;

    // exact: full name match; otherwise substring match (as in gguf_tensor_exists)
    bool tensor_exists(const std::string& name, bool exact = true) const;

private:
    void reset();
    static bool open_gguf(kcpp_file_probe& p, const std::string& path);
    static bool open_safetensors(kcpp_file_probe& p, std::ifstream& fin, uint64_t file_size);
    static bool looks_like_json(std::ifstream& fin, uint64_t file_size);

    std::string path_;
    std::string error_;
    kcpp_file_format format_ = kcpp_file_format::UNKNOWN;
    int version_             = 0;
    std::vector<kcpp_probe_tensor> tensors_;
    bool tensors_complete_   = false;
    std::map<std::string, std::string> metadata_;

    struct gguf_context*  gguf_ctx_      = nullptr;  // owned when format_ == GGUF
    struct ggml_context*  gguf_meta_ctx_ = nullptr;  // ditto (dummy tensors, no data)
};

#endif  // KCPP_MODEL_FILE_PROBE_H
