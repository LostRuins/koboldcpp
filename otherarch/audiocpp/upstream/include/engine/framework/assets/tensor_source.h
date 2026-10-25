#pragma once

#include "engine/framework/core/module.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::assets {

// Existing loaders remain unsynchronized. Model adapters explicitly opt in before
// sharing a source across sessions; the policy is fixed at source construction.
struct TensorSourceOptions {
    bool synchronized_access = false;
};

enum class TensorStorageType {
    Native,
    F32,
    F16,
    BF16,
    I8,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    NVFP4,
    // drama3-small: i-quants (routed expert banks).
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
};

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
};

struct TensorData {
    core::TensorShape shape;
    ggml_type type = GGML_TYPE_F32;
    std::vector<std::byte> bytes;
};

struct TensorDataF32 {
    core::TensorShape shape;
    std::vector<float> values;
};

struct RawTensorData {
    TensorMetadata metadata;
    std::vector<std::byte> bytes;
};

class TensorSource {
public:
    virtual ~TensorSource() = default;

    [[nodiscard]] virtual const std::filesystem::path & source_path() const noexcept = 0;
    [[nodiscard]] virtual bool has_tensor(std::string_view name) const noexcept = 0;
    [[nodiscard]] virtual TensorMetadata require_metadata(std::string_view name) const = 0;
    [[nodiscard]] virtual std::vector<TensorMetadata> tensors() const = 0;
    virtual void release_storage() const {}
    [[nodiscard]] virtual RawTensorData require_tensor_data(std::string_view name) const = 0;
    [[nodiscard]] virtual std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const = 0;
    [[nodiscard]] virtual std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const = 0;
    [[nodiscard]] std::vector<float> require_f32(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const {
        return require_f32(name, std::optional<std::vector<int64_t>>(std::vector<int64_t>(expected_shape)));
    }
    [[nodiscard]] std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const {
        return optional_f32(name, std::optional<std::vector<int64_t>>(std::vector<int64_t>(expected_shape)));
    }
    [[nodiscard]] TensorDataF32 require_f32_tensor(std::string_view name) const;
    [[nodiscard]] TensorDataF32 require_f32_tensor(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const;
    [[nodiscard]] TensorData require_tensor(
        std::string_view name,
        TensorStorageType storage_type,
        std::initializer_list<int64_t> expected_shape) const;
    [[nodiscard]] TensorData require_tensor(
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const;
    [[nodiscard]] TensorData require_tensor_as_shape(
        std::string_view name,
        TensorStorageType storage_type,
        std::initializer_list<int64_t> expected_source_shape,
        std::initializer_list<int64_t> tensor_shape) const;
    virtual void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const;
    virtual void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const;
    [[nodiscard]] std::optional<TensorDataF32> optional_f32_tensor(std::string_view name) const;
    [[nodiscard]] std::optional<TensorDataF32> optional_f32_tensor(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const;
    [[nodiscard]] std::optional<RawTensorData> optional_tensor_data(std::string_view name) const;
    [[nodiscard]] std::optional<std::string> find_tensor_name(
        std::initializer_list<std::string_view> candidates) const;
    [[nodiscard]] std::string require_tensor_name(
        std::initializer_list<std::string_view> candidates) const;
    [[nodiscard]] virtual int64_t require_i64_scalar(std::string_view name) const = 0;
};

[[nodiscard]] TensorStorageType parse_tensor_storage_type(std::string_view value);
[[nodiscard]] ggml_type ggml_type_for_tensor_dtype(std::string_view dtype);
[[nodiscard]] ggml_type ggml_type_for_tensor_storage(TensorStorageType storage_type);
[[nodiscard]] TensorStorageType tensor_storage_type_for_dtype(std::string_view dtype);
[[nodiscard]] std::vector<float> tensor_data_to_f32(
    std::string_view name,
    const TensorData & tensor);
[[nodiscard]] TensorStorageType resolve_tensor_storage_type(
    const TensorSource & source,
    std::string_view name,
    TensorStorageType requested_type);
inline void require_tensor_shape(
    const TensorSource & source,
    std::string_view name,
    const std::vector<int64_t> & expected_shape) {
    const auto metadata = source.require_metadata(name);
    if (metadata.shape != expected_shape) {
        throw std::runtime_error("tensor shape mismatch for " + std::string(name));
    }
}
inline void require_tensor_shape(
    const TensorSource & source,
    std::string_view name,
    std::initializer_list<int64_t> expected_shape) {
    require_tensor_shape(source, name, std::vector<int64_t>(expected_shape));
}
// When requested, retain the exact upload bytes for reuse instead of reconverting.
void set_backend_tensor_from_f32_parallel(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type,
    std::vector<std::byte> * retained_bytes = nullptr);
std::shared_ptr<const TensorSource> open_tensor_source(const std::filesystem::path & path);
// Finds a logical checkpoint by stem, preferring safetensors over PyTorch and GGUF.
std::optional<std::filesystem::path> find_tensor_file(
    const std::filesystem::path & directory, std::string_view stem);
std::shared_ptr<const TensorSource> open_tensor_source(
    const std::filesystem::path & path, TensorSourceOptions options);
std::shared_ptr<const TensorSource> open_tensor_source(
    const std::filesystem::path & path, std::string_view tensor_prefix, TensorSourceOptions options);
std::shared_ptr<const TensorSource> open_tensor_source(
    const std::filesystem::path & path,
    std::string_view tensor_prefix);
std::shared_ptr<const TensorSource> make_prefixed_tensor_source(
    std::shared_ptr<const TensorSource> source,
    std::string_view tensor_prefix);
std::shared_ptr<const TensorSource> make_weight_norm_folded_tensor_source(
    std::shared_ptr<const TensorSource> source,
    std::vector<std::string> patterns);
struct TensorSourceInput {
    std::filesystem::path path;
    std::string tensor_prefix;
};
struct GgufEmbeddedFile {
    std::filesystem::path source_path;
    std::filesystem::path destination;
};
struct GgufEmbeddedModelSpec {
    std::string family;
    std::string json;
};
struct GgufTensorTypeOverride {
    std::string pattern;
    TensorStorageType storage_type = TensorStorageType::Native;
};
struct WeightAdapter;
enum class WeightMergeMode;
struct GgufConversionOptions {
    std::optional<TensorStorageType> bnb_nf4_type;
    std::vector<std::string> excluded_tensor_prefixes;
    std::vector<GgufTensorTypeOverride> type_overrides;
    std::vector<std::string> folded_weight_norm_patterns;
    // Bound tensor names; applied before conversion and quantization.
    std::shared_ptr<const WeightAdapter> weight_adapter;
    std::optional<WeightMergeMode> weight_adapter_merge_mode;
};
void convert_tensor_sources_to_gguf(const std::vector<TensorSourceInput> & inputs,
                                    const std::filesystem::path & output_path, TensorStorageType weight_type,
                                    bool overwrite = false, bool embed_sidecars = true,
                                    const std::filesystem::path & sidecar_root = {},
                                    const std::vector<GgufEmbeddedFile> & extra_sidecars = {},
                                    const std::optional<GgufEmbeddedModelSpec> & model_spec = std::nullopt,
                                    GgufConversionOptions options = {});
void convert_tensor_source_to_gguf(const std::filesystem::path & input_path, const std::filesystem::path & output_path,
                                   TensorStorageType weight_type, bool overwrite = false, bool embed_sidecars = true);
[[nodiscard]] bool gguf_has_embedded_sidecars(const std::filesystem::path & path);
// The destinations of a GGUF's embedded sidecars, in the order it stores them.
// Read from the file itself, so a caller re-emitting a package does not have to
// infer the set from whatever is sitting in the materialisation cache. Throws on
// the same malformed metadata that materialising the sidecars would reject: this
// answers what the package holds, and a package whose byte ranges do not fit its
// blob holds nothing anyone can read.
[[nodiscard]] std::vector<std::string> gguf_embedded_sidecar_names(const std::filesystem::path & path);
[[nodiscard]] std::optional<GgufEmbeddedModelSpec> read_gguf_embedded_model_spec(const std::filesystem::path & path);
[[nodiscard]] std::filesystem::path materialize_gguf_sidecars(const std::filesystem::path & path);
struct PreparedModelDirectory {
    std::filesystem::path model_root;
    std::optional<std::filesystem::path> standalone_gguf;
};
// *.gguf files directly inside `directory`, sorted by name; empty when it is not a directory.
[[nodiscard]] std::vector<std::filesystem::path> directory_gguf_files(const std::filesystem::path & directory);
// The one GGUF a model directory stands for: `model.gguf` when present, otherwise the sole
// *.gguf in it. Published GGUF packages keep their release name (vevo2-q8_0.gguf), so requiring
// `model.gguf` would silently push those directories onto the safetensors path. Returns nullopt
// when the directory holds no GGUF, or several with no `model.gguf` to disambiguate them.
[[nodiscard]] std::optional<std::filesystem::path> find_directory_gguf(const std::filesystem::path & directory);
[[nodiscard]] PreparedModelDirectory
prepare_model_directory(const std::filesystem::path & model_path,
    const std::filesystem::path & gguf_relative_path = "model.gguf");
std::vector<std::filesystem::path> indexed_tensor_source_shard_paths(const std::filesystem::path & index_path,
    const std::filesystem::path & model_root);
std::shared_ptr<const TensorSource> open_indexed_tensor_source(const std::filesystem::path & index_path,
    const std::filesystem::path & model_root);
std::shared_ptr<const TensorSource> open_indexed_tensor_source(
    const std::filesystem::path & index_path, const std::filesystem::path & model_root,
    TensorSourceOptions options);

}  // namespace engine::assets
