#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/assets/torch_bin.h"

#include "engine/framework/io/binary.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"

#include <gguf.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace engine::assets {
namespace {

enum class TensorContainer { Safetensors, Torch, Gguf };
constexpr std::pair<std::string_view, TensorContainer> kTensorContainers[] = {
    {".safetensors", TensorContainer::Safetensors},
    {".pt", TensorContainer::Torch},
    {".pth", TensorContainer::Torch},
    {".bin", TensorContainer::Torch},
    {".ckpt", TensorContainer::Torch},
    {".gguf", TensorContainer::Gguf},
};

constexpr int64_t kParallelF32ConvertElements = 1ll << 20;
constexpr int64_t kF32ConvertChunkElements = 1ll << 16;
// Quantization chunks are whole ROWS, so this is a budget rounded down to a row
// count rather than an exact split. A row of a large tensor is a few thousand
// elements, which puts a chunk in the same size range as the conversions above.
constexpr int64_t kQuantizeChunkElements = 1ll << 20;

bool tensor_type_override_matches(std::string_view name, std::string_view pattern) {
    if (pattern.empty()) {
        return false;
    }
    if (pattern.back() == '*') {
        pattern.remove_suffix(1);
        return name.substr(0, pattern.size()) == pattern;
    }
    return name == pattern;
}

std::optional<TensorStorageType> find_tensor_type_override(
    std::string_view name,
    const std::vector<GgufTensorTypeOverride> & overrides) {
    for (const auto & rule : overrides) {
        if (tensor_type_override_matches(name, rule.pattern)) {
            return rule.storage_type;
        }
    }
    return std::nullopt;
}

core::TensorShape shape_from_dims(const std::vector<int64_t> & dims) {
    if (dims.empty() || dims.size() > core::kMaxTensorRank) {
        throw std::runtime_error("tensor rank must be between 1 and 4");
    }
    switch (dims.size()) {
        case 1:
            return core::TensorShape::from_dims({dims[0]});
        case 2:
            return core::TensorShape::from_dims({dims[0], dims[1]});
        case 3:
            return core::TensorShape::from_dims({dims[0], dims[1], dims[2]});
        case 4:
            return core::TensorShape::from_dims({dims[0], dims[1], dims[2], dims[3]});
        default:
            throw std::runtime_error("unsupported tensor rank");
    }
}

void validate_expected_shape(
    std::string_view name,
    const std::vector<int64_t> & actual_shape,
    const std::optional<std::vector<int64_t>> & expected_shape) {
    if (expected_shape.has_value() && actual_shape != *expected_shape) {
        throw std::runtime_error("tensor shape mismatch for " + std::string(name));
    }
}

std::string lower_ascii(std::string_view value) {
    std::string out(value);
    for (char & ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool raw_dtype_matches_ggml_type(std::string_view dtype, ggml_type type) {
    const std::string normalized = lower_ascii(dtype);
    return (normalized == "f32" && type == GGML_TYPE_F32) ||
           (normalized == "f16" && type == GGML_TYPE_F16) ||
           (normalized == "bf16" && type == GGML_TYPE_BF16) ||
           (normalized == "i8" && type == GGML_TYPE_I8) ||
           (normalized == "q4_0" && type == GGML_TYPE_Q4_0) ||
           (normalized == "q4_1" && type == GGML_TYPE_Q4_1) ||
           (normalized == "q5_0" && type == GGML_TYPE_Q5_0) ||
           (normalized == "q5_1" && type == GGML_TYPE_Q5_1) ||
           (normalized == "q2_k" && type == GGML_TYPE_Q2_K) ||
           (normalized == "q3_k" && type == GGML_TYPE_Q3_K) ||
           (normalized == "q4_k" && type == GGML_TYPE_Q4_K) ||
           (normalized == "q5_k" && type == GGML_TYPE_Q5_K) ||
           (normalized == "q6_k" && type == GGML_TYPE_Q6_K) ||
           (normalized == "q8_0" && type == GGML_TYPE_Q8_0) ||
           (normalized == "nvfp4" && type == GGML_TYPE_NVFP4) ||
           (normalized == "iq2_xxs" && type == GGML_TYPE_IQ2_XXS) ||
           (normalized == "iq2_xs" && type == GGML_TYPE_IQ2_XS) ||
           (normalized == "iq2_s" && type == GGML_TYPE_IQ2_S) ||
           (normalized == "iq3_xxs" && type == GGML_TYPE_IQ3_XXS) ||
           (normalized == "iq3_s" && type == GGML_TYPE_IQ3_S) ||
           (normalized == "iq4_nl" && type == GGML_TYPE_IQ4_NL) ||
           (normalized == "iq4_xs" && type == GGML_TYPE_IQ4_XS);
}

ggml_type parse_ggml_type_for_tensor_dtype(std::string_view dtype) {
    const std::string normalized = lower_ascii(dtype);
    if (normalized == "f32" || normalized == "float32") return GGML_TYPE_F32;
    if (normalized == "f16" || normalized == "float16" || normalized == "fp16") return GGML_TYPE_F16;
    if (normalized == "bf16" || normalized == "bfloat16") return GGML_TYPE_BF16;
    if (normalized == "i8" || normalized == "int8") return GGML_TYPE_I8;
    if (normalized == "bool" || normalized == "boolean") return GGML_TYPE_I8;
    if (normalized == "i16" || normalized == "int16") return GGML_TYPE_I16;
    if (normalized == "i32" || normalized == "int32") return GGML_TYPE_I32;
    if (normalized == "i64" || normalized == "int64") return GGML_TYPE_I64;
    if (normalized == "q4_0") return GGML_TYPE_Q4_0;
    if (normalized == "q4_1") return GGML_TYPE_Q4_1;
    if (normalized == "q5_0") return GGML_TYPE_Q5_0;
    if (normalized == "q5_1") return GGML_TYPE_Q5_1;
    if (normalized == "q8_0") return GGML_TYPE_Q8_0;
    if (normalized == "q2_k") return GGML_TYPE_Q2_K;
    if (normalized == "q3_k") return GGML_TYPE_Q3_K;
    if (normalized == "q4_k") return GGML_TYPE_Q4_K;
    if (normalized == "q5_k") return GGML_TYPE_Q5_K;
    if (normalized == "q6_k") return GGML_TYPE_Q6_K;
    if (normalized == "nvfp4") return GGML_TYPE_NVFP4;
    if (normalized == "iq2_xxs") return GGML_TYPE_IQ2_XXS;
    if (normalized == "iq2_xs") return GGML_TYPE_IQ2_XS;
    if (normalized == "iq2_s") return GGML_TYPE_IQ2_S;
    if (normalized == "iq3_xxs") return GGML_TYPE_IQ3_XXS;
    if (normalized == "iq3_s") return GGML_TYPE_IQ3_S;
    if (normalized == "iq4_nl") return GGML_TYPE_IQ4_NL;
    if (normalized == "iq4_xs") return GGML_TYPE_IQ4_XS;
    throw std::runtime_error("unsupported tensor dtype for GGUF: " + std::string(dtype));
}

std::string dtype_for_ggml_type(ggml_type type) {
    const char * name = ggml_type_name(type);
    if (name == nullptr) {
        throw std::runtime_error("GGUF tensor has an unknown ggml type");
    }
    return lower_ascii(name);
}

void validate_raw_tensor_byte_size(std::string_view name, const core::TensorShape & shape, ggml_type type, size_t bytes) {
    const size_t expected = static_cast<size_t>(shape.prefix_elements()) *
                            ggml_row_size(type, shape.last_dim());
    if (bytes != expected) {
        throw std::runtime_error("tensor byte size mismatch for " + std::string(name));
    }
}

std::vector<std::byte> f32_bytes(const std::vector<float> & values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<std::byte> f16_bytes(const std::vector<float> & values) {
    std::vector<ggml_fp16_t> f16_values(values.size());
    ggml_fp32_to_fp16_row(values.data(), f16_values.data(), static_cast<int64_t>(values.size()));
    std::vector<std::byte> bytes(values.size() * sizeof(ggml_fp16_t));
    std::memcpy(bytes.data(), f16_values.data(), bytes.size());
    return bytes;
}

std::vector<std::byte> bf16_bytes(const std::vector<float> & values) {
    std::vector<ggml_bf16_t> bf16_values(values.size());
    ggml_fp32_to_bf16_row(values.data(), bf16_values.data(), static_cast<int64_t>(values.size()));
    std::vector<std::byte> bytes(values.size() * sizeof(ggml_bf16_t));
    std::memcpy(bytes.data(), bf16_values.data(), bytes.size());
    return bytes;
}

std::vector<std::byte> quantize_f32_rows(
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type) {
    if (!ggml_is_quantized(type)) {
        throw std::runtime_error("tensor quantization target is not a quantized ggml type");
    }
    if (ggml_quantize_requires_imatrix(type)) {
        throw std::runtime_error("tensor quantization target requires an importance matrix: " + std::string(name));
    }
    if (shape.rank < 2) {
        throw std::runtime_error("quantized tensor must have rank >= 2: " + std::string(name));
    }
    const int64_t elements_per_row = shape.last_dim();
    if (elements_per_row % ggml_blck_size(type) != 0) {
        throw std::runtime_error("quantized tensor row size is not divisible by block size: " + std::string(name));
    }
    const int64_t rows = shape.prefix_elements();
    if (rows <= 0 || elements_per_row <= 0 ||
        static_cast<size_t>(rows * elements_per_row) != values.size()) {
        throw std::runtime_error("quantized tensor shape does not match F32 value count: " + std::string(name));
    }
    std::vector<std::byte> bytes(static_cast<size_t>(rows) * ggml_row_size(type, elements_per_row));

    // ⚠ ROWS IN PARALLEL. ggml_quantize_chunk takes an ELEMENT offset and writes
    // at (offset / n_per_row) * row_size, so a chunk reads and writes only its
    // own rows and the slices never touch. This is the whole cost of a
    // conversion -- a k-quant searches per block -- and left serial it pegged
    // one core while the other fifteen idled: bf16, which only changes format,
    // took 37s on an 8B model where q4_k took over seventeen minutes of CPU.
    //
    // ggml_quantize_init is called inside ggml_quantize_chunk and takes a
    // critical section, so it is safe from several threads; calling it once here
    // keeps the workers from queueing on it.
    ggml_quantize_init(type);

    // Chunked by a row count rather than by a thread count, and with no
    // num_threads clause, so OMP_NUM_THREADS still decides how many run -- the
    // same arrangement the f32 conversions above use. Sizing the split to the
    // hardware instead would quietly override a caller who had limited it.
    const int64_t rows_per_chunk = std::max<int64_t>(1, kQuantizeChunkElements / elements_per_row);
    const int64_t chunks = (rows + rows_per_chunk - 1) / rows_per_chunk;

    std::vector<size_t> written(static_cast<size_t>(chunks), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(chunks > 1)
#endif
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        const int64_t first_row = chunk * rows_per_chunk;
        const int64_t count = std::min(rows_per_chunk, rows - first_row);
        written[static_cast<size_t>(chunk)] = ggml_quantize_chunk(
            type,
            values.data(),
            bytes.data(),
            first_row * elements_per_row,
            count,
            elements_per_row,
            nullptr);
    }

    const size_t total = std::accumulate(written.begin(), written.end(), size_t{0});
    if (total != bytes.size()) {
        throw std::runtime_error("quantized tensor byte size mismatch: " + std::string(name));
    }
    return bytes;
}

void set_tensor_bytes(ggml_tensor * tensor, const void * data, size_t bytes, std::string_view name) {
    if (tensor == nullptr) {
        throw std::runtime_error("cannot upload to a null backend tensor: " + std::string(name));
    }
    if (bytes != ggml_nbytes(tensor)) {
        throw std::runtime_error("backend tensor byte size mismatch for " + std::string(name));
    }
    ggml_backend_tensor_set(tensor, data, 0, bytes);
}

std::vector<float> decode_tensor_data_f32(std::string_view name, const TensorData & tensor) {
    if (tensor.type == GGML_TYPE_F32) {
        if (tensor.bytes.size() != static_cast<size_t>(tensor.shape.num_elements()) * sizeof(float)) {
            throw std::runtime_error("invalid F32 tensor byte size: " + std::string(name));
        }
        std::vector<float> values(static_cast<size_t>(tensor.shape.num_elements()));
        std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
        return values;
    }
    if (tensor.type == GGML_TYPE_F16) {
        if (tensor.bytes.size() != static_cast<size_t>(tensor.shape.num_elements()) * sizeof(ggml_fp16_t)) {
            throw std::runtime_error("invalid F16 tensor byte size: " + std::string(name));
        }
        std::vector<float> values(static_cast<size_t>(tensor.shape.num_elements()));
        ggml_fp16_to_fp32_row(
            reinterpret_cast<const ggml_fp16_t *>(tensor.bytes.data()),
            values.data(),
            tensor.shape.num_elements());
        return values;
    }
    if (tensor.type == GGML_TYPE_BF16) {
        if (tensor.bytes.size() != static_cast<size_t>(tensor.shape.num_elements()) * sizeof(ggml_bf16_t)) {
            throw std::runtime_error("invalid BF16 tensor byte size: " + std::string(name));
        }
        std::vector<float> values(static_cast<size_t>(tensor.shape.num_elements()));
        ggml_bf16_to_fp32_row(
            reinterpret_cast<const ggml_bf16_t *>(tensor.bytes.data()),
            values.data(),
            tensor.shape.num_elements());
        return values;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor.type);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::runtime_error("tensor type is not readable as F32 data: " + std::string(name));
    }
    if (tensor.shape.rank < 2) {
        throw std::runtime_error("quantized tensor must have rank >= 2: " + std::string(name));
    }
    const int64_t cols = tensor.shape.last_dim();
    const int64_t rows = tensor.shape.prefix_elements();
    const size_t row_bytes = ggml_row_size(tensor.type, cols);
    if (tensor.bytes.size() != static_cast<size_t>(rows) * row_bytes) {
        throw std::runtime_error("quantized tensor byte size mismatch: " + std::string(name));
    }
    std::vector<float> values(static_cast<size_t>(rows * cols));
    const std::byte * src = tensor.bytes.data();
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(
            src + static_cast<std::ptrdiff_t>(row * static_cast<int64_t>(row_bytes)),
            values.data() + static_cast<std::ptrdiff_t>(row * cols),
            cols);
    }
    return values;
}

void set_backend_tensor_from_f32(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type,
    std::vector<std::byte> * retained_bytes = nullptr) {
    if (type == GGML_TYPE_F32) {
        set_tensor_bytes(tensor, values.data(), values.size() * sizeof(float), name);
        if (retained_bytes) *retained_bytes = f32_bytes(values);
        return;
    }
    if (type == GGML_TYPE_F16) {
        auto bytes = f16_bytes(values);
        set_tensor_bytes(tensor, bytes.data(), bytes.size(), name);
        if (retained_bytes) *retained_bytes = std::move(bytes);
        return;
    }
    if (type == GGML_TYPE_BF16) {
        auto bytes = bf16_bytes(values);
        set_tensor_bytes(tensor, bytes.data(), bytes.size(), name);
        if (retained_bytes) *retained_bytes = std::move(bytes);
        return;
    }
    auto bytes = quantize_f32_rows(name, values, shape, type);
    set_tensor_bytes(tensor, bytes.data(), bytes.size(), name);
    if (retained_bytes) *retained_bytes = std::move(bytes);
}

}  // namespace

void set_backend_tensor_from_f32_parallel(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type,
    std::vector<std::byte> * retained_bytes) {
    if (static_cast<int64_t>(values.size()) < kParallelF32ConvertElements ||
        (type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)) {
        set_backend_tensor_from_f32(tensor, name, values, shape, type, retained_bytes);
        return;
    }

    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> converted(values.size());
        const int64_t count = static_cast<int64_t>(values.size());
        const int64_t chunks = (count + kF32ConvertChunkElements - 1) / kF32ConvertChunkElements;
        #pragma omp parallel for schedule(static)
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            const int64_t offset = chunk * kF32ConvertChunkElements;
            const int64_t length = std::min(kF32ConvertChunkElements, count - offset);
            ggml_fp32_to_fp16_row(values.data() + offset, converted.data() + offset, length);
        }
        set_tensor_bytes(tensor, converted.data(), converted.size() * sizeof(ggml_fp16_t), name);
        if (retained_bytes) {
            const auto * begin = reinterpret_cast<const std::byte *>(converted.data());
            retained_bytes->assign(begin, begin + converted.size() * sizeof(ggml_fp16_t));
        }
        return;
    }

    std::vector<ggml_bf16_t> converted(values.size());
    const int64_t count = static_cast<int64_t>(values.size());
    const int64_t chunks = (count + kF32ConvertChunkElements - 1) / kF32ConvertChunkElements;
    #pragma omp parallel for schedule(static)
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        const int64_t offset = chunk * kF32ConvertChunkElements;
        const int64_t length = std::min(kF32ConvertChunkElements, count - offset);
        ggml_fp32_to_bf16_row(values.data() + offset, converted.data() + offset, length);
    }
    set_tensor_bytes(tensor, converted.data(), converted.size() * sizeof(ggml_bf16_t), name);
    if (retained_bytes) {
        const auto * begin = reinterpret_cast<const std::byte *>(converted.data());
        retained_bytes->assign(begin, begin + converted.size() * sizeof(ggml_bf16_t));
    }
}

namespace {

std::vector<std::byte> encode_f32_tensor_data(
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return f32_bytes(values);
    }
    if (type == GGML_TYPE_F16) {
        return f16_bytes(values);
    }
    if (type == GGML_TYPE_BF16) {
        return bf16_bytes(values);
    }
    return quantize_f32_rows(name, values, shape, type);
}

bool dtype_equals(std::string_view actual, std::string_view expected) {
    return lower_ascii(actual) == lower_ascii(expected);
}

int64_t checked_element_count(std::string_view name, const std::vector<int64_t> & shape) {
    int64_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("tensor shape contains a non-positive dimension: " + std::string(name));
        }
        if (count > std::numeric_limits<int64_t>::max() / dim) {
            throw std::runtime_error("tensor element count overflow: " + std::string(name));
        }
        count *= dim;
    }
    return count;
}

std::vector<uint8_t> raw_u8_values(const RawTensorData & data) {
    if (!dtype_equals(data.metadata.dtype, "U8")) {
        throw std::runtime_error("BNB NF4 tensor dtype mismatch for " + data.metadata.name);
    }
    std::vector<uint8_t> values(data.bytes.size());
    std::memcpy(values.data(), data.bytes.data(), data.bytes.size());
    return values;
}

std::vector<float> raw_f32_values(const RawTensorData & data) {
    if (!dtype_equals(data.metadata.dtype, "F32")) {
        throw std::runtime_error("BNB NF4 tensor dtype mismatch for " + data.metadata.name);
    }
    if (data.bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("BNB NF4 F32 tensor byte size mismatch: " + data.metadata.name);
    }
    std::vector<float> values(data.bytes.size() / sizeof(float));
    std::memcpy(values.data(), data.bytes.data(), data.bytes.size());
    return values;
}

struct BnbNf4QuantState {
    std::vector<int64_t> shape;
    int64_t blocksize = 0;
    int64_t nested_blocksize = 0;
    float nested_offset = 0.0F;
};

BnbNf4QuantState parse_bnb_nf4_quant_state(const RawTensorData & data) {
    const auto bytes = raw_u8_values(data);
    const auto root = engine::io::json::parse(std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
    BnbNf4QuantState state;
    const auto quant_type = engine::io::json::require_string(root, "quant_type");
    const auto dtype = engine::io::json::require_string(root, "dtype");
    state.shape = engine::io::json::require_i64_array(root, "shape");
    state.blocksize = engine::io::json::require_i64(root, "blocksize");
    state.nested_blocksize = engine::io::json::require_i64(root, "nested_blocksize");
    state.nested_offset = engine::io::json::require_f32(root, "nested_offset");
    if (quant_type != "nf4") {
        throw std::runtime_error("BNB quant_state is not NF4: " + data.metadata.name);
    }
    if (state.blocksize <= 0 || state.nested_blocksize <= 0) {
        throw std::runtime_error("BNB NF4 quant_state contains invalid block sizes: " + data.metadata.name);
    }
    if (dtype != "bfloat16" && dtype != "float16" && dtype != "float32") {
        throw std::runtime_error("BNB NF4 quant_state contains unsupported dequant dtype: " + dtype);
    }
    return state;
}

std::string bnb_nf4_quant_state_name(std::string_view name) {
    return std::string(name) + ".quant_state.bitsandbytes__nf4";
}

std::optional<std::string> bnb_nf4_helper_base_name(const TensorSource & source, std::string_view name) {
    static constexpr std::string_view suffixes[] = {
        ".absmax",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_map",
        ".quant_state.bitsandbytes__nf4",
    };
    const std::string value(name);
    for (const auto suffix : suffixes) {
        if (value.size() <= suffix.size() ||
            value.compare(value.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        const std::string base = value.substr(0, value.size() - suffix.size());
        if (source.has_tensor(base) && source.has_tensor(bnb_nf4_quant_state_name(base))) {
            return base;
        }
    }
    return std::nullopt;
}

bool is_bnb_nf4_weight(const TensorSource & source, const TensorMetadata & metadata) {
    return dtype_equals(metadata.dtype, "U8") && source.has_tensor(bnb_nf4_quant_state_name(metadata.name));
}

RawTensorData convert_bnb_nf4_weight(
    const TensorSource & source,
    const TensorMetadata & metadata,
    TensorStorageType storage_type) {
    const ggml_type output_type = ggml_type_for_tensor_storage(storage_type);
    const auto packed = source.require_tensor_data(metadata.name);
    const auto state = parse_bnb_nf4_quant_state(source.require_tensor_data(bnb_nf4_quant_state_name(metadata.name)));
    const int64_t elements = checked_element_count(metadata.name, state.shape);
    if (packed.metadata.shape != std::vector<int64_t>{(elements + 1) / 2, 1}) {
        throw std::runtime_error("BNB NF4 packed tensor shape mismatch: " + metadata.name);
    }
    const auto absmax = raw_u8_values(source.require_tensor_data(std::string(metadata.name) + ".absmax"));
    const auto nested_absmax = raw_f32_values(source.require_tensor_data(std::string(metadata.name) + ".nested_absmax"));
    const auto nested_quant_map =
        raw_f32_values(source.require_tensor_data(std::string(metadata.name) + ".nested_quant_map"));
    const auto quant_map = raw_f32_values(source.require_tensor_data(std::string(metadata.name) + ".quant_map"));
    if (nested_quant_map.size() != 256 || quant_map.size() != 16) {
        throw std::runtime_error("BNB NF4 quant maps have unexpected size: " + metadata.name);
    }
    const int64_t expected_nested =
        (static_cast<int64_t>(absmax.size()) + state.nested_blocksize - 1) / state.nested_blocksize;
    if (static_cast<int64_t>(nested_absmax.size()) != expected_nested) {
        throw std::runtime_error("BNB NF4 nested absmax length mismatch: " + metadata.name);
    }
    std::vector<float> scales(absmax.size());
    for (size_t i = 0; i < absmax.size(); ++i) {
        scales[i] = nested_quant_map[absmax[i]] *
            nested_absmax[static_cast<size_t>(static_cast<int64_t>(i) / state.nested_blocksize)] +
            state.nested_offset;
    }
    const auto packed_values = raw_u8_values(packed);
    if (static_cast<int64_t>(packed_values.size()) != (elements + 1) / 2) {
        throw std::runtime_error("BNB NF4 packed byte count mismatch: " + metadata.name);
    }
    const int64_t block_count = (elements + state.blocksize - 1) / state.blocksize;
    if (static_cast<int64_t>(scales.size()) != block_count) {
        throw std::runtime_error("BNB NF4 absmax block count mismatch: " + metadata.name);
    }
    std::vector<float> values(static_cast<size_t>(elements));
    int64_t out = 0;
    for (const uint8_t byte : packed_values) {
        values[static_cast<size_t>(out)] =
            quant_map[(byte >> 4U) & 0x0FU] * scales[static_cast<size_t>(out / state.blocksize)];
        ++out;
        if (out < elements) {
            values[static_cast<size_t>(out)] =
                quant_map[byte & 0x0FU] * scales[static_cast<size_t>(out / state.blocksize)];
            ++out;
        }
    }
    RawTensorData out_data;
    out_data.metadata = {metadata.name, dtype_for_ggml_type(output_type), state.shape};
    out_data.bytes = encode_f32_tensor_data(metadata.name, values, shape_from_dims(state.shape), output_type);
    return out_data;
}

class SafeTensorSource final : public TensorSource {
public:
    SafeTensorSource(std::filesystem::path path, TensorSourceOptions options)
        : index_(engine::io::load_safetensors_index(path)),
          synchronized_access_(options.synchronized_access), bytes_(engine::io::read_binary_blob(path)) {}

    const std::filesystem::path & source_path() const noexcept override {
        return index_.source_path;
    }

    bool has_tensor(std::string_view name) const noexcept override {
        return index_.tensors.find(std::string(name)) != index_.tensors.end();
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        return TensorMetadata{info->name, info->dtype, info->shape};
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(index_.tensors.size());
        for (const auto & [name, info] : index_.tensors) {
            out.push_back({name, info.dtype, info.shape});
        }
        std::sort(out.begin(), out.end(), [](const TensorMetadata & lhs, const TensorMetadata & rhs) {
            return lhs.name < rhs.name;
        });
        return out;
    }

    void release_storage() const override {
        auto lock = lock_storage();
        bytes_ = engine::io::BinaryBlob();
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        auto lock = lock_storage();
        const auto [data, byte_size] = require_data_range(*info);
        RawTensorData tensor;
        tensor.metadata = TensorMetadata{info->name, info->dtype, info->shape};
        tensor.bytes.resize(byte_size);
        std::memcpy(tensor.bytes.data(), data, byte_size);
        discard_data_range(*info);
        return tensor;
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        validate_expected_shape(name, info->shape, expected_shape);
        const auto shape = shape_from_dims(expected_shape);
        const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
        auto lock = lock_storage();
        const auto [data, byte_size] = require_data_range(*info);
        if (raw_dtype_matches_ggml_type(info->dtype, type)) {
            validate_raw_tensor_byte_size(name, shape, type, byte_size);
            set_tensor_bytes(tensor, data, byte_size, name);
            discard_data_range(*info);
            return;
        }
        const ggml_type raw_type = ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(info->dtype));
        std::vector<std::byte> raw_bytes(byte_size);
        std::memcpy(raw_bytes.data(), data, byte_size);
        const auto values = decode_tensor_data_f32(name, TensorData{shape, raw_type, std::move(raw_bytes)});
        set_backend_tensor_from_f32(tensor, name, values, shape, type);
        discard_data_range(*info);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        set_backend_tensor(tensor, name, TensorStorageType::F32, expected_shape);
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto tensor = require_tensor_data(name);
        validate_expected_shape(name, tensor.metadata.shape, expected_shape);
        const ggml_type type = ggml_type_for_tensor_dtype(tensor.metadata.dtype);
        const auto physical_shape = tensor.metadata.shape.empty()
            ? shape_from_dims({1})
            : shape_from_dims(tensor.metadata.shape);
        return decode_tensor_data_f32(name, TensorData{physical_shape, type, tensor.bytes});
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        if (info->dtype != "I64" || info->data_end - info->data_begin != sizeof(int64_t)) {
            throw std::runtime_error("tensor is not an I64 scalar: " + std::string(name));
        }
        auto lock = lock_storage();
        const auto [data, byte_size] = require_data_range(*info);
        (void) byte_size;
        int64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

private:
    const engine::io::SafeTensorInfo * find_info(std::string_view name) const noexcept {
        const auto it = index_.tensors.find(std::string(name));
        if (it == index_.tensors.end()) {
            return nullptr;
        }
        return &it->second;
    }

    std::pair<const std::byte *, size_t> require_data_range(const engine::io::SafeTensorInfo & info) const {
        if (bytes_.empty()) {
            bytes_ = engine::io::read_binary_blob(index_.source_path);
        }
        const size_t byte_size = info.data_end - info.data_begin;
        // Written as subtractions against the total so nothing can overflow.
        // `header_bytes + data_begin + byte_size > total` wraps if any term is
        // large, and a wrapped comparison passes -- handing back a span over
        // memory past the end of the blob. Parse-time validation now rejects the
        // inputs that made that reachable; this is the last gate before a raw
        // pointer escapes, so it checks anyway.
        const size_t total = bytes_.size();
        if (index_.header_bytes > total || byte_size > total - index_.header_bytes ||
            info.data_begin > total - index_.header_bytes - byte_size) {
            throw std::runtime_error("tensor data range is out of bounds: " + info.name);
        }
        const size_t data_offset = index_.header_bytes + info.data_begin;
        return {bytes_.data() + static_cast<std::ptrdiff_t>(data_offset), byte_size};
    }

    void discard_data_range(const engine::io::SafeTensorInfo & info) const noexcept {
        bytes_.discard_range(index_.header_bytes + info.data_begin, info.data_end - info.data_begin);
    }

    engine::io::SafeTensorIndex index_;
    // Keep mappings alive until each copy or synchronous backend upload completes.
    const bool synchronized_access_;
    mutable std::mutex storage_mutex_;
    std::unique_lock<std::mutex> lock_storage() const {
        std::unique_lock<std::mutex> lock(storage_mutex_, std::defer_lock);
        if (synchronized_access_) { lock.lock(); }
        return lock;
    }
    mutable engine::io::BinaryBlob bytes_;
};

struct GgufTensorInfo {
    std::string logical_name;
    std::string physical_name;
    std::string dtype;
    std::vector<int64_t> shape;
    ggml_type type = GGML_TYPE_F32;
    size_t data_offset = 0;
    size_t byte_size = 0;
};

class GgufTensorSource final : public TensorSource {
public:
    GgufTensorSource(std::filesystem::path path, TensorSourceOptions options)
        : source_path_(engine::io::normalize_path(path)), synchronized_access_(options.synchronized_access) {
        ggml_context * tensor_context = nullptr;
        gguf_context * gguf = gguf_init_from_file(
            source_path_.string().c_str(),
            gguf_init_params{true, &tensor_context});
        if (gguf == nullptr || tensor_context == nullptr) {
            if (gguf != nullptr) gguf_free(gguf);
            if (tensor_context != nullptr) ggml_free(tensor_context);
            throw std::runtime_error("failed to read GGUF tensor source: " + source_path_.string());
        }
        try {
            data_begin_ = gguf_get_data_offset(gguf);
            const int64_t logical_names_key = gguf_find_key(gguf, "audiocpp.tensor_names");
            if (logical_names_key >= 0 &&
                (gguf_get_kv_type(gguf, logical_names_key) != GGUF_TYPE_ARRAY ||
                 gguf_get_arr_type(gguf, logical_names_key) != GGUF_TYPE_STRING ||
                 gguf_get_arr_n(gguf, logical_names_key) != static_cast<size_t>(gguf_get_n_tensors(gguf)))) {
                throw std::runtime_error("GGUF audiocpp.tensor_names metadata is invalid");
            }
            const int64_t count = gguf_get_n_tensors(gguf);
            const int64_t ranks_key = gguf_find_key(gguf, "audiocpp.tensor_ranks");
            const int64_t shapes_key = gguf_find_key(gguf, "audiocpp.tensor_shapes");
            const bool has_exact_shapes = ranks_key >= 0 && shapes_key >= 0;
            if ((ranks_key >= 0) != (shapes_key >= 0) ||
                (has_exact_shapes &&
                 (gguf_get_kv_type(gguf, ranks_key) != GGUF_TYPE_ARRAY ||
                  gguf_get_arr_type(gguf, ranks_key) != GGUF_TYPE_INT32 ||
                  gguf_get_arr_n(gguf, ranks_key) != static_cast<size_t>(count) ||
                  gguf_get_kv_type(gguf, shapes_key) != GGUF_TYPE_ARRAY ||
                  gguf_get_arr_type(gguf, shapes_key) != GGUF_TYPE_INT64))) {
                throw std::runtime_error("GGUF audiocpp exact tensor shape metadata is invalid");
            }
            size_t shape_cursor = 0;
            infos_.reserve(static_cast<size_t>(count));
            for (int64_t i = 0; i < count; ++i) {
                GgufTensorInfo info;
                info.physical_name = gguf_get_tensor_name(gguf, i);
                info.logical_name = logical_names_key >= 0
                    ? gguf_get_arr_str(gguf, logical_names_key, static_cast<size_t>(i))
                    : info.physical_name;
                info.type = gguf_get_tensor_type(gguf, i);
                info.dtype = dtype_for_ggml_type(info.type);
                info.data_offset = gguf_get_tensor_offset(gguf, i);
                info.byte_size = gguf_get_tensor_size(gguf, i);
                const ggml_tensor * tensor = ggml_get_tensor(tensor_context, info.physical_name.c_str());
                if (tensor == nullptr) {
                    throw std::runtime_error("GGUF tensor metadata is missing: " + info.physical_name);
                }
                if (has_exact_shapes) {
                    const auto * ranks = static_cast<const int32_t *>(gguf_get_arr_data(gguf, ranks_key));
                    const auto * shapes = static_cast<const int64_t *>(gguf_get_arr_data(gguf, shapes_key));
                    const int32_t tensor_rank = ranks[i];
                    if (tensor_rank < 0 || tensor_rank > static_cast<int32_t>(core::kMaxTensorRank) ||
                        shape_cursor + static_cast<size_t>(tensor_rank) > gguf_get_arr_n(gguf, shapes_key)) {
                        throw std::runtime_error("GGUF contains invalid exact tensor dimensions");
                    }
                    if (tensor_rank > 0) {
                        info.shape.assign(shapes + shape_cursor, shapes + shape_cursor + tensor_rank);
                    } else {
                        info.shape.clear();
                    }
                    shape_cursor += static_cast<size_t>(tensor_rank);
                } else {
                    const int dimensions = ggml_n_dims(tensor);
                    info.shape.reserve(static_cast<size_t>(dimensions));
                    for (int dim = dimensions - 1; dim >= 0; --dim) {
                        info.shape.push_back(tensor->ne[dim]);
                    }
                }
                if (!info_by_name_.emplace(info.logical_name, infos_.size()).second) {
                    throw std::runtime_error("GGUF contains duplicate logical tensor name: " + info.logical_name);
                }
                infos_.push_back(std::move(info));
            }
            if (has_exact_shapes && shape_cursor != gguf_get_arr_n(gguf, shapes_key)) {
                throw std::runtime_error("GGUF exact tensor shape metadata has trailing dimensions");
            }
        } catch (...) {
            gguf_free(gguf);
            ggml_free(tensor_context);
            throw;
        }
        gguf_free(gguf);
        ggml_free(tensor_context);
        bytes_ = engine::io::read_binary_blob(source_path_);
    }

    const std::filesystem::path & source_path() const noexcept override { return source_path_; }

    bool has_tensor(std::string_view name) const noexcept override {
        return info_by_name_.find(std::string(name)) != info_by_name_.end();
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        const auto & info = require_info(name);
        return {info.logical_name, info.dtype, info.shape};
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(infos_.size());
        for (const auto & info : infos_) out.push_back({info.logical_name, info.dtype, info.shape});
        std::sort(out.begin(), out.end(), [](const TensorMetadata & lhs, const TensorMetadata & rhs) {
            return lhs.name < rhs.name;
        });
        return out;
    }

    void release_storage() const override {
        auto lock = lock_storage();
        bytes_ = engine::io::BinaryBlob();
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & info = require_info(name);
        auto lock = lock_storage();
        const auto [data, byte_size] = require_data_range(info);
        RawTensorData out;
        out.metadata = {info.logical_name, info.dtype, info.shape};
        out.bytes.resize(byte_size);
        std::memcpy(out.bytes.data(), data, byte_size);
        bytes_.discard_range(data_begin_ + info.data_offset, byte_size);
        return out;
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        const auto & info = require_info(name);
        validate_expected_shape(name, info.shape, expected_shape);
        const auto shape = shape_from_dims(expected_shape);
        const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
        auto lock = lock_storage();
        const auto [data, byte_size] = require_data_range(info);
        if (info.type == type) {
            validate_raw_tensor_byte_size(name, shape, type, byte_size);
            set_tensor_bytes(tensor, data, byte_size, name);
            bytes_.discard_range(data_begin_ + info.data_offset, byte_size);
            return;
        }
        std::vector<std::byte> raw_bytes(byte_size);
        std::memcpy(raw_bytes.data(), data, byte_size);
        const auto values = decode_tensor_data_f32(name, TensorData{shape, info.type, std::move(raw_bytes)});
        set_backend_tensor_from_f32(tensor, name, values, shape, type);
        bytes_.discard_range(data_begin_ + info.data_offset, byte_size);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        set_backend_tensor(tensor, name, TensorStorageType::F32, expected_shape);
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto tensor = require_tensor_data(name);
        validate_expected_shape(name, tensor.metadata.shape, expected_shape);
        const auto physical_shape = tensor.metadata.shape.empty()
            ? shape_from_dims({1})
            : shape_from_dims(tensor.metadata.shape);
        return decode_tensor_data_f32(
            name,
            TensorData{physical_shape, require_info(name).type, tensor.bytes});
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        const auto & info = require_info(name);
        if (info.type != GGML_TYPE_I64 || info.byte_size != sizeof(int64_t)) {
            throw std::runtime_error("tensor is not an I64 scalar: " + std::string(name));
        }
        auto lock = lock_storage();
        const auto [data, byte_size] = require_data_range(info);
        (void) byte_size;
        int64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

private:
    const GgufTensorInfo & require_info(std::string_view name) const {
        const auto it = info_by_name_.find(std::string(name));
        if (it == info_by_name_.end()) throw std::runtime_error("missing tensor: " + std::string(name));
        return infos_[it->second];
    }

    std::pair<const std::byte *, size_t> require_data_range(const GgufTensorInfo & info) const {
        if (bytes_.empty()) bytes_ = engine::io::read_binary_blob(source_path_);
        const size_t offset = data_begin_ + info.data_offset;
        if (offset > bytes_.size() || info.byte_size > bytes_.size() - offset) {
            throw std::runtime_error("GGUF tensor data range is out of bounds: " + info.logical_name);
        }
        return {bytes_.data() + static_cast<std::ptrdiff_t>(offset), info.byte_size};
    }

    std::filesystem::path source_path_;
    size_t data_begin_ = 0;
    std::vector<GgufTensorInfo> infos_;
    std::unordered_map<std::string, size_t> info_by_name_;
    // Sessions share this source, including lazy remapping and storage release.
    const bool synchronized_access_;
    mutable std::mutex storage_mutex_;
    std::unique_lock<std::mutex> lock_storage() const {
        std::unique_lock<std::mutex> lock(storage_mutex_, std::defer_lock);
        if (synchronized_access_) { lock.lock(); }
        return lock;
    }
    mutable engine::io::BinaryBlob bytes_;
};

std::unordered_map<std::string, std::string> parse_indexed_tensor_weight_map(
    const std::filesystem::path & index_path) {
    const auto root = engine::io::json::parse_file(index_path);
    const auto & object = root.require("weight_map").as_object();
    std::unordered_map<std::string, std::string> weight_map;
    weight_map.reserve(object.size());
    for (const auto & [name, value] : object) {
        weight_map.emplace(name, value.as_string());
    }
    if (weight_map.empty()) {
        throw std::runtime_error("indexed tensor source has an empty weight_map: " + index_path.string());
    }
    return weight_map;
}

std::vector<std::filesystem::path> indexed_tensor_source_shard_paths_from_weight_map(
    const std::filesystem::path & model_root,
    const std::unordered_map<std::string, std::string> & weight_map) {
    std::set<std::string> shard_names;
    for (const auto & [_, file_name] : weight_map) {
        shard_names.insert(file_name);
    }
    std::vector<std::filesystem::path> paths;
    paths.reserve(shard_names.size());
    for (const auto & name : shard_names) {
        const auto path = model_root / name;
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("missing indexed tensor shard: " + path.string());
        }
        paths.push_back(engine::io::normalize_path(path));
    }
    return paths;
}

class IndexedTensorSource final : public TensorSource {
public:
    IndexedTensorSource(
        std::filesystem::path index_path,
        std::unordered_map<std::string, std::string> weight_map,
        std::unordered_map<std::string, std::shared_ptr<const TensorSource>> shard_sources)
        : index_path_(std::move(index_path)),
          weight_map_(std::move(weight_map)),
          shard_sources_(std::move(shard_sources)) {}

    const std::filesystem::path & source_path() const noexcept override {
        return index_path_;
    }

    bool has_tensor(std::string_view name) const noexcept override {
        const auto route = weight_map_.find(std::string(name));
        if (route == weight_map_.end()) {
            return false;
        }
        const auto source = shard_sources_.find(route->second);
        return source != shard_sources_.end() && source->second->has_tensor(name);
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        return source_for(name)->require_metadata(name);
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(weight_map_.size());
        for (const auto & [name, _] : weight_map_) {
            out.push_back(require_metadata(name));
        }
        std::sort(out.begin(), out.end(), [](const TensorMetadata & lhs, const TensorMetadata & rhs) {
            return lhs.name < rhs.name;
        });
        return out;
    }

    void release_storage() const override {
        for (const auto & [_, source] : shard_sources_) {
            source->release_storage();
        }
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        return source_for(name)->require_tensor_data(name);
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return source_for(name)->require_f32(name, expected_shape);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        source_for(name)->set_backend_tensor(tensor, name, storage_type, expected_shape);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        source_for(name)->set_backend_f32_tensor(tensor, name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return source_for(name)->require_i64_scalar(name);
    }

private:
    std::shared_ptr<const TensorSource> source_for(std::string_view name) const {
        const auto route = weight_map_.find(std::string(name));
        if (route == weight_map_.end()) {
            throw std::runtime_error("missing indexed tensor route: " + std::string(name));
        }
        const auto source = shard_sources_.find(route->second);
        if (source == shard_sources_.end()) {
            throw std::runtime_error("missing indexed tensor shard source: " + route->second);
        }
        return source->second;
    }

    std::filesystem::path index_path_;
    std::unordered_map<std::string, std::string> weight_map_;
    std::unordered_map<std::string, std::shared_ptr<const TensorSource>> shard_sources_;
};

class CompositeTensorSource final : public TensorSource {
public:
    struct Component {
        std::string prefix;
        std::shared_ptr<const TensorSource> source;
    };

    CompositeTensorSource(std::filesystem::path source_path, std::vector<Component> components)
        : source_path_(std::move(source_path)), components_(std::move(components)) {
        for (size_t component_index = 0; component_index < components_.size(); ++component_index) {
            auto & component = components_[component_index];
            std::replace(component.prefix.begin(), component.prefix.end(), '\\', '/');
            while (!component.prefix.empty() && component.prefix.front() == '/') component.prefix.erase(0, 1);
            while (!component.prefix.empty() && component.prefix.back() == '/') component.prefix.pop_back();
            if (component.prefix == "." || component.prefix.find("..") != std::string::npos) {
                throw std::runtime_error("invalid packed GGUF tensor prefix: " + component.prefix);
            }
            for (const auto & tensor : component.source->tensors()) {
                const std::string logical_name = component.prefix.empty()
                    ? tensor.name
                    : component.prefix + "/" + tensor.name;
                if (!routes_.emplace(logical_name, Route{component_index, tensor.name}).second) {
                    throw std::runtime_error("duplicate packed GGUF tensor name: " + logical_name);
                }
            }
        }
    }

    const std::filesystem::path & source_path() const noexcept override { return source_path_; }
    bool has_tensor(std::string_view name) const noexcept override {
        return routes_.find(std::string(name)) != routes_.end();
    }
    TensorMetadata require_metadata(std::string_view name) const override {
        const auto & route = require_route(name);
        auto metadata = components_[route.component].source->require_metadata(route.name);
        metadata.name = std::string(name);
        return metadata;
    }
    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> result;
        result.reserve(routes_.size());
        for (const auto & [name, _] : routes_) result.push_back(require_metadata(name));
        std::sort(result.begin(), result.end(), [](const auto & lhs, const auto & rhs) { return lhs.name < rhs.name; });
        return result;
    }
    void release_storage() const override {
        for (const auto & component : components_) component.source->release_storage();
    }
    RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & route = require_route(name);
        auto data = components_[route.component].source->require_tensor_data(route.name);
        data.metadata.name = std::string(name);
        return data;
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto & route = require_route(name);
        return components_[route.component].source->require_f32(route.name, expected_shape);
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }
    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        const auto & route = require_route(name);
        components_[route.component].source->set_backend_tensor(tensor, route.name, storage_type, expected_shape);
    }
    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        set_backend_tensor(tensor, name, TensorStorageType::F32, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        const auto & route = require_route(name);
        return components_[route.component].source->require_i64_scalar(route.name);
    }

private:
    struct Route {
        size_t component = 0;
        std::string name;
    };
    const Route & require_route(std::string_view name) const {
        const auto it = routes_.find(std::string(name));
        if (it == routes_.end()) throw std::runtime_error("missing packed tensor: " + std::string(name));
        return it->second;
    }

    std::filesystem::path source_path_;
    std::vector<Component> components_;
    std::unordered_map<std::string, Route> routes_;
};

class PrefixedTensorSourceView final : public TensorSource {
public:
    PrefixedTensorSourceView(std::shared_ptr<const TensorSource> source, std::string prefix)
        : source_(std::move(source)), prefix_(std::move(prefix)) {
        std::replace(prefix_.begin(), prefix_.end(), '\\', '/');
        while (!prefix_.empty() && prefix_.front() == '/') prefix_.erase(0, 1);
        while (!prefix_.empty() && prefix_.back() == '/') prefix_.pop_back();
        if (prefix_.empty()) throw std::runtime_error("packed tensor source prefix is empty");
        const std::string marker = prefix_ + "/";
        for (const auto & tensor : source_->tensors()) {
            if (tensor.name.rfind(marker, 0) == 0) {
                routes_.emplace(tensor.name.substr(marker.size()), tensor.name);
            }
        }
        if (routes_.empty()) throw std::runtime_error("packed GGUF namespace does not exist: " + prefix_);
    }

    const std::filesystem::path & source_path() const noexcept override { return source_->source_path(); }
    bool has_tensor(std::string_view name) const noexcept override {
        return routes_.find(std::string(name)) != routes_.end();
    }
    TensorMetadata require_metadata(std::string_view name) const override {
        auto metadata = source_->require_metadata(require_name(name));
        metadata.name = std::string(name);
        return metadata;
    }
    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> result;
        result.reserve(routes_.size());
        for (const auto & [name, _] : routes_) result.push_back(require_metadata(name));
        std::sort(result.begin(), result.end(), [](const auto & lhs, const auto & rhs) { return lhs.name < rhs.name; });
        return result;
    }
    void release_storage() const override { source_->release_storage(); }
    RawTensorData require_tensor_data(std::string_view name) const override {
        auto data = source_->require_tensor_data(require_name(name));
        data.metadata.name = std::string(name);
        return data;
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return source_->require_f32(require_name(name), expected_shape);
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }
    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        source_->set_backend_tensor(tensor, require_name(name), storage_type, expected_shape);
    }
    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        set_backend_tensor(tensor, name, TensorStorageType::F32, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        return source_->require_i64_scalar(require_name(name));
    }

private:
    const std::string & require_name(std::string_view name) const {
        const auto it = routes_.find(std::string(name));
        if (it == routes_.end()) throw std::runtime_error("missing tensor in namespace " + prefix_ + ": " + std::string(name));
        return it->second;
    }
    std::shared_ptr<const TensorSource> source_;
    std::string prefix_;
    std::unordered_map<std::string, std::string> routes_;
};

}  // namespace

ggml_type ggml_type_for_tensor_dtype(std::string_view dtype) {
    return parse_ggml_type_for_tensor_dtype(dtype);
}

TensorDataF32 TensorSource::require_f32_tensor(std::string_view name) const {
    const auto metadata = require_metadata(name);
    return TensorDataF32{
        shape_from_dims(metadata.shape),
        require_f32(name, metadata.shape),
    };
}

TensorDataF32 TensorSource::require_f32_tensor(
    std::string_view name,
    std::initializer_list<int64_t> expected_shape) const {
    const auto values = require_f32(name, expected_shape);
    return TensorDataF32{
        shape_from_dims(std::vector<int64_t>(expected_shape)),
        std::move(values),
    };
}

TensorData TensorSource::require_tensor(
    std::string_view name,
    TensorStorageType storage_type,
    std::initializer_list<int64_t> expected_shape) const {
    return require_tensor_as_shape(name, storage_type, expected_shape, expected_shape);
}

TensorData TensorSource::require_tensor(
    std::string_view name,
    TensorStorageType storage_type,
    const std::vector<int64_t> & expected_shape) const {
    const core::TensorShape shape = shape_from_dims(expected_shape);
    const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
    const auto raw = require_tensor_data(name);
    validate_expected_shape(name, raw.metadata.shape, expected_shape);
    if (raw_dtype_matches_ggml_type(raw.metadata.dtype, type)) {
        validate_raw_tensor_byte_size(name, shape, type, raw.bytes.size());
        return TensorData{shape, type, raw.bytes};
    }
    const ggml_type raw_type = ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(raw.metadata.dtype));
    const auto values = decode_tensor_data_f32(name, TensorData{shape, raw_type, raw.bytes});
    return TensorData{shape, type, encode_f32_tensor_data(name, values, shape, type)};
}

TensorData TensorSource::require_tensor_as_shape(
    std::string_view name,
    TensorStorageType storage_type,
    std::initializer_list<int64_t> expected_source_shape,
    std::initializer_list<int64_t> tensor_shape) const {
    const std::vector<int64_t> expected(expected_source_shape);
    const core::TensorShape shape = shape_from_dims(std::vector<int64_t>(tensor_shape));
    if (shape.num_elements() != std::accumulate(expected.begin(), expected.end(), int64_t{1}, std::multiplies<int64_t>())) {
        throw std::runtime_error("tensor source shape element count mismatch for " + std::string(name));
    }
    const core::TensorShape source_shape = shape_from_dims(expected);
    const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
    const auto raw = require_tensor_data(name);
    validate_expected_shape(name, raw.metadata.shape, expected);
    if (raw.metadata.shape == std::vector<int64_t>(tensor_shape) &&
        raw_dtype_matches_ggml_type(raw.metadata.dtype, type)) {
        validate_raw_tensor_byte_size(name, shape, type, raw.bytes.size());
        return TensorData{shape, type, raw.bytes};
    }
    const ggml_type raw_type = ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(raw.metadata.dtype));
    const auto values = decode_tensor_data_f32(name, TensorData{source_shape, raw_type, raw.bytes});
    return TensorData{shape, type, encode_f32_tensor_data(name, values, shape, type)};
}

void TensorSource::set_backend_tensor(
    ggml_tensor * tensor,
    std::string_view name,
    TensorStorageType storage_type,
    const std::vector<int64_t> & expected_shape) const {
    TensorData data;
    switch (expected_shape.size()) {
        case 1:
            data = require_tensor(name, storage_type, {expected_shape[0]});
            break;
        case 2:
            data = require_tensor(name, storage_type, {expected_shape[0], expected_shape[1]});
            break;
        case 3:
            data = require_tensor(name, storage_type, {expected_shape[0], expected_shape[1], expected_shape[2]});
            break;
        case 4:
            data = require_tensor(
                name,
                storage_type,
                {expected_shape[0], expected_shape[1], expected_shape[2], expected_shape[3]});
            break;
        default:
            throw std::runtime_error("tensor rank must be between 1 and 4");
    }
    set_tensor_bytes(tensor, data.bytes.data(), data.bytes.size(), name);
}

void TensorSource::set_backend_f32_tensor(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<int64_t> & expected_shape) const {
    const auto values = require_f32(name, std::optional<std::vector<int64_t>>(expected_shape));
    set_tensor_bytes(tensor, values.data(), values.size() * sizeof(float), name);
}

std::optional<TensorDataF32> TensorSource::optional_f32_tensor(std::string_view name) const {
    if (!has_tensor(name)) {
        return std::nullopt;
    }
    return require_f32_tensor(name);
}

std::optional<TensorDataF32> TensorSource::optional_f32_tensor(
    std::string_view name,
    std::initializer_list<int64_t> expected_shape) const {
    if (!has_tensor(name)) {
        return std::nullopt;
    }
    return require_f32_tensor(name, expected_shape);
}

std::optional<RawTensorData> TensorSource::optional_tensor_data(std::string_view name) const {
    if (!has_tensor(name)) {
        return std::nullopt;
    }
    return require_tensor_data(name);
}

std::optional<std::string> TensorSource::find_tensor_name(
    std::initializer_list<std::string_view> candidates) const {
    for (const auto candidate : candidates) {
        if (has_tensor(candidate)) {
            return std::string(candidate);
        }
    }
    return std::nullopt;
}

std::string TensorSource::require_tensor_name(
    std::initializer_list<std::string_view> candidates) const {
    const auto match = find_tensor_name(candidates);
    if (match.has_value()) {
        return *match;
    }
    throw std::runtime_error("none of the candidate tensor names were found");
}

TensorStorageType parse_tensor_storage_type(std::string_view value) {
    const std::string normalized = lower_ascii(value);
    if (normalized == "native" || normalized == "source" || normalized == "auto" || normalized == "orig" ||
        normalized == "original") {
        return TensorStorageType::Native;
    }
    if (normalized == "f32" || normalized == "float32") {
        return TensorStorageType::F32;
    }
    if (normalized == "f16" || normalized == "float16" || normalized == "fp16") {
        return TensorStorageType::F16;
    }
    if (normalized == "bf16" || normalized == "bfloat16") {
        return TensorStorageType::BF16;
    }
    if (normalized == "q4_0") {
        return TensorStorageType::Q4_0;
    }
    if (normalized == "q4_1") {
        return TensorStorageType::Q4_1;
    }
    if (normalized == "q5_0") {
        return TensorStorageType::Q5_0;
    }
    if (normalized == "q5_1") {
        return TensorStorageType::Q5_1;
    }
    if (normalized == "q2_k") {
        return TensorStorageType::Q2_K;
    }
    if (normalized == "q3_k") {
        return TensorStorageType::Q3_K;
    }
    if (normalized == "q4_k") {
        return TensorStorageType::Q4_K;
    }
    if (normalized == "q5_k") {
        return TensorStorageType::Q5_K;
    }
    if (normalized == "q6_k") {
        return TensorStorageType::Q6_K;
    }
    if (normalized == "q8_0") {
        return TensorStorageType::Q8_0;
    }
    if (normalized == "nvfp4") {
        return TensorStorageType::NVFP4;
    }
    if (normalized == "iq2_xxs") {
        return TensorStorageType::IQ2_XXS;
    }
    if (normalized == "iq2_xs") {
        return TensorStorageType::IQ2_XS;
    }
    if (normalized == "iq2_s") {
        return TensorStorageType::IQ2_S;
    }
    if (normalized == "iq3_xxs") {
        return TensorStorageType::IQ3_XXS;
    }
    if (normalized == "iq3_s") {
        return TensorStorageType::IQ3_S;
    }
    if (normalized == "iq4_nl") {
        return TensorStorageType::IQ4_NL;
    }
    if (normalized == "iq4_xs") {
        return TensorStorageType::IQ4_XS;
    }
    throw std::runtime_error("unsupported tensor storage type: " + std::string(value));
}

ggml_type ggml_type_for_tensor_storage(TensorStorageType storage_type) {
    switch (storage_type) {
        case TensorStorageType::Native:
            throw std::runtime_error("native tensor storage type must be resolved before creating a ggml tensor");
        case TensorStorageType::F32:
            return GGML_TYPE_F32;
        case TensorStorageType::F16:
            return GGML_TYPE_F16;
        case TensorStorageType::BF16:
            return GGML_TYPE_BF16;
        case TensorStorageType::I8:
            return GGML_TYPE_I8;
        case TensorStorageType::Q4_0:
            return GGML_TYPE_Q4_0;
        case TensorStorageType::Q4_1:
            return GGML_TYPE_Q4_1;
        case TensorStorageType::Q5_0:
            return GGML_TYPE_Q5_0;
        case TensorStorageType::Q5_1:
            return GGML_TYPE_Q5_1;
        case TensorStorageType::Q2_K:
            return GGML_TYPE_Q2_K;
        case TensorStorageType::Q3_K:
            return GGML_TYPE_Q3_K;
        case TensorStorageType::Q4_K:
            return GGML_TYPE_Q4_K;
        case TensorStorageType::Q5_K:
            return GGML_TYPE_Q5_K;
        case TensorStorageType::Q6_K:
            return GGML_TYPE_Q6_K;
        case TensorStorageType::Q8_0:
            return GGML_TYPE_Q8_0;
        case TensorStorageType::NVFP4:
            return GGML_TYPE_NVFP4;
        case TensorStorageType::IQ2_XXS:
            return GGML_TYPE_IQ2_XXS;
        case TensorStorageType::IQ2_XS:
            return GGML_TYPE_IQ2_XS;
        case TensorStorageType::IQ2_S:
            return GGML_TYPE_IQ2_S;
        case TensorStorageType::IQ3_XXS:
            return GGML_TYPE_IQ3_XXS;
        case TensorStorageType::IQ3_S:
            return GGML_TYPE_IQ3_S;
        case TensorStorageType::IQ4_NL:
            return GGML_TYPE_IQ4_NL;
        case TensorStorageType::IQ4_XS:
            return GGML_TYPE_IQ4_XS;
    }
    throw std::runtime_error("unknown tensor storage type");
}

TensorStorageType tensor_storage_type_for_dtype(std::string_view dtype) {
    const std::string normalized = lower_ascii(dtype);
    if (normalized == "f32" || normalized == "float32") {
        return TensorStorageType::F32;
    }
    if (normalized == "f16" || normalized == "float16" || normalized == "fp16") {
        return TensorStorageType::F16;
    }
    if (normalized == "bf16" || normalized == "bfloat16") {
        return TensorStorageType::BF16;
    }
    if (normalized == "i8" || normalized == "int8") {
        return TensorStorageType::I8;
    }
    if (normalized == "q4_0") {
        return TensorStorageType::Q4_0;
    }
    if (normalized == "q4_1") {
        return TensorStorageType::Q4_1;
    }
    if (normalized == "q5_0") {
        return TensorStorageType::Q5_0;
    }
    if (normalized == "q5_1") {
        return TensorStorageType::Q5_1;
    }
    if (normalized == "q2_k") {
        return TensorStorageType::Q2_K;
    }
    if (normalized == "q3_k") {
        return TensorStorageType::Q3_K;
    }
    if (normalized == "q4_k") {
        return TensorStorageType::Q4_K;
    }
    if (normalized == "q5_k") {
        return TensorStorageType::Q5_K;
    }
    if (normalized == "q6_k") {
        return TensorStorageType::Q6_K;
    }
    if (normalized == "q8_0") {
        return TensorStorageType::Q8_0;
    }
    if (normalized == "nvfp4") {
        return TensorStorageType::NVFP4;
    }
    if (normalized == "iq2_xxs") {
        return TensorStorageType::IQ2_XXS;
    }
    if (normalized == "iq2_xs") {
        return TensorStorageType::IQ2_XS;
    }
    if (normalized == "iq2_s") {
        return TensorStorageType::IQ2_S;
    }
    if (normalized == "iq3_xxs") {
        return TensorStorageType::IQ3_XXS;
    }
    if (normalized == "iq3_s") {
        return TensorStorageType::IQ3_S;
    }
    if (normalized == "iq4_nl") {
        return TensorStorageType::IQ4_NL;
    }
    if (normalized == "iq4_xs") {
        return TensorStorageType::IQ4_XS;
    }
    throw std::runtime_error("unsupported native tensor dtype: " + std::string(dtype));
}

TensorStorageType resolve_tensor_storage_type(
    const TensorSource & source,
    std::string_view name,
    TensorStorageType requested_type) {
    if (requested_type != TensorStorageType::Native) {
        return requested_type;
    }
    return tensor_storage_type_for_dtype(source.require_metadata(name).dtype);
}

std::vector<float> tensor_data_to_f32(std::string_view name, const TensorData & tensor) {
    return decode_tensor_data_f32(name, tensor);
}

std::shared_ptr<const TensorSource> open_tensor_source(const std::filesystem::path & path) {
    return open_tensor_source(path, TensorSourceOptions{});
}

std::optional<std::filesystem::path> find_tensor_file(
    const std::filesystem::path & directory, std::string_view stem) {
    for (const auto & entry : kTensorContainers) {
        const auto candidate = directory / (std::string(stem) + std::string(entry.first));
        if (io::is_existing_file(candidate)) return candidate;
    }
    return std::nullopt;
}

std::shared_ptr<const TensorSource> open_tensor_source(
    const std::filesystem::path & path, TensorSourceOptions options) {
    const std::string file_name = lower_ascii(path.filename().string());
    static constexpr std::string_view kIndexSuffix = ".safetensors.index.json";
    if (file_name.size() >= kIndexSuffix.size() &&
        file_name.compare(file_name.size() - kIndexSuffix.size(), kIndexSuffix.size(), kIndexSuffix) == 0) {
        return open_indexed_tensor_source(path, path.parent_path(), options);
    }
    const std::string extension = lower_ascii(path.extension().string());
    for (const auto & entry : kTensorContainers) {
        if (extension != entry.first) continue;
        switch (entry.second) {
            case TensorContainer::Safetensors: return std::make_shared<SafeTensorSource>(path, options);
            // kcpp: PyTorch pickle/bin loading is converter baggage. Runtime
            // packages for selected models use safetensors or GGUF.
            case TensorContainer::Torch:
                throw std::runtime_error("PyTorch .bin tensor sources are not enabled in the KoboldCpp Audio.cpp integration: " +
                                         path.string());
            case TensorContainer::Gguf: return std::make_shared<GgufTensorSource>(path, options);
        }
    }
    throw std::runtime_error("unsupported tensor source format: " + path.string());
}

std::shared_ptr<const TensorSource> open_tensor_source(
    const std::filesystem::path & path,
    std::string_view tensor_prefix) {
    auto source = open_tensor_source(path);
    return make_prefixed_tensor_source(std::move(source), tensor_prefix);
}

std::shared_ptr<const TensorSource> open_tensor_source(
    const std::filesystem::path & path, std::string_view tensor_prefix, TensorSourceOptions options) {
    return make_prefixed_tensor_source(open_tensor_source(path, options), tensor_prefix);
}

std::shared_ptr<const TensorSource> make_prefixed_tensor_source(
    std::shared_ptr<const TensorSource> source,
    std::string_view tensor_prefix) {
    if (source == nullptr) {
        throw std::runtime_error("prefixed tensor source requires a source");
    }
    if (tensor_prefix.empty()) return source;
    return std::make_shared<PrefixedTensorSourceView>(
        std::move(source), std::string(tensor_prefix));
}

namespace {

// `with_contents == false` validates the same metadata but skips copying each
// sidecar's bytes out of the blob -- 38 MB for the Kokoro package, per call, when
// the caller only wants the names. Validation is deliberately not skipped with it:
// a caller that reads the names of a malformed package must fail the way a caller
// that reads its contents does.
std::vector<std::pair<std::string, std::string>> read_gguf_embedded_sidecars(
    const std::filesystem::path & path,
    bool with_contents) {
    ggml_context * tensor_context = nullptr;
    gguf_context * gguf = gguf_init_from_file(
        path.string().c_str(), gguf_init_params{true, &tensor_context});
    if (gguf == nullptr) {
        if (tensor_context != nullptr) ggml_free(tensor_context);
        throw std::runtime_error("failed to read GGUF metadata: " + path.string());
    }
    std::vector<std::pair<std::string, std::string>> result;
    try {
        const int64_t names_key = gguf_find_key(gguf, "audiocpp.embedded_files.names");
        const int64_t contents_key = gguf_find_key(gguf, "audiocpp.embedded_files.contents");
        const int64_t offsets_key = gguf_find_key(gguf, "audiocpp.embedded_files.offsets");
        const int64_t data_key = gguf_find_key(gguf, "audiocpp.embedded_files.data");
        const bool blob_layout = offsets_key >= 0 || data_key >= 0;
        if (names_key < 0 && contents_key < 0 && !blob_layout) {
            gguf_free(gguf);
            if (tensor_context != nullptr) ggml_free(tensor_context);
            return result;
        }
        if (blob_layout) {
            if (names_key < 0 || offsets_key < 0 || data_key < 0 ||
                gguf_get_kv_type(gguf, names_key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, names_key) != GGUF_TYPE_STRING ||
                gguf_get_kv_type(gguf, offsets_key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, offsets_key) != GGUF_TYPE_UINT64 ||
                gguf_get_arr_n(gguf, offsets_key) != gguf_get_arr_n(gguf, names_key) + 1 ||
                gguf_get_kv_type(gguf, data_key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, data_key) != GGUF_TYPE_UINT8) {
                throw std::runtime_error("GGUF embedded binary sidecar metadata is invalid");
            }
        } else if (names_key < 0 || contents_key < 0 ||
            gguf_get_kv_type(gguf, names_key) != GGUF_TYPE_ARRAY ||
            gguf_get_kv_type(gguf, contents_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(gguf, names_key) != GGUF_TYPE_STRING ||
            gguf_get_arr_type(gguf, contents_key) != GGUF_TYPE_STRING ||
            gguf_get_arr_n(gguf, names_key) != gguf_get_arr_n(gguf, contents_key)) {
            throw std::runtime_error("GGUF embedded sidecar metadata is invalid");
        }
        const size_t count = gguf_get_arr_n(gguf, names_key);
        const auto * offsets = blob_layout
            ? static_cast<const uint64_t *>(gguf_get_arr_data(gguf, offsets_key))
            : nullptr;
        const auto * data = blob_layout
            ? static_cast<const char *>(gguf_get_arr_data(gguf, data_key))
            : nullptr;
        const size_t data_size = blob_layout ? gguf_get_arr_n(gguf, data_key) : 0;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            std::string name = gguf_get_arr_str(gguf, names_key, i);
            const std::filesystem::path relative(name);
            const auto normalized = relative.lexically_normal();
            if (name.empty() || relative.is_absolute() || normalized.empty() ||
                *normalized.begin() == "..") {
                throw std::runtime_error("GGUF contains an unsafe embedded sidecar name: " + name);
            }
            if (blob_layout) {
                if (offsets[i] > offsets[i + 1] || offsets[i + 1] > data_size) {
                    throw std::runtime_error("GGUF embedded sidecar byte range is invalid");
                }
                std::string content;
                const size_t length = static_cast<size_t>(offsets[i + 1] - offsets[i]);
                if (with_contents && length > 0) content.assign(data + offsets[i], length);
                result.emplace_back(normalized.generic_string(), std::move(content));
            } else {
                result.emplace_back(
                    normalized.generic_string(),
                    with_contents ? std::string(gguf_get_arr_str(gguf, contents_key, i)) : std::string());
            }
        }
    } catch (...) {
        gguf_free(gguf);
        if (tensor_context != nullptr) ggml_free(tensor_context);
        throw;
    }
    gguf_free(gguf);
    if (tensor_context != nullptr) ggml_free(tensor_context);
    return result;
}

}  // namespace

bool gguf_has_embedded_sidecars(const std::filesystem::path & path) {
    return !read_gguf_embedded_sidecars(path, false).empty();
}

std::vector<std::string> gguf_embedded_sidecar_names(const std::filesystem::path & path) {
    const auto sidecars = read_gguf_embedded_sidecars(path, false);
    std::vector<std::string> names;
    names.reserve(sidecars.size());
    for (const auto & sidecar : sidecars) names.push_back(sidecar.first);
    return names;
}

std::optional<GgufEmbeddedModelSpec> read_gguf_embedded_model_spec(const std::filesystem::path & path) {
    ggml_context * tensor_context = nullptr;
    gguf_context * gguf = gguf_init_from_file(path.string().c_str(), gguf_init_params{true, &tensor_context});
    if (gguf == nullptr) {
        if (tensor_context != nullptr)
            ggml_free(tensor_context);
        throw std::runtime_error("failed to read GGUF metadata: " + path.string());
    }
    std::optional<GgufEmbeddedModelSpec> result;
    try {
        const int64_t version_key = gguf_find_key(gguf, "audiocpp.model_spec.version");
        const int64_t family_key = gguf_find_key(gguf, "audiocpp.model_spec.family");
        const int64_t json_key = gguf_find_key(gguf, "audiocpp.model_spec.json");
        if (version_key >= 0 || family_key >= 0 || json_key >= 0) {
            if (version_key < 0 || family_key < 0 || json_key < 0 ||
                gguf_get_kv_type(gguf, version_key) != GGUF_TYPE_UINT32 ||
                gguf_get_kv_type(gguf, family_key) != GGUF_TYPE_STRING ||
                gguf_get_kv_type(gguf, json_key) != GGUF_TYPE_STRING) {
                throw std::runtime_error("GGUF embedded model spec metadata is invalid");
            }
            const uint32_t version = gguf_get_val_u32(gguf, version_key);
            if (version != 1) {
                throw std::runtime_error("unsupported GGUF embedded model spec version: " +
                                         std::to_string(version));
            }
            GgufEmbeddedModelSpec spec;
            spec.family = gguf_get_val_str(gguf, family_key);
            spec.json = gguf_get_val_str(gguf, json_key);
            if (spec.family.empty() || spec.json.empty()) {
                throw std::runtime_error("GGUF embedded model spec is empty");
            }
            result = std::move(spec);
        }
    } catch (...) {
        gguf_free(gguf);
        if (tensor_context != nullptr)
            ggml_free(tensor_context);
        throw;
    }
    gguf_free(gguf);
    if (tensor_context != nullptr)
        ggml_free(tensor_context);
    return result;
}

std::filesystem::path materialize_gguf_sidecars(const std::filesystem::path & path) {
    const auto canonical = engine::io::normalize_path(path);
    const auto sidecars = read_gguf_embedded_sidecars(canonical, true);
    if (sidecars.empty()) {
        throw std::runtime_error("GGUF does not contain embedded model sidecars: " + canonical.string());
    }
    const auto fingerprint_source =
        canonical.string() + ":" + std::to_string(std::filesystem::file_size(canonical)) + ":" +
        std::to_string(static_cast<long long>(std::filesystem::last_write_time(canonical).time_since_epoch().count()));
    std::ostringstream fingerprint;
    fingerprint << std::hex << std::hash<std::string>{}(fingerprint_source);
    const auto root = std::filesystem::temp_directory_path() / "audiocpp-gguf" / fingerprint.str();
    std::filesystem::create_directories(root);
    for (const auto & [name, content] : sidecars) {
        const auto output_path = root / name;
        if (engine::io::is_existing_file(output_path) && engine::io::read_text_file(output_path) == content) {
            continue;
        }
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("failed to materialize GGUF sidecar: " + output_path.string());
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!output)
            throw std::runtime_error("failed to write GGUF sidecar: " + output_path.string());
    }
    return root;
}

std::vector<std::filesystem::path> directory_gguf_files(const std::filesystem::path & directory) {
    std::vector<std::filesystem::path> files;
    if (!engine::io::is_existing_directory(directory)) {
        return files;
    }
    for (const auto & entry : std::filesystem::directory_iterator(directory)) {
        const auto & candidate = entry.path();
        if (engine::io::is_existing_file(candidate) && lower_ascii(candidate.extension().string()) == ".gguf") {
            files.push_back(candidate);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::optional<std::filesystem::path> find_directory_gguf(const std::filesystem::path & directory) {
    const auto named = directory / "model.gguf";
    if (engine::io::is_existing_file(named)) {
        return named;
    }
    auto files = directory_gguf_files(directory);
    if (files.size() != 1) {
        return std::nullopt;
    }
    return files.front();
}

PreparedModelDirectory prepare_model_directory(const std::filesystem::path & model_path,
    const std::filesystem::path & gguf_relative_path) {
    std::filesystem::path gguf_path;
    if (engine::io::is_existing_directory(model_path)) {
        gguf_path = model_path / gguf_relative_path;
        if (!engine::io::is_existing_file(gguf_path)) {
            gguf_path = find_directory_gguf(model_path).value_or(gguf_path);
        }
    } else if (engine::io::is_existing_file(model_path)) {
        gguf_path = model_path;
    } else {
        throw std::runtime_error("model path does not exist: " + model_path.string());
    }
    if (engine::io::is_existing_file(gguf_path) && lower_ascii(gguf_path.extension().string()) == ".gguf") {
        const auto canonical_gguf = engine::io::normalize_path(gguf_path);
        if (gguf_has_embedded_sidecars(gguf_path)) {
            return {materialize_gguf_sidecars(gguf_path), canonical_gguf};
        }
        const auto external_root =
            engine::io::is_existing_directory(model_path) ? model_path : model_path.parent_path();
        return {engine::io::normalize_path(external_root), canonical_gguf};
    }
    const auto root = engine::io::is_existing_directory(model_path) ? model_path : model_path.parent_path();
    return {engine::io::normalize_path(root), std::nullopt};
}

void convert_tensor_sources_to_gguf(const std::vector<TensorSourceInput> & inputs,
                                    const std::filesystem::path & output_path, TensorStorageType weight_type,
                                    bool overwrite, bool embed_sidecars,
                                    const std::filesystem::path & requested_sidecar_root,
                                    const std::vector<GgufEmbeddedFile> & extra_sidecars,
                                    const std::optional<GgufEmbeddedModelSpec> & model_spec,
                                    GgufConversionOptions options) {
    if (inputs.empty()) {
        throw std::runtime_error("GGUF conversion requires at least one tensor source");
    }
    if (engine::io::is_existing_file(output_path) && !overwrite) {
        throw std::runtime_error("GGUF output already exists: " + output_path.string());
    }
    std::vector<CompositeTensorSource::Component> components;
    components.reserve(inputs.size());
    for (const auto & input : inputs) {
        if (!engine::io::is_existing_file(input.path)) {
            throw std::runtime_error("input tensor source does not exist: " + input.path.string());
        }
        try {
            components.push_back({input.tensor_prefix, open_tensor_source(input.path)});
        } catch (const std::exception & error) {
            throw std::runtime_error("failed to open tensor source " + input.path.string() + ": " + error.what());
        }
    }
    const auto sidecar_root = engine::io::normalize_path(
        requested_sidecar_root.empty() ? inputs.front().path.parent_path() : requested_sidecar_root);
    if (!engine::io::is_existing_directory(sidecar_root)) {
        throw std::runtime_error("GGUF sidecar root is not a directory: " + sidecar_root.string());
    }
    std::shared_ptr<const TensorSource> source =
        std::make_shared<CompositeTensorSource>(sidecar_root, std::move(components));
    if (!options.folded_weight_norm_patterns.empty()) {
        source = make_weight_norm_folded_tensor_source(
            std::move(source),
            options.folded_weight_norm_patterns);
    }
    std::unordered_set<std::string> adapted_tensors;
    if (options.weight_adapter) {
        const auto & adapter = *options.weight_adapter;
        const auto mode = options.weight_adapter_merge_mode.value_or(adapter.updates.empty()
            ? WeightMergeMode::DeltaF32 : adapter.updates.begin()->second.merge_mode);
        WeightAdapterBindings bindings;
        for (const auto & tensor : source->tensors()) {
            bindings.add(tensor.name, tensor.name);
        }
        for (const auto & entry : adapter.updates) {
            if (entry.second.merge_mode != mode) {
                throw std::runtime_error("GGUF conversion requires one adapter merge policy");
            }
        }
        auto plan = bind_weight_adapter(*source, adapter, bindings, mode);
        for (const auto & entry : plan.updates) adapted_tensors.insert(entry.first);
        for (const auto & entry : plan.replacements) adapted_tensors.insert(entry.first);
        source = make_weight_adapter_source(std::move(source), std::move(plan), "gguf.lora");
    }
    const auto metadata = source->tensors();
    if (metadata.empty()) {
        throw std::runtime_error("cannot convert an empty tensor source to GGUF");
    }
    auto is_excluded = [&options](std::string_view name) {
        for (const auto & prefix : options.excluded_tensor_prefixes) {
            if (!prefix.empty() && name.rfind(prefix, 0) == 0) {
                return true;
            }
        }
        return false;
    };
    std::vector<TensorMetadata> convertible_metadata;
    convertible_metadata.reserve(metadata.size());
    for (const auto & item : metadata) {
        const bool zero_element_tensor =
            std::any_of(item.shape.begin(), item.shape.end(), [](int64_t dim) { return dim == 0; });
        if (zero_element_tensor || is_excluded(item.name)) {
            continue;
        }
        convertible_metadata.push_back(item);
    }
    if (convertible_metadata.empty()) {
        throw std::runtime_error("cannot convert a tensor source with only zero-element tensors to GGUF");
    }

    const bool preserve_source_dtype = weight_type == TensorStorageType::Native;
    const ggml_type requested_type = preserve_source_dtype ? GGML_TYPE_COUNT : ggml_type_for_tensor_storage(weight_type);
    const bool convert_bnb_nf4 = options.bnb_nf4_type.has_value();
    const ggml_type requested_bnb_nf4_type =
        convert_bnb_nf4 ? ggml_type_for_tensor_storage(*options.bnb_nf4_type) : GGML_TYPE_COUNT;
    if (!preserve_source_dtype && ggml_is_quantized(requested_type) && ggml_quantize_requires_imatrix(requested_type)) {
        throw std::runtime_error("requested GGUF quantization requires an importance matrix");
    }
    if (convert_bnb_nf4 &&
        (!ggml_is_quantized(requested_bnb_nf4_type) || ggml_quantize_requires_imatrix(requested_bnb_nf4_type))) {
        throw std::runtime_error("BNB NF4 GGUF conversion currently requires a row-wise quantized target type");
    }
    if (convert_bnb_nf4) {
        convertible_metadata.clear();
        convertible_metadata.reserve(metadata.size());
        for (const auto & item : metadata) {
            const bool zero_element_tensor =
                std::any_of(item.shape.begin(), item.shape.end(), [](int64_t dim) { return dim == 0; });
            if (zero_element_tensor || is_excluded(item.name) ||
                bnb_nf4_helper_base_name(*source, item.name).has_value()) {
                continue;
            }
            if (is_bnb_nf4_weight(*source, item)) {
                const auto state =
                    parse_bnb_nf4_quant_state(source->require_tensor_data(bnb_nf4_quant_state_name(item.name)));
                convertible_metadata.push_back(
                    {item.name, dtype_for_ggml_type(requested_bnb_nf4_type), state.shape});
            } else {
                convertible_metadata.push_back(item);
            }
        }
        if (convertible_metadata.empty()) {
            throw std::runtime_error("cannot convert a tensor source with no storable tensors to GGUF");
        }
    }

    const size_t context_bytes = std::max<size_t>(
        1024 * 1024,
        convertible_metadata.size() * (ggml_tensor_overhead() + 256));
    ggml_context * tensor_context = ggml_init(ggml_init_params{context_bytes, nullptr, true});
    if (tensor_context == nullptr) {
        throw std::runtime_error("failed to allocate GGUF tensor metadata context");
    }
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        ggml_free(tensor_context);
        throw std::runtime_error("failed to allocate GGUF context");
    }

    struct OutputTensor {
        TensorMetadata metadata;
        std::string physical_name;
        ggml_type type = GGML_TYPE_F32;
        TensorStorageType storage_type = TensorStorageType::Native;
    };
    std::vector<OutputTensor> outputs;
    outputs.reserve(convertible_metadata.size());
    std::vector<std::string> logical_names;
    logical_names.reserve(convertible_metadata.size());
    std::unordered_set<std::string> physical_names;

    try {
        gguf_set_val_str(gguf, "general.architecture", "audiocpp");
        gguf_set_val_str(gguf, "general.name", sidecar_root.filename().string().c_str());
        gguf_set_val_str(gguf, "audiocpp.tensor_name_format", "native");
        gguf_set_val_str(gguf, "audiocpp.source_format",
                         inputs.size() == 1 ? lower_ascii(inputs.front().path.extension().string()).c_str() : "packed");
        const std::string output_weight_type =
            preserve_source_dtype ? "orig" : lower_ascii(ggml_type_name(requested_type));
        gguf_set_val_str(gguf, "audiocpp.weight_type", output_weight_type.c_str());
        if (convert_bnb_nf4) {
            gguf_set_val_str(gguf, "audiocpp.bnb_nf4_type", ggml_type_name(requested_bnb_nf4_type));
        }
        if (model_spec.has_value()) {
            gguf_set_val_u32(gguf, "audiocpp.model_spec.version", 1);
            gguf_set_val_str(gguf, "audiocpp.model_spec.family", model_spec->family.c_str());
            gguf_set_val_str(gguf, "audiocpp.model_spec.json", model_spec->json.c_str());
        }

        std::vector<std::string> source_names;
        std::vector<std::string> source_paths;
        std::vector<const char *> source_name_ptrs;
        std::vector<const char *> source_path_ptrs;
        source_names.reserve(inputs.size());
        source_paths.reserve(inputs.size());
        for (const auto & input : inputs) {
            source_names.push_back(input.tensor_prefix);
            std::error_code relative_error;
            const auto relative_path = std::filesystem::relative(input.path, sidecar_root, relative_error);
            source_paths.push_back((relative_error || relative_path.empty()
                ? std::filesystem::absolute(input.path).lexically_normal()
                : relative_path).generic_string());
        }
        for (const auto & value : source_names) source_name_ptrs.push_back(value.c_str());
        for (const auto & value : source_paths) source_path_ptrs.push_back(value.c_str());
        gguf_set_arr_str(gguf, "audiocpp.tensor_sources.names", source_name_ptrs.data(), source_name_ptrs.size());
        gguf_set_arr_str(gguf, "audiocpp.tensor_sources.paths", source_path_ptrs.data(), source_path_ptrs.size());

        std::vector<std::string> embedded_names;
        std::vector<const char *> embedded_name_ptrs;
        std::vector<uint64_t> embedded_offsets{0};
        std::vector<uint8_t> embedded_data;
        if (embed_sidecars) {
            constexpr uintmax_t kMaxEmbeddedSidecarBytes = 64u * 1024u * 1024u;
            std::vector<std::filesystem::path> sidecar_paths;
            std::error_code walk_error;
            std::unordered_set<std::string> explicit_destinations;
            for (const auto & sidecar : extra_sidecars) {
                const auto normalized = sidecar.destination.lexically_normal();
                if (normalized.empty() || normalized.is_absolute() || *normalized.begin() == "..") {
                    throw std::runtime_error("invalid embedded sidecar destination: " + sidecar.destination.string());
                }
                if (!engine::io::is_existing_file(sidecar.source_path)) {
                    throw std::runtime_error("embedded sidecar source does not exist: " + sidecar.source_path.string());
                }
                if (!explicit_destinations.insert(normalized.generic_string()).second) {
                    throw std::runtime_error(
                        "duplicate embedded sidecar destination: " + normalized.generic_string());
                }
            }
            for (std::filesystem::recursive_directory_iterator it(sidecar_root, walk_error), end;
                 !walk_error && it != end; it.increment(walk_error)) {
                if (it->is_directory() &&
                    (it->path().filename() == ".cache" || it->path().filename() == ".git")) {
                    it.disable_recursion_pending();
                    continue;
                }
                if (!it->is_regular_file()) continue;
                const auto & path = it->path();
                const std::string extension = lower_ascii(path.extension().string());
                if (extension == ".safetensors" || extension == ".gguf" || extension == ".bin" ||
                    extension == ".pt" || extension == ".pth" || path == output_path ||
                    it->file_size() > kMaxEmbeddedSidecarBytes) {
                    continue;
                }
                const auto relative = std::filesystem::relative(path, sidecar_root).generic_string();
                if (explicit_destinations.find(relative) != explicit_destinations.end()) continue;
                sidecar_paths.push_back(path);
            }
            if (walk_error) {
                throw std::runtime_error("failed to enumerate model sidecars: " + walk_error.message());
            }
            std::sort(sidecar_paths.begin(), sidecar_paths.end());
            for (const auto & path : sidecar_paths) {
                const auto relative = std::filesystem::relative(path, sidecar_root);
                const std::string content = engine::io::read_text_file(path);
                embedded_names.push_back(relative.generic_string());
                embedded_data.insert(embedded_data.end(), content.begin(), content.end());
                embedded_offsets.push_back(static_cast<uint64_t>(embedded_data.size()));
            }
            for (const auto & sidecar : extra_sidecars) {
                const std::string content = engine::io::read_text_file(sidecar.source_path);
                embedded_names.push_back(sidecar.destination.lexically_normal().generic_string());
                embedded_data.insert(embedded_data.end(), content.begin(), content.end());
                embedded_offsets.push_back(static_cast<uint64_t>(embedded_data.size()));
            }
            if (embedded_names.empty()) {
                throw std::runtime_error("no GGUF sidecars were found; provide "
                                         "--root/--sidecar for a standalone GGUF "
                                         "or pass --no-sidecars to explicitly create a "
                                         "tensor-only container");
            }
            embedded_name_ptrs.reserve(embedded_names.size());
            for (const auto & value : embedded_names)
                embedded_name_ptrs.push_back(value.c_str());
            if (!embedded_names.empty()) {
                gguf_set_arr_str(gguf, "audiocpp.embedded_files.names", embedded_name_ptrs.data(),
                                 embedded_name_ptrs.size());
                gguf_set_arr_data(gguf, "audiocpp.embedded_files.offsets", GGUF_TYPE_UINT64, embedded_offsets.data(),
                                  embedded_offsets.size());
                gguf_set_arr_data(gguf, "audiocpp.embedded_files.data", GGUF_TYPE_UINT8, embedded_data.data(),
                                  embedded_data.size());
            }
        }

        for (size_t i = 0; i < convertible_metadata.size(); ++i) {
            const auto & item = convertible_metadata[i];
            if (item.shape.size() > core::kMaxTensorRank) {
                throw std::runtime_error("GGUF supports tensor ranks from 0 to 4: " + item.name);
            }
            const bool bnb_nf4_weight = convert_bnb_nf4 && source->has_tensor(bnb_nf4_quant_state_name(item.name));
            const ggml_type source_type = bnb_nf4_weight
                ? requested_bnb_nf4_type
                : ggml_type_for_tensor_dtype(item.dtype);
            const bool source_is_float = source_type == GGML_TYPE_F32 || source_type == GGML_TYPE_F16 ||
                source_type == GGML_TYPE_BF16 || ggml_is_quantized(source_type);
            const std::string normalized_name = lower_ascii(item.name);
            const bool name_is_weight = item.name.size() >= 7 &&
                item.name.compare(item.name.size() - 7, 7, ".weight") == 0;
            const bool is_lookup_table = normalized_name.find("embed") != std::string::npos ||
                normalized_name.find("codebook") != std::string::npos;
            const bool can_quantize = !preserve_source_dtype && source_is_float && name_is_weight && item.shape.size() == 2 &&
                !is_lookup_table && item.shape.back() % ggml_blck_size(requested_type) == 0;
            const bool use_requested = !preserve_source_dtype &&
                (!ggml_is_quantized(requested_type) ? source_is_float && !item.shape.empty() : can_quantize);
            const bool use_f16_lookup = !preserve_source_dtype && ggml_is_quantized(requested_type) &&
                source_is_float && is_lookup_table;
            const auto type_override = find_tensor_type_override(item.name, options.type_overrides);
            const bool use_override = type_override.has_value() && *type_override != TensorStorageType::Native;
            const ggml_type override_type = use_override ? ggml_type_for_tensor_storage(*type_override) : source_type;
            if (use_override && ggml_is_quantized(override_type)) {
                // An explicit override may quantize [expert, output, input] banks.
                // Keep the default policy and ordinary 3-D kernels unchanged.
                const bool is_expert_bank = item.shape.size() == 3 && name_is_weight &&
                    normalized_name.find(".mlp.experts.") != std::string::npos;
                const bool can_quantize_override =
                    source_is_float && (item.shape.size() == 2 || is_expert_bank) &&
                    item.shape.back() % ggml_blck_size(override_type) == 0;
                if (!can_quantize_override) {
                    throw std::runtime_error("GGUF tensor type override cannot quantize tensor: " + item.name);
                }
            }
            const ggml_type output_type = bnb_nf4_weight
                ? requested_bnb_nf4_type
                : (type_override.has_value()
                ? override_type
                : (use_requested
                ? requested_type
                : (use_f16_lookup ? GGML_TYPE_F16 : source_type)));
            const TensorStorageType output_storage = bnb_nf4_weight
                ? *options.bnb_nf4_type
                : (type_override.has_value()
                ? *type_override
                : (use_requested
                ? weight_type
                : (use_f16_lookup ? TensorStorageType::F16 : TensorStorageType::Native)));

            std::string physical_name = item.name;
            if (physical_name.size() >= GGML_MAX_NAME || !physical_names.insert(physical_name).second) {
                physical_name = "_audiocpp." + std::to_string(i);
                if (!physical_names.insert(physical_name).second) {
                    throw std::runtime_error("failed to create a unique GGUF tensor alias");
                }
            }
            // GGML requires at least one physical dimension. Preserve a safetensors
            // rank-0 scalar as a one-element tensor and record rank 0 in the exact
            // audio.cpp shape metadata below.
            core::TensorShape shape = item.shape.empty()
                ? shape_from_dims({1})
                : shape_from_dims(item.shape);
            const auto dims = core::to_ggml_dims(shape);
            ggml_tensor * tensor = ggml_new_tensor(
                tensor_context,
                output_type,
                static_cast<int>(shape.rank),
                dims.data());
            if (tensor == nullptr) {
                throw std::runtime_error("failed to create GGUF tensor metadata: " + item.name);
            }
            ggml_set_name(tensor, physical_name.c_str());
            gguf_add_tensor(gguf, tensor);
            outputs.push_back({item, std::move(physical_name), output_type, output_storage});
            logical_names.push_back(item.name);
        }

        std::vector<const char *> logical_name_ptrs;
        logical_name_ptrs.reserve(logical_names.size());
        for (const auto & name : logical_names) logical_name_ptrs.push_back(name.c_str());
        gguf_set_arr_str(
            gguf,
            "audiocpp.tensor_names",
            logical_name_ptrs.data(),
            logical_name_ptrs.size());
        std::vector<int32_t> exact_ranks;
        std::vector<int64_t> exact_shapes;
        exact_ranks.reserve(convertible_metadata.size());
        for (const auto & item : convertible_metadata) {
            exact_ranks.push_back(static_cast<int32_t>(item.shape.size()));
            exact_shapes.insert(exact_shapes.end(), item.shape.begin(), item.shape.end());
        }
        gguf_set_arr_data(gguf, "audiocpp.tensor_ranks", GGUF_TYPE_INT32, exact_ranks.data(), exact_ranks.size());
        gguf_set_arr_data(gguf, "audiocpp.tensor_shapes", GGUF_TYPE_INT64, exact_shapes.data(), exact_shapes.size());

        const auto parent = output_path.parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        auto temporary_path = output_path;
        temporary_path += ".tmp";
        std::filesystem::remove(temporary_path);
        if (!gguf_write_to_file(gguf, temporary_path.string().c_str(), true)) {
            throw std::runtime_error("failed to write GGUF metadata: " + temporary_path.string());
        }

        try {
            std::ofstream output(temporary_path, std::ios::binary | std::ios::app);
            if (!output) throw std::runtime_error("failed to append GGUF tensor data");
            size_t cursor = 0;
            std::vector<char> padding;
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto & item = outputs[i];
                const size_t offset = gguf_get_tensor_offset(gguf, static_cast<int64_t>(i));
                if (offset < cursor) throw std::runtime_error("GGUF tensor offsets are not monotonic");
                padding.assign(offset - cursor, '\0');
                if (!padding.empty()) output.write(padding.data(), static_cast<std::streamsize>(padding.size()));

                RawTensorData raw;
                if (convert_bnb_nf4 && source->has_tensor(bnb_nf4_quant_state_name(item.metadata.name))) {
                    raw = convert_bnb_nf4_weight(*source, item.metadata, item.storage_type);
                } else if (item.storage_type == TensorStorageType::Native &&
                           adapted_tensors.count(item.metadata.name) == 0) {
                    raw = source->require_tensor_data(item.metadata.name);
                } else {
                    const auto tensor = source->require_tensor(
                        item.metadata.name,
                        item.storage_type,
                        item.metadata.shape);
                    raw.metadata = item.metadata;
                    raw.bytes = tensor.bytes;
                }
                const size_t expected_size = gguf_get_tensor_size(gguf, static_cast<int64_t>(i));
                if (raw.bytes.size() != expected_size) {
                    throw std::runtime_error("GGUF converted tensor byte size mismatch: " + item.metadata.name);
                }
                output.write(
                    reinterpret_cast<const char *>(raw.bytes.data()),
                    static_cast<std::streamsize>(raw.bytes.size()));
                if (!output) throw std::runtime_error("failed to write GGUF tensor: " + item.metadata.name);
                cursor = offset + raw.bytes.size();
            }
            output.close();
            if (!output) throw std::runtime_error("failed to finalize GGUF tensor data");
            if (engine::io::is_existing_file(output_path)) std::filesystem::remove(output_path);
            std::filesystem::rename(temporary_path, output_path);
        } catch (...) {
            std::filesystem::remove(temporary_path);
            throw;
        }
    } catch (...) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw;
    }
    gguf_free(gguf);
    ggml_free(tensor_context);
}

void convert_tensor_source_to_gguf(
    const std::filesystem::path & input_path,
    const std::filesystem::path & output_path,
    TensorStorageType weight_type,
    bool overwrite,
    bool embed_sidecars) {
    convert_tensor_sources_to_gguf(
        {{input_path, ""}}, output_path, weight_type, overwrite, embed_sidecars, input_path.parent_path());
}

std::vector<std::filesystem::path> indexed_tensor_source_shard_paths(
    const std::filesystem::path & index_path,
    const std::filesystem::path & model_root) {
    return indexed_tensor_source_shard_paths_from_weight_map(
        model_root,
        parse_indexed_tensor_weight_map(index_path));
}

std::shared_ptr<const TensorSource> open_indexed_tensor_source(
    const std::filesystem::path & index_path,
    const std::filesystem::path & model_root) {
    return open_indexed_tensor_source(index_path, model_root, TensorSourceOptions{});
}

std::shared_ptr<const TensorSource> open_indexed_tensor_source(
    const std::filesystem::path & index_path, const std::filesystem::path & model_root,
    TensorSourceOptions options) {
    const auto weight_map = parse_indexed_tensor_weight_map(index_path);
    std::unordered_map<std::string, std::shared_ptr<const TensorSource>> shard_sources;
    for (const auto & path : indexed_tensor_source_shard_paths_from_weight_map(model_root, weight_map)) {
        shard_sources.emplace(path.filename().string(), open_tensor_source(path, options));
    }
    return std::make_shared<IndexedTensorSource>(
        index_path,
        weight_map,
        std::move(shard_sources));
}

}  // namespace engine::assets
