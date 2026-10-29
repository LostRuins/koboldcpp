#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/load_progress.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/trace.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::core {

class BackendWeightStore {
public:
    BackendWeightStore(
        ggml_backend_t backend,
        BackendType backend_type,
        std::string name,
        size_t context_bytes,
        ggml_backend_buffer_type_t buffer_type = nullptr)
        : backend_(backend),
          backend_type_(backend_type),
          name_(std::move(name)),
          buffer_type_(buffer_type) {
        if (backend_ == nullptr) {
            throw std::runtime_error(name_ + " backend is not initialized");
        }
        if (context_bytes == 0) {
            throw std::runtime_error(name_ + " context bytes must be non-zero");
        }
        ggml_init_params params{context_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize " + name_ + " weight context");
        }
    }

    BackendWeightStore(const BackendWeightStore &) = delete;
    BackendWeightStore & operator=(const BackendWeightStore &) = delete;

    ~BackendWeightStore() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    TensorValue load_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        std::initializer_list<int64_t> expected_shape) {
        return load_tensor(source, tensor_name, storage_type, std::vector<int64_t>(expected_shape));
    }

    TensorValue load_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) {
        const auto shape = shape_from_dims(expected_shape);
        const auto resolved_type = type_for_storable_loaded_tensor(source, tensor_name, storage_type, shape);
        auto value = make_backend_tensor(shape, assets::ggml_type_for_tensor_storage(resolved_type));
        PendingUpload upload;
        upload.tensor = value.tensor;
        upload.source = &source;
        upload.name = std::string(tensor_name);
        upload.storage_type = resolved_type;
        upload.expected_shape = std::vector<int64_t>(expected_shape);
        upload.kind = PendingUploadKind::Tensor;
        pending_.push_back(std::move(upload));
        return value;
    }

    TensorValue load_tensor_as_shape(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_source_shape,
        const TensorShape & tensor_shape) {
        const auto resolved_type = type_for_storable_loaded_tensor(source, tensor_name, storage_type, tensor_shape);
        auto value = make_backend_tensor(tensor_shape, assets::ggml_type_for_tensor_storage(resolved_type));
        const auto data = require_tensor_as_shape(source, tensor_name, resolved_type, expected_source_shape, tensor_shape);
        append_upload(value.tensor, data.bytes);
        return value;
    }

    TensorValue load_f32_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        std::initializer_list<int64_t> expected_shape) {
        return load_f32_tensor(source, tensor_name, std::vector<int64_t>(expected_shape));
    }

    TensorValue load_f32_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        const std::vector<int64_t> & expected_shape) {
        const auto shape = shape_from_dims(expected_shape);
        auto value = make_backend_tensor(shape, GGML_TYPE_F32);
        PendingUpload upload;
        upload.tensor = value.tensor;
        upload.source = &source;
        upload.name = std::string(tensor_name);
        upload.storage_type = assets::TensorStorageType::F32;
        upload.expected_shape = std::vector<int64_t>(expected_shape);
        upload.kind = PendingUploadKind::F32;
        pending_.push_back(std::move(upload));
        return value;
    }

    TensorValue make_tensor(
        const TensorShape & shape,
        ggml_type type,
        const void * data,
        size_t bytes) {
        auto value = make_backend_tensor(shape, type);
        std::vector<std::byte> owned(bytes);
        std::memcpy(owned.data(), data, bytes);
        append_upload(value.tensor, std::move(owned));
        return value;
    }

    TensorValue make_f32(const TensorShape & shape, std::vector<float> values) {
        auto value = make_backend_tensor(shape, GGML_TYPE_F32);
        if (static_cast<int64_t>(values.size()) != shape.num_elements()) {
            throw std::runtime_error(name_ + " F32 tensor value count does not match shape");
        }
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        append_upload(value.tensor, std::move(bytes));
        return value;
    }

    TensorValue make_from_f32(
        const TensorShape & shape,
        assets::TensorStorageType storage_type,
        std::vector<float> values) {
        const ggml_type type = type_for_storable_derived_tensor(shape, storage_type);
        auto value = make_backend_tensor(shape, type);
        if (static_cast<int64_t>(values.size()) != shape.num_elements()) {
            throw std::runtime_error(name_ + " derived tensor value count does not match shape");
        }
        append_upload(value.tensor, values_to_bytes(shape, type, values));
        return value;
    }

    void upload() {
        if (buffer_ != nullptr) {
            throw std::runtime_error(name_ + " weights were already uploaded");
        }
        buffer_ = buffer_type_ != nullptr
            ? ggml_backend_alloc_ctx_tensors_from_buft(ctx_.get(), buffer_type_)
            : ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate " + name_ + " backend weight buffer");
        }
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // The header context is host RAM by construction; the buffer is where
        // the weights actually landed (device, or a host fallback), by name.
        debug::timing_log_context_reservation(name_, ctx_.get());
        debug::timing_log_scalar(
            name_ + ".buffer_mb",
            static_cast<double>(ggml_backend_buffer_get_size(buffer_)) / (1024.0 * 1024.0));
        debug::timing_log_scalar(name_ + ".buffer_name", std::string_view(ggml_backend_buffer_name(buffer_)));
        uint64_t pending_bytes = 0;
        for (const auto & upload : pending_) {
            pending_bytes += ggml_nbytes(upload.tensor);
        }
        engine::core::register_weight_bytes(pending_bytes);
        engine::core::emit_weight_upload_progress(name_);
        for (auto & upload : pending_) {
            if (upload.kind == PendingUploadKind::Tensor) {
                upload.source->set_backend_tensor(
                    upload.tensor,
                    upload.name,
                    upload.storage_type,
                    upload.expected_shape);
            } else if (upload.kind == PendingUploadKind::F32) {
                upload.source->set_backend_f32_tensor(upload.tensor, upload.name, upload.expected_shape);
            } else {
                upload_bytes(upload.tensor, upload.bytes);
                upload.bytes.clear();
                upload.bytes.shrink_to_fit();
            }
            // Per-tensor progress report (llama.cpp pace): the tensor reads and
            // backend copies below are the long pole of a model load, so each
            // finished tensor advances the uploaded-bytes curve.
            engine::core::add_uploaded_weight_bytes(ggml_nbytes(upload.tensor));
            engine::core::emit_weight_upload_progress(name_);
        }
        pending_.clear();
        pending_.shrink_to_fit();
    }

private:
    struct GgmlContextDeleter {
        void operator()(ggml_context * ctx) const noexcept {
            if (ctx != nullptr) {
                ggml_free(ctx);
            }
        }
    };

    enum class PendingUploadKind {
        Tensor,
        F32,
        Bytes,
    };

    struct PendingUpload {
        ggml_tensor * tensor = nullptr;
        const assets::TensorSource * source = nullptr;
        std::string name;
        assets::TensorStorageType storage_type = assets::TensorStorageType::F32;
        std::vector<int64_t> expected_shape;
        PendingUploadKind kind = PendingUploadKind::Bytes;
        std::vector<std::byte> bytes;
    };

    static TensorShape shape_from_dims(const std::vector<int64_t> & dims) {
        if (dims.size() == 0 || dims.size() > kMaxTensorRank) {
            throw std::runtime_error("weight tensor rank must be between 1 and 4");
        }
        auto it = dims.begin();
        switch (dims.size()) {
            case 1:
                return TensorShape::from_dims({*it});
            case 2: {
                const int64_t d0 = *it++;
                return TensorShape::from_dims({d0, *it});
            }
            case 3: {
                const int64_t d0 = *it++;
                const int64_t d1 = *it++;
                return TensorShape::from_dims({d0, d1, *it});
            }
            case 4: {
                const int64_t d0 = *it++;
                const int64_t d1 = *it++;
                const int64_t d2 = *it++;
                return TensorShape::from_dims({d0, d1, d2, *it});
            }
        }
        throw std::runtime_error("weight tensor rank is unsupported");
    }

    ggml_type type_for_derived_storage(assets::TensorStorageType storage_type) const {
        if (storage_type == assets::TensorStorageType::Native) {
            return GGML_TYPE_F32;
        }
        return assets::ggml_type_for_tensor_storage(storage_type);
    }

    assets::TensorStorageType type_for_storable_loaded_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const TensorShape & shape) const {
        const auto resolved_type = backend_safe_loaded_storage_type(source, tensor_name, storage_type);
        const ggml_type type = assets::ggml_type_for_tensor_storage(resolved_type);
        if (!ggml_is_quantized(type)) {
            return resolved_type;
        }
        // A GGUF may store a source matrix in quantized form even when a model
        // reshapes it into a convolution kernel. Quantized GGML rows cannot end
        // in a non-block-sized dimension (for example a 1-wide Conv1D kernel),
        // so decode such native tensors to F32 for the reshaped backend weight.
        if (shape.rank < 2 || shape.last_dim() % ggml_blck_size(type) != 0) {
            return assets::TensorStorageType::F32;
        }
        return resolved_type;
    }

    assets::TensorStorageType backend_safe_loaded_storage_type(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type) const {
        if (storage_type != assets::TensorStorageType::Native) {
            return storage_type;
        }
        const auto native_type = assets::tensor_storage_type_for_dtype(source.require_metadata(tensor_name).dtype);
        if ((backend_type_ == BackendType::Vulkan || backend_type_ == BackendType::Metal) &&
            native_type == assets::TensorStorageType::BF16) {
            return assets::TensorStorageType::F16;
        }
        return native_type;
    }

    ggml_type type_for_storable_derived_tensor(
        const TensorShape & shape,
        assets::TensorStorageType storage_type) const {
        const ggml_type type = type_for_derived_storage(storage_type);
        if (!ggml_is_quantized(type)) {
            return type;
        }
        // GGML quantized tensors are row-block encoded; small conv kernels and 1-D tensors
        // cannot be represented in Q8_0 layout, so derived F32 weights keep exact storage.
        if (shape.rank < 2 || shape.last_dim() % ggml_blck_size(type) != 0) {
            return GGML_TYPE_F32;
        }
        return type;
    }

    assets::TensorData require_tensor_as_shape(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & source_shape,
        const TensorShape & tensor_shape) const {
        if (source_shape.empty() || source_shape.size() > kMaxTensorRank) {
            throw std::runtime_error(name_ + " source tensor rank must be between 1 and 4");
        }
        if (tensor_shape.rank == 1) {
            if (source_shape.size() == 1) {
                return source.require_tensor_as_shape(
                    tensor_name, storage_type, {source_shape[0]}, {tensor_shape.dims[0]});
            }
        } else if (tensor_shape.rank == 2) {
            if (source_shape.size() == 2) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1]},
                    {tensor_shape.dims[0], tensor_shape.dims[1]});
            }
            if (source_shape.size() == 3) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1], source_shape[2]},
                    {tensor_shape.dims[0], tensor_shape.dims[1]});
            }
        } else if (tensor_shape.rank == 3) {
            if (source_shape.size() == 2) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1]},
                    {tensor_shape.dims[0], tensor_shape.dims[1], tensor_shape.dims[2]});
            }
            if (source_shape.size() == 3) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1], source_shape[2]},
                    {tensor_shape.dims[0], tensor_shape.dims[1], tensor_shape.dims[2]});
            }
        } else if (tensor_shape.rank == 4) {
            if (source_shape.size() == 4) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1], source_shape[2], source_shape[3]},
                    {tensor_shape.dims[0], tensor_shape.dims[1], tensor_shape.dims[2], tensor_shape.dims[3]});
            }
        }
        throw std::runtime_error(name_ + " unsupported tensor reshape rank");
    }

    std::vector<std::byte> values_to_bytes(
        const TensorShape & shape,
        ggml_type type,
        const std::vector<float> & values) const {
        if (type == GGML_TYPE_F32) {
            std::vector<std::byte> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            return bytes;
        }
        if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> converted(values.size());
            ggml_fp32_to_fp16_row(values.data(), converted.data(), static_cast<int64_t>(values.size()));
            std::vector<std::byte> bytes(converted.size() * sizeof(ggml_fp16_t));
            std::memcpy(bytes.data(), converted.data(), bytes.size());
            return bytes;
        }
        if (type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> converted(values.size());
            ggml_fp32_to_bf16_row(values.data(), converted.data(), static_cast<int64_t>(values.size()));
            std::vector<std::byte> bytes(converted.size() * sizeof(ggml_bf16_t));
            std::memcpy(bytes.data(), converted.data(), bytes.size());
            return bytes;
        }
        if (!ggml_is_quantized(type)) {
            throw std::runtime_error(name_ + " unsupported derived tensor storage type");
        }
        if (ggml_quantize_requires_imatrix(type)) {
            throw std::runtime_error(name_ + " derived tensor storage type requires an importance matrix");
        }
        if (shape.rank < 2) {
            throw std::runtime_error(name_ + " cannot quantize rank-1 derived tensor");
        }
        const int64_t elements_per_row = shape.last_dim();
        if (elements_per_row % ggml_blck_size(type) != 0) {
            throw std::runtime_error(name_ + " derived tensor row size is not divisible by quant block size");
        }
        const int64_t rows = shape.prefix_elements();
        std::vector<std::byte> bytes(static_cast<size_t>(rows) * ggml_row_size(type, elements_per_row));
        const size_t written = ggml_quantize_chunk(
            type,
            values.data(),
            bytes.data(),
            0,
            rows,
            elements_per_row,
            nullptr);
        if (written != bytes.size()) {
            throw std::runtime_error(name_ + " derived quantized tensor byte size mismatch");
        }
        return bytes;
    }

    TensorValue make_backend_tensor(const TensorShape & shape, ggml_type type) {
        if (buffer_ != nullptr) {
            throw std::runtime_error(name_ + " cannot add weights after upload");
        }
        const auto dims = to_ggml_dims(shape);
        auto * tensor = ggml_new_tensor(ctx_.get(), type, static_cast<int>(shape.rank), dims.data());
        if (tensor == nullptr) {
            throw std::runtime_error("failed to create " + name_ + " weight tensor");
        }
        return wrap_tensor(tensor, shape, type);
    }

    void append_upload(ggml_tensor * tensor, std::vector<std::byte> bytes) {
        if (tensor == nullptr) {
            throw std::runtime_error(name_ + " cannot upload a null tensor");
        }
        if (bytes.size() != ggml_nbytes(tensor)) {
            throw std::runtime_error(name_ + " tensor byte size mismatch");
        }
        PendingUpload upload;
        upload.tensor = tensor;
        upload.kind = PendingUploadKind::Bytes;
        upload.bytes = std::move(bytes);
        pending_.push_back(std::move(upload));
    }

    void upload_bytes(ggml_tensor * tensor, const std::vector<std::byte> & bytes) const {
        if (tensor == nullptr) {
            throw std::runtime_error(name_ + " cannot upload a null tensor");
        }
        if (bytes.size() != ggml_nbytes(tensor)) {
            throw std::runtime_error(name_ + " tensor byte size mismatch");
        }
        ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
    }

    ggml_backend_t backend_ = nullptr;
    BackendType backend_type_ = BackendType::Cpu;
    std::string name_;
    ggml_backend_buffer_type_t buffer_type_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<PendingUpload> pending_;
};

}  // namespace engine::core
