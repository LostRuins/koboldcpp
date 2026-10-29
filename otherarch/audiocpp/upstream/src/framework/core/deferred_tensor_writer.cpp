#include "engine/framework/core/deferred_tensor_writer.h"

namespace engine::core {

void DeferredTensorWriter::queue_f32(const TensorValue & tensor, const std::vector<float> & values) {
    pending_f32_.push_back(PendingF32Tensor{tensor, values});
}

void DeferredTensorWriter::queue_i32(const TensorValue & tensor, const std::vector<int32_t> & values) {
    pending_i32_.push_back(PendingI32Tensor{tensor, values});
}

TensorValue DeferredTensorWriter::make_f32_tensor(
    ModuleBuildContext & ctx,
    const TensorShape & shape,
    const std::vector<float> & values) {
    auto tensor = core::make_tensor(ctx, GGML_TYPE_F32, shape);
    queue_f32(tensor, values);
    return tensor;
}

TensorValue DeferredTensorWriter::make_i32_tensor(
    ModuleBuildContext & ctx,
    const TensorShape & shape,
    const std::vector<int32_t> & values) {
    auto tensor = core::make_tensor(ctx, GGML_TYPE_I32, shape);
    queue_i32(tensor, values);
    return tensor;
}

void DeferredTensorWriter::flush() const {
    for (const auto & item : pending_f32_) {
        core::write_tensor_f32(item.tensor, item.values);
    }
    for (const auto & item : pending_i32_) {
        core::write_tensor_i32(item.tensor, item.values);
    }
}

}  // namespace engine::core
