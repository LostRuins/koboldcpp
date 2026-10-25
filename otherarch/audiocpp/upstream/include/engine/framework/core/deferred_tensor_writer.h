#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <vector>

namespace engine::core {

class DeferredTensorWriter {
public:
    struct PendingF32Tensor {
        TensorValue tensor;
        std::vector<float> values;
    };

    struct PendingI32Tensor {
        TensorValue tensor;
        std::vector<int32_t> values;
    };

    void queue_f32(const TensorValue & tensor, const std::vector<float> & values);
    void queue_i32(const TensorValue & tensor, const std::vector<int32_t> & values);

    TensorValue make_f32_tensor(
        ModuleBuildContext & ctx,
        const TensorShape & shape,
        const std::vector<float> & values);

    TensorValue make_i32_tensor(
        ModuleBuildContext & ctx,
        const TensorShape & shape,
        const std::vector<int32_t> & values);

    void flush() const;

private:
    std::vector<PendingF32Tensor> pending_f32_;
    std::vector<PendingI32Tensor> pending_i32_;
};

}  // namespace engine::core
