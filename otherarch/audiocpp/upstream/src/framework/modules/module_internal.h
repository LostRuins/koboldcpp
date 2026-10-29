#pragma once

#include "engine/framework/modules/structural_modules.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace engine::modules::internal {

inline void validate_sequence_input(const core::TensorValue & input, int64_t hidden_size, const char * name) {
    core::validate_rank_between(input, 3, 3, name);
    core::validate_last_dim(input, hidden_size, name);
}

inline core::TensorValue concat_range(
    core::ModuleBuildContext & ctx,
    const std::vector<core::TensorValue> & values,
    size_t begin,
    size_t end,
    int axis) {
    if (begin + 1 == end) {
        return values[begin];
    }
    const size_t mid = begin + (end - begin) / 2;
    return ConcatModule({axis}).build(
        ctx,
        concat_range(ctx, values, begin, mid, axis),
        concat_range(ctx, values, mid, end, axis));
}

inline core::TensorValue concat_all(
    core::ModuleBuildContext & ctx,
    const std::vector<core::TensorValue> & values,
    int axis) {
    if (values.empty()) {
        throw std::runtime_error("concat_all requires at least one tensor");
    }
    return concat_range(ctx, values, 0, values.size(), axis);
}

}  // namespace engine::modules::internal
