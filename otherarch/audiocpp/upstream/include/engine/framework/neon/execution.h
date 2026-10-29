#pragma once

#include "engine/framework/neon/operators.h"
#include "engine/framework/neon/tensor.h"

#include <initializer_list>
#include <memory>

namespace engine::neon {

bool available() noexcept;

// A prepared data-flow plan. Operator implementations own math and packing;
// this class owns tensor lifetimes, reusable workspace and the worker team.
class ExecutionPlan {
public:
    class Value {
    private:
        size_t index_;
        const void *owner_;
        Value(size_t index, const void *owner) : index_(index), owner_(owner) {}
        friend class ExecutionPlan;
    };

    ExecutionPlan();
    ~ExecutionPlan();
    Value input();
    Value add(const Operator &operation, std::initializer_list<Value> inputs);
    void set_output(Value value, Layout layout = Layout::ChannelFirst);
    // Retained intermediate results remain read-only and valid until the next run.
    void retain(Value value);
    TensorView view(Value value) const;
    // Validates and plans allocations before dispatch. Output is owned by this
    // plan and remains valid until the next run. One plan cannot run concurrently.
    const Tensor &run(const std::vector<TensorView> &inputs, int threads);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::neon
