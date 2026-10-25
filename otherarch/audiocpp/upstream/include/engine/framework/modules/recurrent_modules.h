#pragma once

#include "engine/framework/core/module.h"

#include <vector>

namespace engine::modules {

struct LSTMCellConfig {
    int64_t input_size = 0;
    int64_t hidden_size = 0;
};

struct LSTMCellWeights {
    core::TensorValue weight_ih;
    core::TensorValue weight_hh;
    core::TensorValue bias_ih;
    core::TensorValue bias_hh;
};

struct LSTMCellOutputs {
    core::TensorValue hidden;
    core::TensorValue cell;
};

struct LSTMSequenceConfig {
    int64_t input_size = 0;
    int64_t hidden_size = 0;
    bool reverse = false;
    // Opt-in [time, batch, features] layout instead of [time, features].
    bool batched = false;
    // Put the non-activation operand first in commutative gate multiplies so
    // backends can fuse a following sigmoid with the multiply.
    bool fusion_friendly_gate_mul = false;
};

struct LSTMSequenceWeights {
    LSTMCellWeights cell;
};

struct LSTMStackWeights {
    std::vector<LSTMCellWeights> layers;
};

struct LSTMSequenceOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue cell;
};

class LSTMCellModule {
public:
    explicit LSTMCellModule(LSTMCellConfig config);

    const LSTMCellConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    LSTMCellOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & hidden,
        const core::TensorValue & cell,
        const LSTMCellWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    LSTMCellConfig config_;
};

class LSTMSequenceModule {
public:
    explicit LSTMSequenceModule(LSTMSequenceConfig config);

    const LSTMSequenceConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    LSTMSequenceOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & initial_hidden,
        const core::TensorValue & initial_cell,
        const LSTMSequenceWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    LSTMSequenceConfig config_;
};

struct BidirectionalLSTMWeights {
    LSTMCellWeights forward;
    LSTMCellWeights reverse;
};

struct BidirectionalLSTMOutputs {
    core::TensorValue sequence;
    core::TensorValue forward_hidden;
    core::TensorValue forward_cell;
    core::TensorValue reverse_hidden;
    core::TensorValue reverse_cell;
};

class BidirectionalLSTMModule {
public:
    explicit BidirectionalLSTMModule(LSTMSequenceConfig config);

    const LSTMSequenceConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    BidirectionalLSTMOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & initial_forward_hidden,
        const core::TensorValue & initial_forward_cell,
        const core::TensorValue & initial_reverse_hidden,
        const core::TensorValue & initial_reverse_cell,
        const BidirectionalLSTMWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    LSTMSequenceConfig config_;
};

}  // namespace engine::modules
