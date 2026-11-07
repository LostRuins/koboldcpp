#include "engine/framework/modules/recurrent_modules.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>
#include <vector>

namespace engine::modules {

namespace {

const core::ModulePortSpec kLstmInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"hidden", core::PortKind::Activation, false},
    {"cell", core::PortKind::Activation, false},
    {"weight_ih", core::PortKind::Parameter, false},
    {"weight_hh", core::PortKind::Parameter, false},
    {"bias_ih", core::PortKind::Parameter, false},
    {"bias_hh", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kLstmOutputs[] = {
    {"hidden", core::PortKind::Activation, false},
    {"cell", core::PortKind::Activation, false},
};

const core::ModuleSchema kLstmSchema = {
    "LSTMCell",
    "nn.recurrent",
    kLstmInputs,
    7,
    kLstmOutputs,
    2,
    "Applies a single LSTM cell update over [batch, features] inputs.",
};

const core::ModulePortSpec kLstmSequenceInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"initial_hidden", core::PortKind::Activation, false},
    {"initial_cell", core::PortKind::Activation, false},
    {"weight_ih", core::PortKind::Parameter, false},
    {"weight_hh", core::PortKind::Parameter, false},
    {"bias_ih", core::PortKind::Parameter, false},
    {"bias_hh", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kLstmSequenceOutputs[] = {
    {"sequence", core::PortKind::Activation, false},
    {"hidden", core::PortKind::Activation, false},
    {"cell", core::PortKind::Activation, false},
};

const core::ModuleSchema kLstmSequenceSchema = {
    "LSTMSequence",
    "nn.recurrent",
    kLstmSequenceInputs,
    7,
    kLstmSequenceOutputs,
    3,
    "Applies an LSTM over [time, features], or opt-in [time, batch, features] sequences.",
};

const core::ModulePortSpec kBidirLstmInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"initial_forward_hidden", core::PortKind::Activation, false},
    {"initial_forward_cell", core::PortKind::Activation, false},
    {"initial_reverse_hidden", core::PortKind::Activation, false},
    {"initial_reverse_cell", core::PortKind::Activation, false},
    {"forward_weight_ih", core::PortKind::Parameter, false},
    {"forward_weight_hh", core::PortKind::Parameter, false},
    {"forward_bias_ih", core::PortKind::Parameter, false},
    {"forward_bias_hh", core::PortKind::Parameter, false},
    {"reverse_weight_ih", core::PortKind::Parameter, false},
    {"reverse_weight_hh", core::PortKind::Parameter, false},
    {"reverse_bias_ih", core::PortKind::Parameter, false},
    {"reverse_bias_hh", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kBidirLstmOutputs[] = {
    {"sequence", core::PortKind::Activation, false},
    {"forward_hidden", core::PortKind::Activation, false},
    {"forward_cell", core::PortKind::Activation, false},
    {"reverse_hidden", core::PortKind::Activation, false},
    {"reverse_cell", core::PortKind::Activation, false},
};

const core::ModuleSchema kBidirLstmSchema = {
    "BidirectionalLSTM",
    "nn.recurrent",
    kBidirLstmInputs,
    13,
    kBidirLstmOutputs,
    5,
    "Applies forward and reverse LSTM passes over a full [time, features] sequence and concatenates outputs.",
};

void validate_weight_shapes(const LSTMCellConfig & config, const LSTMCellWeights & weights) {
    const int64_t gates = 4 * config.hidden_size;
    core::validate_shape(weights.weight_ih, core::TensorShape::from_dims({gates, config.input_size}), "weight_ih");
    core::validate_shape(weights.weight_hh, core::TensorShape::from_dims({gates, config.hidden_size}), "weight_hh");
    core::validate_shape(weights.bias_ih, core::TensorShape::from_dims({gates}), "bias_ih");
    core::validate_shape(weights.bias_hh, core::TensorShape::from_dims({gates}), "bias_hh");
}

core::TensorValue gate_slice(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & gates,
    int64_t hidden_size,
    int64_t gate_index) {
    return SliceModule({1, gate_index * hidden_size, hidden_size}).build(ctx, gates);
}

void validate_sequence_config(const LSTMSequenceConfig & config) {
    if (config.input_size <= 0 || config.hidden_size <= 0) {
        throw std::runtime_error("LSTMSequenceConfig dimensions must be positive");
    }
}

LSTMCellOutputs build_lstm_cell_from_projected_input(
    core::ModuleBuildContext & ctx,
    const LSTMCellConfig & config,
    const core::TensorValue & projected_input,
    const core::TensorValue & hidden,
    const core::TensorValue & cell,
    const LSTMCellWeights & weights,
    bool fusion_friendly_gate_mul) {
    const int64_t gates = 4 * config.hidden_size;
    core::validate_shape(projected_input, core::TensorShape::from_dims({hidden.shape.dims[0], gates}), "projected_input");
    core::validate_shape(hidden, core::TensorShape::from_dims({hidden.shape.dims[0], config.hidden_size}), "hidden");
    core::validate_shape(cell, core::TensorShape::from_dims({hidden.shape.dims[0], config.hidden_size}), "cell");
    validate_weight_shapes(config, weights);

    const auto projected_hidden = LinearModule({config.hidden_size, gates, true}).build(
        ctx,
        hidden,
        {weights.weight_hh, weights.bias_hh});
    const auto gate_values = AddModule().build(ctx, projected_input, projected_hidden);

    const auto input_gate = SigmoidModule().build(ctx, gate_slice(ctx, gate_values, config.hidden_size, 0));
    const auto forget_gate = SigmoidModule().build(ctx, gate_slice(ctx, gate_values, config.hidden_size, 1));
    const auto candidate = TanhModule().build(ctx, gate_slice(ctx, gate_values, config.hidden_size, 2));
    const auto output_gate = SigmoidModule().build(ctx, gate_slice(ctx, gate_values, config.hidden_size, 3));

    const auto kept_cell = fusion_friendly_gate_mul
        ? MulModule().build(ctx, cell, forget_gate)
        : MulModule().build(ctx, forget_gate, cell);
    const auto written_cell = fusion_friendly_gate_mul
        ? MulModule().build(ctx, candidate, input_gate)
        : MulModule().build(ctx, input_gate, candidate);
    const auto new_cell = AddModule().build(ctx, kept_cell, written_cell);
    const auto cell_activation = TanhModule().build(ctx, new_cell);
    const auto new_hidden = fusion_friendly_gate_mul
        ? MulModule().build(ctx, cell_activation, output_gate)
        : MulModule().build(ctx, output_gate, cell_activation);

    return {new_hidden, new_cell};
}

LSTMSequenceOutputs build_lstm_sequence(
    core::ModuleBuildContext & ctx,
    const LSTMSequenceConfig & config,
    const core::TensorValue & input,
    const core::TensorValue & initial_hidden,
    const core::TensorValue & initial_cell,
    const LSTMCellWeights & weights) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    const int64_t batch = config.batched ? input.shape.dims[1] : 1;
    core::validate_shape(input, config.batched
        ? core::TensorShape::from_dims({input.shape.dims[0], batch, config.input_size})
        : core::TensorShape::from_dims({input.shape.dims[0], config.input_size}), "input");
    core::validate_shape(initial_hidden, core::TensorShape::from_dims({batch, config.hidden_size}), "initial_hidden");
    core::validate_shape(initial_cell, core::TensorShape::from_dims({batch, config.hidden_size}), "initial_cell");
    validate_weight_shapes({config.input_size, config.hidden_size}, weights);

    const int64_t frames = input.shape.dims[0];
    const int64_t gates = 4 * config.hidden_size;
    const auto projected_inputs = LinearModule({config.input_size, gates, true}).build(
        ctx,
        input,
        {weights.weight_ih, weights.bias_ih});
    std::vector<core::TensorValue> steps(static_cast<size_t>(frames));
    auto hidden = initial_hidden;
    auto cell = initial_cell;
    const ConcatModule concat_rows({0});

    for (int64_t step = 0; step < frames; ++step) {
        const int64_t t = config.reverse ? (frames - 1 - step) : step;
        auto projected_x_t = SliceModule({0, t, 1}).build(ctx, projected_inputs);
        if (config.batched) {
            projected_x_t = core::reshape_tensor(ctx, projected_x_t, core::TensorShape::from_dims({batch, gates}));
        }
        const auto outputs = build_lstm_cell_from_projected_input(
            ctx,
            {config.input_size, config.hidden_size},
            projected_x_t,
            hidden,
            cell,
            weights,
            config.fusion_friendly_gate_mul);
        hidden = outputs.hidden;
        cell = outputs.cell;
        steps[static_cast<size_t>(t)] = config.batched
            ? core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, batch, config.hidden_size}))
            : hidden;
    }

    if (config.batched) {
        // Avoid copying a growing prefix at every timestep of a long sequence.
        while (steps.size() > 1) {
            std::vector<core::TensorValue> merged;
            merged.reserve((steps.size() + 1) / 2);
            for (size_t i = 0; i < steps.size(); i += 2) {
                merged.push_back(i + 1 < steps.size() ? concat_rows.build(ctx, steps[i], steps[i + 1]) : steps[i]);
            }
            steps = std::move(merged);
        }
        return {steps.front(), hidden, cell};
    }

    auto sequence = steps[0];
    for (int64_t t = 1; t < frames; ++t) {
        sequence = concat_rows.build(ctx, sequence, steps[static_cast<size_t>(t)]);
    }
    return LSTMSequenceOutputs{sequence, hidden, cell};
}

}  // namespace

LSTMCellModule::LSTMCellModule(LSTMCellConfig config) : config_(config) {
    if (config_.input_size <= 0 || config_.hidden_size <= 0) {
        throw std::runtime_error("LSTMCellConfig dimensions must be positive");
    }
}

const LSTMCellConfig & LSTMCellModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & LSTMCellModule::schema() const noexcept {
    return static_schema();
}

LSTMCellOutputs LSTMCellModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & hidden,
    const core::TensorValue & cell,
    const LSTMCellWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, 2, "input");
    core::validate_shape(hidden, core::TensorShape::from_dims({input.shape.dims[0], config_.hidden_size}), "hidden");
    core::validate_shape(cell, core::TensorShape::from_dims({input.shape.dims[0], config_.hidden_size}), "cell");
    core::validate_shape(input, core::TensorShape::from_dims({input.shape.dims[0], config_.input_size}), "input");
    validate_weight_shapes(config_, weights);

    const int64_t gates = 4 * config_.hidden_size;
    const auto projected_input = LinearModule({config_.input_size, gates, true}).build(
        ctx,
        input,
        {weights.weight_ih, weights.bias_ih});
    return build_lstm_cell_from_projected_input(ctx, config_, projected_input, hidden, cell, weights, false);
}

const core::ModuleSchema & LSTMCellModule::static_schema() noexcept {
    return kLstmSchema;
}

LSTMSequenceModule::LSTMSequenceModule(LSTMSequenceConfig config) : config_(config) {
    validate_sequence_config(config_);
}

const LSTMSequenceConfig & LSTMSequenceModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & LSTMSequenceModule::schema() const noexcept {
    return static_schema();
}

LSTMSequenceOutputs LSTMSequenceModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & initial_hidden,
    const core::TensorValue & initial_cell,
    const LSTMSequenceWeights & weights) const {
    return build_lstm_sequence(ctx, config_, input, initial_hidden, initial_cell, weights.cell);
}

const core::ModuleSchema & LSTMSequenceModule::static_schema() noexcept {
    return kLstmSequenceSchema;
}

BidirectionalLSTMModule::BidirectionalLSTMModule(LSTMSequenceConfig config) : config_(config) {
    validate_sequence_config(config_);
}

const LSTMSequenceConfig & BidirectionalLSTMModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & BidirectionalLSTMModule::schema() const noexcept {
    return static_schema();
}

BidirectionalLSTMOutputs BidirectionalLSTMModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & initial_forward_hidden,
    const core::TensorValue & initial_forward_cell,
    const core::TensorValue & initial_reverse_hidden,
    const core::TensorValue & initial_reverse_cell,
    const BidirectionalLSTMWeights & weights) const {
    LSTMSequenceConfig forward_config = config_;
    forward_config.reverse = false;
    LSTMSequenceConfig reverse_config = config_;
    reverse_config.reverse = true;
    const auto forward = build_lstm_sequence(ctx, forward_config, input, initial_forward_hidden, initial_forward_cell, weights.forward);
    const auto reverse = build_lstm_sequence(ctx, reverse_config, input, initial_reverse_hidden, initial_reverse_cell, weights.reverse);
    const auto sequence = ConcatModule({config_.batched ? 2 : 1}).build(ctx, forward.sequence, reverse.sequence);
    return {sequence, forward.hidden, forward.cell, reverse.hidden, reverse.cell};
}

const core::ModuleSchema & BidirectionalLSTMModule::static_schema() noexcept {
    return kBidirLstmSchema;
}

}  // namespace engine::modules
