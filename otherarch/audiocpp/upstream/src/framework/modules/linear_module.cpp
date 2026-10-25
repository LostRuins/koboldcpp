#include "engine/framework/modules/linear_module.h"

#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kLinearInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"weight", core::PortKind::Parameter, false},
    {"bias", core::PortKind::Parameter, true},
};

const core::ModulePortSpec kLinearOutputs[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kLinearSchema = {
    "Linear",
    "nn.primitive",
    kLinearInputs,
    3,
    kLinearOutputs,
    1,
    "Applies an affine projection over the last logical dimension.",
};

core::TensorShape flatten_to_matrix_shape(const core::TensorShape & shape) {
    if (shape.rank == 1) {
        return core::TensorShape::from_dims({1, shape.last_dim()});
    }

    return core::TensorShape::from_dims({shape.prefix_elements(), shape.last_dim()});
}

void validate_weight_shape(const LinearConfig & config, const LinearWeights & weights) {
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config.out_features, config.in_features}),
        "weight");

    if (!config.use_bias) {
        return;
    }

    if (!weights.bias.has_value()) {
        throw std::runtime_error("bias is required when LinearConfig.use_bias is true");
    }

    core::validate_shape(
        *weights.bias,
        core::TensorShape::from_dims({config.out_features}),
        "bias");
}

}  // namespace

LinearModule::LinearModule(LinearConfig config) : config_(config) {
    if (config_.in_features <= 0 || config_.out_features <= 0) {
        throw std::runtime_error("LinearConfig features must be positive");
    }
}

const LinearConfig & LinearModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & LinearModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue LinearModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LinearWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }

    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.in_features, "input");
    validate_weight_shape(config_, weights);
    if (config_.cuda_f16_output && (ctx.backend_type != core::BackendType::Cuda ||
            input.type != GGML_TYPE_F16 || weights.weight.type != GGML_TYPE_F16 ||
            config_.precision != GGML_PREC_DEFAULT || config_.cuda_nvfp4_f16_activation_lowering)) {
        throw std::runtime_error("FP16 linear output requires CUDA, FP16 operands, and default precision");
    }
    const core::TensorValue contiguous_input = core::ensure_backend_addressable_layout(ctx, input);

    const core::TensorShape matrix_input_shape = flatten_to_matrix_shape(contiguous_input.shape);
    core::TensorValue matrix_input = core::reshape_tensor(ctx, contiguous_input, matrix_input_shape);

    // kcpp: Audio.cpp adds private F16-output and backend-lowering variants.
    // Build the ordinary matmul and represent F16 output with a standard cast.
    ggml_tensor * projected_raw =
        ggml_mul_mat(ctx.ggml, weights.weight.tensor, matrix_input.tensor);
    if (config_.precision != GGML_PREC_DEFAULT) {
        ggml_mul_mat_set_prec(projected_raw, config_.precision);
    }
    if (config_.cuda_f16_output) {
        projected_raw = ggml_cast(ctx.ggml, projected_raw, GGML_TYPE_F16);
    }
    core::TensorValue projected = core::wrap_tensor(
        projected_raw,
        core::TensorShape::from_dims({matrix_input_shape.at(0), config_.out_features}),
        config_.cuda_f16_output ? GGML_TYPE_F16 : GGML_TYPE_F32);
    if (config_.use_bias) {
        ggml_tensor * bias = config_.cuda_f16_output
            ? ggml_cast(ctx.ggml, weights.bias->tensor, GGML_TYPE_F16) : weights.bias->tensor;
        ggml_tensor * biased_raw = ggml_add(ctx.ggml, projected.tensor, bias);
        projected = core::wrap_tensor(
            biased_raw, projected.shape,
            config_.cuda_f16_output ? GGML_TYPE_F16 : GGML_TYPE_F32);
    }

    return core::reshape_tensor(ctx, projected, input.shape.with_last_dim(config_.out_features));
}

const core::ModuleSchema & LinearModule::static_schema() noexcept {
    return kLinearSchema;
}

}  // namespace engine::modules
