#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kActivationInputs[] = {
    {"input", core::PortKind::Activation, false},
};

const core::ModulePortSpec kActivationOutputs[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kReluSchema = {
    "ReLU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies rectified linear activation elementwise.",
};

const core::ModuleSchema kLeakyReluSchema = {
    "LeakyReLU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies leaky rectified linear activation elementwise.",
};

const core::ModuleSchema kSigmoidSchema = {
    "Sigmoid",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies sigmoid activation elementwise.",
};

const core::ModuleSchema kTanhSchema = {
    "Tanh",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies tanh activation elementwise.",
};

const core::ModuleSchema kSqrtSchema = {
    "Sqrt",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies square root elementwise.",
};

const core::ModuleSchema kBf16RoundSchema = {
    "Bf16Round",
    "tensor.precision",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Rounds activations to BF16 values while retaining F32 storage.",
};

const core::ModuleSchema kGeluSchema = {
    "GELU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies GELU activation elementwise.",
};

const core::ModuleSchema kSiluSchema = {
    "SiLU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies SiLU activation elementwise.",
};

const core::ModuleSchema kGegluSchema = {
    "GEGLU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies exact-erf GELU to the second half and multiplies by the first half.",
};

const core::ModuleSchema kSeluSchema = {
    "SELU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies SELU activation elementwise.",
};

const core::ModuleSchema kSwooshLSchema = {
    "SwooshL",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies Zipformer SwooshL activation elementwise.",
};

const core::ModuleSchema kSwooshRSchema = {
    "SwooshR",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies Zipformer SwooshR activation elementwise.",
};

const core::ModuleSchema kEluSchema = {
    "ELU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies ELU activation elementwise.",
};

const core::ModuleSchema kSoftmaxSchema = {
    "Softmax",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Applies softmax over the last physical dimension.",
};

const core::ModuleSchema kGLUSchema = {
    "GLU",
    "nn.activation",
    kActivationInputs,
    1,
    kActivationOutputs,
    1,
    "Splits the last dimension in half and applies sigmoid gating.",
};

const core::ModulePortSpec kSnakeInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"alpha", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kSnakeBetaInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"alpha", core::PortKind::Parameter, false},
    {"beta", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kAliasFreeActivationInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"alpha", core::PortKind::Parameter, false},
    {"inv_beta", core::PortKind::Parameter, false},
    {"up_filter_even", core::PortKind::Parameter, false},
    {"up_filter_odd", core::PortKind::Parameter, false},
    {"down_filter", core::PortKind::Parameter, false},
};

const core::ModuleSchema kSnake1dSchema = {
    "Snake1d",
    "nn.activation",
    kSnakeInputs,
    2,
    kActivationOutputs,
    1,
    "Applies Snake activation over channel-time tensors using per-channel alpha.",
};

const core::ModuleSchema kSnakeBeta1dSchema = {
    "SnakeBeta1d",
    "nn.activation",
    kSnakeBetaInputs,
    3,
    kActivationOutputs,
    1,
    "Applies SnakeBeta activation over channel-time tensors using per-channel alpha and beta.",
};

const core::ModuleSchema kAliasFreeActivationSchema = {
    "AliasFreeActivation",
    "nn.activation",
    kAliasFreeActivationInputs,
    6,
    kActivationOutputs,
    1,
    "Applies filtered upsample, nonlinear activation, and filtered downsample to channel-time tensors.",
};

template <typename Fn>
core::TensorValue build_unary(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    Fn fn) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::wrap_tensor(fn(ctx.ggml, contiguous.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorShape make_snake_alpha_shape(const core::TensorShape & input, int64_t hidden_size) {
    core::TensorShape shape = {};
    shape.rank = input.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    shape.dims[shape.rank - 2] = hidden_size;
    return shape;
}

core::TensorValue ensure_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

bool same_shape(const core::TensorShape & lhs, const core::TensorShape & rhs) {
    if (lhs.rank != rhs.rank) {
        return false;
    }
    for (size_t i = 0; i < lhs.rank; ++i) {
        if (lhs.dims[i] != rhs.dims[i]) {
            return false;
        }
    }
    return true;
}

ggml_tensor * repeat_frame(ggml_context * ctx, ggml_tensor * x, int64_t frame, int64_t count) {
    ggml_tensor * src = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], static_cast<size_t>(frame) * x->nb[0]);
    return ggml_repeat(ctx, src, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, count, x->ne[1]));
}

ggml_tensor * replicate_pad_left_ct(ggml_context * ctx, ggml_tensor * x, int64_t count) {
    if (count <= 0) {
        return x;
    }
    return ggml_concat(ctx, repeat_frame(ctx, x, 0, count), x, 0);
}

ggml_tensor * depthwise_conv_transpose_causal_ct(
    core::ModuleBuildContext & ctx,
    ggml_tensor * x,
    const core::TensorValue & even_filter,
    const core::TensorValue & odd_filter,
    int64_t phase_kernel,
    int64_t upsample_ratio) {
    const int64_t frames = x->ne[0];
    const int64_t channels = x->ne[1];
    ggml_tensor * input4 = ggml_reshape_4d(
        ctx.ggml,
        core::has_backend_addressable_layout(x) ? x : ggml_cont(ctx.ggml, x),
        frames,
        1,
        channels,
        1);

    auto convolve_phase = [&](const core::TensorValue & phase_filter) {
        ggml_tensor * raw = ggml_conv_2d_dw_direct(
            ctx.ggml,
            phase_filter.tensor,
            input4,
            1,
            1,
            static_cast<int>(phase_kernel - 1),
            0,
            1,
            1);
        raw = core::has_backend_addressable_layout(raw) ? raw : ggml_cont(ctx.ggml, raw);
        auto * reshaped = ggml_reshape_2d(ctx.ggml, raw, raw->ne[0], raw->ne[2]);
        return ggml_view_2d(ctx.ggml, reshaped, frames, reshaped->ne[1], reshaped->nb[1], 0);
    };

    ggml_tensor * even = convolve_phase(even_filter);
    ggml_tensor * odd = convolve_phase(odd_filter);
    ggml_tensor * even3 = ggml_reshape_3d(
        ctx.ggml,
        core::has_backend_addressable_layout(even) ? even : ggml_cont(ctx.ggml, even),
        1,
        frames,
        channels);
    ggml_tensor * odd3 = ggml_reshape_3d(
        ctx.ggml,
        core::has_backend_addressable_layout(odd) ? odd : ggml_cont(ctx.ggml, odd),
        1,
        frames,
        channels);
    return ggml_reshape_2d(ctx.ggml, ggml_concat(ctx.ggml, even3, odd3, 0), frames * upsample_ratio, channels);
}

ggml_tensor * depthwise_conv_causal_ct(
    core::ModuleBuildContext & ctx,
    ggml_tensor * x,
    const core::TensorValue & filter,
    int64_t kernel_size,
    int64_t stride) {
    ggml_tensor * padded = replicate_pad_left_ct(ctx.ggml, x, kernel_size - 1);
    ggml_tensor * input4 = ggml_reshape_4d(
        ctx.ggml,
        core::has_backend_addressable_layout(padded) ? padded : ggml_cont(ctx.ggml, padded),
        padded->ne[0],
        1,
        padded->ne[1],
        1);
    ggml_tensor * out = ggml_conv_2d_dw_direct(ctx.ggml, filter.tensor, input4, static_cast<int>(stride), 1, 0, 0, 1, 1);
    out = core::has_backend_addressable_layout(out) ? out : ggml_cont(ctx.ggml, out);
    return ggml_reshape_2d(ctx.ggml, out, out->ne[0], out->ne[2]);
}

core::TensorValue build_swoosh(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    float offset,
    float constant) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto shifted = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, contiguous.tensor, 1.0F, -offset),
        input.shape,
        GGML_TYPE_F32);
    auto activated = core::wrap_tensor(ggml_softplus(ctx.ggml, shifted.tensor), input.shape, GGML_TYPE_F32);
    auto residual = core::wrap_tensor(ggml_scale(ctx.ggml, contiguous.tensor, -0.08F), input.shape, GGML_TYPE_F32);
    auto summed = core::wrap_tensor(ggml_add(ctx.ggml, activated.tensor, residual.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, summed.tensor, 1.0F, constant), input.shape, GGML_TYPE_F32);
}

}  // namespace

const core::ModuleSchema & ReluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ReluModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_unary(ctx, input, ggml_relu);
}

const core::ModuleSchema & ReluModule::static_schema() noexcept {
    return kReluSchema;
}

LeakyReluModule::LeakyReluModule(LeakyReluConfig config) : config_(config) {
}

const LeakyReluConfig & LeakyReluModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & LeakyReluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue LeakyReluModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::wrap_tensor(
        ggml_leaky_relu(ctx.ggml, contiguous.tensor, config_.negative_slope, false),
        input.shape,
        GGML_TYPE_F32);
}

const core::ModuleSchema & LeakyReluModule::static_schema() noexcept {
    return kLeakyReluSchema;
}

const core::ModuleSchema & SigmoidModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SigmoidModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_unary(ctx, input, ggml_sigmoid);
}

const core::ModuleSchema & SigmoidModule::static_schema() noexcept {
    return kSigmoidSchema;
}

const core::ModuleSchema & TanhModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue TanhModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_unary(ctx, input, ggml_tanh);
}

const core::ModuleSchema & TanhModule::static_schema() noexcept {
    return kTanhSchema;
}

const core::ModuleSchema & SqrtModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SqrtModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_unary(ctx, input, ggml_sqrt);
}

const core::ModuleSchema & SqrtModule::static_schema() noexcept {
    return kSqrtSchema;
}

const core::ModuleSchema & Bf16RoundModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Bf16RoundModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    // kcpp: Audio.cpp carries ggml_round_bf16 as an unofficial GGML op.
    // KoboldCpp keeps its GGML syncable, so express the same graph with the
    // standard casts already supported by every KCPP backend.
    auto * rounded_bf16 = ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_BF16);
    auto * rounded_f32 = ggml_cast(ctx.ggml, rounded_bf16, GGML_TYPE_F32);
    return core::wrap_tensor(rounded_f32, input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & Bf16RoundModule::static_schema() noexcept {
    return kBf16RoundSchema;
}

GeluModule::GeluModule(GeluConfig config) : config_(config) {
}

const GeluConfig & GeluModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & GeluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue GeluModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    switch (config_.approximation) {
        case GeluApproximation::ExactErf:
            return build_unary(ctx, input, ggml_gelu_erf);
        case GeluApproximation::Tanh:
            return build_unary(ctx, input, ggml_gelu);
        case GeluApproximation::Quick:
            return build_unary(ctx, input, ggml_gelu_quick);
        default:
            throw std::runtime_error("Unsupported GELU approximation mode");
    }
}

const core::ModuleSchema & GeluModule::static_schema() noexcept {
    return kGeluSchema;
}

const core::ModuleSchema & GegluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue GegluModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    if (input.shape.last_dim() % 2 != 0 ||
        (input.type != GGML_TYPE_F16 && input.type != GGML_TYPE_F32)) {
        throw std::runtime_error("GEGLU requires an even last dimension and F16 or F32 input");
    }
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto output_shape = input.shape;
    output_shape.dims[output_shape.rank - 1] /= 2;
    return core::wrap_tensor(ggml_geglu_erf_swapped(ctx.ggml, contiguous.tensor), output_shape, input.type);
}

const core::ModuleSchema & GegluModule::static_schema() noexcept {
    return kGegluSchema;
}

const core::ModuleSchema & SiluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SiluModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (preserve_input_type_) {
        core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
        const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
        return core::wrap_tensor(ggml_silu(ctx.ggml, contiguous.tensor), input.shape, input.type);
    }
    return build_unary(ctx, input, ggml_silu);
}

const core::ModuleSchema & SiluModule::static_schema() noexcept {
    return kSiluSchema;
}

const core::ModuleSchema & SeluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SeluModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    constexpr float kAlpha = 1.6732632423543772848170429916717F;
    constexpr float kScale = 1.0507009873554804934193349852946F;
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto positive = ggml_relu(ctx.ggml, contiguous.tensor);
    auto negative = ggml_scale(ctx.ggml, ggml_relu(ctx.ggml, ggml_scale(ctx.ggml, contiguous.tensor, -1.0F)), -1.0F);
    negative = ggml_scale(ctx.ggml, ggml_expm1(ctx.ggml, negative), kAlpha);
    return core::wrap_tensor(
        ggml_scale(ctx.ggml, ggml_add(ctx.ggml, positive, negative), kScale),
        input.shape,
        GGML_TYPE_F32);
}

const core::ModuleSchema & SeluModule::static_schema() noexcept {
    return kSeluSchema;
}

const core::ModuleSchema & SwooshLModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SwooshLModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_swoosh(ctx, input, 4.0F, -0.035F);
}

const core::ModuleSchema & SwooshLModule::static_schema() noexcept {
    return kSwooshLSchema;
}

const core::ModuleSchema & SwooshRModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SwooshRModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_swoosh(ctx, input, 1.0F, -0.313261687F);
}

const core::ModuleSchema & SwooshRModule::static_schema() noexcept {
    return kSwooshRSchema;
}

const core::ModuleSchema & EluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue EluModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_unary(ctx, input, ggml_elu);
}

const core::ModuleSchema & EluModule::static_schema() noexcept {
    return kEluSchema;
}

const core::ModuleSchema & SoftmaxModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SoftmaxModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return build_unary(ctx, input, ggml_soft_max);
}

const core::ModuleSchema & SoftmaxModule::static_schema() noexcept {
    return kSoftmaxSchema;
}

GLUModule::GLUModule(GLUConfig config) : config_(config) {}

const core::ModuleSchema & GLUModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue GLUModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    if (input.shape.last_dim() % 2 != 0) {
        throw std::runtime_error("GLU input last dimension must be even");
    }

    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const auto flat = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({contiguous.shape.num_elements() / contiguous.shape.last_dim(), contiguous.shape.last_dim()}));
    const int64_t hidden = flat.shape.last_dim() / 2;
    auto lhs = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, flat.tensor, hidden, flat.shape.dims[0], flat.tensor->nb[1], 0),
        core::TensorShape::from_dims({flat.shape.dims[0], hidden}),
        GGML_TYPE_F32);
    auto rhs = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, flat.tensor, hidden, flat.shape.dims[0], flat.tensor->nb[1], hidden * sizeof(float)),
        core::TensorShape::from_dims({flat.shape.dims[0], hidden}),
        GGML_TYPE_F32);
    if (config_.contiguous_gate) {
        rhs = core::wrap_tensor(ggml_cont(ctx.ggml, rhs.tensor), rhs.shape, GGML_TYPE_F32);
    }
    rhs = core::wrap_tensor(ggml_sigmoid(ctx.ggml, rhs.tensor), rhs.shape, GGML_TYPE_F32);
    auto output = core::wrap_tensor(ggml_mul(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);

    auto output_shape = input.shape;
    output_shape.dims[output_shape.rank - 1] = hidden;
    return core::reshape_tensor(ctx, output, output_shape);
}

const core::ModuleSchema & GLUModule::static_schema() noexcept {
    return kGLUSchema;
}

Snake1dModule::Snake1dModule(Snake1dConfig config) : config_(config) {
    if (config_.hidden_size <= 0) {
        throw std::runtime_error("Snake1dConfig.hidden_size must be positive");
    }
}

const Snake1dConfig & Snake1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Snake1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Snake1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Snake1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, core::kMaxTensorRank, "input");
    if (input.shape.dims[input.shape.rank - 2] != config_.hidden_size) {
        throw std::runtime_error("Snake1d input hidden dimension mismatch");
    }

    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const auto input_f32 = ensure_f32(ctx, contiguous);
    core::TensorValue alpha_broadcast = {};
    const auto alpha_shape = make_snake_alpha_shape(input.shape, config_.hidden_size);
    if (same_shape(weights.alpha.shape, input.shape) || same_shape(weights.alpha.shape, alpha_shape)) {
        alpha_broadcast = ensure_f32(ctx, weights.alpha);
    } else {
        core::validate_shape(weights.alpha, core::TensorShape::from_dims({config_.hidden_size}), "alpha");
        alpha_broadcast = core::reshape_tensor(ctx, ensure_f32(ctx, weights.alpha), alpha_shape);
    }
    const auto ax = core::wrap_tensor(ggml_mul(ctx.ggml, input_f32.tensor, alpha_broadcast.tensor), input_f32.shape, GGML_TYPE_F32);
    const auto s = core::wrap_tensor(ggml_sin(ctx.ggml, ax.tensor), input_f32.shape, GGML_TYPE_F32);
    if (weights.inverse_alpha) {
        core::TensorValue inverse;
        if (same_shape(weights.inverse_alpha->shape, alpha_shape)) {
            inverse = ensure_f32(ctx, *weights.inverse_alpha);
        } else {
            core::validate_shape(*weights.inverse_alpha,
                core::TensorShape::from_dims({config_.hidden_size}), "inverse_alpha");
            inverse = core::reshape_tensor(ctx, ensure_f32(ctx, *weights.inverse_alpha), alpha_shape);
        }
        const auto squared = ggml_sqr(ctx.ggml, s.tensor);
        const auto fraction = ggml_mul(ctx.ggml, squared, inverse.tensor);
        return core::wrap_tensor(ggml_add(ctx.ggml, input_f32.tensor, fraction), input.shape, GGML_TYPE_F32);
    }
    const auto s2 = core::wrap_tensor(ggml_mul(ctx.ggml, s.tensor, s.tensor), input_f32.shape, GGML_TYPE_F32);
    const auto frac = core::wrap_tensor(ggml_div(ctx.ggml, s2.tensor, alpha_broadcast.tensor), input_f32.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, input_f32.tensor, frac.tensor), input_f32.shape, GGML_TYPE_F32);
}

PReluModule::PReluModule(PReluConfig config) : config_(config) {}

const core::ModuleSchema & PReluModule::schema() const noexcept {
    return static_schema();
}

const core::ModuleSchema & PReluModule::static_schema() noexcept {
    static const core::ModulePortSpec inputs[] = {
        {"input", core::PortKind::Activation, false},
        {"slope", core::PortKind::Parameter, false},
    };
    static const core::ModuleSchema schema = {
        "PReLU", "nn.activation", inputs, 2, kActivationOutputs, 1,
        "Applies learned scalar or channel-wise negative slopes.",
    };
    return schema;
}

core::TensorValue PReluModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & slope) const {
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    if (config_.channel_axis >= input.shape.rank) {
        throw std::runtime_error("PReLU channel axis is outside input rank");
    }
    auto alpha = slope;
    if (!same_shape(alpha.shape, input.shape)) {
        const auto channels = alpha.shape.num_elements();
        if (channels != 1 && channels != input.shape.dims[config_.channel_axis]) {
            throw std::runtime_error("PReLU slope must be scalar or match the channel dimension");
        }
        auto shape = input.shape;
        for (size_t axis = 0; axis < shape.rank; ++axis) shape.dims[axis] = 1;
        shape.dims[config_.channel_axis] = channels;
        alpha = core::reshape_tensor(ctx, alpha, shape);
        alpha = RepeatModule({input.shape}).build(ctx, alpha);
    }
    const auto x = core::ensure_backend_addressable_layout(ctx, input);
    const auto positive = ReluModule().build(ctx, x);
    auto negative = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, positive.tensor), x.shape);
    negative = MulModule().build(ctx, negative, alpha);
    return AddModule().build(ctx, positive, negative);
}

const core::ModuleSchema & Snake1dModule::static_schema() noexcept {
    return kSnake1dSchema;
}

SnakeBeta1dModule::SnakeBeta1dModule(SnakeBeta1dConfig config) : config_(config) {
    if (config_.hidden_size <= 0) {
        throw std::runtime_error("SnakeBeta1dConfig.hidden_size must be positive");
    }
}

const SnakeBeta1dConfig & SnakeBeta1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & SnakeBeta1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SnakeBeta1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SnakeBeta1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, core::kMaxTensorRank, "input");
    if (input.shape.dims[input.shape.rank - 2] != config_.hidden_size) {
        throw std::runtime_error("SnakeBeta1d input hidden dimension mismatch");
    }
    core::validate_shape(weights.alpha, core::TensorShape::from_dims({config_.hidden_size}), "alpha");
    core::validate_shape(weights.beta, core::TensorShape::from_dims({config_.hidden_size}), "beta");

    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const auto input_f32 = ensure_f32(ctx, contiguous);
    const auto channel_shape = make_snake_alpha_shape(input.shape, config_.hidden_size);
    auto alpha = core::reshape_tensor(ctx, ensure_f32(ctx, weights.alpha), channel_shape);
    auto beta = core::reshape_tensor(ctx, ensure_f32(ctx, weights.beta), channel_shape);
    if (config_.logscale) {
        alpha = core::wrap_tensor(ggml_exp(ctx.ggml, alpha.tensor), alpha.shape, GGML_TYPE_F32);
        beta = core::wrap_tensor(ggml_exp(ctx.ggml, beta.tensor), beta.shape, GGML_TYPE_F32);
    }
    const auto ax = core::wrap_tensor(ggml_mul(ctx.ggml, input_f32.tensor, alpha.tensor), input_f32.shape, GGML_TYPE_F32);
    const auto s = core::wrap_tensor(ggml_sin(ctx.ggml, ax.tensor), input_f32.shape, GGML_TYPE_F32);
    const auto s2 = core::wrap_tensor(ggml_mul(ctx.ggml, s.tensor, s.tensor), input_f32.shape, GGML_TYPE_F32);
    const auto denom = core::wrap_tensor(ggml_scale_bias(ctx.ggml, beta.tensor, 1.0F, 1.0e-9F), beta.shape, GGML_TYPE_F32);
    const auto periodic = core::wrap_tensor(ggml_div(ctx.ggml, s2.tensor, denom.tensor), input_f32.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, input_f32.tensor, periodic.tensor), input_f32.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & SnakeBeta1dModule::static_schema() noexcept {
    return kSnakeBeta1dSchema;
}

AliasFreeActivationModule::AliasFreeActivationModule(AliasFreeActivationConfig config) : config_(config) {
    if (config_.channels <= 0) {
        throw std::runtime_error("AliasFreeActivationConfig.channels must be positive");
    }
    if (config_.kernel_size <= 0) {
        throw std::runtime_error("AliasFreeActivationConfig.kernel_size must be positive");
    }
    if (config_.upsample_ratio <= 0 || config_.kernel_size % config_.upsample_ratio != 0) {
        throw std::runtime_error("AliasFreeActivationConfig.upsample_ratio must divide kernel_size");
    }
}

const AliasFreeActivationConfig & AliasFreeActivationModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & AliasFreeActivationModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue AliasFreeActivationModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AliasFreeActivationWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    if (input.shape.dims[1] != config_.channels) {
        throw std::runtime_error("AliasFreeActivation input channel mismatch");
    }
    const int64_t phase_kernel = config_.kernel_size / config_.upsample_ratio;
    core::validate_shape(weights.alpha, core::TensorShape::from_dims({config_.channels}), "alpha");
    core::validate_shape(weights.inv_beta, core::TensorShape::from_dims({config_.channels}), "inv_beta");
    core::validate_shape(weights.up_filter_even, core::TensorShape::from_dims({config_.channels, 1, 1, phase_kernel}), "up_filter_even");
    core::validate_shape(weights.up_filter_odd, core::TensorShape::from_dims({config_.channels, 1, 1, phase_kernel}), "up_filter_odd");
    core::validate_shape(weights.down_filter, core::TensorShape::from_dims({config_.channels, 1, 1, config_.kernel_size}), "down_filter");

    auto input_ct = ggml_reshape_2d(
        ctx.ggml,
        core::ensure_backend_addressable_layout(ctx, input).tensor,
        input.shape.dims[2],
        input.shape.dims[1]);
    ggml_tensor * up = depthwise_conv_transpose_causal_ct(
        ctx,
        input_ct,
        weights.up_filter_even,
        weights.up_filter_odd,
        phase_kernel,
        config_.upsample_ratio);
    up = ggml_scale(ctx.ggml, up, static_cast<float>(config_.upsample_ratio));
    ggml_tensor * alpha = ggml_reshape_2d(ctx.ggml, weights.alpha.tensor, 1, config_.channels);
    ggml_tensor * inv_beta = ggml_reshape_2d(ctx.ggml, weights.inv_beta.tensor, 1, config_.channels);
    ggml_tensor * periodic = nullptr;
    switch (config_.kind) {
        case AliasFreeActivationKind::SnakeBeta:
            periodic = ggml_sqr(ctx.ggml, ggml_sin(ctx.ggml, ggml_mul(ctx.ggml, up, alpha)));
            break;
        default:
            throw std::runtime_error("Unsupported alias-free activation kind");
    }
    ggml_tensor * activated = ggml_add(ctx.ggml, up, ggml_mul(ctx.ggml, periodic, inv_beta));
    ggml_tensor * output_ct = depthwise_conv_causal_ct(
        ctx,
        activated,
        weights.down_filter,
        config_.kernel_size,
        config_.upsample_ratio);
    return core::wrap_tensor(
        ggml_reshape_3d(
            ctx.ggml,
            core::has_backend_addressable_layout(output_ct) ? output_ct : ggml_cont(ctx.ggml, output_ct),
            output_ct->ne[0],
            output_ct->ne[1],
            1),
        core::TensorShape::from_dims({1, output_ct->ne[1], output_ct->ne[0]}),
        GGML_TYPE_F32);
}

const core::ModuleSchema & AliasFreeActivationModule::static_schema() noexcept {
    return kAliasFreeActivationSchema;
}

}  // namespace engine::modules
