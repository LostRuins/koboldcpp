#include "attention_internal.h"

namespace engine::modules {

using namespace attention::internal;

PositionalEncodingModule::PositionalEncodingModule(PositionalEncodingConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "PositionalEncodingConfig.hidden_size");
}

const PositionalEncodingConfig & PositionalEncodingModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & PositionalEncodingModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue PositionalEncodingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PositionalEncodingWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");

    const int64_t frames = input.shape.dims[input.shape.rank - 2];
    core::validate_shape(weights.encoding, core::TensorShape::from_dims({frames, config_.hidden_size}), "encoding");

    core::TensorShape repeated_shape = {};
    repeated_shape.rank = input.shape.rank;
    for (size_t i = 0; i + 2 < input.shape.rank; ++i) {
        repeated_shape.dims[i] = 1;
    }
    repeated_shape.dims[input.shape.rank - 2] = frames;
    repeated_shape.dims[input.shape.rank - 1] = config_.hidden_size;

    auto encoding = core::reshape_tensor(ctx, weights.encoding, repeated_shape);
    auto expanded = core::wrap_tensor(ggml_repeat(ctx.ggml, encoding.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, expanded.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & PositionalEncodingModule::static_schema() noexcept {
    return kPositionalEncodingSchema;
}

}  // namespace engine::modules
