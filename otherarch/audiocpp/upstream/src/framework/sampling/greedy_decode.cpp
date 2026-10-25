#include "engine/framework/sampling/greedy_decode.h"

#include <stdexcept>

namespace engine::sampling {

namespace {

const engine::core::ModulePortSpec kSingleInput[] = {
    {"input", engine::core::PortKind::Activation, false},
};

const engine::core::ModulePortSpec kSingleOutput[] = {
    {"output", engine::core::PortKind::Activation, false},
};

const engine::core::ModuleSchema kGreedyDecodeSchema = {
    "GreedyDecode",
    "sampling.decode",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Takes argmax over the vocabulary axis for each decoding step.",
};

}

const engine::core::ModuleSchema & GreedyDecodeModule::schema() const noexcept {
    return static_schema();
}

engine::core::TensorValue GreedyDecodeModule::build(engine::core::ModuleBuildContext & ctx, const engine::core::TensorValue & logits) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    engine::core::validate_rank_between(logits, 3, 3, "logits");

    auto flat = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, logits),
        engine::core::TensorShape::from_dims({logits.shape.num_elements() / logits.shape.last_dim(), logits.shape.last_dim()}));
    auto argmax = engine::core::wrap_tensor(
        ggml_argmax(ctx.ggml, flat.tensor),
        engine::core::TensorShape::from_dims({flat.shape.dims[0]}),
        GGML_TYPE_I32);
    return engine::core::reshape_tensor(ctx, argmax, engine::core::TensorShape::from_dims({logits.shape.dims[0], logits.shape.dims[1]}));
}

const engine::core::ModuleSchema & GreedyDecodeModule::static_schema() noexcept {
    return kGreedyDecodeSchema;
}

}  // namespace engine::sampling
