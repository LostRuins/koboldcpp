#include "engine/framework/modules/lookup_modules.h"

#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kEmbeddingInputs[] = {
    {"indices", core::PortKind::Activation, false},
    {"weight", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kEmbeddingOutputs[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kEmbeddingSchema = {
    "Embedding",
    "nn.lookup",
    kEmbeddingInputs,
    2,
    kEmbeddingOutputs,
    1,
    "Looks up token or frame ids in an embedding table.",
};

const core::ModuleSchema kPitchEmbedSchema = {
    "PitchEmbed",
    "nn.lookup",
    kEmbeddingInputs,
    2,
    kEmbeddingOutputs,
    1,
    "Looks up per-frame pitch bins in an embedding table.",
};

const core::ModuleSchema kEnergyEmbedSchema = {
    "EnergyEmbed",
    "nn.lookup",
    kEmbeddingInputs,
    2,
    kEmbeddingOutputs,
    1,
    "Looks up per-frame energy bins in an embedding table.",
};

const core::ModuleSchema kCodebookLookupSchema = {
    "CodebookLookup",
    "nn.lookup",
    kEmbeddingInputs,
    2,
    kEmbeddingOutputs,
    1,
    "Looks up vector-quantizer codebook entries from discrete indices.",
};

core::TensorShape embedding_output_shape(const core::TensorShape & index_shape, int64_t embedding_dim) {
    if (index_shape.rank == 0 || index_shape.rank >= core::kMaxTensorRank) {
        throw std::runtime_error("Embedding indices rank must be between 1 and 3");
    }

    core::TensorShape output = {};
    output.rank = index_shape.rank + 1;
    for (size_t i = 0; i < index_shape.rank; ++i) {
        output.dims[i] = index_shape.dims[i];
    }
    output.dims[output.rank - 1] = embedding_dim;
    return output;
}

}  // namespace

EmbeddingModule::EmbeddingModule(EmbeddingConfig config) : config_(config) {
    if (config_.num_embeddings <= 0 || config_.embedding_dim <= 0) {
        throw std::runtime_error("EmbeddingConfig dimensions must be positive");
    }
}

const EmbeddingConfig & EmbeddingModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & EmbeddingModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue EmbeddingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & indices,
    const core::TensorValue & weight) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (indices.type != GGML_TYPE_I32) {
        throw std::runtime_error("Embedding indices must be GGML_TYPE_I32");
    }
    core::validate_rank_between(indices, 1, core::kMaxTensorRank - 1, "indices");
    core::validate_shape(
        weight,
        core::TensorShape::from_dims({config_.num_embeddings, config_.embedding_dim}),
        "weight");

    const auto output_shape = embedding_output_shape(indices.shape, config_.embedding_dim);
    const auto flat_indices_shape = core::TensorShape::from_dims({indices.shape.num_elements()});
    const auto flat_indices = core::reshape_tensor(ctx, indices, flat_indices_shape);
    const auto flat_output = core::wrap_tensor(
        ggml_get_rows(ctx.ggml, weight.tensor, flat_indices.tensor),
        core::TensorShape::from_dims({indices.shape.num_elements(), config_.embedding_dim}),
        GGML_TYPE_F32);
    return core::reshape_tensor(ctx, flat_output, output_shape);
}

const core::ModuleSchema & EmbeddingModule::static_schema() noexcept {
    return kEmbeddingSchema;
}

PitchEmbedModule::PitchEmbedModule(IndexedEmbeddingConfig config) : config_(config) {
    if (config_.num_embeddings <= 0 || config_.embedding_dim <= 0) {
        throw std::runtime_error("IndexedEmbeddingConfig dimensions must be positive");
    }
}

const IndexedEmbeddingConfig & PitchEmbedModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & PitchEmbedModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue PitchEmbedModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & indices,
    const core::TensorValue & weight) const {
    return EmbeddingModule({config_.num_embeddings, config_.embedding_dim}).build(ctx, indices, weight);
}

const core::ModuleSchema & PitchEmbedModule::static_schema() noexcept {
    return kPitchEmbedSchema;
}

EnergyEmbedModule::EnergyEmbedModule(IndexedEmbeddingConfig config) : config_(config) {
    if (config_.num_embeddings <= 0 || config_.embedding_dim <= 0) {
        throw std::runtime_error("IndexedEmbeddingConfig dimensions must be positive");
    }
}

const IndexedEmbeddingConfig & EnergyEmbedModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & EnergyEmbedModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue EnergyEmbedModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & indices,
    const core::TensorValue & weight) const {
    return EmbeddingModule({config_.num_embeddings, config_.embedding_dim}).build(ctx, indices, weight);
}

const core::ModuleSchema & EnergyEmbedModule::static_schema() noexcept {
    return kEnergyEmbedSchema;
}

CodebookLookupModule::CodebookLookupModule(IndexedEmbeddingConfig config) : config_(config) {
    if (config_.num_embeddings <= 0 || config_.embedding_dim <= 0) {
        throw std::runtime_error("IndexedEmbeddingConfig dimensions must be positive");
    }
}

const IndexedEmbeddingConfig & CodebookLookupModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & CodebookLookupModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue CodebookLookupModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & indices,
    const core::TensorValue & weight) const {
    return EmbeddingModule({config_.num_embeddings, config_.embedding_dim}).build(ctx, indices, weight);
}

const core::ModuleSchema & CodebookLookupModule::static_schema() noexcept {
    return kCodebookLookupSchema;
}

}  // namespace engine::modules
