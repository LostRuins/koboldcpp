#pragma once

#include "engine/framework/core/module.h"

#include <cstdint>

namespace engine::modules {

struct EmbeddingConfig {
    int64_t num_embeddings = 0;
    int64_t embedding_dim = 0;
};

class EmbeddingModule {
public:
    explicit EmbeddingModule(EmbeddingConfig config);

    const EmbeddingConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & indices,
        const core::TensorValue & weight) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    EmbeddingConfig config_;
};

struct IndexedEmbeddingConfig {
    int64_t num_embeddings = 0;
    int64_t embedding_dim = 0;
};

class PitchEmbedModule {
public:
    explicit PitchEmbedModule(IndexedEmbeddingConfig config);

    const IndexedEmbeddingConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & indices,
        const core::TensorValue & weight) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    IndexedEmbeddingConfig config_;
};

class EnergyEmbedModule {
public:
    explicit EnergyEmbedModule(IndexedEmbeddingConfig config);

    const IndexedEmbeddingConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & indices,
        const core::TensorValue & weight) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    IndexedEmbeddingConfig config_;
};

class CodebookLookupModule {
public:
    explicit CodebookLookupModule(IndexedEmbeddingConfig config);

    const IndexedEmbeddingConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & indices,
        const core::TensorValue & weight) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    IndexedEmbeddingConfig config_;
};

}  // namespace engine::modules
