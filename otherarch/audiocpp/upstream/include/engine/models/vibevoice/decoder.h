#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/vibevoice/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models::vibevoice {

class VibeVoiceQwen2PrefillGraph;
class VibeVoiceQwen2CachedStepGraph;
class VibeVoiceQwen2CachedBatchStepGraph;
class VibeVoiceQwen2EmbeddingGraph;

class VibeVoiceQwen2CachedState final {
public:
    VibeVoiceQwen2CachedState();
    ~VibeVoiceQwen2CachedState();

    VibeVoiceQwen2CachedState(const VibeVoiceQwen2CachedState &) = delete;
    VibeVoiceQwen2CachedState & operator=(const VibeVoiceQwen2CachedState &) = delete;
    VibeVoiceQwen2CachedState(VibeVoiceQwen2CachedState &&) noexcept;
    VibeVoiceQwen2CachedState & operator=(VibeVoiceQwen2CachedState &&) noexcept;

private:
    friend class VibeVoiceQwen2WeightsRuntime;
    friend class VibeVoiceQwen2CachedBatchStepGraph;

    std::unique_ptr<VibeVoiceQwen2CachedStepGraph> graph_;
    runtime::TransformerKVState pending_state_;
    const VibeVoiceQwen2CachedBatchStepGraph * batch_owner_ = nullptr;
    bool graph_has_state_ = false;
};

struct VibeVoiceDecoderLogits {
    std::vector<float> values;
    int64_t vocab_size = 0;
};

struct VibeVoiceTokenEmbeddings {
    std::vector<float> values;
    int64_t steps = 0;
    int64_t hidden_size = 0;
};

struct VibeVoiceDecoderHidden {
    std::vector<float> values;
    int64_t dims = 0;
};

struct VibeVoiceDecoderResult {
    VibeVoiceDecoderLogits logits;
    VibeVoiceDecoderHidden last_hidden;
};

struct VibeVoiceDecoderPrefillOutput {
    VibeVoiceDecoderResult result;
    runtime::TransformerKVState state;
};

struct VibeVoiceQwen2MLPWeights {
    modules::LinearWeights gate_proj;
    modules::LinearWeights up_proj;
    modules::LinearWeights down_proj;
};

struct VibeVoiceQwen2LayerWeights {
    assets::TensorDataF32 input_norm;
    modules::AttentionWeights self_attention;
    assets::TensorDataF32 post_norm;
    VibeVoiceQwen2MLPWeights mlp;
};

struct VibeVoiceQwen2Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    core::TensorValue lm_head;
    std::vector<VibeVoiceQwen2LayerWeights> layers;
    assets::TensorDataF32 norm;
};

struct VibeVoiceDecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

class VibeVoiceQwen2WeightsRuntime final {
public:
    VibeVoiceQwen2WeightsRuntime(
        std::shared_ptr<const VibeVoiceAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t weight_context_bytes = 256ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 128ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);

    ~VibeVoiceQwen2WeightsRuntime();

    VibeVoiceQwen2WeightsRuntime(const VibeVoiceQwen2WeightsRuntime &) = delete;
    VibeVoiceQwen2WeightsRuntime & operator=(const VibeVoiceQwen2WeightsRuntime &) = delete;

    const VibeVoiceAssets & assets() const noexcept;
    const VibeVoiceQwen2Weights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::ConstantTensorCache & constants() const noexcept;
    ggml_type cache_type() const noexcept;
    int threads() const noexcept;

    VibeVoiceTokenEmbeddings embed_tokens(const std::vector<int32_t> & input_ids) const;
    VibeVoiceDecoderPrefillOutput prefill_embeddings(const std::vector<float> & embeddings, int64_t steps) const;
    std::vector<VibeVoiceDecoderPrefillOutput> prefill_embeddings_batch(
        const std::vector<std::vector<float>> & embeddings,
        int64_t steps) const;
    void reset_cached_state(VibeVoiceQwen2CachedState & state, runtime::TransformerKVState prefill_state) const;
    VibeVoiceDecoderResult cached_step(
        const std::vector<float> & embedding,
        VibeVoiceQwen2CachedState & state,
        int64_t cache_capacity) const;
    std::vector<VibeVoiceDecoderResult> cached_step_batch(
        const std::vector<std::vector<float>> & embeddings,
        const std::vector<VibeVoiceQwen2CachedState *> & states,
        int64_t cache_capacity) const;
    void release_prompt_graphs() const;

private:
    VibeVoiceQwen2CachedBatchStepGraph * find_cached_batch_graph(
        const VibeVoiceQwen2CachedState & state) const;
    void export_and_drop_cached_batch_graph(const VibeVoiceQwen2CachedBatchStepGraph * owner) const;

    std::shared_ptr<const VibeVoiceAssets> assets_;
    std::shared_ptr<const VibeVoiceQwen2Weights> weights_;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    mutable std::unique_ptr<VibeVoiceQwen2EmbeddingGraph> embedding_graph_;
    mutable std::unique_ptr<VibeVoiceQwen2PrefillGraph> prefill_graph_;
    mutable std::vector<std::unique_ptr<VibeVoiceQwen2CachedBatchStepGraph>> cached_batch_graphs_;
    ggml_backend_t backend_ = nullptr;
    ggml_type cache_type_ = GGML_TYPE_F16;
    int threads_ = 1;
};

VibeVoiceQwen2Weights load_vibevoice_decoder_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceQwen2LayerWeights & weights,
    const VibeVoiceQwen2Config & config,
    core::ConstantTensorCache & constants,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt);

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer_static_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceQwen2LayerWeights & weights,
    const VibeVoiceQwen2Config & config,
    core::ConstantTensorCache & constants,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask);

}  // namespace engine::models::vibevoice
