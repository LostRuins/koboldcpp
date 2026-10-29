#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/models/neutts/assets.h"

#include <cstddef>
#include <memory>

#include <ggml-backend.h>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::neutts {

struct NeuTTSQwen3Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    modules::CausalDecoderWeights decoder;
};

modules::CausalDecoderConfig make_neutts_qwen3_config(
    const NeuTTSBackboneConfig & config,
    core::BackendType backend_type);

NeuTTSQwen3Weights load_neutts_backbone_weights(
    const NeuTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type);

}  // namespace engine::models::neutts
