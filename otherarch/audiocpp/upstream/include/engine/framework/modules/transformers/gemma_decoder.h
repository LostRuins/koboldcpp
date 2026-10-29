#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/projected_grouped_self_attention.h"
#include "engine/framework/modules/linear_module.h"

#include <ggml-backend.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::modules {

struct GemmaDecoderStackConfig {
    int64_t hidden_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    int64_t sliding_window_pattern = 0;
    float rope_theta = 1000000.0F;
    float local_rope_theta = 10000.0F;
    float rope_freq_scale = 1.0F;
    float rms_norm_eps = 1.0e-6F;
    float query_pre_attn_scalar = 256.0F;
    bool scale_embeddings = true;
    bool use_fast_cuda_projection = true;
};

struct GemmaDecoderLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue post_attention_norm;
    core::TensorValue pre_feedforward_norm;
    core::TensorValue post_feedforward_norm;
    ProjectedGroupedSelfAttentionWeights self_attention;
    GatedFeedForwardWeights feed_forward;
};

struct GemmaDecoderStackWeights {
    core::TensorValue embed_tokens;
    std::vector<GemmaDecoderLayerWeights> layers;
    core::TensorValue norm;
};

struct GemmaDecoderLayerWeightNames {
    std::string input_norm = "input_layernorm.weight";
    std::string post_attention_norm = "post_attention_layernorm.weight";
    std::string pre_feedforward_norm = "pre_feedforward_layernorm.weight";
    std::string post_feedforward_norm = "post_feedforward_layernorm.weight";
    std::string q_norm = "self_attn.q_norm.weight";
    std::string k_norm = "self_attn.k_norm.weight";
    std::string q_proj = "self_attn.q_proj";
    std::string k_proj = "self_attn.k_proj";
    std::string v_proj = "self_attn.v_proj";
    std::string o_proj = "self_attn.o_proj";
    std::string gate_proj = "mlp.gate_proj";
    std::string up_proj = "mlp.up_proj";
    std::string down_proj = "mlp.down_proj";
};

struct GemmaDecoderWeightBinding {
    std::string model_prefix = "model";
    std::string layers_prefix = "layers";
    std::string embed_tokens = "embed_tokens.weight";
    std::string final_norm = "norm.weight";
    GemmaDecoderLayerWeightNames layer;
    assets::TensorStorageType embedding_storage_type = assets::TensorStorageType::Native;
    assets::TensorStorageType norm_storage_type = assets::TensorStorageType::Native;
    assets::TensorStorageType projection_storage_type = assets::TensorStorageType::Native;
};

struct GemmaDecoderStackOutputs {
    core::TensorValue hidden;
    std::vector<core::TensorValue> captured_hidden_states;
};

using GemmaDecoderTensorSourceResolver = std::function<const assets::TensorSource &(std::string_view)>;

GemmaDecoderStackWeights load_gemma_decoder_stack_weights(
    core::BackendWeightStore & store,
    const GemmaDecoderStackConfig & config,
    const GemmaDecoderWeightBinding & binding,
    const GemmaDecoderTensorSourceResolver & resolver);

class GemmaDecoderStackModule {
public:
    explicit GemmaDecoderStackModule(GemmaDecoderStackConfig config);

    const GemmaDecoderStackConfig & config() const noexcept;

    GemmaDecoderStackOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input_ids,
        const core::TensorValue & positions,
        const core::TensorValue & additive_attention_mask,
        const GemmaDecoderStackWeights & weights,
        bool capture_input_and_layers = false) const;

private:
    GemmaDecoderStackConfig config_;
};

class GemmaDecoderComponent {
public:
    static GemmaDecoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        ggml_backend_t backend,
        core::BackendType backend_type,
        GemmaDecoderStackConfig config,
        GemmaDecoderWeightBinding binding,
        size_t weight_context_bytes);

    static GemmaDecoderComponent load_from_resolver(
        ggml_backend_t backend,
        core::BackendType backend_type,
        GemmaDecoderStackConfig config,
        GemmaDecoderWeightBinding binding,
        size_t weight_context_bytes,
        const GemmaDecoderTensorSourceResolver & resolver);

    GemmaDecoderComponent() = default;
    GemmaDecoderComponent(
        GemmaDecoderStackConfig config,
        std::shared_ptr<core::BackendWeightStore> store,
        std::shared_ptr<const GemmaDecoderStackWeights> weights);

    const GemmaDecoderStackConfig & config() const noexcept;
    const std::shared_ptr<const GemmaDecoderStackWeights> & weights() const noexcept;
    const std::shared_ptr<core::BackendWeightStore> & store() const noexcept;

    GemmaDecoderStackOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input_ids,
        const core::TensorValue & positions,
        const core::TensorValue & additive_attention_mask,
        bool capture_input_and_layers = false) const;

private:
    GemmaDecoderStackConfig config_;
    std::shared_ptr<core::BackendWeightStore> store_;
    std::shared_ptr<const GemmaDecoderStackWeights> weights_;
};

}  // namespace engine::modules
