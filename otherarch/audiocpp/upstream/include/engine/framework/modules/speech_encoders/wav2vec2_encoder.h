#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::modules {

enum class Wav2Vec2FeatureExtractorNorm {
    LayerNormEveryLayer,
    FirstLayerGroupNorm,
};

enum class Wav2Vec2EncoderLayerNormOrder {
    PreNorm,
    PostNorm,
};

struct Wav2Vec2EncoderConfig {
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t num_hidden_layers = 24;
    int64_t output_hidden_layer = 24;
    int64_t num_attention_heads = 16;
    int64_t conv_in_channels = 1;
    int64_t num_conv_pos_embeddings = 128;
    int64_t num_conv_pos_embedding_groups = 16;
    std::vector<int64_t> conv_dim{512, 512, 512, 512, 512, 512, 512};
    std::vector<int64_t> conv_kernel{10, 3, 3, 3, 3, 2, 2};
    std::vector<int64_t> conv_stride{5, 2, 2, 2, 2, 2, 2};
    float layer_norm_eps = 1.0e-5F;
    bool apply_positional_embedding = true;
    bool apply_encoder_input_layer_norm = false;
    bool apply_final_layer_norm = true;
    bool pad_odd_tokens_with_attention_mask = false;
    bool materialize_output = false;
    bool release_graph_after_encode = false;
    int64_t final_projection_size = 0;
    Wav2Vec2FeatureExtractorNorm feature_extractor_norm = Wav2Vec2FeatureExtractorNorm::LayerNormEveryLayer;
    Wav2Vec2EncoderLayerNormOrder encoder_layer_norm_order = Wav2Vec2EncoderLayerNormOrder::PreNorm;
    bool record_final_layer_after_final_norm = false;
};

struct Wav2Vec2EncoderLayerWeightNames {
    std::string pre_attention_layer_norm = "layer_norm";
    std::string attention = "attention";
    std::string post_attention_layer_norm = "self_attn_layer_norm";
    std::string feed_forward_intermediate = "feed_forward.intermediate_dense";
    std::string feed_forward_output = "feed_forward.output_dense";
    std::string final_layer_norm = "final_layer_norm";
};

struct Wav2Vec2EncoderWeightBinding {
    std::string feature_extractor_layers = "feature_extractor.conv_layers";
    std::string feature_extractor_conv = "conv";
    std::string feature_extractor_layer_norm = "layer_norm";
    std::string feature_projection_layer_norm = "feature_projection.layer_norm";
    std::string feature_projection_projection = "feature_projection.projection";
    std::string positional_conv = "encoder.pos_conv_embed.conv";
    std::string encoder_layer_norm = "encoder.layer_norm";
    std::string encoder_layers = "encoder.layers";
    std::string final_projection = "final_proj";
    Wav2Vec2EncoderLayerWeightNames layer;
    assets::TensorStorageType conv_storage_type = assets::TensorStorageType::F32;
    assets::TensorStorageType positional_conv_storage_type = assets::TensorStorageType::F32;
    assets::TensorStorageType projection_storage_type = assets::TensorStorageType::F32;
    assets::TensorStorageType attention_storage_type = assets::TensorStorageType::F32;
    assets::TensorStorageType feed_forward_storage_type = assets::TensorStorageType::F32;
    assets::TensorStorageType final_projection_storage_type = assets::TensorStorageType::F32;
};

struct Wav2Vec2EncoderOutput {
    std::vector<float> hidden_states;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct Wav2Vec2EncoderLayerOutput {
    std::vector<int64_t> layer_indices;
    std::vector<std::vector<float>> hidden_states;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct Wav2Vec2EncoderRunConfig {
    int64_t output_hidden_layer = -1;
    bool apply_final_projection = false;
};

struct Wav2Vec2EncoderWeights {
    Wav2Vec2EncoderConfig config;
    std::filesystem::path source_path;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, core::TensorValue> tensors;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

class Wav2Vec2EncoderRuntime {
public:
    static Wav2Vec2EncoderRuntime load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        core::BackendConfig backend,
        Wav2Vec2EncoderConfig config = {});
    static Wav2Vec2EncoderRuntime load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        Wav2Vec2EncoderConfig config = {});
    static Wav2Vec2EncoderRuntime load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        Wav2Vec2EncoderConfig config,
        Wav2Vec2EncoderWeightBinding binding);

    Wav2Vec2EncoderRuntime() = default;
    Wav2Vec2EncoderRuntime(
        std::shared_ptr<const Wav2Vec2EncoderWeights> weights,
        core::BackendConfig backend);

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const Wav2Vec2EncoderWeights> & weights() const noexcept;
    int64_t hidden_size() const noexcept;
    int64_t loaded_tensor_count() const noexcept;
    int64_t parameter_count() const noexcept;
    Wav2Vec2EncoderOutput encode(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples) const;
    Wav2Vec2EncoderOutput encode(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples,
        Wav2Vec2EncoderRunConfig run_config) const;
    Wav2Vec2EncoderLayerOutput encode_layers(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples,
        const std::vector<int64_t> & output_layers) const;
    void release_runtime_graph();

private:
    struct State;

    std::shared_ptr<const Wav2Vec2EncoderWeights> weights_;
    core::BackendConfig backend_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules
