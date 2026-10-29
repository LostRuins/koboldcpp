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

struct WavlmEncoderConfig {
    int64_t hidden_size = 768;
    int64_t intermediate_size = 3072;
    int64_t num_hidden_layers = 12;
    int64_t output_hidden_layer = 12;
    int64_t num_attention_heads = 12;
    int64_t num_buckets = 320;
    int64_t max_distance = 800;
    int64_t num_conv_pos_embeddings = 128;
    int64_t num_conv_pos_embedding_groups = 16;
    std::vector<int64_t> conv_dim{512, 512, 512, 512, 512, 512, 512};
    std::vector<int64_t> conv_kernel{10, 3, 3, 3, 3, 2, 2};
    std::vector<int64_t> conv_stride{5, 2, 2, 2, 2, 2, 2};
    float layer_norm_eps = 1.0e-5F;
    bool normalize_input = false;
    bool conv_feature_layer_norm = false;
    bool transformer_layer_norm_first = false;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
};

struct WavlmEncoderOutput {
    std::vector<float> hidden_states;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct WavlmEncoderLayerOutput {
    std::vector<int64_t> layer_indices;
    std::vector<std::vector<float>> hidden_states;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

class WavlmPositionBiasCache;

struct WavlmEncoderWeights {
    WavlmEncoderConfig config;
    std::filesystem::path source_path;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, core::TensorValue> tensors;
    std::vector<float> relative_attention_embedding;
    std::shared_ptr<WavlmPositionBiasCache> position_bias_cache;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

class WavlmEncoderComponent {
public:
    static WavlmEncoderComponent load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        core::BackendConfig backend,
        WavlmEncoderConfig config = {});
    static WavlmEncoderComponent load_from_tensor_source(
        const assets::TensorSource & source,
        core::BackendConfig backend,
        WavlmEncoderConfig config = {});

    WavlmEncoderComponent() = default;
    WavlmEncoderComponent(
        std::shared_ptr<const WavlmEncoderWeights> weights,
        core::BackendConfig backend,
        bool reuse_graph = true);

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const WavlmEncoderWeights> & weights() const noexcept;
    int64_t hidden_size() const noexcept;
    int64_t loaded_tensor_count() const noexcept;
    int64_t parameter_count() const noexcept;

    WavlmEncoderOutput encode(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples) const;

    WavlmEncoderLayerOutput encode_layers(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples,
        const std::vector<int64_t> & output_layers) const;
    void release_runtime_graph();

private:
    struct State;

    std::shared_ptr<const WavlmEncoderWeights> weights_;
    core::BackendConfig backend_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules
