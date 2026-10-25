#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::modules {

struct WhisperFrontendComponentConfig {
    std::string name = "framework.whisper_frontend";
    size_t weight_context_bytes = 256ull * 1024ull * 1024ull;
    size_t graph_context_bytes = 512ull * 1024ull * 1024ull;
    assets::TensorStorageType matmul_weight_storage_type = assets::TensorStorageType::Native;
    assets::TensorStorageType conv_weight_storage_type = assets::TensorStorageType::Native;
};

struct WhisperFrontendComponentWeights {
    WhisperEmbeddingConfig config;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    WhisperEmbeddingWeights embedding;
};

class WhisperFrontendComponent {
public:
    static WhisperFrontendComponent load_openai_layout(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        WhisperEmbeddingConfig config,
        WhisperFrontendComponentConfig component_config = {});

    static WhisperFrontendComponent load_hf_encoder_layout(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        WhisperFrontendComponentConfig component_config = {});

    WhisperFrontendComponent() = default;
    WhisperFrontendComponent(
        std::shared_ptr<const WhisperFrontendComponentWeights> weights,
        WhisperFrontendComponentConfig component_config);

    const WhisperEmbeddingConfig & config() const;
    int64_t channels() const noexcept;
    std::vector<float> encode_log_mel(const std::vector<float> & log_mel) const;
    void release_runtime_graph() const;

private:
    struct State;

    std::shared_ptr<const WhisperFrontendComponentWeights> weights_;
    WhisperFrontendComponentConfig component_config_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules
