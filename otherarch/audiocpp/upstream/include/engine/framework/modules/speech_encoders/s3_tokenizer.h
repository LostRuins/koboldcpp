#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

struct S3TokenizerConfig {
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    std::string tensor_prefix = "tokenizer";
    int64_t input_mels = 128;
    int64_t channels = 1280;
    int64_t heads = 20;
    int64_t head_dim = 64;
    int64_t conv_stride = 2;
    int64_t conv_padding = 1;
    int64_t conv_kernel = 3;
    int64_t fsmn_kernel = 31;
    int64_t fsmn_padding = 15;
    int64_t quantizer_dim = 8;
    int64_t quantizer_base = 3;
    int64_t quantizer_offset = 1;
    float quantizer_scale = 0.9990000128746033F;
    size_t weight_context_bytes = 768ull * 1024ull * 1024ull;
};

struct S3TokenizerOutputs {
    std::vector<int32_t> tokens;
    int64_t token_count = 0;
};

struct S3TokenizerWeights {
    struct BlockWeights {
        NormWeights attn_ln;
        LinearWeights attn_qkv_packed;
        LinearWeights attn_out;
        DepthwiseConv1dWeights fsmn;
        NormWeights mlp_ln;
        LinearWeights mlp_fc1;
        LinearWeights mlp_fc2;
    };

    S3TokenizerConfig config;
    Conv1dWeights conv1;
    Conv1dWeights conv2;
    std::vector<BlockWeights> blocks;
    LinearWeights quantizer_project_down;
    const core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<core::BackendWeightStore> store;
};

class S3TokenizerComponent {
public:
    static S3TokenizerComponent load_from_source(
        std::shared_ptr<const assets::TensorSource> source,
        const core::ExecutionContext & execution_context,
        S3TokenizerConfig config = {});

    S3TokenizerComponent() = default;
    S3TokenizerComponent(
        std::shared_ptr<const S3TokenizerWeights> weights,
        const core::ExecutionContext & execution_context);
    ~S3TokenizerComponent();
    S3TokenizerComponent(S3TokenizerComponent &&) noexcept;
    S3TokenizerComponent & operator=(S3TokenizerComponent &&) noexcept;
    S3TokenizerComponent(const S3TokenizerComponent &) = delete;
    S3TokenizerComponent & operator=(const S3TokenizerComponent &) = delete;

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const S3TokenizerWeights> & weights() const noexcept;
    S3TokenizerOutputs tokenize(
        const runtime::AudioBuffer & audio,
        std::optional<int64_t> max_len) const;
    void release_runtime_cache() const;

private:
    struct State;

    std::shared_ptr<const S3TokenizerWeights> weights_;
    const core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules
