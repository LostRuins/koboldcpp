#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/codecs/s3gen_runtime.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::chatterbox_turbo {

// Loads and runs Chatterbox Turbo's S3Gen half (speech tokens -> waveform) by delegating to the
// shared framework S3Gen runtime's meanflow decoder and HiFT vocoder. The native, repacked GGUF (see
// tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py) stores S3Gen tensors
// under the same names those loaders already expect, so no name translation is needed here.
//
// Only the built-in default voice baked into the T3 GGUF's `conds.gen.*` tensors is supported.
// The turbo checkpoint's `s3.se` speaker encoder and `s3.tok` speech tokenizer sections are not
// used by this family.
class ChatterboxTurboS3Gen {
public:
    static std::shared_ptr<ChatterboxTurboS3Gen> load(
        std::shared_ptr<const engine::assets::TensorSource> s3gen_source,
        const engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

    // speech_tokens: T3 output (S3 codebook ids, < 6561; caller strips control tokens).
    engine::codecs::s3gen::S3GenInferenceOutputs synthesize(
        const engine::codecs::s3gen::S3GenConditioning & ref_dict,
        const std::vector<int32_t> & speech_tokens,
        uint64_t flow_seed,
        uint64_t vocoder_seed) const;

private:
    std::unique_ptr<engine::codecs::s3gen::S3GenRuntime> runtime_;
};

}  // namespace engine::community_models::chatterbox_turbo
