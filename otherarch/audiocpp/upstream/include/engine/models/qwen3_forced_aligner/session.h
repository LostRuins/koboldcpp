#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/framework/modules/speech_encoders/qwen3_audio_encoder_runtime.h"
#include "engine/models/qwen3_asr/frontend_whisper.h"
#include "engine/models/qwen3_asr/thinker.h"
#include "engine/models/qwen3_asr/tokenizer_text.h"
#include "engine/models/qwen3_forced_aligner/processor.h"

#include <cstddef>
#include <memory>

namespace engine::models::qwen3_forced_aligner {

class Qwen3ForcedAlignerSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    Qwen3ForcedAlignerSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult run_single(const runtime::TaskRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets_;
    size_t audio_encoder_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t thinker_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t thinker_decode_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t thinker_weight_context_bytes_ = 64ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType audio_encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType thinker_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::models::qwen3_asr::Qwen3ASRTextTokenizer tokenizer_;
    engine::models::qwen3_asr::Qwen3ASRWhisperFrontend frontend_;
    modules::Qwen3AudioEncoderRuntime audio_encoder_;
    engine::models::qwen3_asr::Qwen3ASRThinkerRuntime thinker_;
    Qwen3ForcedAlignProcessor processor_;
    int64_t run_index_ = 0;
};

}  // namespace engine::models::qwen3_forced_aligner
