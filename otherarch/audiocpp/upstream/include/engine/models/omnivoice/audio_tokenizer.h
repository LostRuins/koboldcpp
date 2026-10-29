#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/types.h"

#include <memory>

namespace engine::models::omnivoice {

struct OmniVoiceReferenceAudioOptions {
    bool preprocess_prompt = true;
    bool has_reference_text = false;
};

struct OmniVoiceHiggsAudioV2TokenizerRuntimeStats {
    bool encoder_graph_rebuilt = false;
    int64_t encoder_frame_capacity = 0;
    int64_t encoder_acoustic_sample_capacity = 0;
    int64_t encoder_semantic_sample_capacity = 0;
    double encoder_rebuild_ms = 0.0;
    bool decoder_graph_rebuilt = false;
    int64_t decoder_frame_capacity = 0;
    int64_t decoder_codebook_capacity = 0;
    double decoder_rebuild_ms = 0.0;
};

class OmniVoiceHiggsAudioV2TokenizerRuntime {
public:
    OmniVoiceHiggsAudioV2TokenizerRuntime(
        std::shared_ptr<const OmniVoiceAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~OmniVoiceHiggsAudioV2TokenizerRuntime();

    OmniVoiceAudioTokens encode_reference_audio(
        const runtime::AudioBuffer & audio,
        const OmniVoiceReferenceAudioOptions & options);
    runtime::AudioBuffer decode_audio_tokens(const OmniVoiceGeneratedAudioTokens & audio_tokens);
    void release_runtime_graphs();
    const OmniVoiceHiggsAudioV2TokenizerRuntimeStats & last_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::omnivoice
