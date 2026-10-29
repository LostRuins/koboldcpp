#pragma once

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/models/breeze_tts/assets.h"
#include "engine/models/breeze_tts/speech_decoder.h"
#include "engine/models/breeze_tts/text_encoder.h"
#include "engine/models/breeze_tts/tokenizer_text.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::breeze_tts {

struct BreezeGenerationRequest {
    std::string text;
    std::string instruction;
    std::string reference_text;
    std::optional<engine::runtime::AudioBuffer> reference_audio;
    std::optional<BreezeSpeechCodes> reference_codes;
    float guidance_scale = 1.0F;
    float temperature = 0.9F;
    float depth_temperature = 0.9F;
    int64_t top_k = 50;
    float top_p = 1.0F;
    int64_t max_tokens = 1500;
    uint64_t seed = 0;
};

// BreezeTTS 2's reference inference rounds activations to bf16 (and keeps a bf16
// KV cache). That is what the model was trained with, but on backends without a
// cheap fused cast it costs a visible share of the AR loop, so the choice is
// explicit: 'auto' keeps the reference behavior on CUDA/HIP/Vulkan and stays on
// the faster f32 path on Metal, 'on'/'off' force it either way.
enum class Bf16ActivationMode {
    Auto,
    On,
    Off,
};

class BreezeGeneratorRuntime {
public:
    BreezeGeneratorRuntime(
        std::shared_ptr<const BreezeTTSAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        engine::core::AttentionPreference attention_preference = engine::core::AttentionPreference::Auto,
        Bf16ActivationMode bf16_activations = Bf16ActivationMode::Auto);
    ~BreezeGeneratorRuntime();

    engine::runtime::AudioBuffer generate(
        const BreezeGenerationRequest & request,
        BreezeSpeechCodes * generated_codes = nullptr);
    BreezeSpeechCodes encode_reference(const engine::runtime::AudioBuffer & audio) const;
    engine::runtime::AudioBuffer generate_stream(
        const BreezeGenerationRequest & request,
        engine::runtime::StreamingAudioConfig config,
        engine::runtime::StreamEventCallback sink,
        BreezeSpeechCodes * generated_codes = nullptr);
    void end_stream();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::breeze_tts
