#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/types.h"

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models {

namespace qwen3_tts {

struct Qwen3SpeechTokenizerDecoderWeights;
class Qwen3SpeechTokenizerDecoderGraph;
class RknnSpeechDecoder;
class Qwen3SpeechDecoderStreamState;

// Optional Rockchip NPU backend for the decoder stage: the path(s) of the
// stateful step export. Empty `rknn_model_path` keeps the ggml graph.
// `window_path` (optional) is a larger-B export used to voice each whole
// streaming window from a zeroed cache — window semantics matching the ggml
// streaming path, at the cost of re-voicing the left context per event.
struct Qwen3SpeechTokenizerDecoderRknn {
    std::string model_path;
    std::string first_model_path;
    std::string window_path;
};

class Qwen3SpeechTokenizerDecoderRuntime {
public:
    static constexpr int64_t kLeftContextFrames = 25;
    Qwen3SpeechTokenizerDecoderRuntime(
        std::shared_ptr<const Qwen3TTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type,
        Qwen3TTSPerfMode perf_mode,
        Qwen3SpeechTokenizerDecoderRknn rknn = {},
        bool stateful_stream = false);
    ~Qwen3SpeechTokenizerDecoderRuntime();

    runtime::AudioBuffer decode(const Qwen3SpeechCodes & codec_codes) const;
    runtime::AudioBuffer decode_and_trim_reference(
        const Qwen3SpeechCodes & reference_codes,
        const Qwen3SpeechCodes & generated_codes) const;
    runtime::AudioBuffer decode_stream_chunk(
        const Qwen3SpeechCodes & codec_codes,
        int64_t context_frames,
        int64_t frames_per_chunk) const;

    // Streaming state hooks, wired to the streaming audio controller's
    // reset/drain callbacks. Reset clears the GGML stateful mode; on the stateful
    // RKNN path they zero the carried cache and voice the sub-group tail left
    // in the buffer at the end of an utterance.
    void reset_stream() const;
    runtime::AudioBuffer flush_stream() const;

private:
    std::shared_ptr<const Qwen3TTSAssets> assets_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<const Qwen3SpeechTokenizerDecoderWeights> weights_;
    size_t graph_arena_bytes_ = 0;
    Qwen3TTSPerfMode perf_mode_ = Qwen3TTSPerfMode::Standard;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    std::unique_ptr<Qwen3SpeechDecoderStreamState> stream_state_;
    mutable int64_t stream_graph_capacity_ = 0;
    mutable std::map<int64_t, std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph>> stateful_graphs_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> graph_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> stream_graph_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> stream_initial_graph_;
    // Non-null when the decoder stage runs on the NPU; then the ggml weights
    // are never loaded.
    mutable std::unique_ptr<RknnSpeechDecoder> rknn_;
    // Always present to keep this public class layout identical when the private
    // Strix Halo compile definition differs between translation units.
    mutable std::array<std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph>, 2> optimized_graphs_;
};

}  // namespace qwen3_tts
}  // namespace engine::models
