#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::qwen3_tts {

// Rockchip NPU (RKNN) backend for the speech-tokenizer decoder.
//
// The graph is the stateful step export of the same decoder (see
// qwen3-tts-axera-russian/scripts/export_vocoder_stateful.py): one call voices a
// fixed group of B token frames and carries every history-dependent tail
// (conv caches + the sliding-window KV) through a flat cache tensor in graph I/O,
// so there are no chunk seams and no left-context window. Inputs
// `audio_codes [1,16,B] i64`, `freqs_in [B,head_dim/2] f32` (RoPE angles,
// reduced mod 2*pi on the host), `cache_valid [1] f32` (the COUNT of filled KV
// slots, not a flag) and `cache_in [cache_numel] f32`; outputs `audio_values`
// and `cache_out`.
//
// All decode entry points are const-callable from one session thread; the
// streaming state (cache, absolute token position, unvoiced-token buffer) lives
// here and is reset per utterance.
class RknnSpeechDecoder {
public:
    static constexpr int64_t kSamplesPerToken = 1920;  // 24 kHz / 12.5 Hz

    // `model_path` voices every group; `first_model_path` (optional) is a
    // smaller-B export used for the first group only, which is what a listener
    // waits for. The cache layout is B-independent, so the small model's cache
    // feeds the big one. `num_quantizers` and `sliding_window` come from the
    // checkpoint's speech_tokenizer_config.
    //
    // `window_path` (optional) switches streaming to window semantics: the same
    // stateful export run at zero cache and `cache_valid` 0 over the whole
    // [left context + new] window, which is exactly the full decode over the
    // window — the function the ggml streaming path computes per chunk, context
    // re-voiced every event. The first window (no context) rides the
    // first-group model when one is loaded. The context frames' audio is
    // dropped, flush is a no-op, and the stateful streaming state stays unused.
    // A `window_path` naming the same file as `model_path` (compared as files,
    // as the residency accounting compares them, not as strings) aliases the
    // main engine (one resident context voices groups offline and windows
    // streaming).
    RknnSpeechDecoder(
        const std::string & model_path,
        const std::string & first_model_path,
        const std::string & window_path,
        int64_t num_quantizers,
        int64_t head_dim,
        int64_t sliding_window,
        float rope_theta);
    ~RknnSpeechDecoder();

    RknnSpeechDecoder(const RknnSpeechDecoder &) = delete;
    RknnSpeechDecoder & operator=(const RknnSpeechDecoder &) = delete;

    // Tokens voiced per NPU step (the first-group model when one is loaded).
    int64_t group_tokens() const;
    int64_t first_group_tokens() const;

    // Whole-utterance decode from a clean state. `codes` is row-major
    // [frames][num_quantizers]. The trailing partial group is zero-padded and
    // its audio truncated, which is the export's own flush convention.
    std::vector<float> decode(const int32_t * codes, int64_t frames, int64_t num_quantizers);

    // Streaming step. `codes` holds `context_frames + new_frames` row-major
    // frames of which the first `context_frames` were already voiced and live in
    // the cache; only the tail is fed. `flush` voices the buffered remainder
    // (< one group) with zero padding and resets the utterance. With a window
    // engine loaded the context is fed too (re-voiced, not cache-covered), its
    // audio is dropped, and `flush` returns nothing.
    std::vector<float> step(
        const int32_t * codes,
        int64_t context_frames,
        int64_t new_frames,
        int64_t num_quantizers,
        bool flush);

    // Start a new utterance: zero the cache, rewind the position, drop the
    // unvoiced buffer.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::qwen3_tts
