#include "engine/models/qwen3_tts/rknn_speech_decoder.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/rknn/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::qwen3_tts {

namespace {

#if defined(ENGINE_HAVE_RKNN)
using Clock = std::chrono::steady_clock;

constexpr int64_t kSampleRate = 24000;

// Prefix of every message this sidecar raises, shared with the plumbing in
// engine::rknn so a failure names the stage that failed.
const char * const kLabel = "Qwen3 speech decoder RKNN";
#endif

}  // namespace

struct RknnSpeechDecoder::Impl {
#if defined(ENGINE_HAVE_RKNN)
    // One compiled step graph. The cache layout is B-independent, so the
    // first-group (small-B) model hands its cache straight to the big one.
    struct Step {
        engine::rknn::Engine engine;
        int64_t group = 0;
        int64_t cache_numel = 0;
        int64_t freqs_width = 0;
        int64_t cache_valid_elems = 1;
        uint32_t codes_size = 0;  // byte width of audio_codes at the declared type
        int64_t freqs_elems = 0;
        bool codes_pass_through = true;

        void load(const std::string & path, int64_t num_quantizers, int64_t head_dim_half) {
            // Engine::load reads the image, inits the context, picks the cores
            // (all three by name, AUTO fallback, AUDIOCPP_RKNN_CORE_MASK
            // override) and caches the tensor attributes. Measured here (same
            // binary, both masks, 300 Russian characters, greedy): AUTO keeps
            // the engine on Core0 (55/0/0 in /sys/kernel/debug/rknpu/load),
            // 0_1_2 is accepted and spreads it (41/15/16) for 1.8% on the
            // vocoder (21766 -> 21383 ms). The engine reads its weights per
            // step, so the step is DRAM-bound, not scheduler-bound: the mask is
            // right, the headroom is not there.
            engine.load(path, kLabel);
            if (engine.input_count() != 4 || engine.output_count() != 2) {
                throw std::runtime_error(
                    std::string(kLabel) + ": expected the 4-input/2-output stateful export, got " +
                    std::to_string(engine.input_count()) + "/" + std::to_string(engine.output_count()));
            }
            int64_t codes_groups = 0;
            for (uint32_t index = 0; index < engine.input_count(); ++index) {
                const rknn_tensor_attr & attr = engine.input_attr(index);
                const std::string name(attr.name);
                debug::timing_log_scalar(
                    "qwen3_tts.speech_decoder.rknn.input_attr" + std::to_string(index),
                    name + " type=" + std::to_string(attr.type) + " size=" + std::to_string(attr.size) +
                        " elems=" + std::to_string(attr.n_elems) + " fmt=" + std::to_string(attr.fmt) +
                        " dims=" + std::to_string(attr.n_dims));
                if (name == "audio_codes" && attr.n_dims == 3) {
                    codes_groups = attr.dims[1];
                    group = static_cast<int64_t>(attr.dims[2]);
                    codes_size = static_cast<uint32_t>(attr.n_elems * sizeof(int64_t));
                    codes_pass_through = attr.type == RKNN_TENSOR_INT64;
                } else if (name == "freqs_in" && attr.n_dims == 2) {
                    freqs_width = static_cast<int64_t>(attr.dims[1]);
                    freqs_elems = static_cast<int64_t>(attr.n_elems);
                } else if (name == "cache_valid") {
                    cache_valid_elems = static_cast<int64_t>(attr.n_elems);
                } else if (name == "cache_in") {
                    cache_numel = static_cast<int64_t>(attr.n_elems);
                }
            }
            if (group <= 0 || cache_numel <= 0 || freqs_width <= 0) {
                throw std::runtime_error(std::string(kLabel) + ": model " + path + " is not the stateful export");
            }
            if (codes_groups != num_quantizers) {
                throw std::runtime_error(
                    std::string(kLabel) + ": model has " + std::to_string(codes_groups) +
                    " codebooks, checkpoint has " + std::to_string(num_quantizers));
            }
            if (freqs_width != head_dim_half) {
                throw std::runtime_error(
                    std::string(kLabel) + ": model RoPE width " + std::to_string(freqs_width) +
                    " does not match head_dim/2 " + std::to_string(head_dim_half));
            }
        }

        // One NPU step: `codes` is codebook-major [16, group] (zero-padded),
        // `n_valid` counts the KV slots that hold real keys.
        void run(
            const int64_t * codes,
            const float * freqs,
            float cache_valid,
            const float * cache,
            std::vector<float> & audio,
            std::vector<float> & cache_out) {
            // With pass_through = 0 the runtime wants the width of the declared
            // type, not attr.size, which reports the engine's native fp16 width.
            std::vector<engine::rknn::Input> inputs(4);
            for (uint32_t index = 0; index < inputs.size(); ++index) {
                inputs[index].index = index;
                inputs[index].type = RKNN_TENSOR_FLOAT32;
            }
            inputs[0].buf = codes;
            inputs[0].size = codes_size;
            inputs[0].pass_through = codes_pass_through ? 1 : 0;
            inputs[0].type = RKNN_TENSOR_INT64;
            inputs[1].buf = freqs;
            inputs[1].size = static_cast<uint32_t>(freqs_elems * sizeof(float));
            inputs[2].buf = &cache_valid;
            inputs[2].size = static_cast<uint32_t>(cache_valid_elems * sizeof(float));
            inputs[3].buf = cache;
            inputs[3].size = static_cast<uint32_t>(cache_numel * sizeof(float));
            std::vector<std::vector<float>> outputs;
            engine.run(inputs, outputs);
            if (outputs.size() != 2) {
                throw std::runtime_error(std::string(kLabel) + ": expected the 4-input/2-output stateful export");
            }
            audio.swap(outputs[0]);
            cache_out.swap(outputs[1]);
        }
    };

    Step main;
    Step first;
    // Streaming window engine: a stateful export run at a zeroed cache so each
    // call is a full decode over its whole input. It may be the main model's
    // file itself, which is aliased rather than loaded a second time.
    Step window;
    bool window_from_main = false;
    int64_t num_quantizers = 0;
    int64_t kv_len = 0;
    std::vector<float> inv_freq;
    std::vector<float> cache;
    std::vector<int64_t> buf;  // unvoiced tokens, frame-major [tokens][16]
    int64_t pos = 0;
    // Leading frames of `buf` that are reference context: voiced for the cache,
    // never emitted as audio.
    int64_t priming = 0;
    double compute_ms = 0.0;
    int64_t steps = 0;

    void reset_state() {
        std::fill(cache.begin(), cache.end(), 0.0F);
        buf.clear();
        pos = 0;
        priming = 0;
    }
#endif  // ENGINE_HAVE_RKNN

#if defined(ENGINE_HAVE_RKNN)
    // Voice `tokens` (codebook-major, may be zero-padded) on `step`, appending
    // `tokens * 1920` samples to `out`.
    void voice(Step & step, int64_t tokens, const std::vector<int64_t> & codes, std::vector<float> & out);
#endif
};

#if defined(ENGINE_HAVE_RKNN)
void RknnSpeechDecoder::Impl::voice(
    Step & step,
    int64_t tokens,
    const std::vector<int64_t> & codes,
    std::vector<float> & out) {
    if (tokens > step.group) {
        throw std::runtime_error("Qwen3 speech decoder RKNN: group larger than the compiled model");
    }
    // RoPE angles, reduced mod 2*pi: feeding raw positions past ~1e3 loses
    // fp16 precision, and cos/sin as graph inputs explode on the NPU.
    std::vector<float> freqs(static_cast<size_t>(step.group * step.freqs_width), 0.0F);
    for (int64_t frame = 0; frame < tokens; ++frame) {
        const double position = static_cast<double>(pos + frame);
        for (int64_t i = 0; i < step.freqs_width; ++i) {
            freqs[static_cast<size_t>(frame * step.freqs_width + i)] = static_cast<float>(
                std::fmod(position * inv_freq[static_cast<size_t>(i)], 2.0 * M_PI));
        }
    }
    // cache_valid is the COUNT of filled KV slots. A boolean flag tells every
    // step after the first that all kv_len slots hold real keys, and the
    // never-written zero keys score 0 (not -inf) in the softmax: measured
    // stepwise SNR 48.8 dB at B=64 and 10.2 dB at B=16, vs 111.4 / 103.7 dB
    // with the count.
    const float cache_valid = static_cast<float>(std::min(pos, kv_len));
    std::vector<float> audio;
    std::vector<float> cache_out;
    const auto start = Clock::now();
    step.run(codes.data(), freqs.data(), cache_valid, cache.data(), audio, cache_out);
    compute_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    ++steps;
    if (static_cast<int64_t>(cache_out.size()) != step.cache_numel) {
        throw std::runtime_error("Qwen3 speech decoder RKNN: cache_out size changed between steps");
    }
    cache.swap(cache_out);
    cache.resize(main.cache_numel);
    const int64_t valid = tokens * kSamplesPerToken;
    if (valid > static_cast<int64_t>(audio.size())) {
        throw std::runtime_error("Qwen3 speech decoder RKNN: step returned a short waveform");
    }
    out.insert(out.end(), audio.begin(), audio.begin() + valid);
    pos += tokens;
}
#endif  // ENGINE_HAVE_RKNN

RknnSpeechDecoder::RknnSpeechDecoder(
    const std::string & model_path,
    const std::string & first_model_path,
    const std::string & window_path,
    int64_t num_quantizers,
    int64_t head_dim,
    int64_t sliding_window,
    float rope_theta)
    : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_HAVE_RKNN)
    if (model_path.empty()) {
        throw std::runtime_error("Qwen3 speech decoder RKNN: empty model path");
    }
    impl_->num_quantizers = num_quantizers;
    // The exported KV window is sliding_window - 1 keys (the current query is
    // not part of the history).
    impl_->kv_len = sliding_window - 1;
    impl_->inv_freq.resize(static_cast<size_t>(head_dim / 2));
    for (size_t i = 0; i < impl_->inv_freq.size(); ++i) {
        impl_->inv_freq[i] = static_cast<float>(
            std::pow(static_cast<double>(rope_theta), -static_cast<double>(i) / (head_dim / 2)));
    }
    impl_->main.load(model_path, num_quantizers, head_dim / 2);
    impl_->cache.assign(static_cast<size_t>(impl_->main.cache_numel), 0.0F);
    if (!first_model_path.empty()) {
        impl_->first.load(first_model_path, num_quantizers, head_dim / 2);
        if (impl_->first.cache_numel != impl_->main.cache_numel) {
            throw std::runtime_error("Qwen3 speech decoder RKNN: first-group cache size differs from the main model");
        }
    }
    if (!window_path.empty()) {
        // Files are compared, not strings: the residency accounting canonicalizes paths
        // (`engine_resources`), so a string comparison here lets the same file spelled two
        // ways load two contexts and be counted as one.
        std::error_code ec;
        if (std::filesystem::equivalent(window_path, model_path, ec)) {
            // The main engine voices the windows too (its B covers the default
            // 25 + 12): one resident context, not two.
            impl_->window_from_main = true;
        } else {
            impl_->window.load(window_path, num_quantizers, head_dim / 2);
            if (impl_->window.cache_numel != impl_->main.cache_numel) {
                throw std::runtime_error("Qwen3 speech decoder RKNN: window cache size differs from the main model");
            }
        }
    }
    debug::timing_log_scalar("qwen3_tts.speech_decoder.rknn.group", impl_->main.group);
    debug::timing_log_scalar(
        "qwen3_tts.speech_decoder.rknn.first_group",
        impl_->first.engine.loaded() ? impl_->first.group : 0);
    debug::timing_log_scalar(
        "qwen3_tts.speech_decoder.rknn.window_group",
        impl_->window_from_main
            ? impl_->main.group
            : (impl_->window.engine.loaded() ? impl_->window.group : 0));
    debug::timing_log_scalar("qwen3_tts.speech_decoder.rknn.cache_numel", impl_->main.cache_numel);
#else
    (void)model_path;
    (void)first_model_path;
    (void)window_path;
    (void)num_quantizers;
    (void)head_dim;
    (void)sliding_window;
    (void)rope_theta;
    throw std::runtime_error("Qwen3 speech decoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

RknnSpeechDecoder::~RknnSpeechDecoder() = default;

int64_t RknnSpeechDecoder::group_tokens() const {
#if defined(ENGINE_HAVE_RKNN)
    return impl_->main.group;
#else
    return 0;
#endif
}

int64_t RknnSpeechDecoder::first_group_tokens() const {
#if defined(ENGINE_HAVE_RKNN)
    return impl_->first.engine.loaded() ? impl_->first.group : impl_->main.group;
#else
    return 0;
#endif
}

void RknnSpeechDecoder::reset() {
#if defined(ENGINE_HAVE_RKNN)
    impl_->reset_state();
#endif
}

std::vector<float> RknnSpeechDecoder::decode(const int32_t * codes, int64_t frames, int64_t num_quantizers) {
#if defined(ENGINE_HAVE_RKNN)
    auto & s = *impl_;
    if (frames <= 0 || num_quantizers != s.num_quantizers) {
        throw std::runtime_error("Qwen3 speech decoder RKNN: invalid codec shape");
    }
    reset();
    std::vector<float> out;
    out.reserve(static_cast<size_t>(frames * kSamplesPerToken));
    std::vector<int64_t> group;
    for (int64_t start = 0; start < frames; start += s.main.group) {
        const int64_t n = std::min<int64_t>(s.main.group, frames - start);
        group.assign(static_cast<size_t>(s.num_quantizers * s.main.group), 0);
        for (int64_t frame = 0; frame < n; ++frame) {
            for (int64_t g = 0; g < s.num_quantizers; ++g) {
                group[static_cast<size_t>(g * s.main.group + frame)] =
                    codes[(start + frame) * num_quantizers + g];
            }
        }
        s.voice(s.main, n, group, out);
    }
    debug::timing_log_scalar("qwen3_tts.speech_decoder.rknn.compute_ms", s.compute_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder.rknn.steps", s.steps);
    return out;
#else
    (void)codes;
    (void)frames;
    (void)num_quantizers;
    throw std::runtime_error("Qwen3 speech decoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

std::vector<float> RknnSpeechDecoder::step(
    const int32_t * codes,
    int64_t context_frames,
    int64_t new_frames,
    int64_t num_quantizers,
    bool flush) {
#if defined(ENGINE_HAVE_RKNN)
    auto & s = *impl_;
    if (num_quantizers != s.num_quantizers || context_frames < 0 || new_frames < 0) {
        throw std::runtime_error("Qwen3 speech decoder RKNN: invalid streaming codec shape");
    }
    std::vector<float> out;
    // Windowed streaming: voice the whole [context + new] window on the
    // fixed-shape engine from a zeroed cache — a standalone full decode over
    // the window, the function the ggml streaming path computes per chunk,
    // context re-voiced every event — and drop the context's audio. Nothing is
    // buffered, so flush voices nothing.
    if (s.window_from_main || s.window.engine.loaded()) {
        if (codes == nullptr || flush) {
            return out;
        }
        const int64_t total = context_frames + new_frames;
        // The context-free first window rides the small first-group engine:
        // fewer frames per step is the audio the listener waits for.
        Impl::Step & voice_step =
            s.first.engine.loaded() && context_frames == 0 && total <= s.first.group
            ? s.first
            : (s.window_from_main ? s.main : s.window);
        if (total > voice_step.group) {
            throw std::runtime_error(
                "Qwen3 speech decoder RKNN: window of " + std::to_string(total) +
                " frames exceeds the " + std::to_string(voice_step.group) + "-frame window engine");
        }
        std::vector<int64_t> fed(static_cast<size_t>(s.num_quantizers) * voice_step.group, 0);
        for (int64_t frame = 0; frame < total; ++frame) {
            for (int64_t g = 0; g < s.num_quantizers; ++g) {
                fed[static_cast<size_t>(g * voice_step.group + frame)] = codes[frame * num_quantizers + g];
            }
        }
        // RoPE positions restart at 0 every window: each event is a full decode
        // of its own window, not a continuation of the last one.
        std::vector<float> freqs(static_cast<size_t>(voice_step.group * voice_step.freqs_width), 0.0F);
        for (int64_t frame = 0; frame < total; ++frame) {
            for (int64_t i = 0; i < voice_step.freqs_width; ++i) {
                freqs[static_cast<size_t>(frame * voice_step.freqs_width + i)] = static_cast<float>(
                    std::fmod(static_cast<double>(frame) * s.inv_freq[static_cast<size_t>(i)], 2.0 * M_PI));
            }
        }
        std::vector<float> audio;
        std::vector<float> cache_out;
        const auto start = Clock::now();
        voice_step.run(fed.data(), freqs.data(), 0.0F, s.cache.data(), audio, cache_out);
        s.compute_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        ++s.steps;
        const int64_t keep = total * kSamplesPerToken;
        const int64_t drop = context_frames * kSamplesPerToken;
        if (keep > static_cast<int64_t>(audio.size())) {
            throw std::runtime_error("Qwen3 speech decoder RKNN: window engine returned a short waveform");
        }
        out.assign(audio.begin() + drop, audio.begin() + keep);
        return out;
    }
    // The window carries left context, and a stateful model cannot voice a token
    // twice (the cache would hold it twice and pos would run ahead), so the
    // context is normally skipped as already covered by the cache. The first
    // window of an utterance is the exception: voice cloning seeds it with the
    // tail of the *reference* clip, which was never voiced, so voice it once and
    // drop its audio. Right context must stay disabled for this path: those
    // frames would be voiced here and re-fed as new tokens by the next window.
    int64_t priming = 0;
    if (codes != nullptr) {
        const bool seed = s.pos == 0 && s.buf.empty();
        priming = seed ? context_frames : 0;
        for (int64_t frame = seed ? 0 : context_frames; frame < context_frames + new_frames; ++frame) {
            for (int64_t g = 0; g < s.num_quantizers; ++g) {
                s.buf.push_back(codes[frame * num_quantizers + g]);
            }
        }
    }
    const auto voiced = static_cast<int64_t>(s.buf.size()) / s.num_quantizers;
    int64_t done = 0;
    for (;;) {
        // The first group of an utterance runs on the small-B model: the cache
        // layout is B-independent, so its cache feeds the big one directly, and
        // 16 tokens is 1.28 s of audio the listener does not have to wait for.
        const bool first = s.first.engine.loaded() && s.pos == 0;
        const int64_t group_tokens = first ? s.first.group : s.main.group;
        if (voiced - done < group_tokens) {
            break;
        }
        std::vector<int64_t> group(static_cast<size_t>(s.num_quantizers * group_tokens), 0);
        for (int64_t frame = 0; frame < group_tokens; ++frame) {
            for (int64_t g = 0; g < s.num_quantizers; ++g) {
                group[static_cast<size_t>(g * group_tokens + frame)] = s.buf[static_cast<size_t>((done + frame) * s.num_quantizers + g)];
            }
        }
        done += group_tokens;
        s.voice(first ? s.first : s.main, group_tokens, group, out);
    }
    s.buf.erase(s.buf.begin(), s.buf.begin() + done * s.num_quantizers);
    s.priming += priming;
    int64_t voiced_total = done;
    if (flush) {
        const int64_t tail = static_cast<int64_t>(s.buf.size()) / s.num_quantizers;
        if (tail > 0) {
            std::vector<int64_t> group(static_cast<size_t>(s.num_quantizers * s.main.group), 0);
            for (int64_t frame = 0; frame < tail; ++frame) {
                for (int64_t g = 0; g < s.num_quantizers; ++g) {
                    group[static_cast<size_t>(g * s.main.group + frame)] = s.buf[static_cast<size_t>(frame * s.num_quantizers + g)];
                }
            }
            s.voice(s.main, tail, group, out);
        }
        voiced_total += tail;
        s.buf.clear();
    }
    // Reference-context tokens voiced into the cache are conditioning only: their
    // own audio is dropped, and the count carries across calls when a window is
    // smaller than one group.
    const int64_t primed = std::min(s.priming, voiced_total);
    s.priming -= primed;
    const int64_t drop = primed * kSamplesPerToken;
    if (drop > static_cast<int64_t>(out.size())) {
        throw std::runtime_error("Qwen3 speech decoder RKNN: reference context was not voiced");
    }
    out.erase(out.begin(), out.begin() + drop);
    if (flush) {
        s.reset_state();
    }
    return out;
#else
    (void)codes;
    (void)context_frames;
    (void)new_frames;
    (void)num_quantizers;
    (void)flush;
    throw std::runtime_error("Qwen3 speech decoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

}  // namespace engine::models::qwen3_tts
