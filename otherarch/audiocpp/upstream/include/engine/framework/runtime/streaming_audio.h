#pragma once

#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::runtime {

enum class StreamingAudioChunkPolicy { Fixed, Grow };

inline StreamingAudioChunkPolicy parse_streaming_audio_chunk_policy(std::string_view value) {
    if (value == "fixed") return StreamingAudioChunkPolicy::Fixed;
    if (value == "grow") return StreamingAudioChunkPolicy::Grow;
    throw std::invalid_argument("stream_chunk_policy must be fixed or grow");
}

struct StreamingAudioConfig {
    StreamingAudioChunkPolicy policy = StreamingAudioChunkPolicy::Grow;
    size_t frames_per_chunk = 12;
    size_t frame_width = 1;
    size_t left_context_frames = 0;
    size_t right_context_frames = 0;
};

// A borrowed, frame-major window. The decoder returns ONLY the new audio,
// excluding context and graph padding. References do not advance first_frame.
template<class Scalar>
struct StreamingAudioWindow {
    const Scalar * data;
    size_t frame_width;
    size_t left_frames;
    size_t new_frames;
    size_t right_frames;
    size_t first_frame;
    bool final;

    size_t frames() const { return left_frames + new_frames + right_frames; }
};

// Session-owned orchestration, independent of decoder architecture/backend.
// Graphs and weights stay in the supplied decoder. No asynchronous work is
// started here; callbacks run on the session's generation thread.
template<class Scalar>
class StreamingAudioController {
public:
    using Window = StreamingAudioWindow<Scalar>;
    using Decode = std::function<AudioBuffer(const Window &)>;

    void begin(StreamingAudioConfig config, Decode decode, StreamEventCallback sink = {},
               const std::vector<Scalar> & reference = {},
               std::function<void()> reset_decoder = {},
               std::function<AudioBuffer()> drain_decoder = {}) {
        if (active_) throw std::logic_error("streaming audio request is already active");
        if (!decode || config.frames_per_chunk == 0 || config.frame_width == 0 ||
            reference.size() % config.frame_width != 0) {
            throw std::invalid_argument("invalid streaming audio configuration or reference frames");
        }
        reset();
        config_ = config;
        if (reset_decoder) reset_decoder();
        decode_ = std::move(decode);
        sink_ = std::move(sink);
        drain_ = std::move(drain_decoder);
        target_ = config.policy == StreamingAudioChunkPolicy::Grow ? 1 : config.frames_per_chunk;
        context_ = std::min(config.left_context_frames, reference.size() / config.frame_width);
        pending_.assign(reference.end() - context_ * config.frame_width, reference.end());
        active_ = true;
    }

    void push(const std::vector<Scalar> & frames) {
        require_active();
        if (frames.size() % config_.frame_width != 0) {
            throw std::invalid_argument("streaming audio input contains an incomplete frame");
        }
        try {
            // Ingest incrementally even when a diffusion model supplies its
            // complete latent sequence, bounding the controller's input buffer.
            for (size_t i = 0; i < frames.size(); i += config_.frame_width) {
                pending_.insert(pending_.end(), frames.begin() + i,
                                frames.begin() + i + config_.frame_width);
                if (available() >= target_ + config_.right_context_frames) flush(false);
            }
        } catch (...) {
            reset();
            throw;
        }
    }

    AudioBuffer finish() {
        require_active();
        try {
            while (available() != 0) flush(true);
            if (drain_) {
                const auto start = Clock::now();
                auto tail = drain_();
                decode_ms_ += elapsed(start);
                publish(std::move(tail));
            }
            auto result = std::move(audio_);
            active_ = false;
            pending_.clear();
            decode_ = {};
            sink_ = {};
            drain_ = {};
            return result;
        } catch (...) {
            reset();
            throw;
        }
    }

    void reset() {
        active_ = false;
        pending_.clear();
        audio_ = {};
        context_ = first_frame_ = target_ = 0;
        decode_ms_ = publish_ms_ = 0;
        decode_ = {};
        sink_ = {};
        drain_ = {};
    }

    double decode_ms() const { return decode_ms_; }
    double publish_ms() const { return publish_ms_; }

private:
    using Clock = std::chrono::steady_clock;
    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    void require_active() const {
        if (!active_) throw std::logic_error("streaming audio request is not active");
    }
    size_t available() const { return pending_.size() / config_.frame_width - context_; }

    void flush(bool finishing) {
        const size_t count = std::min(target_, available());
        const size_t right = std::min(config_.right_context_frames, available() - count);
        const auto start = Clock::now();
        auto audio = decode_({pending_.data(), config_.frame_width, context_, count, right,
                              first_frame_, finishing && count == available()});
        decode_ms_ += elapsed(start);
        publish(std::move(audio));
        first_frame_ += count;
        const size_t keep = std::min(config_.left_context_frames, context_ + count);
        pending_.erase(pending_.begin(),
                       pending_.begin() + (context_ + count - keep) * config_.frame_width);
        context_ = keep;
        if (config_.policy == StreamingAudioChunkPolicy::Grow) {
            target_ += std::min(target_, config_.frames_per_chunk - target_);
        }
    }

    void publish(AudioBuffer audio) {
        if (audio.samples.empty()) return;
        append_audio_buffer(audio_, audio);
        if (sink_) {
            const auto start = Clock::now();
            StreamEvent event;
            event.audio_output = std::move(audio);
            sink_(event);
            publish_ms_ += elapsed(start);
        }
    }

    StreamingAudioConfig config_;
    Decode decode_;
    StreamEventCallback sink_;
    std::function<AudioBuffer()> drain_;
    std::vector<Scalar> pending_;
    AudioBuffer audio_;
    size_t context_ = 0, first_frame_ = 0, target_ = 0;
    double decode_ms_ = 0, publish_ms_ = 0;
    bool active_ = false;
};

} // namespace engine::runtime
