#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/partial_text.h"
#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::runtime {

struct AsrStreamMetadata {
    std::vector<VoiceActivityEvent> voice_activity;
    bool is_final = false;
};

// Incremental-audio lifecycle and transcript publication. Models own buffering,
// decoding and commitment boundaries, not callback/return bookkeeping.
class StreamingAsrSessionBase : public IStreamingVoiceTaskSession {
public:
    void start_stream(const TaskRequest & request) final {
        require_streaming_mode();
        reset();
        stream_started_at_ = std::chrono::steady_clock::now();
        begin_stream(request);
        stream_active_ = true;
    }

    void set_stream_event_sink(StreamEventCallback sink) final {
        sink_ = std::move(sink);
    }

    void reset() final {
        stream_active_ = false;
        stream_result_ = {};
        stream_result_.text_output = Transcript{};
        pending_event_ = {};
        partials_.reset();
        reset_stream_state();
    }

    StreamEvent process_audio_chunk(const AudioChunk & chunk) final {
        require_active_stream();
        pending_event_ = {};
        consume_audio(chunk);
        return std::exchange(pending_event_, {});
    }

    TaskResult finalize() final {
        require_active_stream();
        complete_stream(stream_result_);
        auto result = std::move(stream_result_);
        stream_active_ = false;
        stream_result_ = {};
        partials_.reset();
        pending_event_ = {};
        debug::timing_log_scalar("session.wall_ms",
            debug::elapsed_ms(stream_started_at_, std::chrono::steady_clock::now()));
        return result;
    }

protected:
    virtual void begin_stream(const TaskRequest & request) = 0;
    virtual void consume_audio(const AudioChunk & chunk) = 0;
    // Flush model state and attach final-only metadata. The accumulated transcript
    // is already present; implementations need not reconstruct or return it.
    virtual void complete_stream(TaskResult & result) = 0;
    virtual void reset_stream_state() = 0;

    const Transcript & committed_transcript() const { return *stream_result_.text_output; }

    // A decoder supplies the committed cumulative transcript. Timestamps in
    // this update are cumulative too; they replace the previous published set.
    void publish_transcript(Transcript transcript,
                            std::vector<WordTimestamp> words = {},
                            AsrStreamMetadata metadata = {}) {
        stream_result_.text_output = std::move(transcript);
        const auto & text = *stream_result_.text_output;
        StreamEvent event;
        event.word_timestamps = std::move(words);
        event.voice_activity = std::move(metadata.voice_activity);
        event.is_final = metadata.is_final;
        auto delta = partials_.publish(text.text);
        if (!delta.empty()) {
            event.partial_text = Transcript{std::move(delta), text.language};
        }
        stream_result_.word_timestamps = event.word_timestamps;
        if (event.word_timestamps.empty()) pending_event_.word_timestamps.clear();
        publish_event(std::move(event));
    }

    // A finalized segment supplies only new text and timestamps. Joining words
    // or punctuation remains the model's decision, not a language heuristic here.
    void append_transcript(Transcript segment,
                           std::vector<WordTimestamp> words = {},
                           bool include_snapshot = false) {
        auto & text = *stream_result_.text_output;
        text.text += segment.text;
        if (!segment.language.empty()) text.language = std::move(segment.language);
        stream_result_.word_timestamps.insert(stream_result_.word_timestamps.end(),
                                             words.begin(), words.end());
        StreamEvent event;
        auto delta = partials_.publish(text.text);
        if (!delta.empty()) event.partial_text = Transcript{std::move(delta), text.language};
        if (include_snapshot) event.partial_text_snapshot = text;
        event.word_timestamps = std::move(words);
        publish_event(std::move(event), true);
    }

    void publish_hypothesis(Transcript transcript) {
        StreamEvent event;
        event.partial_text_snapshot = std::move(transcript);
        publish_event(std::move(event));
    }

    void publish_metadata(AsrStreamMetadata metadata) {
        StreamEvent event;
        event.voice_activity = std::move(metadata.voice_activity);
        event.is_final = metadata.is_final;
        publish_event(std::move(event));
    }

    double stream_wall_ms() const {
        return debug::elapsed_ms(stream_started_at_, std::chrono::steady_clock::now());
    }

private:
    // Timestamp lists have two existing contracts: cumulative decoder updates
    // replace the pending list; committed segments append disjoint word spans.
    // Keeping this distinction here avoids model-local callback/return branches.
    void publish_event(StreamEvent event, bool append_words = false) {
        if (sink_) {
            if (event.partial_text || event.partial_text_snapshot ||
                !event.word_timestamps.empty() || !event.voice_activity.empty() ||
                event.is_final) sink_(event);
            return;
        }
        if (event.partial_text) {
            if (!pending_event_.partial_text) pending_event_.partial_text = Transcript{};
            pending_event_.partial_text->text += event.partial_text->text;
            pending_event_.partial_text->language = std::move(event.partial_text->language);
        }
        if (event.partial_text_snapshot) {
            pending_event_.partial_text_snapshot = std::move(event.partial_text_snapshot);
        }
        if (!event.word_timestamps.empty()) {
            if (append_words) {
                pending_event_.word_timestamps.insert(pending_event_.word_timestamps.end(),
                    event.word_timestamps.begin(), event.word_timestamps.end());
            } else {
                pending_event_.word_timestamps = std::move(event.word_timestamps);
            }
        }
        pending_event_.voice_activity.insert(pending_event_.voice_activity.end(),
            event.voice_activity.begin(), event.voice_activity.end());
        pending_event_.is_final = pending_event_.is_final || event.is_final;
    }

    TaskResult stream_result_;

    void require_streaming_mode() const {
        if (run_mode() != RunMode::Streaming) {
            throw std::runtime_error(family() + " requires a streaming session");
        }
    }

    void require_active_stream() const {
        require_streaming_mode();
        if (!stream_active_) {
            throw std::runtime_error(family() + " requires start_stream before processing or finalizing audio");
        }
    }

    bool stream_active_ = false;
    StreamEventCallback sink_;
    StreamEvent pending_event_;
    PartialTextPublisher partials_;
    std::chrono::steady_clock::time_point stream_started_at_{};
};

} // namespace engine::runtime
