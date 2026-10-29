#pragma once

#include "engine/framework/runtime/session.h"

#include <optional>
#include <stdexcept>
#include <utility>

namespace engine::runtime {

// Opt-in lifecycle for synchronous, text-driven audio streaming. Generation
// publishes events through the sink before start_stream returns its final result.
class StreamingTtsSessionBase : public IStreamingVoiceTaskSession {
public:
    StreamingPolicy streaming_policy() const final {
        return {StreamingInputKind::None, StreamingOutputKind::FinalResult};
    }

    void start_stream(const TaskRequest & request) final {
        if (run_mode() != RunMode::Streaming) {
            throw std::runtime_error(family() + " start_stream requires a streaming session");
        }
        reset();
        try {
            result_ = generate_stream(request);
        } catch (...) {
            reset();
            throw;
        }
    }

    void set_stream_event_sink(StreamEventCallback sink) final {
        sink_ = std::move(sink);
    }

    void reset() final {
        result_.reset();
        reset_stream_state();
    }

    StreamEvent process_audio_chunk(const AudioChunk &) final {
        throw std::runtime_error(family() + " streaming accepts text, not audio chunks");
    }

    TaskResult finalize() final {
        if (!result_) {
            throw std::runtime_error(family() + " stream has not completed");
        }
        auto result = std::move(*result_);
        reset();
        return result;
    }

protected:
    virtual TaskResult generate_stream(const TaskRequest & request) = 0;
    virtual void reset_stream_state() = 0;

    const StreamEventCallback & stream_event_sink() const { return sink_; }

private:
    StreamEventCallback sink_;
    std::optional<TaskResult> result_;
};

} // namespace engine::runtime
