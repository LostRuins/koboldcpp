#pragma once

#include <cstddef>
#include <string>

namespace engine::runtime {

// Turns a running transcript into the increments a partial is contracted to be.
//
// A streaming session's partial_text is the text decoded since the last one.
// The CLI appends them into a scrolling transcript (see PartialTextRenderer)
// and the reference server forwards each as an OpenAI-shaped
// `transcript.text.delta`, which is incremental by specification. A family that
// holds a whole running transcript therefore has to publish the difference, and
// every family that did so carried its own copy of the arithmetic.
//
// The copies disagreed and two of them were wrong, which is why this is shared
// rather than duplicated once more:
//
//   - A decode can end part way through a UTF-8 sequence, because a tokenizer
//     falls back to bytes for text its vocabulary does not cover. Publishing
//     that puts half a code point on the wire, and it reaches a JSON encoder
//     as invalid UTF-8. The tail is held back for the update that completes it.
//
//   - A decode can revise text already published rather than only extending it.
//     Nothing can retract a delta that has gone out, so the client's transcript
//     is wrong either way; starting the next delta at the divergence at least
//     keeps it from being spliced into the middle of a character.
//
// Not thread safe: a session owns one of these and publishes from the thread
// driving the stream.
class PartialTextPublisher {
public:
    // The text newly decoded since the last call. Empty when nothing is new, or
    // when everything new is an incomplete character still being held.
    std::string publish(const std::string & transcript);

    // Forgets what has gone out, for a session starting a fresh stream.
    void reset() { published_.clear(); }

    // What has actually been published, which trails `transcript` whenever a
    // character is being held back.
    const std::string & published() const { return published_; }

private:
    std::string published_;
};

// Longest common prefix of two transcripts, backed off so the result never
// lands inside a UTF-8 sequence. Exposed for tests and for callers that need
// the offset rather than the text.
std::size_t transcript_common_prefix(const std::string & lhs, const std::string & rhs);

// How much of `text` is safe to publish: everything up to the last complete
// UTF-8 sequence. Equal to text.size() unless the tail is a partial character.
std::size_t transcript_publishable_end(const std::string & text);

}  // namespace engine::runtime
