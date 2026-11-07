#include "engine/community_models/mms_forced_aligner/ctc_alignment.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::community_models::mms_forced_aligner {

namespace {

constexpr int64_t kFrameStrideSamples = 320;
constexpr double kSampleRate16k = 16000.0;

int64_t checked_multiply(int64_t lhs, int64_t rhs, const char * what) {
    if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
        throw std::runtime_error(std::string("MMS forced aligner overflow while sizing ") + what);
    }
    return lhs * rhs;
}

int32_t state_label(const std::vector<int32_t> & targets, int64_t state, int32_t blank_id) {
    // state 2k+1 -> targets[k]; even states are blank.
    return state % 2 == 0 ? blank_id : targets[static_cast<size_t>(state / 2)];
}

// Two-bit backpointer per cell: 0 = stay, 1 = advance-one, 2 = advance-two.
class PackedBackpointers {
public:
    PackedBackpointers(int64_t cells) : bytes_(static_cast<size_t>((cells + 3) / 4)) {}

    void set(int64_t index, uint8_t move) {
        const size_t byte = static_cast<size_t>(index) / 4;
        const size_t shift = static_cast<size_t>(index % 4) * 2;
        bytes_[byte] = static_cast<uint8_t>((bytes_[byte] & ~(uint8_t{3} << shift)) | (move << shift));
    }

    uint8_t get(int64_t index) const {
        const size_t byte = static_cast<size_t>(index) / 4;
        const size_t shift = static_cast<size_t>(index % 4) * 2;
        return static_cast<uint8_t>((bytes_[byte] >> shift) & uint8_t{3});
    }

private:
    std::vector<uint8_t> bytes_;
};

}  // namespace

CtcForcedAlignment ctc_forced_align(
    const MmsLogProbabilities & log_probs,
    const std::vector<int32_t> & targets,
    int32_t blank_id,
    const CtcAlignmentLimits & limits) {
    const int64_t frames = log_probs.frames;
    const int64_t classes = log_probs.classes;
    if (log_probs.data == nullptr || frames <= 0 || classes <= 0) {
        throw std::runtime_error("MMS forced aligner requires nonempty log probabilities");
    }
    if (blank_id < 0 || blank_id >= classes) {
        throw std::runtime_error("MMS forced aligner blank id is outside the class range");
    }
    const int64_t token_count = static_cast<int64_t>(targets.size());
    if (token_count == 0) {
        throw std::runtime_error("MMS forced aligner requires a nonempty target sequence");
    }
    if (token_count > limits.max_target_tokens) {
        throw std::runtime_error(
            "MMS forced aligner target sequence has " + std::to_string(token_count) +
            " tokens, exceeding max_target_tokens=" + std::to_string(limits.max_target_tokens));
    }
    int64_t repeated_adjacent = 0;
    for (int64_t index = 1; index < token_count; ++index) {
        const int32_t current = targets[static_cast<size_t>(index)];
        const int32_t previous = targets[static_cast<size_t>(index - 1)];
        if (current < 0 || current >= classes || previous < 0 || previous >= classes) {
            throw std::runtime_error("MMS forced aligner target id is outside the class range");
        }
        if (current == blank_id) {
            throw std::runtime_error("MMS forced aligner targets must not contain the blank id");
        }
        if (current == previous) {
            ++repeated_adjacent;
        }
    }
    if (targets.front() < 0 || targets.front() >= classes) {
        throw std::runtime_error("MMS forced aligner target id is outside the class range");
    }
    if (targets.front() == blank_id) {
        throw std::runtime_error("MMS forced aligner targets must not contain the blank id");
    }
    if (frames < token_count + repeated_adjacent) {
        throw std::runtime_error(
            "MMS forced aligner frame count " + std::to_string(frames) +
            " is too short for " + std::to_string(token_count) +
            " targets with " + std::to_string(repeated_adjacent) +
            " adjacent repeats (need at least " + std::to_string(token_count + repeated_adjacent) + ")");
    }

    const int64_t states = 2 * token_count + 1;
    const int64_t cells = checked_multiply(frames, states, "the alignment matrix");
    if (cells > limits.max_alignment_cells) {
        throw std::runtime_error(
            "MMS forced aligner alignment requires " + std::to_string(cells) +
            " cells (frames=" + std::to_string(frames) +
            ", targets=" + std::to_string(token_count) + "), exceeding max_alignment_cells=" +
            std::to_string(limits.max_alignment_cells));
    }

    const float negative_infinity = -std::numeric_limits<float>::infinity();
    std::vector<float> previous(static_cast<size_t>(states), negative_infinity);
    std::vector<float> current(static_cast<size_t>(states), negative_infinity);

    const auto frame_log = [&](int64_t frame, int64_t cls) {
        return log_probs.data[frame * classes + cls];
    };

    // Frame 0 can only be the leading blank or the first target.
    previous[0] = frame_log(0, blank_id);
    previous[1] = frame_log(0, targets.front());

    PackedBackpointers backpointers(cells);
    for (int64_t frame = 1; frame < frames; ++frame) {
        std::fill(current.begin(), current.end(), negative_infinity);
        for (int64_t state = 0; state < states; ++state) {
            const float emission = frame_log(frame, state_label(targets, state, blank_id));
            float best = previous[static_cast<size_t>(state)];  // stay
            uint8_t move = 0;
            if (state > 0 && previous[static_cast<size_t>(state - 1)] > best) {
                best = previous[static_cast<size_t>(state - 1)];  // advance-one
                move = 1;
            }
            const bool target_state = state % 2 == 1;
            if (target_state && state >= 3) {
                const int64_t target_index = state / 2;
                if (targets[static_cast<size_t>(target_index)] !=
                    targets[static_cast<size_t>(target_index - 1)]) {
                    const float via_two = previous[static_cast<size_t>(state - 2)];
                    if (via_two > best) {  // advance-two
                        best = via_two;
                        move = 2;
                    }
                }
            }
            current[static_cast<size_t>(state)] = best + emission;
            backpointers.set(frame * states + state, move);
        }
        std::swap(previous, current);
    }

    // Termination: final blank state or final target state; target wins ties.
    const int64_t final_target_state = states - 2;
    const int64_t final_blank_state = states - 1;
    const float target_score = previous[static_cast<size_t>(final_target_state)];
    const float blank_score = previous[static_cast<size_t>(final_blank_state)];
    int64_t final_state = blank_score > target_score ? final_blank_state : final_target_state;
    const float score = std::max(target_score, blank_score);
    if (!std::isfinite(score)) {
        throw std::runtime_error("MMS forced aligner produced a non-finite path score");
    }

    CtcForcedAlignment alignment;
    alignment.state_path.resize(static_cast<size_t>(frames));
    alignment.label_path.resize(static_cast<size_t>(frames));
    alignment.frame_log_probs.resize(static_cast<size_t>(frames));
    alignment.score = score;

    int64_t state = final_state;
    for (int64_t frame = frames - 1; frame >= 0; --frame) {
        alignment.state_path[static_cast<size_t>(frame)] = static_cast<int32_t>(state);
        const int32_t label = state % 2 == 0 ? blank_id : targets[static_cast<size_t>(state / 2)];
        alignment.label_path[static_cast<size_t>(frame)] = label;
        alignment.frame_log_probs[static_cast<size_t>(frame)] = frame_log(frame, label);
        if (frame > 0) {
            const uint8_t move = backpointers.get(frame * states + state);
            if (move == 0) {
                // stay
            } else if (move == 1) {
                state -= 1;
            } else {
                state -= 2;
            }
        }
    }
    return alignment;
}

std::vector<MmsWordSpan> mms_word_spans_from_alignment(
    const CtcForcedAlignment & alignment,
    const MmsPreparedText & prepared,
    int32_t blank_id,
    float merge_threshold_sec) {
    if (alignment.state_path.empty() || alignment.state_path.size() != alignment.label_path.size() ||
        alignment.label_path.size() != alignment.frame_log_probs.size()) {
        throw std::runtime_error("MMS forced aligner alignment path is incomplete");
    }
    if (prepared.target_ids.empty() || prepared.target_ids.size() != prepared.target_to_word.size()) {
        throw std::runtime_error("MMS forced aligner prepared text is incomplete");
    }

    // Run-length segments over the label path.
    struct Segment {
        int32_t label;
        int64_t start;
        int64_t end;
    };
    std::vector<Segment> segments;
    for (int64_t frame = 0; frame < static_cast<int64_t>(alignment.label_path.size()); ++frame) {
        const int32_t label = alignment.label_path[static_cast<size_t>(frame)];
        if (segments.empty() || segments.back().label != label) {
            segments.push_back({label, frame, frame});
        } else {
            segments.back().end = frame;
        }
    }

    // Walk the target sequence in order; every target must appear as a label.
    const int64_t token_count = static_cast<int64_t>(prepared.target_ids.size());
    std::vector<int64_t> target_segment(token_count, -1);
    int64_t segment_index = 0;
    for (int64_t target_index = 0; target_index < token_count; ++target_index) {
        const int32_t target = prepared.target_ids[static_cast<size_t>(target_index)];
        while (segment_index < static_cast<int64_t>(segments.size()) &&
               segments[static_cast<size_t>(segment_index)].label != target) {
            ++segment_index;
        }
        if (segment_index >= static_cast<int64_t>(segments.size())) {
            throw std::runtime_error(
                "MMS forced aligner alignment path is missing target '" +
                std::to_string(target) + "'");
        }
        target_segment[static_cast<size_t>(target_index)] = segment_index;
        // Letters of one word occupy consecutive segments; consuming only the
        // segment run keeps the walk aligned with the target order.
        ++segment_index;
    }

    // Per-word spans over contiguous letter segments plus boundary blanks.
    // The pinned reference decides full-vs-midpoint blank sharing from the
    // target-interval index *including* <star> intervals: the first interval is
    // always a leading star, so a word is the full edge only when it is the
    // first or last interval (no star on that side).
    std::vector<MmsWordSpan> spans;
    const int64_t word_count = static_cast<int64_t>(prepared.original_words.size());
    const int64_t segment_count = static_cast<int64_t>(segments.size());
    int64_t interval_count = word_count;
    for (int64_t t = 0; t < token_count; ++t) {
        if (prepared.target_to_word[static_cast<size_t>(t)] == -1) {
            ++interval_count;
        }
    }
    int64_t interval_index = 0;
    int64_t target_index = 0;
    while (target_index < token_count) {
        if (prepared.target_to_word[static_cast<size_t>(target_index)] == -1) {
            // A <star> target completes its own interval; words are never the
            // first interval in the validated star placements.
            ++interval_index;
            ++target_index;
            continue;
        }
        const int32_t word_index = prepared.target_to_word[static_cast<size_t>(target_index)];
        const int64_t first_segment = target_segment[static_cast<size_t>(target_index)];
        int64_t last_segment = first_segment;
        while (target_index + 1 < token_count &&
               prepared.target_to_word[static_cast<size_t>(target_index + 1)] == word_index) {
            ++target_index;
            last_segment = target_segment[static_cast<size_t>(target_index)];
        }
        ++target_index;

        MmsWordSpan span;
        span.word_index = word_index;
        int64_t start_frame = segments[static_cast<size_t>(first_segment)].start;
        int64_t end_frame = segments[static_cast<size_t>(last_segment)].end;
        const bool first_interval = interval_index == 0;
        const bool last_interval = interval_index == interval_count - 1;
        if (first_segment > 0 && segments[static_cast<size_t>(first_segment - 1)].label == blank_id) {
            const auto & blank = segments[static_cast<size_t>(first_segment - 1)];
            start_frame = first_interval ? blank.start : (blank.start + blank.end) / 2;
        }
        if (last_segment + 1 < segment_count && segments[static_cast<size_t>(last_segment + 1)].label == blank_id) {
            const auto & blank = segments[static_cast<size_t>(last_segment + 1)];
            end_frame = last_interval ? blank.end : (blank.start + blank.end) / 2;
        }
        ++interval_index;
        span.start_frame = start_frame;
        span.end_frame = end_frame;

        double sum = 0.0;
        for (int64_t frame = start_frame; frame <= end_frame; ++frame) {
            sum += alignment.frame_log_probs[static_cast<size_t>(frame)];
        }
        span.log_prob = static_cast<float>(sum / static_cast<double>(end_frame - start_frame + 1));
        spans.push_back(span);
    }
    if (spans.size() != static_cast<size_t>(word_count)) {
        throw std::runtime_error("MMS forced aligner word span reconstruction lost words");
    }

    // Optional gap merging (frame units; 320 samples at 16 kHz per frame).
    // Frame spans are inclusive: a word in [a, e] is followed by [s, t] with a
    // real gap of (s - e - 1) frames. Merge clips the later word to start at
    // previous.end + 1 so the exclusive-end sample ranges stay contiguous
    // (previous.end + 1 gives a 320-frame overlap, not the intended join).
    if (merge_threshold_sec > 0.0F) {
        const int64_t merge_frames = static_cast<int64_t>(
            std::llround(static_cast<double>(merge_threshold_sec) * kSampleRate16k /
                         static_cast<double>(kFrameStrideSamples)));
        for (size_t index = 1; index < spans.size(); ++index) {
            const int64_t previous_end = spans[index - 1].end_frame;
            const int64_t gap = spans[index].start_frame - previous_end - 1;
            if (gap > 0 && gap <= merge_frames) {
                spans[index].start_frame = previous_end + 1;
            }
        }
    }
    return spans;
}

}  // namespace engine::community_models::mms_forced_aligner
