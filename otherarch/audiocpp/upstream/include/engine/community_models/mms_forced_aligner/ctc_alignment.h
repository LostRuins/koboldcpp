#pragma once

#include "engine/community_models/mms_forced_aligner/text_processor.h"

#include <cstdint>
#include <vector>

namespace engine::community_models::mms_forced_aligner {

struct CtcAlignmentLimits {
    // Hard cap on DP cells (frames x states) allocated per request.
    int64_t max_alignment_cells = 50000000;
    // Hard cap on flattened CTC target tokens per request.
    int64_t max_target_tokens = 8192;
};

struct MmsLogProbabilities {
    const float * data = nullptr;
    int64_t frames = 0;
    int64_t classes = 0;
};

struct CtcForcedAlignment {
    // Per-frame index into the logical state sequence
    // [blank, t0, blank, t1, ..., blank, t_{L-1}, blank].
    std::vector<int32_t> state_path;
    // Per-frame emitted class id (blank_id or a target).
    std::vector<int32_t> label_path;
    // Per-frame log probability of the chosen state's class.
    std::vector<float> frame_log_probs;
    // Total path log probability.
    float score = 0.0F;
};

// Viterbi forced alignment over the standard CTC state sequence with stay,
// advance-one, and advance-two (non-blank, non-repeated) transitions, two
// rolling score rows, and packed two-bit backpointers. Deterministic tie
// order: stay beats advance-one beats advance-two (strict `>` updates);
// on an exact final tie the last target state wins over the final blank.
// Fails before allocation when targets exceed max_target_tokens or when
// frames x states exceed max_alignment_cells.
CtcForcedAlignment ctc_forced_align(
    const MmsLogProbabilities & log_probs,
    const std::vector<int32_t> & targets,
    int32_t blank_id,
    const CtcAlignmentLimits & limits);

struct MmsWordSpan {
    int32_t word_index = 0;
    // Inclusive frame range over the emission sequence.
    int64_t start_frame = 0;
    int64_t end_frame = 0;
    // Mean selected-frame log probability over the word's span.
    float log_prob = 0.0F;
};

// Reconstructs per-word spans from the alignment path: merges repeated
// states, associates letter spans with their original words (stars excluded),
// and shares boundary blank intervals at their integer midpoints (full blank
// at the transcript edges). frame -> 16 kHz sample conversion is stride 320;
// `merge_threshold_sec` moves a later start to the previous end when the gap
// is at or below the threshold.
std::vector<MmsWordSpan> mms_word_spans_from_alignment(
    const CtcForcedAlignment & alignment,
    const MmsPreparedText & prepared,
    int32_t blank_id,
    float merge_threshold_sec);

}  // namespace engine::community_models::mms_forced_aligner
