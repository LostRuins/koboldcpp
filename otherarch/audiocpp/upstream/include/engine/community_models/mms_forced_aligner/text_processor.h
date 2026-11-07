#pragma once

#include "engine/community_models/mms_forced_aligner/assets.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::mms_forced_aligner {

enum class MmsTextNormalization {
    Latin,
    PreRomanized,
};

enum class MmsStarFrequency {
    Segment,
    Edges,
};

struct MmsTextProcessorOptions {
    MmsTextNormalization normalization = MmsTextNormalization::Latin;
    MmsStarFrequency star_frequency = MmsStarFrequency::Segment;
};

struct MmsPreparedText {
    std::string canonical_language;
    // Original whitespace-delimited word strings, in order, exactly as the
    // caller supplied them (surviving words only).
    std::vector<std::string> original_words;
    // Normalized ASCII letter forms per surviving original word.
    std::vector<std::string> normalized_words;
    // Flattened CTC target ids (real classes and virtual <star>, id 31).
    std::vector<int32_t> target_ids;
    // Per target id: original word index, or -1 for the virtual <star>.
    std::vector<int32_t> target_to_word;
};

// Canonicalizes nl/nld -> nld and en/eng -> eng; rejects any other code.
std::string mms_canonical_language(const std::string & language);

// Maps a transcript to CTC targets. `segment` star placement follows the
// pinned reference: one <star> before every word, none trailing; `edges`
// places a single <star> at the start and end. Standalone digit-only words
// are dropped; digits inside words are removed; punctuation acts as a
// separator; unsupported non-Latin letters are rejected with an error.
// Throws when no alignable targets remain.
MmsPreparedText prepare_mms_text(
    const MmsVocabulary & vocab,
    const std::string & text,
    const std::string & language,
    const MmsTextProcessorOptions & options);

}  // namespace engine::community_models::mms_forced_aligner
