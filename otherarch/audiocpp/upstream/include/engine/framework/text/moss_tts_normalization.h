#pragma once

// Robust input normalisation for the MOSS-TTS family, ported from the
// tts_robust_normalizer_single_script.py that ships with MOSS-TTS-v1.5.
//
// Clean prose passes through unchanged. What this is for is text that arrives
// as markdown, carries URLs or file paths, mixes CJK and Latin without spaces,
// or repeats punctuation for emphasis -- all of which the model reads literally
// otherwise.

#include <string>

namespace engine::text {

std::string normalize_moss_tts_text(const std::string & text);

}  // namespace engine::text
