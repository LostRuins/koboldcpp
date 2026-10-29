#pragma once

#include <string>

namespace engine::community_models::soprano_tts {

// Full port of the reference Soprano text normalizer (ekwek1/soprano
// soprano/utils/text_normalizer.py, adapted from tortoise-tts). Runs the same
// stage chain in the same order so the LM only ever sees the exact text forms
// it was trained on:
//
//   1. expand_preunicode_special_characters (em-dash -> " - ")
//   2. convert_to_ascii                     (unidecode table fold)
//   3. normalize_newlines                   (ensure ".!?" line endings)
//   4. normalize_numbers                    (dates, phones, times, money,
//                                            decimals, fractions, ordinals,
//                                            years, cardinals)
//   5. normalize_special                    (links, dashes, "A.B", brackets)
//   6. expand_abbreviations                 (mr. -> mister, CPU -> c p u, ...)
//   7. normalize_mixedcase                  (CamelCase -> camel case word)
//   8. expand_special_characters            (@, &, %, :, ;, /, <, >, ...)
//   9. lowercase
//  10. remove_unknown_characters            (training character whitelist)
//  11. collapse_whitespace                  (incl. space before punctuation)
//  12. dedup_punctuation                    (word.. -> word., keep "...")
//  13. collapse_triple_letters              (goooood -> good)
//
// Documented deviations from the Python reference:
//   * convert_to_ascii folds Latin-1/Latin-Extended, punctuation, currency,
//     and letterlike codepoints (see soprano_unidecode_table.inc). Codepoints
//     outside those ranges (CJK, Cyrillic, ...) are dropped here and then
//     stripped by the character whitelist, instead of being romanized by the
//     full unidecode tables.
//
// Reuse notes (why existing repo components could not be used instead):
//   * engine::text::normalize_english_numbers / normalize_english_text follow
//     wetext semantics, not the inflect semantics Soprano was trained on.
//     Measured with a probe against the built library: "1234" -> "one
//     thousand two hundred thirty four" (reference: "twelve thirty-four"),
//     "3:00" -> "three,oh zero" (reference: "three o'clock"), "$2.47" ->
//     "two point four seven dollars" (reference: "two dollars, forty-seven
//     cents"), "1/2" -> "one half" (reference: "one over two"), "1050" ->
//     "one thousand fifty" (reference: "ten fifty"). Reusing them would feed
//     the LM out-of-distribution text.
//   * inflect_v2/frontend.cpp embeds its own private cardinal/ordinal with
//     British "and" spelling; the framework's english_cardinal* helpers are
//     anonymous-namespace internals with space-compound compounds and
//     1900-2099-only year handling. Per-model text frontends with their own
//     local helpers are the repo convention (echo_tts, neutts, supertonic,
//     magpie, and nemotron_asr each carry private text helpers).
//   * engine::text::unicode_normalization is NFKD decomposition only; it
//     cannot transliterate ligatures, "ss" for eszett, currency names, or
//     dash/quote folds the way unidecode does, so the generated table is
//     required for byte parity.
//   * engine::text::replace_all IS reused where semantics match (the em-dash
//     rule). The only local duplicate is a 15-line regex-replace loop
//     mirroring the framework's private normalize_english_regex; exporting
//     that from the framework's public header for a single consumer would be
//     an API change with no net code saving.
std::string clean_soprano_text(const std::string & text);

}  // namespace engine::community_models::soprano_tts
