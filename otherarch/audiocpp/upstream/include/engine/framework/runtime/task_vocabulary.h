#pragma once

#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <string_view>

namespace engine::runtime {

/// One task kind, with every name any layer accepts for it.
///
/// The canonical token is what the C ABI and the CLI take and what `to_string`
/// returns. The aliases are the spellings a model spec may use: the two
/// vocabularies differ for historical reasons (`gen` vs `music`/`sfx`/`edit`),
/// and this is the only place that difference is written down.
/// Every view here must refer to a string literal.
///
/// The C ABI hands `token.data()` straight to callers as a `const char *`, so a
/// view over anything that is not NUL-terminated and statically allocated would
/// return a pointer into a temporary or an unterminated buffer. That holds for
/// the table below; it is a constraint on anything added to it.
struct TaskVocabularyEntry {
    VoiceTaskKind kind;
    std::string_view token;
    /// Spec-side spellings, `aliases[alias_count]` onwards unused.
    std::string_view aliases[4];
    std::size_t alias_count;
};

/// Every task kind this build knows, in enum order.
///
/// This is the single definition of the task vocabulary. `to_string`,
/// `parse_voice_task_kind`, the model-spec schema's allowed task set, the
/// model-spec parser and the C ABI's enumeration all read it, so they cannot
/// disagree with one another.
///
/// They used to. The schema accepted `codec`, which no parser mapped to a kind,
/// so `model_specs/miocodec.json` threw at load; the spec parser accepted
/// `audio_generation`, which the schema rejected, so a spec using it failed
/// validation instead; and `include/audiocpp.h` documented `"diarization"` and
/// `"alignment"`, which nothing accepts. Four hand-maintained lists of the same
/// fourteen things drift in four directions, and nothing was comparing them.
const TaskVocabularyEntry * task_vocabulary(std::size_t & count) noexcept;

/// The canonical token for a spec-side task name, or an empty view when the
/// name names no task kind.
///
/// `"music"` -> `"gen"`. Exposed because a caller that reads `model_specs/*.json`
/// -- a package browser, an installer, a binding building a picker before
/// anything is loaded -- needs the mapping and has no model to ask.
std::string_view task_token_for_spec_name(std::string_view spec_task) noexcept;

/// Whether a spec may declare this task name.
bool is_spec_task_name(std::string_view value) noexcept;

}  // namespace engine::runtime
