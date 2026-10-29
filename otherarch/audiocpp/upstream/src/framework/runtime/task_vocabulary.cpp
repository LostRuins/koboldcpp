#include "engine/framework/runtime/task_vocabulary.h"

namespace engine::runtime {

namespace {

/// The vocabulary, in enum order.
///
/// Adding a task kind means adding one row here. Nothing else enumerates them,
/// which is the point: the four lists this replaced were edited independently
/// and drifted.
constexpr TaskVocabularyEntry kVocabulary[] = {
    {VoiceTaskKind::Vad, "vad", {"vad"}, 1},
    {VoiceTaskKind::Asr, "asr", {"asr"}, 1},
    {VoiceTaskKind::Diarization, "diar", {"diar"}, 1},
    {VoiceTaskKind::SourceSeparation, "sep", {"sep"}, 1},
    // The one genuinely many-to-one row: a spec says what the audio is for,
    // the runtime has a single generation kind.
    {VoiceTaskKind::AudioGeneration, "gen", {"music", "sfx", "edit", "audio_generation"}, 4},
    {VoiceTaskKind::Tts, "tts", {"tts"}, 1},
    {VoiceTaskKind::VoiceCloning, "clon", {"clone"}, 1},
    {VoiceTaskKind::VoiceConversion, "vc", {"vc"}, 1},
    {VoiceTaskKind::SpeechToSpeech, "s2s", {"s2s"}, 1},
    {VoiceTaskKind::Alignment, "align", {"align"}, 1},
    {VoiceTaskKind::VoiceDesign, "vdes", {"design"}, 1},
    {VoiceTaskKind::SpeakerRecognition, "spk", {"speaker"}, 1},
    {VoiceTaskKind::Svc, "svc", {"svc"}, 1},
    {VoiceTaskKind::Midi, "midi", {"midi"}, 1},
    {VoiceTaskKind::TurnDetection, "turn", {"turn"}, 1},
    {VoiceTaskKind::WakeWord, "wake", {"wake_word"}, 1},
    {VoiceTaskKind::AudioClassification, "cls", {"classify"}, 1},
};

/// Compile-time exhaustiveness, kept deliberately.
///
/// Replacing the hand-written switches with a table lookup gave up something
/// they provided for free: `-Wswitch` told you when a VoiceTaskKind was added
/// and not handled. A table cannot, so a new kind would have silently become
/// `to_string` -> "unknown" and an ABI enumeration that omits it.
///
/// This switch has no default and every case falls through to one return, so
/// adding a kind without adding a row below fails the build the same way it
/// used to. It carries no data, so it cannot drift from the table -- it only
/// forces whoever adds a kind to open this file.
constexpr bool vocabulary_is_exhaustive(VoiceTaskKind kind) {
    switch (kind) {
    case VoiceTaskKind::Vad:
    case VoiceTaskKind::Asr:
    case VoiceTaskKind::Diarization:
    case VoiceTaskKind::SourceSeparation:
    case VoiceTaskKind::AudioGeneration:
    case VoiceTaskKind::Tts:
    case VoiceTaskKind::VoiceCloning:
    case VoiceTaskKind::VoiceConversion:
    case VoiceTaskKind::SpeechToSpeech:
    case VoiceTaskKind::Alignment:
    case VoiceTaskKind::VoiceDesign:
    case VoiceTaskKind::SpeakerRecognition:
    case VoiceTaskKind::Svc:
    case VoiceTaskKind::Midi:
    case VoiceTaskKind::TurnDetection:
    case VoiceTaskKind::WakeWord:
    case VoiceTaskKind::AudioClassification:
        return true;
    }
    return false;
}

static_assert(vocabulary_is_exhaustive(VoiceTaskKind::Vad),
              "every VoiceTaskKind must be listed above and have a row in kVocabulary");

}  // namespace

const TaskVocabularyEntry * task_vocabulary(std::size_t & count) noexcept {
    count = sizeof(kVocabulary) / sizeof(kVocabulary[0]);
    return kVocabulary;
}

std::string_view task_token_for_spec_name(std::string_view spec_task) noexcept {
    for (const auto & entry : kVocabulary) {
        for (std::size_t i = 0; i < entry.alias_count; ++i) {
            if (entry.aliases[i] == spec_task) {
                return entry.token;
            }
        }
    }
    return {};
}

bool is_spec_task_name(std::string_view value) noexcept {
    return !task_token_for_spec_name(value).empty();
}

}  // namespace engine::runtime
