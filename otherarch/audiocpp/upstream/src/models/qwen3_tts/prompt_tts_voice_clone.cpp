#include "engine/models/qwen3_tts/prompt_tts_voice_clone.h"

#include <stdexcept>
#include <string>

namespace engine::models::qwen3_tts {
namespace {

void require_text_token_limit(size_t actual, int64_t limit, const char * what) {
    if (limit <= 0) {
        throw std::runtime_error(std::string("Qwen3 voice clone ") + what + " token limit must be positive");
    }
    if (actual > static_cast<size_t>(limit)) {
        throw std::runtime_error(
            std::string("Qwen3 voice clone ") + what + " token count "
            + std::to_string(actual) + " exceeds limit " + std::to_string(limit));
    }
}

}  // namespace

Qwen3TTSVoiceClonePromptBuilder::Qwen3TTSVoiceClonePromptBuilder(
    const Qwen3TextTokenizer & tokenizer,
    const Qwen3SpeechTokenizerEncoderRuntime & speech_encoder,
    const Qwen3TTSEcapaTdnnEncoderRuntime & speaker_encoder,
    int64_t text_token_limit)
    : tokenizer_(tokenizer),
      speech_encoder_(speech_encoder),
      speaker_encoder_(speaker_encoder),
      text_token_limit_(text_token_limit) {}

Qwen3VoiceClonePrompt Qwen3TTSVoiceClonePromptBuilder::build_voice_prompt(const Qwen3VoiceCloneInput & input) const {
    Qwen3VoiceClonePrompt prompt;
    prompt.speaker_embedding = speaker_encoder_.encode(input.reference_audio);
    prompt.icl_mode = input.mode == Qwen3VoiceCloneMode::Icl;
    if (prompt.icl_mode) {
        if (input.reference_text.empty()) {
            throw std::runtime_error("Qwen3 voice clone ICL mode requires reference text");
        }
        prompt.reference_codes = speech_encoder_.encode(input.reference_audio);
        prompt.reference_text_ids = tokenizer_.encode(tokenizer_.build_reference_prompt(input.reference_text));
        require_text_token_limit(prompt.reference_text_ids.size(), text_token_limit_, "reference text");
    }
    return prompt;
}

Qwen3TalkerPrefill Qwen3TTSVoiceClonePromptBuilder::build_prefill(
    const Qwen3TTSRequest & request,
    const Qwen3VoiceClonePrompt & prompt) const {
    Qwen3TalkerPrefill prefill;
    prefill.prompt_mode = Qwen3TalkerPromptMode::VoiceClone;
    prefill.input_ids = tokenizer_.encode(tokenizer_.build_assistant_prompt(request.text));
    require_text_token_limit(prefill.input_ids.size(), text_token_limit_, "text");
    prefill.reference_ids = prompt.reference_text_ids;
    prefill.reference_codes = prompt.reference_codes;
    prefill.speaker_embedding = prompt.speaker_embedding;
    prefill.language = request.language;
    prefill.icl_mode = prompt.icl_mode;
    prefill.x_vector_only_mode = !prompt.icl_mode;
    return prefill;
}

}  // namespace engine::models::qwen3_tts
