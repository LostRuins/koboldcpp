#pragma once

// Prompt assembly shared by the MOSS `moss_tts_delay` family.
//
// Every member of the family wraps a `<user_inst>` block in the same ChatML turn
// and splices reference audio into it the same way -- the checkpoints ship a
// byte-identical chat_template.jinja, and the block's field list is shared. What
// differs between them is the TEXT of that block: MOSS-TTS-v1.5 normalises the
// spoken text and carries a token budget, MOSS-TTSD pins Tokens and Scene to
// "None" and renders one entry per speaker. So the rendering stays with each
// model and the assembly lives here, rather than each model re-deriving the
// delay pattern and the placeholder splitting.

#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/framework/decoders/moss_tts_delay/config.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::decoders {

// The placeholder a rendered block leaves where a reference span belongs. Taken
// from the reference processor's AUDIO_PLACEHOLDER so a rendered block can be
// compared against it directly.
inline constexpr const char * kMossAudioPlaceholder = "<|audio|>";

// One speaker's reference codes, [n_vq][frames] as the codec returns them.
struct MossReferenceAudio {
    std::vector<std::vector<int32_t>> codes;
    int64_t frames = 0;
};

// An unset field renders as the literal "None", and so does one that is present
// but blank -- the reference does str(None) on the dataclass field, and a field
// trimmed to nothing is not a value the template can carry.
std::string moss_prompt_field(const std::optional<std::string> & value);

// Delay-patterns one reference's codes: row t of codebook v carries frame t - v,
// and the rest is audio_pad_code. Returns frames + n_vq - 1 rows.
std::vector<std::vector<int32_t>> moss_apply_delay_pattern(
    const MossReferenceAudio & reference, int64_t num_codebooks, int32_t audio_pad_code);

// The generation prefix for an already-rendered `<user_inst>` block: the chat
// wrapper, the block with each placeholder expanded into a reference span, and
// the assistant header.
//
// `model_label` only names the model in error messages.
codecs::MossTokenRows build_moss_generation_prefix(
    const std::string & user_inst,
    const std::vector<MossReferenceAudio> & references,
    const MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view model_label);

// The prefix for CONTINUATION, where the model is handed audio to carry on from
// rather than asked to start fresh. Used by MOSS-TTSD for cloning: the user turn
// names the speakers and this appends an assistant turn holding the reference
// recording, which the generated dialogue continues.
//
// ⚠ THE ASSISTANT SPAN IS DELIBERATELY UNFINISHED. A completed span is `frames`
// generation slots, then n_vq - 1 delay slots as the codebooks retire, then
// audio_end. This one stops after the `frames` generation slots: the reference
// truncates in continuation mode, so the codebooks are still mid-flight and the
// model's first sampled row continues the delay pattern instead of starting a
// new one. Emitting the delay slots or the audio_end would tell the model the
// audio had ended, and it would begin again rather than carry on.
codecs::MossTokenRows build_moss_continuation_prefix(
    const std::string & user_inst,
    const std::vector<MossReferenceAudio> & references,
    const MossReferenceAudio & assistant_audio,
    const MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view model_label);

}  // namespace engine::decoders
