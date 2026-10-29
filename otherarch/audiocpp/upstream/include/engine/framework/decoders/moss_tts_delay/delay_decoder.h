#pragma once

#include "engine/framework/decoders/moss_tts_delay/config.h"
#include "engine/framework/decoders/moss_tts_delay/heads.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <cstdint>
#include <random>
#include <vector>

namespace engine::decoders {

// Defaults from MossTTSDelayModel.generate. The model card warns that this family is
// sensitive to them, and at a generic TTS preset it collapses into an immediate
// end-of-speech, so these travel with the checkpoint rather than with the caller.
struct MossTtsDelaySamplingOptions {
    float text_temperature = 1.5F;
    float text_top_p = 1.0F;
    int text_top_k = 50;
    float audio_temperature = 1.5F;
    float audio_top_p = 0.6F;
    int audio_top_k = 50;
    float audio_repetition_penalty = 1.1F;
    // Greedy decoding, used by the parity tests: it removes the RNG from the comparison so
    // any divergence is a real divergence.
    bool do_sample = true;
};

// MOSS-VoiceGenerator has no reference recording to anchor duration against, so left
// unbounded it either retires the codebooks on the first frame or keeps talking well past
// the text. These bounds gate the two decisions the model would otherwise make freely:
// starting the flush, and ending the turn. They do not touch pauses inside an utterance.
struct MossTtsDelayLengthBounds {
    int64_t min_frames = 0;  // 0 disables the floor
    int64_t max_frames = 0;  // 0 disables the ceiling
};

struct MossTtsDelayRow {
    int32_t text_token = 0;
    std::vector<int32_t> codes;  // n_vq entries, audio_pad_code where nothing was sampled
};

// The delay-pattern state machine. Codebook i is delayed by i steps: it stays padded until
// the audio has been running for more than i steps, and after the text ends there is an
// n_vq-step flush window in which the codebooks retire one by one. Ported from
// MossTTSDelayModel.generate with batch size one.
class MossTtsDelayDecoder {
public:
    MossTtsDelayDecoder(
        MossTtsDelayConfig config,
        MossTtsDelaySamplingOptions sampling,
        uint32_t seed,
        MossTtsDelayLengthBounds bounds = {});

    // Consumes one step's logits (modified in place while masking) and returns the row to
    // feed back into the model.
    MossTtsDelayRow step(MossTtsDelayStepLogits & logits);

    // Seeds the repetition penalty with the prompt's audio rows, [rows][n_vq]
    // row-major. The reference penalises against every earlier row including the
    // prompt's; for a prompt that carries no audio that is all pad and makes no
    // difference, but a cloning prompt carries the reference recording's codes
    // and without this they are never penalised. Kept apart from the generated
    // history so extract_audio_codes() still returns only what was generated.
    void seed_prompt_codes(const int32_t * codes, int64_t rows);

    // ⚠ CONTINUATION NEEDS THE STATE MACHINE TOLD, NOT JUST THE PENALTY SEEDED.
    // When the prompt already carries assistant audio -- MOSS-TTSD clones this
    // way -- the model will not emit an audio-start, because the prefix ended
    // mid-span. Left at its initial state the decoder therefore believes no
    // audio has begun: in_audio_ stays false and audio_length_ stays 0, so
    // `started = audio_length_ > codebook` masks every codebook but the first
    // and the delay ramp restarts from nothing. The model then begins a fresh
    // utterance instead of carrying on, and re-speaks the text the prompt audio
    // had already covered.
    //
    // `prompt_audio_frames` is the number of audio frames the prefix carries.
    void begin_continuation(int64_t prompt_audio_frames);

    bool stopped() const noexcept { return stopped_; }
    int64_t steps() const noexcept { return step_index_; }

    // Strips the delay pattern and the padding rows, yielding [n_vq, frames] row-major.
    std::vector<int32_t> extract_audio_codes(int64_t & codebooks_out, int64_t & frames_out) const;

private:
    int32_t sample_text(std::vector<float> & logits);
    int32_t sample_code(std::vector<float> & logits, int64_t codebook);

    MossTtsDelayConfig config_;
    MossTtsDelaySamplingOptions sampling_;
    MossTtsDelayLengthBounds bounds_;
    std::mt19937 rng_;
    engine::sampling::HfSamplerScratch sampler_scratch_;
    uint64_t sample_call_index_ = 0;

    int64_t step_index_ = 0;
    bool stopped_ = false;
    bool in_audio_ = false;
    int64_t audio_length_ = 0;
    // Counts down the flush window once the text side has finished. The sentinel means
    // "not flushing"; the reference uses INT64_MAX for the same purpose.
    static constexpr int64_t kNotDelaying = -1;
    int64_t delayed_length_ = kNotDelaying;

    std::vector<MossTtsDelayRow> history_;
    std::vector<MossTtsDelayRow> prompt_history_;
};

}  // namespace engine::decoders
