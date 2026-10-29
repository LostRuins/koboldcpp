#pragma once
#include "engine/framework/audio/conversion.h"
#include "engine/models/chatterbox/components.h"
#include <algorithm>
#include <stdexcept>

namespace engine::community_models::kitten_tts2 {

// Shared by live cloning and the offline preset preparer. The full decoder
// repeats short references to six seconds, then uses at most ten seconds.
inline models::chatterbox::EmbedReferenceOutputs prepare_decoder_reference(
    runtime::AudioBuffer wav24,
    const models::chatterbox::S3TokenizerComponent & tokenizer,
    const models::chatterbox::CAMPPlusEncoderComponent & camp) {
    if (wav24.sample_rate != 24000 || wav24.channels != 1 || wav24.samples.empty())
        throw std::runtime_error("Kitten decoder reference must be non-empty mono 24 kHz audio");
    const size_t original = wav24.samples.size();
    const size_t repeats = (144000 + original - 1) / original;
    const size_t length = std::min<size_t>(240000, original * std::max<size_t>(1, repeats));
    wav24.samples.resize(length);
    for (size_t i = original; i < length; ++i) wav24.samples[i] = wav24.samples[i % original];
    runtime::AudioBuffer decoder16{16000, 1, audio::resample_mono_torchaudio_sinc_hann(
        wav24.samples, 24000, 16000, audio::torchaudio_sinc_hann_float32_options())};
    return tokenizer.embed_reference_from_wavs(camp, wav24, decoder16);
}

} // namespace engine::community_models::kitten_tts2
