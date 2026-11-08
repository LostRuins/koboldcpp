#pragma once

#include "engine/community_models/mms_forced_aligner/assets.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/speech_encoders/wav2vec2_encoder.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::mms_forced_aligner {

struct MmsWav2Vec2CtcConfig {
    double window_sec = 30.0;
    double context_sec = 2.0;
};

struct MmsEmissionOutput {
    // Row-major [frames, 32] CPU log probabilities: the 31 real CTC classes
    // plus the virtual <star> class (id 31) with log-probability 0.0.
    std::vector<float> log_probs;
    int64_t frames = 0;
    int64_t classes = 32;
};

// Model-free helpers; unit-tested without a checkpoint.

// (x - mean) / sqrt(variance + 1e-7) over the whole buffer, matching the
// Wav2Vec2 feature extractor; accumulates in double.
std::vector<float> mms_normalize_waveform_16k(const std::vector<float> & waveform);

// Stable per-row log-softmax over `classes` logits per frame, then appends the
// virtual <star> column (log-probability 0.0). Output is [frames, classes + 1].
std::vector<float> mms_log_softmax_and_star(const float * logits, int64_t frames, int64_t classes);

class MmsWav2Vec2CtcRuntime {
public:
    MmsWav2Vec2CtcRuntime(
        std::shared_ptr<const MmsForcedAlignerAssets> assets,
        core::BackendConfig backend,
        engine::assets::TensorStorageType weight_storage_type,
        MmsWav2Vec2CtcConfig config = {});

    // Mixdown/resample to 16 kHz, normalize, window with left/right context,
    // encode through the MMS Wav2Vec2 encoder with the CTC head, and convert to
    // [frames, 32] CPU log probabilities. Mirrors the reference
    // ctc-forced-aligner generate_emissions (30s center + 2s context windows;
    // short audio runs un-windowed).
    MmsEmissionOutput compute(const runtime::AudioBuffer & audio) const;

private:
    void load_encoder() const;

    std::shared_ptr<const MmsForcedAlignerAssets> assets_;
    core::BackendConfig backend_;
    engine::assets::TensorStorageType weight_storage_type_;
    MmsWav2Vec2CtcConfig config_;
    // Lazily loaded on first compute; a loaded encoder is non-null.
    mutable std::unique_ptr<modules::Wav2Vec2EncoderRuntime> mms_encoder_;
};

}  // namespace engine::community_models::mms_forced_aligner
