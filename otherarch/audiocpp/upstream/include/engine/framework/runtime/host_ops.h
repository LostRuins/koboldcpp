#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::runtime {

struct FloatHostTensor {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

struct IntHostTensor {
    std::vector<int32_t> values;
    std::vector<int64_t> shape;
};

class PCMInput {
public:
    FloatHostTensor compute(const std::vector<int16_t> & input, int64_t batch, int64_t samples) const;
};

class Resample {
public:
    FloatHostTensor compute(
        const std::vector<float> & input,
        int64_t batch,
        int64_t channels,
        int64_t input_samples,
        int64_t output_samples) const;
};

class Chunker {
public:
    FloatHostTensor compute(
        const std::vector<float> & input,
        int64_t batch,
        int64_t frames,
        int64_t hidden_size,
        int64_t chunk_size,
        int64_t hop) const;
};

class OverlapAdd {
public:
    FloatHostTensor compute(
        const std::vector<float> & input,
        int64_t batch,
        int64_t chunk_count,
        int64_t chunk_size,
        int64_t hidden_size,
        int64_t hop) const;
};

class LengthRegulator {
public:
    FloatHostTensor compute(
        const std::vector<float> & input,
        const std::vector<int32_t> & durations,
        int64_t batch,
        int64_t frames,
        int64_t hidden_size,
        int64_t target_frames) const;
};

class Sampler {
public:
    IntHostTensor compute(
        const std::vector<float> & logits,
        const std::vector<float> & uniforms,
        int64_t batch,
        int64_t steps,
        int64_t vocab_size) const;
};

class AutoregressiveSampler {
public:
    IntHostTensor compute(
        const std::vector<float> & logits,
        const std::vector<float> & uniforms,
        int64_t batch,
        int64_t steps,
        int64_t vocab_size) const;
};

class VectorQuantizer {
public:
    FloatHostTensor compute(
        const std::vector<float> & input,
        const std::vector<float> & codebook,
        int64_t batch,
        int64_t steps,
        int64_t dim,
        int64_t codebook_size) const;
};

class MonotonicAlignment {
public:
    FloatHostTensor compute(
        const std::vector<float> & scores,
        int64_t batch,
        int64_t text_steps,
        int64_t audio_steps) const;
};

class BeamSearch {
public:
    IntHostTensor compute(
        const std::vector<float> & logits,
        int64_t batch,
        int64_t steps,
        int64_t vocab_size,
        int64_t beam_size) const;
};

class CTCDecoder {
public:
    IntHostTensor compute(
        const std::vector<float> & logits,
        int64_t batch,
        int64_t steps,
        int64_t vocab_size,
        int32_t blank_id) const;
};

}  // namespace engine::runtime
