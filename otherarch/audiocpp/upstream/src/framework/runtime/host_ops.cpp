#include "engine/framework/runtime/host_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace engine::runtime {

namespace {

int64_t checked_product(std::initializer_list<int64_t> dims) {
    int64_t result = 1;
    for (const int64_t dim : dims) {
        result *= dim;
    }
    return result;
}

float stable_softmax_pick(const float * row, int64_t vocab_size, float uniform) {
    float max_value = row[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
        max_value = std::max(max_value, row[i]);
    }

    std::vector<float> probs(static_cast<size_t>(vocab_size));
    float sum = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        probs[static_cast<size_t>(i)] = std::exp(row[i] - max_value);
        sum += probs[static_cast<size_t>(i)];
    }

    float cumulative = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[static_cast<size_t>(i)] / sum;
        if (uniform <= cumulative || i == vocab_size - 1) {
            return static_cast<float>(i);
        }
    }
    return 0.0f;
}

}  // namespace

FloatHostTensor PCMInput::compute(const std::vector<int16_t> & input, int64_t batch, int64_t samples) const {
    if (static_cast<int64_t>(input.size()) != checked_product({batch, samples})) {
        throw std::runtime_error("PCMInput input size mismatch");
    }

    FloatHostTensor result;
    result.shape = {batch, samples};
    result.values.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result.values[i] = static_cast<float>(input[i]) / 32768.0f;
    }
    return result;
}

FloatHostTensor Resample::compute(
    const std::vector<float> & input,
    int64_t batch,
    int64_t channels,
    int64_t input_samples,
    int64_t output_samples) const {
    if (static_cast<int64_t>(input.size()) != checked_product({batch, channels, input_samples})) {
        throw std::runtime_error("Resample input size mismatch");
    }

    FloatHostTensor result;
    result.shape = {batch, channels, output_samples};
    result.values.resize(static_cast<size_t>(checked_product({batch, channels, output_samples})));

    const float scale = static_cast<float>(input_samples) / static_cast<float>(output_samples);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t in_base = static_cast<size_t>((b * channels + c) * input_samples);
            const size_t out_base = static_cast<size_t>((b * channels + c) * output_samples);
            for (int64_t i = 0; i < output_samples; ++i) {
                float x = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
                int64_t x0 = static_cast<int64_t>(std::floor(x));
                int64_t x1 = x0 + 1;
                const float w1 = x - static_cast<float>(x0);
                const float w0 = 1.0f - w1;
                x0 = std::clamp<int64_t>(x0, 0, input_samples - 1);
                x1 = std::clamp<int64_t>(x1, 0, input_samples - 1);
                result.values[out_base + static_cast<size_t>(i)] =
                    input[in_base + static_cast<size_t>(x0)] * w0 +
                    input[in_base + static_cast<size_t>(x1)] * w1;
            }
        }
    }

    return result;
}

FloatHostTensor Chunker::compute(
    const std::vector<float> & input,
    int64_t batch,
    int64_t frames,
    int64_t hidden_size,
    int64_t chunk_size,
    int64_t hop) const {
    if (static_cast<int64_t>(input.size()) != checked_product({batch, frames, hidden_size})) {
        throw std::runtime_error("Chunker input size mismatch");
    }
    if (chunk_size <= 0) {
        throw std::runtime_error("Chunker chunk_size must be positive");
    }
    if (hop <= 0) {
        throw std::runtime_error("Chunker hop must be positive");
    }
    if (frames < chunk_size) {
        throw std::runtime_error("Chunker frames must be >= chunk_size");
    }
    const int64_t chunk_count = ((frames - chunk_size) / hop) + 1;
    FloatHostTensor result;
    result.shape = {batch, chunk_count, chunk_size, hidden_size};
    result.values.resize(static_cast<size_t>(checked_product({batch, chunk_count, chunk_size, hidden_size})));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int64_t start = chunk * hop;
            for (int64_t t = 0; t < chunk_size; ++t) {
                for (int64_t h = 0; h < hidden_size; ++h) {
                    const size_t src = static_cast<size_t>(((b * frames) + (start + t)) * hidden_size + h);
                    const size_t dst = static_cast<size_t>((((b * chunk_count) + chunk) * chunk_size + t) * hidden_size + h);
                    result.values[dst] = input[src];
                }
            }
        }
    }
    return result;
}

FloatHostTensor OverlapAdd::compute(
    const std::vector<float> & input,
    int64_t batch,
    int64_t chunk_count,
    int64_t chunk_size,
    int64_t hidden_size,
    int64_t hop) const {
    if (static_cast<int64_t>(input.size()) != checked_product({batch, chunk_count, chunk_size, hidden_size})) {
        throw std::runtime_error("OverlapAdd input size mismatch");
    }
    const int64_t output_frames = (chunk_count - 1) * hop + chunk_size;
    FloatHostTensor result;
    result.shape = {batch, output_frames, hidden_size};
    result.values.assign(static_cast<size_t>(checked_product({batch, output_frames, hidden_size})), 0.0f);
    std::vector<float> counts(result.values.size(), 0.0f);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int64_t start = chunk * hop;
            for (int64_t t = 0; t < chunk_size; ++t) {
                for (int64_t h = 0; h < hidden_size; ++h) {
                    const size_t src = static_cast<size_t>((((b * chunk_count) + chunk) * chunk_size + t) * hidden_size + h);
                    const size_t dst = static_cast<size_t>(((b * output_frames) + (start + t)) * hidden_size + h);
                    result.values[dst] += input[src];
                    counts[dst] += 1.0f;
                }
            }
        }
    }

    for (size_t i = 0; i < result.values.size(); ++i) {
        result.values[i] /= std::max(counts[i], 1.0f);
    }
    return result;
}

FloatHostTensor LengthRegulator::compute(
    const std::vector<float> & input,
    const std::vector<int32_t> & durations,
    int64_t batch,
    int64_t frames,
    int64_t hidden_size,
    int64_t target_frames) const {
    if (static_cast<int64_t>(input.size()) != checked_product({batch, frames, hidden_size}) ||
        static_cast<int64_t>(durations.size()) != checked_product({batch, frames})) {
        throw std::runtime_error("LengthRegulator input size mismatch");
    }

    FloatHostTensor result;
    result.shape = {batch, target_frames, hidden_size};
    result.values.assign(static_cast<size_t>(checked_product({batch, target_frames, hidden_size})), 0.0f);

    for (int64_t b = 0; b < batch; ++b) {
        int64_t out_t = 0;
        for (int64_t t = 0; t < frames; ++t) {
            const int32_t repeat = std::max<int32_t>(durations[static_cast<size_t>(b * frames + t)], 0);
            for (int32_t r = 0; r < repeat && out_t < target_frames; ++r, ++out_t) {
                for (int64_t h = 0; h < hidden_size; ++h) {
                    result.values[static_cast<size_t>(((b * target_frames) + out_t) * hidden_size + h)] =
                        input[static_cast<size_t>(((b * frames) + t) * hidden_size + h)];
                }
            }
        }
    }
    return result;
}

IntHostTensor Sampler::compute(
    const std::vector<float> & logits,
    const std::vector<float> & uniforms,
    int64_t batch,
    int64_t steps,
    int64_t vocab_size) const {
    if (static_cast<int64_t>(logits.size()) != checked_product({batch, steps, vocab_size}) ||
        static_cast<int64_t>(uniforms.size()) != checked_product({batch, steps})) {
        throw std::runtime_error("Sampler input size mismatch");
    }

    IntHostTensor result;
    result.shape = {batch, steps};
    result.values.resize(static_cast<size_t>(batch * steps));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < steps; ++t) {
            const float * row = logits.data() + static_cast<size_t>((b * steps + t) * vocab_size);
            result.values[static_cast<size_t>(b * steps + t)] =
                static_cast<int32_t>(stable_softmax_pick(row, vocab_size, uniforms[static_cast<size_t>(b * steps + t)]));
        }
    }
    return result;
}

IntHostTensor AutoregressiveSampler::compute(
    const std::vector<float> & logits,
    const std::vector<float> & uniforms,
    int64_t batch,
    int64_t steps,
    int64_t vocab_size) const {
    return Sampler().compute(logits, uniforms, batch, steps, vocab_size);
}

FloatHostTensor VectorQuantizer::compute(
    const std::vector<float> & input,
    const std::vector<float> & codebook,
    int64_t batch,
    int64_t steps,
    int64_t dim,
    int64_t codebook_size) const {
    if (static_cast<int64_t>(input.size()) != checked_product({batch, steps, dim}) ||
        static_cast<int64_t>(codebook.size()) != checked_product({codebook_size, dim})) {
        throw std::runtime_error("VectorQuantizer input size mismatch");
    }
    FloatHostTensor result;
    result.shape = {batch, steps, dim};
    result.values.resize(input.size());

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < steps; ++t) {
            const float * x = input.data() + static_cast<size_t>((b * steps + t) * dim);
            int64_t best_index = 0;
            float best_distance = std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < codebook_size; ++c) {
                const float * code = codebook.data() + static_cast<size_t>(c * dim);
                float distance = 0.0f;
                for (int64_t d = 0; d < dim; ++d) {
                    const float diff = x[d] - code[d];
                    distance += diff * diff;
                }
                if (distance < best_distance) {
                    best_distance = distance;
                    best_index = c;
                }
            }
            const float * code = codebook.data() + static_cast<size_t>(best_index * dim);
            std::copy(code, code + dim, result.values.begin() + static_cast<ptrdiff_t>((b * steps + t) * dim));
        }
    }
    return result;
}

FloatHostTensor MonotonicAlignment::compute(
    const std::vector<float> & scores,
    int64_t batch,
    int64_t text_steps,
    int64_t audio_steps) const {
    if (static_cast<int64_t>(scores.size()) != checked_product({batch, text_steps, audio_steps})) {
        throw std::runtime_error("MonotonicAlignment input size mismatch");
    }
    FloatHostTensor result;
    result.shape = {batch, text_steps, audio_steps};
    result.values.assign(scores.size(), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        int64_t last = 0;
        for (int64_t t = 0; t < text_steps; ++t) {
            int64_t best = last;
            float best_score = scores[static_cast<size_t>((b * text_steps + t) * audio_steps + last)];
            for (int64_t a = last + 1; a < audio_steps; ++a) {
                const float score = scores[static_cast<size_t>((b * text_steps + t) * audio_steps + a)];
                if (score > best_score) {
                    best_score = score;
                    best = a;
                }
            }
            result.values[static_cast<size_t>((b * text_steps + t) * audio_steps + best)] = 1.0f;
            last = std::min(best, audio_steps - 1);
        }
    }
    return result;
}

IntHostTensor BeamSearch::compute(
    const std::vector<float> & logits,
    int64_t batch,
    int64_t steps,
    int64_t vocab_size,
    int64_t beam_size) const {
    if (static_cast<int64_t>(logits.size()) != checked_product({batch, steps, vocab_size})) {
        throw std::runtime_error("BeamSearch input size mismatch");
    }
    if (beam_size <= 0) {
        throw std::runtime_error("BeamSearch beam_size must be positive");
    }
    if (beam_size > vocab_size) {
        throw std::runtime_error("BeamSearch beam_size must be <= vocab_size");
    }

    IntHostTensor result;
    result.shape = {batch, beam_size, steps};
    result.values.assign(static_cast<size_t>(checked_product({batch, beam_size, steps})), 0);

    for (int64_t b = 0; b < batch; ++b) {
        using Beam = std::pair<float, std::vector<int32_t>>;
        std::vector<Beam> beams = {{0.0f, {}}};
        for (int64_t t = 0; t < steps; ++t) {
            const float * row = logits.data() + static_cast<size_t>((b * steps + t) * vocab_size);
            std::vector<int64_t> indices(static_cast<size_t>(vocab_size));
            for (int64_t i = 0; i < vocab_size; ++i) {
                indices[static_cast<size_t>(i)] = i;
            }
            std::partial_sort(
                indices.begin(),
                indices.begin() + static_cast<ptrdiff_t>(beam_size),
                indices.end(),
                [&](int64_t lhs, int64_t rhs) { return row[lhs] > row[rhs]; });

            std::vector<Beam> next;
            for (const auto & beam : beams) {
                for (int64_t i = 0; i < beam_size; ++i) {
                    const int64_t token = indices[static_cast<size_t>(i)];
                    auto sequence = beam.second;
                    sequence.push_back(static_cast<int32_t>(token));
                    next.emplace_back(beam.first + row[token], std::move(sequence));
                }
            }
            std::partial_sort(
                next.begin(),
                next.begin() + static_cast<ptrdiff_t>(std::min<int64_t>(beam_size, static_cast<int64_t>(next.size()))),
                next.end(),
                [](const Beam & lhs, const Beam & rhs) { return lhs.first > rhs.first; });
            next.resize(static_cast<size_t>(beam_size));
            beams = std::move(next);
        }

        for (int64_t beam_index = 0; beam_index < beam_size; ++beam_index) {
            for (int64_t t = 0; t < steps; ++t) {
                result.values[static_cast<size_t>(((b * beam_size) + beam_index) * steps + t)] =
                    beams[static_cast<size_t>(beam_index)].second[static_cast<size_t>(t)];
            }
        }
    }
    return result;
}

IntHostTensor CTCDecoder::compute(
    const std::vector<float> & logits,
    int64_t batch,
    int64_t steps,
    int64_t vocab_size,
    int32_t blank_id) const {
    if (static_cast<int64_t>(logits.size()) != checked_product({batch, steps, vocab_size})) {
        throw std::runtime_error("CTCDecoder input size mismatch");
    }
    IntHostTensor result;
    result.shape = {batch, steps};
    result.values.assign(static_cast<size_t>(batch * steps), blank_id);
    for (int64_t b = 0; b < batch; ++b) {
        int32_t previous = blank_id;
        int64_t out_index = 0;
        for (int64_t t = 0; t < steps; ++t) {
            const float * row = logits.data() + static_cast<size_t>((b * steps + t) * vocab_size);
            int32_t best = 0;
            for (int64_t v = 1; v < vocab_size; ++v) {
                if (row[v] > row[best]) {
                    best = static_cast<int32_t>(v);
                }
            }
            if (best != blank_id && best != previous && out_index < steps) {
                result.values[static_cast<size_t>(b * steps + out_index)] = best;
                ++out_index;
            }
            previous = best;
        }
    }
    return result;
}

}  // namespace engine::runtime
