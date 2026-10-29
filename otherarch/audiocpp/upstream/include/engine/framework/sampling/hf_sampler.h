#pragma once

#include "engine/framework/sampling/torch_random.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

namespace engine::sampling {

struct HfSamplingOptions {
    bool do_sample = true;
    float temperature = 1.0F;
    int64_t top_k = 0;
    float top_p = 1.0F;
    float min_p = 0.0F;
    int64_t min_tokens_to_keep = 1;
    float repetition_penalty = 1.0F;
};

struct HfTorchSamplingState {
    const TorchCudaSamplingPolicy * policy = nullptr;
    uint64_t seed = 0;
    uint64_t call_index = 0;
    uint64_t offset_blocks = 0;
    bool use_offset_blocks = false;
    // Preserve full-population CUDA RNG coordinates when scores are a contiguous slice.
    uint64_t population_size = 0;
    uint64_t token_index_offset = 0;
};

class HfSamplerScratch {
public:
    void reserve_vocab(size_t vocab_size);
    void reset_vocab(size_t vocab_size);

private:
    friend class HfSampler;
    friend class HfLogitsProcessor;
    friend class HfTokenSampler;
    std::vector<float> scores_;
    std::vector<int32_t> candidates_;
    std::vector<double> weights_;
    std::vector<uint32_t> seen_;
    uint32_t seen_generation_ = 1;
    bool probabilities_ready_ = false;
    const float * probabilities_scores_data_ = nullptr;
    size_t probabilities_scores_size_ = 0;
};

class HfLogitsProcessor {
public:
    static int32_t argmax(const float * logits, size_t vocab_size, std::string_view context);

    static void apply_repetition_penalty(
        std::vector<float> & scores,
        const std::vector<int32_t> & history,
        float penalty,
        HfSamplerScratch & scratch);

    static void apply_repetition_penalty(
        std::vector<float> & scores,
        const int32_t * history,
        size_t history_size,
        float penalty,
        HfSamplerScratch & scratch);

    static void apply_top_k(
        std::vector<float> & scores,
        int64_t top_k,
        int64_t min_tokens_to_keep,
        HfSamplerScratch & scratch);

    static void apply_top_p(
        std::vector<float> & scores,
        float top_p,
        int64_t min_tokens_to_keep,
        HfSamplerScratch & scratch);

    static void apply_min_p(
        std::vector<float> & scores,
        float min_p,
        int64_t min_tokens_to_keep,
        HfSamplerScratch & scratch);

    static void apply_temperature(std::vector<float> & scores, float temperature);

    static void build_candidates(
        const std::vector<float> & scores,
        HfSamplerScratch & scratch,
        std::string_view context);
};

class HfTokenSampler {
public:
    static int32_t sample_from_processed_scores(
        const std::vector<float> & scores,
        HfSamplerScratch & scratch,
        std::mt19937 & fallback_rng,
        const HfTorchSamplingState * torch_state,
        std::string_view context,
        bool use_ready_candidates = false);
};

class HfSampler {
public:
    int32_t sample(
        const std::vector<float> & logits,
        const std::vector<int32_t> & history,
        const HfSamplingOptions & options,
        HfSamplerScratch & scratch,
        std::mt19937 & fallback_rng,
        const HfTorchSamplingState * torch_state,
        std::string_view context) const;
};

}  // namespace engine::sampling
