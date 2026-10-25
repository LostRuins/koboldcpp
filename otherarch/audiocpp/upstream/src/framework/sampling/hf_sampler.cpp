#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::sampling {

namespace {

std::string context_message(std::string_view context, std::string_view message) {
    if (context.empty()) {
        return std::string(message);
    }
    std::string out(context);
    out += " ";
    out += message;
    return out;
}

void require_min_tokens(int64_t min_tokens_to_keep) {
    if (min_tokens_to_keep <= 0) {
        throw std::runtime_error("HF sampler min_tokens_to_keep must be positive");
    }
}

class TorchCpuMt19937 {
public:
    explicit TorchCpuMt19937(uint64_t seed) {
        state_[0] = static_cast<uint32_t>(seed & 0xffffffffU);
        for (int index = 1; index < kStateN; ++index) {
            state_[index] =
                1812433253U * (state_[index - 1] ^ (state_[index - 1] >> 30U)) + static_cast<uint32_t>(index);
        }
        left_ = 1;
        next_ = 0;
    }

    uint32_t random() {
        if (--left_ == 0) {
            next_state();
        }
        uint32_t y = state_[next_++];
        y ^= y >> 11U;
        y ^= (y << 7U) & 0x9d2c5680U;
        y ^= (y << 15U) & 0xefc60000U;
        y ^= y >> 18U;
        return y;
    }

    uint64_t random64() {
        const uint32_t high = random();
        const uint32_t low = random();
        return (static_cast<uint64_t>(high) << 32U) | static_cast<uint64_t>(low);
    }

    void discard(uint64_t count) {
        for (uint64_t index = 0; index < count; ++index) {
            (void)random();
        }
    }

    void discard_exponential_draws(uint64_t count) {
        discard(count * 2ULL);
    }

private:
    static constexpr int kStateN = 624;
    static constexpr int kStateM = 397;
    static constexpr uint32_t kMatrixA = 0x9908b0dfU;
    static constexpr uint32_t kUpperMask = 0x80000000U;
    static constexpr uint32_t kLowerMask = 0x7fffffffU;

    static uint32_t mix_bits(uint32_t lhs, uint32_t rhs) {
        return (lhs & kUpperMask) | (rhs & kLowerMask);
    }

    static uint32_t twist(uint32_t lhs, uint32_t rhs) {
        return (mix_bits(lhs, rhs) >> 1U) ^ ((rhs & 1U) != 0U ? kMatrixA : 0U);
    }

    void next_state() {
        uint32_t * p = state_.data();
        left_ = kStateN;
        next_ = 0;

        for (int index = kStateN - kStateM + 1; --index != 0; ++p) {
            *p = p[kStateM] ^ twist(p[0], p[1]);
        }
        for (int index = kStateM; --index != 0; ++p) {
            *p = p[kStateM - kStateN] ^ twist(p[0], p[1]);
        }
        *p = p[kStateM - kStateN] ^ twist(p[0], state_[0]);
    }

    int left_ = 1;
    uint32_t next_ = 0;
    std::array<uint32_t, kStateN> state_{};
};

float torch_cpu_exponential_float(TorchCpuMt19937 & rng) {
    constexpr uint64_t mask = (uint64_t{1} << std::numeric_limits<double>::digits) - 1U;
    constexpr double divisor = 1.0 / static_cast<double>(uint64_t{1} << std::numeric_limits<double>::digits);
    const double uniform = static_cast<double>(rng.random64() & mask) * divisor;
    return static_cast<float>(-std::log1p(-uniform));
}

int32_t sample_torch_cpu_multinomial(
    const std::vector<float> & scores,
    uint64_t seed,
    uint64_t call_index,
    std::string_view context) {
    TorchCpuMt19937 rng(seed);
    rng.discard_exponential_draws(call_index * static_cast<uint64_t>(scores.size()));

    float max_score = -std::numeric_limits<float>::infinity();
    for (const float score : scores) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error(context_message(context, "sampler has no finite logits"));
    }
    double best_rank = -std::numeric_limits<double>::infinity();
    int32_t best_token = -1;
    for (size_t token = 0; token < scores.size(); ++token) {
        const float exponential = torch_cpu_exponential_float(rng);
        if (!std::isfinite(scores[token])) {
            continue;
        }
        const double weight = std::exp(static_cast<double>(scores[token] - max_score));
        const double rank = weight / static_cast<double>(exponential);
        if (rank > best_rank) {
            best_rank = rank;
            best_token = static_cast<int32_t>(token);
        }
    }
    if (best_token < 0) {
        throw std::runtime_error(context_message(context, "CPU Torch sampler failed to select a token"));
    }
    return best_token;
}

}  // namespace

void HfSamplerScratch::reserve_vocab(size_t vocab_size) {
    scores_.reserve(vocab_size);
    candidates_.reserve(vocab_size);
    weights_.reserve(vocab_size);
    seen_.reserve(vocab_size);
}

void HfSamplerScratch::reset_vocab(size_t vocab_size) {
    candidates_.clear();
    weights_.clear();
    probabilities_ready_ = false;
    probabilities_scores_data_ = nullptr;
    probabilities_scores_size_ = 0;
    if (seen_.size() != vocab_size) {
        seen_.assign(vocab_size, 0);
        seen_generation_ = 1;
    } else if (seen_generation_ == 0) {
        std::fill(seen_.begin(), seen_.end(), 0);
        seen_generation_ = 1;
    }
}

int32_t HfLogitsProcessor::argmax(const float * logits, size_t vocab_size, std::string_view context) {
    if (logits == nullptr || vocab_size == 0) {
        throw std::runtime_error(context_message(context, "sampler cannot select from empty logits"));
    }
    size_t best = 0;
    for (size_t index = 1; index < vocab_size; ++index) {
        if (logits[index] > logits[best]) {
            best = index;
        }
    }
    return static_cast<int32_t>(best);
}

void HfLogitsProcessor::apply_repetition_penalty(
    std::vector<float> & scores,
    const std::vector<int32_t> & history,
    float penalty,
    HfSamplerScratch & scratch) {
    apply_repetition_penalty(scores, history.data(), history.size(), penalty, scratch);
}

void HfLogitsProcessor::apply_repetition_penalty(
    std::vector<float> & scores,
    const int32_t * history,
    size_t history_size,
    float penalty,
    HfSamplerScratch & scratch) {
    if (penalty == 1.0F || history_size == 0) {
        return;
    }
    if (!(penalty > 0.0F) || !std::isfinite(penalty)) {
        throw std::runtime_error("HF sampler repetition_penalty must be finite and positive");
    }
    scratch.reset_vocab(scores.size());
    const uint32_t generation = scratch.seen_generation_++;
    for (size_t index = 0; index < history_size; ++index) {
        const int32_t token = history[index];
        if (token < 0 || static_cast<size_t>(token) >= scores.size()) {
            continue;
        }
        const size_t token_index = static_cast<size_t>(token);
        if (scratch.seen_[token_index] == generation) {
            continue;
        }
        scratch.seen_[token_index] = generation;
        float & score = scores[token_index];
        score = score < 0.0F ? score * penalty : score / penalty;
    }
    scratch.probabilities_ready_ = false;
    scratch.probabilities_scores_data_ = nullptr;
    scratch.probabilities_scores_size_ = 0;
}

void HfLogitsProcessor::apply_top_k(
    std::vector<float> & scores,
    int64_t top_k,
    int64_t min_tokens_to_keep,
    HfSamplerScratch & scratch) {
    require_min_tokens(min_tokens_to_keep);
    if (top_k <= 0 || scores.empty()) {
        return;
    }
    const int64_t keep = std::min<int64_t>(
        static_cast<int64_t>(scores.size()),
        std::max(top_k, min_tokens_to_keep));
    if (keep >= static_cast<int64_t>(scores.size())) {
        return;
    }
    auto & order = scratch.candidates_;
    order.resize(scores.size());
    for (size_t index = 0; index < scores.size(); ++index) {
        order[index] = static_cast<int32_t>(index);
    }
    auto kth = order.begin() + static_cast<std::ptrdiff_t>(keep - 1);
    std::nth_element(order.begin(), kth, order.end(), [&](int32_t lhs, int32_t rhs) {
        return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)];
    });
    const float threshold = scores[static_cast<size_t>(*kth)];
    for (float & score : scores) {
        if (score < threshold) {
            score = -std::numeric_limits<float>::infinity();
        }
    }
    scratch.probabilities_ready_ = false;
    scratch.probabilities_scores_data_ = nullptr;
    scratch.probabilities_scores_size_ = 0;
}

void HfLogitsProcessor::apply_top_p(
    std::vector<float> & scores,
    float top_p,
    int64_t min_tokens_to_keep,
    HfSamplerScratch & scratch) {
    require_min_tokens(min_tokens_to_keep);
    if (!(top_p < 1.0F)) {
        return;
    }
    if (top_p < 0.0F || !std::isfinite(top_p)) {
        throw std::runtime_error("HF sampler top_p must be finite and in [0, 1]");
    }
    if (scores.empty()) {
        return;
    }
    auto & sorted = scratch.candidates_;
    sorted.clear();
    sorted.reserve(scores.size());
    for (size_t index = 0; index < scores.size(); ++index) {
        if (std::isfinite(scores[index])) {
            sorted.push_back(static_cast<int32_t>(index));
        }
    }
    if (sorted.empty()) {
        return;
    }
    std::sort(sorted.begin(), sorted.end(), [&](int32_t lhs, int32_t rhs) {
        const float lhs_score = scores[static_cast<size_t>(lhs)];
        const float rhs_score = scores[static_cast<size_t>(rhs)];
        if (lhs_score == rhs_score) {
            return lhs < rhs;
        }
        return lhs_score < rhs_score;
    });
    const float max_score = scores[static_cast<size_t>(sorted.back())];
    auto & weights = scratch.weights_;
    weights.resize(sorted.size());
    double total = 0.0;
    for (size_t index = 0; index < sorted.size(); ++index) {
        const int32_t token = sorted[index];
        weights[index] = std::exp(static_cast<double>(scores[static_cast<size_t>(token)] - max_score));
        total += weights[index];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("HF sampler top-p probability mass is invalid");
    }
    double cumulative = 0.0;
    size_t kept_count = 0;
    const double remove_mass = 1.0 - static_cast<double>(top_p);
    const size_t min_keep = std::min<size_t>(static_cast<size_t>(min_tokens_to_keep), sorted.size());
    const size_t protected_from = sorted.size() - min_keep;
    for (size_t index = 0; index < sorted.size(); ++index) {
        const int32_t token = sorted[index];
        cumulative += weights[index] / total;
        if (index < protected_from && cumulative <= remove_mass) {
            scores[static_cast<size_t>(token)] = -std::numeric_limits<float>::infinity();
        } else {
            sorted[kept_count] = token;
            weights[kept_count] = weights[index];
            ++kept_count;
        }
    }
    sorted.resize(kept_count);
    weights.resize(kept_count);
    scratch.probabilities_ready_ = true;
    scratch.probabilities_scores_data_ = scores.data();
    scratch.probabilities_scores_size_ = scores.size();
}

void HfLogitsProcessor::apply_min_p(
    std::vector<float> & scores,
    float min_p,
    int64_t min_tokens_to_keep,
    HfSamplerScratch & scratch) {
    require_min_tokens(min_tokens_to_keep);
    if (min_p < 0.0F || min_p > 1.0F || !std::isfinite(min_p)) {
        throw std::runtime_error("HF sampler min_p must be finite and in [0, 1]");
    }
    if (min_p == 0.0F) {
        return;
    }
    if (scores.empty()) {
        return;
    }
    auto & order = scratch.candidates_;
    order.clear();
    order.reserve(scores.size());
    for (size_t index = 0; index < scores.size(); ++index) {
        if (std::isfinite(scores[index])) {
            order.push_back(static_cast<int32_t>(index));
        }
    }
    if (order.empty()) {
        return;
    }
    const size_t min_keep = std::min<size_t>(
        static_cast<size_t>(min_tokens_to_keep), order.size());
    std::partial_sort(
        order.begin(),
        order.begin() + static_cast<std::ptrdiff_t>(min_keep),
        order.end(),
        [&](int32_t lhs, int32_t rhs) {
            const float lhs_score = scores[static_cast<size_t>(lhs)];
            const float rhs_score = scores[static_cast<size_t>(rhs)];
            return lhs_score == rhs_score ? lhs < rhs : lhs_score > rhs_score;
        });
    const float max_score = scores[static_cast<size_t>(order.front())];
    const float threshold = max_score + std::log(min_p);
    const float protected_threshold =
        scores[static_cast<size_t>(order[min_keep - 1])];
    for (float & score : scores) {
        if (score < threshold && score < protected_threshold) {
            score = -std::numeric_limits<float>::infinity();
        }
    }
    scratch.probabilities_ready_ = false;
    scratch.probabilities_scores_data_ = nullptr;
    scratch.probabilities_scores_size_ = 0;
}

void HfLogitsProcessor::apply_temperature(std::vector<float> & scores, float temperature) {
    if (!(temperature > 0.0F) || !std::isfinite(temperature)) {
        throw std::runtime_error("HF sampler temperature must be finite and positive");
    }
    if (temperature == 1.0F) {
        return;
    }
    for (float & score : scores) {
        score /= temperature;
    }
}

void HfLogitsProcessor::build_candidates(
    const std::vector<float> & scores,
    HfSamplerScratch & scratch,
    std::string_view context) {
    scratch.candidates_.clear();
    scratch.weights_.clear();
    scratch.candidates_.reserve(scores.size());
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < scores.size(); ++index) {
        if (std::isfinite(scores[index])) {
            scratch.candidates_.push_back(static_cast<int32_t>(index));
            max_score = std::max(max_score, scores[index]);
        }
    }
    if (scratch.candidates_.empty() || !std::isfinite(max_score)) {
        throw std::runtime_error(context_message(context, "sampler has no finite logits"));
    }
    scratch.weights_.reserve(scratch.candidates_.size());
    for (const int32_t token : scratch.candidates_) {
        const double weight = std::exp(static_cast<double>(scores[static_cast<size_t>(token)] - max_score));
        scratch.weights_.push_back(weight);
    }
    scratch.probabilities_ready_ = true;
    scratch.probabilities_scores_data_ = scores.data();
    scratch.probabilities_scores_size_ = scores.size();
}

int32_t HfTokenSampler::sample_from_processed_scores(
    const std::vector<float> & scores,
        HfSamplerScratch & scratch,
        std::mt19937 & fallback_rng,
        const HfTorchSamplingState * torch_state,
        std::string_view context,
        bool use_ready_candidates) {
    const bool candidates_match_scores =
        use_ready_candidates &&
        scratch.probabilities_ready_ &&
        scratch.probabilities_scores_data_ == scores.data() &&
        scratch.probabilities_scores_size_ == scores.size();
    if (torch_state != nullptr && torch_state->policy != nullptr && torch_state->policy->cuda_fast_path) {
        if (!candidates_match_scores) {
            HfLogitsProcessor::build_candidates(scores, scratch, context);
        }
        const uint64_t population_size = torch_state->population_size == 0
            ? static_cast<uint64_t>(scores.size())
            : torch_state->population_size;
        if (torch_state->token_index_offset + scores.size() > population_size) {
            throw std::runtime_error(context_message(context, "CUDA sampler population range is invalid"));
        }
        double best_rank = -std::numeric_limits<double>::infinity();
        int32_t best_token = -1;
        for (size_t index = 0; index < scratch.candidates_.size(); ++index) {
            const int32_t token = scratch.candidates_[index];
            const uint64_t population_token =
                torch_state->token_index_offset + static_cast<uint64_t>(token);
            const float exponential = torch_state->use_offset_blocks
                ? torch_cuda_tensor_iterator_exponential_element_at_offset(
                      torch_state->seed,
                      population_size,
                      population_token,
                      torch_state->offset_blocks,
                      torch_state->policy->multiprocessor_count,
                      torch_state->policy->max_threads_per_multiprocessor)
                : torch_cuda_tensor_iterator_exponential_element(
                      torch_state->seed,
                      population_size,
                      population_token,
                      torch_state->call_index,
                      torch_state->policy->multiprocessor_count,
                      torch_state->policy->max_threads_per_multiprocessor);
            const double rank = scratch.weights_[index] / static_cast<double>(exponential);
            if (rank > best_rank) {
                best_rank = rank;
                best_token = token;
            }
        }
        if (best_token < 0) {
            throw std::runtime_error(context_message(context, "CUDA sampler failed to select a token"));
        }
        return best_token;
    }
    if (torch_state != nullptr) {
        return sample_torch_cpu_multinomial(scores, torch_state->seed, torch_state->call_index, context);
    }
    HfLogitsProcessor::build_candidates(scores, scratch, context);
    std::discrete_distribution<size_t> distribution(scratch.weights_.begin(), scratch.weights_.end());
    return scratch.candidates_[distribution(fallback_rng)];
}

int32_t HfSampler::sample(
    const std::vector<float> & logits,
    const std::vector<int32_t> & history,
    const HfSamplingOptions & options,
    HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const HfTorchSamplingState * torch_state,
    std::string_view context) const {
    if (logits.empty()) {
        throw std::runtime_error(context_message(context, "sampler cannot select from empty logits"));
    }
    const bool needs_repetition_penalty = options.repetition_penalty != 1.0F && !history.empty();
    const bool needs_sampling_processors =
        options.do_sample &&
        (options.temperature != 1.0F || options.top_k > 0 ||
         options.top_p < 1.0F || options.min_p > 0.0F);
    if (!needs_repetition_penalty && !needs_sampling_processors) {
        if (!options.do_sample) {
            return HfLogitsProcessor::argmax(logits.data(), logits.size(), context);
        }
        scratch.probabilities_ready_ = false;
        return HfTokenSampler::sample_from_processed_scores(logits, scratch, fallback_rng, torch_state, context);
    }
    auto & scores = scratch.scores_;
    scores.assign(logits.begin(), logits.end());
    scratch.probabilities_ready_ = false;
    HfLogitsProcessor::apply_repetition_penalty(scores, history, options.repetition_penalty, scratch);
    if (!options.do_sample) {
        return HfLogitsProcessor::argmax(scores.data(), scores.size(), context);
    }
    HfLogitsProcessor::apply_temperature(scores, options.temperature);
    HfLogitsProcessor::apply_top_k(scores, options.top_k, options.min_tokens_to_keep, scratch);
    HfLogitsProcessor::apply_top_p(scores, options.top_p, options.min_tokens_to_keep, scratch);
    HfLogitsProcessor::apply_min_p(scores, options.min_p, options.min_tokens_to_keep, scratch);
    return HfTokenSampler::sample_from_processed_scores(scores, scratch, fallback_rng, torch_state, context, true);
}

}  // namespace engine::sampling
