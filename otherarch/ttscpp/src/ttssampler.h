#ifndef sampler_h
#define sampler_h

#include <stdint.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

// currently this is only built to support single sequence output sampling without beam search.
struct sampler {
    // These default configurations are based on the generation configuration for Parler TTS Mini (version 1.0)
    uint32_t n_output_heads = 9;
    uint32_t eos_token_id = 1024;
    uint32_t vocab_size = 1088;
    float temperature = 1.0f;
    uint32_t top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    std::vector<int32_t> last_token_ids;
    std::vector<uint32_t> repetition_counts;
    bool do_sample = true;
    bool apply_softmax = true;

    void sample(float * logits, std::vector<uint32_t> & output_tokens);
    void softmax(float * logits, std::vector<std::vector<size_t>> picks, std::vector<uint32_t> max_indices);
    void max(float * logits, std::vector<uint32_t> & output_tokens);
    std::vector<std::vector<size_t>> topk(float * logits, bool performed_softmax);
    void topp(float * logits, std::vector<std::vector<size_t>> & picks, std::vector<float> & max_head_probs);
    void reset();
};

#endif
