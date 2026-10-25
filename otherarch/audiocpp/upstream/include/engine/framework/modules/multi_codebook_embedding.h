#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::modules {

struct MultiCodebookEmbeddingSpec {
    int64_t hidden_size = 0;
    int64_t num_codebooks = 0;
    int64_t vocab_size = 0;
    int64_t pad_token_id = 0;
    std::vector<int64_t> codebook_sizes;
    std::string tensor_prefix = "audio_embeddings";
};

class MultiCodebookEmbedding {
public:
    MultiCodebookEmbedding(const assets::TensorSource & source, MultiCodebookEmbeddingSpec spec);

    int64_t hidden_size() const noexcept { return hidden_size_; }
    int64_t num_codebooks() const noexcept { return num_codebooks_; }
    int32_t pad_token_id() const noexcept { return pad_token_id_; }
    int64_t codebook_size(int64_t codebook) const;
    const float * embedding(int64_t codebook, int32_t code) const;
    void add_bias(const int32_t * codes, float * bias) const;
    std::vector<float> bias_for(const int32_t * codes) const;

private:
    int64_t hidden_size_ = 0;
    int64_t num_codebooks_ = 0;
    int32_t pad_token_id_ = 0;
    std::vector<std::vector<float>> embeddings_;
};

}  // namespace engine::modules
