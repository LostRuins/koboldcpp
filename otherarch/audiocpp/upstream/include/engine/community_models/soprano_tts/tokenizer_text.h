#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::soprano_tts {

// Byte-level BPE tokenizer for the Soprano text prompt. The LM expects the text
// wrapped as a single prompt: "[STOP][TEXT]<speech>[START]". Before encoding,
// text is normalized with the full reference `clean_text` chain (ekwek1/soprano,
// see text_normalizer.h): numbers, symbols, and abbreviations are expanded,
// punctuation is canonicalized, and the text is folded to the training
// character whitelist so the model only renders pauses for sentence marks.
class SopranoTextTokenizer {
public:
    explicit SopranoTextTokenizer(const std::filesystem::path & tokenizer_json_path);

    std::vector<int32_t> encode_text(const std::string & text) const;
    std::string decode_ids(const std::vector<int32_t> & ids) const;

    int32_t bos_id() const noexcept { return bos_id_; }
    int32_t eos_id() const noexcept { return eos_id_; }
    int32_t stop_id() const noexcept { return stop_id_; }
    int32_t text_id() const noexcept { return text_id_; }
    int32_t start_id() const noexcept { return start_id_; }
    int64_t vocab_size() const noexcept { return static_cast<int64_t>(id_to_token_.size()); }

private:
    std::vector<int32_t> apply_prompt(const std::vector<int32_t> & speech_tokens) const;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::pair<int32_t, int32_t>> merges_;  // rank-ordered (a_id, b_id)

    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t stop_id_ = -1;
    int32_t text_id_ = -1;
    int32_t start_id_ = -1;
};

}  // namespace engine::community_models::soprano_tts
