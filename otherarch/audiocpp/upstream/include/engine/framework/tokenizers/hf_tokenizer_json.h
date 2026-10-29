#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::tokenizers {

class HuggingFaceTokenizerJson {
public:
    HuggingFaceTokenizerJson(
        std::vector<std::string> id_to_token,
        std::string metaspace_replacement,
        bool trim_leading_space,
        bool byte_level = false);

    const std::vector<std::string> & id_to_token() const noexcept;
    std::string decode_ids(const std::vector<int32_t> & ids) const;
    std::string decode_ids(const std::vector<int32_t> & ids, int32_t skip_token_id) const;

private:
    std::vector<std::string> id_to_token_;
    std::string metaspace_replacement_;
    bool trim_leading_space_ = false;
    bool byte_level_ = false;
};

std::shared_ptr<HuggingFaceTokenizerJson> load_huggingface_tokenizer_json(
    const std::filesystem::path & path);

}  // namespace engine::tokenizers
