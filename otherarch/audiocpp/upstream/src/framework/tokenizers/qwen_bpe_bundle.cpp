#include "engine/framework/tokenizers/qwen_bpe_bundle.h"

#include <stdexcept>
#include <utility>

namespace engine::tokenizers {

std::shared_ptr<LlamaBpeTokenizer> load_qwen_bpe_tokenizer(
    const engine::assets::ResourceBundle & bundle) {
    LlamaBpeTokenizerSpec spec;
    spec.tokenizer_config_path = bundle.require_file("tokenizer_config");
    if (const auto * path = bundle.find_file("vocab")) {
        spec.vocab_path = *path;
    }
    if (const auto * path = bundle.find_file("merges")) {
        spec.merges_path = *path;
    }
    if (const auto * path = bundle.find_file("tokenizer_json")) {
        spec.tokenizer_json_path = *path;
    }
    spec.pre_type = LlamaBpePreTokenizer::Qwen2;
    return load_llama_bpe_tokenizer(spec);
}

int64_t require_added_token_id(
    const engine::assets::ResourceBundle & bundle,
    std::string_view content) {
    const auto tokenizer = bundle.parse_json("tokenizer_json");
    for (const auto & item : tokenizer.require("added_tokens").as_array()) {
        const auto * token_content = item.find("content");
        const auto * token_id = item.find("id");
        if (token_content != nullptr && token_content->is_string() &&
            token_id != nullptr && token_id->is_number() &&
            token_content->as_string() == content) {
            return token_id->as_i64();
        }
    }
    throw std::runtime_error("tokenizer.json is missing token: " + std::string(content));
}

}  // namespace engine::tokenizers
