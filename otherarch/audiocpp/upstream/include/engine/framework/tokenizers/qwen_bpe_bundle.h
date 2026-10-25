#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <memory>
#include <string_view>

namespace engine::tokenizers {

// Load the Qwen2-pretokenized BPE tokenizer referenced by a model bundle:
// tokenizer_config.json plus vocab.json/merges.txt or tokenizer.json.
std::shared_ptr<LlamaBpeTokenizer> load_qwen_bpe_tokenizer(
    const engine::assets::ResourceBundle & bundle);

// Look up a special token's id from the bundle's tokenizer.json
// added_tokens list; throws when the token is absent.
int64_t require_added_token_id(
    const engine::assets::ResourceBundle & bundle,
    std::string_view content);

}  // namespace engine::tokenizers
