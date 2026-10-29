#pragma once

#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::chatterbox_turbo {

// Builds a GPT2-style BPE tokenizer from the vocab.json/merges.txt/special_tokens.json sidecar
// files embedded in the repacked native GGUF (see
// tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py, which splits the
// upstream tokenizer.ggml.tokens/merges GGUF metadata arrays into these plain files at conversion
// time -- the trailing `[laugh]`/`[sigh]`/... emotion tags are pulled out into
// special_tokens_path so they are matched as atomic tokens rather than shredded into byte-level
// BPE pieces).
std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_chatterbox_turbo_tokenizer(
    const std::filesystem::path & vocab_path,
    const std::filesystem::path & merges_path,
    const std::filesystem::path & special_tokens_path);

// Direct port of tts_turbo.py::punc_norm: capitalizes the first letter, collapses whitespace,
// replaces punctuation characters uncommon in the training data, and ensures a trailing
// sentence-ending punctuation mark.
std::string chatterbox_turbo_punc_norm(const std::string & text);

std::vector<int32_t> encode_chatterbox_turbo_text(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & text);

}  // namespace engine::community_models::chatterbox_turbo
