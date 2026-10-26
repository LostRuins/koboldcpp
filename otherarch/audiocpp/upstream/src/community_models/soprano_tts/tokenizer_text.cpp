#include "engine/community_models/soprano_tts/tokenizer_text.h"

#include "engine/community_models/soprano_tts/text_normalizer.h"
#include "engine/framework/io/json.h"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::community_models::soprano_tts {
namespace {

namespace json = engine::io::json;

int32_t token_id(const std::unordered_map<std::string, int32_t> & token_to_id,
                 const std::string & token) {
    const auto it = token_to_id.find(token);
    if (it == token_to_id.end()) {
        throw std::runtime_error("Soprano tokenizer missing token: " + token);
    }
    return it->second;
}

std::vector<std::string> pre_tokenize(const std::string & text) {
    // GPT-2-style byte-level pre-tokenizer: split on whitespace and digits,
    // keeping punctuation attached.
    std::vector<std::string> parts;
    std::string cur;
    for (const char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            // Flush current word, then keep the space as its own piece.
            if (!cur.empty()) {
                parts.push_back(std::move(cur));
                cur.clear();
            }
            parts.emplace_back(1, ' ');
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) parts.push_back(std::move(cur));
    return parts;
}

}  // namespace

SopranoTextTokenizer::SopranoTextTokenizer(const std::filesystem::path & path) {
    const auto root = json::parse_file(path);
    const auto & model = root.require("model");
    const auto & vocab = model.require("vocab");
    id_to_token_.resize(vocab.as_object().size());
    for (const auto & entry : vocab.as_object()) {
        const auto id = static_cast<int32_t>(entry.second.as_i64());
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
            id_to_token_[static_cast<size_t>(id)] = entry.first;
        }
        token_to_id_.emplace(entry.first, id);
    }
    for (const auto & added : root.require("added_tokens").as_array()) {
        const auto token = json::require_string(added, "content");
        const auto id = json::require_i64(added, "id");
        token_to_id_[token] = static_cast<int32_t>(id);
    }
    if (const auto * merges = model.find("merges")) {
        int32_t rank = 0;
        for (const auto & merge : merges->as_array()) {
            // Soprano ships merges as a list of two-element ["a","b"] arrays.
            const auto & pair = merge.as_array();
            if (pair.size() < 2) {
                continue;
            }
            const auto a = token_to_id_.find(pair[0].as_string());
            const auto b = token_to_id_.find(pair[1].as_string());
            if (a != token_to_id_.end() && b != token_to_id_.end()) {
                merges_.emplace_back(a->second, b->second);
                ++rank;
            }
        }
    }
    stop_id_ = token_id(token_to_id_, "[STOP]");
    text_id_ = token_id(token_to_id_, "[TEXT]");
    start_id_ = token_id(token_to_id_, "[START]");
    // config bos/eos are both the STOP token id (3).
    const auto eos_it = token_to_id_.find("[STOP]");
    eos_id_ = (eos_it != token_to_id_.end()) ? eos_it->second : 3;
    bos_id_ = eos_id_;
}
std::vector<int32_t> SopranoTextTokenizer::encode_text(const std::string & raw) const {
    const std::string text = clean_soprano_text(raw);
    std::vector<int32_t> tokens;
    const auto pieces = pre_tokenize(text);
    for (const auto & piece : pieces) {
        // Character-level: look up each character directly in the vocab.
        std::vector<int32_t> word;
        word.reserve(piece.size());
        for (const char ch : piece) {
            std::string ch_str(1, ch);
            const auto it = token_to_id_.find(ch_str);
            if (it != token_to_id_.end()) {
                word.push_back(it->second);
            } else {
                // Unknown character → [UNK]
                const auto unk = token_to_id_.find("[UNK]");
                if (unk != token_to_id_.end()) {
                    word.push_back(unk->second);
                }
            }
        }
        // BPE: repeatedly apply the lowest-rank adjacent merge.
        for (;;) {
            int64_t best_rank = -1;
            size_t best_pos = 0;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                for (size_t r = 0; r < merges_.size(); ++r) {
                    if (merges_[r].first == word[i] && merges_[r].second == word[i + 1]) {
                        if (best_rank < 0 || static_cast<int64_t>(r) < best_rank) {
                            best_rank = static_cast<int64_t>(r);
                            best_pos = i;
                        }
                        break;
                    }
                }
            }
            if (best_rank < 0) {
                break;
            }
            const auto & a_str = id_to_token_[static_cast<size_t>(word[best_pos])];
            const auto & b_str = id_to_token_[static_cast<size_t>(word[best_pos + 1])];
            const std::string merged_str = a_str + b_str;
            const auto it = token_to_id_.find(merged_str);
            if (it == token_to_id_.end()) {
                break;
            }
            word[best_pos] = it->second;
            word.erase(word.begin() + static_cast<ptrdiff_t>(best_pos + 1));
        }
        tokens.insert(tokens.end(), word.begin(), word.end());
    }
    return apply_prompt(tokens);
}

std::vector<int32_t> SopranoTextTokenizer::apply_prompt(const std::vector<int32_t> & speech) const {
    std::vector<int32_t> out;
    out.reserve(speech.size() + 4);
    out.push_back(stop_id_);
    out.push_back(text_id_);
    out.insert(out.end(), speech.begin(), speech.end());
    out.push_back(start_id_);
    return out;
}

std::string SopranoTextTokenizer::decode_ids(const std::vector<int32_t> & ids) const {
    std::string out;
    for (const int32_t id : ids) {
        if (id <= 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
            continue;
        }
        const auto & token = id_to_token_[static_cast<size_t>(id)];
        if (token.size() == 1) {
            const int byte = static_cast<unsigned char>(token[0]) - 1;
            if (byte >= 0) {
                out.push_back(static_cast<char>(byte));
            } else {
                out += token;
            }
        } else {
            out += token;
        }
    }
    return out;
}

}  // namespace engine::community_models::soprano_tts
