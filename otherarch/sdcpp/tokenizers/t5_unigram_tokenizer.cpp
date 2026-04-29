#include "t5_unigram_tokenizer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <regex>
#include <sstream>

#include "json.hpp"
#include "tokenize_util.h"
#include "util.h"
#include "vocab/vocab.h"

// Port from: https://github.com/google/sentencepiece/blob/master/src/unigram_model.h
// and https://github.com/google/sentencepiece/blob/master/src/unigram_model.h.
// Original License: https://github.com/google/sentencepiece/blob/master/LICENSE
//
// Since tokenization is not the bottleneck in SD, performance was not a major consideration
// during the migration.

MetaspacePreTokenizer::MetaspacePreTokenizer(const std::string replacement, bool add_prefix_space)
    : replacement(replacement), add_prefix_space(add_prefix_space) {}

std::string MetaspacePreTokenizer::tokenize(const std::string& input) const {
    std::string tokens;
    std::stringstream ss(input);

    if (add_prefix_space) {
        tokens += replacement;
    }

    std::string token;
    bool first_token = true;
    while (std::getline(ss, token, ' ')) {
        if (!first_token) {
            tokens += replacement + token;
        } else {
            tokens += token;
        }

        first_token = false;
    }

    return tokens;
}

void T5UniGramTokenizer::InitializePieces(const std::string& json_str) {
    nlohmann::json data;

    try {
        data = nlohmann::json::parse(json_str);
    } catch (const nlohmann::json::parse_error&) {
        status_ = INVLIAD_JSON;
        return;
    }
    if (!data.contains("model")) {
        status_ = INVLIAD_JSON;
        return;
    }
    nlohmann::json model = data["model"];
    if (!model.contains("vocab")) {
        status_ = INVLIAD_JSON;
        return;
    }
    if (model.contains("unk_id")) {
        UNK_TOKEN_ID = model["unk_id"];
    }

    replacement      = data["pre_tokenizer"]["replacement"];
    add_prefix_space = data["pre_tokenizer"]["add_prefix_space"];

    pre_tokenizer = MetaspacePreTokenizer(replacement, add_prefix_space);

    for (const auto& item : model["vocab"]) {
        if (item.size() != 2 || !item[0].is_string() || !item[1].is_number_float()) {
            status_ = INVLIAD_JSON;
            return;
        }
        std::string piece = item[0];
        if (piece.empty()) {
            piece = "<empty_token>";
        }
        float score = item[1];
        piece_score_pairs.emplace_back(piece, score);
    }
}

void T5UniGramTokenizer::BuildTrie(std::vector<std::pair<std::string, int>>* pieces) {
    if (status_ != OK) {
        return;
    }

    if (pieces->empty()) {
        status_ = NO_PIECES_LOADED;
        return;
    }

    std::sort(pieces->begin(), pieces->end());

    std::vector<const char*> key(pieces->size());
    std::vector<int> value(pieces->size());
    for (size_t i = 0; i < pieces->size(); ++i) {
        key[i]   = (*pieces)[i].first.data();
        value[i] = (*pieces)[i].second;
    }

    trie_ = std::unique_ptr<Darts::DoubleArray>(new Darts::DoubleArray());
    if (trie_->build(key.size(), const_cast<char**>(&key[0]), nullptr, &value[0]) != 0) {
        status_ = BUILD_DOUBLE_ARRAY_FAILED;
        return;
    }

    const int kMaxTrieResultsSize = 1024;
    std::vector<Darts::DoubleArray::result_pair_type> results(kMaxTrieResultsSize);
    trie_results_size_ = 0;
    for (const auto& p : *pieces) {
        const size_t num_nodes = trie_->commonPrefixSearch(
            p.first.data(), results.data(), results.size(), p.first.size());
        trie_results_size_ = std::max(trie_results_size_, static_cast<int>(num_nodes));
    }

    if (trie_results_size_ == 0) {
        status_ = NO_ENTRY_FOUND;
    }
}

float T5UniGramTokenizer::GetScoreInlined(int id) const {
    return piece_score_pairs[id].second;
}

bool T5UniGramTokenizer::IsUnusedInlined(int id) const {
    (void)id;
    return false;
}

bool T5UniGramTokenizer::IsUserDefinedInlined(int id) const {
    (void)id;
    return false;
}

size_t T5UniGramTokenizer::OneCharLen(const char* src) const {
    return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

EncodeResult T5UniGramTokenizer::EncodeOptimized(const std::string& normalized) const {
    if (status() != OK || normalized.empty()) {
        return {};
    }

    struct BestPathNode {
        int id                = -1;
        float best_path_score = 0;
        int starts_at         = -1;
    };

    const int size        = static_cast<int>(normalized.size());
    const float unk_score = min_score() - kUnkPenalty;
    std::vector<BestPathNode> best_path_ends_at(size + 1);

    int starts_at = 0;
    while (starts_at < size) {
        std::size_t node_pos                 = 0;
        std::size_t key_pos                  = starts_at;
        const auto best_path_score_till_here = best_path_ends_at[starts_at].best_path_score;
        bool has_single_node                 = false;
        const int mblen                      = std::min<int>(static_cast<int>(OneCharLen(normalized.data() + starts_at)), size - starts_at);
        while (key_pos < static_cast<size_t>(size)) {
            const int ret = trie_->traverse(normalized.data(), node_pos, key_pos, key_pos + 1);
            if (ret == -2) {
                break;
            }
            if (ret >= 0) {
                if (IsUnusedInlined(ret)) {
                    continue;
                }
                auto& target_node                    = best_path_ends_at[key_pos];
                const auto length                    = static_cast<int>(key_pos - starts_at);
                const auto score                     = IsUserDefinedInlined(ret) ? (length * max_score_ - 0.1f) : GetScoreInlined(ret);
                const auto candidate_best_path_score = score + best_path_score_till_here;
                if (target_node.starts_at == -1 || candidate_best_path_score > target_node.best_path_score) {
                    target_node.best_path_score = static_cast<float>(candidate_best_path_score);
                    target_node.starts_at       = starts_at;
                    target_node.id              = ret;
                }
                if (!has_single_node && length == mblen) {
                    has_single_node = true;
                }
            }
        }
        if (!has_single_node) {
            auto& target_node                    = best_path_ends_at[starts_at + mblen];
            const auto candidate_best_path_score = unk_score + best_path_score_till_here;
            if (target_node.starts_at == -1 || candidate_best_path_score > target_node.best_path_score) {
                target_node.best_path_score = candidate_best_path_score;
                target_node.starts_at       = starts_at;
                target_node.id              = UNK_TOKEN_ID;
            }
        }
        starts_at += mblen;
    }

    EncodeResult results;
    int ends_at = size;
    while (ends_at > 0) {
        const auto& node = best_path_ends_at[ends_at];
        results.emplace_back(normalized.substr(node.starts_at, ends_at - node.starts_at), node.id);
        ends_at = node.starts_at;
    }
    std::reverse(results.begin(), results.end());
    return results;
}

T5UniGramTokenizer::T5UniGramTokenizer(bool is_umt5) {
    add_bos_token = false;
    add_eos_token = true;

    if (is_umt5) {
        PAD_TOKEN_ID = 0;
        EOS_TOKEN_ID = 1;
        BOS_TOKEN_ID = 2;
        UNK_TOKEN_ID = 3;

        PAD_TOKEN = "<pad>";
        EOS_TOKEN = "</s>";
        BOS_TOKEN = "<s>";
        UNK_TOKEN = "<unk>";
    } else {
        PAD_TOKEN_ID = 0;
        EOS_TOKEN_ID = 1;
        UNK_TOKEN_ID = 2;

        PAD_TOKEN = "<pad>";
        EOS_TOKEN = "</s>";
        UNK_TOKEN = "<unk>";
    }

    special_tokens = {
        "<pad>",
        "</s>",
        "<unk>",
    };

    if (is_umt5) {
        special_tokens.push_back("<s>");
    }

    if (is_umt5) {
        InitializePieces(load_umt5_tokenizer_json());
    } else {
        InitializePieces(load_t5_tokenizer_json());
    }

    min_score_ = FLT_MAX;
    max_score_ = FLT_MIN;

    std::vector<std::pair<std::string, int>> pieces;
    for (int i = 0; i < static_cast<int>(piece_score_pairs.size()); i++) {
        const auto& sp = piece_score_pairs[i];

        min_score_ = std::min(min_score_, sp.second);
        max_score_ = std::max(max_score_, sp.second);

        pieces.emplace_back(sp.first, i);
    }

    BuildTrie(&pieces);
}

T5UniGramTokenizer::~T5UniGramTokenizer() = default;

std::string T5UniGramTokenizer::decode_token(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(piece_score_pairs.size())) {
        return "";
    }

    const std::string& piece = piece_score_pairs[token_id].first;
    if (piece == "<empty_token>") {
        return "";
    }
    return piece;
}

std::string T5UniGramTokenizer::normalize(const std::string& input) const {
    // Ref: https://github.com/huggingface/tokenizers/blob/1ff56c0c70b045f0cd82da1af9ac08cd4c7a6f9f/bindings/python/py_src/tokenizers/implementations/sentencepiece_unigram.py#L29
    // TODO: nmt-nfkc
    std::string normalized = std::regex_replace(input, std::regex(" {2,}"), " ");
    return normalized;
}

std::vector<int> T5UniGramTokenizer::encode(const std::string& input, on_new_token_cb_t on_new_token_cb) {
    std::vector<int32_t> tokens;
    std::vector<std::string> token_strs;
    std::string normalized = normalize(input);
    auto splited_texts     = split_with_special_tokens(normalized, special_tokens);
    if (splited_texts.empty()) {
        splited_texts.push_back(normalized);  // for empty string
    }

    for (auto& splited_text : splited_texts) {
        if (is_special_token(splited_text)) {
            if (on_new_token_cb != nullptr) {
                bool skip = on_new_token_cb(splited_text, tokens);
                if (skip) {
                    token_strs.push_back(splited_text);
                    continue;
                }
            }

            if (splited_text == UNK_TOKEN) {
                tokens.push_back(UNK_TOKEN_ID);
                token_strs.push_back(UNK_TOKEN);
            } else if (splited_text == EOS_TOKEN) {
                tokens.push_back(EOS_TOKEN_ID);
                token_strs.push_back(EOS_TOKEN);
            } else if (splited_text == PAD_TOKEN) {
                tokens.push_back(PAD_TOKEN_ID);
                token_strs.push_back(PAD_TOKEN);
            }
            continue;
        }

        std::string pretokenized = pre_tokenizer.tokenize(splited_text);
        EncodeResult result      = EncodeOptimized(pretokenized);
        for (const auto& item : result) {
            tokens.push_back(item.second);
            token_strs.push_back(item.first);
        }
    }

    std::stringstream ss;
    ss << "[";
    for (const auto& token_str : token_strs) {
        ss << "\"" << token_str << "\", ";
    }
    ss << "]";
    LOG_DEBUG("split prompt \"%s\" to tokens %s", input.c_str(), ss.str().c_str());

    return tokens;
}
