#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <regex>

#include "util.h"

void Tokenizer::add_special_token(const std::string& token) {
    special_tokens.push_back(token);
}

bool Tokenizer::is_special_token(const std::string& token) const {
    for (const auto& special_token : special_tokens) {
        if (special_token == token) {
            return true;
        }
    }
    return false;
}

std::string Tokenizer::normalize(const std::string& text) const {
    return text;
}

std::vector<int> Tokenizer::tokenize(const std::string& text,
                                     on_new_token_cb_t on_new_token_cb,
                                     bool padding,
                                     size_t min_length,
                                     size_t max_length,
                                     bool allow_overflow_expand) {
    std::vector<int> tokens = encode(text, on_new_token_cb);
    if (padding) {
        pad_tokens(tokens, nullptr, nullptr, min_length, max_length, allow_overflow_expand);
    }
    return tokens;
}

void Tokenizer::pad_tokens(std::vector<int>& tokens,
                           std::vector<float>* weights,
                           std::vector<float>* mask,
                           size_t min_length,
                           size_t max_length,
                           bool allow_overflow_expand) {
    const bool use_weights = weights != nullptr;
    const bool use_mask    = mask != nullptr;

    if (use_weights && tokens.size() != weights->size()) {
        LOG_ERROR("tokens size != weights size");
        return;
    }

    const size_t bos_count           = add_bos_token ? 1 : 0;
    const size_t eos_count           = add_eos_token ? 1 : 0;
    const size_t special_token_count = bos_count + eos_count;

    auto build_sequence = [&](size_t begin,
                              size_t count,
                              size_t target_length,
                              std::vector<int>& out_tokens,
                              std::vector<float>& out_weights,
                              std::vector<float>& out_mask) {
        const size_t base_length  = count + special_token_count;
        const size_t final_length = std::max(target_length, base_length);

        out_tokens.clear();
        out_weights.clear();
        out_mask.clear();

        out_tokens.reserve(final_length);
        if (use_weights) {
            out_weights.reserve(final_length);
        }
        if (use_mask) {
            out_mask.reserve(final_length);
        }

        if (add_bos_token) {
            out_tokens.push_back(BOS_TOKEN_ID);
            if (use_weights) {
                out_weights.push_back(1.0f);
            }
            if (use_mask) {
                out_mask.push_back(1.0f);
            }
        }

        for (size_t i = 0; i < count; ++i) {
            out_tokens.push_back(tokens[begin + i]);
            if (use_weights) {
                out_weights.push_back((*weights)[begin + i]);
            }
            if (use_mask) {
                out_mask.push_back(1.0f);
            }
        }

        if (add_eos_token) {
            out_tokens.push_back(EOS_TOKEN_ID);
            if (use_weights) {
                out_weights.push_back(1.0f);
            }
            if (use_mask) {
                out_mask.push_back(1.0f);
            }
        }

        if (final_length > out_tokens.size()) {
            const size_t pad_count = final_length - out_tokens.size();
            if (pad_left) {
                out_tokens.insert(out_tokens.begin(), pad_count, PAD_TOKEN_ID);

                if (use_weights) {
                    out_weights.insert(out_weights.begin(), pad_count, 1.0f);
                }
                if (use_mask) {
                    out_mask.insert(out_mask.begin(), pad_count, 0.0f);
                }
            } else {
                out_tokens.insert(out_tokens.end(), pad_count, PAD_TOKEN_ID);

                if (use_weights) {
                    out_weights.insert(out_weights.end(), pad_count, 1.0f);
                }
                if (use_mask) {
                    out_mask.insert(out_mask.end(), pad_count, 0.0f);
                }
            }
        }
    };

    const size_t single_length    = std::max(min_length, tokens.size() + special_token_count);
    const bool exceeds_max_length = max_length > 0 && single_length > max_length;

    std::vector<int> new_tokens;
    std::vector<float> new_weights;
    std::vector<float> new_mask;

    if (!exceeds_max_length) {
        build_sequence(0, tokens.size(), min_length, new_tokens, new_weights, new_mask);
    } else if (!allow_overflow_expand) {
        build_sequence(0, tokens.size(), 0, new_tokens, new_weights, new_mask);

        new_tokens.resize(max_length);
        if (use_weights) {
            new_weights.resize(max_length);
        }
        if (use_mask) {
            new_mask.resize(max_length);
        }

        if (add_eos_token && !new_tokens.empty()) {
            new_tokens.back() = EOS_TOKEN_ID;
            if (use_weights) {
                new_weights.back() = 1.0f;
            }
            if (use_mask) {
                new_mask.back() = 1.0f;
            }
        }
    } else if (min_length > special_token_count) {
        const size_t tokens_per_chunk = min_length - special_token_count;
        size_t offset                 = 0;

        while (offset < tokens.size()) {
            const size_t remaining = tokens.size() - offset;
            const size_t take      = std::min(tokens_per_chunk, remaining);

            std::vector<int> chunk_tokens;
            std::vector<float> chunk_weights;
            std::vector<float> chunk_mask;

            build_sequence(offset, take, min_length, chunk_tokens, chunk_weights, chunk_mask);

            new_tokens.insert(new_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
            if (use_weights) {
                new_weights.insert(new_weights.end(), chunk_weights.begin(), chunk_weights.end());
            }
            if (use_mask) {
                new_mask.insert(new_mask.end(), chunk_mask.begin(), chunk_mask.end());
            }

            offset += take;
        }
    } else {
        build_sequence(0, tokens.size(), min_length, new_tokens, new_weights, new_mask);
    }

    tokens = std::move(new_tokens);
    if (use_weights) {
        *weights = std::move(new_weights);
    }
    if (use_mask) {
        *mask = std::move(new_mask);
    }
}

static std::string clean_up_tokenization(std::string& text) {
    std::regex pattern(R"( ,)");
    return std::regex_replace(text, pattern, ",");
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;

    for (int token_id : tokens) {
        if (token_id == BOS_TOKEN_ID || token_id == EOS_TOKEN_ID || token_id == PAD_TOKEN_ID) {
            continue;
        }

        std::string piece = decode_token(token_id);
        if (!end_of_word_suffix.empty() && ends_with(piece, end_of_word_suffix)) {
            piece.erase(piece.size() - end_of_word_suffix.size());
            text += piece + " ";
        } else {
            text += piece;
        }
    }

    text = clean_up_tokenization(text);
    return trim(text);
}
