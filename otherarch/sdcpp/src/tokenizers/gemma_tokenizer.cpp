#include "gemma_tokenizer.h"

#include "core/util.h"
#include "ggml.h"
#include "json.hpp"
#include "vocab/vocab.h"

std::string GemmaTokenizer::normalize(const std::string& text) const {
    std::string normalized = text;
    size_t pos             = 0;
    while ((pos = normalized.find(' ', pos)) != std::string::npos) {
        normalized.replace(pos, 1, "\xE2\x96\x81");
        pos += 3;
    }
    return normalized;
}

void GemmaTokenizer::load_from_merges(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
    nlohmann::json vocab;
    try {
        vocab = nlohmann::json::parse(vocab_utf8_str);
    } catch (const nlohmann::json::parse_error&) {
        GGML_ABORT("invalid vocab json str");
    }
    for (const auto& [key, value] : vocab.items()) {
        std::u32string token = utf8_to_utf32(key);
        int i                = value;
        encoder[token]       = i;
        decoder[i]           = token;
    }
    encoder_len = static_cast<int>(vocab.size());
    LOG_DEBUG("vocab size: %d", encoder_len);

    std::vector<std::u32string> merges = split_utf32(merges_utf8_str);
    std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
    for (const auto& merge : merges) {
        size_t space_pos = merge.find(' ');
        merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
    }
    LOG_DEBUG("merges size %zu", merge_pairs.size());

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

GemmaTokenizer::GemmaTokenizer(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
    byte_level_bpe = false;
    byte_fallback  = true;
    add_bos_token  = true;
    pad_left       = true;
    PAD_TOKEN      = "<pad>";
    EOS_TOKEN      = "<eos>";
    BOS_TOKEN      = "<bos>";
    UNK_TOKEN      = "<unk>";

    PAD_TOKEN_ID = 0;
    EOS_TOKEN_ID = 1;
    BOS_TOKEN_ID = 2;
    UNK_TOKEN_ID = 3;

    std::vector<std::string> special_tokens_before_merge = {
        PAD_TOKEN,
        EOS_TOKEN,
        BOS_TOKEN,
        UNK_TOKEN,
        "<mask>",
        "[multimodal]",
    };
    for (int i = 0; i <= 98; i++) {
        special_tokens_before_merge.push_back("<unused" + std::to_string(i) + ">");
    }
    special_tokens_before_merge.push_back("<start_of_turn>");
    special_tokens_before_merge.push_back("<end_of_turn>");
    for (int i = 1; i <= 31; i++) {
        special_tokens_before_merge.push_back(std::string(i, '\n'));
    }
    for (int i = 2; i <= 31; i++) {
        std::string whitespace_token;
        for (int j = 0; j < i; j++) {
            whitespace_token += "\xE2\x96\x81";
        }
        special_tokens_before_merge.push_back(whitespace_token);
    }
    std::vector<std::string> html_tokens = {
        "<table>",
        "<caption>",
        "<thead>",
        "<tbody>",
        "<tfoot>",
        "<tr>",
        "<th>",
        "<td>",
        "</table>",
        "</caption>",
        "</thead>",
        "</tbody>",
        "</tfoot>",
        "</tr>",
        "</th>",
        "</td>",
        "<h1>",
        "<h2>",
        "<h3>",
        "<h4>",
        "<h5>",
        "<h6>",
        "<blockquote>",
        "</h1>",
        "</h2>",
        "</h3>",
        "</h4>",
        "</h5>",
        "</h6>",
        "</blockquote>",
        "<strong>",
        "<em>",
        "<b>",
        "<i>",
        "<u>",
        "<s>",
        "<sub>",
        "<sup>",
        "<code>",
        "</strong>",
        "</em>",
        "</b>",
        "</i>",
        "</u>",
        "</s>",
        "</sub>",
        "</sup>",
        "</code>",
        "<a>",
        "<html>",
        "<body>",
        "<img>",
        "<span>",
        "<bbox>",
        "<ul>",
        "<li>",
        "<div>",
        "<iframe>",
        "<footer>",
        "</a>",
        "</html>",
        "</body>",
        "</img>",
        "</span>",
        "</bbox>",
        "</ul>",
        "</li>",
        "</div>",
        "</iframe>",
        "</footer>",
    };
    special_tokens_before_merge.insert(special_tokens_before_merge.end(),
                                       html_tokens.begin(),
                                       html_tokens.end());
    for (int i = 0; i <= 0xFF; i++) {
        char hex_buf[16];
        snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", i);
        special_tokens_before_merge.push_back(hex_buf);
    }

    std::vector<std::string> special_tokens_after_merge = {
        "<start_of_image>",
        "<end_of_image>",
    };
    for (int i = 1; i <= 31; i++) {
        special_tokens_after_merge.insert(special_tokens_after_merge.begin() + i - 1,
                                          std::string(i, '\t'));
    }
    for (int i = 99; i <= 6241; i++) {
        special_tokens_after_merge.push_back("<unused" + std::to_string(i) + ">");
    }
    special_tokens_after_merge.push_back("<image_soft_token>");

    special_tokens = special_tokens_before_merge;
    special_tokens.insert(special_tokens.end(),
                          special_tokens_after_merge.begin(),
                          special_tokens_after_merge.end());

    if (merges_utf8_str.size() > 0 && vocab_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str, vocab_utf8_str);
    } else {
        load_from_merges(load_gemma_merges(), load_gemma_vocab_json());
    }
}

std::string Gemma2Tokenizer::normalize(const std::string& text) const {
    std::string normalized = text;
    size_t pos             = 0;
    while ((pos = normalized.find(' ', pos)) != std::string::npos) {
        normalized.replace(pos, 1, "\xE2\x96\x81");
        pos += 3;
    }
    return normalized;
}

void Gemma2Tokenizer::load_from_merges(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
    nlohmann::json vocab;
    try {
        vocab = nlohmann::json::parse(vocab_utf8_str);
    } catch (const nlohmann::json::parse_error&) {
        GGML_ABORT("invalid vocab json str");
    }
    for (const auto& [key, value] : vocab.items()) {
        std::u32string token = utf8_to_utf32(key);
        int i                = value;
        encoder[token]       = i;
        decoder[i]           = token;
    }
    encoder_len = static_cast<int>(vocab.size());
    LOG_DEBUG("vocab size: %d", encoder_len);

    std::vector<std::u32string> merges = split_utf32(merges_utf8_str);
    std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
    for (const auto& merge : merges) {
        size_t space_pos = merge.find(' ');
        merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
    }
    LOG_DEBUG("merges size %zu", merge_pairs.size());

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

Gemma2Tokenizer::Gemma2Tokenizer(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
    byte_level_bpe = false;
    byte_fallback  = true;
    add_bos_token  = true;
    PAD_TOKEN      = "<pad>";
    EOS_TOKEN      = "<eos>";
    BOS_TOKEN      = "<bos>";
    UNK_TOKEN      = "<unk>";

    PAD_TOKEN_ID = 0;
    EOS_TOKEN_ID = 1;
    BOS_TOKEN_ID = 2;
    UNK_TOKEN_ID = 3;

    std::vector<std::string> special_tokens_before_merge = {
        PAD_TOKEN,
        EOS_TOKEN,
        BOS_TOKEN,
        UNK_TOKEN,
        "<mask>",
        "<2mass>",
        "[@BOS@]",
    };
    for (int i = 0; i <= 98; i++) {
        special_tokens_before_merge.push_back("<unused" + std::to_string(i) + ">");
    }
    special_tokens_before_merge.push_back("<start_of_turn>");
    special_tokens_before_merge.push_back("<end_of_turn>");
    for (int i = 1; i <= 31; i++) {
        special_tokens_before_merge.push_back(std::string(i, '\n'));
    }
    for (int i = 2; i <= 31; i++) {
        std::string whitespace_token;
        for (int j = 0; j < i; j++) {
            whitespace_token += "\xE2\x96\x81";
        }
        special_tokens_before_merge.push_back(whitespace_token);
    }
    std::vector<std::string> html_tokens = {
        "<table>",
        "<caption>",
        "<thead>",
        "<tbody>",
        "<tfoot>",
        "<tr>",
        "<th>",
        "<td>",
        "</table>",
        "</caption>",
        "</thead>",
        "</tbody>",
        "</tfoot>",
        "</tr>",
        "</th>",
        "</td>",
        "<h1>",
        "<h2>",
        "<h3>",
        "<h4>",
        "<h5>",
        "<h6>",
        "<blockquote>",
        "</h1>",
        "</h2>",
        "</h3>",
        "</h4>",
        "</h5>",
        "</h6>",
        "</blockquote>",
        "<strong>",
        "<em>",
        "<b>",
        "<i>",
        "<u>",
        "<s>",
        "<sub>",
        "<sup>",
        "<code>",
        "</strong>",
        "</em>",
        "</b>",
        "</i>",
        "</u>",
        "</s>",
        "</sub>",
        "</sup>",
        "</code>",
    };
    special_tokens_before_merge.insert(special_tokens_before_merge.end(),
                                       html_tokens.begin(),
                                       html_tokens.end());
    for (int i = 0; i <= 0xFF; i++) {
        char hex_buf[16];
        snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", i);
        special_tokens_before_merge.push_back(hex_buf);
    }

    std::vector<std::string> special_tokens_after_merge = {
        "[toxicity=0]",
    };
    for (int i = 1; i <= 31; i++) {
        special_tokens_after_merge.insert(special_tokens_after_merge.begin() + i - 1,
                                          std::string(i, '\t'));
    }
    for (int i = 99; i <= 99; i++) {
        special_tokens_after_merge.push_back("<unused" + std::to_string(i) + ">");
    }

    special_tokens = special_tokens_before_merge;
    special_tokens.insert(special_tokens.end(),
                          special_tokens_after_merge.begin(),
                          special_tokens_after_merge.end());

    if (merges_utf8_str.size() > 0 && vocab_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str, vocab_utf8_str);
    } else {
        load_from_merges(load_gemma2_merges(), load_gemma2_vocab_json());
    }
}