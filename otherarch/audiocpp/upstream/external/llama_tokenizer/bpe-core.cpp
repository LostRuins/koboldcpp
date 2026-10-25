#include "bpe-core.h"

#include "unicode.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llama_tokenizer_vendor {
namespace {

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() = delete;
};

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

enum FragmentBufferVariantType {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT,
};

struct fragment_buffer_variant {
    fragment_buffer_variant(int32_t token_id)
        : type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
          token(token_id),
          raw_text(dummy_),
          offset(0),
          length(0) {}

    fragment_buffer_variant(const std::string & source_text, int64_t source_offset, int64_t source_length)
        : type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
          token(-1),
          raw_text(source_text),
          offset(source_offset),
          length(source_length) {
        if (source_offset < 0 || source_length < 1 || static_cast<uint64_t>(source_offset + source_length) > raw_text.length()) {
            throw std::runtime_error("invalid raw fragment passed to llama tokenizer");
        }
    }

    const FragmentBufferVariantType type;
    const int32_t token;
    const std::string dummy_;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

struct llm_tokenizer_bpe {
    explicit llm_tokenizer_bpe(PreTokenizerType pre_type);

    std::vector<std::string> regex_exprs;
    bool byte_encode = true;
};

struct llm_tokenizer_bpe_session {
    llm_tokenizer_bpe_session(const BpeVocabulary & vocab, const llm_tokenizer_bpe & tokenizer)
        : vocab(vocab), tokenizer(tokenizer) {}

    void tokenize(const std::string & text, std::vector<int32_t> & output);

private:
    void add_new_bigram(int left, int right);

    const BpeVocabulary & vocab;
    const llm_tokenizer_bpe & tokenizer;
    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;
    llm_bigram_bpe::queue work_queue;
};

std::string pair_key(const std::string & left, const std::string & right) {
    std::string key = left;
    key.push_back('\0');
    key += right;
    return key;
}

llm_tokenizer_bpe::llm_tokenizer_bpe(PreTokenizerType pre_type) {
    switch (pre_type) {
        case PreTokenizerType::Llama3:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Jais2:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",
            };
            break;
        case PreTokenizerType::Dbrx:
        case PreTokenizerType::Smaug:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::DeepseekLlm:
            regex_exprs = {
                "[\r\n]",
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                "\\s+$",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}+",
            };
            break;
        case PreTokenizerType::Deepseek3Llm:
        case PreTokenizerType::HunyuanDense:
        case PreTokenizerType::JoyaiLlm:
            regex_exprs = {
                "\\p{N}{1,3}",
                "[一-龥぀-ゟ゠-ヿ]+",
                "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Youtu:
            regex_exprs = {
                "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::DeepseekCoder:
            regex_exprs = {
                "[\r\n]",
                "\\s?\\p{L}+",
                "\\s?\\p{P}+",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}",
            };
            break;
        case PreTokenizerType::Falcon:
            regex_exprs = {
                "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "[0-9][0-9][0-9]",
            };
            break;
        case PreTokenizerType::Starcoder:
        case PreTokenizerType::Refact:
        case PreTokenizerType::CommandR:
        case PreTokenizerType::Smollm:
        case PreTokenizerType::Codeshell:
        case PreTokenizerType::Exaone:
        case PreTokenizerType::Minerva:
            regex_exprs = {
                "\\p{N}",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
            break;
        case PreTokenizerType::Gpt2:
        case PreTokenizerType::Mpt:
        case PreTokenizerType::Olmo:
        case PreTokenizerType::Jais:
        case PreTokenizerType::Trillion:
        case PreTokenizerType::GraniteDocling:
            regex_exprs = {
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
            break;
        case PreTokenizerType::StableLm2:
        case PreTokenizerType::Qwen2:
        case PreTokenizerType::Hunyuan:
        case PreTokenizerType::SolarOpen:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Qwen35:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Poro:
        case PreTokenizerType::Bloom:
        case PreTokenizerType::Gpt3Finnish:
            regex_exprs = {
                " ?[^(\\s|.,!?…。，、।۔،)]+",
            };
            break;
        case PreTokenizerType::Chatglm4:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Viking:
            regex_exprs = {
                " ?[^(\\s|.,!?…。，、।۔،)]+",
                "\\p{N}",
            };
            break;
        case PreTokenizerType::Tekken:
            regex_exprs = {
                "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Chameleon:
            regex_exprs = {
                "<sentinel:[0-9]+>",
                "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
                "([\\t\\n]|    |  )",
                "\\p{N}",
                "[\\p{P}!-/:-@\\[-`{-~]",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
            break;
        case PreTokenizerType::Gpt4o:
        case PreTokenizerType::MinimaxM2:
            regex_exprs = {
                "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::TinyAya:
            regex_exprs = {
                "\\d{1,3}(?=(?:\\d{3})*\\b)",
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::KimiK2:
            regex_exprs = {
                "\\p{Han}+",
            };
            break;
        case PreTokenizerType::SuperBpe:
            regex_exprs = {
                "\\p{N}+",
                "(?=(\\d{3})+(?!\\d))",
            };
            break;
        case PreTokenizerType::BailingMoe:
            regex_exprs = {
                "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::SeedCoder:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Grok2:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Afmoe:
            regex_exprs = {
                "\\p{AFMoE_digits}",
                "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
                "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::ExaoneMoe:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreTokenizerType::Gemma4:
            regex_exprs = {
                "[^\\n]+|[\\n]+",
            };
            byte_encode = false;
            break;
        case PreTokenizerType::SarvamMoe:
            regex_exprs = {
                "[^\\n]+|[\\n]+",
            };
            byte_encode = false;
            break;
        default:
            regex_exprs = {
                "[\\p{P}\\$\\+<=>\\^~\\|]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "\\p{N}+",
                "[0-9][0-9][0-9]",
            };
            break;
    }
}

void llm_tokenizer_bpe_session::add_new_bigram(int left, int right) {
    if (left == -1 || right == -1) {
        return;
    }
    std::string left_token(symbols[left].text, symbols[left].n);
    std::string right_token(symbols[right].text, symbols[right].n);

    const int rank_found = vocab.find_bpe_rank(left_token, right_token);
    if (rank_found < 0) {
        return;
    }

    llm_bigram_bpe bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.text = left_token + right_token;
    bigram.size = left_token.size() + right_token.size();
    bigram.rank = rank_found;
    work_queue.push(std::move(bigram));
}

void llm_tokenizer_bpe_session::tokenize(const std::string & text, std::vector<int32_t> & output) {
    int final_prev_index = -1;
    const auto word_collection = unicode_regex_split(text, tokenizer.regex_exprs, tokenizer.byte_encode);

    symbols_final.clear();
    for (const auto & word : word_collection) {
        work_queue = llm_bigram_bpe::queue();
        symbols.clear();

        int index = 0;
        size_t offset = 0;
        while (offset < word.size()) {
            llm_symbol sym;
            const size_t char_len = std::min(word.size() - offset, static_cast<size_t>(unicode_len_utf8(word[offset])));
            sym.text = word.c_str() + offset;
            sym.n = char_len;
            offset += sym.n;
            sym.prev = index - 1;
            sym.next = offset == word.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
            add_new_bigram(i - 1, i);
        }

        while (!work_queue.empty()) {
            auto bigram = work_queue.pop_move();

            auto & left_symbol = symbols[bigram.left];
            auto & right_symbol = symbols[bigram.right];

            if (left_symbol.n == 0 || right_symbol.n == 0) {
                continue;
            }
            const std::string left_token(left_symbol.text, left_symbol.n);
            const std::string right_token(right_symbol.text, right_symbol.n);
            if (left_token + right_token != bigram.text) {
                continue;
            }

            left_symbol.n += right_symbol.n;
            right_symbol.n = 0;
            left_symbol.next = right_symbol.next;
            if (right_symbol.next >= 0) {
                symbols[right_symbol.next].prev = bigram.left;
            }

            add_new_bigram(left_symbol.prev, bigram.left);
            add_new_bigram(bigram.left, left_symbol.next);
        }

        for (auto & sym : symbols) {
            if (sym.n > 0) {
                sym.prev = final_prev_index;
                sym.next = -1;
                if (final_prev_index != -1) {
                    symbols_final[final_prev_index].next = symbols_final.size();
                }
                symbols_final.emplace_back(sym);
                final_prev_index = static_cast<int>(symbols_final.size() - 1);
            }
        }
    }

    symbols = symbols_final;
    if (!symbols.empty()) {
        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            if (symbol.n == 0) {
                continue;
            }

            const std::string str(symbol.text, symbol.n);
            const int32_t token = vocab.text_to_token(str);
            if (token < 0) {
                for (auto j = str.begin(); j != str.end(); ++j) {
                    int32_t token_multibyte = -1;
                    if (tokenizer.byte_encode) {
                        token_multibyte = vocab.text_to_token(std::string(1, *j));
                    } else {
                        static const char * hex = "0123456789ABCDEF";
                        const uint8_t ch = static_cast<uint8_t>(*j);
                        const char buf[7] = {'<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0};
                        token_multibyte = vocab.text_to_token(buf);
                    }
                    if (token_multibyte >= 0) {
                        output.push_back(token_multibyte);
                    }
                }
            } else {
                output.push_back(token);
            }
        }
    }
}

void tokenizer_st_partition(
    const BpeVocabulary & vocab,
    std::forward_list<fragment_buffer_variant> & buffer,
    bool parse_special) {
    for (const int32_t special_id : vocab.special_tokens_sorted) {
        const auto & data = vocab.require_token_data(special_id);
        const auto & text = data.text;

        if (!parse_special && (data.attr & (TOKEN_ATTR_CONTROL | TOKEN_ATTR_UNKNOWN))) {
            continue;
        }

        auto it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = *it;
            if (fragment.type != FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                ++it;
                continue;
            }

            const auto & raw_text = fragment.raw_text;
            auto raw_text_base_offset = fragment.offset;
            auto raw_text_base_length = fragment.length;

            while (true) {
                const auto match = std::string_view(raw_text.data(), raw_text_base_offset + raw_text_base_length).find(
                    text,
                    raw_text_base_offset);
                if (match == std::string::npos) {
                    break;
                }

                const auto source = std::distance(buffer.begin(), it);
                if (match > raw_text_base_offset) {
                    const int64_t left_reminder_offset = raw_text_base_offset;
                    int64_t left_reminder_length = static_cast<int64_t>(match - raw_text_base_offset);
                    if (data.attr & TOKEN_ATTR_LSTRIP) {
                        while (left_reminder_length > 0 &&
                               std::isspace(static_cast<unsigned char>(raw_text[left_reminder_offset + left_reminder_length - 1]))) {
                            left_reminder_length--;
                        }
                    }
                    if (left_reminder_length > 0) {
                        buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                        ++it;
                    }
                }

                buffer.emplace_after(it, special_id);
                ++it;

                if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                    int64_t right_reminder_offset = static_cast<int64_t>(match + text.length());
                    int64_t right_reminder_length = static_cast<int64_t>(
                        raw_text_base_length - ((match - raw_text_base_offset) + text.length()));
                    if (data.attr & TOKEN_ATTR_RSTRIP) {
                        while (right_reminder_length > 0 &&
                               std::isspace(static_cast<unsigned char>(raw_text[right_reminder_offset]))) {
                            right_reminder_offset++;
                            right_reminder_length--;
                        }
                    }
                    if (right_reminder_length > 0) {
                        buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                        ++it;
                    }

                    if (source == 0) {
                        buffer.erase_after(buffer.before_begin());
                    } else {
                        buffer.erase_after(std::next(buffer.begin(), source - 1));
                    }
                    raw_text_base_offset = right_reminder_offset;
                    raw_text_base_length = right_reminder_length;
                } else {
                    if (source == 0) {
                        buffer.erase_after(buffer.before_begin());
                    } else {
                        buffer.erase_after(std::next(buffer.begin(), source - 1));
                    }
                    break;
                }
            }
            ++it;
        }
    }
}

}  // namespace

int32_t BpeVocabulary::text_to_token(const std::string & text) const {
    const auto it = token_to_id.find(text);
    return it == token_to_id.end() ? -1 : it->second;
}

int BpeVocabulary::find_bpe_rank(const std::string & left, const std::string & right) const {
    const auto it = bpe_ranks.find(pair_key(left, right));
    return it == bpe_ranks.end() ? -1 : it->second;
}

const TokenData & BpeVocabulary::require_token_data(int32_t token_id) const {
    const auto it = id_to_token.find(token_id);
    if (it == id_to_token.end()) {
        throw std::runtime_error("llama tokenizer saw unknown token id: " + std::to_string(token_id));
    }
    return it->second;
}

void rebuild_special_tokens_cache(BpeVocabulary & vocab) {
    vocab.special_tokens_sorted.clear();
    for (const auto & [token_id, token_data] : vocab.id_to_token) {
        if (token_data.attr & (TOKEN_ATTR_CONTROL | TOKEN_ATTR_USER_DEFINED | TOKEN_ATTR_UNKNOWN)) {
            vocab.special_tokens_sorted.push_back(token_id);
        }
    }
    std::sort(
        vocab.special_tokens_sorted.begin(),
        vocab.special_tokens_sorted.end(),
        [&vocab](const int32_t a, const int32_t b) {
            return vocab.require_token_data(a).text.size() > vocab.require_token_data(b).text.size();
        });
}

std::vector<int32_t> tokenize_bpe(const BpeVocabulary & vocab, const std::string & text, bool parse_special) {
    if (text.empty()) {
        return {};
    }
    llm_tokenizer_bpe tokenizer(vocab.pre_type);
    llm_tokenizer_bpe_session session(vocab, tokenizer);

    std::forward_list<fragment_buffer_variant> fragment_buffer;
    fragment_buffer.emplace_front(text, 0, static_cast<int64_t>(text.size()));
    tokenizer_st_partition(vocab, fragment_buffer, parse_special);

    std::vector<int32_t> output;
    for (const auto & fragment : fragment_buffer) {
        if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN) {
            output.push_back(fragment.token);
        } else {
            session.tokenize(
                fragment.raw_text.substr(static_cast<size_t>(fragment.offset), static_cast<size_t>(fragment.length)),
                output);
        }
    }
    return output;
}

std::string decode_bpe(const BpeVocabulary & vocab, const std::vector<int32_t> & token_ids, bool skip_special_tokens) {
    std::string decoded;
    for (const int32_t token_id : token_ids) {
        const auto & token = vocab.require_token_data(token_id);
        if (skip_special_tokens && (token.attr & (TOKEN_ATTR_CONTROL | TOKEN_ATTR_USER_DEFINED | TOKEN_ATTR_UNKNOWN))) {
            continue;
        }
        if (token.attr & (TOKEN_ATTR_CONTROL | TOKEN_ATTR_USER_DEFINED | TOKEN_ATTR_UNKNOWN)) {
            decoded += token.text;
            continue;
        }

        const auto cpts = unicode_cpts_from_utf8(token.text);
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            try {
                decoded.push_back(static_cast<char>(unicode_utf8_to_byte(utf8)));
            } catch (const std::out_of_range &) {
                decoded += utf8;
            }
        }
    }
    return decoded;
}

}  // namespace llama_tokenizer_vendor
