#include "sdcpp_gguf_tokenizer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "gguf.h"
#include "vendor/nlohmann/json.hpp"

namespace kcpp_sd {
namespace {

using json = nlohmann::json;

struct tokenizer_metadata {
    std::string model;
    std::string pre;
    std::vector<std::string> tokens;
    std::vector<int32_t> token_types;
    std::vector<std::string> merges;
    std::vector<float> scores;  // Reserved for SentencePiece/Unigram support.
    int bos_id = -1;
    int eos_id = -1;
    int unk_id = -1;
    bool add_bos = false;
    bool add_eos = false;
    bool add_space_prefix = false;
    bool normalizer_nfc = false;
    bool normalizer_lowercase = false;
    bool normalizer_strip_accents = false;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("GGUF tokenizer metadata: " + message);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

int64_t find_key(const gguf_context* context, const char* key, bool required = true) {
    const int64_t index = gguf_find_key(context, key);
    if (required && index < 0) {
        fail(std::string("missing '") + key + "'");
    }
    return index;
}

bool read_bool(const gguf_context* context, const char* key, bool fallback = false) {
    const int64_t index = find_key(context, key, false);
    if (index < 0) {
        return fallback;
    }
    require(gguf_get_kv_type(context, index) == GGUF_TYPE_BOOL, std::string("'") + key + "' must be a boolean");
    return gguf_get_val_bool(context, index);
}

int read_id(const gguf_context* context, const char* key) {
    const int64_t index = find_key(context, key, false);
    if (index < 0) {
        return -1;
    }
    int64_t value = -1;
    switch (gguf_get_kv_type(context, index)) {
        case GGUF_TYPE_UINT32:
            value = gguf_get_val_u32(context, index);
            break;
        case GGUF_TYPE_INT32:
            value = gguf_get_val_i32(context, index);
            break;
        case GGUF_TYPE_UINT64:
            value = static_cast<int64_t>(gguf_get_val_u64(context, index));
            break;
        case GGUF_TYPE_INT64:
            value = gguf_get_val_i64(context, index);
            break;
        default:
            fail(std::string("'") + key + "' must be an integer");
    }
    require(value >= 0 && value <= INT_MAX, std::string("'") + key + "' is outside int32 range");
    return static_cast<int>(value);
}

std::string read_string(const gguf_context* context, const char* key, bool required = true) {
    const int64_t index = find_key(context, key, required);
    if (index < 0) {
        return {};
    }
    require(gguf_get_kv_type(context, index) == GGUF_TYPE_STRING, std::string("'") + key + "' must be a string");
    return gguf_get_val_str(context, index);
}

std::vector<std::string> read_string_array(const gguf_context* context, const char* key, bool required = true) {
    const int64_t index = find_key(context, key, required);
    if (index < 0) {
        return {};
    }
    require(gguf_get_kv_type(context, index) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(context, index) == GGUF_TYPE_STRING,
            std::string("'") + key + "' must be a string array");
    const size_t count = gguf_get_arr_n(context, index);
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(gguf_get_arr_str(context, index, i));
    }
    return result;
}

std::vector<int32_t> read_token_types(const gguf_context* context, size_t token_count) {
    const int64_t index = find_key(context, "tokenizer.ggml.token_type", false);
    if (index < 0) {
        return std::vector<int32_t>(token_count, 1);  // NORMAL
    }
    require(gguf_get_kv_type(context, index) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(context, index) == GGUF_TYPE_INT32,
            "'tokenizer.ggml.token_type' must be an int32 array");
    require(gguf_get_arr_n(context, index) == token_count, "token and token-type arrays have different lengths");
    const auto* values = static_cast<const int32_t*>(gguf_get_arr_data(context, index));
    return std::vector<int32_t>(values, values + token_count);
}

std::vector<float> read_scores(const gguf_context* context, size_t token_count) {
    const int64_t index = find_key(context, "tokenizer.ggml.scores", false);
    if (index < 0) {
        return {};
    }
    require(gguf_get_kv_type(context, index) == GGUF_TYPE_ARRAY,
            "'tokenizer.ggml.scores' must be an array");
    const gguf_type type = gguf_get_arr_type(context, index);
    require(type == GGUF_TYPE_FLOAT32 || type == GGUF_TYPE_INT32,
            "'tokenizer.ggml.scores' must be a float32 or int32 array");
    require(gguf_get_arr_n(context, index) == token_count, "token and score arrays have different lengths");
    std::vector<float> result(token_count);
    if (type == GGUF_TYPE_FLOAT32) {
        const auto* values = static_cast<const float*>(gguf_get_arr_data(context, index));
        result.assign(values, values + token_count);
    } else {
        const auto* values = static_cast<const int32_t*>(gguf_get_arr_data(context, index));
        for (size_t i = 0; i < token_count; ++i) {
            result[i] = static_cast<float>(values[i]);
        }
    }
    return result;
}

tokenizer_metadata read_metadata(const gguf_context* context) {
    tokenizer_metadata data;
    data.model = read_string(context, "tokenizer.ggml.model");
    data.pre = read_string(context, "tokenizer.ggml.pre", false);
    data.tokens = read_string_array(context, "tokenizer.ggml.tokens");
    data.token_types = read_token_types(context, data.tokens.size());
    data.merges = read_string_array(context, "tokenizer.ggml.merges", false);
    data.scores = read_scores(context, data.tokens.size());
    data.bos_id = read_id(context, "tokenizer.ggml.bos_token_id");
    data.eos_id = read_id(context, "tokenizer.ggml.eos_token_id");
    data.unk_id = read_id(context, "tokenizer.ggml.unknown_token_id");
    data.add_bos = read_bool(context, "tokenizer.ggml.add_bos_token");
    data.add_eos = read_bool(context, "tokenizer.ggml.add_eos_token");
    data.add_space_prefix = read_bool(context, "tokenizer.ggml.add_space_prefix");
    data.normalizer_nfc = read_bool(context, "tokenizer.ggml.normalizer.nfc");
    data.normalizer_lowercase = read_bool(context, "tokenizer.ggml.normalizer.lowercase");
    data.normalizer_strip_accents = read_bool(context, "tokenizer.ggml.normalizer.strip_accents");
    return data;
}

std::string bpe_pattern(const std::string& pre) {
    if (pre == "qwen35") {
        return R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    }
    if (pre == "qwen2" || pre == "deepseek-r1-qwen" || pre == "kormo" || pre == "f2llmv2") {
        return R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    }
    if (pre == "gpt-2" || pre == "phi-2" || pre == "jina-es" || pre == "jina-de" ||
        pre == "gigachat" || pre == "jina-v2-es" || pre == "jina-v2-de" || pre == "a.x-4.0" ||
        pre == "mellum" || pre == "modern-bert") {
        return R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
    }
    if (pre == "gpt-4o" || pre == "llama4" || pre == "kanana2" || pre == "talkie") {
        return R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    }
    if (pre == "bailingmoe" || pre == "bailingmoe2" || pre == "llada-moe") {
        return R"('(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)";
    }
    fail("unsupported tokenizer.ggml.pre '" + pre + "'");
}

void add_template_special(json& processor, const std::string& token, int id, bool before_sequence) {
    processor["special_tokens"][token] = {
        {"id", token},
        {"ids", json::array({id})},
        {"tokens", json::array({token})},
    };
    json item = {{"SpecialToken", {{"id", token}, {"type_id", 0}}}};
    if (before_sequence) {
        processor["single"].insert(processor["single"].begin(), std::move(item));
    } else {
        processor["single"].push_back(std::move(item));
    }
}

std::string serialize_bpe(tokenizer_metadata data, tokenizer_profile profile) {
    require(data.model == "gpt2", "unsupported tokenizer.ggml.model '" + data.model + "' (BPE/gpt2 expected)");
    require(!data.merges.empty(), "missing or empty 'tokenizer.ggml.merges'");
    require(!data.normalizer_strip_accents, "strip-accents normalization is unsupported");

    if (profile == tokenizer_profile::MOSS_IMAGE) {
        require(data.tokens.size() >= 248077, "token array is too small for the MOSS-Image profile");
        require(data.tokens[248044] == "<|endoftext|>" &&
                    data.tokens[248045] == "<|im_start|>" &&
                    data.tokens[248046] == "<|im_end|>",
                "base vocabulary is not compatible with MOSS-Image/Qwen3.5");
        data.pre = "qwen2";
        static const std::array<const char*, 7> moss_tokens = {
            "<|audio_start|>",
            "<|audio_end|>",
            "<tts_pad>",
            "<tts_text_bos>",
            "<tts_text_eod>",
            "<tts_text_bos_single>",
            "<|audio_pad|>",
        };
        for (size_t i = 0; i < moss_tokens.size(); ++i) {
            data.tokens[248070 + i] = moss_tokens[i];
            data.token_types[248070 + i] = 3;  // CONTROL
        }
    }

    json config = {
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens", json::array()},
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           json::array({
               {{"type", "Split"},
                {"pattern", {{"Regex", bpe_pattern(data.pre)}}},
                {"behavior", "Isolated"},
                {"invert", false}},
               {{"type", "ByteLevel"},
                {"add_prefix_space", data.add_space_prefix},
                {"trim_offsets", true},
                {"use_regex", false}},
           })}}},
        {"post_processor", nullptr},
        {"decoder", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}}},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", ""},
          {"end_of_word_suffix", ""},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", false},
          {"vocab", json::object()},
          {"merges", data.merges}}},
    };

    auto& vocab = config["model"]["vocab"];
    for (size_t id = 0; id < data.tokens.size(); ++id) {
        const auto found = vocab.find(data.tokens[id]);
        require(found == vocab.end() || found->get<size_t>() == id, "duplicate token string has conflicting IDs");
        vocab[data.tokens[id]] = id;
        const int32_t type = data.token_types[id];
        if (type == 3 || type == 4) {  // CONTROL or USER_DEFINED
            config["added_tokens"].push_back({
                {"id", id},
                {"content", data.tokens[id]},
                {"single_word", false},
                {"lstrip", false},
                {"rstrip", false},
                {"normalized", false},
                {"special", type == 3},
            });
        }
    }

    if (data.unk_id >= 0) {
        require(static_cast<size_t>(data.unk_id) < data.tokens.size(), "unknown token ID exceeds the token array");
        config["model"]["unk_token"] = data.tokens[data.unk_id];
    }

    if (data.normalizer_nfc && data.normalizer_lowercase) {
        config["normalizer"] = {{"type", "Sequence"},
                                {"normalizers", json::array({{{"type", "NFC"}}, {{"type", "Lowercase"}}})}};
    } else if (data.normalizer_nfc) {
        config["normalizer"] = {{"type", "NFC"}};
    } else if (data.normalizer_lowercase) {
        config["normalizer"] = {{"type", "Lowercase"}};
    }

    if (data.add_bos || data.add_eos) {
        json processor = {
            {"type", "TemplateProcessing"},
            {"single", json::array({{{"Sequence", {{"id", "A"}, {"type_id", 0}}}}})},
            {"pair", json::array()},
            {"special_tokens", json::object()},
        };
        if (data.add_bos) {
            require(data.bos_id >= 0 && static_cast<size_t>(data.bos_id) < data.tokens.size(), "BOS token ID exceeds the token array");
            add_template_special(processor, data.tokens[data.bos_id], data.bos_id, true);
        }
        if (data.add_eos) {
            require(data.eos_id >= 0 && static_cast<size_t>(data.eos_id) < data.tokens.size(), "EOS token ID exceeds the token array");
            add_template_special(processor, data.tokens[data.eos_id], data.eos_id, false);
        }
        config["post_processor"] = std::move(processor);
    }

    return config.dump();
}

std::string serialize_unigram(const tokenizer_metadata& data) {
    require(data.model == "t5", "unsupported tokenizer.ggml.model '" + data.model + "' (Unigram/t5 expected)");
    require(data.scores.size() == data.tokens.size(), "missing or incomplete 'tokenizer.ggml.scores'");

    json vocab = json::array();
    for (size_t id = 0; id < data.tokens.size(); ++id) {
        // T5UniGramTokenizer deliberately requires JSON floating-point scores.
        vocab.push_back(json::array({data.tokens[id], static_cast<double>(data.scores[id])}));
    }

    json config = {
        {"pre_tokenizer",
         {{"replacement", "\xE2\x96\x81"},
          {"add_prefix_space", data.add_space_prefix}}},
        {"model",
         {{"type", "Unigram"},
          {"unk_id", data.unk_id >= 0 ? data.unk_id : 3},
          {"vocab", std::move(vocab)}}},
    };
    return config.dump();
}

void serialize_vocab_and_merges(const tokenizer_metadata& data,
                                std::string& vocab_output,
                                std::string& merges_output) {
    require(data.model == "gpt2" || data.model == "gemma4" || data.model == "llama",
            "unsupported tokenizer.ggml.model '" + data.model + "' (Gemma BPE expected)");

    json vocab = json::object();
    for (size_t id = 0; id < data.tokens.size(); ++id) {
        const auto found = vocab.find(data.tokens[id]);
        require(found == vocab.end() || found->get<size_t>() == id,
                "duplicate token string has conflicting IDs");
        vocab[data.tokens[id]] = id;
    }
    vocab_output = vocab.dump();

    merges_output.clear();
    if (!data.merges.empty()) {
        for (const std::string& merge : data.merges) {
            require(merge.find('\n') == std::string::npos && merge.find('\r') == std::string::npos,
                    "merge contains a line break");
            require(merge.find(' ') != std::string::npos, "merge does not contain a token separator");
            merges_output += merge;
            merges_output += '\n';
        }
        return;
    }

    // Gemma 3 GGUFs converted from tokenizer.model use SentencePiece's
    // `llama` representation. SentencePiece stores the BPE rank as each
    // piece's score rather than an explicit merge array. This is the same
    // slow-to-fast reconstruction used to produce the HF tokenizer merges:
    // visit normal pieces by descending score and emit every valid split,
    // rightmost split first.
    require(data.model == "llama", "missing or empty 'tokenizer.ggml.merges'");
    require(data.scores.size() == data.tokens.size(), "missing or incomplete 'tokenizer.ggml.scores'");

    std::unordered_set<std::string> token_set(data.tokens.begin(), data.tokens.end());
    std::vector<size_t> piece_ids(data.tokens.size());
    std::iota(piece_ids.begin(), piece_ids.end(), 0);
    std::stable_sort(piece_ids.begin(), piece_ids.end(), [&](size_t left, size_t right) {
        return data.scores[left] > data.scores[right];
    });

    size_t merge_count = 0;
    for (size_t id : piece_ids) {
        if (data.token_types[id] != 1) {  // NORMAL
            continue;
        }
        const std::string& piece = data.tokens[id];
        std::vector<size_t> split_offsets;
        for (size_t offset = 1; offset < piece.size(); ++offset) {
            const unsigned char byte = static_cast<unsigned char>(piece[offset]);
            if ((byte & 0xC0) != 0x80) {
                split_offsets.push_back(offset);
            }
        }
        for (auto split = split_offsets.rbegin(); split != split_offsets.rend(); ++split) {
            const std::string left = piece.substr(0, *split);
            const std::string right = piece.substr(*split);
            if (token_set.find(left) == token_set.end() || token_set.find(right) == token_set.end()) {
                continue;
            }
            require(left.find_first_of("\r\n") == std::string::npos &&
                        right.find_first_of("\r\n") == std::string::npos,
                    "reconstructed merge contains a line break");
            require(left.find(' ') == std::string::npos,
                    "reconstructed left-hand token contains a literal space");
            merges_output += left;
            merges_output += ' ';
            merges_output += right;
            merges_output += '\n';
            ++merge_count;
        }
    }
    require(merge_count > 0, "could not reconstruct merges from SentencePiece scores");
}

template <typename Serializer>
bool extract(const std::string& gguf_path, std::string& error, Serializer&& serializer) {
    error.clear();
    try {
        gguf_context* context = gguf_init_from_file(gguf_path.c_str(), {/*.no_alloc =*/ true, /*.ctx =*/ nullptr});
        require(context != nullptr, "cannot open '" + gguf_path + "'");
        std::unique_ptr<gguf_context, decltype(&gguf_free)> owner(context, &gguf_free);
        serializer(context);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace

bool gguf_tokenizer_extractor::to_hf_json(const std::string& gguf_path,
                                          std::string& output,
                                          std::string& error,
                                          tokenizer_profile profile) {
    output.clear();
    return extract(gguf_path, error, [&](const gguf_context* context) {
        const int64_t hf_json_index = find_key(context, "tokenizer.huggingface.json", false);
        if (hf_json_index >= 0 && profile == tokenizer_profile::DEFAULT) {
            require(gguf_get_kv_type(context, hf_json_index) == GGUF_TYPE_STRING,
                    "'tokenizer.huggingface.json' must be a string");
            output = gguf_get_val_str(context, hf_json_index);
            return;
        }

        const tokenizer_metadata data = read_metadata(context);
        if (data.model == "gpt2") {
            output = serialize_bpe(data, profile);
            return;
        }
        fail("unsupported tokenizer.ggml.model '" + data.model + "'; currently supported: gpt2/BPE");
    });
}

bool gguf_tokenizer_extractor::to_unigram_json(const std::string& gguf_path,
                                                std::string& output,
                                                std::string& error) {
    output.clear();
    return extract(gguf_path, error, [&](const gguf_context* context) {
        output = serialize_unigram(read_metadata(context));
    });
}

bool gguf_tokenizer_extractor::to_vocab_and_merges(const std::string& gguf_path,
                                                    std::string& vocab_output,
                                                    std::string& merges_output,
                                                    std::string& error) {
    vocab_output.clear();
    merges_output.clear();
    return extract(gguf_path, error, [&](const gguf_context* context) {
        serialize_vocab_and_merges(read_metadata(context), vocab_output, merges_output);
    });
}

}  // namespace kcpp_sd
