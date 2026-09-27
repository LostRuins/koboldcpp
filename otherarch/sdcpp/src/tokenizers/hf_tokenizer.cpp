#include "hf_tokenizer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "core/regex.h"
#include "core/util.h"
#include "json.hpp"
#include "utf8proc.h"

using TokenizerJSON = nlohmann::json;

static void tokenizer_require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("tokenizer.json: " + message);
    }
}

static std::string tokenizer_utf8(int32_t codepoint) {
    utf8proc_uint8_t bytes[4];
    auto count = utf8proc_encode_char(codepoint, bytes);
    return std::string(reinterpret_cast<const char*>(bytes), count);
}

static bool tokenizer_error(std::string* error, const std::string& message) {
    if (error) {
        *error = "tokenizer.json: " + message;
    } else {
        LOG_ERROR("tokenizer.json: %s", message.c_str());
    }
    return false;
}

static bool tokenizer_next(const std::string& text, size_t& offset, int32_t& codepoint, std::string* error) {
    auto size = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset), text.size() - offset, &codepoint);
    if (size <= 0) {
        return tokenizer_error(error, "invalid UTF-8 input");
    }
    offset += size;
    return true;
}

static int tokenizer_id(const TokenizerJSON& value) {
    tokenizer_require(value.is_number_integer(), "token ID must be an integer");
    auto id = value.get<int64_t>();
    tokenizer_require(id >= 0 && id <= INT_MAX, "token ID outside int32 range");
    return static_cast<int>(id);
}

static uint64_t tokenizer_pair(int left, int right) {
    return (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);
}

struct HFTokenizer::Impl {
    struct Pattern {
        std::string literal;
        std::shared_ptr<sd::Regex> regex;

        explicit Pattern(const TokenizerJSON& config) {
            tokenizer_require(config.is_object() && config.size() == 1, "invalid String/Regex pattern");
            if (config.contains("String")) {
                literal = config.at("String").get<std::string>();
            } else {
                tokenizer_require(config.contains("Regex"), "unsupported pattern");
                regex = std::make_shared<sd::Regex>();
                std::string error;
                bool ok = regex->compile(config.at("Regex").get<std::string>(), &error);
                tokenizer_require(ok, "invalid regex: " + error);
            }
        }

        bool matches(const std::string& text, std::vector<sd::Regex::Match>& result, std::string* error) const {
            result.clear();
            if (regex) {
                std::string regex_error;
                if (!regex->find_matches(text, result, &regex_error)) {
                    return tokenizer_error(error, "regex search failed: " + regex_error);
                }
            } else if (literal.empty()) {
                size_t offset = 0;
                for (;;) {
                    result.emplace_back(offset, offset);
                    if (offset == text.size()) {
                        break;
                    }
                    int32_t cp;
                    if (!tokenizer_next(text, offset, cp, error)) {
                        result.clear();
                        return false;
                    }
                }
            } else {
                size_t offset = 0;
                while ((offset = text.find(literal, offset)) != std::string::npos) {
                    result.emplace_back(offset, offset + literal.size());
                    offset += literal.size();
                }
            }
            return true;
        }

        bool replace(const std::string& text, const std::string& replacement, std::string& result, std::string* error) const {
            result.clear();
            size_t offset = 0;
            std::vector<sd::Regex::Match> found;
            if (!matches(text, found, error)) {
                return false;
            }
            for (const auto& match : found) {
                result.append(text, offset, match.first - offset);
                result += replacement;
                offset = match.second;
            }
            result.append(text, offset, std::string::npos);
            return true;
        }

        bool split(const std::string& text, const std::string& behavior, bool invert, std::vector<std::string>& result, std::string* error) const {
            result.clear();
            struct Part {
                size_t start, end;
                bool matched;
            };
            std::vector<Part> parts;
            size_t offset = 0;
            std::vector<sd::Regex::Match> found;
            if (!matches(text, found, error)) {
                return false;
            }
            for (const auto& match : found) {
                if (match.first > offset) {
                    parts.push_back({offset, match.first, invert});
                }
                parts.push_back({match.first, match.second, !invert});
                offset = match.second;
            }
            if (offset < text.size()) {
                parts.push_back({offset, text.size(), invert});
            }
            if (behavior == "MergedWithNext") {
                std::reverse(parts.begin(), parts.end());
            }
            std::vector<Part> merged;
            bool previous = false;
            for (const auto& part : parts) {
                bool join = (behavior == "Contiguous" && part.matched == previous) ||
                            ((behavior == "MergedWithPrevious" || behavior == "MergedWithNext") && part.matched && !previous);
                if (join && !merged.empty()) {
                    merged.back().start = std::min(merged.back().start, part.start);
                    merged.back().end   = std::max(merged.back().end, part.end);
                } else if (behavior != "Removed" || !part.matched) {
                    merged.push_back(part);
                }
                previous = part.matched;
            }
            if (behavior == "MergedWithNext") {
                std::reverse(merged.begin(), merged.end());
            }
            for (const auto& part : merged) {
                if (part.start != part.end) {
                    result.push_back(text.substr(part.start, part.end - part.start));
                }
            }
            return true;
        }
    };

    struct Step {
        std::string type, content, behavior;
        std::shared_ptr<Pattern> pattern;
        bool invert = false, prefix_space = false;
    };

    struct Trie {
        struct Node {
            std::unordered_map<unsigned char, size_t> children;
            int id = -1;
        };
        std::vector<Node> nodes{1};

        void add(const std::string& text, int id) {
            size_t index = 0;
            for (unsigned char c : text) {
                auto found = nodes[index].children.find(c);
                if (found == nodes[index].children.end()) {
                    size_t next = nodes.size();
                    nodes[index].children.emplace(c, next);
                    nodes.emplace_back();
                    index = next;
                } else {
                    index = found->second;
                }
            }
            nodes[index].id = id;
        }

        std::pair<size_t, int> match(const std::string& text, size_t start) const {
            size_t index = 0;
            std::pair<size_t, int> result{start, -1};
            for (size_t end = start; end < text.size(); ++end) {
                auto found = nodes[index].children.find(static_cast<unsigned char>(text[end]));
                if (found == nodes[index].children.end()) {
                    break;
                }
                index = found->second;
                if (nodes[index].id >= 0) {
                    result = {end + 1, nodes[index].id};
                }
            }
            return result;
        }
    };

    struct Merge {
        size_t rank;
        int id;
    };
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, int> added_vocab;
    std::unordered_map<int, std::string> tokens;
    std::unordered_map<uint64_t, Merge> merges;
    std::unordered_set<int> special_ids;
    std::vector<std::string> custom_tokens;
    std::vector<Step> normalizers, pre_tokenizers, decoders;
    Trie raw_added, normalized_added;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<int32_t, unsigned char> byte_decoder;
    std::string suffix;
    int unk       = -1;
    bool fuse_unk = false, byte_fallback = false, ignore_merges = false, has_decoder = false;

    Impl() {
        int extra = 256;
        for (int byte = 0; byte < 256; ++byte) {
            int cp             = ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174) ? byte : extra++;
            byte_encoder[byte] = tokenizer_utf8(cp);
            byte_decoder[cp]   = static_cast<unsigned char>(byte);
        }
    }

    static void parse_steps(const TokenizerJSON& config, const std::string& stage, std::vector<Step>& out, int depth = 0) {
        tokenizer_require(depth < 32, stage + " nesting is too deep");
        if (config.is_null()) {
            return;
        }
        Step step;
        step.type = config.at("type").get<std::string>();
        if (step.type == "Sequence") {
            const char* key = stage == "normalizer" ? "normalizers" : stage == "pre_tokenizer" ? "pretokenizers"
                                                                                               : "decoders";
            for (const auto& child : config.at(key)) {
                parse_steps(child, stage, out, depth + 1);
            }
            return;
        }
        if ((stage == "normalizer" || stage == "decoder") && step.type == "Replace") {
            step.pattern = std::make_shared<Pattern>(config.at("pattern"));
            step.content = config.at("content").get<std::string>();
        } else if (stage == "normalizer" && (step.type == "NFC" || step.type == "Lowercase")) {
        } else if (stage == "pre_tokenizer" && step.type == "Split") {
            step.pattern  = std::make_shared<Pattern>(config.at("pattern"));
            step.behavior = config.at("behavior").get<std::string>();
            tokenizer_require(step.behavior == "Removed" || step.behavior == "Isolated" || step.behavior == "Contiguous" || step.behavior == "MergedWithPrevious" || step.behavior == "MergedWithNext", "unsupported Split behavior: " + step.behavior);
            step.invert = config.value("invert", false);
        } else if ((stage == "pre_tokenizer" || stage == "decoder") && step.type == "ByteLevel") {
            step.prefix_space = config.value("add_prefix_space", true);
            if (stage == "pre_tokenizer" && config.value("use_regex", true)) {
                step.pattern = std::make_shared<Pattern>(TokenizerJSON{{"Regex", R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"}});
            }
        } else if (stage == "decoder" && (step.type == "ByteFallback" || step.type == "Fuse")) {
        } else {
            tokenizer_require(false, "unsupported " + stage + ": " + step.type);
        }
        out.push_back(std::move(step));
    }

    bool normalize(std::string text, std::string& result, std::string* error) const {
        result.clear();
        for (const auto& step : normalizers) {
            if (step.type == "Replace") {
                std::string replaced;
                if (!step.pattern->replace(text, step.content, replaced, error)) {
                    return false;
                }
                text = std::move(replaced);
            } else if (step.type == "NFC") {
                utf8proc_uint8_t* output = nullptr;
                auto size                = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()), text.size(), &output, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
                std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> buffer(output, &std::free);
                if (size < 0) {
                    return tokenizer_error(error, std::string("NFC normalization failed: ") + utf8proc_errmsg(size));
                }
                text.assign(reinterpret_cast<const char*>(output), size);
            } else {
                std::string lower;
                for (size_t i = 0; i < text.size();) {
                    int32_t cp;
                    if (!tokenizer_next(text, i, cp, error)) {
                        return false;
                    }
                    // Rust char::to_lowercase uses full, context-free lowercase. U+0130 expands.
                    lower += cp == 0x130 ? "i\xcc\x87" : tokenizer_utf8(utf8proc_tolower(cp));
                }
                text = std::move(lower);
            }
        }
        result = std::move(text);
        return true;
    }

    bool pre_tokenize(const std::string& text, std::vector<std::string>& result, std::string* error) const {
        result.clear();
        std::vector<std::string> pieces{text};
        for (const auto& step : pre_tokenizers) {
            std::vector<std::string> next;
            for (auto piece : pieces) {
                if (piece.empty()) {
                    continue;
                }
                std::vector<std::string> split;
                if (step.type == "Split") {
                    if (!step.pattern->split(piece, step.behavior, step.invert, split, error)) {
                        return false;
                    }
                    next.insert(next.end(), split.begin(), split.end());
                } else {
                    if (step.prefix_space && piece.front() != ' ') {
                        piece.insert(piece.begin(), ' ');
                    }
                    if (step.pattern) {
                        if (!step.pattern->split(piece, "Isolated", false, split, error)) {
                            return false;
                        }
                    } else {
                        split.push_back(piece);
                    }
                    for (const auto& part : split) {
                        std::string encoded;
                        for (unsigned char byte : part) {
                            encoded += byte_encoder[byte];
                        }
                        next.push_back(std::move(encoded));
                    }
                }
            }
            pieces = std::move(next);
        }
        result = std::move(pieces);
        return true;
    }

    bool bpe(const std::string& text, std::vector<int>& ids, std::string* error) const {
        ids.clear();
        if (ignore_merges) {
            auto found = vocab.find(text);
            if (found != vocab.end()) {
                ids.push_back(found->second);
                return true;
            }
        }
        bool pending_unk = false;
        for (size_t i = 0; i < text.size();) {
            int32_t cp;
            size_t end = i;
            if (!tokenizer_next(text, end, cp, error)) {
                ids.clear();
                return false;
            }
            std::string symbol = text.substr(i, end - i);
            if (end == text.size()) {
                symbol += suffix;
            }
            i          = end;
            auto found = vocab.find(symbol);
            if (found != vocab.end()) {
                if (pending_unk) {
                    ids.push_back(unk);
                    pending_unk = false;
                }
                ids.push_back(found->second);
                continue;
            }
            if (byte_fallback) {
                std::vector<int> bytes;
                for (unsigned char byte : symbol) {
                    const char* hex   = "0123456789ABCDEF";
                    std::string token = "<0x00>";
                    token[3]          = hex[byte >> 4];
                    token[4]          = hex[byte & 15];
                    auto fallback     = vocab.find(token);
                    if (fallback == vocab.end()) {
                        break;
                    }
                    bytes.push_back(fallback->second);
                }
                if (bytes.size() == symbol.size()) {
                    ids.insert(ids.end(), bytes.begin(), bytes.end());
                    continue;
                }
            }
            if (unk >= 0) {
                if (pending_unk && !fuse_unk) {
                    ids.push_back(unk);
                }
                pending_unk = true;
            }
        }
        if (pending_unk) {
            ids.push_back(unk);
        }
        struct Symbol {
            int id;
            size_t prev, next, generation = 0;
            bool alive = true;
        };
        struct Candidate {
            size_t rank, left, right, left_generation, right_generation;
            int id;
            bool operator<(const Candidate& other) const {
                return rank != other.rank ? rank > other.rank : left > other.left;
            }
        };
        const size_t none = ids.size();
        std::vector<Symbol> symbols;
        for (size_t i = 0; i < ids.size(); ++i) {
            symbols.push_back({ids[i], i == 0 ? none : i - 1, i + 1});
        }
        std::priority_queue<Candidate> queue;
        auto push = [&](size_t left) {
            if (left == none || symbols[left].next == none) {
                return;
            }
            size_t right = symbols[left].next;
            auto found   = merges.find(tokenizer_pair(symbols[left].id, symbols[right].id));
            if (found != merges.end()) {
                queue.push({found->second.rank, left, right, symbols[left].generation, symbols[right].generation, found->second.id});
            }
        };
        for (size_t i = 0; i < symbols.size(); ++i) {
            push(i);
        }
        while (!queue.empty()) {
            Candidate item = queue.top();
            queue.pop();
            auto& left  = symbols[item.left];
            auto& right = symbols[item.right];
            if (!left.alive || !right.alive || left.next != item.right || left.generation != item.left_generation || right.generation != item.right_generation) {
                continue;
            }
            left.id   = item.id;
            left.next = right.next;
            ++left.generation;
            right.alive = false;
            if (left.next != none) {
                symbols[left.next].prev = item.left;
            }
            push(left.prev);
            push(item.left);
        }
        ids.clear();
        for (const auto& symbol : symbols) {
            if (symbol.alive) {
                ids.push_back(symbol.id);
            }
        }
        return true;
    }

    int lookup(const std::string& token) const {
        auto added = added_vocab.find(token);
        if (added != added_vocab.end()) {
            return added->second;
        }
        auto found = vocab.find(token);
        return found == vocab.end() ? -1 : found->second;
    }

    void add_token(const std::string& token, int id, bool added = false) {
        auto old = tokens.find(id);
        tokenizer_require(old == tokens.end() || old->second == token, "conflicting token ID " + std::to_string(id));
        int old_id = lookup(token);
        tokenizer_require(old_id < 0 || old_id == id, "conflicting ID for token " + token);
        tokens[id]                           = token;
        (added ? added_vocab : vocab)[token] = id;
    }
};

HFTokenizer::HFTokenizer(const std::string& path)
    : impl_(new Impl) {
    std::ifstream stream(path, std::ios::binary);
    tokenizer_require(stream.good(), "cannot open " + path);
    TokenizerJSON config;
    stream >> config;
    tokenizer_require(config.value("version", std::string("1.0")) == "1.0", "unsupported version");
    tokenizer_require(config.value("padding", TokenizerJSON()).is_null(), "JSON padding is unsupported; padding is controlled by the text encoder");
    tokenizer_require(config.value("truncation", TokenizerJSON()).is_null(), "JSON truncation is unsupported; truncation is controlled by the text encoder");
    const auto& model = config.at("model");
    tokenizer_require(model.at("type") == "BPE", "only BPE models are supported");
    tokenizer_require(model.value("dropout", TokenizerJSON()).is_null() || model.at("dropout") == 0, "BPE dropout is unsupported");
    const auto& prefix = model.value("continuing_subword_prefix", TokenizerJSON());
    tokenizer_require(prefix.is_null() || prefix == "", "nonempty continuing_subword_prefix is unsupported");
    const auto& suffix   = model.value("end_of_word_suffix", TokenizerJSON());
    impl_->suffix        = suffix.is_null() ? "" : suffix.get<std::string>();
    impl_->fuse_unk      = model.value("fuse_unk", false);
    impl_->byte_fallback = model.value("byte_fallback", false);
    impl_->ignore_merges = model.value("ignore_merges", false);
    tokenizer_require(model.at("vocab").is_object(), "BPE vocab must be an object");
    impl_->vocab.reserve(model.at("vocab").size());
    impl_->tokens.reserve(model.at("vocab").size());
    for (const auto& entry : model.at("vocab").items()) {
        impl_->add_token(entry.key(), tokenizer_id(entry.value()));
    }
    if (!model.value("unk_token", TokenizerJSON()).is_null()) {
        UNK_TOKEN  = model.at("unk_token").get<std::string>();
        impl_->unk = impl_->lookup(UNK_TOKEN);
        tokenizer_require(impl_->unk >= 0, "unk_token is absent from vocab");
    }
    UNK_TOKEN_ID = impl_->unk;
    tokenizer_require(model.at("merges").is_array(), "BPE merges must be an array");
    impl_->merges.reserve(model.at("merges").size());
    size_t rank = 0;
    for (const auto& merge : model.at("merges")) {
        std::string left, right;
        if (merge.is_string()) {
            auto value = merge.get<std::string>();
            auto space = value.find(' ');
            tokenizer_require(space != std::string::npos && value.find(' ', space + 1) == std::string::npos, "invalid legacy BPE merge");
            left  = value.substr(0, space);
            right = value.substr(space + 1);
        } else {
            tokenizer_require(merge.is_array() && merge.size() == 2, "BPE merge must contain two tokens");
            left  = merge.at(0).get<std::string>();
            right = merge.at(1).get<std::string>();
        }
        int a = impl_->lookup(left), b = impl_->lookup(right), id = impl_->lookup(left + right);
        tokenizer_require(a >= 0 && b >= 0 && id >= 0, "BPE merge references a missing vocab token");
        impl_->merges[tokenizer_pair(a, b)] = {rank++, id};
    }
    Impl::parse_steps(config.value("normalizer", TokenizerJSON()), "normalizer", impl_->normalizers);
    Impl::parse_steps(config.value("pre_tokenizer", TokenizerJSON()), "pre_tokenizer", impl_->pre_tokenizers);
    impl_->has_decoder = !config.value("decoder", TokenizerJSON()).is_null();
    Impl::parse_steps(config.value("decoder", TokenizerJSON()), "decoder", impl_->decoders);
    size_t next_added_id = impl_->vocab.size();
    for (const auto& token : config.value("added_tokens", TokenizerJSON::array())) {
        for (const char* flag : {"single_word", "lstrip", "rstrip"}) {
            tokenizer_require(!token.value(flag, false), std::string("added_tokens.") + flag + "=true is unsupported");
        }
        auto content = token.at("content").get<std::string>();
        tokenizer_require(!content.empty(), "empty added token is unsupported");
        int id = tokenizer_id(token.at("id"));
        if (impl_->lookup(content) < 0) {
            tokenizer_require(static_cast<size_t>(id) == next_added_id++, "nonconsecutive added token IDs would be reassigned by Hugging Face tokenizers");
        }
        impl_->add_token(content, id, true);
        if (token.value("special", false)) {
            special_tokens.push_back(content);
            impl_->special_ids.insert(id);
        }
        bool normalized     = token.value("normalized", true);
        std::string pattern = content;
        if (normalized) {
            std::string error;
            bool ok = impl_->normalize(content, pattern, &error);
            tokenizer_require(ok, error);
        }
        tokenizer_require(!pattern.empty(), "added token normalizes to an empty string");
        (normalized ? impl_->normalized_added : impl_->raw_added).add(pattern, id);
    }
    const auto& processor = config.value("post_processor", TokenizerJSON());
    BOS_TOKEN_ID = EOS_TOKEN_ID = -1;
    if (!processor.is_null()) {
        auto type    = processor.at("type").get<std::string>();
        auto special = [&](const TokenizerJSON& pair) {
            tokenizer_require(pair.is_array() && pair.size() == 2, "invalid postprocessor special token");
            int id = tokenizer_id(pair.at(1));
            tokenizer_require(impl_->lookup(pair.at(0).get<std::string>()) == id, "postprocessor token/ID does not match vocab");
            return id;
        };
        if (type == "RobertaProcessing") {
            BOS_TOKEN_ID = special(processor.at("cls"));
            EOS_TOKEN_ID = special(processor.at("sep"));
        } else if (type == "TemplateProcessing") {
            bool seen_sequence = false;
            for (const auto& item : processor.at("single")) {
                if (item.contains("Sequence")) {
                    tokenizer_require(!seen_sequence && item.at("Sequence").at("id") == "A", "single template must contain exactly one sequence A");
                    seen_sequence = true;
                } else {
                    auto name         = item.at("SpecialToken").at("id").get<std::string>();
                    const auto& token = processor.at("special_tokens").at(name);
                    tokenizer_require(token.at("ids").size() == 1 && token.at("tokens").size() == 1, "multi-ID template special tokens are unsupported");
                    int id      = special(TokenizerJSON::array({token.at("tokens").at(0), token.at("ids").at(0)}));
                    int& target = seen_sequence ? EOS_TOKEN_ID : BOS_TOKEN_ID;
                    tokenizer_require(target < 0, "single template supports at most one prefix and one suffix token");
                    target = id;
                }
            }
            tokenizer_require(seen_sequence, "single template has no sequence A");
        } else {
            tokenizer_require(type == "ByteLevel", "unsupported post_processor: " + type);
        }
    }
    add_bos_token = BOS_TOKEN_ID >= 0;
    add_eos_token = EOS_TOKEN_ID >= 0;
    BOS_TOKEN     = decode_token(BOS_TOKEN_ID);
    EOS_TOKEN     = decode_token(EOS_TOKEN_ID);
    set_padding(0, false);
}

HFTokenizer::~HFTokenizer() = default;

void HFTokenizer::set_padding(int token_id, bool left) {
    PAD_TOKEN_ID = token_id;
    PAD_TOKEN    = decode_token(token_id);
    pad_left     = left;
}

void HFTokenizer::validate_vocab_size(int64_t embedding_rows) const {
    tokenizer_require(embedding_rows > 0, "text encoder has no token embedding rows");
    for (const auto& token : impl_->tokens) {
        tokenizer_require(token.first < embedding_rows, "token ID " + std::to_string(token.first) + " exceeds text encoder vocabulary (" + std::to_string(embedding_rows) + ")");
    }
    tokenizer_require(PAD_TOKEN_ID >= 0 && PAD_TOKEN_ID < embedding_rows, "padding ID exceeds text encoder vocabulary");
}

int HFTokenizer::token_to_id(const std::string& token) const {
    return impl_->lookup(token);
}

void HFTokenizer::add_special_token(const std::string& token) {
    Tokenizer::add_special_token(token);
    if (!token.empty()) {
        impl_->custom_tokens.push_back(token);
    }
}

bool HFTokenizer::encode(const std::string& text, std::vector<int>& tokens, on_new_token_cb_t callback, std::string* error) {
    tokens.clear();
    if (error) {
        error->clear();
    }
    for (size_t i = 0; i < text.size();) {
        int32_t cp;
        if (!tokenizer_next(text, i, cp, error)) {
            return false;
        }
    }
    std::vector<int> result;
    Impl::Trie raw_custom, normalized_custom;
    if (callback) {
        for (size_t index = 0; index < impl_->custom_tokens.size(); ++index) {
            const auto& token = impl_->custom_tokens[index];
            raw_custom.add(token, static_cast<int>(index));
            std::string normalized;
            if (!impl_->normalize(token, normalized, error)) {
                return false;
            }
            if (!normalized.empty()) {
                normalized_custom.add(normalized, static_cast<int>(index));
            }
        }
    }
    auto encode_plain = [&](const std::string& value) {
        std::vector<std::string> pieces;
        if (!impl_->pre_tokenize(value, pieces, error)) {
            return false;
        }
        for (auto& piece : pieces) {
            if (callback && callback(piece, result)) {
                continue;
            }
            std::vector<int> ids;
            if (!impl_->bpe(piece, ids, error)) {
                return false;
            }
            result.insert(result.end(), ids.begin(), ids.end());
        }
        return true;
    };
    auto extract = [&](const std::string& value, const Impl::Trie& added, const Impl::Trie& custom_tokens, const auto& encode_gap) {
        size_t start = 0, i = 0;
        while (i < value.size()) {
            auto match      = added.match(value, i);
            auto custom     = custom_tokens.match(value, i);
            bool use_custom = custom.second >= 0 && custom.first >= match.first;
            if (match.second < 0 && !use_custom) {
                ++i;
                continue;
            }
            if (!encode_gap(value.substr(start, i - start))) {
                return false;
            }
            if (use_custom) {
                auto token = impl_->custom_tokens[custom.second];
                if (!callback(token, result)) {
                    if (match.second >= 0 && match.first == custom.first) {
                        result.push_back(match.second);
                    } else if (!encode_gap(value.substr(i, custom.first - i))) {
                        return false;
                    }
                }
            } else {
                result.push_back(match.second);
            }
            start = i = use_custom ? custom.first : match.first;
        }
        return encode_gap(value.substr(start));
    };
    auto encode_normalized = [&](const std::string& value) {
        std::string normalized;
        if (!impl_->normalize(value, normalized, error)) {
            return false;
        }
        return extract(normalized, impl_->normalized_added, normalized_custom, encode_plain);
    };
    if (!extract(text, impl_->raw_added, raw_custom, encode_normalized)) {
        return false;
    }
    std::stringstream ss;
    ss << "[";
    for (int id : result) {
        auto token = impl_->tokens.find(id);
        if (token != impl_->tokens.end()) {
            ss << "\"" << token->second << "\", ";
        } else {
            ss << "\"<id:" << id << ">\", ";
        }
    }
    ss << "]";
    LOG_VERBOSE("split prompt \"%s\" to %zu tokens %s", text.c_str(), result.size(), ss.str().c_str());
    tokens = std::move(result);
    return true;
}

std::string HFTokenizer::decode_token(int id) const {
    auto found = impl_->tokens.find(id);
    return found == impl_->tokens.end() ? "" : found->second;
}

static std::string tokenizer_lossy_utf8(const std::string& bytes, bool fallback) {
    std::string result;
    for (size_t i = 0; i < bytes.size();) {
        int32_t cp;
        auto count = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(bytes.data() + i), bytes.size() - i, &cp);
        if (count > 0) {
            result.append(bytes, i, count);
            i += count;
        } else if (fallback) {
            result.clear();
            for (size_t j = 0; j < bytes.size(); ++j) {
                result += "\xef\xbf\xbd";
            }
            return result;
        } else {
            result += "\xef\xbf\xbd";
            unsigned char lead = bytes[i++];
            size_t expected    = lead >= 0xc2 && lead <= 0xdf ? 2 : lead >= 0xe0 && lead <= 0xef ? 3
                                                                : lead >= 0xf0 && lead <= 0xf4   ? 4
                                                                                                 : 1;
            for (size_t j = 1; j < expected && i < bytes.size(); ++j) {
                unsigned char c = bytes[i];
                if (c < 0x80 || c > 0xbf || (j == 1 && ((lead == 0xe0 && c < 0xa0) || (lead == 0xed && c > 0x9f) || (lead == 0xf0 && c < 0x90) || (lead == 0xf4 && c > 0x8f)))) {
                    break;
                }
                ++i;
            }
        }
    }
    return result;
}

bool HFTokenizer::decode(const std::vector<int>& ids, std::string& text, std::string* error) const {
    text.clear();
    if (error) {
        error->clear();
    }
    std::vector<std::string> pieces;
    for (int id : ids) {
        if (!impl_->special_ids.count(id) && impl_->tokens.count(id)) {
            pieces.push_back(decode_token(id));
        }
    }
    for (const auto& step : impl_->decoders) {
        if (step.type == "Replace") {
            for (auto& piece : pieces) {
                std::string replaced;
                if (!step.pattern->replace(piece, step.content, replaced, error)) {
                    return false;
                }
                piece = std::move(replaced);
            }
        } else if (step.type == "ByteLevel" || step.type == "Fuse") {
            std::string joined;
            for (const auto& piece : pieces) {
                std::string bytes;
                if (step.type == "ByteLevel") {
                    for (size_t i = 0; i < piece.size();) {
                        int32_t cp;
                        if (!tokenizer_next(piece, i, cp, error)) {
                            return false;
                        }
                        auto found = impl_->byte_decoder.find(cp);
                        if (found == impl_->byte_decoder.end()) {
                            bytes = piece;
                            break;
                        }
                        bytes += static_cast<char>(found->second);
                    }
                } else {
                    bytes = piece;
                }
                joined += bytes;
            }
            pieces = {step.type == "ByteLevel" ? tokenizer_lossy_utf8(joined, false) : joined};
        } else {
            std::vector<std::string> decoded;
            std::string bytes;
            auto flush = [&] {
                if (!bytes.empty()) {
                    decoded.push_back(tokenizer_lossy_utf8(bytes, true));
                    bytes.clear();
                }
            };
            for (const auto& piece : pieces) {
                auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                                            : c >= 'a' && c <= 'f'   ? c - 'a' + 10
                                                                                                     : -1; };
                if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>' && hex(piece[3]) >= 0 && hex(piece[4]) >= 0) {
                    bytes += static_cast<char>((hex(piece[3]) << 4) | hex(piece[4]));
                } else {
                    flush();
                    decoded.push_back(piece);
                }
            }
            flush();
            pieces = std::move(decoded);
        }
    }
    std::string result;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i && !impl_->has_decoder) {
            result += ' ';
        }
        result += pieces[i];
    }
    text = std::move(result);
    return true;
}
