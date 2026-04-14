// bpe.h, Qwen3/GPT-2 byte-level BPE tokenizer (CPU-only, no dependencies)
// Loads vocab.json + merges.txt from HuggingFace tokenizer directory.
// Handles: byte-level encoding, GPT-2 regex pre-tokenizer, BPE merges,
//          special token recognition (<|endoftext|>).

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <climits>

#include "gguf.h"

// GPT-2 byte-level encoding table
// Maps byte [0..255] -> Unicode char for BPE vocab keys.
// Printable ASCII stays as-is, control/space bytes get remapped.
static void build_byte_encoder(std::string byte2str[256]) {
    // Standard GPT-2 byte encoder
    int bs[256], cs[256], n = 0, total = 0;
    // Printable ranges that map to themselves
    for (int b = '!'; b <= '~'; b++) { bs[total] = b; cs[total] = b; total++; }
    for (int b = 0xA1; b <= 0xAC; b++) { bs[total] = b; cs[total] = b; total++; }
    for (int b = 0xAE; b <= 0xFF; b++) { bs[total] = b; cs[total] = b; total++; }
    // Remaining bytes get mapped to 256+
    bool used[256] = {};
    for (int i = 0; i < total; i++) used[bs[i]] = true;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) { bs[total] = b; cs[total] = 256 + n; n++; total++; }
    }
    assert(total == 256);
    // Convert codepoints to UTF-8 strings
    for (int i = 0; i < 256; i++) {
        int cp = cs[i];
        char buf[4];
        int len;
        if (cp < 0x80) { buf[0] = (char)cp; len = 1; }
        else if (cp < 0x800) { buf[0] = (char)(0xC0 | (cp >> 6)); buf[1] = (char)(0x80 | (cp & 0x3F)); len = 2; }
        else { buf[0] = (char)(0xE0 | (cp >> 12)); buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[2] = (char)(0x80 | (cp & 0x3F)); len = 3; }
        byte2str[bs[i]] = std::string(buf, len);
    }
}

// UTF-8 helpers
static int utf8_codepoint(const char *s, int *advance) {
    unsigned char c = s[0];
    if (c < 0x80) { *advance = 1; return c; }
    if ((c & 0xE0) == 0xC0) { *advance = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) { *advance = 3; return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    if ((c & 0xF8) == 0xF0) { *advance = 4; return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); }
    *advance = 1; return c;  // fallback
}

// Unicode category checks (simplified but covers Latin + common scripts)
static bool is_letter(int cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0x80) return false;
    // Latin Extended: U+00C0-U+00D6, U+00D8-U+00F6, U+00F8-U+01BF + Latin Extended-A/B
    if (cp >= 0xC0 && cp <= 0x024F && cp != 0xD7 && cp != 0xF7) return true;
    // Common CJK, Cyrillic, Greek, Arabic, etc., treat as letters
    if (cp >= 0x0370 && cp <= 0x1FFF) return true;  // Greek, Cyrillic, Armenian, etc.
    if (cp >= 0x2C00 && cp <= 0x2DFF) return true;  // Georgian, etc.
    if (cp >= 0x3040 && cp <= 0x9FFF) return true;  // CJK
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;  // Korean
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;  // CJK compatibility
    if (cp >= 0x10000) return true;                 // SMP, mostly letters/symbols
    return false;
}

static bool is_digit(int cp) { return cp >= '0' && cp <= '9'; }

static bool is_whitespace(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C
        || cp == 0xA0 || cp == 0x2000 || cp == 0x2001 || cp == 0x2002 || cp == 0x200B;
}

static bool is_newline(int cp) { return cp == '\n' || cp == '\r'; }

// GPT-2 pre-tokenizer regex (manual implementation)
// Pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//          \s?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Splits text into non-overlapping chunks (on original text, not byte-encoded).
static std::vector<std::string> gpt2_pre_tokenize(const std::string &text) {
    std::vector<std::string> chunks;
    const char *s = text.c_str();
    int len = (int)text.size();
    int i = 0;

    while (i < len) {
        int adv;
        int cp = utf8_codepoint(s + i, &adv);

        // Rule 1: Contractions 's 't 're 've 'm 'll 'd
        if ((cp == '\'' || cp == 0x2019) && i + adv < len) {
            const char *rest = s + i + adv;
            int rlen = len - i - adv;
            auto try_match = [&](const char *suffix, int slen) -> bool {
                if (rlen >= slen) {
                    // case-insensitive compare
                    for (int k = 0; k < slen; k++) {
                        char c1 = rest[k], c2 = suffix[k];
                        if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
                        if (c1 != c2) return false;
                    }
                    // next char should NOT be a letter
                    if (rlen > slen) {
                        int a2;
                        int cp2 = utf8_codepoint(rest + slen, &a2);
                        if (is_letter(cp2)) return false;
                    }
                    chunks.push_back(std::string(s + i, adv + slen));
                    i += adv + slen;
                    return true;
                }
                return false;
            };
            if (try_match("ll", 2)) continue;
            if (try_match("re", 2)) continue;
            if (try_match("ve", 2)) continue;
            if (try_match("s", 1)) continue;
            if (try_match("t", 1)) continue;
            if (try_match("m", 1)) continue;
            if (try_match("d", 1)) continue;
        }

        // Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_letter(cp)) {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_letter(cp2)) break;
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }
        if (!is_newline(cp) && !is_letter(cp) && !is_digit(cp) && !is_whitespace(cp)) {
            // Non-letter/number/space, check if followed by letters
            int start = i;
            int after = i + adv;
            if (after < len) {
                int a2;
                int cp2 = utf8_codepoint(s + after, &a2);
                if (is_letter(cp2)) {
                    i = after + a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (!is_letter(cp3)) break;
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
            }
        }

        // Rule 3: \p{N}+  (digits, consume consecutively)
        if (is_digit(cp)) {
            int start = i;
            while (i < len && is_digit((unsigned char)s[i])) i++;
            // GPT-2 regex matches single \p{N}, let's match one at a time
            // to be safe, but in practice consecutive digits usually merge anyway.
            // The regex is \p{N} (single digit), so split each digit:
            for (int j = start; j < i; j++) {
                chunks.push_back(std::string(s + j, 1));
            }
            continue;
        }

        // Rule 5: \s*[\r\n]+  (newlines with optional leading whitespace)
        if (is_newline(cp)) {
            int start = i;
            while (i < len && is_newline((unsigned char)s[i])) i++;
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        // Rule 6: whitespace handling
        // Regex order: \s+(?!\S) first (trailing whitespace), then \s+ as fallback
        // \s+(?!\S) backtracks: consumes whitespace NOT followed by non-whitespace
        // This peels off leading spaces, leaving the last space to combine with the next word
        if (is_whitespace(cp)) {
            int start = i;
            // Find end of whitespace run
            int ws_end = i + adv;
            while (ws_end < len && is_whitespace((unsigned char)s[ws_end]) && !is_newline((unsigned char)s[ws_end]))
                ws_end++;
            // Check what follows the whitespace run
            bool followed_by_non_ws = (ws_end < len && !is_whitespace((unsigned char)s[ws_end]) && !is_newline((unsigned char)s[ws_end]));
            if (followed_by_non_ws && ws_end - start > 1) {
                // \s+(?!\S) matches all but the last space
                // Leave one space for the next iteration to combine with word
                int trailing = ws_end - 1;
                chunks.push_back(std::string(s + start, trailing - start));
                i = trailing;
                continue;
            }
            // Single space followed by word: combine space + word as one chunk
            i = start + adv;
            if (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (is_letter(cp2)) {
                    i += a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (!is_letter(cp3)) break;
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
                if (is_digit(cp2)) {
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
                if (!is_whitespace(cp2) && !is_newline(cp2)) {
                    int pstart = start;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (is_whitespace(cp3) || is_letter(cp3) || is_digit(cp3)) break;
                        i += a3;
                    }
                    while (i < len && is_newline((unsigned char)s[i])) i++;
                    chunks.push_back(std::string(s + pstart, i - pstart));
                    continue;
                }
            }
            // Trailing whitespace (end of string or before newline), consume all
            i = ws_end;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_whitespace(cp2)) break;
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        // Rule 4: [^\s\p{L}\p{N}]+[\r\n]*  (punctuation/symbols)
        {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (is_whitespace(cp2) || is_letter(cp2) || is_digit(cp2) || is_newline(cp2)) break;
                i += a2;
            }
            // trailing newlines
            while (i < len && is_newline((unsigned char)s[i])) i++;
            chunks.push_back(std::string(s + start, i - start));
        }
    }
    return chunks;
}

// BPE tokenizer struct
struct BPETokenizer {
    std::unordered_map<std::string, int> vocab;  // token_str -> id
    std::unordered_map<std::string, int> merges; // "a b" -> rank
    std::string byte2str[256];                   // byte -> GPT-2 UTF-8 string
    int eos_id;                                  // <|endoftext|> = 151643
    int n_vocab;
    std::vector<std::string> id_to_str;          // id -> token_str (reverse vocab)
};

// Minimal JSON parser for vocab.json ({"str": int, ...})
static bool load_vocab_json(const std::string &path, std::unordered_map<std::string, int> &vocab) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string data(sz, '\0');
    fread(&data[0], 1, sz, f);
    fclose(f);

    size_t i = 0;
    auto skip_ws = [&]() { while (i < sz && (data[i] == ' ' || data[i] == '\n' || data[i] == '\r' || data[i] == '\t')) i++; };

    skip_ws();
    if (data[i] != '{') return false;
    i++;

    while (i < sz) {
        skip_ws();
        if (data[i] == '}') break;
        if (data[i] == ',') { i++; skip_ws(); }
        if (data[i] != '"') return false;
        i++;

        // Parse key (JSON string with escape handling)
        std::string key;
        while (i < sz && data[i] != '"') {
            if (data[i] == '\\') {
                i++;
                if (i >= sz) return false;
                switch (data[i]) {
                    case '"': key += '"'; break;
                    case '\\': key += '\\'; break;
                    case '/': key += '/'; break;
                    case 'n': key += '\n'; break;
                    case 'r': key += '\r'; break;
                    case 't': key += '\t'; break;
                    case 'u': {
                        // \uXXXX -> UTF-8
                        if (i + 4 >= sz) return false;
                        char hex[5] = {data[i+1], data[i+2], data[i+3], data[i+4], 0};
                        int cp = (int)strtol(hex, nullptr, 16);
                        i += 4;
                        if (cp < 0x80) { key += (char)cp; }
                        else if (cp < 0x800) { key += (char)(0xC0 | (cp >> 6)); key += (char)(0x80 | (cp & 0x3F)); }
                        else { key += (char)(0xE0 | (cp >> 12)); key += (char)(0x80 | ((cp >> 6) & 0x3F)); key += (char)(0x80 | (cp & 0x3F)); }
                        break;
                    }
                    default: key += data[i]; break;
                }
            } else {
                key += data[i];
            }
            i++;
        }
        i++; // skip closing "

        skip_ws();
        if (data[i] != ':') return false;
        i++;
        skip_ws();

        // Parse integer value
        bool neg = false;
        if (data[i] == '-') { neg = true; i++; }
        int val = 0;
        while (i < sz && data[i] >= '0' && data[i] <= '9') {
            val = val * 10 + (data[i] - '0');
            i++;
        }
        if (neg) val = -val;

        vocab[key] = val;
    }
    return true;
}

// Load merges from merges.txt
static bool load_merges(const std::string &path, std::unordered_map<std::string, int> &merges) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    int rank = 0;
    // Skip header line (#version: 0.2)
    if (fgets(line, sizeof(line), f) == nullptr) { fclose(f); return false; }
    while (fgets(line, sizeof(line), f)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        merges[std::string(line)] = rank++;
    }
    fclose(f);
    return true;
}

// Load tokenizer from directory (vocab.json + merges.txt)
static bool load_bpe_tokenizer(BPETokenizer *tok, const char *dir) {
    build_byte_encoder(tok->byte2str);

    std::string vocab_path = std::string(dir) + "/vocab.json";
    std::string merges_path = std::string(dir) + "/merges.txt";

    if (!load_vocab_json(vocab_path, tok->vocab)) {
        fprintf(stderr, "ERROR: failed to load %s\n", vocab_path.c_str());
        return false;
    }
    if (!load_merges(merges_path, tok->merges)) {
        fprintf(stderr, "ERROR: failed to load %s\n", merges_path.c_str());
        return false;
    }

    tok->n_vocab = (int)tok->vocab.size();
    tok->eos_id = 151643;  // <|endoftext|>

    // Build reverse vocab (id -> token string)
    tok->id_to_str.resize(tok->n_vocab);
    for (auto &kv : tok->vocab) {
        if (kv.second >= 0 && kv.second < tok->n_vocab)
            tok->id_to_str[kv.second] = kv.first;
    }

    fprintf(stderr, "[BPE] Loaded: %d vocab, %d merges\n", tok->n_vocab, (int)tok->merges.size());
    return true;
}

// Load tokenizer from GGUF KV (tokenizer.ggml.tokens + tokenizer.ggml.merges)
static bool load_bpe_from_gguf(BPETokenizer *tok, const char *gguf_path) {
    build_byte_encoder(tok->byte2str);

    struct gguf_init_params gp = { true, NULL };
    struct gguf_context * ctx = gguf_init_from_file(gguf_path, gp);
    if (!ctx) {
        fprintf(stderr, "[BPE] failed to open %s\n", gguf_path);
        return false;
    }

    int64_t tok_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    int64_t mrg_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (tok_key < 0 || mrg_key < 0) {
        fprintf(stderr, "[BPE] tokenizer not found in %s\n", gguf_path);
        gguf_free(ctx);
        return false;
    }

    int n_tokens = (int)gguf_get_arr_n(ctx, tok_key);
    int n_merges = (int)gguf_get_arr_n(ctx, mrg_key);

    for (int i = 0; i < n_tokens; i++) {
        const char * s = gguf_get_arr_str(ctx, tok_key, (size_t)i);
        tok->vocab[std::string(s)] = i;
    }

    for (int i = 0; i < n_merges; i++) {
        const char * s = gguf_get_arr_str(ctx, mrg_key, (size_t)i);
        tok->merges[std::string(s)] = i;
    }

    gguf_free(ctx);

    tok->n_vocab = (int)tok->vocab.size();
    tok->eos_id = 151643;

    tok->id_to_str.resize(tok->n_vocab);
    for (auto &kv : tok->vocab) {
        if (kv.second >= 0 && kv.second < tok->n_vocab)
            tok->id_to_str[kv.second] = kv.first;
    }

    fprintf(stderr, "[BPE] Loaded from GGUF: %d vocab, %d merges\n", tok->n_vocab, n_merges);
    return true;
}

// Byte-level encode: raw text bytes -> GPT-2 BPE string
static std::string byte_level_encode(const BPETokenizer *tok, const std::string &text) {
    std::string out;
    for (unsigned char c : text) out += tok->byte2str[c];
    return out;
}

// BPE merge algorithm
// Input: list of symbols (strings). Merges pairs by priority.
static std::vector<std::string> bpe_merge(const std::unordered_map<std::string, int> &merge_rank,
                                           const std::vector<std::string> &symbols) {
    if (symbols.size() <= 1) return symbols;

    std::vector<std::string> work = symbols;

    while (work.size() > 1) {
        // Find the pair with lowest rank (highest priority)
        int best_rank = INT_MAX;
        int best_pos = -1;
        for (int i = 0; i < (int)work.size() - 1; i++) {
            std::string key = work[i] + " " + work[i + 1];
            auto it = merge_rank.find(key);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;  // no more merges

        // Merge the pair
        std::string merged = work[best_pos] + work[best_pos + 1];
        work[best_pos] = merged;
        work.erase(work.begin() + best_pos + 1);
    }
    return work;
}

// Encode a single pre-tokenized chunk -> token ids
static void encode_chunk(const BPETokenizer *tok, const std::string &chunk,
                          std::vector<int> &ids) {
    // Byte-level encode
    std::string encoded = byte_level_encode(tok, chunk);

    // Split into individual UTF-8 characters (each is a BPE symbol)
    std::vector<std::string> symbols;
    const char *s = encoded.c_str();
    int len = (int)encoded.size();
    int i = 0;
    while (i < len) {
        int adv;
        utf8_codepoint(s + i, &adv);
        symbols.push_back(std::string(s + i, adv));
        i += adv;
    }

    // Apply BPE merges
    std::vector<std::string> merged = bpe_merge(tok->merges, symbols);

    // Look up in vocab
    for (const auto &piece : merged) {
        auto it = tok->vocab.find(piece);
        if (it != tok->vocab.end()) {
            ids.push_back(it->second);
        } else {
            // Fallback: encode each byte individually (should not happen with byte-level BPE)
            fprintf(stderr, "[BPE] WARNING: unknown token '%s'\n", piece.c_str());
            for (unsigned char c : piece) {
                auto it2 = tok->vocab.find(std::string(1, c));
                if (it2 != tok->vocab.end()) ids.push_back(it2->second);
            }
        }
    }
}

// Full encode: text -> token ids
// Handles special tokens (<|endoftext|>) embedded in text.
// Appends EOS at end (post-processor behavior).
static std::vector<int> bpe_encode(const BPETokenizer *tok, const std::string &text,
                                    bool add_eos = true) {
    std::vector<int> ids;
    const std::string special = "<|endoftext|>";

    // Split on special tokens
    size_t pos = 0;
    while (pos < text.size()) {
        size_t found = text.find(special, pos);
        std::string segment = (found == std::string::npos)
            ? text.substr(pos) : text.substr(pos, found - pos);

        // Pre-tokenize and encode the segment
        if (!segment.empty()) {
            auto chunks = gpt2_pre_tokenize(segment);
            for (const auto &chunk : chunks) {
                encode_chunk(tok, chunk, ids);
            }
        }

        if (found == std::string::npos) break;
        ids.push_back(tok->eos_id);  // <|endoftext|>
        pos = found + special.size();
    }

    if (add_eos) ids.push_back(tok->eos_id);
    return ids;
}
