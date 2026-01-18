#ifndef tokenizer_h
#define tokenizer_h

#include <unordered_map>
#include <stdint.h>
#include <map>
#include <unordered_set>
#include <regex>
#include <queue>
#include "ttsutil.h"

struct token_trie {
    bool has_value = false;
    uint32_t token;
    std::map<char, struct token_trie> children;

    void add(const std::string & gram, uint32_t token);
    void _add(const std::string & gram, uint32_t new_token, size_t index);
    const struct token_trie * traverse(const char c) const;
};

static std::regex duped_spaces("\\s{2,}");
static std::regex spaces("\\s");

struct result {
    uint32_t token;
    size_t offset;
    float score;
};

// much of this is implemented in llama.cpp, but in order to simplify this for my use case, I reimplementing here.
// There are several important simplifications here:
// 1. I only implement unigram tokenization
// 2. I don't need to support detokenization
struct unigram_tokenizer {
    unigram_tokenizer(std::unordered_map<std::string, uint32_t> vocab, uint32_t unk_token, float unk_token_score, std::vector<float> scores): vocab(vocab), unk_token(unk_token), unk_token_score(unk_token_score), scores(scores) {};
    ~unigram_tokenizer() = default;

    std::unordered_map<std::string, uint32_t> vocab;
    std::vector<float> scores;
    struct token_trie root_trie;
    uint32_t unk_token;
    float unk_token_score;
    uint32_t eos_token = 1;
    bool dedupe_spaces = true;
    bool init = false;

    void initialize_tokenizer();
    void tokenize(const std::string & text, std::vector<uint32_t> & tokens);
};

// For intializing a new tokenizer from a gguf file meta
unigram_tokenizer * unigram_tokenizer_from_gguf(gguf_context * meta);

// While this functions like a tokenizer, no token ids are assigned as the token ids never need to be used in the context in which this is
// currently being used. This tokenizer pattern is currently being used by the phonemizer to break up a word into its relevant graphemes.
// As such, only the graphemes need to be returned.
struct single_pass_tokenizer {
    single_pass_tokenizer(std::vector<std::string> tkns): tokens(tkns) {
        max_size = 0;
        for (auto token : tkns) {
            token_vocab.insert(token);
            if (token.size() > max_size) {
                max_size = token.size();
            }
        }
    }
    size_t max_size;
    uint32_t unknown_id = 0;
    std::vector<std::string> tokens;
    std::unordered_set<std::string> token_vocab;
    void tokenize(const std::string & text, std::vector<uint32_t> & token_ids);
    void token_split(const std::string & text, std::vector<std::string> & tokens);
};

single_pass_tokenizer * single_pass_tokenizer_from_gguf(gguf_context * meta, std::string key_name = "phonemizer.graphemes");

struct bpe_symbol;

struct bpe_merge {
    bpe_symbol * a;
    bpe_symbol * b;
    int rank;
    int new_size;

    bpe_symbol * merge();
};

struct bpe_merge_comp{
    bool operator() (const bpe_merge & a, const bpe_merge & b);
};

struct pair_hash {
    size_t operator() (const std::pair<std::string, std::string> & p) const;
};

struct bpe_symbol {
    bpe_symbol(const char * token): token(token) {};
    const char* token;
    int size = 1;
    int pos;
    bpe_symbol * next = nullptr;
    bpe_symbol * last = nullptr;

    void add_merges(std::priority_queue<bpe_merge, std::vector<bpe_merge>, bpe_merge_comp> & merges, std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> & rank_map, bool only_forward = false);
    std::string as_str();
};

struct pair_builder {
    pair_builder(std::string word) {
        bpe_symbol * last = nullptr;
        for (int i = 0; i < word.size(); i++) {
            int increment = 0;
            // make sure we process each utf-8 character.
            while(i + increment + 1 < word.size() && (word[i+increment+1] & 0b11000000) == 0b10000000) {
                ++increment;
            }
            bpe_symbol * part = new bpe_symbol(word.data()+i);
            part->pos = i;
            part->size += increment;
            i += increment;
            if (last) {
                last->next = part;
                part->last = last;
            }
            last = part;
            parts.push_back(part);
        }
    }

    ~pair_builder() {
        for (auto p : parts) {
            delete p;
        }
    }

    void join_pairs(std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> & rank_map);
    std::vector<bpe_symbol*> parts;
};

struct bpe_tokenizer {
    bpe_tokenizer(std::unordered_map<std::string, uint32_t> & tokens_to_ids, std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> & ranks, uint32_t bos, uint32_t eos): tokens_to_ids(tokens_to_ids), ranks(ranks), eos_token_id(eos), bos_token_id(bos) {};
    std::unordered_map<std::string, uint32_t> tokens_to_ids;
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> ranks;
    uint32_t eos_token_id;
    uint32_t bos_token_id;

    void tokenize(const std::string & text, std::vector<uint32_t> & token_ids);
    void bpe_tokenize(std::string chunk, std::vector<uint32_t> & token_ids);
};

bpe_tokenizer * bpe_tokenizer_from_gguf(gguf_context * meta, std::string base_name = "tokenizer.ggml");

#endif
