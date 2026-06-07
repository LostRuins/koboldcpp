#ifndef __SD_TOKENIZERS_T5_UNIGRAM_TOKENIZER_H__
#define __SD_TOKENIZERS_T5_UNIGRAM_TOKENIZER_H__

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "darts.h"
#include "tokenizer.h"

class MetaspacePreTokenizer {
private:
    std::string replacement;
    bool add_prefix_space;

public:
    MetaspacePreTokenizer(const std::string replacement = " ", bool add_prefix_space = true);

    std::string tokenize(const std::string& input) const;
};

using EncodeResult = std::vector<std::pair<std::string, int>>;

class T5UniGramTokenizer : public Tokenizer {
public:
    enum Status {
        OK,
        NO_PIECES_LOADED,
        NO_ENTRY_FOUND,
        BUILD_DOUBLE_ARRAY_FAILED,
        PIECE_ALREADY_DEFINED,
        INVLIAD_JSON
    };

protected:
    MetaspacePreTokenizer pre_tokenizer;
    std::vector<std::pair<std::string, float>> piece_score_pairs;
    float min_score_ = 0.0f;
    float max_score_ = 0.0f;
    std::unique_ptr<Darts::DoubleArray> trie_;
    int trie_results_size_ = 0;
    Status status_         = OK;
    float kUnkPenalty      = 10.0f;
    std::string replacement;
    bool add_prefix_space = true;

    void InitializePieces(const std::string& json_str);
    void BuildTrie(std::vector<std::pair<std::string, int>>* pieces);
    float GetScoreInlined(int id) const;
    bool IsUnusedInlined(int id) const;
    bool IsUserDefinedInlined(int id) const;
    size_t OneCharLen(const char* src) const;
    EncodeResult EncodeOptimized(const std::string& normalized) const;

    float min_score() const { return min_score_; }
    float max_score() const { return max_score_; }
    Status status() const { return status_; }
    std::string decode_token(int token_id) const override;
    std::string normalize(const std::string& input) const override;

public:
    explicit T5UniGramTokenizer(bool is_umt5 = false);
    ~T5UniGramTokenizer();

    std::vector<int> encode(const std::string& input, on_new_token_cb_t on_new_token_cb = nullptr) override;
};

#endif  // __SD_TOKENIZERS_T5_UNIGRAM_TOKENIZER_H__
