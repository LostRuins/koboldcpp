#pragma once

#include "engine/framework/tokenizers/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::tokenizers {

enum class SentencePieceType : int32_t {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
};

struct SentencePiecePiece {
    int id = 0;
    float score = 0.0F;
    std::string text;
    SentencePieceType type = SentencePieceType::Normal;
};

std::vector<SentencePiecePiece> load_sentencepiece_model(const std::filesystem::path & model_path);
std::vector<int32_t> tokenize_sentencepiece(
    const std::vector<SentencePiecePiece> & pieces,
    const std::string & text);
std::string decode_sentencepiece(
    const std::vector<SentencePiecePiece> & pieces,
    const std::vector<int32_t> & token_ids);

class SentencePieceTokenizer final : public ITokenizer {
public:
    explicit SentencePieceTokenizer(std::vector<SentencePiecePiece> pieces);

    std::string family() const override;
    TokenizedText tokenize(const std::string & text) const override;

private:
    std::vector<SentencePiecePiece> pieces_;
};

std::shared_ptr<SentencePieceTokenizer> load_sentencepiece_tokenizer(
    const std::filesystem::path & model_path);

}  // namespace engine::tokenizers
