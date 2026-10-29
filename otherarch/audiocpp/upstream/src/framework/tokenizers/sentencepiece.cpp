#include "engine/framework/tokenizers/sentencepiece.h"

#include "sentencepiece_processor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::tokenizers {
namespace {

using ProcessorPtr = std::shared_ptr<sentencepiece::SentencePieceProcessor>;

std::mutex & processor_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<size_t, ProcessorPtr> & processor_cache() {
    static std::unordered_map<size_t, ProcessorPtr> cache;
    return cache;
}

void require_status(const sentencepiece::util::Status & status, const std::string & context) {
    if (!status.ok()) {
        throw std::runtime_error(context + ": " + status.ToString());
    }
}

size_t hash_combine(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

size_t fingerprint_piece(const SentencePiecePiece & piece) {
    size_t seed = std::hash<int>{}(piece.id);
    seed = hash_combine(seed, std::hash<std::string>{}(piece.text));
    seed = hash_combine(seed, std::hash<int>{}(static_cast<int>(piece.type)));
    seed = hash_combine(seed, std::hash<int32_t>{}(static_cast<int32_t>(piece.score * 1000000.0F)));
    return seed;
}

size_t fingerprint_pieces(const std::vector<SentencePiecePiece> & pieces) {
    size_t seed = std::hash<size_t>{}(pieces.size());
    for (const auto & piece : pieces) {
        seed = hash_combine(seed, fingerprint_piece(piece));
    }
    return seed;
}

ProcessorPtr load_processor(const std::filesystem::path & model_path) {
    auto processor = std::make_shared<sentencepiece::SentencePieceProcessor>();
    require_status(processor->Load(model_path.string()), "failed to load SentencePiece model");
    return processor;
}

std::vector<SentencePiecePiece> extract_pieces(const sentencepiece::SentencePieceProcessor & processor) {
    std::vector<SentencePiecePiece> pieces;
    const int piece_count = processor.GetPieceSize();
    pieces.reserve(static_cast<size_t>(piece_count));
    for (int id = 0; id < piece_count; ++id) {
        SentencePiecePiece piece;
        piece.id = id;
        piece.score = processor.GetScore(id);
        piece.text = processor.IdToPiece(id);
        if (processor.IsUnknown(id)) {
            piece.type = SentencePieceType::Unknown;
        } else if (processor.IsControl(id)) {
            piece.type = SentencePieceType::Control;
        } else if (processor.IsUnused(id)) {
            piece.type = SentencePieceType::Unused;
        } else if (processor.IsByte(id)) {
            piece.type = SentencePieceType::Byte;
        } else {
            piece.type = SentencePieceType::Normal;
        }
        pieces.push_back(std::move(piece));
    }
    return pieces;
}

ProcessorPtr lookup_processor_for_pieces(const std::vector<SentencePiecePiece> & pieces) {
    const auto fingerprint = fingerprint_pieces(pieces);
    std::lock_guard<std::mutex> lock(processor_cache_mutex());
    const auto it = processor_cache().find(fingerprint);
    if (it == processor_cache().end()) {
        throw std::runtime_error("SentencePiece processor cache miss for the provided piece inventory");
    }
    return it->second;
}

void cache_processor_for_pieces(const std::vector<SentencePiecePiece> & pieces, ProcessorPtr processor) {
    const auto fingerprint = fingerprint_pieces(pieces);
    std::lock_guard<std::mutex> lock(processor_cache_mutex());
    processor_cache()[fingerprint] = std::move(processor);
}

}  // namespace

std::vector<SentencePiecePiece> load_sentencepiece_model(const std::filesystem::path & model_path) {
    auto processor = load_processor(model_path);
    auto pieces = extract_pieces(*processor);
    cache_processor_for_pieces(pieces, std::move(processor));
    return pieces;
}

std::vector<int32_t> tokenize_sentencepiece(
    const std::vector<SentencePiecePiece> & pieces,
    const std::string & text) {
    const auto processor = lookup_processor_for_pieces(pieces);
    std::vector<int> ids;
    require_status(processor->Encode(text, &ids), "failed to tokenize with SentencePiece");

    std::vector<int32_t> output;
    output.reserve(ids.size());
    for (const int id : ids) {
        output.push_back(static_cast<int32_t>(id));
    }
    return output;
}

std::string decode_sentencepiece(
    const std::vector<SentencePiecePiece> & pieces,
    const std::vector<int32_t> & token_ids) {
    const auto processor = lookup_processor_for_pieces(pieces);
    std::vector<int> ids;
    ids.reserve(token_ids.size());
    for (const int32_t id : token_ids) {
        ids.push_back(static_cast<int>(id));
    }
    std::string output;
    require_status(processor->Decode(ids, &output), "failed to decode with SentencePiece");
    return output;
}

SentencePieceTokenizer::SentencePieceTokenizer(std::vector<SentencePiecePiece> pieces)
    : pieces_(std::move(pieces)) {}

std::string SentencePieceTokenizer::family() const {
    return "sentencepiece";
}

TokenizedText SentencePieceTokenizer::tokenize(const std::string & text) const {
    TokenizedText result;
    result.text = text;
    result.token_ids = tokenize_sentencepiece(pieces_, text);
    return result;
}

std::shared_ptr<SentencePieceTokenizer> load_sentencepiece_tokenizer(
    const std::filesystem::path & model_path) {
    return std::make_shared<SentencePieceTokenizer>(load_sentencepiece_model(model_path));
}

}  // namespace engine::tokenizers
