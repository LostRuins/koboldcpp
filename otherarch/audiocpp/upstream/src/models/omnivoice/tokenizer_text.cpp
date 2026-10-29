#include "engine/models/omnivoice/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <regex>
#include <stdexcept>
#include <utility>

namespace engine::models::omnivoice {

struct OmniVoiceTextTokenizer::Impl {
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    std::regex nonverbal_pattern{
        R"(\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\])"};
};

namespace {

std::shared_ptr<const OmniVoiceTextTokenizer::Impl> load_impl(const OmniVoiceAssets & assets) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
    spec.tokenizer_json_path = assets.resources.require_file("tokenizer_json");
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;

    auto impl = std::make_shared<OmniVoiceTextTokenizer::Impl>();
    impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    return impl;
}

}  // namespace

OmniVoiceTextTokenizer::OmniVoiceTextTokenizer(std::shared_ptr<const OmniVoiceAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("OmniVoice text tokenizer requires assets");
    }
    impl_ = load_impl(*assets);
}

std::vector<int32_t> OmniVoiceTextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer->encode(text);
}

std::vector<int32_t> OmniVoiceTextTokenizer::encode_with_nonverbal_tags(const std::string & text) const {
    std::vector<int32_t> ids;
    std::sregex_iterator begin(text.begin(), text.end(), impl_->nonverbal_pattern);
    std::sregex_iterator end;
    size_t last_end = 0;
    for (auto it = begin; it != end; ++it) {
        const auto & match = *it;
        const size_t match_start = static_cast<size_t>(match.position());
        if (match_start > last_end) {
            auto segment_ids = encode(text.substr(last_end, match_start - last_end));
            ids.insert(ids.end(), segment_ids.begin(), segment_ids.end());
        }
        auto tag_ids = encode(match.str());
        ids.insert(ids.end(), tag_ids.begin(), tag_ids.end());
        last_end = static_cast<size_t>(match.position() + match.length());
    }
    if (last_end < text.size()) {
        auto segment_ids = encode(text.substr(last_end));
        ids.insert(ids.end(), segment_ids.begin(), segment_ids.end());
    }
    return ids.empty() ? encode(text) : ids;
}

int32_t OmniVoiceTextTokenizer::token_id(const std::string & token) const {
    const auto token_id = impl_->tokenizer->find_token_id(token);
    if (!token_id.has_value()) {
        throw std::runtime_error("OmniVoice tokenizer does not contain token: " + token);
    }
    return *token_id;
}

}  // namespace engine::models::omnivoice
