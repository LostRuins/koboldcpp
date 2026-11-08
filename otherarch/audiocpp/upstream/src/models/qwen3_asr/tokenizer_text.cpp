#include "engine/models/qwen3_asr/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {

struct Qwen3ASRTextTokenizer::Impl {
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

namespace {

std::shared_ptr<const Qwen3ASRTextTokenizer::Impl> load_impl(const Qwen3ASRAssets & assets) {
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
    if (const auto * path = assets.resources.find_file("vocab")) {
        spec.vocab_path = *path;
    }
    if (const auto * path = assets.resources.find_file("merges")) {
        spec.merges_path = *path;
    }
    if (const auto * path = assets.resources.find_file("tokenizer_json")) {
        spec.tokenizer_json_path = *path;
    }
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;

    auto impl = std::make_shared<Qwen3ASRTextTokenizer::Impl>();
    impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    return impl;
}

std::string default_chat_prompt(const std::string & context) {
    return "<|im_start|>system\n" + context +
        "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace

Qwen3ASRTextTokenizer::Qwen3ASRTextTokenizer(std::shared_ptr<const Qwen3ASRAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Qwen3 ASR tokenizer requires assets");
    }
    impl_ = load_impl(*assets_);
}

Qwen3ASRPrompt Qwen3ASRTextTokenizer::build_prompt(
    const std::string & context,
    const std::string & language,
    const int64_t audio_feature_tokens) const {
    std::string prompt = default_chat_prompt(context);
    if (!language.empty() && language != "Auto") {
        prompt += "language " + language + "<asr_text>";
    }
    return build_raw_audio_prompt(prompt, audio_feature_tokens);
}

Qwen3ASRPrompt Qwen3ASRTextTokenizer::build_raw_audio_prompt(
    const std::string & text,
    const int64_t audio_feature_tokens) const {
    if (audio_feature_tokens <= 0) {
        throw std::runtime_error("Qwen3 ASR prompt requires positive audio feature token count");
    }
    const auto ids = impl_->tokenizer->encode(text);
    const int32_t audio_token = static_cast<int32_t>(assets_->config.text_decoder.audio_token_id);
    std::vector<int32_t> expanded;
    expanded.reserve(ids.size() + static_cast<size_t>(audio_feature_tokens));
    Qwen3ASRPrompt result;
    for (const int32_t id : ids) {
        if (id == audio_token) {
            for (int64_t i = 0; i < audio_feature_tokens; ++i) {
                result.audio_token_positions.push_back(static_cast<int32_t>(expanded.size()));
                expanded.push_back(id);
            }
        } else {
            expanded.push_back(id);
        }
    }
    result.input_ids = std::move(expanded);
    result.attention_mask.assign(result.input_ids.size(), 1);
    return result;
}

std::string Qwen3ASRTextTokenizer::decode(const std::vector<int32_t> & token_ids) const {
    std::vector<int32_t> filtered;
    filtered.reserve(token_ids.size());
    for (const int32_t id : token_ids) {
        if (id == assets_->config.text_decoder.pad_token_id ||
            std::find(
                assets_->config.text_decoder.eos_token_ids.begin(),
                assets_->config.text_decoder.eos_token_ids.end(),
                static_cast<int64_t>(id)) != assets_->config.text_decoder.eos_token_ids.end() ||
            impl_->tokenizer->is_control_token_id(id)) {
            continue;
        }
        filtered.push_back(id);
    }
    return impl_->tokenizer->decode(filtered);
}

}  // namespace engine::models::qwen3_asr
