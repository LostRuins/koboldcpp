#include "engine/models/qwen3_tts/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>

namespace engine::models::qwen3_tts {

struct Qwen3TextTokenizer::Impl {
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

namespace {

std::shared_ptr<const Qwen3TextTokenizer::Impl> load_impl(const Qwen3TTSAssets & assets) {
    auto impl = std::make_shared<Qwen3TextTokenizer::Impl>();
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = assets.resources.require_file("vocab");
    spec.merges_path = assets.resources.require_file("merges");
    spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    return impl;
}

}  // namespace

Qwen3TextTokenizer::Qwen3TextTokenizer(std::shared_ptr<const Qwen3TTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 text tokenizer requires assets");
    }
    impl_ = load_impl(*assets);
}

std::string Qwen3TextTokenizer::build_assistant_prompt(const std::string & text) const {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string Qwen3TextTokenizer::build_reference_prompt(const std::string & text) const {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n";
}

std::string Qwen3TextTokenizer::build_instruct_prompt(const std::string & text) const {
    return "<|im_start|>user\n" + text + "<|im_end|>\n";
}

std::vector<int32_t> Qwen3TextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer->encode(text);
}

}  // namespace engine::models::qwen3_tts
