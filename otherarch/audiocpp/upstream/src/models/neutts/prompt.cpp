#include "engine/models/neutts/prompt.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::models::neutts {
namespace {

std::shared_ptr<const NeuTTSAssets> require_assets(
    std::shared_ptr<const NeuTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("NeuTTS prompt builder requires assets");
    }
    return assets;
}

int32_t require_token_id(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("NeuTTS tokenizer missing token: " + token);
    }
    return *id;
}

bool contains(const std::vector<std::string> & values, const std::string & value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string checked_emotion(const NeuTTSAssets & assets, std::string emotion) {
    if (emotion.empty()) {
        emotion = "neutral";
    }
    if (!contains(assets.backbone.supported_emotions, emotion)) {
        throw std::runtime_error("unsupported NeuTTS emotion: " + emotion);
    }
    return emotion;
}

}  // namespace

struct NeuTTSPromptBuilder::Impl {
    explicit Impl(std::shared_ptr<const NeuTTSAssets> input_assets)
        : assets(require_assets(std::move(input_assets))) {
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_json_path = assets->resources.require_file("tokenizer_json");
        spec.tokenizer_config_path = assets->resources.require_file("tokenizer_config");
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);

        text_prompt_start = require_token_id(*tokenizer, "<|TEXT_PROMPT_START|>");
        text_prompt_end = require_token_id(*tokenizer, "<|TEXT_PROMPT_END|>");
        speech_generation_start = require_token_id(*tokenizer, "<|SPEECH_GENERATION_START|>");
        speech_generation_end = require_token_id(*tokenizer, "<|SPEECH_GENERATION_END|>");
        speech_token_start = require_token_id(*tokenizer, "<|speech_0|>");
        speech_token_end = require_token_id(*tokenizer, "<|speech_65535|>");
    }

    std::shared_ptr<const NeuTTSAssets> assets;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t text_prompt_start = 0;
    int32_t text_prompt_end = 0;
    int32_t speech_generation_start = 0;
    int32_t speech_generation_end = 0;
    int32_t speech_token_start = 0;
    int32_t speech_token_end = 0;
};

std::string normalize_neutts_text(std::string text) {
    replace_all(text, u8"‘", "'");
    replace_all(text, u8"’", "'");
    replace_all(text, u8"“", "\"");
    replace_all(text, u8"”", "\"");
    replace_all(text, u8"　", " ");
    return text;
}

NeuTTSPromptBuilder::NeuTTSPromptBuilder(std::shared_ptr<const NeuTTSAssets> assets)
    : impl_(std::make_shared<Impl>(std::move(assets))) {}

NeuTTSPrompt NeuTTSPromptBuilder::build(
    const std::string & text,
    const std::string & speaker,
    const std::string & emotion) const {
    const auto speaker_it = impl_->assets->speakers.find(speaker.empty() ? "emily" : speaker);
    if (speaker_it == impl_->assets->speakers.end()) {
        throw std::runtime_error("unsupported NeuTTS speaker: " + speaker);
    }
    const auto resolved_emotion = checked_emotion(*impl_->assets, emotion);
    const bool has_emotion_token = resolved_emotion != "neutral";
    const auto & prompt = speaker_it->second;

    NeuTTSPrompt out;
    out.speaker = speaker_it->first;
    out.emotion = resolved_emotion;
    out.normalized_reference_text = normalize_neutts_text(prompt.reference_text);
    out.normalized_input_text = normalize_neutts_text(text);
    out.speech_token_start = impl_->speech_token_start;
    out.speech_token_end = impl_->speech_token_end;
    out.speech_generation_end = impl_->speech_generation_end;

    std::vector<int32_t> text_ids;
    if (has_emotion_token) {
        text_ids = impl_->tokenizer->encode(out.normalized_reference_text, true);
        const std::string emotion_token = "<|" + resolved_emotion + "|>";
        std::string upper = emotion_token;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
        text_ids.push_back(require_token_id(*impl_->tokenizer, upper));
        auto input_ids = impl_->tokenizer->encode(out.normalized_input_text, true);
        text_ids.insert(text_ids.end(), input_ids.begin(), input_ids.end());
    } else {
        text_ids = impl_->tokenizer->encode(
            out.normalized_reference_text + " " + out.normalized_input_text,
            true);
    }

    out.token_ids.reserve(text_ids.size() + prompt.speech_codes.size() + 3);
    out.token_ids.push_back(impl_->text_prompt_start);
    out.token_ids.insert(out.token_ids.end(), text_ids.begin(), text_ids.end());
    out.token_ids.push_back(impl_->text_prompt_end);
    out.token_ids.push_back(impl_->speech_generation_start);
    for (const int32_t code : prompt.speech_codes) {
        if (code < 0 || impl_->speech_token_start + code > impl_->speech_token_end) {
            throw std::runtime_error("NeuTTS speaker prompt code out of range for speaker: " + out.speaker);
        }
        out.token_ids.push_back(impl_->speech_token_start + code);
    }
    return out;
}

}  // namespace engine::models::neutts
