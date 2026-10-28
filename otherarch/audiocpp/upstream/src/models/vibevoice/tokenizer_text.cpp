#include "engine/models/vibevoice/tokenizer_text.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::vibevoice {
namespace {

constexpr const char * kSystemPrompt =
    " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";

std::shared_ptr<const VibeVoiceAssets> require_assets(std::shared_ptr<const VibeVoiceAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VibeVoice text tokenizer requires assets");
    }
    return assets;
}

int32_t require_token_id(const engine::tokenizers::LlamaBpeTokenizer & tokenizer, const std::string & token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("VibeVoice tokenizer missing token: " + token);
    }
    return *id;
}

std::string trim_ascii(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.pop_back();
    }
    return text;
}

std::vector<VibeVoicePromptLine> parse_script(const std::string & script) {
    static const std::regex line_regex(R"(^Speaker\s+(\d+)\s*:\s*(.*)$)", std::regex::icase);
    std::vector<VibeVoicePromptLine> lines;
    size_t start = 0;
    while (start <= script.size()) {
        const size_t end = script.find('\n', start);
        const std::string raw_line = end == std::string::npos
            ? script.substr(start)
            : script.substr(start, end - start);
        const std::string line = trim_ascii(raw_line);
        if (!line.empty()) {
            std::smatch match;
            if (std::regex_match(line, match, line_regex)) {
                VibeVoicePromptLine parsed;
                parsed.speaker_id = std::stoi(match[1].str());
                parsed.text = " " + trim_ascii(match[2].str());
                lines.push_back(std::move(parsed));
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    if (lines.empty()) {
        throw std::runtime_error("VibeVoice prompt has no valid Speaker N: lines");
    }
    int min_speaker = lines.front().speaker_id;
    for (const auto & line : lines) {
        min_speaker = std::min(min_speaker, line.speaker_id);
    }
    if (min_speaker > 0) {
        for (auto & line : lines) {
            line.speaker_id -= 1;
        }
    }
    return lines;
}

std::vector<int> collect_speaker_ids(const std::vector<VibeVoicePromptLine> & lines) {
    std::set<int> ids;
    for (const auto & line : lines) {
        ids.insert(line.speaker_id);
    }
    return {ids.begin(), ids.end()};
}

void append_tokens(
    VibeVoicePromptEncoding & encoding,
    const std::vector<int32_t> & tokens,
    const bool speech_mask_value) {
    encoding.input_ids.insert(encoding.input_ids.end(), tokens.begin(), tokens.end());
    encoding.speech_input_mask.insert(encoding.speech_input_mask.end(), tokens.size(), speech_mask_value ? 1U : 0U);
}

void append_token(VibeVoicePromptEncoding & encoding, int32_t token, bool speech_mask_value) {
    encoding.input_ids.push_back(token);
    encoding.speech_input_mask.push_back(speech_mask_value ? 1U : 0U);
}

int64_t speech_token_count(size_t samples, int64_t compression_ratio) {
    if (compression_ratio <= 0) {
        throw std::runtime_error("VibeVoice speech compression ratio must be positive");
    }
    return static_cast<int64_t>((samples + static_cast<size_t>(compression_ratio) - 1) /
                                static_cast<size_t>(compression_ratio));
}

std::vector<float> convert_prompt_audio_to_mono_resampled(
    const runtime::AudioBuffer & audio,
    int target_sample_rate_hz) {
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate == target_sample_rate_hz || mono.empty()) {
        return mono;
    }
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
    options.output_padding = 256;
    options.reject_empty_output = true;
    options.warning_context = "VibeVoice prompt audio";
    options.fallback_description = "linear resampling";
    return engine::audio::resample_mono_soxr_or_linear(mono, audio.sample_rate, target_sample_rate_hz, options);
}

int64_t speech_token_count(const runtime::AudioBuffer & audio, int64_t compression_ratio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("VibeVoice prompt audio must be non-empty");
    }
    const auto waveform = convert_prompt_audio_to_mono_resampled(audio, 24000);
    if (waveform.empty()) {
        throw std::runtime_error("VibeVoice prompt audio resampled to empty waveform");
    }
    return speech_token_count(waveform.size(), compression_ratio);
}

}  // namespace

struct VibeVoiceTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const VibeVoiceAssets> input_assets)
        : assets(std::move(input_assets)),
          tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              assets->resources.require_file("tokenizer_vocab"),
              assets->resources.require_file("tokenizer_merges"),
              assets->resources.require_file("tokenizer_config"),
              assets->resources.find_file("tokenizer_json") == nullptr
                  ? std::nullopt
                  : std::optional<std::filesystem::path>(*assets->resources.find_file("tokenizer_json")),
              engine::tokenizers::LlamaBpePreTokenizer::Qwen2,
          }),
          speech_start(require_token_id(tokenizer, "<|vision_start|>")),
          speech_end(require_token_id(tokenizer, "<|vision_end|>")),
          speech_diffusion(require_token_id(tokenizer, "<|vision_pad|>")),
          eos(require_token_id(tokenizer, "<|endoftext|>")),
          pad(require_token_id(tokenizer, "<|image_pad|>")) {}

    std::shared_ptr<const VibeVoiceAssets> assets;
    engine::tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t speech_start = 0;
    int32_t speech_end = 0;
    int32_t speech_diffusion = 0;
    int32_t eos = 0;
    int32_t pad = 0;
};

VibeVoiceTextTokenizer::VibeVoiceTextTokenizer(std::shared_ptr<const VibeVoiceAssets> assets)
    : impl_(std::make_shared<Impl>(require_assets(std::move(assets)))) {}

std::vector<int32_t> VibeVoiceTextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer.encode(text);
}

VibeVoicePromptEncoding VibeVoiceTextTokenizer::encode_prompt(const VibeVoiceRequest & request) const {
    VibeVoicePromptEncoding encoding;
    encoding.parsed_script = parse_script(request.text);
    encoding.speaker_ids = collect_speaker_ids(encoding.parsed_script);

    append_tokens(encoding, encode(kSystemPrompt), false);

    if (!request.speakers.empty()) {
        append_tokens(encoding, encode(" Voice input:\n"), false);
        for (size_t index = 0; index < request.speakers.size(); ++index) {
            append_tokens(encoding, encode(" Speaker " + std::to_string(index) + ":"), false);
            append_token(encoding, impl_->speech_start, false);
            const int64_t speech_tokens = speech_token_count(
                request.speakers[index].audio,
                impl_->assets->processor.speech_tok_compress_ratio);
            encoding.speech_prompt_token_counts.push_back(speech_tokens);
            for (int64_t i = 0; i < speech_tokens; ++i) {
                append_token(encoding, impl_->speech_diffusion, true);
            }
            append_token(encoding, impl_->speech_end, false);
            append_tokens(encoding, encode("\n"), false);
        }
    }

    append_tokens(encoding, encode(" Text input:\n"), false);
    for (const auto & line : encoding.parsed_script) {
        append_tokens(
            encoding,
            encode(" Speaker " + std::to_string(line.speaker_id) + ":" + line.text + "\n"),
            false);
    }
    append_tokens(encoding, encode(" Speech output:\n"), false);
    append_token(encoding, impl_->speech_start, false);
    encoding.attention_mask.assign(encoding.input_ids.size(), 1);
    return encoding;
}

int32_t VibeVoiceTextTokenizer::eos_id() const noexcept {
    return impl_->eos;
}

int32_t VibeVoiceTextTokenizer::pad_id() const noexcept {
    return impl_->pad;
}

int32_t VibeVoiceTextTokenizer::speech_start_id() const noexcept {
    return impl_->speech_start;
}

int32_t VibeVoiceTextTokenizer::speech_end_id() const noexcept {
    return impl_->speech_end;
}

int32_t VibeVoiceTextTokenizer::speech_diffusion_id() const noexcept {
    return impl_->speech_diffusion;
}

}  // namespace engine::models::vibevoice
