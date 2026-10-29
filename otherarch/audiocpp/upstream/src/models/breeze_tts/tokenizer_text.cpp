#include "engine/models/breeze_tts/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::models::breeze_tts {
namespace {

constexpr const char * kDefaultInstruction = "Speak clearly and naturally.";
constexpr const char * kBos = "<bos>";
constexpr const char * kSpeaker0 = "[S0]";
constexpr const char * kInstructionBos = "<ins_bos>";
constexpr const char * kInstructionEos = "<ins_eos>";

std::vector<engine::tokenizers::LlamaBpeAddedToken> breeze_special_tokens(const BreezeTTSConfig & config) {
    return {
        {kSpeaker0, 262146},
        {"[S1]", 262147},
        {"[S2]", 262148},
        {"[S3]", 262149},
        {"[S4]", 262150},
        {"[S5]", 262151},
        {"[S6]", 262152},
        {"[S7]", 262153},
        {"[S8]", 262154},
        {"[S9]", 262155},
        {kInstructionBos, 262156},
        {kInstructionEos, 262157},
        {"<|AUDIO|>", static_cast<int32_t>(config.audio_token_id)},
        {"<|audio_eos|>", static_cast<int32_t>(config.audio_eos_token_id)},
    };
}

void append_tokens(BreezePromptBranch & out, const std::vector<int32_t> & ids, bool text) {
    out.input_ids.insert(out.input_ids.end(), ids.begin(), ids.end());
    out.text_mask.insert(out.text_mask.end(), ids.size(), text ? uint8_t{1} : uint8_t{0});
    if (text) {
        out.text_segment_lengths.push_back(static_cast<int64_t>(ids.size()));
        out.text_segments.push_back(ids);
    }
}

void append_audio_placeholders(BreezePromptBranch & out, int32_t audio_token, int32_t audio_eos, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("BreezeTTS clone requires encoded reference audio frames");
    }
    out.input_ids.insert(out.input_ids.end(), static_cast<size_t>(frames), audio_token);
    out.text_mask.insert(out.text_mask.end(), static_cast<size_t>(frames), uint8_t{0});
    out.input_ids.push_back(audio_eos);
    out.text_mask.push_back(uint8_t{0});
}

}  // namespace

struct BreezeTextTokenizer::Impl {
    Impl(std::shared_ptr<const BreezeTTSAssets> assets)
        : assets(std::move(assets)),
          tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              {},
              {},
              this->assets->resources.require_file("tokenizer_config_json"),
              this->assets->resources.require_file("tokenizer_json"),
              engine::tokenizers::LlamaBpePreTokenizer::Gemma4,
              breeze_special_tokens(this->assets->config),
              "\xE2\x96\x81"}) {
        audio_token = static_cast<int32_t>(this->assets->config.audio_token_id);
        audio_eos = static_cast<int32_t>(this->assets->config.audio_eos_token_id);
    }

    std::vector<int32_t> encode_text_segment(const std::string & text) const {
        return tokenizer.encode(std::string(kBos) + text, true);
    }

    std::shared_ptr<const BreezeTTSAssets> assets;
    engine::tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t audio_token = 0;
    int32_t audio_eos = 0;
};

BreezeTextTokenizer::BreezeTextTokenizer(std::shared_ptr<const BreezeTTSAssets> assets)
    : impl_(std::make_unique<Impl>(std::move(assets))) {}

BreezeTextTokenizer::~BreezeTextTokenizer() = default;

BreezePromptBranch BreezeTextTokenizer::build_tts_instruction(
    const std::string & text,
    const std::string & instruction) const {
    const std::string actual_instruction = instruction.empty() ? kDefaultInstruction : instruction;
    BreezePromptBranch out;
    append_tokens(out, impl_->encode_text_segment(
        std::string(kSpeaker0) + kInstructionBos + actual_instruction + kInstructionEos + text), true);
    return out;
}

BreezePromptBranch BreezeTextTokenizer::build_tts_plain(const std::string & text) const {
    BreezePromptBranch out;
    append_tokens(out, impl_->encode_text_segment(std::string(kSpeaker0) + text), true);
    return out;
}

BreezePromptBranch BreezeTextTokenizer::build_clone(
    const std::string & text,
    const std::string & instruction,
    const std::string & reference_text,
    int64_t reference_audio_frames) const {
    const std::string actual_instruction = instruction.empty() ? kDefaultInstruction : instruction;
    BreezePromptBranch out;
    append_tokens(out, impl_->encode_text_segment(std::string(kSpeaker0) + reference_text), true);
    append_audio_placeholders(out, impl_->audio_token, impl_->audio_eos, reference_audio_frames);
    append_tokens(out, impl_->encode_text_segment(
        std::string(kSpeaker0) + kInstructionBos + actual_instruction + kInstructionEos + text), true);
    return out;
}

BreezePromptBranch BreezeTextTokenizer::build_clone_negative(
    const std::string & text,
    const std::string & reference_text,
    int64_t reference_audio_frames) const {
    BreezePromptBranch out;
    append_tokens(out, impl_->encode_text_segment(std::string(kSpeaker0) + reference_text), true);
    append_audio_placeholders(out, impl_->audio_token, impl_->audio_eos, reference_audio_frames);
    append_tokens(out, impl_->encode_text_segment(std::string(kSpeaker0) + text), true);
    return out;
}

int32_t BreezeTextTokenizer::audio_token_id() const noexcept {
    return impl_->audio_token;
}

int32_t BreezeTextTokenizer::audio_eos_token_id() const noexcept {
    return impl_->audio_eos;
}

}  // namespace engine::models::breeze_tts
