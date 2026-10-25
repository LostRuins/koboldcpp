#include "engine/community_models/chatterbox_turbo/tts.h"

#include "engine/community_models/chatterbox_turbo/text_tokenizer_turbo.h"

#include <cstring>
#include <stdexcept>

namespace engine::community_models::chatterbox_turbo {

namespace {

// Chatterbox Turbo's speech-token control ids (tts_turbo.py / s3gen/const.py). Not exposed via
// GGUF metadata, so mirrored here from the upstream Python reference.
constexpr int32_t kS3GenSilenceToken = 4299;
constexpr int32_t kOovTokenThreshold = 6561;  // speech_tokens = speech_tokens[speech_tokens < 6561]

std::vector<int32_t> require_i32_array(
    const engine::assets::TensorSource & source,
    const std::string & name,
    int64_t expected_count) {
    const auto raw = source.require_tensor_data(name);
    if (raw.metadata.dtype != "i32") {
        throw std::runtime_error("Chatterbox Turbo expected an i32 tensor for " + name + ", got " + raw.metadata.dtype);
    }
    if (raw.bytes.size() != static_cast<size_t>(expected_count) * sizeof(int32_t)) {
        throw std::runtime_error("Chatterbox Turbo tensor size mismatch for " + name);
    }
    std::vector<int32_t> out(static_cast<size_t>(expected_count));
    std::memcpy(out.data(), raw.bytes.data(), raw.bytes.size());
    return out;
}

}  // namespace

ChatterboxTurboTTSComponent::ChatterboxTurboTTSComponent(
    std::shared_ptr<const ChatterboxTurboAssets> assets,
    const engine::core::ExecutionContext & execution_context)
    : assets_(std::move(assets)) {
    tokenizer_ = load_chatterbox_turbo_tokenizer(
        assets_->resources.require_file("tokenizer_vocab"),
        assets_->resources.require_file("tokenizer_merges"),
        assets_->resources.require_file("tokenizer_special_tokens"));

    auto t3_weights = load_t3_turbo_inference_weights(*assets_->t3_turbo_weights, execution_context);
    t3_ = std::make_unique<T3TurboInferenceComponent>(t3_weights, execution_context);

    s3gen_ = ChatterboxTurboS3Gen::load(assets_->s3gen_weights, execution_context);

    const auto & conds = *assets_->builtin_conditionals_turbo;
    builtin_speaker_embedding_ = conds.require_f32("t3.speaker_emb", {t3_weights->speaker_embed_size});
    builtin_cond_prompt_speech_tokens_ = require_i32_array(
        conds, "t3.speech_prompt_tokens", static_cast<int64_t>(conds.require_metadata("t3.speech_prompt_tokens").shape.at(0)));

    const auto prompt_token_shape = conds.require_metadata("gen.prompt_token").shape;
    const auto prompt_feat_shape = conds.require_metadata("gen.prompt_feat").shape;  // [frames, mel_dim]
    const auto embedding_shape = conds.require_metadata("gen.embedding").shape;

    builtin_ref_dict_.prompt_tokens = require_i32_array(conds, "gen.prompt_token", prompt_token_shape.at(0));
    builtin_ref_dict_.prompt_token_count = prompt_token_shape.at(0);
    builtin_ref_dict_.prompt_feat = conds.require_f32("gen.prompt_feat", prompt_feat_shape);
    builtin_ref_dict_.prompt_feat_frames = prompt_feat_shape.at(0);
    builtin_ref_dict_.prompt_feat_dims = prompt_feat_shape.at(1);
    builtin_ref_dict_.embedding = conds.require_f32("gen.embedding", embedding_shape);
    builtin_ref_dict_.embedding_size = embedding_shape.at(0);
}

engine::codecs::s3gen::S3GenInferenceOutputs ChatterboxTurboTTSComponent::generate(
    const std::string & text,
    const ChatterboxTurboGenerateConfig & config) const {
    T3TurboGenerateRequest request;
    request.speaker_embedding = builtin_speaker_embedding_;
    request.cond_prompt_speech_tokens = builtin_cond_prompt_speech_tokens_;
    request.text_tokens = encode_chatterbox_turbo_text(*tokenizer_, text);
    request.max_new_tokens = config.max_new_tokens;
    request.temperature = config.temperature;
    request.top_p = config.top_p;
    request.top_k = config.top_k;
    request.repetition_penalty = config.repetition_penalty;
    request.seed = config.seed;

    const auto t3_outputs = t3_->generate_speech_tokens(request);

    std::vector<int32_t> speech_tokens;
    speech_tokens.reserve(t3_outputs.predicted_tokens.size() + 3);
    for (int32_t token : t3_outputs.predicted_tokens) {
        if (token < kOovTokenThreshold) {
            speech_tokens.push_back(token);
        }
    }
    speech_tokens.push_back(kS3GenSilenceToken);
    speech_tokens.push_back(kS3GenSilenceToken);
    speech_tokens.push_back(kS3GenSilenceToken);

    return s3gen_->synthesize(builtin_ref_dict_, speech_tokens, config.seed, config.seed);
}

}  // namespace engine::community_models::chatterbox_turbo
