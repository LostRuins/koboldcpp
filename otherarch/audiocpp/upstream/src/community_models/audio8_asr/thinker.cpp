#include "engine/community_models/audio8_asr/thinker.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::audio8_asr {
namespace {

namespace modules = engine::modules;

runtime::GreedyCausalDecoderSpec make_qwen2_decoder_spec(const Audio8ASRDecoderConfig & config) {
    runtime::GreedyCausalDecoderSpec spec;
    // Qwen2-style decoder: attention biases, no Q/K norms, RoPE theta 1e6.
    spec.decoder.stack.hidden_size = config.hidden_size;
    spec.decoder.stack.num_attention_heads = config.num_attention_heads;
    spec.decoder.stack.num_key_value_heads = config.num_key_value_heads;
    spec.decoder.stack.head_dim = config.head_dim;
    spec.decoder.stack.intermediate_size = config.intermediate_size;
    spec.decoder.stack.layers = config.num_hidden_layers;
    spec.decoder.stack.rms_norm_eps = config.rms_norm_eps;
    spec.decoder.stack.rope_theta = config.rope_theta;
    spec.decoder.stack.use_qk_norm = false;
    spec.decoder.stack.runtime.static_cache.update_mode =
        modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    spec.decoder.logits_size = config.vocab_size;
    spec.decoder.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    spec.vocab_size = config.vocab_size;
    spec.max_position_embeddings = config.max_position_embeddings;
    spec.tie_word_embeddings = config.tie_word_embeddings;
    spec.attention_bias = true;
    spec.token_embedding_tensor = "language_model.model.embed_tokens.weight";
    spec.lm_head_tensor = "language_model.lm_head.weight";
    spec.final_norm_tensor = "language_model.model.norm.weight";
    spec.layer_prefix = "language_model.model.layers";
    spec.eos_token_ids = config.eos_token_ids;
    return spec;
}

}  // namespace

Audio8Qwen2ThinkerRuntime::Audio8Qwen2ThinkerRuntime(
    std::shared_ptr<const assets::TensorSource> weights_source,
    const Audio8ASRDecoderConfig & config,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : qwen2_runtime_(
          std::move(weights_source),
          make_qwen2_decoder_spec(config),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type),
      config_(std::make_shared<const Audio8ASRDecoderConfig>(config)) {}

Audio8Qwen2ThinkerRuntime::~Audio8Qwen2ThinkerRuntime() = default;

Audio8ASRGeneratedTokens Audio8Qwen2ThinkerRuntime::generate(
    const Audio8ASRPrompt & prompt,
    const Audio8ASRAudioEmbeddings & audio_embeddings,
    const Audio8ASRGenerationOptions & options) {
    if (prompt.input_ids.empty()) {
        throw std::runtime_error("Audio8 ASR thinker prompt is empty");
    }
    if (audio_embeddings.hidden_size != config_->hidden_size ||
        audio_embeddings.tokens != static_cast<int64_t>(prompt.audio_token_positions.size()) ||
        static_cast<int64_t>(audio_embeddings.values.size()) != audio_embeddings.tokens * config_->hidden_size) {
        throw std::runtime_error("Audio8 ASR audio embeddings do not match the prompt placeholders");
    }
    for (const int32_t position : prompt.audio_token_positions) {
        if (position < 0 || position >= static_cast<int32_t>(prompt.input_ids.size())) {
            throw std::runtime_error("Audio8 ASR audio placeholder position out of range");
        }
    }
    runtime::GreedyCausalDecoderRuntime::Prompt decoder_prompt;
    decoder_prompt.input_ids = prompt.input_ids;
    decoder_prompt.injection.values = audio_embeddings.values;
    decoder_prompt.injection.tokens = audio_embeddings.tokens;
    decoder_prompt.injection.positions = prompt.audio_token_positions;
    Audio8ASRGeneratedTokens out;
    out.token_ids = qwen2_runtime_.generate(decoder_prompt, options.max_new_tokens);
    return out;
}

}  // namespace engine::community_models::audio8_asr
