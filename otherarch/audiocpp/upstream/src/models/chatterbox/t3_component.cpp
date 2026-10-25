#include "components/t3_runtime.h"

#include "engine/framework/debug/trace.h"

#include <chrono>

namespace {

std::vector<int32_t> repeat_tokens(const std::vector<int32_t> & tokens, int64_t batch) {
    std::vector<int32_t> repeated;
    repeated.reserve(static_cast<size_t>(batch) * tokens.size());
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        repeated.insert(repeated.end(), tokens.begin(), tokens.end());
    }
    return repeated;
}

std::vector<float> repeat_values(const std::vector<float> & values, int64_t batch) {
    std::vector<float> repeated;
    repeated.reserve(static_cast<size_t>(batch) * values.size());
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        repeated.insert(repeated.end(), values.begin(), values.end());
    }
    return repeated;
}

std::vector<float> extract_step_embeddings(
    const std::vector<float> & sequence_embeddings,
    int64_t batch,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t step_index) {
    std::vector<float> step(static_cast<size_t>(batch * hidden_size), 0.0f);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        const float * src = sequence_embeddings.data() + static_cast<ptrdiff_t>(((batch_index * seq_len) + step_index) * hidden_size);
        float * dst = step.data() + static_cast<ptrdiff_t>(batch_index * hidden_size);
        std::copy(src, src + hidden_size, dst);
    }
    return step;
}

std::vector<float> build_positioned_token_embeddings(
    const std::vector<float> & token_table,
    int64_t token_rows,
    const std::vector<float> & position_table,
    int64_t hidden_size,
    const std::vector<int32_t> & tokens,
    int64_t batch,
    bool zero_second_batch_tokens) {
    const int64_t seq_len = batch == 0 ? 0 : static_cast<int64_t>(tokens.size()) / batch;
    auto embeddings = engine::models::chatterbox::gather_rows(token_table, token_rows, hidden_size, tokens);
    if (zero_second_batch_tokens && batch > 1) {
        std::fill(
            embeddings.begin() + static_cast<ptrdiff_t>(seq_len * hidden_size),
            embeddings.begin() + static_cast<ptrdiff_t>(2 * seq_len * hidden_size),
            0.0f);
    }

    std::vector<int32_t> positions;
    positions.reserve(static_cast<size_t>(batch * seq_len));
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t pos = 0; pos < seq_len; ++pos) {
            positions.push_back(static_cast<int32_t>(pos));
        }
    }
    const auto position_embeddings = engine::models::chatterbox::gather_rows(
        position_table,
        static_cast<int64_t>(position_table.size()) / hidden_size,
        hidden_size,
        positions);
    for (size_t i = 0; i < embeddings.size(); ++i) {
        embeddings[i] += position_embeddings[i];
    }
    return embeddings;
}

}  // namespace

namespace engine::models::chatterbox {
struct T3InferenceComponent::State {
    void release_runtime_graphs() {
        std::lock_guard<std::mutex> lock(mutex);
        runner.reset();
        prefill_runner.reset();
    }

    void release_runtime_cache() {
        std::lock_guard<std::mutex> lock(mutex);
        owner.reset();
        runner.reset();
        prefill_runner.reset();
        prefix_cache.reset();
    }

    std::shared_ptr<T3DecodeBackendOwner> get_owner(
        const engine::core::BackendConfig & backend,
        const T3InferenceWeights & weights) {
        if (!owner || !same_backend(owner->config(), backend)) {
            owner = std::make_shared<T3DecodeBackendOwner>(weights, backend);
            runner.reset();
            prefill_runner.reset();
            prefix_cache.reset();
        }
        return owner;
    }

    struct PrefixCache {
        int64_t batch = 0;
        std::vector<float> speaker_embedding;
        std::vector<int32_t> cond_prompt_speech_tokens;
        std::vector<float> emotion_adv;
        std::vector<float> cond_embeddings;
        int64_t cond_length = 0;
        T3BackendCacheState cache;
    };

    std::shared_ptr<T3DecodeBackendRunner> get_runner(
        int64_t batch,
        int64_t cache_steps,
        const engine::core::BackendConfig & backend,
        const T3InferenceWeights & weights) {
        if (runner && runner->matches(batch, cache_steps, backend)) {
            engine::debug::timing_log_scalar("chatterbox.t3.decode.graph.rebuild_ms", 0.0);
            return runner;
        }
        auto owner = get_owner(backend, weights);
        runner.reset();
        const auto rebuild_started = std::chrono::steady_clock::now();
        runner = std::make_shared<T3DecodeBackendRunner>(weights, batch, cache_steps, std::move(owner));
        engine::debug::timing_log_scalar(
            "chatterbox.t3.decode.graph.rebuild_ms",
            std::to_string(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - rebuild_started).count()));
        return runner;
    }

    std::shared_ptr<T3PrefillBackendRunner> get_prefill_runner(
        int64_t batch,
        int64_t prefix_steps,
        int64_t seq_len,
        const engine::core::BackendConfig & backend,
        const T3InferenceWeights & weights) {
        if (prefill_runner && prefill_runner->matches(batch, prefix_steps, seq_len, backend)) {
            engine::debug::timing_log_scalar("chatterbox.t3.prefill.graph.rebuild_ms", 0.0);
            return prefill_runner;
        }
        auto owner = get_owner(backend, weights);
        prefill_runner.reset();
        const auto rebuild_started = std::chrono::steady_clock::now();
        prefill_runner = std::make_shared<T3PrefillBackendRunner>(
            weights,
            batch,
            prefix_steps,
            seq_len,
            std::move(owner));
        engine::debug::timing_log_scalar(
            "chatterbox.t3.prefill.graph.rebuild_ms",
            std::to_string(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - rebuild_started).count()));
        return prefill_runner;
    }

    mutable std::mutex mutex;
    std::shared_ptr<T3DecodeBackendOwner> owner;
    std::optional<PrefixCache> prefix_cache;
    std::shared_ptr<T3PrefillBackendRunner> prefill_runner;
    std::shared_ptr<T3DecodeBackendRunner> runner;
};

struct T3InferenceComponent::ConditioningInput {
    std::vector<float> speaker_embedding;
    std::vector<int32_t> cond_prompt_speech_tokens;
    std::vector<float> emotion_adv;
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> speech_tokens;
    int64_t batch = 1;
    int64_t cond_prompt_length = 0;
    int64_t text_length = 0;
    int64_t speech_length = 0;
    float guidance_scale = 0.0f;
};

struct T3InferenceComponent::PreparedInput {
    std::vector<float> cond_embeddings;
    std::vector<float> text_embeddings;
    std::vector<float> speech_embeddings;
    std::vector<float> combined_embeddings;
    int64_t cond_length = 0;
    int64_t batch = 0;
    int64_t total_length = 0;
    int64_t hidden_size = 0;
};

T3InferenceComponent::T3InferenceComponent(
    std::shared_ptr<const T3InferenceWeights> weights,
    const engine::core::ExecutionContext & execution_context)
    : weights_(std::move(weights)),
      execution_context_(&execution_context),
      state_(std::make_shared<State>()) {
    if (!weights_) {
        throw std::runtime_error("T3InferenceComponent requires weights");
    }
}

T3InferenceComponent::PreparedInput T3InferenceComponent::prepare_conditioning_inputs(const ConditioningInput & input) const {
    const int64_t hidden_size = weights_->hidden_size;
    const int64_t speaker_embed_size = weights_->speaker_embed_size;
    const int64_t num_heads = weights_->perceiver_num_heads;
    const int64_t query_tokens = weights_->perceiver_query_tokens;
    const auto spkr_enc_weight = engine::assets::tensor_data_to_f32("cond_enc.spkr_enc.weight", weights_->spkr_enc_weight);
    const auto spkr_enc_bias = engine::assets::tensor_data_to_f32("cond_enc.spkr_enc.bias", weights_->spkr_enc_bias);
    const auto emotion_adv_weight = engine::assets::tensor_data_to_f32(
        "cond_enc.emotion_adv_fc.weight",
        weights_->emotion_adv_weight);
    const auto perceiver_pre_attention_query = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.pre_attention_query",
        weights_->perceiver_pre_attention_query);
    const auto perceiver_norm_weight = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.norm.weight",
        weights_->perceiver_norm_weight);
    const auto perceiver_norm_bias = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.norm.bias",
        weights_->perceiver_norm_bias);
    const auto perceiver_to_q_weight = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.to_q.weight",
        weights_->perceiver_to_q_weight);
    const auto perceiver_to_q_bias = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.to_q.bias",
        weights_->perceiver_to_q_bias);
    const auto perceiver_to_k_weight = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.to_k.weight",
        weights_->perceiver_to_k_weight);
    const auto perceiver_to_k_bias = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.to_k.bias",
        weights_->perceiver_to_k_bias);
    const auto perceiver_to_v_weight = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.to_v.weight",
        weights_->perceiver_to_v_weight);
    const auto perceiver_to_v_bias = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.to_v.bias",
        weights_->perceiver_to_v_bias);
    const auto perceiver_proj_out_weight = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.proj_out.weight",
        weights_->perceiver_proj_out_weight);
    const auto perceiver_proj_out_bias = engine::assets::tensor_data_to_f32(
        "cond_enc.perceiver.attn.proj_out.bias",
        weights_->perceiver_proj_out_bias);
    const auto text_embedding_weight = engine::assets::tensor_data_to_f32("text_emb.weight", weights_->text_embedding_weight);
    const auto speech_embedding_weight = engine::assets::tensor_data_to_f32("speech_emb.weight", weights_->speech_embedding_weight);
    const auto text_position_weight = engine::assets::tensor_data_to_f32(
        "text_pos_emb.emb.weight",
        weights_->text_position_weight);
    const auto speech_position_weight = engine::assets::tensor_data_to_f32(
        "speech_pos_emb.emb.weight",
        weights_->speech_position_weight);

    const auto cond_prompt_emb = gather_rows(
        speech_embedding_weight,
        weights_->speech_vocab,
        hidden_size,
        input.cond_prompt_speech_tokens);
    std::vector<int32_t> cond_prompt_positions;
    cond_prompt_positions.reserve(static_cast<size_t>(input.batch * input.cond_prompt_length));
    for (int64_t b = 0; b < input.batch; ++b) {
        for (int64_t t = 0; t < input.cond_prompt_length; ++t) {
            cond_prompt_positions.push_back(static_cast<int32_t>(t));
        }
    }
    const auto cond_prompt_pos = gather_rows(
        speech_position_weight,
        weights_->speech_position_vocab,
        hidden_size,
        cond_prompt_positions);
    std::vector<float> cond_prompt_sum = cond_prompt_emb;
    for (size_t i = 0; i < cond_prompt_sum.size(); ++i) {
        cond_prompt_sum[i] += cond_prompt_pos[i];
    }

    std::vector<float> query(static_cast<size_t>(input.batch * query_tokens * hidden_size), 0.0f);
    for (int64_t b = 0; b < input.batch; ++b) {
        std::copy(
            perceiver_pre_attention_query.begin(),
            perceiver_pre_attention_query.begin() + static_cast<ptrdiff_t>(query_tokens * hidden_size),
            query.begin() + static_cast<ptrdiff_t>(b * query_tokens * hidden_size));
    }
    auto query_norm = layer_norm_rows(
        query,
        input.batch * query_tokens,
        hidden_size,
        perceiver_norm_weight,
        perceiver_norm_bias,
        1.0e-5f);
    auto cond_prompt_norm = layer_norm_rows(
        cond_prompt_sum,
        input.batch * input.cond_prompt_length,
        hidden_size,
        perceiver_norm_weight,
        perceiver_norm_bias,
        1.0e-5f);
    auto pre_att = attention_qkv_batched(
        query_norm,
        input.batch,
        query_tokens,
        cond_prompt_norm,
        input.cond_prompt_length,
        hidden_size,
        num_heads,
        perceiver_to_q_weight,
        perceiver_to_q_bias,
        perceiver_to_k_weight,
        perceiver_to_k_bias,
        perceiver_to_v_weight,
        perceiver_to_v_bias,
        perceiver_proj_out_weight,
        perceiver_proj_out_bias);
    for (size_t i = 0; i < pre_att.size(); ++i) {
        pre_att[i] += query[i];
    }
    auto pre_att_norm = layer_norm_rows(
        pre_att,
        input.batch * query_tokens,
        hidden_size,
        perceiver_norm_weight,
        perceiver_norm_bias,
        1.0e-5f);
    auto attn = attention_qkv_batched(
        pre_att_norm,
        input.batch,
        query_tokens,
        pre_att_norm,
        query_tokens,
        hidden_size,
        num_heads,
        perceiver_to_q_weight,
        perceiver_to_q_bias,
        perceiver_to_k_weight,
        perceiver_to_k_bias,
        perceiver_to_v_weight,
        perceiver_to_v_bias,
        perceiver_proj_out_weight,
        perceiver_proj_out_bias);
    for (size_t i = 0; i < attn.size(); ++i) {
        attn[i] += pre_att[i];
    }

    const auto speaker_cond = linear_rows(
        input.speaker_embedding,
        input.batch,
        speaker_embed_size,
        spkr_enc_weight,
        hidden_size,
        &spkr_enc_bias);
    const auto emotion_cond = linear_rows(
        input.emotion_adv,
        input.batch,
        1,
        emotion_adv_weight,
        hidden_size,
        nullptr);

    PreparedInput outputs;
    outputs.batch = input.batch;
    outputs.cond_length = 1 + query_tokens + 1;
    outputs.hidden_size = hidden_size;
    outputs.total_length = outputs.cond_length + input.text_length + input.speech_length;
    outputs.cond_embeddings.assign(static_cast<size_t>(input.batch * outputs.cond_length * hidden_size), 0.0f);
    for (int64_t b = 0; b < input.batch; ++b) {
        float * dst = outputs.cond_embeddings.data() + static_cast<ptrdiff_t>(b * outputs.cond_length * hidden_size);
        std::copy(
            speaker_cond.begin() + static_cast<ptrdiff_t>(b * hidden_size),
            speaker_cond.begin() + static_cast<ptrdiff_t>((b + 1) * hidden_size),
            dst);
        std::copy(
            attn.begin() + static_cast<ptrdiff_t>(b * query_tokens * hidden_size),
            attn.begin() + static_cast<ptrdiff_t>((b + 1) * query_tokens * hidden_size),
            dst + hidden_size);
        std::copy(
            emotion_cond.begin() + static_cast<ptrdiff_t>(b * hidden_size),
            emotion_cond.begin() + static_cast<ptrdiff_t>((b + 1) * hidden_size),
            dst + ((1 + query_tokens) * hidden_size));
    }

    std::vector<int32_t> text_positions;
    text_positions.reserve(static_cast<size_t>(input.batch * input.text_length));
    for (int64_t b = 0; b < input.batch; ++b) {
        for (int64_t t = 0; t < input.text_length; ++t) {
            text_positions.push_back(static_cast<int32_t>(t));
        }
    }
    auto text_emb = gather_rows(text_embedding_weight, weights_->text_vocab, hidden_size, input.text_tokens);
    const auto text_pos = gather_rows(text_position_weight, weights_->text_position_vocab, hidden_size, text_positions);
    if (input.guidance_scale > 0.0f && input.batch > 1) {
        std::fill(
            text_emb.begin() + static_cast<ptrdiff_t>(input.text_length * hidden_size),
            text_emb.begin() + static_cast<ptrdiff_t>(2 * input.text_length * hidden_size),
            0.0f);
    }
    for (size_t i = 0; i < text_emb.size(); ++i) {
        text_emb[i] += text_pos[i];
    }
    outputs.text_embeddings = std::move(text_emb);

    std::vector<int32_t> speech_positions;
    speech_positions.reserve(static_cast<size_t>(input.batch * input.speech_length));
    for (int64_t b = 0; b < input.batch; ++b) {
        for (int64_t t = 0; t < input.speech_length; ++t) {
            speech_positions.push_back(static_cast<int32_t>(t));
        }
    }
    auto speech_emb = gather_rows(speech_embedding_weight, weights_->speech_vocab, hidden_size, input.speech_tokens);
    const auto speech_pos = gather_rows(speech_position_weight, weights_->speech_position_vocab, hidden_size, speech_positions);
    for (size_t i = 0; i < speech_emb.size(); ++i) {
        speech_emb[i] += speech_pos[i];
    }
    outputs.speech_embeddings = std::move(speech_emb);

    outputs.combined_embeddings.assign(static_cast<size_t>(input.batch * outputs.total_length * hidden_size), 0.0f);
    for (int64_t b = 0; b < input.batch; ++b) {
        float * dst = outputs.combined_embeddings.data() + static_cast<ptrdiff_t>(b * outputs.total_length * hidden_size);
        std::copy(
            outputs.cond_embeddings.begin() + static_cast<ptrdiff_t>(b * outputs.cond_length * hidden_size),
            outputs.cond_embeddings.begin() + static_cast<ptrdiff_t>((b + 1) * outputs.cond_length * hidden_size),
            dst);
        std::copy(
            outputs.text_embeddings.begin() + static_cast<ptrdiff_t>(b * input.text_length * hidden_size),
            outputs.text_embeddings.begin() + static_cast<ptrdiff_t>((b + 1) * input.text_length * hidden_size),
            dst + outputs.cond_length * hidden_size);
        std::copy(
            outputs.speech_embeddings.begin() + static_cast<ptrdiff_t>(b * input.speech_length * hidden_size),
            outputs.speech_embeddings.begin() + static_cast<ptrdiff_t>((b + 1) * input.speech_length * hidden_size),
            dst + (outputs.cond_length + input.text_length) * hidden_size);
    }
    return outputs;
}

T3GenerateOutputs T3InferenceComponent::generate_speech_tokens(const T3GenerateRequest & request) const {
    constexpr int32_t kStartSpeechToken = 6561;
    constexpr int32_t kStopSpeechToken = 6562;
    constexpr int64_t kDecodeCapacityChunk = 256;
    const auto & backend_config = execution_context_->config();
    set_t3_host_thread_count(static_cast<size_t>(std::max(1, backend_config.threads)));
    const int64_t hidden_size = weights_->hidden_size;
    const int64_t speech_vocab = weights_->speech_vocab;
    const int64_t num_heads = weights_->num_heads;
    const bool use_cfg = request.guidance_scale > 0.0f;
    const int64_t batch = use_cfg ? 2 : 1;
    TorchMt19937 rng(request.seed);

    std::vector<int32_t> generated_ids = request.initial_speech_tokens;
    if (generated_ids.empty()) {
        generated_ids.push_back(kStartSpeechToken);
    }

    T3GenerateOutputs outputs;
    outputs.predicted_tokens.reserve(static_cast<size_t>(request.max_new_tokens));

    const auto batched_cond_prompt_tokens = repeat_tokens(request.cond_prompt_speech_tokens, batch);
    const auto batched_speaker_embedding = repeat_values(request.speaker_embedding, batch);
    const auto batched_emotion_adv = repeat_values(request.emotion_adv, batch);
    const auto batched_text_tokens = repeat_tokens(request.text_tokens, batch);
    const auto batched_speech_tokens = repeat_tokens(generated_ids, batch);
    const int64_t text_length = static_cast<int64_t>(request.text_tokens.size());
    const int64_t speech_length = static_cast<int64_t>(generated_ids.size());
    const int64_t dynamic_length = text_length + speech_length + 1;

    State::PrefixCache prefix_cache;
    bool prefix_cache_hit = false;
    double prefix_cache_build_ms = 0.0;
    std::shared_ptr<T3DecodeBackendRunner> runner;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        prefix_cache_hit = state_->prefix_cache.has_value() &&
            state_->prefix_cache->batch == batch &&
            state_->prefix_cache->speaker_embedding == batched_speaker_embedding &&
            state_->prefix_cache->cond_prompt_speech_tokens == batched_cond_prompt_tokens &&
            state_->prefix_cache->emotion_adv == batched_emotion_adv;
        if (!prefix_cache_hit) {
            const auto prefix_build_started = std::chrono::steady_clock::now();
            ConditioningInput prefix_case;
            prefix_case.batch = batch;
            prefix_case.cond_prompt_length = static_cast<int64_t>(request.cond_prompt_speech_tokens.size());
            prefix_case.text_length = 0;
            prefix_case.speech_length = 0;
            prefix_case.guidance_scale = 0.0f;
            prefix_case.cond_prompt_speech_tokens = batched_cond_prompt_tokens;
            prefix_case.speaker_embedding = batched_speaker_embedding;
            prefix_case.emotion_adv = batched_emotion_adv;

            auto prefix_prepared = prepare_conditioning_inputs(prefix_case);
            runner = state_->get_runner(batch, prefix_prepared.cond_length, backend_config, *weights_);
            T3BackendCacheState initial_cache;
            initial_cache.batch = batch;
            initial_cache.hidden_size = hidden_size;
            initial_cache.num_heads = num_heads;
            initial_cache.head_dim = hidden_size / num_heads;
            initial_cache.steps = 0;
            initial_cache.layers = std::vector<T3BackendLayerState>(weights_->layers.size());
            runner->import_state(std::move(initial_cache));
            runner->set_capture_cache_state(true);
            for (int64_t pos = 0; pos < prefix_prepared.cond_length; ++pos) {
                runner->step(
                    extract_step_embeddings(
                        prefix_prepared.cond_embeddings,
                        batch,
                        prefix_prepared.cond_length,
                        hidden_size,
                        pos),
                    pos);
            }
            runner->set_capture_cache_state(false);
            State::PrefixCache new_prefix_cache;
            new_prefix_cache.batch = batch;
            new_prefix_cache.speaker_embedding = batched_speaker_embedding;
            new_prefix_cache.cond_prompt_speech_tokens = batched_cond_prompt_tokens;
            new_prefix_cache.emotion_adv = batched_emotion_adv;
            new_prefix_cache.cond_embeddings = std::move(prefix_prepared.cond_embeddings);
            new_prefix_cache.cond_length = prefix_prepared.cond_length;
            new_prefix_cache.cache = runner->export_state();
            state_->prefix_cache = std::move(new_prefix_cache);
            prefix_cache_build_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefix_build_started).count();
        }
        prefix_cache = *state_->prefix_cache;
    }

    const auto text_embedding_weight = engine::assets::tensor_data_to_f32("text_emb.weight", weights_->text_embedding_weight);
    const auto speech_embedding_weight = engine::assets::tensor_data_to_f32("speech_emb.weight", weights_->speech_embedding_weight);
    const auto text_position_weight = engine::assets::tensor_data_to_f32(
        "text_pos_emb.emb.weight",
        weights_->text_position_weight);
    const auto speech_position_weight = engine::assets::tensor_data_to_f32(
        "speech_pos_emb.emb.weight",
        weights_->speech_position_weight);

    const auto text_embeddings = build_positioned_token_embeddings(
        text_embedding_weight,
        weights_->text_vocab,
        text_position_weight,
        hidden_size,
        batched_text_tokens,
        batch,
        use_cfg);
    const auto speech_embeddings = build_positioned_token_embeddings(
        speech_embedding_weight,
        weights_->speech_vocab,
        speech_position_weight,
        hidden_size,
        batched_speech_tokens,
        batch,
        false);

    std::vector<float> sequence_embeddings(static_cast<size_t>(batch * dynamic_length * hidden_size), 0.0f);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        float * dst = sequence_embeddings.data() + static_cast<ptrdiff_t>(batch_index * dynamic_length * hidden_size);
        if (text_length > 0) {
            const float * text_src = text_embeddings.data() + static_cast<ptrdiff_t>(batch_index * text_length * hidden_size);
            std::copy(text_src, text_src + static_cast<ptrdiff_t>(text_length * hidden_size), dst);
        }
        const float * speech_src = speech_embeddings.data() + static_cast<ptrdiff_t>(batch_index * speech_length * hidden_size);
        std::copy(
            speech_src,
            speech_src + static_cast<ptrdiff_t>(speech_length * hidden_size),
            dst + static_cast<ptrdiff_t>(text_length * hidden_size));
        const float * token_emb = speech_embedding_weight.data() + static_cast<ptrdiff_t>(kStartSpeechToken * hidden_size);
        const float * pos_emb = speech_position_weight.data();
        float * bos_dst = dst + static_cast<ptrdiff_t>((text_length + speech_length) * hidden_size);
        for (int64_t i = 0; i < hidden_size; ++i) {
            bos_dst[i] = token_emb[i] + pos_emb[i];
        }
    }

    const int64_t total_length = prefix_cache.cond_length + text_length + speech_length;
    std::vector<float> last_logits;
    auto step_started = std::chrono::steady_clock::now();
    T3PrefillBackendOutput prefill_output;
    {
        std::shared_ptr<T3PrefillBackendRunner> prefill_runner;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            prefill_runner = state_->get_prefill_runner(
                batch,
                prefix_cache.cond_length,
                dynamic_length,
                backend_config,
                *weights_);
        }
        prefill_output = prefill_runner->run(sequence_embeddings, prefix_cache.cache);
    }
    outputs.prefill_runner_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - step_started).count();
    last_logits = std::move(prefill_output.logits);
    const int64_t prefill_cache_steps = total_length + 1;
    const int64_t max_decode_cache_steps = prefill_cache_steps + request.max_new_tokens;
    const int64_t initial_decode_cache_steps =
        prefill_cache_steps + std::min<int64_t>(request.max_new_tokens, kDecodeCapacityChunk);
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        runner = state_->get_runner(batch, initial_decode_cache_steps, backend_config, *weights_);
    }
    const auto decoder_cache_clone_started = std::chrono::steady_clock::now();
    runner->import_state(prefill_output.cache);
    outputs.decoder_cache_clone_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - decoder_cache_clone_started).count();

    std::vector<float> logits(static_cast<size_t>(speech_vocab), 0.0f);
    std::vector<float> sampling_probs;
    std::vector<size_t> sampling_order;
    std::vector<uint8_t> sampling_mask;
    for (int64_t step = 0; step < request.max_new_tokens; ++step) {
        std::copy_n(last_logits.data(), speech_vocab, logits.data());
        if (use_cfg) {
            for (int64_t i = 0; i < speech_vocab; ++i) {
                const float cond = last_logits[static_cast<size_t>(i)];
                const float uncond = last_logits[static_cast<size_t>(speech_vocab + i)];
                logits[static_cast<size_t>(i)] = cond + request.guidance_scale * (cond - uncond);
            }
        }
        auto sampling_started = std::chrono::steady_clock::now();
        apply_repetition_penalty_in_place(logits, generated_ids, request.repetition_penalty, sampling_mask);
        if (request.do_sample) {
            if (request.temperature != 1.0f) {
                for (float & value : logits) {
                    value /= request.temperature;
                }
            }
            apply_min_p_in_place(logits, request.min_p, sampling_probs);
            apply_top_p_in_place(logits, request.top_p, sampling_probs, sampling_order, sampling_mask);
        }
        const int32_t next_token = sample_from_logits(logits, request.do_sample, rng);
        outputs.sampling_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - sampling_started).count();
        generated_ids.push_back(next_token);

        if (request.stop_on_eos && next_token == kStopSpeechToken) {
            outputs.hit_eos = true;
            break;
        }
        auto next_embed_started = std::chrono::steady_clock::now();
        std::vector<float> next_embed(static_cast<size_t>(batch * hidden_size), 0.0f);
        for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
            const float * token_emb = speech_embedding_weight.data() + static_cast<ptrdiff_t>(next_token * hidden_size);
            const float * pos_emb = speech_position_weight.data() + static_cast<ptrdiff_t>(step + 1) * hidden_size;
            float * dst = next_embed.data() + static_cast<ptrdiff_t>(batch_index * hidden_size);
            for (int64_t i = 0; i < hidden_size; ++i) {
                dst[i] = token_emb[i] + pos_emb[i];
            }
        }
        outputs.next_embed_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - next_embed_started).count();
        step_started = std::chrono::steady_clock::now();
        if (runner->valid_steps() >= runner->cache_capacity_steps()) {
            const int64_t next_capacity = std::min<int64_t>(
                max_decode_cache_steps,
                runner->cache_capacity_steps() + kDecodeCapacityChunk);
            auto cache_state = runner->export_state_from_device();
            {
                std::lock_guard<std::mutex> lock(state_->mutex);
                runner = state_->get_runner(batch, next_capacity, backend_config, *weights_);
            }
            runner->import_state(cache_state);
        }
        last_logits = runner->step(next_embed, total_length + 1 + step);
        outputs.decode_runner_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - step_started).count();
    }

    const size_t initial_count = request.initial_speech_tokens.empty() ? 1 : request.initial_speech_tokens.size();
    if (generated_ids.size() > initial_count) {
        outputs.predicted_tokens.assign(generated_ids.begin() + static_cast<ptrdiff_t>(initial_count), generated_ids.end());
    } else {
        outputs.predicted_tokens.clear();
    }
    outputs.token_count = static_cast<int64_t>(outputs.predicted_tokens.size());
    outputs.prefix_cache_build_ms = prefix_cache_build_ms;
    return outputs;
}

void T3InferenceComponent::release_runtime_graphs() const {
    state_->release_runtime_graphs();
}

void T3InferenceComponent::release_runtime_cache() const {
    state_->release_runtime_cache();
}

}  // namespace engine::models::chatterbox
