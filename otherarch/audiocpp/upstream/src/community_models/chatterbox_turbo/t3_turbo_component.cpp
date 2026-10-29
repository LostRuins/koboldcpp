#include "components/t3_turbo_runtime.h"

#include <chrono>
#include <mutex>

namespace engine::community_models::chatterbox_turbo {

struct T3TurboInferenceComponent::State {
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

    std::shared_ptr<T3TurboBackendOwner> get_owner(
        const engine::core::BackendConfig & backend,
        const T3TurboInferenceWeights & weights) {
        if (!owner || !same_backend(owner->config(), backend)) {
            owner = std::make_shared<T3TurboBackendOwner>(weights, backend);
            runner.reset();
            prefill_runner.reset();
            prefix_cache.reset();
        }
        return owner;
    }

    struct PrefixCache {
        std::vector<float> speaker_embedding;
        std::vector<int32_t> cond_prompt_speech_tokens;
        int64_t cond_length = 0;
        T3TurboCacheState cache;
    };

    std::shared_ptr<T3TurboDecodeBackendRunner> get_runner(
        int64_t cache_steps,
        const engine::core::BackendConfig & backend,
        const T3TurboInferenceWeights & weights) {
        if (runner && runner->matches(cache_steps, backend)) {
            return runner;
        }
        auto held_owner = get_owner(backend, weights);
        runner.reset();
        runner = std::make_shared<T3TurboDecodeBackendRunner>(weights, cache_steps, std::move(held_owner));
        return runner;
    }

    std::shared_ptr<T3TurboPrefillBackendRunner> get_prefill_runner(
        int64_t prefix_steps,
        int64_t seq_len,
        const engine::core::BackendConfig & backend,
        const T3TurboInferenceWeights & weights) {
        if (prefill_runner && prefill_runner->matches(prefix_steps, seq_len, backend)) {
            return prefill_runner;
        }
        auto held_owner = get_owner(backend, weights);
        prefill_runner.reset();
        prefill_runner = std::make_shared<T3TurboPrefillBackendRunner>(weights, prefix_steps, seq_len, std::move(held_owner));
        return prefill_runner;
    }

    mutable std::mutex mutex;
    std::shared_ptr<T3TurboBackendOwner> owner;
    std::optional<PrefixCache> prefix_cache;
    std::shared_ptr<T3TurboPrefillBackendRunner> prefill_runner;
    std::shared_ptr<T3TurboDecodeBackendRunner> runner;
};

T3TurboInferenceComponent::T3TurboInferenceComponent(
    std::shared_ptr<const T3TurboInferenceWeights> weights,
    const engine::core::ExecutionContext & execution_context)
    : weights_(std::move(weights)), execution_context_(&execution_context), state_(std::make_shared<State>()) {
    if (!weights_) {
        throw std::runtime_error("T3TurboInferenceComponent requires weights");
    }
}

namespace {

// cond_embeds = [ spkr_proj(1 token), speech_emb(cond_prompt_speech_tokens) (speech_cond_prompt_len tokens) ]
// No perceiver resampler, no emotion conditioning, no per-segment position embeddings: Chatterbox
// Turbo's GPT2 backbone applies one absolute "wpe" position embedding across the whole combined
// [cond; text; speech] sequence, added host-side below (see t3.py prepare_input_embeds with
// hp.input_pos_emb=None: HF's GPT2Model applies its own wpe once over inputs_embeds).
std::vector<float> build_cond_embeddings(
    const T3TurboInferenceWeights & weights,
    const std::vector<float> & speaker_embedding,
    const std::vector<int32_t> & cond_prompt_speech_tokens) {
    const int64_t hidden = weights.hidden_size;
    const auto spkr_enc_weight = engine::assets::tensor_data_to_f32("cond.spkr_enc.weight", weights.spkr_enc_weight);
    const auto spkr_enc_bias = engine::assets::tensor_data_to_f32("cond.spkr_enc.bias", weights.spkr_enc_bias);
    const auto speech_embedding_weight =
        engine::assets::tensor_data_to_f32("speech_emb.weight", weights.speech_embedding_weight);

    std::vector<float> spkr_proj(static_cast<size_t>(hidden), 0.0f);
    for (int64_t out_index = 0; out_index < hidden; ++out_index) {
        double value = static_cast<double>(spkr_enc_bias[static_cast<size_t>(out_index)]);
        const float * w = spkr_enc_weight.data() + static_cast<ptrdiff_t>(out_index * weights.speaker_embed_size);
        for (int64_t in_index = 0; in_index < weights.speaker_embed_size; ++in_index) {
            value += static_cast<double>(w[in_index]) * static_cast<double>(speaker_embedding[static_cast<size_t>(in_index)]);
        }
        spkr_proj[static_cast<size_t>(out_index)] = static_cast<float>(value);
    }

    const auto cond_prompt_emb = gather_rows(speech_embedding_weight, weights.speech_vocab, hidden, cond_prompt_speech_tokens);

    std::vector<float> cond_embeddings;
    cond_embeddings.reserve(spkr_proj.size() + cond_prompt_emb.size());
    cond_embeddings.insert(cond_embeddings.end(), spkr_proj.begin(), spkr_proj.end());
    cond_embeddings.insert(cond_embeddings.end(), cond_prompt_emb.begin(), cond_prompt_emb.end());
    return cond_embeddings;
}

void add_wpe_in_place(
    std::vector<float> & embeddings,
    const std::vector<float> & wpe_weight,
    int64_t hidden,
    int64_t start_position) {
    const int64_t seq_len = static_cast<int64_t>(embeddings.size()) / hidden;
    for (int64_t t = 0; t < seq_len; ++t) {
        const float * pos_row = wpe_weight.data() + static_cast<ptrdiff_t>((start_position + t) * hidden);
        float * dst = embeddings.data() + static_cast<ptrdiff_t>(t * hidden);
        for (int64_t i = 0; i < hidden; ++i) {
            dst[i] += pos_row[i];
        }
    }
}

}  // namespace

T3TurboGenerateOutputs T3TurboInferenceComponent::generate_speech_tokens(const T3TurboGenerateRequest & request) const {
    const auto & backend_config = execution_context_->config();
    const int64_t hidden_size = weights_->hidden_size;
    const int64_t speech_vocab = weights_->speech_vocab;
    TurboMt19937 rng(request.seed == 0 ? 0x2A2A2A2AU : request.seed);

    const auto wpe_weight = engine::assets::tensor_data_to_f32("wpe.weight", weights_->wpe_weight);
    const auto text_embedding_weight =
        engine::assets::tensor_data_to_f32("text_emb.weight", weights_->text_embedding_weight);
    const auto speech_embedding_weight =
        engine::assets::tensor_data_to_f32("speech_emb.weight", weights_->speech_embedding_weight);

    std::vector<int32_t> generated_ids = request.initial_speech_tokens;
    if (generated_ids.empty()) {
        generated_ids.push_back(kTurboStartSpeechToken);
    }

    T3TurboGenerateOutputs outputs;
    outputs.predicted_tokens.reserve(static_cast<size_t>(request.max_new_tokens));

    State::PrefixCache prefix_cache;
    double prefix_cache_build_ms = 0.0;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const bool prefix_cache_hit = state_->prefix_cache.has_value() &&
            state_->prefix_cache->speaker_embedding == request.speaker_embedding &&
            state_->prefix_cache->cond_prompt_speech_tokens == request.cond_prompt_speech_tokens;
        if (!prefix_cache_hit) {
            const auto started = std::chrono::steady_clock::now();
            auto cond_embeddings =
                build_cond_embeddings(*weights_, request.speaker_embedding, request.cond_prompt_speech_tokens);
            const int64_t cond_length = static_cast<int64_t>(cond_embeddings.size()) / hidden_size;
            add_wpe_in_place(cond_embeddings, wpe_weight, hidden_size, 0);

            auto runner = state_->get_runner(cond_length, backend_config, *weights_);
            T3TurboCacheState initial_cache;
            initial_cache.hidden_size = hidden_size;
            initial_cache.num_heads = weights_->num_heads;
            initial_cache.head_dim = hidden_size / weights_->num_heads;
            initial_cache.layers = std::vector<T3TurboLayerCacheState>(weights_->layers.size());
            runner->import_state(initial_cache);
            runner->set_capture_cache_state(true);
            for (int64_t pos = 0; pos < cond_length; ++pos) {
                std::vector<float> step(
                    cond_embeddings.begin() + static_cast<ptrdiff_t>(pos * hidden_size),
                    cond_embeddings.begin() + static_cast<ptrdiff_t>((pos + 1) * hidden_size));
                runner->step(step, pos);
            }
            runner->set_capture_cache_state(false);

            State::PrefixCache new_cache;
            new_cache.speaker_embedding = request.speaker_embedding;
            new_cache.cond_prompt_speech_tokens = request.cond_prompt_speech_tokens;
            new_cache.cond_length = cond_length;
            new_cache.cache = runner->export_state();
            state_->prefix_cache = std::move(new_cache);
            prefix_cache_build_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        }
        prefix_cache = *state_->prefix_cache;
    }

    const int64_t text_length = static_cast<int64_t>(request.text_tokens.size());
    const int64_t speech_length = static_cast<int64_t>(generated_ids.size());
    const int64_t dynamic_length = text_length + speech_length;

    std::vector<float> dynamic_embeddings(static_cast<size_t>(dynamic_length * hidden_size), 0.0f);
    if (text_length > 0) {
        const auto text_emb = gather_rows(text_embedding_weight, weights_->text_vocab, hidden_size, request.text_tokens);
        std::copy(text_emb.begin(), text_emb.end(), dynamic_embeddings.begin());
    }
    {
        const auto speech_emb = gather_rows(speech_embedding_weight, weights_->speech_vocab, hidden_size, generated_ids);
        std::copy(speech_emb.begin(), speech_emb.end(), dynamic_embeddings.begin() + static_cast<ptrdiff_t>(text_length * hidden_size));
    }
    add_wpe_in_place(dynamic_embeddings, wpe_weight, hidden_size, prefix_cache.cond_length);

    auto step_started = std::chrono::steady_clock::now();
    T3TurboPrefillOutput prefill_output;
    {
        std::shared_ptr<T3TurboPrefillBackendRunner> prefill_runner;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            prefill_runner = state_->get_prefill_runner(prefix_cache.cond_length, dynamic_length, backend_config, *weights_);
        }
        prefill_output = prefill_runner->run(dynamic_embeddings, prefix_cache.cache);
    }
    outputs.prefill_runner_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - step_started).count();

    std::vector<float> last_logits = std::move(prefill_output.logits);
    const int64_t prefill_cache_steps = prefix_cache.cond_length + dynamic_length;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto runner = state_->get_runner(prefill_cache_steps + request.max_new_tokens, backend_config, *weights_);
        runner->import_state(prefill_output.cache);
    }

    std::vector<float> logits(static_cast<size_t>(speech_vocab), 0.0f);
    std::vector<float> sampling_probs;
    std::vector<size_t> sampling_order;
    std::vector<uint8_t> sampling_mask;
    for (int64_t step = 0; step < request.max_new_tokens; ++step) {
        std::copy_n(last_logits.data(), speech_vocab, logits.data());
        apply_repetition_penalty_in_place(logits, generated_ids, request.repetition_penalty, sampling_mask);
        if (request.do_sample) {
            if (request.temperature != 1.0f && request.temperature > 0.0f) {
                for (float & value : logits) {
                    value /= request.temperature;
                }
            }
            apply_top_k_in_place(logits, request.top_k);
            apply_top_p_in_place(logits, request.top_p, sampling_probs, sampling_order, sampling_mask);
        }
        const int32_t next_token = sample_from_logits(logits, request.do_sample, rng);
        generated_ids.push_back(next_token);

        if (request.stop_on_eos && next_token == kTurboStopSpeechToken) {
            outputs.hit_eos = true;
            break;
        }

        std::vector<float> next_embed(speech_embedding_weight.begin() + static_cast<ptrdiff_t>(next_token * hidden_size),
            speech_embedding_weight.begin() + static_cast<ptrdiff_t>((next_token + 1) * hidden_size));
        const float * pos_row = wpe_weight.data() +
            static_cast<ptrdiff_t>((prefill_cache_steps + step) * hidden_size);
        for (int64_t i = 0; i < hidden_size; ++i) {
            next_embed[static_cast<size_t>(i)] += pos_row[i];
        }

        step_started = std::chrono::steady_clock::now();
        std::shared_ptr<T3TurboDecodeBackendRunner> runner;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            runner = state_->runner;
        }
        last_logits = runner->step(next_embed, prefill_cache_steps + step);
        outputs.decode_runner_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - step_started).count();
    }

    const size_t initial_count = request.initial_speech_tokens.empty() ? 1 : request.initial_speech_tokens.size();
    if (generated_ids.size() > initial_count) {
        outputs.predicted_tokens.assign(generated_ids.begin() + static_cast<ptrdiff_t>(initial_count), generated_ids.end());
    }
    outputs.token_count = static_cast<int64_t>(outputs.predicted_tokens.size());
    outputs.prefix_cache_build_ms = prefix_cache_build_ms;
    return outputs;
}

void T3TurboInferenceComponent::release_runtime_graphs() const {
    state_->release_runtime_graphs();
}

void T3TurboInferenceComponent::release_runtime_cache() const {
    state_->release_runtime_cache();
}

}  // namespace engine::community_models::chatterbox_turbo
