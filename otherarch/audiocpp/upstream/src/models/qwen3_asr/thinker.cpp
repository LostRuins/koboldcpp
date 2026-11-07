#include "engine/models/qwen3_asr/thinker.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/greedy_decode.h"


#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::models::qwen3_asr {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct TextLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct ThinkerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<TextLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue lm_head;
};

struct PrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

modules::DecoderLayerWeights to_qwen3_layer_weights(const TextLayerWeights & weights) {
    modules::DecoderLayerWeights out;
    out.input_norm = {weights.input_norm, std::nullopt};
    out.self_attention.q_weight = weights.q_proj;
    out.self_attention.k_weight = weights.k_proj;
    out.self_attention.v_weight = weights.v_proj;
    out.self_attention.out_weight = weights.o_proj;
    out.q_norm = {weights.q_norm, std::nullopt};
    out.k_norm = {weights.k_norm, std::nullopt};
    out.post_norm = {weights.post_norm, std::nullopt};
    out.mlp.gate_proj = {weights.gate_proj, std::nullopt};
    out.mlp.up_proj = {weights.up_proj, std::nullopt};
    out.mlp.down_proj = {weights.down_proj, std::nullopt};
    return out;
}

modules::CausalDecoderConfig make_qwen3_decoder_config(
    const Qwen3ASRTextDecoderConfig & config,
    modules::CausalDecoderLogitsMode logits_mode) {
    modules::CausalDecoderConfig out;
    out.stack.hidden_size = config.hidden_size;
    out.stack.num_attention_heads = config.num_attention_heads;
    out.stack.num_key_value_heads = config.num_key_value_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.num_hidden_layers;
    out.stack.rms_norm_eps = config.rms_norm_eps;
    out.stack.rope_theta = config.rope_theta;
    out.stack.use_qk_norm = true;
    out.stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_size = config.output_size;
    out.logits_mode = logits_mode;
    return out;
}

modules::CausalDecoderWeights make_qwen3_decoder_weights(const ThinkerWeights & weights) {
    modules::CausalDecoderWeights out;
    out.stack.layers.reserve(weights.layers.size());
    for (const auto & layer : weights.layers) {
        out.stack.layers.push_back(to_qwen3_layer_weights(layer));
    }
    out.final_norm = {weights.norm, std::nullopt};
    out.lm_head = {weights.lm_head, std::nullopt};
    return out;
}

int64_t head_dim(const Qwen3ASRTextDecoderConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("Qwen3 ASR thinker attention config is invalid");
    }
    return config.head_dim;
}

core::TensorValue prompt_embeddings(
    core::ModuleBuildContext & ctx,
    const ThinkerWeights & weights,
    const Qwen3ASRTextDecoderConfig & config,
    ggml_tensor * token_ids,
    ggml_tensor * audio_embeddings,
    ggml_tensor * audio_positions,
    int64_t prompt_steps,
    int64_t audio_tokens) {
    auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(ctx, ids, weights.token_embedding);
    if (audio_tokens > 0) {
        auto audio = core::wrap_tensor(
            audio_embeddings,
            core::TensorShape::from_dims({audio_tokens, config.hidden_size}),
            GGML_TYPE_F32);
        auto positions = core::wrap_tensor(
            audio_positions,
            core::TensorShape::from_dims({audio_tokens}),
            GGML_TYPE_I64);
        x = core::wrap_tensor(
            ggml_set_rows(ctx.ggml, x.tensor, audio.tensor, positions.tensor),
            x.shape,
            GGML_TYPE_F32);
    }
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, prompt_steps, config.hidden_size}));
}

ThinkerWeights load_weights(
    const Qwen3ASRAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.text_decoder;
    const auto & source = *assets.model_weights;
    const std::string model_prefix = assets.config.hf_transformers_layout
        ? "model.language_model"
        : "thinker.model";
    ThinkerWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "qwen3_asr.thinker.weights",
        weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        model_prefix + ".embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    const int64_t dim = head_dim(config);
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = model_prefix + ".layers." + std::to_string(layer);
        TextLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
        w.q_proj = weights.store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.num_attention_heads * dim, config.hidden_size});
        w.k_proj = weights.store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        w.v_proj = weights.store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        w.o_proj = weights.store->load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.hidden_size, config.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
        w.gate_proj = weights.store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.up_proj = weights.store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.down_proj = weights.store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, model_prefix + ".norm.weight", {config.hidden_size});
    const std::string lm_head_name = assets.config.hf_transformers_layout
        ? "lm_head.weight"
        : "thinker.lm_head.weight";
    // Fine-tuned checkpoints saved with tied embeddings (e.g. QwenCleo-ASR) can omit
    // thinker.lm_head.weight in the legacy layout; the official checkpoints still ship it.
    const bool tie_output = assets.config.tie_word_embeddings &&
        (assets.config.hf_transformers_layout || !source.has_tensor(lm_head_name));
    if (tie_output) {
        if (config.output_size != config.vocab_size) {
            throw std::runtime_error("Qwen3 ASR tied output embedding requires output_size == vocab_size");
        }
        weights.lm_head = weights.token_embedding;
    } else {
        weights.lm_head = weights.store->load_tensor(
            source,
            lm_head_name,
            storage_type,
            {config.output_size, config.hidden_size});
    }
    weights.store->upload();
    return weights;
}

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("Qwen3 ASR thinker cannot select from empty logits");
    }
    size_t best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

bool is_eos(const Qwen3ASRTextDecoderConfig & config, int32_t token) {
    return std::find(config.eos_token_ids.begin(), config.eos_token_ids.end(), static_cast<int64_t>(token)) !=
        config.eos_token_ids.end();
}

class ThinkerWeightsRuntime {
public:
    ThinkerWeightsRuntime(
        std::shared_ptr<const Qwen3ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<ThinkerWeights>(load_weights(*assets_, backend_, backend_type_, weight_context_bytes, storage_type))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Qwen3 ASR thinker weights runtime requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Qwen3 ASR thinker backend is not initialized");
        }
    }

    const Qwen3ASRAssets & assets() const noexcept {
        return *assets_;
    }

    const ThinkerWeights & weights() const noexcept {
        return *weights_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int threads() const noexcept {
        return threads_;
    }

private:
    std::shared_ptr<const Qwen3ASRAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const ThinkerWeights> weights_;
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

class PrefillGraph {
public:
    PrefillGraph(
        std::shared_ptr<ThinkerWeightsRuntime> runtime,
        int64_t prompt_steps,
        int64_t audio_tokens,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          audio_tokens_(audio_tokens) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("Qwen3 ASR thinker prefill requires positive prompt length");
        }
        if (audio_tokens_ < 0 || audio_tokens_ > prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker prefill audio token count is invalid");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 ASR thinker prefill graph context");
        }
        const auto & config = runtime_->assets().config.text_decoder;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "qwen3_asr.thinker.prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        audio_embeddings_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, std::max<int64_t>(audio_tokens_, 1));
        audio_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I64, std::max<int64_t>(audio_tokens_, 1));
        auto x = prompt_embeddings(
            ctx,
            weights,
            config,
            token_ids_,
            audio_embeddings_,
            audio_positions_,
            prompt_steps_,
            audio_tokens_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        const auto decoder_weights = make_qwen3_decoder_weights(weights);
        auto decoder_out = modules::CausalDecoderModule(
                               make_qwen3_decoder_config(config, modules::CausalDecoderLogitsMode::LastStep))
                               .build(ctx, x, positions, decoder_weights);
        for (const auto & layer : decoder_out.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("Qwen3 ASR thinker prefill decoder did not return K/V state");
            }
            // The graph allocator recycles intermediates, and the decoder K/V is an
            // intermediate that run() has to read back afterwards. Copy each one into
            // a tensor of its own and mark it as a graph output, which is what keeps
            // it off the reuse list. Same shape as CausalDecoderRuntime prefill.
            auto * key = ggml_cpy(
                ctx_.get(),
                layer.key->tensor,
                ggml_dup_tensor(ctx_.get(), layer.key->tensor));
            auto * value = ggml_cpy(
                ctx_.get(),
                layer.value->tensor,
                ggml_dup_tensor(ctx_.get(), layer.value->tensor));
            ggml_set_output(key);
            ggml_set_output(value);
            keys_.push_back(key);
            values_.push_back(value);
        }
        logits_ = decoder_out.logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        for (auto * key : keys_) {
            ggml_build_forward_expand(graph_, key);
        }
        for (auto * value : values_) {
            ggml_build_forward_expand(graph_, value);
        }
        // unique_ptr so a throw below (or from later ctor statements) frees
        // whatever the allocator already reserved; a throwing constructor
        // runs no destructor, and each failed rebuild used to leak its
        // partially reserved arena
        const auto try_alloc = [&]() {
            gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend())));
            return gallocr_ != nullptr &&
                ggml_gallocr_reserve(gallocr_.get(), graph_) &&
                ggml_gallocr_alloc_graph(gallocr_.get(), graph_);
        };
        // On failure the device may simply be full of idle cached pool
        // buffers (the legacy pool never shrinks on its own); trim and retry
        // once before declaring the size impossible
        if (!try_alloc() &&
            (engine::core::trim_backend_pools(runtime_->backend()), !try_alloc())) {
            // Size, not a fault: the graph scales with prompt_steps_, which the
            // caller controls through the transcription prompt and the length of
            // the audio. Say which, and by how much, so the remedy is obvious.
            throw engine::runtime::CapacityError(
                "Qwen3 ASR prefill graph does not fit in device memory at this size ("
                + std::to_string(prompt_steps_) + " prompt steps, of which "
                + std::to_string(audio_tokens_) + " are audio tokens); "
                "shorten the transcription prompt or the audio");
        }
        position_ids_ = modules::decoder_position_ids(prompt_steps_);
        debug::timing_log_scalar("qwen3_asr.thinker.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("qwen3_asr.thinker.prefill_prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
    }

    bool matches(const ThinkerWeightsRuntime & runtime, int64_t prompt_steps, int64_t audio_tokens) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps && audio_tokens_ == audio_tokens;
    }

    PrefillOutput run(
        const std::vector<int32_t> & token_ids,
        const std::vector<float> & audio_embeddings,
        const std::vector<int32_t> & audio_positions) {
        const auto & config = runtime_->assets().config.text_decoder;
        if (static_cast<int64_t>(token_ids.size()) != prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker prefill token id count mismatch");
        }
        if (static_cast<int64_t>(audio_embeddings.size()) != audio_tokens_ * config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR thinker prefill audio embedding size mismatch");
        }
        if (static_cast<int64_t>(audio_positions.size()) != audio_tokens_) {
            throw std::runtime_error("Qwen3 ASR thinker prefill audio position count mismatch");
        }
        auto timing_start = Clock::now();
        // Re-uploaded on every run: the graph allocator may hand this leaf out to a
        // later node once the graph has consumed it, so it cannot be written once at
        // build time and left alone.
        ggml_backend_tensor_set(positions_, position_ids_.data(), 0, position_ids_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (audio_tokens_ > 0) {
            std::vector<int64_t> positions(audio_positions.begin(), audio_positions.end());
            ggml_backend_tensor_set(
                audio_embeddings_,
                audio_embeddings.data(),
                0,
                audio_embeddings.size() * sizeof(float));
            ggml_backend_tensor_set(
                audio_positions_,
                positions.data(),
                0,
                positions.size() * sizeof(int64_t));
        }
        debug::timing_log_scalar("qwen3_asr.thinker.prefill_input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar("qwen3_asr.thinker.prefill.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 ASR thinker prefill graph compute failed");
        }
        PrefillOutput out;
        out.logits.resize(static_cast<size_t>(config.output_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps_ * config.num_key_value_heads * head_dim(config));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        debug::timing_log_scalar("qwen3_asr.thinker.prefill_output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    int64_t audio_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * audio_embeddings_ = nullptr;
    ggml_tensor * audio_positions_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<int32_t> position_ids_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

class PromptClassificationGraph {
public:
    PromptClassificationGraph(
        std::shared_ptr<ThinkerWeightsRuntime> runtime,
        int64_t prompt_steps,
        int64_t audio_tokens,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          audio_tokens_(audio_tokens) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("Qwen3 ASR thinker classification requires positive prompt length");
        }
        if (audio_tokens_ < 0 || audio_tokens_ > prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker classification audio token count is invalid");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 ASR thinker classification graph context");
        }
        const auto & config = runtime_->assets().config.text_decoder;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "qwen3_asr.thinker.classify", runtime_->backend_type()};
        token_ids_input_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        audio_embeddings_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, std::max<int64_t>(audio_tokens_, 1));
        audio_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I64, std::max<int64_t>(audio_tokens_, 1));
        auto x = prompt_embeddings(
            ctx,
            weights,
            config,
            token_ids_input_,
            audio_embeddings_,
            audio_positions_,
            prompt_steps_,
            audio_tokens_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        auto decoder_out = modules::CausalDecoderModule(
                               make_qwen3_decoder_config(config, modules::CausalDecoderLogitsMode::AllSteps))
                               .build(ctx, x, positions, make_qwen3_decoder_weights(weights));
        auto token_ids = engine::sampling::GreedyDecodeModule().build(ctx, decoder_out.logits);
        token_ids_ = token_ids.tensor;
        ggml_set_output(token_ids_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, token_ids_);
        const auto try_alloc = [&]() {
            gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend())));
            return gallocr_ != nullptr &&
                ggml_gallocr_reserve(gallocr_.get(), graph_) &&
                ggml_gallocr_alloc_graph(gallocr_.get(), graph_);
        };
        if (!try_alloc() &&
            (engine::core::trim_backend_pools(runtime_->backend()), !try_alloc())) {
            throw std::runtime_error("failed to allocate Qwen3 ASR thinker classification graph");
        }
        position_ids_ = modules::decoder_position_ids(prompt_steps_);
        debug::timing_log_scalar("qwen3_asr.thinker.classify.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("qwen3_asr.thinker.classify_prompt_steps", prompt_steps_);
    }

    ~PromptClassificationGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
    }

    bool matches(const ThinkerWeightsRuntime & runtime, int64_t prompt_steps, int64_t audio_tokens) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps && audio_tokens_ == audio_tokens;
    }

    std::vector<int32_t> run(
        const std::vector<int32_t> & input_ids,
        const std::vector<float> & audio_embeddings,
        const std::vector<int32_t> & audio_positions) {
        const auto & config = runtime_->assets().config.text_decoder;
        if (static_cast<int64_t>(input_ids.size()) != prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker classification token id count mismatch");
        }
        if (static_cast<int64_t>(audio_embeddings.size()) != audio_tokens_ * config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR thinker classification audio embedding size mismatch");
        }
        if (static_cast<int64_t>(audio_positions.size()) != audio_tokens_) {
            throw std::runtime_error("Qwen3 ASR thinker classification audio position count mismatch");
        }
        auto timing_start = Clock::now();
        // See PrefillGraph::run: leaves are not pinned by the graph allocator.
        ggml_backend_tensor_set(positions_, position_ids_.data(), 0, position_ids_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(token_ids_input_, input_ids.data(), 0, input_ids.size() * sizeof(int32_t));
        if (audio_tokens_ > 0) {
            std::vector<int64_t> positions(audio_positions.begin(), audio_positions.end());
            ggml_backend_tensor_set(
                audio_embeddings_,
                audio_embeddings.data(),
                0,
                audio_embeddings.size() * sizeof(float));
            ggml_backend_tensor_set(
                audio_positions_,
                positions.data(),
                0,
                positions.size() * sizeof(int64_t));
        }
        debug::timing_log_scalar("qwen3_asr.thinker.classify_input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar("qwen3_asr.thinker.classify.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 ASR thinker classification graph compute failed");
        }
        std::vector<int32_t> token_ids(static_cast<size_t>(prompt_steps_));
        timing_start = Clock::now();
        ggml_backend_tensor_get(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("qwen3_asr.thinker.classify_output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return token_ids;
    }

private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    int64_t audio_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_input_ = nullptr;
    ggml_tensor * audio_embeddings_ = nullptr;
    ggml_tensor * audio_positions_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * token_ids_ = nullptr;
    std::vector<int32_t> position_ids_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

class DecodeGraph {
public:
    DecodeGraph(std::shared_ptr<ThinkerWeightsRuntime> runtime, int64_t cache_steps, size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("Qwen3 ASR thinker decode requires positive cache length");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 ASR thinker decode graph context");
        }
        const auto & config = runtime_->assets().config.text_decoder;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "qwen3_asr.thinker.decode", runtime_->backend_type()};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto token_id = core::wrap_tensor(token_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token_id, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto decoder_out = modules::CausalDecoderModule(
                               make_qwen3_decoder_config(config, modules::CausalDecoderLogitsMode::LastStep))
                               .build_static_cache_tail(
                                   ctx,
                                   graph_,
                                   x,
                                   positions,
                                   make_qwen3_decoder_weights(weights),
                                   cache_steps_,
                                   attention_mask,
                                   cache_slot);
        step_cache_ = std::move(decoder_out.cache);
        logits_ = decoder_out.logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            engine::core::trim_backend_pools(runtime_->backend());
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        }
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3 ASR thinker decode graph");
        }
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        debug::timing_log_scalar("qwen3_asr.thinker.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("qwen3_asr.thinker.decode_cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const ThinkerWeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_.get() == &runtime && cache_steps_ >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    std::vector<float> run_step(int32_t token) {
        const auto & config = runtime_->assets().config.text_decoder;
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(int32_t));
        modules::write_decoder_cached_step_mask(
            attention_mask_,
            attention_mask_values_,
            cache_steps_,
            step_cache_.valid_steps(),
            step_cache_.valid_steps());
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 ASR thinker decode graph compute failed");
        }
        std::vector<float> logits(static_cast<size_t>(config.output_size));
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        step_cache_.advance_after_direct_append(1);
        return logits;
    }

private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_values_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

}  // namespace

struct Qwen3ASRThinkerRuntime::Impl {
    Impl(
        std::shared_ptr<const Qwen3ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : weights(std::make_shared<ThinkerWeightsRuntime>(
              std::move(assets),
              execution,
              weight_context_bytes,
              storage_type)),
          prefill_graph_arena_bytes(prefill_graph_arena_bytes),
          decode_graph_arena_bytes(decode_graph_arena_bytes) {}

    void validate_prompt_audio(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings) const {
        const auto & config = weights->assets().config.text_decoder;
        if (audio_embeddings.hidden_size != config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR audio embedding hidden size mismatch");
        }
        if (audio_embeddings.tokens != static_cast<int64_t>(prompt.audio_token_positions.size())) {
            throw std::runtime_error("Qwen3 ASR audio embedding token count does not match prompt placeholders");
        }
        if (static_cast<int64_t>(audio_embeddings.values.size()) != audio_embeddings.tokens * config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR audio embedding value count mismatch");
        }
        for (const int32_t position : prompt.audio_token_positions) {
            if (position < 0 || position >= static_cast<int32_t>(prompt.input_ids.size())) {
                throw std::runtime_error("Qwen3 ASR audio placeholder position out of range");
            }
        }
    }

    Qwen3ASRGeneratedTokens generate(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings,
        const Qwen3ASRGenerationOptions & options) {
        const auto & config = weights->assets().config.text_decoder;
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("Qwen3 ASR thinker prompt is empty");
        }
        if (options.max_new_tokens <= 0) {
            throw std::runtime_error("Qwen3 ASR max_new_tokens must be positive");
        }
        const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
        if (prompt_steps + options.max_new_tokens > config.max_position_embeddings) {
            throw std::runtime_error("Qwen3 ASR thinker request exceeds max_position_embeddings");
        }
        auto timing_start = Clock::now();
        validate_prompt_audio(prompt, audio_embeddings);
        debug::timing_log_scalar("qwen3_asr.thinker.prompt_prepare_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (prefill_graph == nullptr || !prefill_graph->matches(*weights, prompt_steps, audio_embeddings.tokens)) {
            // drop the old graph (and its cuda_graphs cache entry) before
            // allocating the replacement: assigning over a live unique_ptr
            // holds both arenas at once at the rebuild peak
            prefill_graph.reset();
            prefill_graph = std::make_unique<PrefillGraph>(
                weights,
                prompt_steps,
                audio_embeddings.tokens,
                prefill_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("qwen3_asr.thinker.prefill.graph.build_ms", 0.0);
            debug::trace_log_scalar("qwen3_asr.thinker.prefill_prompt_steps", prompt_steps);
        }
        timing_start = Clock::now();
        auto prefill = prefill_graph->run(
            prompt.input_ids,
            audio_embeddings.values,
            prompt.audio_token_positions);
        debug::timing_log_scalar("qwen3_asr.thinker.prefill_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        const int64_t required_cache_steps = prompt_steps + options.max_new_tokens;
        if (decode_graph == nullptr || !decode_graph->can_run(*weights, required_cache_steps)) {
            decode_graph.reset();
            decode_graph = std::make_unique<DecodeGraph>(weights, required_cache_steps, decode_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("qwen3_asr.thinker.decode.graph.build_ms", 0.0);
            debug::trace_log_scalar("qwen3_asr.thinker.decode_cache_steps", required_cache_steps);
        }
        decode_graph->import_state(prefill.kv_state);

        Qwen3ASRGeneratedTokens out;
        std::vector<float> logits = std::move(prefill.logits);
        timing_start = Clock::now();
        for (int64_t step = 0; step < options.max_new_tokens; ++step) {
            const int32_t token = argmax_index(logits);
            if (is_eos(config, token)) {
                break;
            }
            out.token_ids.push_back(token);
            logits = decode_graph->run_step(token);
        }
        debug::timing_log_scalar("qwen3_asr.thinker.decode_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

    std::vector<int32_t> classify_prompt(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings) {
        const auto & config = weights->assets().config.text_decoder;
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("Qwen3 ASR thinker classification prompt is empty");
        }
        const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
        if (prompt_steps > config.max_position_embeddings) {
            throw std::runtime_error("Qwen3 ASR thinker classification request exceeds max_position_embeddings");
        }
        auto timing_start = Clock::now();
        validate_prompt_audio(prompt, audio_embeddings);
        debug::timing_log_scalar("qwen3_asr.thinker.classify_prompt_prepare_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (classification_graph == nullptr ||
            !classification_graph->matches(*weights, prompt_steps, audio_embeddings.tokens)) {
            classification_graph.reset();
            classification_graph = std::make_unique<PromptClassificationGraph>(
                weights,
                prompt_steps,
                audio_embeddings.tokens,
                prefill_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("qwen3_asr.thinker.classify.graph.build_ms", 0.0);
            debug::trace_log_scalar("qwen3_asr.thinker.classify_prompt_steps", prompt_steps);
        }
        timing_start = Clock::now();
        auto token_ids = classification_graph->run(
            prompt.input_ids,
            audio_embeddings.values,
            prompt.audio_token_positions);
        debug::timing_log_scalar("qwen3_asr.thinker.classify_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return token_ids;
    }

    std::shared_ptr<ThinkerWeightsRuntime> weights;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    std::unique_ptr<PrefillGraph> prefill_graph;
    std::unique_ptr<DecodeGraph> decode_graph;
    std::unique_ptr<PromptClassificationGraph> classification_graph;
};

Qwen3ASRThinkerRuntime::Qwen3ASRThinkerRuntime(
    std::shared_ptr<const Qwen3ASRAssets> assets,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

Qwen3ASRThinkerRuntime::~Qwen3ASRThinkerRuntime() = default;

Qwen3ASRGeneratedTokens Qwen3ASRThinkerRuntime::generate(
    const Qwen3ASRPrompt & prompt,
    const Qwen3ASRAudioEmbeddings & audio_embeddings,
    const Qwen3ASRGenerationOptions & options) {
    return impl_->generate(prompt, audio_embeddings, options);
}

std::vector<int32_t> Qwen3ASRThinkerRuntime::classify_prompt(
    const Qwen3ASRPrompt & prompt,
    const Qwen3ASRAudioEmbeddings & audio_embeddings) {
    return impl_->classify_prompt(prompt, audio_embeddings);
}

}  // namespace engine::models::qwen3_asr
