#include "engine/models/omnivoice/generator.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <utility>
#include <vector>

namespace engine::models::omnivoice {
namespace {

namespace assets_ns = engine::assets;
namespace modules = engine::modules;
namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;
constexpr float kMaskedAttentionBias = std::numeric_limits<float>::lowest();

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_weight_storage_type(assets_ns::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets_ns::TensorStorageType::Native:
        case assets_ns::TensorStorageType::F32:
        case assets_ns::TensorStorageType::F16:
        case assets_ns::TensorStorageType::BF16:
        case assets_ns::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "OmniVoice generator weight_type currently supports only native, f32, f16, bf16, and q8_0");
    }
}

int64_t checked_positive(int64_t value, const char * name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("OmniVoice generator expected positive ") + name);
    }
    return value;
}

int64_t head_dim(const OmniVoiceQwen3DiffusionConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("OmniVoice generator attention configuration is invalid");
    }
    return config.head_dim;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    if (input.shape.rank != 4) {
        throw std::runtime_error("OmniVoice generator repeat_kv_heads expects rank-4 input");
    }
    auto contiguous = ensure_contiguous(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = ensure_contiguous(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask,
    OmniVoiceGeneratorPerfMode perf_mode) {
    if (perf_mode == OmniVoiceGeneratorPerfMode::FlashAttention) {
        return modules::ScaledDotProductAttentionModule({
            dim,
            ctx.backend_type == core::BackendType::Cuda
                ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
                : modules::ScaledDotProductAttentionLowering::Flash,
            GGML_PREC_F32,
            modules::AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    }
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            attention_mask.tensor,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    auto context = matmul.build(ctx, attn, v_heads);
    return modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
}

struct Qwen3DiffusionLayerWeights {
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

struct Qwen3DiffusionWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue audio_embedding;
    std::vector<Qwen3DiffusionLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue audio_head;
};

Qwen3DiffusionWeights load_weights(
    const OmniVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets_ns::TensorStorageType storage_type) {
    validate_weight_storage_type(storage_type);
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    Qwen3DiffusionWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "omnivoice.generator.weights",
        weight_context_bytes);
    weights.text_embedding = weights.store->load_tensor(
        source,
        "llm.embed_tokens.weight",
        storage_type,
        {config.llm.vocab_size, config.llm.hidden_size});
    weights.audio_embedding = weights.store->load_tensor(
        source,
        "audio_embeddings.weight",
        storage_type,
        {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size});
    const int64_t dim = head_dim(config.llm);
    weights.layers.reserve(static_cast<size_t>(config.llm.num_hidden_layers));
    for (int64_t layer = 0; layer < config.llm.num_hidden_layers; ++layer) {
        const std::string prefix = "llm.layers." + std::to_string(layer);
        Qwen3DiffusionLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.llm.hidden_size});
        w.q_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.q_proj.weight",
            storage_type,
            {config.llm.num_attention_heads * dim, config.llm.hidden_size});
        w.k_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.k_proj.weight",
            storage_type,
            {config.llm.num_key_value_heads * dim, config.llm.hidden_size});
        w.v_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.v_proj.weight",
            storage_type,
            {config.llm.num_key_value_heads * dim, config.llm.hidden_size});
        w.o_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.o_proj.weight",
            storage_type,
            {config.llm.hidden_size, config.llm.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(
            source,
            prefix + ".post_attention_layernorm.weight",
            {config.llm.hidden_size});
        w.gate_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.gate_proj.weight",
            storage_type,
            {config.llm.intermediate_size, config.llm.hidden_size});
        w.up_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.up_proj.weight",
            storage_type,
            {config.llm.intermediate_size, config.llm.hidden_size});
        w.down_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.down_proj.weight",
            storage_type,
            {config.llm.hidden_size, config.llm.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "llm.norm.weight", {config.llm.hidden_size});
    weights.audio_head = weights.store->load_tensor(
        source,
        "audio_heads.weight",
        storage_type,
        {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size});
    weights.store->upload();
    return weights;
}

core::TensorValue build_embeddings(
    core::ModuleBuildContext & ctx,
    const OmniVoiceConfig & config,
    const Qwen3DiffusionWeights & weights,
    const core::TensorValue & text_ids,
    const std::array<core::TensorValue, 8> & audio_ids,
    const core::TensorValue & audio_mask,
    const core::TensorValue & text_mask) {
    auto text_embeds = modules::EmbeddingModule({config.llm.vocab_size, config.llm.hidden_size})
                           .build(ctx, text_ids, weights.text_embedding);
    core::TensorValue audio_embeds = modules::EmbeddingModule(
                                         {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size})
                                         .build(ctx, audio_ids[0], weights.audio_embedding);
    for (int64_t codebook = 1; codebook < config.num_audio_codebook; ++codebook) {
        auto embed = modules::EmbeddingModule(
                         {config.num_audio_codebook * config.audio_vocab_size, config.llm.hidden_size})
                         .build(ctx, audio_ids[static_cast<size_t>(codebook)], weights.audio_embedding);
        audio_embeds = modules::AddModule{}.build(ctx, audio_embeds, embed);
    }
    const auto broadcast_shape =
        core::TensorShape::from_dims({text_ids.shape.dims[0], text_ids.shape.dims[1], config.llm.hidden_size});
    auto audio_mask_expanded = modules::RepeatModule({broadcast_shape}).build(ctx, audio_mask);
    auto text_mask_expanded = modules::RepeatModule({broadcast_shape}).build(ctx, text_mask);
    auto text_part = modules::MulModule{}.build(ctx, text_embeds, text_mask_expanded);
    auto audio_part = modules::MulModule{}.build(ctx, audio_embeds, audio_mask_expanded);
    return modules::AddModule{}.build(ctx, text_part, audio_part);
}

core::TensorValue decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const Qwen3DiffusionLayerWeights & weights,
    const OmniVoiceQwen3DiffusionConfig & config,
    const core::TensorValue & attention_mask,
    OmniVoiceGeneratorPerfMode perf_mode = OmniVoiceGeneratorPerfMode::Standard) {
    const int64_t dim = head_dim(config);
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    const modules::LinearModule q_proj(
        binding::linear_config(config.hidden_size, config.num_attention_heads * dim, false));
    const modules::LinearModule k_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, false));
    const modules::LinearModule v_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, false));
    const modules::LinearModule o_proj(
        binding::linear_config(config.num_attention_heads * dim, config.hidden_size, false));
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::RMSNormModule head_norm({dim, config.rms_norm_eps, true, false});
    auto x_norm = hidden_norm.build(ctx, input, binding::norm_data(ctx, weights.input_norm));
    auto q = q_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.q_proj));
    auto k = k_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.k_proj));
    auto v = v_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.v_proj));
    q = head_norm.build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), binding::norm_data(ctx, weights.q_norm));
    k = head_norm.build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), binding::norm_data(ctx, weights.k_norm));
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k), kv_repeats);
    auto v_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask, perf_mode);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, binding::linear_data(ctx, weights.o_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(ctx, weights.post_norm));
    auto gate = modules::LinearModule(
                    binding::linear_config(config.hidden_size, config.intermediate_size, false))
                    .build(ctx, ff_in, binding::linear_data(ctx, weights.gate_proj));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(
                  binding::linear_config(config.hidden_size, config.intermediate_size, false))
                  .build(ctx, ff_in, binding::linear_data(ctx, weights.up_proj));
    auto ff = modules::LinearModule(
                  binding::linear_config(config.intermediate_size, config.hidden_size, false))
                  .build(ctx, modules::MulModule{}.build(ctx, gate, up), binding::linear_data(ctx, weights.down_proj));
    return modules::AddModule{}.build(ctx, x, ff);
}

class Qwen3DiffusionWeightsRuntime {
public:
    Qwen3DiffusionWeightsRuntime(
        std::shared_ptr<const OmniVoiceAssets> assets,
        core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        assets_ns::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          weights_(std::make_shared<Qwen3DiffusionWeights>(
              load_weights(*assets_, backend_, backend_type_, weight_context_bytes, storage_type))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("OmniVoice generator requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("OmniVoice generator backend is not initialized");
        }
    }

    const OmniVoiceAssets & assets() const noexcept {
        return *assets_;
    }

    const Qwen3DiffusionWeights & weights() const noexcept {
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
    std::shared_ptr<const OmniVoiceAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const Qwen3DiffusionWeights> weights_;
};

struct PackedInputs {
    int64_t style_tokens = 0;
    int64_t text_tokens = 0;
    int64_t reference_frames = 0;
    int64_t target_frames = 0;
    std::vector<int32_t> conditional_text_ids;
    std::vector<int32_t> unconditional_text_ids;
    std::array<std::vector<int32_t>, 8> conditional_audio_ids;
    std::array<std::vector<int32_t>, 8> unconditional_audio_ids;
};

class Qwen3DiffusionGraph {
public:
    virtual ~Qwen3DiffusionGraph() = default;
    virtual bool matches(const Qwen3DiffusionWeightsRuntime & runtime, int64_t total_tokens, int64_t target_frames) const = 0;
    virtual void rebuild(int64_t total_token_capacity, int64_t target_frame_capacity) = 0;
    virtual void prepare_request(const PackedInputs & inputs) = 0;
    virtual void set_guidance_scale(float value) noexcept = 0;
    virtual void update_generated_target_tokens(const PackedInputs & inputs) = 0;
    virtual const std::vector<float> & compute_logits(double & compute_ms, double & readback_ms) = 0;
    virtual int64_t total_token_capacity() const noexcept = 0;
    virtual int64_t target_frame_capacity() const noexcept = 0;
    virtual double rebuild_clear_ms() const noexcept = 0;
    virtual double rebuild_build_ms() const noexcept = 0;
    virtual double rebuild_alloc_ms() const noexcept = 0;
    virtual double rebuild_init_ms() const noexcept = 0;
};

class Qwen3DiffusionForwardGraph final : public Qwen3DiffusionGraph {
public:
    Qwen3DiffusionForwardGraph(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t total_token_capacity,
        int64_t target_frame_capacity,
        OmniVoiceGeneratorPerfMode perf_mode)
        : runtime_(std::move(runtime)),
          graph_arena_bytes_(graph_arena_bytes),
          perf_mode_(perf_mode) {
        rebuild(total_token_capacity, target_frame_capacity);
    }

    ~Qwen3DiffusionForwardGraph() {
        clear_graph();
    }

    bool matches(
        const Qwen3DiffusionWeightsRuntime & runtime,
        int64_t total_tokens,
        int64_t target_frames) const override {
        return runtime_.get() == &runtime &&
            total_tokens_capacity_ >= total_tokens &&
            target_frame_capacity_ >= target_frames;
    }

    void rebuild(int64_t total_token_capacity, int64_t target_frame_capacity) override {
        if (total_token_capacity <= 0 || target_frame_capacity <= 0) {
            throw std::runtime_error("OmniVoice generator total token capacity is invalid");
        }
        const auto clear_start = Clock::now();
        clear_graph();
        const auto clear_end = Clock::now();
        rebuild_clear_ms_ = engine::debug::elapsed_ms(clear_start, clear_end);
        total_tokens_capacity_ = total_token_capacity;
        target_frame_capacity_ = target_frame_capacity;

        const auto build_start = Clock::now();
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice generator input context");
        }
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice generator graph context");
        }

        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.forward", runtime_->backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "omnivoice.generator.forward.inputs",
            runtime_->backend_type()};

        std::array<core::TensorValue, 8> audio_id_values = {};

        auto text_ids =
            core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2, total_tokens_capacity_}));
        text_ids_ = text_ids.tensor;
        ggml_set_input(text_ids_);
        for (int64_t codebook = 0; codebook < config.num_audio_codebook; ++codebook) {
            auto audio_ids =
                core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2, total_tokens_capacity_}));
            audio_ids_[static_cast<size_t>(codebook)] = audio_ids.tensor;
            ggml_set_input(audio_ids_[static_cast<size_t>(codebook)]);
            audio_id_values[static_cast<size_t>(codebook)] = audio_ids;
        }
        audio_mask_value_ =
            core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, total_tokens_capacity_, 1}));
        auto audio_mask = audio_mask_value_;
        text_mask_value_ =
            core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, total_tokens_capacity_, 1}));
        auto text_mask = text_mask_value_;
        audio_mask_ = audio_mask.tensor;
        text_mask_ = text_mask.tensor;
        ggml_set_input(audio_mask_);
        ggml_set_input(text_mask_);
        positions_ =
            core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({total_tokens_capacity_})).tensor;
        ggml_set_input(positions_);
        auto positions =
            core::wrap_tensor(positions_, core::TensorShape::from_dims({total_tokens_capacity_}), GGML_TYPE_I32);
        conditional_target_indices_ =
            core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({target_frame_capacity_})).tensor;
        ggml_set_input(conditional_target_indices_);
        unconditional_target_indices_ =
            core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({target_frame_capacity_})).tensor;
        ggml_set_input(unconditional_target_indices_);
        auto attention_mask =
            core::make_tensor(
                input_ctx,
                perf_mode_ == OmniVoiceGeneratorPerfMode::FlashAttention ? GGML_TYPE_F16 : GGML_TYPE_F32,
                core::TensorShape::from_dims({2, 1, total_tokens_capacity_, total_tokens_capacity_}));
        attention_mask_ = attention_mask.tensor;
        guidance_scale_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1})).tensor;
        ggml_set_input(attention_mask_);
        ggml_set_input(guidance_scale_);

        auto x = build_embeddings(ctx, config, weights, text_ids, audio_id_values, audio_mask, text_mask);
        for (const auto & layer : weights.layers) {
            x = decoder_layer(ctx, x, positions, layer, config.llm, attention_mask, perf_mode_);
        }
        x = modules::RMSNormModule({config.llm.hidden_size, config.llm.rms_norm_eps, true, false})
                .build(ctx, x, binding::norm_data(ctx, weights.norm));
        auto conditional_hidden = modules::SliceModule({0, 0, 1}).build(ctx, x);
        conditional_hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, conditional_hidden),
            core::TensorShape::from_dims({total_tokens_capacity_, config.llm.hidden_size}));
        auto conditional_index_value = core::wrap_tensor(
            conditional_target_indices_,
            core::TensorShape::from_dims({target_frame_capacity_}),
            GGML_TYPE_I32);
        conditional_hidden = modules::EmbeddingModule({total_tokens_capacity_, config.llm.hidden_size})
                                 .build(ctx, conditional_index_value, conditional_hidden);
        conditional_hidden = core::reshape_tensor(
            ctx,
            conditional_hidden,
            core::TensorShape::from_dims({1, target_frame_capacity_, config.llm.hidden_size}));
        auto unconditional_hidden = modules::SliceModule({0, 1, 1}).build(ctx, x);
        unconditional_hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, unconditional_hidden),
            core::TensorShape::from_dims({total_tokens_capacity_, config.llm.hidden_size}));
        auto unconditional_index_value = core::wrap_tensor(
            unconditional_target_indices_,
            core::TensorShape::from_dims({target_frame_capacity_}),
            GGML_TYPE_I32);
        unconditional_hidden = modules::EmbeddingModule({total_tokens_capacity_, config.llm.hidden_size})
                                   .build(ctx, unconditional_index_value, unconditional_hidden);
        unconditional_hidden = core::reshape_tensor(
            ctx,
            unconditional_hidden,
            core::TensorShape::from_dims({1, target_frame_capacity_, config.llm.hidden_size}));
        auto hidden = modules::ConcatModule({0}).build(ctx, conditional_hidden, unconditional_hidden);
        auto logits =
            modules::LinearModule(
                binding::linear_config(
                    config.llm.hidden_size,
                    config.num_audio_codebook * config.audio_vocab_size,
                    false))
                .build(ctx, hidden, binding::linear_data(ctx, weights.audio_head));
        auto conditional_logits = modules::SliceModule({0, 0, 1}).build(ctx, logits);
        auto unconditional_logits = modules::SliceModule({0, 1, 1}).build(ctx, logits);
        auto guidance_scale_value = core::wrap_tensor(
            guidance_scale_,
            core::TensorShape::from_dims({1}),
            GGML_TYPE_F32);
        guidance_scale_value = core::reshape_tensor(
            ctx,
            guidance_scale_value,
            core::TensorShape::from_dims({1, 1, 1}));
        auto guidance_scale_expanded = modules::RepeatModule({conditional_logits.shape}).build(ctx, guidance_scale_value);
        auto diff = core::wrap_tensor(
            ggml_sub(ctx.ggml, conditional_logits.tensor, unconditional_logits.tensor),
            conditional_logits.shape,
            GGML_TYPE_F32);
        auto scaled_diff = modules::MulModule{}.build(ctx, diff, guidance_scale_expanded);
        auto combined_raw = modules::AddModule{}.build(ctx, conditional_logits, scaled_diff);
        combined_raw = ensure_contiguous(ctx, combined_raw);
        logits_ = combined_raw.tensor;

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        const auto build_end = Clock::now();
        rebuild_build_ms_ = engine::debug::elapsed_ms(build_start, build_end);

        const auto alloc_start = Clock::now();
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), runtime_->backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate OmniVoice generator input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate OmniVoice generator graph buffer");
        }
        const auto alloc_end = Clock::now();
        rebuild_alloc_ms_ = engine::debug::elapsed_ms(alloc_start, alloc_end);

        const auto init_start = Clock::now();
        std::vector<int32_t> position_host(static_cast<size_t>(total_tokens_capacity_), 0);
        for (int64_t i = 0; i < total_tokens_capacity_; ++i) {
            position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, position_host.data(), 0, position_host.size() * sizeof(int32_t));

        text_ids_host_.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0);
        conditional_target_indices_host_.assign(static_cast<size_t>(target_frame_capacity_), 0);
        unconditional_target_indices_host_.assign(static_cast<size_t>(target_frame_capacity_), 0);
        audio_mask_values_host_.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0.0F);
        text_mask_values_host_.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0.0F);
        if (perf_mode_ == OmniVoiceGeneratorPerfMode::FlashAttention) {
            attention_values_host_f16_.assign(
                static_cast<size_t>(2 * total_tokens_capacity_ * total_tokens_capacity_),
                ggml_fp32_to_fp16(kMaskedAttentionBias));
            attention_values_host_.clear();
        } else {
            attention_values_host_.assign(
                static_cast<size_t>(2 * total_tokens_capacity_ * total_tokens_capacity_),
                kMaskedAttentionBias);
            attention_values_host_f16_.clear();
        }
        for (auto & ids : audio_ids_host_) {
            ids.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0);
        }
        current_total_tokens_ = 0;
        current_target_frames_ = 0;
        current_conditional_target_start_ = 0;
        last_target_frames_ = -1;
        last_conditional_target_start_ = -1;
        last_mask_conditional_total_ = -1;
        last_mask_conditional_audio_start_ = -1;
        last_mask_unconditional_total_ = -1;
        guidance_scale_host_ = 0.0F;
        last_guidance_scale_ = std::numeric_limits<float>::quiet_NaN();
        logits_host_.clear();
        const auto init_end = Clock::now();
        rebuild_init_ms_ = engine::debug::elapsed_ms(init_start, init_end);
    }

    void prepare_request(const PackedInputs & inputs) override {
        const int64_t total_tokens =
            inputs.style_tokens + inputs.text_tokens + inputs.reference_frames + inputs.target_frames;
        if (total_tokens <= 0 ||
            total_tokens > total_tokens_capacity_ ||
            inputs.target_frames > target_frame_capacity_) {
            throw std::runtime_error("OmniVoice generator packed input shape exceeds graph capacity");
        }
        if (static_cast<int64_t>(inputs.conditional_text_ids.size()) != total_tokens ||
            static_cast<int64_t>(inputs.unconditional_text_ids.size()) != total_tokens) {
            throw std::runtime_error("OmniVoice generator text id tensor shape is invalid");
        }
        current_total_tokens_ = total_tokens;
        current_target_frames_ = inputs.target_frames;
        current_conditional_target_start_ =
            inputs.style_tokens + inputs.text_tokens + inputs.reference_frames;
        std::fill(
            text_ids_host_.begin(),
            text_ids_host_.end(),
            static_cast<int32_t>(runtime_->assets().config.audio_mask_id));
        for (int64_t pos = 0; pos < total_tokens; ++pos) {
            text_ids_host_[static_cast<size_t>(pos)] = inputs.conditional_text_ids[static_cast<size_t>(pos)];
            text_ids_host_[static_cast<size_t>(total_tokens_capacity_ + pos)] =
                inputs.unconditional_text_ids[static_cast<size_t>(pos)];
        }
        ggml_backend_tensor_set(text_ids_, text_ids_host_.data(), 0, text_ids_host_.size() * sizeof(int32_t));
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & conditional_ids = inputs.conditional_audio_ids[static_cast<size_t>(codebook)];
            const auto & unconditional_ids = inputs.unconditional_audio_ids[static_cast<size_t>(codebook)];
            if (static_cast<int64_t>(conditional_ids.size()) != total_tokens ||
                static_cast<int64_t>(unconditional_ids.size()) != total_tokens) {
                throw std::runtime_error("OmniVoice generator audio id tensor shape is invalid");
            }
            auto & ids = audio_ids_host_[static_cast<size_t>(codebook)];
            std::fill(
                ids.begin(),
                ids.end(),
                static_cast<int32_t>(codebook * runtime_->assets().config.audio_vocab_size +
                                     runtime_->assets().config.audio_mask_id));
            for (int64_t pos = 0; pos < total_tokens; ++pos) {
                ids[static_cast<size_t>(pos)] = conditional_ids[static_cast<size_t>(pos)];
                ids[static_cast<size_t>(total_tokens_capacity_ + pos)] =
                    unconditional_ids[static_cast<size_t>(pos)];
            }
            ggml_backend_tensor_set(
                audio_ids_[static_cast<size_t>(codebook)],
                ids.data(),
                0,
                ids.size() * sizeof(int32_t));
        }
        if (last_conditional_target_start_ != current_conditional_target_start_ ||
            last_target_frames_ != current_target_frames_) {
            for (int64_t frame = 0; frame < inputs.target_frames; ++frame) {
                conditional_target_indices_host_[static_cast<size_t>(frame)] =
                    static_cast<int32_t>(current_conditional_target_start_ + frame);
                unconditional_target_indices_host_[static_cast<size_t>(frame)] =
                    static_cast<int32_t>(frame);
            }
            const size_t target_index_bytes = static_cast<size_t>(inputs.target_frames) * sizeof(int32_t);
            ggml_backend_tensor_set(
                conditional_target_indices_,
                conditional_target_indices_host_.data(),
                0,
                target_index_bytes);
            ggml_backend_tensor_set(
                unconditional_target_indices_,
                unconditional_target_indices_host_.data(),
                0,
                target_index_bytes);
            last_conditional_target_start_ = current_conditional_target_start_;
            last_target_frames_ = current_target_frames_;
        }
        if (last_guidance_scale_ != guidance_scale_host_) {
            ggml_backend_tensor_set(guidance_scale_, &guidance_scale_host_, 0, sizeof(guidance_scale_host_));
            last_guidance_scale_ = guidance_scale_host_;
        }
        const int64_t conditional_total =
            inputs.style_tokens + inputs.text_tokens + inputs.reference_frames + inputs.target_frames;
        const int64_t conditional_audio_start = inputs.style_tokens + inputs.text_tokens;
        const int64_t unconditional_total = inputs.target_frames;
        if (last_mask_conditional_total_ != conditional_total ||
            last_mask_conditional_audio_start_ != conditional_audio_start ||
            last_mask_unconditional_total_ != unconditional_total) {
            upload_runtime_masks(inputs, conditional_total, conditional_audio_start, unconditional_total);
            last_mask_conditional_total_ = conditional_total;
            last_mask_conditional_audio_start_ = conditional_audio_start;
            last_mask_unconditional_total_ = unconditional_total;
        }
    }

    void set_guidance_scale(float value) noexcept override {
        guidance_scale_host_ = value;
    }

    void update_generated_target_tokens(const PackedInputs & inputs) override {
        if (current_total_tokens_ <= 0 || current_target_frames_ != inputs.target_frames) {
            throw std::runtime_error("OmniVoice generator target token update requires a prepared request");
        }
        const size_t conditional_offset_bytes =
            static_cast<size_t>(current_conditional_target_start_) * sizeof(int32_t);
        const size_t unconditional_offset_bytes =
            static_cast<size_t>(total_tokens_capacity_) * sizeof(int32_t);
        const size_t byte_count = static_cast<size_t>(current_target_frames_) * sizeof(int32_t);
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & conditional_ids = inputs.conditional_audio_ids[static_cast<size_t>(codebook)];
            const auto & unconditional_ids = inputs.unconditional_audio_ids[static_cast<size_t>(codebook)];
            auto & host_ids = audio_ids_host_[static_cast<size_t>(codebook)];
            for (int64_t frame = 0; frame < current_target_frames_; ++frame) {
                host_ids[static_cast<size_t>(current_conditional_target_start_ + frame)] =
                    conditional_ids[static_cast<size_t>(current_conditional_target_start_ + frame)];
                host_ids[static_cast<size_t>(total_tokens_capacity_ + frame)] =
                    unconditional_ids[static_cast<size_t>(frame)];
            }
            ggml_backend_tensor_set(
                audio_ids_[static_cast<size_t>(codebook)],
                host_ids.data() + current_conditional_target_start_,
                conditional_offset_bytes,
                byte_count);
            ggml_backend_tensor_set(
                audio_ids_[static_cast<size_t>(codebook)],
                host_ids.data() + total_tokens_capacity_,
                unconditional_offset_bytes,
                byte_count);
        }
    }

    const std::vector<float> & compute_logits(double & compute_ms, double & readback_ms) override {
        if (current_total_tokens_ <= 0 || current_target_frames_ <= 0) {
            throw std::runtime_error("OmniVoice generator compute requires a prepared request");
        }

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice generator graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);

        const size_t logits_count = static_cast<size_t>(
            target_frame_capacity_ * runtime_->assets().config.num_audio_codebook *
            runtime_->assets().config.audio_vocab_size);
        if (logits_host_.size() != logits_count) {
            logits_host_.assign(logits_count, 0.0F);
        }
        const auto readback_start = Clock::now();
        ggml_backend_tensor_get(logits_, logits_host_.data(), 0, logits_host_.size() * sizeof(float));
        const auto readback_end = Clock::now();
        readback_ms += engine::debug::elapsed_ms(readback_start, readback_end);
        return logits_host_;
    }

    int64_t total_token_capacity() const noexcept override {
        return total_tokens_capacity_;
    }
    int64_t target_frame_capacity() const noexcept override {
        return target_frame_capacity_;
    }
    double rebuild_clear_ms() const noexcept override { return rebuild_clear_ms_; }
    double rebuild_build_ms() const noexcept override { return rebuild_build_ms_; }
    double rebuild_alloc_ms() const noexcept override { return rebuild_alloc_ms_; }
    double rebuild_init_ms() const noexcept override { return rebuild_init_ms_; }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
        logits_ = nullptr;
        guidance_scale_ = nullptr;
        attention_mask_ = nullptr;
        conditional_target_indices_ = nullptr;
        unconditional_target_indices_ = nullptr;
        positions_ = nullptr;
        audio_mask_ = nullptr;
        text_mask_ = nullptr;
        for (auto & tensor : audio_ids_) {
            tensor = nullptr;
        }
        text_ids_ = nullptr;
        input_ctx_.reset();
        ctx_.reset();
    }

    void upload_runtime_masks(
        const PackedInputs &,
        int64_t conditional_total,
        int64_t conditional_audio_start,
        int64_t unconditional_total) {
        std::fill(audio_mask_values_host_.begin(), audio_mask_values_host_.end(), 0.0F);
        std::fill(text_mask_values_host_.begin(), text_mask_values_host_.end(), 1.0F);
        for (int64_t pos = 0; pos < conditional_total; ++pos) {
            if (pos >= conditional_audio_start) {
                audio_mask_values_host_[static_cast<size_t>(pos)] = 1.0F;
                text_mask_values_host_[static_cast<size_t>(pos)] = 0.0F;
            }
        }
        for (int64_t pos = 0; pos < unconditional_total; ++pos) {
            const size_t offset = static_cast<size_t>(total_tokens_capacity_ + pos);
            audio_mask_values_host_[offset] = 1.0F;
            text_mask_values_host_[offset] = 0.0F;
        }
        ggml_backend_tensor_set(
            audio_mask_,
            audio_mask_values_host_.data(),
            0,
            audio_mask_values_host_.size() * sizeof(float));
        ggml_backend_tensor_set(
            text_mask_,
            text_mask_values_host_.data(),
            0,
            text_mask_values_host_.size() * sizeof(float));
        if (perf_mode_ == OmniVoiceGeneratorPerfMode::FlashAttention) {
            std::fill(
                attention_values_host_f16_.begin(),
                attention_values_host_f16_.end(),
                ggml_fp32_to_fp16(kMaskedAttentionBias));
        } else {
            std::fill(
                attention_values_host_.begin(),
                attention_values_host_.end(),
                kMaskedAttentionBias);
        }
        const auto write_zero = [&](int batch, int64_t q, int64_t k) {
            const size_t index = static_cast<size_t>(
                k + total_tokens_capacity_ * (q + total_tokens_capacity_ * batch));
            if (perf_mode_ == OmniVoiceGeneratorPerfMode::FlashAttention) {
                attention_values_host_f16_[index] = ggml_fp32_to_fp16(0.0F);
            } else {
                attention_values_host_[index] = 0.0F;
            }
        };
        for (int64_t q = 0; q < conditional_total; ++q) {
            for (int64_t k = 0; k < conditional_total; ++k) {
                write_zero(0, q, k);
            }
        }
        for (int64_t q = conditional_total; q < total_tokens_capacity_; ++q) {
            write_zero(0, q, q);
        }
        for (int64_t q = 0; q < unconditional_total; ++q) {
            for (int64_t k = 0; k < unconditional_total; ++k) {
                write_zero(1, q, k);
            }
        }
        for (int64_t q = unconditional_total; q < total_tokens_capacity_; ++q) {
            write_zero(1, q, q);
        }
        if (perf_mode_ == OmniVoiceGeneratorPerfMode::FlashAttention) {
            ggml_backend_tensor_set(
                attention_mask_,
                attention_values_host_f16_.data(),
                0,
                attention_values_host_f16_.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(
                attention_mask_,
                attention_values_host_.data(),
                0,
                attention_values_host_.size() * sizeof(float));
        }
    }

    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    OmniVoiceGeneratorPerfMode perf_mode_ = OmniVoiceGeneratorPerfMode::Standard;
    int64_t total_tokens_capacity_ = 0;
    int64_t target_frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_tensor * text_ids_ = nullptr;
    std::array<ggml_tensor *, 8> audio_ids_{};
    core::TensorValue audio_mask_value_;
    core::TensorValue text_mask_value_;
    ggml_tensor * audio_mask_ = nullptr;
    ggml_tensor * text_mask_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * conditional_target_indices_ = nullptr;
    ggml_tensor * unconditional_target_indices_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * guidance_scale_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    int64_t current_total_tokens_ = 0;
    int64_t current_target_frames_ = 0;
    int64_t current_conditional_target_start_ = 0;
    int64_t last_target_frames_ = -1;
    int64_t last_conditional_target_start_ = -1;
    int64_t last_mask_conditional_total_ = -1;
    int64_t last_mask_conditional_audio_start_ = -1;
    int64_t last_mask_unconditional_total_ = -1;
    std::vector<int32_t> text_ids_host_;
    std::array<std::vector<int32_t>, 8> audio_ids_host_{};
    std::vector<int32_t> conditional_target_indices_host_;
    std::vector<int32_t> unconditional_target_indices_host_;
    std::vector<float> audio_mask_values_host_;
    std::vector<float> text_mask_values_host_;
    std::vector<float> attention_values_host_;
    std::vector<ggml_fp16_t> attention_values_host_f16_;
    float guidance_scale_host_ = 0.0F;
    float last_guidance_scale_ = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> logits_host_;
    double rebuild_clear_ms_ = 0.0;
    double rebuild_build_ms_ = 0.0;
    double rebuild_alloc_ms_ = 0.0;
    double rebuild_init_ms_ = 0.0;
};

#include "generator_layerwise.inc"

std::pair<int32_t, float> argmax_with_value_excluding_mask(const float * values, int64_t count, int64_t mask_id) {
    int32_t best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < count; ++i) {
        if (i == mask_id) {
            continue;
        }
        const float value = values[static_cast<size_t>(i)];
        if (value > best_value) {
            best_value = value;
            best_index = static_cast<int32_t>(i);
        }
    }
    if (best_index < 0) {
        throw std::runtime_error("OmniVoice generator could not choose a token after masking");
    }
    return {best_index, best_value};
}

std::pair<int32_t, float> best_log_prob_excluding_mask(const float * logits, int64_t count, int64_t mask_id) {
    if (count <= 0) {
        throw std::runtime_error("OmniVoice generator best_log_prob requires positive vocabulary size");
    }
    int32_t best_index = -1;
    float best_logit = -std::numeric_limits<float>::infinity();
    float max_logit = logits[0];
    for (int64_t i = 0; i < count; ++i) {
        const float value = logits[static_cast<size_t>(i)];
        max_logit = std::max(max_logit, value);
        if (i == mask_id) {
            continue;
        }
        if (value > best_logit) {
            best_logit = value;
            best_index = static_cast<int32_t>(i);
        }
    }
    if (best_index < 0) {
        throw std::runtime_error("OmniVoice generator could not choose a token after masking");
    }
    double sum = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        sum += std::exp(static_cast<double>(logits[static_cast<size_t>(i)] - max_logit));
    }
    const float logsum = max_logit + static_cast<float>(std::log(sum));
    return {best_index, best_logit - logsum};
}

void log_softmax_into(const float * values, int64_t count, std::vector<float> & out) {
    if (count <= 0) {
        throw std::runtime_error("OmniVoice generator log_softmax requires positive vocabulary size");
    }
    float max_value = values[0];
    for (int64_t i = 1; i < count; ++i) {
        max_value = std::max(max_value, values[i]);
    }
    double sum = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        sum += std::exp(static_cast<double>(values[i] - max_value));
    }
    const float logsum = max_value + static_cast<float>(std::log(sum));
    out.resize(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        out[static_cast<size_t>(i)] = values[i] - logsum;
    }
}

float uniform_open01(std::mt19937 & rng) {
    static std::uniform_real_distribution<float> dist(1.0e-6F, 1.0F - 1.0e-6F);
    return dist(rng);
}

float gumbel_sample_scalar(float logit, float temperature, std::mt19937 & rng) {
    const float scaled = logit / temperature;
    const float u = uniform_open01(rng);
    const float noise = -std::log(-std::log(u + 1.0e-10F) + 1.0e-10F);
    return scaled + noise;
}

int32_t sample_class(
    const float * log_probs,
    int64_t vocab_size,
    float temperature,
    int64_t mask_id,
    std::mt19937 & rng) {
    const int64_t k = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(0.1 * static_cast<double>(vocab_size))));
    std::vector<int32_t> indices(static_cast<size_t>(vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&](int32_t lhs, int32_t rhs) {
            return log_probs[static_cast<size_t>(lhs)] > log_probs[static_cast<size_t>(rhs)];
        });
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_index = 0;
    for (int64_t i = 0; i < k; ++i) {
        const int32_t token = indices[static_cast<size_t>(i)];
        if (token == mask_id) {
            continue;
        }
        const float score = gumbel_sample_scalar(log_probs[static_cast<size_t>(token)], temperature, rng);
        if (score > best_score) {
            best_score = score;
            best_index = token;
        }
    }
    return best_index;
}

std::vector<double> time_steps(int64_t num_inference_steps, float t_shift) {
    if (num_inference_steps <= 0) {
        throw std::runtime_error("OmniVoice generator num_inference_steps must be positive");
    }
    std::vector<double> out(static_cast<size_t>(num_inference_steps + 1), 0.0);
    for (int64_t i = 0; i <= num_inference_steps; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(num_inference_steps);
        out[static_cast<size_t>(i)] = static_cast<double>(t_shift) * t /
            (1.0 + (static_cast<double>(t_shift) - 1.0) * t);
    }
    return out;
}

std::vector<int64_t> make_schedule(int64_t total_masked_tokens, int64_t num_inference_steps, float t_shift) {
    const auto steps = time_steps(num_inference_steps, t_shift);
    std::vector<int64_t> schedule(static_cast<size_t>(num_inference_steps), 0);
    int64_t remaining = total_masked_tokens;
    for (int64_t step = 0; step < num_inference_steps; ++step) {
        int64_t num = 0;
        if (step == num_inference_steps - 1) {
            num = remaining;
        } else {
            const double fraction = steps[static_cast<size_t>(step + 1)] - steps[static_cast<size_t>(step)];
            num = std::min<int64_t>(
                static_cast<int64_t>(std::ceil(static_cast<double>(total_masked_tokens) * fraction)),
                remaining);
        }
        schedule[static_cast<size_t>(step)] = num;
        remaining -= num;
    }
    return schedule;
}

void validate_prompt_token_ranges(const OmniVoiceAssets & assets, const OmniVoicePrompt & prompt) {
    const int32_t text_vocab_limit = static_cast<int32_t>(assets.config.llm.vocab_size);
    for (size_t i = 0; i < prompt.style_token_ids.size(); ++i) {
        const int32_t token = prompt.style_token_ids[i];
        if (token < 0 || token >= text_vocab_limit) {
            throw std::runtime_error(
                "OmniVoice style token id out of range at index " + std::to_string(i) + ": " + std::to_string(token) +
                " vs llm vocab " + std::to_string(text_vocab_limit));
        }
    }
    for (size_t i = 0; i < prompt.text_token_ids.size(); ++i) {
        const int32_t token = prompt.text_token_ids[i];
        if (token < 0 || token >= text_vocab_limit) {
            throw std::runtime_error(
                "OmniVoice text token id out of range at index " + std::to_string(i) + ": " + std::to_string(token) +
                " vs llm vocab " + std::to_string(text_vocab_limit));
        }
    }
    if (!prompt.reference_audio_tokens.has_value()) {
        return;
    }
    const int32_t audio_vocab_limit = static_cast<int32_t>(assets.config.audio_vocab_size);
    const auto & audio = *prompt.reference_audio_tokens;
    for (size_t i = 0; i < audio.token_ids.size(); ++i) {
        const int32_t token = audio.token_ids[i];
        if (token < 0 || token >= audio_vocab_limit) {
            throw std::runtime_error(
                "OmniVoice reference audio token id out of range at index " + std::to_string(i) + ": " +
                std::to_string(token) + " vs audio vocab " + std::to_string(audio_vocab_limit));
        }
    }
}

PackedInputs pack_initial_inputs(const OmniVoiceAssets & assets, const OmniVoicePrompt & prompt) {
    const int64_t codebooks = checked_positive(assets.config.num_audio_codebook, "num_audio_codebook");
    if (codebooks != 8) {
        throw std::runtime_error("OmniVoice native generator currently expects exactly 8 codebooks");
    }
    PackedInputs packed;
    packed.style_tokens = static_cast<int64_t>(prompt.style_token_ids.size());
    packed.text_tokens = static_cast<int64_t>(prompt.text_token_ids.size());
    packed.reference_frames = 0;
    if (prompt.reference_audio_tokens.has_value()) {
        packed.reference_frames = prompt.reference_audio_tokens->frames;
        if (prompt.reference_audio_tokens->codebooks != codebooks) {
            throw std::runtime_error("OmniVoice reference audio token codebook count does not match generator");
        }
    }
    packed.target_frames = checked_positive(prompt.target_audio_tokens, "target_audio_tokens");
    const int64_t total = packed.style_tokens + packed.text_tokens + packed.reference_frames + packed.target_frames;
    packed.conditional_text_ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    packed.unconditional_text_ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    for (auto & ids : packed.conditional_audio_ids) {
        ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    }
    for (auto & ids : packed.unconditional_audio_ids) {
        ids.assign(static_cast<size_t>(total), static_cast<int32_t>(assets.config.audio_mask_id));
    }

    int64_t pos = 0;
    for (int32_t token : prompt.style_token_ids) {
        packed.conditional_text_ids[static_cast<size_t>(pos)] = token;
        ++pos;
    }
    for (int32_t token : prompt.text_token_ids) {
        packed.conditional_text_ids[static_cast<size_t>(pos)] = token;
        ++pos;
    }
    if (prompt.reference_audio_tokens.has_value()) {
        for (int64_t frame = 0; frame < packed.reference_frames; ++frame) {
            packed.conditional_text_ids[static_cast<size_t>(pos + frame)] =
                prompt.reference_audio_tokens->token_ids[static_cast<size_t>(frame * codebooks)];
            for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
                packed.conditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(pos + frame)] =
                    prompt.reference_audio_tokens->token_ids[static_cast<size_t>(frame * codebooks + codebook)];
            }
        }
        pos += packed.reference_frames;
    }
    for (int64_t frame = 0; frame < packed.target_frames; ++frame) {
        packed.conditional_text_ids[static_cast<size_t>(pos + frame)] = static_cast<int32_t>(assets.config.audio_mask_id);
        for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
            packed.conditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(pos + frame)] =
                static_cast<int32_t>(assets.config.audio_mask_id);
        }
    }

    for (int64_t frame = 0; frame < packed.target_frames; ++frame) {
        packed.unconditional_text_ids[static_cast<size_t>(frame)] = static_cast<int32_t>(assets.config.audio_mask_id);
        for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
            packed.unconditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(frame)] =
                static_cast<int32_t>(assets.config.audio_mask_id);
        }
    }

    const int32_t vocab_stride = static_cast<int32_t>(assets.config.audio_vocab_size);
    for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
        const int32_t offset = static_cast<int32_t>(codebook) * vocab_stride;
        auto & cond_ids = packed.conditional_audio_ids[static_cast<size_t>(codebook)];
        for (int32_t & value : cond_ids) {
            value += offset;
        }
        auto & uncond_ids = packed.unconditional_audio_ids[static_cast<size_t>(codebook)];
        for (int32_t & value : uncond_ids) {
            value += offset;
        }
    }
    return packed;
}

void update_generated_tokens(
    PackedInputs & packed,
    const OmniVoiceAssets & assets,
    const std::vector<int32_t> & generated_tokens) {
    const int64_t codebooks = assets.config.num_audio_codebook;
    const int64_t target_frames = packed.target_frames;
    const int64_t cond_start = packed.style_tokens + packed.text_tokens + packed.reference_frames;
    const int32_t vocab_stride = static_cast<int32_t>(assets.config.audio_vocab_size);
    for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
        const int32_t offset = static_cast<int32_t>(codebook) * vocab_stride;
        for (int64_t frame = 0; frame < target_frames; ++frame) {
            const int32_t token = generated_tokens[static_cast<size_t>(codebook * target_frames + frame)];
            packed.conditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(cond_start + frame)] =
                token + offset;
            packed.unconditional_audio_ids[static_cast<size_t>(codebook)][static_cast<size_t>(frame)] =
                token + offset;
        }
    }
}

}  // namespace

struct OmniVoiceQwen3DiffusionRuntime::Impl {
    std::shared_ptr<const OmniVoiceAssets> assets;
    size_t graph_arena_bytes = 0;
    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime;
    std::unique_ptr<Qwen3DiffusionGraph> forward_graph;
    bool mem_saver = false;
    OmniVoiceGeneratorPerfMode perf_mode = OmniVoiceGeneratorPerfMode::Standard;
    OmniVoiceQwen3DiffusionRuntimeStats last_stats = {};
    std::mt19937 rng{std::random_device{}()};
};

OmniVoiceQwen3DiffusionRuntime::OmniVoiceQwen3DiffusionRuntime(
    std::shared_ptr<const OmniVoiceAssets> assets,
    core::ExecutionContext & execution_context,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets_ns::TensorStorageType weight_storage_type,
    bool mem_saver,
    OmniVoiceGeneratorPerfMode perf_mode)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("OmniVoice generator requires assets");
    }
    if (execution_context.backend() == nullptr) {
        throw std::runtime_error("OmniVoice generator backend is not initialized");
    }
    impl_->assets = std::move(assets);
    impl_->graph_arena_bytes = std::max(prefill_graph_arena_bytes, decode_graph_arena_bytes);
    impl_->runtime = std::make_shared<Qwen3DiffusionWeightsRuntime>(
        impl_->assets,
        execution_context,
        weight_context_bytes,
        weight_storage_type);
    impl_->mem_saver = mem_saver;
    impl_->perf_mode = perf_mode;
}

OmniVoiceQwen3DiffusionRuntime::~OmniVoiceQwen3DiffusionRuntime() = default;

const OmniVoiceQwen3DiffusionRuntimeStats & OmniVoiceQwen3DiffusionRuntime::last_stats() const noexcept {
    return impl_->last_stats;
}

void OmniVoiceQwen3DiffusionRuntime::seed_rng(uint32_t seed) {
    impl_->rng.seed(seed);
}

void OmniVoiceQwen3DiffusionRuntime::release_runtime_graphs() {
    impl_->forward_graph.reset();
}

OmniVoiceGeneratedAudioTokens OmniVoiceQwen3DiffusionRuntime::generate(
    const OmniVoicePrompt & prompt,
    const OmniVoiceGenerationOptions & options) {
    if (prompt.style_token_ids.empty()) {
        throw std::runtime_error("OmniVoice generator requires non-empty style tokens");
    }
    if (prompt.text_token_ids.empty()) {
        throw std::runtime_error("OmniVoice generator requires non-empty text tokens");
    }
    if (prompt.reference_audio_tokens.has_value() &&
        prompt.reference_audio_tokens->codebooks != impl_->assets->config.num_audio_codebook) {
        throw std::runtime_error("OmniVoice generator reference audio token codebooks do not match model configuration");
    }
    validate_prompt_token_ranges(*impl_->assets, prompt);

    PackedInputs packed = pack_initial_inputs(*impl_->assets, prompt);
    const int64_t required_total_tokens =
        packed.style_tokens + packed.text_tokens + packed.reference_frames + packed.target_frames;
    impl_->last_stats.graph_rebuilt = false;
    impl_->last_stats.rebuild_ms = 0.0;
    impl_->last_stats.rebuild_clear_ms = 0.0;
    impl_->last_stats.rebuild_build_ms = 0.0;
    impl_->last_stats.rebuild_alloc_ms = 0.0;
    impl_->last_stats.rebuild_init_ms = 0.0;
    const bool needs_rebuild =
        impl_->forward_graph == nullptr ||
        !impl_->forward_graph->matches(
            *impl_->runtime,
            required_total_tokens,
            packed.target_frames) ||
        impl_->forward_graph->total_token_capacity() != required_total_tokens ||
        impl_->forward_graph->target_frame_capacity() != packed.target_frames;
    if (needs_rebuild) {
        const auto rebuild_start = Clock::now();
        if (impl_->forward_graph == nullptr) {
            if (impl_->mem_saver) {
                impl_->forward_graph = std::make_unique<Qwen3DiffusionLayerwiseGraph>(
                    impl_->runtime,
                    impl_->graph_arena_bytes,
                    required_total_tokens,
                    packed.target_frames);
            } else {
                impl_->forward_graph = std::make_unique<Qwen3DiffusionForwardGraph>(
                    impl_->runtime,
                    impl_->graph_arena_bytes,
                    required_total_tokens,
                    packed.target_frames,
                    impl_->perf_mode);
            }
        } else {
            impl_->forward_graph->rebuild(required_total_tokens, packed.target_frames);
        }
        const auto rebuild_end = Clock::now();
        impl_->last_stats.graph_rebuilt = true;
        impl_->last_stats.rebuild_ms = engine::debug::elapsed_ms(rebuild_start, rebuild_end);
        impl_->last_stats.rebuild_clear_ms = impl_->forward_graph->rebuild_clear_ms();
        impl_->last_stats.rebuild_build_ms = impl_->forward_graph->rebuild_build_ms();
        impl_->last_stats.rebuild_alloc_ms = impl_->forward_graph->rebuild_alloc_ms();
        impl_->last_stats.rebuild_init_ms = impl_->forward_graph->rebuild_init_ms();
    }
    impl_->last_stats.total_token_capacity = impl_->forward_graph->total_token_capacity();
    impl_->last_stats.target_frame_capacity = impl_->forward_graph->target_frame_capacity();

    const int64_t codebooks = impl_->assets->config.num_audio_codebook;
    const int64_t vocab_size = impl_->assets->config.audio_vocab_size;
    const int64_t mask_id = impl_->assets->config.audio_mask_id;
    const int64_t target_frames = packed.target_frames;
    const auto schedule = make_schedule(codebooks * target_frames, options.num_inference_steps, options.t_shift);
    std::vector<int32_t> accepted(static_cast<size_t>(codebooks * target_frames), static_cast<int32_t>(mask_id));
    std::vector<size_t> active_indices(static_cast<size_t>(codebooks * target_frames), 0);
    std::iota(active_indices.begin(), active_indices.end(), 0);
    PackedInputs inference = std::move(packed);
    update_generated_tokens(inference, *impl_->assets, accepted);
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
    double scoring_ms = 0.0;
    double update_ms = 0.0;
    int64_t executed_steps = 0;
    const auto prepare_upload_start = Clock::now();
    impl_->forward_graph->set_guidance_scale(options.guidance_scale);
    impl_->forward_graph->prepare_request(inference);
    const auto prepare_upload_end = Clock::now();
    upload_ms += engine::debug::elapsed_ms(prepare_upload_start, prepare_upload_end);
    for (int64_t step = 0; step < options.num_inference_steps; ++step) {
        const int64_t fill_count = schedule[static_cast<size_t>(step)];
        if (fill_count <= 0) {
            continue;
        }
        ++executed_steps;

        const auto & batched_logits = impl_->forward_graph->compute_logits(compute_ms, readback_ms);
        struct Candidate {
            float score = -std::numeric_limits<float>::infinity();
            int32_t predicted = 0;
            size_t index = 0;
        };
        const auto scoring_start = Clock::now();
        std::vector<Candidate> candidates;
        candidates.reserve(active_indices.size());
        if (options.class_temperature == 0.0F) {
            const size_t candidate_slots = active_indices.size();
            const unsigned desired_threads = static_cast<unsigned>(
                std::max<int>(1, impl_->runtime->threads()));
            const unsigned worker_count = std::min<unsigned>(
                desired_threads,
                static_cast<unsigned>(std::max<size_t>(1, candidate_slots)));
            std::vector<std::vector<Candidate>> staged_candidates(std::max<unsigned>(1, worker_count));
            std::vector<std::thread> workers;
            workers.reserve(worker_count > 0 ? worker_count - 1 : 0);
            auto score_range = [&](unsigned worker_index, size_t begin_index, size_t end_index) {
                auto & local_candidates = staged_candidates[worker_index];
                local_candidates.clear();
                local_candidates.reserve(end_index > begin_index ? end_index - begin_index : 0);
                std::vector<float> conditional_log_probs;
                std::vector<float> unconditional_log_probs;
                std::vector<float> guided_scores;
                std::vector<float> final_log_probs_storage;
                for (size_t slot_index = begin_index; slot_index < end_index; ++slot_index) {
                    const size_t flat_index = active_indices[slot_index];
                    const int64_t codebook = static_cast<int64_t>(flat_index / static_cast<size_t>(target_frames));
                    const int64_t frame = static_cast<int64_t>(flat_index % static_cast<size_t>(target_frames));
                    const size_t conditional_offset = static_cast<size_t>(
                        frame * codebooks * vocab_size + codebook * vocab_size);
                    const float * combined_logits = batched_logits.data() + static_cast<std::ptrdiff_t>(conditional_offset);
                    auto [best_token, best_score] =
                        best_log_prob_excluding_mask(combined_logits, vocab_size, mask_id);
                    Candidate candidate;
                    candidate.score = best_score - static_cast<float>(codebook) * options.layer_penalty_factor;
                    candidate.predicted = best_token;
                    candidate.index = flat_index;
                    local_candidates.push_back(candidate);
                }
            };
            const size_t block = (candidate_slots + worker_count - 1) / worker_count;
            size_t begin = 0;
            for (unsigned worker = 1; worker < worker_count; ++worker) {
                const size_t end = std::min(candidate_slots, begin + block);
                workers.emplace_back(score_range, worker, begin, end);
                begin = end;
            }
            score_range(0, begin, candidate_slots);
            for (auto & worker : workers) {
                worker.join();
            }
            size_t total_candidates = 0;
            for (const auto & local_candidates : staged_candidates) {
                total_candidates += local_candidates.size();
            }
            candidates.reserve(total_candidates);
            for (const auto & local_candidates : staged_candidates) {
                candidates.insert(candidates.end(), local_candidates.begin(), local_candidates.end());
            }
            for (auto & candidate : candidates) {
                if (options.position_temperature > 0.0F) {
                    candidate.score =
                        gumbel_sample_scalar(candidate.score, options.position_temperature, impl_->rng);
                }
            }
        } else {
            std::vector<float> conditional_log_probs;
            std::vector<float> combined_log_probs;
            for (const size_t token_index : active_indices) {
                const int64_t codebook = static_cast<int64_t>(token_index / static_cast<size_t>(target_frames));
                const int64_t frame = static_cast<int64_t>(token_index % static_cast<size_t>(target_frames));
                const size_t conditional_offset = static_cast<size_t>(
                    frame * codebooks * vocab_size + codebook * vocab_size);
                const float * combined_logits = batched_logits.data() + static_cast<std::ptrdiff_t>(conditional_offset);
                log_softmax_into(combined_logits, vocab_size, combined_log_probs);
                auto [best_token, best_score] =
                    argmax_with_value_excluding_mask(combined_log_probs.data(), vocab_size, mask_id);
                Candidate candidate;
                candidate.index = token_index;
                candidate.predicted =
                    sample_class(combined_log_probs.data(), vocab_size, options.class_temperature, mask_id, impl_->rng);
                candidate.score = best_score -
                    static_cast<float>(codebook) * options.layer_penalty_factor;
                if (options.position_temperature > 0.0F) {
                    candidate.score =
                        gumbel_sample_scalar(candidate.score, options.position_temperature, impl_->rng);
                }
                candidates.push_back(candidate);
            }
        }
        if (candidates.empty()) {
            break;
        }
        const int64_t actual_fill = std::min<int64_t>(fill_count, static_cast<int64_t>(candidates.size()));
        const auto candidate_order = [](const Candidate & lhs, const Candidate & rhs) {
            if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
            }
            return lhs.index < rhs.index;
        };
        if (actual_fill < static_cast<int64_t>(candidates.size())) {
            auto top_end = candidates.begin() + actual_fill;
            std::nth_element(candidates.begin(), top_end, candidates.end(), candidate_order);
            std::sort(candidates.begin(), top_end, candidate_order);
        } else {
            std::sort(candidates.begin(), candidates.end(), candidate_order);
        }
        const auto scoring_end = Clock::now();
        scoring_ms += engine::debug::elapsed_ms(scoring_start, scoring_end);
        const auto update_start = Clock::now();
        for (int64_t i = 0; i < actual_fill; ++i) {
            const auto & candidate = candidates[static_cast<size_t>(i)];
            accepted[candidate.index] = candidate.predicted;
        }
        update_generated_tokens(inference, *impl_->assets, accepted);
        const auto upload_start = Clock::now();
        impl_->forward_graph->update_generated_target_tokens(inference);
        const auto upload_end = Clock::now();
        upload_ms += engine::debug::elapsed_ms(upload_start, upload_end);
        active_indices.erase(
            std::remove_if(
                active_indices.begin(),
                active_indices.end(),
                [&](size_t flat_index) { return accepted[flat_index] != static_cast<int32_t>(mask_id); }),
            active_indices.end());
        const auto update_end = Clock::now();
        update_ms += engine::debug::elapsed_ms(update_start, update_end);
    }

    for (int32_t token : accepted) {
        if (token == static_cast<int32_t>(mask_id)) {
            throw std::runtime_error("OmniVoice generator left masked tokens after iterative decoding");
        }
    }

    OmniVoiceGeneratedAudioTokens out;
    out.frames = target_frames;
    out.codebooks = codebooks;
    out.graph_rebuilt = impl_->last_stats.graph_rebuilt;
    out.graph_total_token_capacity = impl_->last_stats.total_token_capacity;
    out.graph_target_frame_capacity = impl_->last_stats.target_frame_capacity;
    out.forward_ms = upload_ms + compute_ms + readback_ms;
    out.upload_ms = upload_ms;
    out.compute_ms = compute_ms;
    out.readback_ms = readback_ms;
    out.scoring_ms = scoring_ms;
    out.update_ms = update_ms;
    out.decode_steps = executed_steps;
    out.token_ids.resize(static_cast<size_t>(target_frames * codebooks), 0);
    for (int64_t frame = 0; frame < target_frames; ++frame) {
        for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
            out.token_ids[static_cast<size_t>(frame * codebooks + codebook)] =
                accepted[static_cast<size_t>(codebook * target_frames + frame)];
        }
    }
    impl_->last_stats.upload_ms = out.upload_ms;
    impl_->last_stats.compute_ms = out.compute_ms;
    impl_->last_stats.readback_ms = out.readback_ms;
    return out;
}

}  // namespace engine::models::omnivoice
