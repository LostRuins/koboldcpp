#include "engine/models/yue2/nar_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace engine::models::yue2 {
namespace {

namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct Yue2NARWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    engine::modules::LinearWeights vae2llm;
    engine::modules::LinearWeights time0;
    engine::modules::LinearWeights time2;
    core::TensorValue latent_pos_embed;
    engine::modules::DecoderStackWeights nar_stack;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights llm2vae;
};

engine::modules::DecoderLayerWeights load_nar_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Yue2ModelConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    engine::modules::DecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_input_layernorm", config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.q_proj.weight",
        storage_type,
        {config.attention_heads * config.head_dim, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.k_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.v_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_self_attn.q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_self_attn.k_norm", config.head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_pre_mlp_layernorm", config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".nar_mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".nar_mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".nar_mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

std::shared_ptr<const Yue2NARWeights> load_nar_weights(
    const Yue2Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<Yue2NARWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "yue2.nar.weights",
        weight_context_bytes);
    const auto & source = *assets.model_weights;
    const auto & config = assets.config.model;
    weights->vae2llm = binding::linear_from_source(
        *weights->store,
        source,
        "vae2llm",
        storage_type,
        config.hidden_size,
        config.latent_dim,
        true);
    weights->time0 = binding::linear_from_source(
        *weights->store,
        source,
        "time_embedder.mlp.0",
        storage_type,
        config.hidden_size,
        256,
        true);
    weights->time2 = binding::linear_from_source(
        *weights->store,
        source,
        "time_embedder.mlp.2",
        storage_type,
        config.hidden_size,
        config.hidden_size,
        true);
    weights->latent_pos_embed = weights->store->load_tensor(
        source,
        "latent_pos_embed.pe",
        storage_type,
        {config.max_latent_frames, config.hidden_size});
    weights->nar_stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights->nar_stack.layers.push_back(load_nar_layer(*weights->store, source, config, storage_type, layer));
    }
    weights->final_norm = binding::norm_weight_from_source(*weights->store, source, "model.norm", config.hidden_size);
    weights->llm2vae = binding::linear_from_source(
        *weights->store,
        source,
        "llm2vae",
        storage_type,
        config.latent_dim,
        config.hidden_size,
        true);
    weights->store->upload();
    return weights;
}

std::vector<std::pair<int64_t, int64_t>> chunk_ranges(int64_t frames, int64_t prefix_tokens, int64_t context) {
    const int64_t chunk = std::min((context - prefix_tokens - 3) / 2, context);
    if (frames < 1 || chunk < 1) {
        throw std::runtime_error("Yue2 NAR chunk leaves no acoustic context");
    }
    std::vector<std::pair<int64_t, int64_t>> out;
    for (int64_t begin = 0; begin < frames; begin += chunk) {
        out.push_back({begin, std::min(begin + chunk, frames)});
    }
    return out;
}

std::vector<float> timestep_features(float shifted_t, int64_t rows) {
    constexpr int64_t kDim = 256;
    std::vector<float> out(static_cast<size_t>(rows * kDim), 0.0F);
    const int64_t half = kDim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half));
        const float value = shifted_t * freq;
        for (int64_t row = 0; row < rows; ++row) {
            out[static_cast<size_t>(row * kDim + i)] = std::cos(value);
            out[static_cast<size_t>(row * kDim + half + i)] = std::sin(value);
        }
    }
    return out;
}

float shifted_t_value(float raw_t, float shift) {
    const float sigmoid = 1.0F / (1.0F + std::exp(-raw_t));
    return shift * sigmoid / (1.0F + (shift - 1.0F) * sigmoid);
}

float logit_clamped(float t) {
    const float safe = std::min(std::max(t, 1.0e-7F), 1.0F - 1.0e-7F);
    return std::min(std::max(std::log(safe / (1.0F - safe)), -20.0F), 20.0F);
}

struct QKVParts {
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
};

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

QKVParts build_qkv_part(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const engine::modules::DecoderLayerWeights & weights,
    const Yue2ModelConfig & config) {
    auto q = engine::modules::LinearModule({config.hidden_size, config.attention_heads * config.head_dim, false})
                 .build(ctx, input, {weights.self_attention.q_weight, std::nullopt});
    auto k = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                 .build(ctx, input, {weights.self_attention.k_weight, std::nullopt});
    auto v = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                 .build(ctx, input, {weights.self_attention.v_weight, std::nullopt});
    q = reshape_heads(ctx, q, config.attention_heads, config.head_dim);
    k = reshape_heads(ctx, k, config.kv_heads, config.head_dim);
    q = engine::modules::RMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
    k = engine::modules::RMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
    return {q, k, reshape_heads(ctx, v, config.kv_heads, config.head_dim)};
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = engine::modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = core::ensure_backend_addressable_layout(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

// The eager lowering below holds the whole score matrix (query steps x kv steps
// x heads x 4 bytes) in one graph tensor, so its peak allocation grows with the
// square of the song length while a single NAR chunk reaches ~12k frames. Both
// Intel and RADV cap a single buffer at 4 GiB, and a single storage binding
// cannot address more than that either, so the query rows are split into tiles
// keeping one tile's scores below this budget. Softmax is per query row, so the
// split is exact.
constexpr int64_t kAttentionScoreTileBytes = 3LL * 1024LL * 1024LL * 1024LL;

int64_t attention_score_bytes_per_row(int64_t kv_steps, int64_t heads, int64_t batch) {
    return kv_steps * heads * batch * static_cast<int64_t>(ggml_type_size(GGML_TYPE_F32));
}

int64_t attention_tile_count(int64_t steps, int64_t score_bytes_per_row, int64_t forced_tile_rows) {
    const int64_t tile_rows = forced_tile_rows > 0
        ? forced_tile_rows
        : std::max<int64_t>(1, kAttentionScoreTileBytes / std::max<int64_t>(1, score_bytes_per_row));
    return std::max<int64_t>(1, (steps + tile_rows - 1) / tile_rows);
}

core::TensorValue mixed_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q,
    const core::TensorValue & k,
    const core::TensorValue & v,
    const core::TensorValue * attention_mask,
    const Yue2ModelConfig & config,
    core::BackendType backend_type,
    bool allow_flash_attention,
    int64_t attention_tile_rows) {
    auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    if (backend_type != core::BackendType::Cpu && allow_flash_attention) {
        auto * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_heads.tensor,
            k_heads.tensor,
            v_heads.tensor,
            attention_mask != nullptr ? attention_mask->tensor : nullptr,
            1.0F / std::sqrt(static_cast<float>(config.head_dim)),
            0.0F,
            0.0F);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        return core::wrap_tensor(
            flash,
            core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], config.head_dim}),
            GGML_TYPE_F32);
    }

    if (backend_type != core::BackendType::Cpu) {
        // Eager lowering for GPUs whose flash kernel is missing or slow. K/V
        // stay grouped: ggml_mul_mat broadcasts the kv heads over the query
        // heads (ne12 % ne02 == 0), so no F16 repeat is materialized (Vulkan
        // has no REPEAT kernel for F16 and the copies would dominate anyway).
        // Layouts (ggml ne order): q {dim, steps, heads, batch},
        // k {dim, kv_steps, kv_heads, batch}, v^T {kv_steps, dim, kv_heads, batch}.
        // v is {dim, kv_heads, kv_steps, batch}; mul_mat wants kv_steps innermost.
        auto * v_t = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, v.tensor, 1, 2, 0, 3));
        const int64_t steps = q_heads.tensor->ne[1];
        const int64_t tiles = attention_tile_count(
            steps,
            attention_score_bytes_per_row(k_heads.tensor->ne[1], q_heads.tensor->ne[2], q_heads.tensor->ne[3]),
            attention_tile_rows);
        // Spread the rows evenly instead of leaving a small remainder tile: the
        // matmul kernels lose throughput on a short tile.
        const int64_t rows_per_tile = steps / tiles;
        const int64_t wide_tiles = steps % tiles;
        ggml_tensor * joined = nullptr;
        int64_t first_row = 0;
        for (int64_t tile = 0; tile < tiles; ++tile) {
            const int64_t rows = rows_per_tile + (tile < wide_tiles ? 1 : 0);
            auto * q_tile = q_heads.tensor;
            auto * mask_tile = attention_mask != nullptr ? attention_mask->tensor : nullptr;
            if (tiles > 1) {
                // A row slice keeps the head stride of the full tensor. Backends
                // derive the batch stride of a row-contiguous mul_mat operand
                // from ne0*ne1 instead of nb2, so the slice is made contiguous
                // rather than handed over as a view.
                q_tile = ggml_cont(
                    ctx.ggml,
                    ggml_view_4d(
                        ctx.ggml,
                        q_heads.tensor,
                        q_heads.tensor->ne[0],
                        rows,
                        q_heads.tensor->ne[2],
                        q_heads.tensor->ne[3],
                        q_heads.tensor->nb[1],
                        q_heads.tensor->nb[2],
                        q_heads.tensor->nb[3],
                        static_cast<size_t>(first_row) * q_heads.tensor->nb[1]));
                if (mask_tile != nullptr) {
                    throw std::runtime_error("Yue2 NAR tiled attention does not support an attention mask");
                }
            }
            auto * scores = ggml_mul_mat(ctx.ggml, k_heads.tensor, q_tile);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            auto * attn = ggml_soft_max_ext(
                ctx.ggml,
                scores,
                mask_tile,
                1.0F / std::sqrt(static_cast<float>(config.head_dim)),
                0.0F);
            auto * context = ggml_mul_mat(ctx.ggml, v_t, attn);
            ggml_mul_mat_set_prec(context, GGML_PREC_F32);
            // Joining tile by tile keeps the tiles sequential in the graph, so
            // the allocator reuses one tile's score memory for the next.
            joined = joined == nullptr ? context : ggml_concat(ctx.ggml, joined, context, 1);
            first_row += rows;
        }
        return core::wrap_tensor(
            ggml_permute(ctx.ggml, joined, 0, 2, 1, 3),
            core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], config.head_dim}),
            GGML_TYPE_F32);
    }

    const int64_t repeats = config.attention_heads / config.kv_heads;
    k_heads = repeat_kv_heads(ctx, k_heads, repeats);
    v_heads = repeat_kv_heads(ctx, v_heads, repeats);
    auto scores = engine::modules::MatMulModule{}.build(
        ctx,
        q_heads,
        engine::modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            attention_mask != nullptr ? attention_mask->tensor : nullptr,
            1.0F / std::sqrt(static_cast<float>(config.head_dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    auto context = engine::modules::MatMulModule{}.build(ctx, attn, v_heads);
    return engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
}

core::TensorValue build_mlp_part(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const engine::modules::DecoderMLPWeights & weights,
    const Yue2ModelConfig & config) {
    auto gate = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, input, {weights.gate_proj.weight, std::nullopt});
    gate = engine::modules::SiluModule{}.build(ctx, gate);
    auto up = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, input, {weights.up_proj.weight, std::nullopt});
    auto gated = engine::modules::MulModule{}.build(ctx, gate, up);
    return engine::modules::LinearModule({config.intermediate_size, config.hidden_size, false})
        .build(ctx, gated, {weights.down_proj.weight, std::nullopt});
}

core::TensorValue build_cached_nar_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const engine::modules::DecoderLayerWeights & nar_weights,
    const core::TensorValue & ar_key,
    const core::TensorValue & ar_value,
    const Yue2ModelConfig & config,
    core::BackendType backend_type,
    bool allow_flash_attention,
    int64_t attention_tile_rows) {
    auto norm = engine::modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, input, nar_weights.input_norm);
    auto qkv = build_qkv_part(ctx, norm, nar_weights, config);
    qkv.q = engine::modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta})
                .build(ctx, qkv.q, positions);
    qkv.k = engine::modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta})
                .build(ctx, qkv.k, positions);
    qkv.k = core::ensure_backend_addressable_layout(ctx, qkv.k);
    qkv.v = core::ensure_backend_addressable_layout(ctx, qkv.v);
    qkv.k = core::wrap_tensor(ggml_cast(ctx.ggml, qkv.k.tensor, GGML_TYPE_F16), qkv.k.shape, GGML_TYPE_F16);
    qkv.v = core::wrap_tensor(ggml_cast(ctx.ggml, qkv.v.tensor, GGML_TYPE_F16), qkv.v.shape, GGML_TYPE_F16);
    auto k = engine::modules::ConcatModule({1}).build(ctx, ar_key, qkv.k);
    auto v = engine::modules::ConcatModule({1}).build(ctx, ar_value, qkv.v);
    auto context =
        mixed_attention(ctx, qkv.q, k, v, nullptr, config, backend_type, allow_flash_attention, attention_tile_rows);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.attention_heads * config.head_dim}));
    auto attn = engine::modules::LinearModule({config.attention_heads * config.head_dim, config.hidden_size, false})
                    .build(ctx, context, {nar_weights.self_attention.out_weight, std::nullopt});
    auto x = engine::modules::AddModule{}.build(ctx, input, attn);
    norm = engine::modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
               .build(ctx, x, nar_weights.post_norm);
    return engine::modules::AddModule{}.build(ctx, x, build_mlp_part(ctx, norm, nar_weights.mlp, config));
}

}  // namespace

struct Yue2NARRuntime::Impl {
    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType weight_type,
        size_t weight_context_bytes,
        size_t graph_arena_bytes,
        bool allow_flash_attention,
        int64_t attention_tile_rows)
        : execution(execution),
          assets(std::move(assets)),
          graph_arena_bytes(graph_arena_bytes),
          allow_flash_attention(allow_flash_attention),
          attention_tile_rows(attention_tile_rows) {
        if (!this->assets) {
            throw std::runtime_error("Yue2 NAR runtime requires assets");
        }
        weights = load_nar_weights(*this->assets, execution, weight_context_bytes, weight_type);
    }

    struct Graph {
        Graph(
            Impl & owner,
            int64_t frames,
            const Yue2ARDevicePrefixState & ar_state)
            : owner(&owner),
              frames(frames),
              ar_length(ar_state.current_end) {
            const auto & config = owner.assets->config.model;
            if (frames <= 0 || ar_length <= 0) {
                throw std::runtime_error("Yue2 NAR graph requires positive shapes");
            }
            if (static_cast<int64_t>(ar_state.keys.size()) != config.layers ||
                static_cast<int64_t>(ar_state.values.size()) != config.layers) {
                throw std::runtime_error("Yue2 NAR prefix cache layer count mismatch");
            }
            nar_length = frames + 2;
            total_length = ar_length + nar_length;
            ggml_init_params input_params{
                ggml_tensor_overhead() * static_cast<size_t>(3 + config.layers * 2),
                nullptr,
                true};
            input_ctx.reset(ggml_init(input_params));
            ggml_init_params graph_params{owner.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(graph_params));
            if (!input_ctx || !ctx) {
                throw std::runtime_error("failed to initialize Yue2 NAR graph context");
            }
            core::ModuleBuildContext input_build{input_ctx.get(), "yue2.nar.input", owner.execution.backend_type()};
            core::ModuleBuildContext build{ctx.get(), "yue2.nar", owner.execution.backend_type()};
            const auto build_start = Clock::now();
            state = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({1, nar_length, config.latent_dim}));
            time = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 256}));
            positions = core::make_tensor(input_build, GGML_TYPE_I32, core::TensorShape::from_dims({nar_length}));
            ggml_set_input(state.tensor);
            ggml_set_input(time.tensor);
            ggml_set_input(positions.tensor);
            ar_keys.reserve(static_cast<size_t>(config.layers));
            ar_values.reserve(static_cast<size_t>(config.layers));
            for (int64_t layer = 0; layer < config.layers; ++layer) {
                const auto & key = ar_state.keys[static_cast<size_t>(layer)];
                const auto & value = ar_state.values[static_cast<size_t>(layer)];
                if (key.shape.rank != 4 || value.shape.rank != 4 ||
                    key.shape.dims[0] != 1 || value.shape.dims[0] != 1 ||
                    key.shape.dims[1] != ar_length || value.shape.dims[1] != ar_length ||
                    key.shape.dims[2] != config.kv_heads || value.shape.dims[2] != config.kv_heads ||
                    key.shape.dims[3] != config.head_dim || value.shape.dims[3] != config.head_dim ||
                    key.type != GGML_TYPE_F16 || value.type != GGML_TYPE_F16) {
                    throw std::runtime_error("Yue2 NAR prefix cache tensor shape mismatch");
                }
                ar_keys.push_back(key);
                ar_values.push_back(value);
            }
            auto nar_hidden = engine::modules::LinearModule({config.latent_dim, config.hidden_size, true})
                                  .build(build, state, owner.weights->vae2llm);
            auto time_hidden = engine::modules::LinearModule({256, config.hidden_size, true})
                                   .build(build, time, owner.weights->time0);
            time_hidden = engine::modules::SiluModule{}.build(build, time_hidden);
            time_hidden = engine::modules::LinearModule({config.hidden_size, config.hidden_size, true})
                              .build(build, time_hidden, owner.weights->time2);
            time_hidden = core::reshape_tensor(build, time_hidden, core::TensorShape::from_dims({1, 1, config.hidden_size}));
            time_hidden = engine::modules::RepeatModule(
                {core::TensorShape::from_dims({1, nar_length, config.hidden_size})})
                              .build(build, time_hidden);
            auto pos = engine::modules::SliceModule({0, 0, nar_length}).build(build, owner.weights->latent_pos_embed);
            pos = core::reshape_tensor(build, pos, core::TensorShape::from_dims({1, nar_length, config.hidden_size}));
            if (pos.type != GGML_TYPE_F32) {
                pos = core::wrap_tensor(ggml_cast(build.ggml, pos.tensor, GGML_TYPE_F32), pos.shape, GGML_TYPE_F32);
            }
            nar_hidden = engine::modules::AddModule{}.build(build, nar_hidden, time_hidden);
            nar_hidden = engine::modules::AddModule{}.build(build, nar_hidden, pos);
            auto hidden = nar_hidden;

            for (int64_t layer = 0; layer < config.layers; ++layer) {
                hidden = build_cached_nar_layer(
                    build,
                    hidden,
                    positions,
                    owner.weights->nar_stack.layers[static_cast<size_t>(layer)],
                    ar_keys[static_cast<size_t>(layer)],
                    ar_values[static_cast<size_t>(layer)],
                    config,
                    owner.execution.backend_type(),
                    owner.allow_flash_attention,
                    owner.attention_tile_rows);
            }
            hidden = engine::modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                         .build(build, hidden, owner.weights->final_norm);
            auto latent = engine::modules::LinearModule({config.hidden_size, config.latent_dim, true})
                              .build(build, hidden, owner.weights->llm2vae);
            auto content = engine::modules::SliceModule({1, 1, frames}).build(build, latent);
            output = core::ensure_backend_addressable_layout(build, content);
            ggml_set_output(output.tensor);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_build_forward_expand(graph, output.tensor);
            engine::debug::timing_log_scalar("yue2.nar.graph.build_ms", engine::debug::elapsed_ms(build_start));
            engine::debug::timing_log_context_reservation("yue2.nar.graph", ctx.get());
            const auto alloc_start = Clock::now();
            input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), owner.execution.backend());
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.execution.backend()));
            if (input_buffer == nullptr || gallocr == nullptr ||
                !ggml_gallocr_reserve(gallocr, graph) ||
                !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("failed to allocate Yue2 NAR graph");
            }
            engine::debug::timing_log_scalar("yue2.nar.graph.alloc_ms", engine::debug::elapsed_ms(alloc_start));
            std::vector<int32_t> pos_values(static_cast<size_t>(nar_length));
            for (int64_t i = 0; i < nar_length; ++i) {
                pos_values[static_cast<size_t>(i)] = static_cast<int32_t>(ar_length + i);
            }
            ggml_backend_tensor_set(positions.tensor, pos_values.data(), 0, pos_values.size() * sizeof(int32_t));
            engine::debug::timing_log_scalar("yue2.nar.graph.frames", frames);
            engine::debug::timing_log_scalar(
                "yue2.nar.graph.buffer_bytes", ggml_gallocr_get_buffer_size(gallocr, 0));
            if (!owner.allow_flash_attention && owner.execution.backend_type() != core::BackendType::Cpu) {
                engine::debug::timing_log_scalar(
                    "yue2.nar.graph.attention_tiles",
                    attention_tile_count(
                        nar_length,
                        attention_score_bytes_per_row(total_length, config.attention_heads, 1),
                        owner.attention_tile_rows));
            }
        }

        ~Graph() {
            core::release_backend_graph_resources(owner->execution.backend(), graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (input_buffer) {
                ggml_backend_buffer_free(input_buffer);
            }
        }

        bool matches(int64_t f, int64_t ar) const noexcept {
            return frames == f && ar_length == ar;
        }

        std::vector<float> run(
            const std::vector<float> & padded_state,
            const std::vector<float> & time_features) {
            const auto & config = owner->assets->config.model;
            if (static_cast<int64_t>(padded_state.size()) != nar_length * config.latent_dim ||
                static_cast<int64_t>(time_features.size()) != 256) {
                throw std::runtime_error("Yue2 NAR step input shape mismatch");
            }
            core::write_tensor_f32(state, padded_state);
            core::write_tensor_f32(time, time_features);
            const auto status = core::compute_backend_graph(owner->execution.backend(), graph, nullptr, "yue2.nar.velocity");
            ggml_backend_synchronize(owner->execution.backend());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Yue2 NAR graph compute failed");
            }
            std::vector<float> out;
            core::read_tensor_f32_into(output.tensor, out);
            return out;
        }

        Impl * owner = nullptr;
        int64_t frames = 0;
        int64_t ar_length = 0;
        int64_t nar_length = 0;
        int64_t total_length = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_backend_buffer_t input_buffer = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        core::TensorValue state;
        core::TensorValue time;
        core::TensorValue positions;
        std::vector<core::TensorValue> ar_keys;
        std::vector<core::TensorValue> ar_values;
        core::TensorValue output;
    };

    std::vector<float> velocity(
        Graph & graph,
        const Yue2ARDevicePrefixState & ar_state,
        const std::vector<float> & state,
        float raw_t) {
        const auto & config = assets->config.model;
        const int64_t frames = static_cast<int64_t>(state.size()) / config.latent_dim;
        if (frames <= 0 || frames * config.latent_dim != static_cast<int64_t>(state.size())) {
            throw std::runtime_error("Yue2 NAR state shape mismatch");
        }
        if (!graph.matches(frames, ar_state.current_end)) {
            throw std::runtime_error("Yue2 NAR graph shape changed during chunk solve");
        }
        std::vector<float> padded(static_cast<size_t>((frames + 2) * config.latent_dim), 0.0F);
        std::copy(state.begin(), state.end(), padded.begin() + static_cast<std::ptrdiff_t>(config.latent_dim));
        const auto shifted = shifted_t_value(raw_t, config.timestep_shift);
        auto features = timestep_features(shifted, 1);
        return graph.run(padded, features);
    }

    std::vector<float> solve_chunk(
        const Yue2ARDevicePrefixState & ar_state,
        const std::vector<float> & noise,
        int64_t ode_steps) {
        auto state = noise;
        const auto & config = assets->config.model;
        const int64_t frames = static_cast<int64_t>(state.size()) / config.latent_dim;
        graph = std::make_unique<Graph>(*this, frames, ar_state);
        auto & chunk_graph = *graph;
        const float dt = 1.0F / static_cast<float>(ode_steps);
        for (int64_t step = 0; step < ode_steps; ++step) {
            const float t = 1.0F - static_cast<float>(step) * dt;
            const auto first = velocity(chunk_graph, ar_state, state, logit_clamped(t));
            std::vector<float> mid(state.size(), 0.0F);
            for (size_t i = 0; i < state.size(); ++i) {
                mid[i] = state[i] - first[i] * (dt / 2.0F);
            }
            const auto second = velocity(chunk_graph, ar_state, mid, logit_clamped(t - dt / 2.0F));
            for (size_t i = 0; i < state.size(); ++i) {
                state[i] -= second[i] * dt;
            }
        }
        return state;
    }

    std::vector<float> synthesize(
        const std::vector<int32_t> & prefix,
        const std::vector<int32_t> & codec,
        const std::function<Yue2ARDevicePrefixState(const std::vector<int32_t> &)> & prefill_state,
        const std::vector<float> & noise,
        uint64_t seed,
        int64_t ode_steps,
        int64_t context) {
        const auto total_start = Clock::now();
        const auto & config = assets->config.model;
        const auto ranges = chunk_ranges(static_cast<int64_t>(codec.size()), static_cast<int64_t>(prefix.size()), context);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.codec_tokens", codec.size());
        engine::debug::timing_log_scalar("yue2.nar.synthesize.chunks", ranges.size());
        engine::debug::timing_log_scalar("yue2.nar.synthesize.context", context);
        std::vector<float> full_noise(static_cast<size_t>(codec.size() * config.latent_dim), 0.0F);
        if (noise.empty()) {
            std::mt19937 rng(static_cast<uint32_t>(seed));
            std::normal_distribution<float> normal(0.0F, 1.0F);
            for (float & value : full_noise) {
                value = normal(rng);
            }
        } else {
            if (noise.size() != full_noise.size()) {
                throw std::runtime_error("Yue2 nar_noise_file frame count does not match semantic codec count");
            }
            full_noise = noise;
        }
        std::vector<float> out;
        out.reserve(full_noise.size());
        double prefill_ms = 0.0;
        double solve_ms = 0.0;
        for (const auto & [begin, end] : ranges) {
            std::vector<int32_t> ar_tokens = prefix;
            ar_tokens.reserve(prefix.size() + static_cast<size_t>(end - begin) + 1);
            for (int64_t i = begin; i < end; ++i) {
                ar_tokens.push_back(codec[static_cast<size_t>(i)] + kCodecOffset);
            }
            ar_tokens.push_back(kMusicEndToken);
            const size_t begin_elem = static_cast<size_t>(begin * config.latent_dim);
            const size_t end_elem = static_cast<size_t>(end * config.latent_dim);
            std::vector<float> noise(end_elem - begin_elem);
            std::copy(full_noise.begin() + static_cast<std::ptrdiff_t>(begin_elem),
                      full_noise.begin() + static_cast<std::ptrdiff_t>(end_elem),
                      noise.begin());
            const auto prefill_start = Clock::now();
            auto ar_state = prefill_state(ar_tokens);
            prefill_ms += engine::debug::elapsed_ms(prefill_start);
            const auto solve_start = Clock::now();
            auto chunk = solve_chunk(ar_state, noise, ode_steps);
            solve_ms += engine::debug::elapsed_ms(solve_start);
            out.insert(out.end(), chunk.begin(), chunk.end());
        }
        engine::debug::timing_log_scalar("yue2.nar.synthesize.prefill_state_ms", prefill_ms);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.solve_chunks_ms", solve_ms);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const Yue2Assets> assets;
    size_t graph_arena_bytes = 0;
    bool allow_flash_attention = true;
    int64_t attention_tile_rows = 0;
    std::shared_ptr<const Yue2NARWeights> weights;
    std::unique_ptr<Graph> graph;
};

Yue2NARRuntime::Yue2NARRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const Yue2Assets> assets,
    assets::TensorStorageType weight_type,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    bool allow_flash_attention,
    int64_t attention_tile_rows)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          weight_type,
          weight_context_bytes,
          graph_arena_bytes,
          allow_flash_attention,
          attention_tile_rows)) {}

Yue2NARRuntime::~Yue2NARRuntime() = default;

std::vector<float> Yue2NARRuntime::synthesize(
    const std::vector<int32_t> & prefix,
    const std::vector<int32_t> & codec,
    const std::function<Yue2ARDevicePrefixState(const std::vector<int32_t> &)> & prefill_state,
    const std::vector<float> & noise,
    uint64_t seed,
    int64_t ode_steps,
    int64_t context) {
    return impl_->synthesize(prefix, codec, prefill_state, noise, seed, ode_steps, context);
}

void Yue2NARRuntime::release_runtime_graphs() {
    impl_->graph.reset();
}

}  // namespace engine::models::yue2
