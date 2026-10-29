#include "engine/models/breeze_tts/text_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/text_encoders/t5_gemma_encoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::breeze_tts {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::T5GemmaEncoderConfig text_config(const BreezeTTSConfig & config) {
    modules::T5GemmaEncoderConfig out;
    out.hidden_size = config.text_hidden_size;
    out.layers = config.text_layers;
    out.attention_heads = config.text_heads;
    out.kv_heads = config.text_kv_heads;
    out.head_dim = config.text_head_dim;
    out.attention_size = config.text_heads * config.text_head_dim;
    out.intermediate_size = config.text_intermediate_size;
    out.vocab_size = config.text_vocab_size;
    out.rope_theta = config.text_rope_theta;
    out.rope_freq_scale = 1.0F / config.text_rope_linear_factor;
    out.rms_norm_eps = config.text_rms_norm_eps;
    out.query_pre_attn_scalar = config.text_query_pre_attn_scalar;
    out.attn_logit_softcap = 0.0F;
    out.scale_embeddings = true;
    out.use_qk_norm = true;
    out.rms_norm_style = modules::T5GemmaRMSNormStyle::Gemma;
    out.layer_rope_theta = config.text_layer_rope_theta;
    out.layer_rope_freq_scale = config.text_layer_rope_freq_scale;
    return out;
}

modules::T5GemmaEncoderLayerWeights load_text_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const BreezeTTSConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "text_encoder.layers." + std::to_string(layer);
    modules::T5GemmaEncoderLayerWeights out;
    out.pre_self_attn_norm = store.load_f32_tensor(source, prefix + ".pre_self_attn_layernorm.weight", {config.text_hidden_size});
    out.post_self_attn_norm = store.load_f32_tensor(source, prefix + ".post_self_attn_layernorm.weight", {config.text_hidden_size});
    out.pre_ff_norm = store.load_f32_tensor(source, prefix + ".pre_feedforward_layernorm.weight", {config.text_hidden_size});
    out.post_ff_norm = store.load_f32_tensor(source, prefix + ".post_feedforward_layernorm.weight", {config.text_hidden_size});
    out.q_proj = binding::linear_from_source(
        store, source, prefix + ".self_attn.q_proj", storage_type, config.text_heads * config.text_head_dim, config.text_hidden_size, false);
    out.k_proj = binding::linear_from_source(
        store, source, prefix + ".self_attn.k_proj", storage_type, config.text_kv_heads * config.text_head_dim, config.text_hidden_size, false);
    out.v_proj = binding::linear_from_source(
        store, source, prefix + ".self_attn.v_proj", storage_type, config.text_kv_heads * config.text_head_dim, config.text_hidden_size, false);
    out.o_proj = binding::linear_from_source(
        store, source, prefix + ".self_attn.o_proj", storage_type, config.text_hidden_size, config.text_heads * config.text_head_dim, false);
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.text_head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.text_head_dim);
    out.gate_proj = binding::linear_from_source(
        store, source, prefix + ".mlp.gate_proj", storage_type, config.text_intermediate_size, config.text_hidden_size, false);
    out.up_proj = binding::linear_from_source(
        store, source, prefix + ".mlp.up_proj", storage_type, config.text_intermediate_size, config.text_hidden_size, false);
    out.down_proj = binding::linear_from_source(
        store, source, prefix + ".mlp.down_proj", storage_type, config.text_hidden_size, config.text_intermediate_size, false);
    return out;
}

struct BreezeT5Gemma2TextWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::T5GemmaEncoderWeights encoder;
    modules::LinearWeights projector;
};

std::shared_ptr<const BreezeT5Gemma2TextWeights> load_text_weights(
    const BreezeTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto out = std::make_shared<BreezeT5Gemma2TextWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "breeze_tts.text_encoder.weights",
        weight_context_bytes);
    const auto & source = *assets.weights;
    const auto & config = assets.config;
    out->encoder.embed_tokens = out->store->load_tensor(
        source,
        "text_encoder.embed_tokens.weight",
        storage_type,
        {config.text_vocab_size, config.text_hidden_size});
    out->encoder.layers.reserve(static_cast<size_t>(config.text_layers));
    for (int64_t layer = 0; layer < config.text_layers; ++layer) {
        out->encoder.layers.push_back(load_text_layer(*out->store, source, config, storage_type, layer));
    }
    out->encoder.norm = out->store->load_f32_tensor(source, "text_encoder.norm.weight", {config.text_hidden_size});
    out->projector = binding::linear_from_source(
        *out->store,
        source,
        "text_encoder_proj",
        storage_type,
        config.hidden_size,
        config.text_hidden_size,
        false);
    out->store->upload();
    return out;
}

std::vector<float> full_attention_mask(int64_t heads, int64_t tokens) {
    return std::vector<float>(static_cast<size_t>(heads * tokens * tokens), 0.0F);
}

}  // namespace

struct BreezeT5Gemma2TextEncoderRuntime::Impl {
    struct Graph {
        Graph(
            core::ExecutionContext & execution,
            size_t graph_arena_bytes,
            const BreezeTTSConfig & config,
            std::shared_ptr<const BreezeT5Gemma2TextWeights> weights,
            int64_t tokens)
            : execution(execution),
              config(config),
              weights(std::move(weights)),
              tokens(tokens) {
            ctx.reset(ggml_init({graph_arena_bytes, nullptr, true}));
            if (ctx == nullptr) {
                throw std::runtime_error("failed to initialize BreezeTTS text encoder graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "breeze_tts.text_encoder", execution.backend_type()};
            input_ids_value = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({1, tokens}));
            positions_value = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({tokens}));
            attention_value = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.text_heads, tokens, tokens}));
            input_ids = input_ids_value.tensor;
            positions = positions_value.tensor;
            attention = attention_value.tensor;
            auto encoded = modules::T5GemmaEncoderModule(text_config(config)).build(
                build,
                input_ids_value,
                positions_value,
                attention_value,
                this->weights->encoder);
            auto projected = modules::LinearModule({config.text_hidden_size, config.hidden_size, false, GGML_PREC_DEFAULT}).build(
                build,
                encoded,
                this->weights->projector);
            output = core::ensure_backend_addressable_layout(build, projected).tensor;
            ggml_set_input(input_ids);
            ggml_set_input(positions);
            ggml_set_input(attention);
            ggml_set_output(output);
            graph = ggml_new_graph_custom(ctx.get(), 65536, false);
            ggml_build_forward_expand(graph, output);
            if (core::is_host_backend(execution.backend())) {
                params_buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), execution.backend());
            }
            galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (galloc == nullptr || !ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
                throw std::runtime_error("failed to allocate BreezeTTS text encoder graph");
            }
        }

        ~Graph() {
            engine::core::release_backend_graph_resources(execution.backend(), graph);
            if (galloc != nullptr) {
                ggml_gallocr_free(galloc);
            }
            if (params_buffer != nullptr) {
                ggml_backend_buffer_free(params_buffer);
            }
        }

        BreezeProjectedText run(const std::vector<int32_t> & ids) {
            if (static_cast<int64_t>(ids.size()) != tokens) {
                throw std::runtime_error("BreezeTTS text encoder graph token count mismatch");
            }
            std::vector<int32_t> pos(static_cast<size_t>(tokens));
            for (int64_t i = 0; i < tokens; ++i) {
                pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            const auto mask = full_attention_mask(config.text_heads, tokens);
            core::write_tensor_i32(input_ids_value, ids);
            core::write_tensor_i32(positions_value, pos);
            core::write_tensor_f32(attention_value, mask);
            core::set_backend_threads(execution.backend(), execution.config().threads);
            if (core::compute_backend_graph(execution.backend(), graph, nullptr, "breeze_tts.text_encoder") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("BreezeTTS text encoder graph compute failed");
            }
            return {tokens, core::read_tensor_f32(output)};
        }

        core::ExecutionContext & execution;
        BreezeTTSConfig config;
        std::shared_ptr<const BreezeT5Gemma2TextWeights> weights;
        int64_t tokens = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        core::TensorValue input_ids_value;
        core::TensorValue positions_value;
        core::TensorValue attention_value;
        ggml_tensor * input_ids = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * attention = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_backend_buffer_t params_buffer = nullptr;
    };

    Impl(
        std::shared_ptr<const BreezeTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(assets)),
          execution(execution),
          graph_arena_bytes(graph_arena_bytes),
          weights(load_text_weights(*this->assets, execution, weight_context_bytes, storage_type)) {}

    BreezeProjectedText encode(const std::vector<int32_t> & input_ids) {
        const auto start = Clock::now();
        const int64_t tokens = static_cast<int64_t>(input_ids.size());
        if (tokens <= 0) {
            throw std::runtime_error("BreezeTTS text encoder requires non-empty input");
        }
        if (graph == nullptr || graph->tokens != tokens) {
            graph = std::make_unique<Graph>(execution, graph_arena_bytes, assets->config, weights, tokens);
        }
        auto out = graph->run(input_ids);
        engine::debug::timing_log_scalar("breeze_tts.text_encoder.total_ms", engine::debug::elapsed_ms(start));
        return out;
    }

    void release_runtime_graphs() {
        graph.reset();
    }

    std::shared_ptr<const BreezeTTSAssets> assets;
    core::ExecutionContext & execution;
    size_t graph_arena_bytes = 0;
    std::shared_ptr<const BreezeT5Gemma2TextWeights> weights;
    std::unique_ptr<Graph> graph;
};

BreezeT5Gemma2TextEncoderRuntime::BreezeT5Gemma2TextEncoderRuntime(
    std::shared_ptr<const BreezeTTSAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, graph_arena_bytes, weight_context_bytes, storage_type)) {}

BreezeT5Gemma2TextEncoderRuntime::~BreezeT5Gemma2TextEncoderRuntime() = default;

BreezeProjectedText BreezeT5Gemma2TextEncoderRuntime::encode(const std::vector<int32_t> & input_ids) {
    return impl_->encode(input_ids);
}

void BreezeT5Gemma2TextEncoderRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::breeze_tts
