#include "engine/models/cosyvoice3/ar.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::cosyvoice3 {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

constexpr float kTopP = 0.8F;
constexpr int64_t kRasWindow = 10;
constexpr float kRasTau = 0.1F;
constexpr std::array<int32_t, 11> kSilentTokens{1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323};
constexpr int64_t kMaxConsecutiveSilentTokens = 5;

modules::CausalDecoderRuntimeConfig make_qwen_config(
    const CosyVoice3Config & config,
    core::BackendType backend_type,
    size_t graph_arena_bytes) {
    modules::CausalDecoderRuntimeConfig out;
    out.trace_name = "cosyvoice3.ar";
    out.prefill_graph_arena_bytes = graph_arena_bytes;
    out.decode_graph_arena_bytes = graph_arena_bytes;
    out.decoder.stack.hidden_size = config.hidden_size;
    out.decoder.stack.num_attention_heads = config.heads;
    out.decoder.stack.num_key_value_heads = config.kv_heads;
    out.decoder.stack.head_dim = config.head_dim;
    out.decoder.stack.intermediate_size = config.intermediate_size;
    out.decoder.stack.layers = config.layers;
    out.decoder.stack.rms_norm_eps = 1.0e-6F;
    out.decoder.stack.rope_theta = 1000000.0F;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.use_qk_norm = false;
    out.decoder.stack.attention_precision = GGML_PREC_F32;
    out.decoder.stack.projection_precision = GGML_PREC_DEFAULT;
    out.decoder.stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.decoder.logits_size = config.speech_token_size + config.speech_reserved_tokens;
    out.decoder.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    out.output_mode = modules::CausalDecoderOutputMode::Logits;
    out.logits_readback_token_ids.reserve(static_cast<size_t>(out.decoder.logits_size));
    for (int32_t token = 0; token < static_cast<int32_t>(out.decoder.logits_size); ++token) {
        out.logits_readback_token_ids.push_back(token);
    }
    if (backend_type == core::BackendType::Metal) {
        out.decoder.lm_head_input_type = GGML_TYPE_F32;
    } else if (backend_type == core::BackendType::Vulkan) {
        out.decoder.lm_head_input_type = GGML_TYPE_F16;
    }
    return out;
}

modules::DecoderLayerWeights load_qwen_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const CosyVoice3Config & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "llm.model.model.layers." + std::to_string(layer);
    modules::DecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.heads * config.head_dim, config.hidden_size});
    out.self_attention.q_bias = store.load_f32_tensor(
        source,
        prefix + ".self_attn.q_proj.bias",
        {config.heads * config.head_dim});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.k_bias = store.load_f32_tensor(
        source,
        prefix + ".self_attn.k_proj.bias",
        {config.kv_heads * config.head_dim});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_bias = store.load_f32_tensor(
        source,
        prefix + ".self_attn.v_proj.bias",
        {config.kv_heads * config.head_dim});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.heads * config.head_dim});
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

struct CosyVoice3Qwen2ARWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::CausalDecoderRuntimeWeights qwen;
    std::vector<float> text_embedding;
    std::vector<float> speech_embedding;
};

std::shared_ptr<CosyVoice3Qwen2ARWeights> load_ar_weights(
    const CosyVoice3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type,
    const modules::CausalDecoderRuntimeConfig & qwen_config) {
    auto weights = std::make_shared<CosyVoice3Qwen2ARWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "cosyvoice3.ar.weights",
        weight_context_bytes);
    const auto & source = *assets.llm_weights;
    const auto & config = assets.config;
    weights->qwen.token_embedding = weights->store->load_tensor(
        source,
        "llm.model.model.embed_tokens.weight",
        storage_type,
        {config.text_vocab_size, config.hidden_size});
    weights->qwen.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights->qwen.stack.layers.push_back(load_qwen_layer(*weights->store, source, config, storage_type, layer));
    }
    weights->qwen.final_norm = binding::norm_weight_from_source(
        *weights->store,
        source,
        "llm.model.model.norm",
        config.hidden_size);
    weights->qwen.lm_head = binding::linear_from_source(
        *weights->store,
        source,
        "llm_decoder",
        storage_type,
        qwen_config.decoder.logits_size,
        config.hidden_size,
        false);
    weights->text_embedding = source.require_f32(
        "llm.model.model.embed_tokens.weight",
        {config.text_vocab_size, config.hidden_size});
    weights->speech_embedding = source.require_f32(
        "speech_embedding.weight",
        {config.speech_token_size + config.speech_reserved_tokens, config.hidden_size});
    weights->store->upload();
    assets.llm_weights->release_storage();
    return weights;
}

void append_embedding_rows(
    std::vector<float> & out,
    const std::vector<float> & table,
    int64_t rows,
    int64_t dim,
    const std::vector<int32_t> & tokens,
    const char * label) {
    for (const int32_t token : tokens) {
        if (token < 0 || token >= rows) {
            throw std::runtime_error(std::string("CosyVoice3 AR ") + label + " token is outside embedding table");
        }
        const size_t begin = static_cast<size_t>(token) * static_cast<size_t>(dim);
        out.insert(out.end(), table.begin() + static_cast<std::ptrdiff_t>(begin), table.begin() + static_cast<std::ptrdiff_t>(begin + static_cast<size_t>(dim)));
    }
}

std::vector<float> embedding_row(
    const std::vector<float> & table,
    int64_t rows,
    int64_t dim,
    int32_t token,
    const char * label) {
    if (token < 0 || token >= rows) {
        throw std::runtime_error(std::string("CosyVoice3 AR ") + label + " token is outside embedding table");
    }
    const size_t begin = static_cast<size_t>(token) * static_cast<size_t>(dim);
    return std::vector<float>(
        table.begin() + static_cast<std::ptrdiff_t>(begin),
        table.begin() + static_cast<std::ptrdiff_t>(begin + static_cast<size_t>(dim)));
}

std::vector<float> log_softmax(const std::vector<float> & logits) {
    float max_value = -std::numeric_limits<float>::infinity();
    for (const float value : logits) {
        if (std::isfinite(value)) {
            max_value = std::max(max_value, value);
        }
    }
    if (!std::isfinite(max_value)) {
        throw std::runtime_error("CosyVoice3 AR logits have no finite value");
    }
    double sum = 0.0;
    for (const float value : logits) {
        if (std::isfinite(value)) {
            sum += std::exp(static_cast<double>(value - max_value));
        }
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("CosyVoice3 AR logits have invalid probability mass");
    }
    const float log_sum = max_value + static_cast<float>(std::log(sum));
    std::vector<float> out(logits.size(), -std::numeric_limits<float>::infinity());
    for (size_t index = 0; index < logits.size(); ++index) {
        if (std::isfinite(logits[index])) {
            out[index] = logits[index] - log_sum;
        }
    }
    return out;
}

int32_t sample_from_scores(
    const std::vector<float> & scores,
    sampling::HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const sampling::TorchCudaSamplingPolicy & policy,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks,
    std::string_view context) {
    const sampling::HfTorchSamplingState torch_state{
        &policy,
        seed,
        sample_call_index,
        rng_offset_blocks,
        true,
    };
    const int32_t token = sampling::HfTokenSampler::sample_from_processed_scores(
        scores,
        scratch,
        fallback_rng,
        policy.cuda_fast_path ? &torch_state : nullptr,
        context);
    ++sample_call_index;
    if (policy.cuda_fast_path) {
        rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(scores.size()),
            policy);
    }
    return token;
}

int32_t nucleus_sample(
    const std::vector<float> & log_probs,
    int64_t top_k,
    sampling::HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const sampling::TorchCudaSamplingPolicy & policy,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks) {
    std::vector<int32_t> order(log_probs.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
        return log_probs[static_cast<size_t>(lhs)] > log_probs[static_cast<size_t>(rhs)];
    });

    std::vector<int32_t> kept;
    std::vector<float> kept_scores;
    kept.reserve(static_cast<size_t>(std::min<int64_t>(top_k, static_cast<int64_t>(order.size()))));
    kept_scores.reserve(kept.capacity());
    double cumulative = 0.0;
    for (const int32_t token : order) {
        const float score = log_probs[static_cast<size_t>(token)];
        if (!std::isfinite(score)) {
            continue;
        }
        if (cumulative < static_cast<double>(kTopP) && static_cast<int64_t>(kept.size()) < top_k) {
            cumulative += std::exp(static_cast<double>(score));
            kept.push_back(token);
            kept_scores.push_back(score);
        } else {
            break;
        }
    }
    if (kept.empty()) {
        throw std::runtime_error("CosyVoice3 AR nucleus sampler has no finite candidates");
    }
    const int32_t local = sample_from_scores(
        kept_scores,
        scratch,
        fallback_rng,
        policy,
        seed,
        sample_call_index,
        rng_offset_blocks,
        "CosyVoice3 AR nucleus sampler");
    return kept[static_cast<size_t>(local)];
}

int32_t ras_sample(
    std::vector<float> log_probs,
    const std::vector<int32_t> & decoded_tokens,
    int64_t top_k,
    sampling::HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const sampling::TorchCudaSamplingPolicy & policy,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks) {
    const int32_t top_id = nucleus_sample(
        log_probs,
        top_k,
        scratch,
        fallback_rng,
        policy,
        seed,
        sample_call_index,
        rng_offset_blocks);
    int64_t repeat_count = 0;
    const size_t window = std::min<size_t>(decoded_tokens.size(), static_cast<size_t>(kRasWindow));
    for (size_t index = decoded_tokens.size() - window; index < decoded_tokens.size(); ++index) {
        if (decoded_tokens[index] == top_id) {
            ++repeat_count;
        }
    }
    if (static_cast<float>(repeat_count) >= static_cast<float>(kRasWindow) * kRasTau) {
        log_probs[static_cast<size_t>(top_id)] = -std::numeric_limits<float>::infinity();
        return sample_from_scores(
            log_probs,
            scratch,
            fallback_rng,
            policy,
            seed,
            sample_call_index,
            rng_offset_blocks,
            "CosyVoice3 AR RAS fallback sampler");
    }
    return top_id;
}

bool is_silent_token(int32_t token) {
    return std::find(kSilentTokens.begin(), kSilentTokens.end(), token) != kSilentTokens.end();
}

}  // namespace

class CosyVoice3Qwen2ARRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          execution_(execution),
          sampling_policy_(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "cosyvoice3.ar.sampling",
              "CosyVoice3 AR",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("CosyVoice3 AR runtime requires assets");
        }
        qwen_config_ = make_qwen_config(assets_->config, execution_.backend_type(), graph_arena_bytes);
        weights_ = load_ar_weights(*assets_, execution_, weight_context_bytes, storage_type, qwen_config_);
        qwen_runtime_ = std::make_unique<modules::CausalDecoderRuntime>(
            execution_,
            qwen_config_,
            weights_->qwen);
    }

    CosyVoice3AROutput generate(const CosyVoice3ARRequest & request) {
        const auto & config = assets_->config;
        if (request.target_text_tokens.empty()) {
            throw std::runtime_error("CosyVoice3 AR target text tokens are empty");
        }
        if (request.top_k <= 0) {
            throw std::runtime_error("CosyVoice3 AR top_k must be positive");
        }
        const int64_t min_len = request.min_tokens >= 0
            ? request.min_tokens
            : static_cast<int64_t>(request.target_text_tokens.size()) * 2;
        const int64_t max_len = request.max_tokens >= 0
            ? request.max_tokens
            : static_cast<int64_t>(request.target_text_tokens.size()) * 20;
        if (min_len < 0 || max_len <= 0 || min_len > max_len) {
            throw std::runtime_error("CosyVoice3 AR token length bounds are invalid");
        }

        const int32_t sos = static_cast<int32_t>(config.speech_token_size);
        const int32_t task_id = static_cast<int32_t>(config.speech_token_size + 2);
        const int64_t speech_embedding_rows = config.speech_token_size + config.speech_reserved_tokens;

        std::vector<int32_t> text_tokens = request.prompt_text_tokens;
        text_tokens.insert(text_tokens.end(), request.target_text_tokens.begin(), request.target_text_tokens.end());

        const int64_t prefill_steps =
            1 + static_cast<int64_t>(text_tokens.size()) + 1 + static_cast<int64_t>(request.prompt_speech_tokens.size());
        std::vector<float> embeddings;
        embeddings.reserve(static_cast<size_t>(prefill_steps * config.hidden_size));
        append_embedding_rows(
            embeddings,
            weights_->speech_embedding,
            speech_embedding_rows,
            config.hidden_size,
            std::vector<int32_t>{sos},
            "sos");
        append_embedding_rows(
            embeddings,
            weights_->text_embedding,
            config.text_vocab_size,
            config.hidden_size,
            text_tokens,
            "text");
        append_embedding_rows(
            embeddings,
            weights_->speech_embedding,
            speech_embedding_rows,
            config.hidden_size,
            std::vector<int32_t>{task_id},
            "task");
        append_embedding_rows(
            embeddings,
            weights_->speech_embedding,
            speech_embedding_rows,
            config.hidden_size,
            request.prompt_speech_tokens,
            "prompt speech");

        const int64_t required_cache_steps = prefill_steps + max_len;
        auto timing_start = Clock::now();
        qwen_runtime_->release_runtime_graphs();
        auto prefill = qwen_runtime_->prefill_embeddings(embeddings, prefill_steps);
        debug::timing_log_scalar("cosyvoice3.ar.prefill.total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        qwen_runtime_->start_decode_embeddings(prefill.state, required_cache_steps);

        sampling::HfSamplerScratch scratch;
        scratch.reserve_vocab(static_cast<size_t>(config.speech_token_size + config.speech_reserved_tokens));
        std::mt19937 fallback_rng(request.seed);
        uint64_t sample_call_index = 0;
        uint64_t rng_offset_blocks = 0;

        CosyVoice3AROutput out;
        std::vector<int32_t> decoded_tokens;
        int64_t consecutive_silent_tokens = 0;
        int64_t filtered_silent_tokens = 0;
        std::vector<float> logits = std::move(prefill.logits);
        timing_start = Clock::now();
        for (int64_t step = 0; step < max_len; ++step) {
            auto log_probs = log_softmax(logits);
            if (step < min_len) {
                log_probs[static_cast<size_t>(sos)] = -std::numeric_limits<float>::infinity();
            }
            const int32_t token = ras_sample(
                std::move(log_probs),
                decoded_tokens,
                request.top_k,
                scratch,
                fallback_rng,
                sampling_policy_,
                request.seed,
                sample_call_index,
                rng_offset_blocks);
            if (token >= config.speech_token_size) {
                break;
            }
            decoded_tokens.push_back(token);
            bool append_token = true;
            if (is_silent_token(token)) {
                ++consecutive_silent_tokens;
                if (consecutive_silent_tokens > kMaxConsecutiveSilentTokens) {
                    append_token = false;
                    ++filtered_silent_tokens;
                }
            } else {
                consecutive_silent_tokens = 0;
            }
            if (append_token) {
                out.speech_tokens.push_back(token);
            }
            logits = qwen_runtime_->decode_embedding(embedding_row(
                weights_->speech_embedding,
                speech_embedding_rows,
                config.hidden_size,
                token,
                "sampled speech")).logits;
        }
        debug::timing_log_scalar("cosyvoice3.ar.decode.total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        (void) prefill_steps;
        (void) filtered_silent_tokens;
        return out;
    }

    void release_graphs() {
        if (qwen_runtime_ != nullptr) {
            qwen_runtime_->release_runtime_graphs();
        }
    }

private:
    std::shared_ptr<const CosyVoice3Assets> assets_;
    core::ExecutionContext & execution_;
    modules::CausalDecoderRuntimeConfig qwen_config_;
    std::shared_ptr<CosyVoice3Qwen2ARWeights> weights_;
    std::unique_ptr<modules::CausalDecoderRuntime> qwen_runtime_;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
};

CosyVoice3Qwen2ARRuntime::CosyVoice3Qwen2ARRuntime(
    std::shared_ptr<const CosyVoice3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

CosyVoice3Qwen2ARRuntime::~CosyVoice3Qwen2ARRuntime() = default;

CosyVoice3AROutput CosyVoice3Qwen2ARRuntime::generate(const CosyVoice3ARRequest & request) {
    return impl_->generate(request);
}

void CosyVoice3Qwen2ARRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::cosyvoice3
