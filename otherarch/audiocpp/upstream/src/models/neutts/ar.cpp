#include "engine/models/neutts/ar.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/neutts/backbone.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::neutts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kNeuTTSGenerationContext = 2048;

std::shared_ptr<const NeuTTSAssets> require_assets(
    std::shared_ptr<const NeuTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("NeuTTS AR runtime requires assets");
    }
    return assets;
}

class ARWeightsRuntime {
public:
    ARWeightsRuntime(
        std::shared_ptr<const NeuTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(require_assets(std::move(assets))),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          device_(execution.config().device),
          weights_(std::make_shared<NeuTTSQwen3Weights>(
              load_neutts_backbone_weights(
                  *assets_,
                  backend_,
                  backend_type_,
                  weight_context_bytes,
                  storage_type))) {
        if (backend_ == nullptr) {
            throw std::runtime_error("NeuTTS AR backend is not initialized");
        }
    }

    const NeuTTSAssets & assets() const noexcept {
        return *assets_;
    }

    const NeuTTSQwen3Weights & weights() const noexcept {
        return *weights_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int device() const noexcept {
        return device_;
    }

private:
    std::shared_ptr<const NeuTTSAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int device_ = 0;
    std::shared_ptr<const NeuTTSQwen3Weights> weights_;
};

modules::CausalDecoderRuntimeConfig make_qwen3_decode_runtime_config(
    const NeuTTSBackboneConfig & backbone,
    core::BackendType backend_type,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes) {
    modules::CausalDecoderRuntimeConfig config;
    config.trace_name = "neutts.ar";
    config.decoder = make_neutts_qwen3_config(backbone, backend_type);
    config.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
    config.decode_graph_arena_bytes = decode_graph_arena_bytes;
    return config;
}

modules::CausalDecoderRuntimeWeights make_qwen3_decode_runtime_weights(const NeuTTSQwen3Weights & weights) {
    modules::CausalDecoderRuntimeWeights out;
    out.token_embedding = weights.token_embedding;
    out.stack = weights.decoder.stack;
    out.final_norm = weights.decoder.final_norm;
    out.lm_head = weights.decoder.lm_head;
    return out;
}

bool is_speech_token(int32_t token, int32_t speech_token_start, int32_t speech_token_end) {
    return token >= speech_token_start && token <= speech_token_end;
}

void mask_eos_before_min_tokens(
    std::vector<float> & logits,
    int32_t eos_token_id,
    int64_t generated_tokens,
    int64_t min_tokens) {
    if (generated_tokens >= min_tokens) {
        return;
    }
    if (eos_token_id < 0 || static_cast<size_t>(eos_token_id) >= logits.size()) {
        throw std::runtime_error("NeuTTS AR eos token is outside the vocabulary");
    }
    logits[static_cast<size_t>(eos_token_id)] = -std::numeric_limits<float>::infinity();
}

void round_finite_logits_to_bf16(std::vector<float> & logits) {
    for (float & value : logits) {
        if (std::isfinite(value)) {
            value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
        }
    }
}

}  // namespace

struct NeuTTSQwen3ARRuntime::Impl {
    Impl(
        std::shared_ptr<const NeuTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : weights(std::make_shared<ARWeightsRuntime>(
              std::move(assets),
              execution,
              weight_context_bytes,
              storage_type)),
          qwen3_runtime(std::make_unique<modules::CausalDecoderRuntime>(
              execution,
              make_qwen3_decode_runtime_config(
                  weights->assets().backbone,
                  weights->backend_type(),
                  prefill_graph_arena_bytes,
                  decode_graph_arena_bytes),
              make_qwen3_decode_runtime_weights(weights->weights()))),
          sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
              weights->backend_type(),
              weights->device(),
              "neutts.ar.sampling",
              "NeuTTS AR",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {}

    NeuTTSGeneratedCodes generate(
        const std::vector<int32_t> & prompt_ids,
        int32_t speech_token_start,
        int32_t speech_token_end,
        int32_t speech_generation_end,
        const NeuTTSGenerationOptions & options,
        const std::function<void(int32_t)> & on_speech_code) {
        const auto & config = weights->assets().backbone;
        if (prompt_ids.empty()) {
            throw std::runtime_error("NeuTTS AR prompt is empty");
        }
        if (options.max_tokens < 0) {
            throw std::runtime_error("NeuTTS AR max_tokens must be non-negative");
        }
        if (options.min_tokens < 0 || (options.max_tokens > 0 && options.min_tokens > options.max_tokens)) {
            throw std::runtime_error("NeuTTS AR min_tokens is invalid");
        }
        if (speech_token_start <= 0 || speech_token_end < speech_token_start ||
            speech_token_end >= config.vocab_size) {
            throw std::runtime_error("NeuTTS AR speech token range is invalid");
        }
        if (speech_generation_end < 0 || speech_generation_end >= config.vocab_size) {
            throw std::runtime_error("NeuTTS AR speech generation end token is invalid");
        }
        const int64_t prompt_steps = static_cast<int64_t>(prompt_ids.size());
        const int64_t generation_context = std::min<int64_t>(config.max_context, kNeuTTSGenerationContext);
        const int64_t max_new_tokens = options.max_tokens > 0 ? options.max_tokens : generation_context - prompt_steps;
        const int64_t required_cache_steps = prompt_steps + max_new_tokens;
        if (required_cache_steps > generation_context) {
            throw std::runtime_error("NeuTTS AR prompt plus max_tokens exceeds model context");
        }
        if (options.min_tokens > max_new_tokens) {
            throw std::runtime_error("NeuTTS AR min_tokens exceeds available generation room");
        }
        debug::trace_log_scalar("neutts.ar.generation_context", generation_context);
        debug::trace_log_scalar("neutts.ar.max_new_tokens.effective", max_new_tokens);
        auto timing_start = Clock::now();
        auto prefill = qwen3_runtime->prefill_tokens(prompt_ids);
        debug::timing_log_scalar(
            "neutts.ar.prefill.total_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        qwen3_runtime->start_decode_tokens(prefill.state, required_cache_steps);

        sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = true;
        sampling_options.temperature = options.temperature;
        sampling_options.top_k = options.top_k;
        sampling_options.top_p = 1.0F;
        sampling_options.min_tokens_to_keep = 1;
        sampling_options.repetition_penalty = 1.0F;
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        scratch.reserve_vocab(static_cast<size_t>(config.vocab_size));
        std::mt19937 fallback_rng(static_cast<uint32_t>(options.seed));
        uint64_t sample_call_index = 0;
        std::vector<int32_t> history(prompt_ids.begin(), prompt_ids.end());
        std::vector<float> logits = std::move(prefill.logits);
        debug::trace_log_scalar("neutts.ar.sampling.seed", options.seed);

        NeuTTSGeneratedCodes out;
        timing_start = Clock::now();
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            mask_eos_before_min_tokens(logits, speech_generation_end, step, options.min_tokens);
            round_finite_logits_to_bf16(logits);
            const sampling::HfTorchSamplingState torch_state{
                &sampling_policy,
                options.seed,
                sample_call_index++,
            };
            const int32_t token = sampler.sample(
                logits,
                history,
                sampling_options,
                scratch,
                fallback_rng,
                sampling_policy.cuda_fast_path ? &torch_state : nullptr,
                "NeuTTS AR");
            if (token == speech_generation_end && step >= options.min_tokens) {
                break;
            }
            history.push_back(token);
            out.token_ids.push_back(token);
            if (is_speech_token(token, speech_token_start, speech_token_end)) {
                out.speech_codes.push_back(token - speech_token_start);
                if (on_speech_code) on_speech_code(out.speech_codes.back());
            }
            logits = qwen3_runtime->decode_token(token).logits;
        }
        debug::timing_log_scalar(
            "neutts.ar.decode.total_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        debug::trace_log_scalar("neutts.ar.generated_tokens", static_cast<int64_t>(out.token_ids.size()));
        debug::trace_log_scalar("neutts.ar.generated_speech_codes", static_cast<int64_t>(out.speech_codes.size()));
        return out;
    }

    std::shared_ptr<ARWeightsRuntime> weights;
    std::unique_ptr<modules::CausalDecoderRuntime> qwen3_runtime;
    sampling::TorchCudaSamplingPolicy sampling_policy;
};

NeuTTSQwen3ARRuntime::NeuTTSQwen3ARRuntime(
    std::shared_ptr<const NeuTTSAssets> assets,
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

NeuTTSQwen3ARRuntime::~NeuTTSQwen3ARRuntime() = default;

NeuTTSGeneratedCodes NeuTTSQwen3ARRuntime::generate(
    const std::vector<int32_t> & prompt_ids,
    int32_t speech_token_start,
    int32_t speech_token_end,
    int32_t speech_generation_end,
    const NeuTTSGenerationOptions & options,
    const std::function<void(int32_t)> & on_speech_code) {
    return impl_->generate(prompt_ids, speech_token_start, speech_token_end, speech_generation_end, options, on_speech_code);
}

void NeuTTSQwen3ARRuntime::release_runtime_graphs() {
    if (impl_ != nullptr && impl_->qwen3_runtime != nullptr) {
        impl_->qwen3_runtime->release_runtime_graphs();
    }
}

}  // namespace engine::models::neutts
