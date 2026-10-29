#include "engine/models/ming3_tts/generator.h"
#include "engine/framework/modules/speech_decoders/ming_flow_runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::ming3_tts {

struct MingGenerator::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source, const MingConfig & config,
         core::ExecutionContext & execution, core::AttentionPreference attention)
        : execution_(execution), store_(execution.backend(), execution.backend_type(), "ming3_tts.ar.weights", 4 * 1024 * 1024),
          flow_(source, config.flow, execution) {
        namespace binding = modules::binding;
        const auto & llm = config.backbone;
        latent_dim_ = config.codec.encoder.latent_dim;
        modules::CausalDecoderRuntimeConfig runtime;
        runtime.trace_name = "ming3_tts.ar";
        runtime.prefill_graph_arena_bytes = 8 * 1024 * 1024;
        runtime.decode_graph_arena_bytes = 8 * 1024 * 1024;
        runtime.return_hidden = true;
        runtime.readback_round_type = GGML_TYPE_BF16;
        auto & c = runtime.decoder.stack;
        c.hidden_size = llm.hidden_size;
        hidden_ = c.hidden_size;
        c.intermediate_size = llm.intermediate_size;
        c.layers = llm.layers;
        c.num_attention_heads = llm.attention_heads;
        c.num_key_value_heads = llm.key_value_heads;
        c.head_dim = c.hidden_size / c.num_attention_heads;
        c.rms_norm_eps = llm.rms_norm_eps;
        c.rope_theta = llm.rope_theta;
        c.use_qk_norm = false;
        c.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
        c.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
        const bool flash = core::resolve_flash_attention(execution.backend(), c.head_dim, attention);
        c.runtime.attention.allow_flash_attention = flash;
        c.runtime.attention.prefill_mode = c.runtime.attention.static_mode = flash
            ? modules::DecoderAttentionMode::FlashGroupedViewKV : modules::DecoderAttentionMode::ManualRepeat;
        c.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        if (flash) c.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
        c.activation_cast.enabled = true;
        c.activation_cast.norm_intermediate_type = GGML_TYPE_BF16;
        c.activation_cast.rope_intermediate_type = GGML_TYPE_BF16;
        c.activation_cast.after_input_norm = true;
        c.activation_cast.after_qkv_projection = true;
        c.activation_cast.after_rope = true;
        c.activation_cast.after_attention = true;
        c.activation_cast.after_attention_output = true;
        c.activation_cast.after_residual = true;
        c.activation_cast.after_ffn_norm = true;
        c.activation_cast.after_mlp_projection = true;
        c.activation_cast.after_mlp_silu = true;
        c.activation_cast.after_mlp_mul = true;
        c.activation_cast.after_output = true;
        runtime.decoder.static_cache_type = flash ? GGML_TYPE_F16 : GGML_TYPE_F32;
        runtime.decoder.logits_size = 2;
        runtime.decoder.use_lm_head_bias = true;
        // Vulkan and Metal use F16 weights without mixed F16/BF16 matrix-vector kernels.
        runtime.decoder.lm_head_input_type = (execution.backend_type() == core::BackendType::Vulkan ||
                                            execution.backend_type() == core::BackendType::Metal)
            ? GGML_TYPE_F32 : GGML_TYPE_BF16;
        const auto storage = assets::TensorStorageType::Native;
        modules::CausalDecoderRuntimeWeights weights;
        vocabulary_ = config.vocabulary_size;
        embedding_ = store_.load_tensor(*source, "model.model.embed_tokens.weight", storage, {vocabulary_, hidden_});
        speaker_projection_ = binding::linear_from_source(store_, *source, "spk_head", storage, hidden_, 192, true);
        std::vector<float> frequencies(c.head_dim / 2);
        for (size_t i = 0; i < frequencies.size(); ++i) {
            frequencies[i] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / c.head_dim);
        }
        auto rounded = frequencies;
        core::round_f32_to_bf16_in_place(rounded);
        for (size_t i = 0; i < frequencies.size(); ++i) frequencies[i] /= rounded[i];
        const auto factors = store_.make_f32(core::TensorShape::from_dims({c.head_dim / 2}), frequencies);
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto p = "model.model.layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(store_, *source, p + ".input_layernorm", c.hidden_size);
            layer.post_norm = binding::norm_weight_from_source(store_, *source, p + ".post_attention_layernorm", c.hidden_size);
            const auto qkv = modules::PackedLinearWeightsBuilder({c.hidden_size, {
                {p + ".self_attn.q_proj.weight", p + ".self_attn.q_proj.bias", c.hidden_size},
                {p + ".self_attn.k_proj.weight", p + ".self_attn.k_proj.bias", c.num_key_value_heads * c.head_dim},
                {p + ".self_attn.v_proj.weight", p + ".self_attn.v_proj.bias", c.num_key_value_heads * c.head_dim}}, true})
                .build(store_, *source, storage);
            layer.self_attention.qkv_weight = qkv.weight;
            layer.self_attention.qkv_bias = qkv.bias;
            layer.self_attention.out_weight = store_.load_tensor(*source, p + ".self_attn.o_proj.weight", storage,
                {c.hidden_size, c.hidden_size});
            layer.mlp.gate_up_proj = modules::PackedLinearWeightsBuilder({c.hidden_size, {
                {p + ".mlp.gate_proj.weight", std::nullopt, c.intermediate_size},
                {p + ".mlp.up_proj.weight", std::nullopt, c.intermediate_size}}, false}).build(store_, *source, storage);
            layer.mlp.down_proj = binding::linear_from_source(store_, *source, p + ".mlp.down_proj", storage,
                c.hidden_size, c.intermediate_size, false);
            layer.rope_frequency_factors = factors;
            weights.stack.layers.push_back(std::move(layer));
        }
        weights.final_norm = binding::norm_weight_from_source(store_, *source, "model.model.norm", c.hidden_size);
        weights.lm_head = binding::linear_from_source(store_, *source, "stop_head", storage, 2, c.hidden_size, true);
        store_.upload();
        decoder_ = std::make_unique<modules::CausalDecoderRuntime>(execution_, runtime, std::move(weights));
    }

    ~Impl() { release_prompt(); }

    void release_prompt() {
        core::release_backend_graph_resources(execution_.backend(), prompt_graph_);
        prompt_graph_ = nullptr;
        prompt_plan_.reset();
        if (prompt_allocator_) ggml_gallocr_free(prompt_allocator_);
        prompt_allocator_ = nullptr;
        if (prompt_context_) ggml_free(prompt_context_);
        prompt_context_ = nullptr;
    }

    std::vector<float> prepare_inputs(const MingPrompt & prompt, const std::vector<std::vector<float>> & speakers,
        const std::vector<float> & prompt_latents, bool zero_speaker) {
        const int64_t steps = static_cast<int64_t>(prompt.ids.size());
        if (steps == 0 || speakers.size() != prompt.speaker_positions.size()) {
            throw std::runtime_error("Ming prompt and speaker conditioning dimensions do not match");
        }
        for (const auto id : prompt.ids) {
            if (id < 0 || id >= vocabulary_) throw std::runtime_error("Ming prompt token is outside the vocabulary");
        }
        std::vector<float> flat_speakers;
        for (const auto & speaker : speakers) {
            if (speaker.size() != 192) throw std::runtime_error("Ming speaker embedding must contain 192 values");
            flat_speakers.insert(flat_speakers.end(), speaker.begin(), speaker.end());
        }
        if (!prompt_graph_ || steps != prompt_steps_ || speakers.size() != prompt_speakers_) {
            release_prompt();
            prompt_context_ = ggml_init({2 * 1024 * 1024, nullptr, true});
            core::ModuleBuildContext ctx{prompt_context_, "ming3_tts.prompt", execution_.backend_type()};
            prompt_ids_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
            ggml_set_input(prompt_ids_.tensor);
            auto embeddings = modules::EmbeddingModule({vocabulary_, hidden_}).build(ctx, prompt_ids_, embedding_);
            prompt_output_ = core::wrap_tensor(ggml_cast(ctx.ggml, embeddings.tensor, GGML_TYPE_F32), embeddings.shape, GGML_TYPE_F32);
            ggml_set_output(prompt_output_.tensor);
            prompt_graph_ = ggml_new_graph_custom(prompt_context_, 128, false);
            ggml_build_forward_expand(prompt_graph_, prompt_output_.tensor);
            if (!speakers.empty()) {
                speaker_input_ = core::make_tensor(ctx, GGML_TYPE_F32,
                    core::TensorShape::from_dims({static_cast<int64_t>(speakers.size()), 192}));
                ggml_set_input(speaker_input_.tensor);
                const auto speaker = modules::Bf16RoundModule{}.build(ctx, speaker_input_);
                speaker_output_ = modules::LinearModule({192, hidden_, true}).build(ctx, speaker, speaker_projection_);
                ggml_set_output(speaker_output_.tensor);
                ggml_build_forward_expand(prompt_graph_, speaker_output_.tensor);
            }
            prompt_allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
            if (!ggml_gallocr_reserve(prompt_allocator_, prompt_graph_) || !ggml_gallocr_alloc_graph(prompt_allocator_, prompt_graph_)) {
                throw std::runtime_error("Ming prompt graph allocation failed");
            }
            core::prepare_host_graph_plan(execution_, prompt_graph_, prompt_plan_);
            prompt_steps_ = steps;
            prompt_speakers_ = speakers.size();
        }
        ggml_backend_tensor_set(prompt_ids_.tensor, prompt.ids.data(), 0, prompt.ids.size() * sizeof(int32_t));
        if (!flat_speakers.empty()) core::write_tensor_f32(speaker_input_, flat_speakers);
        if (core::compute_graph(execution_, prompt_graph_, prompt_plan_, "ming3_tts.prompt") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming prompt graph execution failed");
        }
        auto result = core::read_tensor_f32(prompt_output_.tensor);
        core::round_f32_to_bf16_in_place(result);
        if (!speakers.empty()) {
            auto projected = core::read_tensor_f32(speaker_output_.tensor);
            core::round_f32_to_bf16_in_place(projected);
            if (zero_speaker) std::fill(projected.begin(), projected.end(), 0.0f);
            for (size_t i = 0; i < speakers.size(); ++i) {
                if (prompt.speaker_positions[i] >= prompt.ids.size()) throw std::runtime_error("Ming speaker position is outside the prompt");
                std::copy_n(projected.begin() + i * hidden_, hidden_, result.begin() + prompt.speaker_positions[i] * hidden_);
            }
        }
        if (!prompt_latents.empty()) {
            if (prompt_latents.size() % latent_dim_ != 0 || prompt.latent_position < 0 ||
                static_cast<size_t>(prompt.latent_position) + prompt_latents.size() / latent_dim_ != prompt.ids.size()) {
                throw std::runtime_error("Ming prompt latent positions do not match the acoustic reference");
            }
            const auto projected = flow_.aggregate(prompt_latents);
            std::copy(projected.begin(), projected.end(), result.begin() + prompt.latent_position * hidden_);
        }
        return result;
    }

    std::vector<float> generate(const std::vector<float> & embeddings, const MingGenerationOptions & options,
        uint64_t & offset, const std::vector<float> & initial_history) {
        if (embeddings.empty() || embeddings.size() % hidden_ != 0 || options.max_tokens <= 0) {
            throw std::runtime_error("Ming generation requires complete prompt embeddings and positive max_tokens");
        }
        const int64_t prompt_steps = embeddings.size() / hidden_;
        const int64_t budget = std::min(options.max_tokens, int64_t{512} - prompt_steps);
        if (budget <= 0) throw std::runtime_error("Ming prompt exhausts the 512-position context; reduce text_chunk_size or the reference length");
        auto history = initial_history.empty() ? std::vector<float>(latent_dim_, 0.0f) : initial_history;
        if (history.size() != static_cast<size_t>(latent_dim_)) throw std::runtime_error("Ming history has the wrong latent dimension");
        const auto policy = sampling::resolve_torch_cuda_sampling_policy(execution_.backend_type(),
            execution_.config().device, "ming3_tts", "Ming TTS");
        const auto prefill = decoder_->prefill_embeddings_into_decode_cache(embeddings, prompt_steps, 512);
        modules::CausalDecoderStepResult state{prefill.logits, prefill.hidden};
        std::vector<float> latents;
        for (int64_t step = 0; step < budget; ++step) {
            const auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(latent_dim_, options.seed, offset,
                policy, sampling::TorchRandnPrecision::BFloat16);
            offset += sampling::torch_cuda_tensor_iterator_offset_blocks(latent_dim_, policy);
            const auto sde_noise = sampling::generate_torch_cuda_tensor_iterator_randn(10 * latent_dim_, options.seed, offset,
                policy, sampling::TorchRandnPrecision::BFloat16);
            offset += sampling::torch_cuda_tensor_iterator_offset_blocks(10 * latent_dim_, policy);
            auto patch = flow_.sample(state.hidden, history, noise, sde_noise, options.cfg, options.sigma, options.temperature);
            latents.insert(latents.end(), patch.begin(), patch.end());
            core::round_f32_to_bf16_in_place(state.logits);
            std::vector<float> stop_probability{1.0f / (1.0f + std::exp(state.logits.at(0) - state.logits.at(1)))};
            core::round_f32_to_bf16_in_place(stop_probability);
            if (step > 3 && stop_probability[0] > 0.5f) break;
            if (step + 1 < budget) state = decoder_->decode_embedding(flow_.aggregate(patch));
            history = std::move(patch);
        }
        return latents;
    }

    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    modules::MingFlowRuntime flow_;
    std::unique_ptr<modules::CausalDecoderRuntime> decoder_;
    int64_t hidden_ = 0, latent_dim_ = 0, vocabulary_ = 0, prompt_steps_ = 0;
    size_t prompt_speakers_ = 0;
    modules::LinearWeights speaker_projection_;
    core::TensorValue embedding_, prompt_ids_, prompt_output_, speaker_input_, speaker_output_;
    ggml_context * prompt_context_ = nullptr;
    ggml_cgraph * prompt_graph_ = nullptr;
    ggml_gallocr_t prompt_allocator_ = nullptr;
    core::HostGraphPlan prompt_plan_;
};

MingGenerator::MingGenerator(std::shared_ptr<const assets::TensorSource> source, const MingConfig & config,
    core::ExecutionContext & execution, core::AttentionPreference attention)
    : impl_(std::make_unique<Impl>(std::move(source), config, execution, attention)) {}
MingGenerator::~MingGenerator() = default;
std::vector<float> MingGenerator::prepare_inputs(const MingPrompt & prompt,
    const std::vector<std::vector<float>> & speakers, const std::vector<float> & prompt_latents, bool zero_speaker) {
    return impl_->prepare_inputs(prompt, speakers, prompt_latents, zero_speaker);
}
std::vector<float> MingGenerator::generate(const std::vector<float> & embeddings, const MingGenerationOptions & options,
    uint64_t & random_offset, const std::vector<float> & initial_history) {
    return impl_->generate(embeddings, options, random_offset, initial_history);
}

}  // namespace engine::models::ming3_tts
