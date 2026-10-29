#include "engine/models/pocket_tts/flow_lm.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "engine/models/pocket_tts/voice_state_assets.h"
#include "graph_common.h"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::pocket_tts {
namespace {

modules::TransformerEncoderBlockWeights make_transformer_layer_weights(
    core::ModuleBuildContext & ctx,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const FlowLMConfig & config,
    int64_t layer) {
    return graph_common::make_transformer_block_weights(
        ctx,
        weights.flow.transformer_layers.at(static_cast<size_t>(layer)),
        config.hidden_size);
}

modules::TimedConditionedFlowMLPWeights make_flow_net_weights(
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const FlowLMConfig & config) {
    if (static_cast<int64_t>(weights.flow.flow_net.residual_layers.size()) != config.flow_depth) {
        throw std::runtime_error("PocketTTS FlowNet residual layer count does not match config");
    }
    return weights.flow.flow_net;
}

std::vector<float> replace_bos_sentinel_if_needed(
    const std::vector<float> & bos_embedding,
    const std::vector<float> & input_latent,
    int64_t latent_size) {
    if (static_cast<int64_t>(input_latent.size()) != latent_size) {
        throw std::runtime_error("PocketTTS input latent has unexpected size");
    }
    bool has_nan = false;
    for (float value : input_latent) {
        if (std::isnan(value)) {
            has_nan = true;
            break;
        }
    }
    if (!has_nan) {
        return input_latent;
    }
    if (bos_embedding.size() != static_cast<size_t>(latent_size)) {
        throw std::runtime_error("FlowLM BOS embedding size does not match latent_dim");
    }
    return bos_embedding;
}

std::vector<float> build_step_attention_mask(int64_t valid_steps, int64_t cache_steps) {
    if (valid_steps < 0 || valid_steps > cache_steps) {
        throw std::runtime_error("FlowLM step attention mask requires valid_steps within cache_steps");
    }
    std::vector<float> mask(static_cast<size_t>(cache_steps + 1), -INFINITY);
    for (int64_t index = 0; index < valid_steps; ++index) {
        mask[static_cast<size_t>(index)] = 0.0F;
    }
    mask[static_cast<size_t>(cache_steps)] = 0.0F;
    return mask;
}

core::TensorValue make_cache_prefix_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache_tensor,
    int64_t prefix_steps,
    int64_t num_heads,
    int64_t head_dim) {
    const int64_t step_elems = num_heads * head_dim;
    auto flat = core::wrap_tensor(
        ggml_view_1d(
            ctx.ggml,
            cache_tensor.tensor,
            prefix_steps * step_elems,
            0),
        core::TensorShape::from_dims({prefix_steps * step_elems}),
        GGML_TYPE_F32);
    return core::reshape_tensor(
        ctx,
        flat,
        core::TensorShape::from_dims({1, prefix_steps, num_heads, head_dim}));
}

void write_prompt_attention_mask(
    std::vector<float> & mask,
    int64_t valid_prefix_steps,
    int64_t prefix_capacity,
    int64_t valid_prompt_steps,
    int64_t prompt_capacity) {
    const int64_t total_key_capacity = prefix_capacity + prompt_capacity;
    if (valid_prefix_steps < 0 || valid_prompt_steps < 0 || valid_prompt_steps > prompt_capacity
        || valid_prefix_steps > prefix_capacity || prefix_capacity < 0) {
        throw std::runtime_error("FlowLM prompt attention mask received invalid capacities");
    }
    mask.assign(static_cast<size_t>(prompt_capacity * total_key_capacity), -INFINITY);
    for (int64_t q = 0; q < prompt_capacity; ++q) {
        const size_t row_offset = static_cast<size_t>(q * total_key_capacity);
        const int64_t max_key = prefix_capacity + q;
        if (q >= valid_prompt_steps) {
            mask[row_offset + static_cast<size_t>(max_key)] = 0.0F;
            continue;
        }
        for (int64_t k = 0; k < valid_prefix_steps; ++k) {
            mask[row_offset + static_cast<size_t>(k)] = 0.0F;
        }
        for (int64_t k = prefix_capacity; k <= max_key; ++k) {
            mask[row_offset + static_cast<size_t>(k)] = 0.0F;
        }
    }
}

}  // namespace

class FlowLMWeightsRuntime {
public:
    struct Weights {
        modules::LinearWeights input_linear;
        std::vector<modules::TransformerEncoderBlockWeights> transformer_layers;
        modules::NormWeights out_norm;
        modules::LinearWeights out_eos;
        modules::TimedConditionedFlowMLPWeights flow_net;
    };

    FlowLMWeightsRuntime(
        const FlowLMConfig & config,
        int threads,
        const models::pocket_tts::PocketTTSBackendWeights & weights,
        size_t view_context_bytes) {
        (void) threads;
        weights_.input_linear = weights.flow.input_linear;
        weights_.transformer_layers.reserve(static_cast<size_t>(config.layers));
        ggml_ctx_ = ggml_init({view_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize FlowLM weights view context");
        }
        backend_type_ = weights.backend_type;
        core::ModuleBuildContext ctx = core::ModuleBuildContext{ggml_ctx_, "flow_lm_weights_runtime", backend_type_};
        for (int64_t layer = 0; layer < config.layers; ++layer) {
            weights_.transformer_layers.push_back(make_transformer_layer_weights(ctx, weights, config, layer));
        }
        weights_.out_norm = weights.flow.out_norm;
        weights_.out_eos = weights.flow.out_eos;
        weights_.flow_net = make_flow_net_weights(weights, config);
        const auto & bos = weights.host.bos_emb;
        if (static_cast<int64_t>(bos.size()) != config.latent_size) {
            throw std::runtime_error("flow_lm.bos_emb must have shape [latent_dim]");
        }
        bos_embedding_ = bos;
    }

    ~FlowLMWeightsRuntime() {
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
        }
    }

    FlowLMWeightsRuntime(const FlowLMWeightsRuntime &) = delete;
    FlowLMWeightsRuntime & operator=(const FlowLMWeightsRuntime &) = delete;

    const Weights & weights() const noexcept {
        return weights_;
    }

    const std::vector<float> & bos_embedding() const noexcept {
        return bos_embedding_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

private:
    ggml_context * ggml_ctx_ = nullptr;
    Weights weights_;
    std::vector<float> bos_embedding_;
    core::BackendType backend_type_ = core::BackendType::Cpu;
};

class FlowLMStepRuntime {
public:
    FlowLMStepRuntime(
        const FlowLMConfig & config,
        ggml_backend_t backend,
        int threads,
        std::shared_ptr<const FlowLMWeightsRuntime> weights_runtime,
        int64_t cache_steps,
        int64_t prompt_steps,
        int64_t prompt_prefix_steps,
        size_t graph_context_bytes)
        : config_(config),
          backend_(backend),
          threads_(threads),
          weights_runtime_(std::move(weights_runtime)),
          cache_steps_(std::max<int64_t>(1, cache_steps)),
          prompt_steps_(std::max<int64_t>(0, prompt_steps)),
          prompt_prefix_steps_(std::max<int64_t>(0, prompt_prefix_steps)),
          head_dim_(config.hidden_size / config.num_heads) {
        if (weights_runtime_ == nullptr) {
            throw std::runtime_error("FlowLM step runtime requires weights runtime");
        }
        const auto & weights = weights_runtime_->weights();
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize FlowLM step ggml context");
        }

        core::ModuleBuildContext ctx{ggml_ctx_, "flow_lm_step_runtime", weights_runtime_->backend_type()};

        if (prompt_steps_ > 0) {
            prompt_positions_buffer_.resize(static_cast<size_t>(prompt_steps_));
            prompt_embeddings_buffer_.assign(
                static_cast<size_t>(prompt_steps_ * config_.hidden_size),
                0.0F);
            prompt_embeddings_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, prompt_steps_, config_.hidden_size}));
            prompt_positions_ = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({prompt_steps_}));
            write_prompt_attention_mask(
                prompt_attention_mask_buffer_,
                0,
                prompt_prefix_steps_,
                0,
                prompt_steps_);
            prompt_attention_mask_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_prefix_steps_ + prompt_steps_}));
        }

        latent_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, config_.latent_size}));
        positions_ = core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({1}));
        noise_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config_.latent_size}));
        position_buffer_ = {0};
        start_time_buffer_ = {0.0F};
        end_time_buffer_ = {1.0F};
        start_time_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1}));
        end_time_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1}));
        attention_mask_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1}));
        attention_mask_buffer_ = build_step_attention_mask(0, cache_steps_);

        auto x = modules::LinearModule({config_.latent_size, config_.hidden_size, false}).build(
            ctx,
            latent_,
            weights.input_linear);
        const modules::StreamingTransformerEncoderBlockModule transformer_block({
            config_.hidden_size,
            config_.num_heads,
            config_.intermediate_size,
            config_.norm_eps,
            false,
        });

        keys_.reserve(static_cast<size_t>(config_.layers));
        values_.reserve(static_cast<size_t>(config_.layers));
        std::vector<core::TensorValue> step_cache_keys;
        std::vector<core::TensorValue> step_cache_values;
        step_cache_keys.reserve(static_cast<size_t>(config_.layers));
        step_cache_values.reserve(static_cast<size_t>(config_.layers));

        for (int64_t layer = 0; layer < config_.layers; ++layer) {
            step_cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_})));
            step_cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_})));

            auto outputs = transformer_block.build(
                ctx,
                x,
                positions_,
                weights.transformer_layers[static_cast<size_t>(layer)],
                step_cache_keys.back(),
                step_cache_values.back(),
                attention_mask_);
            x = outputs.output;
            keys_.push_back(outputs.key);
            values_.push_back(outputs.value);
        }
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_,
            config_.num_heads * head_dim_,
            std::move(step_cache_keys),
            std::move(step_cache_values));
        if (prompt_steps_ > 0) {
            prompt_keys_.reserve(static_cast<size_t>(config_.layers));
            prompt_values_.reserve(static_cast<size_t>(config_.layers));
            auto x_prompt = prompt_embeddings_;
            for (int64_t layer = 0; layer < config_.layers; ++layer) {
                std::optional<core::TensorValue> prefix_k = std::nullopt;
                std::optional<core::TensorValue> prefix_v = std::nullopt;
                if (prompt_prefix_steps_ > 0) {
                    prefix_k = make_cache_prefix_view(
                        ctx,
                        step_cache_.key_tensor(static_cast<size_t>(layer)),
                        prompt_prefix_steps_,
                        config_.num_heads,
                        head_dim_);
                    prefix_v = make_cache_prefix_view(
                        ctx,
                        step_cache_.value_tensor(static_cast<size_t>(layer)),
                        prompt_prefix_steps_,
                        config_.num_heads,
                        head_dim_);
                }

                auto outputs = transformer_block.build(
                    ctx,
                    x_prompt,
                    prompt_positions_,
                    weights.transformer_layers[static_cast<size_t>(layer)],
                    prefix_k,
                    prefix_v,
                    prompt_attention_mask_);
                x_prompt = outputs.output;
                prompt_keys_.push_back(outputs.key);
                prompt_values_.push_back(outputs.value);
            }
            prompt_output_ = x_prompt;
        }
        build_transfer_views();

        auto condition = modules::LayerNormModule({config_.hidden_size, config_.norm_eps, true, true}).build(
            ctx,
            x,
            weights.out_norm);
        auto condition_last = modules::SliceModule({1, condition.shape.dims[1] - 1, 1}).build(ctx, condition);
        core::validate_shape(
            condition_last,
            core::TensorShape::from_dims({condition.shape.dims[0], 1, condition.shape.dims[2]}),
            "condition_last");
        condition_ = core::reshape_tensor(
            ctx,
            condition_last,
            core::TensorShape::from_dims({condition.shape.dims[0], condition.shape.dims[2]}));
        auto eos = modules::LinearModule({config_.hidden_size, 1, true}).build(
            ctx,
            condition_,
            weights.out_eos);
        flow_ = modules::TimedConditionedFlowMLPModule({
            config_.latent_size,
            config_.flow_hidden_size,
            config_.hidden_size,
            config_.latent_size,
            config_.flow_depth,
            config_.flow_eps,
            true,
        }).build(ctx, condition_, start_time_, end_time_, noise_, weights.flow_net);
        auto next_latent = modules::ResidualAddModule().build(ctx, noise_, flow_);
        combined_ = modules::ConcatModule({1}).build(ctx, next_latent, eos);
        combined_scratch_.resize(static_cast<size_t>(config_.latent_size + 1));

        params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
        if (params_buffer_ == nullptr) {
            // Under VRAM pressure the CUDA buffer alloc fails and returns null;
            // the write_tensor calls below would then trip GGML_ASSERT and
            // SIGABRT the whole server. Fail the request instead (see #55).
            release_runtime();
            throw std::runtime_error("FlowLM step parameter buffer allocation failed (out of memory)");
        }
        if (prompt_steps_ > 0) {
            core::write_tensor_f32(prompt_embeddings_, prompt_embeddings_buffer_);
            core::write_tensor_i32(prompt_positions_, std::vector<int32_t>(static_cast<size_t>(prompt_steps_), 0));
            core::write_tensor_f32(prompt_attention_mask_, prompt_attention_mask_buffer_);
        }
        core::write_tensor_f32(latent_, std::vector<float>(static_cast<size_t>(config_.latent_size), 0.0F));
        core::write_tensor_i32(positions_, position_buffer_);
        core::write_tensor_f32(noise_, std::vector<float>(static_cast<size_t>(config_.latent_size), 0.0F));
        core::write_tensor_f32(start_time_, start_time_buffer_);
        core::write_tensor_f32(end_time_, end_time_buffer_);
        core::write_tensor_f32(attention_mask_, attention_mask_buffer_);
        core::set_backend_threads(backend_, threads_);
        if (prompt_steps_ > 0) {
            prompt_graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
            ggml_build_forward_expand(prompt_graph_, prompt_output_.tensor);
            for (size_t step = 0; step < prompt_step_key_sources_.size(); ++step) {
                for (size_t layer = 0; layer < prompt_step_key_sources_[step].size(); ++layer) {
                    ggml_build_forward_expand(
                        prompt_graph_,
                        ggml_cpy(
                            ggml_ctx_,
                            prompt_step_key_sources_[step][layer],
                            prompt_step_key_destinations_[step][layer]));
                    ggml_build_forward_expand(
                        prompt_graph_,
                        ggml_cpy(
                            ggml_ctx_,
                            prompt_step_value_sources_[step][layer],
                            prompt_step_value_destinations_[step][layer]));
                }
            }
            prompt_galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if (prompt_galloc_ == nullptr ||
                !ggml_gallocr_reserve(prompt_galloc_, prompt_graph_) ||
                !ggml_gallocr_alloc_graph(prompt_galloc_, prompt_graph_)) {
                release_runtime();
                throw std::runtime_error("FlowLM prompt graph allocation failed (out of memory)");
            }
            if (engine::core::uses_host_graph_plan(backend_)) {
                const auto prompt_plan_started = std::chrono::steady_clock::now();
                prompt_plan_ = engine::core::create_backend_graph_plan_if_host(backend_, prompt_graph_);
                prompt_plan_create_ms_ = engine::debug::elapsed_ms(prompt_plan_started);
                if (prompt_plan_ == nullptr) {
                    release_runtime();
                    throw std::runtime_error("Failed to create FlowLM prompt graph plan");
                }
            }
        }
        step_graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        ggml_build_forward_expand(step_graph_, combined_.tensor);
    }

    ~FlowLMStepRuntime() {
        release_runtime();
    }

    // Free every owned backend resource and null it, so the constructor can
    // call this on an allocation-failure throw path (a throw from the
    // constructor skips the destructor, which would otherwise leak the ggml
    // context and any buffers allocated before the failure). Idempotent.
    void release_runtime() {
        if (prompt_plan_ != nullptr) {
            engine::core::free_backend_graph_plan(backend_, prompt_plan_);
            prompt_plan_ = nullptr;
        }
        if (step_plan_ != nullptr) {
            engine::core::free_backend_graph_plan(backend_, step_plan_);
            step_plan_ = nullptr;
        }
        if (prompt_galloc_ != nullptr) {
            ggml_gallocr_free(prompt_galloc_);
            prompt_galloc_ = nullptr;
        }
        if (step_galloc_ != nullptr) {
            ggml_gallocr_free(step_galloc_);
            step_galloc_ = nullptr;
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
            params_buffer_ = nullptr;
        }
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
            ggml_ctx_ = nullptr;
        }
    }

    void apply_prompt_from_state(
        const std::vector<float> & input_embeddings,
        int64_t steps,
        const FlowLMState & state) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("FlowLM prompt runtime was not initialized with prompt support");
        }
        if (steps <= 0 || steps > prompt_steps_) {
            throw std::runtime_error("FlowLM prompt runtime step count exceeds prepared capacity");
        }
        if (state.current_end < 0 || state.current_end > prompt_prefix_steps_) {
            throw std::runtime_error("FlowLM prompt runtime prefix state exceeds prepared capacity");
        }
        if (state.current_end + steps > cache_steps_) {
            throw std::runtime_error("FlowLM prompt runtime total cache steps exceed prepared capacity");
        }
        step_cache_initialized_ = false;
        reset_timing_counters();

        if (input_embeddings.size() != static_cast<size_t>(steps * config_.hidden_size)) {
            throw std::runtime_error("FlowLM prompt runtime input embedding count mismatch");
        }
        prompt_host_setup_ms_ = engine::debug::measure_ms([&]() {
            std::fill(prompt_embeddings_buffer_.begin(), prompt_embeddings_buffer_.end(), 0.0F);
            std::copy(input_embeddings.begin(), input_embeddings.end(), prompt_embeddings_buffer_.begin());
            core::write_tensor_f32(prompt_embeddings_, prompt_embeddings_buffer_);
            for (int64_t i = 0; i < steps; ++i) {
                prompt_positions_buffer_[static_cast<size_t>(i)] = static_cast<int32_t>(state.current_end + i);
            }
            for (int64_t i = steps; i < prompt_steps_; ++i) {
                prompt_positions_buffer_[static_cast<size_t>(i)] = 0;
            }
            core::write_tensor_i32(prompt_positions_, prompt_positions_buffer_);
            write_prompt_attention_mask(
                prompt_attention_mask_buffer_,
                state.current_end,
                prompt_prefix_steps_,
                steps,
                prompt_steps_);
            core::write_tensor_f32(prompt_attention_mask_, prompt_attention_mask_buffer_);
        });

        prompt_import_ms_ = engine::debug::measure_ms([&]() {
            step_cache_.import_state(state);
        });
        prompt_graph_ms_ = engine::debug::measure_ms([&]() {
            compute_graph(prompt_graph_, prompt_plan_);
        });
        if (steps > 0) {
            prompt_finalize_ms_ = engine::debug::measure_ms([&]() {
                step_cache_.advance_after_direct_append(steps);
            });
        } else {
            prompt_finalize_ms_ = 0.0;
        }
        prompt_mask_refresh_ms_ = engine::debug::measure_ms([&]() {
            attention_mask_buffer_ = build_step_attention_mask(step_cache_.valid_steps(), cache_steps_);
            core::write_tensor_f32(attention_mask_, attention_mask_buffer_);
        });
        step_cache_initialized_ = true;
    }

    FlowLMStepResult run_in_place(
        const std::vector<float> & input_latent,
        const std::vector<float> & noise) {
        if (!step_graph_allocated_) {
            ensure_step_graph_allocated();
        }
        if (!step_cache_initialized_) {
            throw std::runtime_error("FlowLM cached step runtime must be primed with apply_prompt before run_in_place");
        }
        step_input_write_ms_ += engine::debug::measure_ms([&]() {
            const auto latent_values =
                replace_bos_sentinel_if_needed(weights_runtime_->bos_embedding(), input_latent, config_.latent_size);
            core::write_tensor_f32(latent_, latent_values);
            core::write_tensor_f32(noise_, noise);
            position_buffer_[0] = static_cast<int32_t>(step_cache_.current_end());
            core::write_tensor_i32(positions_, position_buffer_);
        });

        step_graph_ms_ += engine::debug::measure_ms([&]() {
            compute_graph(step_graph_, step_plan_);
        });

        FlowLMStepResult result;
        step_output_read_ms_ += engine::debug::measure_ms([&]() {
            core::read_tensor_f32_into(combined_.tensor, combined_scratch_);
        });
        result.next_latent.assign(combined_scratch_.begin(), combined_scratch_.begin() + config_.latent_size);
        result.eos_logit = combined_scratch_[static_cast<size_t>(config_.latent_size)];
        const size_t dst_slot = static_cast<size_t>(step_cache_.valid_steps());
        step_kv_update_ms_ += engine::debug::measure_ms([&]() {
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
                ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
            }
            attention_mask_buffer_[dst_slot] = 0.0F;
            core::write_tensor_f32_slice(attention_mask_, dst_slot, &attention_mask_buffer_[dst_slot], 1);
            step_cache_.advance_after_direct_append(1);
        });
        return result;
    }

    FlowLMState export_state() const {
        return step_cache_.export_state();
    }

    FlowLMRuntimeTiming timing() const {
        return {
            prompt_host_setup_ms_,
            prompt_import_ms_,
            prompt_graph_ms_,
            prompt_finalize_ms_,
            prompt_mask_refresh_ms_,
            step_input_write_ms_,
            step_graph_ms_,
            step_output_read_ms_,
            step_kv_update_ms_,
        };
    }

    FlowLMPlanTiming plan_timing() const {
        return {
            prompt_plan_create_ms_,
            step_plan_create_ms_,
        };
    }

private:
    void reset_timing_counters() {
        prompt_host_setup_ms_ = 0.0;
        prompt_import_ms_ = 0.0;
        prompt_graph_ms_ = 0.0;
        prompt_finalize_ms_ = 0.0;
        prompt_mask_refresh_ms_ = 0.0;
        step_input_write_ms_ = 0.0;
        step_graph_ms_ = 0.0;
        step_output_read_ms_ = 0.0;
        step_kv_update_ms_ = 0.0;
    }

    void build_transfer_views() {
        const int64_t step_elems = config_.num_heads * head_dim_;
        key_sources_.reserve(keys_.size());
        value_sources_.reserve(values_.size());
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            key_sources_.push_back(ggml_view_1d(ggml_ctx_, keys_[layer].tensor, step_elems, 0));
            value_sources_.push_back(ggml_view_1d(ggml_ctx_, values_[layer].tensor, step_elems, 0));
        }

        key_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        value_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            auto & key_slot = key_destinations_[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations_[static_cast<size_t>(slot)];
            key_slot.reserve(keys_.size());
            value_slot.reserve(values_.size());
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * sizeof(float);
            for (size_t layer = 0; layer < keys_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(ggml_ctx_, step_cache_.key_tensor(layer).tensor, step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(ggml_ctx_, step_cache_.value_tensor(layer).tensor, step_elems, byte_offset));
            }
        }

        if (prompt_steps_ > 0) {
            prompt_step_key_sources_.assign(static_cast<size_t>(prompt_steps_), {});
            prompt_step_value_sources_.assign(static_cast<size_t>(prompt_steps_), {});
            prompt_step_key_destinations_.assign(static_cast<size_t>(prompt_steps_), {});
            prompt_step_value_destinations_.assign(static_cast<size_t>(prompt_steps_), {});
            for (int64_t step = 0; step < prompt_steps_; ++step) {
                auto & key_src = prompt_step_key_sources_[static_cast<size_t>(step)];
                auto & value_src = prompt_step_value_sources_[static_cast<size_t>(step)];
                auto & key_dst = prompt_step_key_destinations_[static_cast<size_t>(step)];
                auto & value_dst = prompt_step_value_destinations_[static_cast<size_t>(step)];
                key_src.reserve(prompt_keys_.size());
                value_src.reserve(prompt_values_.size());
                key_dst.reserve(prompt_keys_.size());
                value_dst.reserve(prompt_values_.size());
                const size_t byte_offset = static_cast<size_t>(step * step_elems) * sizeof(float);
                const size_t dst_offset =
                    static_cast<size_t>((prompt_prefix_steps_ + step) * step_elems) * sizeof(float);
                for (size_t layer = 0; layer < prompt_keys_.size(); ++layer) {
                    key_src.push_back(ggml_view_1d(ggml_ctx_, prompt_keys_[layer].tensor, step_elems, byte_offset));
                    value_src.push_back(ggml_view_1d(ggml_ctx_, prompt_values_[layer].tensor, step_elems, byte_offset));
                    key_dst.push_back(ggml_view_1d(ggml_ctx_, step_cache_.key_tensor(layer).tensor, step_elems, dst_offset));
                    value_dst.push_back(ggml_view_1d(ggml_ctx_, step_cache_.value_tensor(layer).tensor, step_elems, dst_offset));
                }
            }
        }
    }

    void ensure_step_graph_allocated() {
        if (step_graph_allocated_) {
            return;
        }
        step_galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (step_galloc_ == nullptr ||
            !ggml_gallocr_reserve(step_galloc_, step_graph_) ||
            !ggml_gallocr_alloc_graph(step_galloc_, step_graph_)) {
            // Lazy per-step allocation: free the partial gallocr so a later
            // retry (step_graph_allocated_ stays false) does not leak it, then
            // fail the request instead of asserting (see #55).
            if (step_galloc_ != nullptr) {
                ggml_gallocr_free(step_galloc_);
                step_galloc_ = nullptr;
            }
            throw std::runtime_error("FlowLM step graph allocation failed (out of memory)");
        }
        if (engine::core::uses_host_graph_plan(backend_)) {
            const auto step_plan_started = std::chrono::steady_clock::now();
            step_plan_ = engine::core::create_backend_graph_plan_if_host(backend_, step_graph_);
            step_plan_create_ms_ = engine::debug::elapsed_ms(step_plan_started);
            if (step_plan_ == nullptr) {
                throw std::runtime_error("Failed to create FlowLM step graph plan");
            }
        }
        step_graph_allocated_ = true;
    }

    void compute_graph(ggml_cgraph * graph, ggml_backend_graph_plan_t plan) {
        engine::core::compute_backend_graph(backend_, graph, plan);
    }

    FlowLMConfig config_;
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const FlowLMWeightsRuntime> weights_runtime_;
    int64_t cache_steps_ = 1;
    int64_t prompt_steps_ = 0;
    int64_t prompt_prefix_steps_ = 0;
    int64_t head_dim_ = 0;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * prompt_graph_ = nullptr;
    ggml_cgraph * step_graph_ = nullptr;
    ggml_gallocr_t prompt_galloc_ = nullptr;
    ggml_gallocr_t step_galloc_ = nullptr;
    ggml_backend_graph_plan_t prompt_plan_ = nullptr;
    ggml_backend_graph_plan_t step_plan_ = nullptr;
    bool step_graph_allocated_ = false;
    core::TensorValue prompt_embeddings_;
    std::vector<float> prompt_embeddings_buffer_;
    core::TensorValue prompt_positions_;
    core::TensorValue prompt_attention_mask_;
    std::vector<int32_t> prompt_positions_buffer_;
    std::vector<float> prompt_attention_mask_buffer_;
    std::vector<core::TensorValue> prompt_keys_;
    std::vector<core::TensorValue> prompt_values_;
    core::TensorValue prompt_output_;
    core::TensorValue latent_;
    core::TensorValue positions_;
    std::vector<int32_t> position_buffer_;
    core::TensorValue noise_;
    core::TensorValue start_time_;
    core::TensorValue end_time_;
    std::vector<float> start_time_buffer_;
    std::vector<float> end_time_buffer_;
    core::TensorValue attention_mask_;
    std::vector<float> attention_mask_buffer_;
    runtime::TransformerKVCache step_cache_;
    std::vector<core::TensorValue> keys_;
    std::vector<core::TensorValue> values_;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    std::vector<std::vector<ggml_tensor *>> prompt_step_key_sources_;
    std::vector<std::vector<ggml_tensor *>> prompt_step_value_sources_;
    std::vector<std::vector<ggml_tensor *>> prompt_step_key_destinations_;
    std::vector<std::vector<ggml_tensor *>> prompt_step_value_destinations_;
    core::TensorValue condition_;
    core::TensorValue flow_;
    core::TensorValue combined_;
    std::vector<float> combined_scratch_;
    bool step_cache_initialized_ = false;
    double prompt_host_setup_ms_ = 0.0;
    double prompt_import_ms_ = 0.0;
    double prompt_graph_ms_ = 0.0;
    double prompt_finalize_ms_ = 0.0;
    double prompt_mask_refresh_ms_ = 0.0;
    double prompt_plan_create_ms_ = 0.0;
    double step_input_write_ms_ = 0.0;
    double step_graph_ms_ = 0.0;
    double step_output_read_ms_ = 0.0;
    double step_kv_update_ms_ = 0.0;
    double step_plan_create_ms_ = 0.0;
};

FlowLM::FlowLM(FlowLMConfig config) : config_(std::move(config)) {}

const FlowLMConfig & FlowLM::config() const noexcept {
    return config_;
}

std::shared_ptr<const FlowLMWeightsRuntime> FlowLM::create_weights_runtime(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    size_t weights_view_context_bytes) const {
    (void) backend;
    return std::make_shared<FlowLMWeightsRuntime>(config_, threads, weights, weights_view_context_bytes);
}

FlowLMState FlowLM::make_empty_state() const {
    FlowLMState state;
    state.current_end = 0;
    state.layers.resize(static_cast<size_t>(config_.layers));
    return state;
}

FlowLMState FlowLM::make_state(const models::pocket_tts::VoiceStateAssets & voice_assets) const {
    if (voice_assets.transformer_layers.empty()) {
        throw std::runtime_error("PocketTTS voice assets do not contain FlowLM cache layers");
    }
    if (static_cast<int64_t>(voice_assets.transformer_layers.size()) != config_.layers) {
        throw std::runtime_error("PocketTTS voice assets layer count does not match FlowLM configuration");
    }

    FlowLMState state;
    state.layers.reserve(voice_assets.transformer_layers.size());
    state.current_end = voice_assets.transformer_layers.front().offset;
    const int64_t head_elems = config_.num_heads * (config_.hidden_size / config_.num_heads);

    for (const auto & layer : voice_assets.transformer_layers) {
        if (layer.offset != state.current_end) {
            throw std::runtime_error("PocketTTS voice cache offsets are inconsistent across layers");
        }
        if (layer.offset < 0 || layer.offset > layer.cached_steps) {
            throw std::runtime_error("PocketTTS voice cache offset must be within cached_steps");
        }
        const size_t expected_elems = static_cast<size_t>(layer.cached_steps * head_elems);
        if (layer.key.size() != expected_elems || layer.value.size() != expected_elems) {
            throw std::runtime_error("PocketTTS voice cache tensor sizes do not match cached_steps * num_heads * head_dim");
        }
        const size_t valid_elems = static_cast<size_t>(layer.offset * head_elems);
        FlowLMCacheState cache;
        cache.valid_steps = layer.offset;
        cache.key.assign(layer.key.begin(), layer.key.begin() + static_cast<std::ptrdiff_t>(valid_elems));
        cache.value.assign(layer.value.begin(), layer.value.begin() + static_cast<std::ptrdiff_t>(valid_elems));
        state.layers.push_back(std::move(cache));
    }
    return state;
}

std::shared_ptr<FlowLMStepRuntime> FlowLM::create_step_runtime(
    ggml_backend_t backend,
    int threads,
    std::shared_ptr<const FlowLMWeightsRuntime> weights,
    int64_t cache_steps,
    int64_t prompt_steps,
    int64_t prompt_prefix_steps,
    size_t step_graph_context_bytes) const {
    return std::make_shared<FlowLMStepRuntime>(
        config_,
        backend,
        threads,
        std::move(weights),
        cache_steps,
        prompt_steps,
        prompt_prefix_steps,
        step_graph_context_bytes);
}

void FlowLM::apply_prompt(
    FlowLMStepRuntime & runtime,
    const std::vector<float> & input_embeddings,
    int64_t steps,
    const FlowLMState & state) const {
    runtime.apply_prompt_from_state(input_embeddings, steps, state);
}

FlowLMState FlowLM::prompt_state(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & input_embeddings,
    int64_t steps,
    const FlowLMState & state,
    size_t weights_view_context_bytes,
    size_t step_graph_context_bytes) const {
    FlowLMState prompted_state;
    const double prompt_ms = engine::debug::measure_ms([&]() {
        (void) manifest;
        auto weights_runtime = create_weights_runtime(backend, threads, weights, weights_view_context_bytes);
        auto runtime = create_step_runtime(
            backend,
            threads,
            std::move(weights_runtime),
            state.current_end + steps,
            steps,
            state.current_end,
            step_graph_context_bytes);
        runtime->apply_prompt_from_state(input_embeddings, steps, state);
        prompted_state = runtime->export_state();
    });
    engine::debug::timing_log_scalar("pocket_tts.flow_lm.prompt_ms", prompt_ms);
    return prompted_state;
}

FlowLMStepResult FlowLM::run_step_in_place(
    FlowLMStepRuntime & runtime,
    const std::vector<float> & input_latent,
    const std::vector<float> & noise) const {
    return runtime.run_in_place(input_latent, noise);
}

FlowLMRuntimeTiming FlowLM::runtime_timing(const FlowLMStepRuntime & runtime) const {
    return runtime.timing();
}

FlowLMPlanTiming FlowLM::runtime_plan_timing(const FlowLMStepRuntime & runtime) const {
    return runtime.plan_timing();
}

}  // namespace engine::models::pocket_tts
