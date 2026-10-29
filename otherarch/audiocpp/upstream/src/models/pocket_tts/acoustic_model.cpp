#include "engine/models/pocket_tts/acoustic_model.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "engine/models/pocket_tts/assets.h"
#include "graph_common.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::pocket_tts {
namespace {

std::vector<float> sample_trunc_normal(std::mt19937 & rng, int64_t count, float stddev, float clamp) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> values(static_cast<size_t>(count));
    for (float & value : values) {
        do {
            value = dist(rng);
        } while (value < -clamp || value > clamp);
    }
    return values;
}

std::vector<float> sample_normal(std::mt19937 & rng, int64_t count, float stddev) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> values(static_cast<size_t>(count));
    for (float & value : values) {
        value = dist(rng);
    }
    return values;
}

void validate_generation_inputs(
    const FlowLMConfig & flow_config,
    const std::vector<float> & text_embeddings,
    const AcousticGenerationConfig & config) {
    if (config.max_steps <= 0) {
        throw std::runtime_error("PocketTTS acoustic max_steps must be positive");
    }
    if (config.temperature <= 0.0F) {
        throw std::runtime_error("PocketTTS acoustic temperature must be positive");
    }
    if (!config.noise_schedule.empty() && config.noise_schedule.size() % static_cast<size_t>(flow_config.latent_size) != 0) {
        throw std::runtime_error("PocketTTS acoustic noise_schedule must be a multiple of latent_size");
    }
    if (!config.noise_schedule.empty()) {
        const size_t scheduled_steps =
            config.noise_schedule.size() / static_cast<size_t>(flow_config.latent_size);
        if (scheduled_steps < static_cast<size_t>(config.max_steps)) {
            throw std::runtime_error("PocketTTS acoustic noise_schedule must provide at least max_steps latent noise vectors");
        }
    }
    if (text_embeddings.size() % static_cast<size_t>(flow_config.hidden_size) != 0) {
        throw std::runtime_error("PocketTTS acoustic text embeddings must be a multiple of hidden_size");
    }
}

std::vector<float> sample_noise_for_step(const FlowLMConfig & flow_config, AcousticStreamState & state) {
    if (!state.config.noise_schedule.empty()) {
        const size_t start = static_cast<size_t>(state.step) * static_cast<size_t>(flow_config.latent_size);
        return std::vector<float>(
            state.config.noise_schedule.begin() + static_cast<ptrdiff_t>(start),
            state.config.noise_schedule.begin() + static_cast<ptrdiff_t>(start + static_cast<size_t>(flow_config.latent_size)));
    }
    if (state.config.noise_clamp > 0.0F) {
        return sample_trunc_normal(
            state.rng,
            flow_config.latent_size,
            std::sqrt(state.config.temperature),
            state.config.noise_clamp);
    }
    return sample_normal(
        state.rng,
        flow_config.latent_size,
        std::sqrt(state.config.temperature));
}

}  // namespace

AcousticModel::AcousticModel(FlowLMConfig config) : flow_lm_(std::move(config)) {}

const FlowLMConfig & AcousticModel::config() const noexcept {
    return flow_lm_.config();
}

AcousticPreparedRuntime AcousticModel::prepare_runtime(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & text_embeddings,
    const FlowLMState & initial_state,
    const AcousticGenerationConfig & config,
    int64_t prompt_capacity,
    int64_t initial_cache_capacity,
    int max_steps_capacity,
    size_t flow_weights_view_context_bytes,
    size_t flow_step_graph_context_bytes) const {
    if (config.max_steps <= 0) {
        throw std::runtime_error("PocketTTS acoustic max_steps must be positive");
    }
    if (text_embeddings.size() % static_cast<size_t>(flow_lm_.config().hidden_size) != 0) {
        throw std::runtime_error("PocketTTS acoustic text embeddings must be a multiple of hidden_size");
    }
    const int64_t prompt_steps = static_cast<int64_t>(text_embeddings.size() / static_cast<size_t>(flow_lm_.config().hidden_size));
    AcousticPreparedRuntime runtime;
    runtime.prompt_steps = prompt_steps;
    const int64_t initial_cache_steps = initial_state.current_end;
    if (prompt_capacity < prompt_steps) {
        throw std::runtime_error("PocketTTS acoustic prompt_capacity must cover prompt_steps");
    }
    if (initial_cache_capacity < initial_cache_steps) {
        throw std::runtime_error("PocketTTS acoustic initial_cache_capacity must cover initial_cache_steps");
    }
    if (max_steps_capacity < config.max_steps) {
        throw std::runtime_error("PocketTTS acoustic max_steps_capacity must cover max_steps");
    }
    const int64_t total_cache_steps =
        initial_cache_capacity + prompt_capacity + static_cast<int64_t>(max_steps_capacity);
    const bool cache_hit =
        runtime_cache_.runtime != nullptr
        && runtime_cache_.backend == backend
        && runtime_cache_.threads == threads
        && runtime_cache_.manifest == &manifest
        && runtime_cache_.prompt_capacity >= prompt_capacity
        // The step runtime bakes the voice-prefix slot layout at creation: prompt KV
        // destinations start at initial_cache_capacity and the prefix attention view
        // spans exactly that many slots. Reusing a runtime built for a different prefix
        // length misaligns the KV slots against the cache validity accounting, so the
        // prefix capacity must match exactly (not merely fit).
        && runtime_cache_.initial_cache_capacity == initial_cache_capacity
        && runtime_cache_.max_steps_capacity >= max_steps_capacity
        && runtime_cache_.flow_weights_view_context_bytes == flow_weights_view_context_bytes
        && runtime_cache_.flow_step_graph_context_bytes == flow_step_graph_context_bytes;
    if (!cache_hit) {
        const bool weights_hit =
            runtime_cache_.weights_runtime != nullptr
            && runtime_cache_.backend == backend
            && runtime_cache_.threads == threads
            && runtime_cache_.manifest == &manifest
            && runtime_cache_.flow_weights_view_context_bytes == flow_weights_view_context_bytes;
        runtime_cache_.backend = backend;
        runtime_cache_.threads = threads;
        runtime_cache_.manifest = &manifest;
        runtime_cache_.prompt_capacity = prompt_capacity;
        runtime_cache_.initial_cache_capacity = initial_cache_capacity;
        runtime_cache_.max_steps_capacity = max_steps_capacity;
        runtime_cache_.flow_weights_view_context_bytes = flow_weights_view_context_bytes;
        runtime_cache_.flow_step_graph_context_bytes = flow_step_graph_context_bytes;
        if (!weights_hit) {
            runtime_cache_.weights_runtime =
                flow_lm_.create_weights_runtime(backend, threads, weights, flow_weights_view_context_bytes);
        }
        runtime_cache_.runtime = flow_lm_.create_step_runtime(
            backend,
            threads,
            runtime_cache_.weights_runtime,
            total_cache_steps,
            prompt_capacity,
            initial_cache_capacity,
            flow_step_graph_context_bytes);
    }
    runtime.step_runtime = runtime_cache_.runtime;
    return runtime;
}

AcousticModelResult AcousticModel::generate(
    const AcousticPreparedRuntime & runtime,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & text_embeddings,
    const FlowLMState & initial_state,
    const AcousticGenerationConfig & config) const {
    AcousticModelResult result;
    const double generate_ms = engine::debug::measure_ms([&]() {
        auto state = start_stream(runtime, manifest, weights, text_embeddings, initial_state, config);
        result.latents.reserve(static_cast<size_t>(config.max_steps) * static_cast<size_t>(flow_lm_.config().latent_size));
        result.eos_logits.reserve(static_cast<size_t>(config.max_steps));

        while (auto step_result = next_stream_step(state)) {
            result.eos_logits.push_back(step_result->eos_logit);
            result.latents.insert(result.latents.end(), step_result->next_latent.begin(), step_result->next_latent.end());
            result.generated_steps += 1;
        }
        const auto flow_timing = flow_lm_.runtime_timing(*runtime.step_runtime);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.prompt.host_setup_ms", flow_timing.prompt_host_setup_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.prompt.import_state_ms", flow_timing.prompt_import_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.prompt.graph.compute_ms", flow_timing.prompt_graph_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.prompt.finalize_ms", flow_timing.prompt_finalize_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.prompt.mask_refresh_ms", flow_timing.prompt_mask_refresh_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.step.input_write_ms", flow_timing.step_input_write_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.step.graph.compute_ms", flow_timing.step_graph_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.step.output_read_ms", flow_timing.step_output_read_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.step.kv_update_ms", flow_timing.step_kv_update_ms);
        const auto plan_timing = flow_lm_.runtime_plan_timing(*runtime.step_runtime);
        engine::debug::timing_log_scalar(
            "pocket_tts.flow_lm.prompt.plan_create_ms",
            plan_timing.prompt_plan_create_ms);
        engine::debug::timing_log_scalar("pocket_tts.flow_lm.step.plan_create_ms", plan_timing.step_plan_create_ms);
    });

    if (result.generated_steps == 0) {
        throw std::runtime_error("PocketTTS acoustic model produced no latents");
    }
    engine::debug::timing_log_scalar("pocket_tts.acoustic.generate_ms", generate_ms);
    return result;
}

AcousticStreamState AcousticModel::start_stream(
    const AcousticPreparedRuntime & runtime,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & text_embeddings,
    const FlowLMState & initial_state,
    const AcousticGenerationConfig & config) const {
    (void) manifest;
    (void) weights;
    validate_generation_inputs(flow_lm_.config(), text_embeddings, config);
    if (runtime.step_runtime == nullptr) {
        throw std::runtime_error("PocketTTS acoustic runtime is not initialized");
    }
    flow_lm_.apply_prompt(*runtime.step_runtime, text_embeddings, runtime.prompt_steps, initial_state);

    AcousticStreamState state;
    state.runtime = runtime;
    state.config = config;
    state.current_input.assign(
        static_cast<size_t>(flow_lm_.config().latent_size),
        std::numeric_limits<float>::quiet_NaN());
    state.rng.seed(config.seed);
    return state;
}

std::optional<FlowLMStepResult> AcousticModel::next_stream_step(AcousticStreamState & state) const {
    if (state.done) {
        return std::nullopt;
    }
    if (state.runtime.step_runtime == nullptr) {
        throw std::runtime_error("PocketTTS acoustic runtime is not initialized");
    }
    if (state.step >= state.config.max_steps) {
        state.done = true;
        return std::nullopt;
    }

    auto noise = sample_noise_for_step(flow_lm_.config(), state);
    auto step_result = flow_lm_.run_step_in_place(
        *state.runtime.step_runtime,
        state.current_input,
        noise);

    const bool is_eos = step_result.eos_logit > state.config.eos_threshold;
    if (is_eos && state.eos_step < 0) {
        state.eos_step = state.step;
    }
    if (state.eos_step >= 0 && state.step >= state.eos_step + state.config.frames_after_eos) {
        state.done = true;
        return std::nullopt;
    }

    state.current_input = step_result.next_latent;
    ++state.step;
    ++state.generated_steps;
    return step_result;
}

void AcousticModel::clear_runtime_cache() const noexcept {
    runtime_cache_ = {};
}

int64_t AcousticModel::prepared_prompt_capacity() const noexcept {
    return runtime_cache_.runtime ? runtime_cache_.prompt_capacity : 0;
}

int AcousticModel::prepared_max_steps_capacity() const noexcept {
    return runtime_cache_.runtime ? runtime_cache_.max_steps_capacity : 0;
}

}  // namespace engine::models::pocket_tts
