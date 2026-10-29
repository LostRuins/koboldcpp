#include "engine/models/vibevoice/diffusion_head.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::vibevoice {
namespace {

namespace binding = modules::binding;

constexpr int64_t kTimestepFrequencyEmbeddingSize = 256;
constexpr float kTimestepMaxPeriod = 10000.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_config(const VibeVoiceDiffusionHeadConfig & config) {
    if (config.hidden_size <= 0 || config.latent_size <= 0 || config.head_layers <= 0) {
        throw std::runtime_error("VibeVoice diffusion head dimensions must be positive");
    }
    if (!(config.head_ffn_ratio > 0.0F)) {
        throw std::runtime_error("VibeVoice diffusion head FFN ratio must be positive");
    }
    if (!(config.rms_norm_eps > 0.0F)) {
        throw std::runtime_error("VibeVoice diffusion head RMS epsilon must be positive");
    }
}

int64_t ffn_dim(const VibeVoiceDiffusionHeadConfig & config) {
    validate_config(config);
    const auto value = static_cast<int64_t>(static_cast<double>(config.hidden_size) * config.head_ffn_ratio);
    if (value <= 0) {
        throw std::runtime_error("VibeVoice diffusion head FFN dimension is invalid");
    }
    return value;
}

std::vector<float> make_timestep_freqs() {
    std::vector<float> freqs(static_cast<size_t>(kTimestepFrequencyEmbeddingSize / 2));
    const float half = static_cast<float>(kTimestepFrequencyEmbeddingSize / 2);
    for (int64_t i = 0; i < kTimestepFrequencyEmbeddingSize / 2; ++i) {
        freqs[static_cast<size_t>(i)] = std::exp(-std::log(kTimestepMaxPeriod) * static_cast<float>(i) / half);
    }
    return freqs;
}

VibeVoiceDiffusionHeadLayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const VibeVoiceDiffusionHeadConfig & config,
    int64_t layer,
    assets::TensorStorageType weight_storage_type) {
    const std::string prefix = "model.prediction_head.layers." + std::to_string(layer);
    const int64_t hidden = config.hidden_size;
    const int64_t ffn = ffn_dim(config);
    VibeVoiceDiffusionHeadLayerWeights weights;
    weights.gate_proj.weight = store.load_tensor(
        source,
        prefix + ".ffn.gate_proj.weight",
        weight_storage_type,
        {ffn, hidden});
    weights.up_proj.weight = store.load_tensor(
        source,
        prefix + ".ffn.up_proj.weight",
        weight_storage_type,
        {ffn, hidden});
    weights.down_proj.weight = store.load_tensor(
        source,
        prefix + ".ffn.down_proj.weight",
        weight_storage_type,
        {hidden, ffn});
    weights.norm = source.require_f32_tensor(prefix + ".norm.weight", {hidden});
    weights.ada_ln.weight = store.load_tensor(
        source,
        prefix + ".adaLN_modulation.1.weight",
        weight_storage_type,
        {3 * hidden, hidden});
    return weights;
}

core::TensorValue repeat_last_dim_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    core::TensorShape shape = {};
    shape.rank = like.shape.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    shape.dims[shape.rank - 1] = value.shape.dims[value.shape.rank - 1];
    auto reshaped = core::reshape_tensor(ctx, value, shape);
    return core::wrap_tensor(ggml_repeat(ctx.ggml, reshaped.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

core::TensorValue add_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(ggml_add(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue mul_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(ggml_mul(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & shift,
    const core::TensorValue & scale,
    const core::TensorValue & ones) {
    auto repeated_ones = repeat_last_dim_like(ctx, ones, scale);
    auto one_plus_scale = add_tensors(ctx, repeated_ones, scale);
    return add_tensors(ctx, mul_tensors(ctx, x, one_plus_scale), shift);
}

core::TensorValue build_timestep_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timesteps,
    const VibeVoiceDiffusionHeadWeights & weights,
    const VibeVoiceDiffusionHeadConfig & config) {
    core::validate_rank_between(timesteps, 1, 1, "timesteps");
    const int64_t batch = timesteps.shape.dims[0];
    auto freqs = core::reshape_tensor(ctx, weights.time_freqs, core::TensorShape::from_dims({1, kTimestepFrequencyEmbeddingSize / 2}));
    freqs = modules::RepeatModule({core::TensorShape::from_dims({batch, kTimestepFrequencyEmbeddingSize / 2})}).build(ctx, freqs);
    auto expanded_time = core::reshape_tensor(ctx, timesteps, core::TensorShape::from_dims({batch, 1}));
    expanded_time = modules::RepeatModule({core::TensorShape::from_dims({batch, kTimestepFrequencyEmbeddingSize / 2})}).build(ctx, expanded_time);
    auto args = modules::MulModule{}.build(ctx, expanded_time, freqs);
    auto cos_part = core::wrap_tensor(ggml_cos(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto embedding = modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
    auto hidden = modules::LinearModule(
                      binding::linear_config(kTimestepFrequencyEmbeddingSize, config.hidden_size, false))
                      .build(ctx, embedding, weights.timestep_fc1);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    return modules::LinearModule(
               binding::linear_config(config.hidden_size, config.hidden_size, false))
        .build(ctx, hidden, weights.timestep_fc2);
}

core::TensorValue build_swiglu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VibeVoiceDiffusionHeadLayerWeights & weights,
    const VibeVoiceDiffusionHeadConfig & config) {
    const int64_t ffn = ffn_dim(config);
    auto gate = modules::LinearModule(
                    binding::linear_config(config.hidden_size, ffn, false))
                    .build(ctx, input, weights.gate_proj);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(
                  binding::linear_config(config.hidden_size, ffn, false))
                  .build(ctx, input, weights.up_proj);
    return modules::LinearModule(
               binding::linear_config(ffn, config.hidden_size, false))
        .build(ctx, modules::MulModule{}.build(ctx, gate, up), weights.down_proj);
}

core::TensorValue build_head_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & c,
    const VibeVoiceDiffusionHeadLayerWeights & weights,
    const VibeVoiceDiffusionHeadConfig & config,
    const core::TensorValue & ones,
    core::ConstantTensorCache & constants) {
    auto modulation = modules::LinearModule(
                          binding::linear_config(config.hidden_size, 3 * config.hidden_size, false))
                          .build(ctx, modules::SiluModule{}.build(ctx, c), weights.ada_ln);
    auto shift = modules::SliceModule({static_cast<int>(modulation.shape.rank - 1), 0, config.hidden_size}).build(ctx, modulation);
    auto scale = modules::SliceModule({static_cast<int>(modulation.shape.rank - 1), config.hidden_size, config.hidden_size}).build(ctx, modulation);
    auto gate = modules::SliceModule({static_cast<int>(modulation.shape.rank - 1), 2 * config.hidden_size, config.hidden_size}).build(ctx, modulation);
    auto normed = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                      .build(ctx, x, binding::norm_data(constants, weights.norm));
    auto ffn_input = modulate(ctx, normed, shift, scale, ones);
    return add_tensors(ctx, x, mul_tensors(ctx, gate, build_swiglu(ctx, ffn_input, weights, config)));
}

core::TensorValue build_final_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & c,
    const VibeVoiceDiffusionHeadWeights & weights,
    const VibeVoiceDiffusionHeadConfig & config) {
    auto modulation = modules::LinearModule(
                          binding::linear_config(config.hidden_size, 2 * config.hidden_size, false))
                          .build(ctx, modules::SiluModule{}.build(ctx, c), weights.final_layer.ada_ln);
    auto shift = modules::SliceModule({static_cast<int>(modulation.shape.rank - 1), 0, config.hidden_size}).build(ctx, modulation);
    auto scale = modules::SliceModule({static_cast<int>(modulation.shape.rank - 1), config.hidden_size, config.hidden_size}).build(ctx, modulation);
    auto normed = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, false, false})
                      .build(ctx, x, {std::nullopt, std::nullopt});
    auto projected = modulate(ctx, normed, shift, scale, weights.ones);
    return modules::LinearModule(
               binding::linear_config(config.hidden_size, config.latent_size, false))
        .build(ctx, projected, weights.final_layer.linear);
}

}  // namespace

class VibeVoiceDiffusionHeadGraph {
public:
    VibeVoiceDiffusionHeadGraph(
        const VibeVoiceDiffusionHeadWeightsRuntime & runtime,
        int64_t frames,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          frames_(frames) {
        if (frames_ <= 0) {
            throw std::runtime_error("VibeVoice diffusion head graph requires positive frames");
        }
        const auto & config = runtime_->assets().config.diffusion_head;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice diffusion head graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.diffusion_head", runtime_->backend_type()};
        auto noisy = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({frames_, config.latent_size}));
        noisy_ = noisy.tensor;
        timesteps_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, frames_);
        auto timestep_value = core::wrap_tensor(timesteps_, core::TensorShape::from_dims({frames_}), GGML_TYPE_F32);
        auto condition = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({frames_, config.hidden_size}));
        condition_ = condition.tensor;

        runtime_->constants().begin_graph();
        auto output = build_vibevoice_diffusion_head(
            ctx,
            noisy,
            timestep_value,
            condition,
            runtime_->weights(),
            config,
            runtime_->constants());
        output = core::ensure_backend_addressable_layout(ctx, output);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        runtime_->constants().finish_graph();
        runtime_->constants().ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice diffusion head graph");
        }
    }

    ~VibeVoiceDiffusionHeadGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const VibeVoiceDiffusionHeadWeightsRuntime & runtime, int64_t frames) const {
        return runtime_ == &runtime && frames_ == frames;
    }

    VibeVoiceDiffusionPrediction run(
        const std::vector<float> & noisy_images,
        int64_t latent_size,
        float timestep,
        const std::vector<float> & condition,
        int64_t condition_hidden_size) {
        const auto & config = runtime_->assets().config.diffusion_head;
        if (latent_size != config.latent_size || condition_hidden_size != config.hidden_size) {
            throw std::runtime_error("VibeVoice diffusion head input dimension mismatch");
        }
        if (static_cast<int64_t>(noisy_images.size()) != frames_ * config.latent_size) {
            throw std::runtime_error("VibeVoice diffusion head noisy latent payload size mismatch");
        }
        if (static_cast<int64_t>(condition.size()) != frames_ * config.hidden_size) {
            throw std::runtime_error("VibeVoice diffusion head condition payload size mismatch");
        }
        ggml_backend_tensor_set(noisy_, noisy_images.data(), 0, noisy_images.size() * sizeof(float));
        std::vector<float> timestep_values(static_cast<size_t>(frames_), timestep);
        ggml_backend_tensor_set(timesteps_, timestep_values.data(), 0, timestep_values.size() * sizeof(float));
        ggml_backend_tensor_set(condition_, condition.data(), 0, condition.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice diffusion head graph compute failed");
        }
        VibeVoiceDiffusionPrediction out;
        out.frames = frames_;
        out.latent_size = config.latent_size;
        out.values.resize(static_cast<size_t>(frames_ * config.latent_size));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    const VibeVoiceDiffusionHeadWeightsRuntime * runtime_ = nullptr;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * noisy_ = nullptr;
    ggml_tensor * timesteps_ = nullptr;
    ggml_tensor * condition_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

VibeVoiceDiffusionHeadWeights load_vibevoice_diffusion_head_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("VibeVoice diffusion head requires model weights");
    }
    const auto & config = assets.config.diffusion_head;
    validate_config(config);
    VibeVoiceDiffusionHeadWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vibevoice.diffusion_head.weights",
        weight_context_bytes);
    weights.noisy_images_proj.weight = weights.store->load_tensor(
        *assets.model_weights,
        "model.prediction_head.noisy_images_proj.weight",
        weight_storage_type,
        {config.hidden_size, config.latent_size});
    weights.cond_proj.weight = weights.store->load_tensor(
        *assets.model_weights,
        "model.prediction_head.cond_proj.weight",
        weight_storage_type,
        {config.hidden_size, config.hidden_size});
    weights.timestep_fc1.weight = weights.store->load_tensor(
        *assets.model_weights,
        "model.prediction_head.t_embedder.mlp.0.weight",
        weight_storage_type,
        {config.hidden_size, kTimestepFrequencyEmbeddingSize});
    weights.timestep_fc2.weight = weights.store->load_tensor(
        *assets.model_weights,
        "model.prediction_head.t_embedder.mlp.2.weight",
        weight_storage_type,
        {config.hidden_size, config.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.head_layers));
    for (int64_t layer = 0; layer < config.head_layers; ++layer) {
        weights.layers.push_back(load_layer_weights(
            *weights.store,
            *assets.model_weights,
            config,
            layer,
            weight_storage_type));
    }
    weights.final_layer.ada_ln.weight = weights.store->load_tensor(
        *assets.model_weights,
        "model.prediction_head.final_layer.adaLN_modulation.1.weight",
        weight_storage_type,
        {2 * config.hidden_size, config.hidden_size});
    weights.final_layer.linear.weight = weights.store->load_tensor(
        *assets.model_weights,
        "model.prediction_head.final_layer.linear.weight",
        weight_storage_type,
        {config.latent_size, config.hidden_size});
    weights.time_freqs = weights.store->make_f32(
        core::TensorShape::from_dims({kTimestepFrequencyEmbeddingSize / 2}),
        make_timestep_freqs());
    weights.ones = weights.store->make_f32(
        core::TensorShape::from_dims({config.hidden_size}),
        std::vector<float>(static_cast<size_t>(config.hidden_size), 1.0F));
    weights.store->upload();
    return weights;
}

VibeVoiceDiffusionHeadWeightsRuntime::VibeVoiceDiffusionHeadWeightsRuntime(
    std::shared_ptr<const VibeVoiceAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t weight_context_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      backend_type_(backend_type),
      threads_(threads) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VibeVoice diffusion head weights runtime requires assets");
    }
    if (threads_ <= 0) {
        throw std::runtime_error("VibeVoice diffusion head weights runtime requires positive thread count");
    }
    const auto backend_started = std::chrono::steady_clock::now();
    backend_ = core::init_backend({backend_type, device, threads_});
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.diffusion_head_backend_init_ms",
        engine::debug::elapsed_ms(backend_started));
    const auto weights_started = std::chrono::steady_clock::now();
    weights_ = std::make_shared<VibeVoiceDiffusionHeadWeights>(
        load_vibevoice_diffusion_head_weights(
            *assets_,
            backend_,
            backend_type,
            weight_context_bytes,
            weight_storage_type));
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.diffusion_head_weights_load_ms",
        engine::debug::elapsed_ms(weights_started));
    constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.diffusion_head.constants",
        constant_context_bytes);
}

VibeVoiceDiffusionHeadWeightsRuntime::~VibeVoiceDiffusionHeadWeightsRuntime() {
    graph_.reset();
    constants_.reset();
    weights_.reset();
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

const VibeVoiceAssets & VibeVoiceDiffusionHeadWeightsRuntime::assets() const noexcept {
    return *assets_;
}

const VibeVoiceDiffusionHeadWeights & VibeVoiceDiffusionHeadWeightsRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t VibeVoiceDiffusionHeadWeightsRuntime::backend() const noexcept {
    return backend_;
}

core::BackendType VibeVoiceDiffusionHeadWeightsRuntime::backend_type() const noexcept {
    return backend_type_;
}

core::ConstantTensorCache & VibeVoiceDiffusionHeadWeightsRuntime::constants() const noexcept {
    return *constants_;
}

int VibeVoiceDiffusionHeadWeightsRuntime::threads() const noexcept {
    return threads_;
}

VibeVoiceDiffusionPrediction VibeVoiceDiffusionHeadWeightsRuntime::predict(
    const std::vector<float> & noisy_images,
    int64_t frames,
    int64_t latent_size,
    float timestep,
    const std::vector<float> & condition,
    int64_t condition_hidden_size) const {
    const auto & config = assets_->config.diffusion_head;
    if (frames <= 0 || latent_size != config.latent_size || condition_hidden_size != config.hidden_size) {
        throw std::runtime_error("VibeVoice diffusion head received invalid input shape");
    }
    if (graph_ == nullptr || !graph_->matches(*this, frames)) {
        graph_.reset();
        graph_ = std::make_unique<VibeVoiceDiffusionHeadGraph>(
            *this,
            frames,
            128ull * 1024ull * 1024ull);
    }
    return graph_->run(noisy_images, latent_size, timestep, condition, condition_hidden_size);
}

core::TensorValue build_vibevoice_diffusion_head(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & noisy_images,
    const core::TensorValue & timesteps,
    const core::TensorValue & condition,
    const VibeVoiceDiffusionHeadWeights & weights,
    const VibeVoiceDiffusionHeadConfig & config,
    core::ConstantTensorCache & constants) {
    validate_config(config);
    core::validate_shape(noisy_images, core::TensorShape::from_dims({noisy_images.shape.dims[0], config.latent_size}), "noisy_images");
    core::validate_shape(condition, core::TensorShape::from_dims({noisy_images.shape.dims[0], config.hidden_size}), "condition");
    core::validate_shape(timesteps, core::TensorShape::from_dims({noisy_images.shape.dims[0]}), "timesteps");
    if (static_cast<int64_t>(weights.layers.size()) != config.head_layers) {
        throw std::runtime_error("VibeVoice diffusion head layer count mismatch");
    }

    auto x = modules::LinearModule(
                 binding::linear_config(config.latent_size, config.hidden_size, false))
                 .build(ctx, noisy_images, weights.noisy_images_proj);
    auto time_embedding = build_timestep_embedding(ctx, timesteps, weights, config);
    auto projected_condition = modules::LinearModule(
                                   binding::linear_config(config.hidden_size, config.hidden_size, false))
                                   .build(ctx, condition, weights.cond_proj);
    auto c = add_tensors(ctx, projected_condition, time_embedding);
    for (const auto & layer : weights.layers) {
        x = build_head_layer(ctx, x, c, layer, config, weights.ones, constants);
    }
    return build_final_layer(ctx, x, c, weights, config);
}

}  // namespace engine::models::vibevoice
