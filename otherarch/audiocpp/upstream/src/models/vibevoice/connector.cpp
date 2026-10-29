#include "engine/models/vibevoice/connector.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/models/vibevoice/tokenizer_audio.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::vibevoice {
namespace {

namespace binding = modules::binding;

constexpr float kConnectorRmsNormEps = 1e-6F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

VibeVoiceConnectorWeights load_connector(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t input_dim,
    int64_t hidden_size,
    assets::TensorStorageType weight_storage_type) {
    if (input_dim <= 0 || hidden_size <= 0) {
        throw std::runtime_error("VibeVoice connector dimensions must be positive");
    }
    VibeVoiceConnectorWeights weights;
    weights.input_dim = input_dim;
    weights.hidden_size = hidden_size;
    weights.fc1.weight = store.load_tensor(
        source,
        prefix + ".fc1.weight",
        weight_storage_type,
        {hidden_size, input_dim});
    weights.fc1.bias = store.load_tensor(
        source,
        prefix + ".fc1.bias",
        assets::TensorStorageType::F32,
        {hidden_size});
    weights.norm = source.require_f32_tensor(prefix + ".norm.weight", {hidden_size});
    weights.fc2.weight = store.load_tensor(
        source,
        prefix + ".fc2.weight",
        weight_storage_type,
        {hidden_size, hidden_size});
    weights.fc2.bias = store.load_tensor(
        source,
        prefix + ".fc2.bias",
        assets::TensorStorageType::F32,
        {hidden_size});
    return weights;
}

}  // namespace

class VibeVoiceConnectorGraph {
public:
    VibeVoiceConnectorGraph(
        const VibeVoiceConnectorWeightsRuntime & runtime,
        const VibeVoiceConnectorWeights & weights,
        core::ConstantTensorCache & constants,
        const char * graph_name,
        int64_t batch_size,
        int64_t frames,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          weights_(&weights),
          constants_(&constants),
          batch_size_(batch_size),
          frames_(frames),
          graph_name_(graph_name) {
        if (batch_size_ <= 0) {
            throw std::runtime_error("VibeVoice connector graph requires positive batch size");
        }
        if (frames_ <= 0) {
            throw std::runtime_error("VibeVoice connector graph requires positive frames");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice connector graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), graph_name_.c_str(), runtime_->backend_type()};
        auto features = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, frames_, weights_->input_dim}));
        input_ = features.tensor;
        constants_->begin_graph();
        auto output = build_vibevoice_connector(ctx, features, *weights_, *constants_);
        output = core::ensure_backend_addressable_layout(ctx, output);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 8192, false);
        ggml_build_forward_expand(graph_, output_);
        constants_->finish_graph();
        constants_->ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice connector graph");
        }
    }

    ~VibeVoiceConnectorGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const VibeVoiceConnectorWeights & weights, int64_t batch_size, int64_t frames) const {
        return weights_ == &weights && batch_size_ == batch_size && frames_ == frames;
    }

    VibeVoiceConnectorOutput run(const std::vector<float> & features, int64_t input_dim) {
        if (input_dim != weights_->input_dim) {
            throw std::runtime_error("VibeVoice connector input_dim mismatch");
        }
        if (static_cast<int64_t>(features.size()) != frames_ * input_dim) {
            throw std::runtime_error("VibeVoice connector feature payload size mismatch");
        }
        ggml_backend_tensor_set(input_, features.data(), 0, features.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice connector graph compute failed");
        }
        VibeVoiceConnectorOutput out;
        out.frames = frames_;
        out.hidden_size = weights_->hidden_size;
        out.values.resize(static_cast<size_t>(frames_ * weights_->hidden_size));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

    std::vector<VibeVoiceConnectorOutput> run_batch(
        const std::vector<VibeVoiceTokenizerLatents> & features) {
        if (static_cast<int64_t>(features.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice connector feature batch size mismatch");
        }
        const size_t per_sample = static_cast<size_t>(frames_ * weights_->input_dim);
        std::vector<float> input(static_cast<size_t>(batch_size_) * per_sample, 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto & item = features[static_cast<size_t>(batch)];
            if (item.frames != frames_ || item.dim != weights_->input_dim) {
                throw std::runtime_error("VibeVoice connector batch feature shape mismatch");
            }
            if (item.values.size() != per_sample) {
                throw std::runtime_error("VibeVoice connector batch feature payload size mismatch");
            }
            std::copy(
                item.values.begin(),
                item.values.end(),
                input.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(per_sample)));
        }
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice connector graph compute failed");
        }
        std::vector<float> output(static_cast<size_t>(batch_size_ * frames_ * weights_->hidden_size), 0.0F);
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        std::vector<VibeVoiceConnectorOutput> out(static_cast<size_t>(batch_size_));
        const size_t output_per_sample = static_cast<size_t>(frames_ * weights_->hidden_size);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto & item = out[static_cast<size_t>(batch)];
            item.frames = frames_;
            item.hidden_size = weights_->hidden_size;
            const auto begin = output.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(output_per_sample));
            item.values.assign(begin, begin + static_cast<std::ptrdiff_t>(output_per_sample));
        }
        return out;
    }

private:
    const VibeVoiceConnectorWeightsRuntime * runtime_ = nullptr;
    const VibeVoiceConnectorWeights * weights_ = nullptr;
    core::ConstantTensorCache * constants_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t frames_ = 0;
    std::string graph_name_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

VibeVoiceConnectorWeightsBundle load_vibevoice_connector_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("VibeVoice connector requires model weights");
    }
    const auto & config = assets.config;
    VibeVoiceConnectorWeightsBundle weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vibevoice.connector.weights",
        weight_context_bytes);
    weights.acoustic = load_connector(
        *weights.store,
        *assets.model_weights,
        "model.acoustic_connector",
        config.acoustic_vae_dim,
        config.decoder.hidden_size,
        weight_storage_type);
    weights.semantic = load_connector(
        *weights.store,
        *assets.model_weights,
        "model.semantic_connector",
        config.semantic_vae_dim,
        config.decoder.hidden_size,
        weight_storage_type);
    weights.store->upload();
    return weights;
}

VibeVoiceConnectorWeightsRuntime::VibeVoiceConnectorWeightsRuntime(
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
        throw std::runtime_error("VibeVoice connector weights runtime requires assets");
    }
    if (threads_ <= 0) {
        throw std::runtime_error("VibeVoice connector weights runtime requires positive thread count");
    }
    const auto backend_started = std::chrono::steady_clock::now();
    backend_ = core::init_backend({backend_type, device, threads_});
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.connector_backend_init_ms",
        engine::debug::elapsed_ms(backend_started));
    const auto weights_started = std::chrono::steady_clock::now();
    weights_ = std::make_shared<VibeVoiceConnectorWeightsBundle>(
        load_vibevoice_connector_weights(
            *assets_,
            backend_,
            backend_type,
            weight_context_bytes,
            weight_storage_type));
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.connector_weights_load_ms",
        engine::debug::elapsed_ms(weights_started));
    acoustic_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.connector.acoustic.constants",
        constant_context_bytes);
    semantic_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.connector.semantic.constants",
        constant_context_bytes);
}

VibeVoiceConnectorWeightsRuntime::~VibeVoiceConnectorWeightsRuntime() {
    semantic_graph_.reset();
    acoustic_graph_.reset();
    semantic_constants_.reset();
    acoustic_constants_.reset();
    weights_.reset();
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

const VibeVoiceAssets & VibeVoiceConnectorWeightsRuntime::assets() const noexcept {
    return *assets_;
}

const VibeVoiceConnectorWeightsBundle & VibeVoiceConnectorWeightsRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t VibeVoiceConnectorWeightsRuntime::backend() const noexcept {
    return backend_;
}

core::BackendType VibeVoiceConnectorWeightsRuntime::backend_type() const noexcept {
    return backend_type_;
}

int VibeVoiceConnectorWeightsRuntime::threads() const noexcept {
    return threads_;
}

VibeVoiceConnectorOutput VibeVoiceConnectorWeightsRuntime::project_acoustic(
    const std::vector<float> & features,
    int64_t frames,
    int64_t input_dim) const {
    if (frames <= 0 || input_dim != weights_->acoustic.input_dim) {
        throw std::runtime_error("VibeVoice acoustic connector received invalid feature shape");
    }
    if (acoustic_graph_ == nullptr || !acoustic_graph_->matches(weights_->acoustic, 1, frames)) {
        acoustic_graph_.reset();
        acoustic_graph_ = std::make_unique<VibeVoiceConnectorGraph>(
            *this,
            weights_->acoustic,
            *acoustic_constants_,
            "vibevoice.connector.acoustic",
            1,
            frames,
            64ull * 1024ull * 1024ull);
    }
    auto out = acoustic_graph_->run(features, input_dim);
    return out;
}

std::vector<VibeVoiceConnectorOutput> VibeVoiceConnectorWeightsRuntime::project_acoustic_batch(
    const std::vector<VibeVoiceTokenizerLatents> & features) const {
    if (features.empty()) {
        throw std::runtime_error("VibeVoice acoustic connector batch requires non-empty features");
    }
    const int64_t frames = features.front().frames;
    if (frames <= 0 || features.front().dim != weights_->acoustic.input_dim) {
        throw std::runtime_error("VibeVoice acoustic connector batch received invalid feature shape");
    }
    for (const auto & item : features) {
        if (item.frames != frames || item.dim != weights_->acoustic.input_dim) {
            throw std::runtime_error("VibeVoice acoustic connector batch requires uniform feature shape");
        }
    }
    const int64_t batch_size = static_cast<int64_t>(features.size());
    if (acoustic_graph_ == nullptr || !acoustic_graph_->matches(weights_->acoustic, batch_size, frames)) {
        acoustic_graph_.reset();
        acoustic_graph_ = std::make_unique<VibeVoiceConnectorGraph>(
            *this,
            weights_->acoustic,
            *acoustic_constants_,
            "vibevoice.connector.acoustic",
            batch_size,
            frames,
            64ull * 1024ull * 1024ull);
    }
    auto out = acoustic_graph_->run_batch(features);
    return out;
}

VibeVoiceConnectorOutput VibeVoiceConnectorWeightsRuntime::project_semantic(
    const std::vector<float> & features,
    int64_t frames,
    int64_t input_dim) const {
    if (frames <= 0 || input_dim != weights_->semantic.input_dim) {
        throw std::runtime_error("VibeVoice semantic connector received invalid feature shape");
    }
    if (semantic_graph_ == nullptr || !semantic_graph_->matches(weights_->semantic, 1, frames)) {
        semantic_graph_.reset();
        semantic_graph_ = std::make_unique<VibeVoiceConnectorGraph>(
            *this,
            weights_->semantic,
            *semantic_constants_,
            "vibevoice.connector.semantic",
            1,
            frames,
            64ull * 1024ull * 1024ull);
    }
    auto out = semantic_graph_->run(features, input_dim);
    return out;
}

std::vector<VibeVoiceConnectorOutput> VibeVoiceConnectorWeightsRuntime::project_semantic_batch(
    const std::vector<VibeVoiceTokenizerLatents> & features) const {
    if (features.empty()) {
        throw std::runtime_error("VibeVoice semantic connector batch requires non-empty features");
    }
    const int64_t frames = features.front().frames;
    if (frames <= 0 || features.front().dim != weights_->semantic.input_dim) {
        throw std::runtime_error("VibeVoice semantic connector batch received invalid feature shape");
    }
    for (const auto & item : features) {
        if (item.frames != frames || item.dim != weights_->semantic.input_dim) {
            throw std::runtime_error("VibeVoice semantic connector batch requires uniform feature shape");
        }
    }
    const int64_t batch_size = static_cast<int64_t>(features.size());
    if (semantic_graph_ == nullptr || !semantic_graph_->matches(weights_->semantic, batch_size, frames)) {
        semantic_graph_.reset();
        semantic_graph_ = std::make_unique<VibeVoiceConnectorGraph>(
            *this,
            weights_->semantic,
            *semantic_constants_,
            "vibevoice.connector.semantic",
            batch_size,
            frames,
            64ull * 1024ull * 1024ull);
    }
    auto out = semantic_graph_->run_batch(features);
    return out;
}

core::TensorValue build_vibevoice_connector(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & features,
    const VibeVoiceConnectorWeights & weights,
    core::ConstantTensorCache & constants) {
    auto hidden = modules::LinearModule(
                      binding::linear_config(weights.input_dim, weights.hidden_size, true))
                      .build(ctx, features, weights.fc1);
    hidden = modules::RMSNormModule({weights.hidden_size, kConnectorRmsNormEps, true, false})
                 .build(ctx, hidden, binding::norm_data(constants, weights.norm));
    return modules::LinearModule(
               binding::linear_config(weights.hidden_size, weights.hidden_size, true))
        .build(ctx, hidden, weights.fc2);
}

}  // namespace engine::models::vibevoice
