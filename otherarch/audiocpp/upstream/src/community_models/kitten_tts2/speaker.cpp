#include "engine/community_models/kitten_tts2/speaker.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace engine::community_models::kitten_tts2 {
namespace {
struct Affine { core::TensorValue scale, bias; };
struct Conv { modules::Conv1dConfig config; modules::Conv1dWeights weights; };
// Each inference graph is short-lived; release backend graph state before its arena.
struct Graph {
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    explicit Graph(ggml_backend_t b) : backend(b) {
        ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
        if (!ctx) throw std::runtime_error("cannot allocate Kitten speaker graph");
        graph = ggml_new_graph_custom(ctx, 2048, false);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    ~Graph() {
        core::release_backend_graph_resources(backend, graph);
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
    }
};
}
struct SpeakerEncoder::Impl {
    const core::ExecutionContext & execution;
    core::BackendWeightStore store;
    Affine waveform_norm;
    std::array<Affine, 3> frontend_norm;
    std::array<Affine, 5> batch_norm;
    std::array<Conv, 3> frontend;
    std::array<Conv, 5> tdnn;
    core::TensorValue embedding_weight, embedding_bias;
    std::vector<float> projection, projection_bias, norm_weight, norm_bias;

    Impl(const assets::TensorSource & source, const assets::TensorSource & lm, const core::ExecutionContext & e)
        : execution(e), store(e.backend(), e.backend_type(), "kitten_tts2.speaker.weights", 1024 * 1024) {
        auto norm = [&](const std::string & p, int64_t n) {
            return Affine{store.load_f32_tensor(source, p + ".weight", {n}),
                store.load_f32_tensor(source, p + ".bias", {n})};
        };
        auto conv = [&](const std::string & p, int64_t in, int64_t out, int64_t k, int dilation = 1) {
            return Conv{{in, out, k, 1, 0, dilation, true},
                {store.load_f32_tensor(source, p + ".weight", {out, in, k}),
                 store.load_f32_tensor(source, p + ".bias", {out})}};
        };
        waveform_norm = norm("sincnet.wav_norm1d", 1);
        // ParamSincFB (cosine then sine pairs), evaluated once from learned cutoffs.
        auto low = source.require_f32("sincnet.conv1d.0.filterbank.low_hz_", {40, 1});
        auto band = source.require_f32("sincnet.conv1d.0.filterbank.band_hz_", {40, 1});
        std::vector<float> filters(80 * 251);
        constexpr double pi = 3.14159265358979323846;
        for (int i = 0; i < 40; ++i) {
            const float lo = 50 + std::abs(low[i]);
            const float hi = std::clamp(lo + 50 + std::abs(band[i]), 50.0f, 8000.0f);
            if (!(hi > lo)) throw std::runtime_error("invalid Kitten sinc filter bandwidth");
            for (int j = 0; j < 125; ++j) {
                const float n = static_cast<float>(2 * pi) * (static_cast<float>(j - 125) / 16000);
                const float window = static_cast<float>(0.54 - 0.46 * std::cos(2 * pi * j / 250));
                const float even = (std::sin(hi*n) - std::sin(lo*n)) / (n/2) * window / (2*(hi-lo));
                const float odd = (std::cos(lo*n) - std::cos(hi*n)) / (n/2) * window / (2*(hi-lo));
                filters[i*251+j] = filters[i*251+250-j] = even;
                filters[(i+40)*251+j] = odd;
                filters[(i+40)*251+250-j] = -odd;
            }
            filters[i*251+125] = 1;
        }
        frontend[0] = {{1, 80, 251, 10, 0, 1, false},
            {store.make_f32(core::TensorShape::from_dims({80,1,251}), std::move(filters)), std::nullopt}};
        frontend[1] = conv("sincnet.conv1d.1", 80, 60, 5);
        frontend[2] = conv("sincnet.conv1d.2", 60, 60, 5);
        for (int i = 0; i < 3; ++i) frontend_norm[i] = norm("sincnet.norm1d." + std::to_string(i), i ? 60 : 80);
        const int channels[] = {60,512,512,512,512,1500}, kernels[] = {5,3,3,1,1}, dilations[] = {1,2,3,1,1};
        for (int i = 0; i < 5; ++i) {
            tdnn[i] = conv("tdnns." + std::to_string(i*3), channels[i], channels[i+1], kernels[i], dilations[i]);
            const auto p = "tdnns." + std::to_string(i*3+2);
            auto scale = source.require_f32(p+".weight", {channels[i+1]});
            auto bias = source.require_f32(p+".bias", {channels[i+1]});
            auto mean = source.require_f32(p+".running_mean", {channels[i+1]});
            auto var = source.require_f32(p+".running_var", {channels[i+1]});
            for (int j = 0; j < channels[i+1]; ++j) {
                scale[j] /= std::sqrt(var[j] + 1e-5f);
                bias[j] -= mean[j] * scale[j];
            }
            batch_norm[i] = {store.make_f32(core::TensorShape::from_dims({channels[i+1]}), std::move(scale)),
                store.make_f32(core::TensorShape::from_dims({channels[i+1]}), std::move(bias))};
        }
        embedding_weight = store.load_f32_tensor(source, "embedding.weight", {512,3000});
        embedding_bias = store.load_f32_tensor(source, "embedding.bias", {512});
        store.upload();
        projection = lm.require_f32("spk_proj.0.weight", {2048,512});
        projection_bias = lm.require_f32("spk_proj.0.bias", {2048});
        norm_weight = lm.require_f32("spk_proj.1.weight", {2048});
        norm_bias = lm.require_f32("spk_proj.1.bias", {2048});
    }
};
SpeakerEncoder::SpeakerEncoder(const assets::TensorSource & source, const assets::TensorSource & lm,
    const core::ExecutionContext & e) : impl_(std::make_unique<Impl>(source, lm, e)) {}
SpeakerEncoder::~SpeakerEncoder() = default;

std::vector<float> SpeakerEncoder::embed(const std::vector<float> & mono_16k) const {
    // Shorter clips cannot traverse all valid TDNN convolutions and statistics pooling.
    if (mono_16k.size() < 16000) throw std::runtime_error("Kitten voice reference must be at least one second");
    const auto & p = *impl_;
    Graph g(p.execution.backend());
    auto * c = g.ctx;
    core::ModuleBuildContext build{c, "kitten_tts2.speaker", p.execution.backend_type()};
    const auto input = core::make_tensor(build, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 1, static_cast<int64_t>(mono_16k.size())}));
    ggml_set_input(input.tensor);
    auto normalize = [&](const core::TensorValue & value, const Affine & weights) {
        // Normalize each channel over time, then apply its learned scalar affine.
        return modules::AdaptiveInstanceNorm1dModule({value.shape.at(1), 1e-5f})
            .build(build, value, {weights.scale, weights.bias});
    };
    auto x = normalize(input, p.waveform_norm);
    const modules::LeakyReluModule activation({0.01f});
    const modules::ReduceMeanModule mean_over_time({-1});
    auto conv = [&](const core::TensorValue & value, const Conv & layer) {
        return modules::Conv1dModule(layer.config).build(build, value, layer.weights);
    };
    for (int i = 0; i < 3; ++i) {
        x = conv(x, p.frontend[i]);
        // Abs and temporal max pooling have no equivalent framework modules.
        if (i == 0) x = core::wrap_tensor(ggml_abs(c, x.tensor), x.shape);
        // A 3x1 window preserves the channel dimension and is equivalent to
        // temporal MaxPool1d(3, 3). CUDA supports POOL_2D, but not POOL_1D.
        auto * pooled = ggml_pool_2d(c, x.tensor, GGML_OP_POOL_MAX, 3, 1, 3, 1, 0, 0);
        x = core::wrap_tensor(pooled, x.shape.with_last_dim(pooled->ne[0]));
        x = activation.build(build, normalize(x, p.frontend_norm[i]));
    }
    for (int i = 0; i < 5; ++i) {
        x = activation.build(build, conv(x, p.tdnn[i]));
        x = modules::BatchNorm1dEvalModule({x.shape.at(1)})
            .build(build, x, {p.batch_norm[i].scale, p.batch_norm[i].bias});
    }
    const auto frames = x.shape.last_dim();
    if (frames < 2) throw std::runtime_error("Kitten speaker reference is too short for statistics pooling");
    const auto mean = mean_over_time.build(build, x);
    // Keep the centered, unbiased variance and its original operation order.
    // Subtract with broadcasting, square and scalar scale have no direct modules.
    auto * diff = ggml_sub(c, x.tensor, mean.tensor);
    const auto squared = core::wrap_tensor(ggml_sqr(c, diff), x.shape);
    const auto variance = mean_over_time.build(build, squared);
    const auto unbiased = core::wrap_tensor(ggml_scale(c, variance.tensor,
        static_cast<float>(frames) / static_cast<float>(frames - 1)), variance.shape);
    const auto stddev = modules::SqrtModule().build(build, unbiased);
    const auto statistics = modules::ConcatModule({1}).build(build, mean, stddev);
    const auto pooled = core::reshape_tensor(build, statistics, core::TensorShape::from_dims({3000}));
    const auto output = modules::LinearModule({3000, 512, true})
        .build(build, pooled, {p.embedding_weight, p.embedding_bias});
    ggml_set_output(output.tensor);
    ggml_build_forward_expand(g.graph, output.tensor);
    if (!ggml_gallocr_alloc_graph(g.allocator, g.graph)) throw std::runtime_error("cannot allocate Kitten speaker buffers");
    ggml_backend_tensor_set(input.tensor, mono_16k.data(), 0, mono_16k.size()*sizeof(float));
    if (core::compute_backend_graph(p.execution.backend(), g.graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Kitten speaker inference failed");
    auto result = core::read_tensor_f32(output.tensor);
    double norm = 0;
    for (float value : result) norm += value*value;
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm < 1e-12) throw std::runtime_error("invalid Kitten speaker embedding");
    for (auto & value : result) value /= static_cast<float>(norm);
    return result;
}

std::vector<float> SpeakerEncoder::project(const std::vector<float> & embedding) const {
    if (embedding.size() != 512) throw std::runtime_error("Kitten speaker embedding must have 512 values");
    const auto & p = *impl_;
    auto input = embedding;
    core::round_f32_to_bf16_in_place(input);
    std::vector<float> out(2048);
    for (size_t row = 0; row < out.size(); ++row) {
        float sum = p.projection_bias[row];
        for (size_t col = 0; col < input.size(); ++col) sum += p.projection[row*512+col] * input[col];
        out[row] = sum;
    }
    core::round_f32_to_bf16_in_place(out);
    const double mean = std::accumulate(out.begin(), out.end(), 0.0) / out.size();
    double variance = 0;
    for (float x : out) variance += (x-mean)*(x-mean);
    const double inv = 1 / std::sqrt(variance/out.size()+1e-5);
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>((out[i]-mean)*inv)*p.norm_weight[i]+p.norm_bias[i];
    core::round_f32_to_bf16_in_place(out);
    return out;
}
}
