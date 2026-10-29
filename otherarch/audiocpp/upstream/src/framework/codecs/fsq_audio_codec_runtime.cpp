#include "engine/framework/codecs/fsq_audio_codec_runtime.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::codecs {
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

void validate_matmul_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "FSQ audio codec weight_type supports only native, f32, f16, bf16, and q8_0");
    }
}

void validate_conv_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
            return;
        default:
            throw std::runtime_error(
                "FSQ audio codec_conv_weight_type supports only native, f32, and f16");
    }
}

modules::NormWeights load_affine_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {channels}),
        store.load_f32_tensor(source, prefix + ".bias", {channels}),
    };
}

FsqAudioCodecResnetBlockWeights load_resnet_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType conv_storage_type) {
    FsqAudioCodecResnetBlockWeights out;
    out.norm1 = load_affine_norm(store, source, prefix + ".norm1", channels);
    out.conv1 = binding::conv1d_from_source(
        store,
        source,
        prefix + ".conv1",
        conv_storage_type,
        channels,
        channels,
        3,
        true);
    out.norm2 = load_affine_norm(store, source, prefix + ".norm2", channels);
    out.conv2 = binding::conv1d_from_source(
        store,
        source,
        prefix + ".conv2",
        conv_storage_type,
        channels,
        channels,
        3,
        true);
    return out;
}

FsqAudioCodecTransformerLayerWeights load_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const FsqAudioCodecConfig & config,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type) {
    FsqAudioCodecTransformerLayerWeights out;
    out.attention_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".input_layernorm",
        config.hidden_size);
    if (backend_type == core::BackendType::Cpu) {
        const auto q_weight = source.require_f32(
            prefix + ".self_attn.q_proj.weight",
            {config.hidden_size, config.hidden_size});
        const auto k_weight = source.require_f32(
            prefix + ".self_attn.k_proj.weight",
            {config.hidden_size, config.hidden_size});
        const auto v_weight = source.require_f32(
            prefix + ".self_attn.v_proj.weight",
            {config.hidden_size, config.hidden_size});
        std::vector<float> qkv_weight;
        qkv_weight.reserve(static_cast<size_t>(3 * config.hidden_size * config.hidden_size));
        qkv_weight.insert(qkv_weight.end(), q_weight.begin(), q_weight.end());
        qkv_weight.insert(qkv_weight.end(), k_weight.begin(), k_weight.end());
        qkv_weight.insert(qkv_weight.end(), v_weight.begin(), v_weight.end());
        out.qkv_proj = modules::LinearWeights{
            store.make_from_f32(
                core::TensorShape::from_dims({3 * config.hidden_size, config.hidden_size}),
                matmul_storage_type,
                std::move(qkv_weight)),
            std::nullopt,
        };
    }
    out.q_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".self_attn.q_proj",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size,
        false);
    out.k_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".self_attn.k_proj",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size,
        false);
    out.v_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".self_attn.v_proj",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size,
        false);
    out.out_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".self_attn.o_proj",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size,
        false);
    out.ffn_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".post_attention_layernorm",
        config.hidden_size);
    out.fc1 = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.fc1",
        matmul_storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.fc2 = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.fc2",
        matmul_storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

int64_t fsq_codebook_size(const std::vector<int64_t> & levels) {
    int64_t size = 1;
    for (const int64_t level : levels) {
        if (level <= 1) {
            throw std::runtime_error("FSQ audio codec levels must be greater than one");
        }
        size *= level;
    }
    return size;
}

std::shared_ptr<const assets::TensorSource> require_source(
    std::shared_ptr<const assets::TensorSource> source) {
    if (source == nullptr) {
        throw std::runtime_error("FSQ audio codec runtime requires a tensor source");
    }
    return source;
}

void validate_config(const FsqAudioCodecConfig & config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.layers <= 0 ||
        config.attention_heads <= 0 || config.head_dim <= 0 || config.quantization_dim <= 0) {
        throw std::runtime_error("FSQ audio codec config has invalid transformer dimensions");
    }
    if (config.hop_length <= 0 || config.prior_blocks < 0 || config.post_blocks < 0) {
        throw std::runtime_error("FSQ audio codec config has invalid decoder dimensions");
    }
    if (config.quantization_levels.empty()) {
        throw std::runtime_error("FSQ audio codec config requires quantization levels");
    }
}

core::TensorValue resnet_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FsqAudioCodecResnetBlockWeights & weights,
    int64_t channels) {
    auto x = modules::GroupNormModule({channels, 32, 1.0e-6F, true, true})
                 .build(ctx, input, weights.norm1);
    x = modules::SiluModule().build(ctx, x);
    x = modules::Conv1dModule({channels, channels, 3, 1, 1, 1, true}).build(ctx, x, weights.conv1);
    x = modules::GroupNormModule({channels, 32, 1.0e-6F, true, true})
            .build(ctx, x, weights.norm2);
    x = modules::SiluModule().build(ctx, x);
    x = modules::Conv1dModule({channels, channels, 3, 1, 1, 1, true}).build(ctx, x, weights.conv2);
    return modules::ResidualAddModule{}.build(ctx, input, x);
}

core::TensorValue apply_codec_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FsqAudioCodecTransformerLayerWeights & weights,
    const FsqAudioCodecConfig & config,
    const core::TensorValue & positions) {
    const ggml_prec matmul_precision = ctx.backend_type == core::BackendType::Cpu
        ? GGML_PREC_F32
        : GGML_PREC_DEFAULT;
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
    if (ctx.backend_type == core::BackendType::Cpu && weights.qkv_proj.has_value()) {
        auto qkv = modules::LinearModule({config.hidden_size, 3 * config.hidden_size, false, matmul_precision})
                       .build(ctx, input, *weights.qkv_proj);
        q = modules::SliceModule({2, 0, config.hidden_size}).build(ctx, qkv);
        k = modules::SliceModule({2, config.hidden_size, config.hidden_size}).build(ctx, qkv);
        v = modules::SliceModule({2, 2 * config.hidden_size, config.hidden_size}).build(ctx, qkv);
    } else {
        q = modules::LinearModule({config.hidden_size, config.hidden_size, false, matmul_precision})
                .build(ctx, input, weights.q_proj);
        k = modules::LinearModule({config.hidden_size, config.hidden_size, false, matmul_precision})
                .build(ctx, input, weights.k_proj);
        v = modules::LinearModule({config.hidden_size, config.hidden_size, false, matmul_precision})
                .build(ctx, input, weights.v_proj);
    }
    const auto head_shape = core::TensorShape::from_dims({
        input.shape.dims[0],
        input.shape.dims[1],
        config.attention_heads,
        config.head_dim,
    });
    q = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), head_shape);
    k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), head_shape);
    v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), head_shape);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    if (ctx.backend_type == core::BackendType::Cpu) {
        v = core::wrap_tensor(ggml_cont(ctx.ggml, v.tensor), v.shape, v.type);
    }
    q = modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);
    const auto attention_lowering = ctx.backend_type == core::BackendType::Cpu
        ? modules::ScaledDotProductAttentionLowering::ExplicitCpuPerHead
        : modules::ScaledDotProductAttentionLowering::Explicit;
    auto context = modules::ScaledDotProductAttentionModule({
        config.head_dim,
        attention_lowering,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k, v);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        input.shape);
    return modules::LinearModule({config.hidden_size, config.hidden_size, false, matmul_precision})
        .build(ctx, context, weights.out_proj);
}

core::TensorValue transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FsqAudioCodecTransformerLayerWeights & weights,
    const FsqAudioCodecConfig & config,
    const core::TensorValue & positions) {
    auto x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                 .build(ctx, input, weights.attention_norm);
    x = apply_codec_attention(
        ctx,
        x,
        weights,
        config,
        positions);
    auto hidden = modules::ResidualAddModule{}.build(ctx, input, x);
    x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
            .build(ctx, hidden, weights.ffn_norm);
    const ggml_prec matmul_precision = ctx.backend_type == core::BackendType::Cpu
        ? GGML_PREC_F32
        : GGML_PREC_DEFAULT;
    x = modules::LinearModule({config.hidden_size, config.intermediate_size, false, matmul_precision})
            .build(ctx, x, weights.fc1);
    x = modules::SiluModule().build(ctx, x);
    x = modules::LinearModule({config.intermediate_size, config.hidden_size, false, matmul_precision})
            .build(ctx, x, weights.fc2);
    return modules::ResidualAddModule{}.build(ctx, hidden, x);
}

class CodecWeightsRuntime {
public:
    CodecWeightsRuntime(
        FsqAudioCodecConfig config,
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType matmul_storage_type,
        assets::TensorStorageType conv_storage_type,
        FsqAudioCodecWeightBinding weight_binding)
        : config_(std::move(config)),
          source_(require_source(std::move(source))),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<FsqAudioCodecDecoderWeights>(
              load_fsq_audio_codec_decoder_weights(
                  *source_,
                  config_,
                  std::move(weight_binding),
                  backend_,
                  backend_type_,
                  weight_context_bytes,
                  matmul_storage_type,
                  conv_storage_type))) {
        validate_config(config_);
        if (backend_ == nullptr) {
            throw std::runtime_error("FSQ audio codec backend is not initialized");
        }
    }

    const FsqAudioCodecConfig & config() const noexcept { return config_; }
    const FsqAudioCodecDecoderWeights & weights() const noexcept { return *weights_; }
    ggml_backend_t backend() const noexcept { return backend_; }
    core::BackendType backend_type() const noexcept { return backend_type_; }
    int threads() const noexcept { return threads_; }

private:
    FsqAudioCodecConfig config_;
    std::shared_ptr<const assets::TensorSource> source_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const FsqAudioCodecDecoderWeights> weights_;
};

class CodecHeadGraph {
public:
    CodecHeadGraph(
        std::shared_ptr<CodecWeightsRuntime> runtime,
        int64_t frames,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          frames_(frames) {
        if (frames_ <= 0) {
            throw std::runtime_error("FSQ audio codec requires positive frame count");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize FSQ audio codec graph context");
        }
        const auto & config = runtime_->config();
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), config.trace_name.c_str(), runtime_->backend_type()};
        levels_ = ggml_new_tensor_3d(
            ctx_.get(),
            GGML_TYPE_F32,
            static_cast<int64_t>(config.quantization_levels.size()),
            frames_,
            1);
        auto x = core::wrap_tensor(
            levels_,
            core::TensorShape::from_dims({1, frames_, static_cast<int64_t>(config.quantization_levels.size())}),
            GGML_TYPE_F32);
        x = modules::LinearModule({
            static_cast<int64_t>(config.quantization_levels.size()),
            config.quantization_dim,
            weights.quantizer_project_out.bias.has_value(),
        }).build(ctx, x, weights.quantizer_project_out);
        x = modules::LinearModule({
            config.quantization_dim,
            config.hidden_size,
            weights.acoustic_fc.bias.has_value(),
        }).build(ctx, x, weights.acoustic_fc);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::Conv1dModule({config.hidden_size, config.hidden_size, 7, 1, 3, 1, true}).build(ctx, x, weights.embed);
        for (const auto & block : weights.prior_blocks) {
            x = resnet_block(ctx, x, block, config.hidden_size);
        }
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, config.attention_heads);
        auto positions = core::wrap_tensor(
            positions_,
            core::TensorShape::from_dims({config.attention_heads}),
            GGML_TYPE_I32);
        for (const auto & layer : weights.transformer_layers) {
            x = transformer_layer(ctx, x, layer, config, positions);
        }
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        for (const auto & block : weights.post_blocks) {
            x = resnet_block(ctx, x, block, config.hidden_size);
        }
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::LayerNormModule({config.hidden_size, config.rms_norm_eps, true, true})
                .build(ctx, x, weights.final_norm);
        x = modules::LinearModule({
            config.hidden_size,
            config.hop_length * 4 + 2,
            weights.istft_head.bias.has_value(),
        }).build(ctx, x, weights.istft_head);
        x = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
        head_ = x.tensor;
        ggml_set_output(head_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, head_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate FSQ audio codec graph");
        }
        positions_values_.resize(static_cast<size_t>(config.attention_heads));
        for (int64_t i = 0; i < config.attention_heads; ++i) {
            positions_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(
            positions_,
            positions_values_.data(),
            0,
            positions_values_.size() * sizeof(int32_t));
        debug::timing_log_scalar(
            config.trace_name + ".graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar(config.trace_name + ".frames", frames_);
    }

    ~CodecHeadGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const CodecWeightsRuntime & runtime, int64_t frames) const {
        return runtime_.get() == &runtime && frames_ == frames;
    }

    FsqAudioCodecHead run(const std::vector<int32_t> & codes) {
        const auto & config = runtime_->config();
        if (static_cast<int64_t>(codes.size()) != frames_) {
            throw std::runtime_error("FSQ audio codec code count mismatch");
        }
        const auto levels = decode_fsq_audio_codec_levels(codes, config.quantization_levels);
        core::write_tensor_f32(core::wrap_tensor(
            levels_,
            core::TensorShape::from_dims({1, frames_, static_cast<int64_t>(config.quantization_levels.size())}),
            GGML_TYPE_F32), levels);
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar(
            config.trace_name + ".graph.compute_ms",
            engine::debug::elapsed_ms(compute_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FSQ audio codec graph compute failed");
        }
        FsqAudioCodecHead out;
        out.frames = frames_;
        out.out_dim = config.hop_length * 4 + 2;
        out.values = core::read_tensor_f32(head_);
        return out;
    }

private:
    std::shared_ptr<CodecWeightsRuntime> runtime_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * levels_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * head_ = nullptr;
    std::vector<int32_t> positions_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

}  // namespace

FsqAudioCodecDecoderWeights load_fsq_audio_codec_decoder_weights(
    const assets::TensorSource & source,
    const FsqAudioCodecConfig & config,
    FsqAudioCodecWeightBinding weight_binding,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    validate_matmul_storage_type(matmul_storage_type);
    validate_conv_storage_type(conv_storage_type);
    validate_config(config);
    FsqAudioCodecDecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        config.trace_name + ".weights",
        weight_context_bytes);
    weights.quantizer_project_out = binding::linear_from_source(
        *weights.store,
        source,
        weight_binding.quantizer_project_out,
        matmul_storage_type,
        config.quantization_dim,
        static_cast<int64_t>(config.quantization_levels.size()),
        true);
    weights.acoustic_fc = binding::linear_from_source(
        *weights.store,
        source,
        weight_binding.acoustic_fc,
        matmul_storage_type,
        config.hidden_size,
        config.quantization_dim,
        true);
    weights.embed = binding::conv1d_from_source(
        *weights.store,
        source,
        weight_binding.embed,
        conv_storage_type,
        config.hidden_size,
        config.hidden_size,
        7,
        true);
    weights.prior_blocks.reserve(static_cast<size_t>(config.prior_blocks));
    for (int64_t block = 0; block < config.prior_blocks; ++block) {
        weights.prior_blocks.push_back(load_resnet_block(
            *weights.store,
            source,
            weight_binding.prior_block_prefix + std::to_string(block),
            config.hidden_size,
            conv_storage_type));
    }
    weights.transformer_layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights.transformer_layers.push_back(load_transformer_layer(
            *weights.store,
            source,
            weight_binding.transformer_layer_prefix + std::to_string(layer),
            config,
            backend_type,
            matmul_storage_type));
    }
    weights.post_blocks.reserve(static_cast<size_t>(config.post_blocks));
    for (int64_t block = 0; block < config.post_blocks; ++block) {
        weights.post_blocks.push_back(load_resnet_block(
            *weights.store,
            source,
            weight_binding.post_block_prefix + std::to_string(block),
            config.hidden_size,
            conv_storage_type));
    }
    weights.final_norm = load_affine_norm(*weights.store, source, weight_binding.final_norm, config.hidden_size);
    weights.istft_head = binding::linear_from_source(
        *weights.store,
        source,
        weight_binding.istft_head,
        matmul_storage_type,
        config.hop_length * 4 + 2,
        config.hidden_size,
        true);
    weights.store->upload();
    return weights;
}

class FsqAudioCodecDecoderRuntime::Impl {
public:
    Impl(
        FsqAudioCodecConfig config,
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType matmul_storage_type,
        assets::TensorStorageType conv_storage_type,
        FsqAudioCodecWeightBinding weight_binding)
        : execution_(&execution),
          weights_(std::make_shared<CodecWeightsRuntime>(
              std::move(config),
              std::move(source),
              execution,
              weight_context_bytes,
              matmul_storage_type,
              conv_storage_type,
              std::move(weight_binding))),
          graph_arena_bytes_(graph_arena_bytes) {}

    FsqAudioCodecHead decode_head(const std::vector<int32_t> & codes) {
        if (codes.empty()) {
            throw std::runtime_error("FSQ audio codec cannot decode an empty code sequence");
        }
        const int64_t frames = static_cast<int64_t>(codes.size());
        if (head_graph_ == nullptr || !head_graph_->matches(*weights_, frames)) {
            head_graph_ = std::make_unique<CodecHeadGraph>(weights_, frames, graph_arena_bytes_);
        }
        return head_graph_->run(codes);
    }

    std::vector<float> decode_audio(const std::vector<int32_t> & codes) {
        auto timing_start = Clock::now();
        auto head = decode_head(codes);
        const auto & config = weights_->config();
        debug::timing_log_scalar(
            config.trace_name + ".head.total_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        const audio::STFTConfig stft_config{
            config.hop_length * 4,
            config.hop_length,
            config.hop_length * 4,
            false,
            audio::STFTPadMode::Constant,
            audio::STFTFamily::Kokoro,
        };
        const auto & window = audio::get_cached_stft_window(stft_config);
        if (execution_->backend_type() == core::BackendType::Cuda) {
            if (cuda_istft_ == nullptr || cuda_istft_frames_ != head.frames) {
                cuda_istft_ = std::make_unique<audio::CudaLogMagnitudePhaseISTFT>(
                    audio::CudaLogMagnitudePhaseISTFTConfig{
                        head.frames,
                        config.hop_length * 4,
                        config.hop_length,
                        head.out_dim,
                        execution_->config().device,
                    });
                cuda_istft_frames_ = head.frames;
            }
            timing_start = Clock::now();
            const auto result = cuda_istft_->compute(head.values, window);
            debug::timing_log_scalar(
                config.trace_name + ".istft.total_ms",
                engine::debug::elapsed_ms(timing_start, Clock::now()));
            return result.audio;
        }
        const auto threads = static_cast<size_t>(std::max(1, execution_->config().threads));
        if (host_istft_ == nullptr || host_istft_frames_ != head.frames) {
            host_istft_ = std::make_unique<audio::HostLogMagnitudePhaseISTFT>(
                audio::HostLogMagnitudePhaseISTFTConfig{
                    head.frames,
                    config.hop_length * 4,
                    config.hop_length,
                    head.out_dim,
                    threads,
            });
            host_istft_frames_ = head.frames;
        }
        timing_start = Clock::now();
        const auto result = host_istft_->compute(head.values, window);
        debug::timing_log_scalar(
            config.trace_name + ".istft.total_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        return result.audio;
    }

    void release_runtime_graphs() {
        head_graph_.reset();
        host_istft_.reset();
        host_istft_frames_ = 0;
        cuda_istft_.reset();
        cuda_istft_frames_ = 0;
    }

private:
    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<CodecWeightsRuntime> weights_;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<CodecHeadGraph> head_graph_;
    std::unique_ptr<audio::HostLogMagnitudePhaseISTFT> host_istft_;
    int64_t host_istft_frames_ = 0;
    std::unique_ptr<audio::CudaLogMagnitudePhaseISTFT> cuda_istft_;
    int64_t cuda_istft_frames_ = 0;
};

FsqAudioCodecDecoderRuntime::FsqAudioCodecDecoderRuntime(
    FsqAudioCodecConfig config,
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    FsqAudioCodecWeightBinding weight_binding)
    : impl_(std::make_unique<Impl>(
          std::move(config),
          std::move(source),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          matmul_storage_type,
          conv_storage_type,
          std::move(weight_binding))) {}

FsqAudioCodecDecoderRuntime::~FsqAudioCodecDecoderRuntime() = default;

FsqAudioCodecDecoderRuntime::FsqAudioCodecDecoderRuntime(FsqAudioCodecDecoderRuntime &&) noexcept = default;

FsqAudioCodecDecoderRuntime & FsqAudioCodecDecoderRuntime::operator=(
    FsqAudioCodecDecoderRuntime &&) noexcept = default;

FsqAudioCodecHead FsqAudioCodecDecoderRuntime::decode_head(const std::vector<int32_t> & codes) {
    return impl_->decode_head(codes);
}

std::vector<float> FsqAudioCodecDecoderRuntime::decode_audio(const std::vector<int32_t> & codes) {
    return impl_->decode_audio(codes);
}

void FsqAudioCodecDecoderRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

std::vector<float> decode_fsq_audio_codec_levels(
    const std::vector<int32_t> & codes,
    const std::vector<int64_t> & levels) {
    if (levels.empty()) {
        throw std::runtime_error("FSQ audio codec levels are empty");
    }
    const int64_t codebook_size = fsq_codebook_size(levels);
    std::vector<float> out(codes.size() * levels.size(), 0.0F);
    for (size_t frame = 0; frame < codes.size(); ++frame) {
        int64_t code = codes[frame];
        if (code < 0 || code >= codebook_size) {
            throw std::runtime_error("FSQ audio codec code is outside the codebook");
        }
        int64_t basis = 1;
        for (size_t dim = 0; dim < levels.size(); ++dim) {
            const int64_t level = levels[dim];
            const int64_t level_index = (code / basis) % level;
            out[frame * levels.size() + dim] =
                static_cast<float>(level_index) * (2.0F / static_cast<float>(level - 1)) - 1.0F;
            basis *= level;
        }
    }
    return out;
}

}  // namespace engine::codecs
