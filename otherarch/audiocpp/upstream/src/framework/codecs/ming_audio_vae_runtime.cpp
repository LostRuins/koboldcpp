#include "engine/framework/codecs/ming_audio_vae_runtime.h"

#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::modules {

namespace {

void load_audio_backbone(core::BackendWeightStore & store, const assets::TensorSource & source,
    const MingBackboneConfig & backbone, const std::string & prefix, core::ExecutionContext & execution,
    core::AttentionPreference attention, bool bf16_residual, modules::DecoderHiddenConfig & config,
    modules::DecoderHiddenWeights & weights, int64_t layers = -1) {
    namespace binding = modules::binding;
    auto & c = config.stack;
    c.hidden_size = backbone.hidden_size;
    c.intermediate_size = backbone.intermediate_size;
    c.layers = layers < 0 ? backbone.layers : layers;
    c.num_attention_heads = backbone.attention_heads;
    c.num_key_value_heads = backbone.key_value_heads;
    c.head_dim = c.hidden_size / c.num_attention_heads;
    c.rms_norm_eps = backbone.rms_norm_eps;
    c.rope_theta = backbone.rope_theta;
    c.use_qk_norm = false;
    c.runtime.attention.allow_flash_attention = core::resolve_flash_attention(execution.backend(), c.head_dim, attention);
    c.runtime.attention.prefill_mode = c.runtime.attention.allow_flash_attention
        ? modules::DecoderAttentionMode::FlashGroupedViewKV : modules::DecoderAttentionMode::ManualRepeat;
    c.activation_cast.enabled = true;
    c.activation_cast.after_input_norm = true;
    c.activation_cast.after_qkv_projection = true;
    c.activation_cast.after_rope = true;
    c.activation_cast.after_attention = true;
    c.activation_cast.after_attention_output = true;
    c.activation_cast.after_ffn_norm = true;
    c.activation_cast.after_mlp_projection = true;
    c.activation_cast.after_mlp_silu = true;
    c.activation_cast.after_mlp_mul = true;
    c.activation_cast.after_residual = bf16_residual;
    c.activation_cast.after_output = bf16_residual;
    if (bf16_residual) {
        c.activation_cast.norm_intermediate_type = GGML_TYPE_BF16;
        c.activation_cast.rope_intermediate_type = GGML_TYPE_BF16;
    }
    config.hidden_mode = modules::CausalDecoderLogitsMode::AllSteps;
    std::vector<float> inverse_frequency(c.head_dim / 2);
    for (size_t i = 0; i < inverse_frequency.size(); ++i) {
        inverse_frequency[i] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / c.head_dim);
    }
    auto rounded_frequency = inverse_frequency;
    core::round_f32_to_bf16_in_place(rounded_frequency);
    for (size_t i = 0; i < inverse_frequency.size(); ++i) inverse_frequency[i] /= rounded_frequency[i];
    const auto rope_factors = store.make_f32(core::TensorShape::from_dims({c.head_dim / 2}), inverse_frequency);
    for (int64_t i = 0; i < c.layers; ++i) {
        const auto p = prefix + ".layers." + std::to_string(i);
        modules::DecoderLayerWeights layer;
        layer.rope_frequency_factors = rope_factors;
        layer.input_norm = binding::norm_weight_from_source(store, source, p + ".input_layernorm", c.hidden_size);
        layer.post_norm = binding::norm_weight_from_source(store, source, p + ".post_attention_layernorm", c.hidden_size);
        auto & a = layer.self_attention;
        const auto q = binding::linear_from_source(store, source, p + ".self_attn.q_proj",
            assets::TensorStorageType::Native, c.hidden_size, c.hidden_size, true);
        const auto k = binding::linear_from_source(store, source, p + ".self_attn.k_proj",
            assets::TensorStorageType::Native, c.num_key_value_heads * c.head_dim, c.hidden_size, true);
        const auto v = binding::linear_from_source(store, source, p + ".self_attn.v_proj",
            assets::TensorStorageType::Native, c.num_key_value_heads * c.head_dim, c.hidden_size, true);
        a.q_weight = q.weight; a.q_bias = q.bias;
        a.k_weight = k.weight; a.k_bias = k.bias;
        a.v_weight = v.weight; a.v_bias = v.bias;
        a.out_weight = store.load_tensor(source, p + ".self_attn.o_proj.weight",
            assets::TensorStorageType::Native, {c.hidden_size, c.hidden_size});
        layer.mlp.gate_proj = binding::linear_from_source(store, source, p + ".mlp.gate_proj",
            assets::TensorStorageType::Native, c.intermediate_size, c.hidden_size, false);
        layer.mlp.up_proj = binding::linear_from_source(store, source, p + ".mlp.up_proj",
            assets::TensorStorageType::Native, c.intermediate_size, c.hidden_size, false);
        layer.mlp.down_proj = binding::linear_from_source(store, source, p + ".mlp.down_proj",
            assets::TensorStorageType::Native, c.hidden_size, c.intermediate_size, false);
        weights.stack.layers.push_back(std::move(layer));
    }
    weights.final_norm = binding::norm_weight_from_source(store, source, prefix + ".norm", c.hidden_size);
}

void write_audio_mask(const core::TensorValue & positions, const core::TensorValue & mask,
    int64_t frames, int64_t sliding_window) {
    std::vector<int32_t> ids(frames);
    std::iota(ids.begin(), ids.end(), 0);
    ggml_backend_tensor_set(positions.tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
    std::vector<float> values(frames * frames);
    for (int64_t row = 0; row < frames; ++row) {
        for (int64_t col = 0; col < frames; ++col) {
            values[row * frames + col] = col > row || (sliding_window > 0 && col <= row - sliding_window)
                ? -std::numeric_limits<float>::infinity() : 0.0f;
        }
    }
    core::write_tensor_f16(mask, values);
}

}  // namespace

struct MingAudioEncoder::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source, const MingAudioCodecConfig & config,
        core::ExecutionContext & execution, core::AttentionPreference attention)
        : execution_(execution), store_(execution.backend(), execution.backend_type(),
              "ming3_tts.vae.encoder.weights", 4 * 1024 * 1024) {
        namespace binding = modules::binding;
        const auto & enc = config.encoder;
        const auto & backbone = enc.backbone;
        input_dim_ = enc.input_dim;
        hop_ = enc.hop_size;
        latent_dim_ = enc.latent_dim;
        patch_ = config.patch_size;
        sliding_window_ = enc.sliding_window;
        bf16_posterior_ = enc.bf16_posterior;
        if (input_dim_ <= 0 || hop_ <= 0 || patch_ <= 0 || latent_dim_ <= 0) {
            throw std::runtime_error("Ming audio encoder requires positive frame and patch dimensions");
        }
        load_audio_backbone(store_, *source, backbone, config.encoder_prefix + ".encoder", execution, attention,
            true, encoder_config_, encoder_weights_);
        // The published VAE constructs its patch aggregator with four Qwen layers.
        load_audio_backbone(store_, *source, backbone, config.encoder_prefix + ".aggregator", execution, attention,
            true, aggregator_config_, aggregator_weights_, 4);
        hidden_ = encoder_config_.stack.hidden_size;
        fc1_ = binding::linear_from_source(store_, *source, config.encoder_prefix + ".fc1",
            assets::TensorStorageType::Native, hidden_, input_dim_, false);
        fc2_ = binding::linear_from_source(store_, *source, config.encoder_prefix + ".fc2",
            assets::TensorStorageType::Native, hidden_, hidden_, true);
        fc3_ = binding::linear_from_source(store_, *source, config.encoder_prefix + ".fc3",
            assets::TensorStorageType::Native, 2 * latent_dim_, hidden_, true);
        const auto cls = source->require_f32(config.encoder_prefix + ".cls_embed", {1, 1, hidden_});
        cls_ = store_.make_f32(core::TensorShape::from_dims({1, 1, hidden_}), cls);
        store_.upload();
    }

    ~Impl() { release_graph(); }

    int64_t latent_frames(size_t samples) const {
        const auto frames = (static_cast<int64_t>(samples) + hop_ - 1) / hop_;
        return (frames + patch_ - 1) / patch_;
    }

    void release_graph() {
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        graph_ = nullptr;
        host_plan_.reset();
        if (allocator_) ggml_gallocr_free(allocator_);
        allocator_ = nullptr;
        if (context_) ggml_free(context_);
        context_ = nullptr;
    }

    void prepare(int64_t frames) {
        if (graph_ && frames_ == frames) return;
        release_graph();
        const int64_t patches = (frames + patch_ - 1) / patch_;
        const int64_t aggregator_frames = patches * (patch_ + 1);
        context_ = ggml_init({8 * 1024 * 1024, nullptr, true});
        core::ModuleBuildContext ctx{context_, "ming3_tts.vae.encoder", execution_.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, input_dim_}));
        ggml_set_input(input_.tensor);
        const modules::Bf16RoundModule round;
        auto x = round.build(ctx, input_);
        x = modules::LinearModule({input_dim_, hidden_, false}).build(ctx, x, fc1_);
        x = round.build(ctx, x);
        x = modules::LinearModule({hidden_, hidden_, true}).build(ctx, x, fc2_);
        x = round.build(ctx, x);
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames}));
        mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, frames, frames}));
        aggregator_positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({aggregator_frames}));
        aggregator_mask_ = core::make_tensor(ctx, GGML_TYPE_F16,
            core::TensorShape::from_dims({1, 1, aggregator_frames, aggregator_frames}));
        for (auto * tensor : {positions_.tensor, mask_.tensor, aggregator_positions_.tensor, aggregator_mask_.tensor}) {
            ggml_set_input(tensor);
            ggml_set_output(tensor);
        }
        x = modules::DecoderHiddenModule(encoder_config_).build(ctx, x, positions_, encoder_weights_, std::nullopt, mask_).hidden;
        x = round.build(ctx, x);
        if (frames != patches * patch_) x = modules::PaddingModule({patches * patch_}).build(ctx, x);
        x = modules::ReshapeModule({core::TensorShape::from_dims({patches, patch_, hidden_})}).build(ctx, x);
        const auto cls = modules::RepeatModule({core::TensorShape::from_dims({patches, 1, hidden_})}).build(ctx, cls_);
        x = modules::ConcatModule({1}).build(ctx, x, cls);
        x = modules::ReshapeModule({core::TensorShape::from_dims({1, aggregator_frames, hidden_})}).build(ctx, x);
        x = modules::DecoderHiddenModule(aggregator_config_).build(ctx, x, aggregator_positions_,
            aggregator_weights_, std::nullopt, aggregator_mask_).hidden;
        x = round.build(ctx, x);
        x = modules::ReshapeModule({core::TensorShape::from_dims({patches, patch_ + 1, hidden_})}).build(ctx, x);
        x = modules::SliceModule({1, patch_, 1}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = modules::ReshapeModule({core::TensorShape::from_dims({1, patches, hidden_})}).build(ctx, x);
        x = modules::LinearModule({hidden_, 2 * latent_dim_, false})
            .build(ctx, x, {fc3_.weight, std::nullopt});
        x = modules::AddModule(false, true).build(ctx, round.build(ctx, x), *fc3_.bias);
        output_ = round.build(ctx, x);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!ggml_gallocr_reserve(allocator_, graph_) || !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error("Ming audio encoder graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph_, host_plan_);
        write_audio_mask(positions_, mask_, frames, sliding_window_);
        write_audio_mask(aggregator_positions_, aggregator_mask_, aggregator_frames, sliding_window_);
        frames_ = frames;
    }

    std::vector<float> encode(const std::vector<float> & waveform, const std::vector<float> & noise) {
        const int64_t patches = latent_frames(waveform.size());
        if (waveform.empty() || noise.size() != static_cast<size_t>(patches * latent_dim_)) {
            throw std::runtime_error("Ming reference encoder requires nonempty audio and channel-major posterior noise");
        }
        const int64_t frames = (static_cast<int64_t>(waveform.size()) + hop_ - 1) / hop_;
        prepare(frames);
        std::vector<float> framed(frames * input_dim_, 0.0f);
        for (int64_t frame = 0; frame < frames; ++frame) {
            const size_t offset = static_cast<size_t>(frame * hop_);
            const size_t count = std::min(static_cast<size_t>(input_dim_), waveform.size() - offset);
            std::copy_n(waveform.begin() + offset, count, framed.begin() + frame * input_dim_);
        }
        core::write_tensor_f32(input_, framed);
        if (core::compute_graph(execution_, graph_, host_plan_, "ming3_tts.vae.encoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming audio encoder graph execution failed");
        }
        const auto params = core::read_tensor_f32(output_.tensor);
        std::vector<float> scale(patches * latent_dim_);
        for (int64_t t = 0; t < patches; ++t) {
            for (int64_t d = 0; d < latent_dim_; ++d) {
                const float x = params[t * 2 * latent_dim_ + latent_dim_ + d];
                scale[t * latent_dim_ + d] = x > 20.0f ? x : std::log1p(std::exp(x));
            }
        }
        if (bf16_posterior_) core::round_f32_to_bf16_in_place(scale);
        for (auto & value : scale) value += 1e-4f;
        if (bf16_posterior_) core::round_f32_to_bf16_in_place(scale);
        for (int64_t t = 0; t < patches; ++t) {
            for (int64_t d = 0; d < latent_dim_; ++d) scale[t * latent_dim_ + d] *= noise[d * patches + t];
        }
        if (bf16_posterior_) core::round_f32_to_bf16_in_place(scale);
        for (int64_t t = 0; t < patches; ++t) {
            for (int64_t d = 0; d < latent_dim_; ++d) scale[t * latent_dim_ + d] += params[t * 2 * latent_dim_ + d];
        }
        if (bf16_posterior_) core::round_f32_to_bf16_in_place(scale);
        return scale;
    }

    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    modules::DecoderHiddenConfig encoder_config_, aggregator_config_;
    modules::DecoderHiddenWeights encoder_weights_, aggregator_weights_;
    modules::LinearWeights fc1_, fc2_, fc3_;
    core::TensorValue cls_, input_, positions_, mask_, aggregator_positions_, aggregator_mask_, output_;
    int64_t hidden_ = 0, input_dim_ = 0, hop_ = 0, latent_dim_ = 0, patch_ = 0, sliding_window_ = 0, frames_ = 0;
    bool bf16_posterior_ = true;
    ggml_context * context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    core::HostGraphPlan host_plan_;
};

struct MingAudioDecoder::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source,
         const MingAudioCodecConfig & config, core::ExecutionContext & execution, core::AttentionPreference attention)
        : execution_(execution), store_(execution.backend(), execution.backend_type(),
              "ming3_tts.vae.decoder.weights", 4 * 1024 * 1024) {
        namespace binding = modules::binding;
        const auto & dec = config.decoder;
        const auto & backbone = dec.backbone;
        latent_dim_ = dec.latent_dim;
        hop_ = dec.output_dim;
        patch_ = config.patch_size;
        fft_size_ = hop_ * 4;
        load_audio_backbone(store_, *source, backbone, config.decoder_prefix + ".decoder", execution, attention,
            false, decoder_config_, weights_);
        const auto & c = decoder_config_.stack;
        sliding_window_ = dec.sliding_window;
        input_projection_ = binding::linear_from_source(store_, *source, config.decoder_prefix + ".fc1",
            assets::TensorStorageType::Native, c.hidden_size, latent_dim_, true);
        output_projection_ = binding::linear_from_source(store_, *source, config.decoder_prefix + ".head.out",
            assets::TensorStorageType::Native, fft_size_ + 2, c.hidden_size, true);
        window_ = source->require_f32(config.decoder_prefix + ".head.istft.window", {fft_size_});
        store_.upload();
    }

    ~Impl() { release_graph(); }

    void release_graph() {
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        graph_ = nullptr;
        host_plan_.reset();
        if (allocator_) ggml_gallocr_free(allocator_);
        allocator_ = nullptr;
        if (context_) ggml_free(context_);
        context_ = nullptr;
    }

    void prepare(int64_t patches) {
        if (graph_ && patches_ == patches) return;
        release_graph();
        const int64_t frames = patches * patch_;
        context_ = ggml_init({8 * 1024 * 1024, nullptr, true});
        core::ModuleBuildContext ctx{context_, "ming3_tts.vae.decoder", execution_.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, patches, latent_dim_}));
        ggml_set_input(input_.tensor);
        auto x = modules::LinearModule({latent_dim_, decoder_config_.stack.hidden_size, true})
            .build(ctx, input_, input_projection_);
        x = modules::Bf16RoundModule{}.build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = modules::Interpolate1dModule({frames, modules::Interpolate1dMode::Linear}).build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames}));
        mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, frames, frames}));
        ggml_set_input(positions_.tensor);
        ggml_set_input(mask_.tensor);
        ggml_set_output(positions_.tensor);
        ggml_set_output(mask_.tensor);
        x = modules::DecoderHiddenModule(decoder_config_).build(ctx, x, positions_, weights_, std::nullopt, mask_).hidden;
        x = modules::LinearModule({decoder_config_.stack.hidden_size, fft_size_ + 2, false})
            .build(ctx, x, {output_projection_.weight, std::nullopt});
        x = modules::AddModule(false, true).build(ctx, modules::Bf16RoundModule{}.build(ctx, x), *output_projection_.bias);
        output_ = modules::Bf16RoundModule{}.build(ctx, x);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!ggml_gallocr_reserve(allocator_, graph_) || !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error("Ming audio decoder graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph_, host_plan_);
        write_audio_mask(positions_, mask_, frames, sliding_window_);
        istft_ = std::make_unique<audio::HostLogMagnitudePhaseISTFT>(audio::HostLogMagnitudePhaseISTFTConfig{
            frames, fft_size_, hop_, fft_size_ + 2, static_cast<size_t>(execution_.config().threads)});
        patches_ = patches;
    }

    std::vector<float> decode(const std::vector<float> & latent) {
        if (latent.empty() || latent.size() % latent_dim_ != 0) {
            throw std::runtime_error("Ming audio decoder requires complete latent frames");
        }
        prepare(static_cast<int64_t>(latent.size() / latent_dim_));
        core::write_tensor_f32(input_, latent);
        if (core::compute_graph(execution_, graph_, host_plan_, "ming3_tts.vae.decoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming audio decoder graph execution failed");
        }
        const auto spectrum = core::read_tensor_f32(output_.tensor);
        return istft_->compute(spectrum, window_).audio;
    }

    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    modules::DecoderHiddenConfig decoder_config_;
    modules::DecoderHiddenWeights weights_;
    modules::LinearWeights input_projection_, output_projection_;
    std::unique_ptr<audio::HostLogMagnitudePhaseISTFT> istft_;
    std::vector<float> window_;
    int64_t latent_dim_ = 0, hop_ = 0, fft_size_ = 0, patch_ = 0, sliding_window_ = 0, patches_ = 0;
    ggml_context * context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    core::HostGraphPlan host_plan_;
    core::TensorValue input_, positions_, mask_, output_;
};

MingAudioEncoder::MingAudioEncoder(std::shared_ptr<const assets::TensorSource> source,
    const MingAudioCodecConfig & config, core::ExecutionContext & execution, core::AttentionPreference attention)
    : impl_(std::make_unique<Impl>(std::move(source), config, execution, attention)) {}
MingAudioEncoder::~MingAudioEncoder() = default;
int64_t MingAudioEncoder::latent_frames(size_t samples) const { return impl_->latent_frames(samples); }
std::vector<float> MingAudioEncoder::encode(const std::vector<float> & waveform, const std::vector<float> & noise) {
    return impl_->encode(waveform, noise);
}

MingAudioDecoder::MingAudioDecoder(std::shared_ptr<const assets::TensorSource> source,
    const MingAudioCodecConfig & config, core::ExecutionContext & execution, core::AttentionPreference attention)
    : impl_(std::make_unique<Impl>(std::move(source), config, execution, attention)) {}
MingAudioDecoder::~MingAudioDecoder() = default;
std::vector<float> MingAudioDecoder::decode(const std::vector<float> & latent) { return impl_->decode(latent); }

}  // namespace engine::modules
