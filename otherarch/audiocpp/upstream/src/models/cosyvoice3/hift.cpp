#include "engine/models/cosyvoice3/hift.h"

#include "engine/framework/modules/vocoders/hift_vocoder.h"

#include <stdexcept>
#include <utility>

namespace engine::models::cosyvoice3 {
namespace {

engine::modules::HiftVocoderConfig make_hift_config(engine::assets::TensorStorageType storage_type) {
    engine::modules::HiftVocoderConfig config;
    config.in_channels = 80;
    config.base_channels = 512;
    config.nb_harmonics = 8;
    config.sampling_rate = 24000;
    config.nsf_alpha = 0.1F;
    config.nsf_sigma = 0.003F;
    config.nsf_voiced_threshold = 10.0F;
    config.upsample_rates = {8, 5, 3};
    config.upsample_kernel_sizes = {16, 11, 7};
    config.istft_n_fft = 16;
    config.istft_hop = 4;
    config.resblock_kernel_sizes = {3, 7, 11};
    config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.source_resblock_kernel_sizes = {7, 7, 11};
    config.source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.lrelu_slope = 0.1F;
    config.audio_limit = 0.99F;
    config.conv_pre_kernel_size = 5;
    config.causal_convolutions = true;
    config.f0_num_class = 1;
    config.f0_in_channels = 80;
    config.f0_cond_channels = 512;
    config.f0_condnet_kernel_sizes = {4, 3, 3, 3, 3};
    config.weight_storage_type = storage_type;
    config.weight_layout = engine::modules::HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
    config.upsample_mode = engine::modules::HiftVocoderUpsampleMode::CausalConv1dNearest;
    config.source_mode = engine::modules::HiftVocoderSourceMode::CausalSineGen2;
    return config;
}

}  // namespace

class CosyVoice3HiftRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType storage_type)
        : assets_(std::move(assets)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("CosyVoice3 HiFT runtime requires assets");
        }
        component_ = engine::modules::HiftVocoderComponent::load_from_tensor_source(
            assets_->hift_weights,
            execution.config(),
            make_hift_config(storage_type));
    }

    engine::runtime::AudioBuffer synthesize(
        const std::vector<float> & mel,
        int64_t frames,
        uint64_t seed) {
        if (frames <= 0 || static_cast<int64_t>(mel.size()) != frames * 80) {
            throw std::runtime_error("CosyVoice3 HiFT mel shape mismatch");
        }
        std::vector<float> channel_major(mel.size());
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t channel = 0; channel < 80; ++channel) {
                channel_major[static_cast<size_t>(channel * frames + frame)] =
                    mel[static_cast<size_t>(frame * 80 + channel)];
            }
        }
        auto out = component_.synthesize(channel_major, frames, seed, 0);
        engine::runtime::AudioBuffer audio;
        audio.sample_rate = static_cast<int>(out.sample_rate);
        audio.channels = 1;
        audio.samples = std::move(out.waveform);
        return audio;
    }

    void release_graphs() {
        component_.release_runtime_cache();
    }

private:
    std::shared_ptr<const CosyVoice3Assets> assets_;
    engine::modules::HiftVocoderComponent component_;
};

CosyVoice3HiftRuntime::CosyVoice3HiftRuntime(
    std::shared_ptr<const CosyVoice3Assets> assets,
    engine::core::ExecutionContext & execution,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage_type)) {}

CosyVoice3HiftRuntime::~CosyVoice3HiftRuntime() = default;

engine::runtime::AudioBuffer CosyVoice3HiftRuntime::synthesize(
    const std::vector<float> & mel,
    int64_t frames,
    uint64_t seed) {
    return impl_->synthesize(mel, frames, seed);
}

void CosyVoice3HiftRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::cosyvoice3
