#include "engine/community_models/chatterbox_turbo/s3gen_turbo.h"

#include <stdexcept>

namespace engine::community_models::chatterbox_turbo {

std::shared_ptr<ChatterboxTurboS3Gen> ChatterboxTurboS3Gen::load(
    std::shared_ptr<const engine::assets::TensorSource> s3gen_source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    auto out = std::make_shared<ChatterboxTurboS3Gen>();
    engine::codecs::s3gen::S3GenConfig config;
    config.decoder.variant = engine::codecs::s3gen::S3FlowVariant::MeanFlow;
    out->runtime_ = std::make_unique<engine::codecs::s3gen::S3GenRuntime>(
        std::move(s3gen_source), execution_context, config,
        engine::codecs::s3gen::S3GenRuntimeOptions{weight_storage_type},
        engine::codecs::s3gen::S3GenWeightBinding{"flow.", "v.", engine::modules::HiftVocoderWeightLayout::Canonical});
    return out;
}

engine::codecs::s3gen::S3GenInferenceOutputs ChatterboxTurboS3Gen::synthesize(
    const engine::codecs::s3gen::S3GenConditioning & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    uint64_t flow_seed,
    uint64_t vocoder_seed) const {
    // n_cfm_timesteps=2 matches tts_turbo.py's ChatterboxTurboTTS.generate default; cfg_rate and
    // cosine_schedule are unused on the meanflow path.
    const auto mel = runtime_->token_to_mel(
        ref_dict,
        speech_tokens,
        static_cast<int64_t>(speech_tokens.size()),
        /*num_steps=*/2,
        /*cfg_rate=*/0.0f,
        /*cosine_schedule=*/false,
        /*full_noise=*/{},
        flow_seed,
        /*timing=*/nullptr);

    const auto voc = runtime_->decode_waveform(mel.mel, mel.frames, vocoder_seed);

    engine::codecs::s3gen::S3GenInferenceOutputs outputs;
    outputs.waveform = voc.waveform;
    outputs.samples = voc.samples;
    outputs.mel = mel.mel;
    outputs.mel_channels = mel.channels;
    outputs.mel_frames = mel.frames;
    return outputs;
}

}  // namespace engine::community_models::chatterbox_turbo
