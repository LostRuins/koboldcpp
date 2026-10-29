#pragma once

#include "engine/framework/codecs/fsq_audio_codec_runtime.h"
#include "engine/models/neutts/assets.h"

namespace engine::models::neutts {

using NeuTTSCodecHead = engine::codecs::FsqAudioCodecHead;
using NeuTTSNeuCodecDecoderRuntime = engine::codecs::FsqAudioCodecDecoderRuntime;

engine::codecs::FsqAudioCodecConfig make_neutts_fsq_audio_codec_config(
    const NeuTTSCodecConfig & config);

}  // namespace engine::models::neutts
