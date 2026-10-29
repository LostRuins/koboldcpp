#include "engine/models/neutts/codec.h"

namespace engine::models::neutts {

engine::codecs::FsqAudioCodecConfig make_neutts_fsq_audio_codec_config(
    const NeuTTSCodecConfig & config) {
    engine::codecs::FsqAudioCodecConfig out;
    out.sample_rate = config.sample_rate;
    out.output_sample_rate = config.output_sample_rate;
    out.hop_length = config.hop_length;
    out.hidden_size = config.hidden_size;
    out.intermediate_size = config.intermediate_size;
    out.layers = config.layers;
    out.attention_heads = config.attention_heads;
    out.kv_heads = config.kv_heads;
    out.head_dim = config.head_dim;
    out.quantization_dim = config.quantization_dim;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.quantization_levels = config.quantization_levels;
    out.trace_name = "neutts.codec";
    return out;
}

}  // namespace engine::models::neutts
