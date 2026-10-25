#include "engine/community_models/kitten_tts2/decoder.h"
#include "engine/framework/codecs/s3gen_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/chatterbox/components.h"

#include <cmath>
#include <stdexcept>

namespace engine::community_models::kitten_tts2 {
namespace s3 = engine::codecs::s3gen;
struct WaveformDecoder::Impl {
    s3::S3GenRuntime runtime;
    Impl(std::shared_ptr<const assets::TensorSource> source, const core::ExecutionContext & context)
        : runtime(std::move(source), context, [] {
              s3::S3GenConfig config;
              config.decoder.variant = s3::S3FlowVariant::MeanFlow;
              return config;
          }(), {assets::TensorStorageType::F32, context.backend_type() == core::BackendType::Cpu}) {}
};
WaveformDecoder::WaveformDecoder(std::shared_ptr<const assets::TensorSource> source,
    const core::ExecutionContext & execution) : impl_(std::make_unique<Impl>(std::move(source), execution)) {}
WaveformDecoder::~WaveformDecoder() = default;

std::vector<float> WaveformDecoder::decode(const std::vector<int32_t> & codes,
    const std::vector<int64_t> & prompt_tokens, const std::vector<float> & prompt_mel,
    const std::vector<float> & embedding, uint64_t seed) {
    if (codes.empty()) throw std::runtime_error("Kitten TTS 2 generated no speech tokens");
    if (prompt_tokens.empty() || prompt_mel.empty() || prompt_mel.size() % 80 || embedding.size() != 192)
        throw std::runtime_error("invalid Kitten TTS 2 decoder conditioning");
    s3::S3GenConditioning ref;
    ref.prompt_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
    ref.prompt_token_count = ref.prompt_tokens.size();
    ref.prompt_feat = prompt_mel;
    ref.prompt_feat_frames = prompt_mel.size() / 80;
    ref.prompt_feat_dims = 80;
    ref.embedding = embedding;
    ref.embedding_size = 192;
    auto tokens = codes;
    tokens.insert(tokens.end(), 3, 4299); // Upstream S3 lookahead preserves the final syllable.
    s3::S3GenTimingBreakdown timing;
    auto result = impl_->runtime.synthesize(
        ref, tokens, tokens.size(), 2, 0.0f, false, {}, seed, seed, &timing);
    debug::timing_log_scalar("kitten_tts2.token2mel_encoder_ms", timing.token2mel_encoder_ms);
    debug::timing_log_scalar("kitten_tts2.token2mel_cfm_ms", timing.token2mel_cfm_ms);
    debug::timing_log_scalar("kitten_tts2.vocoder_ms", timing.vocoder_ms);
    models::chatterbox::apply_s3_trim_fade(result.waveform, 24000);
    if (result.waveform.empty()) throw std::runtime_error("Kitten decoder returned empty audio");
    for (float value : result.waveform)
        if (!std::isfinite(value)) throw std::runtime_error("Kitten decoder returned non-finite audio");
    return result.waveform;
}
}
