#include "engine/models/pocket_tts/audio_decoder.h"

#include "engine/models/pocket_tts/backend_weights.h"
#include "engine/models/pocket_tts/assets.h"

#include <stdexcept>
#include <utility>

namespace engine::models::pocket_tts {
namespace {

std::vector<float> denormalize_latents(
    const std::vector<float> & latents,
    const std::vector<float> & mean,
    const std::vector<float> & std) {
    std::vector<float> output = latents;
    const size_t dim = mean.size();
    for (size_t i = 0; i < output.size(); ++i) {
        const size_t d = i % dim;
        output[i] = output[i] * std[d] + mean[d];
    }
    return output;
}

}  // namespace

AudioDecoder::AudioDecoder(MimiDecoderConfig config) : decoder_(std::move(config)) {}

std::vector<float> AudioDecoder::decode(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & normalized_latents,
    int64_t steps,
    size_t conv_graph_context_bytes,
    size_t transformer_graph_context_bytes,
    size_t tail_graph_context_bytes,
    int64_t full_chunk_frames,
    int64_t stage2_chunk_frames,
    bool use_full_sequence_path) const {
    if (steps <= 0) {
        throw std::runtime_error("PocketTTS audio decoder requires positive step count");
    }
    if (normalized_latents.size() != static_cast<size_t>(steps) * static_cast<size_t>(decoder_.config().latent_size)) {
        throw std::runtime_error("PocketTTS audio decoder latent count does not match steps * latent_size");
    }
    const auto & emb_mean = weights.host.emb_mean;
    const auto & emb_std = weights.host.emb_std;
    if (emb_mean.size() != emb_std.size() || emb_mean.size() != static_cast<size_t>(decoder_.config().latent_size)) {
        throw std::runtime_error("PocketTTS latent normalization stats must match Mimi latent_size");
    }
    auto denormalized = denormalize_latents(normalized_latents, emb_mean, emb_std);
    return decoder_.decode(
        backend,
        threads,
        manifest,
        weights,
        denormalized,
        steps,
        conv_graph_context_bytes,
        transformer_graph_context_bytes,
        tail_graph_context_bytes,
        full_chunk_frames,
        stage2_chunk_frames,
        use_full_sequence_path);
}

void AudioDecoder::reset_streaming_state() const {
    decoder_.reset_streaming_state();
}

std::vector<float> AudioDecoder::decode_streaming_step(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & normalized_latent,
    size_t conv_graph_context_bytes,
    size_t transformer_graph_context_bytes,
    size_t tail_graph_context_bytes) const {
    if (normalized_latent.size() != static_cast<size_t>(decoder_.config().latent_size)) {
        throw std::runtime_error("PocketTTS streaming audio decoder expects one latent step");
    }
    const auto & emb_mean = weights.host.emb_mean;
    const auto & emb_std = weights.host.emb_std;
    if (emb_mean.size() != emb_std.size() || emb_mean.size() != static_cast<size_t>(decoder_.config().latent_size)) {
        throw std::runtime_error("PocketTTS latent normalization stats must match Mimi latent_size");
    }
    auto denormalized = denormalize_latents(normalized_latent, emb_mean, emb_std);
    return decoder_.decode_streaming_step(
        backend,
        threads,
        manifest,
        weights,
        denormalized,
        conv_graph_context_bytes,
        transformer_graph_context_bytes,
        tail_graph_context_bytes);
}

void AudioDecoder::clear_runtime_cache() const noexcept {
    decoder_.clear_runtime_cache();
}

}  // namespace engine::models::pocket_tts
