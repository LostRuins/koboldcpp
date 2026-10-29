#include "engine/models/vibevoice/generator.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/vibevoice/diffusion_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice {
namespace {

int64_t count_speech_mask_positions(const VibeVoicePromptEncoding & encoding) {
    return static_cast<int64_t>(
        std::count(encoding.speech_input_mask.begin(), encoding.speech_input_mask.end(), 1U));
}

VibeVoiceTokenizerLatents tokenizer_latents_from_reference_state(const VibeVoiceReferenceVoiceState & state) {
    if (state.frames <= 0 || state.dim <= 0 ||
        state.acoustic_mean.size() != static_cast<size_t>(state.frames * state.dim)) {
        throw std::runtime_error("VibeVoice cached reference voice state has invalid acoustic mean shape");
    }
    VibeVoiceTokenizerLatents out;
    out.values = state.acoustic_mean;
    out.frames = state.frames;
    out.dim = state.dim;
    return out;
}

std::vector<VibeVoiceTokenizerLatents> resolve_prompt_acoustic_means(
    const std::vector<VibeVoiceSpeakerPrompt> & speakers,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer) {
    std::vector<VibeVoiceTokenizerLatents> means(speakers.size());
    std::vector<runtime::AudioBuffer> missing_audio;
    std::vector<size_t> missing_indices;
    missing_audio.reserve(speakers.size());
    missing_indices.reserve(speakers.size());
    for (size_t index = 0; index < speakers.size(); ++index) {
        const auto & speaker = speakers[index];
        if (speaker.reference_state.has_value()) {
            means[index] = tokenizer_latents_from_reference_state(*speaker.reference_state);
            continue;
        }
        missing_indices.push_back(index);
        missing_audio.push_back(speaker.audio);
    }
    if (!missing_audio.empty()) {
        auto encoded = audio_tokenizer.encode_acoustic_batch(missing_audio);
        if (encoded.size() != missing_indices.size()) {
            throw std::runtime_error("VibeVoice acoustic tokenizer returned unexpected prompt batch size");
        }
        for (size_t index = 0; index < encoded.size(); ++index) {
            means[missing_indices[index]] = std::move(encoded[index]);
        }
    }
    return means;
}

std::vector<float> build_selected_prompt_features(
    const VibeVoiceConnectorOutput & connected,
    const std::vector<int64_t> & token_counts) {
    if (connected.frames <= 0 || connected.hidden_size <= 0) {
        throw std::runtime_error("VibeVoice prompt connector returned invalid shape");
    }
    const int64_t speakers = static_cast<int64_t>(token_counts.size());
    if (speakers <= 0) {
        return {};
    }
    if (connected.frames % speakers != 0) {
        throw std::runtime_error("VibeVoice prompt connector frame count is not divisible by speakers");
    }
    const int64_t padded_frames = connected.frames / speakers;
    std::vector<float> selected;
    int64_t selected_frames = 0;
    for (const int64_t count : token_counts) {
        if (count < 0 || count > padded_frames) {
            throw std::runtime_error("VibeVoice prompt speech mask exceeds acoustic connector frames");
        }
        selected_frames += count;
    }
    selected.reserve(static_cast<size_t>(selected_frames * connected.hidden_size));
    for (int64_t speaker = 0; speaker < speakers; ++speaker) {
        const int64_t count = token_counts[static_cast<size_t>(speaker)];
        for (int64_t frame = 0; frame < count; ++frame) {
            const size_t base = static_cast<size_t>(
                (speaker * padded_frames + frame) * connected.hidden_size);
            selected.insert(
                selected.end(),
                connected.values.begin() + static_cast<std::ptrdiff_t>(base),
                connected.values.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(connected.hidden_size)));
        }
    }
    return selected;
}

void splice_speech_embeddings(
    std::vector<float> & embeddings,
    const VibeVoicePromptEncoding & encoding,
    const std::vector<float> & speech_embeddings,
    int64_t hidden_size) {
    if (hidden_size <= 0) {
        throw std::runtime_error("VibeVoice prompt splice requires positive hidden size");
    }
    const int64_t speech_positions = count_speech_mask_positions(encoding);
    if (static_cast<int64_t>(speech_embeddings.size()) != speech_positions * hidden_size) {
        throw std::runtime_error("VibeVoice prompt speech embedding payload size mismatch");
    }
    if (static_cast<int64_t>(embeddings.size()) != static_cast<int64_t>(encoding.input_ids.size()) * hidden_size) {
        throw std::runtime_error("VibeVoice prompt token embedding payload size mismatch");
    }
    int64_t speech_frame = 0;
    for (size_t token = 0; token < encoding.speech_input_mask.size(); ++token) {
        if (encoding.speech_input_mask[token] == 0U) {
            continue;
        }
        const size_t dst = token * static_cast<size_t>(hidden_size);
        const size_t src = static_cast<size_t>(speech_frame * hidden_size);
        std::copy(
            speech_embeddings.begin() + static_cast<std::ptrdiff_t>(src),
            speech_embeddings.begin() + static_cast<std::ptrdiff_t>(src + static_cast<size_t>(hidden_size)),
            embeddings.begin() + static_cast<std::ptrdiff_t>(dst));
        ++speech_frame;
    }
}

std::vector<float> add_embeddings(
    const VibeVoiceConnectorOutput & lhs,
    const VibeVoiceConnectorOutput & rhs) {
    if (lhs.frames != rhs.frames || lhs.hidden_size != rhs.hidden_size || lhs.values.size() != rhs.values.size()) {
        throw std::runtime_error("VibeVoice generated embedding connector outputs have mismatched shapes");
    }
    std::vector<float> out(lhs.values.size(), 0.0F);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = lhs.values[i] + rhs.values[i];
    }
    return out;
}

std::vector<float> single_token_embedding(
    const VibeVoiceQwen2WeightsRuntime & decoder,
    int32_t token,
    int64_t hidden_size) {
    auto embedding = decoder.embed_tokens({token});
    if (embedding.steps != 1 || embedding.hidden_size != hidden_size ||
        static_cast<int64_t>(embedding.values.size()) != hidden_size) {
        throw std::runtime_error("VibeVoice single token embedding shape mismatch");
    }
    return std::move(embedding.values);
}

int64_t max_generation_steps(const VibeVoicePreparedPrompt & prompt, const VibeVoiceGenerationOptions & options, const VibeVoiceQwen2Config & config) {
    if (prompt.steps <= 0 || config.max_position_embeddings <= prompt.steps) {
        throw std::runtime_error("VibeVoice generation prompt exceeds decoder position capacity");
    }
    int64_t by_length = config.max_position_embeddings - prompt.steps;
    if (options.max_tokens > 0) {
        by_length = std::min(by_length, options.max_tokens);
    }
    const int64_t by_ratio = static_cast<int64_t>(options.max_length_times * static_cast<float>(prompt.steps));
    const int64_t steps = std::min(by_length, by_ratio);
    if (steps <= 0) {
        throw std::runtime_error("VibeVoice generation max steps must be positive");
    }
    return steps;
}

std::vector<float> load_noise_file(const std::string & path, const char * label) {
    if (path.empty()) {
        return {};
    }
    auto values = engine::io::read_f32_file(path);
    if (values.empty()) {
        throw std::runtime_error(std::string("VibeVoice ") + label + " noise file is empty");
    }
    return values;
}

std::vector<float> next_diffusion_noise(
    const std::vector<float> & noise_file_values,
    size_t & noise_file_offset,
    size_t count,
    uint32_t seed,
    uint64_t & rng_index,
    sampling::TorchRandnPrecision precision) {
    if (!noise_file_values.empty()) {
        if (noise_file_offset + count > noise_file_values.size()) {
            throw std::runtime_error("VibeVoice diffusion noise file ran out of values");
        }
        std::vector<float> out(
            noise_file_values.begin() + static_cast<std::ptrdiff_t>(noise_file_offset),
            noise_file_values.begin() + static_cast<std::ptrdiff_t>(noise_file_offset + count));
        noise_file_offset += count;
        return out;
    }
    auto out = engine::sampling::generate_torch_cuda_randn(
        count,
        seed,
        precision,
        rng_index);
    rng_index += static_cast<uint64_t>(count);
    return out;
}

void append_audio(std::vector<float> & output, const runtime::AudioBuffer & chunk) {
    if (chunk.sample_rate != 24000 || chunk.channels != 1) {
        throw std::runtime_error("VibeVoice generated chunk has unexpected audio format");
    }
    output.insert(output.end(), chunk.samples.begin(), chunk.samples.end());
}

}  // namespace

VibeVoicePreparedPrompt prepare_vibevoice_prompt(
    const VibeVoiceRequest & request,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceQwen2WeightsRuntime & decoder,
    uint64_t seed,
    uint64_t start_rng_index,
    const std::vector<float> * prompt_noise_values) {
    auto encoding = text_tokenizer.encode_prompt(request);
    auto token_embeddings = decoder.embed_tokens(encoding.input_ids);
    if (token_embeddings.steps != static_cast<int64_t>(encoding.input_ids.size())) {
        throw std::runtime_error("VibeVoice prompt token embedding step count mismatch");
    }

    uint64_t next_rng_index = start_rng_index;
    if (!request.speakers.empty()) {
        if (encoding.speech_prompt_token_counts.size() != request.speakers.size()) {
            throw std::runtime_error("VibeVoice prompt speaker token count mismatch");
        }
        const int64_t speech_positions = count_speech_mask_positions(encoding);
        int64_t expected_speech_positions = 0;
        for (const int64_t count : encoding.speech_prompt_token_counts) {
            expected_speech_positions += count;
        }
        if (speech_positions != expected_speech_positions) {
            throw std::runtime_error("VibeVoice prompt speech mask count mismatch");
        }
        auto acoustic_means = resolve_prompt_acoustic_means(request.speakers, audio_tokenizer);
        auto acoustic_sample = sample_vibevoice_acoustic_latents_gaussian(
            acoustic_means,
            audio_tokenizer.assets().config.acoustic_tokenizer.fix_std,
            seed,
            start_rng_index,
            sampling::TorchRandnPrecision::BFloat16,
            prompt_noise_values);
        next_rng_index = acoustic_sample.next_rng_index;

        std::vector<float> scaled_features;
        scaled_features.reserve(acoustic_sample.latents.front().values.size() * acoustic_sample.latents.size());
        int64_t padded_frames = acoustic_sample.latents.front().frames;
        const int64_t acoustic_dim = acoustic_sample.latents.front().dim;
        for (const auto & latent : acoustic_sample.latents) {
            if (latent.frames != padded_frames || latent.dim != acoustic_dim) {
                throw std::runtime_error("VibeVoice sampled prompt acoustic latent shape mismatch");
            }
            auto scaled = scale_vibevoice_acoustic_latents_for_connector(
                latent,
                audio_tokenizer.assets().speech_scaling_factor,
                audio_tokenizer.assets().speech_bias_factor);
            scaled_features.insert(scaled_features.end(), scaled.values.begin(), scaled.values.end());
        }
        auto connected = connector.project_acoustic(scaled_features, padded_frames * static_cast<int64_t>(request.speakers.size()), acoustic_dim);
        auto selected = build_selected_prompt_features(connected, encoding.speech_prompt_token_counts);
        splice_speech_embeddings(token_embeddings.values, encoding, selected, token_embeddings.hidden_size);
    } else if (count_speech_mask_positions(encoding) != 0) {
        throw std::runtime_error("VibeVoice prompt has speech mask positions without speaker prompts");
    }

    VibeVoicePreparedPrompt out;
    out.encoding = std::move(encoding);
    out.embeddings = std::move(token_embeddings.values);
    out.steps = token_embeddings.steps;
    out.hidden_size = token_embeddings.hidden_size;
    out.next_rng_index = next_rng_index;
    return out;
}

std::vector<VibeVoicePreparedPrompt> prepare_vibevoice_prompts_batch(
    const std::vector<VibeVoiceRequest> & requests,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceQwen2WeightsRuntime & decoder,
    uint64_t seed,
    uint64_t start_rng_index,
    const std::vector<float> * prompt_noise_values) {
    if (requests.empty()) {
        throw std::runtime_error("VibeVoice batch prompt preparation requires requests");
    }

    std::vector<VibeVoicePreparedPrompt> prompts(requests.size());
    std::vector<VibeVoiceTokenEmbeddings> token_embeddings;
    token_embeddings.reserve(requests.size());
    std::vector<VibeVoiceSpeakerPrompt> speaker_prompts;
    std::vector<size_t> speaker_offsets;
    std::vector<size_t> speaker_counts;
    speaker_offsets.reserve(requests.size());
    speaker_counts.reserve(requests.size());
    int64_t hidden_size = 0;

    for (size_t index = 0; index < requests.size(); ++index) {
        const auto & request = requests[index];
        auto & prompt = prompts[index];
        prompt.encoding = text_tokenizer.encode_prompt(request);

        auto embeddings = decoder.embed_tokens(prompt.encoding.input_ids);
        if (embeddings.steps != static_cast<int64_t>(prompt.encoding.input_ids.size())) {
            throw std::runtime_error("VibeVoice batch prompt token embedding step count mismatch");
        }
        if (hidden_size == 0) {
            hidden_size = embeddings.hidden_size;
        } else if (embeddings.hidden_size != hidden_size) {
            throw std::runtime_error("VibeVoice batch prompt hidden size mismatch");
        }
        if (!request.speakers.empty()) {
            if (prompt.encoding.speech_prompt_token_counts.size() != request.speakers.size()) {
                throw std::runtime_error("VibeVoice batch prompt speaker token count mismatch");
            }
            const int64_t speech_positions = count_speech_mask_positions(prompt.encoding);
            int64_t expected_speech_positions = 0;
            for (const int64_t count : prompt.encoding.speech_prompt_token_counts) {
                expected_speech_positions += count;
            }
            if (speech_positions != expected_speech_positions) {
                throw std::runtime_error("VibeVoice batch prompt speech mask count mismatch");
            }
        } else if (count_speech_mask_positions(prompt.encoding) != 0) {
            throw std::runtime_error("VibeVoice batch prompt has speech mask positions without speaker prompts");
        }

        speaker_offsets.push_back(speaker_prompts.size());
        speaker_counts.push_back(request.speakers.size());
        for (const auto & speaker : request.speakers) {
            speaker_prompts.push_back(speaker);
        }
        token_embeddings.push_back(std::move(embeddings));
    }

    uint64_t next_rng_index = start_rng_index;
    if (!speaker_prompts.empty()) {
        auto acoustic_means = resolve_prompt_acoustic_means(speaker_prompts, audio_tokenizer);
        auto acoustic_sample = sample_vibevoice_acoustic_latents_gaussian(
            acoustic_means,
            audio_tokenizer.assets().config.acoustic_tokenizer.fix_std,
            seed,
            start_rng_index,
            sampling::TorchRandnPrecision::BFloat16,
            prompt_noise_values);
        next_rng_index = acoustic_sample.next_rng_index;

        std::vector<float> scaled_features;
        scaled_features.reserve(acoustic_sample.latents.front().values.size() * acoustic_sample.latents.size());
        const int64_t padded_frames = acoustic_sample.latents.front().frames;
        const int64_t acoustic_dim = acoustic_sample.latents.front().dim;
        for (const auto & latent : acoustic_sample.latents) {
            if (latent.frames != padded_frames || latent.dim != acoustic_dim) {
                throw std::runtime_error("VibeVoice batch sampled prompt acoustic latent shape mismatch");
            }
            auto scaled = scale_vibevoice_acoustic_latents_for_connector(
                latent,
                audio_tokenizer.assets().speech_scaling_factor,
                audio_tokenizer.assets().speech_bias_factor);
            scaled_features.insert(scaled_features.end(), scaled.values.begin(), scaled.values.end());
        }
        auto connected = connector.project_acoustic(
            scaled_features,
            padded_frames * static_cast<int64_t>(speaker_prompts.size()),
            acoustic_dim);

        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            const size_t speaker_count = speaker_counts[request_index];
            if (speaker_count == 0) {
                continue;
            }
            const size_t speaker_offset = speaker_offsets[request_index];
            VibeVoiceConnectorOutput request_connected;
            request_connected.frames = padded_frames * static_cast<int64_t>(speaker_count);
            request_connected.hidden_size = connected.hidden_size;
            const size_t offset = speaker_offset * static_cast<size_t>(padded_frames * connected.hidden_size);
            const size_t count = static_cast<size_t>(request_connected.frames * connected.hidden_size);
            request_connected.values.assign(
                connected.values.begin() + static_cast<std::ptrdiff_t>(offset),
                connected.values.begin() + static_cast<std::ptrdiff_t>(offset + count));
            auto selected = build_selected_prompt_features(
                request_connected,
                prompts[request_index].encoding.speech_prompt_token_counts);
            splice_speech_embeddings(
                token_embeddings[request_index].values,
                prompts[request_index].encoding,
                selected,
                token_embeddings[request_index].hidden_size);
        }
    }

    for (size_t index = 0; index < prompts.size(); ++index) {
        auto & prompt = prompts[index];
        auto & embeddings = token_embeddings[index];
        prompt.embeddings = std::move(embeddings.values);
        prompt.steps = embeddings.steps;
        prompt.hidden_size = embeddings.hidden_size;
        prompt.next_rng_index = next_rng_index;
    }
    return prompts;
}

struct VibeVoiceSamplingCandidate {
    int32_t token = 0;
    float score = 0.0F;
};

int32_t select_vibevoice_constrained_token(
    const VibeVoiceDecoderLogits & logits,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceGenerationOptions & options,
    std::mt19937 & rng) {
    if (logits.vocab_size <= 0 || static_cast<int64_t>(logits.values.size()) != logits.vocab_size) {
        throw std::runtime_error("VibeVoice constrained token selection received invalid logits");
    }
    const int32_t candidates[] = {
        text_tokenizer.speech_start_id(),
        text_tokenizer.speech_end_id(),
        text_tokenizer.speech_diffusion_id(),
        text_tokenizer.eos_id(),
    };
    int32_t best_token = candidates[0];
    float best_score = -std::numeric_limits<float>::infinity();
    for (const int32_t token : candidates) {
        if (token < 0 || token >= logits.vocab_size) {
            throw std::runtime_error("VibeVoice constrained token candidate is outside logits vocabulary");
        }
        const float score = logits.values[static_cast<size_t>(token)];
        if (score > best_score) {
            best_score = score;
            best_token = token;
        }
    }
    if (!options.do_sample) {
        return best_token;
    }
    if (!(options.temperature > 0.0F) || !std::isfinite(options.temperature)) {
        throw std::runtime_error("VibeVoice sampler temperature must be finite and positive");
    }
    if (options.top_k < 0) {
        throw std::runtime_error("VibeVoice sampler top_k must be non-negative");
    }
    if (!(options.top_p > 0.0F && options.top_p <= 1.0F) || !std::isfinite(options.top_p)) {
        throw std::runtime_error("VibeVoice sampler top_p must be finite and in (0, 1]");
    }
    auto candidate_order = [](const VibeVoiceSamplingCandidate & lhs, const VibeVoiceSamplingCandidate & rhs) {
        if (lhs.score == rhs.score) {
            return lhs.token < rhs.token;
        }
        return lhs.score > rhs.score;
    };
    std::vector<VibeVoiceSamplingCandidate> scores;
    scores.reserve(sizeof(candidates) / sizeof(candidates[0]));
    for (const int32_t token : candidates) {
        scores.push_back({token, logits.values[static_cast<size_t>(token)] / options.temperature});
    }
    if (options.top_k > 0 && static_cast<size_t>(options.top_k) < scores.size()) {
        const auto top_end = scores.begin() + static_cast<std::ptrdiff_t>(options.top_k);
        std::nth_element(scores.begin(), top_end, scores.end(), candidate_order);
        std::sort(scores.begin(), top_end, candidate_order);
        scores.erase(top_end, scores.end());
    } else {
        std::sort(scores.begin(), scores.end(), candidate_order);
    }
    const float max_score = scores.front().score;
    std::vector<double> weights(scores.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        weights[i] = std::exp(static_cast<double>(scores[i].score - max_score));
        total += weights[i];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("VibeVoice sampler produced invalid probability mass");
    }
    if (options.top_p < 1.0F) {
        double cumulative = 0.0;
        size_t keep = weights.size();
        for (size_t i = 0; i < weights.size(); ++i) {
            cumulative += weights[i] / total;
            if (cumulative >= static_cast<double>(options.top_p)) {
                keep = i + 1;
                break;
            }
        }
        scores.resize(keep);
        weights.resize(keep);
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return scores[distribution(rng)].token;
}

VibeVoiceResult generate_vibevoice(
    const VibeVoiceRequest & request,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceQwen2WeightsRuntime & decoder,
    const VibeVoiceDiffusionHeadWeightsRuntime & diffusion_head) {
    VibeVoiceQwen2CachedState positive_cache;
    VibeVoiceQwen2CachedState negative_cache;
    return generate_vibevoice(
        request,
        text_tokenizer,
        audio_tokenizer,
        connector,
        decoder,
        diffusion_head,
        positive_cache,
        negative_cache);
}

VibeVoiceResult generate_vibevoice(
    const VibeVoiceRequest & request,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceQwen2WeightsRuntime & decoder,
    const VibeVoiceDiffusionHeadWeightsRuntime & diffusion_head,
    VibeVoiceQwen2CachedState & positive_cache,
    VibeVoiceQwen2CachedState & negative_cache,
    const std::function<void(const runtime::AudioBuffer &)> & on_audio) {
    const auto & config = decoder.assets().config.decoder;
    const auto prompt_noise_values = load_noise_file(request.generation.prompt_noise_file, "prompt");
    auto prompt = prepare_vibevoice_prompt(
        request,
        text_tokenizer,
        audio_tokenizer,
        connector,
        decoder,
        request.generation.seed,
        0,
        prompt_noise_values.empty() ? nullptr : &prompt_noise_values);
    audio_tokenizer.release_prompt_graphs();
    const int64_t max_steps = max_generation_steps(prompt, request.generation, config);
    engine::debug::timing_log_scalar("vibevoice.generate.max_steps", max_steps);
    engine::debug::timing_log_scalar("vibevoice.generate.prompt_steps", prompt.steps);
    auto positive = decoder.prefill_embeddings(prompt.embeddings, prompt.steps);
    decoder.reset_cached_state(positive_cache, std::move(positive.state));
    auto negative_start = single_token_embedding(decoder, text_tokenizer.speech_start_id(), prompt.hidden_size);
    auto negative = decoder.prefill_embeddings(negative_start, 1);
    decoder.reset_cached_state(negative_cache, std::move(negative.state));
    decoder.release_prompt_graphs();
    const int64_t latent_size = audio_tokenizer.assets().config.diffusion_head.latent_size;
    if (latent_size != audio_tokenizer.assets().config.acoustic_vae_dim) {
        throw std::runtime_error("VibeVoice generation latent size mismatch");
    }

    const auto noise_file_values = load_noise_file(request.generation.diffusion_noise_file, "diffusion");
    size_t noise_file_offset = 0;
    uint64_t diffusion_rng_index = prompt.next_rng_index;
    std::vector<float> audio_samples;
    std::vector<int32_t> generated_tokens;
    VibeVoiceTokenizerStreamingState acoustic_streaming_state;
    VibeVoiceTokenizerStreamingState semantic_streaming_state;
    VibeVoiceDPMSolverScheduler diffusion_scheduler(diffusion_head.assets().config.diffusion_head);
    diffusion_scheduler.set_timesteps(request.generation.num_inference_steps);
    std::mt19937 sampler_rng(request.generation.seed);

    auto current = std::move(positive.result);
    for (int64_t step = 0; step < max_steps; ++step) {
        const int32_t token = select_vibevoice_constrained_token(
            current.logits,
            text_tokenizer,
            request.generation,
            sampler_rng);
        generated_tokens.push_back(token);
        if (token == text_tokenizer.eos_id()) {
            break;
        }
        if (token == text_tokenizer.speech_start_id()) {
            negative = decoder.prefill_embeddings(negative_start, 1);
            decoder.reset_cached_state(negative_cache, std::move(negative.state));
            decoder.release_prompt_graphs();
        }
        if (token == text_tokenizer.speech_end_id()) {
            acoustic_streaming_state.set_to_zero();
            semantic_streaming_state.set_to_zero();
        }

        std::vector<float> next_embedding;
        if (token == text_tokenizer.speech_diffusion_id()) {
            const size_t noise_count = static_cast<size_t>(2 * latent_size);
            auto initial_noise = next_diffusion_noise(
                noise_file_values,
                noise_file_offset,
                noise_count,
                request.generation.seed,
                diffusion_rng_index,
                sampling::TorchRandnPrecision::BFloat16);
            VibeVoiceSpeechDiffusionInput diffusion_input;
            diffusion_input.positive_condition = current.last_hidden.values;
            diffusion_input.negative_condition = negative.result.last_hidden.values;
            diffusion_input.initial_speech = std::move(initial_noise);
            diffusion_input.batch_size = 1;
            diffusion_input.hidden_size = prompt.hidden_size;
            diffusion_input.latent_size = latent_size;
            diffusion_input.inference_steps = request.generation.num_inference_steps;
            diffusion_input.guidance_scale = request.generation.guidance_scale;
            auto speech_latents = sample_vibevoice_speech_latents(diffusion_head, diffusion_scheduler, diffusion_input);
            if (speech_latents.size() != 1) {
                throw std::runtime_error("VibeVoice diffusion sampler returned unexpected batch size");
            }

            auto decoder_latents = unscale_vibevoice_acoustic_latents_for_decoder(
                speech_latents.front(),
                audio_tokenizer.assets().speech_scaling_factor,
                audio_tokenizer.assets().speech_bias_factor);
            auto chunk = audio_tokenizer.decode_acoustic_streaming(decoder_latents, acoustic_streaming_state);
            append_audio(audio_samples, chunk);
            if (on_audio) on_audio(chunk);
            auto semantic_features = audio_tokenizer.encode_semantic_streaming(chunk, semantic_streaming_state);
            auto acoustic_embedding = connector.project_acoustic(
                speech_latents.front().values,
                speech_latents.front().frames,
                speech_latents.front().dim);
            auto semantic_embedding = connector.project_semantic(
                semantic_features.values,
                semantic_features.frames,
                semantic_features.dim);
            next_embedding = add_embeddings(acoustic_embedding, semantic_embedding);
            if (static_cast<int64_t>(next_embedding.size()) != prompt.hidden_size) {
                throw std::runtime_error("VibeVoice generated diffusion embedding shape mismatch");
            }
        } else {
            next_embedding = single_token_embedding(decoder, token, prompt.hidden_size);
        }

        if (token == text_tokenizer.speech_diffusion_id()) {
            negative.result = decoder.cached_step(
                next_embedding,
                negative_cache,
                1);
        }
        current = decoder.cached_step(
            next_embedding,
            positive_cache,
            prompt.steps + step + 1);
    }

    if (audio_samples.empty()) {
        throw std::runtime_error("VibeVoice generation produced no audio");
    }

    VibeVoiceResult out;
    out.audio = runtime::AudioBuffer{24000, 1, std::move(audio_samples)};
    out.generated_tokens = std::move(generated_tokens);
    return out;
}

std::vector<VibeVoiceResult> generate_vibevoice_batch(
    const std::vector<VibeVoiceRequest> & requests,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceQwen2WeightsRuntime & decoder,
    const VibeVoiceDiffusionHeadWeightsRuntime & diffusion_head) {
    if (requests.empty()) {
        throw std::runtime_error("VibeVoice batch generation requires at least one request");
    }
    if (requests.size() == 1) {
        return {generate_vibevoice(requests.front(), text_tokenizer, audio_tokenizer, connector, decoder, diffusion_head)};
    }
    std::cerr << "VibeVoice multi-batch generation is experimental; requested batch size "
              << requests.size() << ".\n";

    const auto & first_options = requests.front().generation;
    for (const auto & request : requests) {
        const auto & options = request.generation;
        if (options.max_tokens != first_options.max_tokens ||
            options.guidance_scale != first_options.guidance_scale ||
            options.max_length_times != first_options.max_length_times ||
            options.num_inference_steps != first_options.num_inference_steps ||
            options.seed != first_options.seed ||
            options.do_sample != first_options.do_sample ||
            options.temperature != first_options.temperature ||
            options.top_k != first_options.top_k ||
            options.top_p != first_options.top_p ||
            options.prompt_noise_file != first_options.prompt_noise_file ||
            options.diffusion_noise_file != first_options.diffusion_noise_file) {
            throw std::runtime_error("VibeVoice batch generation requires identical generation options");
        }
    }

    struct SampleState {
        VibeVoicePreparedPrompt prompt;
        VibeVoiceDecoderResult current;
        VibeVoiceDecoderResult negative;
        VibeVoiceQwen2CachedState positive_cache;
        VibeVoiceQwen2CachedState negative_cache;
        VibeVoiceTokenizerStreamingState acoustic_streaming_state;
        VibeVoiceTokenizerStreamingState semantic_streaming_state;
        std::vector<float> audio_samples;
        std::vector<int32_t> generated_tokens;
        std::vector<float> next_embedding;
        int64_t max_steps = 0;
        int64_t cache_capacity = 0;
        int64_t negative_cache_required = 0;
        int32_t token = 0;
        std::mt19937 sampler_rng;
        bool finished = false;
        bool has_next_embedding = false;
    };

    const auto & config = decoder.assets().config.decoder;
    const int64_t latent_size = audio_tokenizer.assets().config.diffusion_head.latent_size;
    if (latent_size != audio_tokenizer.assets().config.acoustic_vae_dim) {
        throw std::runtime_error("VibeVoice batch generation latent size mismatch");
    }

    const auto prompt_noise_values = load_noise_file(first_options.prompt_noise_file, "prompt");
    const auto noise_file_values = load_noise_file(first_options.diffusion_noise_file, "diffusion");
    std::vector<SampleState> states;
    states.reserve(requests.size());
    int64_t hidden_size = 0;
    int64_t max_steps = 0;
    auto prompts = prepare_vibevoice_prompts_batch(
        requests,
        text_tokenizer,
        audio_tokenizer,
        connector,
        decoder,
        first_options.seed,
        0,
        prompt_noise_values.empty() ? nullptr : &prompt_noise_values);
    audio_tokenizer.release_prompt_graphs();
    uint64_t prompt_rng_index = prompts.front().next_rng_index;
    const int64_t prompt_steps = prompts.front().steps;
    std::vector<std::vector<float>> prompt_embeddings;
    prompt_embeddings.reserve(prompts.size());
    for (const auto & prompt : prompts) {
        if (prompt.steps != prompt_steps) {
            throw std::runtime_error("VibeVoice batch generation requires matching prompt lengths");
        }
        prompt_embeddings.push_back(prompt.embeddings);
    }
    auto positive_prefills = decoder.prefill_embeddings_batch(prompt_embeddings, prompt_steps);
    if (positive_prefills.size() != prompts.size()) {
        throw std::runtime_error("VibeVoice batch decoder prefill returned unexpected batch size");
    }
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        SampleState state;
        state.prompt = std::move(prompts[request_index]);
        if (hidden_size == 0) {
            hidden_size = state.prompt.hidden_size;
        } else if (state.prompt.hidden_size != hidden_size) {
            throw std::runtime_error("VibeVoice batch generation prompt hidden size mismatch");
        }
        auto positive = std::move(positive_prefills[request_index]);
        state.current = std::move(positive.result);
        decoder.reset_cached_state(state.positive_cache, std::move(positive.state));
        state.max_steps = max_generation_steps(state.prompt, first_options, config);
        state.cache_capacity = state.prompt.steps + 1;
        state.negative_cache_required = 2;
        state.sampler_rng.seed(first_options.seed + static_cast<uint32_t>(request_index));
        engine::debug::timing_log_scalar(
            "vibevoice.generate.batch.prompt_steps",
            state.prompt.steps);
        max_steps = std::max(max_steps, state.max_steps);
        states.push_back(std::move(state));
    }

    const auto negative_start = single_token_embedding(decoder, text_tokenizer.speech_start_id(), hidden_size);
    const auto negative_template = decoder.prefill_embeddings(negative_start, 1);
    decoder.release_prompt_graphs();
    for (auto & state : states) {
        state.negative = negative_template.result;
        decoder.reset_cached_state(state.negative_cache, negative_template.state);
    }

    VibeVoiceDPMSolverScheduler diffusion_scheduler(diffusion_head.assets().config.diffusion_head);
    diffusion_scheduler.set_timesteps(first_options.num_inference_steps);
    size_t noise_file_offset = 0;
    uint64_t diffusion_rng_index = prompt_rng_index;

    for (int64_t step = 0; step < max_steps; ++step) {
        bool any_active = false;
        for (const auto & state : states) {
            if (!state.finished && step < state.max_steps) {
                any_active = true;
                break;
            }
        }
        if (!any_active) {
            break;
        }

        std::vector<size_t> diffusion_indices;
        for (size_t index = 0; index < states.size(); ++index) {
            auto & state = states[index];
            state.has_next_embedding = false;
            state.next_embedding.clear();
            if (state.finished || step >= state.max_steps) {
                state.finished = true;
                state.token = text_tokenizer.eos_id();
                state.generated_tokens.push_back(state.token);
                state.next_embedding = single_token_embedding(decoder, state.token, state.prompt.hidden_size);
                state.has_next_embedding = true;
                continue;
            }
            const int32_t token = select_vibevoice_constrained_token(
                state.current.logits,
                text_tokenizer,
                first_options,
                state.sampler_rng);
            state.token = token;
            state.generated_tokens.push_back(token);
            if (token == text_tokenizer.eos_id()) {
                state.finished = true;
                state.next_embedding = single_token_embedding(decoder, token, state.prompt.hidden_size);
                state.has_next_embedding = true;
                continue;
            }
            if (token == text_tokenizer.speech_start_id()) {
                state.negative = negative_template.result;
                decoder.reset_cached_state(state.negative_cache, negative_template.state);
                decoder.release_prompt_graphs();
            }
            if (token == text_tokenizer.speech_end_id()) {
                state.acoustic_streaming_state.set_to_zero();
                state.semantic_streaming_state.set_to_zero();
            }
            if (token == text_tokenizer.speech_diffusion_id()) {
                diffusion_indices.push_back(index);
            } else {
                state.next_embedding = single_token_embedding(decoder, token, state.prompt.hidden_size);
                state.has_next_embedding = true;
            }
        }

        if (!diffusion_indices.empty()) {
            const int64_t batch_size = static_cast<int64_t>(diffusion_indices.size());
            auto initial_noise = next_diffusion_noise(
                noise_file_values,
                noise_file_offset,
                static_cast<size_t>(2 * batch_size * latent_size),
                first_options.seed,
                diffusion_rng_index,
                sampling::TorchRandnPrecision::BFloat16);
            VibeVoiceSpeechDiffusionInput diffusion_input;
            diffusion_input.batch_size = batch_size;
            diffusion_input.hidden_size = hidden_size;
            diffusion_input.latent_size = latent_size;
            diffusion_input.inference_steps = first_options.num_inference_steps;
            diffusion_input.guidance_scale = first_options.guidance_scale;
            diffusion_input.initial_speech = std::move(initial_noise);
            diffusion_input.positive_condition.reserve(static_cast<size_t>(batch_size * hidden_size));
            diffusion_input.negative_condition.reserve(static_cast<size_t>(batch_size * hidden_size));
            for (const size_t index : diffusion_indices) {
                const auto & state = states[index];
                diffusion_input.positive_condition.insert(
                    diffusion_input.positive_condition.end(),
                    state.current.last_hidden.values.begin(),
                    state.current.last_hidden.values.end());
                diffusion_input.negative_condition.insert(
                    diffusion_input.negative_condition.end(),
                    state.negative.last_hidden.values.begin(),
                    state.negative.last_hidden.values.end());
            }

            auto speech_latents = sample_vibevoice_speech_latents(diffusion_head, diffusion_scheduler, diffusion_input);
            if (speech_latents.size() != diffusion_indices.size()) {
                throw std::runtime_error("VibeVoice batch diffusion sampler returned unexpected batch size");
            }
            std::vector<VibeVoiceTokenizerLatents> decoder_latents;
            std::vector<VibeVoiceTokenizerStreamingState *> acoustic_states;
            std::vector<VibeVoiceTokenizerStreamingState *> semantic_states;
            decoder_latents.reserve(diffusion_indices.size());
            acoustic_states.reserve(diffusion_indices.size());
            semantic_states.reserve(diffusion_indices.size());
            for (size_t batch = 0; batch < diffusion_indices.size(); ++batch) {
                auto & state = states[diffusion_indices[batch]];
                decoder_latents.push_back(unscale_vibevoice_acoustic_latents_for_decoder(
                    speech_latents[batch],
                    audio_tokenizer.assets().speech_scaling_factor,
                    audio_tokenizer.assets().speech_bias_factor));
                acoustic_states.push_back(&state.acoustic_streaming_state);
                semantic_states.push_back(&state.semantic_streaming_state);
            }
            auto chunks = audio_tokenizer.decode_acoustic_streaming_batch(decoder_latents, acoustic_states);
            if (chunks.size() != diffusion_indices.size()) {
                throw std::runtime_error("VibeVoice batch acoustic decoder returned unexpected batch size");
            }
            for (size_t batch = 0; batch < diffusion_indices.size(); ++batch) {
                auto & state = states[diffusion_indices[batch]];
                const auto & chunk = chunks[batch];
                append_audio(state.audio_samples, chunk);
            }
            auto semantic_features = audio_tokenizer.encode_semantic_streaming_batch(chunks, semantic_states);
            if (semantic_features.size() != diffusion_indices.size()) {
                throw std::runtime_error("VibeVoice batch semantic encoder returned unexpected batch size");
            }
            auto acoustic_embeddings = connector.project_acoustic_batch(speech_latents);
            auto semantic_embeddings = connector.project_semantic_batch(semantic_features);
            if (acoustic_embeddings.size() != diffusion_indices.size() ||
                semantic_embeddings.size() != diffusion_indices.size()) {
                throw std::runtime_error("VibeVoice batch connector returned unexpected batch size");
            }
            for (size_t batch = 0; batch < diffusion_indices.size(); ++batch) {
                auto & state = states[diffusion_indices[batch]];
                state.next_embedding = add_embeddings(acoustic_embeddings[batch], semantic_embeddings[batch]);
                if (static_cast<int64_t>(state.next_embedding.size()) != state.prompt.hidden_size) {
                    throw std::runtime_error("VibeVoice batch generated diffusion embedding shape mismatch");
                }
                state.has_next_embedding = true;
            }
        }

        std::vector<size_t> negative_step_indices;
        std::vector<std::vector<float>> negative_step_embeddings;
        std::vector<VibeVoiceQwen2CachedState *> negative_step_states;
        negative_step_indices.reserve(states.size());
        negative_step_embeddings.reserve(states.size());
        negative_step_states.reserve(states.size());
        for (size_t index = 0; index < states.size(); ++index) {
            auto & state = states[index];
            if (!state.has_next_embedding || state.token != text_tokenizer.speech_diffusion_id()) {
                continue;
            }
            negative_step_indices.push_back(index);
            negative_step_embeddings.push_back(state.next_embedding);
            negative_step_states.push_back(&state.negative_cache);
        }
        if (!negative_step_indices.empty()) {
            int64_t active_cache_capacity = 0;
            for (const size_t index : negative_step_indices) {
                active_cache_capacity = std::max(
                    active_cache_capacity,
                    states[index].negative_cache_required);
            }
            auto negative_outputs = decoder.cached_step_batch(
                negative_step_embeddings,
                negative_step_states,
                active_cache_capacity);
            if (negative_outputs.size() != negative_step_indices.size()) {
                throw std::runtime_error("VibeVoice batch decoder negative cached step returned unexpected batch size");
            }
            for (size_t batch = 0; batch < negative_step_indices.size(); ++batch) {
                states[negative_step_indices[batch]].negative = std::move(negative_outputs[batch]);
            }
        }

        std::vector<size_t> cached_step_indices;
        std::vector<std::vector<float>> cached_step_embeddings;
        std::vector<VibeVoiceQwen2CachedState *> cached_step_states;
        cached_step_indices.reserve(states.size());
        cached_step_embeddings.reserve(states.size());
        cached_step_states.reserve(states.size());
        for (size_t index = 0; index < states.size(); ++index) {
            auto & state = states[index];
            if (!state.has_next_embedding) {
                continue;
            }
            cached_step_indices.push_back(index);
            cached_step_embeddings.push_back(state.next_embedding);
            cached_step_states.push_back(&state.positive_cache);
        }
        if (!cached_step_indices.empty()) {
            int64_t active_cache_capacity = 0;
            for (const size_t index : cached_step_indices) {
                active_cache_capacity = std::max(
                    active_cache_capacity,
                    states[index].prompt.steps + step + 1);
            }
            auto cached_outputs = decoder.cached_step_batch(
                cached_step_embeddings,
                cached_step_states,
                active_cache_capacity);
            if (cached_outputs.size() != cached_step_indices.size()) {
                throw std::runtime_error("VibeVoice batch decoder cached step returned unexpected batch size");
            }
            for (size_t batch = 0; batch < cached_step_indices.size(); ++batch) {
                states[cached_step_indices[batch]].current = std::move(cached_outputs[batch]);
            }
        }
    }

    std::vector<VibeVoiceResult> out;
    out.reserve(states.size());
    for (auto & state : states) {
        if (state.audio_samples.empty()) {
            throw std::runtime_error("VibeVoice batch generation produced no audio");
        }
        VibeVoiceResult result;
        result.audio = runtime::AudioBuffer{24000, 1, std::move(state.audio_samples)};
        result.generated_tokens = std::move(state.generated_tokens);
        out.push_back(std::move(result));
    }
    return out;
}

}  // namespace engine::models::vibevoice
