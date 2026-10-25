#include "engine/models/yue2/pipeline.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/yue2/ar_runtime.h"
#include "engine/models/yue2/nar_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::yue2 {
namespace {

using Clock = std::chrono::steady_clock;

std::string request_text(const Yue2Request & request) {
    std::string text;
    text += cot_instruction(request.cot);
    text += "\n[Tags]\n";
    text += request.style;
    text += "\n[Lyrics]\n";
    text += request.lyrics;
    text += "\n";
    return text;
}

std::vector<int32_t> token_prefixes(
    const Yue2Request & request,
    const Yue2TextTokenizer & tokenizer,
    const std::vector<int32_t> & abc_ids) {
    std::vector<int32_t> out;
    out.push_back(kEodToken);
    auto text_ids = tokenizer.encode(request_text(request));
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    out.push_back(kAbcStartToken);
    if (request.cot == Yue2CotMode::Off) {
        out.push_back(kAbcEndToken);
        out.push_back(kMusicStartToken);
        return out;
    }
    out.insert(out.end(), abc_ids.begin(), abc_ids.end());
    if (!request.abc.empty()) {
        out.push_back(kAbcEndToken);
        out.push_back(kMusicStartToken);
    }
    return out;
}

std::vector<int32_t> negative_prefix(
    const Yue2Request & request,
    const Yue2TextTokenizer & tokenizer,
    const std::vector<int32_t> & abc_ids) {
    std::vector<int32_t> out;
    out.push_back(kEodToken);
    const auto text_ids = tokenizer.encode(cot_instruction(request.cot));
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    if (request.cot == Yue2CotMode::Off) {
        out.push_back(kMusicStartToken);
        return out;
    }
    out.push_back(kAbcStartToken);
    out.insert(out.end(), abc_ids.begin(), abc_ids.end());
    out.push_back(kAbcEndToken);
    out.push_back(kMusicStartToken);
    return out;
}

std::vector<int32_t> codec_from_semantic_tokens(const std::vector<int32_t> & tokens) {
    std::vector<int32_t> out;
    out.reserve(tokens.size());
    for (const int32_t token : tokens) {
        if (token >= kCodecOffset && token < kCodecOffset + kCodecSize) {
            out.push_back(token - kCodecOffset);
        }
    }
    return out;
}

std::vector<int32_t> semantic_tokens_from_codec(const std::vector<int32_t> & codec) {
    std::vector<int32_t> out;
    out.reserve(codec.size());
    for (const int32_t index : codec) {
        if (index < 0 || index >= kCodecSize) {
            throw std::runtime_error("Yue2 semantic prefix codec index is out of range");
        }
        out.push_back(kCodecOffset + index);
    }
    return out;
}

Yue2ARSamplingWindow abc_window(const Yue2GenerationConfig & generation) {
    return Yue2ARSamplingWindow{
        0,
        kEodToken,
        kAbcEndToken,
        generation.abc.min_tokens,
        generation.abc.max_tokens,
        generation.abc,
    };
}

Yue2ARSamplingWindow semantic_window(const Yue2GenerationConfig & generation) {
    return Yue2ARSamplingWindow{
        kCodecOffset,
        kCodecOffset + kCodecSize,
        kMusicEndToken,
        generation.semantic.min_tokens,
        generation.semantic.max_tokens,
        generation.semantic,
    };
}

}  // namespace

class Yue2PipelineRuntime::Impl {
public:
    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType model_weight_type,
        assets::TensorStorageType vae_weight_type,
        size_t model_weight_context_bytes,
        size_t vae_weight_context_bytes,
        size_t ar_prefill_graph_arena_bytes,
        size_t ar_decode_graph_arena_bytes,
        size_t nar_graph_arena_bytes,
        size_t vae_graph_arena_bytes,
        core::AttentionPreference attention_preference,
        int64_t nar_attention_tile_rows)
        : execution(&execution),
          assets(std::move(assets)),
          tokenizer(this->assets->tiktoken_path),
          model_weight_type(model_weight_type),
          vae_weight_type(vae_weight_type),
          model_weight_context_bytes(model_weight_context_bytes),
          vae_weight_context_bytes(vae_weight_context_bytes),
          ar_prefill_graph_arena_bytes(ar_prefill_graph_arena_bytes),
          ar_decode_graph_arena_bytes(ar_decode_graph_arena_bytes),
          nar_graph_arena_bytes(nar_graph_arena_bytes),
          vae_graph_arena_bytes(vae_graph_arena_bytes),
          nar_attention_tile_rows(nar_attention_tile_rows) {
        if (!this->assets) {
            throw std::runtime_error("Yue2 pipeline requires assets");
        }
        (void) this->nar_graph_arena_bytes;
        allow_flash_attention = core::resolve_flash_attention(
            execution.backend(),
            this->assets->config.model.head_dim,
            attention_preference);
        // Intel Vulkan: ggml-vulkan's flash kernels lose to the NAR's eager
        // lowering by 2.2x (Arc Pro B70, 275 s -> 126 s NAR); explicit
        // flash/eager still win.
        if (attention_preference == core::AttentionPreference::Auto && allow_flash_attention &&
            core::vulkan_device_is_intel(execution.backend())) {
            allow_flash_attention = false;
        }
        engine::debug::trace_log_scalar("yue2.attention.allow_flash", allow_flash_attention);
    }

    Yue2Plan plan(const Yue2Request & request) {
        Yue2Plan out;
        out.cot = request.cot;
        if (!request.abc.empty()) {
            out.abc = request.abc;
            out.abc_ids = tokenizer.encode(request.abc);
        }
        out.prefix = token_prefixes(request, tokenizer, out.abc_ids);
        return out;
    }

    Yue2SemanticResult generate_semantic(const Yue2Request & request, Yue2Plan plan) {
        Yue2SemanticResult out;
        out.plan = std::move(plan);
        generate_plan_abc(request, out.plan);
        const auto forced = semantic_tokens_from_codec(request.semantic_prefix);
        if (!forced.empty() &&
            static_cast<int64_t>(forced.size()) >= request.generation.semantic.max_tokens) {
            // The prefix already fills the window: nothing would be sampled, so
            // the AR prefill and its logits are skipped. stop_after=audio still
            // prefills the stream once, in the NAR stage, for its conditioning.
            out.tokens = forced;
            out.truncated = true;
            engine::debug::timing_log_scalar("yue2.semantic.tokens", out.tokens.size());
            engine::debug::timing_log_scalar("yue2.semantic.truncated", out.truncated);
            return out;
        }
        ensure_ar();
        const auto neg = negative_prefix(request, tokenizer, out.plan.abc_ids);
        out.tokens = ar->generate_cfg(
            out.plan.prefix,
            neg,
            semantic_window(request.generation),
            request_guidance_scale(request),
            request.seed,
            forced);
        out.truncated = static_cast<int64_t>(out.tokens.size()) >= request.generation.semantic.max_tokens;
        engine::debug::timing_log_scalar("yue2.semantic.tokens", out.tokens.size());
        engine::debug::timing_log_scalar("yue2.semantic.truncated", out.truncated);
        ar->release_runtime_graphs();
        return out;
    }

    std::vector<float> synthesize_latents(
        const Yue2SemanticResult & semantic,
        const Yue2GenerationConfig & generation,
        const std::vector<float> & noise,
        uint64_t seed) {
        const auto codec = codec_from_semantic_tokens(semantic.tokens);
        if (codec.empty()) {
            throw std::runtime_error("Yue2 semantic generation produced no codec tokens");
        }
        ensure_nar();
        ensure_ar();
        return nar->synthesize(
            semantic.plan.prefix,
            codec,
            [this](const std::vector<int32_t> & tokens) {
                return ar->prefill_device_state(tokens);
            },
            noise,
            seed,
            generation.ode_steps,
            generation.context);
    }

    runtime::AudioBuffer decode_audio(const std::vector<float> & latents, int64_t frames) {
        ensure_vae();
        const int64_t channels = assets->config.model.latent_dim;
        if (frames <= 0 || static_cast<int64_t>(latents.size()) != frames * channels) {
            throw std::runtime_error("Yue2 VAE latent shape mismatch");
        }
        const int64_t core_frames = assets->config.vae.decode_core_frames;
        const int64_t halo_frames = assets->config.vae.decode_halo_frames;
        const int64_t ratio = assets->config.vae.downsampling_ratio;
        if (core_frames <= 0 || halo_frames < 0 || ratio <= 0) {
            throw std::runtime_error("Yue2 VAE tile configuration is invalid");
        }
        engine::debug::timing_log_scalar("yue2.vae_decode.latent_frames", frames);

        auto make_planar_tile = [&](int64_t begin, int64_t end) {
            const int64_t tile_frames = end - begin;
            std::vector<float> planar(static_cast<size_t>(channels * tile_frames), 0.0F);
            for (int64_t t = 0; t < tile_frames; ++t) {
                for (int64_t c = 0; c < channels; ++c) {
                    planar[static_cast<size_t>(c * tile_frames + t)] =
                        latents[static_cast<size_t>((begin + t) * channels + c)];
                }
            }
            return planar;
        };

        if (frames <= core_frames) {
            const auto decode_start = Clock::now();
            auto audio = vae->decode(make_planar_tile(0, frames), 1, frames).front();
            engine::debug::timing_log_scalar("yue2.vae_decode.tiles", 1);
            engine::debug::timing_log_scalar("yue2.vae_decode.tile_decode_ms", engine::debug::elapsed_ms(decode_start));
            engine::debug::timing_log_scalar("yue2.vae_decode.output_frames", static_cast<int64_t>(audio.samples.size()) / audio.channels);
            return audio;
        }

        const int64_t total_output_frames = frames * ratio - 64;
        if (total_output_frames <= 0) {
            throw std::runtime_error("Yue2 VAE output frame count is invalid");
        }
        runtime::AudioBuffer audio;
        audio.sample_rate = assets->config.vae.sample_rate;
        audio.channels = static_cast<int>(assets->config.vae.channels);
        audio.samples.assign(static_cast<size_t>(total_output_frames * audio.channels), 0.0F);
        int64_t tiles = 0;
        double tile_decode_ms = 0.0;
        for (int64_t start = 0; start < frames; start += core_frames) {
            const int64_t end = std::min(frames, start + core_frames);
            const int64_t left = std::max<int64_t>(0, start - halo_frames);
            const int64_t right = std::min(frames, end + halo_frames);
            const auto tile_decode_start = Clock::now();
            auto tile_audio = vae->decode(make_planar_tile(left, right), 1, right - left).front();
            tile_decode_ms += engine::debug::elapsed_ms(tile_decode_start);
            const int64_t out_start = start * ratio;
            const int64_t out_end = std::min(end * ratio, total_output_frames);
            const int64_t copy_frames = out_end - out_start;
            const int64_t crop_start = (start - left) * ratio;
            if (copy_frames <= 0 ||
                crop_start < 0 ||
                crop_start + copy_frames > static_cast<int64_t>(tile_audio.samples.size()) / tile_audio.channels) {
                throw std::runtime_error("Yue2 VAE tile did not cover output core");
            }
            for (int64_t t = 0; t < copy_frames; ++t) {
                for (int64_t c = 0; c < audio.channels; ++c) {
                    audio.samples[static_cast<size_t>((out_start + t) * audio.channels + c)] =
                        tile_audio.samples[static_cast<size_t>((crop_start + t) * tile_audio.channels + c)];
                }
            }
            ++tiles;
        }
        engine::debug::timing_log_scalar("yue2.vae_decode.tiles", tiles);
        engine::debug::timing_log_scalar("yue2.vae_decode.tile_decode_ms", tile_decode_ms);
        engine::debug::timing_log_scalar("yue2.vae_decode.output_frames", total_output_frames);
        return audio;
    }

    Yue2RunResult run(const Yue2Request & request) {
        if (engine::debug::trace_log_enabled()) {
            std::ostringstream settings;
            settings << "seed=" << request.seed
                     << " cot=" << cot_mode_name(request.cot)
                     << " stop_after=" << stop_after_name(request.stop_after)
                     << " guidance_scale=" << request_guidance_scale(request)
                     << " num_inference_steps=" << request.generation.ode_steps
                     << " context=" << request.generation.context
                     << " abc=" << (request.abc.empty() ?
                         (request.cot == Yue2CotMode::Off ? "none" : "generated") : "provided")
                     << " nar_noise=" << (request.nar_noise.empty() ? "generated" : "provided")
                     << " semantic_prefix_frames=" << request.semantic_prefix.size();
            engine::debug::trace_log_scalar("yue2.request", settings.str());
            for (const bool abc : {true, false}) {
                if (abc && (request.cot == Yue2CotMode::Off || !request.abc.empty())) {
                    continue;
                }
                const auto & sampling = abc ? request.generation.abc : request.generation.semantic;
                std::ostringstream config;
                config << "temperature=" << sampling.temperature
                       << " top_p=" << sampling.top_p
                       << " top_k=" << sampling.top_k
                       << " repetition_penalty=" << sampling.repetition_penalty
                       << " penalty_window=" << sampling.penalty_window
                       << " min_tokens=" << sampling.min_tokens
                       << " max_tokens=" << sampling.max_tokens;
                engine::debug::trace_log_scalar(abc ? "yue2.sampling.abc" : "yue2.sampling.semantic", config.str());
            }
        }
        const auto plan_start = Clock::now();
        auto planned = plan(request);
        engine::debug::timing_log_scalar("yue2.plan_ms", engine::debug::elapsed_ms(plan_start, Clock::now()));
        Yue2RunResult out;
        if (request.stop_after == Yue2StopAfter::Abc) {
            generate_plan_abc(request, planned);
            ar.reset();
            attach_plan_abc(request, planned, out);
            return out;
        }
        const auto semantic_start = Clock::now();
        auto semantic = generate_semantic(request, std::move(planned));
        engine::debug::timing_log_scalar("yue2.semantic_ms", engine::debug::elapsed_ms(semantic_start, Clock::now()));
        if (request.export_semantic) {
            out.semantic_codes = codec_from_semantic_tokens(semantic.tokens);
            out.semantic_truncated = semantic.truncated;
        }
        attach_plan_abc(request, semantic.plan, out);
        if (request.stop_after == Yue2StopAfter::Semantic) {
            if (request.export_semantic && out.semantic_codes.empty()) {
                throw std::runtime_error("Yue2 semantic generation produced no codec tokens");
            }
            ar.reset();
            return out;
        }
        const auto nar_start = Clock::now();
        auto latents = synthesize_latents(semantic, request.generation, request.nar_noise, request.seed);
        engine::debug::timing_log_scalar("yue2.nar_ms", engine::debug::elapsed_ms(nar_start, Clock::now()));
        const int64_t frames = static_cast<int64_t>(latents.size()) / assets->config.model.latent_dim;
        ar.reset();
        nar.reset();
        const auto vae_start = Clock::now();
        auto audio = decode_audio(latents, frames);
        engine::debug::timing_log_scalar("yue2.vae_decode_ms", engine::debug::elapsed_ms(vae_start, Clock::now()));
        if (vae) {
            vae->release_runtime_graphs();
        }
        out.audio = std::move(audio);
        return out;
    }

    void release_runtime_graphs() {
        if (vae) {
            vae->release_runtime_graphs();
        }
        if (ar) {
            ar->release_runtime_graphs();
        }
        if (nar) {
            nar->release_runtime_graphs();
        }
    }

private:
    void generate_plan_abc(const Yue2Request & request, Yue2Plan & plan) {
        if (request.cot == Yue2CotMode::Off || !request.abc.empty()) {
            return;
        }
        ensure_ar();
        const auto abc_start = Clock::now();
        plan.abc_ids = ar->generate(plan.prefix, abc_window(request.generation), request.seed);
        engine::debug::timing_log_scalar("yue2.semantic.abc_generate_ms", engine::debug::elapsed_ms(abc_start));
        plan.truncated = static_cast<int64_t>(plan.abc_ids.size()) >= request.generation.abc.max_tokens;
        plan.prefix.insert(plan.prefix.end(), plan.abc_ids.begin(), plan.abc_ids.end());
        plan.prefix.push_back(kAbcEndToken);
        plan.prefix.push_back(kMusicStartToken);
        engine::debug::timing_log_scalar("yue2.semantic.abc_generated_tokens", plan.abc_ids.size());
        engine::debug::timing_log_scalar("yue2.semantic.abc_truncated", plan.truncated);
        ar->release_runtime_graphs();
    }

    void attach_plan_abc(const Yue2Request & request, const Yue2Plan & plan, Yue2RunResult & out) {
        if (request.cot == Yue2CotMode::Off || !request.abc.empty() || plan.abc_ids.empty()) {
            return;
        }
        const auto decode_start = Clock::now();
        out.plan_abc_text = tokenizer.decode(plan.abc_ids);
        engine::debug::timing_log_scalar("yue2.plan.abc_decode_ms", engine::debug::elapsed_ms(decode_start, Clock::now()));
        out.plan_abc_truncated = plan.truncated;
    }

    void ensure_ar() {
        if (ar) {
            return;
        }
        const auto start = Clock::now();
        ar = std::make_unique<Yue2ARRuntime>(
            *execution,
            assets,
            model_weight_type,
            model_weight_context_bytes,
            ar_prefill_graph_arena_bytes,
            ar_decode_graph_arena_bytes);
        engine::debug::timing_log_scalar("yue2.ar.init_ms", engine::debug::elapsed_ms(start));
    }

    void ensure_vae() {
        if (vae) {
            return;
        }
        codecs::OobleckAudioVaeConfig config;
        config.sample_rate = assets->config.vae.sample_rate;
        config.audio_channels = assets->config.vae.channels;
        config.encoder_latent_dim = assets->config.vae.encoder_latent_dim;
        config.decoder_latent_dim = assets->config.vae.latent_dim;
        config.encoder_prefix = "encoder";
        config.decoder_prefix = "decoder";
        codecs::OobleckAudioVaeRuntimeOptions options;
        options.weight_context_bytes = vae_weight_context_bytes;
        options.graph_arena_bytes = vae_graph_arena_bytes;
        options.weight_storage_type = vae_weight_type;
        const auto start = Clock::now();
        vae = std::make_unique<codecs::OobleckAudioVaeRuntime>(
            assets->vae_weights,
            *execution,
            std::move(config),
            options);
        engine::debug::timing_log_scalar("yue2.vae.init_ms", engine::debug::elapsed_ms(start));
    }

    void ensure_nar() {
        if (nar) {
            return;
        }
        const auto start = Clock::now();
        nar = std::make_unique<Yue2NARRuntime>(
            *execution,
            assets,
            model_weight_type,
            model_weight_context_bytes,
            nar_graph_arena_bytes,
            allow_flash_attention,
            nar_attention_tile_rows);
        engine::debug::timing_log_scalar("yue2.nar.init_ms", engine::debug::elapsed_ms(start));
    }

    core::ExecutionContext * execution = nullptr;
    std::shared_ptr<const Yue2Assets> assets;
    Yue2TextTokenizer tokenizer;
    bool allow_flash_attention = true;
    assets::TensorStorageType model_weight_type = assets::TensorStorageType::Native;
    assets::TensorStorageType vae_weight_type = assets::TensorStorageType::Native;
    size_t model_weight_context_bytes = 0;
    size_t vae_weight_context_bytes = 0;
    size_t ar_prefill_graph_arena_bytes = 0;
    size_t ar_decode_graph_arena_bytes = 0;
    size_t nar_graph_arena_bytes = 0;
    size_t vae_graph_arena_bytes = 0;
    int64_t nar_attention_tile_rows = 0;
    std::unique_ptr<codecs::OobleckAudioVaeRuntime> vae;
    std::unique_ptr<Yue2ARRuntime> ar;
    std::unique_ptr<Yue2NARRuntime> nar;
};

Yue2PipelineRuntime::Yue2PipelineRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const Yue2Assets> assets,
    assets::TensorStorageType model_weight_type,
    assets::TensorStorageType vae_weight_type,
    size_t model_weight_context_bytes,
    size_t vae_weight_context_bytes,
    size_t ar_prefill_graph_arena_bytes,
    size_t ar_decode_graph_arena_bytes,
    size_t nar_graph_arena_bytes,
    size_t vae_graph_arena_bytes,
    core::AttentionPreference attention_preference,
    int64_t nar_attention_tile_rows)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          model_weight_type,
          vae_weight_type,
          model_weight_context_bytes,
          vae_weight_context_bytes,
          ar_prefill_graph_arena_bytes,
          ar_decode_graph_arena_bytes,
          nar_graph_arena_bytes,
          vae_graph_arena_bytes,
          attention_preference,
          nar_attention_tile_rows)) {}

Yue2PipelineRuntime::~Yue2PipelineRuntime() = default;

Yue2Plan Yue2PipelineRuntime::plan(const Yue2Request & request) {
    return impl_->plan(request);
}

Yue2SemanticResult Yue2PipelineRuntime::generate_semantic(const Yue2Request & request, Yue2Plan plan) {
    return impl_->generate_semantic(request, std::move(plan));
}

std::vector<float> Yue2PipelineRuntime::synthesize_latents(
    const Yue2SemanticResult & semantic,
    const Yue2GenerationConfig & generation,
    uint64_t seed) {
    static const std::vector<float> empty_noise;
    return impl_->synthesize_latents(semantic, generation, empty_noise, seed);
}

runtime::AudioBuffer Yue2PipelineRuntime::decode_audio(const std::vector<float> & latents, int64_t frames) {
    return impl_->decode_audio(latents, frames);
}

Yue2RunResult Yue2PipelineRuntime::run(const Yue2Request & request) {
    return impl_->run(request);
}

void Yue2PipelineRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::yue2
