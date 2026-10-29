#include "engine/models/ming3_tts/session.h"

#include "engine/models/ming3_tts/assets.h"
#include "engine/framework/codecs/ming_audio_vae_runtime.h"
#include "engine/models/ming3_tts/generator.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/io/text.h"
#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace engine::models::ming3_tts {
namespace {

// The official runtime cuts text into fragments of 50 semantic units, where a hanzi, kana or Hangul syllable counts like a
// Latin word. A 300-codepoint chunk of such text holds several of those fragments, its prompt leaves too few of the 512
// positions for its speech, and the end of the chunk is dropped without an error.
int64_t default_text_chunk_size(std::string_view text) {
    text::utf8_codepoint_count(text, "Ming TTS text");
    size_t cjk = 0;
    size_t other = 0;
    for (size_t pos = 0; pos < text.size();) {
        const auto ch = static_cast<unsigned char>(text[pos]);
        if (ch < 0x80U) {
            other += std::isalnum(ch) != 0;
            ++pos;
        } else if ((ch & 0xE0U) == 0xC0U) {
            ++other;
            pos += 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            const uint32_t cp = ((ch & 0x0FU) << 12) | ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 6) |
                (static_cast<unsigned char>(text[pos + 2]) & 0x3FU);
            cjk += (cp >= 0x3040U && cp <= 0x30FFU) || (cp >= 0x3400U && cp <= 0x4DBFU) ||
                (cp >= 0x4E00U && cp <= 0x9FFFU) || (cp >= 0xAC00U && cp <= 0xD7A3U);
            pos += 3;
        } else {
            pos += 4;
        }
    }
    return cjk > other ? 50 : 300;
}

struct MingReference {
    runtime::AudioBuffer audio;
    std::vector<float> latents;
    std::vector<float> speaker;
};

class MingSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    MingSession(runtime::TaskSpec task, const runtime::SessionOptions & options,
        std::shared_ptr<const MingAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          prompts_({{}, {}, assets_->resources.require_file("tokenizer_config"),
              assets_->resources.require_file("tokenizer_json"), tokenizers::LlamaBpePreTokenizer::Qwen2}),
          generator_(assets_->model_weights, assets_->config, execution_context()),
          decoder_(assets_->model_weights, assets_->config.codec,
              execution_context()) {
        runtime::validate_spec_backed_session_options(options, *contract_, "ming3_tts", "Ming TTS");
        sample_rate_ = assets_->config.codec.sample_rate;
        reference_cache_ = runtime::parse_bool_option(runtime::find_option(options.options,
            {"ming3_tts.reference_cache"}).value_or("true"), "ming3_tts.reference_cache");
    }

    std::string family() const override { return "ming3_tts"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Ming TTS run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Ming TTS");
        if (!request.text_input || request.text_input->text.empty()) throw runtime::InvalidRequestError("Ming TTS requires text input");
        const auto start = std::chrono::steady_clock::now();
        const auto & opts = request.options;
        MingGenerationOptions sampling;
        sampling.seed = runtime::parse_u64_option(opts, {"seed"}).value_or(1895);
        sampling.max_tokens = runtime::parse_positive_i64_option(opts, {"max_tokens"}, 1000);
        sampling.cfg = runtime::parse_finite_float_option(opts, {"guidance_scale"}).value_or(2.0f);
        sampling.sigma = runtime::parse_finite_float_option(opts, {"sigma"}).value_or(0.25f);
        sampling.temperature = runtime::parse_finite_float_option(opts, {"temperature"}).value_or(0.0f);
        if (sampling.sigma < 0 || sampling.temperature < 0) throw runtime::InvalidRequestError("Ming sigma and temperature must be nonnegative");
        uint64_t random_offset = 0;
        MingPromptOptions prompt_options;
        prompt_options.template_name = runtime::find_option(opts, {"template_name"}).value_or(
            task_.task == runtime::VoiceTaskKind::VoiceDesign ? "voice_design" : "tts");
        // The language option, else the transcript's (the server's top-level "language", the CLI's --language).
        prompt_options.language = runtime::find_option(opts, {"language"}).value_or(request.text_input->language);
        prompt_options.instruct = runtime::find_option(opts, {"instruct"}).value_or("");
        prompt_options.instruction_schema = runtime::find_option(opts, {"instruction_schema"}).value_or("");
        for (const auto * key : {"speed", "pitch", "volume", "emotion", "dialect"}) {
            if (const auto value = runtime::find_option(opts, {key})) prompt_options.controls.emplace(key, *value);
        }
        if (const auto voice = runtime::find_option(opts, {"voice_id"})) prompt_options.controls.emplace("ip", *voice);
        if (prompt_options.template_name == "voice_design" || prompt_options.template_name == "style") {
            prompt_options.controls["style"] = std::move(prompt_options.instruct);
            prompt_options.instruct.clear();
        }
        std::vector<std::vector<float>> speakers;
        std::vector<float> reference_latents;
        std::string reference_text = runtime::find_option(opts, {"reference_text"}).value_or("");
        const bool use_speaker = runtime::parse_bool_option(runtime::find_option(opts, {"use_speaker_embedding"})
            .value_or("true"), "use_speaker_embedding");
        bool zero_speaker = runtime::parse_bool_option(runtime::find_option(opts, {"zero_speaker_embedding"})
            .value_or("false"), "zero_speaker_embedding");
        const auto embedding_file = runtime::find_option(opts, {"speaker_embedding_file"});
        const auto embedding_files = request.option_arrays.find("speaker_embedding_file");
        if (zero_speaker) {
            speakers.emplace_back(192, 0.0f);
        } else if (embedding_file || embedding_files != request.option_arrays.end()) {
            if (use_speaker) {
                std::vector<std::string> paths;
                const auto value = embedding_file ? io::trim_ascii_whitespace(*embedding_file) : std::string{};
                if (embedding_files != request.option_arrays.end()) {
                    paths = embedding_files->second;
                } else if (!value.empty() && value.front() == '[') {
                    const auto parsed = io::json::parse(value);
                    for (const auto & item : parsed.as_array()) paths.push_back(item.as_string());
                } else {
                    paths.push_back(*embedding_file);
                }
                for (const auto & path : paths) {
                    const auto source = assets::open_tensor_source(path);
                    const auto name = source->require_tensor_name({"spk_emb", "speaker_embedding", "embedding", "emb", "tensor"});
                    const auto & shape = source->require_metadata(name).shape;
                    if (shape != std::vector<int64_t>{192} && shape != std::vector<int64_t>{1, 192}) {
                        throw runtime::InvalidRequestError("Ming speaker_embedding_file requires shape [192] or [1, 192]: " + path);
                    }
                    speakers.push_back(source->require_f32(name));
                }
            }
        } else if (request.voice && request.voice->speaker && request.voice->speaker->audio) {
            const auto reference = prepare_reference(*request.voice->speaker->audio, sampling.seed, random_offset);
            if (use_speaker) speakers.push_back(reference.speaker);
            if (!reference_text.empty()) reference_latents = reference.latents;
        } else if (!reference_text.empty()) {
            throw runtime::InvalidRequestError("Ming reference_text requires voice_ref audio");
        }
        const bool raw_text = runtime::parse_bool_option(runtime::find_option(opts, {"raw_text"}).value_or("false"), "raw_text");
        const auto chunk_size = text::parse_text_chunk_size_override(opts);
        const bool design = prompt_options.template_name == "voice_design" || prompt_options.template_name == "style" ||
            prompt_options.template_name == "freeform";
        // Later voice design and freeform chunks also carry the first chunk as their reference in the 512-position
        // context, so their chunks stay near the official 24-unit fragments: about 120 codepoints of Latin text.
        const int64_t default_chunk_size = design ? std::min<int64_t>(default_text_chunk_size(request.text_input->text), 120)
            : default_text_chunk_size(request.text_input->text);
        const auto chunks = raw_text ? std::vector<std::string>{request.text_input->text}
            : text::split_text_chunks(request.text_input->text, chunk_size.value_or(default_chunk_size),
                text::parse_text_chunk_mode_override(opts).value_or(text::TextChunkMode::TagAware));
        debug::trace_log_scalar("ming3_tts.text.chunk_count", chunks.size());
        const bool trim = runtime::parse_bool_option(runtime::find_option(opts, {"trim_silence"}).value_or("true"), "trim_silence");
        std::vector<float> waveform;
        const auto latent_dim = assets_->config.codec.encoder.latent_dim;
        std::vector<float> history;
        if (!reference_latents.empty()) history.assign(reference_latents.end() - latent_dim, reference_latents.end());
        for (const auto & chunk : chunks) {
            debug::trace_log_scalar("ming3_tts.text.chunk", io::json::stringify_string(chunk));
            const auto prompt = prompts_.build(chunk, prompt_options, speakers.size(), reference_text,
                reference_latents.size() / latent_dim);
            const auto embeddings = generator_.prepare_inputs(prompt, speakers, reference_latents, zero_speaker);
            const auto latent = generator_.generate(embeddings, sampling, random_offset, history);
            auto audio = decoder_.decode(latent);
            const size_t frame = static_cast<size_t>(sample_rate_ / 10);
            const size_t frames = audio.size() / frame;
            size_t last = frames;
            while (last > 0) {
                double score = 0;
                for (size_t i = (last - 1) * frame; i < last * frame; ++i) score += std::abs(audio[i]);
                if (score / frame > 1e-3) break;
                --last;
            }
            if (trim) {
                const size_t tail = static_cast<size_t>(sample_rate_ * 3 / 10);
                audio.resize(std::min(frames ? frames * frame : audio.size(), last * frame + tail));
            }
            waveform.insert(waveform.end(), audio.begin(), audio.end());
            // As in the official runtime, later chunks of a split voice design or freeform request take the first
            // generated chunk as their reference, so the designed voice stays the same across chunks. The reference
            // ends where its speech ends: its last latent seeds the decode history, and starting from the trailing
            // silence made some chunks stop at once.
            if (design && chunks.size() > 1 && &chunk == &chunks.front() && last > 0) {
                const auto reference = encode_reference(runtime::AudioBuffer{sample_rate_, 1,
                    std::vector<float>(audio.begin(), audio.begin() + last * frame)}, sampling.seed, random_offset);
                speakers = {reference.speaker};
                reference_latents = reference.latents;
                reference_text = chunk;
                zero_speaker = false;
                history.assign(reference_latents.end() - latent_dim, reference_latents.end());
            }
        }
        if (waveform.empty()) throw runtime::InvalidRequestError("Ming generated no audio; check the text and max_tokens");
        runtime::TaskResult result;
        result.audio_output = runtime::AudioBuffer{sample_rate_, 1, std::move(waveform)};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        return result;
    }

private:
    MingReference prepare_reference(const runtime::AudioBuffer & audio, uint64_t seed, uint64_t & random_offset) {
        const auto * cached = cache().find<MingReference>("ming3_tts.reference");
        if (reference_cache_ && cached && cached->audio.sample_rate == audio.sample_rate && cached->audio.channels == audio.channels &&
            cached->audio.samples == audio.samples) return *cached;
        auto reference = encode_reference(audio, seed, random_offset);
        if (reference_cache_) cache().put("ming3_tts.reference", reference);
        return reference;
    }

    MingReference encode_reference(const runtime::AudioBuffer & audio, uint64_t seed, uint64_t & random_offset) {
        if (!encoder_) encoder_ = std::make_unique<modules::MingAudioEncoder>(assets_->model_weights,
            assets_->config.codec, execution_context());
        if (!speaker_) {
            modules::CampplusEncoderConfig config;
            config.weight_layout = modules::CampplusEncoderWeightLayout::Fused;
            config.normalize_partial_segment_by_full_length = true;
            speaker_ = std::make_unique<modules::CampplusEncoderComponent>(
                modules::CampplusEncoderComponent::load_from_tensor_source(
                    assets_->speaker_weights, options().backend, config));
        }
        const auto features = audio::ReferenceAudioFrontend::campplus_fbank_16k(
            audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                audio.samples, audio.sample_rate, audio.channels, 16000));
        MingReference reference;
        reference.audio = audio;
        reference.speaker = speaker_->embed_from_features(features.values, features.frames, features.feature_dim).embedding;
        auto waveform = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            audio.samples, audio.sample_rate, audio.channels, sample_rate_);
        const auto & codec = assets_->config.codec;
        const size_t alignment = codec.encoder.hop_size * codec.patch_size;
        if (waveform.size() % alignment) waveform.insert(waveform.begin(), alignment - waveform.size() % alignment, 0.0f);
        const auto policy = sampling::resolve_torch_cuda_sampling_policy(execution_context().backend_type(),
            options().backend.device, "ming3_tts", "Ming TTS");
        const auto count = encoder_->latent_frames(waveform.size()) * codec.encoder.latent_dim;
        const auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(count, seed, random_offset,
            policy, sampling::TorchRandnPrecision::BFloat16);
        random_offset += sampling::torch_cuda_tensor_iterator_offset_blocks(count, policy);
        reference.latents = encoder_->encode(waveform, noise);
        return reference;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const MingAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    MingPromptBuilder prompts_;
    MingGenerator generator_;
    modules::MingAudioDecoder decoder_;
    std::unique_ptr<modules::MingAudioEncoder> encoder_;
    std::unique_ptr<modules::CampplusEncoderComponent> speaker_;
    int sample_rate_ = 0;
    bool reference_cache_ = true;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_ming3_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<MingAssets> config;
    config.family = "ming3_tts";
    config.load_assets = load_ming3_tts_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const MingAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        if ((task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning &&
             task.task != runtime::VoiceTaskKind::VoiceDesign) || task.mode != runtime::RunMode::Offline) {
            throw runtime::InvalidRequestError("Ming TTS supports offline tts, clon and voice_design");
        }
        return std::make_unique<MingSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::ming3_tts
