#include "engine/models/fish_audio/generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

FishAudioGenerator::FishAudioGenerator(
    std::shared_ptr<const FishAudioAssets> assets,
    std::unique_ptr<FishAudioDualARRuntime> ar,
    std::unique_ptr<engine::codecs::FishDacCodecRuntime> codec)
    : assets_(std::move(assets)),
      tokenizer_(assets_),
      prompt_builder_(assets_, tokenizer_),
      ar_(std::move(ar)),
      codec_(std::move(codec)) {
    if (assets_ == nullptr || ar_ == nullptr || codec_ == nullptr) {
        throw std::runtime_error("Fish Audio generator requires assets, AR runtime, and codec runtime");
    }
}

FishAudioGenerator::~FishAudioGenerator() = default;

engine::codecs::FishDacCodes FishAudioGenerator::encode_reference(const runtime::AudioBuffer & audio) {
    auto codes = codec_->encode_codes(audio);
    codec_->release_encode_graph();
    return codes;
}

FishAudioGenerationResult FishAudioGenerator::generate(
    const FishAudioRequest & request,
    const std::vector<engine::codecs::FishDacCodes> & reference_codes,
    const std::optional<FishAudioConversationTurn> & previous_turn,
    bool mem_saver,
    bool streaming,
    const std::function<void(const runtime::AudioBuffer &)> & on_audio) {
    engine::debug::trace_log_scalar("fish_audio.request.has_reference", !request.references.empty());
    engine::debug::trace_log_scalar("fish_audio.request.reference_count", static_cast<int64_t>(request.references.size()));
    engine::debug::trace_log_scalar("fish_audio.request.text_chars", static_cast<int64_t>(request.text.size()));
    engine::debug::trace_log_scalar("fish_audio.request.has_previous_turn", previous_turn.has_value());
    engine::debug::trace_log_scalar("fish_audio.sampler.seed", request.generation.seed);
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(request, reference_codes, previous_turn);
    engine::debug::timing_log_scalar(
        "fish_audio.prompt_build_ms",
        engine::debug::elapsed_ms(prompt_start, Clock::now()));

    const auto ar_start = Clock::now();
    FishAudioGenerationResult result;
    double decode_ms = 0.0;
    double callback_ms = 0.0;
    std::function<void(const std::vector<int32_t> &)> on_frame;
    if (streaming) {
        codec_->reset_decode_stream();
        on_frame = [&](const std::vector<int32_t> & frame) {
            const auto start = Clock::now();
            engine::codecs::FishDacCodes codes;
            codes.frames = 1;
            codes.codebooks = static_cast<int64_t>(frame.size());
            codes.codes = frame;
            auto audio = codec_->decode_stream(codes);
            decode_ms += engine::debug::elapsed_ms(start, Clock::now());
            runtime::append_audio_buffer(result.audio, audio);
            if (on_audio) on_audio(audio);
            callback_ms += engine::debug::elapsed_ms(start, Clock::now());
        };
    }
    result.codes = ar_->generate(prompt, request.generation, on_frame);
    engine::debug::trace_log_scalar("fish_audio.generated.frames", result.codes.frames);
    engine::debug::trace_log_scalar("fish_audio.generated.codebooks", result.codes.codebooks);
    engine::debug::timing_log_scalar(
        "fish_audio.ar_generate_ms",
        engine::debug::elapsed_ms(ar_start, Clock::now()) - callback_ms);

    const auto decode_start = Clock::now();
    if (!streaming) {
        result.audio = codec_->decode_codes(result.codes);
        decode_ms = engine::debug::elapsed_ms(decode_start, Clock::now());
    }
    engine::debug::timing_log_scalar(
        "fish_audio.codec_decode_ms",
        decode_ms);
    if (!streaming) codec_->release_runtime_graphs();
    if (mem_saver) {
        ar_->release_runtime_graphs();
    }
    return result;
}

}  // namespace engine::models::fish_audio
