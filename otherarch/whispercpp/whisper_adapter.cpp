#include "model_adapter.h"
#include "otherarch/utils.h"

#include "whisper.cpp"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "otherarch/audiocpp/upstream/generated/kcpp_transcription_registry_headers.inc"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <cstdio>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <mutex>
#include <cinttypes>

#define COMMON_SAMPLE_RATE 16000

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static int whisperdebugmode = 0;
static bool whisper_is_quiet = false;
static whisper_context * whisper_ctx = nullptr;
static std::string whisper_output_text = "";
static std::string whisper_output_metadata = "";
struct audiocpp_transcription_loader_entry {
    const char * family;
    std::shared_ptr<engine::runtime::IVoiceModelLoader> loader;
    engine::runtime::VoiceTaskKind task;
    int sample_rate;
};

static const std::vector<audiocpp_transcription_loader_entry> & audiocpp_transcription_loaders()
{
    static const std::vector<audiocpp_transcription_loader_entry> loaders = {
#include "otherarch/audiocpp/upstream/generated/kcpp_transcription_registry_loaders.inc"
    };
    return loaders;
}

static const audiocpp_transcription_loader_entry * audiocpp_transcription_entry = nullptr;
static std::string audiocpp_transcription_family;
static std::unique_ptr<engine::runtime::ILoadedVoiceModel> audiocpp_transcription_model;
static std::unique_ptr<engine::runtime::IVoiceTaskSession> audiocpp_transcription_session;

int total_transcribe_gens = 0;

bool whispertype_is_speech_model_loaded()
{
    if (audiocpp_transcription_entry != nullptr) {
        return audiocpp_transcription_entry->task == engine::runtime::VoiceTaskKind::Asr;
    }
    return whisper_ctx != nullptr;
}


static bool read_audio(const std::string & b64data, int sample_rate, std::vector<float>& pcmf32)
{
    std::vector<uint8_t> media_data_buffer = kcpp_base64_decode(b64data);

    bool ok = kcpp_decode_audio_from_buf(media_data_buffer.data(), media_data_buffer.size(), sample_rate, pcmf32);
    if (!ok) {
        printf("\nError: Cannot read input audio file.");
        return false;
    }

    if(whisperdebugmode==1 && !whisper_is_quiet)
    {
        printf("\nwav_data_size: %zu",pcmf32.size());
    }

    return true;
}

static std::string output_txt(struct whisper_context * ctx) {

    std::string outtxt = "";
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        outtxt += text;
    }
    return outtxt;
}

void cb_log_disable(enum ggml_log_level , const char * , void * ) { }

static void parse_audiocpp_transcription_options(
    const whisper_generation_inputs & inputs,
    std::unordered_map<std::string, std::string> & options)
{
    const std::string json = inputs.audiocpp_extra_params_json != nullptr
        ? inputs.audiocpp_extra_params_json : "";
    if (json.empty()) {
        return;
    }
    const auto parsed = engine::io::json::parse(json);
    if (!parsed.is_object()) {
        throw std::runtime_error("Audio.cpp transcription options must be a JSON object");
    }
    for (const auto & [key, value] : parsed.as_object()) {
        if (value.is_string()) {
            options[key] = value.as_string();
        } else if (value.is_number()) {
            options[key] = engine::io::json::stringify_number(value.as_number());
        } else if (value.is_bool()) {
            options[key] = value.as_bool() ? "true" : "false";
        }
    }
}

static std::string audiocpp_transcription_metadata_json(
    const engine::runtime::TaskResult & result,
    int sample_rate)
{
    std::ostringstream json;
    json.precision(10);
    json << "{\"family\":" << engine::io::json::stringify_string(audiocpp_transcription_family)
         << ",\"words\":[";
    for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
        const auto & word = result.word_timestamps[i];
        if (i != 0) {
            json << ',';
        }
        json << "{\"word\":" << engine::io::json::stringify_string(word.word)
             << ",\"start\":" << static_cast<double>(word.span.start_sample) / sample_rate
             << ",\"end\":" << static_cast<double>(word.span.end_sample) / sample_rate
             << '}';
    }
    json << "]}";
    return json.str();
}

static std::string audiocpp_transcription_error_json(const std::string & error)
{
    return std::string("{\"family\":") +
        engine::io::json::stringify_string(audiocpp_transcription_family) +
        ",\"error\":" + engine::io::json::stringify_string(error) + "}";
}

bool whispertype_load_model(const whisper_load_model_inputs inputs)
{
    whisper_is_quiet = inputs.quiet;

    std::string modelfile = inputs.model_filename;
    std::string extension = std::filesystem::u8path(modelfile).extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    const std::string architecture = extension == ".gguf" ? gguf_get_model_arch(modelfile) : "";
    const std::string audiocpp_family = architecture == "audiocpp"
        ? gguf_get_string(modelfile, "audiocpp.model_spec.family")
        : architecture;
    audiocpp_transcription_entry = nullptr;
    audiocpp_transcription_family.clear();
    audiocpp_transcription_session.reset();
    audiocpp_transcription_model.reset();
    if (architecture == "audiocpp") {
        const auto & loaders = audiocpp_transcription_loaders();
        const auto it = std::find_if(loaders.begin(), loaders.end(),
            [&audiocpp_family](const auto & entry) { return audiocpp_family == entry.family; });
        if (it == loaders.end()) {
            printf("\nAudio.cpp Transcription Load Error: unsupported model family '%s'.\n",
                audiocpp_family.c_str());
            return false;
        }
        audiocpp_transcription_entry = &*it;
        audiocpp_transcription_family = audiocpp_family;
        printf("\nLoading Audio.cpp %s Model: %s", audiocpp_family.c_str(), modelfile.c_str());
        try {
            engine::runtime::ModelLoadRequest request;
            request.model_path = modelfile;
            request.family_hint = audiocpp_family;
            audiocpp_transcription_model = audiocpp_transcription_entry->loader->load(request);

            engine::runtime::SessionOptions options;
            options.backend.type = engine::core::BackendType::BestAvailable;
            options.backend.device = std::max(0, inputs.kcpp_main_gpu);
            options.backend.threads = 4;
            if (audiocpp_family == "sheetsage2") {
                // A 300-second untiled attention score can exceed 3 GiB.
                options.options["sheetsage2.attention_tile_rows"] = "1024";
            }
            const std::string aligner_filename = inputs.aligner_filename != nullptr
                ? inputs.aligner_filename : "";
            if (!aligner_filename.empty()) {
                if (audiocpp_family == "qwen3_asr") {
                    options.options["qwen3_asr.forced_aligner_path"] = aligner_filename;
                } else {
                    printf("\nWarning: --transcribealigner is only used by Qwen3-ASR; ignoring it for %s.",
                        audiocpp_family.c_str());
                }
            }
            audiocpp_transcription_session = audiocpp_transcription_model->create_task_session(
                {audiocpp_transcription_entry->task, engine::runtime::RunMode::Offline},
                options);
            if (audiocpp_transcription_entry->task == engine::runtime::VoiceTaskKind::Midi) {
                audiocpp_transcription_session->prepare({});
            }
            printf("\nAudio.cpp %s Load Complete.\n", audiocpp_family.c_str());
            return true;
        } catch (const std::exception & error) {
            printf("\nAudio.cpp %s Load Error: %s\n", audiocpp_family.c_str(), error.what());
            audiocpp_transcription_session.reset();
            audiocpp_transcription_model.reset();
            audiocpp_transcription_entry = nullptr;
            audiocpp_transcription_family.clear();
            return false;
        }
    }

    printf("\nLoading Whisper Model: %s",modelfile.c_str());

    whisperdebugmode = inputs.debugmode;
    if (whisperdebugmode!=1) {
        whisper_log_set(cb_log_disable, NULL);
    }

    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = true;
    cparams.flash_attn = false;

    whisper_ctx = whisper_init_from_file_with_params(modelfile.c_str(), cparams);

    if (whisper_ctx == nullptr) {
        printf("\nWhisper Load Error: Failed to initialize whisper context!\n");
        return false;
    }

    printf("\nWhisper Load Complete.\n");

    return true;
}

whisper_generation_outputs whispertype_generate(const whisper_generation_inputs inputs)
{
    whisper_generation_outputs output;
    whisper_output_metadata.clear();

    if (audiocpp_transcription_entry != nullptr) {
        auto * offline = audiocpp_transcription_session
            ? dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(audiocpp_transcription_session.get())
            : nullptr;
        if (offline == nullptr) {
            printf("\nWarning: Audio.cpp %s is not initialized.\n", audiocpp_transcription_family.c_str());
            output.text = "";
            output.status = 0;
            return output;
        }
        if (!whisper_is_quiet) {
            printf("\nAudio.cpp %s Transcribing...", audiocpp_transcription_family.c_str());
        }
        try {
            std::vector<float> pcmf32;
            if (!::read_audio(std::string(inputs.audio_data),
                    audiocpp_transcription_entry->sample_rate, pcmf32)) {
                throw std::runtime_error("failed to decode input audio");
            }
            engine::runtime::TaskRequest request;
            request.audio_input = engine::runtime::AudioBuffer{
                audiocpp_transcription_entry->sample_rate, 1, std::move(pcmf32)};
            parse_audiocpp_transcription_options(inputs, request.options);
            if (audiocpp_transcription_entry->task == engine::runtime::VoiceTaskKind::Asr) {
                // Fixed mode keeps the common text-only path self-contained;
                // VAD/timestamp options may still be selected explicitly.
                if (request.options.find("audio_chunk_mode") == request.options.end()) {
                    request.options["audio_chunk_mode"] = "fixed";
                }
                const std::string langcode = inputs.langcode != nullptr ? inputs.langcode : "";
                if (!langcode.empty() && langcode != "auto" && langcode != "Auto" &&
                    request.options.find("language") == request.options.end() &&
                    audiocpp_transcription_family != "audio8_asr") {
                    request.options["language"] = langcode;
                }
                if (audiocpp_transcription_family == "qwen3_asr") {
                    const std::string request_language =
                        (langcode == "auto" || langcode == "Auto") ? "" : langcode;
                    request.text_input = engine::runtime::Transcript{
                        inputs.prompt != nullptr ? inputs.prompt : "", request_language};
                }
                audiocpp_transcription_session->prepare(
                    engine::runtime::build_preparation_request(request));
            }
            auto result = offline->run(request);
            if (!result.text_output.has_value()) {
                throw std::runtime_error("model produced no text output");
            }
            whisper_output_metadata = audiocpp_transcription_metadata_json(
                result, audiocpp_transcription_entry->sample_rate);
            whisper_output_text = std::move(result.text_output->text);
            const std::string ts = get_timestamp_str();
            if (whisper_is_quiet) {
                printf("\n[%s] Audio.cpp %s Transcription Done.",
                    ts.c_str(), audiocpp_transcription_family.c_str());
            } else {
                printf("\n[%s] Audio.cpp %s Output:\n%s", ts.c_str(),
                    audiocpp_transcription_family.c_str(), whisper_output_text.c_str());
            }
            output.text = whisper_output_text.c_str();
            output.metadata_json = whisper_output_metadata.c_str();
            output.status = 1;
            total_transcribe_gens += 1;
            return output;
        } catch (const std::exception & error) {
            printf("\nAudio.cpp %s Generation Error: %s\n",
                audiocpp_transcription_family.c_str(), error.what());
            output.text = "";
            whisper_output_metadata = audiocpp_transcription_error_json(error.what());
            output.metadata_json = whisper_output_metadata.c_str();
            output.status = 0;
            return output;
        }
    }

    if(whisper_ctx==nullptr)
    {
        printf("\nWarning: KCPP whisper not initialized!\n");
        output.text = "";
        output.status = 0;
        return output;
    }

    if(!whisper_is_quiet)
    {
        printf("\nWhisper Transcribe Generating...");
    }

    const std::string b64data = std::string(inputs.audio_data);
    const std::string initprompt = std::string(inputs.prompt);
    const std::string langcode = std::string(inputs.langcode);

    std::vector<float> pcmf32;               // mono-channel F32 PCM

    if (!::read_audio(b64data, COMMON_SAMPLE_RATE, pcmf32)) {
        printf("\nWhisper: Failed to read input wav data!\n");
        output.text = "";
        output.status = 0;
        return output;
    }

    // run the inference
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.strategy = WHISPER_SAMPLING_GREEDY;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.language         = langcode.c_str();
    wparams.detect_language  = false;
    wparams.n_threads        = 4;
    wparams.n_max_text_ctx   = wparams.n_max_text_ctx;
    wparams.offset_ms        = 0;
    wparams.duration_ms      = 0;
    wparams.token_timestamps = false;
    wparams.thold_pt         = 0.01f;
    wparams.max_len          = 100;
    wparams.split_on_word    = false;
    wparams.audio_ctx        = 0;
    wparams.speed_up         = false;
    wparams.debug_mode       = (whisperdebugmode==1);
    wparams.tdrz_enable      = false;
    wparams.suppress_regex   = nullptr;
    wparams.suppress_non_speech_tokens = inputs.suppress_non_speech;
    wparams.initial_prompt   = initprompt.c_str();
    wparams.greedy.best_of        = -1;
    wparams.beam_search.beam_size = -1;
    wparams.temperature_inc  = 0.2f;
    wparams.temperature      = 0.0f;
    wparams.entropy_thold    = 2.40f;
    wparams.logprob_thold    = -1.00f;
    wparams.no_timestamps    = true;

    if (whisper_full_parallel(whisper_ctx, wparams, pcmf32.data(), pcmf32.size(), 1) != 0) {
        printf("\nWhisper: Failed to process audio!\n");
        output.text = "";
        output.status = 0;
        return output;
    }

    if (!whisper_is_quiet && whisperdebugmode==1) {
        whisper_print_timings(whisper_ctx);
    }

    // output text transcription
    whisper_output_text = output_txt(whisper_ctx);
    std::string ts = get_timestamp_str();
    if(!whisper_is_quiet)
    {
        printf("\n[%s] Whisper Transcribe Output: %s",ts.c_str(),whisper_output_text.c_str());
    } else {
        printf("\n[%s] Whisper Transcribe Done.",ts.c_str());
    }
    output.text = whisper_output_text.c_str();
    output.status = 1;
    total_transcribe_gens += 1;
    return output;
}
