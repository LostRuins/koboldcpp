#include "model_adapter.h"
#include "otherarch/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <mutex>
#include <cinttypes>
#include <filesystem>
#include <memory>

#include "engine/framework/io/json.h"
#include "engine/models/yue2/session.h"

#include "./request.cpp"
#include "./ace-qwen3.cpp"
#include "./dit-vae.cpp"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static int musicdebugmode = 0;
static bool music_is_quiet = false;
static bool musicgen_llm_loaded = false;
static bool musicgen_diffusion_loaded = false;
static std::string musicvulkandeviceenv;
static bool is_audiocpp_yue2 = false;
static std::unique_ptr<engine::runtime::ILoadedVoiceModel> audiocpp_music_model;
static std::unique_ptr<engine::runtime::IVoiceTaskSession> audiocpp_music_session;

static std::string music_output_json_str = "";
static std::string b64_music_output = "";

using music_clock = std::chrono::steady_clock;

static double music_elapsed_seconds(const music_clock::time_point & start)
{
    return std::chrono::duration<double>(music_clock::now() - start).count();
}

bool musictype_load_model(const music_load_model_inputs inputs)
{
    music_is_quiet = inputs.quiet;

    std::string musicllm_filename = inputs.musicllm_filename;
    std::string musicembedding_filename = inputs.musicembedding_filename;
    std::string musicdiffusion_filename = inputs.musicdiffusion_filename;
    std::string musicvae_filename = inputs.musicvae_filename;
    bool lowvram = inputs.lowvram;
    is_audiocpp_yue2 = false;
    musicgen_llm_loaded = false;
    musicgen_diffusion_loaded = false;
    audiocpp_music_session.reset();
    audiocpp_music_model.reset();

    // A two-file LLM+VAE configuration is YuE2. ACE-Step continues to use its
    // LLM plus embedding, diffusion, and VAE configuration below.
    const bool is_yue2_configuration =
        !musicllm_filename.empty() && !musicvae_filename.empty() &&
        musicembedding_filename.empty() && musicdiffusion_filename.empty();
    if (is_yue2_configuration) {
        auto loader = engine::models::yue2::make_yue2_loader();
        engine::runtime::ModelLoadRequest request;
        request.model_path = musicllm_filename;
        request.family_hint = "yue2";
        const auto tokenizer_path = std::filesystem::absolute(
            std::filesystem::u8path(inputs.executable_path != nullptr ? inputs.executable_path : "") /
            "embd_res" / "yue2_qwen_tiktoken.embd");
        request.options["yue2.tiktoken"] = tokenizer_path.u8string();
        try {
            printf("\nLoading Audio.cpp YuE2 Model: %s\n", musicllm_filename.c_str());
            printf("Loading Audio.cpp YuE2 VAE: %s\n", musicvae_filename.c_str());
            audiocpp_music_model = loader->load(request);
            engine::runtime::SessionOptions options;
            options.backend.type = engine::core::BackendType::BestAvailable;
            options.backend.device = std::max(0, inputs.kcpp_main_gpu);
            options.backend.threads = 4;
            options.options["yue2.model_gguf"] = std::filesystem::absolute(
                std::filesystem::u8path(musicllm_filename)).u8string();
            options.options["yue2.vae_gguf"] = std::filesystem::absolute(
                std::filesystem::u8path(musicvae_filename)).u8string();
            audiocpp_music_session = audiocpp_music_model->create_task_session(
                {engine::runtime::VoiceTaskKind::AudioGeneration, engine::runtime::RunMode::Offline},
                options);
            audiocpp_music_session->prepare({});
            is_audiocpp_yue2 = true;
            printf("\nAudio.cpp YuE2 Load Complete.\n");
            return true;
        } catch (const std::exception & error) {
            printf("\nAudio.cpp YuE2 Load Error: %s\n", error.what());
            audiocpp_music_session.reset();
            audiocpp_music_model.reset();
            return false;
        }
    }

    if(lowvram)
    {
        printf("\nMusicGen LowVRAM mode, will swap models at runtime");
    }
    printf("\nLoading Music Gen LLM Model: %s\nLoading Music Gen Embed Model: %s\nLoading Music Gen Diffusion Model: %s\nLoading Music Gen VAE Model: %s\n",
    musicllm_filename.c_str(),musicembedding_filename.c_str(),musicdiffusion_filename.c_str(),musicvae_filename.c_str());
    musicdebugmode = inputs.debugmode;
    bool ok = false;
    if(musicllm_filename!="")
    {
        ok = load_acestep_lm(musicllm_filename,lowvram,musicdebugmode);
        if (!ok) {
            printf("\nFailed to load Music Gen LM Model!\n");
            return false;
        }
        if(lowvram)
        {
            unload_acestep_lm();
        }
        musicgen_llm_loaded = ok;
    }

    if(musicdiffusion_filename!="" && musicembedding_filename!="" && musicvae_filename!="")
    {
        ok = load_acestep_dit(musicembedding_filename,musicdiffusion_filename,lowvram);
        if (!ok) {
            printf("\nFailed to load Music Gen Diffusion, Embed or VAE Model!\n");
            return false;
        }
        if(lowvram)
        {
            unload_acestep_dit_core();
            unload_acestep_dit_others();
        }
        load_acestep_vae_enc(musicvae_filename,lowvram);
        if(lowvram)
        {
            unload_acestep_vae_enc();
        }
        load_acestep_vae_dec(musicvae_filename,lowvram);
        if(lowvram)
        {
            unload_acestep_vae_dec();
        }
        musicgen_diffusion_loaded = ok;
    }

    if(ok)
    {
        printf("\nMusic Gen Load Complete.\n");
    }
    return ok;
}

static std::string audiocpp_artifact_text(const engine::runtime::VoiceArtifact & artifact)
{
    return std::string(
        reinterpret_cast<const char *>(artifact.payload.data()),
        artifact.payload.size());
}

static music_generation_outputs musictype_generate_yue2(const music_generation_inputs inputs)
{
    music_generation_outputs output;
    auto * offline = audiocpp_music_session
        ? dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(audiocpp_music_session.get())
        : nullptr;
    if (offline == nullptr) {
        printf("\nWarning: Audio.cpp YuE2 is not initialized.\n");
        output.status = 0;
        output.music_output_json = "";
        output.data = "";
        return output;
    }

    const auto start_time = music_clock::now();
    try {
        const std::string input_json = inputs.input_json != nullptr && inputs.input_json[0] != '\0'
            ? inputs.input_json : "{}";
        const auto root = engine::io::json::parse(input_json);
        engine::io::json::Value::Object continuation = root.as_object();

        engine::runtime::TaskRequest request;
        std::string lyrics;
        for (const auto & [key, value] : root.as_object()) {
            std::string option_key = key == "caption" ? "style" : key;
            if (option_key == "lyrics") {
                lyrics = value.as_string();
                continue;
            }
            if (value.is_string()) {
                request.options[option_key] = value.as_string();
            } else if (value.is_bool()) {
                request.options[option_key] = value.as_bool() ? "true" : "false";
            } else if (value.is_number()) {
                request.options[option_key] = engine::io::json::stringify_number(value.as_number());
            } else if (value.is_array()) {
                request.options[option_key] = engine::io::json::stringify(value);
            }
        }
        if (!lyrics.empty()) {
            request.text_input = engine::runtime::Transcript{lyrics, ""};
        }

        // Match the existing music adapter's two-phase contract. Planner mode
        // returns a self-contained request containing score and semantic codes;
        // render mode consumes that request and runs through the VAE.
        request.options["stop_after"] = inputs.is_planner_mode ? "semantic" : "audio";
        request.options["export_semantic"] = "true";

        if (!music_is_quiet) {
            printf(inputs.is_planner_mode
                ? "\nAudio.cpp YuE2 Generating Plan and Semantic Codes...\n"
                : "\nAudio.cpp YuE2 Generating Audio...\n");
        }
        auto result = offline->run(request);

        continuation["stop_after"] = engine::io::json::Value::make_string("audio");
        continuation["export_semantic"] = engine::io::json::Value::make_bool(true);
        if (continuation.find("style") == continuation.end()) {
            const auto caption = continuation.find("caption");
            if (caption != continuation.end()) {
                continuation["style"] = caption->second;
            }
        }
        for (const auto & artifact : result.output_artifacts) {
            const auto text = audiocpp_artifact_text(artifact);
            if (artifact.id == "score") {
                continuation["abc"] = engine::io::json::Value::make_string(text);
            } else if (artifact.id == "semantic") {
                continuation["semantic_prefix"] = engine::io::json::parse(text);
            }
        }
        music_output_json_str = engine::io::json::stringify(
            engine::io::json::Value::make_object(std::move(continuation)));

        if (inputs.is_planner_mode) {
            const double elapsed = music_elapsed_seconds(start_time);
            output.status = 1;
            output.music_output_json = music_output_json_str.c_str();
            output.data = "";
            printf("\nYuE2 Generated plan in %.2fs.\n", elapsed);
            return output;
        }

        if (!result.audio_output.has_value() || result.audio_output->samples.empty()) {
            throw std::runtime_error("YuE2 produced no audio");
        }
        const auto & audio = *result.audio_output;
        if (audio.channels != 1 && audio.channels != 2) {
            throw std::runtime_error("YuE2 returned an unsupported channel count");
        }

        if (audio.channels == 2 && inputs.stereo) {
            const int frames = static_cast<int>(audio.samples.size() / 2);
            std::vector<float> planar(audio.samples.size());
            for (int frame = 0; frame < frames; ++frame) {
                planar[frame] = audio.samples[2 * frame];
                planar[frames + frame] = audio.samples[2 * frame + 1];
            }
            b64_music_output = inputs.use_mp3
                ? save_stereo_mp3_base64(planar, frames, audio.sample_rate)
                : save_stereo_wav16_base64(planar, frames, audio.sample_rate);
        } else {
            std::vector<float> mono;
            if (audio.channels == 1) {
                mono = audio.samples;
            } else {
                const size_t frames = audio.samples.size() / 2;
                mono.resize(frames);
                for (size_t frame = 0; frame < frames; ++frame) {
                    mono[frame] = 0.5F * (audio.samples[2 * frame] + audio.samples[2 * frame + 1]);
                }
            }
            b64_music_output = inputs.use_mp3
                ? save_mono_mp3_base64(mono, audio.sample_rate)
                : save_wav16_base64(mono, audio.sample_rate);
        }

        output.status = 1;
        output.music_output_json = music_output_json_str.c_str();
        output.data = b64_music_output.c_str();
        const double elapsed = music_elapsed_seconds(start_time);
        printf("\nYuE2 Generated audio in %.2fs.\n", elapsed);
        return output;
    } catch (const std::exception & error) {
        printf("\nAudio.cpp YuE2 Generation Error: %s\n", error.what());
        output.status = 0;
        output.music_output_json = "";
        output.data = "";
        return output;
    }
}

music_generation_outputs musictype_generate(const music_generation_inputs inputs)
{
    if (is_audiocpp_yue2) {
        return musictype_generate_yue2(inputs);
    }

    music_generation_outputs output;

    if(!musicgen_llm_loaded && !musicgen_diffusion_loaded)
    {
        printf("\nWarning: KCPP music gen not initialized!\n");
        output.status = 0;
        output.music_output_json = "";
        output.data = "";
        return output;
    }

    const auto start_time = music_clock::now();
    if (inputs.is_planner_mode && musicgen_llm_loaded) {
        if (!music_is_quiet) {
            printf("\nMusic Gen Generating Codes...\n");
        }
        music_output_json_str = acestep_prepare_request(inputs);
        if(music_output_json_str=="")
        {
            printf("\nMusic codes generation failed!\n");
            output.status = 0;
            output.music_output_json = "";
            output.data = "";
            return output;
        }
        output.status = 1;
        output.data = "";
        output.music_output_json = music_output_json_str.c_str();
        const double elapsed = music_elapsed_seconds(start_time);
        if (!music_is_quiet) {
            printf("\nMusic Gen Codes Done:\n%s\n",music_output_json_str.c_str());
        }
        printf("\nACE-Step Generated plan in %.2fs.\n", elapsed);
    }
    else if (!inputs.is_planner_mode && musicgen_diffusion_loaded)
    {
        if (!music_is_quiet) {
            printf("\nMusic Gen Generating Audio...");
        }
        b64_music_output = acestep_generate_audio(inputs);
        if(b64_music_output=="")
        {
            printf("\nMusic audio generation failed!\n");
            output.status = 0;
            output.music_output_json = "";
            output.data = "";
            return output;
        }
        output.status = 1;
        output.data = b64_music_output.c_str();
        output.music_output_json = "";
        const double elapsed = music_elapsed_seconds(start_time);
        if (!music_is_quiet) {
            printf("\nMusic Gen Audio Done\n");
        }
        printf("\nACE-Step Generated audio in %.2fs.\n", elapsed);
    }
    else
    {
        printf("\nWarning: KCPP music gen missing requested model (Make sure it was loaded)!\n");
        output.status = 0;
        output.music_output_json = "";
        output.data = "";
        return output;
    }

    return output;
}

const char * musictype_get_family()
{
    if (is_audiocpp_yue2) {
        return "yue2";
    }
    if (musicgen_llm_loaded || musicgen_diffusion_loaded) {
        return "acestep";
    }
    return "none";
}
