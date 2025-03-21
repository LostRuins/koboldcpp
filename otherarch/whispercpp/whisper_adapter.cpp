#include "model_adapter.h"
#include "otherarch/utils.h"

#include "whisper.cpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

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

#define COMMON_SAMPLE_RATE 16000

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static int whisperdebugmode = 0;
static bool whisper_is_quiet = false;
static whisper_context * whisper_ctx = nullptr;
static std::string whisper_output_text = "";

int total_transcribe_gens = 0;

static bool is_wav_buffer(const std::string buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }
    uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    if (chunk_size + 8 != buf.size()) {
        return false;
    }
    return true;
}

static bool read_wav(const std::string & b64data, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo)
{
    drwav wav;
    std::vector<uint8_t> wav_data = kcpp_base64_decode(b64data);

    if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
        printf("error: failed to open WAV file from stdin\n");
        return false;
    }

    if (wav.channels != 1 && wav.channels != 2) {
        printf("WAV file must be mono or stereo\n");
        drwav_uninit(&wav);
        return false;
    }

    if (wav.bitsPerSample != 8 && wav.bitsPerSample != 16 && wav.bitsPerSample != 32) {
        printf("WAV file must be 8-bit, 16-bit or 32-bit. Detected: %d\n",wav.bitsPerSample);
        drwav_uninit(&wav);
        return false;
    }

    const uint64_t n = wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size()/(wav.channels*wav.bitsPerSample/8);

    std::vector<int16_t> pcm16;
    pcm16.resize(n*wav.channels);

     if (wav.bitsPerSample == 8) {
        // Handle 8-bit PCM and convert to 16-bit
        std::vector<uint8_t> pcm8(n * wav.channels);
        drwav_read_pcm_frames(&wav, n, pcm8.data());
        drwav_u8_to_s16(pcm16.data(), pcm8.data(), n * wav.channels);
    } else if (wav.bitsPerSample == 16) {
        // Handle 16-bit PCM directly
        drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    } else if (wav.bitsPerSample == 32) {
        // Handle 32-bit PCM and convert to 16-bit
        std::vector<int32_t> pcm32(n * wav.channels);
        drwav_read_pcm_frames_s32(&wav, n, pcm32.data());
        for (uint64_t i = 0; i < n * wav.channels; ++i) {
            pcm16[i] = static_cast<int16_t>(pcm32[i] >> 16); // Scale down by shifting
        }
    }
    drwav_uninit(&wav);

    std::vector<float> raw_pcm;
    raw_pcm.resize(n);

    if(whisperdebugmode==1 && !whisper_is_quiet)
    {
        printf("\nwav_data_size: %d, n:%d",wav_data.size(),n);
    }

    // convert to mono, float
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            raw_pcm[i] = float(pcm16[i])/32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            raw_pcm[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
        }
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
        if(whisperdebugmode==1 && !whisper_is_quiet)
        {
            printf("\nResample wav from %" PRIu32 " to %" PRIu32 " (in size: %zu)",
            wav.sampleRate, COMMON_SAMPLE_RATE, raw_pcm.size());
        }
        raw_pcm = resample_wav(raw_pcm, wav.sampleRate, COMMON_SAMPLE_RATE);
    }

    uint64_t finalsize = raw_pcm.size();
    pcmf32.resize(finalsize);
    for (uint64_t i = 0; i < finalsize; i++) {
        pcmf32[i] = raw_pcm[i];
    }

    return true;
}

static std::string output_txt(struct whisper_context * ctx, std::vector<std::vector<float>> pcmf32s) {

    std::string outtxt = "";
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        outtxt += text;
    }
    return outtxt;
}

void cb_log_disable(enum ggml_log_level , const char * , void * ) { }

static std::string whisperplatformenv, whisperdeviceenv, whispervulkandeviceenv;
bool whispertype_load_model(const whisper_load_model_inputs inputs)
{
    whisper_is_quiet = inputs.quiet;

    //duplicated from expose.cpp
    int cl_parseinfo = inputs.clblast_info; //first digit is whether configured, second is platform, third is devices
    std::string usingclblast = "GGML_OPENCL_CONFIGURED="+std::to_string(cl_parseinfo>0?1:0);
    putenv((char*)usingclblast.c_str());
    cl_parseinfo = cl_parseinfo%100; //keep last 2 digits
    int platform = cl_parseinfo/10;
    int devices = cl_parseinfo%10;
    whisperplatformenv = "GGML_OPENCL_PLATFORM="+std::to_string(platform);
    whisperdeviceenv = "GGML_OPENCL_DEVICE="+std::to_string(devices);
    putenv((char*)whisperplatformenv.c_str());
    putenv((char*)whisperdeviceenv.c_str());
    std::string vulkan_info_raw = inputs.vulkan_info;
    std::string vulkan_info_str = "";
    for (size_t i = 0; i < vulkan_info_raw.length(); ++i) {
        vulkan_info_str += vulkan_info_raw[i];
        if (i < vulkan_info_raw.length() - 1) {
            vulkan_info_str += ",";
        }
    }
    if(vulkan_info_str!="")
    {
        whispervulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
        putenv((char*)whispervulkandeviceenv.c_str());
    }


    std::string modelfile = inputs.model_filename;
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
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    if (!::read_wav(b64data, pcmf32, pcmf32s, false)) {
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
    whisper_output_text = output_txt(whisper_ctx, pcmf32s);
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
