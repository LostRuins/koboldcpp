#include "model_adapter.h"
#include "otherarch/utils.h"

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

static std::string music_output_json_str = "";
static std::string b64_music_output = "";

bool musictype_load_model(const music_load_model_inputs inputs)
{
    music_is_quiet = inputs.quiet;

    //duplicated from expose.cpp
    std::string vulkan_info_raw = inputs.vulkan_info;
    std::string vulkan_info_str = "";
    for (size_t i = 0; i < vulkan_info_raw.length(); ++i) {
        vulkan_info_str += vulkan_info_raw[i];
        if (i < vulkan_info_raw.length() - 1) {
            vulkan_info_str += ",";
        }
    }
    const char* existingenv = getenv("GGML_VK_VISIBLE_DEVICES");
    if(!existingenv && vulkan_info_str!="")
    {
        musicvulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
        putenv((char*)musicvulkandeviceenv.c_str());
    }

    std::string musicllm_filename = inputs.musicllm_filename;
    std::string musicembedding_filename = inputs.musicembedding_filename;
    std::string musicdiffusion_filename = inputs.musicdiffusion_filename;
    std::string musicvae_filename = inputs.musicvae_filename;
    bool lowvram = inputs.lowvram;
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

music_generation_outputs musictype_generate(const music_generation_inputs inputs)
{
    music_generation_outputs output;

    if(!musicgen_llm_loaded && !musicgen_diffusion_loaded)
    {
        printf("\nWarning: KCPP music gen not initialized!\n");
        output.status = 0;
        output.music_output_json = "";
        output.data = "";
        return output;
    }

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
        if (!music_is_quiet) {
            printf("\nMusic Gen Codes Done:\n%s\n",music_output_json_str.c_str());
        }
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
        if (!music_is_quiet) {
            printf("\nMusic Gen Audio Done\n");
        }
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
