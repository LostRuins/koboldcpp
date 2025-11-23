//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <iostream>
#include <iterator>
#include <queue>
#include <string>
#include <math.h>
#include <cstdint>
#include "expose.h"
#include "model_adapter.cpp"
#include "smart_cache.h"

// Global Smart Cache instance (initialized by Python)
SmartCache::SmartCacheManager* g_smart_cache_manager = nullptr;
SmartCache::SmartCacheMetrics g_smart_cache_metrics;

extern "C"
{

    std::string platformenv, deviceenv, vulkandeviceenv;

    //return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
    static FileFormat file_format = FileFormat::BADFORMAT;
    static FileFormatExtraMeta file_format_meta;

    bool load_model(const load_model_inputs inputs)
    {
        std::string model = inputs.model_filename;
        lora_filename = inputs.lora_filename;
        mmproj_filename = inputs.mmproj_filename;
        draftmodel_filename = inputs.draftmodel_filename;

        int forceversion = inputs.forceversion;

        file_format = check_file_format(model.c_str(),&file_format_meta);

        if(forceversion!=0)
        {
            printf("\nWARNING: FILE FORMAT FORCED TO VER %d\nIf incorrect, loading may fail or crash.\n",forceversion);
            file_format = (FileFormat)forceversion;
        }

        //first digit is whether configured, second is platform, third is devices
        int cl_parseinfo = inputs.clblast_info;

        std::string usingclblast = "GGML_OPENCL_CONFIGURED="+std::to_string(cl_parseinfo>0?1:0);
        putenv((char*)usingclblast.c_str());

        cl_parseinfo = cl_parseinfo%100; //keep last 2 digits
        int platform = cl_parseinfo/10;
        int devices = cl_parseinfo%10;
        platformenv = "GGML_OPENCL_PLATFORM="+std::to_string(platform);
        deviceenv = "GGML_OPENCL_DEVICE="+std::to_string(devices);
        putenv((char*)platformenv.c_str());
        putenv((char*)deviceenv.c_str());

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
            vulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
            putenv((char*)vulkandeviceenv.c_str());
        }

        executable_path = inputs.executable_path;

        if(file_format==FileFormat::GPTJ_1 || file_format==FileFormat::GPTJ_2 || file_format==FileFormat::GPTJ_3 || file_format==FileFormat::GPTJ_4  || file_format==FileFormat::GPTJ_5)
        {
            printf("\n---\nIdentified as Legacy GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format, file_format_meta);
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                if(file_format==FileFormat::GPTJ_1)
                {
                    //if we tried 1 first, then try 3 and lastly 2
                    //otherwise if we tried 3 first, then try 2
                    file_format = FileFormat::GPTJ_4;
                    printf("\n---\nRetrying as Legacy GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format, file_format_meta);
                }

                if (lr == ModelLoadResult::RETRY_LOAD)
                {
                    file_format = FileFormat::GPTJ_3;
                    printf("\n---\nRetrying as Legacy GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format, file_format_meta);
                }

                //lastly try format 2
                if (lr == ModelLoadResult::RETRY_LOAD)
                {
                    file_format = FileFormat::GPTJ_2;
                    printf("\n---\nRetrying as Legacy GPT-J model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format, file_format_meta);
                }
            }

            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else if(file_format==FileFormat::GPT2_1||file_format==FileFormat::GPT2_2||file_format==FileFormat::GPT2_3||file_format==FileFormat::GPT2_4)
        {
            printf("\n---\nIdentified as Legacy GPT-2 model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format, file_format_meta);
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::GPT2_3;
                printf("\n---\nRetrying as Legacy GPT-2 model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                lr = gpttype_load_model(inputs, file_format, file_format_meta);
            }
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::GPT2_2;
                printf("\n---\nRetrying as Legacy GPT-2 model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                lr = gpttype_load_model(inputs, file_format, file_format_meta);
            }
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else if(file_format==FileFormat::NEOX_1 || file_format==FileFormat::NEOX_2 || file_format==FileFormat::NEOX_3 || file_format==FileFormat::NEOX_4 || file_format==FileFormat::NEOX_5 || file_format==FileFormat::NEOX_6 || file_format==FileFormat::NEOX_7)
        {
            printf("\n---\nIdentified as Legacy GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            ModelLoadResult lr = gpttype_load_model(inputs, file_format, file_format_meta);
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                if(file_format==FileFormat::NEOX_2)
                {
                    file_format = FileFormat::NEOX_3;
                    printf("\n---\nRetrying as Legacy GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format, file_format_meta);
                }
                else
                {
                    file_format = FileFormat::NEOX_5;
                    printf("\n---\nRetrying as Legacy GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                    lr = gpttype_load_model(inputs, file_format, file_format_meta);
                }
            }
            if (lr == ModelLoadResult::RETRY_LOAD)
            {
                file_format = FileFormat::NEOX_1;
                printf("\n---\nRetrying as Legacy GPT-NEO-X model: (ver %d)\nAttempting to Load...\n---\n", file_format);
                lr = gpttype_load_model(inputs, file_format, file_format_meta);
            }
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            if(file_format==FileFormat::MPT_1)
            {
                printf("\n---\nIdentified as Legacy MPT model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            }
            else if(file_format==FileFormat::RWKV_1 || file_format==FileFormat::RWKV_2)
            {
                printf("\n---\nIdentified as Legacy RWKV model: (ver %d)\nAttempting to Load...\n---\n", file_format);
            }
            else if(file_format==FileFormat::GGUF_GENERIC)
            {
                printf("\n---\nIdentified as GGUF model.\nAttempting to Load...\n---\n", file_format);
            }
            else if(file_format==FileFormat::GGML || file_format==FileFormat::GGHF || file_format==FileFormat::GGJT || file_format==FileFormat::GGJT_2 || file_format==FileFormat::GGJT_3)
            {
                printf("\n---\nIdentified as Legacy GGML model: (ver %d)\n======\nGGML Models are Outdated: You are STRONGLY ENCOURAGED to obtain a newer GGUF model!\n======\nAttempting to Load...\n---\n", file_format);
            }
            else
            {
                printf("\n---\nUnidentified Model Encountered: (ver %d)\n---\n", file_format);
            }
            ModelLoadResult lr = gpttype_load_model(inputs, file_format, file_format_meta);
            if(file_format==FileFormat::GGML || file_format==FileFormat::GGHF || file_format==FileFormat::GGJT || file_format==FileFormat::GGJT_2 || file_format==FileFormat::GGJT_3)
            {
                //warn a second time
                printf("\n======\nGGML Models are Outdated: You are STRONGLY ENCOURAGED to obtain a newer GGUF model!\n======\n");
            }
            if (lr == ModelLoadResult::FAIL || lr == ModelLoadResult::RETRY_LOAD)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
    }

    generation_outputs generate(const generation_inputs inputs)
    {
        return gpttype_generate(inputs);
    }

    bool sd_load_model(const sd_load_model_inputs inputs)
    {
        return sdtype_load_model(inputs);
    }
    sd_generation_outputs sd_generate(const sd_generation_inputs inputs)
    {
        return sdtype_generate(inputs);
    }

    bool whisper_load_model(const whisper_load_model_inputs inputs)
    {
        return whispertype_load_model(inputs);
    }
    whisper_generation_outputs whisper_generate(const whisper_generation_inputs inputs)
    {
        return whispertype_generate(inputs);
    }

    bool tts_load_model(const tts_load_model_inputs inputs)
    {
        return ttstype_load_model(inputs);
    }
    tts_generation_outputs tts_generate(const tts_generation_inputs inputs)
    {
        return ttstype_generate(inputs);
    }

    bool embeddings_load_model(const embeddings_load_model_inputs inputs)
    {
        return embeddingstype_load_model(inputs);
    }
    embeddings_generation_outputs embeddings_generate(const embeddings_generation_inputs inputs)
    {
        return embeddingstype_generate(inputs);
    }

    const char * new_token(int idx) {
        if (generated_tokens.size() <= idx || idx < 0) return nullptr;

        return generated_tokens[idx].c_str();
    }

    int get_stream_count() {
        return generated_tokens.size();
    }

    bool has_finished() {
        return generation_finished;
    }
    bool has_audio_support()
    {
        return audio_multimodal_supported;
    }
    bool has_vision_support()
    {
        return vision_multimodal_supported;
    }
    float get_last_eval_time() {
        return last_eval_time;
    }
    float get_last_process_time() {
        return last_process_time;
    }
    int get_last_token_count() {
        return last_token_count;
    }
    int get_last_input_count() {
        return last_input_count;
    }
    int get_last_seed()
    {
        return last_seed;
    }
    int get_last_draft_success()
    {
        return last_draft_success;
    }
     int get_last_draft_failed()
    {
        return last_draft_failed;
    }
    int get_total_gens() {
        return total_gens;
    }
    int get_total_img_gens()
    {
        return total_img_gens;
    }
    int get_total_tts_gens()
    {
        return total_tts_gens;
    }
     int get_total_transcribe_gens()
    {
        return total_transcribe_gens;
    }
    int get_last_stop_reason() {
        return (int)last_stop_reason;
    }

    static std::string chat_template = "";
    const char* get_chat_template() {
        chat_template = gpttype_get_chat_template();
        return chat_template.c_str();
    }

    const char* get_pending_output() {
       return gpttype_get_pending_output().c_str();
    }

    bool abort_generate() {
        return gpttype_generate_abort();
    }

    static std::vector<int> toks; //just share a static object for token counting
    token_count_outputs token_count(const char * input, bool addbos)
    {
        std::string inputstr = input;
        token_count_outputs output;
        toks = gpttype_get_token_arr(inputstr,addbos);
        output.count = toks.size();
        output.ids = toks.data(); //this may be slightly unsafe
        return output;
    }

    static std::string detokenized_str = ""; //just share a static object for detokenizing
    const char * detokenize(const token_count_outputs input)
    {
        std::vector<int> input_arr;
        for(int i=0;i<input.count;++i)
        {
            input_arr.push_back(input.ids[i]);
        }
        detokenized_str = gpttype_detokenize(input_arr,false);
        return detokenized_str.c_str();
    }

    static std::vector<TopPicksData> last_logprob_toppicks;
    static std::vector<logprob_item> last_logprob_items;
    last_logprobs_outputs last_logprobs()
    {
        last_logprobs_outputs output;
        last_logprob_items.clear();
        last_logprob_toppicks.clear();
        last_logprob_toppicks = gpttype_get_top_picks_data(); //copy top picks
        for(int i=0;i<last_logprob_toppicks.size();++i)
        {
            logprob_item itm;
            itm.option_count = last_logprob_toppicks[i].tokenid.size();
            itm.selected_token = last_logprob_toppicks[i].selected_token.c_str();
            itm.selected_logprob = last_logprob_toppicks[i].selected_logprob;
            itm.logprobs = last_logprob_toppicks[i].logprobs.data();
            for(int j=0;j<itm.option_count && j<logprobs_max;++j)
            {
                itm.tokens[j] = last_logprob_toppicks[i].tokens[j].c_str();
            }
            last_logprob_items.push_back(itm);
        }
        output.count = last_logprob_items.size();
        output.logprob_items = last_logprob_items.data();
        return output;
    }

    size_t calc_new_state_kv() // returns how much memory a new savestate will cost
    {
        return gpttype_calc_new_state_kv();
    }
    size_t calc_new_state_tokencount()
    {
        return gpttype_calc_new_state_tokencount();
    }
    size_t calc_old_state_kv(int slot) //returns how much memory current savestate is using
    {
        return gpttype_calc_old_state_kv(slot);
    }
    size_t calc_old_state_tokencount(int slot)
    {
        return gpttype_calc_old_state_tokencount(slot);
    }
    size_t save_state_kv(int slot) //triggers the save kv state of current ctx to memory
    {
        return gpttype_save_state_kv(slot);
    }
    bool load_state_kv(int slot) //triggers the load kv state of current ctx to memory
    {
        return gpttype_load_state_kv(slot);
    }
    bool clear_state_kv()
    {
        return gpttype_clear_state_kv(true);
    }

    // =========================================================================
    // Smart Cache Functions
    // =========================================================================

    void* smart_cache_create(double max_ram_gb)
    {
        if (g_smart_cache_manager != nullptr) {
            delete g_smart_cache_manager;
        }
        g_smart_cache_manager = new SmartCache::SmartCacheManager(max_ram_gb);
        g_smart_cache_metrics.reset();
        return g_smart_cache_manager;
    }

    void smart_cache_destroy()
    {
        if (g_smart_cache_manager != nullptr) {
            delete g_smart_cache_manager;
            g_smart_cache_manager = nullptr;
        }
    }

    void smart_cache_set_enabled(bool enabled)
    {
        if (g_smart_cache_manager != nullptr) {
            g_smart_cache_manager->set_enabled(enabled);
        }
    }

    bool smart_cache_is_enabled()
    {
        return (g_smart_cache_manager != nullptr && g_smart_cache_manager->is_enabled());
    }

    int smart_cache_allocate_slot()
    {
        if (g_smart_cache_manager == nullptr) return -1;
        return g_smart_cache_manager->allocate_slot();
    }

    void smart_cache_set_active_slot(int slot_id)
    {
        if (g_smart_cache_manager != nullptr) {
            g_smart_cache_manager->set_active_slot(slot_id);
        }
    }

    void smart_cache_invalidate_slot(int slot_id)
    {
        if (g_smart_cache_manager != nullptr) {
            g_smart_cache_manager->invalidate_slot(slot_id);
        }
    }

    void smart_cache_invalidate_all()
    {
        if (g_smart_cache_manager != nullptr) {
            g_smart_cache_manager->invalidate_all();
        }
    }

    void smart_cache_save_to_slot(int slot_id, const int* tokens, size_t token_count, size_t ram_size_bytes)
    {
        if (g_smart_cache_manager == nullptr || tokens == nullptr) return;
        std::vector<int> token_vec(tokens, tokens + token_count);
        g_smart_cache_manager->save_to_slot(slot_id, token_vec, ram_size_bytes);
    }

    bool smart_cache_evict_lru_slot()
    {
        if (g_smart_cache_manager == nullptr) return false;
        return g_smart_cache_manager->evict_one_lru_slot();
    }

    void smart_cache_evict_to_fit(size_t required_bytes)
    {
        if (g_smart_cache_manager != nullptr) {
            g_smart_cache_manager->evict_lru_slots_to_fit(required_bytes);
        }
    }

    size_t smart_cache_get_total_ram_usage()
    {
        if (g_smart_cache_manager == nullptr) return 0;
        return g_smart_cache_manager->get_total_ram_usage();
    }

    int smart_cache_get_vram_slot_id()
    {
        if (g_smart_cache_manager == nullptr) return -1;
        return g_smart_cache_manager->get_vram_slot_id();
    }

    size_t smart_cache_get_slot_count()
    {
        if (g_smart_cache_manager == nullptr) return 0;
        return g_smart_cache_manager->get_slot_count();
    }

    // Search for best matching slot (returns slot_id or -1)
    // NOTE: This binding is not actively used - logic runs in C++ (gpttype_adapter.cpp)
    int smart_cache_find_best_match(
        const int* prompt_tokens,
        size_t prompt_len,
        int min_tokens,
        int genamt,
        int nctx
    )
    {
        if (g_smart_cache_manager == nullptr || prompt_tokens == nullptr) {
            return -1;
        }

        std::vector<int> prompt_vec(prompt_tokens, prompt_tokens + prompt_len);
        return g_smart_cache_manager->find_best_match(prompt_vec, min_tokens, genamt, nctx);
    }

    // Metrics
    void smart_cache_record_hit(float similarity, size_t tokens_saved)
    {
        g_smart_cache_metrics.record_ram_hit(similarity, tokens_saved);
    }

    void smart_cache_record_miss(float similarity)
    {
        g_smart_cache_metrics.record_ram_miss(similarity);
    }

    void smart_cache_record_context_switch()
    {
        g_smart_cache_metrics.record_context_switch();
    }

    void smart_cache_record_save_to_ram()
    {
        g_smart_cache_metrics.record_save_to_ram();
    }

    // Get statistics (returns JSON string - Python must free it)
    const char* smart_cache_get_stats_json()
    {
        static std::string stats_json;

        char buffer[2048];
        snprintf(buffer, sizeof(buffer),
            "{"
            "\"total_requests\":%llu,"
            "\"requests_skipped\":%llu,"
            "\"vram_reuse\":%llu,"
            "\"ram_hits\":%llu,"
            "\"ram_misses\":%llu,"
            "\"ram_hit_rate\":%.3f,"
            "\"context_switches\":%llu,"
            "\"saves_to_ram\":%llu,"
            "\"tokens_saved\":%llu,"
            "\"avg_similarity_hit\":%.3f,"
            "\"avg_similarity_miss\":%.3f,"
            "\"ram_usage_mb\":%.1f,"
            "\"total_slots\":%zu"
            "}",
            (unsigned long long)g_smart_cache_metrics.total_requests,
            (unsigned long long)g_smart_cache_metrics.requests_skipped,
            (unsigned long long)g_smart_cache_metrics.vram_reuse,
            (unsigned long long)g_smart_cache_metrics.ram_hits,
            (unsigned long long)g_smart_cache_metrics.ram_misses,
            g_smart_cache_metrics.get_ram_hit_rate(),
            (unsigned long long)g_smart_cache_metrics.context_switches,
            (unsigned long long)g_smart_cache_metrics.saves_to_ram,
            (unsigned long long)g_smart_cache_metrics.total_saved_prefill_tokens,
            g_smart_cache_metrics.get_avg_similarity_hit(),
            g_smart_cache_metrics.get_avg_similarity_miss(),
            (g_smart_cache_manager ? g_smart_cache_manager->get_total_ram_usage() / (1024.0 * 1024.0) : 0.0),
            (g_smart_cache_manager ? g_smart_cache_manager->get_slot_count() : 0)
        );

        stats_json = buffer;
        return stats_json.c_str();
    }
}
