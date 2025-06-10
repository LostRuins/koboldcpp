#include "model_adapter.h"
#include "otherarch/utils.h"

#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "src/llama-context.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context * embeddings_ctx = nullptr; //text to codes ctx
static std::string ttsplatformenv, ttsdeviceenv, ttsvulkandeviceenv;
bool embeddings_debug = false;
static int max_batchsize = 512;
static std::string last_output = "";

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model * model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_self_clear(ctx);

    // run model
    if(embeddings_debug)
    {
        printf("\n%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    }

    // run model
    if (llama_decode(ctx, batch) < 0) {
        printf("%s : failed to process\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }
        const float * embd = nullptr;
        int embd_pos = 0;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            if(embd == NULL)
            {
                printf("\nfailed to get token embeddings\n");
            }
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
             if(embd == NULL)
            {
                printf("\nfailed to get sequence embeddings\n");
            }
        }
        float * out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);
    }
}

bool embeddingstype_load_model(const embeddings_load_model_inputs inputs)
{
    //duplicated from expose.cpp
    int cl_parseinfo = inputs.clblast_info; //first digit is whether configured, second is platform, third is devices
    std::string usingclblast = "GGML_OPENCL_CONFIGURED="+std::to_string(cl_parseinfo>0?1:0);
    putenv((char*)usingclblast.c_str());
    cl_parseinfo = cl_parseinfo%100; //keep last 2 digits
    int platform = cl_parseinfo/10;
    int devices = cl_parseinfo%10;
    ttsplatformenv = "GGML_OPENCL_PLATFORM="+std::to_string(platform);
    ttsdeviceenv = "GGML_OPENCL_DEVICE="+std::to_string(devices);
    putenv((char*)ttsplatformenv.c_str());
    putenv((char*)ttsdeviceenv.c_str());
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
        ttsvulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
        putenv((char*)ttsvulkandeviceenv.c_str());
    }

    llama_backend_init();

    std::string modelfile = inputs.model_filename;
    printf("\nLoading Embeddings Model: %s \n",modelfile.c_str());

    embeddings_debug = (inputs.debugmode>0);

    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    const int nthreads = inputs.threads;
    model_params.use_mmap = inputs.use_mmap;
    model_params.use_mlock = false;
    model_params.n_gpu_layers = inputs.gpulayers; //offload if possible
    model_params.split_mode = llama_split_mode::LLAMA_SPLIT_MODE_LAYER;

    llama_model * embeddingsmodel = llama_model_load_from_file(modelfile.c_str(), model_params);
    const int n_ctx_train = llama_model_n_ctx_train(embeddingsmodel);

    max_batchsize = (inputs.embeddingsmaxctx>0?inputs.embeddingsmaxctx:n_ctx_train);
    max_batchsize = (max_batchsize>n_ctx_train?n_ctx_train:max_batchsize);
    ctx_params.embeddings = true;
    ctx_params.n_ubatch = max_batchsize; //max size, must fit
    ctx_params.n_ctx = max_batchsize;
    ctx_params.n_batch = max_batchsize;
    ctx_params.offload_kqv = false;
    ctx_params.n_threads = nthreads;
    ctx_params.n_threads_batch = nthreads;
    ctx_params.flash_attn = inputs.flash_attention;

    embeddings_ctx = llama_init_from_model(embeddingsmodel, ctx_params);

    if (embeddings_ctx == nullptr) {
        printf("\nEmbeddings Model Load Error: Failed to initialize context!\n");
        return false;
    }

    std::vector<int> tmp = {1, 2, 3, 4};
    llama_kv_self_clear(embeddings_ctx);
    auto er = llama_decode(embeddings_ctx, llama_batch_get_one(tmp.data(), tmp.size()));
    if(er!=0)
    {
        printf("\nEmbeddings Model Eval returned nonzero: %d\n",er);
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(embeddingsmodel);

    const int n_ctx = llama_n_ctx(embeddings_ctx);

    if (llama_model_has_encoder(embeddingsmodel) && llama_model_has_decoder(embeddingsmodel)) {
        printf("\n%s: computing embeddings in encoder-decoder models is not supported\n", __func__);
        return false;
    }

    if (n_ctx > n_ctx_train) {
        printf("\n%s: warning: Embeddings model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    printf("\nEmbeddings Model Load Complete.\n");
    return true;
}

embeddings_generation_outputs embeddingstype_generate(const embeddings_generation_inputs inputs)
{
    embeddings_generation_outputs output;

    if(embeddings_ctx==nullptr)
    {
        printf("\nWarning: KCPP Embeddings Model not initialized!\n");
        output.data = "";
        output.status = 0;
        output.count = 0;
        return output;
    }

    double timetaken = 0;
    timer_start();

    llama_kv_self_clear(embeddings_ctx);
    std::string prompt = inputs.prompt;

    // max batch size
    const uint64_t n_batch = max_batchsize;

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> prompt_inputs;
    auto inp = common_tokenize(embeddings_ctx, prompt, true, true);
    if (inp.size() > n_batch) {
        if (inputs.truncate) {
            int oldsize = inp.size();
            //get bos token
            std::vector<int> bos;
            bos = common_tokenize(embeddings_ctx, "", true,true);
            int offset = inp.size() - n_batch + 1;
            inp = std::vector<int>(inp.begin() + offset, inp.end());
            //replace bos into front if exists
            if(bos.size()>0 && inp.size()>0)
            {
                inp[0] = bos[0];
            }
            if(embeddings_debug)
            {
                printf("\n%s: Input too long, truncated from %d to last %d tokens.\n", __func__,oldsize,inp.size());
            }
        } else {
            printf("\n%s: number of tokens in an input (%lld) exceeds embedding size limit for this model (%lld), lower token amount!\n",
                __func__, (long long int) inp.size(), (long long int) n_batch);
            output.data   = "";
            output.status = 0;
            output.count  = 0;
            return output;
        }
    }
    prompt_inputs.push_back(inp);

    if(embeddings_debug)
    {
        print_tok_vec(inp);
    }
    printf("\nGenerating Embeddings for %d tokens...",inp.size());

    // initialize batch
    const int n_prompts = 1;
    const enum llama_pooling_type pooling_type = llama_pooling_type(embeddings_ctx);
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // count number of embeddings
    int n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int k = 0; k < n_prompts; k++) {
            n_embd_count += prompt_inputs[k].size();
        }
    } else {
        n_embd_count = n_prompts;
    }

    // allocate output
    const llama_model * embeddingsmodel = llama_get_model(embeddings_ctx);
    const int n_embd = llama_model_n_embd(embeddingsmodel);
    std::vector<float> embeddings(n_embd_count * n_embd, 0);
    float * emb = embeddings.data();
    int embd_normalize = 2; //euclidean

    // break into batches
    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto & inp = prompt_inputs[k];
        const uint64_t n_toks = inp.size();
        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float * out = emb + e * n_embd;
            batch_decode(embeddings_ctx, batch, out, s, n_embd, embd_normalize);
            e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }
        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + e * n_embd;
    batch_decode(embeddings_ctx, batch, out, s, n_embd, embd_normalize);

    std::string outputarray = "[";
    for (int i = 0; i < n_embd; i++) {
        if (i > 0)
        {
            outputarray += ",";
        }
        outputarray += std::to_string(emb[i]);
    }
    outputarray += "]";
    last_output = outputarray;

    // clean up
    llama_batch_free(batch);

    timetaken = timer_check();
    printf("\nText Embeddings Generated %d values in %.2fs.\n",(int) n_embd,timetaken);

    output.data = last_output.c_str();
    output.status = 1;
    output.count = inp.size();
    return output;
}
