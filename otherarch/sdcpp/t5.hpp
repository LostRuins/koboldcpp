#ifndef __T5_HPP__
#define __T5_HPP__

#include <cfloat>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ggml_extend.hpp"
#include "model.h"
#include "tokenizers/t5_unigram_tokenizer.h"

class T5LayerNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    T5LayerNorm(int64_t hidden_size,
                float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* w = params["weight"];
        x              = ggml_rms_norm(ctx->ggml_ctx, x, eps);
        x              = ggml_mul(ctx->ggml_ctx, x, w);
        return x;
    }
};

struct T5DenseActDense : public UnaryBlock {
public:
    T5DenseActDense(int64_t model_dim, int64_t ff_dim) {
        blocks["wi"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, ff_dim, false));
        blocks["wo"] = std::shared_ptr<GGMLBlock>(new Linear(ff_dim, model_dim, false));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [N, n_token, model_dim]
        auto wi = std::dynamic_pointer_cast<Linear>(blocks["wi"]);
        auto wo = std::dynamic_pointer_cast<Linear>(blocks["wo"]);

        x = wi->forward(ctx, x);
        x = ggml_relu_inplace(ctx->ggml_ctx, x);
        x = wo->forward(ctx, x);
        return x;
    }
};

struct T5DenseGatedActDense : public UnaryBlock {
public:
    T5DenseGatedActDense(int64_t model_dim, int64_t ff_dim) {
        blocks["wi_0"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, ff_dim, false));
        blocks["wi_1"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, ff_dim, false));
        float scale    = 1.f / 32.f;
        // The purpose of the scale here is to prevent NaN issues on some backends(CUDA, ...).
        blocks["wo"] = std::shared_ptr<GGMLBlock>(new Linear(ff_dim, model_dim, false, false, false, scale));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [N, n_token, model_dim]
        auto wi_0 = std::dynamic_pointer_cast<Linear>(blocks["wi_0"]);
        auto wi_1 = std::dynamic_pointer_cast<Linear>(blocks["wi_1"]);
        auto wo   = std::dynamic_pointer_cast<Linear>(blocks["wo"]);

        auto hidden_gelu   = ggml_ext_gelu(ctx->ggml_ctx, wi_0->forward(ctx, x), true);
        auto hidden_linear = wi_1->forward(ctx, x);
        x                  = ggml_mul_inplace(ctx->ggml_ctx, hidden_gelu, hidden_linear);
        x                  = wo->forward(ctx, x);
        return x;
    }
};

struct T5LayerFF : public UnaryBlock {
public:
    T5LayerFF(int64_t model_dim, int64_t ff_dim) {
        blocks["DenseReluDense"] = std::shared_ptr<GGMLBlock>(new T5DenseGatedActDense(model_dim, ff_dim));
        blocks["layer_norm"]     = std::shared_ptr<GGMLBlock>(new T5LayerNorm(model_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [N, n_token, model_dim]
        auto DenseReluDense = std::dynamic_pointer_cast<T5DenseGatedActDense>(blocks["DenseReluDense"]);
        auto layer_norm     = std::dynamic_pointer_cast<T5LayerNorm>(blocks["layer_norm"]);

        auto forwarded_states = layer_norm->forward(ctx, x);
        forwarded_states      = DenseReluDense->forward(ctx, forwarded_states);
        x                     = ggml_add_inplace(ctx->ggml_ctx, forwarded_states, x);
        return x;
    }
};

class T5Attention : public GGMLBlock {
protected:
    int64_t model_dim;
    int64_t inner_dim;
    int64_t num_heads;
    bool using_relative_attention_bias;
    int64_t relative_attention_num_buckets  = 32;
    int64_t relative_attention_max_distance = 128;

public:
    T5Attention(int64_t model_dim,
                int64_t inner_dim,
                int64_t num_heads,
                bool using_relative_attention_bias = false)
        : model_dim(model_dim),
          inner_dim(inner_dim),
          num_heads(num_heads),
          using_relative_attention_bias(using_relative_attention_bias) {
        blocks["q"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, inner_dim, false));
        blocks["k"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, inner_dim, false));
        blocks["v"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, inner_dim, false));
        blocks["o"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, model_dim, false));
        if (using_relative_attention_bias) {
            blocks["relative_attention_bias"] = std::shared_ptr<GGMLBlock>(new Embedding(relative_attention_num_buckets, num_heads));
        }
    }

    ggml_tensor* compute_bias(GGMLRunnerContext* ctx,
                              ggml_tensor* relative_position_bucket) {
        auto relative_attention_bias = std::dynamic_pointer_cast<Embedding>(blocks["relative_attention_bias"]);

        auto values = relative_attention_bias->forward(ctx, relative_position_bucket);            // shape (query_length, key_length, num_heads)
        values      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, values, 2, 0, 1, 3));  // shape (1, num_heads, query_length, key_length)
        return values;
    }

    // x: [N, n_token, model_dim]
    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                  ggml_tensor* x,
                                                  ggml_tensor* past_bias                = nullptr,
                                                  ggml_tensor* mask                     = nullptr,
                                                  ggml_tensor* relative_position_bucket = nullptr) {
        auto q_proj   = std::dynamic_pointer_cast<Linear>(blocks["q"]);
        auto k_proj   = std::dynamic_pointer_cast<Linear>(blocks["k"]);
        auto v_proj   = std::dynamic_pointer_cast<Linear>(blocks["v"]);
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);

        int64_t n_head = num_heads;
        int64_t d_head = inner_dim / n_head;

        auto q = q_proj->forward(ctx, x);
        auto k = k_proj->forward(ctx, x);
        auto v = v_proj->forward(ctx, x);

        if (using_relative_attention_bias && relative_position_bucket != nullptr) {
            past_bias = compute_bias(ctx, relative_position_bucket);
        }
        if (past_bias != nullptr) {
            if (mask != nullptr) {
                mask = ggml_repeat(ctx->ggml_ctx, mask, past_bias);
                mask = ggml_add(ctx->ggml_ctx, mask, past_bias);
            } else {
                mask = past_bias;
            }
        }

        k = ggml_ext_scale(ctx->ggml_ctx, k, ::sqrtf(static_cast<float>(d_head)), true);

        x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, mask);  // [N, n_token, d_head * n_head]

        x = out_proj->forward(ctx, x);  // [N, n_token, model_dim]
        return {x, past_bias};
    }
};

struct T5LayerSelfAttention : public GGMLBlock {
public:
    T5LayerSelfAttention(int64_t model_dim,
                         int64_t inner_dim,
                         int64_t ff_dim,
                         int64_t num_heads,
                         bool using_relative_attention_bias) {
        blocks["SelfAttention"] = std::shared_ptr<GGMLBlock>(new T5Attention(model_dim, inner_dim, num_heads, using_relative_attention_bias));
        blocks["layer_norm"]    = std::shared_ptr<GGMLBlock>(new T5LayerNorm(model_dim));
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                  ggml_tensor* x,
                                                  ggml_tensor* past_bias                = nullptr,
                                                  ggml_tensor* mask                     = nullptr,
                                                  ggml_tensor* relative_position_bucket = nullptr) {
        // x: [N, n_token, model_dim]
        auto SelfAttention = std::dynamic_pointer_cast<T5Attention>(blocks["SelfAttention"]);
        auto layer_norm    = std::dynamic_pointer_cast<T5LayerNorm>(blocks["layer_norm"]);

        auto normed_hidden_state = layer_norm->forward(ctx, x);
        auto ret                 = SelfAttention->forward(ctx, normed_hidden_state, past_bias, mask, relative_position_bucket);
        auto output              = ret.first;
        past_bias                = ret.second;

        x = ggml_add_inplace(ctx->ggml_ctx, output, x);
        return {x, past_bias};
    }
};

struct T5Block : public GGMLBlock {
public:
    T5Block(int64_t model_dim, int64_t inner_dim, int64_t ff_dim, int64_t num_heads, bool using_relative_attention_bias) {
        blocks["layer.0"] = std::shared_ptr<GGMLBlock>(new T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, using_relative_attention_bias));
        blocks["layer.1"] = std::shared_ptr<GGMLBlock>(new T5LayerFF(model_dim, ff_dim));
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                  ggml_tensor* x,
                                                  ggml_tensor* past_bias                = nullptr,
                                                  ggml_tensor* mask                     = nullptr,
                                                  ggml_tensor* relative_position_bucket = nullptr) {
        // x: [N, n_token, model_dim]
        auto layer_0 = std::dynamic_pointer_cast<T5LayerSelfAttention>(blocks["layer.0"]);
        auto layer_1 = std::dynamic_pointer_cast<T5LayerFF>(blocks["layer.1"]);

        auto ret  = layer_0->forward(ctx, x, past_bias, mask, relative_position_bucket);
        x         = ret.first;
        past_bias = ret.second;
        x         = layer_1->forward(ctx, x);
        return {x, past_bias};
    }
};

struct T5Stack : public GGMLBlock {
    int64_t num_layers;

public:
    T5Stack(int64_t num_layers,
            int64_t model_dim,
            int64_t inner_dim,
            int64_t ff_dim,
            int64_t num_heads,
            bool relative_attention = true)
        : num_layers(num_layers) {
        for (int i = 0; i < num_layers; i++) {
            blocks["block." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new T5Block(model_dim, inner_dim, ff_dim, num_heads, (!relative_attention || i == 0)));
        }

        blocks["final_layer_norm"] = std::shared_ptr<GGMLBlock>(new T5LayerNorm(model_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* past_bias                = nullptr,
                         ggml_tensor* attention_mask           = nullptr,
                         ggml_tensor* relative_position_bucket = nullptr) {
        // x: [N, n_token, model_dim]
        for (int i = 0; i < num_layers; i++) {
            auto block = std::dynamic_pointer_cast<T5Block>(blocks["block." + std::to_string(i)]);

            auto ret  = block->forward(ctx, x, past_bias, attention_mask, relative_position_bucket);
            x         = ret.first;
            past_bias = ret.second;
        }

        auto final_layer_norm = std::dynamic_pointer_cast<T5LayerNorm>(blocks["final_layer_norm"]);

        x = final_layer_norm->forward(ctx, x);
        return x;
    }
};

struct T5Params {
    int64_t num_layers      = 24;
    int64_t model_dim       = 4096;
    int64_t ff_dim          = 10240;
    int64_t num_heads       = 64;
    int64_t vocab_size      = 32128;
    bool relative_attention = true;
};

struct T5 : public GGMLBlock {
    T5Params params;

public:
    T5() {}
    T5(T5Params params)
        : params(params) {
        blocks["encoder"] = std::shared_ptr<GGMLBlock>(new T5Stack(params.num_layers,
                                                                   params.model_dim,
                                                                   params.model_dim,
                                                                   params.ff_dim,
                                                                   params.num_heads,
                                                                   params.relative_attention));
        blocks["shared"]  = std::shared_ptr<GGMLBlock>(new Embedding(params.vocab_size,
                                                                     params.model_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids,
                         ggml_tensor* past_bias                = nullptr,
                         ggml_tensor* attention_mask           = nullptr,
                         ggml_tensor* relative_position_bucket = nullptr) {
        // input_ids: [N, n_token]

        auto shared  = std::dynamic_pointer_cast<Embedding>(blocks["shared"]);
        auto encoder = std::dynamic_pointer_cast<T5Stack>(blocks["encoder"]);

        auto x = shared->forward(ctx, input_ids);
        x      = encoder->forward(ctx, x, past_bias, attention_mask, relative_position_bucket);
        return x;
    }
};

struct T5Runner : public GGMLRunner {
    T5Params params;
    T5 model;
    std::vector<int> relative_position_bucket_vec;

    T5Runner(ggml_backend_t backend,
             bool offload_params_to_cpu,
             const String2TensorStorage& tensor_storage_map,
             const std::string prefix,
             bool is_umt5 = false)
        : GGMLRunner(backend, offload_params_to_cpu) {
        if (is_umt5) {
            params.vocab_size         = 256384;
            params.relative_attention = false;
        }
        model = T5(params);
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "t5";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids,
                         ggml_tensor* relative_position_bucket,
                         ggml_tensor* attention_mask = nullptr) {
        size_t N       = input_ids->ne[1];
        size_t n_token = input_ids->ne[0];

        auto hidden_states = model.forward(ctx, input_ids, nullptr, attention_mask, relative_position_bucket);  // [N, n_token, model_dim]
        return hidden_states;
    }

    ggml_cgraph* build_graph(const sd::Tensor<int32_t>& input_ids_tensor,
                             const sd::Tensor<float>& attention_mask_tensor = {}) {
        ggml_cgraph* gf             = ggml_new_graph(compute_ctx);
        ggml_tensor* input_ids      = make_input(input_ids_tensor);
        ggml_tensor* attention_mask = attention_mask_tensor.empty() ? nullptr : make_input(attention_mask_tensor);

        relative_position_bucket_vec = compute_relative_position_bucket(static_cast<int>(input_ids->ne[0]), static_cast<int>(input_ids->ne[0]));

        // for (int i = 0; i < relative_position_bucket_vec.size(); i++) {
        //     if (i % 77 == 0) {
        //         printf("\n");
        //     }
        //     printf("%d ", relative_position_bucket_vec[i]);
        // }

        auto relative_position_bucket = ggml_new_tensor_2d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           input_ids->ne[0],
                                                           input_ids->ne[0]);
        set_backend_tensor_data(relative_position_bucket, relative_position_bucket_vec.data());

        auto runner_ctx            = get_context();
        ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, relative_position_bucket, attention_mask);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<int32_t>& input_ids,
                              const sd::Tensor<float>& attention_mask) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(input_ids, attention_mask);
        };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, true), 3);
    }

    static std::vector<int> _relative_position_bucket(const std::vector<int>& relative_position,
                                                      bool bidirectional = true,
                                                      int num_buckets    = 32,
                                                      int max_distance   = 128) {
        std::vector<int> relative_buckets(relative_position.size(), 0);
        std::vector<int> abs_relative_position = relative_position;

        if (bidirectional) {
            num_buckets = num_buckets / 2;
            for (size_t i = 0; i < relative_position.size(); ++i) {
                if (relative_position[i] > 0) {
                    relative_buckets[i] += num_buckets;
                }
                abs_relative_position[i] = std::abs(relative_position[i]);
            }
        } else {
            for (size_t i = 0; i < relative_position.size(); ++i) {
                abs_relative_position[i] = std::max(-relative_position[i], 0);
            }
        }

        int max_exact = num_buckets / 2;
        std::vector<int> relative_position_if_large(relative_position.size(), 0);

        for (size_t i = 0; i < relative_position.size(); ++i) {
            if (abs_relative_position[i] < max_exact) {
                relative_buckets[i] += abs_relative_position[i];
            } else {
                float log_pos                 = std::log(static_cast<float>(abs_relative_position[i]) / max_exact);
                float log_base                = std::log(static_cast<float>(max_distance) / max_exact);
                relative_position_if_large[i] = max_exact + static_cast<int>((log_pos / log_base) * (num_buckets - max_exact));
                relative_position_if_large[i] = std::min(relative_position_if_large[i], num_buckets - 1);
                relative_buckets[i] += relative_position_if_large[i];
            }
        }

        return relative_buckets;
    }

    std::vector<int> compute_relative_position_bucket(int query_length,
                                                      int key_length) {
        std::vector<int> context_position(query_length);
        std::vector<int> memory_position(key_length);

        for (int i = 0; i < query_length; ++i) {
            context_position[i] = i;
        }
        for (int i = 0; i < key_length; ++i) {
            memory_position[i] = i;
        }

        std::vector<std::vector<int>> relative_position(query_length, std::vector<int>(key_length, 0));
        for (int i = 0; i < query_length; ++i) {
            for (int j = 0; j < key_length; ++j) {
                relative_position[i][j] = memory_position[j] - context_position[i];
            }
        }

        std::vector<int> relative_position_bucket;
        for (int i = 0; i < query_length; ++i) {
            std::vector<int> result = _relative_position_bucket(relative_position[i], true);
            relative_position_bucket.insert(relative_position_bucket.end(), result.begin(), result.end());
        }

        return relative_position_bucket;
    }
};

struct T5Embedder {
    T5UniGramTokenizer tokenizer;
    T5Runner model;

    T5Embedder(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {},
               const std::string prefix                       = "",
               bool is_umt5                                   = false)
        : model(backend, offload_params_to_cpu, tensor_storage_map, prefix, is_umt5), tokenizer(is_umt5) {
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    void alloc_params_buffer() {
        model.alloc_params_buffer();
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> tokenize(std::string text,
                                                                                  size_t max_length = 0,
                                                                                  bool padding      = false) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        std::vector<float> attention_mask;

        tokenizer.pad_tokens(tokens, &weights, &attention_mask, padding ? max_length : 0, padding ? max_length : 100000000, padding);
        for (auto& mask_value : attention_mask) {
            mask_value = mask_value > 0.0f ? 0.0f : -HUGE_VALF;
        }

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights, attention_mask};
    }

    void test() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        {
            std::string text("a lovely cat");
            auto tokens_and_weights     = tokenize(text, 512, true);
            std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
            std::vector<float>& weights = std::get<1>(tokens_and_weights);
            std::vector<float>& masks   = std::get<2>(tokens_and_weights);
            for (auto token : tokens) {
                printf("%d ", token);
            }
            printf("\n");
            auto input_ids      = sd::Tensor<int32_t>::from_vector(tokens);
            auto attention_mask = sd::Tensor<float>::from_vector(masks);
            sd::Tensor<float> out;

            int64_t t0   = ggml_time_ms();
            auto out_opt = model.compute(8, input_ids, attention_mask);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
            LOG_DEBUG("t5 test done in %lldms", t1 - t0);
        }
    }

    static void load_from_file_and_test(const std::string& file_path) {
        // cpu f16: pass
        // cpu f32: pass
        // cuda f16: pass
        // cuda f32: pass
        // cuda q8_0: pass
        // ggml_backend_t backend = ggml_backend_cuda_init(0);
        ggml_backend_t backend    = ggml_backend_cpu_init();
        ggml_type model_data_type = GGML_TYPE_F16;

        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(file_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
            return;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();
        for (auto& [name, tensor_storage] : tensor_storage_map) {
            if (ends_with(name, "weight")) {
                tensor_storage.expected_type = model_data_type;
            }
        }

        std::shared_ptr<T5Embedder> t5 = std::make_shared<T5Embedder>(backend, false, tensor_storage_map, "", true);

        t5->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        t5->get_param_tensors(tensors, "");

        bool success = model_loader.load_tensors(tensors);

        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            return;
        }

        LOG_INFO("t5 model loaded");
        t5->test();
    }
};

#endif  // __T5_HPP__
