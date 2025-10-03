#ifndef __QWENVL_HPP__
#define __QWENVL_HPP__

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "clip.hpp"
#include "ggml_extend.hpp"
#include <nlohmann/json.hpp>
#include "tokenize_util.h"

namespace Qwen {

    class Qwen2Tokenizer {
    private:
        std::map<int, std::u32string> byte_encoder;
        std::map<std::u32string, int> byte_decoder;
        std::map<std::u32string, int> encoder;
        std::map<int, std::u32string> decoder;
        std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
        std::regex pat;
        int encoder_len;
        int bpe_len;

    public:
        const std::string UNK_TOKEN = "<|endoftext|>";
        const std::string EOS_TOKEN = "<|endoftext|>";
        const std::string PAD_TOKEN = "<|endoftext|>";

        const int UNK_TOKEN_ID = 151643;
        const int EOS_TOKEN_ID = 151643;
        const int PAD_TOKEN_ID = 151643;

        std::vector<std::string> special_tokens = {
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<tool_call>",
            "</tool_call>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
        };

    private:
        static std::string strip(const std::string& str) {
            std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
            std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

            if (start == std::string::npos) {
                // String contains only whitespace characters
                return "";
            }

            return str.substr(start, end - start + 1);
        }

        static std::string whitespace_clean(std::string text) {
            text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
            text = strip(text);
            return text;
        }

        static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
            std::set<std::pair<std::u32string, std::u32string>> pairs;
            if (subwords.size() == 0) {
                return pairs;
            }
            std::u32string prev_subword = subwords[0];
            for (int i = 1; i < subwords.size(); i++) {
                std::u32string subword = subwords[i];
                std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
                pairs.insert(pair);
                prev_subword = subword;
            }
            return pairs;
        }

        bool is_special_token(const std::string& token) {
            for (auto& special_token : special_tokens) {
                if (special_token == token) {
                    return true;
                }
            }
            return false;
        }

    public:
        explicit Qwen2Tokenizer(const std::string& merges_utf8_str = "") {
            if (merges_utf8_str.size() > 0) {
                load_from_merges(merges_utf8_str);
            } else {
                load_from_merges(ModelLoader::load_qwen2_merges());
            }
        }

        void load_from_merges(const std::string& merges_utf8_str) {
            auto byte_unicode_pairs = bytes_to_unicode();
            // printf("byte_unicode_pairs have %lu pairs \n", byte_unicode_pairs.size());
            byte_encoder = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }
            // for (auto & pair: byte_unicode_pairs) {
            //     std::cout << pair.first << ": " << pair.second << std::endl;
            // }
            std::vector<std::u32string> merges;
            size_t start = 0;
            size_t pos;
            std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
            while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
                merges.push_back(merges_utf32_str.substr(start, pos - start));
                start = pos + 1;
            }
            LOG_DEBUG("merges size %llu", merges.size());
            merges = std::vector<std::u32string>(merges.begin(), merges.end());
            std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
            for (const auto& merge : merges) {
                size_t space_pos = merge.find(' ');
                merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
                // LOG_DEBUG("%s", utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
                // printf("%s :: %s | %s \n", utf32_to_utf8(merge).c_str(), utf32_to_utf8(merge.substr(0, space_pos)).c_str(),
                //                     utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
            }

            std::vector<std::u32string> vocab;
            for (const auto& pair : byte_unicode_pairs) {
                vocab.push_back(pair.second);
            }
            for (const auto& merge : merge_pairs) {
                vocab.push_back(merge.first + merge.second);
            }
            for (auto& special_token : special_tokens) {
                vocab.push_back(utf8_to_utf32(special_token));
            }

            LOG_DEBUG("vocab size: %llu", vocab.size());
            int i = 0;
            for (const auto& token : vocab) {
                encoder[token] = i;
                decoder[i]     = token;
                i++;
            }
            encoder_len = i;

            int rank = 0;
            for (const auto& merge : merge_pairs) {
                bpe_ranks[merge] = rank++;
            }
            bpe_len = rank;
        };

        std::u32string bpe(const std::u32string& token) {
            std::vector<std::u32string> word;

            for (int i = 0; i < token.size(); i++) {
                word.emplace_back(1, token[i]);
            }

            std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

            if (pairs.empty()) {
                return token;
            }

            while (true) {
                auto min_pair_iter = std::min_element(pairs.begin(),
                                                      pairs.end(),
                                                      [&](const std::pair<std::u32string, std::u32string>& a,
                                                          const std::pair<std::u32string, std::u32string>& b) {
                                                          if (bpe_ranks.find(a) == bpe_ranks.end()) {
                                                              return false;
                                                          } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                                                              return true;
                                                          }
                                                          return bpe_ranks.at(a) < bpe_ranks.at(b);
                                                      });

                const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

                if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                    break;
                }

                std::u32string first  = bigram.first;
                std::u32string second = bigram.second;
                std::vector<std::u32string> new_word;
                int32_t i = 0;

                while (i < word.size()) {
                    auto it = std::find(word.begin() + i, word.end(), first);
                    if (it == word.end()) {
                        new_word.insert(new_word.end(), word.begin() + i, word.end());
                        break;
                    }
                    new_word.insert(new_word.end(), word.begin() + i, it);
                    i = static_cast<int32_t>(std::distance(word.begin(), it));

                    if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                        new_word.push_back(first + second);
                        i += 2;
                    } else {
                        new_word.push_back(word[i]);
                        i += 1;
                    }
                }

                word = new_word;

                if (word.size() == 1) {
                    break;
                }
                pairs = get_pairs(word);
            }

            std::u32string result;
            for (int i = 0; i < word.size(); i++) {
                result += word[i];
                if (i != word.size() - 1) {
                    result += utf8_to_utf32(" ");
                }
            }

            return result;
        }

        std::vector<int> tokenize(std::string text,
                                  on_new_token_cb_t on_new_token_cb = nullptr,
                                  size_t max_length                 = 0,
                                  bool padding                      = false) {
            std::vector<int32_t> tokens = encode(text, on_new_token_cb);

            if (max_length > 0) {
                if (tokens.size() < max_length) {
                    tokens.resize(max_length);
                } else {
                    if (padding) {
                        tokens.insert(tokens.end(), max_length - tokens.size(), PAD_TOKEN_ID);
                    }
                }
            }

            return tokens;
        }

        void pad_tokens(std::vector<int>& tokens,
                        std::vector<float>& weights,
                        size_t max_length = 0,
                        bool padding      = false) {
            if (max_length > 0 && padding) {
                size_t n = std::ceil(tokens.size() * 1.0 / max_length);
                if (n == 0) {
                    n = 1;
                }
                size_t length = max_length * n;
                LOG_DEBUG("token length: %llu", length);
                tokens.insert(tokens.end(), length - tokens.size(), PAD_TOKEN_ID);
                weights.insert(weights.end(), length - weights.size(), 1.0);
            }
        }

        std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb = nullptr) {
            std::string original_text = text;
            std::vector<int32_t> bpe_tokens;
            std::vector<std::string> token_strs;

            auto splited_texts = split_with_special_tokens(text, special_tokens);

            for (auto& splited_text : splited_texts) {
                if (is_special_token(splited_text)) {
                    bpe_tokens.push_back(encoder[utf8_to_utf32(splited_text)]);
                    token_strs.push_back(splited_text);
                    continue;
                }
                auto tokens = token_split(splited_text);
                for (auto& token : tokens) {
                    if (on_new_token_cb != nullptr) {
                        bool skip = on_new_token_cb(token, bpe_tokens);
                        if (skip) {
                            continue;
                        }
                    }

                    std::string token_str = token;
                    std::u32string utf32_token;
                    for (int i = 0; i < token_str.length(); i++) {
                        unsigned char b = token_str[i];
                        utf32_token += byte_encoder[b];
                    }
                    auto bpe_strs = bpe(utf32_token);
                    size_t start  = 0;
                    size_t pos;
                    while ((pos = bpe_strs.find(' ', start)) != std::u32string::npos) {
                        auto bpe_str = bpe_strs.substr(start, pos - start);
                        bpe_tokens.push_back(encoder[bpe_str]);
                        token_strs.push_back(utf32_to_utf8(bpe_str));

                        start = pos + 1;
                    }
                    auto bpe_str = bpe_strs.substr(start, bpe_strs.size() - start);
                    bpe_tokens.push_back(encoder[bpe_str]);
                    token_strs.push_back(utf32_to_utf8(bpe_str));
                }
            }

            std::stringstream ss;
            ss << "[";
            for (auto token : token_strs) {
                ss << "\"" << token << "\", ";
            }
            ss << "]";
            // LOG_DEBUG("split prompt \"%s\" to tokens %s", original_text.c_str(), ss.str().c_str());
            // printf("split prompt \"%s\" to tokens %s \n", original_text.c_str(), ss.str().c_str());
            return bpe_tokens;
        }
    };

    struct Qwen2_5_VLMLP : public GGMLBlock {
    public:
        Qwen2_5_VLMLP(int64_t hidden_size, int64_t intermediate_size, bool bias = false) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, false));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, false));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, false));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto h = gate_proj->forward(ctx, x);
            h      = ggml_silu_inplace(ctx, h);
            h      = ggml_mul_inplace(ctx, h, up_proj->forward(ctx, x));
            h      = down_proj->forward(ctx, h);
            return h;
        }
    };

    struct Qwen2_5_VLAttention : public GGMLBlock {
    protected:
        int64_t head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;

    public:
        Qwen2_5_VLAttention(int64_t hidden_size,
                            int64_t num_heads,
                            int64_t num_kv_heads)
            : num_heads(num_heads), num_kv_heads(num_kv_heads) {
            head_dim = hidden_size / num_heads;
            GGML_ASSERT(num_heads * head_dim == hidden_size);
            blocks["q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_heads * head_dim));
            blocks["k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_kv_heads * head_dim));
            blocks["v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_kv_heads * head_dim));
            blocks["o_proj"] = std::shared_ptr<GGMLBlock>(new Linear(num_heads * head_dim, hidden_size, false));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    ggml_backend_t backend,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);

            auto q = q_proj->forward(ctx, x);  // [N, n_token, num_heads*head_dim]
            auto k = k_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
            auto v = v_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]

            q = ggml_reshape_4d(ctx, q, head_dim, num_heads, n_token, N);     // [N, n_token, num_heads, head_dim]
            k = ggml_reshape_4d(ctx, k, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_reshape_4d(ctx, v, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]

            int sections[4] = {16, 24, 24, 0};
            q               = ggml_rope_multi(ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            k               = ggml_rope_multi(ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);

            q = ggml_cont(ctx, ggml_torch_permute(ctx, q, 0, 2, 1, 3));            // [N, num_heads, n_token, head_dim]
            q = ggml_reshape_3d(ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);  // [N*num_heads, n_token, head_dim]

            k = ggml_cont(ctx, ggml_torch_permute(ctx, k, 0, 2, 1, 3));            // [N, num_kv_heads, n_token, head_dim]
            k = ggml_reshape_3d(ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);  // [N*num_kv_heads, n_token, head_dim]

            x = ggml_nn_attention_ext(ctx, backend, q, k, v, num_heads, nullptr, true, true, false);  // [N, n_token, hidden_size]

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct Qwen2_5_VLBlock : public GGMLBlock {
    public:
        Qwen2_5_VLBlock(int64_t hidden_size,
                        int64_t intermediate_size,
                        int64_t num_heads,
                        int64_t num_kv_heads,
                        float eps = 1e-6f) {
            blocks["self_attn"]                = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLAttention(hidden_size, num_heads, num_kv_heads));
            blocks["mlp"]                      = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLMLP(hidden_size, intermediate_size));
            blocks["input_layernorm"]          = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
            blocks["post_attention_layernorm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    ggml_backend_t backend,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos) {
            // x: [N, n_token, hidden_size]
            auto self_attn                = std::dynamic_pointer_cast<Qwen2_5_VLAttention>(blocks["self_attn"]);
            auto mlp                      = std::dynamic_pointer_cast<Qwen2_5_VLMLP>(blocks["mlp"]);
            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto post_attention_layernorm = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);

            auto residual = x;
            x             = input_layernorm->forward(ctx, x);
            x             = self_attn->forward(ctx, backend, x, input_pos);
            x             = ggml_add_inplace(ctx, x, residual);

            residual = x;
            x        = post_attention_layernorm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx, x, residual);

            return x;
        }
    };

    struct Qwen2_5_VLTextModel : public GGMLBlock {
    protected:
        int64_t num_layers;

    public:
        Qwen2_5_VLTextModel(int64_t num_layers,
                            int64_t vocab_size,
                            int64_t hidden_size,
                            int64_t intermediate_size,
                            int64_t num_heads,
                            int64_t num_kv_heads,
                            float eps = 1e-6f)
            : num_layers(num_layers) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(vocab_size, hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLBlock(hidden_size,
                                                                                                       intermediate_size,
                                                                                                       num_heads,
                                                                                                       num_kv_heads));
            }
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    ggml_backend_t backend,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos) {
            // input_ids: [N, n_token]
            // return: [N, n_token, hidden_size]

            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            auto x = embed_tokens->forward(ctx, input_ids);

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<Qwen2_5_VLBlock>(blocks["layers." + std::to_string(i)]);

                x = block->forward(ctx, backend, x, input_pos);
            }

            x = norm->forward(ctx, x);
            return x;
        }
    };

    struct Qwen2_5_VLParams {
        int64_t num_layers        = 28;
        int64_t hidden_size       = 3584;
        int64_t intermediate_size = 18944;
        int64_t num_heads         = 28;
        int64_t num_kv_heads      = 4;
        int64_t vocab_size        = 152064;
        float rms_norm_eps        = 1e-06f;
    };

    struct Qwen2_5_VL : public GGMLBlock {
        Qwen2_5_VLParams params;

    public:
        Qwen2_5_VL() {}
        Qwen2_5_VL(Qwen2_5_VLParams params)
            : params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLTextModel(params.num_layers,
                                                                                 params.vocab_size,
                                                                                 params.hidden_size,
                                                                                 params.intermediate_size,
                                                                                 params.num_heads,
                                                                                 params.num_kv_heads,
                                                                                 params.rms_norm_eps));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    ggml_backend_t backend,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos) {
            // input_ids: [N, n_token]
            auto model = std::dynamic_pointer_cast<Qwen2_5_VLTextModel>(blocks["model"]);

            auto x = model->forward(ctx, backend, input_ids, input_pos);
            return x;
        }
    };

    struct Qwen2_5_VLRunner : public GGMLRunner {
        Qwen2_5_VLParams params;
        Qwen2_5_VL model;

        std::vector<int> input_pos_vec;

        Qwen2_5_VLRunner(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2GGMLType& tensor_types,
                         const std::string prefix)
            : GGMLRunner(backend, offload_params_to_cpu) {
            model = Qwen2_5_VL(params);
            model.init(params_ctx, tensor_types, prefix);
        }

        std::string get_desc() {
            return "qwenvl2.5";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    ggml_backend_t backend,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos) {
            auto hidden_states = model.forward(ctx, backend, input_ids, input_pos);  // [N, n_token, hidden_size]
            return hidden_states;
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            input_ids = to_backend(input_ids);

            int64_t n_tokens = input_ids->ne[0];
            input_pos_vec.resize(n_tokens * 4);
            for (int i = 0; i < n_tokens; ++i) {
                input_pos_vec[i]                = i;
                input_pos_vec[n_tokens + i]     = i;
                input_pos_vec[2 * n_tokens + i] = i;
                input_pos_vec[3 * n_tokens + i] = 0;
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx,
                                                GGML_TYPE_I32,
                                                n_tokens * 4);
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            struct ggml_tensor* hidden_states = forward(compute_ctx, runtime_backend, input_ids, input_pos);

            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        void compute(const int n_threads,
                     struct ggml_tensor* input_ids,
                     ggml_tensor** output,
                     ggml_context* output_ctx = NULL) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(input_ids);
            };
            GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }
    };

    struct Qwen2_5_VLEmbedder {
        Qwen2Tokenizer tokenizer;
        Qwen2_5_VLRunner model;

        Qwen2_5_VLEmbedder(ggml_backend_t backend,
                           bool offload_params_to_cpu,
                           const String2GGMLType& tensor_types = {},
                           const std::string prefix            = "")
            : model(backend, offload_params_to_cpu, tensor_types, prefix) {
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        void alloc_params_buffer() {
            model.alloc_params_buffer();
        }

        std::tuple<std::vector<int>, std::vector<float>> tokenize(std::string text,
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
                std::vector<int> curr_tokens = tokenizer.tokenize(curr_text, nullptr);
                tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
                weights.insert(weights.end(), curr_tokens.size(), curr_weight);
            }

            tokenizer.pad_tokens(tokens, weights, max_length, padding);

            // for (int i = 0; i < tokens.size(); i++) {
            //     std::cout << tokens[i] << ":" << weights[i] << ", ";
            // }
            // std::cout << std::endl;

            return {tokens, weights};
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = NULL;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != NULL);

            {
                std::string text("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\na lovely cat<|im_end|>\n<|im_start|>assistant\n");
                auto tokens_and_weights     = tokenize(text, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
                struct ggml_tensor* out = NULL;

                int t0 = ggml_time_ms();
                model.compute(8, input_ids, &out, work_ctx);
                int t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("qwen2vl test done in %dms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cpu f16: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file(file_path, "qwen2vl.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto tensor_types = model_loader.tensor_storages_types;
            for (auto& item : tensor_types) {
                // LOG_DEBUG("%s %u", item.first.c_str(), item.second);
                if (ends_with(item.first, "weight")) {
                    item.second = model_data_type;
                }
            }

            std::shared_ptr<Qwen2_5_VLEmbedder> qwenvl = std::shared_ptr<Qwen2_5_VLEmbedder>(new Qwen2_5_VLEmbedder(backend, false, tensor_types, "qwen2vl"));

            qwenvl->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            qwenvl->get_param_tensors(tensors, "qwen2vl");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("qwenvl model loaded");
            qwenvl->test();
        }
    };

};  // Qwen

#endif  // __QWENVL_HPP__