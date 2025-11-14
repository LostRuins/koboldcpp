#ifndef __QWENVL_HPP__
#define __QWENVL_HPP__

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "clip.hpp"
#include "ggml_extend.hpp"
#include <nlohmann/json.hpp>
#include "rope.hpp"
#include "tokenize_util.h"

namespace Qwen {
    constexpr int QWENVL_GRAPH_SIZE = 10240;

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
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto h = gate_proj->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
            h      = down_proj->forward(ctx, h);
            return h;
        }
    };

    struct Qwen2_5_VisionPatchEmbed : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int patch_size;
        int temporal_patch_size;
        int64_t in_channels;
        int64_t embed_dim;

    public:
        Qwen2_5_VisionPatchEmbed(bool llama_cpp_style,
                                 int patch_size          = 14,
                                 int temporal_patch_size = 2,
                                 int64_t in_channels     = 3,
                                 int64_t embed_dim       = 1152)
            : llama_cpp_style(llama_cpp_style),
              patch_size(patch_size),
              temporal_patch_size(temporal_patch_size),
              in_channels(in_channels),
              embed_dim(embed_dim) {
            if (llama_cpp_style) {
                blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
                blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
            } else {
                std::tuple<int, int, int> kernel_size = {(int)temporal_patch_size, (int)patch_size, (int)patch_size};
                blocks["proj"]                        = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                                              embed_dim,
                                                                                              kernel_size,
                                                                                              kernel_size,  // stride
                                                                                              {0, 0, 0},    // padding
                                                                                              {1, 1, 1},    // dilation
                                                                                              false));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [N*grid_t*grid_h*grid_w, in_channels, temporal_patch_size*patch_size*patch_size]
            // return: [N*grid_t*grid_h*grid_w, embed_dim]
            x = ggml_reshape_4d(ctx->ggml_ctx,
                                x,
                                patch_size,
                                patch_size,
                                temporal_patch_size,
                                ggml_nelements(x) / (temporal_patch_size * patch_size * patch_size));

            if (llama_cpp_style) {
                auto proj_0 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.0"]);
                auto proj_1 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.1"]);

                auto x0 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x0      = ggml_reshape_4d(ctx->ggml_ctx, x0, x0->ne[0], x0->ne[1], in_channels, x0->ne[3] / in_channels);
                x0      = proj_0->forward(ctx, x0);

                auto x1 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, 2);
                x1      = ggml_reshape_4d(ctx->ggml_ctx, x1, x1->ne[0], x1->ne[1], in_channels, x1->ne[3] / in_channels);
                x1      = proj_1->forward(ctx, x1);

                x = ggml_add(ctx->ggml_ctx, x0, x1);
            } else {
                auto proj = std::dynamic_pointer_cast<Conv3d>(blocks["proj"]);

                x = proj->forward(ctx, x);
            }

            x = ggml_reshape_2d(ctx->ggml_ctx, x, embed_dim, ggml_nelements(x) / embed_dim);
            return x;
        }
    };

    struct Qwen2_5_VLPatchMerger : public GGMLBlock {
    protected:
        int64_t hidden_size;

    public:
        Qwen2_5_VLPatchMerger(int64_t dim,
                              int64_t context_dim,
                              int64_t spatial_merge_size) {
            hidden_size     = context_dim * spatial_merge_size * spatial_merge_size;
            blocks["ln_q"]  = std::shared_ptr<GGMLBlock>(new RMSNorm(context_dim, 1e-6f));
            blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
            // mlp.1 is nn.GELU()
            blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto ln_q  = std::dynamic_pointer_cast<RMSNorm>(blocks["ln_q"]);
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            x = ln_q->forward(ctx, x);
            x = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
            x = mlp_0->forward(ctx, x);
            x = ggml_gelu(ctx->ggml_ctx, x);
            x = mlp_2->forward(ctx, x);
            return x;
        }
    };

    struct Qwen2_5_VLVisionAttention : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int64_t head_dim;
        int64_t num_heads;

    public:
        Qwen2_5_VLVisionAttention(bool llama_cpp_style,
                                  int64_t hidden_size,
                                  int64_t num_heads)
            : llama_cpp_style(llama_cpp_style), num_heads(num_heads) {
            head_dim = hidden_size / num_heads;
            GGML_ASSERT(num_heads * head_dim == hidden_size);
            if (llama_cpp_style) {
                blocks["q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
            } else {
                blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3));
            }
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto proj       = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            std::vector<ggml_tensor*> qkv_vec;
            if (llama_cpp_style) {
                auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
                auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
                auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);

                auto q = q_proj->forward(ctx, x);
                auto k = k_proj->forward(ctx, x);
                auto v = v_proj->forward(ctx, x);

                qkv_vec = {q, k, v};
            } else {
                auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
                auto qkv      = qkv_proj->forward(ctx, x);
                qkv_vec       = split_qkv(ctx->ggml_ctx, qkv);
            }

            auto q = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
            auto k = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
            auto v = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);  // [N, n_token, n_head, d_head]

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f, false);  // [N, n_token, hidden_size]

            x = proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct Qwen2_5_VLVisionBlock : public GGMLBlock {
    public:
        Qwen2_5_VLVisionBlock(bool llama_cpp_style,
                              int64_t hidden_size,
                              int64_t intermediate_size,
                              int64_t num_heads,
                              float eps = 1e-6f) {
            blocks["attn"]  = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLVisionAttention(llama_cpp_style, hidden_size, num_heads));
            blocks["mlp"]   = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLMLP(hidden_size, intermediate_size, true));
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            auto attn  = std::dynamic_pointer_cast<Qwen2_5_VLVisionAttention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<Qwen2_5_VLMLP>(blocks["mlp"]);
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);

            auto residual = x;
            x             = norm1->forward(ctx, x);
            x             = attn->forward(ctx, x, pe, mask);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = norm2->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct Qwen2_5_VLVisionModel : public GGMLBlock {
    protected:
        int64_t num_layers;
        int64_t spatial_merge_size;
        std::set<int> fullatt_block_indexes;

    public:
        Qwen2_5_VLVisionModel(bool llama_cpp_style,
                              int64_t num_layers,
                              int64_t in_channels,
                              int64_t hidden_size,
                              int64_t out_hidden_size,
                              int64_t intermediate_size,
                              int64_t num_heads,
                              int64_t spatial_merge_size,
                              int64_t patch_size,
                              int64_t temporal_patch_size,
                              int64_t window_size,
                              std::set<int> fullatt_block_indexes = {7, 15, 23, 31},
                              float eps                           = 1e-6f)
            : num_layers(num_layers), fullatt_block_indexes(std::move(fullatt_block_indexes)), spatial_merge_size(spatial_merge_size) {
            blocks["patch_embed"] = std::shared_ptr<GGMLBlock>(new Qwen2_5_VisionPatchEmbed(llama_cpp_style,
                                                                                            patch_size,
                                                                                            temporal_patch_size,
                                                                                            in_channels,
                                                                                            hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLVisionBlock(llama_cpp_style,
                                                                                                             hidden_size,
                                                                                                             intermediate_size,
                                                                                                             num_heads,
                                                                                                             eps));
            }
            blocks["merger"] = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLPatchMerger(out_hidden_size, hidden_size, spatial_merge_size));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* pixel_values,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* window_index,
                                    struct ggml_tensor* window_inverse_index,
                                    struct ggml_tensor* window_mask) {
            // pixel_values: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            // window_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_inverse_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_mask: [grid_h*grid_w, grid_h*grid_w]
            auto patch_embed = std::dynamic_pointer_cast<Qwen2_5_VisionPatchEmbed>(blocks["patch_embed"]);
            auto merger      = std::dynamic_pointer_cast<Qwen2_5_VLPatchMerger>(blocks["merger"]);

            auto x = patch_embed->forward(ctx, pixel_values);

            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] * spatial_merge_size * spatial_merge_size, x->ne[1] / spatial_merge_size / spatial_merge_size, x->ne[2], x->ne[3]);
            x = ggml_get_rows(ctx->ggml_ctx, x, window_index);
            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] / spatial_merge_size / spatial_merge_size, x->ne[1] * spatial_merge_size * spatial_merge_size, x->ne[2], x->ne[3]);

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<Qwen2_5_VLVisionBlock>(blocks["blocks." + std::to_string(i)]);

                auto mask = window_mask;
                if (fullatt_block_indexes.find(i) != fullatt_block_indexes.end()) {
                    mask = nullptr;
                }
                x = block->forward(ctx, x, pe, mask);
            }

            x = merger->forward(ctx, x);

            x = ggml_get_rows(ctx->ggml_ctx, x, window_inverse_index);

            return x;
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

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
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

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);     // [N, n_token, num_heads, head_dim]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]

            int sections[4] = {16, 24, 24, 0};
            q               = ggml_rope_multi(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            k               = ggml_rope_multi(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));  // [N, num_heads, n_token, head_dim]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);      // [N*num_heads, n_token, head_dim]

            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));  // [N, num_kv_heads, n_token, head_dim]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);      // [N*num_kv_heads, n_token, head_dim]

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, true, true, false);  // [N, n_token, hidden_size]

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

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos) {
            // x: [N, n_token, hidden_size]
            auto self_attn                = std::dynamic_pointer_cast<Qwen2_5_VLAttention>(blocks["self_attn"]);
            auto mlp                      = std::dynamic_pointer_cast<Qwen2_5_VLMLP>(blocks["mlp"]);
            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto post_attention_layernorm = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);

            auto residual = x;
            x             = input_layernorm->forward(ctx, x);
            x             = self_attn->forward(ctx, x, input_pos);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = post_attention_layernorm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

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

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    std::vector<std::pair<int, ggml_tensor*>> image_embeds) {
            // input_ids: [N, n_token]
            // return: [N, n_token, hidden_size]

            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            auto x = embed_tokens->forward(ctx, input_ids);

            if (image_embeds.size() > 0) {
                GGML_ASSERT(x->ne[2] == 1);  // N == 1

                auto raw_x              = ggml_cast(ctx->ggml_ctx, x, image_embeds[0].second->type);
                int64_t txt_token_start = 0;
                int64_t txt_token_end   = 0;

                ggml_tensor* input_embed = nullptr;

                for (int i = 0; i < image_embeds.size(); i++) {
                    if (i == 0) {
                        txt_token_start = 0;
                    } else {
                        txt_token_start = image_embeds[i - 1].first + image_embeds[i - 1].second->ne[1];
                    }
                    txt_token_end = image_embeds[i].first;

                    auto txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);
                    if (input_embed == nullptr) {
                        input_embed = txt_embed;
                    } else {
                        input_embed = ggml_concat(ctx->ggml_ctx, input_embed, txt_embed, 1);
                    }

                    auto image_embed = image_embeds[i].second;
                    input_embed      = ggml_concat(ctx->ggml_ctx, input_embed, image_embed, 1);
                }

                txt_token_start = image_embeds[image_embeds.size() - 1].first + image_embeds[image_embeds.size() - 1].second->ne[1];
                txt_token_end   = raw_x->ne[1];

                auto final_txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);

                input_embed = ggml_concat(ctx->ggml_ctx, input_embed, final_txt_embed, 1);
                GGML_ASSERT(raw_x->ne[1] == input_embed->ne[1]);

                x = input_embed;
            }

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<Qwen2_5_VLBlock>(blocks["layers." + std::to_string(i)]);

                x = block->forward(ctx, x, input_pos);
            }

            x = norm->forward(ctx, x);
            return x;
        }
    };

    struct Qwen2_5_VLVisionParams {
        int64_t num_layers                  = 32;
        int64_t hidden_size                 = 1280;
        int64_t intermediate_size           = 3420;
        int64_t num_heads                   = 16;
        int64_t in_channels                 = 3;
        int64_t out_hidden_size             = 3584;
        int64_t temporal_patch_size         = 2;
        int64_t patch_size                  = 14;
        int64_t spatial_merge_size          = 2;
        int64_t window_size                 = 112;
        std::set<int> fullatt_block_indexes = {7, 15, 23, 31};
    };

    struct Qwen2_5_VLParams {
        int64_t num_layers        = 28;
        int64_t hidden_size       = 3584;
        int64_t intermediate_size = 18944;
        int64_t num_heads         = 28;
        int64_t num_kv_heads      = 4;
        int64_t vocab_size        = 152064;
        float rms_norm_eps        = 1e-06f;
        Qwen2_5_VLVisionParams vision;
    };

    struct Qwen2_5_VL : public GGMLBlock {
        bool enable_vision;
        Qwen2_5_VLParams params;

    public:
        Qwen2_5_VL() {}
        Qwen2_5_VL(Qwen2_5_VLParams params, bool enable_vision = false, bool llama_cpp_style = false)
            : enable_vision(enable_vision), params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLTextModel(params.num_layers,
                                                                                 params.vocab_size,
                                                                                 params.hidden_size,
                                                                                 params.intermediate_size,
                                                                                 params.num_heads,
                                                                                 params.num_kv_heads,
                                                                                 params.rms_norm_eps));
            if (enable_vision) {
                blocks["visual"] = std::shared_ptr<GGMLBlock>(new Qwen2_5_VLVisionModel(llama_cpp_style,
                                                                                        params.vision.num_layers,
                                                                                        params.vision.in_channels,
                                                                                        params.vision.hidden_size,
                                                                                        params.vision.out_hidden_size,
                                                                                        params.vision.intermediate_size,
                                                                                        params.vision.num_heads,
                                                                                        params.vision.spatial_merge_size,
                                                                                        params.vision.patch_size,
                                                                                        params.vision.temporal_patch_size,
                                                                                        params.vision.window_size,
                                                                                        params.vision.fullatt_block_indexes));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    std::vector<std::pair<int, ggml_tensor*>> image_embeds) {
            // input_ids: [N, n_token]
            auto model = std::dynamic_pointer_cast<Qwen2_5_VLTextModel>(blocks["model"]);

            auto x = model->forward(ctx, input_ids, input_pos, image_embeds);
            return x;
        }

        struct ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                           struct ggml_tensor* pixel_values,
                                           struct ggml_tensor* pe,
                                           struct ggml_tensor* window_index,
                                           struct ggml_tensor* window_inverse_index,
                                           struct ggml_tensor* window_mask) {
            GGML_ASSERT(enable_vision);
            auto vision_model = std::dynamic_pointer_cast<Qwen2_5_VLVisionModel>(blocks["visual"]);
            return vision_model->forward(ctx, pixel_values, pe, window_index, window_inverse_index, window_mask);
        }
    };

    struct Qwen2_5_VLRunner : public GGMLRunner {
        Qwen2_5_VLParams params;
        bool enable_vision;
        Qwen2_5_VL model;

        std::vector<int> input_pos_vec;
        std::vector<float> window_mask_vec;
        std::vector<int> window_index_vec;
        std::vector<int> window_inverse_index_vec;
        std::vector<float> pe_vec;

        Qwen2_5_VLRunner(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix,
                         bool enable_vision_ = false)
            : GGMLRunner(backend, offload_params_to_cpu), enable_vision(enable_vision_) {
            bool have_vision_weight = false;
            bool llama_cpp_style    = false;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                size_t pos = tensor_name.find("visual.");
                if (pos != std::string::npos) {
                    have_vision_weight = true;
                    if (contains(tensor_name, "attn.q_proj")) {
                        llama_cpp_style = true;
                        break;
                    }
                }
            }
            if (enable_vision && !have_vision_weight) {
                LOG_WARN("no vision weights detected, vision disabled");
                enable_vision = false;
            }
            if (enable_vision) {
                LOG_DEBUG("enable qwen2vl vision");
                if (llama_cpp_style) {
                    LOG_DEBUG("llama.cpp style vision weight");
                }
            }
            model = Qwen2_5_VL(params, enable_vision, llama_cpp_style);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "qwenvl2.5";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    std::vector<std::pair<int, ggml_tensor*>> image_embeds) {
            auto hidden_states = model.forward(ctx, input_ids, input_pos, image_embeds);  // [N, n_token, hidden_size]
            return hidden_states;
        }

        struct ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                           struct ggml_tensor* pixel_values,
                                           struct ggml_tensor* input_pos,
                                           struct ggml_tensor* window_index,
                                           struct ggml_tensor* window_inverse_index,
                                           struct ggml_tensor* window_mask) {
            auto hidden_states = model.vision_forward(ctx, pixel_values, input_pos, window_index, window_inverse_index, window_mask);
            return hidden_states;
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids, std::vector<std::pair<int, ggml_tensor*>> image_embeds) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            input_ids = to_backend(input_ids);

            for (auto& image_embed : image_embeds) {
                image_embed.second = to_backend(image_embed.second);
            }

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

            auto runner_ctx = get_context();

            struct ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, input_pos, image_embeds);

            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        void compute(const int n_threads,
                     struct ggml_tensor* input_ids,
                     std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                     ggml_tensor** output,
                     ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(input_ids, image_embeds);
            };
            GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }

        int64_t get_num_image_tokens(int64_t t, int64_t h, int64_t w) {
            int grid_t     = 1;
            int grid_h     = h / params.vision.patch_size;
            int grid_w     = w / params.vision.patch_size;
            int llm_grid_h = grid_h / params.vision.spatial_merge_size;
            int llm_grid_w = grid_w / params.vision.spatial_merge_size;
            return grid_t * grid_h * grid_w;
        }

        struct ggml_tensor* process_image(struct ggml_context* ctx, struct ggml_tensor* image) {
            // image: [C, H, W]
            // return: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw], grid_t == 1
            int64_t C  = image->ne[2];
            int64_t H  = image->ne[1];
            int64_t W  = image->ne[0];
            int64_t mh = params.vision.spatial_merge_size;
            int64_t mw = params.vision.spatial_merge_size;
            int64_t pt = params.vision.temporal_patch_size;
            int64_t ph = params.vision.patch_size;
            int64_t pw = params.vision.patch_size;

            image = ggml_reshape_4d(ctx, image, pw, mw, (W / mw / pw), H * C);                               // [C*H, (W/mw/pw), mw, pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [mw, C*H, (W/mw/pw), pw]
            image = ggml_reshape_4d(ctx, image, pw * (W / mw / pw), H, C, mw);                               // [mw, C, H, (W/mw/pw)*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [H, mw, C, (W/mw/pw)*pw]
            image = ggml_reshape_4d(ctx, image, pw, (W / mw / pw) * C * mw, ph, mh * (H / mh / ph));         // [(H/mh/ph)*mh, ph, mw*C*(W/mw/pw), pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh, mw*C*(W/mw/pw), ph, pw]
            image = ggml_reshape_4d(ctx, image, pw * ph, (W / mw / pw), C, mw * mh * (H / mh / ph));         // [(H/mh/ph)*mh*mw, C, (W/mw/pw), ph*pw]
            image = ggml_concat(ctx, image, image, 0);                                                       // [(H/mh/ph)*mh*mw, C, (W/mw/pw), pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh*mw, (W/mw/pw), C, pt*ph*pw]
            image = ggml_reshape_4d(ctx, image, pw * ph * pt * C, (W / mw / pw), mw * mh, (H / mh / ph));    // [(H/mh/ph), mh*mw, (W/mw/pw), C*pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph), (W/mw/pw), mh*mw, C*pt*ph*pw]
            image = ggml_reshape_2d(ctx, image, pw * ph * pt * C, mw * mh * (W / mw / pw) * (H / mh / ph));  // [(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            return image;
        }

        struct ggml_cgraph* build_encode_image_graph(struct ggml_tensor* image) {
            struct ggml_cgraph* gf = new_graph_custom(QWENVL_GRAPH_SIZE);

            GGML_ASSERT(image->ne[1] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);
            GGML_ASSERT(image->ne[0] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);

            int grid_t                 = 1;
            int grid_h                 = image->ne[1] / params.vision.patch_size;
            int grid_w                 = image->ne[0] / params.vision.patch_size;
            int llm_grid_h             = grid_h / params.vision.spatial_merge_size;
            int llm_grid_w             = grid_w / params.vision.spatial_merge_size;
            int vit_merger_window_size = params.vision.window_size / params.vision.patch_size / params.vision.spatial_merge_size;

            image = to_backend(image);

            auto pixel_values = process_image(compute_ctx, image);

            // window index
            int inverse_index = 0;
            window_index_vec.resize(llm_grid_h * llm_grid_w);
            window_inverse_index_vec.resize(llm_grid_h * llm_grid_w);
            std::vector<int> seqlens;
            for (int ih = 0; ih < llm_grid_h; ih += vit_merger_window_size) {
                for (int iw = 0; iw < llm_grid_w; iw += vit_merger_window_size) {
                    int win_h = std::min(vit_merger_window_size, llm_grid_h - ih);
                    int win_w = std::min(vit_merger_window_size, llm_grid_w - iw);
                    for (int iy = 0; iy < win_h; iy++) {
                        for (int ix = 0; ix < win_w; ix++) {
                            int index                       = (ih + iy) * llm_grid_w + iw + ix;
                            window_index_vec[inverse_index] = index;
                            window_inverse_index_vec[index] = inverse_index;
                            inverse_index++;
                        }
                    }
                    seqlens.push_back(win_h * win_w * params.vision.spatial_merge_size * params.vision.spatial_merge_size);
                }
            }
            // printf("window_index: ");
            // for (int i : window_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            // printf("window_inverse_index: ");
            // for (int i : window_inverse_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            // printf("seqlens: ");
            // for (int i : seqlens) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            auto window_index         = ggml_new_tensor_1d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           llm_grid_h * llm_grid_w);
            auto window_inverse_index = ggml_new_tensor_1d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           llm_grid_h * llm_grid_w);
            set_backend_tensor_data(window_index, window_index_vec.data());
            set_backend_tensor_data(window_inverse_index, window_inverse_index_vec.data());

            // window mask
            int seq_window_size = (vit_merger_window_size * params.vision.spatial_merge_size) * (vit_merger_window_size * params.vision.spatial_merge_size);
            window_mask_vec.resize((grid_h * grid_w) * (grid_h * grid_w));
            int window_start_index = 0;
            for (int seq_index = 0; seq_index < seqlens.size(); seq_index++) {
                int window_end_index = window_start_index + seqlens[seq_index];
                // LOG_DEBUG("%d %d", window_start_index, window_end_index);
                GGML_ASSERT(window_end_index <= grid_h * grid_w);
                for (int i = window_start_index; i < window_end_index; i++) {
                    for (int j = 0; j < grid_h * grid_w; j++) {
                        float mask_value = -INFINITY;
                        if (j >= window_start_index && j < window_end_index) {
                            mask_value = 0;
                        }
                        GGML_ASSERT((i * (grid_h * grid_w) + j) < window_mask_vec.size());
                        window_mask_vec[i * (grid_h * grid_w) + j] = mask_value;
                    }
                }
                window_start_index = window_end_index;
                // printf("\n");
            }
            // printf("window_mask: \n");
            // for (int i = 0; i < grid_h*grid_w; i++) {
            //     for (int j = 0; j < grid_h*grid_w; j++) {
            //         printf("%f ", window_mask_vec[i * (grid_h * grid_w) + j]);
            //     }
            //     printf("\n");
            // }
            auto window_mask = ggml_new_tensor_2d(compute_ctx,
                                                  GGML_TYPE_F32,
                                                  grid_h * grid_w,
                                                  grid_h * grid_w);
            set_backend_tensor_data(window_mask, window_mask_vec.data());

            // pe
            int head_dim = params.vision.hidden_size / params.vision.num_heads;
            pe_vec       = Rope::gen_qwen2vl_pe(grid_h,
                                                grid_w,
                                                params.vision.spatial_merge_size,
                                                window_inverse_index_vec,
                                                10000.f,
                                                {head_dim / 2, head_dim / 2});
            int pos_len  = pe_vec.size() / head_dim / 2;
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            auto runnter_ctx                  = get_context();
            struct ggml_tensor* hidden_states = vision_forward(&runnter_ctx,
                                                               pixel_values,
                                                               pe,
                                                               window_index,
                                                               window_inverse_index,
                                                               window_mask);
            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        void encode_image(const int n_threads,
                          struct ggml_tensor* image,
                          ggml_tensor** output,
                          ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_encode_image_graph(image);
            };
            GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }
    };

    struct Qwen2_5_VLEmbedder {
        Qwen2Tokenizer tokenizer;
        Qwen2_5_VLRunner model;

        Qwen2_5_VLEmbedder(ggml_backend_t backend,
                           bool offload_params_to_cpu,
                           const String2TensorStorage& tensor_storage_map = {},
                           const std::string prefix                       = "",
                           bool enable_vision                             = false)
            : model(backend, offload_params_to_cpu, tensor_storage_map, prefix, enable_vision) {
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
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);
            bool test_vit              = true;
            bool test_decoder_with_vit = true;

            if (test_decoder_with_vit) {
                ggml_tensor* image_embed = nullptr;
                {
                    auto image = load_tensor_from_file(work_ctx, "qwen2vl_normalized.bin");
                    print_ggml_tensor(image, false, "image");
                    struct ggml_tensor* out = nullptr;

                    int t0 = ggml_time_ms();
                    model.encode_image(8, image, &out, work_ctx);
                    int t1 = ggml_time_ms();

                    print_ggml_tensor(out, false, "image_embed");
                    image_embed = out;
                    LOG_DEBUG("qwen2vl encode_image test done in %dms", t1 - t0);
                }

                std::string placeholder  = "<|image_pad|>";
                std::string img_prompt   = "Picture 1: <|vision_start|>";  // [24669, 220, 16, 25, 220, 151652]
                int64_t num_image_tokens = image_embed->ne[1];
                img_prompt.reserve(num_image_tokens * placeholder.size());
                for (int i = 0; i < num_image_tokens; i++) {
                    img_prompt += placeholder;
                }
                img_prompt += "<|vision_end|>";

                std::vector<std::pair<int, ggml_tensor*>> image_embeds;
                image_embeds.emplace_back(64, image_embed);

                std::string text = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                text += img_prompt;
                text += "change 'flux.cpp' to 'edit.cpp'";
                text += "<|im_end|>\n<|im_start|>assistant\n";

                auto tokens_and_weights     = tokenize(text, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
                struct ggml_tensor* out = nullptr;

                int t0 = ggml_time_ms();
                model.compute(8, input_ids, image_embeds, &out, work_ctx);
                int t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("qwen2vl test done in %dms", t1 - t0);
            } else if (test_vit) {
                // auto image = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 280, 280, 3);
                // ggml_set_f32(image, 0.f);
                auto image = load_tensor_from_file(work_ctx, "qwen2vl_normalized.bin");
                print_ggml_tensor(image, false, "image");
                struct ggml_tensor* out = nullptr;

                int t0 = ggml_time_ms();
                model.encode_image(8, image, &out, work_ctx);
                int t1 = ggml_time_ms();

                print_ggml_tensor(out, false, "out");

                // auto ref_out = load_tensor_from_file(work_ctx, "qwen2vl.bin");
                // ggml_ext_tensor_diff(ref_out, out, 0.01f);

                LOG_DEBUG("qwen2vl test done in %dms", t1 - t0);
            } else {
                std::string text("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\na lovely cat<|im_end|>\n<|im_start|>assistant\n");
                auto tokens_and_weights     = tokenize(text, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
                struct ggml_tensor* out = nullptr;

                int t0 = ggml_time_ms();
                model.compute(8, input_ids, {}, &out, work_ctx);
                int t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("qwen2vl test done in %dms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cpu f16: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_F16;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "qwen2vl.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            for (auto& [name, tensor_storage] : tensor_storage_map) {
                if (ends_with(name, "weight")) {
                    tensor_storage.expected_type = model_data_type;
                }
            }

            std::shared_ptr<Qwen2_5_VLEmbedder> qwenvl = std::make_shared<Qwen2_5_VLEmbedder>(backend,
                                                                                              false,
                                                                                              tensor_storage_map,
                                                                                              "qwen2vl",
                                                                                              true);

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
