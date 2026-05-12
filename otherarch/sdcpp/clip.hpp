#ifndef __CLIP_HPP__
#define __CLIP_HPP__

#include "ggml_extend.hpp"
#include "model.h"
#include "tokenizers/clip_tokenizer.h"

/*================================================ FrozenCLIPEmbedder ================================================*/

// Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py

struct CLIPMLP : public GGMLBlock {
protected:
    bool use_gelu;

public:
    CLIPMLP(int64_t d_model, int64_t intermediate_size) {
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(d_model, intermediate_size));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, d_model));

        if (d_model == 1024 || d_model == 1280) {  // SD 2.x
            use_gelu = true;
        } else {  // SD 1.x
            use_gelu = false;
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, n_token, d_model]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        if (use_gelu) {
            x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
        } else {
            x = ggml_ext_gelu_quick(ctx->ggml_ctx, x, true);
        }
        x = fc2->forward(ctx, x);
        return x;
    }
};

struct CLIPLayer : public GGMLBlock {
protected:
    int64_t d_model;  // hidden_size/embed_dim
    int64_t n_head;
    int64_t intermediate_size;

public:
    CLIPLayer(int64_t d_model,
              int64_t n_head,
              int64_t intermediate_size,
              bool proj_in = false)
        : d_model(d_model),
          n_head(n_head),
          intermediate_size(intermediate_size) {
        blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new MultiheadAttention(d_model, n_head, true, true, proj_in));

        blocks["layer_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));
        blocks["layer_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));

        blocks["mlp"] = std::shared_ptr<GGMLBlock>(new CLIPMLP(d_model, intermediate_size));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* mask = nullptr) {
        // x: [N, n_token, d_model]
        auto self_attn   = std::dynamic_pointer_cast<MultiheadAttention>(blocks["self_attn"]);
        auto layer_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm1"]);
        auto layer_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm2"]);
        auto mlp         = std::dynamic_pointer_cast<CLIPMLP>(blocks["mlp"]);

        x = ggml_add(ctx->ggml_ctx, x, self_attn->forward(ctx, layer_norm1->forward(ctx, x), mask));
        x = ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, layer_norm2->forward(ctx, x)));
        return x;
    }
};

struct CLIPEncoder : public GGMLBlock {
protected:
    int n_layer;

public:
    CLIPEncoder(int n_layer,
                int64_t d_model,
                int64_t n_head,
                int64_t intermediate_size,
                bool proj_in = false)
        : n_layer(n_layer) {
        for (int i = 0; i < n_layer; i++) {
            std::string name = "layers." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new CLIPLayer(d_model, n_head, intermediate_size, proj_in));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* mask                   = nullptr,
                         int clip_skip                       = -1,
                         const std::string& graph_cut_prefix = "") {
        // x: [N, n_token, d_model]
        int layer_idx = n_layer - 1;
        // LOG_DEBUG("clip_skip %d", clip_skip);
        if (clip_skip > 0) {
            layer_idx = n_layer - clip_skip;
        }

        for (int i = 0; i < n_layer; i++) {
            // LOG_DEBUG("layer %d", i);
            if (i == layer_idx + 1) {
                break;
            }
            std::string name = "layers." + std::to_string(i);
            auto layer       = std::dynamic_pointer_cast<CLIPLayer>(blocks[name]);
            x                = layer->forward(ctx, x, mask);  // [N, n_token, d_model]
            if (!graph_cut_prefix.empty()) {
                sd::ggml_graph_cut::mark_graph_cut(x, graph_cut_prefix + ".layers." + std::to_string(i), "x");
            }
            // LOG_DEBUG("layer %d", i);
        }
        return x;
    }
};

class CLIPEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t vocab_size;
    int64_t num_positions;
    bool force_clip_f32;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type token_wtype = GGML_TYPE_F32;
        if (!force_clip_f32) {
            token_wtype = get_type(prefix + "token_embedding.weight", tensor_storage_map, GGML_TYPE_F32);
            if (!support_get_rows(token_wtype)) {
                token_wtype = GGML_TYPE_F32;
            }
        }
        enum ggml_type position_wtype       = GGML_TYPE_F32;
        params["token_embedding.weight"]    = ggml_new_tensor_2d(ctx, token_wtype, embed_dim, vocab_size);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, position_wtype, embed_dim, num_positions);
    }

public:
    CLIPEmbeddings(int64_t embed_dim,
                   int64_t vocab_size    = 49408,
                   int64_t num_positions = 77,
                   bool force_clip_f32   = false)
        : embed_dim(embed_dim),
          vocab_size(vocab_size),
          num_positions(num_positions),
          force_clip_f32(force_clip_f32) {
    }

    ggml_tensor* get_token_embed_weight() {
        return params["token_embedding.weight"];
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids,
                         ggml_tensor* custom_embed_weight) {
        // input_ids: [N, n_token]
        auto token_embed_weight    = params["token_embedding.weight"];
        auto position_embed_weight = params["position_embedding.weight"];

        GGML_ASSERT(input_ids->ne[0] == position_embed_weight->ne[1]);
        input_ids            = ggml_reshape_3d(ctx->ggml_ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto token_embedding = ggml_get_rows(ctx->ggml_ctx, custom_embed_weight != nullptr ? custom_embed_weight : token_embed_weight, input_ids);
        token_embedding      = ggml_reshape_3d(ctx->ggml_ctx, token_embedding, token_embedding->ne[0], token_embedding->ne[1], token_embedding->ne[3]);

        // token_embedding + position_embedding
        auto x = ggml_add(ctx->ggml_ctx,
                          token_embedding,
                          position_embed_weight);  // [N, n_token, embed_dim]
        return x;
    }
};

class CLIPVisionEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim;
    int num_channels;
    int patch_size;
    int image_size;
    int num_patches;
    int64_t num_positions;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type patch_wtype    = GGML_TYPE_F16;
        enum ggml_type class_wtype    = GGML_TYPE_F32;
        enum ggml_type position_wtype = GGML_TYPE_F32;

        params["patch_embedding.weight"]    = ggml_new_tensor_4d(ctx, patch_wtype, patch_size, patch_size, num_channels, embed_dim);
        params["class_embedding"]           = ggml_new_tensor_1d(ctx, class_wtype, embed_dim);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, position_wtype, embed_dim, num_positions);
    }

public:
    CLIPVisionEmbeddings(int64_t embed_dim,
                         int num_channels = 3,
                         int patch_size   = 14,
                         int image_size   = 224)
        : embed_dim(embed_dim),
          num_channels(num_channels),
          patch_size(patch_size),
          image_size(image_size) {
        num_patches   = (image_size / patch_size) * (image_size / patch_size);
        num_positions = num_patches + 1;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* pixel_values) {
        // pixel_values: [N, num_channels, image_size, image_size]
        // return: [N, num_positions, embed_dim]
        GGML_ASSERT(pixel_values->ne[0] == image_size && pixel_values->ne[1] == image_size && pixel_values->ne[2] == num_channels);

        auto patch_embed_weight    = params["patch_embedding.weight"];
        auto class_embed_weight    = params["class_embedding"];
        auto position_embed_weight = params["position_embedding.weight"];

        // concat(patch_embedding, class_embedding) + position_embedding
        ggml_tensor* patch_embedding;
        int64_t N       = pixel_values->ne[3];
        patch_embedding = ggml_ext_conv_2d(ctx->ggml_ctx, pixel_values, patch_embed_weight, nullptr, patch_size, patch_size);  // [N, embed_dim, image_size // pacht_size, image_size // pacht_size]
        patch_embedding = ggml_reshape_3d(ctx->ggml_ctx, patch_embedding, num_patches, embed_dim, N);                          // [N, embed_dim, num_patches]
        patch_embedding = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, patch_embedding, 1, 0, 2, 3));                  // [N, num_patches, embed_dim]
        patch_embedding = ggml_reshape_4d(ctx->ggml_ctx, patch_embedding, 1, embed_dim, num_patches, N);                       // [N, num_patches, embed_dim, 1]

        ggml_tensor* class_embedding = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, embed_dim, N);
        class_embedding              = ggml_repeat(ctx->ggml_ctx, class_embed_weight, class_embedding);      // [N, embed_dim]
        class_embedding              = ggml_reshape_4d(ctx->ggml_ctx, class_embedding, 1, embed_dim, 1, N);  // [N, 1, embed_dim, 1]

        ggml_tensor* x = ggml_concat(ctx->ggml_ctx, class_embedding, patch_embedding, 2);  // [N, num_positions, embed_dim, 1]
        x              = ggml_reshape_3d(ctx->ggml_ctx, x, embed_dim, num_positions, N);   // [N, num_positions, embed_dim]
        x              = ggml_add(ctx->ggml_ctx, x, position_embed_weight);
        return x;  // [N, num_positions, embed_dim]
    }
};

// OPENAI_CLIP_VIT_L_14: https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
// OPEN_CLIP_VIT_H_14: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
// OPEN_CLIP_VIT_BIGG_14: https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/blob/main/config.json (CLIPTextModelWithProjection)

enum CLIPVersion {
    OPENAI_CLIP_VIT_L_14,   // SD 1.x and SDXL
    OPEN_CLIP_VIT_H_14,     // SD 2.x
    OPEN_CLIP_VIT_BIGG_14,  // SDXL
};

class CLIPTextModel : public GGMLBlock {
protected:
    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        if (version == OPEN_CLIP_VIT_BIGG_14) {
            enum ggml_type wtype      = GGML_TYPE_F32;
            params["text_projection"] = ggml_new_tensor_2d(ctx, wtype, projection_dim, hidden_size);
        }
    }

public:
    CLIPVersion version = OPENAI_CLIP_VIT_L_14;
    // network hparams
    int32_t vocab_size        = 49408;
    int32_t n_token           = 77;  // max_position_embeddings
    int32_t hidden_size       = 768;
    int32_t intermediate_size = 3072;
    int32_t n_head            = 12;
    int32_t n_layer           = 12;    // num_hidden_layers
    int32_t projection_dim    = 1280;  // only for OPEN_CLIP_VIT_BIGG_14
    bool with_final_ln        = true;

    CLIPTextModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                  bool with_final_ln  = true,
                  bool force_clip_f32 = false,
                  bool proj_in        = false)
        : version(version), with_final_ln(with_final_ln) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size       = 1024;
            intermediate_size = 4096;
            n_head            = 16;
            n_layer           = 24;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {  // CLIPTextModelWithProjection
            hidden_size       = 1280;
            intermediate_size = 5120;
            n_head            = 20;
            n_layer           = 32;
        }

        blocks["embeddings"]       = std::shared_ptr<GGMLBlock>(new CLIPEmbeddings(hidden_size, vocab_size, n_token, force_clip_f32));
        blocks["encoder"]          = std::shared_ptr<GGMLBlock>(new CLIPEncoder(n_layer, hidden_size, n_head, intermediate_size, proj_in));
        blocks["final_layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size));
    }

    ggml_tensor* get_token_embed_weight() {
        auto embeddings = std::dynamic_pointer_cast<CLIPEmbeddings>(blocks["embeddings"]);
        return embeddings->get_token_embed_weight();
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids,
                         ggml_tensor* tkn_embeddings,
                         ggml_tensor* mask    = nullptr,
                         size_t max_token_idx = 0,
                         bool return_pooled   = false,
                         int clip_skip        = -1) {
        // input_ids: [N, n_token]
        auto embeddings       = std::dynamic_pointer_cast<CLIPEmbeddings>(blocks["embeddings"]);
        auto encoder          = std::dynamic_pointer_cast<CLIPEncoder>(blocks["encoder"]);
        auto final_layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);

        auto x = embeddings->forward(ctx, input_ids, tkn_embeddings);  // [N, n_token, hidden_size]
        sd::ggml_graph_cut::mark_graph_cut(x, "clip_text.prelude", "x");
        x = encoder->forward(ctx, x, mask, return_pooled ? -1 : clip_skip, "clip_text");
        if (return_pooled || with_final_ln) {
            x = final_layer_norm->forward(ctx, x);
        }

        if (return_pooled) {
            auto text_projection = params["text_projection"];
            ggml_tensor* pooled  = ggml_view_1d(ctx->ggml_ctx, x, hidden_size, x->nb[1] * max_token_idx);
            if (text_projection != nullptr) {
                pooled = ggml_ext_linear(ctx->ggml_ctx, pooled, text_projection, nullptr);
            } else {
                LOG_DEBUG("identity projection");
            }
            return pooled;  // [hidden_size, 1, 1]
        }

        return x;  // [N, n_token, hidden_size]
    }
};

class CLIPVisionModel : public GGMLBlock {
public:
    // network hparams
    int32_t num_channels      = 3;
    int32_t patch_size        = 14;
    int32_t image_size        = 224;
    int32_t num_positions     = 257;  // (image_size / patch_size)^2 + 1
    int32_t hidden_size       = 1024;
    int32_t intermediate_size = 4096;
    int32_t n_head            = 16;
    int32_t n_layer           = 24;

public:
    CLIPVisionModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14, bool proj_in = false) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size       = 1280;
            intermediate_size = 5120;
            n_head            = 16;
            n_layer           = 32;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {
            hidden_size       = 1664;
            intermediate_size = 8192;
            n_head            = 16;
            n_layer           = 48;
        }

        blocks["embeddings"]     = std::shared_ptr<GGMLBlock>(new CLIPVisionEmbeddings(hidden_size, num_channels, patch_size, image_size));
        blocks["pre_layernorm"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size));
        blocks["encoder"]        = std::shared_ptr<GGMLBlock>(new CLIPEncoder(n_layer, hidden_size, n_head, intermediate_size, proj_in));
        blocks["post_layernorm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* pixel_values,
                         bool return_pooled = true,
                         int clip_skip      = -1) {
        // pixel_values: [N, num_channels, image_size, image_size]
        auto embeddings     = std::dynamic_pointer_cast<CLIPVisionEmbeddings>(blocks["embeddings"]);
        auto pre_layernorm  = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_layernorm"]);
        auto encoder        = std::dynamic_pointer_cast<CLIPEncoder>(blocks["encoder"]);
        auto post_layernorm = std::dynamic_pointer_cast<LayerNorm>(blocks["post_layernorm"]);

        auto x = embeddings->forward(ctx, pixel_values);  // [N, num_positions, embed_dim]
        x      = pre_layernorm->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(x, "clip_vision.prelude", "x");
        x = encoder->forward(ctx, x, nullptr, clip_skip, "clip_vision");

        auto last_hidden_state = x;

        x = post_layernorm->forward(ctx, x);  // [N, n_token, hidden_size]

        GGML_ASSERT(x->ne[3] == 1);
        if (return_pooled) {
            ggml_tensor* pooled = ggml_cont(ctx->ggml_ctx, ggml_view_2d(ctx->ggml_ctx, x, x->ne[0], x->ne[2], x->nb[2], 0));
            return pooled;  // [N, hidden_size]
        } else {
            // return x;  // [N, n_token, hidden_size]
            return last_hidden_state;  // [N, n_token, hidden_size]
        }
    }
};

class CLIPProjection : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool transpose_weight;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (transpose_weight) {
            params["weight"] = ggml_new_tensor_2d(ctx, wtype, out_features, in_features);
        } else {
            params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        }
    }

public:
    CLIPProjection(int64_t in_features,
                   int64_t out_features,
                   bool transpose_weight = false)
        : in_features(in_features),
          out_features(out_features),
          transpose_weight(transpose_weight) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* w = params["weight"];
        if (transpose_weight) {
            w = ggml_cont(ctx->ggml_ctx, ggml_transpose(ctx->ggml_ctx, w));
        }
        return ggml_ext_linear(ctx->ggml_ctx, x, w, nullptr);
    }
};

class CLIPVisionModelProjection : public GGMLBlock {
public:
    int32_t hidden_size    = 1024;
    int32_t projection_dim = 768;
    int32_t image_size     = 224;

public:
    CLIPVisionModelProjection(CLIPVersion version   = OPENAI_CLIP_VIT_L_14,
                              bool transpose_proj_w = false,
                              bool proj_in          = false) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size    = 1280;
            projection_dim = 1024;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {
            hidden_size = 1664;
        }

        blocks["vision_model"]      = std::shared_ptr<GGMLBlock>(new CLIPVisionModel(version, proj_in));
        blocks["visual_projection"] = std::shared_ptr<GGMLBlock>(new CLIPProjection(hidden_size, projection_dim, transpose_proj_w));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* pixel_values,
                         bool return_pooled = true,
                         int clip_skip      = -1) {
        // pixel_values: [N, num_channels, image_size, image_size]
        // return: [N, projection_dim] if return_pooled else [N, n_token, hidden_size]
        auto vision_model      = std::dynamic_pointer_cast<CLIPVisionModel>(blocks["vision_model"]);
        auto visual_projection = std::dynamic_pointer_cast<CLIPProjection>(blocks["visual_projection"]);

        auto x = vision_model->forward(ctx, pixel_values, return_pooled, clip_skip);  // [N, hidden_size] or [N, n_token, hidden_size]

        if (return_pooled) {
            x = visual_projection->forward(ctx, x);  // [N, projection_dim]
        }

        return x;
    }
};

struct CLIPTextModelRunner : public GGMLRunner {
    CLIPTextModel model;

    std::vector<float> attention_mask_vec;

    CLIPTextModelRunner(ggml_backend_t backend,
                        bool offload_params_to_cpu,
                        const String2TensorStorage& tensor_storage_map,
                        const std::string prefix,
                        CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                        bool with_final_ln  = true,
                        bool force_clip_f32 = false)
        : GGMLRunner(backend, offload_params_to_cpu) {
        bool proj_in = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }
            if (contains(name, "self_attn.in_proj")) {
                proj_in = true;
                break;
            }
        }
        model = CLIPTextModel(version, with_final_ln, force_clip_f32, proj_in);
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "clip";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids,
                         ggml_tensor* embeddings,
                         ggml_tensor* mask,
                         size_t max_token_idx = 0,
                         bool return_pooled   = false,
                         int clip_skip        = -1) {
        size_t N       = input_ids->ne[1];
        size_t n_token = input_ids->ne[0];
        if (input_ids->ne[0] > model.n_token) {
            GGML_ASSERT(input_ids->ne[0] % model.n_token == 0);
            input_ids = ggml_reshape_2d(ctx->ggml_ctx, input_ids, model.n_token, input_ids->ne[0] / model.n_token);
        }

        return model.forward(ctx, input_ids, embeddings, mask, max_token_idx, return_pooled, clip_skip);
    }

    ggml_cgraph* build_graph(const sd::Tensor<int32_t>& input_ids_tensor,
                             int num_custom_embeddings    = 0,
                             void* custom_embeddings_data = nullptr,
                             size_t max_token_idx         = 0,
                             bool return_pooled           = false,
                             int clip_skip                = -1) {
        ggml_cgraph* gf        = new_graph_custom(2048);
        ggml_tensor* input_ids = make_input(input_ids_tensor);

        ggml_tensor* embeddings = nullptr;

        if (num_custom_embeddings > 0 && custom_embeddings_data != nullptr) {
            auto token_embed_weight = model.get_token_embed_weight();
            auto custom_embeddings  = ggml_new_tensor_2d(compute_ctx,
                                                         token_embed_weight->type,
                                                         model.hidden_size,
                                                         num_custom_embeddings);
            set_backend_tensor_data(custom_embeddings, custom_embeddings_data);

            // concatenate custom embeddings
            embeddings = ggml_concat(compute_ctx, token_embed_weight, custom_embeddings, 1);
        }

        int n_tokens = static_cast<int>(input_ids->ne[0]);
        attention_mask_vec.resize(n_tokens * n_tokens);
        for (int i0 = 0; i0 < n_tokens; i0++) {
            for (int i1 = 0; i1 < n_tokens; i1++) {
                float value = 0.f;
                if (i0 > i1) {
                    value = -INFINITY;
                }
                attention_mask_vec[i1 * n_tokens + i0] = value;
            }
        }
        auto attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
        set_backend_tensor_data(attention_mask, attention_mask_vec.data());

        auto runner_ctx = get_context();

        ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, embeddings, attention_mask, max_token_idx, return_pooled, clip_skip);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<int32_t>& input_ids,
                              int num_custom_embeddings,
                              void* custom_embeddings_data,
                              size_t max_token_idx,
                              bool return_pooled,
                              int clip_skip) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(input_ids, num_custom_embeddings, custom_embeddings_data, max_token_idx, return_pooled, clip_skip);
        };
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        if (return_pooled) {
            return take_or_empty(std::move(result));
        }
        return restore_trailing_singleton_dims(std::move(result), 3);
    }
};

#endif  // __CLIP_HPP__
