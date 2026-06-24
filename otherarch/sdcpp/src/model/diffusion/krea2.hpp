#ifndef __SD_MODEL_DIFFUSION_KREA2_HPP__
#define __SD_MODEL_DIFFUSION_KREA2_HPP__

#include <inttypes.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/ggml_graph_cut.h"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace Krea2 {
    constexpr int KREA2_GRAPH_SIZE = 65536;

    struct Krea2Config {
        int patch_size            = 2;
        int64_t in_channels       = 16;
        int64_t out_channels      = 16;
        int64_t features          = 6144;
        int64_t timestep_dim      = 256;
        int64_t text_dim          = 2560;
        int64_t text_layers       = 12;
        int64_t layers            = 28;
        int64_t heads             = 48;
        int64_t kv_heads          = 12;
        int64_t text_heads        = 20;
        int64_t text_kv_heads     = 20;
        int64_t mlp_multiplier    = 4;
        float theta               = 1000.f;
        float norm_eps            = 1e-5f;
        std::vector<int> axes_dim = {32, 48, 48};
        int axes_dim_sum          = 128;

        int64_t head_dim() const {
            return features / heads;
        }

        static int64_t count_blocks(const String2TensorStorage& tensor_storage_map,
                                    const std::string& prefix,
                                    const std::string& block_prefix) {
            int64_t count           = 0;
            std::string full_prefix = prefix.empty() ? block_prefix : prefix + "." + block_prefix;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, full_prefix)) {
                    continue;
                }
                std::string tail = name.substr(full_prefix.size());
                size_t dot       = tail.find('.');
                if (dot == std::string::npos) {
                    continue;
                }
                int block_index = std::atoi(tail.substr(0, dot).c_str());
                count           = std::max<int64_t>(count, block_index + 1);
            }
            return count;
        }

        void update_axes_dim() {
            int64_t dim_head = head_dim();
            int64_t unit     = dim_head / 16;
            axes_dim         = {
                        static_cast<int>(dim_head - 12 * unit),
                        static_cast<int>(6 * unit),
                        static_cast<int>(6 * unit),
            };
            axes_dim_sum = axes_dim[0] + axes_dim[1] + axes_dim[2];
        }

        static Krea2Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                               const std::string& prefix) {
            Krea2Config config;
            int64_t detected_head_dim      = 0;
            int64_t detected_text_head_dim = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "first.weight") && tensor_storage.n_dims == 2) {
                    config.in_channels  = tensor_storage.ne[0] / (config.patch_size * config.patch_size);
                    config.out_channels = config.in_channels;
                    config.features     = tensor_storage.ne[1];
                } else if (ends_with(name, "blocks.0.attn.qknorm.qnorm.scale") && tensor_storage.n_dims == 1) {
                    detected_head_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "blocks.0.attn.wq.weight") && tensor_storage.n_dims == 2) {
                    if (detected_head_dim > 0) {
                        config.heads = tensor_storage.ne[1] / detected_head_dim;
                    }
                } else if (ends_with(name, "blocks.0.attn.wk.weight") && tensor_storage.n_dims == 2) {
                    if (detected_head_dim > 0) {
                        config.kv_heads = tensor_storage.ne[1] / detected_head_dim;
                    }
                } else if (ends_with(name, "txtfusion.projector.weight") && tensor_storage.n_dims == 2) {
                    config.text_layers = tensor_storage.ne[0];
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.prenorm.scale") && tensor_storage.n_dims == 1) {
                    config.text_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale") && tensor_storage.n_dims == 1) {
                    detected_text_head_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.attn.wq.weight") && tensor_storage.n_dims == 2) {
                    if (detected_text_head_dim > 0) {
                        config.text_heads = tensor_storage.ne[1] / detected_text_head_dim;
                    }
                } else if (ends_with(name, "txtfusion.layerwise_blocks.0.attn.wk.weight") && tensor_storage.n_dims == 2) {
                    if (detected_text_head_dim > 0) {
                        config.text_kv_heads = tensor_storage.ne[1] / detected_text_head_dim;
                    }
                } else if (ends_with(name, "last.linear.weight") && tensor_storage.n_dims == 2) {
                    config.out_channels = tensor_storage.ne[1] / (config.patch_size * config.patch_size);
                }
            }

            config.layers = std::max<int64_t>(1, count_blocks(tensor_storage_map, prefix, "blocks."));
            if (detected_head_dim > 0 && config.features > 0) {
                config.heads = config.features / detected_head_dim;
            }
            if (detected_head_dim > 0) {
                std::string wk_name = prefix.empty() ? "blocks.0.attn.wk.weight" : prefix + ".blocks.0.attn.wk.weight";
                auto it             = tensor_storage_map.find(wk_name);
                if (it != tensor_storage_map.end() && it->second.n_dims == 2) {
                    config.kv_heads = it->second.ne[1] / detected_head_dim;
                }
            }
            if (detected_text_head_dim > 0 && config.text_dim > 0) {
                config.text_heads = config.text_dim / detected_text_head_dim;
            }
            if (detected_text_head_dim > 0) {
                std::string wk_name = prefix.empty() ? "txtfusion.layerwise_blocks.0.attn.wk.weight" : prefix + ".txtfusion.layerwise_blocks.0.attn.wk.weight";
                auto it             = tensor_storage_map.find(wk_name);
                if (it != tensor_storage_map.end() && it->second.n_dims == 2) {
                    config.text_kv_heads = it->second.ne[1] / detected_text_head_dim;
                }
            }
            config.update_axes_dim();

            LOG_DEBUG("krea2: layers=%" PRId64 ", features=%" PRId64 ", heads=%" PRId64 ", kv_heads=%" PRId64 ", text_dim=%" PRId64 ", text_layers=%" PRId64 ", text_heads=%" PRId64 ", text_kv_heads=%" PRId64 ", channels=%" PRId64,
                      config.layers,
                      config.features,
                      config.heads,
                      config.kv_heads,
                      config.text_dim,
                      config.text_layers,
                      config.text_heads,
                      config.text_kv_heads,
                      config.in_channels);
            return config;
        }
    };

    __STATIC_INLINE__ int64_t ceil_to_multiple(int64_t value, int64_t multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    class KreaRMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;
        std::string prefix;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            this->prefix    = prefix;
            params["scale"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        }

    public:
        KreaRMSNorm(int64_t hidden_size, float eps = 1e-5f)
            : hidden_size(hidden_size),
              eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* scale = params["scale"];
            scale              = ggml_add(ctx->ggml_ctx, scale, ggml_ext_ones(ctx->ggml_ctx, scale->ne[0], 1, 1, 1));
            x                  = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            x                  = ggml_mul_inplace(ctx->ggml_ctx, x, scale);
            return x;
        }
    };

    class KreaSwiGLU : public UnaryBlock {
    public:
        KreaSwiGLU(int64_t features, int64_t multiplier) {
            int64_t mlp_dim = ceil_to_multiple(((2 * features) / 3) * multiplier, 128);
            blocks["gate"]  = std::make_shared<Linear>(features, mlp_dim, false);
            blocks["up"]    = std::make_shared<Linear>(features, mlp_dim, false);
            blocks["down"]  = std::make_shared<Linear>(mlp_dim, features, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gate = std::dynamic_pointer_cast<Linear>(blocks["gate"]);
            auto up   = std::dynamic_pointer_cast<Linear>(blocks["up"]);
            auto down = std::dynamic_pointer_cast<Linear>(blocks["down"]);

            auto gated = ggml_silu(ctx->ggml_ctx, gate->forward(ctx, x));
            auto up_x  = up->forward(ctx, x);
            x          = ggml_mul(ctx->ggml_ctx, gated, up_x);
            return down->forward(ctx, x);
        }
    };

    class KreaAttention : public GGMLBlock {
    protected:
        int64_t features;
        int64_t heads;
        int64_t kv_heads;
        int64_t head_dim_;

        ggml_tensor* attention_no_rope(GGMLRunnerContext* ctx,
                                       ggml_tensor* q,
                                       ggml_tensor* k,
                                       ggml_tensor* v,
                                       ggml_tensor* mask) {
            int64_t Lq = q->ne[2];
            int64_t Lk = k->ne[2];
            int64_t N  = q->ne[3];
            q          = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, q), head_dim_ * heads, Lq, N);
            k          = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, k), head_dim_ * kv_heads, Lk, N);
            v          = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, v), head_dim_ * kv_heads, Lk, N);
            return ggml_ext_attention_ext(ctx->ggml_ctx,
                                          ctx->backend,
                                          q,
                                          k,
                                          v,
                                          heads,
                                          mask,
                                          false,
                                          ctx->flash_attn_enabled);
        }

    public:
        KreaAttention(int64_t features,
                      int64_t heads,
                      int64_t kv_heads,
                      float eps = 1e-5f)
            : features(features),
              heads(heads),
              kv_heads(kv_heads),
              head_dim_(features / heads) {
            blocks["wq"]           = std::make_shared<Linear>(features, heads * head_dim_, false);
            blocks["wk"]           = std::make_shared<Linear>(features, kv_heads * head_dim_, false);
            blocks["wv"]           = std::make_shared<Linear>(features, kv_heads * head_dim_, false);
            blocks["gate"]         = std::make_shared<Linear>(features, features, false);
            blocks["qknorm.qnorm"] = std::make_shared<KreaRMSNorm>(head_dim_, eps);
            blocks["qknorm.knorm"] = std::make_shared<KreaRMSNorm>(head_dim_, eps);
            blocks["wo"]           = std::make_shared<Linear>(features, features, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe   = nullptr,
                             ggml_tensor* mask = nullptr) {
            auto wq    = std::dynamic_pointer_cast<Linear>(blocks["wq"]);
            auto wk    = std::dynamic_pointer_cast<Linear>(blocks["wk"]);
            auto wv    = std::dynamic_pointer_cast<Linear>(blocks["wv"]);
            auto gate  = std::dynamic_pointer_cast<Linear>(blocks["gate"]);
            auto qnorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["qknorm.qnorm"]);
            auto knorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["qknorm.knorm"]);
            auto wo    = std::dynamic_pointer_cast<Linear>(blocks["wo"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                wo->set_force_prec_f32(true);
            }

            int64_t L = x->ne[1];
            int64_t N = x->ne[2];

            auto q = wq->forward(ctx, x);
            q      = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim_, heads, L, N);
            auto k = wk->forward(ctx, x);
            k      = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim_, kv_heads, L, N);
            auto v = wv->forward(ctx, x);
            v      = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim_, kv_heads, L, N);

            q = qnorm->forward(ctx, q);
            k = knorm->forward(ctx, k);

            auto out = pe != nullptr ? Rope::attention(ctx, q, k, v, pe, mask)
                                     : attention_no_rope(ctx, q, k, v, mask);
            out      = ggml_mul(ctx->ggml_ctx, out, ggml_sigmoid(ctx->ggml_ctx, gate->forward(ctx, x)));
            out      = wo->forward(ctx, out);
            return out;
        }
    };

    class KreaDoubleSharedModulation : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            GGML_UNUSED(prefix);
            params["lin"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim * 6);
        }

    public:
        KreaDoubleSharedModulation(int64_t dim)
            : dim(dim) {}

        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            auto lin = ggml_repeat(ctx->ggml_ctx, params["lin"], vec);
            auto out = ggml_add(ctx->ggml_ctx, vec, lin);
            return ggml_ext_chunk(ctx->ggml_ctx, out, 6, 0);
        }
    };

    class KreaFinalModulation : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            GGML_UNUSED(prefix);
            params["lin"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
        }

    public:
        KreaFinalModulation(int64_t dim)
            : dim(dim) {}

        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            auto out = ggml_add(ctx->ggml_ctx, params["lin"], vec);
            return ggml_ext_chunk(ctx->ggml_ctx, out, 2, 1);
        }
    };

    class KreaTextFusionBlock : public UnaryBlock {
    public:
        KreaTextFusionBlock(int64_t dim,
                            int64_t heads,
                            int64_t kv_heads,
                            int64_t multiplier,
                            float eps) {
            blocks["prenorm"]  = std::make_shared<KreaRMSNorm>(dim, eps);
            blocks["postnorm"] = std::make_shared<KreaRMSNorm>(dim, eps);
            blocks["attn"]     = std::make_shared<KreaAttention>(dim, heads, kv_heads, eps);
            blocks["mlp"]      = std::make_shared<KreaSwiGLU>(dim, multiplier);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto prenorm  = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["prenorm"]);
            auto postnorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["postnorm"]);
            auto attn     = std::dynamic_pointer_cast<KreaAttention>(blocks["attn"]);
            auto mlp      = std::dynamic_pointer_cast<KreaSwiGLU>(blocks["mlp"]);

            x = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, prenorm->forward(ctx, x)));
            x = ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, postnorm->forward(ctx, x)));
            return x;
        }
    };

    class KreaTextFusionTransformer : public UnaryBlock {
    protected:
        Krea2Config config;

    public:
        explicit KreaTextFusionTransformer(Krea2Config config)
            : config(std::move(config)) {
            for (int i = 0; i < 2; ++i) {
                blocks["layerwise_blocks." + std::to_string(i)] = std::make_shared<KreaTextFusionBlock>(this->config.text_dim,
                                                                                                        this->config.text_heads,
                                                                                                        this->config.text_kv_heads,
                                                                                                        this->config.mlp_multiplier,
                                                                                                        this->config.norm_eps);
                blocks["refiner_blocks." + std::to_string(i)]   = std::make_shared<KreaTextFusionBlock>(this->config.text_dim,
                                                                                                      this->config.text_heads,
                                                                                                      this->config.text_kv_heads,
                                                                                                      this->config.mlp_multiplier,
                                                                                                      this->config.norm_eps);
            }
            blocks["projector"] = std::make_shared<Linear>(this->config.text_layers, 1, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* context) override {
            int64_t text_tokens = context->ne[1];
            int64_t batch       = context->ne[2];

            context = ggml_reshape_3d(ctx->ggml_ctx,
                                      context,
                                      config.text_dim,
                                      config.text_layers,
                                      text_tokens * batch);

            for (int i = 0; i < 2; ++i) {
                auto block = std::dynamic_pointer_cast<KreaTextFusionBlock>(blocks["layerwise_blocks." + std::to_string(i)]);
                context    = block->forward(ctx, context);
            }

            context        = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, context, 1, 0, 2, 3));
            auto projector = std::dynamic_pointer_cast<Linear>(blocks["projector"]);
            context        = projector->forward(ctx, context);
            context        = ggml_reshape_3d(ctx->ggml_ctx, context, config.text_dim, text_tokens, batch);

            for (int i = 0; i < 2; ++i) {
                auto block = std::dynamic_pointer_cast<KreaTextFusionBlock>(blocks["refiner_blocks." + std::to_string(i)]);
                context    = block->forward(ctx, context);
            }
            return context;
        }
    };

    class KreaSingleStreamBlock : public UnaryBlock {
    public:
        explicit KreaSingleStreamBlock(Krea2Config config) {
            blocks["mod"]      = std::make_shared<KreaDoubleSharedModulation>(config.features);
            blocks["prenorm"]  = std::make_shared<KreaRMSNorm>(config.features, config.norm_eps);
            blocks["postnorm"] = std::make_shared<KreaRMSNorm>(config.features, config.norm_eps);
            blocks["attn"]     = std::make_shared<KreaAttention>(config.features, config.heads, config.kv_heads, config.norm_eps);
            blocks["mlp"]      = std::make_shared<KreaSwiGLU>(config.features, config.mlp_multiplier);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* vec,
                             ggml_tensor* pe) {
            auto mod      = std::dynamic_pointer_cast<KreaDoubleSharedModulation>(blocks["mod"]);
            auto prenorm  = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["prenorm"]);
            auto postnorm = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["postnorm"]);
            auto attn     = std::dynamic_pointer_cast<KreaAttention>(blocks["attn"]);
            auto mlp      = std::dynamic_pointer_cast<KreaSwiGLU>(blocks["mlp"]);

            auto mods       = mod->forward(ctx, vec);
            auto attn_input = Flux::modulate(ctx->ggml_ctx,
                                             prenorm->forward(ctx, x),
                                             mods[1],
                                             mods[0],
                                             true);
            auto attn_out   = attn->forward(ctx, attn_input, pe);
            x               = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, mods[2]));

            auto mlp_input = Flux::modulate(ctx->ggml_ctx,
                                            postnorm->forward(ctx, x),
                                            mods[4],
                                            mods[3],
                                            true);
            auto mlp_out   = mlp->forward(ctx, mlp_input);
            x              = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, mods[5]));
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_UNUSED(ctx);
            GGML_UNUSED(x);
            GGML_ABORT("KreaSingleStreamBlock requires conditioning");
            return nullptr;
        }
    };

    class KreaTimeMLP : public UnaryBlock {
    public:
        explicit KreaTimeMLP(Krea2Config config) {
            blocks["0"] = std::make_shared<Linear>(config.timestep_dim, config.features, true);
            blocks["2"] = std::make_shared<Linear>(config.features, config.features, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto linear_0 = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);
            x             = linear_0->forward(ctx, x);
            x             = ggml_ext_gelu(ctx->ggml_ctx, x, false);
            x             = linear_2->forward(ctx, x);
            return x;
        }
    };

    class KreaTProj : public UnaryBlock {
    public:
        explicit KreaTProj(Krea2Config config) {
            blocks["1"] = std::make_shared<Linear>(config.features, config.features * 6, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["1"]);
            x             = ggml_ext_gelu(ctx->ggml_ctx, x, false);
            x             = linear_1->forward(ctx, x);
            return x;
        }
    };

    class KreaTextMLP : public UnaryBlock {
    public:
        explicit KreaTextMLP(Krea2Config config) {
            blocks["0"] = std::make_shared<KreaRMSNorm>(config.text_dim, config.norm_eps);
            blocks["1"] = std::make_shared<Linear>(config.text_dim, config.features, true);
            blocks["3"] = std::make_shared<Linear>(config.features, config.features, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto norm     = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["0"]);
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["1"]);
            auto linear_3 = std::dynamic_pointer_cast<Linear>(blocks["3"]);
            x             = norm->forward(ctx, x);
            x             = linear_1->forward(ctx, x);
            x             = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            x             = linear_3->forward(ctx, x);
            return x;
        }
    };

    class KreaLastLayer : public GGMLBlock {
    public:
        explicit KreaLastLayer(Krea2Config config) {
            blocks["norm"]       = std::make_shared<KreaRMSNorm>(config.features, config.norm_eps);
            blocks["linear"]     = std::make_shared<Linear>(config.features, config.patch_size * config.patch_size * config.out_channels, true);
            blocks["modulation"] = std::make_shared<KreaFinalModulation>(config.features);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* vec) {
            auto norm       = std::dynamic_pointer_cast<KreaRMSNorm>(blocks["norm"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto modulation = std::dynamic_pointer_cast<KreaFinalModulation>(blocks["modulation"]);

            auto mods = modulation->forward(ctx, vec);
            x         = Flux::modulate(ctx->ggml_ctx,
                                       norm->forward(ctx, x),
                                       mods[1],
                                       mods[0],
                                       true);
            x         = linear->forward(ctx, x);
            return x;
        }
    };

    class Krea2Model : public GGMLBlock {
    protected:
        Krea2Config config;

    public:
        Krea2Model() = default;
        explicit Krea2Model(Krea2Config config)
            : config(std::move(config)) {
            blocks["first"]     = std::make_shared<Linear>(this->config.patch_size * this->config.patch_size * this->config.in_channels,
                                                       this->config.features,
                                                       true);
            blocks["tmlp"]      = std::make_shared<KreaTimeMLP>(this->config);
            blocks["txtfusion"] = std::make_shared<KreaTextFusionTransformer>(this->config);
            blocks["txtmlp"]    = std::make_shared<KreaTextMLP>(this->config);
            blocks["tproj"]     = std::make_shared<KreaTProj>(this->config);
            for (int i = 0; i < this->config.layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<KreaSingleStreamBlock>(this->config);
            }
            blocks["last"] = std::make_shared<KreaLastLayer>(this->config);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t N = x->ne[3];
            GGML_ASSERT(N == 1);

            auto first     = std::dynamic_pointer_cast<Linear>(blocks["first"]);
            auto tmlp      = std::dynamic_pointer_cast<KreaTimeMLP>(blocks["tmlp"]);
            auto txtfusion = std::dynamic_pointer_cast<KreaTextFusionTransformer>(blocks["txtfusion"]);
            auto txtmlp    = std::dynamic_pointer_cast<KreaTextMLP>(blocks["txtmlp"]);
            auto tproj     = std::dynamic_pointer_cast<KreaTProj>(blocks["tproj"]);
            auto last      = std::dynamic_pointer_cast<KreaLastLayer>(blocks["last"]);

            auto img        = DiT::pad_and_patchify(ctx, x, config.patch_size, config.patch_size, true);
            int64_t img_len = img->ne[1];
            img             = first->forward(ctx, img);

            auto t    = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, static_cast<int>(config.timestep_dim), 10000, 1000.f);
            t         = tmlp->forward(ctx, t);
            t         = ggml_reshape_3d(ctx->ggml_ctx, t, t->ne[0], 1, t->ne[1]);
            auto tvec = tproj->forward(ctx, t);

            auto txt        = txtfusion->forward(ctx, context);
            txt             = txtmlp->forward(ctx, txt);
            int64_t txt_len = txt->ne[1];

            auto hidden_states = ggml_concat(ctx->ggml_ctx, txt, img, 1);
            for (int i = 0; i < config.layers; ++i) {
                auto block    = std::dynamic_pointer_cast<KreaSingleStreamBlock>(blocks["blocks." + std::to_string(i)]);
                hidden_states = block->forward(ctx, hidden_states, tvec, pe);
                sd::ggml_graph_cut::mark_graph_cut(hidden_states, "krea2.blocks." + std::to_string(i), "hidden_states");
            }

            hidden_states = last->forward(ctx, hidden_states, t);
            hidden_states = ggml_ext_slice(ctx->ggml_ctx, hidden_states, 1, txt_len, txt_len + img_len);
            hidden_states = DiT::unpatchify_and_crop(ctx->ggml_ctx, hidden_states, H, W, config.patch_size, config.patch_size, true);
            return hidden_states;
        }
    };

    __STATIC_INLINE__ std::vector<float> gen_krea2_pe(int h,
                                                      int w,
                                                      int patch_size,
                                                      int bs,
                                                      int context_len,
                                                      float theta,
                                                      const std::vector<int>& axes_dim) {
        auto txt_ids = Rope::gen_flux_txt_ids(bs, context_len, 3, {});
        auto img_ids = Rope::gen_flux_img_ids(h, w, patch_size, bs, 3, 0, 0, 0, false);
        auto ids     = Rope::concat_ids(txt_ids, img_ids, bs);
        return Rope::embed_nd(ids, bs, theta, axes_dim);
    }

    struct Krea2Runner : public DiffusionModelRunner {
        Krea2Config config;
        Krea2Model model;
        std::vector<float> pe_vec;

        Krea2Runner(ggml_backend_t backend,
                    const String2TensorStorage& tensor_storage_map      = {},
                    const std::string prefix                            = "",
                    std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(Krea2Config::detect_from_weights(tensor_storage_map, prefix)) {
            model = Krea2Model(config);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "krea2";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf        = new_graph_custom(KREA2_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            pe_vec      = gen_krea2_pe(static_cast<int>(x->ne[1]),
                                       static_cast<int>(x->ne[0]),
                                       config.patch_size,
                                       static_cast<int>(x->ne[3]),
                                       static_cast<int>(context->ne[1]),
                                       config.theta,
                                       config.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = model.forward(&runner_ctx, x, timesteps, context, pe);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context));
        }
    };
}  // namespace Krea2

#endif  // __SD_MODEL_DIFFUSION_KREA2_HPP__
