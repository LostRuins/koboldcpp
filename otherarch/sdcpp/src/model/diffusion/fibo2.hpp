#ifndef __SD_MODEL_DIFFUSION_FIBO2_HPP__
#define __SD_MODEL_DIFFUSION_FIBO2_HPP__

#include "model/diffusion/z_image.hpp"

// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_bria_fibo2.py
namespace Fibo2 {
    struct Fibo2Config {
        int64_t dim                       = 3584;
        int64_t channels                  = 128;
        int64_t text_dim                  = 7680;
        int64_t head_dim                  = 128;
        int64_t depth                     = 44;
        int64_t refiner_depth             = 2;
        int64_t perceiver_depth           = 2;
        int64_t max_gist                  = 512;
        int64_t ffn_dim                   = 9536;
        bool fused_qkv                    = false;
        bool fused_kv                     = false;
        bool fused_ffn                    = false;
        std::vector<int> injection_layers = {6, 14, 22, 30, 38};

        static Fibo2Config detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            Fibo2Config config;
            int64_t depth = 0, refiner_depth = 0, perceiver_depth = 0;
            std::set<int> injection_layers;
            for (const auto& [name, tensor] : weights) {
                if (!starts_with(name, prefix + ".")) {
                    continue;
                }
                auto key = name.substr(prefix.size() + 1);
                if (key == "x_embedder.weight") {
                    config.channels = tensor.ne[0];
                    config.dim      = tensor.ne[1];
                } else if (key == "perceiver.latent_queries") {
                    config.max_gist = tensor.ne[1];
                } else if (key == "perceiver.layers.0.cross_attention.to_k.weight" || key == "perceiver.layers.0.cross_attention.to_kv.weight") {
                    config.text_dim = tensor.ne[0];
                    config.fused_kv = key == "perceiver.layers.0.cross_attention.to_kv.weight";
                } else if (key == "layers.0.attention.norm_q.weight") {
                    config.head_dim = tensor.ne[0];
                } else if (key == "layers.0.attention.qkv.weight") {
                    config.fused_qkv = true;
                } else if (key == "layers.0.feed_forward.w1.weight") {
                    config.ffn_dim = tensor.ne[1];
                } else if (key == "layers.0.feed_forward.w13.weight") {
                    config.ffn_dim   = tensor.ne[1] / 2;
                    config.fused_ffn = true;
                }
                if (starts_with(key, "layers.")) {
                    int index = std::stoi(key.substr(7));
                    depth     = std::max<int64_t>(depth, index + 1);
                    if (ends_with(key, ".cross_attn_gate")) {
                        injection_layers.insert(index);
                    }
                } else if (starts_with(key, "noise_refiner.")) {
                    refiner_depth = std::max<int64_t>(refiner_depth, std::stoi(key.substr(14)) + 1);
                } else if (starts_with(key, "perceiver.layers.")) {
                    perceiver_depth = std::max<int64_t>(perceiver_depth, std::stoi(key.substr(17)) + 1);
                }
            }
            if (depth) {
                config.depth           = depth;
                config.refiner_depth   = refiner_depth;
                config.perceiver_depth = perceiver_depth;
                config.injection_layers.assign(injection_layers.begin(), injection_layers.end());
                LOG_VERBOSE("fibo2: dim = %" PRId64 ", depth = %" PRId64 ", head_dim = %" PRId64 ", text_dim = %" PRId64,
                            config.dim, config.depth, config.head_dim, config.text_dim);
            }
            return config;
        }

        int gist_count(int text_length) const {
            double ratio = std::clamp((text_length - 256.0) / (2048.0 - 256.0), 0.0, 1.0);
            return std::min(static_cast<int>(max_gist), static_cast<int>(std::ceil((256 + ratio * (max_gist - 256)) / 64)) * 64);
        }
    };

    struct Attention : public GGMLBlock {
        int64_t head_dim;
        int64_t heads;
        bool fused_qkv;
        bool fused_kv;

        Attention(const Fibo2Config& config, int64_t kv_dim, bool joint = false)
            : head_dim(config.head_dim), heads(config.dim / config.head_dim), fused_qkv(joint && config.fused_qkv), fused_kv(!joint && config.fused_kv) {
            if (fused_qkv) {
                blocks["qkv"] = std::make_shared<Linear>(config.dim, 3 * config.dim, false);
            } else {
                blocks["to_q"] = std::make_shared<Linear>(config.dim, config.dim, false);
                if (fused_kv) {
                    blocks["to_kv"] = std::make_shared<Linear>(kv_dim, 2 * config.dim, false);
                } else {
                    blocks["to_k"] = std::make_shared<Linear>(kv_dim, config.dim, false);
                    blocks["to_v"] = std::make_shared<Linear>(kv_dim, config.dim, false);
                }
            }
            blocks["to_out.0"] = std::make_shared<Linear>(config.dim, config.dim, false);
            blocks["norm_q"]   = std::make_shared<RMSNorm>(head_dim, 1e-5f);
            blocks["norm_k"]   = std::make_shared<RMSNorm>(head_dim, 1e-5f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* kv, ggml_tensor* pe = nullptr) {
            auto c       = ctx->ggml_ctx;
            auto project = [&](const char* name, ggml_tensor* h) {
                return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h);
            };
            ggml_tensor *q, *k, *v;
            if (fused_qkv) {
                auto parts = ggml_ext_chunk(c, project("qkv", x), 3, 0);
                q          = parts[0];
                k          = parts[1];
                v          = parts[2];
            } else {
                q = project("to_q", x);
                if (fused_kv) {
                    auto parts = ggml_ext_chunk(c, project("to_kv", kv), 2, 0);
                    k          = parts[0];
                    v          = parts[1];
                } else {
                    k = project("to_k", kv);
                    v = project("to_v", kv);
                }
            }
            auto reshape = [&](ggml_tensor* h) {
                return ggml_reshape_4d(c, h, head_dim, heads, h->ne[1], h->ne[2]);
            };
            q            = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"])->forward(ctx, reshape(q));
            k            = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"])->forward(ctx, reshape(k));
            v            = reshape(v);
            auto prepare = [&](ggml_tensor* h) {
                if (pe) {
                    return Rope::apply_rope(ctx, h, pe);
                }
                h = ggml_cont(c, ggml_permute(c, h, 0, 2, 1, 3));
                return ggml_reshape_3d(c, h, head_dim, h->ne[1], heads * h->ne[3]);
            };
            auto y = ggml_ext_attention_ext(ctx, prepare(q), prepare(k), v, heads, nullptr, true, ctx->flash_attn_enabled);
            return std::dynamic_pointer_cast<Linear>(blocks["to_out.0"])->forward(ctx, y);
        }
    };

    struct FeedForward : public ZImage::FeedForward {
        bool fused;

        FeedForward(const Fibo2Config& config)
            : ZImage::FeedForward(config.dim, config.ffn_dim, 1), fused(config.fused_ffn) {
            if (fused) {
                blocks.erase("w1");
                blocks.erase("w3");
                blocks["w13"] = std::make_shared<Linear>(config.dim, 2 * config.ffn_dim, false);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (!fused) {
                return ZImage::FeedForward::forward(ctx, x);
            }
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            if (sd_backend_is(ctx->backend, "Vulkan") || sd_backend_is(ctx->backend, "ROCm")) {
                w2->set_force_prec_f32(true);
            }
            x = std::dynamic_pointer_cast<Linear>(blocks["w13"])->forward(ctx, x);
            return w2->forward(ctx, ggml_swiglu(ctx->ggml_ctx, x));
        }
    };

    struct PerceiverLayer : public GGMLBlock {
        PerceiverLayer(const Fibo2Config& config) {
            for (const char* name : {"cross_attn_norm", "self_attn_norm", "ffn_norm"}) {
                blocks[name] = std::make_shared<RMSNorm>(config.dim, 1e-5f);
            }
            blocks["cross_attention"] = std::make_shared<Attention>(config, config.text_dim);
            blocks["self_attention"]  = std::make_shared<Attention>(config, config.dim);
            blocks["ffn"]             = std::make_shared<FeedForward>(config);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* text) {
            auto norm = [&](const char* name, ggml_tensor* h) {
                return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h);
            };
            auto h = std::dynamic_pointer_cast<Attention>(blocks["cross_attention"])->forward(ctx, norm("cross_attn_norm", x), text);
            x      = ggml_add(ctx->ggml_ctx, x, h);
            h      = norm("self_attn_norm", x);
            h      = std::dynamic_pointer_cast<Attention>(blocks["self_attention"])->forward(ctx, h, h);
            x      = ggml_add(ctx->ggml_ctx, x, h);
            h      = std::dynamic_pointer_cast<FeedForward>(blocks["ffn"])->forward(ctx, norm("ffn_norm", x));
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct Perceiver : public GGMLBlock {
        Fibo2Config config;

        Perceiver(const Fibo2Config& config)
            : config(config) {
            for (int i = 0; i < config.perceiver_depth; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<PerceiverLayer>(config);
            }
            blocks["norm_out"] = std::make_shared<RMSNorm>(config.dim, 1e-5f);
        }

        void init_params(ggml_context* ctx, const String2TensorStorage& weights = {}, const std::string prefix = "") override {
            params["latent_queries"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.dim, config.max_gist);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* text, int gist_count) {
            auto x = ggml_ext_slice(ctx->ggml_ctx, params["latent_queries"], 1, 0, gist_count);
            for (int i = 0; i < config.perceiver_depth; ++i) {
                x = std::dynamic_pointer_cast<PerceiverLayer>(blocks["layers." + std::to_string(i)])->forward(ctx, x, text);
            }
            return std::dynamic_pointer_cast<RMSNorm>(blocks["norm_out"])->forward(ctx, x);
        }
    };

    struct TransformerBlock : public GGMLBlock {
        bool injection;

        TransformerBlock(const Fibo2Config& config, bool injection)
            : injection(injection) {
            blocks["attention"]    = std::make_shared<Attention>(config, config.dim, true);
            blocks["feed_forward"] = std::make_shared<FeedForward>(config);
            for (const char* name : {"attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"}) {
                blocks[name] = std::make_shared<RMSNorm>(config.dim, 1e-5f);
            }
            blocks["adaLN_modulation.0"] = std::make_shared<Linear>(std::min<int64_t>(config.dim, 256), 4 * config.dim);
            if (injection) {
                blocks["cross_attention"]  = std::make_shared<Attention>(config, config.text_dim);
                blocks["cross_attn_norm1"] = std::make_shared<RMSNorm>(config.dim, 1e-5f);
                blocks["cross_attn_norm2"] = std::make_shared<RMSNorm>(config.dim, 1e-5f);
            }
        }

        void init_params(ggml_context* ctx, const String2TensorStorage& weights = {}, const std::string prefix = "") override {
            if (injection) {
                params["cross_attn_gate"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, ggml_tensor* noisy, ggml_tensor* clean, int image_start, int image_count, ggml_tensor* text = nullptr) {
            auto c    = ctx->ggml_ctx;
            auto norm = [&](const char* name, ggml_tensor* h) {
                return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h);
            };
            auto modulation = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.0"]);
            auto mn         = ggml_ext_chunk(c, modulation->forward(ctx, noisy), 4, 0);
            auto mc         = ggml_ext_chunk(c, modulation->forward(ctx, clean), 4, 0);
            auto select     = [&](int index, bool gate) {
                auto n      = gate ? ggml_tanh(c, mn[index]) : mn[index];
                auto k      = gate ? ggml_tanh(c, mc[index]) : mc[index];
                auto repeat = [&](ggml_tensor* h, int64_t count) {
                    return ggml_repeat(c, h, ggml_new_tensor_2d(c, GGML_TYPE_F32, h->ne[0], count));
                };
                auto m = repeat(n, image_count);
                if (image_start) {
                    m = ggml_concat(c, repeat(k, image_start), m, 1);
                }
                int64_t tail = x->ne[1] - image_start - image_count;
                if (tail) {
                    m = ggml_concat(c, m, repeat(k, tail), 1);
                }
                return m;
            };
            auto h = ZImage::modulate(c, norm("attention_norm1", x), select(0, false));
            h      = std::dynamic_pointer_cast<Attention>(blocks["attention"])->forward(ctx, h, h, pe);
            x      = ggml_add(c, x, ggml_mul(c, norm("attention_norm2", h), select(1, true)));
            if (injection) {
                h = std::dynamic_pointer_cast<Attention>(blocks["cross_attention"])->forward(ctx, norm("cross_attn_norm1", x), text);
                h = ggml_mul(c, norm("cross_attn_norm2", h), ggml_tanh(c, params["cross_attn_gate"]));
                x = ggml_add(c, x, h);
            }
            h = ZImage::modulate(c, norm("ffn_norm1", x), select(2, false));
            h = std::dynamic_pointer_cast<FeedForward>(blocks["feed_forward"])->forward(ctx, h);
            return ggml_add(c, x, ggml_mul(c, norm("ffn_norm2", h), select(3, true)));
        }
    };

    struct Fibo2Model : public GGMLBlock {
        Fibo2Config config;

        Fibo2Model(const Fibo2Config& config)
            : config(config) {
            blocks["x_embedder"] = std::make_shared<Linear>(config.channels, config.dim);
            blocks["t_embedder"] = std::make_shared<TimestepEmbedder>(1024, 256, std::min<int64_t>(config.dim, 256));
            blocks["perceiver"]  = std::make_shared<Perceiver>(config);
            for (int i = 0; i < config.refiner_depth; ++i) {
                blocks["noise_refiner." + std::to_string(i)] = std::make_shared<TransformerBlock>(config, false);
            }
            for (int i = 0; i < config.depth; ++i) {
                bool injection                        = std::find(config.injection_layers.begin(), config.injection_layers.end(), i) != config.injection_layers.end();
                blocks["layers." + std::to_string(i)] = std::make_shared<TransformerBlock>(config, injection);
            }
            blocks["final_layer"] = std::make_shared<ZImage::FinalLayer>(config.dim, 1, config.channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* t, ggml_tensor* text, ggml_tensor* pe, int gist_count, const std::vector<ggml_tensor*>& refs) {
            auto c        = ctx->ggml_ctx;
            int64_t width = x->ne[0], height = x->ne[1];
            int image_count = static_cast<int>(width * height);
            // The shared FLUX.2 VAE already packs (channel, dy, dx) into 128 channels.
            auto tokens = [&](ggml_tensor* h) {
                h = ggml_reshape_3d(c, h, h->ne[0] * h->ne[1], config.channels, 1);
                return ggml_cont(c, ggml_permute(c, h, 1, 0, 2, 3));
            };
            x = tokens(x);
            for (auto ref : refs) {
                x = ggml_concat(c, x, tokens(ref), 1);
            }
            x             = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"])->forward(ctx, x);
            auto embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto noisy    = embedder->forward(ctx, t);
            auto clean    = embedder->forward(ctx, ggml_scale(c, t, 0.f));
            auto image_pe = ggml_ext_slice(c, pe, 3, gist_count, pe->ne[3]);
            for (int i = 0; i < config.refiner_depth; ++i) {
                x = std::dynamic_pointer_cast<TransformerBlock>(blocks["noise_refiner." + std::to_string(i)])->forward(ctx, x, image_pe, noisy, clean, 0, image_count);
                sd::ggml_graph_cut::mark_graph_cut(x, "fibo2.noise_refiner." + std::to_string(i), "x");
            }
            auto bundle      = [&](int i) { return ggml_ext_slice(c, text, 0, i * config.text_dim, (i + 1) * config.text_dim); };
            auto gist        = std::dynamic_pointer_cast<Perceiver>(blocks["perceiver"])->forward(ctx, bundle(0), gist_count);
            x                = ggml_concat(c, gist, x, 1);
            int bundle_index = 1;
            for (int i = 0; i < config.depth; ++i) {
                auto block   = std::dynamic_pointer_cast<TransformerBlock>(blocks["layers." + std::to_string(i)]);
                auto context = block->injection ? bundle(bundle_index++) : nullptr;
                x            = block->forward(ctx, x, pe, noisy, clean, gist_count, image_count, context);
                sd::ggml_graph_cut::mark_graph_cut(x, "fibo2.layers." + std::to_string(i), "x");
            }
            x = ggml_ext_slice(c, x, 1, gist_count, gist_count + image_count);
            x = std::dynamic_pointer_cast<ZImage::FinalLayer>(blocks["final_layer"])->forward(ctx, x, noisy);
            x = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));
            return ggml_reshape_4d(c, x, width, height, config.channels, 1);
        }
    };

    struct Fibo2Runner : public DiffusionModelRunner {
        Fibo2Config config;
        Fibo2Model model;
        std::vector<float> pe_values;

        Fibo2Runner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(Fibo2Config::detect_from_weights(weights, prefix)),
              model(config) {
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "fibo2"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& p) override {
            if (!p.x || !p.timesteps || !p.context || p.context->shape()[0] != config.text_dim * 6) {
                LOG_ERROR("Fibo 2.0 requires six text feature bundles");
                return {};
            }
            auto build = [&]() {
                auto graph = new_graph_custom(65536);
                auto x     = make_input(*p.x);
                GGML_ASSERT(x->ne[3] == 1 && x->ne[2] == config.channels);
                int count = config.gist_count(static_cast<int>(p.context->shape()[1]));
                Rope::Embedding embedding;
                for (int i = 0; i < count; ++i) {
                    embedding.ids.push_back({static_cast<float>(i + 1), 0.f, 0.f});
                }
                embedding.positions.append_tokens(count);
                auto append_image = [&](ggml_tensor* image, int plane) {
                    auto ids = Rope::gen_flux_img_ids(static_cast<int>(image->ne[1]), static_cast<int>(image->ne[0]),
                                                      1, 1, 3, plane, 0, 0, false, &embedding.positions);
                    embedding.ids.insert(embedding.ids.end(), ids.begin(), ids.end());
                };
                append_image(x, static_cast<int>(config.max_gist + 1));
                std::vector<ggml_tensor*> refs;
                if (p.ref_latents && p.ref_image_params.pass_to_dit) {
                    for (const auto& ref : *p.ref_latents) {
                        refs.push_back(make_input(ref));
                        append_image(refs.back(), static_cast<int>(config.max_gist + 1 + refs.size()));
                    }
                }
                embedding.values = Rope::embed_nd(embedding.ids, 1, 256.f, {32, 48, 48}, embedding.layout, &embedding.frequencies);
                pe_values        = finish_rope_pe(std::move(embedding));
                auto pe          = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, 64, static_cast<int64_t>(pe_values.size() / 256));
                set_backend_tensor_data(pe, pe_values.data());
                auto ctx = get_context();
                auto out = model.forward(&ctx, x, make_input(*p.timesteps), make_input(*p.context), pe, count, refs);
                ggml_build_forward_expand(graph, out);
                return graph;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(build, n_threads, false), p.x->dim());
        }
    };
}

#endif  // __SD_MODEL_DIFFUSION_FIBO2_HPP__
