#ifndef __SD_MODEL_DIFFUSION_FLUX3_HPP__
#define __SD_MODEL_DIFFUSION_FLUX3_HPP__

#include "core/pe_cache.h"
#include "model/diffusion/flux.hpp"

namespace Flux3 {

    struct Flux3Config {
        int64_t hidden_size = 3072;
        int64_t context_dim = 20480;
        int64_t channels    = 128;
        int64_t mlp_dim     = 9216;
        int heads           = 24;
        int depth           = 5;
        int single_depth    = 28;
        bool flow_map       = false;

        static Flux3Config detect_from_weights(const String2TensorStorage& tensors, const std::string& prefix) {
            Flux3Config config;
            int depth = 0, single_depth = 0;
            bool detected = false;
            for (const auto& [name, tensor] : tensors) {
                if (!starts_with(name, prefix + ".")) {
                    continue;
                }
                std::string key = name.substr(prefix.size() + 1);
                if (key == "txt_in.weight") {
                    detected           = true;
                    config.context_dim = tensor.ne[0];
                    config.hidden_size = tensor.ne[1];
                } else if (key == "emb_in.image.weight") {
                    config.channels = tensor.ne[0];
                } else if (key == "single_blocks.0.mlp_out.weight") {
                    config.mlp_dim = tensor.ne[0];
                } else if (key == "ref_time_in.in_layer.weight") {
                    config.flow_map = true;
                }
                if (starts_with(key, "txt_mode_blocks.")) {
                    depth = std::max(depth, atoi(key.substr(16).c_str()) + 1);
                }
                if (starts_with(key, "single_blocks.")) {
                    single_depth = std::max(single_depth, atoi(key.substr(14).c_str()) + 1);
                }
            }
            if (depth > 0)
                config.depth = depth;
            if (single_depth > 0)
                config.single_depth = single_depth;
            config.heads = static_cast<int>(config.hidden_size / 128);
            if (detected) {
                LOG_VERBOSE("flux3: hidden_size = %" PRId64 ", context_dim = %" PRId64 ", depth = %d, single_depth = %d, flow_map = %s",
                            config.hidden_size, config.context_dim, config.depth, config.single_depth, config.flow_map ? "true" : "false");
            }
            return config;
        }
    };

    static ggml_tensor* token_slice(ggml_context* ctx, ggml_tensor* x, int64_t start, int64_t length) {
        return ggml_view_3d(ctx, x, x->ne[0], length, x->ne[2], x->nb[1], x->nb[2], start * x->nb[1]);
    }

    struct Block : public GGMLBlock {
        int heads;

        Block(const Flux3Config& config)
            : heads(config.heads) {
            for (const auto* name : {"q_proj", "k_proj", "v_proj", "attn_out"}) {
                blocks[name] = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
            }
            blocks["mlp_in"]  = std::make_shared<Linear>(config.hidden_size, 2 * config.mlp_dim, false);
            blocks["mlp_out"] = std::make_shared<Linear>(config.mlp_dim, config.hidden_size, false);
            blocks["norm"]    = std::make_shared<Flux::QKNorm>(config.hidden_size / heads);
        }

        ggml_tensor* linear(GGMLRunnerContext* ctx, const std::string& name, ggml_tensor* x) {
            return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, x);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, const std::vector<Flux::ModulationOut>& mods, const std::vector<int64_t>& lengths) {
            auto gc            = ctx->ggml_ctx;
            ggml_tensor* x_mod = nullptr;
            int64_t offset     = 0;
            for (size_t i = 0; i < lengths.size(); ++i) {
                auto part = token_slice(gc, x, offset, lengths[i]);
                part      = Flux::modulate(gc, ggml_norm(gc, part, 1e-6f), mods[i].shift, mods[i].scale);
                x_mod     = x_mod ? ggml_concat(gc, x_mod, part, 1) : part;
                offset += lengths[i];
            }
            auto split = [&](ggml_tensor* v) {
                return ggml_reshape_4d(gc, v, x->ne[0] / heads, heads, x->ne[1], x->ne[2]);
            };
            auto norm        = std::dynamic_pointer_cast<Flux::QKNorm>(blocks["norm"]);
            auto q           = norm->query_norm(ctx, split(linear(ctx, "q_proj", x_mod)));
            auto k           = norm->key_norm(ctx, split(linear(ctx, "k_proj", x_mod)));
            auto v           = split(linear(ctx, "v_proj", x_mod));
            auto attn        = Rope::attention(ctx, q, k, v, pe, nullptr);
            auto mlp         = ggml_ext_silu_act(gc, linear(ctx, "mlp_in", x_mod));
            auto update      = ggml_add(gc, linear(ctx, "attn_out", attn), linear(ctx, "mlp_out", mlp));
            ggml_tensor* out = nullptr;
            offset           = 0;
            for (size_t i = 0; i < lengths.size(); ++i) {
                auto part = ggml_mul(gc, token_slice(gc, update, offset, lengths[i]), mods[i].gate);
                part      = ggml_add(gc, token_slice(gc, x, offset, lengths[i]), part);
                out       = out ? ggml_concat(gc, out, part, 1) : part;
                offset += lengths[i];
            }
            return out;
        }
    };

    struct Model : public GGMLBlock {
        Flux3Config config;

        Model(const Flux3Config& config)
            : config(config) {
            blocks["txt_in"]  = std::make_shared<Linear>(config.context_dim, config.hidden_size, false);
            blocks["time_in"] = std::make_shared<Flux::MLPEmbedder>(256, config.hidden_size, false);
            if (config.flow_map) {
                blocks["ref_time_in"] = std::make_shared<Flux::MLPEmbedder>(256, config.hidden_size, false);
            }
            for (const std::string mode : {"image", "image_cond", "txt"}) {
                blocks["early_stream_modulations." + mode]  = std::make_shared<Flux::Modulation>(config.hidden_size, false, false);
                blocks["single_stream_modulations." + mode] = std::make_shared<Flux::Modulation>(config.hidden_size, false, false);
                const std::string base                      = mode == "txt" ? "txt_mode_blocks." : "content_mode_blocks." + mode + ".";
                for (int i = 0; i < config.depth; ++i) {
                    blocks[base + std::to_string(i)] = std::make_shared<Block>(config);
                }
                if (mode != "txt") {
                    blocks["emb_in." + mode] = std::make_shared<Linear>(config.channels, config.hidden_size, false);
                }
            }
            for (int i = 0; i < config.single_depth; ++i) {
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<Block>(config);
            }
            blocks["final_layer.image"] = std::make_shared<Flux::LastLayer>(config.hidden_size, 1, config.channels, false, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image, ggml_tensor* text, ggml_tensor* reference, ggml_tensor* timestep, ggml_tensor* zero, ggml_tensor* target, ggml_tensor* pe) {
            auto gc             = ctx->ggml_ctx;
            const int64_t width = image->ne[0], height = image->ne[1];
            auto time_in   = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["time_in"]);
            auto image_vec = time_in->forward(ctx, ggml_ext_timestep_embedding(gc, timestep, 256, 10000, 1000.f));
            auto clean_vec = time_in->forward(ctx, ggml_ext_timestep_embedding(gc, zero, 256, 10000, 1000.f));
            if (config.flow_map) {
                auto ref_time = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["ref_time_in"]);
                auto ref_vec  = ref_time->forward(ctx, ggml_ext_timestep_embedding(gc, target, 256, 10000, 1000.f));
                image_vec     = ggml_add(gc, image_vec, ref_vec);
                clean_vec     = ggml_add(gc, clean_vec, ref_vec);
            }
            std::vector<std::string> modes{"image"};
            std::vector<ggml_tensor*> inputs{DiT::patchify(gc, image, 1, 1)};
            if (reference) {
                modes.push_back("image_cond");
                inputs.push_back(reference);
            }
            modes.push_back("txt");
            inputs.push_back(text);
            std::vector<Flux::ModulationOut> single_mods;
            std::vector<int64_t> lengths;
            ggml_tensor* joint = nullptr;
            int64_t offset     = 0;
            for (size_t m = 0; m < modes.size(); ++m) {
                const auto& mode = modes[m];
                auto vec         = m == 0 ? image_vec : clean_vec;
                auto early       = std::dynamic_pointer_cast<Flux::Modulation>(blocks["early_stream_modulations." + mode])->forward(ctx, vec)[0];
                single_mods.push_back(std::dynamic_pointer_cast<Flux::Modulation>(blocks["single_stream_modulations." + mode])->forward(ctx, vec)[0]);
                lengths.push_back(inputs[m]->ne[1]);
                auto h                 = std::dynamic_pointer_cast<Linear>(blocks[mode == "txt" ? "txt_in" : "emb_in." + mode])->forward(ctx, inputs[m]);
                auto mode_pe           = ggml_view_4d(gc, pe, 2, 2, 64, lengths.back(), pe->nb[1], pe->nb[2], pe->nb[3], offset * pe->nb[3]);
                const std::string base = mode == "txt" ? "txt_mode_blocks." : "content_mode_blocks." + mode + ".";
                for (int i = 0; i < config.depth; ++i) {
                    h = std::dynamic_pointer_cast<Block>(blocks[base + std::to_string(i)])->forward(ctx, h, mode_pe, {early}, {lengths.back()});
                    if (i + 1 < config.depth) {
                        sd::ggml_graph_cut::mark_graph_cut(h, "flux3." + base + std::to_string(i), "h");
                    }
                }
                joint = joint ? ggml_concat(gc, joint, h, 1) : h;
                sd::ggml_graph_cut::mark_graph_cut(joint, "flux3." + base + "joint", "joint");
                offset += lengths.back();
            }
            for (int i = 0; i < config.single_depth; ++i) {
                joint = std::dynamic_pointer_cast<Block>(blocks["single_blocks." + std::to_string(i)])->forward(ctx, joint, pe, single_mods, lengths);
                sd::ggml_graph_cut::mark_graph_cut(joint, "flux3.single_blocks." + std::to_string(i), "joint");
            }
            auto out = std::dynamic_pointer_cast<Flux::LastLayer>(blocks["final_layer.image"])->forward(ctx, token_slice(gc, joint, 0, lengths[0]), image_vec);
            return DiT::unpatchify(gc, out, height, width, 1, 1);
        }
    };

    struct Flux3Runner : public DiffusionModelRunner {
        sd::PECache pe_cache_;
        Flux3Config config;
        Model model;
        sd::Tensor<float> zero = sd::Tensor<float>::zeros({1});

        Flux3Runner(ggml_backend_t backend, const String2TensorStorage& tensors, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(Flux3Config::detect_from_weights(tensors, prefix)),
              model(config) {
            model.init(params_ctx, tensors, prefix);
        }

        std::string get_desc() override { return "flux3"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& params) override {
            if (params.ref_latents && params.ref_image_params.pass_to_dit && params.ref_latents->size() > 10) {
                LOG_ERROR("FLUX.3 supports at most 10 reference images");
                return {};
            }
            const auto* extra = diffusion_extra_as<Flux3DiffusionExtra>(params);
            GGML_ASSERT(params.x && params.context && params.timesteps);
            GGML_ASSERT(!config.flow_map || extra->target_timestep);
            auto get_graph = [&]() {
                auto gf   = new_graph_custom(FLUX_GRAPH_SIZE);
                auto x    = make_input(*params.x);
                auto text = make_input(*params.context);
                GGML_ASSERT(x->ne[3] == 1 && text->ne[2] == 1);
                std::vector<ggml_tensor*> refs;
                ggml_tensor* references = nullptr;
                if (params.ref_latents && params.ref_image_params.pass_to_dit) {
                    for (const auto& ref : *params.ref_latents) {
                        auto r = make_input(ref);
                        refs.push_back(r);
                        r          = DiT::patchify(compute_ctx, r, 1, 1);
                        references = references ? ggml_concat(compute_ctx, references, r, 1) : r;
                    }
                }
                sd::PECacheKey pe_key("flux3.pe");
                pe_key.add_shape(x).add_shape(text).add_shapes(refs).add(circular_x_enabled).add(circular_y_enabled);
                auto pe         = make_input(pe_cache_.get_or_create(pe_key, [&]() {
                    Rope::Embedding embedding;
                    auto append_image = [&](int64_t w, int64_t h, float index) {
                        embedding.positions.append_image(static_cast<int>(h), static_cast<int>(w), 1, 0.5f, 0.5f);
                        for (int64_t y = 0; y < h; ++y) {
                            for (int64_t x = 0; x < w; ++x) {
                                embedding.ids.push_back({0.f, y * 0.5f, x * 0.5f, index});
                            }
                        }
                    };
                    append_image(x->ne[0], x->ne[1], 0.f);
                    for (size_t i = 0; i < refs.size(); ++i) {
                        append_image(refs[i]->ne[0], refs[i]->ne[1], -float(i + 1));
                    }
                    for (int64_t i = 0; i < text->ne[1]; ++i) {
                        embedding.ids.push_back({0.f, 0.f, 0.f, static_cast<float>(i)});
                    }
                    embedding.positions.append_tokens(text->ne[1]);
                    embedding.values     = Rope::embed_nd(embedding.ids, 1, 10000.f, {32, 32, 32, 32}, embedding.layout, &embedding.frequencies);
                    auto values          = finish_rope_pe(std::move(embedding));
                    const int64_t tokens = static_cast<int64_t>(values.size() / 256);
                    return sd::Tensor<float>({2, 2, 64, tokens}, std::move(values));
                }));
                auto runner_ctx = get_context();
                auto out        = model.forward(&runner_ctx, x, text, references, make_input(*params.timesteps), make_input(zero),
                                                config.flow_map ? make_input(*extra->target_timestep) : nullptr, pe);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), params.x->dim());
        }
    };

}  // namespace Flux3

#endif  // __SD_MODEL_DIFFUSION_FLUX3_HPP__
