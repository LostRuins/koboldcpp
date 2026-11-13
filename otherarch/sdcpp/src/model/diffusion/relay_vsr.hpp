#ifndef __SD_MODEL_DIFFUSION_RELAY_VSR_HPP__
#define __SD_MODEL_DIFFUSION_RELAY_VSR_HPP__

#include "model/common/vsr.hpp"

namespace RelayVSR {

    struct WanVSRConfig {
        int dim            = 3072;
        int ffn_dim        = 14336;
        int num_heads      = 24;
        int num_layers     = 30;
        int in_channels    = 48;
        int text_dim       = 4096;
        int text_len       = 512;
        int hidden_dim1    = 768;
        int hidden_dim2    = 3072;
        int upscale_factor = 4;

        static WanVSRConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix, int scale) {
            WanVSRConfig config;
            config.upscale_factor = scale;
            auto weight           = [&](const std::string& name) -> const TensorStorage& { return weights.at(prefix + "." + name); };
            config.dim            = int(weight("condition_embedder.time_embedder.linear_1.weight").ne[1]);
            config.ffn_dim        = int(weight("blocks.0.ffn.net.0.proj.weight").ne[1]);
            config.num_heads      = config.dim / 128;
            config.text_dim       = int(weight("prompt_embeds").ne[0]);
            config.text_len       = int(weight("prompt_embeds").ne[1]);
            config.hidden_dim1    = int(weight("lq_proj.conv1.weight").ne[3]);
            config.hidden_dim2    = int(weight("lq_proj.conv2.weight").ne[3]);
            config.num_layers     = 0;
            while (weights.count(prefix + ".blocks." + std::to_string(config.num_layers) + ".attn1.to_q.weight"))
                ++config.num_layers;
            if ((scale != 2 && scale != 4) || weight("lq_proj.conv1.weight").ne[2] != 3 * (4 * scale) * (4 * scale) ||
                config.dim != 3072 || config.num_layers != 30 || config.text_dim != 4096 ||
                weight("proj_out.weight").ne[1] != 192) {
                throw std::runtime_error("RelayVSR Wan weights do not match the requested scale or architecture");
            }
            LOG_VERBOSE("relayvsr: dim = %d, num_layers = %d, ffn_dim = %d, scale = %d", config.dim, config.num_layers, config.ffn_dim, scale);
            return config;
        }
    };

    class LQ4xProj : public GGMLBlock {
        int hidden_dim1_;
        int hidden_dim2_;

    public:
        explicit LQ4xProj(const WanVSRConfig& config)
            : hidden_dim1_(config.hidden_dim1), hidden_dim2_(config.hidden_dim2) {
            const int patch    = 4 * config.upscale_factor;
            blocks["conv1"]    = std::make_shared<Conv2d>(3 * patch * patch, hidden_dim1_, std::pair<int, int>{1, 1});
            blocks["norm1"]    = std::make_shared<RMSNorm>(hidden_dim1_, 1e-6f);
            blocks["conv2"]    = std::make_shared<Conv2d>(hidden_dim1_, hidden_dim2_, std::pair<int, int>{3, 3}, std::pair<int, int>{2, 2}, std::pair<int, int>{1, 1});
            blocks["norm2"]    = std::make_shared<RMSNorm>(hidden_dim2_, 1e-6f);
            blocks["out_proj"] = std::make_shared<Linear>(hidden_dim2_, config.dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* unshuffled) {
            auto conv1    = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
            auto conv2    = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);
            auto norm1    = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2    = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);
            auto c        = ctx->ggml_ctx;
            auto x        = conv1->forward(ctx, unshuffled);
            int width = int(x->ne[0]), height = int(x->ne[1]);
            x = ggml_silu(c, norm1->forward(ctx, VSR::tokens(c, x)));
            x = conv2->forward(ctx, VSR::image(c, x, width, height));
            x = ggml_silu(c, norm2->forward(ctx, VSR::tokens(c, x)));
            return out_proj->forward(ctx, x);
        }
    };

    class WanAttention : public GGMLBlock {
        int dim_;
        int heads_;

    public:
        explicit WanAttention(const WanVSRConfig& config)
            : dim_(config.dim), heads_(config.num_heads) {
            for (const auto& name : {"to_q", "to_k", "to_v", "to_out.0"})
                blocks[name] = std::make_shared<Linear>(dim_, dim_);
            blocks["norm_q"] = std::make_shared<RMSNorm>(dim_, 1e-6f);
            blocks["norm_k"] = std::make_shared<RMSNorm>(dim_, 1e-6f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context, ggml_tensor* query_rope, ggml_tensor* key_rope, const VSR::LocalAttention* local, const std::string& cache_name) {
            auto to_q      = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k      = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v      = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out_0  = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto norm_q    = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
            auto norm_k    = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
            auto c         = ctx->ggml_ctx;
            auto q         = norm_q->forward(ctx, to_q->forward(ctx, x));
            auto past_k    = ctx->get_cache_tensor(cache_name + ".key");
            auto past_v    = ctx->get_cache_tensor(cache_name + ".value");
            ggml_tensor* k = nullptr;
            ggml_tensor* v = nullptr;
            if (context && past_k) {
                k = past_k;
                v = past_v;
            } else {
                k = norm_k->forward(ctx, to_k->forward(ctx, context ? context : x));
                v = to_v->forward(ctx, context ? context : x);
                if (!context && past_k) {
                    k = ggml_concat(c, past_k, k, 1);
                    v = ggml_concat(c, past_v, v, 1);
                }
                auto retained = [&](ggml_tensor* value) {
                    if (context || value->ne[1] <= x->ne[1] * 3)
                        return value;
                    // The next attention window adds its current frame to these three previous frames.
                    const int64_t keep = x->ne[1] * 3;
                    return ggml_cont(c, ggml_view_2d(c, value, value->ne[0], keep, value->nb[1], (value->ne[1] - keep) * value->nb[1]));
                };
                ctx->cache_tensor(cache_name + ".key", retained(k));
                ctx->cache_tensor(cache_name + ".value", retained(v));
            }
            ggml_tensor* out;
            if (local) {
                q   = VSR::apply_rope(ctx, q, query_rope, heads_);
                k   = VSR::apply_rope(ctx, k, key_rope, heads_);
                out = local->forward(ctx, q, k, v, heads_);
            } else {
                out = ggml_ext_attention_ext(ctx, q, k, v, heads_, nullptr, false, ctx->flash_attn_enabled);
            }
            return to_out_0->forward(ctx, out);
        }
    };

    class WanTransformerBlock : public GGMLBlock {
        int dim_;

        void init_params(ggml_context* ctx, const String2TensorStorage& weights, const std::string prefix) override {
            params["scale_shift_table"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim_, 6, 1);
        }

    public:
        explicit WanTransformerBlock(const WanVSRConfig& config)
            : dim_(config.dim) {
            blocks["norm1"]          = std::make_shared<LayerNorm>(dim_, 1e-6f, false);
            blocks["norm2"]          = std::make_shared<LayerNorm>(dim_, 1e-6f, true);
            blocks["norm3"]          = std::make_shared<LayerNorm>(dim_, 1e-6f, false);
            blocks["attn1"]          = std::make_shared<WanAttention>(config);
            blocks["attn2"]          = std::make_shared<WanAttention>(config);
            blocks["ffn.net.0.proj"] = std::make_shared<Linear>(dim_, config.ffn_dim);
            blocks["ffn.net.2"]      = std::make_shared<Linear>(config.ffn_dim, dim_);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* text, ggml_tensor* timestep_proj, ggml_tensor* qpe, ggml_tensor* kpe, const VSR::LocalAttention& local, const std::string& name) {
            auto norm1          = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto norm2          = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto norm3          = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);
            auto ffn_net_0_proj = std::dynamic_pointer_cast<Linear>(blocks["ffn.net.0.proj"]);
            auto ffn_net_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.net.2"]);
            auto c              = ctx->ggml_ctx;
            auto modulation     = ggml_add(c, timestep_proj, params["scale_shift_table"]);
            auto pieces         = ggml_ext_chunk(ctx, modulation, 6, 1);
            auto attn1          = std::dynamic_pointer_cast<WanAttention>(blocks["attn1"]);
            auto attn2          = std::dynamic_pointer_cast<WanAttention>(blocks["attn2"]);
            auto y              = attn1->forward(ctx, ggml_ext_modulate(ctx, norm1->forward(ctx, x), pieces[0], pieces[1]), nullptr, qpe, kpe, &local, name + ".attn1");
            x                   = ggml_add(c, x, ggml_mul(c, y, pieces[2]));
            x                   = ggml_add(c, x, attn2->forward(ctx, norm2->forward(ctx, x), text, nullptr, nullptr, nullptr, name + ".attn2"));
            y                   = ggml_ext_modulate(ctx, norm3->forward(ctx, x), pieces[3], pieces[4]);
            y                   = ffn_net_0_proj->forward(ctx, y);
            y                   = ffn_net_2->forward(ctx, ggml_ext_gelu(ctx, y));
            return ggml_add(c, x, ggml_mul(c, y, pieces[5]));
        }
    };

    class WanTransformer3DModel : public GGMLBlock {
        WanVSRConfig config_;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["scale_shift_table"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config_.dim, 2, 1);
            params["prompt_embeds"]     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config_.text_dim, config_.text_len);
        }

    public:
        explicit WanTransformer3DModel(const WanVSRConfig& config)
            : config_(config) {
            blocks["patch_embedding"]                           = std::make_shared<Conv3d>(config.in_channels, config.dim, std::tuple<int, int, int>{1, 2, 2}, std::tuple<int, int, int>{1, 2, 2});
            blocks["lq_proj"]                                   = std::make_shared<LQ4xProj>(config);
            blocks["condition_embedder.time_embedder.linear_1"] = std::make_shared<Linear>(256, config.dim);
            blocks["condition_embedder.time_embedder.linear_2"] = std::make_shared<Linear>(config.dim, config.dim);
            blocks["condition_embedder.time_proj"]              = std::make_shared<Linear>(config.dim, config.dim * 6);
            blocks["condition_embedder.text_embedder.linear_1"] = std::make_shared<Linear>(config.text_dim, config.dim);
            blocks["condition_embedder.text_embedder.linear_2"] = std::make_shared<Linear>(config.dim, config.dim);
            for (int i = 0; i < config.num_layers; ++i)
                blocks["blocks." + std::to_string(i)] = std::make_shared<WanTransformerBlock>(config);
            blocks["norm_out"] = std::make_shared<LayerNorm>(config.dim, 1e-6f, false);
            blocks["proj_out"] = std::make_shared<Linear>(config.dim, config.in_channels * 4);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* lq, ggml_tensor* timestep, ggml_tensor* qpe, ggml_tensor* kpe, const VSR::LocalAttention& local) {
            auto patch_embedding                           = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);
            auto condition_embedder_time_embedder_linear_1 = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.time_embedder.linear_1"]);
            auto condition_embedder_time_embedder_linear_2 = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.time_embedder.linear_2"]);
            auto condition_embedder_time_proj              = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.time_proj"]);
            auto condition_embedder_text_embedder_linear_1 = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.text_embedder.linear_1"]);
            auto condition_embedder_text_embedder_linear_2 = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.text_embedder.linear_2"]);
            auto norm_out                                  = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out                                  = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            auto c                                         = ctx->ggml_ctx;
            const int width = int(latent->ne[0]) / 2, height = int(latent->ne[1]) / 2;
            auto x    = patch_embedding->forward(ctx, latent);
            x         = VSR::tokens(c, ggml_reshape_4d(c, x, width, height, config_.dim, 1));
            x         = ggml_add(c, x, std::dynamic_pointer_cast<LQ4xProj>(blocks["lq_proj"])->forward(ctx, lq));
            auto e    = condition_embedder_time_embedder_linear_1->forward(ctx, timestep);
            e         = condition_embedder_time_embedder_linear_2->forward(ctx, ggml_silu(c, e));
            auto e0   = condition_embedder_time_proj->forward(ctx, ggml_silu(c, e));
            e0        = ggml_reshape_3d(c, e0, config_.dim, 6, 1);
            auto text = condition_embedder_text_embedder_linear_1->forward(ctx, params["prompt_embeds"]);
            text      = condition_embedder_text_embedder_linear_2->forward(ctx, ggml_ext_gelu(ctx, text));
            for (int i = 0; i < config_.num_layers; ++i) {
                auto name = "blocks." + std::to_string(i);
                x         = std::dynamic_pointer_cast<WanTransformerBlock>(blocks[name])->forward(ctx, x, text, e0, qpe, kpe, local, name);
                sd::ggml_graph_cut::mark_graph_cut(x, "relayvsr." + name, "hidden");
            }
            e               = ggml_reshape_3d(c, e, config_.dim, 1, 1);
            auto modulation = ggml_add(c, params["scale_shift_table"], e);
            auto pieces     = ggml_ext_chunk(ctx, modulation, 2, 1);
            x               = norm_out->forward(ctx, x);
            x               = ggml_ext_modulate(ctx, x, pieces[0], pieces[1]);
            x               = proj_out->forward(ctx, x);
            x               = VSR::pixel_shuffle(c, x, width, height, 2, false);
            return ggml_reshape_4d(c, x, width * 2, height * 2, 1, config_.in_channels);
        }
    };

    struct WanVSRRunner : GGMLRunner {
        WanVSRConfig config;
        WanTransformer3DModel model;
        VSR::GraphInputs inputs;
        std::map<int, std::unique_ptr<VSR::LocalAttention>> attention;
        sd::Tensor<float> timestep;
        int frame = 0;

        WanVSRRunner(ggml_backend_t backend, const String2TensorStorage& weights, int scale, std::shared_ptr<DeviceResidencyManager> manager)
            : GGMLRunner(backend, manager), config(WanVSRConfig::detect_from_weights(weights, "wan", scale)), model(config), timestep({256}) {
            model.init(params_ctx, weights, "wan");
            for (int i = 0; i < 128; ++i) {
                double angle             = 1000.0 * std::pow(10000.0, -double(i) / 128);
                timestep.data()[i]       = float(std::cos(angle));
                timestep.data()[i + 128] = float(std::sin(angle));
            }
        }

        std::string get_desc() override { return "RelayVSR Wan"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) { model.get_param_tensors(tensors, "wan"); }
        void reset() {
            runner_end();
            free_cache_ctx_and_buffer();
            attention.clear();
            inputs.clear();
            frame = 0;
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& noise, const sd::Tensor<float>& lq) {
            int width = int(noise.shape()[0]) / 2, height = int(noise.shape()[1]) / 2;
            int count = std::min(frame + 1, 4);
            if (!attention.count(count))
                attention[count] = std::make_unique<VSR::LocalAttention>(width, height, std::vector<std::pair<int, int>>(count, {22, 40}), true);
            auto graph = [&]() {
                inputs.begin_graph();
                attention[count]->begin_graph();
                auto gf          = new_graph_custom(MAX_GRAPH_SIZE);
                auto ctx         = get_context(gf);
                const int origin = std::max(0, frame - 7);
                std::vector<int> times;
                for (int i = frame - count + 1; i <= frame; ++i)
                    times.push_back(i - origin);
                auto qpe = inputs.rope(&ctx, width, height, {frame - origin}, {44, 42, 42});
                auto kpe = inputs.rope(&ctx, width, height, times, {44, 42, 42});
                auto out = model.forward(&ctx, make_input(noise), make_input(lq), make_input(timestep), qpe, kpe, *attention[count]);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto prediction = restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), 4);
            if (prediction.empty())
                return {};
            ++frame;
            for (int64_t i = 0; i < prediction.numel(); ++i)
                prediction.data()[i] = noise.data()[i] - prediction.data()[i];
            return prediction;
        }
    };

}  // namespace RelayVSR

#endif  // __SD_MODEL_DIFFUSION_RELAY_VSR_HPP__
