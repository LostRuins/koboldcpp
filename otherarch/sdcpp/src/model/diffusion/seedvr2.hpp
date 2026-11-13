#ifndef __SD_MODEL_DIFFUSION_SEEDVR2_HPP__
#define __SD_MODEL_DIFFUSION_SEEDVR2_HPP__

#include "model/common/vsr.hpp"

namespace SeedVR2 {

    struct SeedVR2Config {
        int dim        = 0;
        int layers     = 0;
        int mm_layers  = 0;
        int hidden     = 0;
        int text_dim   = 0;
        int text_len   = 0;
        int rope_freqs = 0;
        bool swiglu    = false;

        static SeedVR2Config detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            SeedVR2Config config;
            auto at = [&](const std::string& key) -> const TensorStorage& {
                auto found = weights.find(prefix + "." + key);
                if (found == weights.end())
                    throw std::runtime_error("Missing SeedVR2 tensor: " + prefix + "." + key);
                return found->second;
            };
            config.dim        = int(at("vid_in.proj.weight").ne[1]);
            config.hidden     = int(at("blocks.0.mlp.vid.proj_in.weight").ne[1]);
            config.text_dim   = int(at("positive_conditioning").ne[0]);
            config.text_len   = int(at("positive_conditioning").ne[1]);
            config.rope_freqs = int(at("blocks.0.attn.rope.rope.freqs").ne[0]);
            config.swiglu     = weights.count(prefix + ".blocks.0.mlp.vid.proj_in_gate.weight") != 0;
            while (weights.count(prefix + ".blocks." + std::to_string(config.layers) + ".attn.rope.rope.freqs")) {
                if (weights.count(prefix + ".blocks." + std::to_string(config.layers) + ".attn.proj_qkv.vid.weight"))
                    ++config.mm_layers;
                ++config.layers;
            }
            if (at("vid_in.proj.weight").logical_ne0() != 132 || at("vid_out.proj.weight").ne[1] != 64 ||
                config.text_dim != 5120 ||
                !(config.swiglu ? config.dim == 2560 && config.layers == 32 && config.mm_layers == 10 && config.rope_freqs == 21
                                : config.dim == 3072 && config.layers == 36 && config.mm_layers == 36 && config.rope_freqs == 10))
                throw std::runtime_error("Unsupported SeedVR2 architecture");
            LOG_VERBOSE("seedvr2: dim = %d, layers = %d, mm_layers = %d, hidden = %d, rope_freqs = %d", config.dim, config.layers, config.mm_layers, config.hidden, config.rope_freqs);
            return config;
        }
    };

    struct WindowPlan {
        sd::Tensor<int32_t> order;
        sd::Tensor<int32_t> reverse;
        sd::Tensor<float> positions;
        sd::Tensor<float> text_positions;
        std::vector<int> lengths;

        WindowPlan(int width, int height, int frames, int text_len, bool shifted, bool multimodal)
            : order({width * height * frames}), reverse({width * height * frames}), positions({1, 3, width * height * frames}), text_positions({1, 3, text_len}) {
            const double scale = std::sqrt(720.0 * 1280.0 / 256.0 / (width * height));
            const int wh       = int(std::ceil(std::nearbyint(height * scale) / 3));
            const int ww       = int(std::ceil(std::nearbyint(width * scale) / 3));
            const int wt       = (std::min(frames, 30) + 3) / 4;
            const double st    = shifted && wt < frames ? 0.5 : 0;
            const double sh    = shifted && wh < height ? 0.5 : 0;
            const double sw    = shifted && ww < width ? 0.5 : 0;
            const int nt       = st ? int(std::ceil((frames - st) / wt)) + 1 : (frames + wt - 1) / wt;
            const int nh       = sh ? int(std::ceil((height - sh) / wh)) + 1 : (height + wh - 1) / wh;
            const int nw       = sw ? int(std::ceil((width - sw) / ww)) + 1 : (width + ww - 1) / ww;
            int index          = 0;
            for (int iw = 0; iw < nw; ++iw)
                for (int ih = 0; ih < nh; ++ih)
                    for (int it = 0; it < nt; ++it) {
                        int x0 = std::max(0, int((iw - sw) * ww)), x1 = std::min(width, int((iw - sw + 1) * ww));
                        int y0 = std::max(0, int((ih - sh) * wh)), y1 = std::min(height, int((ih - sh + 1) * wh));
                        int t0 = std::max(0, int((it - st) * wt)), t1 = std::min(frames, int((it - st + 1) * wt));
                        if (x1 <= x0 || y1 <= y0 || t1 <= t0)
                            continue;
                        lengths.push_back((x1 - x0) * (y1 - y0) * (t1 - t0));
                        for (int t = t0; t < t1; ++t)
                            for (int y = y0; y < y1; ++y)
                                for (int x = x0; x < x1; ++x, ++index) {
                                    int original             = (t * height + y) * width + x;
                                    order.data()[index]      = original;
                                    reverse.data()[original] = index;
                                    const int pos[]          = {t - t0, y - y0, x - x0};
                                    const int size[]         = {t1 - t0, y1 - y0, x1 - x0};
                                    for (int a = 0; a < 3; ++a)
                                        positions.data()[3 * index + a] = multimodal ? float(pos[a] + (a == 0 ? text_len : 0))
                                                                                     : (size[a] == 1 ? -1.f : -1.f + 2.f * pos[a] / (size[a] - 1));
                                }
                    }
            if (index != width * height * frames)
                throw std::runtime_error("SeedVR2 window partition is incomplete");
            for (int i = 0; i < text_len; ++i)
                for (int a = 0; a < 3; ++a)
                    text_positions.data()[3 * i + a] = multimodal ? float(i) : 0.f;
        }
    };

    class AdaSingle : public GGMLBlock {
        int dim_;
        bool output_;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            for (const auto& layer : output_ ? std::vector<std::string>{"out"} : std::vector<std::string>{"attn", "mlp"}) {
                for (const auto& mode : output_ ? std::vector<std::string>{"shift", "scale"} : std::vector<std::string>{"shift", "scale", "gate"})
                    params[layer + "_" + mode] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_);
            }
        }

    public:
        AdaSingle(int dim, bool output = false)
            : dim_(dim), output_(output) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* emb, const std::string& layer, bool gate = false) {
            auto c = ctx->ggml_ctx;
            // The reference reshapes embedding to [D, 2, 3], not [6, D].
            const int offset = layer == "mlp" ? 3 : 0;
            auto mod         = [&](int i, const char* name) {
                auto value = ggml_cont(c, ggml_view_2d(c, emb, 1, dim_, 6 * emb->nb[0], (offset + i) * emb->nb[0]));
                return ggml_add(c, ggml_reshape_1d(c, value, dim_), params[layer + "_" + name]);
            };
            if (gate)
                return ggml_mul(c, x, mod(2, "gate"));
            // AdaSingle uses scale directly; ggml_ext_modulate adds an identity scale.
            return ggml_add(c, ggml_mul(c, x, mod(1, "scale")), mod(0, "shift"));
        }
    };

    class SRMLP : public UnaryBlock {
        bool swiglu_;

    public:
        explicit SRMLP(const SeedVR2Config& config)
            : swiglu_(config.swiglu) {
            blocks["proj_in"]  = std::make_shared<Linear>(config.dim, config.hidden, !swiglu_);
            blocks["proj_out"] = std::make_shared<Linear>(config.hidden, config.dim, !swiglu_);
            if (swiglu_)
                blocks["proj_in_gate"] = std::make_shared<Linear>(config.dim, config.hidden, false);
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto up     = std::dynamic_pointer_cast<Linear>(blocks["proj_in"])->forward(ctx, x);
            auto hidden = swiglu_ ? ggml_swiglu_split(ctx->ggml_ctx, std::dynamic_pointer_cast<Linear>(blocks["proj_in_gate"])->forward(ctx, x), up)
                                  : ggml_ext_gelu(ctx, up);
            return std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, hidden);
        }
    };

    class NaMMSRTransformerBlock : public GGMLBlock {
        SeedVR2Config config_;
        bool shared_;
        bool last_;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["attn.rope.rope.freqs"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.rope_freqs);
        }

    public:
        NaMMSRTransformerBlock(const SeedVR2Config& config, int index)
            : config_(config), shared_(index >= config.mm_layers), last_(config.swiglu && index == config.layers - 1) {
            for (const auto& branch : shared_ ? std::vector<std::string>{"all"} : std::vector<std::string>{"vid", "txt"}) {
                blocks["attn.proj_qkv." + branch] = std::make_shared<Linear>(config.dim, 3 * config.dim, false);
                blocks["attn.proj_out." + branch] = std::make_shared<Linear>(config.dim, config.dim);
                blocks["attn.norm_q." + branch]   = std::make_shared<RMSNorm>(128, 1e-5f);
                blocks["attn.norm_k." + branch]   = std::make_shared<RMSNorm>(128, 1e-5f);
                blocks["ada." + branch]           = std::make_shared<AdaSingle>(config.dim);
                blocks["mlp." + branch]           = std::make_shared<SRMLP>(config);
            }
        }

        ggml_tensor* rotation(GGMLRunnerContext* ctx, ggml_tensor* positions) {
            auto c      = ctx->ggml_ctx;
            const int n = config_.rope_freqs;
            auto p      = ggml_repeat(c, positions, ggml_new_tensor_3d(c, GGML_TYPE_F32, n, 3, positions->ne[2]));
            auto angle  = ggml_mul(c, p, params["attn.rope.rope.freqs"]);
            angle       = ggml_reshape_2d(c, angle, 3 * n, positions->ne[2]);
            angle       = ggml_pad(c, angle, 64 - 3 * n, 0, 0, 0);
            // Keep tokens out of ne[3]: CUDA concat otherwise launches once per token.
            auto cs     = ggml_reshape_3d(c, ggml_cos(c, angle), 1, 1, 64 * positions->ne[2]);
            auto sn     = ggml_reshape_3d(c, ggml_sin(c, angle), 1, 1, 64 * positions->ne[2]);
            auto matrix = ggml_concat(c, ggml_concat(c, cs, ggml_scale(c, sn, -1.f), 0), ggml_concat(c, sn, cs, 0), 1);
            return ggml_reshape_4d(c, matrix, 2, 2, 64, positions->ne[2]);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* vid, ggml_tensor* txt, ggml_tensor* emb, const WindowPlan& plan, VSR::GraphInputs& inputs) {
            auto c        = ctx->ggml_ctx;
            const int dim = config_.dim, heads = dim / 128;
            auto branch = [&](bool text) -> std::string { return shared_ ? "all" : text ? "txt"
                                                                                        : "vid"; };
            auto linear = [&](const std::string& name, bool text, ggml_tensor* x) {
                return std::dynamic_pointer_cast<Linear>(blocks[name + "." + branch(text)])->forward(ctx, x);
            };
            auto ada = [&](ggml_tensor* x, bool text, const std::string& layer, bool gate = false) {
                if (text && last_)
                    return x;
                return std::dynamic_pointer_cast<AdaSingle>(blocks["ada." + branch(text)])->forward(ctx, x, emb, layer, gate);
            };
            auto vqkv      = linear("attn.proj_qkv", false, ada(ggml_rms_norm(c, vid, 1e-5f), false, "attn"));
            auto tqkv      = linear("attn.proj_qkv", true, ada(ggml_rms_norm(c, txt, 1e-5f), true, "attn"));
            vqkv           = ggml_get_rows(c, vqkv, inputs.input(ctx, plan.order));
            auto vpe       = rotation(ctx, inputs.input(ctx, plan.positions));
            auto tpe       = rotation(ctx, inputs.input(ctx, plan.text_positions));
            auto split_qkv = [&](ggml_tensor* x, int i) {
                int64_t qkv_dim = x->ne[0] / 3;
                return ggml_view_4d(c, x, 128, qkv_dim / 128, x->ne[1], x->ne[2] * x->ne[3],
                                    128 * x->nb[0], x->nb[1], x->nb[2], i * qkv_dim * x->nb[0]);
            };
            auto qkv = [&](ggml_tensor* x, bool text, ggml_tensor* pe) {
                std::vector<ggml_tensor*> parts{nullptr, nullptr, nullptr};
                for (int i = 0; i < 2; ++i) {
                    auto h   = std::dynamic_pointer_cast<RMSNorm>(blocks[std::string(i == 0 ? "attn.norm_q." : "attn.norm_k.") + branch(text)])->forward(ctx, split_qkv(x, i));
                    h        = ggml_reshape_2d(c, h, dim, x->ne[1]);
                    parts[i] = VSR::apply_rope(ctx, h, pe, heads);
                }
                // v must stay 3-D [dim, L, N]: the window loop slices dim 1 as tokens.
                auto v   = split_qkv(x, 2);
                parts[2] = ggml_view_3d(c, v, dim, x->ne[1], x->ne[2], v->nb[2], v->nb[3], 0);
                return parts;
            };
            auto vp = qkv(vqkv, false, vpe), tp = qkv(tqkv, true, tpe);
            std::vector<ggml_tensor*> video_windows;
            ggml_tensor* tout = nullptr;
            int start         = 0;
            for (int length : plan.lengths) {
                std::vector<ggml_tensor*> joined;
                for (int i = 0; i < 3; ++i)
                    joined.push_back(ggml_concat(c, ggml_ext_slice(ctx, vp[i], 1, start, start + length), tp[i], 1));
                auto out = ggml_ext_attention_ext(ctx, joined[0], joined[1], joined[2], heads, nullptr, false, ctx->flash_attn_enabled);
                auto vo  = ggml_ext_slice(ctx, out, 1, 0, length);
                auto to  = ggml_ext_slice(ctx, out, 1, length, length + txt->ne[1]);
                video_windows.push_back(vo);
                tout = tout ? ggml_add(c, tout, to) : to;
                start += length;
            }
            auto vout = ggml_ext_vec_concat(ctx, video_windows, 1);
            vout      = ggml_get_rows(c, vout, inputs.input(ctx, plan.reverse));
            tout      = ggml_scale(c, tout, 1.f / plan.lengths.size());
            vid       = ggml_add(c, vid, ada(linear("attn.proj_out", false, vout), false, "attn", true));
            txt       = ggml_add(c, txt, ada(linear("attn.proj_out", true, tout), true, "attn", true));
            auto mlp  = [&](ggml_tensor* x, bool text) {
                auto h = ada(ggml_rms_norm(c, x, 1e-5f), text, "mlp");
                h      = std::dynamic_pointer_cast<SRMLP>(blocks["mlp." + branch(text)])->forward(ctx, h);
                return ggml_add(c, x, ada(h, text, "mlp", true));
            };
            vid = mlp(vid, false);
            if (!last_)
                txt = mlp(txt, true);
            return {vid, txt};
        }
    };

    class NaDiT : public GGMLBlock {
        SeedVR2Config config_;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["positive_conditioning"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config_.text_dim, config_.text_len);
        }

    public:
        explicit NaDiT(const SeedVR2Config& config)
            : config_(config) {
            blocks["vid_in.proj"]     = std::make_shared<Linear>(132, config.dim);
            blocks["txt_in"]          = std::make_shared<Linear>(config.text_dim, config.dim);
            blocks["emb_in.proj_in"]  = std::make_shared<Linear>(256, config.dim);
            blocks["emb_in.proj_hid"] = std::make_shared<Linear>(config.dim, config.dim);
            blocks["emb_in.proj_out"] = std::make_shared<Linear>(config.dim, 6 * config.dim);
            for (int i = 0; i < config.layers; ++i)
                blocks["blocks." + std::to_string(i)] = std::make_shared<NaMMSRTransformerBlock>(config, i);
            blocks["vid_out.proj"] = std::make_shared<Linear>(config.dim, 64);
            if (config.swiglu) {
                blocks["vid_out_norm"] = std::make_shared<RMSNorm>(config.dim, 1e-5f);
                blocks["vid_out_ada"]  = std::make_shared<AdaSingle>(config.dim, true);
            }
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* input, ggml_tensor* timestep, const std::vector<WindowPlan>& plans, VSR::GraphInputs& inputs) {
            auto linear = [&](const std::string& name, ggml_tensor* x) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, x); };
            auto vid    = linear("vid_in.proj", input);
            auto txt    = linear("txt_in", params["positive_conditioning"]);
            auto emb    = linear("emb_in.proj_in", timestep);
            emb         = linear("emb_in.proj_hid", ggml_silu(ctx->ggml_ctx, emb));
            emb         = linear("emb_in.proj_out", ggml_silu(ctx->ggml_ctx, emb));
            for (int i = 0; i < config_.layers; ++i) {
                auto name          = "blocks." + std::to_string(i);
                std::tie(vid, txt) = std::dynamic_pointer_cast<NaMMSRTransformerBlock>(blocks[name])->forward(ctx, vid, txt, emb, plans[i % 2], inputs);
                sd::ggml_graph_cut::mark_graph_cut(vid, "seedvr2." + name, "vid");
                sd::ggml_graph_cut::mark_graph_cut(txt, "seedvr2." + name, "txt");
            }
            if (config_.swiglu) {
                vid = std::dynamic_pointer_cast<RMSNorm>(blocks["vid_out_norm"])->forward(ctx, vid);
                vid = std::dynamic_pointer_cast<AdaSingle>(blocks["vid_out_ada"])->forward(ctx, vid, emb, "out");
            }
            return linear("vid_out.proj", vid);
        }
    };

    struct SeedVR2Runner : GGMLRunner {
        SeedVR2Config config;
        NaDiT model;
        sd::Tensor<float> timestep;

        SeedVR2Runner(ggml_backend_t backend, const String2TensorStorage& weights, std::shared_ptr<DeviceResidencyManager> manager)
            : GGMLRunner(backend, manager), config(SeedVR2Config::detect_from_weights(weights, "diffusion")), model(config), timestep({256}) {
            model.init(params_ctx, weights, "diffusion");
            for (int i = 0; i < 128; ++i) {
                const double angle       = 1000.0 * std::pow(10000.0, -double(i) / 128);
                timestep.data()[i]       = float(std::sin(angle));
                timestep.data()[i + 128] = float(std::cos(angle));
            }
        }
        std::string get_desc() override { return "SeedVR2 DiT"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) { model.get_param_tensors(tensors, "diffusion"); }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& input, int width, int height, int frames) {
            std::vector<WindowPlan> plans;
            for (bool shifted : {false, true})
                plans.emplace_back(width, height, frames, config.text_len, shifted, config.swiglu);
            VSR::GraphInputs inputs;
            auto graph = [&]() {
                inputs.begin_graph();
                auto gf  = new_graph_custom(MAX_GRAPH_SIZE);
                auto ctx = get_context(gf);
                auto out = model.forward(&ctx, make_input(input), make_input(timestep), plans, inputs);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto output = GGMLRunner::compute(graph, n_threads);
            return output ? *output : sd::Tensor<float>();
        }
    };
}

#endif  // __SD_MODEL_DIFFUSION_SEEDVR2_HPP__
