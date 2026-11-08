#ifndef __SD_MODEL_UPSCALER_H3_LATENT_UPSCALER_HPP__
#define __SD_MODEL_UPSCALER_H3_LATENT_UPSCALER_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_graph_cut.h"
#include "core/ggml_runner.h"
#include "core/tensor.hpp"
#include "core/util.h"
#include "model/common/ggml_block.hpp"
#include "model/upscaler/ltx_latent_upscaler.hpp"
#include "model_loader.h"

// Ported from the community ComfyUI node LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
// (MIT) with checkpoint LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0).

namespace H3VUpsampler {
    constexpr int H3_UPSAMPLER_GRAPH_SIZE = 8192;

    struct H3LatentUpscalerConfig {
        int64_t in_channels  = 24;
        int64_t mid_channels = 512;
        int embed_dim        = 64;
        int temporal_kernel  = 5;
        // Residual and temporal blocks share interleaved checkpoint indices.
        std::vector<int> in_res, in_temporal, out_res, out_temporal;
        bool valid = false;

        static H3LatentUpscalerConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                          const std::string& prefix = "") {
            H3LatentUpscalerConfig config;
            std::string base = prefix;
            if (!base.empty()) {
                base += ".";
            }
            auto find_weight = [&](const std::string& suffix) -> const TensorStorage* {
                auto iter = tensor_storage_map.find(base + suffix);
                if (iter == tensor_storage_map.end()) {
                    return nullptr;
                }
                return &iter->second;
            };

            const TensorStorage* conv_in = find_weight("conv_in.weight");
            if (conv_in == nullptr) {
                LOG_ERROR("h3 latent upsampler: 'conv_in.weight' not found under prefix '%s'", prefix.c_str());
                return config;
            }
            // Bias shapes preserve channel counts when the loader folds 5D conv weights.
            const TensorStorage* conv_in_bias = find_weight("conv_in.bias");
            if (conv_in_bias != nullptr) {
                config.mid_channels = conv_in_bias->ne[0];
            } else if (conv_in->n_dims == 5) {
                config.mid_channels = conv_in->ne[0];
            }
            const TensorStorage* conv_out_bias = find_weight("conv_out.bias");
            if (conv_out_bias != nullptr) {
                config.in_channels = conv_out_bias->ne[0];
            } else if (conv_in->n_dims == 5) {
                config.in_channels = conv_in->ne[1];
            }
            if (config.mid_channels <= 0 || config.in_channels <= 0) {
                LOG_ERROR("h3 latent upsampler: could not infer channel counts from the checkpoint");
                return config;
            }

            const TensorStorage* embed_mid = find_weight("embed.2.weight");
            if (embed_mid != nullptr && embed_mid->n_dims >= 1) {
                config.embed_dim = static_cast<int>(embed_mid->logical_ne0());
            }

            std::set<int> in_res_idx, in_temporal_idx, out_res_idx, out_temporal_idx;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, base)) {
                    continue;
                }
                const std::string rel = name.substr(base.size());
                const bool is_in      = starts_with(rel, "in_blocks.");
                const bool is_out     = starts_with(rel, "out_blocks.");
                if (!is_in && !is_out) {
                    continue;
                }
                const std::string rest = rel.substr(is_in ? 10 : 11);
                const size_t dot       = rest.find('.');
                if (dot == std::string::npos) {
                    continue;
                }
                int idx = 0;
                try {
                    idx = std::stoi(rest.substr(0, dot));
                } catch (...) {
                    continue;
                }
                const std::string part = rest.substr(dot + 1);
                if (starts_with(part, "dwconv.") || starts_with(part, "pwconv.")) {
                    (is_in ? in_temporal_idx : out_temporal_idx).insert(idx);
                } else if (starts_with(part, "in_layers.") || starts_with(part, "out_norm.") ||
                           starts_with(part, "out_layers.") || starts_with(part, "emb_layers.")) {
                    (is_in ? in_res_idx : out_res_idx).insert(idx);
                } else if (starts_with(part, "attn.")) {
                    LOG_ERROR("h3 latent upsampler: attention blocks are not supported by the v1 runner");
                    return config;
                }
            }
            config.in_res       = std::vector<int>(in_res_idx.begin(), in_res_idx.end());
            config.in_temporal  = std::vector<int>(in_temporal_idx.begin(), in_temporal_idx.end());
            config.out_res      = std::vector<int>(out_res_idx.begin(), out_res_idx.end());
            config.out_temporal = std::vector<int>(out_temporal_idx.begin(), out_temporal_idx.end());
            if (config.in_res.empty() || config.out_res.empty()) {
                LOG_ERROR("h3 latent upsampler: no res blocks found in the checkpoint");
                return config;
            }

            auto temporal_kernel_of = [&](std::vector<int> idxs, const char* stage) -> int {
                for (int idx : idxs) {
                    const TensorStorage* dw = find_weight(std::string(stage) + "." + std::to_string(idx) + ".dwconv.weight");
                    if (dw != nullptr) {
                        // kD stays at ne[2] in both [OC, IC, kD, kH, kW] and [kW, kH, kD, OC*IC].
                        return static_cast<int>(dw->ne[2]);
                    }
                }
                return 0;
            };
            const int k_in  = temporal_kernel_of(config.in_temporal, "in_blocks");
            const int k_out = k_in > 0 ? k_in : temporal_kernel_of(config.out_temporal, "out_blocks");
            if (k_out > 0) {
                config.temporal_kernel = k_out;
            } else {
                config.temporal_kernel = 1;
            }

            config.valid = true;
            LOG_VERBOSE("h3 latent upsampler: in_channels = %" PRId64 ", mid_channels = %" PRId64
                        ", in_res = %zu, in_temporal = %zu, out_res = %zu, out_temporal = %zu, temporal_kernel = %d, embed_dim = %d",
                        config.in_channels,
                        config.mid_channels,
                        config.in_res.size(),
                        config.in_temporal.size(),
                        config.out_res.size(),
                        config.out_temporal.size(),
                        config.temporal_kernel,
                        config.embed_dim);
            return config;
        }
    };

    class H3ResBlock : public GGMLBlock {
    protected:
        int64_t channels;
        int64_t embed_dim;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            auto make_conv = [&](const std::string& name) {
                blocks[name] = std::shared_ptr<GGMLBlock>(new Conv3d(channels,
                                                                     channels,
                                                                     {3, 3, 3},
                                                                     {1, 1, 1},
                                                                     {1, 1, 1}));
            };
            make_conv("in_layers.2");
            make_conv("out_layers.2");
            blocks["in_layers.0"]  = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, channels));
            blocks["out_norm"]     = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, channels));
            blocks["emb_layers.1"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, 2 * channels, true));
        }

    public:
        H3ResBlock(int64_t channels, int64_t embed_dim)
            : channels(channels),
              embed_dim(embed_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* emb) {
            auto in_layers  = std::dynamic_pointer_cast<Conv3d>(blocks["in_layers.2"]);
            auto in_norm    = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["in_layers.0"]);
            auto out_layers = std::dynamic_pointer_cast<Conv3d>(blocks["out_layers.2"]);
            auto out_norm   = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["out_norm"]);
            auto emb_layers = std::dynamic_pointer_cast<Linear>(blocks["emb_layers.1"]);

            ggml_tensor* h = in_norm->forward(ctx, x);
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            h              = in_layers->forward(ctx, h);

            // Keep emb intact: every residual block uses the same embedding.
            ggml_tensor* emb_silu = ggml_silu(ctx->ggml_ctx, emb);
            ggml_tensor* emb_out  = emb_layers->forward(ctx, emb_silu);
            ggml_tensor* emb_1d   = ggml_reshape_1d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, emb_out), 2 * channels);
            ggml_tensor* scale    = ggml_reshape_4d(ctx->ggml_ctx, ggml_ext_slice(ctx->ggml_ctx, emb_1d, 0, 0, channels), 1, 1, 1, channels);
            ggml_tensor* shift    = ggml_reshape_4d(ctx->ggml_ctx, ggml_ext_slice(ctx->ggml_ctx, emb_1d, 0, channels, 2 * channels), 1, 1, 1, channels);
            ggml_tensor* gain     = ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f);

            h = out_norm->forward(ctx, h);
            h = ggml_mul_inplace(ctx->ggml_ctx, h, gain);
            h = ggml_add_inplace(ctx->ggml_ctx, h, shift);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = out_layers->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    // Shifted multiply-adds preserve depthwise weights [1, 1, k, C] without
    // expanding them into the dense kernels required by Conv3d.
    class H3TemporalDWConv3d : public UnaryBlock {
    protected:
        int64_t channels;
        int kernel;
        bool has_bias;
        std::string prefix;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, kernel, channels);
            if (has_bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            }
        }

    public:
        H3TemporalDWConv3d(int64_t channels, int kernel, bool has_bias = true)
            : channels(channels),
              kernel(kernel),
              has_bias(has_bias) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            }
            ggml_tensor* w32 = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F32);
            ggml_tensor* xa  = x;
            if (xa->type != GGML_TYPE_F32) {
                xa = ggml_cast(ctx->ggml_ctx, xa, GGML_TYPE_F32);
            }

            const int64_t T = xa->ne[2];
            const int pad   = kernel / 2;

            // ggml_pad only pads the end; concatenate zeros for symmetric padding.
            ggml_tensor* xp = xa;
            if (pad > 0) {
                ggml_tensor* zero = ggml_scale(ctx->ggml_ctx, ggml_ext_slice(ctx->ggml_ctx, xa, 2, 0, 1), 0.0f);
                ggml_tensor* lead = zero;
                for (int i = 1; i < pad; ++i) {
                    lead = ggml_concat(ctx->ggml_ctx, lead, zero, 2);
                }
                xp = ggml_concat(ctx->ggml_ctx, lead, xa, 2);
                xp = ggml_concat(ctx->ggml_ctx, xp, lead, 2);
            }

            ggml_tensor* acc = nullptr;
            for (int j = 0; j < kernel; ++j) {
                ggml_tensor* xs   = ggml_ext_slice(ctx->ggml_ctx, xp, 2, j, j + T);
                ggml_tensor* wj   = ggml_ext_slice(ctx->ggml_ctx, w32, 2, j, j + 1);  // [1, 1, 1, C]
                ggml_tensor* term = ggml_mul(ctx->ggml_ctx, xs, wj);
                acc               = acc == nullptr ? term : ggml_add(ctx->ggml_ctx, acc, term);
            }
            if (has_bias) {
                ggml_tensor* b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
                }
                acc = ggml_add(ctx->ggml_ctx, acc, ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, 1, channels));
            }
            return acc;
        }
    };

    class H3TemporalConv : public GGMLBlock {
    protected:
        int64_t channels;
        int temporal_kernel;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            blocks["dwconv"] = std::shared_ptr<GGMLBlock>(new H3TemporalDWConv3d(channels, temporal_kernel));
            blocks["pwconv"] = std::shared_ptr<GGMLBlock>(new Conv3d(channels,
                                                                     channels,
                                                                     {1, 1, 1},
                                                                     {1, 1, 1},
                                                                     {0, 0, 0}));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, channels));
        }

    public:
        H3TemporalConv(int64_t channels, int temporal_kernel)
            : channels(channels),
              temporal_kernel(temporal_kernel) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto dwconv = std::dynamic_pointer_cast<H3TemporalDWConv3d>(blocks["dwconv"]);
            auto pwconv = std::dynamic_pointer_cast<Conv3d>(blocks["pwconv"]);
            auto norm   = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["norm"]);

            ggml_tensor* h = norm->forward(ctx, x);
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            h              = dwconv->forward(ctx, h);
            h              = pwconv->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    class H3Embed : public GGMLBlock {
    protected:
        int embed_dim;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            blocks["0"] = std::shared_ptr<GGMLBlock>(new Linear(1, embed_dim, true));
            blocks["2"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, true));
        }

    public:
        explicit H3Embed(int embed_dim)
            : embed_dim(embed_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* scale_input) {
            auto l0        = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto l2        = std::dynamic_pointer_cast<Linear>(blocks["2"]);
            ggml_tensor* h = l0->forward(ctx, ggml_reshape_2d(ctx->ggml_ctx, scale_input, 1, 1));
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            return l2->forward(ctx, h);
        }
    };

    class H3LatentUpscaler : public GGMLBlock {
    public:
        H3LatentUpscalerConfig config;

        explicit H3LatentUpscaler(H3LatentUpscalerConfig config_)
            : config(std::move(config_)) {
            for (int idx : config.in_res) {
                blocks["in_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3ResBlock(config.mid_channels,
                                                                                                       config.embed_dim));
            }
            for (int idx : config.in_temporal) {
                blocks["in_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3TemporalConv(config.mid_channels,
                                                                                                           config.temporal_kernel));
            }
            for (int idx : config.out_res) {
                blocks["out_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3ResBlock(config.mid_channels,
                                                                                                        config.embed_dim));
            }
            for (int idx : config.out_temporal) {
                blocks["out_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3TemporalConv(config.mid_channels,
                                                                                                            config.temporal_kernel));
            }
            blocks["conv_in"]  = std::shared_ptr<GGMLBlock>(new Conv3d(config.in_channels,
                                                                       config.mid_channels,
                                                                       {3, 3, 3},
                                                                       {1, 1, 1},
                                                                       {1, 1, 1}));
            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, config.mid_channels));
            blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new Conv3d(config.mid_channels,
                                                                       config.in_channels,
                                                                       {3, 3, 3},
                                                                       {1, 1, 1},
                                                                       {1, 1, 1}));
            blocks["embed"]    = std::shared_ptr<GGMLBlock>(new H3Embed(config.embed_dim));
        }

        ggml_tensor* forward_stage_a(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* scale_input) {
            auto conv_in = std::dynamic_pointer_cast<Conv3d>(blocks["conv_in"]);
            auto embed   = std::dynamic_pointer_cast<H3Embed>(blocks["embed"]);

            ggml_tensor* h = conv_in->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(h, "h3_latent_upscaler.conv_in", "x");

            ggml_tensor* emb = embed->forward(ctx, scale_input);
            std::set<int> temporal_set(config.in_temporal.begin(), config.in_temporal.end());
            int max_idx = 0;
            for (int idx : config.in_res) {
                max_idx = std::max(max_idx, idx);
            }
            for (int idx : config.in_temporal) {
                max_idx = std::max(max_idx, idx);
            }
            for (int i = 0; i <= max_idx; ++i) {
                if (temporal_set.count(i)) {
                    auto block = std::dynamic_pointer_cast<H3TemporalConv>(blocks["in_blocks." + std::to_string(i)]);
                    h          = block->forward(ctx, h);
                } else {
                    auto block = std::dynamic_pointer_cast<H3ResBlock>(blocks["in_blocks." + std::to_string(i)]);
                    h          = block->forward(ctx, h, emb);
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "h3_latent_upscaler.in_blocks." + std::to_string(i), "x");
            }
            return h;
        }

        ggml_tensor* forward_stage_b(GGMLRunnerContext* ctx, ggml_tensor* m, ggml_tensor* scale_input) {
            auto conv_out = std::dynamic_pointer_cast<Conv3d>(blocks["conv_out"]);
            auto norm_out = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["norm_out"]);
            auto embed    = std::dynamic_pointer_cast<H3Embed>(blocks["embed"]);

            ggml_tensor* g   = m;
            ggml_tensor* emb = embed->forward(ctx, scale_input);
            std::set<int> temporal_set(config.out_temporal.begin(), config.out_temporal.end());
            int max_idx = 0;
            for (int idx : config.out_res) {
                max_idx = std::max(max_idx, idx);
            }
            for (int idx : config.out_temporal) {
                max_idx = std::max(max_idx, idx);
            }
            for (int i = 0; i <= max_idx; ++i) {
                if (temporal_set.count(i)) {
                    auto block = std::dynamic_pointer_cast<H3TemporalConv>(blocks["out_blocks." + std::to_string(i)]);
                    g          = block->forward(ctx, g);
                } else {
                    auto block = std::dynamic_pointer_cast<H3ResBlock>(blocks["out_blocks." + std::to_string(i)]);
                    g          = block->forward(ctx, g, emb);
                }
                sd::ggml_graph_cut::mark_graph_cut(g, "h3_latent_upscaler.out_blocks." + std::to_string(i), "x");
            }

            g = norm_out->forward(ctx, g);
            g = ggml_silu_inplace(ctx->ggml_ctx, g);
            g = conv_out->forward(ctx, g);
            sd::ggml_graph_cut::mark_graph_cut(g, "h3_latent_upscaler.conv_out", "x");
            return g;
        }
    };

    struct H3LatentUpscalerRunner : public GGMLRunner {
        H3LatentUpscalerConfig config;
        std::unique_ptr<H3LatentUpscaler> model;
        std::string weight_prefix;

        H3LatentUpscalerRunner(ggml_backend_t backend,
                               const String2TensorStorage& tensor_storage_map,
                               const std::string& prefix                           = "",
                               std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager),
              config(H3LatentUpscalerConfig::detect_from_weights(tensor_storage_map, prefix)),
              weight_prefix(prefix) {
            if (!config.valid || config.embed_dim <= 0 || config.mid_channels <= 0) {
                LOG_ERROR("unsupported H3 latent upsampler config: in_channels=%" PRId64 " mid_channels=%" PRId64 " embed_dim=%d temporal_kernel=%d",
                          config.in_channels,
                          config.mid_channels,
                          config.embed_dim,
                          config.temporal_kernel);
                return;
            }
            model = std::make_unique<H3LatentUpscaler>(config);
            model->init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "h3_latent_upscaler";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
            if (model) {
                model->get_param_tensors(tensors, weight_prefix);
            }
        }

        // x: H3 diffusion latent [W, H, T, C] (or 5D batch 1). The output stays
        // in the same normalized domain for the next sampling pass.
        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<float>& x,
                                  int64_t target_w,
                                  int64_t target_h) {
            if (model == nullptr) {
                LOG_ERROR("H3 latent upsampler is not loaded");
                return {};
            }
            if (x.dim() != 4 && x.dim() != 5) {
                LOG_ERROR("H3 latent upsampler expects a 4D or 5D video latent, got dim=%lld", (long long)x.dim());
                return {};
            }
            if (x.dim() == 5 && x.shape()[4] != 1) {
                LOG_ERROR("H3 latent upsampler currently supports batch size 1, got batch=%lld", (long long)x.shape()[4]);
                return {};
            }
            if (config.in_channels != 24 || x.shape()[3] != config.in_channels) {
                LOG_ERROR("H3 latent upsampler requires 24 channels, got model=%" PRId64 " input=%lld", config.in_channels, (long long)x.shape()[3]);
                return {};
            }
            const int64_t in_w = x.shape()[0];
            const int64_t in_h = x.shape()[1];
            const int64_t T    = x.shape()[2];
            if (target_w % 2 != 0 || target_h % 2 != 0) {
                LOG_ERROR("H3 latent upsampler target latent size must be even (pixel size a multiple of 32), got %" PRId64 "x%" PRId64,
                          target_w,
                          target_h);
                return {};
            }
            const float scale = (static_cast<float>(target_w) / static_cast<float>(in_w) +
                                 static_cast<float>(target_h) / static_cast<float>(in_h)) /
                                2.0f;
            if (scale < 1.0f - 1e-3f) {
                LOG_ERROR("H3 latent upsampler only supports upscaling (scale >= 1.0), got %.3f", scale);
                return {};
            }

            const size_t expected_dim = static_cast<size_t>(x.dim());
            const sd::Tensor<float> mean({1, 1, 1, 24},
                                         {0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
                                          -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
                                          -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
                                          -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
                                          -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
                                          -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f});
            const sd::Tensor<float> stddev({1, 1, 1, 24},
                                           {1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f, 1.7549455165863037f,
                                            1.5636216402053833f, 2.194143533706665f, 0.9653137922286987f, 1.0569885969161987f,
                                            0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
                                            0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f, 0.6936293244361877f,
                                            2.961095094680786f, 2.7694199085235596f, 3.0496184825897217f, 2.1088054180145264f,
                                            3.276226282119751f, 3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f});
            // Training applies the VAE normalization a second time to diffusion latents.
            // Undo only this extra transform after the network.
            const sd::Tensor<float> normalized   = (x - mean) / stddev;
            const sd::Tensor<float> scale_tensor = sd::Tensor<float>::from_vector({scale - 1.0f});
            auto get_graph_a                     = [this, &normalized, &scale_tensor]() -> ggml_cgraph* {
                ggml_cgraph* gf          = new_graph_custom(H3_UPSAMPLER_GRAPH_SIZE);
                ggml_tensor* x_t         = make_input(normalized);
                ggml_tensor* scale_input = make_input(scale_tensor);
                auto runner_ctx          = get_context(gf);
                ggml_tensor* out         = model->forward_stage_a(&runner_ctx, x_t, scale_input);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            std::optional<sd::Tensor<float>> mid_opt = GGMLRunner::compute(get_graph_a, n_threads, false, false);
            runner_end();
            if (!mid_opt.has_value() || mid_opt->empty()) {
                LOG_ERROR("H3 latent upsampler stage A (in-block chain) failed");
                return {};
            }
            sd::Tensor<float> mid = std::move(*mid_opt);

            std::vector<int64_t> mid_shape = mid.shape();
            mid_shape[0]                   = target_w;
            mid_shape[1]                   = target_h;
            sd::Tensor<float> mid_up       = sd::ops::interpolate(mid, mid_shape, sd::ops::InterpolateMode::Bilinear, false, false);
            if (mid_up.empty()) {
                LOG_ERROR("H3 latent upsampler middle spatial interpolate failed: %" PRId64 "x%" PRId64 " -> %" PRId64 "x%" PRId64,
                          in_w, in_h, target_w, target_h);
                return {};
            }

            auto get_graph_b = [this, &mid_up, &scale_tensor]() -> ggml_cgraph* {
                ggml_cgraph* gf          = new_graph_custom(H3_UPSAMPLER_GRAPH_SIZE);
                ggml_tensor* m_t         = make_input(mid_up);
                ggml_tensor* scale_input = make_input(scale_tensor);
                auto runner_ctx          = get_context(gf);
                ggml_tensor* out         = model->forward_stage_b(&runner_ctx, m_t, scale_input);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            std::optional<sd::Tensor<float>> out_opt = GGMLRunner::compute(get_graph_b, n_threads, false, false);
            runner_end();
            if (!out_opt.has_value() || out_opt->empty()) {
                LOG_ERROR("H3 latent upsampler stage B (out-block chain) failed");
                return {};
            }
            sd::Tensor<float> out = std::move(*out_opt);
            if (out.shape()[0] != target_w || out.shape()[1] != target_h || out.shape()[2] != T) {
                LOG_ERROR("H3 latent upsampler output shape mismatch: got %" PRId64 "x%" PRId64 "x%" PRId64 " expected %" PRId64 "x%" PRId64 "x%" PRId64,
                          out.shape()[0], out.shape()[1], out.shape()[2], target_w, target_h, T);
                return {};
            }
            return restore_trailing_singleton_dims(out * stddev + mean, expected_dim);
        }
    };

}  // namespace H3VUpsampler

#endif  // __SD_MODEL_UPSCALER_H3_LATENT_UPSCALER_HPP__
