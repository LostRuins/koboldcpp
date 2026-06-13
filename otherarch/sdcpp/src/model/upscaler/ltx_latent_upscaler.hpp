#ifndef __SD_MODEL_UPSCALER_LTX_LATENT_UPSCALER_HPP__
#define __SD_MODEL_UPSCALER_LTX_LATENT_UPSCALER_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/ggml_graph_cut.h"
#include "core/util.h"
#include "model/diffusion/dit.hpp"
#include "model_loader.h"

namespace LTXVUpsampler {
    constexpr int LTX_UPSAMPLER_GRAPH_SIZE = 10240;

    struct LatentUpsamplerConfig {
        int64_t in_channels      = 128;
        int64_t mid_channels     = 1024;
        int num_blocks_per_stage = 4;
        int dims                 = 3;
        bool spatial_upsample    = true;
        bool temporal_upsample   = false;
        bool rational_resampler  = false;
        float spatial_scale      = 2.f;
        int spatial_up_num       = 2;
        int spatial_down_den     = 1;
        int temporal_up_factor   = 1;

        static LatentUpsamplerConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                         const std::string& prefix = "") {
            LatentUpsamplerConfig config;
            auto find_weight = [&](const std::string& suffix) -> const TensorStorage* {
                std::string name = prefix.empty() ? suffix : prefix + "." + suffix;
                auto iter        = tensor_storage_map.find(name);
                if (iter == tensor_storage_map.end()) {
                    return nullptr;
                }
                return &iter->second;
            };

            bool inferred = false;

            const TensorStorage* initial_norm = find_weight("initial_norm.weight");
            if (initial_norm != nullptr) {
                config.mid_channels = initial_norm->ne[0];
                inferred            = true;
            }

            const TensorStorage* final_conv = find_weight("final_conv.bias");
            if (final_conv != nullptr) {
                config.in_channels = final_conv->ne[0];
                inferred           = true;
            }

            int detected_blocks                 = 0;
            const std::string res_blocks_prefix = prefix.empty() ? "res_blocks." : prefix + ".res_blocks.";
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, res_blocks_prefix)) {
                    continue;
                }
                size_t begin = res_blocks_prefix.size();
                size_t end   = name.find('.', begin);
                if (end == std::string::npos) {
                    continue;
                }
                try {
                    int idx         = std::stoi(name.substr(begin, end - begin));
                    detected_blocks = std::max(detected_blocks, idx + 1);
                } catch (...) {
                }
            }
            if (detected_blocks > 0) {
                config.num_blocks_per_stage = detected_blocks;
                inferred                    = true;
            }

            const TensorStorage* rational_upsampler_weight = find_weight("upsampler.conv.weight");
            const TensorStorage* upsampler_bias            = find_weight("upsampler.0.bias");
            config.rational_resampler                      = rational_upsampler_weight != nullptr;
            int64_t upsampler_out_channels                 = upsampler_bias == nullptr ? 0 : upsampler_bias->ne[0];
            config.spatial_upsample                        = config.rational_resampler || upsampler_out_channels == 4 * config.mid_channels;
            config.temporal_upsample                       = upsampler_out_channels == 2 * config.mid_channels;
            if (config.rational_resampler || upsampler_out_channels > 0) {
                inferred = true;
            }
            if (config.temporal_upsample) {
                config.temporal_up_factor = 2;
            }
            if (rational_upsampler_weight != nullptr) {
                int64_t out_channels = rational_upsampler_weight->ne[3];
                if (config.mid_channels > 0 && out_channels % config.mid_channels == 0) {
                    int64_t ratio = out_channels / config.mid_channels;
                    int num       = static_cast<int>(std::round(std::sqrt(static_cast<double>(ratio))));
                    if (num > 0 && static_cast<int64_t>(num) * num == ratio) {
                        config.spatial_up_num = num;
                    }
                }
                if (config.spatial_up_num == 3) {
                    config.spatial_down_den = 2;
                    config.spatial_scale    = 1.5f;
                } else if (config.spatial_up_num == 4) {
                    config.spatial_down_den = 1;
                    config.spatial_scale    = 4.f;
                } else {
                    config.spatial_down_den = 1;
                    config.spatial_scale    = static_cast<float>(config.spatial_up_num);
                }
            }

            if (inferred) {
                LOG_DEBUG("ltx latent upsampler: in_channels = %" PRId64 ", mid_channels = %" PRId64 ", num_blocks_per_stage = %d, spatial_scale = %.3f, temporal_up_factor = %d, rational_resampler = %d",
                          config.in_channels,
                          config.mid_channels,
                          config.num_blocks_per_stage,
                          config.spatial_scale,
                          config.temporal_up_factor,
                          config.rational_resampler);
            }
            return config;
        }
    };

    class VideoGroupNorm : public GGMLBlock {
    protected:
        int num_groups;
        int64_t num_channels;
        float eps;
        std::string prefix;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        }

    public:
        VideoGroupNorm(int num_groups, int64_t num_channels, float eps = 1e-05f)
            : num_groups(num_groups),
              num_channels(num_channels),
              eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // LTX video latent layout is [W, H, T, C]. ggml_group_norm treats ne[2]
            // as channels, so fold only H/T internally and restore the same layout.
            GGML_ASSERT(x->ne[3] == num_channels);
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t T = x->ne[2];
            x               = ggml_ext_cont(ctx->ggml_ctx, x);
            x               = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, num_channels, 1);
            x               = ggml_group_norm(ctx->ggml_ctx, x, num_groups, eps);

            ggml_tensor* weight = params["weight"];
            ggml_tensor* bias   = params["bias"];
            if (ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, weight, prefix + "weight");
                bias   = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, bias, prefix + "bias");
            }
            weight = ggml_reshape_4d(ctx->ggml_ctx, weight, 1, 1, num_channels, 1);
            bias   = ggml_reshape_4d(ctx->ggml_ctx, bias, 1, 1, num_channels, 1);
            x      = ggml_mul_inplace(ctx->ggml_ctx, x, weight);
            x      = ggml_add_inplace(ctx->ggml_ctx, x, bias);
            return ggml_reshape_4d(ctx->ggml_ctx, x, W, H, T, num_channels);
        }
    };

    class ResBlock : public GGMLBlock {
    public:
        ResBlock(int64_t channels, int dims = 3) {
            GGML_ASSERT(dims == 3);
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv3d(channels, channels, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new VideoGroupNorm(32, channels));
            blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv3d(channels, channels, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new VideoGroupNorm(32, channels));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv1 = std::dynamic_pointer_cast<Conv3d>(blocks["conv1"]);
            auto norm1 = std::dynamic_pointer_cast<VideoGroupNorm>(blocks["norm1"]);
            auto conv2 = std::dynamic_pointer_cast<Conv3d>(blocks["conv2"]);
            auto norm2 = std::dynamic_pointer_cast<VideoGroupNorm>(blocks["norm2"]);

            ggml_tensor* residual = x;

            x = conv1->forward(ctx, x);
            x = norm1->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = conv2->forward(ctx, x);
            x = norm2->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, x, residual);
            return ggml_silu(ctx->ggml_ctx, x);
        }
    };

    class PixelShuffleND : public UnaryBlock {
    protected:
        int upscale_factor;

    public:
        explicit PixelShuffleND(int upscale_factor)
            : upscale_factor(upscale_factor) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_ASSERT(upscale_factor > 0);
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];
            GGML_ASSERT(x->ne[2] % (upscale_factor * upscale_factor) == 0);
            // x: [b*f, c*p1*p2, h, w] -> [b*f, c, h*p1, w*p2]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [b*f, h, w, c*p1*p2]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);          // [b*f, h*w, c*p1*p2]
            return DiT::unpatchify(ctx->ggml_ctx, x, h, w, upscale_factor, upscale_factor, true);
        }
    };

    class TemporalPixelShuffleND : public UnaryBlock {
    protected:
        int upscale_factor;

    public:
        explicit TemporalPixelShuffleND(int upscale_factor)
            : upscale_factor(upscale_factor) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_ASSERT(upscale_factor > 0);
            GGML_ASSERT(x->ne[3] % upscale_factor == 0);
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t F = x->ne[2];
            const int64_t C = x->ne[3] / upscale_factor;

            // x: [b, c*p, f, h, w] -> [b, c, f*p, h, w]
            x = ggml_ext_cont(ctx->ggml_ctx, x);
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, F, upscale_factor, C);
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));
            return ggml_reshape_4d(ctx->ggml_ctx, x, W, H, F * upscale_factor, C);
        }
    };

    class BlurDownsample : public GGMLBlock {
    protected:
        int64_t channels;
        int stride;
        std::vector<float> kernel_data;
        std::string kernel_name;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(ctx);
            SD_UNUSED(tensor_storage_map);
            if (stride == 1) {
                return;
            }
            kernel_name = prefix + "kernel";
        }

    public:
        BlurDownsample(int64_t channels, int stride)
            : channels(channels),
              stride(stride) {
            GGML_ASSERT(stride >= 1);
            static const float binomial[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
            kernel_data.resize(static_cast<size_t>(5 * 5 * channels));
            for (int64_t c = 0; c < channels; ++c) {
                for (int y = 0; y < 5; ++y) {
                    for (int x = 0; x < 5; ++x) {
                        kernel_data[static_cast<size_t>(x + 5 * (y + 5 * c))] =
                            binomial[y] * binomial[x] / 256.f;
                    }
                }
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (stride == 1) {
                return x;
            }
            GGML_ASSERT(ctx != nullptr);
            GGML_ASSERT(!kernel_data.empty());
            GGML_ASSERT(x->ne[2] == channels);
            ggml_tensor* kernel = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F32, 5, 5, 1, channels);
            ggml_set_name(kernel, kernel_name.empty() ? "blur_down.kernel" : kernel_name.c_str());
            ctx->bind_backend_tensor_data(kernel, kernel_data.data());
            if (ctx->conv2d_direct_enabled) {
                return ggml_conv_2d_dw_direct(ctx->ggml_ctx, kernel, x, stride, stride, 2, 2, 1, 1);
            }
            return ggml_conv_2d_dw(ctx->ggml_ctx, kernel, x, stride, stride, 2, 2, 1, 1);
        }
    };

    class SpatialRationalResampler : public GGMLBlock {
    protected:
        int64_t mid_channels;
        int num;
        int den;

    public:
        SpatialRationalResampler(int64_t mid_channels, int num, int den)
            : mid_channels(mid_channels),
              num(num),
              den(den) {
            GGML_ASSERT(num >= 1);
            GGML_ASSERT(den >= 1);
            blocks["conv"]          = std::shared_ptr<GGMLBlock>(new Conv2d(mid_channels, num * num * mid_channels, {3, 3}, {1, 1}, {1, 1}));
            blocks["pixel_shuffle"] = std::shared_ptr<GGMLBlock>(new PixelShuffleND(num));
            blocks["blur_down"]     = std::shared_ptr<GGMLBlock>(new BlurDownsample(mid_channels, den));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv          = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);
            auto pixel_shuffle = std::dynamic_pointer_cast<PixelShuffleND>(blocks["pixel_shuffle"]);
            auto blur_down     = std::dynamic_pointer_cast<BlurDownsample>(blocks["blur_down"]);

            // rearrange(x, "b c f h w -> (b f) c h w")
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
            x = conv->forward(ctx, x);
            x = pixel_shuffle->forward(ctx, x);
            x = blur_down->forward(ctx, x);
            return ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
        }
    };

    class LatentUpsampler : public GGMLBlock {
    public:
        LatentUpsamplerConfig config;

        explicit LatentUpsampler(LatentUpsamplerConfig config)
            : config(std::move(config)) {
            GGML_ASSERT(this->config.dims == 3);
            GGML_ASSERT(this->config.spatial_upsample || this->config.temporal_upsample);

            blocks["initial_conv"] = std::shared_ptr<GGMLBlock>(new Conv3d(this->config.in_channels,
                                                                           this->config.mid_channels,
                                                                           {3, 3, 3},
                                                                           {1, 1, 1},
                                                                           {1, 1, 1}));
            blocks["initial_norm"] = std::shared_ptr<GGMLBlock>(new VideoGroupNorm(32, this->config.mid_channels));
            for (int i = 0; i < this->config.num_blocks_per_stage; ++i) {
                blocks["res_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResBlock(this->config.mid_channels, this->config.dims));
            }
            if (this->config.rational_resampler) {
                blocks["upsampler"] = std::shared_ptr<GGMLBlock>(new SpatialRationalResampler(this->config.mid_channels,
                                                                                              this->config.spatial_up_num,
                                                                                              this->config.spatial_down_den));
            } else if (this->config.temporal_upsample) {
                blocks["upsampler.0"] = std::shared_ptr<GGMLBlock>(new Conv3d(this->config.mid_channels,
                                                                              this->config.temporal_up_factor * this->config.mid_channels,
                                                                              {3, 3, 3},
                                                                              {1, 1, 1},
                                                                              {1, 1, 1}));
                blocks["upsampler.1"] = std::shared_ptr<GGMLBlock>(new TemporalPixelShuffleND(this->config.temporal_up_factor));
            } else {
                blocks["upsampler.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.mid_channels,
                                                                              4 * this->config.mid_channels,
                                                                              {3, 3},
                                                                              {1, 1},
                                                                              {1, 1}));
                blocks["upsampler.1"] = std::shared_ptr<GGMLBlock>(new PixelShuffleND(2));
            }
            for (int i = 0; i < this->config.num_blocks_per_stage; ++i) {
                blocks["post_upsample_res_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResBlock(this->config.mid_channels, this->config.dims));
            }
            blocks["final_conv"] = std::shared_ptr<GGMLBlock>(new Conv3d(this->config.mid_channels,
                                                                         this->config.in_channels,
                                                                         {3, 3, 3},
                                                                         {1, 1, 1},
                                                                         {1, 1, 1}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [b, c, f, h, w]
            // return: [b, c, scaled_f, scaled_h, scaled_w]
            auto initial_conv = std::dynamic_pointer_cast<Conv3d>(blocks["initial_conv"]);
            auto initial_norm = std::dynamic_pointer_cast<VideoGroupNorm>(blocks["initial_norm"]);
            auto final_conv   = std::dynamic_pointer_cast<Conv3d>(blocks["final_conv"]);

            x = initial_conv->forward(ctx, x);
            x = initial_norm->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.initial", "x");

            for (int i = 0; i < config.num_blocks_per_stage; ++i) {
                auto block = std::dynamic_pointer_cast<ResBlock>(blocks["res_blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
                sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.res_blocks." + std::to_string(i), "x");
            }

            if (config.rational_resampler) {
                auto upsampler = std::dynamic_pointer_cast<SpatialRationalResampler>(blocks["upsampler"]);
                x              = upsampler->forward(ctx, x);
            } else if (config.temporal_upsample) {
                auto upsample_conv = std::dynamic_pointer_cast<Conv3d>(blocks["upsampler.0"]);
                auto pixel_shuffle = std::dynamic_pointer_cast<TemporalPixelShuffleND>(blocks["upsampler.1"]);
                x                  = upsample_conv->forward(ctx, x);                    // [b, c*2, f, h, w]
                x                  = pixel_shuffle->forward(ctx, x);                    // [b, c, f*2, h, w]
                x                  = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, x->ne[2]);  // x[:, :, 1:, :, :]
            } else {
                auto upsample_conv = std::dynamic_pointer_cast<Conv2d>(blocks["upsampler.0"]);
                auto pixel_shuffle = std::dynamic_pointer_cast<PixelShuffleND>(blocks["upsampler.1"]);

                // rearrange(x, "b c f h w -> (b f) c h w"),
                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // [b*f, c, h, w]
                x = upsample_conv->forward(ctx, x);                                                      // [b*f, c*4, h, w]
                x = pixel_shuffle->forward(ctx, x);                                                      // [b*f, c, h*2, w*2]
                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // [b*c, f, h, w]
            }
            sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.spatial_up", "x");

            for (int i = 0; i < config.num_blocks_per_stage; ++i) {
                auto block = std::dynamic_pointer_cast<ResBlock>(blocks["post_upsample_res_blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
                sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.post_blocks." + std::to_string(i), "x");
            }

            x = final_conv->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.final", "x");
            return x;
        }
    };

    struct LatentUpsamplerRunner : public GGMLRunner {
        LatentUpsamplerConfig config;
        std::unique_ptr<LatentUpsampler> model;

        LatentUpsamplerRunner(ggml_backend_t backend,
                              const String2TensorStorage& tensor_storage_map,
                              std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager),
              config(LatentUpsamplerConfig::detect_from_weights(tensor_storage_map)) {
            if (config.dims != 3 || (!config.spatial_upsample && !config.temporal_upsample) ||
                config.spatial_up_num < 1 || config.spatial_down_den < 1 || config.temporal_up_factor < 1) {
                LOG_ERROR("unsupported LTX latent upsampler config: dims=%d spatial=%d temporal=%d rational=%d scale=%.3f temporal_factor=%d",
                          config.dims,
                          config.spatial_upsample,
                          config.temporal_upsample,
                          config.rational_resampler,
                          config.spatial_scale,
                          config.temporal_up_factor);
                return;
            }

            model = std::make_unique<LatentUpsampler>(config);
            model->init(params_ctx, tensor_storage_map, "");
        }

        std::string get_desc() override {
            return "ltx_latent_upsampler";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
            if (model) {
                model->get_param_tensors(tensors);
            }
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
            if (!model) {
                return nullptr;
            }
            ggml_cgraph* gf  = new_graph_custom(LTX_UPSAMPLER_GRAPH_SIZE);
            ggml_tensor* x   = make_input(x_tensor);
            auto runner_ctx  = get_context();
            ggml_tensor* out = model->forward(&runner_ctx, x);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<float>& x) {
            if (!model) {
                LOG_ERROR("LTX latent upsampler is not loaded");
                return {};
            }
            if (x.dim() != 4 && x.dim() != 5) {
                LOG_ERROR("LTX latent upsampler expects 4D or 5D video latent, got dim=%lld",
                          (long long)x.dim());
                return {};
            }
            if (x.dim() == 5 && x.shape()[4] != 1) {
                LOG_ERROR("LTX latent upsampler currently supports batch size 1, got batch=%lld",
                          (long long)x.shape()[4]);
                return {};
            }
            if (x.shape()[3] != config.in_channels) {
                LOG_ERROR("LTX latent upsampler expected %" PRId64 " channels, got %lld",
                          config.in_channels,
                          (long long)x.shape()[3]);
                return {};
            }
            size_t expected_dim = static_cast<size_t>(x.dim());
            auto get_graph      = [&]() -> ggml_cgraph* { return build_graph(x); };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), expected_dim);
        }
    };

}  // namespace LTXVUpsampler

#endif  // __SD_MODEL_UPSCALER_LTX_LATENT_UPSCALER_HPP__
