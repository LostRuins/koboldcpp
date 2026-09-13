#ifndef __SD_MODEL_AUDIO_WAV2VEC2_HPP__
#define __SD_MODEL_AUDIO_WAV2VEC2_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "model.h"
#include "model/common/ggml_block.hpp"

namespace Wav2Vec2 {

    struct Wav2Vec2Config {
        int64_t embed_dim             = 1024;
        int64_t conv_dim              = 512;
        int num_heads                 = 16;
        int num_layers                = 24;
        std::string feat_extract_norm = "layer";
        bool conv_bias                = true;
        bool do_normalize             = true;
        bool do_stable_layer_norm     = true;

        static Wav2Vec2Config detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            Wav2Vec2Config config;
            auto it = tensor_storage_map.find(prefix + "encoder.layer_norm.bias");
            if (it == tensor_storage_map.end()) {
                LOG_WARN("wav2vec2: %sencoder.layer_norm.bias not found, using large defaults", prefix.c_str());
                return config;
            }
            config.embed_dim = it->second.ne[0];
            if (config.embed_dim == 1024) {
                config.embed_dim            = 1024;
                config.num_heads            = 16;
                config.num_layers           = 24;
                config.feat_extract_norm    = "layer";
                config.conv_bias            = true;
                config.do_normalize         = true;
                config.do_stable_layer_norm = true;
            } else if (config.embed_dim == 768) {
                config.embed_dim            = 768;
                config.num_heads            = 12;
                config.num_layers           = 12;
                config.feat_extract_norm    = "group";
                config.conv_bias            = false;
                config.do_normalize         = false;
                config.do_stable_layer_norm = false;
            } else {
                LOG_WARN("wav2vec2: unsupported embed_dim %" PRId64 ", using large defaults", config.embed_dim);
                config.embed_dim = 1024;
            }
            return config;
        }
    };

    struct Wav2Vec2NoLayerNormConvLayer : public UnaryBlock {
        Wav2Vec2NoLayerNormConvLayer(int64_t in_channels, int64_t out_channels, int kernel_size, int stride, bool bias) {
            blocks["conv"] = std::make_shared<Conv1d>(in_channels, out_channels, kernel_size, stride, 0, 1, 1, bias, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto conv = std::dynamic_pointer_cast<Conv1d>(blocks["conv"]);
            x         = conv->forward(ctx, x);
            return ggml_gelu_erf_inplace(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, x));
        }
    };

    struct Wav2Vec2LayerNormConvLayer : public UnaryBlock {
        Wav2Vec2LayerNormConvLayer(int64_t in_channels, int64_t out_channels, int kernel_size, int stride, bool bias) {
            blocks["conv"]       = std::make_shared<Conv1d>(in_channels, out_channels, kernel_size, stride, 0, 1, 1, bias, true);
            blocks["layer_norm"] = std::make_shared<LayerNorm>(out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto conv       = std::dynamic_pointer_cast<Conv1d>(blocks["conv"]);
            auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
            x               = conv->forward(ctx, x);
            // LayerNorm normalizes channels: [N, C, L] -> [N, L, C].
            x = ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);
            x = layer_norm->forward(ctx, x);
            x = ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);
            return ggml_gelu_erf_inplace(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, x));
        }
    };

    struct Wav2Vec2GroupNormConvLayer : public UnaryBlock {
        Wav2Vec2GroupNormConvLayer(int64_t in_channels, int64_t out_channels, int kernel_size, int stride, bool bias) {
            blocks["conv"]       = std::make_shared<Conv1d>(in_channels, out_channels, kernel_size, stride, 0, 1, 1, bias, true);
            blocks["layer_norm"] = std::make_shared<GroupNorm>((int)out_channels, out_channels, 1e-05f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto conv       = std::dynamic_pointer_cast<Conv1d>(blocks["conv"]);
            auto layer_norm = std::dynamic_pointer_cast<GroupNorm>(blocks["layer_norm"]);
            x               = conv->forward(ctx, x);
            // ggml GroupNorm needs [N, C, H, W], with H=1 for audio.
            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
            x = layer_norm->forward(ctx, x);
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[2], x->ne[3]);
            return ggml_gelu_erf_inplace(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, x));
        }
    };

    struct Wav2Vec2FeatureEncoder : public UnaryBlock {
        Wav2Vec2FeatureEncoder(const Wav2Vec2Config& config) {
            GGML_ASSERT(config.feat_extract_norm == "layer" || config.feat_extract_norm == "group");
            const int kernels[7] = {10, 3, 3, 3, 3, 2, 2};
            const int strides[7] = {5, 2, 2, 2, 2, 2, 2};
            int64_t in_channels  = 1;
            for (int i = 0; i < 7; ++i) {
                const std::string name = "conv_layers." + std::to_string(i);
                if (config.feat_extract_norm == "layer") {
                    blocks[name] = std::make_shared<Wav2Vec2LayerNormConvLayer>(in_channels, config.conv_dim, kernels[i], strides[i], config.conv_bias);
                } else if (i == 0) {
                    blocks[name] = std::make_shared<Wav2Vec2GroupNormConvLayer>(in_channels, config.conv_dim, kernels[i], strides[i], config.conv_bias);
                } else {
                    blocks[name] = std::make_shared<Wav2Vec2NoLayerNormConvLayer>(in_channels, config.conv_dim, kernels[i], strides[i], config.conv_bias);
                }
                in_channels = config.conv_dim;
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            for (int i = 0; i < 7; ++i) {
                auto conv = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv_layers." + std::to_string(i)]);
                x         = conv->forward(ctx, x);
            }
            return ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);
        }
    };

    struct Wav2Vec2FeatureProjection : public UnaryBlock {
        Wav2Vec2FeatureProjection(const Wav2Vec2Config& config) {
            blocks["layer_norm"] = std::make_shared<LayerNorm>(config.conv_dim);
            blocks["projection"] = std::make_shared<Linear>(config.conv_dim, config.embed_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto ln         = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
            auto projection = std::dynamic_pointer_cast<Linear>(blocks["projection"]);
            x               = ln->forward(ctx, x);
            x               = projection->forward(ctx, x);
            return x;
        }
    };

    class Wav2Vec2PositionalConvEmbedding : public UnaryBlock {
    private:
        int64_t embed_dim_;
        static constexpr int groups_      = 16;
        static constexpr int kernel_size_ = 128;
        std::string weight_g_name_;
        std::string weight_v_name_;

        ggml_tensor* weight(GGMLRunnerContext* ctx) {
            auto g       = params[weight_g_name_];
            auto v       = ggml_cast(ctx->ggml_ctx, params[weight_v_name_], GGML_TYPE_F32);
            auto squared = ggml_mul(ctx->ggml_ctx, v, v);
            // PyTorch weight_norm(dim=2) reduces both channel axes, retaining each kernel tap.
            squared   = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, squared, 2, 0, 1, 3));
            squared   = ggml_reshape_2d(ctx->ggml_ctx, squared, embed_dim_ / groups_ * embed_dim_, kernel_size_);
            auto norm = ggml_sqrt(ctx->ggml_ctx, ggml_sum_rows(ctx->ggml_ctx, squared));
            norm      = ggml_reshape_3d(ctx->ggml_ctx, norm, kernel_size_, 1, 1);
            return ggml_mul(ctx->ggml_ctx, v, ggml_div(ctx->ggml_ctx, g, norm));
        }

    public:
        Wav2Vec2PositionalConvEmbedding(const Wav2Vec2Config& config)
            : embed_dim_(config.embed_dim) {
            GGML_ASSERT(embed_dim_ > 0 && embed_dim_ % groups_ == 0);
        }

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            bool legacy    = tensor_storage_map.count(prefix + "conv.weight_g") > 0;
            weight_g_name_ = legacy ? "conv.weight_g" : "conv.parametrizations.weight.original0";
            weight_v_name_ = legacy ? "conv.weight_v" : "conv.parametrizations.weight.original1";
            auto g         = tensor_storage_map.find(prefix + weight_g_name_);
            auto v         = tensor_storage_map.find(prefix + weight_v_name_);
            GGML_ASSERT(g != tensor_storage_map.end() && v != tensor_storage_map.end());
            GGML_ASSERT(g->second.ne[0] == kernel_size_ && g->second.ne[1] == 1 && g->second.ne[2] == 1 && g->second.ne[3] == 1);
            GGML_ASSERT(v->second.ne[0] == kernel_size_ && v->second.ne[1] == embed_dim_ / groups_ && v->second.ne[2] == embed_dim_ && v->second.ne[3] == 1);

            params[weight_g_name_] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size_, 1, 1);
            params[weight_v_name_] = ggml_new_tensor_3d(ctx, get_type(prefix + weight_v_name_, tensor_storage_map, GGML_TYPE_F16),
                                                        kernel_size_, embed_dim_ / groups_, embed_dim_);
            if (tensor_storage_map.count(prefix + "conv.bias") > 0) {
                params["conv.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim_);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto w = weight(ctx);
            auto b = params.count("conv.bias") > 0 ? params["conv.bias"] : nullptr;
            x      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x      = ggml_ext_conv_1d(ctx->ggml_ctx, x, w, b, 1, kernel_size_ / 2, 1, groups_, true);
            // Apply GELU out of place before cropping to keep graph buffer reuse safe.
            x = ggml_gelu_erf(ctx->ggml_ctx, x);
            x = ggml_view_3d(ctx->ggml_ctx, x, x->ne[0] - 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
            return ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);
        }
    };

    struct Wav2Vec2FeedForward : public UnaryBlock {
        Wav2Vec2FeedForward(const Wav2Vec2Config& config) {
            blocks["intermediate_dense"] = std::make_shared<Linear>(config.embed_dim, config.embed_dim * 4);
            blocks["output_dense"]       = std::make_shared<Linear>(config.embed_dim * 4, config.embed_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto intermediate_dense = std::dynamic_pointer_cast<Linear>(blocks["intermediate_dense"]);
            auto output_dense       = std::dynamic_pointer_cast<Linear>(blocks["output_dense"]);
            x                       = intermediate_dense->forward(ctx, x);
            x                       = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            x                       = output_dense->forward(ctx, x);
            return x;
        }
    };

    struct Wav2Vec2EncoderLayer : public UnaryBlock {
        bool do_stable_layer_norm;

        Wav2Vec2EncoderLayer(const Wav2Vec2Config& config)
            : do_stable_layer_norm(config.do_stable_layer_norm) {
            blocks["attention"]        = std::make_shared<MultiheadAttention>(config.embed_dim, config.num_heads, true, true);
            blocks["layer_norm"]       = std::make_shared<LayerNorm>(config.embed_dim);
            blocks["feed_forward"]     = std::make_shared<Wav2Vec2FeedForward>(config);
            blocks["final_layer_norm"] = std::make_shared<LayerNorm>(config.embed_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto attention        = std::dynamic_pointer_cast<MultiheadAttention>(blocks["attention"]);
            auto layer_norm       = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
            auto feed_forward     = std::dynamic_pointer_cast<Wav2Vec2FeedForward>(blocks["feed_forward"]);
            auto final_layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);

            ggml_tensor* residual = x;
            if (do_stable_layer_norm) {
                x = layer_norm->forward(ctx, x);
                x = attention->forward(ctx, x);
                x = ggml_add(ctx->ggml_ctx, residual, x);
                x = ggml_add(ctx->ggml_ctx, x, feed_forward->forward(ctx, final_layer_norm->forward(ctx, x)));
            } else {
                x = attention->forward(ctx, x);
                x = ggml_add(ctx->ggml_ctx, residual, x);
                x = layer_norm->forward(ctx, x);
                x = final_layer_norm->forward(ctx, ggml_add(ctx->ggml_ctx, x, feed_forward->forward(ctx, x)));
            }
            return x;
        }
    };

    struct Wav2Vec2Encoder : public GGMLBlock {
        int num_layers;
        bool do_stable_layer_norm;

        Wav2Vec2Encoder(const Wav2Vec2Config& config)
            : num_layers(config.num_layers), do_stable_layer_norm(config.do_stable_layer_norm) {
            blocks["pos_conv_embed"] = std::make_shared<Wav2Vec2PositionalConvEmbedding>(config);
            for (int i = 0; i < config.num_layers; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<Wav2Vec2EncoderLayer>(config);
            }
            blocks["layer_norm"] = std::make_shared<LayerNorm>(config.embed_dim);
        }

        // For N == 1, all_layers stacks pre-layer states and the final state as [embed_dim, L, num_layers + 1].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor** all_layers = nullptr) {
            auto pos_conv_embed = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
            auto layer_norm     = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);

            std::vector<ggml_tensor*> collected;
            if (all_layers != nullptr) {
                collected.reserve(num_layers + 1);
            }

            x = ggml_add(ctx->ggml_ctx, x, pos_conv_embed->forward(ctx, x));
            if (!do_stable_layer_norm) {
                x = layer_norm->forward(ctx, x);
            }
            for (int i = 0; i < num_layers; ++i) {
                if (all_layers != nullptr) {
                    collected.push_back(x);
                }
                auto layer = std::dynamic_pointer_cast<Wav2Vec2EncoderLayer>(blocks["layers." + std::to_string(i)]);
                x          = layer->forward(ctx, x);
            }
            if (do_stable_layer_norm) {
                x = layer_norm->forward(ctx, x);
            }
            if (all_layers != nullptr) {
                collected.push_back(x);
                ggml_tensor* stack = collected[0];
                for (size_t i = 1; i < collected.size(); ++i) {
                    stack = ggml_concat(ctx->ggml_ctx, stack, collected[i], 2);
                }
                *all_layers = stack;
            }
            return x;
        }
    };

    struct Wav2Vec2Model : public GGMLBlock {
        Wav2Vec2Config config;

        Wav2Vec2Model() = default;
        Wav2Vec2Model(const Wav2Vec2Config& config_)
            : config(config_) {
            blocks["feature_extractor"]  = std::make_shared<Wav2Vec2FeatureEncoder>(config);
            blocks["feature_projection"] = std::make_shared<Wav2Vec2FeatureProjection>(config);
            blocks["encoder"]            = std::make_shared<Wav2Vec2Encoder>(config);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor** all_layers = nullptr) {
            auto feature_extractor  = std::dynamic_pointer_cast<Wav2Vec2FeatureEncoder>(blocks["feature_extractor"]);
            auto feature_projection = std::dynamic_pointer_cast<Wav2Vec2FeatureProjection>(blocks["feature_projection"]);
            auto encoder            = std::dynamic_pointer_cast<Wav2Vec2Encoder>(blocks["encoder"]);

            x = feature_extractor->forward(ctx, x);
            x = feature_projection->forward(ctx, x);
            x = encoder->forward(ctx, x, all_layers);
            return x;
        }
    };

    class Wav2Vec2ModelRunner : public GGMLRunner {
    private:
        Wav2Vec2Config config;

    public:
        Wav2Vec2Model model;
        std::string weight_prefix;

        Wav2Vec2ModelRunner(ggml_backend_t backend,
                            const String2TensorStorage& tensor_storage_map      = {},
                            const std::string prefix                            = "wav2vec2.",
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager),
              config(Wav2Vec2Config::detect_from_weights(tensor_storage_map, prefix)),
              model(config),
              weight_prefix(prefix) {
            // GGMLBlock appends its own separator; loader prefixes already include one.
            std::string block_prefix = weight_prefix;
            if (!block_prefix.empty() && block_prefix.back() == '.') {
                block_prefix.pop_back();
            }
            model.init(params_ctx, tensor_storage_map, block_prefix);
            LOG_INFO("%s", get_desc().c_str());
        }

        std::string get_desc() override {
            return "wav2vec2";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
            std::string block_prefix = weight_prefix;
            if (!block_prefix.empty() && block_prefix.back() == '.') {
                block_prefix.pop_back();
            }
            model.get_param_tensors(tensors, block_prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& waveform_tensor) {
            ggml_cgraph* gf         = ggml_new_graph(compute_ctx);
            ggml_tensor* waveform   = make_input(waveform_tensor);
            auto runner_ctx         = get_context();
            ggml_tensor* all_layers = nullptr;
            model.forward(&runner_ctx, waveform, &all_layers);
            GGML_ASSERT(all_layers != nullptr);
            ggml_build_forward_expand(gf, all_layers);
            return gf;
        }

        sd::Tensor<float> compute(const int n_threads, const std::vector<float>& mono_waveform) {
            GGML_ASSERT(!mono_waveform.empty());
            const int64_t num_samples = (int64_t)mono_waveform.size();
            sd::Tensor<float> waveform({num_samples, 1, 1});
            std::copy(mono_waveform.begin(), mono_waveform.end(), waveform.data());
            normalize(waveform.data(), num_samples);

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(waveform);
            };
            return take_or_empty(GGMLRunner::compute(get_graph, n_threads, true));
        }

    private:
        static void normalize(float* x, int64_t n) {
            double mean = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                mean += x[i];
            }
            mean /= n;
            double var = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = x[i] - mean;
                var += d * d;
            }
            var /= n;
            const float scale = (float)(1.0 / std::sqrt(var + 1e-7));
            for (int64_t i = 0; i < n; ++i) {
                x[i] = (float)((x[i] - mean) * scale);
            }
        }
    };

}  // namespace Wav2Vec2

#endif  // __SD_MODEL_AUDIO_WAV2VEC2_HPP__
