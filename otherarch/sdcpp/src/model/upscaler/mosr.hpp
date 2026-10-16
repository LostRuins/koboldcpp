#ifndef __SD_MODEL_UPSCALER_MOSR_HPP__
#define __SD_MODEL_UPSCALER_MOSR_HPP__

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "core/util.h"
#include "model/common/ggml_block.hpp"

/*
    ===================================    MoSR  ===================================
    References:
    https://github.com/umzi2/MoSR
    https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/architectures/MoSR/__arch/mosr_arch.py

    Only the PixelShuffle ("ps") upsampler is supported.
*/

struct MoSRConfig {
    int scale       = 4;
    int n_block     = 24;
    int in_ch       = 3;
    int dim         = 64;
    int hidden      = 96;
    int conv_ch     = 64;
    int kernel_size = 7;

    bool is_valid() const {
        // The upscaler feeds RGB tiles, so only 3-channel models are usable.
        return in_ch == 3 && scale > 0 && n_block > 0 && conv_ch <= hidden;
    }

    static MoSRConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                          const std::string& prefix = "") {
        MoSRConfig config;
        auto find_weight = [&](const std::string& suffix) -> const TensorStorage* {
            std::string name = prefix.empty() ? suffix : prefix + "." + suffix;
            auto iter        = tensor_storage_map.find(name);
            if (iter == tensor_storage_map.end() || iter->second.n_dims != 4) {
                return nullptr;
            }
            return &iter->second;
        };

        const TensorStorage* first = find_weight("gblocks.0.weight");
        const TensorStorage* fc1   = find_weight("gblocks.1.fc1.weight");
        const TensorStorage* conv  = find_weight("gblocks.1.conv.weight");
        const TensorStorage* up    = find_weight("upsampler.0.weight");
        if (first == nullptr || fc1 == nullptr || conv == nullptr || up == nullptr) {
            // Missing tensors, or the unsupported "gps"/"dys" upsamplers.
            config.n_block = 0;
            return config;
        }

        int seq_len                      = 0;
        const std::string gblocks_prefix = prefix.empty() ? "gblocks." : prefix + ".gblocks.";
        for (const auto& [name, _] : tensor_storage_map) {
            if (!starts_with(name, gblocks_prefix)) {
                continue;
            }
            size_t pos = name.find('.', gblocks_prefix.size());
            if (pos == std::string::npos) {
                continue;
            }
            try {
                int idx = std::stoi(name.substr(gblocks_prefix.size(), pos - gblocks_prefix.size()));
                seq_len = std::max(seq_len, idx + 1);
            } catch (...) {
            }
        }

        // gblocks = [conv_in, n_block x GatedCNNBlock, conv, Mish, conv, Mish, conv_1x1]
        config.n_block     = seq_len - 6;
        config.in_ch       = (int)first->ne[2];
        config.dim         = (int)first->ne[3];
        config.hidden      = (int)fc1->ne[3] / 2;
        config.conv_ch     = (int)conv->ne[3];
        config.kernel_size = (int)conv->ne[0];
        config.scale       = (int)std::lround(std::sqrt((double)up->ne[3] / config.in_ch));
        if (config.scale * config.scale * config.in_ch != up->ne[3]) {
            config.scale = 0;
        }

        LOG_VERBOSE("mosr: scale = %d, n_block = %d, in_ch = %d, dim = %d, hidden = %d, conv_ch = %d, kernel_size = %d",
                    config.scale,
                    config.n_block,
                    config.in_ch,
                    config.dim,
                    config.hidden,
                    config.conv_ch,
                    config.kernel_size);
        return config;
    }
};

static inline ggml_tensor* mosr_mish(ggml_context* ctx, ggml_tensor* x) {
    return ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)));
}

class MoSRLayerNorm2d : public UnaryBlock {
protected:
    int64_t channels;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    }

public:
    explicit MoSRLayerNorm2d(int64_t channels)
        : channels(channels) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // [N, C, H, W] -> [N, H, W, C] so the norm reduces over channels.
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
        x = ggml_ext_layer_norm(ctx->ggml_ctx, x, params["weight"], params["bias"], 1e-6f);
        return ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
    }
};

class MoSRDepthwiseConv2d : public UnaryBlock {
protected:
    int64_t channels;
    int kernel_size;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        // The direct depthwise kernels of the CUDA and Vulkan backends need an F32 kernel.
        params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kernel_size, kernel_size, 1, channels);
        params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    }

public:
    MoSRDepthwiseConv2d(int64_t channels, int kernel_size)
        : channels(channels), kernel_size(kernel_size) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        const int pad = kernel_size / 2;
        x             = ggml_conv_2d_dw_direct(ctx->ggml_ctx, params["weight"], ggml_ext_cont(ctx->ggml_ctx, x), 1, 1, pad, pad, 1, 1);
        auto b        = ggml_reshape_4d(ctx->ggml_ctx, params["bias"], 1, 1, channels, 1);
        return ggml_add_inplace(ctx->ggml_ctx, x, b);
    }
};

class MoSRGatedCNNBlock : public GGMLBlock {
protected:
    int hidden;
    int conv_ch;

public:
    MoSRGatedCNNBlock(int dim, int hidden, int conv_ch, int kernel_size)
        : hidden(hidden), conv_ch(conv_ch) {
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new MoSRLayerNorm2d(dim));
        blocks["fc1"]  = std::shared_ptr<GGMLBlock>(new Conv2d(dim, hidden * 2, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new MoSRDepthwiseConv2d(conv_ch, kernel_size));
        blocks["fc2"]  = std::shared_ptr<GGMLBlock>(new Conv2d(hidden, dim, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [n, dim, h, w]
        // return: [n, dim, h, w]
        auto norm = std::dynamic_pointer_cast<MoSRLayerNorm2d>(blocks["norm"]);
        auto fc1  = std::dynamic_pointer_cast<Conv2d>(blocks["fc1"]);
        auto conv = std::dynamic_pointer_cast<MoSRDepthwiseConv2d>(blocks["conv"]);
        auto fc2  = std::dynamic_pointer_cast<Conv2d>(blocks["fc2"]);

        auto h = fc1->forward(ctx, norm->forward(ctx, x));

        auto channel_view = [&](int64_t offset, int64_t count) {
            return ggml_view_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], count, h->ne[3],
                                h->nb[1], h->nb[2], h->nb[3], offset * h->nb[2]);
        };
        auto g = channel_view(0, hidden);
        auto i = channel_view(hidden, hidden - conv_ch);
        auto c = conv->forward(ctx, channel_view(2 * hidden - conv_ch, conv_ch));
        if (hidden > conv_ch) {
            c = ggml_concat(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, i), c, 2);
        }

        h = ggml_mul(ctx->ggml_ctx, c, mosr_mish(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, g)));
        h = mosr_mish(ctx->ggml_ctx, fc2->forward(ctx, h));
        return ggml_add(ctx->ggml_ctx, h, ggml_scale_bias(ctx->ggml_ctx, x, 1.f, -0.5f));
    }
};

class MoSRConvBlock : public GGMLBlock {
public:
    MoSRConvBlock(int in_ch, int out_ch) {
        blocks["block.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_ch, out_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["block.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_ch, out_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv11"]  = std::shared_ptr<GGMLBlock>(new Conv2d(in_ch, out_ch, {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto conv0  = std::dynamic_pointer_cast<Conv2d>(blocks["block.0"]);
        auto conv2  = std::dynamic_pointer_cast<Conv2d>(blocks["block.2"]);
        auto conv11 = std::dynamic_pointer_cast<Conv2d>(blocks["conv11"]);

        auto out = mosr_mish(ctx->ggml_ctx, conv0->forward(ctx, x));
        out      = mosr_mish(ctx->ggml_ctx, conv2->forward(ctx, out));
        return ggml_add(ctx->ggml_ctx, out, conv11->forward(ctx, x));
    }
};

class MoSRNet : public GGMLBlock {
protected:
    MoSRConfig config;

    static std::string gblock_name(int idx) {
        return "gblocks." + std::to_string(idx);
    }

    // torch.nn.PixelShuffle: [n, c*r*r, h, w] -> [n, c, h*r, w*r]
    static ggml_tensor* pixel_shuffle(ggml_context* ctx, ggml_tensor* x, int r) {
        const int64_t w = x->ne[0];
        const int64_t h = x->ne[1];
        const int64_t c = x->ne[2] / (r * r);
        const int64_t n = x->ne[3];
        x               = ggml_reshape_4d(ctx, x, w, h, r, r * c * n);       // [n*c*r, r, h, w]
        x               = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [n*c*r, h, w, r]
        x               = ggml_reshape_4d(ctx, x, r * w, h, r, c * n);       // [n*c, r, h, w*r]
        x               = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [n*c, h, r, w*r]
        return ggml_reshape_4d(ctx, x, r * w, r * h, c, n);
    }

public:
    explicit MoSRNet(MoSRConfig config)
        : config(std::move(config)) {
        const int dim = this->config.dim;
        const int nb  = this->config.n_block;

        blocks[gblock_name(0)] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.in_ch, dim, {3, 3}, {1, 1}, {1, 1}));
        for (int i = 1; i <= nb; i++) {
            blocks[gblock_name(i)] = std::shared_ptr<GGMLBlock>(new MoSRGatedCNNBlock(dim,
                                                                                      this->config.hidden,
                                                                                      this->config.conv_ch,
                                                                                      this->config.kernel_size));
        }
        blocks[gblock_name(nb + 1)] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim * 2, {3, 3}, {1, 1}, {1, 1}));
        blocks[gblock_name(nb + 3)] = std::shared_ptr<GGMLBlock>(new Conv2d(dim * 2, dim, {3, 3}, {1, 1}, {1, 1}));
        blocks[gblock_name(nb + 5)] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {1, 1}));
        blocks["shortcut"]          = std::shared_ptr<GGMLBlock>(new MoSRConvBlock(this->config.in_ch, dim));
        blocks["upsampler.0"]       = std::shared_ptr<GGMLBlock>(new Conv2d(dim,
                                                                            this->config.in_ch * this->config.scale * this->config.scale,
                                                                            {3, 3},
                                                                            {1, 1},
                                                                            {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [n, in_ch, h, w]
        // return: [n, in_ch, h*scale, w*scale]
        const int nb   = config.n_block;
        auto conv_in   = std::dynamic_pointer_cast<Conv2d>(blocks[gblock_name(0)]);
        auto conv_mid1 = std::dynamic_pointer_cast<Conv2d>(blocks[gblock_name(nb + 1)]);
        auto conv_mid2 = std::dynamic_pointer_cast<Conv2d>(blocks[gblock_name(nb + 3)]);
        auto conv_out  = std::dynamic_pointer_cast<Conv2d>(blocks[gblock_name(nb + 5)]);
        auto shortcut  = std::dynamic_pointer_cast<MoSRConvBlock>(blocks["shortcut"]);
        auto upsampler = std::dynamic_pointer_cast<Conv2d>(blocks["upsampler.0"]);

        auto skip = shortcut->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(skip, "mosr.shortcut", "skip");

        auto feat = conv_in->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(feat, "mosr.prelude", "feat");
        for (int i = 1; i <= nb; i++) {
            auto block = std::dynamic_pointer_cast<MoSRGatedCNNBlock>(blocks[gblock_name(i)]);
            feat       = block->forward(ctx, feat);
            sd::ggml_graph_cut::mark_graph_cut(feat, "mosr.gblocks." + std::to_string(i), "feat");
        }
        feat = mosr_mish(ctx->ggml_ctx, conv_mid1->forward(ctx, feat));
        feat = mosr_mish(ctx->ggml_ctx, conv_mid2->forward(ctx, feat));
        feat = conv_out->forward(ctx, feat);
        feat = ggml_add(ctx->ggml_ctx, feat, ggml_scale_bias(ctx->ggml_ctx, skip, 1.f, -0.5f));
        sd::ggml_graph_cut::mark_graph_cut(feat, "mosr.body.out", "feat");

        auto out = pixel_shuffle(ctx->ggml_ctx, upsampler->forward(ctx, feat), config.scale);
        sd::ggml_graph_cut::mark_graph_cut(out, "mosr.final", "out");
        return out;
    }
};

struct MoSR : public GGMLRunner {
    MoSRConfig config;
    std::unique_ptr<MoSRNet> mosr_net;

    MoSR(ggml_backend_t backend,
         const String2TensorStorage& tensor_storage_map      = {},
         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          config(MoSRConfig::detect_from_weights(tensor_storage_map)) {
        if (!config.is_valid()) {
            return;
        }
        mosr_net = std::make_unique<MoSRNet>(config);
        mosr_net->init(params_ctx, tensor_storage_map, "");
    }

    std::string get_desc() override {
        return "mosr";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        if (!mosr_net) {
            return;
        }

        mosr_net->get_param_tensors(tensors);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
        if (!mosr_net)
            return nullptr;
        constexpr int kGraphNodes = 1 << 16;  // 65k
        ggml_cgraph* gf           = new_graph_custom(kGraphNodes);
        ggml_tensor* x            = make_input(x_tensor);

        auto runner_ctx  = get_context();
        ggml_tensor* out = mosr_net->forward(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<float>& x) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(x); };
        auto result    = restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), x.dim());
        return result;
    }
};

#endif  // __SD_MODEL_UPSCALER_MOSR_HPP__
