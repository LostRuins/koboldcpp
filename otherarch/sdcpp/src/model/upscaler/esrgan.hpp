#ifndef __SD_MODEL_UPSCALER_ESRGAN_HPP__
#define __SD_MODEL_UPSCALER_ESRGAN_HPP__

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/util.h"

/*
    ===================================    ESRGAN  ===================================
    References:
    https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py

*/

struct ESRGANConfig {
    int scale       = 4;
    int num_block   = 23;
    int num_in_ch   = 3;
    int num_out_ch  = 3;
    int num_feat    = 64;
    int num_grow_ch = 32;

    static ESRGANConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                            const std::string& prefix = "") {
        ESRGANConfig config;
        auto find_weight = [&](const std::string& suffix) -> const TensorStorage* {
            std::string name = prefix.empty() ? suffix : prefix + "." + suffix;
            auto iter        = tensor_storage_map.find(name);
            if (iter == tensor_storage_map.end()) {
                return nullptr;
            }
            return &iter->second;
        };

        int detected_num_block        = 0;
        const std::string body_prefix = prefix.empty() ? "body." : prefix + ".body.";
        for (const auto& [name, _] : tensor_storage_map) {
            if (!starts_with(name, body_prefix)) {
                continue;
            }
            size_t pos = name.find('.', body_prefix.size());
            if (pos == std::string::npos) {
                continue;
            }
            try {
                int idx            = std::stoi(name.substr(body_prefix.size(), pos - body_prefix.size()));
                detected_num_block = std::max(detected_num_block, idx + 1);
            } catch (...) {
            }
        }
        if (detected_num_block > 0) {
            config.num_block = detected_num_block;
        }

        bool has_conv_up2 = find_weight("conv_up2.weight") != nullptr;
        bool has_conv_up1 = find_weight("conv_up1.weight") != nullptr;
        bool has_model_tensor =
            detected_num_block > 0 ||
            find_weight("conv_first.weight") != nullptr ||
            find_weight("conv_hr.weight") != nullptr ||
            find_weight("conv_last.weight") != nullptr;
        if (has_conv_up2) {
            config.scale = 4;
        } else if (has_conv_up1) {
            config.scale = 2;
        } else if (has_model_tensor) {
            config.scale = 1;
        }

        if (has_model_tensor || has_conv_up1 || has_conv_up2) {
            LOG_DEBUG("esrgan: scale = %d, num_block = %d, num_in_ch = %d, num_out_ch = %d, num_feat = %d, num_grow_ch = %d",
                      config.scale,
                      config.num_block,
                      config.num_in_ch,
                      config.num_out_ch,
                      config.num_feat,
                      config.num_grow_ch);
        }
        return config;
    }
};

class ResidualDenseBlock : public GGMLBlock {
protected:
    int num_feat;
    int num_grow_ch;

public:
    ResidualDenseBlock(int num_feat = 64, int num_grow_ch = 32)
        : num_feat(num_feat), num_grow_ch(num_grow_ch) {
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv3"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv4"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv5"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 4 * num_grow_ch, num_feat, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* lrelu(GGMLRunnerContext* ctx, ggml_tensor* x) {
        return ggml_leaky_relu(ctx->ggml_ctx, x, 0.2f, true);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [n, num_feat, h, w]
        // return: [n, num_feat, h, w]

        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);
        auto conv3 = std::dynamic_pointer_cast<Conv2d>(blocks["conv3"]);
        auto conv4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv4"]);
        auto conv5 = std::dynamic_pointer_cast<Conv2d>(blocks["conv5"]);

        auto x1    = lrelu(ctx, conv1->forward(ctx, x));
        auto x_cat = ggml_concat(ctx->ggml_ctx, x, x1, 2);
        auto x2    = lrelu(ctx, conv2->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx->ggml_ctx, x_cat, x2, 2);
        auto x3    = lrelu(ctx, conv3->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx->ggml_ctx, x_cat, x3, 2);
        auto x4    = lrelu(ctx, conv4->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx->ggml_ctx, x_cat, x4, 2);
        auto x5    = conv5->forward(ctx, x_cat);

        x5 = ggml_add(ctx->ggml_ctx, ggml_ext_scale(ctx->ggml_ctx, x5, 0.2f), x);
        return x5;
    }
};

class RRDB : public GGMLBlock {
public:
    RRDB(int num_feat, int num_grow_ch = 32) {
        blocks["rdb1"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
        blocks["rdb2"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
        blocks["rdb3"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [n, num_feat, h, w]
        // return: [n, num_feat, h, w]

        auto rdb1 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb1"]);
        auto rdb2 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb2"]);
        auto rdb3 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb3"]);

        auto out = rdb1->forward(ctx, x);
        out      = rdb2->forward(ctx, out);
        out      = rdb3->forward(ctx, out);

        out = ggml_add(ctx->ggml_ctx, ggml_ext_scale(ctx->ggml_ctx, out, 0.2f), x);
        return out;
    }
};

class RRDBNet : public GGMLBlock {
protected:
    ESRGANConfig config;

public:
    explicit RRDBNet(ESRGANConfig config)
        : config(std::move(config)) {
        blocks["conv_first"] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.num_in_ch, this->config.num_feat, {3, 3}, {1, 1}, {1, 1}));
        for (int i = 0; i < this->config.num_block; i++) {
            std::string name = "body." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new RRDB(this->config.num_feat, this->config.num_grow_ch));
        }
        blocks["conv_body"] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.num_feat, this->config.num_feat, {3, 3}, {1, 1}, {1, 1}));
        if (this->config.scale >= 2) {
            blocks["conv_up1"] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.num_feat, this->config.num_feat, {3, 3}, {1, 1}, {1, 1}));
        }
        if (this->config.scale == 4) {
            blocks["conv_up2"] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.num_feat, this->config.num_feat, {3, 3}, {1, 1}, {1, 1}));
        }
        blocks["conv_hr"]   = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.num_feat, this->config.num_feat, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv_last"] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.num_feat, this->config.num_out_ch, {3, 3}, {1, 1}, {1, 1}));
    }

    int get_scale() { return config.scale; }
    int get_num_block() { return config.num_block; }

    ggml_tensor* lrelu(GGMLRunnerContext* ctx, ggml_tensor* x) {
        return ggml_leaky_relu(ctx->ggml_ctx, x, 0.2f, true);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [n, num_in_ch, h, w]
        // return: [n, num_out_ch, h*scale, w*scale]
        auto conv_first = std::dynamic_pointer_cast<Conv2d>(blocks["conv_first"]);
        auto conv_body  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_body"]);
        auto conv_hr    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_hr"]);
        auto conv_last  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_last"]);

        auto feat = conv_first->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(feat, "esrgan.prelude", "feat");
        auto body_feat = feat;
        for (int i = 0; i < config.num_block; i++) {
            std::string name = "body." + std::to_string(i);
            auto block       = std::dynamic_pointer_cast<RRDB>(blocks[name]);

            body_feat = block->forward(ctx, body_feat);
            sd::ggml_graph_cut::mark_graph_cut(body_feat, "esrgan.body." + std::to_string(i), "feat");
        }
        body_feat = conv_body->forward(ctx, body_feat);
        feat      = ggml_add(ctx->ggml_ctx, feat, body_feat);
        sd::ggml_graph_cut::mark_graph_cut(feat, "esrgan.body.out", "feat");
        // upsample
        if (config.scale >= 2) {
            auto conv_up1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv_up1"]);
            feat          = lrelu(ctx, conv_up1->forward(ctx, ggml_upscale(ctx->ggml_ctx, feat, 2, GGML_SCALE_MODE_NEAREST)));
            sd::ggml_graph_cut::mark_graph_cut(feat, "esrgan.up1", "feat");
            if (config.scale == 4) {
                auto conv_up2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv_up2"]);
                feat          = lrelu(ctx, conv_up2->forward(ctx, ggml_upscale(ctx->ggml_ctx, feat, 2, GGML_SCALE_MODE_NEAREST)));
                sd::ggml_graph_cut::mark_graph_cut(feat, "esrgan.up2", "feat");
            }
        }
        // for all scales
        auto out = conv_last->forward(ctx, lrelu(ctx, conv_hr->forward(ctx, feat)));
        sd::ggml_graph_cut::mark_graph_cut(out, "esrgan.final", "out");
        return out;
    }
};

struct ESRGAN : public GGMLRunner {
    ESRGANConfig config;
    std::unique_ptr<RRDBNet> rrdb_net;

    ESRGAN(ggml_backend_t backend,
           const String2TensorStorage& tensor_storage_map      = {},
           std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          config(ESRGANConfig::detect_from_weights(tensor_storage_map)),
          rrdb_net(std::make_unique<RRDBNet>(config)) {
        rrdb_net->init(params_ctx, tensor_storage_map, "");
    }

    std::string get_desc() override {
        return "esrgan";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        if (!rrdb_net) {
            return;
        }

        rrdb_net->get_param_tensors(tensors);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
        if (!rrdb_net)
            return nullptr;
        constexpr int kGraphNodes = 1 << 16;  // 65k
        ggml_cgraph* gf           = new_graph_custom(kGraphNodes);
        ggml_tensor* x            = make_input(x_tensor);

        auto runner_ctx  = get_context();
        ggml_tensor* out = rrdb_net->forward(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<float>& x) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(x); };
        auto result    = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        return result;
    }
};

#endif  // __SD_MODEL_UPSCALER_ESRGAN_HPP__
