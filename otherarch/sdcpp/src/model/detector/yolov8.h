#ifndef __SD_MODEL_DETECTOR_YOLOV8_H__
#define __SD_MODEL_DETECTOR_YOLOV8_H__

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/util.h"

struct YOLOv8Config {
    std::array<int, 23> out_channels{};
    std::map<int, int> hidden_channels;
    std::map<int, int> repeats;
    int detect_box_channels = 0;
    int detect_cls_channels = 0;
    int reg_max             = 0;
    int num_classes         = 0;
    bool valid              = false;

    static YOLOv8Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                            const std::string& prefix = "") {
        YOLOv8Config config;
        auto full_name = [&](const std::string& name) {
            return prefix.empty() ? name : prefix + "." + name;
        };
        auto find_weight = [&](const std::string& name) -> const TensorStorage* {
            auto iter = tensor_storage_map.find(full_name(name));
            return iter == tensor_storage_map.end() ? nullptr : &iter->second;
        };
        auto conv_out = [&](const std::string& name) -> int {
            const TensorStorage* weight = find_weight(name);
            return weight != nullptr && weight->n_dims == 4 ? static_cast<int>(weight->ne[3]) : 0;
        };

        for (int layer : {0, 1, 3, 5, 7, 16, 19}) {
            config.out_channels[layer] = conv_out("model." + std::to_string(layer) + ".conv.weight");
        }
        for (int layer : {2, 4, 6, 8, 12, 15, 18, 21}) {
            const std::string base        = "model." + std::to_string(layer);
            config.out_channels[layer]    = conv_out(base + ".cv2.conv.weight");
            config.hidden_channels[layer] = conv_out(base + ".cv1.conv.weight") / 2;

            int repeat_count = 0;
            while (find_weight(base + ".m." + std::to_string(repeat_count) + ".cv1.conv.weight") != nullptr) {
                ++repeat_count;
            }
            config.repeats[layer] = repeat_count;
        }
        config.out_channels[9] = conv_out("model.9.cv2.conv.weight");

        config.detect_box_channels = conv_out("model.22.cv2.0.0.conv.weight");
        config.detect_cls_channels = conv_out("model.22.cv3.0.0.conv.weight");
        const int box_outputs      = conv_out("model.22.cv2.0.2.weight");
        config.num_classes         = conv_out("model.22.cv3.0.2.weight");
        config.reg_max             = box_outputs / 4;

        config.valid = config.out_channels[0] > 0 && config.out_channels[9] > 0 &&
                       config.out_channels[15] > 0 && config.out_channels[18] > 0 &&
                       config.out_channels[21] > 0 && config.detect_box_channels > 0 &&
                       config.detect_cls_channels > 0 && box_outputs > 0 && box_outputs % 4 == 0 &&
                       config.num_classes > 0;
        for (int layer : {2, 4, 6, 8, 12, 15, 18, 21}) {
            config.valid = config.valid && config.hidden_channels[layer] > 0 && config.repeats[layer] > 0;
        }

        if (config.valid) {
            LOG_DEBUG("yolov8: classes=%d, reg_max=%d, p3=%d, p4=%d, p5=%d",
                      config.num_classes,
                      config.reg_max,
                      config.out_channels[15],
                      config.out_channels[18],
                      config.out_channels[21]);
        }
        return config;
    }
};

class YOLOConv : public UnaryBlock {
    int out_channels_ = 0;

public:
    YOLOConv(int in_channels, int out_channels, int kernel, int stride = 1)
        : out_channels_(out_channels) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                               out_channels,
                                                               {kernel, kernel},
                                                               {stride, stride},
                                                               {kernel / 2, kernel / 2},
                                                               {1, 1},
                                                               true));
    }

    int out_channels() const {
        return out_channels_;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);
        return ggml_silu_inplace(ctx->ggml_ctx, conv->forward(ctx, x));
    }
};

class YOLOBottleneck : public UnaryBlock {
    bool shortcut_ = false;

public:
    YOLOBottleneck(int channels, bool shortcut)
        : shortcut_(shortcut) {
        blocks["cv1"] = std::shared_ptr<GGMLBlock>(new YOLOConv(channels, channels, 3));
        blocks["cv2"] = std::shared_ptr<GGMLBlock>(new YOLOConv(channels, channels, 3));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto cv1 = std::dynamic_pointer_cast<YOLOConv>(blocks["cv1"]);
        auto cv2 = std::dynamic_pointer_cast<YOLOConv>(blocks["cv2"]);
        auto out = cv2->forward(ctx, cv1->forward(ctx, x));
        return shortcut_ ? ggml_add(ctx->ggml_ctx, x, out) : out;
    }
};

class YOLOC2f : public UnaryBlock {
    int hidden_channels_ = 0;
    int repeats_         = 0;

public:
    YOLOC2f(int in_channels,
            int out_channels,
            int hidden_channels,
            int repeats,
            bool shortcut)
        : hidden_channels_(hidden_channels), repeats_(repeats) {
        blocks["cv1"] = std::shared_ptr<GGMLBlock>(new YOLOConv(in_channels, hidden_channels * 2, 1));
        blocks["cv2"] = std::shared_ptr<GGMLBlock>(new YOLOConv(hidden_channels * (2 + repeats), out_channels, 1));
        for (int i = 0; i < repeats; ++i) {
            blocks["m." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new YOLOBottleneck(hidden_channels, shortcut));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto cv1   = std::dynamic_pointer_cast<YOLOConv>(blocks["cv1"]);
        auto cv2   = std::dynamic_pointer_cast<YOLOConv>(blocks["cv2"]);
        auto split = cv1->forward(ctx, x);

        // split: [N, 2*C, H, W], ggml layout [W, H, 2*C, N].
        auto y0     = ggml_view_4d(ctx->ggml_ctx,
                                   split,
                                   split->ne[0],
                                   split->ne[1],
                                   hidden_channels_,
                                   split->ne[3],
                                   split->nb[1],
                                   split->nb[2],
                                   split->nb[3],
                                   0);
        auto y1     = ggml_view_4d(ctx->ggml_ctx,
                                   split,
                                   split->ne[0],
                                   split->ne[1],
                                   hidden_channels_,
                                   split->ne[3],
                                   split->nb[1],
                                   split->nb[2],
                                   split->nb[3],
                                   static_cast<size_t>(hidden_channels_) * split->nb[2]);
        auto joined = ggml_concat(ctx->ggml_ctx, y0, y1, 2);
        auto last   = y1;
        for (int i = 0; i < repeats_; ++i) {
            auto block = std::dynamic_pointer_cast<YOLOBottleneck>(blocks["m." + std::to_string(i)]);
            last       = block->forward(ctx, last);
            joined     = ggml_concat(ctx->ggml_ctx, joined, last, 2);
        }
        return cv2->forward(ctx, joined);
    }
};

class YOLOSPPF : public UnaryBlock {
public:
    YOLOSPPF(int in_channels, int out_channels) {
        blocks["cv1"] = std::shared_ptr<GGMLBlock>(new YOLOConv(in_channels, in_channels / 2, 1));
        blocks["cv2"] = std::shared_ptr<GGMLBlock>(new YOLOConv(in_channels * 2, out_channels, 1));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto cv1 = std::dynamic_pointer_cast<YOLOConv>(blocks["cv1"]);
        auto cv2 = std::dynamic_pointer_cast<YOLOConv>(blocks["cv2"]);
        x        = cv1->forward(ctx, x);
        auto y1  = ggml_pool_2d(ctx->ggml_ctx, x, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2, 2);
        auto y2  = ggml_pool_2d(ctx->ggml_ctx, y1, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2, 2);
        auto y3  = ggml_pool_2d(ctx->ggml_ctx, y2, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2, 2);
        auto out = ggml_concat(ctx->ggml_ctx, x, y1, 2);
        out      = ggml_concat(ctx->ggml_ctx, out, y2, 2);
        out      = ggml_concat(ctx->ggml_ctx, out, y3, 2);
        return cv2->forward(ctx, out);
    }
};

class YOLODetect : public GGMLBlock {
    int num_classes_ = 0;
    int reg_max_     = 0;

public:
    YOLODetect(const std::array<int, 3>& in_channels,
               int box_channels,
               int cls_channels,
               int reg_max,
               int num_classes)
        : num_classes_(num_classes), reg_max_(reg_max) {
        for (int i = 0; i < 3; ++i) {
            const std::string box = "cv2." + std::to_string(i);
            blocks[box + ".0"]    = std::shared_ptr<GGMLBlock>(new YOLOConv(in_channels[i], box_channels, 3));
            blocks[box + ".1"]    = std::shared_ptr<GGMLBlock>(new YOLOConv(box_channels, box_channels, 3));
            blocks[box + ".2"]    = std::shared_ptr<GGMLBlock>(new Conv2d(box_channels, reg_max * 4, {1, 1}, {1, 1}, {0, 0}, {1, 1}, true));

            const std::string cls = "cv3." + std::to_string(i);
            blocks[cls + ".0"]    = std::shared_ptr<GGMLBlock>(new YOLOConv(in_channels[i], cls_channels, 3));
            blocks[cls + ".1"]    = std::shared_ptr<GGMLBlock>(new YOLOConv(cls_channels, cls_channels, 3));
            blocks[cls + ".2"]    = std::shared_ptr<GGMLBlock>(new Conv2d(cls_channels, num_classes, {1, 1}, {1, 1}, {0, 0}, {1, 1}, true));
        }
    }

    ggml_tensor* forward_scale(GGMLRunnerContext* ctx, ggml_tensor* x, int index) {
        const std::string box = "cv2." + std::to_string(index);
        auto box0             = std::dynamic_pointer_cast<YOLOConv>(blocks[box + ".0"]);
        auto box1             = std::dynamic_pointer_cast<YOLOConv>(blocks[box + ".1"]);
        auto box2             = std::dynamic_pointer_cast<Conv2d>(blocks[box + ".2"]);

        const std::string cls = "cv3." + std::to_string(index);
        auto cls0             = std::dynamic_pointer_cast<YOLOConv>(blocks[cls + ".0"]);
        auto cls1             = std::dynamic_pointer_cast<YOLOConv>(blocks[cls + ".1"]);
        auto cls2             = std::dynamic_pointer_cast<Conv2d>(blocks[cls + ".2"]);

        auto boxes   = box2->forward(ctx, box1->forward(ctx, box0->forward(ctx, x)));
        auto classes = cls2->forward(ctx, cls1->forward(ctx, cls0->forward(ctx, x)));
        return ggml_concat(ctx->ggml_ctx, boxes, classes, 2);
    }

    int output_channels() const {
        return reg_max_ * 4 + num_classes_;
    }
};

class YOLOv8Model : public GGMLBlock {
    YOLOv8Config config_;

    std::shared_ptr<YOLOC2f> make_c2f(int layer, int in_channels, bool shortcut) {
        return std::make_shared<YOLOC2f>(in_channels,
                                         config_.out_channels[layer],
                                         config_.hidden_channels.at(layer),
                                         config_.repeats.at(layer),
                                         shortcut);
    }

public:
    explicit YOLOv8Model(YOLOv8Config config)
        : config_(std::move(config)) {
        blocks["model.0"] = std::make_shared<YOLOConv>(3, config_.out_channels[0], 3, 2);
        blocks["model.1"] = std::make_shared<YOLOConv>(config_.out_channels[0], config_.out_channels[1], 3, 2);
        blocks["model.2"] = make_c2f(2, config_.out_channels[1], true);
        blocks["model.3"] = std::make_shared<YOLOConv>(config_.out_channels[2], config_.out_channels[3], 3, 2);
        blocks["model.4"] = make_c2f(4, config_.out_channels[3], true);
        blocks["model.5"] = std::make_shared<YOLOConv>(config_.out_channels[4], config_.out_channels[5], 3, 2);
        blocks["model.6"] = make_c2f(6, config_.out_channels[5], true);
        blocks["model.7"] = std::make_shared<YOLOConv>(config_.out_channels[6], config_.out_channels[7], 3, 2);
        blocks["model.8"] = make_c2f(8, config_.out_channels[7], true);
        blocks["model.9"] = std::make_shared<YOLOSPPF>(config_.out_channels[8], config_.out_channels[9]);

        blocks["model.12"] = make_c2f(12, config_.out_channels[9] + config_.out_channels[6], false);
        blocks["model.15"] = make_c2f(15, config_.out_channels[12] + config_.out_channels[4], false);
        blocks["model.16"] = std::make_shared<YOLOConv>(config_.out_channels[15], config_.out_channels[16], 3, 2);
        blocks["model.18"] = make_c2f(18, config_.out_channels[16] + config_.out_channels[12], false);
        blocks["model.19"] = std::make_shared<YOLOConv>(config_.out_channels[18], config_.out_channels[19], 3, 2);
        blocks["model.21"] = make_c2f(21, config_.out_channels[19] + config_.out_channels[9], false);
        blocks["model.22"] = std::make_shared<YOLODetect>(
            std::array<int, 3>{config_.out_channels[15], config_.out_channels[18], config_.out_channels[21]},
            config_.detect_box_channels,
            config_.detect_cls_channels,
            config_.reg_max,
            config_.num_classes);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto run = [&](int layer, ggml_tensor* input) {
            return std::dynamic_pointer_cast<UnaryBlock>(blocks["model." + std::to_string(layer)])->forward(ctx, input);
        };

        auto x0 = run(0, x);
        auto x1 = run(1, x0);
        auto x2 = run(2, x1);
        auto x3 = run(3, x2);
        auto x4 = run(4, x3);
        auto x5 = run(5, x4);
        auto x6 = run(6, x5);
        auto x7 = run(7, x6);
        auto x8 = run(8, x7);
        auto x9 = run(9, x8);

        auto x12 = run(12, ggml_concat(ctx->ggml_ctx, ggml_upscale(ctx->ggml_ctx, x9, 2, GGML_SCALE_MODE_NEAREST), x6, 2));
        auto x15 = run(15, ggml_concat(ctx->ggml_ctx, ggml_upscale(ctx->ggml_ctx, x12, 2, GGML_SCALE_MODE_NEAREST), x4, 2));
        auto x16 = run(16, x15);
        auto x18 = run(18, ggml_concat(ctx->ggml_ctx, x16, x12, 2));
        auto x19 = run(19, x18);
        auto x21 = run(21, ggml_concat(ctx->ggml_ctx, x19, x9, 2));

        auto detect = std::dynamic_pointer_cast<YOLODetect>(blocks["model.22"]);
        auto p3     = detect->forward_scale(ctx, x15, 0);
        auto p4     = detect->forward_scale(ctx, x18, 1);
        auto p5     = detect->forward_scale(ctx, x21, 2);
        p3          = ggml_reshape_2d(ctx->ggml_ctx, p3, p3->ne[0] * p3->ne[1], detect->output_channels());
        p4          = ggml_reshape_2d(ctx->ggml_ctx, p4, p4->ne[0] * p4->ne[1], detect->output_channels());
        p5          = ggml_reshape_2d(ctx->ggml_ctx, p5, p5->ne[0] * p5->ne[1], detect->output_channels());
        return ggml_concat(ctx->ggml_ctx, ggml_concat(ctx->ggml_ctx, p3, p4, 0), p5, 0);
    }
};

struct YOLOv8Runner : public GGMLRunner {
    YOLOv8Config config;
    std::unique_ptr<YOLOv8Model> model;

    YOLOv8Runner(ggml_backend_t backend,
                 const String2TensorStorage& tensor_storage_map,
                 std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          config(YOLOv8Config::detect_from_weights(tensor_storage_map)) {
        if (config.valid) {
            model = std::make_unique<YOLOv8Model>(config);
            model->init(params_ctx, tensor_storage_map, "");
        }
    }

    std::string get_desc() override {
        return "yolov8";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        if (model) {
            model->get_param_tensors(tensors);
        }
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& input) {
        if (!model) {
            return nullptr;
        }
        ggml_cgraph* graph = new_graph_custom(1 << 16);
        auto x             = make_input(input);
        auto runner_ctx    = get_context();
        auto output        = model->forward(&runner_ctx, x);
        ggml_build_forward_expand(graph, output);
        return graph;
    }

    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& input) {
        auto get_graph = [&]() { return build_graph(input); };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false));
    }
};

#endif  // __SD_MODEL_DETECTOR_YOLOV8_H__
