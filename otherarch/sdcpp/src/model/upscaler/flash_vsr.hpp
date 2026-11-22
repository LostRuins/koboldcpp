#ifndef __SD_MODEL_UPSCALER_FLASH_VSR_HPP__
#define __SD_MODEL_UPSCALER_FLASH_VSR_HPP__

#include "model/common/vsr.hpp"

namespace RelayVSR {

    struct FlashVSRConfig {
        int dim            = 512;
        int depth          = 10;
        int num_heads      = 8;
        int num_kv_groups  = 2;
        int hidden_dim     = 1408;
        int head_dim       = 1024;
        int upscale_factor = 4;

        static FlashVSRConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix, int scale) {
            FlashVSRConfig config;
            config.upscale_factor = scale;
            auto weight           = [&](const std::string& name) -> const TensorStorage& { return weights.at(prefix + "." + name); };
            config.dim            = int(weight("backbone.blocks.0.attention.query.weight").ne[1]);
            config.num_heads      = config.dim / 64;
            config.num_kv_groups  = int(weight("backbone.blocks.0.attention.key.weight").ne[1]) / 64;
            config.hidden_dim     = int(weight("backbone.blocks.0.feed_forward.down.weight").logical_ne0());
            config.head_dim       = int(weight("spatial_head.0.weight").ne[1]);
            config.depth          = 0;
            while (weights.count(prefix + ".backbone.blocks." + std::to_string(config.depth) + ".attention.query.weight"))
                ++config.depth;
            if ((scale != 2 && scale != 4) || config.dim != 512 || config.depth != 10 || config.num_kv_groups != 2 ||
                weight("condition_encoder.projection.weight").ne[2] != 48 || weight("residual_projection.weight").ne[1] != 768)
                throw std::runtime_error("Unsupported RelayVSR Flash architecture");
            LOG_VERBOSE("flashvsr: dim = %d, depth = %d, hidden_dim = %d, scale = %d", config.dim, config.depth, config.hidden_dim, scale);
            return config;
        }
    };

    class LRFrameEncoder : public GGMLBlock {
    public:
        explicit LRFrameEncoder(const FlashVSRConfig& config) {
            int channels[] = {3, 64, 128, 128};
            for (int i = 0; i < 3; ++i) {
                int stride                                = i < 2 ? 2 : 4 / config.upscale_factor;
                blocks["layers." + std::to_string(i * 2)] = std::make_shared<Conv2d>(channels[i], channels[i + 1], std::pair<int, int>{3, 3}, std::pair<int, int>{stride, stride}, std::pair<int, int>{1, 1});
            }
            blocks["layers.6"] = std::make_shared<Conv2d>(128, config.dim, std::pair<int, int>{1, 1});
            blocks["norm"]     = std::make_shared<RMSNorm>(config.dim, 1e-5f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto layers_6 = std::dynamic_pointer_cast<Conv2d>(blocks["layers.6"]);
            auto norm     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            for (int i = 0; i < 3; ++i)
                x = ggml_silu(ctx->ggml_ctx, std::dynamic_pointer_cast<Conv2d>(blocks["layers." + std::to_string(i * 2)])->forward(ctx, x));
            x = layers_6->forward(ctx, x);
            return norm->forward(ctx, VSR::tokens(ctx->ggml_ctx, x));
        }
    };

    class ConditionEncoder : public GGMLBlock {
    public:
        explicit ConditionEncoder(const FlashVSRConfig& config) {
            blocks["projection"] = std::make_shared<Conv2d>(48, config.dim, std::pair<int, int>{1, 1});
            blocks["norm"]       = std::make_shared<RMSNorm>(config.dim, 1e-5f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto projection = std::dynamic_pointer_cast<Conv2d>(blocks["projection"]);
            auto norm       = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            return norm->forward(ctx, VSR::tokens(ctx->ggml_ctx, projection->forward(ctx, x)));
        }
    };

    class ConditionedGroupedQueryAttention : public GGMLBlock {
        FlashVSRConfig config_;

    public:
        explicit ConditionedGroupedQueryAttention(const FlashVSRConfig& config)
            : config_(config) {
            blocks["query"]  = std::make_shared<Linear>(config.dim, config.dim);
            blocks["output"] = std::make_shared<Linear>(config.dim, config.dim);
            for (const auto& name : {"key", "value", "condition_key", "condition_value"})
                blocks[name] = std::make_shared<Linear>(config.dim, 64 * config.num_kv_groups);
            blocks["key_norm"]   = std::make_shared<RMSNorm>(64, 1e-5f);
            blocks["value_norm"] = std::make_shared<RMSNorm>(64, 1e-5f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* qpe, ggml_tensor* kpe, const VSR::LocalAttention& local, const std::string& name, int condition_slot, int start_slot, int end_slot, bool history) {
            auto query           = std::dynamic_pointer_cast<Linear>(blocks["query"]);
            auto output          = std::dynamic_pointer_cast<Linear>(blocks["output"]);
            auto key_norm        = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);
            auto value_norm      = std::dynamic_pointer_cast<RMSNorm>(blocks["value_norm"]);
            auto c               = ctx->ggml_ctx;
            const bool condition = condition_slot >= 0;
            auto key             = std::dynamic_pointer_cast<Linear>(blocks[condition ? "condition_key" : "key"]);
            auto value           = std::dynamic_pointer_cast<Linear>(blocks[condition ? "condition_value" : "value"]);
            auto q               = query->forward(ctx, x);
            auto k               = key->forward(ctx, x);
            auto v               = value->forward(ctx, x);
            const int64_t count  = x->ne[1];
            k                    = key_norm->forward(ctx, ggml_reshape_3d(c, k, 64, config_.num_kv_groups, count));
            v                    = value_norm->forward(ctx, ggml_reshape_3d(c, v, 64, config_.num_kv_groups, count));
            k                    = ggml_reshape_2d(c, k, 64 * config_.num_kv_groups, count);
            v                    = ggml_reshape_2d(c, v, 64 * config_.num_kv_groups, count);
            auto cache_prefix    = name + (condition ? ".condition." + std::to_string(condition_slot) : ".video");
            auto past_k          = history ? ctx->get_cache_tensor(name + ".video.key") : nullptr;
            auto past_v          = history ? ctx->get_cache_tensor(name + ".video.value") : nullptr;
            ctx->cache_tensor(cache_prefix + ".key", k);
            ctx->cache_tensor(cache_prefix + ".value", v);
            if (!condition) {
                auto join = [&](ggml_tensor* current, ggml_tensor* past, const char* suffix) {
                    auto start = ctx->get_cache_tensor(name + ".condition." + std::to_string(start_slot) + suffix);
                    auto end   = ctx->get_cache_tensor(name + ".condition." + std::to_string(end_slot) + suffix);
                    if (!start || !end)
                        throw std::runtime_error("RelayVSR Flash endpoint cache is missing");
                    auto value = ggml_concat(c, start, end, 1);
                    if (past)
                        value = ggml_concat(c, value, past, 1);
                    return ggml_concat(c, value, current, 1);
                };
                k = join(k, past_k, ".key");
                v = join(v, past_v, ".value");
            }
            q = VSR::apply_rope(ctx, q, qpe, config_.num_heads);
            k = VSR::apply_rope(ctx, k, kpe, config_.num_kv_groups);
            return output->forward(ctx, local.forward(ctx, q, k, v, config_.num_heads));
        }
    };

    class ConditionedTransformerBlock : public GGMLBlock {
    public:
        explicit ConditionedTransformerBlock(const FlashVSRConfig& config) {
            blocks["attention_norm"]            = std::make_shared<RMSNorm>(config.dim, 1e-5f);
            blocks["attention"]                 = std::make_shared<ConditionedGroupedQueryAttention>(config);
            blocks["feed_forward_norm"]         = std::make_shared<RMSNorm>(config.dim, 1e-5f);
            blocks["feed_forward.gate_up_proj"] = std::make_shared<Linear>(config.dim, config.hidden_dim * 2, false);
            blocks["feed_forward.down"]         = std::make_shared<Linear>(config.hidden_dim, config.dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* qpe, ggml_tensor* kpe, const VSR::LocalAttention& local, const std::string& name, int condition_slot, int start_slot, int end_slot, bool history) {
            auto attention_norm            = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm"]);
            auto feed_forward_norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["feed_forward_norm"]);
            auto feed_forward_gate_up_proj = std::dynamic_pointer_cast<Linear>(blocks["feed_forward.gate_up_proj"]);
            auto feed_forward_down         = std::dynamic_pointer_cast<Linear>(blocks["feed_forward.down"]);
            auto c                         = ctx->ggml_ctx;
            auto attention                 = std::dynamic_pointer_cast<ConditionedGroupedQueryAttention>(blocks["attention"]);
            auto y                         = attention->forward(ctx, attention_norm->forward(ctx, x), qpe, kpe, local, name, condition_slot, start_slot, end_slot, history);
            x                              = ggml_add(c, x, y);
            y                              = feed_forward_gate_up_proj->forward(ctx, feed_forward_norm->forward(ctx, x));
            auto parts                     = ggml_ext_chunk(ctx, y, 2, 0);
            y                              = ggml_mul(c, ggml_silu(c, parts[0]), parts[1]);
            return ggml_add(c, x, feed_forward_down->forward(ctx, y));
        }
    };

    class FlashVSRConditionNet : public GGMLBlock {
        FlashVSRConfig config_;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
            params["video_type_embedding"]     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config_.dim, 1, 1);
            params["condition_type_embedding"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config_.dim, 1, 1);
        }

    public:
        explicit FlashVSRConditionNet(const FlashVSRConfig& config)
            : config_(config) {
            blocks["lr_encoder"]        = std::make_shared<LRFrameEncoder>(config);
            blocks["condition_encoder"] = std::make_shared<ConditionEncoder>(config);
            for (int i = 0; i < config.depth; ++i)
                blocks["backbone.blocks." + std::to_string(i)] = std::make_shared<ConditionedTransformerBlock>(config);
            blocks["spatial_head.0"]      = std::make_shared<Linear>(config.dim, config.head_dim);
            blocks["residual_projection"] = std::make_shared<Linear>(config.head_dim, 768);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* input, ggml_tensor* qpe, ggml_tensor* kpe, const VSR::LocalAttention& local, int condition_slot, int start_slot, int end_slot, bool history) {
            auto spatial_head_0      = std::dynamic_pointer_cast<Linear>(blocks["spatial_head.0"]);
            auto residual_projection = std::dynamic_pointer_cast<Linear>(blocks["residual_projection"]);
            auto c                   = ctx->ggml_ctx;
            const bool condition     = condition_slot >= 0;
            ggml_tensor* x;
            if (condition)
                x = std::dynamic_pointer_cast<ConditionEncoder>(blocks["condition_encoder"])->forward(ctx, input);
            else
                x = std::dynamic_pointer_cast<LRFrameEncoder>(blocks["lr_encoder"])->forward(ctx, input);
            x = ggml_add(c, x, params[condition ? "condition_type_embedding" : "video_type_embedding"]);
            for (int i = 0; i < config_.depth; ++i) {
                auto name = "backbone.blocks." + std::to_string(i);
                x         = std::dynamic_pointer_cast<ConditionedTransformerBlock>(blocks[name])->forward(ctx, x, qpe, kpe, local, name, condition_slot, start_slot, end_slot, history);
                sd::ggml_graph_cut::mark_graph_cut(x, "flashvsr." + name, "hidden");
            }
            if (condition)
                return x;
            x = residual_projection->forward(ctx, ggml_silu(c, spatial_head_0->forward(ctx, x)));
            return VSR::pixel_shuffle(c, x, local.width, local.height, 16);
        }
    };

    struct FlashVSRRunner : GGMLRunner {
        FlashVSRConfig config;
        FlashVSRConditionNet model;
        VSR::GraphInputs inputs;
        std::map<int, std::unique_ptr<VSR::LocalAttention>> attention;
        bool history = false;

        FlashVSRRunner(ggml_backend_t backend, const String2TensorStorage& weights, int scale, std::shared_ptr<DeviceResidencyManager> manager)
            : GGMLRunner(backend, manager), config(FlashVSRConfig::detect_from_weights(weights, "flash", scale)), model(config) {
            model.init(params_ctx, weights, "flash");
        }
        std::string get_desc() override { return "RelayVSR Flash"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) { model.get_param_tensors(tensors, "flash"); }
        void reset() {
            runner_end();
            free_cache_ctx_and_buffer();
            attention.clear();
            inputs.clear();
            history = false;
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& input, int condition_slot, int start_slot = 0, int end_slot = 1) {
            const bool condition = condition_slot >= 0;
            const int stride     = condition ? 1 : 16 / config.upscale_factor;
            const int width = int(input.shape()[0]) / stride, height = int(input.shape()[1]) / stride;
            const int planes = condition ? 1 : history ? 4
                                                       : 3;
            if (!attention.count(planes)) {
                std::vector<std::pair<int, int>> windows(planes, {24, 24});
                windows[0] = {48, 48};
                if (!condition)
                    windows[1] = {48, 48};
                attention[planes] = std::make_unique<VSR::LocalAttention>(width, height, windows, false);
            }
            auto graph = [&]() {
                inputs.begin_graph();
                attention[planes]->begin_graph();
                auto gf                = new_graph_custom(MAX_GRAPH_SIZE);
                auto ctx               = get_context(gf);
                auto qpe               = inputs.rope(&ctx, width, height, {0}, {16, 24, 24});
                std::vector<int> times = condition ? std::vector<int>{0} : history ? std::vector<int>{0, 0, -1, 0}
                                                                                   : std::vector<int>{0, 0, 0};
                auto kpe               = inputs.rope(&ctx, width, height, times, {16, 24, 24});
                auto out               = model.forward(&ctx, make_input(input), qpe, kpe, *attention[planes], condition_slot, start_slot, end_slot, history && !condition);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), condition ? 2 : 4);
            if (!condition && !result.empty())
                history = true;
            return result;
        }
    };

}  // namespace RelayVSR

#endif  // __SD_MODEL_UPSCALER_FLASH_VSR_HPP__
