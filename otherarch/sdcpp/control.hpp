#ifndef __CONTROL_HPP__
#define __CONTROL_HPP__

#include "common_block.hpp"
#include "model.h"

#define CONTROL_NET_GRAPH_SIZE 1536

/*
    =================================== ControlNet ===================================
    Reference: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/cldm/cldm.py

*/
class ControlNetBlock : public GGMLBlock {
protected:
    SDVersion version = VERSION_SD1;
    // network hparams
    int in_channels                        = 4;
    int out_channels                       = 4;
    int hint_channels                      = 3;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;   // channels // num_heads
    int context_dim                        = 768;  // 1024 for VERSION_SD2, 2048 for VERSION_SDXL
    bool use_linear_projection             = false;

public:
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_SDXL

    ControlNetBlock(SDVersion version = VERSION_SD1)
        : version(version) {
        if (sd_version_is_sd2(version)) {
            context_dim       = 1024;
            num_head_channels = 64;
            num_heads         = -1;
        } else if (sd_version_is_sdxl(version)) {
            context_dim           = 2048;
            attention_resolutions = {4, 2};
            channel_mult          = {1, 2, 4};
            transformer_depth     = {1, 2, 10};
            num_head_channels     = 64;
            num_heads             = -1;
        } else if (version == VERSION_SVD) {
            in_channels       = 8;
            out_channels      = 4;
            context_dim       = 1024;
            adm_in_channels   = 768;
            num_head_channels = 64;
            num_heads         = -1;
        }

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        // time_embed_1 is nn.SiLU()
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        if (sd_version_is_sdxl(version) || version == VERSION_SVD) {
            blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
            // label_emb_1 is nn.SiLU()
            blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch              = model_channels;
        int input_block_idx = 0;
        int ds              = 1;

        auto get_resblock = [&](int64_t channels, int64_t emb_channels, int64_t out_channels) -> ResBlock* {
            return new ResBlock(channels, emb_channels, out_channels);
        };

        auto get_attention_layer = [&](int64_t in_channels,
                                       int64_t n_head,
                                       int64_t d_head,
                                       int64_t depth,
                                       int64_t context_dim) -> SpatialTransformer* {
            return new SpatialTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear_projection);
        };

        auto make_zero_conv = [&](int64_t channels) {
            return new Conv2d(channels, channels, {1, 1});
        };

        blocks["zero_convs.0.0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(model_channels));

        blocks["input_hint_block.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(hint_channels, 16, {3, 3}, {1, 1}, {1, 1}));
        // nn.SiLU()
        blocks["input_hint_block.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(16, 16, {3, 3}, {1, 1}, {1, 1}));
        // nn.SiLU()
        blocks["input_hint_block.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(16, 32, {3, 3}, {2, 2}, {1, 1}));
        // nn.SiLU()
        blocks["input_hint_block.6"] = std::shared_ptr<GGMLBlock>(new Conv2d(32, 32, {3, 3}, {1, 1}, {1, 1}));
        // nn.SiLU()
        blocks["input_hint_block.8"] = std::shared_ptr<GGMLBlock>(new Conv2d(32, 96, {3, 3}, {2, 2}, {1, 1}));
        // nn.SiLU()
        blocks["input_hint_block.10"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 96, {3, 3}, {1, 1}, {1, 1}));
        // nn.SiLU()
        blocks["input_hint_block.12"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 256, {3, 3}, {2, 2}, {1, 1}));
        // nn.SiLU()
        blocks["input_hint_block.14"] = std::shared_ptr<GGMLBlock>(new Conv2d(256, model_channels, {3, 3}, {1, 1}, {1, 1}));

        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                      n_head,
                                                                                      d_head,
                                                                                      transformer_depth[i],
                                                                                      context_dim));
                }
                blocks["zero_convs." + std::to_string(input_block_idx) + ".0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(ch));
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch));

                blocks["zero_convs." + std::to_string(input_block_idx) + ".0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(ch));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        blocks["middle_block.0"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
        blocks["middle_block.1"] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                  n_head,
                                                                                  d_head,
                                                                                  transformer_depth[transformer_depth.size() - 1],
                                                                                  context_dim));
        blocks["middle_block.2"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));

        // middle_block_out
        blocks["middle_block_out.0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(ch));
    }

    ggml_tensor* resblock_forward(std::string name,
                                  GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* emb) {
        auto block = std::dynamic_pointer_cast<ResBlock>(blocks[name]);
        return block->forward(ctx, x, emb);
    }

    ggml_tensor* attention_layer_forward(std::string name,
                                         GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* context) {
        auto block = std::dynamic_pointer_cast<SpatialTransformer>(blocks[name]);
        return block->forward(ctx, x, context);
    }

    ggml_tensor* input_hint_block_forward(GGMLRunnerContext* ctx,
                                          ggml_tensor* hint,
                                          ggml_tensor* emb,
                                          ggml_tensor* context) {
        int num_input_blocks = 15;
        auto h               = hint;
        for (int i = 0; i < num_input_blocks; i++) {
            if (i % 2 == 0) {
                auto block = std::dynamic_pointer_cast<Conv2d>(blocks["input_hint_block." + std::to_string(i)]);

                h = block->forward(ctx, h);
            } else {
                h = ggml_silu_inplace(ctx->ggml_ctx, h);
            }
        }
        return h;
    }

    std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,
                                      ggml_tensor* hint,
                                      ggml_tensor* guided_hint,
                                      ggml_tensor* timesteps,
                                      ggml_tensor* context,
                                      ggml_tensor* y = nullptr) {
        // x: [N, in_channels, h, w] or [N, in_channels/2, h, w]
        // timesteps: [N,]
        // context: [N, max_position, hidden_size] or [1, max_position, hidden_size]. for example, [N, 77, 768]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        if (context != nullptr) {
            if (context->ne[2] != x->ne[3]) {
                context = ggml_repeat(ctx->ggml_ctx, context, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
            }
        }

        if (y != nullptr) {
            if (y->ne[1] != x->ne[3]) {
                y = ggml_repeat(ctx->ggml_ctx, y, ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
            }
        }

        auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);
        auto zero_convs_0     = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs.0.0"]);

        auto middle_block_out = std::dynamic_pointer_cast<Conv2d>(blocks["middle_block_out.0"]);

        auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);  // [N, time_embed_dim]

        // SDXL/SVD
        if (y != nullptr) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, y);
            label_emb      = ggml_silu_inplace(ctx->ggml_ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);  // [N, time_embed_dim]

            emb = ggml_add(ctx->ggml_ctx, emb, label_emb);  // [N, time_embed_dim]
        }

        std::vector<ggml_tensor*> outs;

        if (guided_hint == nullptr) {
            guided_hint = input_hint_block_forward(ctx, hint, emb, context);
        }
        outs.push_back(guided_hint);

        // input_blocks

        // input block 0
        auto h = input_blocks_0_0->forward(ctx, x);
        h      = ggml_add(ctx->ggml_ctx, h, guided_hint);
        outs.push_back(zero_convs_0->forward(ctx, h));

        // input block 1-11
        size_t len_mults    = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                h                = resblock_forward(name, ctx, h, emb);  // [N, mult*model_channels, h, w]
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    h                = attention_layer_forward(name, ctx, h, context);  // [N, mult*model_channels, h, w]
                }

                auto zero_conv = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs." + std::to_string(input_block_idx) + ".0"]);

                outs.push_back(zero_conv->forward(ctx, h));
            }
            if (i != len_mults - 1) {
                ds *= 2;
                input_block_idx += 1;

                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                auto block       = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = block->forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]

                auto zero_conv = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs." + std::to_string(input_block_idx) + ".0"]);

                outs.push_back(zero_conv->forward(ctx, h));
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = resblock_forward("middle_block.0", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]
        h = attention_layer_forward("middle_block.1", ctx, h, context);  // [N, 4*model_channels, h/8, w/8]
        h = resblock_forward("middle_block.2", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]

        // out
        outs.push_back(middle_block_out->forward(ctx, h));
        return outs;
    }
};

struct ControlNet : public GGMLRunner {
    SDVersion version = VERSION_SD1;
    ControlNetBlock control_net;

    ggml_backend_buffer_t control_buffer = nullptr;
    ggml_context* control_ctx            = nullptr;
    std::vector<ggml_tensor*> control_outputs_ggml;
    ggml_tensor* guided_hint_output_ggml = nullptr;
    std::vector<sd::Tensor<float>> controls;
    sd::Tensor<float> guided_hint;
    bool guided_hint_cached = false;

    ControlNet(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {},
               SDVersion version                              = VERSION_SD1)
        : GGMLRunner(backend, offload_params_to_cpu), control_net(version) {
        control_net.init(params_ctx, tensor_storage_map, "");
    }

    ~ControlNet() override {
        free_control_ctx();
    }

    void alloc_control_ctx(std::vector<ggml_tensor*> outs) {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(outs.size() * ggml_tensor_overhead()) + 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc   = true;
        control_ctx       = ggml_init(params);

        control_outputs_ggml.resize(outs.size() - 1);

        size_t control_buffer_size = 0;

        guided_hint_output_ggml = ggml_dup_tensor(control_ctx, outs[0]);
        control_buffer_size += ggml_nbytes(guided_hint_output_ggml);

        for (int i = 0; i < outs.size() - 1; i++) {
            control_outputs_ggml[i] = ggml_dup_tensor(control_ctx, outs[i + 1]);
            control_buffer_size += ggml_nbytes(control_outputs_ggml[i]);
        }

        control_buffer = ggml_backend_alloc_ctx_tensors(control_ctx, runtime_backend);

        LOG_DEBUG("control buffer size %.2fMB", control_buffer_size * 1.f / 1024.f / 1024.f);
    }

    void free_control_ctx() {
        if (control_buffer != nullptr) {
            ggml_backend_buffer_free(control_buffer);
            control_buffer = nullptr;
        }
        if (control_ctx != nullptr) {
            ggml_free(control_ctx);
            control_ctx = nullptr;
        }
        guided_hint_output_ggml = nullptr;
        guided_hint_cached      = false;
        guided_hint             = {};
        control_outputs_ggml.clear();
        controls.clear();
    }

    std::string get_desc() override {
        return "control_net";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        control_net.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                             const sd::Tensor<float>& hint_tensor,
                             const sd::Tensor<float>& timesteps_tensor,
                             const sd::Tensor<float>& context_tensor = {},
                             const sd::Tensor<float>& y_tensor       = {}) {
        ggml_cgraph* gf = new_graph_custom(CONTROL_NET_GRAPH_SIZE);

        ggml_tensor* x         = make_input(x_tensor);
        ggml_tensor* hint      = nullptr;
        ggml_tensor* timesteps = make_input(timesteps_tensor);
        ggml_tensor* context   = make_optional_input(context_tensor);
        ggml_tensor* y         = make_optional_input(y_tensor);

        ggml_tensor* guided_hint_input = nullptr;
        if (guided_hint_cached && !guided_hint.empty()) {
            guided_hint_input = make_input(guided_hint);
            hint              = nullptr;
        } else {
            hint = make_input(hint_tensor);
        }

        auto runner_ctx = get_context();

        auto outs = control_net.forward(&runner_ctx,
                                        x,
                                        hint,
                                        guided_hint_input,
                                        timesteps,
                                        context,
                                        y);

        if (control_ctx == nullptr) {
            alloc_control_ctx(outs);
        }

        ggml_build_forward_expand(gf, ggml_cpy(compute_ctx, outs[0], guided_hint_output_ggml));
        for (int i = 0; i < outs.size() - 1; i++) {
            ggml_build_forward_expand(gf, ggml_cpy(compute_ctx, outs[i + 1], control_outputs_ggml[i]));
        }

        return gf;
    }

    std::optional<std::vector<sd::Tensor<float>>> compute(int n_threads,
                                                          const sd::Tensor<float>& x,
                                                          const sd::Tensor<float>& hint,
                                                          const sd::Tensor<float>& timesteps,
                                                          const sd::Tensor<float>& context = {},
                                                          const sd::Tensor<float>& y       = {}) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 77, 768]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x, hint, timesteps, context, y);
        };

        auto compute_result = GGMLRunner::compute<float>(get_graph, n_threads, false);
        if (!compute_result.has_value()) {
            return std::nullopt;
        }

        if (guided_hint_output_ggml != nullptr) {
            guided_hint = restore_trailing_singleton_dims(sd::make_sd_tensor_from_ggml<float>(guided_hint_output_ggml),
                                                          4);
        }
        controls.clear();
        controls.reserve(control_outputs_ggml.size());
        for (ggml_tensor* control : control_outputs_ggml) {
            auto control_host = restore_trailing_singleton_dims(sd::make_sd_tensor_from_ggml<float>(control), 4);
            GGML_ASSERT(!control_host.empty());
            controls.push_back(std::move(control_host));
        }
        guided_hint_cached = true;
        return controls;
    }

    bool load_from_file(const std::string& file_path, int n_threads) {
        LOG_INFO("loading control net from '%s'", file_path.c_str());
        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        control_net.get_param_tensors(tensors);
        std::set<std::string> ignore_tensors;

        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(file_path)) {
            LOG_ERROR("init control net model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads);

        if (!success) {
            LOG_ERROR("load control net tensors from model loader failed");
            return false;
        }

        LOG_INFO("control net model loaded");
        return success;
    }
};

#endif  // __CONTROL_HPP__
