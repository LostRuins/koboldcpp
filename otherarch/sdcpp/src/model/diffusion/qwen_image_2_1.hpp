#ifndef __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
#define __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__

#include "model/diffusion/qwen_image.hpp"

namespace Qwen {

    struct QwenImage21Config {
        int64_t in_channels       = 64;
        int64_t out_channels      = 64;
        int64_t hidden_size       = 4096;
        int64_t context_dim       = 4096;
        int64_t head_dim          = 128;
        int64_t intermediate_size = 12288;
        int num_layers            = 32;
        bool fused_mlp            = false;
        std::vector<int> axes_dim = {16, 56, 56};

        static QwenImage21Config detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            QwenImage21Config config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = weights.find(prefix + "." + suffix);
                return it == weights.end() ? nullptr : &it->second;
            };
            if (auto w = find("img_in.weight")) {
                config.in_channels = w->ne[0];
                config.hidden_size = w->ne[1];
            }
            if (auto w = find("proj_out.weight")) {
                config.out_channels = w->ne[1];
            }
            if (auto w = find("txt_in.in_layer.weight")) {
                config.context_dim = w->ne[0];
            }
            if (auto w = find("transformer_blocks.0.attn.norm_q.weight")) {
                config.head_dim = w->ne[0];
            }
            if (auto w = find("transformer_blocks.0.img_mlp.gate_up.weight")) {
                config.intermediate_size = w->ne[1] / 2;
                config.fused_mlp         = true;
            } else if (auto w = find("transformer_blocks.0.img_mlp.proj.weight")) {
                config.intermediate_size = w->ne[1];
            }
            int layers                     = 0;
            const std::string block_prefix = prefix + ".transformer_blocks.";
            for (const auto& [name, _] : weights) {
                if (starts_with(name, block_prefix)) {
                    layers = std::max(layers, atoi(name.substr(block_prefix.size()).c_str()) + 1);
                }
            }
            if (layers > 0) {
                config.num_layers = layers;
                LOG_VERBOSE("qwen_image_2_1: layers = %d, hidden_size = %" PRId64 ", context_dim = %" PRId64,
                            layers, config.hidden_size, config.context_dim);
            }
            return config;
        }
    };

    struct QwenImage21Segment {
        int64_t start;
        int64_t end;
        int64_t context_start;
        int image_index;
    };

    struct QwenImage21Layout {
        std::vector<QwenImage21Segment> segments;
        std::vector<std::vector<float>> positions;
        int64_t prefix_length = 0;

        static QwenImage21Layout build(int64_t text_length,
                                       const sd::Tensor<int32_t>& image_slots,
                                       const std::vector<std::pair<int64_t, int64_t>>& image_shapes) {
            if (image_shapes.empty() || (!image_slots.empty() && image_slots.numel() != text_length)) {
                throw std::runtime_error("Qwen Image 2.1: invalid image token layout");
            }
            QwenImage21Layout layout;
            int64_t position  = 0;
            int next_image    = 0;
            auto append_image = [&](int index, int64_t context_start) {
                auto [height, width] = image_shapes[index];
                int64_t start        = static_cast<int64_t>(layout.positions.size());
                layout.segments.push_back({start, start + height * width, context_start, index});
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        layout.positions.push_back({static_cast<float>(position),
                                                    static_cast<float>(h - (height - height / 2)),
                                                    static_cast<float>(w - (width - width / 2))});
                    }
                }
                position += std::max(height, width);
            };
            for (int64_t i = 0; i < text_length;) {
                int tag       = image_slots.empty() ? 0 : image_slots[i];
                int64_t begin = i++;
                while (i < text_length && (image_slots.empty() ? 0 : image_slots[i]) == tag) {
                    ++i;
                }
                if (tag != 0) {
                    if (tag != next_image + 1 || next_image + 1 >= static_cast<int>(image_shapes.size()) ||
                        (i - begin) * 4 != image_shapes[next_image].first * image_shapes[next_image].second) {
                        throw std::runtime_error("Qwen Image 2.1: vision slots and reference latents must have matching sizes");
                    }
                    append_image(next_image++, begin);
                } else {
                    int64_t start = static_cast<int64_t>(layout.positions.size());
                    layout.segments.push_back({start, start + i - begin, begin, -1});
                    for (int64_t j = begin; j < i; ++j, ++position) {
                        float p = static_cast<float>(position);
                        layout.positions.push_back({p, p, p});
                    }
                }
            }
            if (next_image + 1 != static_cast<int>(image_shapes.size())) {
                throw std::runtime_error("Qwen Image 2.1: missing reference image slots");
            }
            layout.prefix_length = static_cast<int64_t>(layout.positions.size());
            append_image(next_image, text_length);
            return layout;
        }
    };

    class QwenImage21ZeroCenterRMSNorm : public RMSNorm {
    public:
        using RMSNorm::RMSNorm;

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto weight = params["weight"];
            if (ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, weight, prefix + "weight");
            }
            weight = ggml_scale_bias(ctx->ggml_ctx, weight, 1.f, 1.f);
            return ggml_mul(ctx->ggml_ctx, ggml_rms_norm(ctx->ggml_ctx, x, eps), weight);
        }
    };

    class QwenImage21TextProjection : public GGMLBlock {
    public:
        QwenImage21TextProjection(const QwenImage21Config& config) {
            blocks["text_norm"] = std::make_shared<QwenImage21ZeroCenterRMSNorm>(config.context_dim, 1e-6f);
            blocks["in_layer"]  = std::make_shared<Linear>(config.context_dim, config.hidden_size, false);
            blocks["out_layer"] = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<QwenImage21ZeroCenterRMSNorm>(blocks["text_norm"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Linear>(blocks["in_layer"])->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            return std::dynamic_pointer_cast<Linear>(blocks["out_layer"])->forward(ctx, x);
        }
    };

    class QwenImage21Attention : public QwenImageAttention {
    public:
        QwenImage21Attention(const QwenImage21Config& config)
            : QwenImageAttention(config.hidden_size, config.head_dim, config.hidden_size / config.head_dim, 0, 0, false, false) {
            for (const auto* name : {"add_q_proj", "add_k_proj", "add_v_proj", "norm_added_q", "norm_added_k", "to_add_out"}) {
                blocks.erase(name);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, const std::vector<QwenImage21Segment>& segments, const std::vector<ggml_tensor*>& masks) {
            int64_t heads = x->ne[0] / dim_head;
            auto project  = [&](const char* name) {
                auto h = std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, x);
                return ggml_reshape_4d(ctx->ggml_ctx, h, dim_head, heads, x->ne[1], x->ne[2]);
            };
            auto q              = project("to_q");
            auto k              = project("to_k");
            auto v              = project("to_v");
            q                   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"])->forward(ctx, q);
            k                   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"])->forward(ctx, k);
            q                   = Rope::apply_rope(ctx->ggml_ctx, q, pe);
            k                   = Rope::apply_rope(ctx->ggml_ctx, k, pe);
            ggml_tensor* result = nullptr;
            for (size_t i = 0; i < segments.size(); ++i) {
                const auto& segment = segments[i];
                auto sq             = ggml_ext_slice(ctx->ggml_ctx, q, 1, segment.start, segment.end);
                auto sk             = ggml_ext_slice(ctx->ggml_ctx, k, 1, 0, segment.end);
                auto sv             = ggml_ext_slice(ctx->ggml_ctx, v, 2, 0, segment.end);
                auto out            = ggml_ext_attention_ext(ctx, sq, sk, sv, heads, masks[i], true, ctx->flash_attn_enabled);
                result              = result == nullptr ? out : ggml_concat(ctx->ggml_ctx, result, out, 1);
            }
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            if (sd_backend_is(ctx->backend, "Vulkan") || sd_backend_is(ctx->backend, "ROCm")) {
                to_out->set_force_prec_f32(true);
            }
            return to_out->forward(ctx, result);
        }
    };

    class QwenImage21TransformerBlock : public GGMLBlock {
    public:
        QwenImage21TransformerBlock(const QwenImage21Config& config) {
            blocks["img_norm1"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["img_norm2"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["attn"]      = std::make_shared<QwenImage21Attention>(config);
            if (config.fused_mlp) {
                blocks["img_mlp.gate_up"] = std::make_shared<Linear>(config.hidden_size, 2 * config.intermediate_size, false);
            } else {
                blocks["img_mlp.proj"]       = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
                blocks["img_mlp.gate_layer"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
            }
            blocks["img_mlp.out"] = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, false);
        }

        static ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* x, ggml_tensor* params, int64_t prefix_length, bool gate = false) {
            auto rows  = ggml_ext_chunk(ctx, params, 2, 1);
            auto apply = [&](ggml_tensor* part, ggml_tensor* row) {
                row = gate ? ggml_tanh(ctx, row) : ggml_scale_bias(ctx, row, 1.f, 1.f);
                return ggml_mul(ctx, part, row);
            };
            auto target = apply(ggml_ext_slice(ctx, x, 1, prefix_length, x->ne[1]), rows[0]);
            if (prefix_length == 0) {
                return target;
            }
            auto prefix = apply(ggml_ext_slice(ctx, x, 1, 0, prefix_length), rows[1]);
            return ggml_concat(ctx, prefix, target, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const std::vector<ggml_tensor*>& modulation, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks) {
            auto h = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"])->forward(ctx, x);
            h      = modulate(ctx->ggml_ctx, h, modulation[0], layout.prefix_length);
            h      = std::dynamic_pointer_cast<QwenImage21Attention>(blocks["attn"])->forward(ctx, h, pe, layout.segments, masks);
            x      = ggml_add(ctx->ggml_ctx, x, modulate(ctx->ggml_ctx, h, modulation[1], layout.prefix_length, true));
            h      = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"])->forward(ctx, x);
            h      = modulate(ctx->ggml_ctx, h, modulation[2], layout.prefix_length);
            ggml_tensor* gate;
            auto fused = blocks.find("img_mlp.gate_up");
            if (fused != blocks.end()) {
                auto gate_up = std::dynamic_pointer_cast<Linear>(fused->second)->forward(ctx, h);
                auto parts   = ggml_ext_chunk(ctx->ggml_ctx, gate_up, 2, 0);
                gate         = parts[0];
                h            = parts[1];
            } else {
                gate = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.gate_layer"])->forward(ctx, h);
                h    = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.proj"])->forward(ctx, h);
            }
            h = ggml_mul(ctx->ggml_ctx, h, ggml_silu(ctx->ggml_ctx, gate));
            h = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.out"])->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, modulate(ctx->ggml_ctx, h, modulation[3], layout.prefix_length, true));
        }
    };

    class QwenImage21Model : public GGMLBlock {
        QwenImage21Config config;

    public:
        QwenImage21Model(const QwenImage21Config& config)
            : config(config) {
            blocks["time_text_embed.timestep_embedder"] = std::make_shared<TimestepEmbedding>(256, config.hidden_size, 0, 0, false);
            blocks["txt_in"]                            = std::make_shared<QwenImage21TextProjection>(config);
            blocks["img_in"]                            = std::make_shared<Linear>(config.in_channels, config.hidden_size, false);
            blocks["modulation.1"]                      = std::make_shared<Linear>(config.hidden_size, 4 * config.hidden_size, false);
            blocks["norm_out.linear"]                   = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
            blocks["norm_out.norm"]                     = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["proj_out"]                          = std::make_shared<Linear>(config.hidden_size, config.out_channels, false);
            for (int i = 0; i < config.num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<QwenImage21TransformerBlock>(config);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, const std::vector<ggml_tensor*>& refs, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks) {
            auto time = ggml_concat(ctx->ggml_ctx, timestep, ggml_ext_zeros_like(ctx->ggml_ctx, timestep), 0);
            // Runtime flow timesteps already use the [0, 1000] scale.
            time               = ggml_ext_timestep_embedding(ctx->ggml_ctx, time, 256, 10000, 1.f);
            time               = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_text_embed.timestep_embedder"])->forward(ctx, time);
            time               = ggml_silu(ctx->ggml_ctx, time);
            auto modulation    = std::dynamic_pointer_cast<Linear>(blocks["modulation.1"])->forward(ctx, time);
            auto mod           = ggml_ext_chunk(ctx->ggml_ctx, modulation, 4, 0);
            auto text          = std::dynamic_pointer_cast<QwenImage21TextProjection>(blocks["txt_in"])->forward(ctx, context);
            auto img_in        = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            ggml_tensor* joint = nullptr;
            for (const auto& segment : layout.segments) {
                ggml_tensor* h;
                if (segment.image_index < 0) {
                    h = ggml_ext_slice(ctx->ggml_ctx, text, 1, segment.context_start,
                                       segment.context_start + segment.end - segment.start);
                } else {
                    auto image = segment.image_index == static_cast<int>(refs.size()) ? x : refs[segment.image_index];
                    h          = img_in->forward(ctx, DiT::patchify(ctx->ggml_ctx, image, 1, 1));
                }
                joint = joint == nullptr ? h : ggml_concat(ctx->ggml_ctx, joint, h, 1);
            }
            sd::ggml_graph_cut::mark_graph_cut(joint, "qwen_image_2_1.prelude", "joint");
            for (int i = 0; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<QwenImage21TransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                joint      = block->forward(ctx, joint, mod, pe, layout, masks);
                sd::ggml_graph_cut::mark_graph_cut(joint, "qwen_image_2_1.transformer_blocks." + std::to_string(i), "joint");
            }
            joint      = ggml_ext_slice(ctx->ggml_ctx, joint, 1, layout.prefix_length, joint->ne[1]);
            auto scale = std::dynamic_pointer_cast<Linear>(blocks["norm_out.linear"])->forward(ctx, ggml_ext_chunk(ctx->ggml_ctx, time, 2, 1)[0]);
            joint      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out.norm"])->forward(ctx, joint);
            joint      = ggml_mul(ctx->ggml_ctx, joint, ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f));
            joint      = std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, joint);
            return DiT::unpatchify_and_crop(ctx->ggml_ctx, joint, x->ne[1], x->ne[0], 1, 1);
        }
    };

    struct QwenImage21Runner : public DiffusionModelRunner {
        QwenImage21Config config;
        QwenImage21Model model;
        std::vector<float> pe_data;
        std::vector<sd::Tensor<float>> mask_data;

        QwenImage21Runner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(QwenImage21Config::detect_from_weights(weights, prefix)),
              model(config) {
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "qwen_image_2_1"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            const auto& x       = tensor_or_empty(inputs.x);
            const auto& context = tensor_or_empty(inputs.context);
            if (x.empty() || context.empty() || context.dim() < 2 || context.shape()[0] != config.context_dim ||
                tensor_or_empty(inputs.timesteps).numel() != 1 ||
                x.dim() != 4 || x.shape()[3] != 1 || x.shape()[2] != config.in_channels) {
                LOG_ERROR("Qwen Image 2.1 requires an image latent and text conditioning with batch size 1");
                return {};
            }
            static const std::vector<sd::Tensor<float>> empty_refs;
            const auto& refs = inputs.ref_latents && inputs.ref_image_params.pass_to_dit ? *inputs.ref_latents : empty_refs;
            std::vector<std::pair<int64_t, int64_t>> shapes;
            for (const auto& ref : refs) {
                if (ref.dim() != 4 || ref.shape()[2] != config.in_channels || ref.shape()[3] != 1) {
                    LOG_ERROR("Qwen Image 2.1: invalid reference latent shape");
                    return {};
                }
                shapes.emplace_back(ref.shape()[1], ref.shape()[0]);
            }
            shapes.emplace_back(x.shape()[1], x.shape()[0]);
            const auto* extra = std::get_if<QwenImage21DiffusionExtra>(&inputs.extra);
            QwenImage21Layout layout;
            try {
                layout = QwenImage21Layout::build(context.shape()[1], tensor_or_empty(extra ? extra->image_slots : nullptr), shapes);
            } catch (const std::exception& error) {
                LOG_ERROR("%s", error.what());
                return {};
            }
            pe_data = Rope::embed_nd(layout.positions, 1, 10000.f, config.axes_dim);
            mask_data.clear();
            for (const auto& segment : layout.segments) {
                sd::Tensor<float> mask;
                if (segment.image_index < 0) {
                    mask = sd::Tensor<float>::zeros({segment.end, segment.end - segment.start});
                    for (int64_t q = segment.start; q < segment.end; ++q) {
                        for (int64_t k = q + 1; k < segment.end; ++k) {
                            mask[k + segment.end * (q - segment.start)] = -INFINITY;
                        }
                    }
                }
                mask_data.push_back(std::move(mask));
            }
            auto build = [&]() {
                auto graph = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE * 2);
                auto pe    = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2, layout.positions.size());
                set_backend_tensor_data(pe, pe_data.data());
                std::vector<ggml_tensor*> masks, ref_inputs;
                for (const auto& mask : mask_data) {
                    masks.push_back(mask.empty() ? nullptr : make_input(mask));
                }
                for (const auto& ref : refs) {
                    ref_inputs.push_back(make_input(ref));
                }
                auto ctx = get_context();
                auto out = model.forward(&ctx, make_input(x), make_input(*inputs.timesteps), make_input(context),
                                         ref_inputs, pe, layout, masks);
                ggml_build_forward_expand(graph, out);
                return graph;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(build, n_threads, false), x.dim());
        }
    };
}

#endif  // __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
