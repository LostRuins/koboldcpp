#ifndef __SD_MODEL_DIFFUSION_WAN_AUDIO_HPP__
#define __SD_MODEL_DIFFUSION_WAN_AUDIO_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model/common/ggml_block.hpp"

namespace WAN {

    class WanCausalConv1d : public UnaryBlock {
    private:
        int kernel_size_;

    public:
        WanCausalConv1d(int64_t in_dim,
                        int64_t out_dim,
                        int kernel_size = 3,
                        int stride      = 1)
            : kernel_size_(kernel_size) {
            blocks["conv"] = std::make_shared<Conv1d>(in_dim, out_dim, kernel_size, stride, 0, 1, 1, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // Replicate the first sample for causal left padding.
            if (kernel_size_ > 1) {
                auto first = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, 1);
                for (int i = 0; i < kernel_size_ - 1; i++) {
                    x = ggml_concat(ctx->ggml_ctx, first, x, 0);
                }
            }
            return std::dynamic_pointer_cast<Conv1d>(blocks["conv"])->forward(ctx, x);
        }
    };

    class WanMotionEncoder : public GGMLBlock {
    private:
        int64_t hidden_dim_;
        int num_token_;
        bool need_global_;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // The padding token is combined with F32 activations.
            params["padding_tokens"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim_);
        }

        ggml_tensor* conv_norm_silu(GGMLRunnerContext* ctx,
                                    ggml_tensor* x,
                                    const std::string& conv_key,
                                    const std::string& norm_key,
                                    bool to_conv_layout) {
            x = std::dynamic_pointer_cast<WanCausalConv1d>(blocks[conv_key])->forward(ctx, x);
            x = ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);
            x = std::dynamic_pointer_cast<LayerNorm>(blocks[norm_key])->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (to_conv_layout) {
                x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            }
            return x;
        }

    public:
        WanMotionEncoder(int64_t in_dim,
                         int64_t hidden_dim,
                         int num_token,
                         bool need_global = true)
            : hidden_dim_(hidden_dim), num_token_(num_token), need_global_(need_global) {
            blocks["conv1_local"] = std::make_shared<WanCausalConv1d>(in_dim, hidden_dim / 4 * num_token);
            if (need_global) {
                blocks["conv1_global"] = std::make_shared<WanCausalConv1d>(in_dim, hidden_dim / 4);
            }
            blocks["norm1"] = std::make_shared<LayerNorm>(hidden_dim / 4, 1e-6f, false);
            blocks["conv2"] = std::make_shared<WanCausalConv1d>(hidden_dim / 4, hidden_dim / 2, 3, 2);
            blocks["norm2"] = std::make_shared<LayerNorm>(hidden_dim / 2, 1e-6f, false);
            blocks["conv3"] = std::make_shared<WanCausalConv1d>(hidden_dim / 2, hidden_dim, 3, 2);
            blocks["norm3"] = std::make_shared<LayerNorm>(hidden_dim, 1e-6f, false);
            if (need_global) {
                blocks["final_linear"] = std::make_shared<Linear>(hidden_dim, hidden_dim);
            }
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto local = std::dynamic_pointer_cast<WanCausalConv1d>(blocks["conv1_local"])->forward(ctx, x);
            auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            std::vector<ggml_tensor*> tokens;
            // Each token group is normalized independently over channels.
            for (auto& group : ggml_ext_chunk(ctx->ggml_ctx, local, num_token_, 1)) {
                ggml_tensor* s = ggml_permute(ctx->ggml_ctx, group, 1, 0, 2, 3);
                s              = norm1->forward(ctx, s);
                s              = ggml_silu(ctx->ggml_ctx, s);
                s              = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, s, 1, 0, 2, 3));
                s              = conv_norm_silu(ctx, s, "conv2", "norm2", true);
                s              = conv_norm_silu(ctx, s, "conv3", "norm3", false);
                tokens.push_back(ggml_reshape_3d(ctx->ggml_ctx, s, s->ne[0], 1, s->ne[1]));
            }
            auto padding = ggml_reshape_3d(ctx->ggml_ctx, params["padding_tokens"], hidden_dim_, 1, 1);
            padding      = ggml_repeat(ctx->ggml_ctx, padding, tokens[0]);
            tokens.push_back(padding);
            ggml_tensor* local_out = ggml_ext_vec_concat(ctx->ggml_ctx, tokens, 1);

            if (!need_global_) {
                return {local_out, nullptr};
            }
            ggml_tensor* g = conv_norm_silu(ctx, x, "conv1_global", "norm1", true);
            g              = conv_norm_silu(ctx, g, "conv2", "norm2", true);
            g              = conv_norm_silu(ctx, g, "conv3", "norm3", false);
            g              = std::dynamic_pointer_cast<Linear>(blocks["final_linear"])->forward(ctx, g);
            return {local_out, g};
        }
    };

    class WanCausalAudioEncoder : public GGMLBlock {
    private:
        int num_layers_;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // Preserve the checkpoint shape for loading; layer mixing requires F32.
            auto it = tensor_storage_map.find(prefix + "weights");
            if (it != tensor_storage_map.end()) {
                params["weights"] = ggml_new_tensor(ctx, GGML_TYPE_F32, it->second.n_dims, it->second.ne);
            } else {
                params["weights"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_layers_);
            }
        }

    public:
        WanCausalAudioEncoder(int64_t audio_dim,
                              int64_t dim,
                              int num_token,
                              int num_layers = 25)
            : num_layers_(num_layers) {
            blocks["encoder"] = std::make_shared<WanMotionEncoder>(audio_dim, dim, num_token, true);
        }

        // features: [layers, frames, audio_dim]; outputs: [T, tokens+1, dim] and [T, dim].
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* features) {
            auto weights = ggml_silu(ctx->ggml_ctx, params["weights"]);
            auto x       = ggml_mul(ctx->ggml_ctx, features, ggml_reshape_3d(ctx->ggml_ctx, weights, 1, 1, num_layers_));
            x            = ggml_div(ctx->ggml_ctx, x, ggml_sum(ctx->ggml_ctx, weights));
            // Move the layer axis to ggml dimension 0 for reduction.
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
            x = ggml_sum_rows(ctx->ggml_ctx, x);
            x = ggml_reshape_2d(ctx->ggml_ctx, x, x->ne[1], x->ne[2]);
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            return std::dynamic_pointer_cast<WanMotionEncoder>(blocks["encoder"])->forward(ctx, x);
        }
    };

    class WanAudioInjector : public GGMLBlock {
    private:
        int64_t dim_;

    public:
        WanAudioInjector(int64_t dim,
                         int64_t num_heads,
                         int count,
                         bool qk_norm = true,
                         float eps    = 1e-6f)
            : dim_(dim) {
            for (int i = 0; i < count; i++) {
                blocks["injector." + std::to_string(i)] =
                    std::make_shared<WanT2VCrossAttention>(dim, num_heads, qk_norm, eps);
                blocks["injector_adain_layers." + std::to_string(i) + ".linear"] =
                    std::make_shared<Linear>(dim, dim * 2);
            }
            // S2V AdaLayerNorm uses its own epsilon, independent of attention norms.
            blocks["adain_norm"] = std::make_shared<LayerNorm>(dim, 1e-5f, false);
        }

        // Inject into the video prefix; trailing reference tokens pass through unchanged.
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t seq_len,
                             int64_t T,
                             int injector_id,
                             ggml_tensor* audio_local,
                             ggml_tensor* audio_global) {
            int64_t n_tok   = seq_len / T;
            int64_t n_token = x->ne[1];

            auto adain_linear = std::dynamic_pointer_cast<Linear>(blocks["injector_adain_layers." + std::to_string(injector_id) + ".linear"]);
            auto injector     = std::dynamic_pointer_cast<WanT2VCrossAttention>(blocks["injector." + std::to_string(injector_id)]);
            auto adain_norm   = std::dynamic_pointer_cast<LayerNorm>(blocks["adain_norm"]);

            auto temb  = ggml_silu(ctx->ggml_ctx, audio_global);
            temb       = adain_linear->forward(ctx, temb);
            auto shift = ggml_ext_slice(ctx->ggml_ctx, temb, 0, 0, dim_);
            auto scale = ggml_ext_slice(ctx->ggml_ctx, temb, 0, dim_, dim_ * 2);
            shift      = ggml_reshape_3d(ctx->ggml_ctx, shift, dim_, 1, T);
            scale      = ggml_reshape_3d(ctx->ggml_ctx, scale, dim_, 1, T);

            auto x_vid = ggml_ext_slice(ctx->ggml_ctx, x, 1, 0, seq_len);
            auto h     = ggml_reshape_3d(ctx->ggml_ctx, x_vid, dim_, n_tok, T);
            h          = adain_norm->forward(ctx, h);
            h          = ggml_add(ctx->ggml_ctx, h, ggml_mul(ctx->ggml_ctx, h, scale));
            h          = ggml_add(ctx->ggml_ctx, h, shift);

            auto res = injector->forward(ctx, h, audio_local, 0);
            res      = ggml_reshape_2d(ctx->ggml_ctx, res, dim_, seq_len);

            auto x_head = ggml_add(ctx->ggml_ctx, x_vid, res);
            if (seq_len < n_token) {
                auto x_tail = ggml_ext_slice(ctx->ggml_ctx, x, 1, seq_len, n_token);
                return ggml_concat(ctx->ggml_ctx, x_head, x_tail, 1);
            }
            return x_head;
        }
    };

}  // namespace WAN

#endif  // __SD_MODEL_DIFFUSION_WAN_AUDIO_HPP__
