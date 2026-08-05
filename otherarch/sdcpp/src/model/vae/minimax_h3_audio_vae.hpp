#ifndef __SD_MODEL_VAE_MINIMAX_H3_AUDIO_VAE_HPP__
#define __SD_MODEL_VAE_MINIMAX_H3_AUDIO_VAE_HPP__

#include <array>
#include <string>
#include <vector>

#include "model/vae/audio_vae.hpp"
#include "model/vae/ltx_audio_vae.hpp"

namespace MiniMaxH3 {

    struct AudioSnake1D : public UnaryBlock {
        int64_t channels;

        explicit AudioSnake1D(int64_t channels)
            : channels(channels) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            params["alpha"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto alpha       = params["alpha"];
            auto oscillation = ggml_sin(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, alpha));
            oscillation      = ggml_mul(ctx->ggml_ctx, oscillation, oscillation);
            auto eps         = ggml_ext_scale(ctx->ggml_ctx, ggml_ext_ones(ctx->ggml_ctx, 1, 1, 1, 1), 1e-9f);
            return ggml_add(ctx->ggml_ctx,
                            x,
                            ggml_div(ctx->ggml_ctx, oscillation, ggml_add(ctx->ggml_ctx, alpha, eps)));
        }
    };

    struct AudioEncoderResidualUnit : public GGMLBlock {
        int64_t channels;

        AudioEncoderResidualUnit(int64_t channels, int dilation)
            : channels(channels) {
            blocks["block.0"] = std::make_shared<AudioSnake1D>(channels);
            blocks["block.1"] = std::make_shared<LTXV::Conv1D>(channels,
                                                               channels,
                                                               7,
                                                               1,
                                                               3 * dilation,
                                                               dilation);
            blocks["block.2"] = std::make_shared<AudioSnake1D>(channels);
            blocks["block.3"] = std::make_shared<LTXV::Conv1D>(channels, channels, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto act1  = std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.0"]);
            auto conv1 = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["block.1"]);
            auto act2  = std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.2"]);
            auto conv2 = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["block.3"]);
            auto h     = conv2->forward(ctx, act2->forward(ctx, conv1->forward(ctx, act1->forward(ctx, x))));
            if (x->ne[0] != h->ne[0]) {
                int64_t pad = (x->ne[0] - h->ne[0]) / 2;
                x           = ggml_ext_slice(ctx->ggml_ctx, x, 0, pad, x->ne[0] - pad);
            }
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct AudioEncoderBlock : public GGMLBlock {
        int64_t out_channels;

        AudioEncoderBlock(int64_t out_channels, int stride)
            : out_channels(out_channels) {
            int64_t in_channels = out_channels / 2;
            blocks["block.0"]   = std::make_shared<AudioEncoderResidualUnit>(in_channels, 1);
            blocks["block.1"]   = std::make_shared<AudioEncoderResidualUnit>(in_channels, 3);
            blocks["block.2"]   = std::make_shared<AudioEncoderResidualUnit>(in_channels, 9);
            blocks["block.3"]   = std::make_shared<AudioSnake1D>(in_channels);
            blocks["block.4"]   = std::make_shared<LTXV::Conv1D>(in_channels,
                                                               out_channels,
                                                               2 * stride,
                                                               stride,
                                                               static_cast<int>(std::ceil(stride / 2.f)));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            for (int i = 0; i < 3; ++i) {
                auto unit = std::dynamic_pointer_cast<AudioEncoderResidualUnit>(blocks["block." + std::to_string(i)]);
                x         = unit->forward(ctx, x);
            }
            auto act  = std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.3"]);
            auto conv = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["block.4"]);
            return conv->forward(ctx, act->forward(ctx, x));
        }
    };

    struct AudioEncoder : public GGMLBlock {
        static constexpr std::array<int, 5> strides = {2, 4, 4, 5, 5};

        AudioEncoder() {
            int64_t channels  = 64;
            blocks["block.0"] = std::make_shared<LTXV::Conv1D>(1, channels, 7, 1, 3);
            for (size_t i = 0; i < strides.size(); ++i) {
                channels *= 2;
                blocks["block." + std::to_string(i + 1)] = std::make_shared<AudioEncoderBlock>(channels, strides[i]);
            }
            blocks["block.6"] = std::make_shared<AudioSnake1D>(channels);
            blocks["block.7"] = std::make_shared<LTXV::Conv1D>(channels, 2048, 3, 1, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto input = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["block.0"]);
            x          = input->forward(ctx, x);
            for (size_t i = 0; i < strides.size(); ++i) {
                auto block = std::dynamic_pointer_cast<AudioEncoderBlock>(blocks["block." + std::to_string(i + 1)]);
                x          = block->forward(ctx, x);
            }
            auto act = std::dynamic_pointer_cast<AudioSnake1D>(blocks["block.6"]);
            auto out = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["block.7"]);
            return out->forward(ctx, act->forward(ctx, x));
        }
    };

    struct AudioGeGLUMLP : public GGMLBlock {
        AudioGeGLUMLP(int64_t hidden_size, int64_t intermediate_size) {
            blocks["norm"] = std::make_shared<LayerNorm>(hidden_size);
            blocks["w0"]   = std::make_shared<Linear>(hidden_size, intermediate_size, true);
            blocks["w1"]   = std::make_shared<Linear>(hidden_size, intermediate_size, true);
            blocks["w2"]   = std::make_shared<Linear>(intermediate_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto w0   = std::dynamic_pointer_cast<Linear>(blocks["w0"]);
            auto w1   = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2   = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            x         = norm->forward(ctx, x);
            auto gate = ggml_ext_gelu(ctx->ggml_ctx, w0->forward(ctx, x), true);
            return w2->forward(ctx, ggml_mul(ctx->ggml_ctx, gate, w1->forward(ctx, x)));
        }
    };

    struct AudioCausalAttention : public GGMLBlock {
        static constexpr int64_t in_channels  = 2048;
        static constexpr int64_t out_channels = 32;
        static constexpr int64_t num_head     = 8;
        static constexpr int64_t head_dim     = in_channels / num_head;

        AudioCausalAttention() {
            blocks["qkv"]  = std::make_shared<Linear>(in_channels, in_channels * 3, false);
            blocks["proj"] = std::make_shared<Linear>(out_channels, out_channels, true);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
            params["q_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
            params["v_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto qkv_layer  = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto proj       = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            auto qkv        = ggml_ext_chunk(ctx->ggml_ctx, qkv_layer->forward(ctx, x), 3, 0);
            auto bias_shape = [&](ggml_tensor* bias) {
                return ggml_reshape_4d(ctx->ggml_ctx, bias, bias->ne[0], 1, 1, 1);
            };
            auto q = ggml_add(ctx->ggml_ctx, qkv[0], bias_shape(params["q_bias"]));
            auto k = qkv[1];
            auto v = ggml_add(ctx->ggml_ctx, qkv[2], bias_shape(params["v_bias"]));

            int64_t sequence = x->ne[1];
            auto mask        = ggml_diag_mask_inf(ctx->ggml_ctx,
                                                  ggml_ext_zeros(ctx->ggml_ctx, sequence, sequence, 1, 1),
                                                  0);
            auto attn_out    = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                      ctx->backend,
                                                      q,
                                                      k,
                                                      v,
                                                      num_head,
                                                      mask,
                                                      false,
                                                      ctx->flash_attn_enabled);
            int64_t batch    = attn_out->ne[2] * attn_out->ne[3];
            attn_out         = ggml_reshape_4d(ctx->ggml_ctx, attn_out, head_dim, num_head, sequence, batch);
            attn_out         = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, attn_out, 1, 0, 2, 3));
            attn_out         = ggml_mean(ctx->ggml_ctx, attn_out);
            attn_out         = ggml_reshape_3d(ctx->ggml_ctx, attn_out, head_dim, sequence, batch);

            constexpr int64_t pool = head_dim / out_channels;
            attn_out               = ggml_reshape_4d(ctx->ggml_ctx, attn_out, pool, out_channels, sequence, batch);
            attn_out               = ggml_mean(ctx->ggml_ctx, attn_out);
            attn_out               = ggml_reshape_3d(ctx->ggml_ctx, attn_out, out_channels, sequence, batch);
            return proj->forward(ctx, attn_out);
        }
    };

    struct AudioAttentionProjection : public GGMLBlock {
        AudioAttentionProjection() {
            blocks["norm1"] = std::make_shared<LayerNorm>(2048);
            blocks["attn"]  = std::make_shared<AudioCausalAttention>();
            blocks["proj"]  = std::make_shared<Linear>(2048, 32, true);
            blocks["norm3"] = std::make_shared<LayerNorm>(2048);
            blocks["norm2"] = std::make_shared<LayerNorm>(32);
            blocks["mlp"]   = std::make_shared<AudioGeGLUMLP>(32, 64);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto attn  = std::dynamic_pointer_cast<AudioCausalAttention>(blocks["attn"]);
            auto proj  = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            auto norm3 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);
            auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto mlp   = std::dynamic_pointer_cast<AudioGeGLUMLP>(blocks["mlp"]);
            x          = ggml_add(ctx->ggml_ctx,
                                  proj->forward(ctx, norm3->forward(ctx, x)),
                                  attn->forward(ctx, norm1->forward(ctx, x)));
            return ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, norm2->forward(ctx, x)));
        }
    };

    struct AudioAMPBlock : public GGMLBlock {
        int channels;

        AudioAMPBlock(int channels,
                      int kernel_size,
                      const std::array<int, 3>& dilations)
            : channels(channels) {
            for (int i = 0; i < 3; ++i) {
                blocks["activations." + std::to_string(i * 2)] =
                    std::make_shared<LTXV::Activation1D>(channels);
                blocks["activations." + std::to_string(i * 2 + 1)] =
                    std::make_shared<LTXV::Activation1D>(channels);
                blocks["convs1." + std::to_string(i)] =
                    std::make_shared<LTXV::Conv1D>(channels,
                                                   channels,
                                                   kernel_size,
                                                   1,
                                                   (kernel_size * dilations[i] - dilations[i]) / 2,
                                                   dilations[i]);
                blocks["convs2." + std::to_string(i)] =
                    std::make_shared<LTXV::Conv1D>(channels,
                                                   channels,
                                                   kernel_size,
                                                   1,
                                                   kernel_size / 2);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            for (int i = 0; i < 3; ++i) {
                auto act1 = std::dynamic_pointer_cast<LTXV::Activation1D>(
                    blocks["activations." + std::to_string(i * 2)]);
                auto act2 = std::dynamic_pointer_cast<LTXV::Activation1D>(
                    blocks["activations." + std::to_string(i * 2 + 1)]);
                auto conv1 = std::dynamic_pointer_cast<LTXV::Conv1D>(
                    blocks["convs1." + std::to_string(i)]);
                auto conv2 = std::dynamic_pointer_cast<LTXV::Conv1D>(
                    blocks["convs2." + std::to_string(i)]);

                auto h = conv1->forward(ctx, act1->forward(ctx, x));
                h      = conv2->forward(ctx, act2->forward(ctx, h));
                x      = ggml_add(ctx->ggml_ctx, x, h);
            }
            return x;
        }
    };

    struct BigVGAN : public GGMLBlock {
        static constexpr int initial_channels                     = 1024;
        static constexpr int num_kernels                          = 3;
        static constexpr int num_upsamples                        = 7;
        static constexpr std::array<int, num_upsamples> rates     = {5, 5, 2, 2, 2, 2, 2};
        static constexpr std::array<int, num_upsamples> kernels   = {9, 9, 4, 4, 4, 4, 4};
        static constexpr std::array<int, num_kernels> res_kernels = {3, 7, 11};

        BigVGAN() {
            blocks["conv_pre"] = std::make_shared<LTXV::Conv1D>(2048,
                                                                initial_channels,
                                                                7,
                                                                1,
                                                                3);
            int channels       = initial_channels;
            for (int i = 0; i < num_upsamples; ++i) {
                int next_channels = initial_channels / (1 << (i + 1));
                blocks["ups." + std::to_string(i) + ".0"] =
                    std::make_shared<LTXV::ConvTranspose1D>(channels,
                                                            next_channels,
                                                            kernels[i],
                                                            rates[i],
                                                            (kernels[i] - rates[i]) / 2);
                for (int j = 0; j < num_kernels; ++j) {
                    blocks["resblocks." + std::to_string(i * num_kernels + j)] =
                        std::make_shared<AudioAMPBlock>(next_channels,
                                                        res_kernels[j],
                                                        std::array<int, 3>{1, 3, 5});
                }
                channels = next_channels;
            }
            blocks["activation_post"] = std::make_shared<LTXV::Activation1D>(channels);
            blocks["conv_post"]       = std::make_shared<LTXV::Conv1D>(channels,
                                                                 1,
                                                                 7,
                                                                 1,
                                                                 3,
                                                                 1,
                                                                 false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv_pre = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["conv_pre"]);
            x             = conv_pre->forward(ctx, x);
            for (int i = 0; i < num_upsamples; ++i) {
                auto up = std::dynamic_pointer_cast<LTXV::ConvTranspose1D>(
                    blocks["ups." + std::to_string(i) + ".0"]);
                x = up->forward(ctx, x);

                ggml_tensor* sum = nullptr;
                for (int j = 0; j < num_kernels; ++j) {
                    auto block = std::dynamic_pointer_cast<AudioAMPBlock>(
                        blocks["resblocks." + std::to_string(i * num_kernels + j)]);
                    auto value = block->forward(ctx, x);
                    sum        = sum == nullptr ? value : ggml_add(ctx->ggml_ctx, sum, value);
                }
                x = ggml_ext_scale(ctx->ggml_ctx, sum, 1.f / num_kernels);
            }
            auto activation = std::dynamic_pointer_cast<LTXV::Activation1D>(blocks["activation_post"]);
            auto conv_post  = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["conv_post"]);
            return ggml_clamp(ctx->ggml_ctx,
                              conv_post->forward(ctx, activation->forward(ctx, x)),
                              -1.f,
                              1.f);
        }
    };

    struct AudioVAE : public GGMLBlock {
        static constexpr int kLatentChannels = 32;

        AudioVAE() {
            blocks["encoder"]     = std::make_shared<AudioEncoder>();
            blocks["pre_block"]   = std::make_shared<AudioAttentionProjection>();
            blocks["mean_proj"]   = std::make_shared<LTXV::Conv1D>(kLatentChannels, kLatentChannels, 1);
            blocks["logs_proj"]   = std::make_shared<LTXV::Conv1D>(kLatentChannels, kLatentChannels, 1);
            blocks["dec_in_proj"] = std::make_shared<LTXV::Conv1D>(kLatentChannels,
                                                                   2048,
                                                                   1);
            blocks["decoder"]     = std::make_shared<BigVGAN>();
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            params["latents_mean"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kLatentChannels);
            params["latents_std"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kLatentChannels);
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* waveform) {
            GGML_ASSERT(waveform->ne[1] == 2);
            auto encoder   = std::dynamic_pointer_cast<AudioEncoder>(blocks["encoder"]);
            auto pre       = std::dynamic_pointer_cast<AudioAttentionProjection>(blocks["pre_block"]);
            auto mean_proj = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["mean_proj"]);

            waveform = ggml_reshape_3d(ctx->ggml_ctx, waveform, waveform->ne[0], 1, waveform->ne[1]);
            auto x   = encoder->forward(ctx, waveform);  // [B*S, 2048, T]
            x        = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x        = pre->forward(ctx, x);
            x        = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            auto z   = mean_proj->forward(ctx, x);

            auto mean = ggml_reshape_4d(ctx->ggml_ctx, params["latents_mean"], 1, kLatentChannels, 1, 1);
            auto std  = ggml_reshape_4d(ctx->ggml_ctx, params["latents_std"], 1, kLatentChannels, 1, 1);
            z         = ggml_div(ctx->ggml_ctx, ggml_sub(ctx->ggml_ctx, z, mean), std);
            return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 0, 2, 1, 3));
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* latent) {
            GGML_ASSERT(latent->ne[1] == 2 && latent->ne[2] == kLatentChannels);
            latent = ggml_cont(ctx->ggml_ctx,
                               ggml_permute(ctx->ggml_ctx, latent, 0, 2, 1, 3));

            auto mean = ggml_reshape_4d(ctx->ggml_ctx,
                                        params["latents_mean"],
                                        1,
                                        kLatentChannels,
                                        1,
                                        1);
            auto std  = ggml_reshape_4d(ctx->ggml_ctx,
                                        params["latents_std"],
                                        1,
                                        kLatentChannels,
                                        1,
                                        1);
            latent    = ggml_add(ctx->ggml_ctx,
                                 ggml_mul(ctx->ggml_ctx, latent, std),
                                 mean);

            auto dec_in           = std::dynamic_pointer_cast<LTXV::Conv1D>(blocks["dec_in_proj"]);
            auto decoder          = std::dynamic_pointer_cast<BigVGAN>(blocks["decoder"]);
            int64_t streams       = latent->ne[2] * latent->ne[3];
            latent                = ggml_reshape_3d(ctx->ggml_ctx,
                                                    latent,
                                                    latent->ne[0],
                                                    latent->ne[1],
                                                    streams);
            ggml_tensor* waveform = nullptr;
            for (int64_t stream = 0; stream < streams; ++stream) {
                auto stream_latent   = ggml_ext_slice(ctx->ggml_ctx, latent, 2, stream, stream + 1);
                auto stream_waveform = decoder->forward(ctx, dec_in->forward(ctx, stream_latent));
                waveform             = waveform == nullptr
                                           ? stream_waveform
                                           : ggml_concat(ctx->ggml_ctx, waveform, stream_waveform, 2);
            }
            return ggml_reshape_4d(ctx->ggml_ctx,
                                   waveform,
                                   waveform->ne[0],
                                   streams,
                                   1,
                                   1);
        }
    };

    struct AudioVAERunner : public ::AudioVAERunner {
        AudioVAE model;
        std::string weight_prefix;

        AudioVAERunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map,
                       const std::string& prefix                           = "",
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : ::AudioVAERunner(backend, weight_manager),
              weight_prefix(prefix) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            model.get_param_tensors(tensors, weight_prefix);
        }

        size_t get_params_mem_size() override {
            return model.get_params_mem_size();
        }

        std::string get_desc() override {
            return "minimax_h3_audio_vae";
        }

        int output_sample_rate() const override {
            return 32000;
        }

        sd::Tensor<float> encode(int n_threads,
                                 const sd::Tensor<float>& waveform) override {
            int64_t t0     = ggml_time_ms();
            auto get_graph = [&]() -> ggml_cgraph* {
                auto input      = make_input(waveform);
                auto runner_ctx = get_context();
                auto latent     = model.encode(&runner_ctx, input);
                auto graph      = new_graph_custom(655360);
                ggml_build_forward_expand(graph, latent);
                return graph;
            };
            auto result = restore_trailing_singleton_dims(
                GGMLRunner::compute<float>(get_graph, n_threads, false, false, false),
                4);
            int64_t t1 = ggml_time_ms();
            LOG_INFO("MiniMax-H3 audio VAE encode completed, taking %.2fs",
                     (t1 - t0) / 1000.f);
            return result;
        }

        sd::Tensor<float> decode(int n_threads,
                                 const sd::Tensor<float>& latent_tensor) override {
            int64_t t0     = ggml_time_ms();
            auto get_graph = [&]() -> ggml_cgraph* {
                auto latent     = make_input(latent_tensor);
                auto runner_ctx = get_context();
                auto waveform   = model.decode(&runner_ctx, latent);
                auto graph      = new_graph_custom(655360);
                ggml_build_forward_expand(graph, waveform);
                return graph;
            };
            auto result = restore_trailing_singleton_dims(
                GGMLRunner::compute<float>(get_graph, n_threads, false, false, false),
                4);
            int64_t t1 = ggml_time_ms();
            LOG_INFO("MiniMax-H3 audio VAE decode completed, taking %.2fs",
                     (t1 - t0) / 1000.f);
            return result;
        }
    };

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_VAE_MINIMAX_H3_AUDIO_VAE_HPP__
