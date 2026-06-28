#ifndef __SD_MODEL_DIFFUSION_SEFI_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_SEFI_IMAGE_HPP__

#include <memory>

#include "model/common/block.hpp"

namespace SefiImage {
    struct SefiImageConfig {
        int64_t semantic_channels        = 16;
        int64_t texture_latent_channels  = 32;
        int64_t timestep_guidance_in_dim = 256;
        int64_t hidden_size              = 3072;
        float timestep_shift_alpha       = 0.3f;
        float delta_t                    = 0.1f;

        int64_t packed_texture_channels(int patch_size) const {
            return texture_latent_channels * patch_size * patch_size;
        }

        int64_t packed_input_channels(int patch_size) const {
            return semantic_channels + packed_texture_channels(patch_size);
        }

        static SefiImageConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                   const std::string& prefix) {
            SefiImageConfig config;
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "dual_time_embed.semantic_embedder.linear_1.weight") && tensor_storage.n_dims == 2) {
                    config.timestep_guidance_in_dim = tensor_storage.ne[0];
                    config.hidden_size              = tensor_storage.ne[1] * 2;
                }
            }
            LOG_DEBUG("sefi_image: semantic_channels = %" PRId64 ", texture_latent_channels = %" PRId64 ", hidden_size = %" PRId64,
                      config.semantic_channels,
                      config.texture_latent_channels,
                      config.hidden_size);
            return config;
        }
    };

    struct SefiTimestepEmbedding : public GGMLBlock {
    public:
        SefiTimestepEmbedding(int64_t in_channels, int64_t time_embed_dim) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim, false));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim, false));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* sample) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            sample = linear_1->forward(ctx, sample);
            sample = ggml_silu_inplace(ctx->ggml_ctx, sample);
            sample = linear_2->forward(ctx, sample);
            return sample;
        }
    };

    struct SefiDualTimestepEmbeddings : public GGMLBlock {
    public:
        SefiDualTimestepEmbeddings(int64_t in_channels, int64_t embedding_dim) {
            GGML_ASSERT(embedding_dim % 2 == 0);
            int64_t half_dim            = embedding_dim / 2;
            blocks["semantic_embedder"] = std::make_shared<SefiTimestepEmbedding>(in_channels, half_dim);
            blocks["texture_embedder"]  = std::make_shared<SefiTimestepEmbedding>(in_channels, half_dim);
            timestep_guidance_in_dim    = in_channels;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* timestep_sem,
                             ggml_tensor* timestep_tex) {
            auto semantic_embedder = std::dynamic_pointer_cast<SefiTimestepEmbedding>(blocks["semantic_embedder"]);
            auto texture_embedder  = std::dynamic_pointer_cast<SefiTimestepEmbedding>(blocks["texture_embedder"]);

            auto sem_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep_sem, (int)timestep_guidance_in_dim, 10000, 1.f);
            auto tex_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep_tex, (int)timestep_guidance_in_dim, 10000, 1.f);
            auto sem_emb  = semantic_embedder->forward(ctx, sem_proj);
            auto tex_emb  = texture_embedder->forward(ctx, tex_proj);
            return ggml_concat(ctx->ggml_ctx, sem_emb, tex_emb, 0);
        }

    private:
        int64_t timestep_guidance_in_dim = 256;
    };
}  // namespace SefiImage

#endif  // __SD_MODEL_DIFFUSION_SEFI_IMAGE_HPP__
