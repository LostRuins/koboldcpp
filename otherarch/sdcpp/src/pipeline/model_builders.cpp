#include "model_builders.h"

#include <cstring>
#include <utility>

#include "conditioning/conditioner.hpp"
#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "extensions/generation_extension.h"
#include "model/adapter/ip_adapter.hpp"
#include "model/diffusion/anima.hpp"
#include "model/diffusion/boogu.hpp"
#include "model/diffusion/control.hpp"
#include "model/diffusion/ernie_image.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/hidream_o1.hpp"
#include "model/diffusion/hunyuan.hpp"
#include "model/diffusion/ideogram4.hpp"
#include "model/diffusion/krea2.hpp"
#include "model/diffusion/lens.hpp"
#include "model/diffusion/lingbot_video.hpp"
#include "model/diffusion/ltxv.hpp"
#include "model/diffusion/mage_flow.hpp"
#include "model/diffusion/minimax_h3.hpp"
#include "model/diffusion/minit2i.hpp"
#include "model/diffusion/mmdit.hpp"
#include "model/diffusion/model.hpp"
#include "model/diffusion/pid.hpp"
#include "model/diffusion/qwen_image.hpp"
#include "model/diffusion/sensenova_u1.h"
#include "model/diffusion/unet.hpp"
#include "model/diffusion/wan.hpp"
#include "model/diffusion/z_image.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "model/vae/hunyuan_vae.hpp"
#include "model/vae/ltx_audio_vae.hpp"
#include "model/vae/ltx_vae.hpp"
#include "model/vae/mage_vae.hpp"
#include "model/vae/minimax_h3_audio_vae.hpp"
#include "model/vae/minimax_h3_vae.hpp"
#include "model/vae/tae.hpp"
#include "model/vae/vae.hpp"
#include "model/vae/wan_vae.hpp"

namespace sd::model_builders {

    static bool ensure_backend_pair(SDBackendManager& backends, SDBackendModule module) {
        if (backends.runtime_backend(module) == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
            return false;
        }
        if (backends.params_backend(module) == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
            return false;
        }
        return true;
    }

    static SDVersion sd_vae_format_to_version(enum sd_vae_format_t format, SDVersion fallback) {
        switch (format) {
            case SD_VAE_FORMAT_FLUX:
                return VERSION_FLUX;
            case SD_VAE_FORMAT_SD3:
                return VERSION_SD3;
            case SD_VAE_FORMAT_FLUX2:
                return VERSION_FLUX2;
            case SD_VAE_FORMAT_WAN:
                return VERSION_WAN2;
            case SD_VAE_FORMAT_AUTO:
            default:
                return fallback;
        }
    }

    bool build_core_runners(const Context& ctx, CoreRunners& runners) {
        const auto* sd_ctx_params      = &ctx.params;
        const auto& tensor_storage_map = ctx.tensor_storage_map;
        const auto version             = ctx.version;
        const auto& weight_manager     = ctx.weight_manager;
        CoreRunners result;
        if (!ensure_backend_pair(ctx.backends, SDBackendModule::TE) ||
            !ensure_backend_pair(ctx.backends, SDBackendModule::DIFFUSION)) {
            return false;
        }

        if (sd_version_is_sd3(version)) {
            result.conditioner = std::make_shared<SD3CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                   tensor_storage_map,
                                                                   weight_manager);
            result.diffusion   = std::make_shared<MMDiTRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                             tensor_storage_map,
                                                             "model.diffusion_model",
                                                             weight_manager);
        } else if (sd_version_is_pid(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Pid::PiDRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                tensor_storage_map,
                                                                "model.diffusion_model.net",
                                                                weight_manager);
        } else if (sd_version_is_ideogram4(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Ideogram4::Ideogram4Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            weight_manager);
        } else if (sd_version_is_krea2(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               true,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Krea2::Krea2Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model",
                                                                    weight_manager);
        } else if (sd_version_is_flux(version)) {
            bool is_chroma = false;
            for (auto pair : tensor_storage_map) {
                if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                    is_chroma = true;
                    break;
                }
            }
            if (is_chroma) {
                result.conditioner = std::make_shared<T5CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                      tensor_storage_map,
                                                                      false,
                                                                      1,
                                                                      false,
                                                                      weight_manager,
                                                                      sd_ctx_params->model_args);
            } else if (version == VERSION_OVIS_IMAGE) {
                result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                   tensor_storage_map,
                                                                   version,
                                                                   "",
                                                                   false,
                                                                   weight_manager);
            } else {
                result.conditioner = std::make_shared<FluxCLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                        tensor_storage_map,
                                                                        weight_manager);
            }
            result.diffusion = std::make_shared<Flux::FluxRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                  tensor_storage_map,
                                                                  "model.diffusion_model",
                                                                  version,
                                                                  weight_manager,
                                                                  sd_ctx_params->model_args);
        } else if (sd_version_is_flux2(version) || sd_version_is_sefi_image(version)) {
            bool is_chroma     = false;
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Flux::FluxRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                  tensor_storage_map,
                                                                  "model.diffusion_model",
                                                                  version,
                                                                  weight_manager,
                                                                  sd_ctx_params->model_args);
        } else if (sd_version_is_ltxav(version)) {
            result.conditioner = std::make_shared<LTXAVEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 "text_encoders.llm",
                                                                 "text_embedding_projection",
                                                                 weight_manager);
            result.diffusion   = std::make_shared<LTXV::LTXAVRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model",
                                                                   weight_manager);
        } else if (sd_version_is_minimax_h3(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               true,
                                                               weight_manager);
            result.diffusion   = std::make_shared<MiniMaxH3::MiniMaxH3Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            weight_manager);
        } else if (sd_version_is_hunyuan_video(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Hunyuan::HunyuanVideoRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                             tensor_storage_map,
                                                                             "model.diffusion_model",
                                                                             version,
                                                                             weight_manager);
        } else if (sd_version_is_wan(version)) {
            result.conditioner = std::make_shared<T5CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                  tensor_storage_map,
                                                                  true,
                                                                  0,
                                                                  true,
                                                                  weight_manager);
            result.diffusion   = std::make_shared<WAN::WanRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                tensor_storage_map,
                                                                "model.diffusion_model",
                                                                version,
                                                                weight_manager);
            if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                result.high_noise_diffusion = std::make_shared<WAN::WanRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                               tensor_storage_map,
                                                                               "model.high_noise_diffusion_model",
                                                                               version,
                                                                               weight_manager);
            }
            if (result.diffusion->get_desc() == "Wan2.1-I2V-14B" ||
                result.diffusion->get_desc() == "Wan2.1-FLF2V-14B" ||
                result.diffusion->get_desc() == "Wan2.1-I2V-1.3B") {
                if (!ensure_backend_pair(ctx.backends, SDBackendModule::CLIP_VISION)) {
                    return false;
                }
                result.clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(ctx.backends.runtime_backend(SDBackendModule::CLIP_VISION),
                                                                                tensor_storage_map,
                                                                                weight_manager);
            }
        } else if (sd_version_is_lingbot_video(version)) {
            bool enable_vision = false;
            for (const auto& [name, _] : tensor_storage_map) {
                if (starts_with(name, "text_encoders.llm.visual.")) {
                    enable_vision = true;
                    break;
                }
            }
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               enable_vision,
                                                               weight_manager);
            result.diffusion   = std::make_shared<LingBotVideo::LingBotVideoRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                                  tensor_storage_map,
                                                                                  "model.diffusion_model",
                                                                                  weight_manager,
                                                                                  sd_ctx_params->model_args);
        } else if (sd_version_is_qwen_image(version)) {
            bool enable_vision = version != VERSION_QWEN_IMAGE_LAYERED;
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               enable_vision,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Qwen::QwenImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.diffusion_model",
                                                                       version,
                                                                       weight_manager,
                                                                       sd_ctx_params->model_args);
        } else if (sd_version_is_mage_flow(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               true,
                                                               weight_manager);
            result.diffusion   = std::make_shared<MageFlow::MageFlowRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                          tensor_storage_map,
                                                                          "model.diffusion_model",
                                                                          weight_manager);
        } else if (sd_version_is_longcat(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               true,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Flux::FluxRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                  tensor_storage_map,
                                                                  "model.diffusion_model",
                                                                  version,
                                                                  weight_manager,
                                                                  sd_ctx_params->model_args);
        } else if (version == VERSION_HIDREAM_O1) {
            result.conditioner = std::make_shared<HiDreamO1::HiDreamO1Conditioner>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                                   tensor_storage_map,
                                                                                   weight_manager);
            result.diffusion   = std::make_shared<HiDreamO1::HiDreamO1Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model",
                                                                            weight_manager);
        } else if (sd_version_is_minit2i(version)) {
            result.conditioner = std::make_shared<MiniT2IConditioner>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                      tensor_storage_map,
                                                                      weight_manager);
            result.diffusion   = std::make_shared<MiniT2I::MiniT2IRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                        tensor_storage_map,
                                                                        "model.diffusion_model.model.net",
                                                                        weight_manager);
        } else if (sd_version_is_sensenova_u1(version)) {
            result.conditioner = std::make_shared<SenseNovaU1Conditioner>();
            result.diffusion   = std::make_shared<SenseNovaU1::SenseNovaU1Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                                tensor_storage_map,
                                                                                "",
                                                                                weight_manager);
        } else if (sd_version_is_anima(version)) {
            result.conditioner = std::make_shared<AnimaConditioner>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    weight_manager);
            result.diffusion   = std::make_shared<Anima::AnimaRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model",
                                                                    weight_manager);
        } else if (sd_version_is_z_image(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager);
            result.diffusion   = std::make_shared<ZImage::ZImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                      tensor_storage_map,
                                                                      "model.diffusion_model",
                                                                      version,
                                                                      weight_manager);
        } else if (sd_version_is_boogu_image(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               true,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Boogu::BooguImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         version,
                                                                         weight_manager);
        } else if (sd_version_is_ernie_image(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager);
            result.diffusion   = std::make_shared<ErnieImage::ErnieImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                              tensor_storage_map,
                                                                              "model.diffusion_model",
                                                                              weight_manager);
        } else if (sd_version_is_lens(version)) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager);
            result.diffusion   = std::make_shared<Lens::LensRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                  tensor_storage_map,
                                                                  "model.diffusion_model",
                                                                  weight_manager);
        } else {  // SD1.x SD2.x SDXL
            std::map<std::string, std::string> embbeding_map;
            for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
            }
            result.conditioner = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                                     tensor_storage_map,
                                                                                     embbeding_map,
                                                                                     version,
                                                                                     weight_manager);
            result.diffusion   = std::make_shared<UNetModelRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                 tensor_storage_map,
                                                                 "model.diffusion_model",
                                                                 version,
                                                                 weight_manager);
            if (sd_ctx_params->diffusion_conv_direct) {
                LOG_INFO("Using Conv2d direct in the diffusion model");
                result.diffusion->set_conv2d_direct_enabled(true);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->ip_adapter_path)) > 0 && result.clip_vision == nullptr) {
            if (!ensure_backend_pair(ctx.backends, SDBackendModule::CLIP_VISION)) {
                return false;
            }
            result.clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(ctx.backends.runtime_backend(SDBackendModule::CLIP_VISION),
                                                                            tensor_storage_map,
                                                                            weight_manager);
        }

        if (strlen(SAFE_STR(sd_ctx_params->ip_adapter_path)) > 0) {
            result.ip_adapter = std::make_shared<IPAdapter::IPAdapterRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                             tensor_storage_map,
                                                                             "ip_adapter",
                                                                             weight_manager);
        }
        if (result.conditioner) {
            result.conditioner->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.diffusion) {
            result.diffusion->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.high_noise_diffusion) {
            result.high_noise_diffusion->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.clip_vision) {
            result.clip_vision->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.ip_adapter) {
            result.ip_adapter->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        runners = std::move(result);
        return true;
    }

    bool build_vae_runners(const Context& ctx, const VAEOptions& options, VAERunners& runners) {
        const auto* sd_ctx_params      = &ctx.params;
        const auto& tensor_storage_map = ctx.tensor_storage_map;
        const auto version             = ctx.version;
        const auto& weight_manager     = ctx.weight_manager;
        VAERunners result;
        if (!ensure_backend_pair(ctx.backends, SDBackendModule::VAE)) {
            return false;
        }

        auto create_tae = [&](bool decode_only) -> std::shared_ptr<VAE> {
            if (sd_version_uses_wan_vae(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version) || sd_version_is_minimax_h3(version)) {
                return std::make_shared<TinyVideoAutoEncoder>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                              tensor_storage_map,
                                                              "decoder",
                                                              decode_only,
                                                              version,
                                                              weight_manager);

            } else {
                auto model = std::make_shared<TinyImageAutoEncoder>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                    tensor_storage_map,
                                                                    "decoder.layers",
                                                                    decode_only,
                                                                    version,
                                                                    weight_manager);
                return model;
            }
        };

        sd_vae_format_t vae_format = sd_ctx_params->vae_format;
        if (vae_format < SD_VAE_FORMAT_AUTO || vae_format >= SD_VAE_FORMAT_COUNT) {
            LOG_WARN("invalid VAE format override, using auto");
            vae_format = SD_VAE_FORMAT_AUTO;
        }
        SDVersion vae_version = version;
        if (sd_version_is_pid(version) && vae_format != SD_VAE_FORMAT_AUTO) {
            vae_version = sd_vae_format_to_version(vae_format, vae_version);
        }

        auto create_vae = [&]() -> std::shared_ptr<VAE> {
            if (sd_version_is_ltxav(version)) {
                return std::make_shared<LTXVideoVAE>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                     tensor_storage_map,
                                                     "first_stage_model",
                                                     false,
                                                     version,
                                                     weight_manager);
            } else if (sd_version_is_minimax_h3(version)) {
                return std::make_shared<MiniMaxH3VAE::MiniMaxH3VideoVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                               tensor_storage_map,
                                                                               "first_stage_model",
                                                                               weight_manager);
            } else if (sd_version_is_mage_flow(vae_version)) {
                return std::make_shared<MageVAE::MageVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                tensor_storage_map,
                                                                "first_stage_model",
                                                                weight_manager);
            } else if (sd_version_uses_hunyuan_video_vae(vae_version)) {
                return std::make_shared<Hunyuan::HunyuanVideoVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                        tensor_storage_map,
                                                                        "first_stage_model",
                                                                        false,
                                                                        vae_version,
                                                                        weight_manager);
            } else if (sd_version_uses_wan_vae(vae_version)) {
                return std::make_shared<WAN::WanVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                           tensor_storage_map,
                                                           "first_stage_model",
                                                           false,
                                                           vae_version,
                                                           weight_manager);
            } else {
                auto model = std::make_shared<AutoEncoderKL>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                             tensor_storage_map,
                                                             "first_stage_model",
                                                             false,
                                                             false,
                                                             vae_version,
                                                             weight_manager);
                if (sd_version_is_sdxl(version) &&
                    (strlen(SAFE_STR(sd_ctx_params->vae_path)) == 0 || sd_ctx_params->force_sdxl_vae_conv_scale || options.external_vae_is_invalid)) {
                    float vae_conv_2d_scale = 1.f / 32.f;
                    LOG_WARN(
                        "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, "
                        "using Conv2D scale %.3f",
                        vae_conv_2d_scale);
                    model->set_conv2d_scale(vae_conv_2d_scale);
                }
                return model;
            }
        };

        if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1 || sd_version_is_minit2i(version) || sd_version_is_sensenova_u1(version)) {
            LOG_INFO("using FakeVAE");
            result.vae = std::make_shared<FakeVAE>(version,
                                                   ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                   weight_manager);
        } else if (options.use_tae && !options.tae_preview_only) {
            LOG_INFO("using TAE for encoding / decoding");
            result.vae = create_tae(false);
        } else {
            LOG_INFO("using VAE for encoding / decoding");
            result.vae = create_vae();
            if (options.use_tae && options.tae_preview_only) {
                LOG_INFO("using TAE for preview");
                result.preview = create_tae(true);
            }
        }

        if (options.use_audio_vae) {
            if (sd_version_is_minimax_h3(version)) {
                result.audio = std::make_shared<MiniMaxH3::AudioVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                           tensor_storage_map,
                                                                           "",
                                                                           weight_manager);
            } else {
                result.audio = std::make_shared<LTXV::LTXAudioVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                         tensor_storage_map,
                                                                         "",
                                                                         weight_manager);
            }
        }

        if (sd_ctx_params->vae_conv_direct) {
            LOG_INFO("Using Conv2d direct in the vae model");
            result.vae->set_conv2d_direct_enabled(true);
            if (result.preview) {
                result.preview->set_conv2d_direct_enabled(true);
            }
        }
        if (result.vae) {
            result.vae->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.preview) {
            result.preview->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.audio) {
            result.audio->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        runners = std::move(result);
        return true;
    }

    bool build_control_net_runner(const Context& ctx, std::shared_ptr<ControlNet>& runner) {
        const auto* sd_ctx_params      = &ctx.params;
        const auto& tensor_storage_map = ctx.tensor_storage_map;
        const auto version             = ctx.version;
        const auto& weight_manager     = ctx.weight_manager;
        if (!ensure_backend_pair(ctx.backends, SDBackendModule::CONTROL_NET)) {
            return false;
        }
        auto control_net = std::make_shared<ControlNet>(ctx.backends.runtime_backend(SDBackendModule::CONTROL_NET),
                                                        tensor_storage_map,
                                                        version,
                                                        "",
                                                        weight_manager);
        if (sd_ctx_params->diffusion_conv_direct) {
            LOG_INFO("Using Conv2d direct in the control net");
            control_net->set_conv2d_direct_enabled(true);
        }
        control_net->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        runner = std::move(control_net);
        return true;
    }

    bool build_extension_runners(const GenerationExtensionInitContext& ctx,
                                 std::vector<std::shared_ptr<GenerationExtension>>& extensions) {
        std::vector<std::shared_ptr<GenerationExtension>> result;
        for (auto extension : {create_photomaker_extension(), create_pulid_extension()}) {
            if (!extension->init(ctx)) {
                return false;
            }
            if (extension->is_enabled()) {
                result.push_back(std::move(extension));
            }
        }
        extensions = std::move(result);
        return true;
    }

}  // namespace sd::model_builders
