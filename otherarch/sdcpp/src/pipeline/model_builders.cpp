#include "model_builders.h"

#include <cmath>
#include <cstring>
#include <utility>

#include "conditioning/conditioner.hpp"
#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "extensions/generation_extension.h"
#include "model/adapter/ip_adapter.hpp"
#include "model/audio/wav2vec2.hpp"
#include "model/diffusion/control.hpp"
#include "model/diffusion/model.hpp"
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
#include "model_definition.h"

namespace sd::model_builders {

    bool ensure_backend_pair(SDBackendManager& backends, SDBackendModule module) {
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

    bool build_core_runners(const Context& ctx, CoreRunners& runners) try {
        const auto* sd_ctx_params      = &ctx.params;
        const auto& tensor_storage_map = ctx.tensor_storage_map;
        const auto version             = ctx.version;
        const auto& weight_manager     = ctx.weight_manager;
        CoreRunners result;
        TokenizerConfig tokenizers(sd_ctx_params->tokenizer);
        if (!ensure_backend_pair(ctx.backends, SDBackendModule::TE) ||
            !ensure_backend_pair(ctx.backends, SDBackendModule::DIFFUSION)) {
            return false;
        }

        const auto* definition = find_model_definition(version);
        if (!definition || !definition->build || !definition->build(ctx, tokenizers, result)) {
            LOG_ERROR("failed to build model runners for %s", sd_version_name(version));
            return false;
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
        tokenizers.validate_usage();
        runners = std::move(result);
        return true;
    } catch (const std::exception& error) {
        LOG_ERROR("failed to build model runners: %s", error.what());
        return false;
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
            if (find_model_traits(version)->video_tae) {
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
        if (find_model_traits(version)->vae_format_override && vae_format != SD_VAE_FORMAT_AUTO) {
            vae_version = sd_vae_format_to_version(vae_format, vae_version);
        }

        auto create_vae = [&]() -> std::shared_ptr<VAE> {
            if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::LTXAV) {
                return std::make_shared<LTXVideoVAE>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                     tensor_storage_map,
                                                     "first_stage_model",
                                                     false,
                                                     version,
                                                     weight_manager);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::MINIMAX_H3) {
                return std::make_shared<MiniMaxH3VAE::MiniMaxH3VideoVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                               tensor_storage_map,
                                                                               "first_stage_model",
                                                                               weight_manager);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::MAGE) {
                return std::make_shared<MageVAE::MageVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                tensor_storage_map,
                                                                "first_stage_model",
                                                                weight_manager);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::HUNYUAN) {
                return std::make_shared<Hunyuan::HunyuanVideoVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                        tensor_storage_map,
                                                                        "first_stage_model",
                                                                        false,
                                                                        vae_version,
                                                                        weight_manager);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::WAN) {
                return std::make_shared<WAN::WanVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                           tensor_storage_map,
                                                           "first_stage_model",
                                                           false,
                                                           vae_version,
                                                           weight_manager);
            } else {
                auto model             = std::make_shared<AutoEncoderKL>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                             tensor_storage_map,
                                                             "first_stage_model",
                                                             false,
                                                             false,
                                                             vae_version,
                                                             weight_manager);
                const auto* definition = find_model_definition(version);
                if (definition && definition->configure_vae) {
                    definition->configure_vae(ctx, options, *model);
                }
                return model;
            }
        };

        if (find_model_traits(version)->fake_vae) {
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
            if (find_model_traits(version)->vae_kind == ModelVAEKind::MINIMAX_H3) {
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
