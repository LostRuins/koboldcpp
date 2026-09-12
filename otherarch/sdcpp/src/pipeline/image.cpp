#include "generation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>

#include "core/rng.hpp"
#include "diffusion_engine.h"
#include "model/vae/vae.hpp"
#include "request.h"
#include "runtime/denoiser.hpp"
#include "upscaler.h"

namespace sd::pipeline {

    struct CircularAxesState {
        bool circular_x = false;
        bool circular_y = false;
    };

    static CircularAxesState configure_image_vae_axes(StableDiffusionGGML* sd,
                                                      const sd_img_gen_params_t* sd_img_gen_params,
                                                      const GenerationRequest& request) {
        CircularAxesState original_axes = {sd->circular_x, sd->circular_y};

        if (!sd_img_gen_params->vae_tiling_params.enabled) {
            if (sd->first_stage_model) {
                sd->first_stage_model->set_circular_axes(sd->circular_x, sd->circular_y);
            }
            if (sd->preview_vae) {
                sd->preview_vae->set_circular_axes(sd->circular_x, sd->circular_y);
            }
            return original_axes;
        }

        int tile_size_x, tile_size_y;
        float overlap;
        int latent_size_x = request.width / request.vae_scale_factor;
        int latent_size_y = request.height / request.vae_scale_factor;
        sd->first_stage_model->get_tile_sizes(tile_size_x,
                                              tile_size_y,
                                              overlap,
                                              sd_img_gen_params->vae_tiling_params,
                                              latent_size_x,
                                              latent_size_y);

        sd->circular_x = sd->circular_x && (tile_size_x >= latent_size_x);
        sd->circular_y = sd->circular_y && (tile_size_y >= latent_size_y);

        if (sd->first_stage_model) {
            sd->first_stage_model->set_circular_axes(sd->circular_x, sd->circular_y);
        }
        if (sd->preview_vae) {
            sd->preview_vae->set_circular_axes(sd->circular_x, sd->circular_y);
        }

        sd->circular_x = original_axes.circular_x && (tile_size_x < latent_size_x);
        sd->circular_y = original_axes.circular_y && (tile_size_y < latent_size_y);

        return original_axes;
    }

    static void restore_image_vae_axes(StableDiffusionGGML* sd, const CircularAxesState& original_axes) {
        sd->circular_x = original_axes.circular_x;
        sd->circular_y = original_axes.circular_y;
    }

    class ImageVaeAxesGuard {
    private:
        StableDiffusionGGML* sd = nullptr;
        CircularAxesState original_axes;

    public:
        ImageVaeAxesGuard(StableDiffusionGGML* sd,
                          const sd_img_gen_params_t* sd_img_gen_params,
                          const GenerationRequest& request)
            : sd(sd),
              original_axes(configure_image_vae_axes(sd, sd_img_gen_params, request)) {}

        ~ImageVaeAxesGuard() {
            restore_image_vae_axes(sd, original_axes);
        }

        ImageVaeAxesGuard(const ImageVaeAxesGuard&)            = delete;
        ImageVaeAxesGuard& operator=(const ImageVaeAxesGuard&) = delete;
    };

    sd::Tensor<float> ensure_image_tensor_channels(sd::Tensor<float> image, int channels) {
        if (image.empty()) {
            return image;
        }
        GGML_ASSERT(image.dim() == 4);
        int64_t current_channels = image.shape()[2];
        if (current_channels == channels) {
            return image;
        }
        if (channels == 4) {
            sd::Tensor<float> alpha = sd::full<float>({image.shape()[0], image.shape()[1], 1, image.shape()[3]}, 1.f);
            if (current_channels == 3) {
                return sd::ops::concat(image, alpha, 2);
            }
            if (current_channels == 1) {
                sd::Tensor<float> rgb = sd::ops::concat(image, image, 2);
                rgb                   = sd::ops::concat(rgb, image, 2);
                return sd::ops::concat(rgb, alpha, 2);
            }
        }
        if (channels == 3 && current_channels >= 3) {
            return sd::ops::slice(image, 2, 0, 3);
        }
        GGML_ABORT("cannot convert image tensor from %lld to %d channels",
                   (long long)current_channels,
                   channels);
    }

    static std::optional<ImageGenerationLatents> prepare_image_generation_latents(StableDiffusionGGML* sd,
                                                                                  const sd_img_gen_params_t* sd_img_gen_params,
                                                                                  GenerationRequest* request,
                                                                                  SamplePlan* plan,
                                                                                  const RefImageParams& ref_image_params) {
        int64_t prepare_start_ms = ggml_time_ms();

        sd::Tensor<float> init_image_tensor;
        sd::Tensor<float> control_image_tensor;
        sd::Tensor<float> mask_image_tensor;
        int image_channels = sd->get_image_channels();

        if (sd_img_gen_params->init_image.data != nullptr) {
            LOG_INFO("IMG2IMG");

            if (request->strength < 1.f) {
                bool strength_as_noise_level = false;
                bool force_first_sigma       = false;
                for (const auto& [key, value] : parse_key_value_args(sd_img_gen_params->sample_params.extra_sample_args, "img2img arg")) {
                    if (key == "strength_as_noise_level") {
                        if (!parse_strict_bool(value, strength_as_noise_level)) {
                            LOG_WARN("ignoring invalid img2img sample arg '%s=%s'", key.c_str(), value.c_str());
                        }
                    } else if (key == "force_first_sigma") {
                        if (!parse_strict_bool(value, force_first_sigma)) {
                            LOG_WARN("ignoring invalid img2img sample arg '%s=%s'", key.c_str(), value.c_str());
                        }
                    }
                }

                size_t t_enc;
                float target_sigma = -1;
                if (!strength_as_noise_level) {
                    t_enc = static_cast<size_t>(plan->sample_steps * request->strength);
                    if (t_enc == static_cast<size_t>(plan->sample_steps)) {
                        t_enc--;
                    }
                } else {
                    LOG_VERBOSE("Interpreting denoise strength as relative noise level");
                    // assume x_noised = K * (x * (1-noise_level) + noise * noise_level) = K * lerp(x, noise, noise_level)
                    // K = 1, noise_level = sigma for flow models
                    // K = 1+sigma, noise_level=sigma/(1+sigma) for diffusion models
                    float target_noise_level = request->strength;
                    target_sigma             = sd->denoiser->noise_level_to_sigma(target_noise_level);
                    size_t start_index       = 0;
                    for (size_t i = 0; i < plan->sigmas.size(); ++i) {
                        if (plan->sigmas[i] <= target_sigma) {
                            start_index = i;
                            break;
                        }
                    }

                    if (start_index >= plan->sigmas.size() - 1) {
                        start_index = plan->sigmas.size() - 2;  // Leave at least 1 step
                    }
                    t_enc = plan->sample_steps - start_index - 1;
                }
                LOG_INFO("target t_enc is %zu steps", t_enc);
                std::vector<float> sigma_sched;
                sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());

                if (target_sigma > 0 && force_first_sigma && strength_as_noise_level) {
                    LOG_VERBOSE("force_first_sigma to %.4f (from %.4f)", target_sigma, sigma_sched[0]);
                    sigma_sched[0] = target_sigma;
                }

                plan->sigmas       = std::move(sigma_sched);
                plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
            }

            init_image_tensor = ensure_image_tensor_channels(sd_image_to_tensor(sd_img_gen_params->init_image, request->width, request->height),
                                                             image_channels);
        }

        if (sd_img_gen_params->mask_image.data != nullptr) {
            mask_image_tensor = sd_image_to_tensor(sd_img_gen_params->mask_image, request->width, request->height);
            mask_image_tensor = sd::ops::round(mask_image_tensor);
        }

        if (sd_img_gen_params->control_image.data != nullptr) {
            control_image_tensor = sd_image_to_tensor(sd_img_gen_params->control_image, request->width, request->height);
        }

        if (init_image_tensor.empty() || mask_image_tensor.empty()) {
            if (sd_version_is_inpaint(sd->version)) {
                LOG_WARN("inpainting model requires both an init image and a mask image.");
            }
        }

        if (mask_image_tensor.empty()) {
            mask_image_tensor = sd::full<float>({request->width, request->height, 1, 1}, 1.f);
        }

        sd::Tensor<float> latent_mask = sd::ops::interpolate(mask_image_tensor,
                                                             {request->width / request->vae_scale_factor,
                                                              request->height / request->vae_scale_factor,
                                                              1,
                                                              1},
                                                             sd::ops::InterpolateMode::NearestMax);

        sd::Tensor<float> init_latent;
        sd::Tensor<float> control_latent;
        if (init_image_tensor.empty()) {
            if (sd->version == VERSION_QWEN_IMAGE_LAYERED) {
                init_latent = sd->generate_init_latent(request->width, request->height, request->qwen_image_layers + 1, true);
            } else {
                init_latent = sd->generate_init_latent(request->width, request->height);
            }
        } else {
            init_latent = sd->encode_first_stage(init_image_tensor);
            if (init_latent.empty()) {
                LOG_ERROR("failed to encode init image");
                return std::nullopt;
            }
        }

        if (sd->animatediff_num_frames > 1 &&
            init_latent.dim() >= 4 && init_latent.shape()[3] == 1) {
            int n_frames = sd->animatediff_num_frames;
            std::vector<int64_t> shape(init_latent.shape().begin(), init_latent.shape().end());
            shape[3] = n_frames;
            if (!init_image_tensor.empty()) {
                sd::Tensor<float> replicated(shape);
                for (int f = 0; f < n_frames; ++f) {
                    sd::ops::slice_assign(&replicated, 3, f, f + 1, init_latent);
                }
                init_latent = std::move(replicated);
            } else {
                init_latent = sd::Tensor<float>(std::move(shape));
            }
        }

        if (!control_image_tensor.empty()) {
            control_latent = sd->encode_first_stage(control_image_tensor);
            if (control_latent.empty()) {
                LOG_ERROR("failed to encode control image");
                return std::nullopt;
            }
        }

        std::vector<sd::Tensor<float>> ref_images;
        for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
            ref_images.push_back(ensure_image_tensor_channels(sd_image_to_tensor(sd_img_gen_params->ref_images[i]),
                                                              image_channels));
        }

        if (ref_images.empty() && sd_version_is_unet_edit(sd->version)) {
            LOG_WARN("This model needs at least one reference image; using an empty reference");
            ref_images.push_back(sd::zeros<float>({request->width, request->height, image_channels, 1}));
            request->guidance.img_cfg = request->guidance.txt_cfg;
            request->use_img_uncond   = false;
        }

        if (!ref_images.empty()) {
            LOG_INFO("EDIT mode");
        }

        std::vector<sd::Tensor<float>> ref_latents;
        for (size_t i = 0; i < ref_images.size(); i++) {
            if (sd->version == VERSION_HIDREAM_O1) {
                continue;
            }
            sd::Tensor<float> ref_latent;
            if (ref_image_params.resize_before_vae && !sd_version_is_pid(sd->version)) {
                LOG_VERBOSE("auto resize ref images");
                double vae_width;
                double vae_height;
                if (ref_image_params.resize_vae_to_target) {
                    vae_width  = request->width;
                    vae_height = request->height;
                } else {
                    int target_pixels  = ref_image_params.vae_input_max_pixels > 0 ? ref_image_params.vae_input_max_pixels : 1024 * 1024;
                    int vae_image_size = std::min(target_pixels, request->width * request->height);
                    vae_width          = sqrt(vae_image_size * ref_images[i].shape()[0] / ref_images[i].shape()[1]);
                    vae_height         = vae_width * ref_images[i].shape()[1] / ref_images[i].shape()[0];
                }

                int factor = sd_version_is_qwen_image(sd->version) ? 32 : 16;
                vae_height = round(vae_height / factor) * factor;
                vae_width  = round(vae_width / factor) * factor;

                auto resized_ref_img = sd::ops::interpolate(ref_images[i],
                                                            {static_cast<int>(vae_width),
                                                             static_cast<int>(vae_height),
                                                             ref_images[i].shape()[2],
                                                             ref_images[i].shape()[3]});

                LOG_VERBOSE("resize vae ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64,
                            static_cast<int>(i),
                            ref_images[i].shape()[1],
                            ref_images[i].shape()[0],
                            resized_ref_img.shape()[1],
                            resized_ref_img.shape()[0]);

                ref_latent = sd->encode_first_stage(resized_ref_img);
            } else {
                ref_latent = sd->encode_first_stage(ref_images[i]);
            }
            if (ref_latent.empty()) {
                LOG_ERROR("failed to encode reference image %d", static_cast<int>(i));
                return std::nullopt;
            }

            ref_latents.push_back(std::move(ref_latent));
        }

        if (sd_version_is_pid(sd->version)) {
            if (ref_latents.empty()) {
                LOG_ERROR("PiD requires a reference image");
                return std::nullopt;
            }
        }

        sd::Tensor<float> concat_latent;
        sd::Tensor<float> img_uncond_concat_latent;
        if (sd_version_is_inpaint(sd->version)) {
            sd::Tensor<float> masked_init_latent;

            if (sd->version != VERSION_FLEX_2) {
                if (!init_image_tensor.empty()) {
                    auto masked_image  = ((1.0f - mask_image_tensor) * (init_image_tensor - 0.5f)) + 0.5f;
                    masked_init_latent = sd->encode_first_stage(masked_image);
                    if (masked_init_latent.empty()) {
                        LOG_ERROR("failed to encode masked init image");
                        return std::nullopt;
                    }
                } else {
                    masked_init_latent = sd::Tensor<float>::zeros_like(init_latent);
                }
            } else {
                masked_init_latent = ((1.0f - latent_mask) * init_latent);
            }

            auto uncond_masked_init_latent = sd::Tensor<float>::zeros_like(masked_init_latent);

            if (sd->version == VERSION_FLUX_FILL) {
                auto mask = mask_image_tensor.reshape({request->vae_scale_factor,
                                                       request->width / request->vae_scale_factor,
                                                       request->vae_scale_factor,
                                                       request->height / request->vae_scale_factor});
                mask      = mask.permute({1, 3, 0, 2}).reshape({request->width / request->vae_scale_factor, request->height / request->vae_scale_factor, request->vae_scale_factor * request->vae_scale_factor, 1});

                concat_latent            = sd::ops::concat(masked_init_latent, mask, 2);
                img_uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, mask, 2);
            } else if (sd->version == VERSION_FLEX_2) {
                concat_latent = sd::ops::concat(masked_init_latent, latent_mask, 2);
                if (!control_latent.empty()) {
                    concat_latent = sd::ops::concat(concat_latent, control_latent, 2);
                } else {
                    concat_latent = sd::ops::concat(concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
                }

                img_uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, latent_mask, 2);
                img_uncond_concat_latent = sd::ops::concat(img_uncond_concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
            } else {  // SD1.x SD2.x SDXL inpaint
                concat_latent            = sd::ops::concat(latent_mask, masked_init_latent, 2);
                img_uncond_concat_latent = sd::ops::concat(latent_mask, uncond_masked_init_latent, 2);
            }
        }
        if (sd_version_is_unet_edit(sd->version)) {
            concat_latent            = sd::ops::interpolate<float>(ref_latents[0], init_latent.shape());
            img_uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
        }
        if (sd->version == VERSION_FLUX_CONTROLS) {
            if (!control_latent.empty()) {
                concat_latent = control_latent;
            } else {
                concat_latent = sd::Tensor<float>::zeros_like(init_latent);
            }
            img_uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
        }

        if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0) {
            int64_t t1 = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);
        }

        ImageGenerationLatents latents;
        latents.init_latent              = std::move(init_latent);
        latents.concat_latent            = std::move(concat_latent);
        latents.img_uncond_concat_latent = std::move(img_uncond_concat_latent);
        latents.control_image            = std::move(control_image_tensor);
        latents.ref_images               = std::move(ref_images);
        latents.ref_latents              = std::move(ref_latents);

        if (sd_version_is_inpaint(sd->version)) {
            latent_mask = sd::ops::max_pool_2d(latent_mask,
                                               {3, 3},
                                               {1, 1},
                                               {1, 1});
        }
        latents.denoise_mask = std::move(latent_mask);

        return latents;
    }

    static std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds(StableDiffusionGGML* sd,
                                                                                const sd_img_gen_params_t* sd_img_gen_params,
                                                                                GenerationRequest* request,
                                                                                SamplePlan* plan,
                                                                                ImageGenerationLatents* latents,
                                                                                const RefImageParams& ref_image_params) {
        ConditionerRunnerEndOnExit conditioner_runner_end{sd->cond_stage_model.get()};

        ConditionerParams condition_params;
        condition_params.text      = request->prompt;
        condition_params.clip_skip = request->clip_skip;
        condition_params.width     = request->width;
        condition_params.height    = request->height;
        if (ref_image_params.pass_to_vlm) {
            condition_params.ref_images = &latents->ref_images;
        }

        condition_params.ref_image_params = ref_image_params;

        sd->prepare_generation_extensions(request->pm_params,
                                          request->pulid_params,
                                          condition_params,
                                          plan->total_steps);
        sd->compute_ip_adapter_tokens(sd_img_gen_params->ip_adapter_image, sd_img_gen_params->ip_adapter_strength);
        int64_t prepare_start_ms         = ggml_time_ms();
        condition_params.zero_out_masked = false;
        auto cond                        = sd->cond_stage_model->get_learned_condition(sd->n_threads,
                                                                                       condition_params);
        if (cond.c_concat.empty() && ref_image_params.pass_to_dit) {
            cond.c_concat = latents->concat_latent;  // TODO: optimize
        }

        bool use_ref_latent_img_cfg = request->use_img_uncond &&
                                      !latents->ref_images.empty() &&
                                      sd_version_supports_ref_latent_img_cfg(sd->version);

        SDCondition uncond;
        if (request->use_uncond || request->use_high_noise_uncond) {
            if (sd_version_is_ideogram4(sd->version)) {
                uncond.c_vector = sd::Tensor<float>::from_vector({1.0f});
            } else if (sd_version_is_minit2i(sd->version)) {
                // MiniT2I derives the unconditional signal from the same T5 hidden
                // states with a zeroed prompt mask, so no extra text encode is needed.
                uncond.c_crossattn = cond.c_crossattn;
                uncond.c_vector    = sd::Tensor<float>::zeros_like(cond.c_vector);
            } else if (sd_version_is_sensenova_u1(sd->version)) {
                auto* sensenova_conditioner = static_cast<SenseNovaU1Conditioner*>(sd->cond_stage_model.get());
                uncond                      = sensenova_conditioner->get_unconditional_condition(request->negative_prompt);
            } else {
                bool zero_out_masked = false;
                if (sd_version_is_sdxl(sd->version) &&
                    request->negative_prompt.empty() &&
                    !sd->is_using_edm_v_parameterization) {
                    zero_out_masked = true;
                }
                condition_params.text            = request->negative_prompt;
                condition_params.zero_out_masked = zero_out_masked;
                uncond                           = sd->cond_stage_model->get_learned_condition(sd->n_threads,
                                                                                               condition_params);
            }
            if (uncond.c_concat.empty() && ref_image_params.pass_to_dit) {
                uncond.c_concat = latents->concat_latent;  // TODO: optimize
            }
        }

        SDCondition img_uncond;
        if (request->use_img_uncond) {
            if ((request->use_uncond || request->use_high_noise_uncond) && (latents->ref_images.empty() || !use_ref_latent_img_cfg)) {
                img_uncond = SDCondition(uncond.c_crossattn, uncond.c_vector, latents->img_uncond_concat_latent);
            } else {
                bool zero_out_masked = false;
                if (sd_version_is_sdxl(sd->version) &&
                    request->negative_prompt.empty() &&
                    !sd->is_using_edm_v_parameterization) {
                    zero_out_masked = true;
                }
                condition_params.text            = request->negative_prompt;
                condition_params.zero_out_masked = zero_out_masked;
                std::vector<sd::Tensor<float>> empty_ref_images;
                if (use_ref_latent_img_cfg) {
                    condition_params.ref_images = &empty_ref_images;
                }
                img_uncond = sd->cond_stage_model->get_learned_condition(sd->n_threads,
                                                                         condition_params);
                if (img_uncond.c_concat.empty() && ref_image_params.pass_to_dit) {
                    img_uncond.c_concat = latents->img_uncond_concat_latent;  // TODO: optimize
                }
            }
        }

        int64_t t1 = ggml_time_ms();
        LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

        ImageGenerationEmbeds embeds;
        embeds.img_uncond = std::move(img_uncond);
        embeds.cond       = std::move(cond);
        embeds.uncond     = std::move(uncond);

        return embeds;
    }

    static sd_image_t* decode_image_outputs(StableDiffusionGGML* sd,
                                            const GenerationRequest& request,
                                            const std::vector<sd::Tensor<float>>& final_latents,
                                            int* num_images_out) {
        if (final_latents.empty()) {
            LOG_ERROR("no latent images to decode");
            return nullptr;
        }
        if (final_latents.size() > static_cast<size_t>(request.batch_count)) {
            LOG_ERROR("expected at most %d latents, got %zu", request.batch_count, final_latents.size());
            return nullptr;
        }
        if (final_latents.size() < static_cast<size_t>(request.batch_count)) {
            LOG_INFO("decoding %zu/%d latents", final_latents.size(), request.batch_count);
        } else {
            LOG_INFO("decoding %zu latents", final_latents.size());
        }
        std::vector<sd::Tensor<float>> decoded_images;
        int64_t t0     = ggml_time_ms();
        bool cancelled = false;

        for (size_t i = 0; i < final_latents.size(); i++) {
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling latent decodings");
                cancelled = true;
                break;
            }
            int64_t t1 = ggml_time_ms();
            if (sd->version == VERSION_QWEN_IMAGE_LAYERED) {
                int qwen_image_latent_layers = request.qwen_image_layers + 1;
                if (final_latents[i].dim() < 5 || final_latents[i].shape()[2] < qwen_image_latent_layers) {
                    LOG_ERROR("qwen image layered expected at least %d latent layers, got shape dim=%d",
                              qwen_image_latent_layers,
                              final_latents[i].dim());
                    return nullptr;
                }
                for (int layer_index = 0; layer_index < qwen_image_latent_layers; layer_index++) {
                    if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                        LOG_ERROR("cancelling latent decodings");
                        cancelled = true;
                        break;
                    }
                    sd::Tensor<float> layer_latent = sd::ops::slice(final_latents[i], 2, layer_index, layer_index + 1);
                    layer_latent.squeeze_(2);
                    sd::Tensor<float> image = sd->decode_first_stage(layer_latent);
                    if (image.empty()) {
                        LOG_ERROR("decode_first_stage failed for latent %zu layer %d", i + 1, layer_index + 1);
                        return nullptr;
                    }
                    decoded_images.push_back(std::move(image));
                }
                if (cancelled) {
                    break;
                }
            } else if (sd->animatediff_num_frames > 1 &&
                       final_latents[i].dim() >= 4 &&
                       final_latents[i].shape()[3] == sd->animatediff_num_frames) {
                int n_frames = sd->animatediff_num_frames;
                for (int f = 0; f < n_frames; ++f) {
                    if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                        LOG_ERROR("cancelling latent decodings");
                        cancelled = true;
                        break;
                    }
                    sd::Tensor<float> frame_latent = sd::ops::slice(final_latents[i], 3, f, f + 1);
                    sd::Tensor<float> image        = sd->decode_first_stage(frame_latent);
                    if (image.empty()) {
                        LOG_ERROR("decode_first_stage failed for AnimateDiff frame %d/%d", f + 1, n_frames);
                        return nullptr;
                    }
                    decoded_images.push_back(std::move(image));
                }
            } else {
                sd::Tensor<float> image = sd->decode_first_stage(final_latents[i]);
                if (image.empty()) {
                    LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
                    return nullptr;
                }
                decoded_images.push_back(std::move(image));
            }
            int64_t t2 = ggml_time_ms();
            LOG_INFO("latent %zu decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
        }

        int64_t t4 = ggml_time_ms();
        LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t0) * 1.0f / 1000);
        if (decoded_images.empty()) {
            LOG_ERROR(cancelled ? "cancelled before any latent images were decoded" : "no decoded images");
            return nullptr;
        }

        int image_count           = static_cast<int>(decoded_images.size());
        sd_image_t* result_images = (sd_image_t*)calloc(image_count, sizeof(sd_image_t));
        if (result_images == nullptr) {
            return nullptr;
        }
        if (num_images_out != nullptr) {
            *num_images_out = image_count;
        }

        for (size_t i = 0; i < decoded_images.size(); i++) {
            result_images[i] = tensor_to_sd_image(decoded_images[i]);
        }

        return result_images;
    }

    static sd::Tensor<float> upscale_hires_latent(StableDiffusionGGML* sd,
                                                  const sd::Tensor<float>& latent,
                                                  const GenerationRequest& request,
                                                  UpscalerGGML* upscaler) {
        if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling hires latent upscale");
            return {};
        }

        auto get_hires_latent_target_shape = [&]() {
            std::vector<int64_t> target_shape = latent.shape();
            if (target_shape.size() < 2) {
                target_shape.clear();
                return target_shape;
            }
            target_shape[0] = request.hires.target_width / request.vae_scale_factor;
            target_shape[1] = request.hires.target_height / request.vae_scale_factor;
            return target_shape;
        };

        if (request.hires.upscaler == SD_HIRES_UPSCALER_LATENT ||
            request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST ||
            request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT ||
            request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_ANTIALIASED ||
            request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC ||
            request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED) {
            std::vector<int64_t> target_shape = get_hires_latent_target_shape();
            if (target_shape.empty()) {
                LOG_ERROR("latent has invalid shape for hires upscale");
                return {};
            }

            sd::ops::InterpolateMode mode = sd::ops::InterpolateMode::Nearest;
            bool antialias                = false;
            switch (request.hires.upscaler) {
                case SD_HIRES_UPSCALER_LATENT:
                    mode = sd::ops::InterpolateMode::Bilinear;
                    break;
                case SD_HIRES_UPSCALER_LATENT_NEAREST:
                    mode = sd::ops::InterpolateMode::Nearest;
                    break;
                case SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT:
                    mode = sd::ops::InterpolateMode::NearestExact;
                    break;
                case SD_HIRES_UPSCALER_LATENT_ANTIALIASED:
                    mode      = sd::ops::InterpolateMode::Bilinear;
                    antialias = true;
                    break;
                case SD_HIRES_UPSCALER_LATENT_BICUBIC:
                    mode = sd::ops::InterpolateMode::Bicubic;
                    break;
                case SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED:
                    mode      = sd::ops::InterpolateMode::Bicubic;
                    antialias = true;
                    break;
                default:
                    break;
            }

            LOG_INFO("hires %s upscale %" PRId64 "x%" PRId64 " -> %" PRId64 "x%" PRId64,
                     sd_hires_upscaler_name(request.hires.upscaler),
                     latent.shape()[0],
                     latent.shape()[1],
                     target_shape[0],
                     target_shape[1]);

            return sd::ops::interpolate(latent, target_shape, mode, false, antialias);
        } else if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL ||
                   request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS ||
                   request.hires.upscaler == SD_HIRES_UPSCALER_NEAREST) {
            if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL && upscaler == nullptr) {
                LOG_ERROR("hires model upscaler context is null");
                return {};
            }

            sd::Tensor<float> decoded = sd->decode_first_stage(latent);
            if (decoded.empty()) {
                LOG_ERROR("decode_first_stage failed before hires %s upscale",
                          sd_hires_upscaler_name(request.hires.upscaler));
                return {};
            }
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling hires image upscale");
                return {};
            }

            sd::Tensor<float> upscaled_tensor;
            if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
                upscaled_tensor = upscaler->upscale_tensor(decoded);
                if (upscaled_tensor.empty()) {
                    LOG_ERROR("hires model upscale failed");
                    return {};
                }

                if (upscaled_tensor.shape()[0] != request.hires.target_width ||
                    upscaled_tensor.shape()[1] != request.hires.target_height) {
                    upscaled_tensor = sd::ops::interpolate(upscaled_tensor,
                                                           {request.hires.target_width,
                                                            request.hires.target_height,
                                                            upscaled_tensor.shape()[2],
                                                            upscaled_tensor.shape()[3]});
                }
            } else {
                sd::ops::InterpolateMode mode = request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS
                                                    ? sd::ops::InterpolateMode::Lanczos
                                                    : sd::ops::InterpolateMode::Nearest;
                LOG_INFO("hires %s image upscale %" PRId64 "x%" PRId64 " -> %dx%d",
                         sd_hires_upscaler_name(request.hires.upscaler),
                         decoded.shape()[0],
                         decoded.shape()[1],
                         request.hires.target_width,
                         request.hires.target_height);
                upscaled_tensor = sd::ops::interpolate(decoded,
                                                       {request.hires.target_width,
                                                        request.hires.target_height,
                                                        decoded.shape()[2],
                                                        decoded.shape()[3]},
                                                       mode);
                upscaled_tensor = sd::ops::clamp(upscaled_tensor, 0.0f, 1.0f);
            }

            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling hires latent encode");
                return {};
            }
            sd::Tensor<float> upscaled_latent = sd->encode_first_stage(upscaled_tensor);
            if (upscaled_latent.empty()) {
                LOG_ERROR("encode_first_stage failed after hires %s upscale",
                          sd_hires_upscaler_name(request.hires.upscaler));
            }
            return upscaled_latent;
        }

        LOG_ERROR("unsupported hires upscaler '%s'", sd_hires_upscaler_name(request.hires.upscaler));
        return {};
    }

    bool generate_image(StableDiffusionGGML* sd,
                        const sd_img_gen_params_t* sd_img_gen_params,
                        sd_image_t** images_out,
                        int* num_images_out) {
        if (images_out != nullptr) {
            *images_out = nullptr;
        }
        if (num_images_out != nullptr) {
            *num_images_out = 0;
        }
        if (sd == nullptr || sd_img_gen_params == nullptr) {
            return false;
        }

        // MiniMax-H3 is video-only. Its denoiser always splits the packed latent into a video and an
        // audio half, and only generate_video ever computes the audio length, so reaching this
        // function with an H3 checkpoint is guaranteed to die on
        // GGML_ASSERT(!audio_input_cache.empty()) with a core dump, after the several minutes it
        // takes to load the weights, and with nothing in the output pointing at the missing --mode.
        // (The AnimateDiff path below routes vid_gen back through here, but that is SD1.5 plus a
        // motion module, never H3.)
        if (sd_version_is_minimax_h3(sd->version)) {
            LOG_ERROR("MiniMax-H3 is a video model and cannot be run in img_gen mode; use --mode vid_gen");
            return false;
        }

        sd->reset_cancel_flag();

        int64_t t0            = ggml_time_ms();
        sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
        GenerationRequest request(sd, sd_img_gen_params);
        LOG_INFO("generate_image %dx%d", request.width, request.height);

        sd->rng->manual_seed(request.seed);
        sd->sampler_rng->manual_seed(request.seed);
        sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
        if (!sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count))
            return false;
        sd->apply_circular_axes(sd_img_gen_params->circular_x, sd_img_gen_params->circular_y);

        const RefImageParams ref_image_params = sd->resolve_ref_image_params(sd_img_gen_params->ref_image_args);

        ImageVaeAxesGuard axes_guard(sd, sd_img_gen_params, request);

        SamplePlan plan(sd, sd_img_gen_params, request);
        auto latents_opt = prepare_image_generation_latents(sd,
                                                            sd_img_gen_params,
                                                            &request,
                                                            &plan,
                                                            ref_image_params);
        if (!latents_opt.has_value()) {
            return false;
        }
        ImageGenerationLatents latents = std::move(*latents_opt);

        auto embeds_opt = prepare_image_generation_embeds(sd,
                                                          sd_img_gen_params,
                                                          &request,
                                                          &plan,
                                                          &latents,
                                                          ref_image_params);
        if (!embeds_opt.has_value()) {
            return false;
        }
        ImageGenerationEmbeds embeds = std::move(*embeds_opt);

        std::vector<sd::Tensor<float>> final_latents;
        int64_t denoise_start = ggml_time_ms();
        for (int b = 0; b < request.batch_count; b++) {
            sd_cancel_mode_t cancel = sd->get_cancel_flag();
            if (cancel == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation");
                return false;
            }
            if (cancel == SD_CANCEL_NEW_LATENTS) {
                LOG_INFO("cancelling new latent generation, returning %zu/%d completed latents",
                         final_latents.size(),
                         request.batch_count);
                break;
            }

            int64_t sampling_start = ggml_time_ms();
            int64_t cur_seed       = request.seed + b;
            LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

            sd->rng->manual_seed(cur_seed);
            sd->sampler_rng->manual_seed(cur_seed);
            sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd->rng);

            sd::Tensor<float> x_0 = sd->sample(sd->diffusion_model,
                                               true,
                                               latents.init_latent,
                                               std::move(noise),
                                               embeds.cond,
                                               embeds.uncond,
                                               embeds.img_uncond,
                                               latents.control_image,
                                               request.control_strength,
                                               request.guidance,
                                               plan.eta,
                                               request.shifted_timestep,
                                               plan.sample_method,
                                               sd->is_flow_denoiser(),
                                               plan.extra_sample_args,
                                               plan.sigmas,
                                               latents.ref_latents,
                                               ref_image_params,
                                               latents.denoise_mask,
                                               sd::Tensor<float>(),
                                               1.f,
                                               0,
                                               static_cast<float>(request.fps),
                                               request.cache_params,
                                               true);
            int64_t sampling_end  = ggml_time_ms();
            if (!x_0.empty()) {
                LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
                final_latents.push_back(std::move(x_0));
                continue;
            }

            LOG_ERROR("sampling for image %d/%d failed after %.2fs",
                      b + 1,
                      request.batch_count,
                      (sampling_end - sampling_start) * 1.0f / 1000);
            return false;
        }
        int64_t denoise_end = ggml_time_ms();
        LOG_INFO("generating %zu latent images completed, taking %.2fs",
                 final_latents.size(),
                 (denoise_end - denoise_start) * 1.0f / 1000);
        if (final_latents.empty()) {
            LOG_ERROR("no latent images generated");
            return false;
        }

        if (request.hires.enabled && request.hires.target_width > 0) {
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation before hires fix");
                return false;
            }
            LOG_INFO("hires fix: upscaling to %dx%d", request.hires.target_width, request.hires.target_height);

            std::unique_ptr<UpscalerGGML> hires_upscaler;
            if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
                if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                    LOG_ERROR("cancelling generation before hires model load");
                    return false;
                }
                LOG_INFO("hires fix: loading model upscaler from '%s'", request.hires.model_path);
                hires_upscaler                    = std::make_unique<UpscalerGGML>(sd->n_threads,
                                                                false,
                                                                request.hires.upscale_tile_size,
                                                                sd->backend_spec,
                                                                sd->params_backend_spec);
                const size_t max_graph_vram_bytes = sd->max_graph_vram_bytes_for_module(SDBackendModule::UPSCALER);
                hires_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
                if (!hires_upscaler->load_from_file(request.hires.model_path,
                                                    sd->n_threads)) {
                    LOG_ERROR("load hires model upscaler failed");
                    return false;
                }
            }

            int hires_scheduler_steps = 0;
            std::vector<float> hires_sigma_sched =
                make_hires_sigma_schedule(sd,
                                          request.hires,
                                          sd_img_gen_params->sample_params,
                                          plan.sample_method,
                                          plan.sample_steps,
                                          sd->get_image_seq_len(request.hires.target_height, request.hires.target_width),
                                          &hires_scheduler_steps);
            LOG_INFO("hires fix: scheduler_steps=%d, denoising_strength=%.2f, sigma_sched_size=%zu%s",
                     hires_scheduler_steps,
                     request.hires.denoising_strength,
                     hires_sigma_sched.size(),
                     request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");

            std::vector<sd::Tensor<float>> hires_final_latents;
            int64_t hires_denoise_start = ggml_time_ms();
            for (int b = 0; b < (int)final_latents.size(); b++) {
                if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                    LOG_ERROR("cancelling generation during hires fix");
                    return false;
                }
                int64_t cur_seed = request.seed + b;
                sd->rng->manual_seed(cur_seed);
                sd->sampler_rng->manual_seed(cur_seed);

                sd::Tensor<float> upscaled = upscale_hires_latent(sd,
                                                                  final_latents[b],
                                                                  request,
                                                                  hires_upscaler.get());
                if (upscaled.empty()) {
                    return false;
                }

                sd::Tensor<float> noise = sd::randn_like<float>(upscaled, sd->rng);

                sd::Tensor<float> hires_denoise_mask;
                if (!latents.denoise_mask.empty()) {
                    std::vector<int64_t> mask_shape = latents.denoise_mask.shape();
                    mask_shape[0]                   = upscaled.shape()[0];
                    mask_shape[1]                   = upscaled.shape()[1];
                    hires_denoise_mask              = sd::ops::interpolate(latents.denoise_mask,
                                                                           mask_shape,
                                                                           sd::ops::InterpolateMode::NearestMax);
                }

                int64_t hires_sample_start = ggml_time_ms();
                sd::Tensor<float> x_0      = sd->sample(sd->diffusion_model,
                                                        true,
                                                        upscaled,
                                                        std::move(noise),
                                                        embeds.cond,
                                                        embeds.uncond,
                                                        embeds.img_uncond,
                                                        latents.control_image,
                                                        request.control_strength,
                                                        request.guidance,
                                                        plan.eta,
                                                        request.shifted_timestep,
                                                        plan.sample_method,
                                                        sd->is_flow_denoiser(),
                                                        plan.extra_sample_args,
                                                        hires_sigma_sched,
                                                        latents.ref_latents,
                                                        ref_image_params,
                                                        hires_denoise_mask,
                                                        sd::Tensor<float>(),
                                                        1.f,
                                                        0,
                                                        static_cast<float>(request.fps),
                                                        request.cache_params,
                                                        false);
                int64_t hires_sample_end   = ggml_time_ms();
                if (!x_0.empty()) {
                    LOG_INFO("hires sampling %d/%d completed, taking %.2fs",
                             b + 1,
                             (int)final_latents.size(),
                             (hires_sample_end - hires_sample_start) * 1.0f / 1000);
                    hires_final_latents.push_back(std::move(x_0));
                    continue;
                }

                LOG_ERROR("hires sampling for image %d/%d failed after %.2fs",
                          b + 1,
                          (int)final_latents.size(),
                          (hires_sample_end - hires_sample_start) * 1.0f / 1000);
                return false;
            }
            int64_t hires_denoise_end = ggml_time_ms();
            LOG_INFO("hires fix completed, taking %.2fs", (hires_denoise_end - hires_denoise_start) * 1.0f / 1000);

            final_latents = std::move(hires_final_latents);
        }

        int num_images = 0;
        auto result    = decode_image_outputs(sd, request, final_latents, &num_images);
        if (result == nullptr) {
            return false;
        }

        sd->lora_stat();

        int64_t t1 = ggml_time_ms();
        LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
        if (num_images_out != nullptr) {
            *num_images_out = num_images;
        }
        if (images_out != nullptr) {
            *images_out = result;
        } else {
            free_sd_images(result, num_images);
        }
        return true;
    }

}  // namespace sd::pipeline
