#include "runtime/guidance.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>

#include "core/util.h"

namespace sd::guidance {

    static bool has_tensor(const sd::Tensor<float>* tensor) {
        return tensor != nullptr && !tensor->empty();
    }

    bool is_adaptive_projected_guidance_enabled(const AdaptiveProjectedGuidanceParams& params) {
        return params.eta != 1.0f || params.momentum != 0.0f || params.norm_threshold > 0.0f;
    }

    AdaptiveProjectedGuidanceParams parse_adaptive_projected_guidance_args(const char* extra_sample_args) {
        AdaptiveProjectedGuidanceParams params;
        for (const auto& [key, value] : parse_key_value_args(extra_sample_args, "extra sample arg")) {
            float parsed = 0.0f;
            if (key == "apg_eta") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.eta = parsed;
            } else if (key == "apg_momentum") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.momentum = parsed;
            } else if (key == "apg_norm_threshold") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.norm_threshold = parsed;
            } else if (key == "apg_norm_threshold_smoothing") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.norm_threshold_smoothing = parsed;
            }
        }
        return params;
    }

    bool parse_skip_layer_guidance_uncond_arg(const char* extra_sample_args) {
        bool uncond = false;
        for (const auto& [key, value] : parse_key_value_args(extra_sample_args, "extra sample arg")) {
            if (key == "slg_uncond") {
                if (!parse_strict_bool(value, uncond)) {
                    LOG_WARN("ignoring invalid SLG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                }
            }
        }
        return uncond;
    }

    ClassifierFreeGuidance::ClassifierFreeGuidance(float guidance_scale,
                                                   float image_guidance_scale)
        : guidance_scale_(guidance_scale),
          image_guidance_scale_(image_guidance_scale) {
    }

    GuiderOutput ClassifierFreeGuidance::forward(const GuidanceInput& input,
                                                 GuiderOutput previous) const {
        (void)previous;

        GuiderOutput output;
        if (!has_tensor(input.pred_cond)) {
            return output;
        }

        const sd::Tensor<float>& pred_cond = *input.pred_cond;
        output.pred                        = pred_cond;
        if (has_tensor(input.pred_uncond)) {
            const sd::Tensor<float>& pred_uncond = *input.pred_uncond;
            if (has_tensor(input.pred_img_uncond)) {
                const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
                output.pred                              = pred_img_uncond +
                              image_guidance_scale_ * (pred_uncond - pred_img_uncond) +
                              guidance_scale_ * (pred_cond - pred_uncond);

            } else {
                output.pred = pred_uncond + guidance_scale_ * (pred_cond - pred_uncond);
            }
        } else if (has_tensor(input.pred_img_uncond)) {
            const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
            output.pred                              = pred_img_uncond + guidance_scale_ * (pred_cond - pred_img_uncond);
        }

        return output;
    }

    AdaptiveProjectedGuidance::AdaptiveProjectedGuidance(float guidance_scale,
                                                         float image_guidance_scale,
                                                         AdaptiveProjectedGuidanceParams params)
        : guidance_scale_(guidance_scale),
          image_guidance_scale_(image_guidance_scale),
          params_(params) {
    }

    static sd::Tensor<float> calculate_guidance_delta(const sd::Tensor<float>& pred_cond,
                                                      const sd::Tensor<float>* pred_uncond,
                                                      const sd::Tensor<float>* pred_img_uncond,
                                                      float guidance_scale,
                                                      float image_guidance_scale) {
        if (pred_img_uncond != nullptr) {
            if (pred_uncond != nullptr && guidance_scale == 1.0f) {
                return *pred_uncond - *pred_img_uncond;
            }
            if (pred_uncond != nullptr) {
                return pred_cond +
                       (*pred_uncond * (image_guidance_scale - guidance_scale) +
                        *pred_img_uncond * (1.0f - image_guidance_scale)) /
                           (guidance_scale - 1.0f);
            }
            return pred_cond - *pred_img_uncond;
        }
        return pred_cond - *pred_uncond;
    }

    GuiderOutput AdaptiveProjectedGuidance::forward(const GuidanceInput& input,
                                                    GuiderOutput previous) const {
        (void)previous;

        GuiderOutput output;
        if (!has_tensor(input.pred_cond)) {
            return output;
        }

        const sd::Tensor<float>& pred_cond = *input.pred_cond;
        output.pred                        = pred_cond;
        if (has_tensor(input.pred_uncond)) {
            const sd::Tensor<float>& pred_uncond = *input.pred_uncond;
            if (has_tensor(input.pred_img_uncond)) {
                const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
                output.pred                              = pred_img_uncond +
                              image_guidance_scale_ * (pred_uncond - pred_img_uncond) +
                              guidance_scale_ * (pred_cond - pred_uncond);
            } else {
                output.pred = pred_uncond + guidance_scale_ * (pred_cond - pred_uncond);
            }
        } else if (has_tensor(input.pred_img_uncond)) {
            const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
            output.pred                              = pred_img_uncond + guidance_scale_ * (pred_cond - pred_img_uncond);
        }
        if (!has_tensor(input.pred_uncond) && !has_tensor(input.pred_img_uncond)) {
            return output;
        }

        const sd::Tensor<float>* pred_uncond     = input.pred_uncond;
        const sd::Tensor<float>* pred_img_uncond = input.pred_img_uncond;

        sd::Tensor<float> deltas = calculate_guidance_delta(pred_cond,
                                                            pred_uncond,
                                                            pred_img_uncond,
                                                            guidance_scale_,
                                                            image_guidance_scale_);
        if (params_.momentum != 0.0f) {
            if (momentum_buffer_.shape() != deltas.shape()) {
                momentum_buffer_ = sd::Tensor<float>::zeros_like(deltas);
            }
            deltas += params_.momentum * momentum_buffer_;
            momentum_buffer_ = deltas;
        }

        float diff_norm = 0.0f;
        if (params_.norm_threshold > 0.0f) {
            diff_norm = std::sqrt((deltas * deltas).sum());
        }

        float apg_scale_factor = 1.0f;
        if (params_.norm_threshold > 0.0f) {
            if (diff_norm > 0.0f) {
                if (params_.norm_threshold_smoothing <= 0.0f) {
                    apg_scale_factor = std::min(1.0f, params_.norm_threshold / diff_norm);
                } else {
                    float x          = params_.norm_threshold / diff_norm;
                    apg_scale_factor = x / std::pow(1.0f + std::pow(x, 1.0f / params_.norm_threshold_smoothing),
                                                    params_.norm_threshold_smoothing);
                }
            }
        }

        deltas *= apg_scale_factor;
        if (params_.eta != 1.0f) {
            float cond_norm_sq = (pred_cond * pred_cond).sum();
            if (cond_norm_sq != 0.0f) {
                float projection_scale = (pred_cond * deltas).sum() / cond_norm_sq;
                deltas += (params_.eta - 1.0f) * (projection_scale * pred_cond);
            }
        }

        output.pred = pred_cond;
        if (pred_uncond != nullptr) {
            if (guidance_scale_ != 1.0f) {
                output.pred = pred_cond + (guidance_scale_ - 1.0f) * deltas;
            } else if (pred_img_uncond != nullptr) {
                output.pred = pred_cond + (image_guidance_scale_ - 1.0f) * deltas;
            }
        } else if (pred_img_uncond != nullptr) {
            output.pred = *pred_img_uncond + guidance_scale_ * deltas;
        }

        return output;
    }

    SkipLayerGuidance::SkipLayerGuidance(std::vector<int> layers,
                                         float scale,
                                         float start,
                                         float stop)
        : layers_(std::move(layers)),
          scale_(scale),
          start_(start),
          stop_(stop) {
    }

    bool SkipLayerGuidance::is_enabled_for_step(const GuidanceInput& input) const {
        if (layers_.empty() || input.schedule_size == 0) {
            return false;
        }

        int start_step = static_cast<int>(start_ * static_cast<float>(input.schedule_size));
        int stop_step  = static_cast<int>(stop_ * static_cast<float>(input.schedule_size));
        return input.step > start_step && input.step < stop_step;
    }

    const std::vector<int>& SkipLayerGuidance::layers() const {
        return layers_;
    }

    GuiderOutput SkipLayerGuidance::forward(const GuidanceInput& input,
                                            GuiderOutput output) const {
        if (scale_ == 0.0f || !is_enabled_for_step(input) || !input.predict_skip_layer) {
            return output;
        }

        if (output.pred.empty() || !has_tensor(input.pred_cond)) {
            return GuiderOutput();
        }

        output.pred_skip_layer = input.predict_skip_layer();
        if (output.pred_skip_layer.empty()) {
            return GuiderOutput();
        }

        output.pred += (*input.pred_cond - output.pred_skip_layer) * scale_;
        return output;
    }

}  // namespace sd::guidance
