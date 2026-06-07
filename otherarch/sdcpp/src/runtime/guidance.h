#ifndef __SD_RUNTIME_GUIDANCE_H__
#define __SD_RUNTIME_GUIDANCE_H__

#include <cstddef>
#include <functional>
#include <vector>

#include "core/tensor.hpp"

namespace sd::guidance {

    struct GuiderOutput {
        sd::Tensor<float> pred;
        sd::Tensor<float> pred_cond;
        sd::Tensor<float> pred_uncond;
        sd::Tensor<float> pred_img_cond;
        sd::Tensor<float> pred_skip_layer;
    };

    struct AdaptiveProjectedGuidanceParams {
        float eta                      = 1.0f;
        float momentum                 = 0.0f;
        float norm_threshold           = 0.0f;
        float norm_threshold_smoothing = 0.0f;
    };

    AdaptiveProjectedGuidanceParams parse_adaptive_projected_guidance_args(const char* extra_sample_args);
    bool is_adaptive_projected_guidance_enabled(const AdaptiveProjectedGuidanceParams& params);
    bool parse_skip_layer_guidance_uncond_arg(const char* extra_sample_args);

    struct GuidanceInput {
        int step                                 = 0;
        size_t schedule_size                     = 0;
        const sd::Tensor<float>* pred_cond       = nullptr;
        const sd::Tensor<float>* pred_uncond     = nullptr;
        const sd::Tensor<float>* pred_img_uncond = nullptr;

        std::function<sd::Tensor<float>()> predict_skip_layer;
    };

    class BaseGuidance {
    public:
        virtual ~BaseGuidance()                                   = default;
        virtual GuiderOutput forward(const GuidanceInput& input,
                                     GuiderOutput previous) const = 0;
    };

    class ClassifierFreeGuidance : public BaseGuidance {
        float guidance_scale_       = 1.0f;
        float image_guidance_scale_ = 1.0f;

    public:
        ClassifierFreeGuidance(float guidance_scale,
                               float image_guidance_scale);

        GuiderOutput forward(const GuidanceInput& input,
                             GuiderOutput previous) const override;
    };

    class AdaptiveProjectedGuidance : public BaseGuidance {
        float guidance_scale_       = 1.0f;
        float image_guidance_scale_ = 1.0f;
        AdaptiveProjectedGuidanceParams params_;
        mutable sd::Tensor<float> momentum_buffer_;

    public:
        AdaptiveProjectedGuidance(float guidance_scale,
                                  float image_guidance_scale,
                                  AdaptiveProjectedGuidanceParams params);

        GuiderOutput forward(const GuidanceInput& input,
                             GuiderOutput previous) const override;
    };

    class SkipLayerGuidance : public BaseGuidance {
        std::vector<int> layers_;
        float scale_ = 0.0f;
        float start_ = 0.0f;
        float stop_  = 1.0f;

    public:
        SkipLayerGuidance(std::vector<int> layers,
                          float scale,
                          float start,
                          float stop);

        bool is_enabled_for_step(const GuidanceInput& input) const;
        const std::vector<int>& layers() const;

        GuiderOutput forward(const GuidanceInput& input,
                             GuiderOutput previous) const override;
    };

}  // namespace sd::guidance

#endif  // __SD_RUNTIME_GUIDANCE_H__
