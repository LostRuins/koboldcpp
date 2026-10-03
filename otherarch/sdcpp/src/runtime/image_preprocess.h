#ifndef __SD_RUNTIME_IMAGE_PREPROCESS_H__
#define __SD_RUNTIME_IMAGE_PREPROCESS_H__

#include <map>
#include <string>
#include <vector>

#include "core/tensor.hpp"
#include "stable-diffusion.h"

namespace sd {

    enum class ImageTarget {
        Init,
        End,
        Mask,
        Control,
        Ref,
        IPAdapter,
        ID,
        ControlFrame,
    };

    enum class ImageResizeMode {
        Auto,
        None,
        Stretch,
        Crop,
        CropResize,
        FitPad,
    };

    struct ImageTransform {
        int source_width  = 0;
        int source_height = 0;
        int x             = 0;
        int y             = 0;
        int crop_width    = 0;
        int crop_height   = 0;
        int resize_width  = 0;
        int resize_height = 0;
        int width         = 0;
        int height        = 0;
        int pad_x         = 0;
        int pad_y         = 0;
    };

    struct ImagePreprocessRule {
        ImageTarget target = ImageTarget::Init;
        int index          = -1;
        std::map<std::string, std::string> options;
    };

    class ImagePreprocessor {
        std::vector<ImagePreprocessRule> rules_;
        mutable bool valid_ = true;
        ImageTransform init_transform_;
        bool has_init_transform_ = false;
        bool prepared_           = false;
        std::vector<sd_image_t> owned_images_;
        std::vector<sd_image_t> ref_images_;
        std::vector<sd_image_t> id_images_;
        std::vector<sd_image_t> control_frames_;

        bool fail(const std::string& message) const;
        std::map<std::string, std::string> resolve_options(ImageTarget target, int index) const;
        Tensor<float> apply_transform(const Tensor<float>& image, const std::map<std::string, std::string>& options, ImageTransform plan, const std::string& label, ops::InterpolateMode default_filter) const;

        bool prepare_image(sd_image_t& image, ImageTarget target, int index, int width, int height);
        bool prepare_array(sd_image_t*& images, int count, ImageTarget target, std::vector<sd_image_t>& storage, int width = -1, int height = -1);

    public:
        explicit ImagePreprocessor(const char* rules = nullptr);
        ~ImagePreprocessor();
        ImagePreprocessor(const ImagePreprocessor&)            = delete;
        ImagePreprocessor& operator=(const ImagePreprocessor&) = delete;
        bool prepare_inputs(sd_img_gen_params_t& params, int width, int height);
        bool prepare_inputs(sd_vid_gen_params_t& params, int width, int height);
        bool is_valid() const { return valid_; }
        bool validate_inputs(const sd_img_gen_params_t& params) const;
        bool validate_inputs(const sd_vid_gen_params_t& params) const;
        Tensor<float> apply_geometry(const Tensor<float>& image, ImageTarget target, int index, int width, int height, ImageResizeMode default_mode = ImageResizeMode::Stretch, ops::InterpolateMode default_filter = ops::InterpolateMode::Nearest, ImageTransform* plan_out = nullptr) const;
        Tensor<float> preprocess_input(sd_image_t image, ImageTarget target, int index = 0, int width = -1, int height = -1);
    };

}  // namespace sd

#endif  // __SD_RUNTIME_IMAGE_PREPROCESS_H__
