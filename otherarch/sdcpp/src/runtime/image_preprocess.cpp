#include "image_preprocess.h"
#include "core/util.h"

#include <climits>
#include <set>

namespace sd {

    static constexpr std::pair<const char*, ImageTarget> image_targets[] = {
        {"init", ImageTarget::Init},
        {"end", ImageTarget::End},
        {"mask", ImageTarget::Mask},
        {"control", ImageTarget::Control},
        {"ref", ImageTarget::Ref},
        {"ip-adapter", ImageTarget::IPAdapter},
        {"id", ImageTarget::ID},
        {"control-frame", ImageTarget::ControlFrame},
    };

    static constexpr std::pair<const char*, ImageResizeMode> image_resize_modes[] = {
        {"auto", ImageResizeMode::Auto},
        {"none", ImageResizeMode::None},
        {"stretch", ImageResizeMode::Stretch},
        {"crop", ImageResizeMode::Crop},
        {"crop-resize", ImageResizeMode::CropResize},
        {"fit-pad", ImageResizeMode::FitPad},
    };

    template <typename T, size_t N>
    static bool parse_enum(const std::string& text, const std::pair<const char*, T> (&names)[N], T& value) {
        for (const auto& entry : names) {
            if (text == entry.first) {
                value = entry.second;
                return true;
            }
        }
        return false;
    }

    template <typename T, size_t N>
    static const char* enum_name(T value, const std::pair<const char*, T> (&names)[N]) {
        for (const auto& entry : names) {
            if (value == entry.second)
                return entry.first;
        }
        return "unknown";
    }

    template <typename T>
    static bool one_of(T value, std::initializer_list<T> choices) {
        return std::find(choices.begin(), choices.end(), value) != choices.end();
    }

    static bool one_of(const std::string& value, std::initializer_list<const char*> choices) {
        for (const char* choice : choices) {
            if (value == choice)
                return true;
        }
        return false;
    }

    static ImageResizeMode resolve_mode(const std::map<std::string, std::string>& options, ImageResizeMode default_mode) {
        auto it              = options.find("mode");
        ImageResizeMode mode = ImageResizeMode::Auto;
        if (it != options.end())
            parse_enum(it->second, image_resize_modes, mode);
        if (mode != ImageResizeMode::Auto)
            return mode;
        return options.count("width") && default_mode == ImageResizeMode::None ? ImageResizeMode::Stretch : default_mode;
    }

    bool ImagePreprocessor::fail(const std::string& message) const {
        LOG_ERROR("image preprocessing: %s", message.c_str());
        valid_ = false;
        return false;
    }

    ImagePreprocessor::ImagePreprocessor(const char* text) {
        if (text == nullptr || trim(text).empty())
            return;
        for (const auto& part : split_string(text, ';')) {
            ImagePreprocessRule rule;
            std::set<std::string> keys;
            if (trim(part).empty()) {
                fail("empty rule");
                return;
            }
            for (const auto& entry : split_string(part, ',')) {
                size_t equal = entry.find('=');
                if (equal == std::string::npos) {
                    fail("expected key=value: " + entry);
                    return;
                }
                std::string key   = trim(entry.substr(0, equal));
                std::string value = trim(entry.substr(equal + 1));
                bool ok           = !value.empty() && keys.insert(key).second;
                int number        = 0;
                if (key == "target") {
                    ok &= parse_enum(value, image_targets, rule.target);
                } else if (key == "index") {
                    ok &= parse_strict_int(value, rule.index) && rule.index >= 0;
                } else {
                    if (key == "mode") {
                        ImageResizeMode mode;
                        ok &= parse_enum(value, image_resize_modes, mode);
                    } else if (key == "filter") {
                        ok &= one_of(value, {"auto", "nearest", "nearest-exact", "bilinear", "bicubic", "lanczos"});
                    } else if (key == "antialias") {
                        ok &= one_of(value, {"auto", "true", "false"});
                    } else if (key == "canny") {
                        ok &= one_of(value, {"true", "false"});
                    } else if (key == "anchor") {
                        ok &= one_of(value, {"center", "top", "bottom", "left", "right"});
                    } else if (key == "width" || key == "height") {
                        ok &= parse_strict_int(value, number) && number > 0;
                    } else if (key == "pad_color") {
                        ok &= value.size() == 7 || value.size() == 9;
                        ok &= !value.empty() && value[0] == '#';
                        for (size_t i = 1; i < value.size(); ++i)
                            ok &= std::isxdigit(static_cast<unsigned char>(value[i])) != 0;
                    } else {
                        ok = false;
                    }
                    rule.options[key] = value;
                }
                if (!ok) {
                    fail("invalid or duplicate option: " + entry);
                    return;
                }
            }
            if (!keys.count("target") || rule.options.empty() ||
                rule.options.count("width") != rule.options.count("height") ||
                (rule.index >= 0 && !one_of(rule.target, {ImageTarget::Ref, ImageTarget::ID, ImageTarget::ControlFrame}))) {
                fail("invalid target, index, or incomplete dimensions: " + part);
                return;
            }
            rules_.push_back(std::move(rule));
        }
        for (const auto& rule : rules_) {
            const auto options = resolve_options(rule.target, std::max(0, rule.index));
            if (options.count("antialias") && options.at("antialias") == "true" && options.count("filter") &&
                one_of(options.at("filter"), {"nearest", "nearest-exact"})) {
                fail("antialias requires bilinear, bicubic, or lanczos");
                return;
            }
        }
    }

    std::map<std::string, std::string> ImagePreprocessor::resolve_options(ImageTarget target, int index) const {
        std::map<std::string, std::string> options;
        for (int specificity = 0; specificity < 2; ++specificity) {
            for (const auto& rule : rules_) {
                if (rule.target == target &&
                    rule.index == (specificity == 0 ? -1 : index)) {
                    for (const auto& entry : rule.options)
                        options[entry.first] = entry.second;
                }
            }
        }
        return options;
    }

    bool ImagePreprocessor::validate_inputs(const sd_img_gen_params_t& params) const {
        const std::map<ImageTarget, int> counts = {
            {ImageTarget::Init, params.init_image.data != nullptr},
            {ImageTarget::Mask, params.mask_image.data != nullptr},
            {ImageTarget::Control, params.control_image.data != nullptr},
            {ImageTarget::IPAdapter, params.ip_adapter_image.data != nullptr},
            {ImageTarget::Ref, params.ref_images != nullptr ? params.ref_images_count : 0},
            {ImageTarget::ID, params.pm_params.id_images != nullptr ? params.pm_params.id_images_count : 0},
        };
        for (const auto& rule : rules_) {
            auto it   = counts.find(rule.target);
            int count = it == counts.end() ? 0 : it->second;
            if (count <= 0 || rule.index >= count) {
                return fail(std::string("rule targets an unavailable image: ") + enum_name(rule.target, image_targets));
            }
        }
        return valid_;
    }

    bool ImagePreprocessor::validate_inputs(const sd_vid_gen_params_t& params) const {
        const std::map<ImageTarget, int> counts = {
            {ImageTarget::Init, params.init_image.data != nullptr},
            {ImageTarget::End, params.end_image.data != nullptr},
            {ImageTarget::Ref, params.ref_images != nullptr ? params.ref_images_count : 0},
            {ImageTarget::ControlFrame, params.control_frames != nullptr ? params.control_frames_size : 0},
        };
        for (const auto& rule : rules_) {
            auto it   = counts.find(rule.target);
            int count = it == counts.end() ? 0 : it->second;
            if (count <= 0 || rule.index >= count)
                return fail(std::string("rule targets an unavailable video input: ") + enum_name(rule.target, image_targets));
        }
        return valid_;
    }

    static int anchor_offset(int remaining, const std::string& anchor, bool horizontal) {
        if (anchor == (horizontal ? "left" : "top"))
            return 0;
        if (anchor == (horizontal ? "right" : "bottom"))
            return remaining;
        return remaining / 2;
    }

    Tensor<float> ImagePreprocessor::apply_transform(const Tensor<float>& image, const std::map<std::string, std::string>& options, ImageTransform p, const std::string& label, ops::InterpolateMode default_filter) const {
        auto value = [&](const char* key, const char* fallback) {
            auto it = options.find(key);
            return it == options.end() ? std::string(fallback) : it->second;
        };
        std::string filter        = value("filter", "auto");
        ops::InterpolateMode mode = default_filter;
        if (filter == "nearest")
            mode = ops::InterpolateMode::Nearest;
        if (filter == "nearest-exact")
            mode = ops::InterpolateMode::NearestExact;
        if (filter == "bilinear")
            mode = ops::InterpolateMode::Bilinear;
        if (filter == "bicubic")
            mode = ops::InterpolateMode::Bicubic;
        if (filter == "lanczos")
            mode = ops::InterpolateMode::Lanczos;
        bool filtered  = ops::is_2d_filter_interpolate_mode(mode);
        bool antialias = value("antialias", "auto") == "true" ||
                         (value("antialias", "auto") == "auto" && filtered &&
                          (p.resize_width < p.crop_width || p.resize_height < p.crop_height));
        if (antialias && !filtered) {
            fail(label + ": antialias requires bilinear, bicubic, or lanczos");
            return {};
        }
        auto cropped = ops::slice(ops::slice(image, 0, p.x, p.x + p.crop_width), 1, p.y, p.y + p.crop_height);
        int channels = static_cast<int>(image.shape()[2]);
        bool resize  = p.resize_width != p.crop_width || p.resize_height != p.crop_height;
        if (resize && channels == 4 && filtered) {
            for (int64_t i = 0, pixels = cropped.shape()[0] * cropped.shape()[1]; i < pixels; ++i) {
                for (int c = 0; c < 3; ++c)
                    cropped[i + c * pixels] *= cropped[i + 3 * pixels];
            }
        }
        auto resized = ops::interpolate(cropped, {p.resize_width, p.resize_height, channels, 1}, mode, false, antialias);
        if (resize && channels == 4 && filtered) {
            for (int64_t i = 0, pixels = resized.shape()[0] * resized.shape()[1]; i < pixels; ++i) {
                float alpha = std::clamp(resized[i + 3 * pixels], 0.f, 1.f);
                for (int c = 0; c < 3; ++c)
                    resized[i + c * pixels] = alpha > 1e-6f ? resized[i + c * pixels] / alpha : 0.f;
            }
        }
        resized = ops::clamp(resized, 0.f, 1.f);
        Tensor<float> output({p.width, p.height, channels, 1});
        std::string color = value("pad_color", "#000000ff");
        if (color.size() == 7)
            color += "ff";
        uint8_t rgba[4];
        for (int c = 0; c < 4; ++c)
            rgba[c] = static_cast<uint8_t>(std::strtoul(color.substr(1 + c * 2, 2).c_str(), nullptr, 16));
        for (int c = 0; c < channels; ++c) {
            float fill = rgba[channels == 1 ? 0 : c] / 255.f;
            for (int y = 0; y < p.height; ++y) {
                for (int x = 0; x < p.width; ++x) {
                    output.index(x, y, c, 0) = x >= p.pad_x && x < p.pad_x + p.resize_width && y >= p.pad_y && y < p.pad_y + p.resize_height
                                                   ? resized.index(x - p.pad_x, y - p.pad_y, c, 0)
                                                   : fill;
                }
            }
        }
        LOG_INFO("preprocess %s: %dx%d crop=(%d,%d,%d,%d) resize=%dx%d pad=(%d,%d) output=%dx%d filter=%s(%d) antialias=%s",
                 label.c_str(), p.source_width, p.source_height, p.x, p.y, p.crop_width, p.crop_height,
                 p.resize_width, p.resize_height, p.pad_x, p.pad_y, p.width, p.height, filter.c_str(), static_cast<int>(mode), BOOL_STR(antialias));
        return output;
    }

    Tensor<float> ImagePreprocessor::apply_geometry(const Tensor<float>& image, ImageTarget target, int index, int width, int height, ImageResizeMode default_mode, ops::InterpolateMode default_filter, ImageTransform* plan_out) const {
        if (!valid_ || image.empty())
            return {};
        const std::string label = std::string(enum_name(target, image_targets)) + "[" + std::to_string(index) + "]";
        auto options            = resolve_options(target, index);
        if (image.dim() != 4 || image.shape()[3] != 1 || image.shape()[2] < 1 || image.shape()[2] > 4) {
            fail(label + ": expected one image with 1 to 4 channels");
            return {};
        }
        ImageTransform p;
        p.source_width = p.crop_width = static_cast<int>(image.shape()[0]);
        p.source_height = p.crop_height = static_cast<int>(image.shape()[1]);
        int target_width                = width > 0 ? width : p.source_width;
        int target_height               = height > 0 ? height : p.source_height;
        if (options.count("width")) {
            parse_strict_int(options.at("width"), target_width);
            parse_strict_int(options.at("height"), target_height);
        }
        ImageResizeMode mode = resolve_mode(options, default_mode);
        std::string anchor   = options.count("anchor") ? options.at("anchor") : "center";
        p.width = p.resize_width = target_width;
        p.height = p.resize_height = target_height;
        if (mode == ImageResizeMode::None) {
            if (options.count("width") && (target_width != p.source_width || target_height != p.source_height)) {
                fail(label + ": mode=none conflicts with requested dimensions");
                return {};
            }
            p.width = p.resize_width = p.source_width;
            p.height = p.resize_height = p.source_height;
        } else if (mode == ImageResizeMode::Crop || mode == ImageResizeMode::CropResize) {
            if (mode == ImageResizeMode::Crop) {
                p.crop_width  = target_width;
                p.crop_height = target_height;
            } else if (int64_t(p.source_width) * target_height > int64_t(p.source_height) * target_width) {
                p.crop_width = std::max(1, static_cast<int>(int64_t(p.source_height) * target_width / target_height));
            } else {
                p.crop_height = std::max(1, static_cast<int>(int64_t(p.source_width) * target_height / target_width));
            }
            if (p.crop_width > p.source_width || p.crop_height > p.source_height) {
                fail(label + ": crop exceeds source dimensions");
                return {};
            }
            p.x = anchor_offset(p.source_width - p.crop_width, anchor, true);
            p.y = anchor_offset(p.source_height - p.crop_height, anchor, false);
        } else if (mode == ImageResizeMode::FitPad) {
            double scale    = std::min(double(target_width) / p.source_width, double(target_height) / p.source_height);
            p.resize_width  = std::max(1, std::min(target_width, static_cast<int>(std::round(p.source_width * scale))));
            p.resize_height = std::max(1, std::min(target_height, static_cast<int>(std::round(p.source_height * scale))));
            p.pad_x         = anchor_offset(target_width - p.resize_width, anchor, true);
            p.pad_y         = anchor_offset(target_height - p.resize_height, anchor, false);
        }
        if (p.width <= 0 || p.height <= 0) {
            fail(label + ": invalid output dimensions");
            return {};
        }
        uint64_t max_pixels = std::min<uint64_t>(INT64_MAX, SIZE_MAX / sizeof(float)) / static_cast<uint64_t>(image.shape()[2]);
        if (uint64_t(p.width) * p.height > max_pixels || uint64_t(p.resize_width) * p.resize_height > max_pixels) {
            fail(label + ": image allocation size overflows");
            return {};
        }
        if (plan_out != nullptr)
            *plan_out = p;
        return apply_transform(image, options, p, label, default_filter);
    }

    Tensor<float> ImagePreprocessor::preprocess_input(sd_image_t image, ImageTarget target, int index, int width, int height) {
        if (image.data == nullptr || image.width == 0 || image.height == 0 || image.width > INT_MAX || image.height > INT_MAX || image.channel < 1 || image.channel > 4) {
            fail(std::string(enum_name(target, image_targets)) + ": invalid input image");
            return {};
        }
        auto tensor = sd_image_to_tensor(image);
        if (target == ImageTarget::Mask && has_init_transform_) {
            auto options = resolve_options(target, index);
            if (image.width != init_transform_.source_width || image.height != init_transform_.source_height) {
                fail("mask and init source dimensions must match");
                return {};
            }
            bool geometry_override = options.count("width") || options.count("anchor") ||
                                     (options.count("mode") && options.at("mode") != "auto");
            if (geometry_override) {
                ImageTransform p;
                auto init_options            = resolve_options(ImageTarget::Init, 0);
                ImageResizeMode default_mode = resolve_mode(init_options, ImageResizeMode::CropResize);
                auto result                  = apply_geometry(tensor, target, index, init_transform_.width, init_transform_.height, default_mode, ops::InterpolateMode::NearestExact, &p);
                if (result.empty())
                    return {};
                const auto& q = init_transform_;
                if (p.x != q.x || p.y != q.y || p.crop_width != q.crop_width || p.crop_height != q.crop_height ||
                    p.resize_width != q.resize_width || p.resize_height != q.resize_height || p.pad_x != q.pad_x || p.pad_y != q.pad_y || p.width != q.width || p.height != q.height) {
                    fail("mask geometry conflicts with init; configure geometry on init and filter on mask");
                    return {};
                }
                return result;
            }
            return apply_transform(tensor, options, init_transform_, "mask[0]", ops::InterpolateMode::NearestExact);
        }
        auto result = apply_geometry(tensor, target, index, width, height, width > 0 ? ImageResizeMode::CropResize : ImageResizeMode::None,
                                     target == ImageTarget::Mask ? ops::InterpolateMode::NearestExact : ops::InterpolateMode::Nearest,
                                     target == ImageTarget::Init ? &init_transform_ : nullptr);
        if (target == ImageTarget::Init)
            has_init_transform_ = !result.empty();
        return result;
    }

    ImagePreprocessor::~ImagePreprocessor() {
        for (const auto& image : owned_images_)
            std::free(image.data);
    }

    bool ImagePreprocessor::prepare_image(sd_image_t& image, ImageTarget target, int index, int width, int height) {
        if (image.data == nullptr)
            return true;
        auto options = resolve_options(target, index);
        bool canny   = options.count("canny") && options.at("canny") == "true";
        auto tensor  = preprocess_input(image, target, index, width, height);
        if (tensor.empty())
            return false;
        auto output = tensor_to_sd_image(tensor);
        if (output.data == nullptr)
            return fail("could not allocate input preprocessing buffer");
        owned_images_.push_back(output);
        if (canny && !preprocess_canny(output, 0.08f, 0.08f, 0.8f, 1.f, false))
            return fail("Canny preprocessing failed");
        image = output;
        return true;
    }

    bool ImagePreprocessor::prepare_array(sd_image_t*& images, int count, ImageTarget target, std::vector<sd_image_t>& storage, int width, int height) {
        if (count < 0 || (count > 0 && images == nullptr))
            return fail(std::string("invalid image array: ") + enum_name(target, image_targets));
        if (count == 0)
            return true;
        storage.assign(images, images + count);
        for (int i = 0; i < count; ++i) {
            if (storage[i].data == nullptr)
                return fail(std::string("empty image in array: ") + enum_name(target, image_targets));
            if (!prepare_image(storage[i], target, i, width, height))
                return false;
        }
        images = storage.data();
        return true;
    }

    bool ImagePreprocessor::prepare_inputs(sd_img_gen_params_t& params, int width, int height) {
        if (prepared_)
            return fail("inputs have already been prepared");
        prepared_ = true;
        if (!valid_ || !validate_inputs(params))
            return false;
        if (!prepare_image(params.init_image, ImageTarget::Init, 0, width, height) ||
            !prepare_image(params.mask_image, ImageTarget::Mask, 0, width, height) ||
            !prepare_image(params.control_image, ImageTarget::Control, 0, width, height) ||
            !prepare_image(params.ip_adapter_image, ImageTarget::IPAdapter, 0, -1, -1) ||
            !prepare_array(params.ref_images, params.ref_images_count, ImageTarget::Ref, ref_images_) ||
            !prepare_array(params.pm_params.id_images, params.pm_params.id_images_count, ImageTarget::ID, id_images_))
            return false;
        params.image_preprocess = {};
        return true;
    }

    bool ImagePreprocessor::prepare_inputs(sd_vid_gen_params_t& params, int width, int height) {
        if (prepared_)
            return fail("inputs have already been prepared");
        prepared_ = true;
        if (!valid_ || !validate_inputs(params))
            return false;
        if (!prepare_image(params.init_image, ImageTarget::Init, 0, width, height) ||
            !prepare_image(params.end_image, ImageTarget::End, 0, width, height) ||
            !prepare_array(params.ref_images, params.ref_images_count, ImageTarget::Ref, ref_images_) ||
            !prepare_array(params.control_frames, params.control_frames_size, ImageTarget::ControlFrame, control_frames_, width, height))
            return false;
        params.image_preprocess = {};
        return true;
    }

}  // namespace sd
