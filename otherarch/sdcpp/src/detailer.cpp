#include "detailer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

#include "core/util.h"
#include "json.hpp"
#include "model_loader.h"

struct adetailer_ctx_t {
    ADetailerGGML* detailer = nullptr;
};

struct LetterboxInput {
    sd::Tensor<float> tensor;
    float scale = 1.f;
    int pad_x   = 0;
    int pad_y   = 0;
};

struct Mask {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> data;
};

struct CropRegion {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};

static std::string trim_copy(const std::string& value) {
    size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool parse_bool(const std::string& value, bool* result) {
    const std::string lower = lower_copy(trim_copy(value));
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
        *result = true;
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
        *result = false;
        return true;
    }
    return false;
}

static bool parse_int(const std::string& value, int* result) {
    try {
        size_t used = 0;
        int parsed  = std::stoi(trim_copy(value), &used);
        if (used != trim_copy(value).size()) {
            return false;
        }
        *result = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_float(const std::string& value, float* result) {
    try {
        size_t used  = 0;
        float parsed = std::stof(trim_copy(value), &used);
        if (used != trim_copy(value).size() || !std::isfinite(parsed)) {
            return false;
        }
        *result = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

static float image_channel(const sd_image_t& image, int x, int y, int channel) {
    if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channel == 0) {
        return 0.f;
    }
    x                  = std::clamp(x, 0, static_cast<int>(image.width) - 1);
    y                  = std::clamp(y, 0, static_cast<int>(image.height) - 1);
    int source_channel = image.channel == 1 ? 0 : std::min(channel, static_cast<int>(image.channel) - 1);
    size_t index       = (static_cast<size_t>(y) * image.width + x) * image.channel + source_channel;
    return image.data[index] / 255.f;
}

static float bilinear_channel(const sd_image_t& image, float x, float y, int channel) {
    int x0   = static_cast<int>(std::floor(x));
    int y0   = static_cast<int>(std::floor(y));
    int x1   = x0 + 1;
    int y1   = y0 + 1;
    float wx = x - x0;
    float wy = y - y0;
    float a  = image_channel(image, x0, y0, channel) * (1.f - wx) + image_channel(image, x1, y0, channel) * wx;
    float b  = image_channel(image, x0, y1, channel) * (1.f - wx) + image_channel(image, x1, y1, channel) * wx;
    return a * (1.f - wy) + b * wy;
}

static LetterboxInput make_letterbox_input(sd_image_t image, int input_size) {
    LetterboxInput result;
    result.tensor      = sd::full<float>({input_size, input_size, 3, 1}, 114.f / 255.f);
    result.scale       = std::min(static_cast<float>(input_size) / image.width,
                                  static_cast<float>(input_size) / image.height);
    int resized_width  = std::max(1, static_cast<int>(std::round(image.width * result.scale)));
    int resized_height = std::max(1, static_cast<int>(std::round(image.height * result.scale)));
    result.pad_x       = (input_size - resized_width) / 2;
    result.pad_y       = (input_size - resized_height) / 2;

    for (int y = 0; y < resized_height; ++y) {
        float source_y = (y + 0.5f) / result.scale - 0.5f;
        for (int x = 0; x < resized_width; ++x) {
            float source_x = (x + 0.5f) / result.scale - 0.5f;
            for (int c = 0; c < 3; ++c) {
                result.tensor.index(x + result.pad_x, y + result.pad_y, c, 0) = bilinear_channel(image, source_x, source_y, c);
            }
        }
    }
    return result;
}

static float logistic(float x) {
    if (x >= 0.f) {
        float z = std::exp(-x);
        return 1.f / (1.f + z);
    }
    float z = std::exp(x);
    return z / (1.f + z);
}

static float dfl_expectation(const float* values, int64_t anchor_count, int channel_offset, int reg_max, int64_t anchor) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < reg_max; ++i) {
        maximum = std::max(maximum, values[anchor + anchor_count * (channel_offset + i)]);
    }
    float denominator = 0.f;
    float numerator   = 0.f;
    for (int i = 0; i < reg_max; ++i) {
        float probability = std::exp(values[anchor + anchor_count * (channel_offset + i)] - maximum);
        denominator += probability;
        numerator += probability * i;
    }
    return denominator > 0.f ? numerator / denominator : 0.f;
}

static float box_iou(const ADetailerDetection& a, const ADetailerDetection& b) {
    float x1           = std::max(a.x1, b.x1);
    float y1           = std::max(a.y1, b.y1);
    float x2           = std::min(a.x2, b.x2);
    float y2           = std::min(a.y2, b.y2);
    float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float area_a       = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float area_b       = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    float union_area   = area_a + area_b - intersection;
    return union_area > 0.f ? intersection / union_area : 0.f;
}

static std::vector<ADetailerDetection> decode_detections(const sd::Tensor<float>& raw,
                                                         const YOLOv8Config& config,
                                                         const LetterboxInput& letterbox,
                                                         sd_image_t image,
                                                         const ADetailerParams& params) {
    std::vector<ADetailerDetection> candidates;
    if (raw.empty() || raw.dim() < 2) {
        return candidates;
    }
    int64_t anchor_count = raw.shape()[0];
    int64_t channels     = raw.shape()[1];
    if (channels != config.reg_max * 4 + config.num_classes) {
        LOG_ERROR("unexpected YOLOv8 output channels: %lld", static_cast<long long>(channels));
        return candidates;
    }

    const std::array<int, 3> strides = {8, 16, 32};
    std::array<int64_t, 4> offsets   = {0, 0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        int grid       = params.input_size / strides[i];
        offsets[i + 1] = offsets[i] + static_cast<int64_t>(grid) * grid;
    }
    if (offsets[3] != anchor_count) {
        LOG_ERROR("unexpected YOLOv8 anchor count: %lld (expected %lld)",
                  static_cast<long long>(anchor_count),
                  static_cast<long long>(offsets[3]));
        return candidates;
    }

    const float* values = raw.data();
    for (int scale_index = 0; scale_index < 3; ++scale_index) {
        int stride = strides[scale_index];
        int grid   = params.input_size / stride;
        for (int64_t anchor = offsets[scale_index]; anchor < offsets[scale_index + 1]; ++anchor) {
            int64_t local = anchor - offsets[scale_index];
            int grid_x    = static_cast<int>(local % grid);
            int grid_y    = static_cast<int>(local / grid);

            float confidence = 0.f;
            int class_id     = 0;
            for (int c = 0; c < config.num_classes; ++c) {
                float score = logistic(values[anchor + anchor_count * (config.reg_max * 4 + c)]);
                if (score > confidence) {
                    confidence = score;
                    class_id   = c;
                }
            }
            if (confidence < params.confidence) {
                continue;
            }

            float left     = dfl_expectation(values, anchor_count, 0 * config.reg_max, config.reg_max, anchor);
            float top      = dfl_expectation(values, anchor_count, 1 * config.reg_max, config.reg_max, anchor);
            float right    = dfl_expectation(values, anchor_count, 2 * config.reg_max, config.reg_max, anchor);
            float bottom   = dfl_expectation(values, anchor_count, 3 * config.reg_max, config.reg_max, anchor);
            float center_x = (grid_x + 0.5f) * stride;
            float center_y = (grid_y + 0.5f) * stride;

            ADetailerDetection detection;
            detection.x1         = (center_x - left * stride - letterbox.pad_x) / letterbox.scale;
            detection.y1         = (center_y - top * stride - letterbox.pad_y) / letterbox.scale;
            detection.x2         = (center_x + right * stride - letterbox.pad_x) / letterbox.scale;
            detection.y2         = (center_y + bottom * stride - letterbox.pad_y) / letterbox.scale;
            detection.x1         = std::clamp(detection.x1, 0.f, static_cast<float>(image.width));
            detection.y1         = std::clamp(detection.y1, 0.f, static_cast<float>(image.height));
            detection.x2         = std::clamp(detection.x2, 0.f, static_cast<float>(image.width));
            detection.y2         = std::clamp(detection.y2, 0.f, static_cast<float>(image.height));
            detection.confidence = confidence;
            detection.class_id   = class_id;
            if (detection.x2 > detection.x1 && detection.y2 > detection.y1) {
                candidates.push_back(detection);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.confidence > b.confidence;
    });
    std::vector<ADetailerDetection> selected;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& kept : selected) {
            if (candidate.class_id == kept.class_id && box_iou(candidate, kept) > params.nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            selected.push_back(candidate);
            if (static_cast<int>(selected.size()) >= params.max_detections) {
                break;
            }
        }
    }
    return selected;
}

static float detection_area(const ADetailerDetection& detection) {
    return std::max(0.f, detection.x2 - detection.x1) * std::max(0.f, detection.y2 - detection.y1);
}

static void filter_and_sort_detections(std::vector<ADetailerDetection>* detections,
                                       int width,
                                       int height,
                                       const ADetailerParams& params) {
    const float image_area = static_cast<float>(width) * height;
    detections->erase(std::remove_if(detections->begin(), detections->end(), [&](const auto& detection) {
                          float ratio = detection_area(detection) / image_area;
                          return ratio < params.mask_min_ratio || ratio > params.mask_max_ratio;
                      }),
                      detections->end());

    if (params.mask_k_largest > 0 && static_cast<int>(detections->size()) > params.mask_k_largest) {
        std::partial_sort(detections->begin(),
                          detections->begin() + params.mask_k_largest,
                          detections->end(),
                          [](const auto& a, const auto& b) { return detection_area(a) > detection_area(b); });
        detections->resize(params.mask_k_largest);
    }

    if (params.sort_by == ADETAILER_SORT_LEFT_TO_RIGHT) {
        std::sort(detections->begin(), detections->end(), [](const auto& a, const auto& b) { return a.x1 < b.x1; });
    } else if (params.sort_by == ADETAILER_SORT_CENTER_TO_EDGE) {
        const float cx = width * 0.5f;
        const float cy = height * 0.5f;
        std::sort(detections->begin(), detections->end(), [&](const auto& a, const auto& b) {
            float ax = (a.x1 + a.x2) * 0.5f - cx;
            float ay = (a.y1 + a.y2) * 0.5f - cy;
            float bx = (b.x1 + b.x2) * 0.5f - cx;
            float by = (b.y1 + b.y2) * 0.5f - cy;
            return ax * ax + ay * ay < bx * bx + by * by;
        });
    } else if (params.sort_by == ADETAILER_SORT_AREA) {
        std::sort(detections->begin(), detections->end(), [](const auto& a, const auto& b) {
            return detection_area(a) > detection_area(b);
        });
    }
}

static Mask bbox_mask(const ADetailerDetection& detection, int width, int height) {
    Mask mask;
    mask.width  = width;
    mask.height = height;
    mask.data.assign(static_cast<size_t>(width) * height, 0);
    int x1 = std::clamp(static_cast<int>(std::floor(detection.x1)), 0, width);
    int y1 = std::clamp(static_cast<int>(std::floor(detection.y1)), 0, height);
    int x2 = std::clamp(static_cast<int>(std::ceil(detection.x2)), 0, width);
    int y2 = std::clamp(static_cast<int>(std::ceil(detection.y2)), 0, height);
    for (int y = y1; y < y2; ++y) {
        std::fill(mask.data.begin() + static_cast<size_t>(y) * width + x1,
                  mask.data.begin() + static_cast<size_t>(y) * width + x2,
                  255);
    }
    return mask;
}

static Mask offset_mask(const Mask& source, int x_offset, int y_offset) {
    Mask output{source.width, source.height, std::vector<uint8_t>(source.data.size(), 0)};
    for (int y = 0; y < source.height; ++y) {
        int target_y = y - y_offset;
        if (target_y < 0 || target_y >= source.height) {
            continue;
        }
        for (int x = 0; x < source.width; ++x) {
            int target_x = x + x_offset;
            if (target_x >= 0 && target_x < source.width) {
                output.data[static_cast<size_t>(target_y) * source.width + target_x] = source.data[static_cast<size_t>(y) * source.width + x];
            }
        }
    }
    return output;
}

static Mask morphology_mask(const Mask& source, int amount) {
    if (amount == 0) {
        return source;
    }
    int kernel = std::abs(amount);
    int before = kernel / 2;
    int after  = kernel - before - 1;
    Mask output{source.width, source.height, std::vector<uint8_t>(source.data.size(), amount > 0 ? 0 : 255)};
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            uint8_t value = amount > 0 ? 0 : 255;
            for (int ky = -before; ky <= after; ++ky) {
                for (int kx = -before; kx <= after; ++kx) {
                    int sx         = x + kx;
                    int sy         = y + ky;
                    uint8_t sample = (sx >= 0 && sx < source.width && sy >= 0 && sy < source.height)
                                         ? source.data[static_cast<size_t>(sy) * source.width + sx]
                                         : 0;
                    value          = amount > 0 ? std::max(value, sample) : std::min(value, sample);
                }
            }
            output.data[static_cast<size_t>(y) * source.width + x] = value;
        }
    }
    return output;
}

static Mask gaussian_blur_mask(const Mask& source, int radius) {
    if (radius <= 0) {
        return source;
    }
    float sigma       = static_cast<float>(radius);
    int kernel_radius = std::max(1, static_cast<int>(std::ceil(2.5f * sigma)));
    std::vector<float> kernel(kernel_radius * 2 + 1);
    float sum = 0.f;
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        float value               = std::exp(-(i * i) / (2.f * sigma * sigma));
        kernel[i + kernel_radius] = value;
        sum += value;
    }
    for (float& value : kernel) {
        value /= sum;
    }

    std::vector<float> horizontal(source.data.size());
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            float value = 0.f;
            for (int k = -kernel_radius; k <= kernel_radius; ++k) {
                int sx = std::clamp(x + k, 0, source.width - 1);
                value += source.data[static_cast<size_t>(y) * source.width + sx] * kernel[k + kernel_radius];
            }
            horizontal[static_cast<size_t>(y) * source.width + x] = value;
        }
    }

    Mask output{source.width, source.height, std::vector<uint8_t>(source.data.size())};
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            float value = 0.f;
            for (int k = -kernel_radius; k <= kernel_radius; ++k) {
                int sy = std::clamp(y + k, 0, source.height - 1);
                value += horizontal[static_cast<size_t>(sy) * source.width + x] * kernel[k + kernel_radius];
            }
            output.data[static_cast<size_t>(y) * source.width + x] = static_cast<uint8_t>(std::clamp(value, 0.f, 255.f) + 0.5f);
        }
    }
    return output;
}

static std::vector<Mask> make_masks(const std::vector<ADetailerDetection>& detections,
                                    int width,
                                    int height,
                                    const ADetailerParams& params) {
    std::vector<Mask> masks;
    for (const auto& detection : detections) {
        Mask mask = bbox_mask(detection, width, height);
        if (params.x_offset != 0 || params.y_offset != 0) {
            mask = offset_mask(mask, params.x_offset, params.y_offset);
        }
        mask = morphology_mask(mask, params.dilate_erode);
        if (std::any_of(mask.data.begin(), mask.data.end(), [](uint8_t value) { return value != 0; })) {
            masks.push_back(std::move(mask));
        }
    }
    if (params.merge_masks && !masks.empty()) {
        Mask merged{width, height, std::vector<uint8_t>(static_cast<size_t>(width) * height, 0)};
        for (const Mask& mask : masks) {
            for (size_t i = 0; i < merged.data.size(); ++i) {
                merged.data[i] = std::max(merged.data[i], mask.data[i]);
            }
        }
        masks = {std::move(merged)};
    }
    if (params.invert_mask) {
        for (Mask& mask : masks) {
            for (uint8_t& value : mask.data) {
                value = 255 - value;
            }
        }
    }
    return masks;
}

static bool mask_bbox(const Mask& mask, CropRegion* region) {
    int x1 = mask.width;
    int y1 = mask.height;
    int x2 = 0;
    int y2 = 0;
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.data[static_cast<size_t>(y) * mask.width + x] != 0) {
                x1 = std::min(x1, x);
                y1 = std::min(y1, y);
                x2 = std::max(x2, x + 1);
                y2 = std::max(y2, y + 1);
            }
        }
    }
    if (x2 <= x1 || y2 <= y1) {
        return false;
    }
    *region = {x1, y1, x2, y2};
    return true;
}

static CropRegion expand_crop(CropRegion crop,
                              int image_width,
                              int image_height,
                              int padding,
                              int target_width,
                              int target_height) {
    crop.x1 = std::max(0, crop.x1 - padding);
    crop.y1 = std::max(0, crop.y1 - padding);
    crop.x2 = std::min(image_width, crop.x2 + padding);
    crop.y2 = std::min(image_height, crop.y2 + padding);

    float target_aspect = static_cast<float>(target_width) / target_height;
    int width           = crop.x2 - crop.x1;
    int height          = crop.y2 - crop.y1;
    int desired_width   = width;
    int desired_height  = height;
    if (static_cast<float>(width) / height < target_aspect) {
        desired_width = static_cast<int>(std::ceil(height * target_aspect));
    } else {
        desired_height = static_cast<int>(std::ceil(width / target_aspect));
    }
    desired_width  = std::min(desired_width, image_width);
    desired_height = std::min(desired_height, image_height);
    int center_x   = (crop.x1 + crop.x2) / 2;
    int center_y   = (crop.y1 + crop.y2) / 2;
    crop.x1        = std::clamp(center_x - desired_width / 2, 0, image_width - desired_width);
    crop.y1        = std::clamp(center_y - desired_height / 2, 0, image_height - desired_height);
    crop.x2        = crop.x1 + desired_width;
    crop.y2        = crop.y1 + desired_height;
    return crop;
}

static sd_image_t resize_crop_image(const sd_image_t& source, const CropRegion& crop, int width, int height) {
    sd_image_t output = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, nullptr};
    output.data       = static_cast<uint8_t*>(malloc(static_cast<size_t>(width) * height * 3));
    if (output.data == nullptr) {
        return {0, 0, 0, nullptr};
    }
    float scale_x = static_cast<float>(crop.x2 - crop.x1) / width;
    float scale_y = static_cast<float>(crop.y2 - crop.y1) / height;
    for (int y = 0; y < height; ++y) {
        float source_y = crop.y1 + (y + 0.5f) * scale_y - 0.5f;
        for (int x = 0; x < width; ++x) {
            float source_x = crop.x1 + (x + 0.5f) * scale_x - 0.5f;
            for (int c = 0; c < 3; ++c) {
                float value                                               = bilinear_channel(source, source_x, source_y, c);
                output.data[(static_cast<size_t>(y) * width + x) * 3 + c] = static_cast<uint8_t>(std::clamp(value * 255.f, 0.f, 255.f) + 0.5f);
            }
        }
    }
    return output;
}

static sd_image_t resize_crop_mask(const Mask& source, const CropRegion& crop, int width, int height) {
    sd_image_t output = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1, nullptr};
    output.data       = static_cast<uint8_t*>(malloc(static_cast<size_t>(width) * height));
    if (output.data == nullptr) {
        return {0, 0, 0, nullptr};
    }
    float scale_x = static_cast<float>(crop.x2 - crop.x1) / width;
    float scale_y = static_cast<float>(crop.y2 - crop.y1) / height;
    for (int y = 0; y < height; ++y) {
        int source_y = std::clamp(crop.y1 + static_cast<int>((y + 0.5f) * scale_y), 0, source.height - 1);
        for (int x = 0; x < width; ++x) {
            int source_x                                    = std::clamp(crop.x1 + static_cast<int>((x + 0.5f) * scale_x), 0, source.width - 1);
            output.data[static_cast<size_t>(y) * width + x] = source.data[static_cast<size_t>(source_y) * source.width + source_x];
        }
    }
    return output;
}

static sd_image_t copy_as_rgb(const sd_image_t& source) {
    CropRegion full{0, 0, static_cast<int>(source.width), static_cast<int>(source.height)};
    return resize_crop_image(source, full, source.width, source.height);
}

static void composite_crop(sd_image_t* destination,
                           const sd_image_t& generated,
                           const Mask& feather_mask,
                           const CropRegion& crop) {
    int crop_width  = crop.x2 - crop.x1;
    int crop_height = crop.y2 - crop.y1;
    float scale_x   = static_cast<float>(generated.width) / crop_width;
    float scale_y   = static_cast<float>(generated.height) / crop_height;
    for (int y = crop.y1; y < crop.y2; ++y) {
        float source_y = (y - crop.y1 + 0.5f) * scale_y - 0.5f;
        for (int x = crop.x1; x < crop.x2; ++x) {
            float alpha = feather_mask.data[static_cast<size_t>(y) * feather_mask.width + x] / 255.f;
            if (alpha <= 0.f) {
                continue;
            }
            float source_x           = (x - crop.x1 + 0.5f) * scale_x - 0.5f;
            size_t destination_index = (static_cast<size_t>(y) * destination->width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float generated_value                    = bilinear_channel(generated, source_x, source_y, c) * 255.f;
                float original_value                     = destination->data[destination_index + c];
                float blended                            = original_value * (1.f - alpha) + generated_value * alpha;
                destination->data[destination_index + c] = static_cast<uint8_t>(std::clamp(blended, 0.f, 255.f) + 0.5f);
            }
        }
    }
}

static std::vector<std::string> split_prompts(const std::string& prompt) {
    std::vector<std::string> result;
    size_t start = 0;
    while (true) {
        size_t separator = prompt.find("[SEP]", start);
        result.push_back(trim_copy(prompt.substr(start, separator == std::string::npos ? std::string::npos : separator - start)));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 5;
    }
    return result;
}

static std::string resolve_prompt(const char* prompt_template, const char* base_prompt, int index) {
    std::string base     = base_prompt == nullptr ? "" : base_prompt;
    std::string templ    = prompt_template == nullptr ? "" : prompt_template;
    auto prompts         = split_prompts(templ);
    std::string selected = prompts[std::min(index, static_cast<int>(prompts.size()) - 1)];
    if (selected.empty()) {
        return base;
    }
    size_t position = 0;
    while ((position = selected.find("[PROMPT]", position)) != std::string::npos) {
        selected.replace(position, 8, base);
        position += base.size();
    }
    return selected;
}

static bool prompt_is_skip(const std::string& prompt) {
    return trim_copy(prompt) == "[SKIP]";
}

static bool validate_params(const ADetailerParams& params) {
    return params.input_size >= 32 && params.input_size % 32 == 0 &&
           params.confidence >= 0.f && params.confidence <= 1.f &&
           params.nms_threshold >= 0.f && params.nms_threshold <= 1.f &&
           params.max_detections > 0 && params.mask_k_largest >= 0 &&
           params.mask_min_ratio >= 0.f && params.mask_max_ratio <= 1.f &&
           params.mask_min_ratio <= params.mask_max_ratio && params.mask_blur >= 0 &&
           params.inpaint_padding >= 0 && params.inpaint_width > 0 && params.inpaint_height > 0 &&
           params.denoising_strength >= 0.f && params.denoising_strength <= 1.f && params.steps >= 0 &&
           (params.cfg_scale < 0.f || std::isfinite(params.cfg_scale));
}

static std::vector<std::string> parse_class_names(const std::string& value, int class_count) {
    std::vector<std::string> names(static_cast<size_t>(class_count));
    try {
        nlohmann::json parsed = nlohmann::json::parse(value);
        if (parsed.is_array()) {
            for (size_t i = 0; i < names.size() && i < parsed.size(); ++i) {
                if (parsed[i].is_string()) {
                    names[i] = parsed[i].get<std::string>();
                }
            }
        } else if (parsed.is_object()) {
            for (const auto& item : parsed.items()) {
                int class_id = 0;
                if (parse_int(item.key(), &class_id) && class_id >= 0 && class_id < class_count &&
                    item.value().is_string()) {
                    names[static_cast<size_t>(class_id)] = item.value().get<std::string>();
                }
            }
        }
    } catch (const std::exception&) {
    }
    return names;
}

ADetailerGGML::ADetailerGGML(int n_threads,
                             std::string backend_spec,
                             std::string params_backend_spec)
    : n_threads(n_threads > 0 ? n_threads : 1),
      backend_spec(std::move(backend_spec)),
      params_backend_spec(std::move(params_backend_spec)) {
}

ADetailerGGML::~ADetailerGGML() {
    model_manager.reset();
    detector.reset();
}

bool ADetailerGGML::load_from_file(const std::string& detector_path) {
    std::string error;
    if (!backend_manager.init(backend_spec.c_str(), params_backend_spec.c_str(), nullptr, &error)) {
        LOG_ERROR("ADetailer backend config failed: %s", error.c_str());
        return false;
    }
    ggml_backend_t backend        = backend_manager.runtime_backend(SDBackendModule::DETECTOR);
    ggml_backend_t params_backend = backend_manager.params_backend(SDBackendModule::DETECTOR);
    if (backend == nullptr || params_backend == nullptr) {
        LOG_ERROR("failed to initialize detector backend");
        return false;
    }

    model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(n_threads);
    model_manager->set_enable_mmap(false);
    ModelLoader& loader = model_manager->loader();
    if (!loader.init_from_file(detector_path)) {
        LOG_ERROR("failed to load ADetailer detector: '%s'", detector_path.c_str());
        return false;
    }

    detector = std::make_shared<YOLOv8Runner>(backend, loader.get_tensor_storage_map(), model_manager);
    if (!detector || !detector->model || !detector->config.valid) {
        LOG_ERROR("unsupported YOLOv8 detection weights: '%s'", detector_path.c_str());
        return false;
    }

    class_names.assign(static_cast<size_t>(detector->config.num_classes), "");
    auto names_item = loader.get_metadata().find("yolov8.names");
    if (names_item != loader.get_metadata().end()) {
        class_names = parse_class_names(names_item->second, detector->config.num_classes);
    }

    std::map<std::string, ggml_tensor*> tensors;
    detector->get_param_tensors(tensors);
    if (!model_manager->register_param_tensors("YOLOv8",
                                               std::move(tensors),
                                               backend_manager.params_backend_is_disk(SDBackendModule::DETECTOR)
                                                   ? ModelManager::ResidencyMode::Disk
                                                   : ModelManager::ResidencyMode::ParamBackend,
                                               backend,
                                               params_backend) ||
        !model_manager->validate_registered_tensors()) {
        LOG_ERROR("failed to register YOLOv8 detector tensors");
        return false;
    }
    return true;
}

std::vector<ADetailerDetection> ADetailerGGML::predict(sd_image_t image,
                                                       const ADetailerParams& params) {
    LetterboxInput input  = make_letterbox_input(image, params.input_size);
    int64_t start         = ggml_time_ms();
    sd::Tensor<float> raw = detector->compute(n_threads, input.tensor);
    detector->free_compute_buffer();
    if (raw.empty()) {
        LOG_ERROR("YOLOv8 detector inference failed");
        return {};
    }
    auto detections = decode_detections(raw, detector->config, input, image, params);
    LOG_INFO("ADetailer detected %zu object(s), taking %.2fs",
             detections.size(),
             (ggml_time_ms() - start) / 1000.f);
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        std::string object    = "class_" + std::to_string(detection.class_id);
        if (detection.class_id >= 0 && static_cast<size_t>(detection.class_id) < class_names.size() &&
            !class_names[static_cast<size_t>(detection.class_id)].empty()) {
            object = class_names[static_cast<size_t>(detection.class_id)];
        }
        LOG_INFO("ADetailer detection %zu: object=%s, class_id=%d, confidence=%.3f, bbox=[x1=%.1f, y1=%.1f, x2=%.1f, y2=%.1f]",
                 i + 1,
                 object.c_str(),
                 detection.class_id,
                 detection.confidence,
                 detection.x1,
                 detection.y1,
                 detection.x2,
                 detection.y2);
    }
    return detections;
}

static bool parse_adetailer_params(const sd_adetailer_params_t& public_params,
                                   const sd_img_gen_params_t& inpaint_params,
                                   ADetailerParams* params) {
    if (params == nullptr) {
        return false;
    }
    *params                    = {};
    params->prompt             = public_params.prompt;
    params->negative_prompt    = public_params.negative_prompt;
    params->inpaint_width      = inpaint_params.width;
    params->inpaint_height     = inpaint_params.height;
    params->denoising_strength = inpaint_params.strength;
    const char* extra_ad_args  = public_params.extra_ad_args;
    if (extra_ad_args == nullptr || extra_ad_args[0] == '\0') {
        return validate_params(*params);
    }

    std::stringstream stream(extra_ad_args);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim_copy(item);
        if (item.empty()) {
            continue;
        }
        size_t equals = item.find('=');
        if (equals == std::string::npos) {
            LOG_ERROR("invalid --extra-ad-args item '%s'; expected key=value", item.c_str());
            return false;
        }
        std::string key   = lower_copy(trim_copy(item.substr(0, equals)));
        std::string value = trim_copy(item.substr(equals + 1));
        bool ok           = true;
        if (key == "input_size") {
            ok = parse_int(value, &params->input_size);
        } else if (key == "confidence") {
            ok = parse_float(value, &params->confidence);
        } else if (key == "nms" || key == "nms_threshold") {
            ok = parse_float(value, &params->nms_threshold);
        } else if (key == "max_detections") {
            ok = parse_int(value, &params->max_detections);
        } else if (key == "mask_k_largest") {
            ok = parse_int(value, &params->mask_k_largest);
        } else if (key == "mask_min_ratio") {
            ok = parse_float(value, &params->mask_min_ratio);
        } else if (key == "mask_max_ratio") {
            ok = parse_float(value, &params->mask_max_ratio);
        } else if (key == "dilate_erode") {
            ok = parse_int(value, &params->dilate_erode);
        } else if (key == "x_offset") {
            ok = parse_int(value, &params->x_offset);
        } else if (key == "y_offset") {
            ok = parse_int(value, &params->y_offset);
        } else if (key == "merge_masks") {
            ok = parse_bool(value, &params->merge_masks);
        } else if (key == "invert_mask") {
            ok = parse_bool(value, &params->invert_mask);
        } else if (key == "mask_blur") {
            ok = parse_int(value, &params->mask_blur);
        } else if (key == "inpaint_padding") {
            ok = parse_int(value, &params->inpaint_padding);
        } else if (key == "inpaint_width") {
            ok = parse_int(value, &params->inpaint_width);
        } else if (key == "inpaint_height") {
            ok = parse_int(value, &params->inpaint_height);
        } else if (key == "denoising_strength") {
            ok = parse_float(value, &params->denoising_strength);
        } else if (key == "steps") {
            ok = parse_int(value, &params->steps);
        } else if (key == "cfg_scale") {
            ok = parse_float(value, &params->cfg_scale);
        } else if (key == "sample_method") {
            params->sample_method = str_to_sample_method(value.c_str());
            ok                    = params->sample_method != SAMPLE_METHOD_COUNT;
        } else if (key == "scheduler") {
            params->scheduler = str_to_scheduler(value.c_str());
            ok                = params->scheduler != SCHEDULER_COUNT;
        } else if (key == "sort_by") {
            std::string sort = lower_copy(value);
            if (sort == "none") {
                params->sort_by = ADETAILER_SORT_NONE;
            } else if (sort == "left_to_right" || sort == "left-to-right") {
                params->sort_by = ADETAILER_SORT_LEFT_TO_RIGHT;
            } else if (sort == "center_to_edge" || sort == "center-to-edge") {
                params->sort_by = ADETAILER_SORT_CENTER_TO_EDGE;
            } else if (sort == "area") {
                params->sort_by = ADETAILER_SORT_AREA;
            } else {
                ok = false;
            }
        } else if (key == "mask_mode") {
            std::string mode = lower_copy(value);
            if (mode == "none") {
                params->merge_masks = false;
                params->invert_mask = false;
            } else if (mode == "merge") {
                params->merge_masks = true;
                params->invert_mask = false;
            } else if (mode == "merge_invert" || mode == "merge-invert") {
                params->merge_masks = true;
                params->invert_mask = true;
            } else {
                ok = false;
            }
        } else {
            LOG_ERROR("unknown --extra-ad-args key '%s'", key.c_str());
            return false;
        }
        if (!ok) {
            LOG_ERROR("invalid --extra-ad-args value '%s=%s'", key.c_str(), value.c_str());
            return false;
        }
    }
    if (!validate_params(*params)) {
        LOG_ERROR("invalid --extra-ad-args parameter range");
        return false;
    }
    return true;
}

adetailer_ctx_t* new_adetailer_ctx(const char* detector_path,
                                   int n_threads,
                                   const char* backend,
                                   const char* params_backend) {
    if (detector_path == nullptr || detector_path[0] == '\0') {
        return nullptr;
    }
    auto* context = static_cast<adetailer_ctx_t*>(calloc(1, sizeof(adetailer_ctx_t)));
    if (context == nullptr) {
        return nullptr;
    }
    context->detailer = new ADetailerGGML(n_threads, SAFE_STR(backend), SAFE_STR(params_backend));
    if (context->detailer == nullptr || !context->detailer->load_from_file(detector_path)) {
        delete context->detailer;
        free(context);
        return nullptr;
    }
    return context;
}

void free_adetailer_ctx(adetailer_ctx_t* context) {
    if (context == nullptr) {
        return;
    }
    delete context->detailer;
    context->detailer = nullptr;
    free(context);
}

bool adetail_image(adetailer_ctx_t* context,
                   sd_ctx_t* sd_ctx,
                   sd_image_t input_image,
                   const sd_adetailer_params_t* public_params,
                   const sd_img_gen_params_t* inpaint_params,
                   sd_image_t** images_out,
                   int* num_images_out) {
    if (images_out != nullptr) {
        *images_out = nullptr;
    }
    if (num_images_out != nullptr) {
        *num_images_out = 0;
    }
    if (context == nullptr || context->detailer == nullptr || sd_ctx == nullptr || public_params == nullptr ||
        inpaint_params == nullptr || input_image.data == nullptr) {
        return false;
    }

    ADetailerParams params;
    if (!parse_adetailer_params(*public_params, *inpaint_params, &params)) {
        return false;
    }

    auto detections = context->detailer->predict(input_image, params);
    filter_and_sort_detections(&detections, input_image.width, input_image.height, params);
    auto masks = make_masks(detections, input_image.width, input_image.height, params);

    sd_image_t current = copy_as_rgb(input_image);
    if (current.data == nullptr) {
        return false;
    }

    int applied = 0;
    for (size_t i = 0; i < masks.size(); ++i) {
        std::string prompt          = resolve_prompt(params.prompt, inpaint_params->prompt, static_cast<int>(i));
        std::string negative_prompt = resolve_prompt(params.negative_prompt,
                                                     inpaint_params->negative_prompt,
                                                     static_cast<int>(i));
        if (prompt_is_skip(prompt)) {
            continue;
        }

        CropRegion crop;
        if (!mask_bbox(masks[i], &crop)) {
            continue;
        }
        crop                   = expand_crop(crop,
                                             current.width,
                                             current.height,
                                             params.inpaint_padding,
                                             params.inpaint_width,
                                             params.inpaint_height);
        sd_image_t local_image = resize_crop_image(current, crop, params.inpaint_width, params.inpaint_height);
        sd_image_t local_mask  = resize_crop_mask(masks[i], crop, params.inpaint_width, params.inpaint_height);
        if (local_image.data == nullptr || local_mask.data == nullptr) {
            free(local_image.data);
            free(local_mask.data);
            free(current.data);
            return false;
        }

        sd_img_gen_params_t generation = *inpaint_params;
        generation.prompt              = prompt.c_str();
        generation.negative_prompt     = negative_prompt.c_str();
        generation.init_image          = local_image;
        generation.mask_image          = local_mask;
        generation.width               = params.inpaint_width;
        generation.height              = params.inpaint_height;
        generation.strength            = params.denoising_strength;
        generation.seed                = inpaint_params->seed + static_cast<int64_t>(i);
        generation.batch_count         = 1;
        generation.ref_images          = nullptr;
        generation.ref_images_count    = 0;
        generation.control_image       = {};
        generation.pm_params           = {};
        generation.pulid_params        = {};
        generation.hires.enabled       = false;
        if (params.steps > 0) {
            generation.sample_params.sample_steps        = params.steps;
            generation.sample_params.custom_sigmas       = nullptr;
            generation.sample_params.custom_sigmas_count = 0;
        }
        if (params.cfg_scale >= 0.f) {
            generation.sample_params.guidance.txt_cfg = params.cfg_scale;
        }
        if (params.sample_method != SAMPLE_METHOD_COUNT) {
            generation.sample_params.sample_method = params.sample_method;
        }
        if (params.scheduler != SCHEDULER_COUNT) {
            generation.sample_params.scheduler = params.scheduler;
        }

        sd_image_t* generated = nullptr;
        int generated_count   = 0;
        bool success          = generate_image(sd_ctx, &generation, &generated, &generated_count);
        free(local_image.data);
        free(local_mask.data);
        if (!success || generated_count <= 0 || generated == nullptr || generated[0].data == nullptr) {
            free_sd_images(generated, generated_count);
            free(current.data);
            return false;
        }

        Mask feather = gaussian_blur_mask(masks[i], params.mask_blur);
        composite_crop(&current, generated[0], feather, crop);
        free_sd_images(generated, generated_count);
        ++applied;
    }

    auto* outputs = static_cast<sd_image_t*>(calloc(1, sizeof(sd_image_t)));
    if (outputs == nullptr) {
        free(current.data);
        return false;
    }
    outputs[0] = current;
    if (images_out != nullptr) {
        *images_out = outputs;
    } else {
        free_sd_images(outputs, 1);
    }
    if (num_images_out != nullptr) {
        *num_images_out = 1;
    }
    LOG_INFO("ADetailer applied %d mask(s)", applied);
    return true;
}
