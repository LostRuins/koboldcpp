#ifndef __SD_DETAILER_H__
#define __SD_DETAILER_H__

#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "model/detector/yolov8.h"
#include "model_manager.h"
#include "stable-diffusion.h"

struct ADetailerDetection {
    float x1         = 0.f;
    float y1         = 0.f;
    float x2         = 0.f;
    float y2         = 0.f;
    float confidence = 0.f;
    int class_id     = 0;
};

enum ADetailerSort {
    ADETAILER_SORT_NONE,
    ADETAILER_SORT_LEFT_TO_RIGHT,
    ADETAILER_SORT_CENTER_TO_EDGE,
    ADETAILER_SORT_AREA,
};

struct ADetailerParams {
    const char* prompt            = nullptr;
    const char* negative_prompt   = nullptr;
    int input_size                = 640;
    float confidence              = 0.3f;
    float nms_threshold           = 0.45f;
    int max_detections            = 100;
    int mask_k_largest            = 0;
    float mask_min_ratio          = 0.f;
    float mask_max_ratio          = 1.f;
    int dilate_erode              = 4;
    int x_offset                  = 0;
    int y_offset                  = 0;
    bool merge_masks              = false;
    bool invert_mask              = false;
    int mask_blur                 = 4;
    int inpaint_padding           = 32;
    int inpaint_width             = 512;
    int inpaint_height            = 512;
    float denoising_strength      = 0.4f;
    int steps                     = 0;
    float cfg_scale               = -1.f;
    sample_method_t sample_method = SAMPLE_METHOD_COUNT;
    scheduler_t scheduler         = SCHEDULER_COUNT;
    ADetailerSort sort_by         = ADETAILER_SORT_NONE;
};

struct ADetailerGGML {
    SDBackendManager backend_manager;
    std::shared_ptr<ModelManager> model_manager;
    std::shared_ptr<YOLOv8Runner> detector;
    std::vector<std::string> class_names;
    int n_threads = 1;
    std::string backend_spec;
    std::string params_backend_spec;

    ADetailerGGML(int n_threads,
                  std::string backend_spec,
                  std::string params_backend_spec);
    ~ADetailerGGML();

    bool load_from_file(const std::string& detector_path);
    std::vector<ADetailerDetection> predict(sd_image_t image,
                                            const ADetailerParams& params);
};

#endif  // __SD_DETAILER_H__
