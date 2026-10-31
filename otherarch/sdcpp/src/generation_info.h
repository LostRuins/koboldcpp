#ifndef __SD_GENERATION_INFO_H__
#define __SD_GENERATION_INFO_H__

#include <vector>

#include "json.hpp"
#include "stable-diffusion.h"

namespace sd::pipeline {
    struct GenerationRequest;
}

namespace sd {
    nlohmann::json generation_info(const pipeline::GenerationRequest& request, const sd_img_gen_params_t& params);
    nlohmann::json generation_info(const pipeline::GenerationRequest& request, const sd_vid_gen_params_t& params);
    nlohmann::json adetailer_generation_info(const sd_adetailer_params_t& params, const sd_img_gen_params_t& inpaint_params);
    nlohmann::json upscale_generation_info(uint32_t upscale_factor);
    char* copy_generation_info(const nlohmann::json& info);
    // Consumes images even on failure.
    bool make_image_generation_result(sd_image_t* images, int count, const std::vector<nlohmann::json>& infos, sd_image_generation_result_t** result_out);
}

#endif  // __SD_GENERATION_INFO_H__
