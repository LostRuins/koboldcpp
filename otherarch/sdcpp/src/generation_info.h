#ifndef __SD_GENERATION_INFO_H__
#define __SD_GENERATION_INFO_H__

#include <vector>

#include "config.h"
#include "json.hpp"
#include "stable-diffusion.h"

namespace sd::pipeline {
    struct GenerationRequest;
}

namespace sd {
    nlohmann::json generation_info(const pipeline::GenerationRequest& request, const sd::ImageConfig& params);
    nlohmann::json generation_info(const pipeline::GenerationRequest& request, const sd::VideoConfig& params);
    nlohmann::json adetailer_generation_info(const sd::ADetailConfig& params, const sd::ImageConfig& inpaint_params);
    nlohmann::json upscale_generation_info(uint32_t upscale_factor);

}

#endif  // __SD_GENERATION_INFO_H__
