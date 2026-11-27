#ifndef SD_OPTIONS_H
#define SD_OPTIONS_H

#include "api.h"
#include "config.h"

namespace sd::api {
    enum Operation {
        Context     = 1,
        Image       = 2,
        Video       = 4,
        Upscaler    = 8,
        Upscale     = 16,
        VidUpscaler = 32,
        VidUpscale  = 64,
        ADetailer   = 128,
        ADetail     = 256
    };

    sd_status_t resolve(const sd_opt_t* params, Operation operation, OptConfig& config);
    bool is_set(const sd_opt_t* params, const char* key);
    bool has_prefix(const sd_opt_t* params, const char* prefix);
}

#endif  // SD_OPTIONS_H
