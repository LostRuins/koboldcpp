#ifndef __KCPP_SD_EXTENSIONS_H__
#define __KCPP_SD_EXTENSIONS_H__

#include "stable-diffusion.h"
#include <vector>

namespace kcpp_sd {

    struct sd_model_info {
        int version;
        int vae_scale_factor;
        int spatial_multiple;
    };

    sd_model_info sd_get_model_info(sd_ctx_t* ctx);

    void set_lora_cache(sd_ctx_t *ctx, bool enable);

    void apply_loras(sd_ctx_t *ctx, const std::vector<sd_lora_t>& lora_specs);

    void set_sd_quiet(bool quiet);

    void set_sd_log_level(int log);

}

#endif
