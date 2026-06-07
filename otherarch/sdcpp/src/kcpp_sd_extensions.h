#ifndef __KCPP_SD_EXTENSIONS_H__
#define __KCPP_SD_EXTENSIONS_H__

#include "stable-diffusion.h"
#include <vector>

namespace kcpp_sd {

    struct model_info {
        bool is_chroma;
        bool is_flux1;
        bool is_flux2;
        bool is_kontext;
        bool is_qwenimg;
        bool is_sd1;
        bool is_sd2;
        bool is_sdxs;
        bool is_sdxl;
        bool is_wan;
        bool is_zimage;
        bool is_ltx;
        int vae_scale_factor;
        int spatial_multiple;
    };

    model_info get_model_info(sd_ctx_t* ctx);

    void SetCircularAxesAll(sd_ctx_t* ctx, bool circular_x, bool circular_y);

    void set_lora_cache(sd_ctx_t *ctx, bool enable);

    void apply_loras(sd_ctx_t *ctx, const std::vector<sd_lora_t>& lora_specs);

    void set_sd_quiet(bool quiet);

    void set_sd_log_level(int log);

}

#endif
