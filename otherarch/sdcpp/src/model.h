#ifndef __MODEL_H__
#define __MODEL_H__

#include <string>
#include <utility>
#include <vector>

#include "core/ordered_map.hpp"
#include "ggml-backend.h"
#include "ggml.h"
#include "model_io/tensor_storage.h"

enum SDVersion {
    VERSION_SD1,
    VERSION_SD1_INPAINT,
    VERSION_SD1_PIX2PIX,
    VERSION_SD1_TINY_UNET,
    VERSION_SD2,
    VERSION_SD2_INPAINT,
    VERSION_SD2_TINY_UNET,
    VERSION_SDXS_512_DS,
    VERSION_SDXS_09,
    VERSION_SDXL,
    VERSION_SDXL_INPAINT,
    VERSION_SDXL_PIX2PIX,
    VERSION_SDXL_VEGA,
    VERSION_SDXL_SSD1B,
    VERSION_SVD,
    VERSION_SD3,
    VERSION_FLUX,
    VERSION_FLUX_FILL,
    VERSION_FLUX_CONTROLS,
    VERSION_FLEX_2,
    VERSION_CHROMA_RADIANCE,
    VERSION_WAN2,
    VERSION_WAN2_2_I2V,
    VERSION_WAN2_2_TI2V,
    VERSION_QWEN_IMAGE,
    VERSION_ANIMA,
    VERSION_FLUX2,
    VERSION_FLUX2_KLEIN,
    VERSION_LTXAV,
    VERSION_HIDREAM_O1,
    VERSION_Z_IMAGE,
    VERSION_OVIS_IMAGE,
    VERSION_ERNIE_IMAGE,
    VERSION_LENS,
    VERSION_LONGCAT,
    VERSION_PID,
    VERSION_IDEOGRAM4,
    VERSION_COUNT,
};

static inline bool sd_version_is_sd1(SDVersion version) {
    if (version == VERSION_SD1 || version == VERSION_SD1_INPAINT || version == VERSION_SD1_PIX2PIX || version == VERSION_SD1_TINY_UNET || version == VERSION_SDXS_512_DS) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_sd2(SDVersion version) {
    if (version == VERSION_SD2 || version == VERSION_SD2_INPAINT || version == VERSION_SD2_TINY_UNET || version == VERSION_SDXS_09) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_sdxl(SDVersion version) {
    if (version == VERSION_SDXL || version == VERSION_SDXL_INPAINT || version == VERSION_SDXL_PIX2PIX || version == VERSION_SDXL_SSD1B || version == VERSION_SDXL_VEGA) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_unet(SDVersion version) {
    if (sd_version_is_sd1(version) ||
        sd_version_is_sd2(version) ||
        sd_version_is_sdxl(version)) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_sd3(SDVersion version) {
    if (version == VERSION_SD3) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_flux(SDVersion version) {
    if (version == VERSION_FLUX ||
        version == VERSION_FLUX_FILL ||
        version == VERSION_FLUX_CONTROLS ||
        version == VERSION_FLEX_2 ||
        version == VERSION_OVIS_IMAGE ||
        version == VERSION_CHROMA_RADIANCE) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_flux2(SDVersion version) {
    if (version == VERSION_FLUX2 || version == VERSION_FLUX2_KLEIN) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_ltxav(SDVersion version) {
    if (version == VERSION_LTXAV) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_wan(SDVersion version) {
    if (version == VERSION_WAN2 || version == VERSION_WAN2_2_I2V || version == VERSION_WAN2_2_TI2V) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_qwen_image(SDVersion version) {
    if (version == VERSION_QWEN_IMAGE) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_anima(SDVersion version) {
    if (version == VERSION_ANIMA) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_z_image(SDVersion version) {
    if (version == VERSION_Z_IMAGE) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_longcat(SDVersion version) {
    if (version == VERSION_LONGCAT) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_ernie_image(SDVersion version) {
    if (version == VERSION_ERNIE_IMAGE) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_lens(SDVersion version) {
    if (version == VERSION_LENS) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_pid(SDVersion version) {
    if (version == VERSION_PID) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_ideogram4(SDVersion version) {
    if (version == VERSION_IDEOGRAM4) {
        return true;
    }
    return false;
}

static inline bool sd_version_uses_flux2_vae(SDVersion version) {
    if (sd_version_is_flux2(version) || sd_version_is_ernie_image(version) || sd_version_is_lens(version) || sd_version_is_ideogram4(version)) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_inpaint(SDVersion version) {
    if (version == VERSION_SD1_INPAINT ||
        version == VERSION_SD2_INPAINT ||
        version == VERSION_SDXL_INPAINT ||
        version == VERSION_FLUX_FILL ||
        version == VERSION_FLEX_2) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_dit(SDVersion version) {
    if (sd_version_is_flux(version) ||
        sd_version_is_flux2(version) ||
        sd_version_is_ltxav(version) ||
        sd_version_is_sd3(version) ||
        sd_version_is_wan(version) ||
        sd_version_is_qwen_image(version) ||
        version == VERSION_HIDREAM_O1 ||
        sd_version_is_anima(version) ||
        sd_version_is_z_image(version) ||
        sd_version_is_ernie_image(version) ||
        sd_version_is_lens(version) ||
        sd_version_is_longcat(version) ||
        sd_version_is_pid(version) ||
        sd_version_is_ideogram4(version)) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_unet_edit(SDVersion version) {
    return version == VERSION_SD1_PIX2PIX || version == VERSION_SDXL_PIX2PIX;
}

static inline bool sd_version_is_control(SDVersion version) {
    return version == VERSION_FLUX_CONTROLS || version == VERSION_FLEX_2;
}

static bool sd_version_is_inpaint_or_unet_edit(SDVersion version) {
    return sd_version_is_unet_edit(version) || sd_version_is_inpaint(version) || sd_version_is_control(version);
}

enum PMVersion {
    PM_VERSION_1,
    PM_VERSION_2,
};

typedef OrderedMap<std::string, TensorStorage> String2TensorStorage;
using TensorTypeRules = std::vector<std::pair<std::string, ggml_type>>;

#endif  // __MODEL_H__
