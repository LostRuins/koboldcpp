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
    VERSION_WAN2_2_S2V,
    VERSION_LINGBOT_VIDEO,
    VERSION_QWEN_IMAGE,
    VERSION_QWEN_IMAGE_LAYERED,
    VERSION_QWEN_IMAGE_2_1,
    VERSION_HUNYUAN_VIDEO,
    VERSION_ANIMA,
    VERSION_FLUX2,
    VERSION_FLUX2_KLEIN,
    VERSION_LTXAV,
    VERSION_MINIMAX_H3,
    VERSION_HIDREAM_O1,
    VERSION_Z_IMAGE,
    VERSION_BOOGU_IMAGE,
    VERSION_OVIS_IMAGE,
    VERSION_ERNIE_IMAGE,
    VERSION_LENS,
    VERSION_MINIT2I,
    VERSION_LONGCAT,
    VERSION_PID,
    VERSION_IDEOGRAM4,
    VERSION_SEFI_IMAGE,
    VERSION_KREA2,
    VERSION_MAGE_FLOW,
    VERSION_SENSENOVA_U1_5,
    VERSION_LLADA_IMAGE,
    VERSION_ESRGAN,
    VERSION_PIXART,
    VERSION_MING_IMAGE,
    VERSION_Z_IMAGE_L2P,
    VERSION_IRIS,
    VERSION_FLUX3,
    VERSION_MOSS_IMAGE,
    VERSION_FIBO2,
    VERSION_COUNT,
};

enum class ModelVAEKind { KL,
                          FLUX,
                          FLUX2,
                          WAN,
                          HUNYUAN,
                          LTXAV,
                          MINIMAX_H3,
                          MAGE };

struct ModelTraits {
    bool is_dit                    = false;
    bool supports_video_generation = false;
    bool is_inpaint                = false;
    bool is_unet_edit              = false;
    bool is_control                = false;
    ModelVAEKind vae_kind          = ModelVAEKind::KL;
    bool video_tae                 = false;
    bool fake_vae                  = false;
    bool vae_format_override       = false;
};

const ModelTraits* find_model_traits(SDVersion version);
const char* sd_version_name(SDVersion version);

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

static inline bool sd_version_is_flux3(SDVersion version) {
    return version == VERSION_FLUX3;
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

static inline bool sd_version_is_minimax_h3(SDVersion version) {
    return version == VERSION_MINIMAX_H3;
}

static inline bool sd_version_is_wan(SDVersion version) {
    if (version == VERSION_WAN2 || version == VERSION_WAN2_2_I2V || version == VERSION_WAN2_2_TI2V || version == VERSION_WAN2_2_S2V) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_lingbot_video(SDVersion version) {
    if (version == VERSION_LINGBOT_VIDEO) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_qwen_image(SDVersion version) {
    if (version == VERSION_QWEN_IMAGE || version == VERSION_QWEN_IMAGE_LAYERED || version == VERSION_QWEN_IMAGE_2_1) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_hunyuan_video(SDVersion version) {
    if (version == VERSION_HUNYUAN_VIDEO) {
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

static inline bool sd_version_is_z_image_l2p(SDVersion version) {
    return version == VERSION_Z_IMAGE_L2P;
}

static inline bool sd_version_is_llada_image(SDVersion version) {
    if (version == VERSION_LLADA_IMAGE) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_boogu_image(SDVersion version) {
    if (version == VERSION_BOOGU_IMAGE) {
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

static inline bool sd_version_is_minit2i(SDVersion version) {
    if (version == VERSION_MINIT2I) {
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

static inline bool sd_version_is_sefi_image(SDVersion version) {
    if (version == VERSION_SEFI_IMAGE) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_krea2(SDVersion version) {
    if (version == VERSION_KREA2) {
        return true;
    }
    return false;
}

static inline bool sd_version_is_mage_flow(SDVersion version) {
    return version == VERSION_MAGE_FLOW;
}

static inline bool sd_version_is_sensenova_u1(SDVersion version) {
    return version == VERSION_SENSENOVA_U1_5;
}

static inline bool sd_version_is_pixart(SDVersion version) {
    return version == VERSION_PIXART;
}

static inline bool sd_version_supports_video_generation(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->supports_video_generation;
}

static inline bool sd_version_supports_image_generation(SDVersion version) {
    return !sd_version_supports_video_generation(version);
}

static inline bool sd_version_uses_flux_vae(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->vae_kind == ModelVAEKind::FLUX;
}

static inline bool sd_version_uses_flux2_vae(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->vae_kind == ModelVAEKind::FLUX2;
}

static inline bool sd_version_uses_wan_vae(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->vae_kind == ModelVAEKind::WAN;
}

static inline bool sd_version_uses_hunyuan_video_vae(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->vae_kind == ModelVAEKind::HUNYUAN;
}

static inline bool sd_version_is_inpaint(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->is_inpaint;
}

static inline bool sd_version_is_dit(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->is_dit;
}

static inline bool sd_version_is_unet_edit(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->is_unet_edit;
}

static inline bool sd_version_is_control(SDVersion version) {
    const auto* traits = find_model_traits(version);
    return traits && traits->is_control;
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
