#ifndef __SD_MODEL_DEFINITION_H__
#define __SD_MODEL_DEFINITION_H__

#include <limits>
#include "model.h"
#include "stable-diffusion.h"

class TokenizerConfig;
struct VAE;
namespace sd::model_builders {
    struct Context;
    struct CoreRunners;
    struct VAEOptions;
}

struct ModelSamplingPolicy {
    prediction_t default_prediction       = EPS_PRED;
    float default_flow_shift              = std::numeric_limits<float>::infinity();
    sample_method_t default_sample_method = EULER_A_SAMPLE_METHOD;
    scheduler_t default_scheduler         = DISCRETE_SCHEDULER;

    bool probe_v_prediction = false;

    prediction_t (*prediction_from_weights)(const String2TensorStorage&) = nullptr;
    float (*flow_shift_from_weights)(const String2TensorStorage&)        = nullptr;
};

struct ModelGeometry {
    int diffusion_down_factor = 8;
    int latent_channels       = 4;
    int image_channels        = 3;
};

struct ModelDefinition {
    SDVersion version = VERSION_COUNT;
    const char* name  = nullptr;
    ModelTraits traits;
    ModelSamplingPolicy sampling;
    ModelGeometry geometry;
    const char* ref_image_preset                                                                                = "default";
    void (*configure_vae)(const sd::model_builders::Context&, const sd::model_builders::VAEOptions&, VAE&)      = nullptr;
    bool (*build)(const sd::model_builders::Context&, const TokenizerConfig&, sd::model_builders::CoreRunners&) = nullptr;
};

const ModelDefinition* find_model_definition(SDVersion version);

#endif  // __SD_MODEL_DEFINITION_H__
