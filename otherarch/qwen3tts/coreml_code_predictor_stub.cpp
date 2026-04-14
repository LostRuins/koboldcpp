#include "coreml_code_predictor.h"

namespace qwen3_tts {

CoreMLCodePredictor::CoreMLCodePredictor() {}
CoreMLCodePredictor::~CoreMLCodePredictor() {}

bool CoreMLCodePredictor::load(const std::string &, int32_t) {
    return false;
}

void CoreMLCodePredictor::unload() {}

bool CoreMLCodePredictor::is_loaded() const {
    return false;
}

const std::string & CoreMLCodePredictor::get_error() const {
    static const std::string err = "CoreML predictor only supported on Apple platforms";
    return err;
}

bool CoreMLCodePredictor::predict_step(int32_t,
                                       const float *,
                                       int32_t,
                                       int32_t,
                                       std::vector<float> &) {
    return false;
}

} // namespace qwen3_tts
