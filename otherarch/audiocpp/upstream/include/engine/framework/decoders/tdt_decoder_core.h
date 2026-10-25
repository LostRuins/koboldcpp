#pragma once

#include "engine/framework/decoders/tdt_types.h"

namespace engine::decoders {

class TdtDecoderCore {
public:
    virtual ~TdtDecoderCore() = default;

    virtual void reset_state() = 0;
    virtual void predict_start(int32_t blank_id) = 0;
    virtual void predict_token(int32_t token_id) = 0;
    virtual TdtJointStep joint_step_argmax(const float * encoder_frame) = 0;
    virtual TdtPredictorStateSnapshot snapshot_state() const = 0;
    virtual void restore_state(const TdtPredictorStateSnapshot & snapshot) = 0;
};

}  // namespace engine::decoders
