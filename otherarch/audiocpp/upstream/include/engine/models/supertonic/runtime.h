#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/supertonic/assets.h"
#include "engine/models/supertonic/session.h"

#include <cstddef>
#include <memory>
#include <string>

namespace engine::models::supertonic {

class SupertonicRuntime {
public:
    SupertonicRuntime(
        std::shared_ptr<const SupertonicAssets> assets,
        core::BackendConfig backend_config,
        assets::TensorStorageType weight_storage_type,
        std::size_t style_cache_slots = 4);
    ~SupertonicRuntime();

    runtime::AudioBuffer synthesize(
        const std::string & text,
        const SupertonicGenerationOptions & options,
        const SupertonicTextTokenizer & tokenizer);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::supertonic
