#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/cosyvoice3/assets.h"

#include <memory>
#include <vector>

namespace engine::models::cosyvoice3 {

class CosyVoice3HiftRuntime {
public:
    CosyVoice3HiftRuntime(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType storage_type);
    ~CosyVoice3HiftRuntime();

    CosyVoice3HiftRuntime(const CosyVoice3HiftRuntime &) = delete;
    CosyVoice3HiftRuntime & operator=(const CosyVoice3HiftRuntime &) = delete;

    engine::runtime::AudioBuffer synthesize(
        const std::vector<float> & mel,
        int64_t frames,
        uint64_t seed);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::cosyvoice3
