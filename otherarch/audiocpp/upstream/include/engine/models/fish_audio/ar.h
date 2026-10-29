#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/types.h"

#include <memory>
#include <functional>

namespace engine::models::fish_audio {

class FishAudioDualARRuntime {
public:
    FishAudioDualARRuntime(
        std::shared_ptr<const FishAudioAssets> assets,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~FishAudioDualARRuntime();

    engine::codecs::FishDacCodes generate(
        const FishAudioPrompt & prompt,
        const FishAudioGenerationOptions & options,
        const std::function<void(const std::vector<int32_t> &)> & on_frame = {});
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_audio
