#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::modules {

struct RmvpePitchExtractorConfig {
    std::string trace_prefix = "framework.rmvpe";
    size_t weight_context_bytes = 256ull * 1024ull * 1024ull;
};

class RmvpePitchExtractorComponent {
public:
    RmvpePitchExtractorComponent() = default;
    RmvpePitchExtractorComponent(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        assets::TensorStorageType storage_type,
        RmvpePitchExtractorConfig config = {});
    ~RmvpePitchExtractorComponent();

    RmvpePitchExtractorComponent(RmvpePitchExtractorComponent &&) noexcept;
    RmvpePitchExtractorComponent & operator=(RmvpePitchExtractorComponent &&) noexcept;
    RmvpePitchExtractorComponent(const RmvpePitchExtractorComponent &) = delete;
    RmvpePitchExtractorComponent & operator=(const RmvpePitchExtractorComponent &) = delete;

    std::vector<float> infer_16k_mono(
        const std::vector<float> & waveform_16k,
        float threshold,
        size_t threads) const;

private:
    struct State;

    std::shared_ptr<State> state_;
};

}  // namespace engine::modules
