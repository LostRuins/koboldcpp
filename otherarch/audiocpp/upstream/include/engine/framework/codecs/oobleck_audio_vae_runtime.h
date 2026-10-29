#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::codecs {

struct OobleckAudioVaeConfig {
    int64_t sample_rate = 48000;
    int64_t audio_channels = 2;
    int64_t channels = 64;
    int64_t encoder_latent_dim = 128;
    int64_t decoder_latent_dim = 64;
    std::vector<int64_t> c_mults{1, 2, 4, 8, 16, 32};
    std::vector<int64_t> strides{2, 2, 4, 4, 5, 6};
    bool use_snake = true;
    bool snake_logscale = true;
    bool final_tanh = false;
    std::string encoder_prefix = "encoder";
    std::string decoder_prefix = "decoder";
};

struct OobleckAudioVaeRuntimeOptions {
    size_t graph_arena_bytes = 512ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 1400ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
};

class OobleckAudioVaeRuntime {
public:
    OobleckAudioVaeRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        OobleckAudioVaeConfig config = {},
        OobleckAudioVaeRuntimeOptions options = {});
    ~OobleckAudioVaeRuntime();

    OobleckAudioVaeRuntime(const OobleckAudioVaeRuntime &) = delete;
    OobleckAudioVaeRuntime & operator=(const OobleckAudioVaeRuntime &) = delete;
    OobleckAudioVaeRuntime(OobleckAudioVaeRuntime &&) noexcept;
    OobleckAudioVaeRuntime & operator=(OobleckAudioVaeRuntime &&) noexcept;

    std::vector<float> encode_planar(const std::vector<float> & planar_audio, int64_t frames);
    std::vector<float> decode_planar(const std::vector<float> & latents, int64_t batch, int64_t latent_frames);
    std::vector<runtime::AudioBuffer> decode(const std::vector<float> & latents, int64_t batch, int64_t latent_frames);

    void prepare_encode(int64_t frames);
    void prepare_decode(int64_t batch, int64_t latent_frames);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::codecs
