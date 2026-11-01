#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/sheetsage/types.h"

#include <memory>
#include <vector>

namespace engine::models::sheetsage {

class Mert2EncoderRuntime {
public:
    Mert2EncoderRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        SheetSage2DecoderConfig config = {},
        SheetSage2DecoderRuntimeOptions options = {});
    ~Mert2EncoderRuntime();

    Mert2EncoderRuntime(const Mert2EncoderRuntime &) = delete;
    Mert2EncoderRuntime & operator=(const Mert2EncoderRuntime &) = delete;
    Mert2EncoderRuntime(Mert2EncoderRuntime &&) noexcept;
    Mert2EncoderRuntime & operator=(Mert2EncoderRuntime &&) noexcept;

    std::vector<float> encode_mel(
        const std::vector<float> & normalized_mel,
        int64_t mel_frames);
    void prepare(int64_t batch, int64_t mel_frames);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class SheetSage2DecoderRuntime {
public:
    SheetSage2DecoderRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        SheetSage2DecoderConfig config = {},
        SheetSage2DecoderRuntimeOptions options = {});
    ~SheetSage2DecoderRuntime();

    SheetSage2DecoderRuntime(const SheetSage2DecoderRuntime &) = delete;
    SheetSage2DecoderRuntime & operator=(const SheetSage2DecoderRuntime &) = delete;
    SheetSage2DecoderRuntime(SheetSage2DecoderRuntime &&) noexcept;
    SheetSage2DecoderRuntime & operator=(SheetSage2DecoderRuntime &&) noexcept;

    std::vector<float> decode_logits(
        const std::vector<float> & mixed_encoder_state,
        int64_t memory_steps,
        const std::vector<int32_t> & decoder_input_ids);
    void reset_cached_decode(
        const std::vector<float> & mixed_encoder_state,
        int64_t memory_steps,
        int64_t cache_steps);
    std::vector<float> prefill_cached_decode(const std::vector<int32_t> & token_ids);
    std::vector<float> decode_cached_step(int32_t token);

    void prepare(int64_t batch, int64_t memory_steps, int64_t decoder_steps);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sheetsage
