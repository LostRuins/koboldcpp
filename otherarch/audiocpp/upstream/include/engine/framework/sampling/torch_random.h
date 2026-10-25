#pragma once

#include "engine/framework/core/backend.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace engine::sampling {

struct TorchCudaSamplingPolicy {
    int64_t multiprocessor_count = 1;
    int64_t max_threads_per_multiprocessor = 256;
    bool cuda_fast_path = false;
    int cuda_device_index = 0;
};

enum class TorchCudaSamplingPolicyFailureMode {
    StrictCuda,
    FallbackToDefault,
};

enum class TorchRandnPrecision {
    Float32,
    BFloat16,
};

TorchCudaSamplingPolicy resolve_torch_cuda_sampling_policy(
    engine::core::BackendType backend_type,
    int device_index,
    std::string_view log_category,
    std::string_view model_name,
    TorchCudaSamplingPolicyFailureMode failure_mode = TorchCudaSamplingPolicyFailureMode::StrictCuda);

uint64_t torch_cuda_tensor_iterator_offset_blocks(
    uint64_t total_elements,
    const TorchCudaSamplingPolicy & policy);

void fill_torch_cuda_randn(
    float * output,
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32,
    uint64_t start_index = 0);

std::vector<float> generate_torch_cuda_randn(
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32,
    uint64_t start_index = 0);

void fill_torch_cuda_tensor_iterator_randn(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32);

std::vector<float> generate_torch_cuda_tensor_iterator_randn(
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32);

void fill_torch_cuda_uniform(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t start_index = 0);

std::vector<float> generate_torch_cuda_uniform(size_t count, uint64_t seed, uint64_t start_index = 0);

// True when the CUDA build carries the in-graph-side top-k exponential
// sampler (device logits in, sampled codes out; no host logits readback).
bool torch_cuda_sample_topk_exponential_pairs_available();

// Samples one code per song from device-resident logits laid out as
// [cond_0; uncond_0; cond_1; ...] rows: bf16-rounds both branches, mixes with
// guidance_scale, applies top-k and the torch exponential ranking, matching
// the CPU path sample-for-sample (up to logf ULP differences).
// seeds/offset_blocks may be null after the first call of a frame: the
// device keeps the frame constants and offset_step_blocks advances the
// per-call RNG offset (call_step * blocks-per-call).
void torch_cuda_sample_topk_exponential_pairs(
    const void * device_logits_f32,
    int64_t songs,
    int64_t vocab,
    float guidance_scale,
    int64_t top_k,
    const uint64_t * seeds,
    const uint64_t * offset_blocks,
    uint64_t offset_step_blocks,
    const TorchCudaSamplingPolicy & policy,
    int32_t * out_codes);

// GPU-resident depth frame: per-codebook sampled codes and cond-row hiddens
// stay on device (residual ids for the next codebook are filled device-side);
// one host sync per frame in torch_cuda_depth_frame_end. All calls enqueue on
// the given backend stream.
void * torch_cuda_backend_stream(void * ggml_backend);
void torch_cuda_depth_frame_ensure(int64_t songs, int64_t levels, int64_t hidden_size, const TorchCudaSamplingPolicy & policy);
void torch_cuda_depth_frame_begin(const uint64_t * seeds, const uint64_t * offset_blocks, int64_t songs, void * stream);
void torch_cuda_depth_frame_sample(
    const void * device_logits_f32,
    int64_t level_index,
    int64_t songs,
    int64_t vocab,
    float guidance_scale,
    int64_t top_k,
    const TorchCudaSamplingPolicy & policy,
    void * stream);
void torch_cuda_depth_frame_residual_fill(
    void * residual_ids_i32,
    int64_t previous_levels,
    int64_t songs,
    int64_t audio_vocab,
    void * stream);
void torch_cuda_depth_frame_accumulate_hidden(
    const void * hidden_f32,
    int64_t level_index,
    int64_t songs,
    int64_t hidden_size,
    void * stream);
void torch_cuda_depth_frame_end(
    int32_t * host_codes,
    float * host_hidden,
    int64_t levels,
    int64_t songs,
    int64_t hidden_size,
    void * stream);

float torch_cuda_tensor_iterator_exponential_element(
    uint64_t seed,
    uint64_t total_elements,
    uint64_t element_index,
    uint64_t call_index,
    int64_t multiprocessor_count,
    int64_t max_threads_per_multiprocessor);

float torch_cuda_tensor_iterator_exponential_element_at_offset(
    uint64_t seed,
    uint64_t total_elements,
    uint64_t element_index,
    uint64_t offset_blocks,
    int64_t multiprocessor_count,
    int64_t max_threads_per_multiprocessor);

}  // namespace engine::sampling
