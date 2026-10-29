#include "engine/framework/sampling/torch_random.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/dynamic_library.h"
#ifdef ENGINE_HAS_CUDA_TORCH_RANDOM
#include "torch_random_cuda_runtime.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace engine::sampling {
namespace {

constexpr uint32_t kPhiloxM0 = 0xD2511F53U;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57U;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9U;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85U;
constexpr float kInvTwoPow32 = 2.3283064365386963e-10F;
constexpr float kInvTwoPow32TwoPi = 1.4629180792671596e-09F;

struct Philox4 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

void mul_hi_lo(uint32_t lhs, uint32_t rhs, uint32_t & hi, uint32_t & lo) {
    const uint64_t product = static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs);
    lo = static_cast<uint32_t>(product);
    hi = static_cast<uint32_t>(product >> 32U);
}

Philox4 philox_round(Philox4 counter, uint32_t key0, uint32_t key1) {
    uint32_t hi0 = 0;
    uint32_t lo0 = 0;
    uint32_t hi1 = 0;
    uint32_t lo1 = 0;
    mul_hi_lo(kPhiloxM0, counter.x, hi0, lo0);
    mul_hi_lo(kPhiloxM1, counter.z, hi1, lo1);
    return Philox4{
        hi1 ^ counter.y ^ key0,
        lo1,
        hi0 ^ counter.w ^ key1,
        lo0,
    };
}

Philox4 philox_4x32_10(Philox4 counter, uint64_t seed) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
    for (int round = 0; round < 10; ++round) {
        counter = philox_round(counter, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

void box_muller(uint32_t uniform0, uint32_t uniform1, float & normal0, float & normal1) {
    const float radius_input =
        static_cast<float>(uniform0) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
    const float angle =
        static_cast<float>(uniform1) * kInvTwoPow32TwoPi + (kInvTwoPow32TwoPi * 0.5F);
    const float radius = std::sqrt(-2.0F * std::log(radius_input));
    normal0 = radius * std::sin(angle);
    normal1 = radius * std::cos(angle);
}

float torch_cuda_randn_element(uint64_t seed, uint64_t index) {
    const Philox4 counter{
        0U,
        0U,
        static_cast<uint32_t>(index),
        static_cast<uint32_t>(index >> 32U),
    };
    const Philox4 random = philox_4x32_10(counter, seed);
    float normal0 = 0.0F;
    float normal1 = 0.0F;
    box_muller(random.x, random.y, normal0, normal1);
    return normal0;
}

float torch_cuda_uniform_element(uint64_t seed, uint64_t index) {
    const Philox4 counter{
        0U,
        0U,
        static_cast<uint32_t>(index),
        static_cast<uint32_t>(index >> 32U),
    };
    const Philox4 random = philox_4x32_10(counter, seed);
    return static_cast<float>(random.x) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
}

float torch_cuda_uniform_tensor_iterator_element(
    uint64_t seed,
    uint64_t sequence,
    uint64_t offset_blocks,
    int component) {
    const Philox4 counter{
        static_cast<uint32_t>(offset_blocks),
        static_cast<uint32_t>(offset_blocks >> 32U),
        static_cast<uint32_t>(sequence),
        static_cast<uint32_t>(sequence >> 32U),
    };
    const Philox4 random = philox_4x32_10(counter, seed);
    uint32_t value = random.x;
    switch (component) {
    case 0:
        value = random.x;
        break;
    case 1:
        value = random.y;
        break;
    case 2:
        value = random.z;
        break;
    case 3:
        value = random.w;
        break;
    default:
        throw std::invalid_argument("torch CUDA TensorIterator uniform component is invalid");
    }
    return static_cast<float>(value) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
}

uint64_t torch_cuda_tensor_iterator_stride(uint64_t total_elements, const TorchCudaSamplingPolicy & policy) {
    if (policy.multiprocessor_count <= 0 || policy.max_threads_per_multiprocessor <= 0) {
        throw std::invalid_argument("torch CUDA TensorIterator randn requires CUDA device properties");
    }
    constexpr uint64_t block_size = 256;
    uint64_t grid = (total_elements + block_size - 1) / block_size;
    uint64_t blocks_per_sm = static_cast<uint64_t>(policy.max_threads_per_multiprocessor) / block_size;
    if (blocks_per_sm == 0) {
        blocks_per_sm = 1;
    }
    const uint64_t grid_cap = static_cast<uint64_t>(policy.multiprocessor_count) * blocks_per_sm;
    grid = std::max<uint64_t>(1, std::min(grid_cap, grid));
    return block_size * grid;
}

float round_to_bfloat16(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    float rounded = 0.0F;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

}  // namespace

void fill_torch_cuda_randn(
    float * output,
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision,
    uint64_t start_index) {
    if (output == nullptr && count != 0) {
        throw std::invalid_argument("torch CUDA randn output pointer is null");
    }
    for (size_t index = 0; index < count; ++index) {
        float value = torch_cuda_randn_element(seed, start_index + static_cast<uint64_t>(index));
        if (precision == TorchRandnPrecision::BFloat16) {
            value = round_to_bfloat16(value);
        }
        output[index] = value;
    }
}

std::vector<float> generate_torch_cuda_randn(
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision,
    uint64_t start_index) {
    std::vector<float> output(count);
    fill_torch_cuda_randn(output.data(), output.size(), seed, precision, start_index);
    return output;
}

void fill_torch_cuda_tensor_iterator_randn(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision) {
    if (output == nullptr && count != 0) {
        throw std::invalid_argument("torch CUDA TensorIterator randn output pointer is null");
    }
    if (count == 0) {
        return;
    }
#ifdef ENGINE_HAS_CUDA_TORCH_RANDOM
    if (policy.cuda_fast_path) {
        detail::fill_torch_cuda_tensor_iterator_randn_cuda(
            output,
            count,
            seed,
            offset_blocks,
            policy,
            precision);
        return;
    }
#else
    if (policy.cuda_fast_path) {
        throw std::runtime_error("torch CUDA TensorIterator randn fast path was requested but CUDA runtime was not built");
    }
#endif
    constexpr uint64_t unroll_factor = 4;
    const uint64_t total = static_cast<uint64_t>(count);
    const uint64_t stride = torch_cuda_tensor_iterator_stride(total, policy);
    const uint64_t loop_count = (total + stride * unroll_factor - 1) / (stride * unroll_factor);
    const int64_t parallel_count = static_cast<int64_t>(stride);
#pragma omp parallel for if (parallel_count >= 65536)
    for (int64_t sequence_index = 0; sequence_index < parallel_count; ++sequence_index) {
        const uint64_t sequence = static_cast<uint64_t>(sequence_index);
        for (uint64_t loop_index = 0; loop_index < loop_count; ++loop_index) {
            const uint64_t first_index = loop_index * unroll_factor * stride + sequence;
            if (first_index >= total) {
                continue;
            }
            const Philox4 random = philox_4x32_10(
                Philox4{
                    static_cast<uint32_t>(offset_blocks + loop_index),
                    static_cast<uint32_t>((offset_blocks + loop_index) >> 32U),
                    static_cast<uint32_t>(sequence),
                    static_cast<uint32_t>(sequence >> 32U),
                },
                seed);
            float normal0 = 0.0F;
            float normal1 = 0.0F;
            float normal2 = 0.0F;
            float normal3 = 0.0F;
            box_muller(random.x, random.y, normal0, normal1);
            box_muller(random.z, random.w, normal2, normal3);
            if (precision == TorchRandnPrecision::BFloat16) {
                normal0 = round_to_bfloat16(normal0);
                normal1 = round_to_bfloat16(normal1);
                normal2 = round_to_bfloat16(normal2);
                normal3 = round_to_bfloat16(normal3);
            }
            output[static_cast<size_t>(first_index)] = normal0;
            const uint64_t second_index = first_index + stride;
            if (second_index < total) {
                output[static_cast<size_t>(second_index)] = normal1;
            }
            const uint64_t third_index = second_index + stride;
            if (third_index < total) {
                output[static_cast<size_t>(third_index)] = normal2;
            }
            const uint64_t fourth_index = third_index + stride;
            if (fourth_index < total) {
                output[static_cast<size_t>(fourth_index)] = normal3;
            }
        }
    }
}

std::vector<float> generate_torch_cuda_tensor_iterator_randn(
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision) {
    std::vector<float> output(count);
    fill_torch_cuda_tensor_iterator_randn(output.data(), output.size(), seed, offset_blocks, policy, precision);
    return output;
}

void fill_torch_cuda_uniform(float * output, size_t count, uint64_t seed, uint64_t start_index) {
    if (output == nullptr && count != 0) {
        throw std::invalid_argument("torch CUDA uniform output pointer is null");
    }
    for (size_t index = 0; index < count; ++index) {
        output[index] = torch_cuda_uniform_element(seed, start_index + static_cast<uint64_t>(index));
    }
}

std::vector<float> generate_torch_cuda_uniform(size_t count, uint64_t seed, uint64_t start_index) {
    std::vector<float> output(count);
    fill_torch_cuda_uniform(output.data(), output.size(), seed, start_index);
    return output;
}

namespace {

void log_default_policy(std::string_view category, std::string_view reason) {
    engine::debug::log_message(
        engine::debug::LogLevel::Warning,
        category,
        std::string("using default Torch RNG layout policy ")
            + "(multiprocessor_count=1, max_threads_per_multiprocessor=256): " + std::string(reason));
}

// ENGINE_TORCH_SAMPLING_POLICY pins the TensorIterator RNG layout instead of
// probing the CUDA device, making the noise realization identical across
// backends (CUDA/HIP/CPU) and machines. Accepted values: "default" (1x256)
// or "<multiprocessor_count>x<max_threads_per_multiprocessor>" (e.g.
// "68x1024"). Unset keeps the legacy behavior (device probe on CUDA, default
// layout elsewhere). The pinned layout never uses the CUDA fast path so every
// backend computes the same Philox element mapping on the host.
std::optional<TorchCudaSamplingPolicy> pinned_policy_from_env(std::string_view log_category) {
    const char * value = std::getenv("ENGINE_TORCH_SAMPLING_POLICY");
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    TorchCudaSamplingPolicy policy;
    std::string text(value);
    if (text != "default") {
        const auto cross = text.find('x');
        if (cross == std::string::npos) {
            throw std::runtime_error(
                "ENGINE_TORCH_SAMPLING_POLICY must be \"default\" or \"<sm>x<threads>\", got: " + text);
        }
        try {
            policy.multiprocessor_count = std::stoll(text.substr(0, cross));
            policy.max_threads_per_multiprocessor = std::stoll(text.substr(cross + 1));
        } catch (const std::exception &) {
            throw std::runtime_error(
                "ENGINE_TORCH_SAMPLING_POLICY must be \"default\" or \"<sm>x<threads>\", got: " + text);
        }
        if (policy.multiprocessor_count <= 0 || policy.max_threads_per_multiprocessor <= 0) {
            throw std::runtime_error("ENGINE_TORCH_SAMPLING_POLICY values must be positive: " + text);
        }
    }
    policy.cuda_fast_path = false;
    engine::debug::log_message(
        engine::debug::LogLevel::Warning,
        log_category,
        "using pinned Torch RNG layout policy from ENGINE_TORCH_SAMPLING_POLICY "
        "(multiprocessor_count=" + std::to_string(policy.multiprocessor_count)
        + ", max_threads_per_multiprocessor=" + std::to_string(policy.max_threads_per_multiprocessor)
        + ")");
    return policy;
}

}  // namespace

TorchCudaSamplingPolicy resolve_torch_cuda_sampling_policy(
    engine::core::BackendType backend_type,
    int device_index,
    std::string_view log_category,
    std::string_view model_name,
    TorchCudaSamplingPolicyFailureMode failure_mode) {
    if (const auto pinned = pinned_policy_from_env(log_category)) {
        return *pinned;
    }
    TorchCudaSamplingPolicy policy;
    if (backend_type != engine::core::BackendType::Cuda) {
        log_default_policy(log_category, "backend is not CUDA");
        return policy;
    }
    // CUDA runtime probe via dynamic library (works with both static and GGML_BACKEND_DL builds)
    const engine::io::DynamicLibraryHandle driver = engine::io::open_dynamic_library(
        {"libcuda.so.1", "libcuda.so", "libcuda.dylib", "nvcuda.dll"});
    if (driver == nullptr) {
        if (failure_mode == TorchCudaSamplingPolicyFailureMode::FallbackToDefault) {
            log_default_policy(log_category, "CUDA driver library was not found");
            return policy;
        }
        throw std::runtime_error(std::string(model_name) +
                                 " CUDA sampling policy probe failed: CUDA driver library was not found");
    }
    using CuInitFn = int (*)(unsigned int);
    using CuDeviceGetFn = int (*)(int *, int);
    using CuDeviceGetAttributeFn = int (*)(int *, int, int);
    auto cu_init = reinterpret_cast<CuInitFn>(engine::io::dynamic_library_symbol(driver, "cuInit"));
    auto cu_device_get = reinterpret_cast<CuDeviceGetFn>(engine::io::dynamic_library_symbol(driver, "cuDeviceGet"));
    auto cu_device_get_attribute =
        reinterpret_cast<CuDeviceGetAttributeFn>(engine::io::dynamic_library_symbol(driver, "cuDeviceGetAttribute"));
    if (cu_init == nullptr || cu_device_get == nullptr || cu_device_get_attribute == nullptr) {
        engine::io::close_dynamic_library(driver);
        if (failure_mode == TorchCudaSamplingPolicyFailureMode::FallbackToDefault) {
            log_default_policy(log_category, "CUDA driver symbols were not resolved");
            return policy;
        }
        throw std::runtime_error(std::string(model_name) +
                                 " CUDA sampling policy probe failed: CUDA driver symbols were not resolved");
    }

    int device = 0;
    int multiprocessor_count = 0;
    int max_threads_per_multiprocessor = 0;
    constexpr int kCuDeviceAttributeMultiprocessorCount = 16;
    constexpr int kCuDeviceAttributeMaxThreadsPerMultiprocessor = 39;
    const bool ok = cu_init(0) == 0 && cu_device_get(&device, device_index) == 0 &&
                    cu_device_get_attribute(&multiprocessor_count, kCuDeviceAttributeMultiprocessorCount, device) == 0 &&
                    cu_device_get_attribute(
                        &max_threads_per_multiprocessor,
                        kCuDeviceAttributeMaxThreadsPerMultiprocessor,
                        device) == 0;
    engine::io::close_dynamic_library(driver);
    if (!ok || multiprocessor_count <= 0 || max_threads_per_multiprocessor <= 0) {
        if (failure_mode == TorchCudaSamplingPolicyFailureMode::FallbackToDefault) {
            log_default_policy(log_category, "CUDA device attributes were not queried");
            return policy;
        }
        throw std::runtime_error(std::string(model_name) +
                                 " CUDA sampling policy probe failed: CUDA device attributes are invalid");
    }

    policy.multiprocessor_count = multiprocessor_count;
    policy.max_threads_per_multiprocessor = max_threads_per_multiprocessor;
    policy.cuda_fast_path = true;
    policy.cuda_device_index = device_index;
    return policy;
}

uint64_t torch_cuda_tensor_iterator_offset_blocks(
    uint64_t total_elements,
    const TorchCudaSamplingPolicy & policy) {
    if (total_elements == 0) {
        throw std::invalid_argument("torch CUDA TensorIterator offset requires elements");
    }
    if (policy.multiprocessor_count <= 0 || policy.max_threads_per_multiprocessor <= 0) {
        throw std::invalid_argument("torch CUDA TensorIterator offset requires CUDA device properties");
    }
    constexpr uint64_t block_size = 256;
    constexpr uint64_t unroll_factor = 4;
    uint64_t grid = (total_elements + block_size - 1) / block_size;
    uint64_t blocks_per_sm = static_cast<uint64_t>(policy.max_threads_per_multiprocessor) / block_size;
    if (blocks_per_sm == 0) {
        blocks_per_sm = 1;
    }
    const uint64_t grid_cap = static_cast<uint64_t>(policy.multiprocessor_count) * blocks_per_sm;
    grid = std::max<uint64_t>(1, std::min(grid_cap, grid));
    const uint64_t stride = block_size * grid;
    return ((total_elements - 1) / (stride * unroll_factor) + 1);
}

bool torch_cuda_sample_topk_exponential_pairs_available() {
#ifdef ENGINE_HAS_CUDA_TORCH_RANDOM
    return true;
#else
    return false;
#endif
}

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
    int32_t * out_codes) {
#ifdef ENGINE_HAS_CUDA_TORCH_RANDOM
    detail::sample_topk_exponential_pairs_cuda(
        device_logits_f32, songs, vocab, guidance_scale, top_k,
        seeds, offset_blocks, offset_step_blocks, policy, out_codes);
#else
    (void) device_logits_f32;
    (void) songs;
    (void) vocab;
    (void) guidance_scale;
    (void) top_k;
    (void) seeds;
    (void) offset_blocks;
    (void) policy;
    (void) out_codes;
    throw std::runtime_error("torch CUDA top-k exponential sampler is unavailable in this build");
#endif
}

#ifdef ENGINE_HAS_CUDA_TORCH_RANDOM
#define ENGINE_TORCH_RANDOM_CUDA_ONLY(...) __VA_ARGS__
#else
#define ENGINE_TORCH_RANDOM_CUDA_ONLY(...) \
    throw std::runtime_error("torch CUDA depth frame runtime is unavailable in this build")
#endif

void * torch_cuda_backend_stream(void * ggml_backend) {
#ifdef ENGINE_HAS_CUDA_TORCH_RANDOM
    return core::backend_cuda_stream(static_cast<ggml_backend_t>(ggml_backend));
#else
    (void) ggml_backend;
    return nullptr;
#endif
}

void torch_cuda_depth_frame_ensure(int64_t songs, int64_t levels, int64_t hidden_size, const TorchCudaSamplingPolicy & policy) {
    (void) songs; (void) levels; (void) hidden_size; (void) policy;
    ENGINE_TORCH_RANDOM_CUDA_ONLY(detail::depth_frame_ensure_cuda(songs, levels, hidden_size, policy));
}

void torch_cuda_depth_frame_begin(const uint64_t * seeds, const uint64_t * offset_blocks, int64_t songs, void * stream) {
    (void) seeds; (void) offset_blocks; (void) songs; (void) stream;
    ENGINE_TORCH_RANDOM_CUDA_ONLY(detail::depth_frame_begin_cuda(seeds, offset_blocks, songs, stream));
}

void torch_cuda_depth_frame_sample(
    const void * device_logits_f32,
    int64_t level_index,
    int64_t songs,
    int64_t vocab,
    float guidance_scale,
    int64_t top_k,
    const TorchCudaSamplingPolicy & policy,
    void * stream) {
    (void) device_logits_f32; (void) level_index; (void) songs; (void) vocab;
    (void) guidance_scale; (void) top_k; (void) policy; (void) stream;
    ENGINE_TORCH_RANDOM_CUDA_ONLY(detail::depth_frame_sample_cuda(
        device_logits_f32, level_index, songs, vocab, guidance_scale, top_k, policy, stream));
}

void torch_cuda_depth_frame_residual_fill(
    void * residual_ids_i32,
    int64_t previous_levels,
    int64_t songs,
    int64_t audio_vocab,
    void * stream) {
    (void) residual_ids_i32; (void) previous_levels; (void) songs; (void) audio_vocab; (void) stream;
    ENGINE_TORCH_RANDOM_CUDA_ONLY(detail::depth_frame_residual_fill_cuda(
        residual_ids_i32, previous_levels, songs, audio_vocab, stream));
}

void torch_cuda_depth_frame_accumulate_hidden(
    const void * hidden_f32,
    int64_t level_index,
    int64_t songs,
    int64_t hidden_size,
    void * stream) {
    (void) hidden_f32; (void) level_index; (void) songs; (void) hidden_size; (void) stream;
    ENGINE_TORCH_RANDOM_CUDA_ONLY(detail::depth_frame_accumulate_hidden_cuda(
        hidden_f32, level_index, songs, hidden_size, stream));
}

void torch_cuda_depth_frame_end(
    int32_t * host_codes,
    float * host_hidden,
    int64_t levels,
    int64_t songs,
    int64_t hidden_size,
    void * stream) {
    (void) host_codes; (void) host_hidden; (void) levels; (void) songs; (void) hidden_size; (void) stream;
    ENGINE_TORCH_RANDOM_CUDA_ONLY(detail::depth_frame_end_cuda(
        host_codes, host_hidden, levels, songs, hidden_size, stream));
}

float torch_cuda_tensor_iterator_exponential_element(
    uint64_t seed,
    uint64_t total_elements,
    uint64_t element_index,
    uint64_t call_index,
    int64_t multiprocessor_count,
    int64_t max_threads_per_multiprocessor) {
    const uint64_t offset_blocks =
        call_index * torch_cuda_tensor_iterator_offset_blocks(total_elements, TorchCudaSamplingPolicy{
            multiprocessor_count,
            max_threads_per_multiprocessor,
            false,
            0,
        });
    return torch_cuda_tensor_iterator_exponential_element_at_offset(
        seed,
        total_elements,
        element_index,
        offset_blocks,
        multiprocessor_count,
        max_threads_per_multiprocessor);
}

float torch_cuda_tensor_iterator_exponential_element_at_offset(
    uint64_t seed,
    uint64_t total_elements,
    uint64_t element_index,
    uint64_t offset_blocks,
    int64_t multiprocessor_count,
    int64_t max_threads_per_multiprocessor) {
    if (total_elements == 0 || element_index >= total_elements) {
        throw std::invalid_argument("torch CUDA TensorIterator exponential element index is out of range");
    }
    if (multiprocessor_count <= 0 || max_threads_per_multiprocessor <= 0) {
        throw std::invalid_argument("torch CUDA TensorIterator exponential requires CUDA device properties");
    }
    constexpr uint64_t block_size = 256;
    constexpr uint64_t unroll_factor = 4;
    uint64_t grid = (total_elements + block_size - 1) / block_size;
    uint64_t blocks_per_sm = static_cast<uint64_t>(max_threads_per_multiprocessor) / block_size;
    if (blocks_per_sm == 0) {
        blocks_per_sm = 1;
    }
    const uint64_t grid_cap = static_cast<uint64_t>(multiprocessor_count) * blocks_per_sm;
    grid = std::max<uint64_t>(1, std::min(grid_cap, grid));
    const uint64_t stride = block_size * grid;
    const uint64_t counter_offset = ((total_elements - 1) / (stride * unroll_factor) + 1) * unroll_factor;
    const uint64_t chunk = element_index / stride;
    const int component = static_cast<int>(chunk % unroll_factor);
    const uint64_t loop_index = chunk / unroll_factor;
    const uint64_t sequence = element_index % stride;
    (void)counter_offset;
    const float uniform =
        torch_cuda_uniform_tensor_iterator_element(seed, sequence, offset_blocks + loop_index, component);
    return -std::log(uniform);
}

}  // namespace engine::sampling
