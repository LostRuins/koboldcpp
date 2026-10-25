#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>

namespace engine::core {

// Session-option vocabulary shared by families that lower attention with
// ggml_flash_attn_ext: "auto" (default), "flash", or "eager".
//
// Background: ggml-cuda only instantiates the MMA/wmma flash-attention kernels
// for compute capability >= 8.0 (Ampere). On older GPUs such as Volta/sm70
// (e.g. Tesla V100) a graph containing GGML_OP_FLASH_ATTN_EXT fails at compute
// time with "no device code compatible with CUDA arch 700", even when ggml was
// built with that arch enabled. The eager (explicit matmul + softmax) lowering
// computes the same operation with generic ops and runs everywhere (output
// logits may differ at ulp level, as with any kernel change).
enum class AttentionPreference {
    Auto,
    Flash,
    Eager,
};

// Parses a "<family>.attention" session-option value. Throws std::runtime_error
// naming option_name on invalid input.
AttentionPreference parse_attention_preference(const std::string & value, const char * option_name);

// Resolves whether flash attention may be used for the given backend and head
// dimension. Flash forces true, Eager forces false, Auto gates on the CUDA
// device compute capability (Volta/Turing resolve to eager; see the .cpp for
// why supports_op cannot be used). A null or non-CUDA backend, or a device
// query failure, preserves historical behavior (true).
//
// Adopting in other families (currently wired for higgs_audio_tts and
// breeze_tts only): resolve once per runtime with the model's head_dim and
// the family's "<family>.attention" session option, then switch the
// Decoder prefill/static modes (or SDPA/GQA lowerings) between flash and
// their ManualRepeat/Explicit equivalents based on the result.
bool resolve_flash_attention(ggml_backend_t backend, int64_t head_dim, AttentionPreference preference);

// True when the backend's device is an Intel GPU on the Vulkan backend. Does
// not influence resolve_flash_attention(); it is a building block for families
// whose eager lowering has been measured faster than ggml-vulkan's flash
// kernels on Intel (yue2 NAR: 2.2x on Arc Battlemage). Families whose eager
// path materializes an F16 ggml_repeat must not use it on Vulkan (no kernel).
bool vulkan_device_is_intel(ggml_backend_t backend);

}  // namespace engine::core
