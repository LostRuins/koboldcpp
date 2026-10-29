#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::codecs {

struct SnacDecoderConfig {
  int64_t sample_rate = 24000;
  int64_t codebook_size = 4096;
  int64_t codebook_dim = 8;
  int64_t latent_dim = 768;
  int input_kernel_size = 7;
  int output_kernel_size = 7;
  int residual_kernel_size = 7;
  std::vector<int> quantizer_strides = {4, 2, 1};
  std::vector<int64_t> decoder_channels = {1024, 512, 256, 128, 64};
  std::vector<int> upsample_strides = {8, 8, 4, 2};
  std::vector<int> residual_dilations = {1, 3, 9};
  std::string trace_name = "snac";
};

struct SnacDecoderWeightBinding {
  std::string quantizer_prefix = "quantizer.quantizers.";
  std::string decoder_prefix = "decoder.model.";
};

struct SnacDecoderRuntimeOptions {
  size_t weight_context_bytes = 4ull * 1024ull * 1024ull;
  size_t graph_context_bytes = 4ull * 1024ull * 1024ull;
  assets::TensorStorageType weight_storage_type =
      assets::TensorStorageType::Native;
};

struct SnacCodes {
  std::vector<std::vector<int32_t>> codebooks;
};

class SnacDecoderRuntime {
public:
  SnacDecoderRuntime(SnacDecoderConfig config,
                     std::shared_ptr<const assets::TensorSource> source,
                     core::ExecutionContext &execution,
                     SnacDecoderRuntimeOptions options = {},
                     SnacDecoderWeightBinding weight_binding = {});
  ~SnacDecoderRuntime();

  SnacDecoderRuntime(const SnacDecoderRuntime &) = delete;
  SnacDecoderRuntime &operator=(const SnacDecoderRuntime &) = delete;
  SnacDecoderRuntime(SnacDecoderRuntime &&) noexcept;
  SnacDecoderRuntime &operator=(SnacDecoderRuntime &&) noexcept;

  runtime::AudioBuffer decode(const SnacCodes &codes, uint64_t seed);
  void release_runtime_graphs();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::codecs
