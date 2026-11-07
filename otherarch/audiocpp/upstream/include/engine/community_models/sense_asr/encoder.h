#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/sense_asr/assets.h"
#include "engine/community_models/sense_asr/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::sense_asr {

class SenseAsrEncoderRuntime {
public:
  SenseAsrEncoderRuntime(std::shared_ptr<const SenseAsrAssets> assets,
                         engine::core::ExecutionContext &execution_context,
                         size_t graph_arena_bytes,
                         engine::assets::TensorStorageType weight_storage =
                             engine::assets::TensorStorageType::F32);
  ~SenseAsrEncoderRuntime();

  SenseAsrEncoderRuntime(const SenseAsrEncoderRuntime &) = delete;
  SenseAsrEncoderRuntime &operator=(const SenseAsrEncoderRuntime &) = delete;
  SenseAsrEncoderRuntime(SenseAsrEncoderRuntime &&) noexcept;
  SenseAsrEncoderRuntime &operator=(SenseAsrEncoderRuntime &&) noexcept;

  void prepare_capacity(int64_t frames);

  void set_query_tokens(std::vector<int32_t> query_tokens);

  SenseAsrEncoderOutput encode(const SenseAsrAudioFeatures &features);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::community_models::sense_asr
