#pragma once

#include "engine/models/maya1/assets.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::maya1 {

struct Maya1GenerationOptions {
  int64_t max_tokens = 2048;
  int64_t min_tokens = 28;
  float temperature = 0.4F;
  float top_p = 0.9F;
  float repetition_penalty = 1.1F;
  uint64_t seed = 0;
};

struct Maya1GenerationResult {
  std::vector<int32_t> snac_tokens;
};

class Maya1LlamaGenerator {
public:
  Maya1LlamaGenerator(std::shared_ptr<const Maya1Assets> assets,
                 core::ExecutionContext &execution,
                 size_t prefill_graph_arena_bytes,
                 size_t decode_graph_arena_bytes, size_t weight_context_bytes,
                 assets::TensorStorageType weight_storage_type);
  ~Maya1LlamaGenerator();

  Maya1GenerationResult generate(const std::vector<int32_t> &prompt_ids,
                                 const Maya1GenerationOptions &options);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::maya1
