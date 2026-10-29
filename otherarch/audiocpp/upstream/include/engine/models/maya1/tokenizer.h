#pragma once

#include "engine/models/maya1/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::maya1 {

class Maya1Tokenizer {
public:
  explicit Maya1Tokenizer(std::shared_ptr<const Maya1Assets> assets);
  ~Maya1Tokenizer();

  std::vector<int32_t> build_prompt(const std::string &description,
                                    const std::string &text) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::maya1
