#include "engine/models/maya1/tokenizer.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>

namespace engine::models::maya1 {
namespace {

constexpr int32_t kCodeStart = 128257;
constexpr int32_t kStartHeader = 128259;
constexpr int32_t kEndHeader = 128260;
constexpr int32_t kStartAudio = 128261;
constexpr int32_t kBos = 128000;
constexpr int32_t kTextEnd = 128009;

} // namespace

struct Maya1Tokenizer::Impl {
  explicit Impl(const Maya1Assets &assets)
      : tokenizer({
            {},
            {},
            assets.resources.require_file("tokenizer_config"),
            assets.resources.require_file("tokenizer_json"),
            tokenizers::LlamaBpePreTokenizer::Llama3,
        }) {}

  tokenizers::LlamaBpeTokenizer tokenizer;
};

Maya1Tokenizer::Maya1Tokenizer(std::shared_ptr<const Maya1Assets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("Maya1 tokenizer requires assets");
  }
  impl_ = std::make_unique<Impl>(*assets);
}

Maya1Tokenizer::~Maya1Tokenizer() = default;

std::vector<int32_t>
Maya1Tokenizer::build_prompt(const std::string &description,
                             const std::string &text) const {
  if (description.empty()) {
    throw std::runtime_error("Maya1 requires a non-empty voice description");
  }
  if (text.empty()) {
    throw std::runtime_error("Maya1 requires non-empty text");
  }
  std::vector<int32_t> out{kBos, kStartHeader, kBos};
  auto content = impl_->tokenizer.encode(
      "<description=\"" + description + "\"> " + text, true);
  out.insert(out.end(), content.begin(), content.end());
  out.insert(out.end(), {kTextEnd, kEndHeader, kStartAudio, kCodeStart});
  return out;
}

} // namespace engine::models::maya1
