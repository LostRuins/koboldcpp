#ifndef __SD_CONDITIONING_PROMPT_INPUT_H__
#define __SD_CONDITIONING_PROMPT_INPUT_H__

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.hpp"

namespace sd {

    struct PromptPart {
        enum class Type { TEXT,
                          IMAGE };

        Type type = Type::TEXT;
        std::string content;
        std::shared_ptr<const Tensor<float>> image;
        float weight = 1.f;
    };

    using PromptInput       = std::vector<PromptPart>;
    using PromptImageLoader = std::function<Tensor<float>(const std::string&)>;

    class PromptInputTemplate {
        const PromptInput* input_;
        std::map<std::string, size_t> image_markers_;

    public:
        std::string text;

        explicit PromptInputTemplate(const PromptInput* input);
        PromptInput expand(const std::string& formatted_text, float weight = 1.f) const;
    };

    PromptInput parse_prompt_input(const std::string& text);
    bool prompt_has_images(const PromptInput& input);
    std::string prompt_input_text(const PromptInput& input);
    bool resolve_prompt_images(PromptInput& input, const PromptImageLoader& loader);

}  // namespace sd

#endif  // __SD_CONDITIONING_PROMPT_INPUT_H__
