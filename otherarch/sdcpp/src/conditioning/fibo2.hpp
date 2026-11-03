#ifndef __SD_CONDITIONING_FIBO2_HPP__
#define __SD_CONDITIONING_FIBO2_HPP__

#include "conditioning/conditioner.hpp"

struct Fibo2Conditioner : public LLMEmbedder {
    using JSON = nlohmann::ordered_json;

    Fibo2Conditioner(ggml_backend_t backend, const String2TensorStorage& weights, std::shared_ptr<RunnerWeightManager> weight_manager, const TokenizerConfig& tokenizers)
        : LLMEmbedder(backend, weights, VERSION_FIBO2, "", true, weight_manager, tokenizers) {
        if (llm->config.hidden_size != 2560 || llm->config.num_layers != 36) {
            throw std::runtime_error("Fibo 2.0 requires its Qwen3-VL-4B text encoder");
        }
    }

    static bool empty_value(const JSON& value) {
        return value.is_null() ||
               (value.is_string() && value.get_ref<const std::string&>().empty()) ||
               ((value.is_array() || value.is_object()) && value.empty());
    }

    static JSON clean_caption(const JSON& value) {
        if (value.is_object()) {
            auto result = JSON::object();
            for (auto it = value.begin(); it != value.end(); ++it) {
                auto cleaned = clean_caption(it.value());
                if (!empty_value(cleaned)) {
                    result[it.key()] = std::move(cleaned);
                }
            }
            return result;
        }
        if (value.is_array()) {
            auto result = JSON::array();
            for (const auto& item : value) {
                auto cleaned = clean_caption(item);
                if (!empty_value(cleaned)) {
                    result.push_back(std::move(cleaned));
                }
            }
            return result;
        }
        return value;
    }

    static std::pair<JSON, JSON> parse_prompt(const std::string& prompt) {
        auto caption = JSON::parse(prompt, nullptr, false);
        if (!caption.is_object()) {
            return {prompt, nullptr};
        }
        JSON instruction;
        if (caption.contains("edit_instruction")) {
            instruction = caption["edit_instruction"];
            caption.erase("edit_instruction");
        }
        if (caption.contains("structured_caption")) {
            caption = JSON(caption["structured_caption"]);
            if (!caption.is_object()) {
                return {caption, instruction};
            }
        }
        bool short_form = caption.size() == 2 && caption.contains("short_description") && caption.contains("text_render");
        caption         = clean_caption(caption);
        for (const char* score : {"aesthetic_score", "preference_score"}) {
            caption.erase(score);
            for (auto& value : caption) {
                if (value.is_object()) {
                    value.erase(score);
                }
            }
        }
        caption = clean_caption(caption);
        if (short_form) {
            caption = JSON{{"short_description", caption.value("short_description", JSON(""))},
                           {"text_render", caption.value("text_render", JSON::array())}};
        }
        return {caption, instruction};
    }

    static std::string format_prompt(const std::string& text, int num_images) {
        auto [caption, instruction] = parse_prompt(text);
        if (num_images > 0) {
            if (instruction.is_null()) {
                caption     = nullptr;
                instruction = text;
            }
            auto edit = JSON::object();
            for (int i = 0; i < num_images; ++i) {
                edit[num_images == 1 ? "image" : "image_" + std::to_string(i + 1)] = "<|vision_start|><|image_pad|><|vision_end|>";
            }
            if (!empty_value(caption)) {
                edit["structured_caption"] = caption;
            }
            edit["edit_instruction"] = instruction;
            return edit.dump();
        }
        if (text.empty()) {
            return R"({"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""})";
        }
        if (caption.is_string()) {
            return caption.get<std::string>();
        }
        return empty_value(caption) ? "{}" : caption.dump();
    }

    static sd::Tensor<float> make_bundles(sd::Tensor<float> states, const std::set<int>& layers) {
        constexpr int width         = 2560;
        constexpr int bundles[6][3] = {{9, 20, 31}, {5, 16, 27}, {7, 18, 29}, {9, 20, 31}, {11, 22, 33}, {13, 24, 35}};
        for (int64_t token = 0; token < states.shape()[1]; ++token) {
            for (size_t layer = 0; layer < layers.size(); ++layer) {
                float sum      = 0.f;
                int64_t offset = token * states.shape()[0] + layer * width;
                for (int i = 0; i < width; ++i) {
                    sum += states[offset + i] * states[offset + i];
                }
                float scale = 1.f / std::sqrt(sum / width + 1e-6f);
                for (int i = 0; i < width; ++i) {
                    states[offset + i] *= scale;
                }
            }
        }
        sd::Tensor<float> result({6 * 3 * width, states.shape()[1]});
        for (int b = 0; b < 6; ++b) {
            for (int j = 0; j < 3; ++j) {
                int index = static_cast<int>(std::distance(layers.begin(), layers.find(bundles[b][j])));
                for (int64_t token = 0; token < states.shape()[1]; ++token) {
                    std::copy_n(states.data() + token * states.shape()[0] + index * width, width,
                                result.data() + token * result.shape()[0] + (b * 3 + j) * width);
                }
            }
        }
        return result;
    }

    SDCondition get_learned_condition(int n_threads, const ConditionerParams& p) override {
        int image_count = p.ref_images ? static_cast<int>(p.ref_images->size()) : 0;
        if (image_count && !llm->enable_vision) {
            LOG_ERROR("Fibo 2.0 editing requires vision weights");
            return {};
        }
        std::string prompt = "<|im_start|>user\n" + format_prompt(p.text, image_count) + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<std::pair<int, sd::Tensor<float>>> images;
        LLM::DeepStackImageEmbeds deepstack;
        std::vector<LLM::ImageGrid> grids;
        size_t marker_start = 0;
        for (int i = 0; i < image_count; ++i) {
            const auto& image = (*p.ref_images)[i];
            int factor        = llm->config.vision.patch_size * llm->config.vision.spatial_merge_size;
            int height = static_cast<int>(image.shape()[1]), width = static_cast<int>(image.shape()[0]);
            int h = std::max(factor, static_cast<int>(std::round(static_cast<double>(height) / factor)) * factor);
            int w = std::max(factor, static_cast<int>(std::round(static_cast<double>(width) / factor)) * factor);
            resize_image_dims(height, width, h, w, factor,
                              p.ref_image_params.vlm_min_size,
                              p.ref_image_params.vlm_max_size,
                              p.ref_image_params.vlm_resize_mode);
            auto pixels = sd::ops::interpolate(image, {w, h, image.shape()[2], image.shape()[3]}, sd::ops::InterpolateMode::Bicubic, false, true);
            auto rgb    = sd::Tensor<float>({w, h, 3, 1});
            for (int64_t pixel = 0; pixel < int64_t(w) * h; ++pixel) {
                float alpha = pixels.shape()[2] == 4 ? pixels[pixel + 3 * w * h] : 1.f;
                for (int channel = 0; channel < 3; ++channel) {
                    rgb[pixel + channel * w * h] = 2.f * std::clamp(pixels[pixel + channel * w * h] * alpha + 1.f - alpha, 0.f, 1.f) - 1.f;
                }
            }
            auto outputs = llm->encode_image_outputs(n_threads, rgb, false);
            if (!expand_vision_marker(prompt, marker_start, std::move(outputs),
                                      h / llm->config.vision.patch_size, w / llm->config.vision.patch_size,
                                      images, deepstack, grids, 4096)) {
                return {};
            }
        }
        const std::set<int> layers = {5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35};
        auto states                = encode_prompt(n_threads, prompt, {0, 0}, 0, 0, images, layers, 0, false, 4096, deepstack, grids);
        if (states.empty()) {
            return {};
        }
        SDCondition result;
        result.c_crossattn = make_bundles(std::move(states), layers);
        return result;
    }
};

#endif  // __SD_CONDITIONING_FIBO2_HPP__
