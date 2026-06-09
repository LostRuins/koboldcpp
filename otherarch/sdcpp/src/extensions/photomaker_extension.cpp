#include "extensions/generation_extension.h"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>

#include "core/tensor_ggml.hpp"
#include "core/util.h"
#include "model/adapter/lora.hpp"
#include "model/adapter/pmid.hpp"

static std::tuple<std::vector<int>, std::vector<float>, std::vector<bool>>
tokenize_photomaker_trigger(FrozenCLIPEmbedderWithCustomWords& clip_conditioner,
                            const std::string& text,
                            int trigger_token_count,
                            int32_t image_token) {
    auto tokens_and_weights           = clip_conditioner.tokenize(text);
    std::vector<int> source_tokens    = std::move(tokens_and_weights.first);
    std::vector<float> source_weights = std::move(tokens_and_weights.second);

    if (!source_tokens.empty() && source_tokens.front() == clip_conditioner.tokenizer.BOS_TOKEN_ID) {
        source_tokens.erase(source_tokens.begin());
        source_weights.erase(source_weights.begin());
    }
    if (!source_tokens.empty() && source_tokens.back() == clip_conditioner.tokenizer.EOS_TOKEN_ID) {
        source_tokens.pop_back();
        source_weights.pop_back();
    }

    std::vector<int> tokens;
    std::vector<float> weights;
    int32_t class_idx = -1;
    for (size_t i = 0; i < source_tokens.size(); i++) {
        int token = source_tokens[i];
        if (token == image_token) {
            if (!tokens.empty()) {
                class_idx          = static_cast<int32_t>(tokens.size()) - 1;
                int class_token    = tokens.back();
                float class_weight = weights.back();
                for (int j = 1; j < trigger_token_count; j++) {
                    tokens.push_back(class_token);
                    weights.push_back(class_weight);
                }
            }
            continue;
        }
        tokens.push_back(token);
        weights.push_back(source_weights[i]);
    }

    clip_conditioner.tokenizer.pad_tokens(tokens,
                                          &weights,
                                          nullptr,
                                          clip_conditioner.text_model->model.n_token,
                                          clip_conditioner.text_model->model.n_token,
                                          true);
    std::vector<bool> class_token_mask;
    for (int i = 0; i < tokens.size(); i++) {
        class_token_mask.push_back(class_idx + 1 <= i && i < class_idx + 1 + trigger_token_count);
    }

    return std::make_tuple(tokens, weights, class_token_mask);
}

static std::tuple<SDCondition, std::vector<bool>>
get_photomaker_condition_with_trigger(FrozenCLIPEmbedderWithCustomWords& clip_conditioner,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params,
                                      const std::string& trigger_word,
                                      int trigger_token_count) {
    auto image_tokens = clip_conditioner.convert_token_to_id(trigger_word);
    GGML_ASSERT(image_tokens.size() == 1);
    auto tokens_and_weights         = tokenize_photomaker_trigger(clip_conditioner,
                                                                  conditioner_params.text,
                                                                  trigger_token_count,
                                                                  image_tokens[0]);
    std::vector<int>& tokens        = std::get<0>(tokens_and_weights);
    std::vector<float>& weights     = std::get<1>(tokens_and_weights);
    std::vector<bool>& trigger_mask = std::get<2>(tokens_and_weights);
    auto cond                       = clip_conditioner.get_learned_condition_common(n_threads,
                                                                                    tokens,
                                                                                    weights,
                                                                                    conditioner_params.clip_skip,
                                                                                    conditioner_params.width,
                                                                                    conditioner_params.height,
                                                                                    conditioner_params.zero_out_masked);
    return std::make_tuple(std::move(cond), trigger_mask);
}

static std::string remove_photomaker_trigger_from_prompt(FrozenCLIPEmbedderWithCustomWords& clip_conditioner,
                                                         const std::string& prompt,
                                                         const std::string& trigger_word) {
    auto image_tokens = clip_conditioner.convert_token_to_id(trigger_word);
    GGML_ASSERT(image_tokens.size() == 1);
    auto tokens_and_weights  = clip_conditioner.tokenize(prompt);
    std::vector<int>& tokens = tokens_and_weights.first;
    auto it                  = std::find(tokens.begin(), tokens.end(), image_tokens[0]);
    GGML_ASSERT(it != tokens.end());
    tokens.erase(it);
    return clip_conditioner.decode(tokens);
}

struct PhotoMakerExtension : public GenerationExtension {
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;
    bool enabled = false;
    std::string model_path;
    std::string trigger_word = "img";
    SDCondition id_condition;
    int start_merge_step = -1;

    const char* name() const override {
        return "photomaker";
    }

    bool is_enabled() const override {
        return enabled;
    }

    bool init(const GenerationExtensionInitContext& ctx) override {
        model_path = SAFE_STR(ctx.params->photo_maker_path);
        if (model_path.empty()) {
            return true;
        }

        if (!ctx.ensure_backend_pair(SDBackendModule::PHOTOMAKER)) {
            return false;
        }

        PMVersion pm_version = std::strstr(model_path.c_str(), "v2") != nullptr ? PM_VERSION_2 : PM_VERSION_1;
        pmid_model           = std::make_shared<PhotoMakerIDEncoder>(ctx.backend_for(SDBackendModule::PHOTOMAKER),
                                                           ctx.params_backend_for(SDBackendModule::PHOTOMAKER),
                                                           ctx.tensor_storage_map,
                                                           "pmid",
                                                           ctx.version,
                                                           pm_version);
        if (pm_version == PM_VERSION_2) {
            LOG_INFO("using PhotoMaker Version 2");
        }

        pmid_lora               = std::make_shared<LoraModel>("pmid",
                                                ctx.backend_for(SDBackendModule::PHOTOMAKER),
                                                ctx.params_backend_for(SDBackendModule::PHOTOMAKER),
                                                model_path,
                                                "",
                                                ctx.version);
        auto lora_tensor_filter = [&](const std::string& tensor_name) {
            return starts_with(tensor_name, "lora.model");
        };
        if (!pmid_lora->load_from_file(ctx.n_threads, lora_tensor_filter)) {
            LOG_WARN("load photomaker lora tensors from %s failed", model_path.c_str());
            return false;
        }

        LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", model_path.c_str());
        if (!ctx.model_loader.init_from_file_and_convert_name(model_path, "pmid.")) {
            LOG_WARN("loading stacked ID embedding from '%s' failed", model_path.c_str());
            return true;
        }

        enabled = true;
        return true;
    }

    void collect_param_tensors(GenerationExtensionTensorContext& ctx) override {
        if (!enabled || pmid_model == nullptr) {
            return;
        }

        std::map<std::string, ggml_tensor*> temp;
        pmid_model->get_param_tensors(temp, "pmid");
        bool do_mmap = ctx.module_can_mmap(SDBackendModule::PHOTOMAKER);
        for (const auto& [key, tensor] : temp) {
            ctx.tensors[key] = tensor;
            if (do_mmap) {
                ctx.mmap_able_tensors[key] = tensor;
            }
        }
    }

    void add_ignore_tensors(std::set<std::string>& ignore_tensors) const override {
        if (!enabled) {
            return;
        }
        ignore_tensors.insert("pmid.unet.");
    }

    bool alloc_params_buffer() override {
        if (!enabled || pmid_model == nullptr) {
            return true;
        }
        return pmid_model->alloc_params_buffer();
    }

    size_t get_params_buffer_size() const override {
        if (!enabled || pmid_model == nullptr) {
            return 0;
        }
        return pmid_model->get_params_buffer_size();
    }

    void reset_runtime_condition() override {
        id_condition     = {};
        start_merge_step = -1;
    }

    bool prepare_condition(GenerationExtensionConditionContext& ctx) override {
        reset_runtime_condition();
        if (!enabled || pmid_model == nullptr || pmid_lora == nullptr) {
            return false;
        }

        if (!pmid_lora->applied) {
            int64_t t0 = ggml_time_ms();
            pmid_lora->apply(ctx.tensors, ctx.version, ctx.n_threads);
            int64_t t1         = ggml_time_ms();
            pmid_lora->applied = true;
            LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
            if (ctx.free_params_immediately) {
                pmid_lora->free_params_buffer();
            }
        }

        bool pmv2 = pmid_model->get_version() == PM_VERSION_2;
        if (ctx.pm_params.id_images_count <= 0 || ctx.pm_params.id_images == nullptr) {
            LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
            LOG_WARN("Turn off PhotoMaker for this request");
            return false;
        }
        auto* clip_conditioner = dynamic_cast<FrozenCLIPEmbedderWithCustomWords*>(ctx.conditioner);
        if (clip_conditioner == nullptr) {
            LOG_WARN("PhotoMaker requires FrozenCLIPEmbedderWithCustomWords conditioner");
            LOG_WARN("Turn off PhotoMaker for this request");
            return false;
        }

        int clip_image_size        = 224;
        pmid_model->style_strength = ctx.pm_params.style_strength;
        sd::Tensor<float> id_image_tensor;
        for (int i = 0; i < ctx.pm_params.id_images_count; i++) {
            auto id_image           = sd_image_to_tensor(ctx.pm_params.id_images[i]);
            auto processed_id_image = clip_preprocess(id_image, clip_image_size, clip_image_size);
            if (id_image_tensor.empty()) {
                id_image_tensor = processed_id_image;
            } else {
                id_image_tensor = sd::ops::concat(id_image_tensor, processed_id_image, 3);
            }
        }

        int64_t t0                        = ggml_time_ms();
        int trigger_token_count           = pmv2 ? 2 * ctx.pm_params.id_images_count : ctx.pm_params.id_images_count;
        auto cond_tup                     = get_photomaker_condition_with_trigger(*clip_conditioner,
                                                                                  ctx.n_threads,
                                                                                  ctx.condition_params,
                                                                                  trigger_word,
                                                                                  trigger_token_count);
        SDCondition prepared_id_condition = std::get<0>(cond_tup);
        auto class_tokens_mask            = std::get<1>(cond_tup);
        if (std::find(class_tokens_mask.begin(), class_tokens_mask.end(), true) == class_tokens_mask.end()) {
            LOG_WARN("PhotoMaker trigger word '%s' was not found in prompt", trigger_word.c_str());
            LOG_WARN("Turn off PhotoMaker for this request");
            return false;
        }

        sd::Tensor<float> id_embeds;
        if (pmv2 && ctx.pm_params.id_embed_path != nullptr) {
            try {
                id_embeds = sd::load_tensor_from_file_as_tensor<float>(ctx.pm_params.id_embed_path);
            } catch (const std::exception&) {
                id_embeds = {};
            }
        }
        if (pmv2 && id_embeds.empty()) {
            LOG_WARN("Provided PhotoMaker images, but NO valid ID embeds file for PM v2");
            LOG_WARN("Turn off PhotoMaker for this request");
            return false;
        }
        if (pmv2 && ctx.pm_params.id_images_count != id_embeds.shape()[1]) {
            LOG_WARN("PhotoMaker image count (%d) does NOT match ID embeds (%d). You should run face_detect.py again.",
                     ctx.pm_params.id_images_count,
                     static_cast<int>(id_embeds.shape()[1]));
            LOG_WARN("Turn off PhotoMaker for this request");
            return false;
        }

        auto res = pmid_model->compute(ctx.n_threads,
                                       id_image_tensor,
                                       prepared_id_condition.c_crossattn,
                                       id_embeds,
                                       class_tokens_mask);
        if (res.empty()) {
            LOG_ERROR("Photomaker ID Stacking failed");
            LOG_WARN("Turn off PhotoMaker for this request");
            return false;
        }

        prepared_id_condition.c_crossattn = std::move(res);
        int64_t t1                        = ggml_time_ms();
        id_condition                      = std::move(prepared_id_condition);
        start_merge_step                  = int(ctx.pm_params.style_strength / 100.f * ctx.total_steps);
        ctx.condition_params.text         = remove_photomaker_trigger_from_prompt(*clip_conditioner,
                                                                                  ctx.condition_params.text,
                                                                                  trigger_word);
        LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
        LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);

        if (ctx.free_params_immediately) {
            pmid_model->free_params_buffer();
        }
        return true;
    }

    const SDCondition& before_condition(int step,
                                        const SDCondition& condition) const override {
        if (!id_condition.empty() && start_merge_step != -1 && step > start_merge_step) {
            return id_condition;
        }
        return condition;
    }
};

std::shared_ptr<GenerationExtension> create_photomaker_extension() {
    return std::make_shared<PhotoMakerExtension>();
}
