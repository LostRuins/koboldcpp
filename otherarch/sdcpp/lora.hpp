#ifndef __LORA_HPP__
#define __LORA_HPP__

#include <mutex>
#include "ggml_extend.hpp"

#define LORA_GRAPH_BASE_SIZE 10240

struct LoraModel : public GGMLRunner {
    float multiplier = 1.0f;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::map<ggml_tensor*, ggml_tensor*> original_tensor_to_final_tensor;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed                = false;
    bool applied                    = false;
    bool tensor_preprocessed        = false;
    std::vector<int> zero_index_vec = {0};
    ggml_tensor* zero_index         = nullptr;

    LoraModel(ggml_backend_t backend,
              const std::string& file_path = "",
              std::string prefix           = "",
              SDVersion version            = VERSION_COUNT)
        : file_path(file_path), GGMLRunner(backend, false) {
        prefix = "lora." + prefix;
        if (!model_loader.init_from_file_and_convert_name(file_path, prefix, version)) {
            load_failed = true;
        }
    }

    std::string get_desc() override {
        return "lora";
    }

    bool load_from_file(bool filter_tensor, int n_threads) {
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        std::unordered_map<std::string, TensorStorage> tensors_to_create;
        std::mutex lora_mutex;
        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            if (dry_run) {
                const std::string& name = tensor_storage.name;

                if (filter_tensor && !contains(name, "lora.model")) {
                    return true;
                }

                {
                    std::lock_guard<std::mutex> lock(lora_mutex);
                    tensors_to_create[name] = tensor_storage;
                }
            } else {
                const std::string& name = tensor_storage.name;
                auto iter               = lora_tensors.find(name);
                if (iter != lora_tensors.end()) {
                    *dst_tensor = iter->second;
                }
            }
            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, n_threads);

        for (const auto& pair : tensors_to_create) {
            const auto& name         = pair.first;
            const auto& ts           = pair.second;
            struct ggml_tensor* real = ggml_new_tensor(params_ctx,
                                                       ts.type,
                                                       ts.n_dims,
                                                       ts.ne);
            lora_tensors[name]       = real;
        }

        alloc_params_buffer();

        dry_run = false;
        model_loader.load_tensors(on_new_tensor_cb, n_threads);

        LOG_DEBUG("finished loaded lora");
        return true;
    }

    ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* a) {
        auto out = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out      = ggml_get_rows(ctx, out, zero_index);
        out      = ggml_reshape(ctx, out, a);
        // auto out = ggml_cast(ctx, a, GGML_TYPE_F32);
        return out;
    }

    void preprocess_lora_tensors(const std::map<std::string, ggml_tensor*>& model_tensors) {
        if (tensor_preprocessed) {
            return;
        }
        tensor_preprocessed = true;
        // I really hate these hardcoded processes.
        if (model_tensors.find("cond_stage_model.1.transformer.text_model.encoder.layers.0.self_attn.in_proj.weight") != model_tensors.end()) {
            std::map<std::string, ggml_tensor*> new_lora_tensors;
            for (auto& [old_name, tensor] : lora_tensors) {
                std::string new_name = old_name;

                if (contains(new_name, "cond_stage_model.1.transformer.text_model.encoder.layers")) {
                    std::vector<std::pair<std::string, std::string>> qkv_name_map = {
                        {"self_attn.q_proj.weight", "self_attn.in_proj.weight"},
                        {"self_attn.q_proj.bias", "self_attn.in_proj.bias"},
                        {"self_attn.k_proj.weight", "self_attn.in_proj.weight.1"},
                        {"self_attn.k_proj.bias", "self_attn.in_proj.bias.1"},
                        {"self_attn.v_proj.weight", "self_attn.in_proj.weight.2"},
                        {"self_attn.v_proj.bias", "self_attn.in_proj.bias.2"},
                    };
                    for (auto kv : qkv_name_map) {
                        size_t pos = new_name.find(kv.first);
                        if (pos != std::string::npos) {
                            new_name.replace(pos, kv.first.size(), kv.second);
                        }
                    }
                }

                new_lora_tensors[new_name] = tensor;
            }

            lora_tensors = std::move(new_lora_tensors);
        }
    }

    ggml_tensor* get_lora_diff(const std::string& model_tensor_name, std::set<std::string>& applied_lora_tensors) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }

            std::string lora_down_name = "lora." + key + ".lora_down";
            std::string lora_up_name   = "lora." + key + ".lora_up";
            std::string lora_mid_name  = "lora." + key + ".lora_mid";
            std::string scale_name     = "lora." + key + ".scale";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lora_up   = nullptr;
            ggml_tensor* lora_mid  = nullptr;
            ggml_tensor* lora_down = nullptr;

            auto iter = lora_tensors.find(lora_up_name);
            if (iter != lora_tensors.end()) {
                lora_up = to_f32(compute_ctx, iter->second);
            }

            iter = lora_tensors.find(lora_mid_name);
            if (iter != lora_tensors.end()) {
                lora_mid = to_f32(compute_ctx, iter->second);
            }

            iter = lora_tensors.find(lora_down_name);
            if (iter != lora_tensors.end()) {
                lora_down = to_f32(compute_ctx, iter->second);
            }

            if (lora_up == nullptr || lora_down == nullptr) {
                break;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);

            if (lora_mid) {
                applied_lora_tensors.insert(lora_mid_name);
            }

            float scale_value = 1.0f;

            int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
            iter         = lora_tensors.find(scale_name);
            if (iter != lora_tensors.end()) {
                scale_value = ggml_ext_backend_tensor_get_f32(iter->second);
                applied_lora_tensors.insert(scale_name);
            } else {
                iter = lora_tensors.find(alpha_name);
                if (iter != lora_tensors.end()) {
                    float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                    scale_value = alpha / rank;
                    // LOG_DEBUG("rank %s %ld %.2f %.2f", alpha_name.c_str(), rank, alpha, scale_value);
                    applied_lora_tensors.insert(alpha_name);
                }
            }
            scale_value *= multiplier;

            auto curr_updown = ggml_ext_merge_lora(compute_ctx, lora_down, lora_up, lora_mid);
            curr_updown      = ggml_scale_inplace(compute_ctx, curr_updown, scale_value);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(compute_ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }

            index++;
        }

        // diff
        if (updown == nullptr) {
            std::string lora_diff_name = "lora." + model_tensor_name + ".diff";

            if (lora_tensors.find(lora_diff_name) != lora_tensors.end()) {
                updown = to_f32(compute_ctx, lora_tensors[lora_diff_name]);
                applied_lora_tensors.insert(lora_diff_name);
            }
        }

        return updown;
    }

    ggml_tensor* get_loha_diff(const std::string& model_tensor_name, std::set<std::string>& applied_lora_tensors) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            std::string hada_1_down_name = "lora." + key + ".hada_w1_b";
            std::string hada_1_mid_name  = "lora." + key + ".hada_t1";
            std::string hada_1_up_name   = "lora." + key + ".hada_w1_a";
            std::string hada_2_down_name = "lora." + key + ".hada_w2_b";
            std::string hada_2_mid_name  = "lora." + key + ".hada_t2";
            std::string hada_2_up_name   = "lora." + key + ".hada_w2_a";
            std::string alpha_name       = "lora." + key + ".alpha";

            ggml_tensor* hada_1_mid  = nullptr;  // tau for tucker decomposition
            ggml_tensor* hada_1_up   = nullptr;
            ggml_tensor* hada_1_down = nullptr;

            ggml_tensor* hada_2_mid  = nullptr;  // tau for tucker decomposition
            ggml_tensor* hada_2_up   = nullptr;
            ggml_tensor* hada_2_down = nullptr;

            auto iter = lora_tensors.find(hada_1_down_name);
            if (iter != lora_tensors.end()) {
                hada_1_down = to_f32(compute_ctx, iter->second);
            }

            iter = lora_tensors.find(hada_1_up_name);
            if (iter != lora_tensors.end()) {
                hada_1_up = to_f32(compute_ctx, iter->second);
            }

            iter = lora_tensors.find(hada_1_mid_name);
            if (iter != lora_tensors.end()) {
                hada_1_mid = to_f32(compute_ctx, iter->second);
                hada_1_up  = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, hada_1_up));
            }

            iter = lora_tensors.find(hada_2_down_name);
            if (iter != lora_tensors.end()) {
                hada_2_down = to_f32(compute_ctx, iter->second);
            }

            iter = lora_tensors.find(hada_2_up_name);
            if (iter != lora_tensors.end()) {
                hada_2_up = to_f32(compute_ctx, iter->second);
            }

            iter = lora_tensors.find(hada_2_mid_name);
            if (iter != lora_tensors.end()) {
                hada_2_mid = to_f32(compute_ctx, iter->second);
                hada_2_up  = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, hada_2_up));
            }

            if (hada_1_up == nullptr || hada_1_down == nullptr || hada_2_up == nullptr || hada_2_down == nullptr) {
                break;
            }

            applied_lora_tensors.insert(hada_1_down_name);
            applied_lora_tensors.insert(hada_1_up_name);
            applied_lora_tensors.insert(hada_2_down_name);
            applied_lora_tensors.insert(hada_2_up_name);
            applied_lora_tensors.insert(alpha_name);

            if (hada_1_mid) {
                applied_lora_tensors.insert(hada_1_mid_name);
            }

            if (hada_2_mid) {
                applied_lora_tensors.insert(hada_2_mid_name);
            }

            float scale_value = 1.0f;

            // calc_scale
            // TODO: .dora_scale?
            int64_t rank = hada_1_down->ne[ggml_n_dims(hada_1_down) - 1];
            iter         = lora_tensors.find(alpha_name);
            if (iter != lora_tensors.end()) {
                float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                scale_value = alpha / rank;
                applied_lora_tensors.insert(alpha_name);
            }
            scale_value *= multiplier;

            struct ggml_tensor* updown_1 = ggml_ext_merge_lora(compute_ctx, hada_1_down, hada_1_up, hada_1_mid);
            struct ggml_tensor* updown_2 = ggml_ext_merge_lora(compute_ctx, hada_2_down, hada_2_up, hada_2_mid);
            auto curr_updown             = ggml_mul_inplace(compute_ctx, updown_1, updown_2);
            curr_updown                  = ggml_scale_inplace(compute_ctx, curr_updown, scale_value);
            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(compute_ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    ggml_tensor* get_lokr_diff(const std::string& model_tensor_name, std::set<std::string>& applied_lora_tensors) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            std::string lokr_w1_name   = "lora." + key + ".lokr_w1";
            std::string lokr_w1_a_name = "lora." + key + ".lokr_w1_a";
            std::string lokr_w1_b_name = "lora." + key + ".lokr_w1_b";
            std::string lokr_w2_name   = "lora." + key + ".lokr_w2";
            std::string lokr_w2_a_name = "lora." + key + ".lokr_w2_a";
            std::string lokr_w2_b_name = "lora." + key + ".lokr_w2_b";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lokr_w1   = nullptr;
            ggml_tensor* lokr_w1_a = nullptr;
            ggml_tensor* lokr_w1_b = nullptr;
            ggml_tensor* lokr_w2   = nullptr;
            ggml_tensor* lokr_w2_a = nullptr;
            ggml_tensor* lokr_w2_b = nullptr;

            auto iter = lora_tensors.find(lokr_w1_name);
            if (iter != lora_tensors.end()) {
                lokr_w1 = to_f32(compute_ctx, iter->second);
            }

            iter = lora_tensors.find(lokr_w2_name);
            if (iter != lora_tensors.end()) {
                lokr_w2 = to_f32(compute_ctx, iter->second);
            }

            int64_t rank = 1;
            if (lokr_w1 == nullptr) {
                iter = lora_tensors.find(lokr_w1_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_a = to_f32(compute_ctx, iter->second);
                }

                iter = lora_tensors.find(lokr_w1_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_b = to_f32(compute_ctx, iter->second);
                }

                if (lokr_w1_a == nullptr || lokr_w1_b == nullptr) {
                    break;
                }

                rank = lokr_w1_b->ne[ggml_n_dims(lokr_w1_b) - 1];

                lokr_w1 = ggml_ext_merge_lora(compute_ctx, lokr_w1_b, lokr_w1_a);
            }

            if (lokr_w2 == nullptr) {
                iter = lora_tensors.find(lokr_w2_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_a = to_f32(compute_ctx, iter->second);
                }

                iter = lora_tensors.find(lokr_w2_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_b = to_f32(compute_ctx, iter->second);
                }

                if (lokr_w2_a == nullptr || lokr_w2_b == nullptr) {
                    break;
                }

                rank = lokr_w2_b->ne[ggml_n_dims(lokr_w2_b) - 1];

                lokr_w2 = ggml_ext_merge_lora(compute_ctx, lokr_w2_b, lokr_w2_a);
            }

            if (!lokr_w1_a) {
                applied_lora_tensors.insert(lokr_w1_name);
            } else {
                applied_lora_tensors.insert(lokr_w1_a_name);
                applied_lora_tensors.insert(lokr_w1_b_name);
            }

            if (!lokr_w2_a) {
                applied_lora_tensors.insert(lokr_w2_name);
            } else {
                applied_lora_tensors.insert(lokr_w2_a_name);
                applied_lora_tensors.insert(lokr_w2_b_name);
            }

            float scale_value = 1.0f;
            iter              = lora_tensors.find(alpha_name);
            if (iter != lora_tensors.end()) {
                float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                scale_value = alpha / rank;
                applied_lora_tensors.insert(alpha_name);
            }

            if (rank == 1) {
                scale_value = 1.0f;
            }

            scale_value *= multiplier;

            auto curr_updown = ggml_ext_kronecker(compute_ctx, lokr_w1, lokr_w2);
            curr_updown      = ggml_scale_inplace(compute_ctx, curr_updown, scale_value);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(compute_ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    struct ggml_cgraph* build_lora_graph(const std::map<std::string, ggml_tensor*>& model_tensors, SDVersion version) {
        size_t lora_graph_size = LORA_GRAPH_BASE_SIZE + lora_tensors.size() * 10;
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, lora_graph_size, false);

        zero_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        set_backend_tensor_data(zero_index, zero_index_vec.data());
        ggml_build_forward_expand(gf, zero_index);

        preprocess_lora_tensors(model_tensors);

        original_tensor_to_final_tensor.clear();

        std::set<std::string> applied_lora_tensors;
        for (auto it : model_tensors) {
            std::string model_tensor_name = it.first;
            ggml_tensor* model_tensor     = it.second;

            // lora
            ggml_tensor* updown = get_lora_diff(model_tensor_name, applied_lora_tensors);
            // loha
            if (updown == nullptr) {
                updown = get_loha_diff(model_tensor_name, applied_lora_tensors);
            }

            // lokr
            if (updown == nullptr) {
                updown = get_lokr_diff(model_tensor_name, applied_lora_tensors);
            }

            if (updown == nullptr) {
                continue;
            }

            ggml_tensor* original_tensor = model_tensor;
            if (!ggml_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                model_tensor = ggml_dup_tensor(compute_ctx, model_tensor);
                set_backend_tensor_data(model_tensor, original_tensor->data);
            }

            if (ggml_nelements(updown) < ggml_nelements(model_tensor)) {
                if (ggml_n_dims(updown) == 2 && ggml_n_dims(model_tensor) == 2 && updown->ne[0] == model_tensor->ne[0]) {
                    LOG_WARN("pad for %s", model_tensor_name.c_str());
                    auto pad_tensor = ggml_ext_zeros(compute_ctx, updown->ne[0], model_tensor->ne[1] - updown->ne[1], 1, 1);
                    updown          = ggml_concat(compute_ctx, updown, pad_tensor, 1);
                }
            }

            GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(model_tensor));
            updown = ggml_reshape(compute_ctx, updown, model_tensor);
            ggml_tensor* final_tensor;
            if (model_tensor->type != GGML_TYPE_F32 && model_tensor->type != GGML_TYPE_F16) {
                final_tensor = to_f32(compute_ctx, model_tensor);
                final_tensor = ggml_add_inplace(compute_ctx, final_tensor, updown);
                final_tensor = ggml_cpy(compute_ctx, final_tensor, model_tensor);
            } else {
                final_tensor = ggml_add_inplace(compute_ctx, model_tensor, updown);
            }
            ggml_build_forward_expand(gf, final_tensor);
            if (!ggml_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                original_tensor_to_final_tensor[original_tensor] = final_tensor;
            }
        }
        size_t total_lora_tensors_count   = 0;
        size_t applied_lora_tensors_count = 0;

        for (auto& kv : lora_tensors) {
            total_lora_tensors_count++;
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                LOG_WARN("unused lora tensor |%s|", kv.first.c_str());
                print_ggml_tensor(kv.second, true);
                // exit(0);
            } else {
                applied_lora_tensors_count++;
            }
        }
        /* Don't worry if this message shows up twice in the logs per LoRA,
         * this function is called once to calculate the required buffer size
         * and then again to actually generate a graph to be used */
        if (applied_lora_tensors_count != total_lora_tensors_count) {
            LOG_WARN("Only (%lu / %lu) LoRA tensors will be applied",
                     applied_lora_tensors_count, total_lora_tensors_count);
        } else {
            LOG_DEBUG("(%lu / %lu) LoRA tensors will be applied",
                      applied_lora_tensors_count, total_lora_tensors_count);
        }

        return gf;
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, SDVersion version, int n_threads) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_lora_graph(model_tensors, version);
        };
        GGMLRunner::compute(get_graph, n_threads, false);
        for (auto item : original_tensor_to_final_tensor) {
            ggml_tensor* original_tensor = item.first;
            ggml_tensor* final_tensor    = item.second;

            ggml_backend_tensor_copy(final_tensor, original_tensor);
        }
        original_tensor_to_final_tensor.clear();
        GGMLRunner::free_compute_buffer();
    }
};

#endif  // __LORA_HPP__
