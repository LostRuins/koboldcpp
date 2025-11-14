#ifndef __LORA_HPP__
#define __LORA_HPP__

#include <mutex>
#include "ggml_extend.hpp"

#define LORA_GRAPH_BASE_SIZE 10240

struct LoraModel : public GGMLRunner {
    std::string lora_id;
    float multiplier = 1.0f;
    std::unordered_map<std::string, struct ggml_tensor*> lora_tensors;
    std::map<ggml_tensor*, ggml_tensor*> original_tensor_to_final_tensor;
    std::set<std::string> applied_lora_tensors;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed         = false;
    bool applied             = false;
    bool tensor_preprocessed = false;

    typedef std::function<bool(const std::string&)> filter_t;

    LoraModel(const std::string& lora_id,
              ggml_backend_t backend,
              const std::string& file_path = "",
              std::string prefix           = "",
              SDVersion version            = VERSION_COUNT)
        : lora_id(lora_id), file_path(file_path), GGMLRunner(backend, false) {
        prefix = "lora." + prefix;
        if (!model_loader.init_from_file_and_convert_name(file_path, prefix, version)) {
            load_failed = true;
        }
    }

    std::string get_desc() override {
        return "lora";
    }

    bool load_from_file(int n_threads, filter_t filter = nullptr) {
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

                if (filter && !filter(name)) {
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

        if (tensors_to_create.empty()) {
            return true;
        }

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

    void preprocess_lora_tensors(const std::map<std::string, ggml_tensor*>& model_tensors) {
        if (tensor_preprocessed) {
            return;
        }
        tensor_preprocessed = true;
        // I really hate these hardcoded processes.
        if (model_tensors.find("cond_stage_model.1.transformer.text_model.encoder.layers.0.self_attn.in_proj.weight") != model_tensors.end()) {
            std::unordered_map<std::string, ggml_tensor*> new_lora_tensors;
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

    ggml_tensor* get_lora_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
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
                lora_up = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(lora_mid_name);
            if (iter != lora_tensors.end()) {
                lora_mid = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(lora_down_name);
            if (iter != lora_tensors.end()) {
                lora_down = ggml_ext_cast_f32(ctx, iter->second);
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

            auto curr_updown = ggml_ext_merge_lora(ctx, lora_down, lora_up, lora_mid);
            curr_updown      = ggml_scale_inplace(ctx, curr_updown, scale_value);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }

            index++;
        }
        return updown;
    }

    ggml_tensor* get_raw_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }

            std::string diff_name = "lora." + key + ".diff";

            ggml_tensor* curr_updown = nullptr;

            auto iter = lora_tensors.find(diff_name);
            if (iter != lora_tensors.end()) {
                curr_updown = ggml_ext_cast_f32(ctx, iter->second);
            } else {
                break;
            }

            applied_lora_tensors.insert(diff_name);

            float scale_value = 1.0f;
            scale_value *= multiplier;

            curr_updown = ggml_scale_inplace(ctx, curr_updown, scale_value);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }

            index++;
        }
        return updown;
    }

    ggml_tensor* get_loha_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
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
                hada_1_down = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_1_up_name);
            if (iter != lora_tensors.end()) {
                hada_1_up = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_1_mid_name);
            if (iter != lora_tensors.end()) {
                hada_1_mid = ggml_ext_cast_f32(ctx, iter->second);
                hada_1_up  = ggml_cont(ctx, ggml_transpose(ctx, hada_1_up));
            }

            iter = lora_tensors.find(hada_2_down_name);
            if (iter != lora_tensors.end()) {
                hada_2_down = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_2_up_name);
            if (iter != lora_tensors.end()) {
                hada_2_up = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_2_mid_name);
            if (iter != lora_tensors.end()) {
                hada_2_mid = ggml_ext_cast_f32(ctx, iter->second);
                hada_2_up  = ggml_cont(ctx, ggml_transpose(ctx, hada_2_up));
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

            struct ggml_tensor* updown_1 = ggml_ext_merge_lora(ctx, hada_1_down, hada_1_up, hada_1_mid);
            struct ggml_tensor* updown_2 = ggml_ext_merge_lora(ctx, hada_2_down, hada_2_up, hada_2_mid);
            auto curr_updown             = ggml_mul_inplace(ctx, updown_1, updown_2);
            curr_updown                  = ggml_scale_inplace(ctx, curr_updown, scale_value);
            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    ggml_tensor* get_lokr_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
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
                lokr_w1 = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(lokr_w2_name);
            if (iter != lora_tensors.end()) {
                lokr_w2 = ggml_ext_cast_f32(ctx, iter->second);
            }

            int64_t rank = 1;
            if (lokr_w1 == nullptr) {
                iter = lora_tensors.find(lokr_w1_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_a = ggml_ext_cast_f32(ctx, iter->second);
                }

                iter = lora_tensors.find(lokr_w1_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_b = ggml_ext_cast_f32(ctx, iter->second);
                }

                if (lokr_w1_a == nullptr || lokr_w1_b == nullptr) {
                    break;
                }

                rank = lokr_w1_b->ne[ggml_n_dims(lokr_w1_b) - 1];

                lokr_w1 = ggml_ext_merge_lora(ctx, lokr_w1_b, lokr_w1_a);
            }

            if (lokr_w2 == nullptr) {
                iter = lora_tensors.find(lokr_w2_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_a = ggml_ext_cast_f32(ctx, iter->second);
                }

                iter = lora_tensors.find(lokr_w2_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_b = ggml_ext_cast_f32(ctx, iter->second);
                }

                if (lokr_w2_a == nullptr || lokr_w2_b == nullptr) {
                    break;
                }

                rank = lokr_w2_b->ne[ggml_n_dims(lokr_w2_b) - 1];

                lokr_w2 = ggml_ext_merge_lora(ctx, lokr_w2_b, lokr_w2_a);
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

            auto curr_updown = ggml_ext_kronecker(ctx, lokr_w1, lokr_w2);
            curr_updown      = ggml_scale_inplace(ctx, curr_updown, scale_value);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    ggml_tensor* get_weight_diff(const std::string& model_tensor_name, ggml_context* ctx, ggml_tensor* model_tensor, bool with_lora = true) {
        // lora
        ggml_tensor* diff = nullptr;
        if (with_lora) {
            diff = get_lora_weight_diff(model_tensor_name, ctx);
        }
        // diff
        if (diff == nullptr) {
            diff = get_raw_weight_diff(model_tensor_name, ctx);
        }
        // loha
        if (diff == nullptr) {
            diff = get_loha_weight_diff(model_tensor_name, ctx);
        }
        // lokr
        if (diff == nullptr) {
            diff = get_lokr_weight_diff(model_tensor_name, ctx);
        }
        if (diff != nullptr) {
            if (ggml_nelements(diff) < ggml_nelements(model_tensor)) {
                if (ggml_n_dims(diff) == 2 && ggml_n_dims(model_tensor) == 2 && diff->ne[0] == model_tensor->ne[0]) {
                    LOG_WARN("pad for %s", model_tensor_name.c_str());
                    auto pad_tensor = ggml_ext_zeros(ctx, diff->ne[0], model_tensor->ne[1] - diff->ne[1], 1, 1);
                    diff            = ggml_concat(ctx, diff, pad_tensor, 1);
                }
            }

            GGML_ASSERT(ggml_nelements(diff) == ggml_nelements(model_tensor));
            diff = ggml_reshape(ctx, diff, model_tensor);
        }
        return diff;
    }

    ggml_tensor* get_out_diff(ggml_context* ctx,
                              ggml_tensor* x,
                              WeightAdapter::ForwardParams forward_params,
                              const std::string& model_tensor_name) {
        ggml_tensor* out_diff = nullptr;
        int index             = 0;
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

            bool is_conv2d = forward_params.op_type == WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;

            auto iter = lora_tensors.find(lora_up_name);
            if (iter != lora_tensors.end()) {
                lora_up = iter->second;
                if (is_conv2d && lora_up->type != GGML_TYPE_F16) {
                    lora_up = ggml_cast(ctx, lora_up, GGML_TYPE_F16);
                }
            }

            iter = lora_tensors.find(lora_mid_name);
            if (iter != lora_tensors.end()) {
                lora_mid = iter->second;
                if (is_conv2d && lora_mid->type != GGML_TYPE_F16) {
                    lora_mid = ggml_cast(ctx, lora_mid, GGML_TYPE_F16);
                }
            }

            iter = lora_tensors.find(lora_down_name);
            if (iter != lora_tensors.end()) {
                lora_down = iter->second;
                if (is_conv2d && lora_down->type != GGML_TYPE_F16) {
                    lora_down = ggml_cast(ctx, lora_down, GGML_TYPE_F16);
                }
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

            ggml_tensor* lx;
            if (!is_conv2d) {
                lx = ggml_ext_linear(ctx, x, lora_down, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
                if (lora_mid) {
                    lx = ggml_ext_linear(ctx, lx, lora_mid, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
                }
                lx = ggml_ext_linear(ctx, lx, lora_up, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
            } else {  // OP_CONV2D
                lx = ggml_ext_conv_2d(ctx,
                                      x,
                                      lora_down,
                                      nullptr,
                                      forward_params.conv2d.s0,
                                      forward_params.conv2d.s1,
                                      forward_params.conv2d.p0,
                                      forward_params.conv2d.p1,
                                      forward_params.conv2d.d0,
                                      forward_params.conv2d.d1,
                                      forward_params.conv2d.direct,
                                      forward_params.conv2d.scale);
                if (lora_mid) {
                    lx = ggml_ext_conv_2d(ctx,
                                          lx,
                                          lora_mid,
                                          nullptr,
                                          1,
                                          1,
                                          0,
                                          0,
                                          1,
                                          1,
                                          forward_params.conv2d.direct,
                                          forward_params.conv2d.scale);
                }
                lx = ggml_ext_conv_2d(ctx,
                                      lx,
                                      lora_up,
                                      nullptr,
                                      1,
                                      1,
                                      0,
                                      0,
                                      1,
                                      1,
                                      forward_params.conv2d.direct,
                                      forward_params.conv2d.scale);
            }

            auto curr_out_diff = ggml_scale_inplace(ctx, lx, scale_value);

            if (out_diff == nullptr) {
                out_diff = curr_out_diff;
            } else {
                out_diff = ggml_concat(ctx, out_diff, curr_out_diff, ggml_n_dims(out_diff) - 1);
            }

            index++;
        }
        return out_diff;
    }

    struct ggml_cgraph* build_lora_graph(const std::map<std::string, ggml_tensor*>& model_tensors, SDVersion version) {
        size_t lora_graph_size = LORA_GRAPH_BASE_SIZE + lora_tensors.size() * 10;
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, lora_graph_size, false);

        preprocess_lora_tensors(model_tensors);

        original_tensor_to_final_tensor.clear();
        applied_lora_tensors.clear();

        for (auto it : model_tensors) {
            std::string model_tensor_name = it.first;
            ggml_tensor* model_tensor     = it.second;

            // lora
            ggml_tensor* diff = get_weight_diff(model_tensor_name, compute_ctx, model_tensor);
            if (diff == nullptr) {
                continue;
            }

            ggml_tensor* original_tensor = model_tensor;
            if (!ggml_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                model_tensor = ggml_dup_tensor(compute_ctx, model_tensor);
                set_backend_tensor_data(model_tensor, original_tensor->data);
            }

            ggml_tensor* final_tensor;
            if (model_tensor->type != GGML_TYPE_F32 && model_tensor->type != GGML_TYPE_F16) {
                final_tensor = ggml_ext_cast_f32(compute_ctx, model_tensor);
                final_tensor = ggml_add_inplace(compute_ctx, final_tensor, diff);
                final_tensor = ggml_cpy(compute_ctx, final_tensor, model_tensor);
            } else {
                final_tensor = ggml_add_inplace(compute_ctx, model_tensor, diff);
            }
            ggml_build_forward_expand(gf, final_tensor);
            if (!ggml_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                original_tensor_to_final_tensor[original_tensor] = final_tensor;
            }
        }
        return gf;
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, SDVersion version, int n_threads) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_lora_graph(model_tensors, version);
        };
        GGMLRunner::compute(get_graph, n_threads, false);
        stat();
        for (auto item : original_tensor_to_final_tensor) {
            ggml_tensor* original_tensor = item.first;
            ggml_tensor* final_tensor    = item.second;

            ggml_backend_tensor_copy(final_tensor, original_tensor);
        }
        original_tensor_to_final_tensor.clear();
        GGMLRunner::free_compute_buffer();
    }

    void stat(bool at_runntime = false) {
        size_t total_lora_tensors_count   = 0;
        size_t applied_lora_tensors_count = 0;

        for (auto& kv : lora_tensors) {
            total_lora_tensors_count++;
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                if (!at_runntime) {
                    LOG_WARN("unused lora tensor |%s|", kv.first.c_str());
                    print_ggml_tensor(kv.second, true);
                }
            } else {
                applied_lora_tensors_count++;
            }
        }
        /* Don't worry if this message shows up twice in the logs per LoRA,
         * this function is called once to calculate the required buffer size
         * and then again to actually generate a graph to be used */
        if (!at_runntime && applied_lora_tensors_count != total_lora_tensors_count) {
            LOG_WARN("Only (%lu / %lu) LoRA tensors have been applied, lora_file_path = %s",
                     applied_lora_tensors_count, total_lora_tensors_count, file_path.c_str());
        } else {
            LOG_INFO("(%lu / %lu) LoRA tensors have been applied, lora_file_path = %s",
                     applied_lora_tensors_count, total_lora_tensors_count, file_path.c_str());
        }
    }
};

struct MultiLoraAdapter : public WeightAdapter {
protected:
    std::vector<std::shared_ptr<LoraModel>> lora_models;

public:
    explicit MultiLoraAdapter(const std::vector<std::shared_ptr<LoraModel>>& lora_models)
        : lora_models(lora_models) {
    }

    ggml_tensor* patch_weight(ggml_context* ctx, ggml_tensor* weight, const std::string& weight_name, bool with_lora) {
        for (auto& lora_model : lora_models) {
            ggml_tensor* diff = lora_model->get_weight_diff(weight_name, ctx, weight, with_lora);
            if (diff == nullptr) {
                continue;
            }

            if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                weight = ggml_ext_cast_f32(ctx, weight);
            }
            weight = ggml_add(ctx, weight, diff);
        }
        return weight;
    }

    ggml_tensor* patch_weight(ggml_context* ctx, ggml_tensor* weight, const std::string& weight_name) override {
        return patch_weight(ctx, weight, weight_name, true);
    }

    ggml_tensor* forward_with_lora(ggml_context* ctx,
                                   ggml_tensor* x,
                                   ggml_tensor* w,
                                   ggml_tensor* b,
                                   const std::string& prefix,
                                   WeightAdapter::ForwardParams forward_params) override {
        w = patch_weight(ctx, w, prefix + "weight", false);
        if (b) {
            b = patch_weight(ctx, b, prefix + "bias", false);
        }
        ggml_tensor* out;
        if (forward_params.op_type == ForwardParams::op_type_t::OP_LINEAR) {
            out = ggml_ext_linear(ctx, x, w, b, forward_params.linear.force_prec_f32, forward_params.linear.scale);
        } else {  // OP_CONV2D
            out = ggml_ext_conv_2d(ctx,
                                   x,
                                   w,
                                   b,
                                   forward_params.conv2d.s0,
                                   forward_params.conv2d.s1,
                                   forward_params.conv2d.p0,
                                   forward_params.conv2d.p1,
                                   forward_params.conv2d.d0,
                                   forward_params.conv2d.d1,
                                   forward_params.conv2d.direct,
                                   forward_params.conv2d.scale);
        }
        for (auto& lora_model : lora_models) {
            ggml_tensor* out_diff = lora_model->get_out_diff(ctx, x, forward_params, prefix + "weight");
            if (out_diff == nullptr) {
                continue;
            }
            out = ggml_add_inplace(ctx, out, out_diff);
        }
        return out;
    }

    size_t get_extra_graph_size() override {
        size_t lora_tensor_num = 0;
        for (auto& lora_model : lora_models) {
            lora_tensor_num += lora_model->lora_tensors.size();
        }
        return LORA_GRAPH_BASE_SIZE + lora_tensor_num * 10;
    }
};

#endif  // __LORA_HPP__
