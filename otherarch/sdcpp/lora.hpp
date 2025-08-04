#ifndef __LORA_HPP__
#define __LORA_HPP__

#include "ggml_extend.hpp"

#define LORA_GRAPH_BASE_SIZE 10240

struct LoraModel : public GGMLRunner {
    enum lora_t {
        REGULAR      = 0,
        DIFFUSERS    = 1,
        DIFFUSERS_2  = 2,
        DIFFUSERS_3  = 3,
        TRANSFORMERS = 4,
        LORA_TYPE_COUNT
    };

    const std::string lora_ups[LORA_TYPE_COUNT] = {
        ".lora_up",
        "_lora.up",
        ".lora_B",
        ".lora.up",
        ".lora_linear_layer.up",
    };

    const std::string lora_downs[LORA_TYPE_COUNT] = {
        ".lora_down",
        "_lora.down",
        ".lora_A",
        ".lora.down",
        ".lora_linear_layer.down",
    };

    const std::string lora_pre[LORA_TYPE_COUNT] = {
        "lora.",
        "",
        "",
        "",
        "",
    };

    const std::map<std::string, std::string> alt_names = {
        // mmdit
        {"final_layer.adaLN_modulation.1", "norm_out.linear"},
        {"pos_embed", "pos_embed.proj"},
        {"final_layer.linear", "proj_out"},
        {"y_embedder.mlp.0", "time_text_embed.text_embedder.linear_1"},
        {"y_embedder.mlp.2", "time_text_embed.text_embedder.linear_2"},
        {"t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1"},
        {"t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2"},
        {"x_block.mlp.fc1", "ff.net.0.proj"},
        {"x_block.mlp.fc2", "ff.net.2"},
        {"context_block.mlp.fc1", "ff_context.net.0.proj"},
        {"context_block.mlp.fc2", "ff_context.net.2"},
        {"x_block.adaLN_modulation.1", "norm1.linear"},
        {"context_block.adaLN_modulation.1", "norm1_context.linear"},
        {"context_block.attn.proj", "attn.to_add_out"},
        {"x_block.attn.proj", "attn.to_out.0"},
        {"x_block.attn2.proj", "attn2.to_out.0"},
        // flux
        // singlestream
        {"linear2", "proj_out"},
        {"modulation.lin", "norm.linear"},
        // doublestream
        {"txt_attn.proj", "attn.to_add_out"},
        {"img_attn.proj", "attn.to_out.0"},
        {"txt_mlp.0", "ff_context.net.0.proj"},
        {"txt_mlp.2", "ff_context.net.2"},
        {"img_mlp.0", "ff.net.0.proj"},
        {"img_mlp.2", "ff.net.2"},
        {"txt_mod.lin", "norm1_context.linear"},
        {"img_mod.lin", "norm1.linear"},
    };

    const std::map<std::string, std::string> qkv_prefixes = {
        // mmdit
        {"context_block.attn.qkv", "attn.add_"},  // suffix "_proj"
        {"x_block.attn.qkv", "attn.to_"},
        {"x_block.attn2.qkv", "attn2.to_"},
        // flux
        // doublestream
        {"txt_attn.qkv", "attn.add_"},  // suffix "_proj"
        {"img_attn.qkv", "attn.to_"},
    };
    const std::map<std::string, std::string> qkvm_prefixes = {
        // flux
        // singlestream
        {"linear1", ""},
    };

    const std::string* type_fingerprints = lora_ups;

    float multiplier = 1.0f;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed                = false;
    bool applied                    = false;
    std::vector<int> zero_index_vec = {0};
    ggml_tensor* zero_index         = NULL;
    enum lora_t type                = REGULAR;

    LoraModel(ggml_backend_t backend,
              const std::string& file_path = "",
              const std::string prefix     = "")
        : file_path(file_path), GGMLRunner(backend) {
        if (!model_loader.init_from_file(file_path, prefix)) {
            load_failed = true;
        }
    }

    std::string get_desc() {
        return "lora";
    }

    bool load_from_file(bool filter_tensor = false) {
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;

            if (filter_tensor && !contains(name, "lora")) {
                // LOG_INFO("skipping LoRA tesnor '%s'", name.c_str());
                return true;
            }
            // LOG_INFO("%s", name.c_str());
            for (int i = 0; i < LORA_TYPE_COUNT; i++) {
                if (name.find(type_fingerprints[i]) != std::string::npos) {
                    type = (lora_t)i;
                    break;
                }
            }

            if (dry_run) {
                struct ggml_tensor* real = ggml_new_tensor(params_ctx,
                                                           tensor_storage.type,
                                                           tensor_storage.n_dims,
                                                           tensor_storage.ne);
                lora_tensors[name]       = real;
            } else {
                auto real   = lora_tensors[name];
                *dst_tensor = real;
            }

            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, backend);
        alloc_params_buffer();
        // exit(0);
        dry_run = false;
        model_loader.load_tensors(on_new_tensor_cb, backend);

        LOG_DEBUG("lora type: \"%s\"/\"%s\"", lora_downs[type].c_str(), lora_ups[type].c_str());

        LOG_DEBUG("finished loaded lora");
        return true;
    }

    ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* a) {
        auto out = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out      = ggml_get_rows(ctx, out, zero_index);
        out      = ggml_reshape(ctx, out, a);
        return out;
    }

    std::vector<std::string> to_lora_keys(std::string blk_name, SDVersion version) {
        std::vector<std::string> keys;
        // if (!sd_version_is_sd3(version) || blk_name != "model.diffusion_model.pos_embed") {
        size_t k_pos = blk_name.find(".weight");
        if (k_pos == std::string::npos) {
            return keys;
        }
        blk_name = blk_name.substr(0, k_pos);
        // }
        keys.push_back(blk_name);
        keys.push_back("lora." + blk_name);
        if (sd_version_is_dit(version)) {
            if (blk_name.find("model.diffusion_model") != std::string::npos) {
                blk_name.replace(blk_name.find("model.diffusion_model"), sizeof("model.diffusion_model") - 1, "transformer");
            }

            if (blk_name.find(".single_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".single_blocks"), sizeof(".single_blocks") - 1, ".single_transformer_blocks");
            }
            if (blk_name.find(".double_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".double_blocks"), sizeof(".double_blocks") - 1, ".transformer_blocks");
            }

            if (blk_name.find(".joint_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".joint_blocks"), sizeof(".joint_blocks") - 1, ".transformer_blocks");
            }

            if (blk_name.find("text_encoders.clip_l") != std::string::npos) {
                blk_name.replace(blk_name.find("text_encoders.clip_l"), sizeof("text_encoders.clip_l") - 1, "cond_stage_model");
            }

            for (const auto& item : alt_names) {
                size_t match = blk_name.find(item.first);
                if (match != std::string::npos) {
                    blk_name = blk_name.substr(0, match) + item.second;
                }
            }
            for (const auto& prefix : qkv_prefixes) {
                size_t match = blk_name.find(prefix.first);
                if (match != std::string::npos) {
                    std::string split_blk = "SPLIT|" + blk_name.substr(0, match) + prefix.second;
                    keys.push_back(split_blk);
                }
            }
            for (const auto& prefix : qkvm_prefixes) {
                size_t match = blk_name.find(prefix.first);
                if (match != std::string::npos) {
                    std::string split_blk = "SPLIT_L|" + blk_name.substr(0, match) + prefix.second;
                    keys.push_back(split_blk);
                }
            }
            keys.push_back(blk_name);
        }

        std::vector<std::string> ret;
        for (std::string& key : keys) {
            ret.push_back(key);
            replace_all_chars(key, '.', '_');
            // fix for some sdxl lora, like lcm-lora-xl
            if (key == "model_diffusion_model_output_blocks_2_2_conv") {
                ret.push_back("model_diffusion_model_output_blocks_2_1_conv");
            }
            ret.push_back(key);
        }
        return ret;
    }

    struct ggml_cgraph* build_lora_graph(std::map<std::string, struct ggml_tensor*> model_tensors, SDVersion version) {
        size_t lora_graph_size = LORA_GRAPH_BASE_SIZE + lora_tensors.size() * 10;
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, lora_graph_size, false);

        zero_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        set_backend_tensor_data(zero_index, zero_index_vec.data());
        ggml_build_forward_expand(gf, zero_index);

        std::set<std::string> applied_lora_tensors;
        for (auto it : model_tensors) {
            std::string k_tensor       = it.first;
            struct ggml_tensor* weight = model_tensors[it.first];

            std::vector<std::string> keys = to_lora_keys(k_tensor, version);
            if (keys.size() == 0)
                continue;

            for (auto& key : keys) {
                bool is_qkv_split = starts_with(key, "SPLIT|");
                if (is_qkv_split) {
                    key = key.substr(sizeof("SPLIT|") - 1);
                }
                bool is_qkvm_split = starts_with(key, "SPLIT_L|");
                if (is_qkvm_split) {
                    key = key.substr(sizeof("SPLIT_L|") - 1);
                }
                struct ggml_tensor* updown = NULL;
                float scale_value          = 1.0f;
                std::string fk             = lora_pre[type] + key;
                if (lora_tensors.find(fk + ".hada_w1_a") != lora_tensors.end()) {
                    // LoHa mode

                    // TODO: split qkv convention for LoHas (is it ever used?)
                    if (is_qkv_split || is_qkvm_split) {
                        LOG_ERROR("Split qkv isn't supported for LoHa models.");
                        break;
                    }
                    std::string alpha_name = "";

                    ggml_tensor* hada_1_mid  = NULL;  // tau for tucker decomposition
                    ggml_tensor* hada_1_up   = NULL;
                    ggml_tensor* hada_1_down = NULL;

                    ggml_tensor* hada_2_mid  = NULL;  // tau for tucker decomposition
                    ggml_tensor* hada_2_up   = NULL;
                    ggml_tensor* hada_2_down = NULL;

                    std::string hada_1_mid_name  = "";
                    std::string hada_1_down_name = "";
                    std::string hada_1_up_name   = "";

                    std::string hada_2_mid_name  = "";
                    std::string hada_2_down_name = "";
                    std::string hada_2_up_name   = "";

                    hada_1_down_name = fk + ".hada_w1_b";
                    hada_1_up_name   = fk + ".hada_w1_a";
                    hada_1_mid_name  = fk + ".hada_t1";
                    if (lora_tensors.find(hada_1_down_name) != lora_tensors.end()) {
                        hada_1_down = to_f32(compute_ctx, lora_tensors[hada_1_down_name]);
                    }
                    if (lora_tensors.find(hada_1_up_name) != lora_tensors.end()) {
                        hada_1_up = to_f32(compute_ctx, lora_tensors[hada_1_up_name]);
                    }
                    if (lora_tensors.find(hada_1_mid_name) != lora_tensors.end()) {
                        hada_1_mid = to_f32(compute_ctx, lora_tensors[hada_1_mid_name]);
                        applied_lora_tensors.insert(hada_1_mid_name);
                        hada_1_up = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, hada_1_up));
                    }

                    hada_2_down_name = fk + ".hada_w2_b";
                    hada_2_up_name   = fk + ".hada_w2_a";
                    hada_2_mid_name  = fk + ".hada_t2";
                    if (lora_tensors.find(hada_2_down_name) != lora_tensors.end()) {
                        hada_2_down = to_f32(compute_ctx, lora_tensors[hada_2_down_name]);
                    }
                    if (lora_tensors.find(hada_2_up_name) != lora_tensors.end()) {
                        hada_2_up = to_f32(compute_ctx, lora_tensors[hada_2_up_name]);
                    }
                    if (lora_tensors.find(hada_2_mid_name) != lora_tensors.end()) {
                        hada_2_mid = to_f32(compute_ctx, lora_tensors[hada_2_mid_name]);
                        applied_lora_tensors.insert(hada_2_mid_name);
                        hada_2_up = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, hada_2_up));
                    }

                    alpha_name = fk + ".alpha";

                    applied_lora_tensors.insert(hada_1_down_name);
                    applied_lora_tensors.insert(hada_1_up_name);
                    applied_lora_tensors.insert(hada_2_down_name);
                    applied_lora_tensors.insert(hada_2_up_name);

                    applied_lora_tensors.insert(alpha_name);
                    if (hada_1_up == NULL || hada_1_down == NULL || hada_2_up == NULL || hada_2_down == NULL) {
                        continue;
                    }

                    struct ggml_tensor* updown_1 = ggml_merge_lora(compute_ctx, hada_1_down, hada_1_up, hada_1_mid);
                    struct ggml_tensor* updown_2 = ggml_merge_lora(compute_ctx, hada_2_down, hada_2_up, hada_2_mid);
                    updown                       = ggml_mul_inplace(compute_ctx, updown_1, updown_2);

                    // calc_scale
                    // TODO: .dora_scale?
                    int64_t rank = hada_1_down->ne[ggml_n_dims(hada_1_down) - 1];
                    if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                        float alpha = ggml_backend_tensor_get_f32(lora_tensors[alpha_name]);
                        scale_value = alpha / rank;
                    }
                } else if (lora_tensors.find(fk + ".lokr_w1") != lora_tensors.end() || lora_tensors.find(fk + ".lokr_w1_a") != lora_tensors.end()) {
                    // LoKr mode

                    // TODO: split qkv convention for LoKrs (is it ever used?)
                    if (is_qkv_split || is_qkvm_split) {
                        LOG_ERROR("Split qkv isn't supported for LoKr models.");
                        break;
                    }

                    std::string alpha_name = fk + ".alpha";

                    ggml_tensor* lokr_w1 = NULL;
                    ggml_tensor* lokr_w2 = NULL;

                    std::string lokr_w1_name = "";
                    std::string lokr_w2_name = "";

                    lokr_w1_name = fk + ".lokr_w1";
                    lokr_w2_name = fk + ".lokr_w2";

                    if (lora_tensors.find(lokr_w1_name) != lora_tensors.end()) {
                        lokr_w1 = to_f32(compute_ctx, lora_tensors[lokr_w1_name]);
                        applied_lora_tensors.insert(lokr_w1_name);
                    } else {
                        ggml_tensor* down     = NULL;
                        ggml_tensor* up       = NULL;
                        std::string down_name = lokr_w1_name + "_b";
                        std::string up_name   = lokr_w1_name + "_a";
                        if (lora_tensors.find(down_name) != lora_tensors.end()) {
                            // w1 should not be low rank normally, sometimes w1 and w2 are swapped
                            down = to_f32(compute_ctx, lora_tensors[down_name]);
                            applied_lora_tensors.insert(down_name);

                            int64_t rank = down->ne[ggml_n_dims(down) - 1];
                            if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                                float alpha = ggml_backend_tensor_get_f32(lora_tensors[alpha_name]);
                                scale_value = alpha / rank;
                            }
                        }
                        if (lora_tensors.find(up_name) != lora_tensors.end()) {
                            up = to_f32(compute_ctx, lora_tensors[up_name]);
                            applied_lora_tensors.insert(up_name);
                        }
                        lokr_w1 = ggml_merge_lora(compute_ctx, down, up);
                    }
                    if (lora_tensors.find(lokr_w2_name) != lora_tensors.end()) {
                        lokr_w2 = to_f32(compute_ctx, lora_tensors[lokr_w2_name]);
                        applied_lora_tensors.insert(lokr_w2_name);
                    } else {
                        ggml_tensor* down     = NULL;
                        ggml_tensor* up       = NULL;
                        std::string down_name = lokr_w2_name + "_b";
                        std::string up_name   = lokr_w2_name + "_a";
                        if (lora_tensors.find(down_name) != lora_tensors.end()) {
                            down = to_f32(compute_ctx, lora_tensors[down_name]);
                            applied_lora_tensors.insert(down_name);

                            int64_t rank = down->ne[ggml_n_dims(down) - 1];
                            if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                                float alpha = ggml_backend_tensor_get_f32(lora_tensors[alpha_name]);
                                scale_value = alpha / rank;
                            }
                        }
                        if (lora_tensors.find(up_name) != lora_tensors.end()) {
                            up = to_f32(compute_ctx, lora_tensors[up_name]);
                            applied_lora_tensors.insert(up_name);
                        }
                        lokr_w2 = ggml_merge_lora(compute_ctx, down, up);
                    }

                    // Technically it might be unused, but I believe it's the expected behavior
                    applied_lora_tensors.insert(alpha_name);

                    updown = ggml_kronecker(compute_ctx, lokr_w1, lokr_w2);

                } else {
                    // LoRA mode
                    ggml_tensor* lora_mid  = NULL;  // tau for tucker decomposition
                    ggml_tensor* lora_up   = NULL;
                    ggml_tensor* lora_down = NULL;

                    std::string alpha_name         = "";
                    std::string scale_name         = "";
                    std::string split_q_scale_name = "";
                    std::string lora_mid_name      = "";
                    std::string lora_down_name     = "";
                    std::string lora_up_name       = "";

                    if (is_qkv_split) {
                        std::string suffix  = "";
                        auto split_q_d_name = fk + "q" + suffix + lora_downs[type] + ".weight";

                        if (lora_tensors.find(split_q_d_name) == lora_tensors.end()) {
                            suffix         = "_proj";
                            split_q_d_name = fk + "q" + suffix + lora_downs[type] + ".weight";
                        }
                        if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                            // print_ggml_tensor(it.second, true);  //[3072, 21504, 1, 1]
                            // find qkv and mlp up parts in LoRA model
                            auto split_k_d_name = fk + "k" + suffix + lora_downs[type] + ".weight";
                            auto split_v_d_name = fk + "v" + suffix + lora_downs[type] + ".weight";

                            auto split_q_u_name = fk + "q" + suffix + lora_ups[type] + ".weight";
                            auto split_k_u_name = fk + "k" + suffix + lora_ups[type] + ".weight";
                            auto split_v_u_name = fk + "v" + suffix + lora_ups[type] + ".weight";

                            auto split_q_scale_name = fk + "q" + suffix + ".scale";
                            auto split_k_scale_name = fk + "k" + suffix + ".scale";
                            auto split_v_scale_name = fk + "v" + suffix + ".scale";

                            auto split_q_alpha_name = fk + "q" + suffix + ".alpha";
                            auto split_k_alpha_name = fk + "k" + suffix + ".alpha";
                            auto split_v_alpha_name = fk + "v" + suffix + ".alpha";

                            ggml_tensor* lora_q_down = NULL;
                            ggml_tensor* lora_q_up   = NULL;
                            ggml_tensor* lora_k_down = NULL;
                            ggml_tensor* lora_k_up   = NULL;
                            ggml_tensor* lora_v_down = NULL;
                            ggml_tensor* lora_v_up   = NULL;

                            lora_q_down = to_f32(compute_ctx, lora_tensors[split_q_d_name]);

                            if (lora_tensors.find(split_q_u_name) != lora_tensors.end()) {
                                lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);
                            }

                            if (lora_tensors.find(split_k_d_name) != lora_tensors.end()) {
                                lora_k_down = to_f32(compute_ctx, lora_tensors[split_k_d_name]);
                            }

                            if (lora_tensors.find(split_k_u_name) != lora_tensors.end()) {
                                lora_k_up = to_f32(compute_ctx, lora_tensors[split_k_u_name]);
                            }

                            if (lora_tensors.find(split_v_d_name) != lora_tensors.end()) {
                                lora_v_down = to_f32(compute_ctx, lora_tensors[split_v_d_name]);
                            }

                            if (lora_tensors.find(split_v_u_name) != lora_tensors.end()) {
                                lora_v_up = to_f32(compute_ctx, lora_tensors[split_v_u_name]);
                            }

                            float q_rank = lora_q_up->ne[0];
                            float k_rank = lora_k_up->ne[0];
                            float v_rank = lora_v_up->ne[0];

                            float lora_q_scale = 1;
                            float lora_k_scale = 1;
                            float lora_v_scale = 1;

                            if (lora_tensors.find(split_q_scale_name) != lora_tensors.end()) {
                                lora_q_scale = ggml_backend_tensor_get_f32(lora_tensors[split_q_scale_name]);
                                applied_lora_tensors.insert(split_q_scale_name);
                            }
                            if (lora_tensors.find(split_k_scale_name) != lora_tensors.end()) {
                                lora_k_scale = ggml_backend_tensor_get_f32(lora_tensors[split_k_scale_name]);
                                applied_lora_tensors.insert(split_k_scale_name);
                            }
                            if (lora_tensors.find(split_v_scale_name) != lora_tensors.end()) {
                                lora_v_scale = ggml_backend_tensor_get_f32(lora_tensors[split_v_scale_name]);
                                applied_lora_tensors.insert(split_v_scale_name);
                            }

                            if (lora_tensors.find(split_q_alpha_name) != lora_tensors.end()) {
                                float lora_q_alpha = ggml_backend_tensor_get_f32(lora_tensors[split_q_alpha_name]);
                                applied_lora_tensors.insert(split_q_alpha_name);
                                lora_q_scale = lora_q_alpha / q_rank;
                            }
                            if (lora_tensors.find(split_k_alpha_name) != lora_tensors.end()) {
                                float lora_k_alpha = ggml_backend_tensor_get_f32(lora_tensors[split_k_alpha_name]);
                                applied_lora_tensors.insert(split_k_alpha_name);
                                lora_k_scale = lora_k_alpha / k_rank;
                            }
                            if (lora_tensors.find(split_v_alpha_name) != lora_tensors.end()) {
                                float lora_v_alpha = ggml_backend_tensor_get_f32(lora_tensors[split_v_alpha_name]);
                                applied_lora_tensors.insert(split_v_alpha_name);
                                lora_v_scale = lora_v_alpha / v_rank;
                            }

                            ggml_scale_inplace(compute_ctx, lora_q_down, lora_q_scale);
                            ggml_scale_inplace(compute_ctx, lora_k_down, lora_k_scale);
                            ggml_scale_inplace(compute_ctx, lora_v_down, lora_v_scale);

                            // print_ggml_tensor(lora_q_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_k_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_v_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_q_up, true);    //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_k_up, true);    //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_v_up, true);    //[R, 3072, 1, 1]

                            // these need to be stitched together this way:
                            //                          |q_up,0   ,0   |
                            //                          |0   ,k_up,0   |
                            //                          |0   ,0   ,v_up|
                            // (q_down,k_down,v_down) . (q   ,k   ,v)

                            // up_concat will be [9216, R*3, 1, 1]
                            // down_concat will be [R*3, 3072, 1, 1]
                            ggml_tensor* lora_down_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_down, lora_k_down, 1), lora_v_down, 1);

                            ggml_tensor* z = ggml_dup_tensor(compute_ctx, lora_q_up);
                            ggml_scale(compute_ctx, z, 0);
                            ggml_tensor* zz = ggml_concat(compute_ctx, z, z, 1);

                            ggml_tensor* q_up = ggml_concat(compute_ctx, lora_q_up, zz, 1);
                            ggml_tensor* k_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, z, lora_k_up, 1), z, 1);
                            ggml_tensor* v_up = ggml_concat(compute_ctx, zz, lora_v_up, 1);
                            // print_ggml_tensor(q_up, true);  //[R, 9216, 1, 1]
                            // print_ggml_tensor(k_up, true);  //[R, 9216, 1, 1]
                            // print_ggml_tensor(v_up, true);  //[R, 9216, 1, 1]
                            ggml_tensor* lora_up_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, q_up, k_up, 0), v_up, 0);
                            // print_ggml_tensor(lora_up_concat, true);  //[R*3, 9216, 1, 1]

                            lora_down = ggml_cont(compute_ctx, lora_down_concat);
                            lora_up   = ggml_cont(compute_ctx, lora_up_concat);

                            applied_lora_tensors.insert(split_q_u_name);
                            applied_lora_tensors.insert(split_k_u_name);
                            applied_lora_tensors.insert(split_v_u_name);

                            applied_lora_tensors.insert(split_q_d_name);
                            applied_lora_tensors.insert(split_k_d_name);
                            applied_lora_tensors.insert(split_v_d_name);
                        }
                    } else if (is_qkvm_split) {
                        auto split_q_d_name = fk + "attn.to_q" + lora_downs[type] + ".weight";
                        if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                            // print_ggml_tensor(it.second, true);  //[3072, 21504, 1, 1]
                            // find qkv and mlp up parts in LoRA model
                            auto split_k_d_name = fk + "attn.to_k" + lora_downs[type] + ".weight";
                            auto split_v_d_name = fk + "attn.to_v" + lora_downs[type] + ".weight";

                            auto split_q_u_name = fk + "attn.to_q" + lora_ups[type] + ".weight";
                            auto split_k_u_name = fk + "attn.to_k" + lora_ups[type] + ".weight";
                            auto split_v_u_name = fk + "attn.to_v" + lora_ups[type] + ".weight";

                            auto split_m_d_name = fk + "proj_mlp" + lora_downs[type] + ".weight";
                            auto split_m_u_name = fk + "proj_mlp" + lora_ups[type] + ".weight";

                            auto split_q_scale_name = fk + "attn.to_q" + ".scale";
                            auto split_k_scale_name = fk + "attn.to_k" + ".scale";
                            auto split_v_scale_name = fk + "attn.to_v" + ".scale";
                            auto split_m_scale_name = fk + "proj_mlp" + ".scale";

                            auto split_q_alpha_name = fk + "attn.to_q" + ".alpha";
                            auto split_k_alpha_name = fk + "attn.to_k" + ".alpha";
                            auto split_v_alpha_name = fk + "attn.to_v" + ".alpha";
                            auto split_m_alpha_name = fk + "proj_mlp" + ".alpha";

                            ggml_tensor* lora_q_down = NULL;
                            ggml_tensor* lora_q_up   = NULL;
                            ggml_tensor* lora_k_down = NULL;
                            ggml_tensor* lora_k_up   = NULL;
                            ggml_tensor* lora_v_down = NULL;
                            ggml_tensor* lora_v_up   = NULL;

                            ggml_tensor* lora_m_down = NULL;
                            ggml_tensor* lora_m_up   = NULL;

                            lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);

                            if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                                lora_q_down = to_f32(compute_ctx, lora_tensors[split_q_d_name]);
                            }

                            if (lora_tensors.find(split_q_u_name) != lora_tensors.end()) {
                                lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);
                            }

                            if (lora_tensors.find(split_k_d_name) != lora_tensors.end()) {
                                lora_k_down = to_f32(compute_ctx, lora_tensors[split_k_d_name]);
                            }

                            if (lora_tensors.find(split_k_u_name) != lora_tensors.end()) {
                                lora_k_up = to_f32(compute_ctx, lora_tensors[split_k_u_name]);
                            }

                            if (lora_tensors.find(split_v_d_name) != lora_tensors.end()) {
                                lora_v_down = to_f32(compute_ctx, lora_tensors[split_v_d_name]);
                            }

                            if (lora_tensors.find(split_v_u_name) != lora_tensors.end()) {
                                lora_v_up = to_f32(compute_ctx, lora_tensors[split_v_u_name]);
                            }

                            if (lora_tensors.find(split_m_d_name) != lora_tensors.end()) {
                                lora_m_down = to_f32(compute_ctx, lora_tensors[split_m_d_name]);
                            }

                            if (lora_tensors.find(split_m_u_name) != lora_tensors.end()) {
                                lora_m_up = to_f32(compute_ctx, lora_tensors[split_m_u_name]);
                            }

                            float q_rank = lora_q_up->ne[0];
                            float k_rank = lora_k_up->ne[0];
                            float v_rank = lora_v_up->ne[0];
                            float m_rank = lora_v_up->ne[0];

                            float lora_q_scale = 1;
                            float lora_k_scale = 1;
                            float lora_v_scale = 1;
                            float lora_m_scale = 1;

                            if (lora_tensors.find(split_q_scale_name) != lora_tensors.end()) {
                                lora_q_scale = ggml_backend_tensor_get_f32(lora_tensors[split_q_scale_name]);
                                applied_lora_tensors.insert(split_q_scale_name);
                            }
                            if (lora_tensors.find(split_k_scale_name) != lora_tensors.end()) {
                                lora_k_scale = ggml_backend_tensor_get_f32(lora_tensors[split_k_scale_name]);
                                applied_lora_tensors.insert(split_k_scale_name);
                            }
                            if (lora_tensors.find(split_v_scale_name) != lora_tensors.end()) {
                                lora_v_scale = ggml_backend_tensor_get_f32(lora_tensors[split_v_scale_name]);
                                applied_lora_tensors.insert(split_v_scale_name);
                            }
                            if (lora_tensors.find(split_m_scale_name) != lora_tensors.end()) {
                                lora_m_scale = ggml_backend_tensor_get_f32(lora_tensors[split_m_scale_name]);
                                applied_lora_tensors.insert(split_m_scale_name);
                            }

                            if (lora_tensors.find(split_q_alpha_name) != lora_tensors.end()) {
                                float lora_q_alpha = ggml_backend_tensor_get_f32(lora_tensors[split_q_alpha_name]);
                                applied_lora_tensors.insert(split_q_alpha_name);
                                lora_q_scale = lora_q_alpha / q_rank;
                            }
                            if (lora_tensors.find(split_k_alpha_name) != lora_tensors.end()) {
                                float lora_k_alpha = ggml_backend_tensor_get_f32(lora_tensors[split_k_alpha_name]);
                                applied_lora_tensors.insert(split_k_alpha_name);
                                lora_k_scale = lora_k_alpha / k_rank;
                            }
                            if (lora_tensors.find(split_v_alpha_name) != lora_tensors.end()) {
                                float lora_v_alpha = ggml_backend_tensor_get_f32(lora_tensors[split_v_alpha_name]);
                                applied_lora_tensors.insert(split_v_alpha_name);
                                lora_v_scale = lora_v_alpha / v_rank;
                            }
                            if (lora_tensors.find(split_m_alpha_name) != lora_tensors.end()) {
                                float lora_m_alpha = ggml_backend_tensor_get_f32(lora_tensors[split_m_alpha_name]);
                                applied_lora_tensors.insert(split_m_alpha_name);
                                lora_m_scale = lora_m_alpha / m_rank;
                            }

                            ggml_scale_inplace(compute_ctx, lora_q_down, lora_q_scale);
                            ggml_scale_inplace(compute_ctx, lora_k_down, lora_k_scale);
                            ggml_scale_inplace(compute_ctx, lora_v_down, lora_v_scale);
                            ggml_scale_inplace(compute_ctx, lora_m_down, lora_m_scale);

                            // print_ggml_tensor(lora_q_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_k_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_v_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_m_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_q_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_k_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_v_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_m_up, true);  //[R, 12288, 1, 1]

                            // these need to be stitched together this way:
                            //                                 |q_up,0   ,0   ,0   |
                            //                                 |0   ,k_up,0   ,0   |
                            //                                 |0   ,0   ,v_up,0   |
                            //                                 |0   ,0   ,0   ,m_up|
                            // (q_down,k_down,v_down,m_down) . (q   ,k   ,v   ,m)

                            // up_concat will be [21504, R*4, 1, 1]
                            // down_concat will be [R*4, 3072, 1, 1]

                            ggml_tensor* lora_down_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_down, lora_k_down, 1), ggml_concat(compute_ctx, lora_v_down, lora_m_down, 1), 1);
                            // print_ggml_tensor(lora_down_concat, true);  //[3072, R*4, 1, 1]

                            // this also means that if rank is bigger than 672, it is less memory efficient to do it this way (should be fine)
                            // print_ggml_tensor(lora_q_up, true);  //[3072, R, 1, 1]
                            ggml_tensor* z     = ggml_dup_tensor(compute_ctx, lora_q_up);
                            ggml_tensor* mlp_z = ggml_dup_tensor(compute_ctx, lora_m_up);
                            ggml_scale(compute_ctx, z, 0);
                            ggml_scale(compute_ctx, mlp_z, 0);
                            ggml_tensor* zz = ggml_concat(compute_ctx, z, z, 1);

                            ggml_tensor* q_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_up, zz, 1), mlp_z, 1);
                            ggml_tensor* k_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, z, lora_k_up, 1), ggml_concat(compute_ctx, z, mlp_z, 1), 1);
                            ggml_tensor* v_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, zz, lora_v_up, 1), mlp_z, 1);
                            ggml_tensor* m_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, zz, z, 1), lora_m_up, 1);
                            // print_ggml_tensor(q_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(k_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(v_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(m_up, true);  //[R, 21504, 1, 1]

                            ggml_tensor* lora_up_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, q_up, k_up, 0), ggml_concat(compute_ctx, v_up, m_up, 0), 0);
                            // print_ggml_tensor(lora_up_concat, true);  //[R*4, 21504, 1, 1]

                            lora_down = ggml_cont(compute_ctx, lora_down_concat);
                            lora_up   = ggml_cont(compute_ctx, lora_up_concat);

                            applied_lora_tensors.insert(split_q_u_name);
                            applied_lora_tensors.insert(split_k_u_name);
                            applied_lora_tensors.insert(split_v_u_name);
                            applied_lora_tensors.insert(split_m_u_name);

                            applied_lora_tensors.insert(split_q_d_name);
                            applied_lora_tensors.insert(split_k_d_name);
                            applied_lora_tensors.insert(split_v_d_name);
                            applied_lora_tensors.insert(split_m_d_name);
                        }
                    } else {
                        lora_up_name   = fk + lora_ups[type] + ".weight";
                        lora_down_name = fk + lora_downs[type] + ".weight";
                        lora_mid_name  = fk + ".lora_mid.weight";

                        alpha_name = fk + ".alpha";
                        scale_name = fk + ".scale";

                        if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                            lora_up = to_f32(compute_ctx, lora_tensors[lora_up_name]);
                        }

                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            lora_down = to_f32(compute_ctx, lora_tensors[lora_down_name]);
                        }

                        if (lora_tensors.find(lora_mid_name) != lora_tensors.end()) {
                            lora_mid = to_f32(compute_ctx, lora_tensors[lora_mid_name]);
                            applied_lora_tensors.insert(lora_mid_name);
                        }

                        applied_lora_tensors.insert(lora_up_name);
                        applied_lora_tensors.insert(lora_down_name);
                        applied_lora_tensors.insert(alpha_name);
                        applied_lora_tensors.insert(scale_name);
                    }

                    if (lora_up == NULL || lora_down == NULL) {
                        continue;
                    }
                    // calc_scale
                    // TODO: .dora_scale?
                    int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
                    if (lora_tensors.find(scale_name) != lora_tensors.end()) {
                        scale_value = ggml_backend_tensor_get_f32(lora_tensors[scale_name]);
                    } else if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                        float alpha = ggml_backend_tensor_get_f32(lora_tensors[alpha_name]);
                        scale_value = alpha / rank;
                    }

                    updown = ggml_merge_lora(compute_ctx, lora_down, lora_up, lora_mid);
                }
                scale_value *= multiplier;
                updown = ggml_reshape(compute_ctx, updown, weight);
                GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
                updown = ggml_scale_inplace(compute_ctx, updown, scale_value);
                ggml_tensor* final_weight;
                if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                    // final_weight = ggml_new_tensor(compute_ctx, GGML_TYPE_F32, ggml_n_dims(weight), weight->ne);
                    // final_weight = ggml_cpy(compute_ctx, weight, final_weight);
                    final_weight = to_f32(compute_ctx, weight);
                    final_weight = ggml_add_inplace(compute_ctx, final_weight, updown);
                    final_weight = ggml_cpy(compute_ctx, final_weight, weight);
                } else {
                    final_weight = ggml_add_inplace(compute_ctx, weight, updown);
                }
                // final_weight = ggml_add_inplace(compute_ctx, weight, updown);  // apply directly
                ggml_build_forward_expand(gf, final_weight);
                break;
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
            LOG_WARN("Only (%lu / %lu) LoRA tensors have been applied",
                     applied_lora_tensors_count, total_lora_tensors_count);
        } else {
            LOG_DEBUG("(%lu / %lu) LoRA tensors applied successfully",
                      applied_lora_tensors_count, total_lora_tensors_count);
        }

        return gf;
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, SDVersion version, int n_threads) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_lora_graph(model_tensors, version);
        };
        GGMLRunner::compute(get_graph, n_threads, true);
    }
};

#endif  // __LORA_HPP__
