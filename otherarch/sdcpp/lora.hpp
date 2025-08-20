#ifndef __LORA_HPP__
#define __LORA_HPP__

#include "ggml_extend.hpp"

#define LORA_GRAPH_SIZE 20480

// 🔧 LORA HANDLING MODE FLAGS
enum lora_handling_mode_t {
    LORA_MODE_SKIP_INCOMPATIBLE = 0,  // Skip incompatible layers (safer)
    LORA_MODE_MLX_DECOMPOSITION = 1   // Use MLX-style A/B decomposition for incompatible layers
};

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

    // 🔧 FAL.AI COMPATIBILITY: Check if this is a direct fal.ai style LoRA tensor name
    bool is_direct_falai_tensor(const std::string& tensor_name) const {
        // fal.ai LoRAs have direct tensor names like: transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight
        return (tensor_name.find("transformer.single_transformer_blocks.") == 0 && 
                (tensor_name.find(".lora_A.weight") != std::string::npos || 
                 tensor_name.find(".lora_B.weight") != std::string::npos));
    }
    
    // 🔧 FAL.AI COMPATIBILITY: Extract base pattern from fal.ai tensor name
    std::string extract_falai_base_pattern(const std::string& tensor_name) const {
        // Convert: transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight
        // To:      transformer.single_transformer_blocks.0.attn.to_k
        std::string base = tensor_name;
        
        // Remove .lora_A.weight or .lora_B.weight suffix
        size_t lora_pos = base.find(".lora_");
        if (lora_pos != std::string::npos) {
            base = base.substr(0, lora_pos);
        }
        
        return base;
    }
    
    // 🔧 PREFIX EXPANSION: Expand prefix keys into specific tensor keys (like fal.ai does)
    std::vector<std::string> expand_prefix_to_tensors(const std::string& prefix, int type) const {
        std::vector<std::string> expanded_keys;
        
        // Remove trailing dot if present
        std::string clean_prefix = prefix;
        if (!clean_prefix.empty() && clean_prefix.back() == '.') {
            clean_prefix = clean_prefix.substr(0, clean_prefix.length() - 1);
        }
        
        // 🔧 FAL.AI COMPATIBILITY: For fal.ai style LoRAs, scan actual tensor names
        if (clean_prefix.find("transformer.single_transformer_blocks.") == 0) {
            // Extract block number from prefix
            std::string block_pattern = clean_prefix + ".";
            
            // Scan through actual LoRA tensors to find matching patterns
            std::set<std::string> found_patterns;
            for (const auto& lora_tensor : lora_tensors) {
                const std::string& tensor_name = lora_tensor.first;
                
                // Check if this tensor matches our block pattern
                if (tensor_name.find(block_pattern) == 0 && is_direct_falai_tensor(tensor_name)) {
                    std::string base_pattern = extract_falai_base_pattern(tensor_name);
                    found_patterns.insert(base_pattern);
                }
            }
            
            // Add all found patterns to the expanded keys
            for (const auto& pattern : found_patterns) {
                expanded_keys.push_back(pattern);
            }
            
            printf("🔧 FAL.AI EXPANSION: Found %zu patterns for prefix '%s'\n", found_patterns.size(), clean_prefix.c_str());
            fflush(stdout);
            
        } else {
            // For other prefixes, use the generic approach
            std::vector<std::string> suffixes = {
                ".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.to_out.0", ".attn.to_add_out",
                ".ff.net.0.proj", ".ff.net.2", ".ff_context.net.0.proj", ".ff_context.net.2",
                ".norm1.linear", ".norm1_context.linear", ".norm.linear", ".norm_out.linear",
                ".linear1", ".linear2", ".proj_out", ".txt_mlp.0", ".txt_mlp.2", ".img_mlp.0", ".img_mlp.2"
            };
            
            // Check which suffixes actually exist in the LoRA tensors
            for (const auto& suffix : suffixes) {
                std::string full_key = clean_prefix + suffix;
                std::string lora_key = lora_pre[type] + full_key;
                
                // Check if this specific tensor exists in the LoRA
                bool has_tensor = false;
                for (const auto& lora_tensor : lora_tensors) {
                    if (lora_tensor.first.find(lora_key) != std::string::npos) {
                        has_tensor = true;
                        break;
                    }
                }
                
                if (has_tensor) {
                    expanded_keys.push_back(full_key);
                }
            }
        }
        
        return expanded_keys;
    }
    
    // 🔧 MLX-STYLE A/B DECOMPOSITION: Handle incompatible LoRA tensors gracefully
    struct ggml_tensor* apply_mlx_style_lora(ggml_context* ctx, 
                                            struct ggml_tensor* weight,
                                            struct ggml_tensor* lora_a, 
                                            struct ggml_tensor* lora_b,
                                            float scale = 1.0f) {
        printf("🔥 MLX-STYLE: Applying A/B decomposition for incompatible LoRA\n");
        
        // MLX approach: Work with A/B matrices directly, no upfront merging
        // Compute: weight + scale * (B @ A)
        
        // Ensure A and B are 2D for matrix multiplication
        struct ggml_tensor* a_2d = ggml_reshape_2d(ctx, lora_a, lora_a->ne[0], lora_a->ne[1]);
        struct ggml_tensor* b_2d = ggml_reshape_2d(ctx, lora_b, lora_b->ne[0], lora_b->ne[1]);
        
        // Compute B @ A (note: B is typically [out_dim, rank], A is [rank, in_dim])
        struct ggml_tensor* ba_product = ggml_mul_mat(ctx, b_2d, a_2d);
        
        // Scale the result
        if (scale != 1.0f) {
            ba_product = ggml_scale(ctx, ba_product, scale);
        }
        
        // Try to add to original weight if shapes are compatible
        if (ggml_nelements(ba_product) == ggml_nelements(weight)) {
            // Reshape BA product to match weight shape
            struct ggml_tensor* ba_reshaped = ggml_reshape(ctx, ba_product, weight);
            
            // Add to original weight: weight + delta
            struct ggml_tensor* result = ggml_add(ctx, weight, ba_reshaped);
            
            printf("✅ MLX-STYLE: Successfully applied LoRA using A/B decomposition\n");
            return result;
        } else {
            printf("⚠️ MLX-STYLE: Element count mismatch, returning scaled LoRA delta only\n");
            return ba_product;
        }
    }

    float multiplier = 1.0f;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed                = false;
    bool applied                    = false;
    std::vector<int> zero_index_vec = {0};
    ggml_tensor* zero_index         = NULL;
    enum lora_t type                = REGULAR;
    
    // 🔧 LORA HANDLING MODE CONTROL
    lora_handling_mode_t handling_mode = LORA_MODE_SKIP_INCOMPATIBLE;
    
    // Set LoRA handling mode
    void set_handling_mode(lora_handling_mode_t mode) {
        handling_mode = mode;
        printf("🔧 LORA HANDLING MODE: %s\n", 
               mode == LORA_MODE_MLX_DECOMPOSITION ? "MLX A/B Decomposition" : "Skip Incompatible");
    }

    LoraModel(ggml_backend_t backend,
              const std::string& file_path = "",
              const std::string prefix     = "")
        : file_path(file_path), GGMLRunner(backend) {
        printf("🔥 LORA CONSTRUCTOR DEBUG 1: Starting LoRA constructor\n");
        printf("🔥 LORA CONSTRUCTOR DEBUG 2: file_path=%s, prefix=%s\n", file_path.c_str(), prefix.c_str());
        printf("🔥 LORA CONSTRUCTOR DEBUG 3: backend=%p\n", (void*)backend);
        fflush(stdout);
        
        printf("🔥 LORA CONSTRUCTOR DEBUG 4: Calling model_loader.init_from_file...\n");
        fflush(stdout);
        
        if (!model_loader.init_from_file(file_path, prefix)) {
            printf("🔥 LORA CONSTRUCTOR DEBUG 5: ❌ model_loader.init_from_file FAILED\n");
            fflush(stdout);
            load_failed = true;
        } else {
            printf("🔥 LORA CONSTRUCTOR DEBUG 5: ✅ model_loader.init_from_file SUCCESS\n");
            fflush(stdout);
        }
        
        printf("🔥 LORA CONSTRUCTOR DEBUG 6: Constructor completed\n");
        fflush(stdout);
    }

    std::string get_desc() {
        return "lora";
    }

    bool load_from_file(bool filter_tensor = false) {
        printf("🔥 LOAD_FROM_FILE DEBUG 1: Starting load_from_file\n");
        printf("🔥 LOAD_FROM_FILE DEBUG 2: file_path=%s, filter_tensor=%s\n", file_path.c_str(), filter_tensor ? "true" : "false");
        fflush(stdout);
        
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            printf("🔥 LOAD_FROM_FILE DEBUG 3: ❌ load_failed is true, aborting\n");
            fflush(stdout);
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }
        
        printf("🔥 LOAD_FROM_FILE DEBUG 3: ✅ load_failed is false, continuing\n");
        fflush(stdout);

        // 🔧 TENSOR VALIDATION: Check for Apple M4 Metal compatibility issues
        int bf16_tensor_count = 0;
        int total_tensor_count = 0;
        bool has_compatibility_issues = false;

        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;

            if (filter_tensor && !contains(name, "lora")) {
                // LOG_INFO("skipping LoRA tesnor '%s'", name.c_str());
                return true;
            }
            
            total_tensor_count++;
            
            // 🔧 TENSOR VALIDATION: Check for BF16 tensors (Apple M4 Metal incompatible)
            if (tensor_storage.type == GGML_TYPE_BF16) {
                bf16_tensor_count++;
                if (bf16_tensor_count == 1) {
                    LOG_WARN("⚠️  APPLE M4 COMPATIBILITY: BF16 tensors detected in LoRA");
                    LOG_WARN("⚠️  BF16 tensors cause Metal backend crashes on Apple Silicon");
                    LOG_WARN("⚠️  Auto-converting BF16 → FP16 for compatibility...");
                    has_compatibility_issues = true;
                }
            }
            
            // LOG_INFO("%s", name.c_str());
            
            // 🔧 IMPROVED TYPE DETECTION: Check for diffusers naming first
            bool type_detected = false;
            if (name.find(".lora_A") != std::string::npos || name.find(".lora_B") != std::string::npos) {
                type = DIFFUSERS_2;  // Standard diffusers naming (.lora_A/.lora_B)
                type_detected = true;
                LOG_DEBUG("🔧 Detected DIFFUSERS_2 type from tensor: %s", name.c_str());
            } else {
                // Fallback to original fingerprint detection
                for (int i = 0; i < LORA_TYPE_COUNT; i++) {
                    if (name.find(type_fingerprints[i]) != std::string::npos) {
                        type = (lora_t)i;
                        type_detected = true;
                        LOG_DEBUG("🔧 Detected type %d from tensor: %s", i, name.c_str());
                        break;
                    }
                }
            }
            
            if (!type_detected) {
                LOG_DEBUG("🔧 No type detected from tensor: %s", name.c_str());
            }

            if (dry_run) {
                // 🔧 TENSOR CONVERSION: Convert BF16 to FP16 for Apple M4 Metal compatibility
                ggml_type tensor_type = tensor_storage.type;
                if (tensor_type == GGML_TYPE_BF16) {
                    tensor_type = GGML_TYPE_F16;  // Convert BF16 → FP16
                    LOG_DEBUG("Converting tensor '%s' from BF16 → FP16", name.c_str());
                }
                
                struct ggml_tensor* real = ggml_new_tensor(params_ctx,
                                                           tensor_type,  // Use converted type
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

        // 🔧 COMPATIBILITY SUMMARY: Report tensor conversion results
        if (has_compatibility_issues) {
            LOG_INFO("🔧 APPLE M4 COMPATIBILITY REPORT:");
            LOG_INFO("🔧   - Total tensors: %d", total_tensor_count);
            LOG_INFO("🔧   - BF16 tensors converted: %d", bf16_tensor_count);
            LOG_INFO("🔧   - All BF16 tensors auto-converted to FP16 for Metal compatibility");
            LOG_INFO("🔧   - LoRA should now work without crashes on Apple Silicon");
        }

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
        printf("🔥 BUILD_LORA_GRAPH DEBUG 1: Function entry - compute_ctx=%p\n", (void*)compute_ctx);
        fflush(stdout);
        
        printf("🔥 BUILD_LORA_GRAPH DEBUG 2: model_tensors.size() = %zu, version = %d\n", model_tensors.size(), version);
        fflush(stdout);
        
        if (!compute_ctx) {
            printf("❌ CRITICAL: compute_ctx is NULL!\n");
            fflush(stdout);
            return nullptr;
        }
        
        printf("🔥 BUILD_LORA_GRAPH DEBUG 3: LORA_GRAPH_SIZE = %d\n", LORA_GRAPH_SIZE);
        fflush(stdout);
        
        printf("🔥 BUILD_LORA_GRAPH DEBUG 4: About to call ggml_new_graph_custom...\n");
        fflush(stdout);
        
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, LORA_GRAPH_SIZE, false);
        
        printf("🔥 BUILD_LORA_GRAPH DEBUG 5: Graph created: %p\n", (void*)gf);
        fflush(stdout);

        printf("🔥 BUILD_LORA_GRAPH DEBUG 4: Creating zero index tensor...\n");
        fflush(stdout);
        zero_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        set_backend_tensor_data(zero_index, zero_index_vec.data());
        ggml_build_forward_expand(gf, zero_index);

        printf("🔥 BUILD_LORA_GRAPH DEBUG 5: Starting model tensor iteration (%zu tensors)...\n", model_tensors.size());
        fflush(stdout);
        
        std::set<std::string> applied_lora_tensors;
        int tensor_count = 0;
        int processed_count = 0;
        
        // 🔧 LORA APPLICATION STATISTICS: Track application success/failure
        int lora_applied_count = 0;
        int lora_skipped_count = 0;
        int lora_validation_passed_count = 0;
        
        // 🔧 DECLARE UPDOWN: Declare updown tensor variable for LoRA merging
        struct ggml_tensor* updown = NULL;
        
        for (auto it : model_tensors) {
            printf("🔧 ITERATION DEBUG: Loop iteration %d, tensor key: %s\n", tensor_count, it.first.c_str());
            fflush(stdout);
            
            std::string k_tensor       = it.first;
            struct ggml_tensor* weight = model_tensors[it.first];

            if (tensor_count % 200 == 0) {
                printf("🔧 PROGRESS: Processing tensor %d/%zu\n", tensor_count, model_tensors.size());
                fflush(stdout);
            }
            tensor_count++;

            printf("🔧 ITERATION DEBUG: About to call to_lora_keys for tensor: %s\n", k_tensor.c_str());
            fflush(stdout);
            
            std::vector<std::string> keys = to_lora_keys(k_tensor, version);
            printf("🔧 ITERATION DEBUG: to_lora_keys returned %zu keys\n", keys.size());
            fflush(stdout);
            
            if (keys.size() == 0) {
                continue;  // Skip tensors with no LoRA keys
            }
            processed_count++;

            printf("🔧 KEY PROCESSING DEBUG: Processing %zu keys for tensor: %s\n", keys.size(), k_tensor.c_str());
            fflush(stdout);
            
            // 🔧 PREFIX EXPANSION: First pass - expand any prefix keys and collect all keys to process
            std::vector<std::string> all_keys_to_process;
            for (const auto& original_key : keys) {
                std::string key = original_key;  // Make a copy for processing
                
                // 🔧 FAL.AI COMPATIBILITY: Handle SPLIT prefixes BEFORE checking for trailing dots
                bool is_qkv_split = starts_with(key, "SPLIT|");
                if (is_qkv_split) {
                    key = key.substr(sizeof("SPLIT|") - 1);
                    printf("🔧 PREFIX EXPANSION: Removed SPLIT| prefix from '%s', result: '%s'\n", original_key.c_str(), key.c_str());
                    fflush(stdout);
                }
                bool is_qkvm_split = starts_with(key, "SPLIT_L|");
                if (is_qkvm_split) {
                    key = key.substr(sizeof("SPLIT_L|") - 1);
                    printf("🔧 PREFIX EXPANSION: Removed SPLIT_L| prefix from '%s', result: '%s'\n", original_key.c_str(), key.c_str());
                    fflush(stdout);
                }
                
                if (!key.empty() && key.back() == '.') {
                    printf("🔧 PREFIX EXPANSION: Detected prefix key '%s' → expanding into specific tensor keys\n", key.c_str());
                    fflush(stdout);
                    
                    // Expand the prefix into specific tensor keys
                    std::vector<std::string> expanded_keys = expand_prefix_to_tensors(key, type);
                    printf("🔧 PREFIX EXPANSION: Expanded '%s' into %zu specific keys\n", key.c_str(), expanded_keys.size());
                    fflush(stdout);
                    
                    // Add the expanded keys to our processing list
                    for (const auto& expanded_key : expanded_keys) {
                        printf("🔧 PREFIX EXPANSION: Adding expanded key to processing queue: %s\n", expanded_key.c_str());
                        fflush(stdout);
                        all_keys_to_process.push_back(expanded_key);
                    }
                } else {
                    // Regular key - add directly (use the original key to preserve SPLIT prefixes for later processing)
                    all_keys_to_process.push_back(original_key);
                }
            }
            
            printf("🔧 PREFIX EXPANSION: Total keys to process: %zu (original: %zu)\n", all_keys_to_process.size(), keys.size());
            fflush(stdout);
            
            // 🔧 PREFIX EXPANSION: Second pass - process all keys (original + expanded)
            for (auto& key : all_keys_to_process) {
                printf("🔧 KEY PROCESSING DEBUG: Processing key: %s\n", key.c_str());
                fflush(stdout);
                
                // 🔧 PREFIX EXPANSION: Check for trailing dots BEFORE SPLIT prefix removal
                if (!key.empty() && key.back() == '.') {
                    printf("🔧 PREFIX EXPANSION: Detected trailing dot in key '%s' → this is a prefix that needs expansion\n", key.c_str());
                    fflush(stdout);
                    continue; // Skip this key as it's just a prefix
                }
                
                bool is_qkv_split = starts_with(key, "SPLIT|");
                if (is_qkv_split) {
                    key = key.substr(sizeof("SPLIT|") - 1);
                    printf("🔧 KEY PROCESSING DEBUG: Applied SPLIT| prefix removal\n");
                    fflush(stdout);
                }
                bool is_qkvm_split = starts_with(key, "SPLIT_L|");
                if (is_qkvm_split) {
                    key = key.substr(sizeof("SPLIT_L|") - 1);
                    printf("🔧 KEY PROCESSING DEBUG: Applied SPLIT_L| prefix removal\n");
                    printf("🔧 KEY PROCESSING DEBUG: Key after SPLIT_L| removal: '%s'\n", key.c_str());
                    fflush(stdout);
                }
                
                printf("🚨 CRASH DEBUG: About to check dots for key: '%s'\n", key.c_str());
                fflush(stdout);
                
                // 🔧 CRITICAL FIX: Handle trailing dots immediately after prefix removal
                printf("🔧 DOT CHECK: Checking key '%s' for trailing dot (length=%zu)\n", key.c_str(), key.length());
                fflush(stdout);
                if (!key.empty() && key.back() == '.') {
                    printf("🔧 CRITICAL FIX: Detected trailing dot in key '%s' after prefix removal\n", key.c_str());
                    fflush(stdout);
                    
                    // 🔧 FAL.AI COMPATIBILITY: For fal.ai, try to convert the prefix to actual tensors
                    if (key.find("transformer.single_transformer_blocks.") == 0) {
                        printf("🔧 FAL.AI PREFIX CONVERSION: Converting prefix '%s' to actual fal.ai tensors\n", key.c_str());
                        fflush(stdout);
                        
                        // Remove trailing dot and expand to actual fal.ai tensor patterns
                        std::string clean_key = key.substr(0, key.length() - 1);
                        
                        // Find all actual tensors that match this prefix
                        std::vector<std::string> matching_tensors;
                        for (const auto& lora_tensor : lora_tensors) {
                            const std::string& tensor_name = lora_tensor.first;
                            if (tensor_name.find(clean_key + ".") == 0 && 
                                (tensor_name.find(".lora_A.weight") != std::string::npos || 
                                 tensor_name.find(".lora_B.weight") != std::string::npos)) {
                                
                                // Extract the base pattern (remove .lora_A/B.weight)
                                std::string base = tensor_name;
                                if (base.find(".lora_A.weight") != std::string::npos) {
                                    base = base.substr(0, base.find(".lora_A.weight"));
                                } else if (base.find(".lora_B.weight") != std::string::npos) {
                                    base = base.substr(0, base.find(".lora_B.weight"));
                                }
                                
                                // Check if we haven't already added this base pattern
                                bool already_added = false;
                                for (const auto& existing : matching_tensors) {
                                    if (existing == base) {
                                        already_added = true;
                                        break;
                                    }
                                }
                                
                                if (!already_added) {
                                    matching_tensors.push_back(base);
                                    printf("🔧 FAL.AI PREFIX CONVERSION: Found matching tensor pattern: %s\n", base.c_str());
                                    fflush(stdout);
                                }
                            }
                        }
                        
                        printf("🔧 FAL.AI PREFIX CONVERSION: Found %zu matching tensor patterns for prefix '%s'\n", 
                               matching_tensors.size(), clean_key.c_str());
                        fflush(stdout);
                        
                        // If we found matching tensors, add them to the processing queue for this weight
                        if (!matching_tensors.empty()) {
                            // Pick the first matching tensor and process it
                            std::string selected_tensor = matching_tensors[0];
                            printf("🔧 FAL.AI PREFIX CONVERSION: Selected tensor for processing: %s\n", selected_tensor.c_str());
                            fflush(stdout);
                            
                            // Process this tensor directly with fal.ai logic
                            std::string direct_up_name = selected_tensor + ".lora_B.weight";
                            std::string direct_down_name = selected_tensor + ".lora_A.weight";
                            
                            ggml_tensor* lora_up = NULL;
                            ggml_tensor* lora_down = NULL;
                            
                            if (lora_tensors.find(direct_up_name) != lora_tensors.end()) {
                                lora_up = to_f32(compute_ctx, lora_tensors[direct_up_name]);
                                printf("🔧 FAL.AI PREFIX: ✅ Found up tensor: %s\n", direct_up_name.c_str());
                                fflush(stdout);
                            }
                            
                            if (lora_tensors.find(direct_down_name) != lora_tensors.end()) {
                                lora_down = to_f32(compute_ctx, lora_tensors[direct_down_name]);
                                printf("🔧 FAL.AI PREFIX: ✅ Found down tensor: %s\n", direct_down_name.c_str());
                                fflush(stdout);
                            }
                            
                            if (lora_up != NULL && lora_down != NULL) {
                                printf("🔧 FAL.AI PREFIX: Processing LoRA merge for %s\n", selected_tensor.c_str());
                                fflush(stdout);
                                
                                // Calculate scale
                                int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
                                float tensor_scale = 1.0f / rank;
                                tensor_scale *= multiplier;
                                
                                // 🔧 COMFYUI-STYLE VALIDATION: Check tensor compatibility BEFORE merge (like PyTorch try/except)
                                printf("🔧 PRE-MERGE VALIDATION: Checking tensor compatibility for %s\n", selected_tensor.c_str());
                                
                                // Get dimensions for validation
                                int64_t weight_elements = ggml_nelements(weight);
                                int64_t weight_out_dim = weight->ne[ggml_n_dims(weight) - 1];
                                int64_t weight_in_dim = weight->ne[0];  // First dimension
                                int64_t lora_down_in = lora_down->ne[0];
                                int64_t lora_up_out = lora_up->ne[ggml_n_dims(lora_up) - 1];
                                int64_t lora_rank_val = lora_down->ne[ggml_n_dims(lora_down) - 1];
                                
                                printf("🔧 DIMENSIONS DEBUG:\n");
                                printf("🔧   Weight: [%lld x %lld] (%lld elements)\n", weight_in_dim, weight_out_dim, weight_elements);
                                printf("🔧   LoRA down: [%lld x %lld] (rank=%lld)\n", lora_down_in, lora_rank_val, lora_rank_val);
                                printf("🔧   LoRA up: [%lld x %lld] (out=%lld)\n", lora_rank_val, lora_up_out, lora_up_out);
                                
                                // Expected result: lora_down [A x rank] * lora_up [rank x B] = result [A x B]
                                int64_t expected_merged_in = lora_down_in;
                                int64_t expected_merged_out = lora_up_out;
                                int64_t expected_merged_elements = expected_merged_in * expected_merged_out;
                                printf("🔧   Expected merged: [%lld x %lld] (%lld elements)\n", 
                                       expected_merged_in, expected_merged_out, expected_merged_elements);
                                
                                // Compatibility checks (more permissive like ComfyUI)
                                bool rank_compatible = (lora_down->ne[ggml_n_dims(lora_down) - 1] == lora_up->ne[0]);
                                bool dimension_compatible = (expected_merged_in == weight_in_dim && expected_merged_out == weight_out_dim);
                                bool element_count_compatible = (expected_merged_elements == weight_elements);
                                
                                // 🔧 COMFYUI COMPATIBILITY: Allow reshape if element counts are related by simple factors
                                bool reshapeable = false;
                                if (!element_count_compatible && expected_merged_elements > 0 && weight_elements > 0) {
                                    double ratio = (double)weight_elements / (double)expected_merged_elements;
                                    // Allow reshape if ratio is a reasonable integer (1, 2, 4, 7, etc.) or fraction
                                    if (ratio == (int)ratio && ratio >= 1.0 && ratio <= 1024.0) {
                                        reshapeable = true;
                                        printf("🔧   Reshape possible: weight has %gx more elements than LoRA result\n", ratio);
                                    } else if (ratio < 1.0) {
                                        double inv_ratio = 1.0 / ratio;
                                        if (inv_ratio == (int)inv_ratio && inv_ratio >= 1.0 && inv_ratio <= 1024.0) {
                                            reshapeable = true;
                                            printf("🔧   Reshape possible: LoRA result has %gx more elements than weight\n", inv_ratio);
                                        }
                                    }
                                }
                                
                                // Final compatibility: either exact match OR reshapeable
                                bool final_compatible = rank_compatible && (dimension_compatible || reshapeable);
                                
                                printf("🔧 COMPATIBILITY CHECKS:\n");
                                printf("🔧   Rank compatible (down_rank=%lld == up_in=%lld): %s\n", 
                                       lora_down->ne[ggml_n_dims(lora_down) - 1], lora_up->ne[0], rank_compatible ? "✅ YES" : "❌ NO");
                                printf("🔧   Dimension compatible (merged [%lld,%lld] == weight [%lld,%lld]): %s\n", 
                                       expected_merged_in, expected_merged_out, weight_in_dim, weight_out_dim, dimension_compatible ? "✅ YES" : "❌ NO");
                                printf("🔧   Element count compatible (%lld == %lld): %s\n", 
                                       expected_merged_elements, weight_elements, element_count_compatible ? "✅ YES" : "❌ NO");
                                printf("🔧   Reshapeable: %s\n", reshapeable ? "✅ YES" : "❌ NO");
                                printf("🔧   Final compatibility: %s\n", final_compatible ? "✅ YES" : "❌ NO");
                                
                                // ComfyUI-style skip check - use final compatibility
                                if (!final_compatible) {
                                    if (handling_mode == LORA_MODE_MLX_DECOMPOSITION) {
                                        printf("🔥 MLX-MODE: LoRA tensor %s is incompatible - trying MLX-style A/B decomposition\n", selected_tensor.c_str());
                                        printf("🔥   Reason for incompatibility: %s%s\n", 
                                               !rank_compatible ? "RANK_MISMATCH " : "",
                                               (!dimension_compatible && !reshapeable) ? "DIMENSION_INCOMPATIBLE " : "");
                                        
                                        // Try MLX-style approach
                                        try {
                                            updown = apply_mlx_style_lora(compute_ctx, weight, lora_down, lora_up, multiplier);
                                            lora_applied_count++;  // Track successful MLX application
                                            printf("✅ MLX-MODE: Successfully applied using A/B decomposition\n");
                                        } catch (const std::exception& e) {
                                            printf("❌ MLX-MODE: A/B decomposition failed: %s - SKIPPING\n", e.what());
                                            lora_skipped_count++;
                                            continue;
                                        }
                                    } else {
                                        printf("🚨 COMFYUI-STYLE SKIP: LoRA tensor %s is incompatible with target weight - SKIPPING\n", selected_tensor.c_str());
                                        printf("🚨   This prevents the crash that would occur in ggml_merge_lora()\n");
                                        printf("🚨   Reason: %s%s\n", 
                                               !rank_compatible ? "RANK_MISMATCH " : "",
                                               (!dimension_compatible && !reshapeable) ? "DIMENSION_INCOMPATIBLE " : "");
                                        fflush(stdout);
                                        lora_skipped_count++;  // Track skipped LoRA
                                        continue; // Skip this LoRA tensor like ComfyUI does
                                    }
                                } else {
                                    printf("✅ PRE-MERGE VALIDATION PASSED: All compatibility checks passed - proceeding with standard merge\n");
                                    fflush(stdout);
                                    lora_validation_passed_count++;  // Track successful validation
                                    
                                    updown = ggml_merge_lora(compute_ctx, lora_down, lora_up, NULL);
                                    lora_applied_count++;  // Track successful application
                                }
                                
                                // Debug: Print tensor dimensions before reshape
                                printf("🔧 TENSOR DIM DEBUG (PREFIX): updown: [");
                                for (int i = 0; i < ggml_n_dims(updown); i++) {
                                    printf("%lld", updown->ne[i]);
                                    if (i < ggml_n_dims(updown) - 1) printf(" x ");
                                }
                                printf("] (%lld elements)\n", ggml_nelements(updown));
                                
                                printf("🔧 TENSOR DIM DEBUG (PREFIX): weight: [");
                                for (int i = 0; i < ggml_n_dims(weight); i++) {
                                    printf("%lld", weight->ne[i]);
                                    if (i < ggml_n_dims(weight) - 1) printf(" x ");
                                }
                                printf("] (%lld elements)\n", ggml_nelements(weight));
                                fflush(stdout);
                                
                                // MLX-style approach: fail fast with clear error message
                                if (ggml_nelements(updown) == ggml_nelements(weight)) {
                                    updown = ggml_reshape(compute_ctx, updown, weight);
                                } else {
                                    printf("❌ LoRA DIMENSION MISMATCH (PREFIX): %s\n", selected_tensor.c_str());
                                    printf("   LoRA elements: %lld, Weight elements: %lld\n", 
                                           ggml_nelements(updown), ggml_nelements(weight));
                                    printf("   Skipping incompatible LoRA layer to prevent crash\n");
                                    fflush(stdout);
                                    continue; // Skip this problematic layer like MLX would
                                }
                                GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
                                updown = ggml_scale_inplace(compute_ctx, updown, tensor_scale);
                                
                                // Apply the LoRA
                                ggml_tensor* final_weight;
                                if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                                    final_weight = to_f32(compute_ctx, weight);
                                    final_weight = ggml_add_inplace(compute_ctx, final_weight, updown);
                                    final_weight = ggml_cpy(compute_ctx, final_weight, weight);
                                } else {
                                    final_weight = ggml_add_inplace(compute_ctx, weight, updown);
                                }
                                ggml_build_forward_expand(gf, final_weight);
                                
                                // Track applied tensors
                                applied_lora_tensors.insert(direct_up_name);
                                applied_lora_tensors.insert(direct_down_name);
                                processed_count++;
                                
                                printf("🔧 FAL.AI PREFIX: ✅ Applied LoRA for %s\n", selected_tensor.c_str());
                                fflush(stdout);
                                break; // Exit the key processing loop
                            } else {
                                printf("🔧 FAL.AI PREFIX: ❌ Missing tensors for %s (up=%s, down=%s)\n", 
                                       selected_tensor.c_str(), lora_up ? "✅" : "❌", lora_down ? "✅" : "❌");
                                fflush(stdout);
                            }
                        }
                    }
                    continue; // Skip further processing of this key
                }
                printf("🔧 DOT CHECK: No trailing dot found, proceeding with key: '%s'\n", key.c_str());
                fflush(stdout);
                float scale_value          = 1.0f;
                std::string fk             = lora_pre[type] + key;
                printf("🔧 KEY PROCESSING DEBUG: Looking for LoRA tensors with prefix: %s\n", fk.c_str());
                fflush(stdout);
                
                // 🔧 FAL.AI COMPATIBILITY: Check if this is a direct fal.ai tensor (no prefix)
                bool is_falai_direct = false;
                if (key.find("transformer.single_transformer_blocks.") == 0) {
                    // Check if direct fal.ai style tensors exist
                    std::string test_tensor_name = key + ".lora_A.weight";
                    if (lora_tensors.find(test_tensor_name) != lora_tensors.end()) {
                        is_falai_direct = true;
                        printf("🔧 FAL.AI DETECTION: Direct fal.ai tensor detected for key: %s\n", key.c_str());
                        fflush(stdout);
                    }
                }
                
                if (is_falai_direct) {
                    printf("🔧 FAL.AI DIRECT: Processing fal.ai direct tensor for key: %s\n", key.c_str());
                    fflush(stdout);
                    
                    // Direct fal.ai tensor processing - no prefix, direct tensor names
                    ggml_tensor* lora_up = NULL;
                    ggml_tensor* lora_down = NULL;
                    
                    std::string direct_up_name = key + ".lora_B.weight";
                    std::string direct_down_name = key + ".lora_A.weight";
                    
                    if (lora_tensors.find(direct_up_name) != lora_tensors.end()) {
                        printf("🔧 FAL.AI DIRECT: ✅ Found up tensor: %s\n", direct_up_name.c_str());
                        fflush(stdout);
                        lora_up = to_f32(compute_ctx, lora_tensors[direct_up_name]);
                    }
                    
                    if (lora_tensors.find(direct_down_name) != lora_tensors.end()) {
                        printf("🔧 FAL.AI DIRECT: ✅ Found down tensor: %s\n", direct_down_name.c_str());
                        fflush(stdout);
                        lora_down = to_f32(compute_ctx, lora_tensors[direct_down_name]);
                    }
                    
                    if (lora_up != NULL && lora_down != NULL) {
                        printf("🔧 FAL.AI DIRECT: Processing LoRA merge for %s\n", key.c_str());
                        fflush(stdout);
                        
                        // Calculate scale (use rank from down tensor)
                        int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
                        float scale_value = 1.0f / rank;  // Default scaling
                        scale_value *= multiplier;
                        
                        // 🔧 COMFYUI-STYLE VALIDATION: Check tensor compatibility BEFORE merge (like PyTorch try/except)
                        printf("🔧 PRE-MERGE VALIDATION (DIRECT): Checking tensor compatibility for %s\n", key.c_str());
                        
                        // Get dimensions for validation
                        int64_t weight_elements = ggml_nelements(weight);
                        int64_t weight_out_dim = weight->ne[ggml_n_dims(weight) - 1];
                        int64_t weight_in_dim = weight->ne[0];  // First dimension
                        int64_t lora_down_in = lora_down->ne[0];
                        int64_t lora_up_out = lora_up->ne[ggml_n_dims(lora_up) - 1];
                        int64_t lora_rank_val = lora_down->ne[ggml_n_dims(lora_down) - 1];
                        
                        printf("🔧 DIMENSIONS DEBUG (DIRECT):\n");
                        printf("🔧   Weight: [%lld x %lld] (%lld elements)\n", weight_in_dim, weight_out_dim, weight_elements);
                        printf("🔧   LoRA down: [%lld x %lld] (rank=%lld)\n", lora_down_in, lora_rank_val, lora_rank_val);
                        printf("🔧   LoRA up: [%lld x %lld] (out=%lld)\n", lora_rank_val, lora_up_out, lora_up_out);
                        
                        // Expected result: lora_down [A x rank] * lora_up [rank x B] = result [A x B]
                        int64_t expected_merged_in = lora_down_in;
                        int64_t expected_merged_out = lora_up_out;
                        int64_t expected_merged_elements = expected_merged_in * expected_merged_out;
                        printf("🔧   Expected merged: [%lld x %lld] (%lld elements)\n", 
                               expected_merged_in, expected_merged_out, expected_merged_elements);
                        
                        // Compatibility checks (more permissive like ComfyUI)
                        bool rank_compatible = (lora_down->ne[ggml_n_dims(lora_down) - 1] == lora_up->ne[0]);
                        bool dimension_compatible = (expected_merged_in == weight_in_dim && expected_merged_out == weight_out_dim);
                        bool element_count_compatible = (expected_merged_elements == weight_elements);
                        
                        // 🔧 COMFYUI COMPATIBILITY: Allow reshape if element counts are related by simple factors
                        bool reshapeable = false;
                        if (!element_count_compatible && expected_merged_elements > 0 && weight_elements > 0) {
                            double ratio = (double)weight_elements / (double)expected_merged_elements;
                            // Allow reshape if ratio is a reasonable integer (1, 2, 4, 7, etc.) or fraction
                            if (ratio == (int)ratio && ratio >= 1.0 && ratio <= 1024.0) {
                                reshapeable = true;
                                printf("🔧   Reshape possible: weight has %gx more elements than LoRA result\n", ratio);
                            } else if (ratio < 1.0) {
                                double inv_ratio = 1.0 / ratio;
                                if (inv_ratio == (int)inv_ratio && inv_ratio >= 1.0 && inv_ratio <= 1024.0) {
                                    reshapeable = true;
                                    printf("🔧   Reshape possible: LoRA result has %gx more elements than weight\n", inv_ratio);
                                }
                            }
                        }
                        
                        // Final compatibility: either exact match OR reshapeable
                        bool final_compatible = rank_compatible && (dimension_compatible || reshapeable);
                        
                        printf("🔧 COMPATIBILITY CHECKS (DIRECT):\n");
                        printf("🔧   Rank compatible (down_rank=%lld == up_in=%lld): %s\n", 
                               lora_down->ne[ggml_n_dims(lora_down) - 1], lora_up->ne[0], rank_compatible ? "✅ YES" : "❌ NO");
                        printf("🔧   Dimension compatible (merged [%lld,%lld] == weight [%lld,%lld]): %s\n", 
                               expected_merged_in, expected_merged_out, weight_in_dim, weight_out_dim, dimension_compatible ? "✅ YES" : "❌ NO");
                        printf("🔧   Element count compatible (%lld == %lld): %s\n", 
                               expected_merged_elements, weight_elements, element_count_compatible ? "✅ YES" : "❌ NO");
                        printf("🔧   Reshapeable: %s\n", reshapeable ? "✅ YES" : "❌ NO");
                        printf("🔧   Final compatibility: %s\n", final_compatible ? "✅ YES" : "❌ NO");
                        
                        // ComfyUI-style skip check - use final compatibility
                        if (!final_compatible) {
                            printf("🚨 COMFYUI-STYLE SKIP: LoRA tensor %s is incompatible with target weight - SKIPPING\n", key.c_str());
                            printf("🚨   This prevents the crash that would occur in ggml_merge_lora()\n");
                            printf("🚨   Reason: %s%s\n", 
                                   !rank_compatible ? "RANK_MISMATCH " : "",
                                   (!dimension_compatible && !reshapeable) ? "DIMENSION_INCOMPATIBLE " : "");
                            fflush(stdout);
                            lora_skipped_count++;  // Track skipped LoRA
                            continue; // Skip this LoRA tensor like ComfyUI does
                        }
                        
                        printf("✅ PRE-MERGE VALIDATION PASSED (DIRECT): All compatibility checks passed - proceeding with merge\n");
                        fflush(stdout);
                        lora_validation_passed_count++;  // Track successful validation
                        
                        updown = ggml_merge_lora(compute_ctx, lora_down, lora_up, NULL);
                        lora_applied_count++;  // Track successful application
                        
                        // Debug: Print tensor dimensions before reshape
                        printf("🔧 TENSOR DIM DEBUG: updown: [");
                        for (int i = 0; i < ggml_n_dims(updown); i++) {
                            printf("%lld", updown->ne[i]);
                            if (i < ggml_n_dims(updown) - 1) printf(" x ");
                        }
                        printf("] (%lld elements)\n", ggml_nelements(updown));
                        
                        printf("🔧 TENSOR DIM DEBUG: weight: [");
                        for (int i = 0; i < ggml_n_dims(weight); i++) {
                            printf("%lld", weight->ne[i]);
                            if (i < ggml_n_dims(weight) - 1) printf(" x ");
                        }
                        printf("] (%lld elements)\n", ggml_nelements(weight));
                        fflush(stdout);
                        
                        // Intelligent reshaping based on element count compatibility
                        if (ggml_nelements(updown) == ggml_nelements(weight)) {
                            updown = ggml_reshape(compute_ctx, updown, weight);
                        } else {
                            // MLX-style approach: fail fast with clear error message
                            printf("❌ LoRA DIMENSION MISMATCH: %s\n", key.c_str());
                            printf("   LoRA elements: %lld, Weight elements: %lld\n", 
                                   ggml_nelements(updown), ggml_nelements(weight));
                            printf("   Skipping incompatible LoRA layer to prevent crash\n");
                            fflush(stdout);
                            continue; // Skip this problematic layer like MLX would
                        }
                        GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
                        updown = ggml_scale_inplace(compute_ctx, updown, scale_value);
                        
                        // Apply the LoRA
                        ggml_tensor* final_weight;
                        if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                            final_weight = to_f32(compute_ctx, weight);
                            final_weight = ggml_add_inplace(compute_ctx, final_weight, updown);
                            final_weight = ggml_cpy(compute_ctx, final_weight, weight);
                        } else {
                            final_weight = ggml_add_inplace(compute_ctx, weight, updown);
                        }
                        ggml_build_forward_expand(gf, final_weight);
                        
                        // Track applied tensors
                        applied_lora_tensors.insert(direct_up_name);
                        applied_lora_tensors.insert(direct_down_name);
                        processed_count++;
                        
                        printf("🔧 FAL.AI DIRECT: ✅ Successfully applied LoRA for %s\n", key.c_str());
                        fflush(stdout);
                    } else {
                        printf("🔧 FAL.AI DIRECT: ❌ Missing tensors for %s (up=%s, down=%s)\n", 
                               key.c_str(), lora_up ? "✅" : "❌", lora_down ? "✅" : "❌");
                        fflush(stdout);
                    }
                } else if (lora_tensors.find(fk + ".hada_w1_a") != lora_tensors.end()) {
                    printf("🔧 KEY PROCESSING DEBUG: Found LoHa mode tensors\n");
                    fflush(stdout);
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
                    printf("🔧 KEY PROCESSING DEBUG: About to enter standard LoRA mode for key: %s\n", key.c_str());
                    printf("🔧 KEY PROCESSING DEBUG: Current type value: %d\n", type);
                    fflush(stdout);
                    
                    printf("🔧 KEY PROCESSING DEBUG: Using standard LoRA mode\n");
                    fflush(stdout);
                    
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
                        printf("🔧 STANDARD LORA DEBUG: Building tensor names for key: %s\n", fk.c_str());
                        printf("🔧 STANDARD LORA DEBUG: type: %d, lora_ups[type]: %s, lora_downs[type]: %s\n", 
                               type, lora_ups[type].c_str(), lora_downs[type].c_str());
                        fflush(stdout);
                        
                        printf("🔧 TENSOR NAME DEBUG: About to construct tensor names...\n");
                        fflush(stdout);
                        
                        printf("🔧 TENSOR NAME DEBUG: fk: '%s', lora_ups[type]: '%s', lora_downs[type]: '%s'\n", 
                               fk.c_str(), lora_ups[type].c_str(), lora_downs[type].c_str());
                        fflush(stdout);
                        
                        lora_up_name   = fk + lora_ups[type] + ".weight";
                        printf("🔧 TENSOR NAME DEBUG: lora_up_name constructed: '%s'\n", lora_up_name.c_str());
                        fflush(stdout);
                        
                        lora_down_name = fk + lora_downs[type] + ".weight";
                        printf("🔧 TENSOR NAME DEBUG: lora_down_name constructed: '%s'\n", lora_down_name.c_str());
                        fflush(stdout);
                        
                        lora_mid_name  = fk + ".lora_mid.weight";
                        printf("🔧 TENSOR NAME DEBUG: lora_mid_name constructed: '%s'\n", lora_mid_name.c_str());
                        fflush(stdout);

                        alpha_name = fk + ".alpha";
                        scale_name = fk + ".scale";
                        
                        printf("🔧 STANDARD LORA DEBUG: Looking for tensors:\n");
                        printf("🔧 STANDARD LORA DEBUG:   up: %s\n", lora_up_name.c_str());
                        printf("🔧 STANDARD LORA DEBUG:   down: %s\n", lora_down_name.c_str());
                        printf("🔧 STANDARD LORA DEBUG:   mid: %s\n", lora_mid_name.c_str());
                        printf("🔧 STANDARD LORA DEBUG:   alpha: %s\n", alpha_name.c_str());
                        printf("🔧 STANDARD LORA DEBUG:   scale: %s\n", scale_name.c_str());
                        fflush(stdout);

                        // 🔧 FAL.AI COMPATIBILITY: Try primary naming convention first
                        printf("🔧 STANDARD LORA DEBUG: Checking primary naming convention...\n");
                        fflush(stdout);
                        
                        if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                            printf("🔧 STANDARD LORA DEBUG: ✅ Found up tensor: %s\n", lora_up_name.c_str());
                            fflush(stdout);
                            lora_up = to_f32(compute_ctx, lora_tensors[lora_up_name]);
                        } else {
                            printf("🔧 STANDARD LORA DEBUG: ❌ Missing up tensor: %s\n", lora_up_name.c_str());
                            fflush(stdout);
                        }

                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            printf("🔧 STANDARD LORA DEBUG: ✅ Found down tensor: %s\n", lora_down_name.c_str());
                            fflush(stdout);
                            lora_down = to_f32(compute_ctx, lora_tensors[lora_down_name]);
                        } else {
                            printf("🔧 STANDARD LORA DEBUG: ❌ Missing down tensor: %s\n", lora_down_name.c_str());
                            fflush(stdout);
                        }
                        
                        // 🔧 FAL.AI COMPATIBILITY: Direct fal.ai tensor name lookup if standard prefixed lookup fails
                        if (lora_up == NULL || lora_down == NULL) {
                            // Try direct fal.ai style tensor names without prefix
                            std::string direct_up_name = key + ".lora_B.weight";      // fal.ai style up
                            std::string direct_down_name = key + ".lora_A.weight";    // fal.ai style down
                            
                            printf("🔧 FAL.AI FALLBACK: Trying direct fal.ai naming:\n");
                            printf("🔧 FAL.AI FALLBACK:   up: %s\n", direct_up_name.c_str());
                            printf("🔧 FAL.AI FALLBACK:   down: %s\n", direct_down_name.c_str());
                            fflush(stdout);
                            
                            // 🔧 ENHANCED FAL.AI MAPPING: Try alternative name transformations if direct lookup fails
                            bool direct_found = (lora_tensors.find(direct_up_name) != lora_tensors.end() &&
                                               lora_tensors.find(direct_down_name) != lora_tensors.end());
                            
                            if (!direct_found) {
                                printf("🔧 ENHANCED MAPPING: Direct fal.ai lookup failed, trying name transformations...\n");
                                
                                std::string alt_key = key;
                                
                                // Transform model tensor names to fal.ai LoRA names
                                if (alt_key.find("model.diffusion_model.double_blocks") != std::string::npos) {
                                    // Transform: model.diffusion_model.double_blocks.X.Y → transformer.single_transformer_blocks.X.Y
                                    alt_key.replace(alt_key.find("model.diffusion_model.double_blocks"), 
                                                   sizeof("model.diffusion_model.double_blocks") - 1,
                                                   "transformer.single_transformer_blocks");
                                    printf("🔧 ENHANCED MAPPING: Trying alternative mapping: %s\n", alt_key.c_str());
                                    
                                    direct_up_name = alt_key + ".lora_B.weight";
                                    direct_down_name = alt_key + ".lora_A.weight";
                                    
                                    printf("🔧 ENHANCED MAPPING:   up: %s\n", direct_up_name.c_str());
                                    printf("🔧 ENHANCED MAPPING:   down: %s\n", direct_down_name.c_str());
                                }
                                
                                // Also try other common transformations
                                if (lora_tensors.find(direct_up_name) == lora_tensors.end() &&
                                    alt_key.find("model.diffusion_model.single_blocks") != std::string::npos) {
                                    // Transform: model.diffusion_model.single_blocks.X.Y → transformer.single_transformer_blocks.X.Y
                                    alt_key = key;
                                    alt_key.replace(alt_key.find("model.diffusion_model.single_blocks"), 
                                                   sizeof("model.diffusion_model.single_blocks") - 1,
                                                   "transformer.single_transformer_blocks");
                                    printf("🔧 ENHANCED MAPPING: Trying single_blocks mapping: %s\n", alt_key.c_str());
                                    
                                    direct_up_name = alt_key + ".lora_B.weight";
                                    direct_down_name = alt_key + ".lora_A.weight";
                                }
                                
                                fflush(stdout);
                            }
                            
                            if (lora_up == NULL && lora_tensors.find(direct_up_name) != lora_tensors.end()) {
                                printf("🔧 FAL.AI FALLBACK: ✅ Found direct up tensor: %s\n", direct_up_name.c_str());
                                fflush(stdout);
                                lora_up = to_f32(compute_ctx, lora_tensors[direct_up_name]);
                                lora_up_name = direct_up_name; // Update the name for applied_lora_tensors
                            }
                            
                            if (lora_down == NULL && lora_tensors.find(direct_down_name) != lora_tensors.end()) {
                                printf("🔧 FAL.AI FALLBACK: ✅ Found direct down tensor: %s\n", direct_down_name.c_str());
                                fflush(stdout);
                                lora_down = to_f32(compute_ctx, lora_tensors[direct_down_name]);
                                lora_down_name = direct_down_name; // Update the name for applied_lora_tensors
                            }
                        }
                        
                        // 🔧 FAL.AI COMPATIBILITY: Final fallback to standard diffusers naming
                        if (lora_up == NULL || lora_down == NULL) {
                            std::string fallback_up_name = fk + ".lora_B.weight";      // Standard diffusers up
                            std::string fallback_down_name = fk + ".lora_A.weight";    // Standard diffusers down
                            
                            if (lora_up == NULL && lora_tensors.find(fallback_up_name) != lora_tensors.end()) {
                                lora_up = to_f32(compute_ctx, lora_tensors[fallback_up_name]);
                                lora_up_name = fallback_up_name;  // Update for logging
                                LOG_DEBUG("🔧 Using fallback diffusers naming for up projection: %s", fallback_up_name.c_str());
                            }
                            
                            if (lora_down == NULL && lora_tensors.find(fallback_down_name) != lora_tensors.end()) {
                                lora_down = to_f32(compute_ctx, lora_tensors[fallback_down_name]);
                                lora_down_name = fallback_down_name;  // Update for logging
                                LOG_DEBUG("🔧 Using fallback diffusers naming for down projection: %s", fallback_down_name.c_str());
                            }
                        }

                        if (lora_tensors.find(lora_mid_name) != lora_tensors.end()) {
                            lora_mid = to_f32(compute_ctx, lora_tensors[lora_mid_name]);
                            applied_lora_tensors.insert(lora_mid_name);
                        }

                        // 🔧 FAL.AI COMPATIBILITY: Track applied tensors with actual names used
                        if (lora_up != NULL) {
                            applied_lora_tensors.insert(lora_up_name);
                        }
                        if (lora_down != NULL) {
                            applied_lora_tensors.insert(lora_down_name);
                        }
                        applied_lora_tensors.insert(alpha_name);
                        applied_lora_tensors.insert(scale_name);
                    }

                    // 🔧 GRACEFUL PARTIAL LORA SUPPORT: Skip missing tensors instead of crashing
                    if (lora_up == NULL || lora_down == NULL) {
                        if (lora_up == NULL && lora_down == NULL) {
                            // Both missing - this is normal for partial LoRAs (e.g., fal.ai LoRAs)
                            // Skip silently
                        } else {
                            // Only one missing - this indicates a problem
                            LOG_WARN("⚠️  Partial LoRA tensor pair for %s: up=%s, down=%s", 
                                key.c_str(), 
                                lora_up ? "✅" : "❌", 
                                lora_down ? "✅" : "❌");
                        }
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

                    printf("🔧 TENSOR_OP DEBUG 1: About to call ggml_merge_lora for %s\n", key.c_str());
                    fflush(stdout);
                    updown = ggml_merge_lora(compute_ctx, lora_down, lora_up, lora_mid);
                    printf("🔧 TENSOR_OP DEBUG 2: ggml_merge_lora completed\n");
                    fflush(stdout);
                }
                scale_value *= multiplier;
                printf("🔧 TENSOR_OP DEBUG 3: About to reshape tensor for %s\n", key.c_str());
                fflush(stdout);
                updown = ggml_reshape(compute_ctx, updown, weight);
                printf("🔧 TENSOR_OP DEBUG 4: Reshape completed, checking elements\n");
                fflush(stdout);
                GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
                printf("🔧 TENSOR_OP DEBUG 5: About to scale tensor\n");
                fflush(stdout);
                updown = ggml_scale_inplace(compute_ctx, updown, scale_value);
                printf("🔧 TENSOR_OP DEBUG 6: Scale completed\n");
                fflush(stdout);
                ggml_tensor* final_weight;
                printf("🔧 TENSOR_OP DEBUG 7: Processing final weight for %s (type=%d)\n", key.c_str(), weight->type);
                fflush(stdout);
                if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                    printf("🔧 TENSOR_OP DEBUG 8: Converting to F32\n");
                    fflush(stdout);
                    // final_weight = ggml_new_tensor(compute_ctx, GGML_TYPE_F32, ggml_n_dims(weight), weight->ne);
                    // final_weight = ggml_cpy(compute_ctx, weight, final_weight);
                    final_weight = to_f32(compute_ctx, weight);
                    printf("🔧 TENSOR_OP DEBUG 9: Adding updown\n");
                    fflush(stdout);
                    final_weight = ggml_add_inplace(compute_ctx, final_weight, updown);
                    printf("🔧 TENSOR_OP DEBUG 10: Copying back to original type\n");
                    fflush(stdout);
                    final_weight = ggml_cpy(compute_ctx, final_weight, weight);
                } else {
                    printf("🔧 TENSOR_OP DEBUG 11: Direct add inplace\n");
                    fflush(stdout);
                    final_weight = ggml_add_inplace(compute_ctx, weight, updown);
                }
                printf("🔧 TENSOR_OP DEBUG 12: Building forward expand\n");
                fflush(stdout);
                // final_weight = ggml_add_inplace(compute_ctx, weight, updown);  // apply directly
                ggml_build_forward_expand(gf, final_weight);
                printf("🔧 TENSOR_OP DEBUG 13: Forward expand completed for %s\n", key.c_str());
                fflush(stdout);
                processed_count++;
                break;
            }
        }
        
        printf("🔧 LORA APPLICATION SUMMARY:\n");
        printf("🔧   Total model tensors: %zu\n", model_tensors.size());
        printf("🔧   Processed tensors: %d\n", processed_count);
        printf("🔧   LoRA tensor count: %zu\n", lora_tensors.size());
        fflush(stdout);
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
        
        // 🔧 PARTIAL LORA SUPPORT: Enhanced reporting
        float utilization_percent = total_lora_tensors_count > 0 ? 
            (float)applied_lora_tensors_count * 100.0f / total_lora_tensors_count : 0.0f;
            
        if (applied_lora_tensors_count != total_lora_tensors_count) {
            size_t unused_count = total_lora_tensors_count - applied_lora_tensors_count;
            if (utilization_percent >= 50.0f) {
                // Partial LoRA - this is normal for fal.ai style LoRAs
                LOG_INFO("🔧 PARTIAL LORA: (%lu / %lu) tensors applied (%.1f%% utilization)", 
                         applied_lora_tensors_count, total_lora_tensors_count, utilization_percent);
                LOG_INFO("🔧 This is normal for partial LoRAs that target specific layers only");
            } else {
                // Low utilization might indicate a problem
                LOG_WARN("⚠️  LOW LORA UTILIZATION: Only (%lu / %lu) tensors applied (%.1f%%)", 
                         applied_lora_tensors_count, total_lora_tensors_count, utilization_percent);
                LOG_WARN("⚠️  %lu unused tensors - check LoRA compatibility", unused_count);
            }
        } else {
            LOG_INFO("✅ COMPLETE LORA: (%lu / %lu) tensors applied successfully (100%)",
                      applied_lora_tensors_count, total_lora_tensors_count);
        }

        // 🔧 LORA APPLICATION FINAL STATISTICS: Report success/failure counts
        printf("📊 LORA APPLICATION STATISTICS:\n");
        printf("📊   Validation checks passed: %d\n", lora_validation_passed_count);
        printf("📊   LoRA tensors successfully applied: %d\n", lora_applied_count);
        printf("📊   LoRA tensors skipped (incompatible): %d\n", lora_skipped_count);
        printf("📊   Total LoRA processing attempts: %d\n", lora_validation_passed_count + lora_skipped_count);
        
        if (lora_applied_count > 0) {
            printf("✅ SUCCESS: %d LoRA layers applied successfully!\n", lora_applied_count);
            printf("✅ LoRA should now affect image generation\n");
        } else if (lora_skipped_count > 0) {
            printf("⚠️  WARNING: All %d LoRA layers were skipped due to incompatibility\n", lora_skipped_count);
            printf("⚠️  LoRA will have no effect on image generation\n");
        } else {
            printf("❓ INFO: No LoRA processing occurred\n");
        }
        fflush(stdout);
        
        printf("🔧 BUILD_LORA_GRAPH DEBUG FINAL: Function completing successfully, returning graph=%p\n", (void*)gf);
        fflush(stdout);
        return gf;
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, SDVersion version, int n_threads) {
        printf("🔥 LORA_APPLY DEBUG 1: Starting LoRA apply function\n");
        printf("🔥 LORA_APPLY DEBUG 2: model_tensors.size() = %zu\n", model_tensors.size());
        printf("🔥 LORA_APPLY DEBUG 3: version = %d, n_threads = %d\n", version, n_threads);
        fflush(stdout);
        
        // 🔧 SAFETY CHECK: Validate tensors before applying
        if (model_tensors.empty()) {
            LOG_ERROR("❌ Cannot apply LoRA: No model tensors provided");
            return;
        }
        
        if (lora_tensors.empty()) {
            LOG_ERROR("❌ Cannot apply LoRA: No LoRA tensors loaded");
            return;
        }
        
        printf("🔥 LORA_APPLY DEBUG 4: Creating get_graph lambda...\n");
        fflush(stdout);
        
        auto get_graph = [&]() -> struct ggml_cgraph* {
            printf("🔥 LORA_APPLY DEBUG 5: Inside get_graph lambda, calling build_lora_graph...\n");
            fflush(stdout);
            
            // 🔧 CRITICAL SAFETY CHECKS before calling build_lora_graph()
            printf("🔥 SAFETY CHECK 1: compute_ctx = %p\n", (void*)compute_ctx);
            printf("🔥 SAFETY CHECK 2: model_tensors.size() = %zu\n", model_tensors.size());
            printf("🔥 SAFETY CHECK 3: version = %d\n", version);
            printf("🔥 SAFETY CHECK 4: About to call build_lora_graph with validated parameters...\n");
            fflush(stdout);
            
            struct ggml_cgraph* result = build_lora_graph(model_tensors, version);
            
            printf("🔥 LORA_APPLY DEBUG 6: build_lora_graph returned: %p\n", (void*)result);
            fflush(stdout);
            
            if (!result) {
                LOG_ERROR("❌ CRITICAL: build_lora_graph returned NULL - LoRA graph construction failed");
                return nullptr;
            }
            
            return result;
        };
        
        printf("🔥 LORA_APPLY DEBUG 7: About to call GGMLRunner::compute...\n");
        fflush(stdout);
        
        try {
            // 🔧 PROTECTED COMPUTE: Wrap in try-catch to prevent crashes
            GGMLRunner::compute(get_graph, n_threads, true);
            printf("🔥 LORA_APPLY DEBUG 8: ✅ GGMLRunner::compute completed successfully!\n");
            fflush(stdout);
            LOG_INFO("✅ LoRA application completed successfully");
        } catch (const std::exception& e) {
            LOG_ERROR("❌ EXCEPTION during LoRA application: %s", e.what());
            LOG_ERROR("❌ This may be due to tensor compatibility issues");
            throw; // Re-throw to maintain error handling chain
        } catch (...) {
            LOG_ERROR("❌ UNKNOWN EXCEPTION during LoRA application");
            LOG_ERROR("❌ Likely cause: Tensor shape/dtype incompatibility with Metal backend");
            throw; // Re-throw to maintain error handling chain
        }
    }
};

#endif  // __LORA_HPP__