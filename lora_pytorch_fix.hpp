#ifndef __LORA_PYTORCH_FIX_HPP__
#define __LORA_PYTORCH_FIX_HPP__

#include <string>
#include <vector>
#include <map>
#include <set>

/*
 * PyTorch-Style LoRA Name Resolution for KoboldCpp
 * 
 * This implements the flexible tensor discovery system from ComfyUI/PyTorch
 * to fix LoRA loading issues in KoboldCpp.
 */

class PyTorchLoRAResolver {
public:
    enum LoRAArchitecture {
        FLUX,
        SDXL,
        SD15,
        UNKNOWN
    };

    struct LoRANamingVariant {
        std::string up_suffix;
        std::string down_suffix;
        std::string prefix;
        std::string description;
    };

    // PyTorch-style naming conventions (from ComfyUI lora.py)
    static const std::vector<LoRANamingVariant> NAMING_VARIANTS;

    /**
     * Detect LoRA architecture from tensor names
     */
    static LoRAArchitecture detect_architecture(const std::map<std::string, ggml_tensor*>& lora_tensors) {
        for (const auto& [name, tensor] : lora_tensors) {
            // Flux detection
            if (name.find("transformer.single_transformer_blocks.") != std::string::npos ||
                name.find("single_blocks.") != std::string::npos) {
                return FLUX;
            }
            // SDXL detection
            if (name.find("lora_unet_") != std::string::npos ||
                name.find("diffusion_model.") != std::string::npos) {
                return SDXL;
            }
        }
        return UNKNOWN;
    }

    /**
     * Generate all possible tensor name variants for a base key
     */
    static std::vector<std::string> generate_tensor_name_variants(
        const std::string& base_key, 
        bool is_up_tensor,
        LoRAArchitecture arch = UNKNOWN
    ) {
        std::vector<std::string> variants;
        
        for (const auto& variant : NAMING_VARIANTS) {
            std::string suffix = is_up_tensor ? variant.up_suffix : variant.down_suffix;
            
            // Try with prefix
            if (!variant.prefix.empty()) {
                variants.push_back(variant.prefix + base_key + suffix + ".weight");
            }
            
            // Try without prefix
            variants.push_back(base_key + suffix + ".weight");
            
            // Try without .weight suffix (for some formats)
            variants.push_back(base_key + suffix);
        }

        // Add architecture-specific variants
        if (arch == FLUX) {
            add_flux_variants(variants, base_key, is_up_tensor);
        } else if (arch == SDXL) {
            add_sdxl_variants(variants, base_key, is_up_tensor);
        }

        return variants;
    }

    /**
     * Find matching tensor name from variants
     */
    static std::string find_matching_tensor(
        const std::vector<std::string>& variants,
        const std::map<std::string, ggml_tensor*>& lora_tensors
    ) {
        for (const std::string& variant : variants) {
            if (lora_tensors.find(variant) != lora_tensors.end()) {
                return variant;
            }
        }
        return "";
    }

    /**
     * Map model keys to LoRA keys (PyTorch-style)
     */
    static std::map<std::string, std::string> build_key_mapping(
        const std::map<std::string, ggml_tensor*>& model_tensors,
        const std::map<std::string, ggml_tensor*>& lora_tensors
    ) {
        std::map<std::string, std::string> key_map;
        LoRAArchitecture arch = detect_architecture(lora_tensors);
        
        for (const auto& [model_key, tensor] : model_tensors) {
            if (!model_key.ends_with(".weight")) continue;
            
            std::string base_key = model_key.substr(0, model_key.length() - 7); // Remove ".weight"
            
            // Generate possible LoRA key variants
            std::vector<std::string> lora_key_variants = generate_lora_key_variants(base_key, arch);
            
            // Find matching LoRA key
            for (const std::string& lora_key : lora_key_variants) {
                auto up_variants = generate_tensor_name_variants(lora_key, true, arch);
                auto down_variants = generate_tensor_name_variants(lora_key, false, arch);
                
                std::string up_match = find_matching_tensor(up_variants, lora_tensors);
                std::string down_match = find_matching_tensor(down_variants, lora_tensors);
                
                if (!up_match.empty() && !down_match.empty()) {
                    key_map[model_key] = lora_key;
                    break;
                }
            }
        }
        
        return key_map;
    }

    /**
     * Load LoRA tensors with PyTorch-style flexibility
     */
    static bool load_lora_tensor_pair(
        const std::string& base_key,
        const std::map<std::string, ggml_tensor*>& lora_tensors,
        ggml_context* compute_ctx,
        ggml_tensor*& lora_up,
        ggml_tensor*& lora_down,
        LoRAArchitecture arch = UNKNOWN
    ) {
        // Try to detect architecture if not provided
        if (arch == UNKNOWN) {
            arch = detect_architecture(lora_tensors);
        }

        auto up_variants = generate_tensor_name_variants(base_key, true, arch);
        auto down_variants = generate_tensor_name_variants(base_key, false, arch);

        std::string up_name = find_matching_tensor(up_variants, lora_tensors);
        std::string down_name = find_matching_tensor(down_variants, lora_tensors);

        if (!up_name.empty() && !down_name.empty()) {
            printf("🔧 PYTORCH RESOLVER: ✅ Found tensor pair for %s\n", base_key.c_str());
            printf("🔧 PYTORCH RESOLVER:   up: %s\n", up_name.c_str());
            printf("🔧 PYTORCH RESOLVER:   down: %s\n", down_name.c_str());
            
            lora_up = to_f32(compute_ctx, lora_tensors.at(up_name));
            lora_down = to_f32(compute_ctx, lora_tensors.at(down_name));
            return true;
        }

        printf("🔧 PYTORCH RESOLVER: ❌ No tensor pair found for %s\n", base_key.c_str());
        return false;
    }

private:
    static void add_flux_variants(std::vector<std::string>& variants, const std::string& base_key, bool is_up) {
        std::string suffix = is_up ? ".lora_B" : ".lora_A";
        
        // Flux-specific patterns
        variants.push_back("transformer." + base_key + suffix + ".weight");
        variants.push_back("lora_transformer_" + replace_dots_with_underscores(base_key) + suffix.substr(1));
        variants.push_back("lycoris_" + replace_dots_with_underscores(base_key) + suffix.substr(1));
        
        // Direct Flux patterns (no transformer prefix)
        if (base_key.find("single_transformer_blocks.") != std::string::npos) {
            variants.push_back(base_key + suffix + ".weight");
        }
    }

    static void add_sdxl_variants(std::vector<std::string>& variants, const std::string& base_key, bool is_up) {
        std::string suffix = is_up ? ".lora_B" : ".lora_A";
        
        // SDXL-specific patterns
        variants.push_back("lora_unet_" + replace_dots_with_underscores(base_key) + suffix.substr(1) + ".weight");
        variants.push_back("diffusion_model." + base_key + suffix + ".weight");
    }

    static std::vector<std::string> generate_lora_key_variants(const std::string& model_key, LoRAArchitecture arch) {
        std::vector<std::string> variants;
        
        // Direct mapping
        variants.push_back(model_key);
        
        if (arch == FLUX) {
            // Model key: diffusion_model.single_blocks.0.linear1
            // LoRA key: transformer.single_transformer_blocks.0.linear1
            if (model_key.find("diffusion_model.single_blocks.") == 0) {
                std::string flux_key = model_key;
                flux_key = replace_first(flux_key, "diffusion_model.single_blocks.", "transformer.single_transformer_blocks.");
                variants.push_back(flux_key);
                
                // Also try without transformer prefix
                std::string direct_key = replace_first(model_key, "diffusion_model.", "");
                direct_key = replace_first(direct_key, "single_blocks.", "single_transformer_blocks.");
                variants.push_back(direct_key);
            }
        }
        
        return variants;
    }

    static std::string replace_dots_with_underscores(const std::string& str) {
        std::string result = str;
        std::replace(result.begin(), result.end(), '.', '_');
        return result;
    }

    static std::string replace_first(const std::string& str, const std::string& from, const std::string& to) {
        std::string result = str;
        size_t start_pos = result.find(from);
        if (start_pos != std::string::npos) {
            result.replace(start_pos, from.length(), to);
        }
        return result;
    }
};

// Define the naming variants (from ComfyUI)
const std::vector<PyTorchLoRAResolver::LoRANamingVariant> PyTorchLoRAResolver::NAMING_VARIANTS = {
    {".lora_up", ".lora_down", "lora.", "Standard LoRA"},
    {"_lora.up", "_lora.down", "", "Diffusers LoRA"},
    {".lora_B", ".lora_A", "", "Alternative LoRA"},
    {".lora.up", ".lora.down", "", "Diffusers v3 LoRA"},
    {".lora_B", ".lora_A", "", "Mochi LoRA"},
    {".lora_linear_layer.up", ".lora_linear_layer.down", "", "Transformers LoRA"}
};

#endif // __LORA_PYTORCH_FIX_HPP__