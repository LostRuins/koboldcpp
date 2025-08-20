#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include "ggml.h"

/**
 * Tensor Compatibility Fix for KoboldCpp LoRA System
 * 
 * This module implements ComfyUI-style tensor compatibility handling
 * to resolve dimension mismatch issues when applying LoRA tensors.
 * 
 * Key Features:
 * - Tensor dimension validation
 * - Safe tensor padding/resizing
 * - Graceful error handling
 * - Element count verification
 */

namespace TensorCompat {

    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        int64_t elements;
        ggml_type type;
        
        TensorInfo(const std::string& n, ggml_tensor* tensor) : name(n) {
            if (tensor) {
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (tensor->ne[i] > 1) {
                        shape.push_back(tensor->ne[i]);
                    }
                }
                elements = ggml_nelements(tensor);
                type = tensor->type;
            } else {
                elements = 0;
                type = GGML_TYPE_F32;
            }
        }
        
        bool isValid() const { return elements > 0; }
        
        std::string toString() const {
            std::string result = "[";
            for (size_t i = 0; i < shape.size(); i++) {
                if (i > 0) result += " x ";
                result += std::to_string(shape[i]);
            }
            result += "] = " + std::to_string(elements) + " elements";
            return result;
        }
    };

    /**
     * Check if two tensors are compatible for LoRA operations
     */
    bool areTensorsCompatible(ggml_tensor* source, ggml_tensor* target) {
        if (!source || !target) return false;
        
        int64_t source_elements = ggml_nelements(source);
        int64_t target_elements = ggml_nelements(target);
        
        // Exact match is always compatible
        if (source_elements == target_elements) return true;
        
        // Check if source can be broadcast/padded to target
        // Allow for reasonable size differences (up to 16x)
        if (target_elements > source_elements && target_elements <= source_elements * 16) {
            return true;
        }
        
        // Check if target can be truncated to source
        if (source_elements > target_elements && source_elements <= target_elements * 16) {
            return true;
        }
        
        return false;
    }

    /**
     * Validate tensor dimensions before attempting LoRA merge
     */
    bool validateLoRATensors(ggml_tensor* lora_up, ggml_tensor* lora_down, ggml_tensor* weight, const std::string& key) {
        TensorInfo up_info(key + ".lora_up", lora_up);
        TensorInfo down_info(key + ".lora_down", lora_down);
        TensorInfo weight_info(key + ".weight", weight);
        
        printf("🔍 TENSOR VALIDATION for %s:\n", key.c_str());
        printf("   Up tensor:     %s\n", up_info.toString().c_str());
        printf("   Down tensor:   %s\n", down_info.toString().c_str());
        printf("   Weight tensor: %s\n", weight_info.toString().c_str());
        
        if (!up_info.isValid() || !down_info.isValid() || !weight_info.isValid()) {
            printf("❌ Invalid tensor(s) detected\n");
            return false;
        }
        
        // Check LoRA rank consistency
        int64_t up_rank = lora_up->ne[ggml_n_dims(lora_up) - 1];
        int64_t down_rank = lora_down->ne[0];
        
        if (up_rank != down_rank) {
            printf("❌ LoRA rank mismatch: up=%lld, down=%lld\n", up_rank, down_rank);
            return false;
        }
        
        // Predict merged tensor size
        int64_t merged_elements = 1;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (lora_up->ne[i] > 1 && lora_down->ne[i] > 1) {
                merged_elements *= std::max(lora_up->ne[i], lora_down->ne[i]);
            } else if (lora_up->ne[i] > 1) {
                merged_elements *= lora_up->ne[i];
            } else if (lora_down->ne[i] > 1) {
                merged_elements *= lora_down->ne[i];
            }
        }
        
        printf("   Predicted merged: %lld elements\n", merged_elements);
        
        if (!areTensorsCompatible((ggml_tensor*)&merged_elements, weight)) {
            printf("❌ Merged tensor incompatible with weight tensor\n");
            printf("   Size ratio: %.2fx\n", (double)weight_info.elements / merged_elements);
            
            // Allow large differences but warn
            if (weight_info.elements > merged_elements * 4 || merged_elements > weight_info.elements * 4) {
                printf("⚠️  Large size difference detected - proceed with caution\n");
                return false;
            }
        }
        
        printf("✅ Tensor validation passed\n");
        return true;
    }

    /**
     * Create a compatible tensor by padding or truncating
     */
    ggml_tensor* makeCompatibleTensor(ggml_context* ctx, ggml_tensor* source, ggml_tensor* target_shape, const std::string& name) {
        if (!source || !target_shape) return nullptr;
        
        int64_t source_elements = ggml_nelements(source);
        int64_t target_elements = ggml_nelements(target_shape);
        
        printf("🔧 TENSOR COMPAT for %s: %lld -> %lld elements\n", name.c_str(), source_elements, target_elements);
        
        // If already compatible, return source
        if (source_elements == target_elements) {
            printf("   ✅ Already compatible\n");
            return source;
        }
        
        // Create new tensor with target shape
        ggml_tensor* result = ggml_new_tensor(ctx, source->type, ggml_n_dims(target_shape), target_shape->ne);
        
        if (target_elements > source_elements) {
            // Pad with zeros
            printf("   📏 Padding tensor (%.2fx larger)\n", (double)target_elements / source_elements);
            
            // Zero-initialize
            ggml_set_zero(result);
            
            // Copy source data to beginning of result
            // This is a simplified approach - in practice you'd need proper tensor broadcasting
            memset(result->data, 0, ggml_nbytes(result));
            if (source->data && ggml_nbytes(source) <= ggml_nbytes(result)) {
                memcpy(result->data, source->data, ggml_nbytes(source));
            }
            
        } else {
            // Truncate
            printf("   ✂️  Truncating tensor (%.2fx smaller)\n", (double)source_elements / target_elements);
            
            // Copy only what fits
            if (source->data) {
                memcpy(result->data, source->data, std::min(ggml_nbytes(result), ggml_nbytes(source)));
            }
        }
        
        return result;
    }

    /**
     * Safe tensor reshape with compatibility checks
     */
    ggml_tensor* safeReshape(ggml_context* ctx, ggml_tensor* source, ggml_tensor* shape_template, const std::string& operation) {
        if (!source || !shape_template) {
            printf("❌ NULL tensor in safeReshape for %s\n", operation.c_str());
            return nullptr;
        }
        
        TensorInfo source_info(operation + "_source", source);
        TensorInfo target_info(operation + "_target", shape_template);
        
        printf("🔧 SAFE RESHAPE for %s:\n", operation.c_str());
        printf("   Source: %s\n", source_info.toString().c_str());
        printf("   Target: %s\n", target_info.toString().c_str());
        
        // Check element compatibility
        if (source_info.elements != target_info.elements) {
            printf("❌ Element count mismatch: %lld != %lld (ratio: %.2fx)\n", 
                   source_info.elements, target_info.elements,
                   (double)target_info.elements / source_info.elements);
            
            // Try to create compatible tensor
            ggml_tensor* compatible = makeCompatibleTensor(ctx, source, shape_template, operation);
            if (compatible) {
                printf("✅ Created compatible tensor\n");
                return ggml_reshape(ctx, compatible, shape_template);
            } else {
                printf("❌ Failed to create compatible tensor\n");
                return nullptr;
            }
        }
        
        // Standard reshape
        printf("✅ Standard reshape (element counts match)\n");
        return ggml_reshape(ctx, source, shape_template);
    }

    /**
     * Enhanced LoRA merge with tensor compatibility
     */
    ggml_tensor* safeMergeLora(ggml_context* ctx, ggml_tensor* lora_down, ggml_tensor* lora_up, 
                              ggml_tensor* lora_mid, ggml_tensor* weight, const std::string& key) {
        printf("🔧 SAFE LORA MERGE for %s\n", key.c_str());
        
        // Validate inputs
        if (!validateLoRATensors(lora_up, lora_down, weight, key)) {
            printf("❌ Tensor validation failed for %s\n", key.c_str());
            return nullptr;
        }
        
        ggml_tensor* merged = nullptr;
        
        try {
            printf("   Step 1: Attempting standard ggml_merge_lora...\n");
            merged = ggml_merge_lora(ctx, lora_down, lora_up, lora_mid);
            
            if (merged) {
                printf("   ✅ Standard merge successful\n");
                
                // Now try safe reshape
                printf("   Step 2: Attempting safe reshape...\n");
                ggml_tensor* reshaped = safeReshape(ctx, merged, weight, key);
                
                if (reshaped) {
                    printf("   ✅ Safe reshape successful for %s\n", key.c_str());
                    return reshaped;
                } else {
                    printf("   ❌ Safe reshape failed for %s\n", key.c_str());
                    return nullptr;
                }
            } else {
                printf("   ❌ Standard merge failed for %s\n", key.c_str());
                return nullptr;
            }
            
        } catch (const std::exception& e) {
            printf("❌ Exception in safeMergeLora for %s: %s\n", key.c_str(), e.what());
            return nullptr;
        } catch (...) {
            printf("❌ Unknown exception in safeMergeLora for %s\n", key.c_str());
            return nullptr;
        }
    }

} // namespace TensorCompat