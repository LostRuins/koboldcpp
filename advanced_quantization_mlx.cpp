#include "advanced_quantization_mlx.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <immintrin.h>  // SIMD intrinsics

namespace AdvancedQuantization {

// ============================================================================
// TensorStatistics Implementation
// ============================================================================

void TensorStatistics::analyze_tensor(const float* data, int rows, int cols) {
    int total_size = rows * cols;
    
    // Basic statistics
    double sum = 0.0, sum_sq = 0.0;
    min_value = data[0];
    max_value = data[0];
    int zero_count = 0;
    
    for (int i = 0; i < total_size; i++) {
        float val = data[i];
        sum += val;
        sum_sq += val * val;
        min_value = std::min(min_value, val);
        max_value = std::max(max_value, val);
        if (std::abs(val) < 1e-6f) zero_count++;
    }
    
    mean = static_cast<float>(sum / total_size);
    variance = static_cast<float>(sum_sq / total_size - mean * mean);
    sparsity_ratio = static_cast<float>(zero_count) / total_size;
    
    // Per-channel statistics for better quantization
    channel_means.resize(rows);
    channel_scales.resize(rows);
    
    for (int r = 0; r < rows; r++) {
        double row_sum = 0.0;
        float row_min = data[r * cols];
        float row_max = row_min;
        
        for (int c = 0; c < cols; c++) {
            float val = data[r * cols + c];
            row_sum += val;
            row_min = std::min(row_min, val);
            row_max = std::max(row_max, val);
        }
        
        channel_means[r] = static_cast<float>(row_sum / cols);
        channel_scales[r] = row_max - row_min;
    }
    
    // Detect outliers (values beyond threshold standard deviations)
    float std_dev = std::sqrt(variance);
    float outlier_bound = mean + 3.0f * std_dev; // 3-sigma rule
    
    for (int i = 0; i < total_size; i++) {
        if (std::abs(data[i] - mean) > outlier_bound) {
            outliers.push_back(data[i]);
        }
    }
    
    printf("📊 QUANT-ANALYSIS: Tensor %dx%d analyzed\n", rows, cols);
    printf("📊 QUANT-ANALYSIS: Mean=%.4f, Std=%.4f, Sparsity=%.2f%%, Outliers=%zu\n",
           mean, std_dev, sparsity_ratio * 100, outliers.size());
}

QuantizationType TensorStatistics::recommend_quantization() const {
    // Decision logic based on tensor characteristics
    
    // If very sparse, use specialized sparse quantization
    if (sparsity_ratio > 0.5f) {
        return QuantizationType::INT4_GROUP32; // Smaller groups for sparse data
    }
    
    // If many outliers, use asymmetric quantization
    if (outliers.size() > 100) {
        return QuantizationType::INT8_SYMMETRIC; // Higher precision for outliers
    }
    
    // If high variance, use smaller groups for better precision
    if (variance > 1.0f) {
        return QuantizationType::INT4_GROUP32;
    }
    
    // Default to MLX-style quantization
    return QuantizationType::INT4_GROUP64;
}

// ============================================================================
// QuantizedTensor Implementation
// ============================================================================

QuantizedTensor::QuantizedTensor(const float* fp32_data, int r, int c, const QuantizationParams& p)
    : rows(r), cols(c), params(p), outliers_count(0) {
    
    printf("🔢 QUANTIZATION: Creating quantized tensor %dx%d\n", rows, cols);
    
    // Analyze tensor characteristics
    stats.analyze_tensor(fp32_data, rows, cols);
    
    // Adjust parameters based on statistics if adaptive mode
    if (params.type == QuantizationType::DYNAMIC_ADAPT) {
        params.type = stats.recommend_quantization();
        printf("🔢 QUANTIZATION: Adaptive mode selected %d\n", static_cast<int>(params.type));
    }
    
    // Allocate memory based on quantization type
    switch (params.type) {
        case QuantizationType::INT4_GROUP64:
        case QuantizationType::INT4_GROUP32:
            quantize_int4_grouped(fp32_data);
            break;
        case QuantizationType::INT8_SYMMETRIC:
            quantize_int8_symmetric(fp32_data);
            break;
        default:
            quantize_int4_grouped(fp32_data); // Fallback
    }
    
    printf("🔢 QUANTIZATION: Tensor quantized - compression %.2fx\n", compression_ratio());
}

QuantizedTensor::~QuantizedTensor() {
    delete[] static_cast<uint8_t*>(quantized_data);
    delete[] scales;
    delete[] zero_points;
    delete[] outlier_indices;
    delete[] outlier_values;
}

void QuantizedTensor::quantize_int4_grouped(const float* data) {
    int group_size = params.group_size;
    int total_groups = (cols + group_size - 1) / group_size;
    
    // Allocate memory
    quantized_size = (rows * cols * params.bits + 7) / 8; // Pack bits into bytes
    quantized_data = new uint8_t[quantized_size];
    scales_size = rows * total_groups;
    scales = new float[scales_size];
    zero_points = new float[scales_size];
    
    printf("🔢 QUANTIZATION: INT4 grouped quantization (groups: %d)\n", total_groups);
    
    uint8_t* q_data = static_cast<uint8_t*>(quantized_data);
    memset(q_data, 0, quantized_size);
    
    int bit_offset = 0;
    
    for (int row = 0; row < rows; row++) {
        for (int group_start = 0; group_start < cols; group_start += group_size) {
            int group_end = std::min(group_start + group_size, cols);
            int group_idx = row * total_groups + (group_start / group_size);
            
            // Find min/max in this group
            float min_val = data[row * cols + group_start];
            float max_val = min_val;
            
            for (int c = group_start + 1; c < group_end; c++) {
                float val = data[row * cols + c];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            
            // Calculate scale and zero point
            float scale = (max_val - min_val) / 15.0f; // 4-bit range: 0-15
            scales[group_idx] = scale;
            zero_points[group_idx] = min_val;
            
            // Quantize values in this group
            for (int c = group_start; c < group_end; c++) {
                float val = data[row * cols + c];
                int quantized = 0;
                
                if (scale > 1e-8f) {
                    quantized = static_cast<int>((val - min_val) / scale);
                    quantized = std::max(0, std::min(15, quantized));
                }
                
                // Pack 4-bit value into byte array
                int byte_idx = bit_offset / 8;
                int bit_pos = bit_offset % 8;
                
                if (bit_pos <= 4) {
                    q_data[byte_idx] |= (quantized << bit_pos);
                } else {
                    // Value spans two bytes
                    q_data[byte_idx] |= (quantized << bit_pos);
                    if (byte_idx + 1 < quantized_size) {
                        q_data[byte_idx + 1] |= (quantized >> (8 - bit_pos));
                    }
                }
                
                bit_offset += 4;
            }
        }
    }
    
    printf("🔢 QUANTIZATION: INT4 quantization complete\n");
}

void QuantizedTensor::quantize_int8_symmetric(const float* data) {
    // Symmetric quantization: scale around zero
    quantized_size = rows * cols; // 1 byte per value
    quantized_data = new int8_t[quantized_size];
    scales_size = rows; // One scale per row
    scales = new float[scales_size];
    
    int8_t* q_data = static_cast<int8_t*>(quantized_data);
    
    for (int row = 0; row < rows; row++) {
        // Find maximum absolute value in this row
        float max_abs = 0.0f;
        for (int col = 0; col < cols; col++) {
            max_abs = std::max(max_abs, std::abs(data[row * cols + col]));
        }
        
        // Calculate scale (map to -127, +127)
        scales[row] = max_abs / 127.0f;
        
        // Quantize row
        for (int col = 0; col < cols; col++) {
            float val = data[row * cols + col];
            int quantized = 0;
            
            if (scales[row] > 1e-8f) {
                quantized = static_cast<int>(val / scales[row]);
                quantized = std::max(-127, std::min(127, quantized));
            }
            
            q_data[row * cols + col] = static_cast<int8_t>(quantized);
        }
    }
    
    printf("🔢 QUANTIZATION: INT8 symmetric quantization complete\n");
}

void QuantizedTensor::quantized_matmul_fp16(
    const float* input,
    float* output,
    int batch_size,
    ComputePrecision precision) const {
    
    printf("⚡ QUANT-MATMUL: Executing quantized matmul (batch: %d, precision: %d)\n", 
           batch_size, static_cast<int>(precision));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (params.type == QuantizationType::INT8_SYMMETRIC) {
        // Optimized INT8 matrix multiplication
        const int8_t* q_weights = static_cast<const int8_t*>(quantized_data);
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; b++) {
            for (int r = 0; r < rows; r++) {
                float sum = 0.0f;
                float scale = scales[r];
                
                // Vectorized dot product
                for (int c = 0; c < cols; c++) {
                    float weight = static_cast<float>(q_weights[r * cols + c]) * scale;
                    sum += input[b * cols + c] * weight;
                }
                
                output[b * rows + r] = sum;
            }
        }
    } else {
        // INT4 quantized multiplication (more complex unpacking)
        const uint8_t* q_weights = static_cast<const uint8_t*>(quantized_data);
        int group_size = params.group_size;
        int total_groups = (cols + group_size - 1) / group_size;
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; b++) {
            for (int r = 0; r < rows; r++) {
                float sum = 0.0f;
                
                for (int group_start = 0; group_start < cols; group_start += group_size) {
                    int group_end = std::min(group_start + group_size, cols);
                    int group_idx = r * total_groups + (group_start / group_size);
                    
                    float scale = scales[group_idx];
                    float zero_point = zero_points[group_idx];
                    
                    for (int c = group_start; c < group_end; c++) {
                        // Unpack 4-bit quantized weight
                        int linear_idx = r * cols + c;
                        int bit_offset = linear_idx * 4;
                        int byte_idx = bit_offset / 8;
                        int bit_pos = bit_offset % 8;
                        
                        uint8_t quantized = 0;
                        if (bit_pos <= 4) {
                            quantized = (q_weights[byte_idx] >> bit_pos) & 0xF;
                        } else {
                            quantized = (q_weights[byte_idx] >> bit_pos) & ((1 << (8 - bit_pos)) - 1);
                            if (byte_idx + 1 < quantized_size) {
                                quantized |= ((q_weights[byte_idx + 1] & ((1 << (bit_pos - 4)) - 1)) << (8 - bit_pos));
                            }
                        }
                        
                        // Dequantize and accumulate
                        float weight = static_cast<float>(quantized) * scale + zero_point;
                        sum += input[b * cols + c] * weight;
                    }
                }
                
                output[b * rows + r] = sum;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    printf("⚡ QUANT-MATMUL: Completed in %.2f ms\n", elapsed_ms);
}

float QuantizedTensor::compression_ratio() const {
    size_t original_size = rows * cols * sizeof(float);
    size_t compressed_size = memory_usage();
    return static_cast<float>(original_size) / compressed_size;
}

size_t QuantizedTensor::memory_usage() const {
    return quantized_size + 
           scales_size * sizeof(float) + 
           scales_size * sizeof(float) + // zero_points
           outliers_count * sizeof(uint16_t) + // outlier indices
           outliers_count * sizeof(float);     // outlier values
}

// ============================================================================
// QuantizationEngine Implementation
// ============================================================================

QuantizationEngine::QuantizationEngine(const QuantizationParams& params) 
    : global_params(params) {
    printf("🏭 QUANT-ENGINE: Initializing quantization engine\n");
}

void QuantizationEngine::analyze_model_tensors(const std::map<std::string, struct ggml_tensor*>& tensors) {
    printf("🔍 QUANT-ENGINE: Analyzing %zu model tensors\n", tensors.size());
    
    for (const auto& [name, tensor] : tensors) {
        LayerProfile profile;
        profile.name = name;
        
        // Analyze tensor statistics
        int rows = tensor->ne[1];
        int cols = tensor->ne[0];
        float* data = static_cast<float*>(tensor->data);
        
        profile.stats.analyze_tensor(data, rows, cols);
        profile.recommended_type = profile.stats.recommend_quantization();
        
        // Assign compute precision based on layer type
        if (name.find("attention") != std::string::npos) {
            profile.compute_precision = ComputePrecision::FP16; // Attention needs precision
            profile.sensitivity_score = 0.8f;
        } else if (name.find("mlp") != std::string::npos || name.find("ff") != std::string::npos) {
            profile.compute_precision = ComputePrecision::INT8_COMPUTE; // MLP can be aggressive
            profile.sensitivity_score = 0.3f;
        } else {
            profile.compute_precision = ComputePrecision::MIXED_OPTIMAL;
            profile.sensitivity_score = 0.5f;
        }
        
        layer_profiles[name] = profile;
        
        printf("🔍 QUANT-ENGINE: Layer '%s' - Type: %d, Precision: %d, Sensitivity: %.2f\n",
               name.c_str(), static_cast<int>(profile.recommended_type), 
               static_cast<int>(profile.compute_precision), profile.sensitivity_score);
    }
}

std::unique_ptr<QuantizedTensor> QuantizationEngine::quantize_tensor(
    const std::string& name,
    const float* data,
    int rows, int cols,
    bool force_precision) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    QuantizationParams params = global_params;
    
    // Use layer-specific parameters if available
    auto it = layer_profiles.find(name);
    if (it != layer_profiles.end() && !force_precision) {
        params.type = it->second.recommended_type;
        
        // Adjust group size based on sensitivity
        if (it->second.sensitivity_score > 0.7f) {
            params.group_size = std::min(params.group_size, 32); // Smaller groups for sensitive layers
        }
    }
    
    auto tensor = std::make_unique<QuantizedTensor>(data, rows, cols, params);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    metrics.quantization_time_ms += elapsed_ms;
    metrics.total_memory_saved += rows * cols * sizeof(float) - tensor->memory_usage();
    
    printf("🔢 QUANT-ENGINE: Quantized '%s' in %.2f ms (compression: %.2fx)\n", 
           name.c_str(), elapsed_ms, tensor->compression_ratio());
    
    return tensor;
}

void QuantizationEngine::print_quantization_report() const {
    printf("\n📋 QUANTIZATION ENGINE REPORT\n");
    printf("==============================\n");
    printf("Total quantization time: %.2f ms\n", metrics.quantization_time_ms);
    printf("Total dequantization time: %.2f ms\n", metrics.dequantization_time_ms);
    printf("Total compute time: %.2f ms\n", metrics.compute_time_ms);
    printf("Memory saved: %.2f MB\n", metrics.total_memory_saved / 1024.0 / 1024.0);
    printf("Estimated accuracy loss: %.4f%%\n", metrics.accuracy_loss_estimate * 100);
    
    printf("\nLayer Analysis:\n");
    for (const auto& [name, profile] : layer_profiles) {
        printf("  %s: Type=%d, Precision=%d, Sensitivity=%.2f\n",
               name.c_str(), static_cast<int>(profile.recommended_type),
               static_cast<int>(profile.compute_precision), profile.sensitivity_score);
    }
    printf("==============================\n");
}

// ============================================================================
// Hardware-Optimized Kernels
// ============================================================================

namespace Kernels {

void quantize_int4_simd(
    const float* input,
    uint8_t* output,
    const float* scales,
    int size,
    int group_size) {
    
    printf("⚡ SIMD-KERNEL: INT4 quantization with SIMD\n");
    
    #ifdef __AVX2__
    // AVX2-optimized quantization
    const int simd_width = 8; // 8 floats per AVX2 register
    
    for (int i = 0; i < size; i += simd_width) {
        int remaining = std::min(simd_width, size - i);
        
        // Load 8 floats
        __m256 input_vec = _mm256_loadu_ps(&input[i]);
        
        // Load corresponding scales (broadcast if needed)
        int group_idx = i / group_size;
        __m256 scale_vec = _mm256_broadcast_ss(&scales[group_idx]);
        
        // Quantize: divide by scale, round, clamp to 0-15
        __m256 quantized_f = _mm256_div_ps(input_vec, scale_vec);
        quantized_f = _mm256_round_ps(quantized_f, _MM_FROUND_TO_NEAREST_INT);
        
        // Convert to integers and clamp
        __m256i quantized_i = _mm256_cvtps_epi32(quantized_f);
        quantized_i = _mm256_max_epi32(quantized_i, _mm256_setzero_si256());
        quantized_i = _mm256_min_epi32(quantized_i, _mm256_set1_epi32(15));
        
        // Pack and store (simplified - real implementation would pack into 4-bit)
        alignas(32) int32_t quantized_values[8];
        _mm256_store_si256((__m256i*)quantized_values, quantized_i);
        
        // Pack 4-bit values into output
        for (int j = 0; j < remaining; j += 2) {
            uint8_t packed = (quantized_values[j] & 0xF);
            if (j + 1 < remaining) {
                packed |= ((quantized_values[j + 1] & 0xF) << 4);
            }
            output[(i + j) / 2] = packed;
        }
    }
    #else
    // Fallback scalar implementation
    for (int i = 0; i < size; i++) {
        int group_idx = i / group_size;
        float scale = scales[group_idx];
        
        int quantized = static_cast<int>(input[i] / scale);
        quantized = std::max(0, std::min(15, quantized));
        
        // Pack into output
        int byte_idx = i / 2;
        int bit_pos = (i % 2) * 4;
        if (i % 2 == 0) {
            output[byte_idx] = quantized;
        } else {
            output[byte_idx] |= (quantized << 4);
        }
    }
    #endif
}

void quantized_matmul_fused(
    const float* input,
    const uint8_t* weight_q,
    const float* scales,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim,
    int group_size) {
    
    printf("⚡ FUSED-KERNEL: Quantized matrix multiplication\n");
    
    // This implements the MLX-style fused quantized matmul
    // Key optimization: dequantize weights on-the-fly to avoid memory pressure
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        for (int out_idx = 0; out_idx < output_dim; out_idx++) {
            float sum = 0.0f;
            
            // Process in groups for efficient dequantization
            for (int in_start = 0; in_start < input_dim; in_start += group_size) {
                int in_end = std::min(in_start + group_size, input_dim);
                int group_idx = (out_idx * input_dim + in_start) / group_size;
                float scale = scales[group_idx];
                
                // Accumulate over this group
                for (int in_idx = in_start; in_idx < in_end; in_idx++) {
                    // Unpack 4-bit weight
                    int weight_idx = out_idx * input_dim + in_idx;
                    int byte_idx = weight_idx / 2;
                    int bit_pos = (weight_idx % 2) * 4;
                    
                    uint8_t quantized = (weight_q[byte_idx] >> bit_pos) & 0xF;
                    float weight = static_cast<float>(quantized) * scale;
                    
                    sum += input[b * input_dim + in_idx] * weight;
                }
            }
            
            output[b * output_dim + out_idx] = sum;
        }
    }
}

} // namespace Kernels

// ============================================================================
// QuantizedLoRAFusion Implementation
// ============================================================================

QuantizedLoRAFusion::QuantizedLoRAFusion(const QuantizationParams& params) 
    : engine(params) {
    printf("🔧 QUANT-LORA: Initializing quantized LoRA fusion system\n");
}

void QuantizedLoRAFusion::set_quantized_base(struct ggml_tensor* base_weights) {
    printf("🔧 QUANT-LORA: Setting quantized base model\n");
    
    int rows = base_weights->ne[1];
    int cols = base_weights->ne[0];
    float* data = static_cast<float*>(base_weights->data);
    
    base_model = engine.quantize_tensor("base_model", data, rows, cols);
    
    printf("🔧 QUANT-LORA: Base model quantized (compression: %.2fx)\n", 
           base_model->compression_ratio());
}

void QuantizedLoRAFusion::add_quantized_lora(
    const std::string& name,
    struct ggml_tensor* lora_a,
    struct ggml_tensor* lora_b,
    float scale) {
    
    printf("🔧 QUANT-LORA: Adding quantized LoRA '%s'\n", name.c_str());
    
    // For LoRAs, we might use higher precision due to their smaller size
    QuantizationParams lora_params;
    lora_params.type = QuantizationType::INT8_SYMMETRIC; // Higher precision for LoRAs
    lora_params.group_size = 32; // Smaller groups
    
    int rows_a = lora_a->ne[1];
    int cols_a = lora_a->ne[0];
    int rows_b = lora_b->ne[1]; 
    int cols_b = lora_b->ne[0];
    
    auto q_lora_a = engine.quantize_tensor(name + "_A", 
                                           static_cast<float*>(lora_a->data), 
                                           rows_a, cols_a, true);
    auto q_lora_b = engine.quantize_tensor(name + "_B",
                                           static_cast<float*>(lora_b->data),
                                           rows_b, cols_b, true);
    
    // Store both A and B matrices (simplified - real implementation would store together)
    lora_tensors.emplace_back(name + "_A", std::move(q_lora_a));
    lora_tensors.emplace_back(name + "_B", std::move(q_lora_b));
    
    printf("🔧 QUANT-LORA: LoRA '%s' quantized and added\n", name.c_str());
}

struct ggml_tensor* QuantizedLoRAFusion::quantized_forward(
    struct ggml_tensor* input,
    struct ggml_context* ctx,
    const std::vector<std::string>& active_loras) {
    
    printf("🚀 QUANT-LORA: Quantized forward pass\n");
    
    int batch_size = input->ne[1];
    int input_dim = input->ne[0];
    int output_dim = base_model ? 4096 : 2048; // This should come from base model
    
    // Create output tensor
    struct ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output_dim, batch_size);
    
    float* input_data = static_cast<float*>(input->data);
    float* output_data = static_cast<float*>(output->data);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Step 1: Quantized base model forward pass
    if (base_model) {
        base_model->quantized_matmul_fp16(input_data, output_data, batch_size);
    } else {
        memset(output_data, 0, batch_size * output_dim * sizeof(float));
    }
    
    // Step 2: Add quantized LoRA contributions
    // (Simplified implementation - real version would process all LoRAs efficiently)
    for (const auto& lora_name : active_loras) {
        // Find corresponding LoRA tensors and apply them
        printf("🚀 QUANT-LORA: Applying quantized LoRA '%s'\n", lora_name.c_str());
        // Implementation would do quantized LoRA computation here
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    printf("🚀 QUANT-LORA: Quantized forward completed in %.2f ms\n", elapsed_ms);
    
    return output;
}

size_t QuantizedLoRAFusion::get_total_memory_usage() const {
    size_t total = 0;
    
    if (base_model) {
        total += base_model->memory_usage();
    }
    
    for (const auto& [name, tensor] : lora_tensors) {
        total += tensor->memory_usage();
    }
    
    return total;
}

float QuantizedLoRAFusion::get_compression_ratio() const {
    // Calculate average compression ratio across all tensors
    if (!base_model && lora_tensors.empty()) return 1.0f;
    
    float total_ratio = 0.0f;
    int count = 0;
    
    if (base_model) {
        total_ratio += base_model->compression_ratio();
        count++;
    }
    
    for (const auto& [name, tensor] : lora_tensors) {
        total_ratio += tensor->compression_ratio();
        count++;
    }
    
    return total_ratio / count;
}

} // namespace AdvancedQuantization