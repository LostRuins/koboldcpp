#ifndef __ADVANCED_QUANTIZATION_MLX_HPP__
#define __ADVANCED_QUANTIZATION_MLX_HPP__

#include "ggml_extend.hpp"
#include <vector>
#include <memory>
#include <cstdint>

/**
 * Advanced Quantization System - MLX Inspired
 * 
 * Implements MLX-style quantization optimizations:
 * 1. 4-bit quantization with group-wise scaling
 * 2. Mixed precision computation (int4 storage, fp16 compute, fp32 accumulate)
 * 3. Hardware-optimized kernels for quantized operations
 * 4. Dynamic precision based on sensitivity analysis
 */

namespace AdvancedQuantization {

// ============================================================================
// Quantization Schemes
// ============================================================================

enum class QuantizationType {
    INT4_GROUP64,    // 4-bit with 64-element groups (MLX default)
    INT4_GROUP32,    // 4-bit with 32-element groups (higher precision)
    INT8_SYMMETRIC,  // 8-bit symmetric quantization  
    FP16_MIXED,      // Mixed FP16/FP32 precision
    DYNAMIC_ADAPT    // Dynamically adapt based on tensor statistics
};

enum class ComputePrecision {
    FP32,           // Full 32-bit precision
    FP16,           // Half precision
    BF16,           // Brain Float 16 (good for training)
    INT8_COMPUTE,   // Integer compute (fastest)
    MIXED_OPTIMAL   // Optimal mixed precision per operation
};

// ============================================================================
// Quantization Parameters and Statistics
// ============================================================================

struct QuantizationParams {
    QuantizationType type = QuantizationType::INT4_GROUP64;
    int group_size = 64;
    int bits = 4;
    bool symmetric = false;
    float zero_point = 0.0f;
    
    // MLX-style adaptive parameters
    bool enable_outlier_detection = true;
    float outlier_threshold = 3.0f; // Standard deviations
    int min_group_size = 16;
    int max_group_size = 256;
};

struct TensorStatistics {
    float mean = 0.0f;
    float variance = 0.0f;
    float min_value = 0.0f;
    float max_value = 0.0f;
    float sparsity_ratio = 0.0f;    // Percentage of zeros
    std::vector<float> outliers;
    
    // Per-channel statistics for better quantization
    std::vector<float> channel_means;
    std::vector<float> channel_scales;
    
    // Sensitivity analysis for dynamic precision
    float gradient_magnitude = 0.0f;
    float weight_importance = 1.0f;
    
    void analyze_tensor(const float* data, int rows, int cols);
    QuantizationType recommend_quantization() const;
};

// ============================================================================
// Quantized Tensor Storage
// ============================================================================

class QuantizedTensor {
private:
    void* quantized_data;           // Packed quantized weights
    float* scales;                  // Per-group scaling factors
    float* zero_points;             // Per-group zero points (for asymmetric)
    uint16_t* outlier_indices;      // Indices of outlier values
    float* outlier_values;          // Full precision outlier values
    
    QuantizationParams params;
    TensorStatistics stats;
    
    int rows, cols;
    size_t quantized_size;
    size_t scales_size;
    size_t outliers_count;

public:
    QuantizedTensor(const float* fp32_data, int r, int c, const QuantizationParams& p);
    ~QuantizedTensor();
    
    // Quantization process
    void quantize_int4_grouped(const float* data);
    void quantize_int8_symmetric(const float* data);
    void quantize_with_outliers(const float* data);
    
    // Dequantization
    void dequantize_to_buffer(float* output) const;
    void dequantize_row_range(int start_row, int end_row, float* output) const;
    
    // Hardware-optimized operations
    void quantized_matmul_fp16(
        const float* input,      // [batch, cols]
        float* output,           // [batch, rows]  
        int batch_size,
        ComputePrecision precision = ComputePrecision::MIXED_OPTIMAL
    ) const;
    
    // Memory and performance info
    size_t memory_usage() const;
    float compression_ratio() const;
    
    const QuantizationParams& get_params() const { return params; }
    const TensorStatistics& get_stats() const { return stats; }
};

// ============================================================================
// Advanced Quantization Engine
// ============================================================================

class QuantizationEngine {
private:
    struct LayerProfile {
        std::string name;
        TensorStatistics stats;
        QuantizationType recommended_type;
        ComputePrecision compute_precision;
        float sensitivity_score;
        size_t access_frequency;
    };
    
    std::unordered_map<std::string, LayerProfile> layer_profiles;
    QuantizationParams global_params;
    
    // Performance tracking
    struct PerformanceMetrics {
        double quantization_time_ms = 0.0;
        double dequantization_time_ms = 0.0;
        double compute_time_ms = 0.0;
        size_t total_memory_saved = 0;
        double accuracy_loss_estimate = 0.0;
    } metrics;

public:
    QuantizationEngine(const QuantizationParams& params = QuantizationParams{});
    
    // Analysis and profiling
    void analyze_model_tensors(const std::map<std::string, struct ggml_tensor*>& tensors);
    void profile_layer_sensitivity(const std::string& layer_name, 
                                   const float* gradients, 
                                   int size);
    
    // Quantization strategies
    std::unique_ptr<QuantizedTensor> quantize_tensor(
        const std::string& name,
        const float* data,
        int rows, int cols,
        bool force_precision = false
    );
    
    // Model-wide quantization
    std::unordered_map<std::string, std::unique_ptr<QuantizedTensor>> quantize_full_model(
        const std::map<std::string, struct ggml_tensor*>& model_tensors
    );
    
    // Dynamic precision adjustment
    void update_precision_based_on_performance();
    void adjust_precision_for_layer(const std::string& layer_name, ComputePrecision new_precision);
    
    // Performance monitoring
    const PerformanceMetrics& get_metrics() const { return metrics; }
    void reset_metrics();
    void print_quantization_report() const;
    
    // Hardware-specific optimizations
    void enable_simd_optimizations();
    void enable_gpu_quantization();
    bool is_hardware_quantization_supported() const;
};

// ============================================================================
// Hardware-Optimized Quantization Kernels
// ============================================================================

namespace Kernels {

// SIMD-optimized quantization
void quantize_int4_simd(
    const float* input,
    uint8_t* output,
    const float* scales,
    int size,
    int group_size
);

void dequantize_int4_simd(
    const uint8_t* input,
    float* output,
    const float* scales,
    int size,
    int group_size
);

// Fused quantized matrix multiplication
void quantized_matmul_fused(
    const float* input,          // [batch, input_dim] 
    const uint8_t* weight_q,     // Quantized weights
    const float* scales,         // Quantization scales
    float* output,               // [batch, output_dim]
    int batch_size,
    int input_dim,
    int output_dim,
    int group_size
);

// Mixed precision accumulation (crucial for numerical stability)
void mixed_precision_accumulate(
    const float* input_fp32,     // High precision input
    const uint16_t* weight_fp16, // Half precision weights
    float* output_fp32,          // Full precision output
    int batch_size,
    int input_dim,
    int output_dim
);

} // namespace Kernels

// ============================================================================
// Integration with Multi-LoRA System
// ============================================================================

class QuantizedLoRAFusion {
private:
    std::unique_ptr<QuantizedTensor> base_model;
    std::vector<std::pair<std::string, std::unique_ptr<QuantizedTensor>>> lora_tensors;
    QuantizationEngine engine;
    
public:
    QuantizedLoRAFusion(const QuantizationParams& params = QuantizationParams{});
    
    // Set base model with automatic quantization
    void set_quantized_base(struct ggml_tensor* base_weights);
    
    // Add LoRA with precision optimization
    void add_quantized_lora(
        const std::string& name,
        struct ggml_tensor* lora_a,
        struct ggml_tensor* lora_b,
        float scale
    );
    
    // Optimized forward pass combining quantized base + LoRAs
    struct ggml_tensor* quantized_forward(
        struct ggml_tensor* input,
        struct ggml_context* ctx,
        const std::vector<std::string>& active_loras = {}
    );
    
    // Performance analysis
    void analyze_quantization_impact();
    void benchmark_vs_fp32();
    
    // Memory optimization
    size_t get_total_memory_usage() const;
    float get_compression_ratio() const;
    void optimize_memory_layout();
};

// ============================================================================
// Calibration and Fine-tuning Support
// ============================================================================

class QuantizationCalibrator {
private:
    struct CalibrationSample {
        std::vector<float> input;
        std::vector<float> expected_output;
        float loss_weight;
    };
    
    std::vector<CalibrationSample> calibration_data;
    QuantizationEngine* engine;
    
public:
    QuantizationCalibrator(QuantizationEngine* eng) : engine(eng) {}
    
    // Add calibration samples
    void add_calibration_sample(
        const float* input, int input_size,
        const float* output, int output_size,
        float weight = 1.0f
    );
    
    // Run calibration to find optimal quantization parameters
    void calibrate_quantization_parameters();
    void find_optimal_group_sizes();
    void analyze_quantization_error();
    
    // Post-training quantization fine-tuning
    void fine_tune_quantized_model(int num_epochs = 10);
    
    // Export optimized parameters
    void save_calibration_results(const std::string& filename);
    void load_calibration_results(const std::string& filename);
};

// ============================================================================
// Utility Functions
// ============================================================================

namespace Utils {

// Convert between different quantization formats
void convert_int4_to_int8(const uint8_t* int4_data, int8_t* int8_data, int size);
void convert_fp16_to_int4(const uint16_t* fp16_data, uint8_t* int4_data, int size);

// Analyze quantization error
float compute_quantization_error(const float* original, const float* quantized, int size);
void analyze_error_distribution(const float* errors, int size);

// Hardware detection
bool supports_int4_instructions();
bool supports_mixed_precision();
int get_optimal_group_size_for_hardware();

} // namespace Utils

} // namespace AdvancedQuantization

#endif // __ADVANCED_QUANTIZATION_MLX_HPP__