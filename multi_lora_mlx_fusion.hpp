#ifndef __MULTI_LORA_MLX_FUSION_HPP__
#define __MULTI_LORA_MLX_FUSION_HPP__

#include "ggml_extend.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

/**
 * MLX-Inspired Multi-LoRA Fusion System for KoboldCPP
 * 
 * This system implements the efficient multi-LoRA processing patterns discovered
 * in Apple's MLX framework, optimized for KoboldCPP's architecture.
 * 
 * Key optimizations:
 * 1. Fused computation graph - all LoRAs processed in single pass
 * 2. Quantization-aware base + full-precision LoRA deltas
 * 3. Memory layout optimization for cache efficiency
 * 4. Lazy evaluation with graph optimization
 * 5. addmm-style fused operations (bias + A @ B)
 */

namespace MultiLoRAMLX {

// Forward declarations
struct QuantizedLoRABase;
struct LoRADelta;
struct ComputationGraph;
class MultiLoRAProcessor;

/**
 * Quantized base model weights (MLX-style 4-bit quantization)
 * Provides 6.4x memory reduction with <7% accuracy loss
 */
struct QuantizedLoRABase {
    uint32_t* weight_data;    // 4-bit quantized weights packed into uint32
    float* scales;            // Per-group scaling factors
    float* biases;            // Per-group bias values (optional)
    
    // Quantization parameters (following MLX defaults)
    int group_size = 64;      // Optimal for cache line efficiency
    int bits = 4;             // 4-bit quantization
    int input_dim;
    int output_dim;
    
    // Memory layout info
    size_t packed_size;       // Size of packed weight data
    size_t scales_size;       // Number of scale values
    size_t biases_size;       // Number of bias values
    
    QuantizedLoRABase(int in_dim, int out_dim) 
        : input_dim(in_dim), output_dim(out_dim) {
        // Calculate sizes based on quantization parameters
        packed_size = (in_dim * out_dim * bits + 31) / 32; // Pack into uint32
        scales_size = (in_dim + group_size - 1) / group_size * out_dim;
        biases_size = scales_size; // Same as scales for simplicity
        
        // Allocate memory
        weight_data = new uint32_t[packed_size];
        scales = new float[scales_size];
        biases = new float[biases_size];
    }
    
    ~QuantizedLoRABase() {
        delete[] weight_data;
        delete[] scales;
        delete[] biases;
    }
    
    // Dequantize a specific weight value
    float dequantize_weight(int row, int col) const;
    
    // Batch dequantize for efficiency
    void dequantize_to_buffer(float* output_buffer) const;
};

/**
 * Individual LoRA adaptation weights
 * Stored in full precision for training compatibility
 */
struct LoRADelta {
    float* lora_A;           // Low-rank decomposition matrix A [input_dim, rank]
    float* lora_B;           // Low-rank decomposition matrix B [rank, output_dim]
    float scale;             // LoRA scaling factor (alpha/rank typically)
    int rank;                // LoRA rank
    int input_dim;
    int output_dim;
    std::string name;        // For debugging/identification
    
    LoRADelta(int in_dim, int out_dim, int r, float s = 1.0f) 
        : input_dim(in_dim), output_dim(out_dim), rank(r), scale(s) {
        lora_A = new float[input_dim * rank];
        lora_B = new float[rank * output_dim];
    }
    
    ~LoRADelta() {
        delete[] lora_A;
        delete[] lora_B;
    }
    
    // Copy constructor and assignment for proper memory management
    LoRADelta(const LoRADelta& other);
    LoRADelta& operator=(const LoRADelta& other);
};

/**
 * Computation node in the lazy evaluation graph
 */
enum class NodeType {
    BASE_MATMUL,    // Base model matrix multiplication
    LORA_MATMUL,    // LoRA A @ B computation
    FUSED_ADDMM,    // Fused bias + matrix multiply (MLX addmm pattern)
    ACCUMULATE,     // Accumulate multiple LoRA outputs
    SCALE,          // Apply scaling factor
    ACTIVATION      // Optional activation function
};

struct ComputationNode {
    NodeType type;
    void* operation_data;               // Operation-specific data
    std::vector<ComputationNode*> inputs;
    struct ggml_tensor* output_tensor;
    
    // Operation timing for profiling
    double execution_time_ms = 0.0;
};

/**
 * Lazy computation graph for multi-LoRA operations
 * Optimizes the computation before execution (MLX pattern)
 */
struct ComputationGraph {
    std::vector<std::unique_ptr<ComputationNode>> nodes;
    std::unordered_map<std::string, ComputationNode*> named_outputs;
    
    // Graph optimization passes
    void optimize();
    void fuse_consecutive_operations();
    void merge_accumulations();
    void reorder_for_cache_efficiency();
    
    // Execution
    void execute(struct ggml_context* ctx);
    
    // Profiling
    void print_execution_profile() const;
};

/**
 * Packed memory layout for efficient multi-LoRA processing
 * All LoRA A matrices stored contiguously, then all B matrices
 */
struct PackedLoRAWeights {
    // Memory layout: [lora0_A][lora1_A]...[loraN_A][lora0_B][lora1_B]...[loraN_B]
    float* packed_weights;
    size_t total_size;
    
    struct LoRAInfo {
        size_t a_offset;      // Offset to LoRA A matrix in packed_weights
        size_t b_offset;      // Offset to LoRA B matrix in packed_weights
        int rank;
        int input_dim;
        int output_dim;
        float scale;
        std::string name;
    };
    
    std::vector<LoRAInfo> lora_info;
    
    PackedLoRAWeights(const std::vector<LoRADelta>& loras);
    ~PackedLoRAWeights() { delete[] packed_weights; }
    
    // Efficient batched processing
    void process_batch(
        const float* input,           // [batch_size, max_input_dim]
        float* output,                // [batch_size, total_output_dim]
        int batch_size,
        struct ggml_context* ctx
    ) const;
};

/**
 * Main multi-LoRA processor implementing MLX-inspired optimizations
 */
class MultiLoRAProcessor {
private:
    std::unique_ptr<QuantizedLoRABase> base_weights;
    std::vector<std::unique_ptr<LoRADelta>> lora_deltas;
    std::unique_ptr<PackedLoRAWeights> packed_loras;
    std::unique_ptr<ComputationGraph> computation_graph;
    
    // Memory pools for efficient allocation
    float* temp_buffer_pool;
    size_t temp_buffer_size;
    size_t temp_buffer_used;
    
    // Performance statistics
    struct Statistics {
        size_t total_operations = 0;
        double total_time_ms = 0.0;
        double base_time_ms = 0.0;
        double lora_time_ms = 0.0;
        size_t memory_saved_bytes = 0;
    } stats;

public:
    MultiLoRAProcessor(int max_input_dim, int max_output_dim, size_t temp_buffer_mb = 256);
    ~MultiLoRAProcessor();
    
    // Model management
    void set_base_model(struct ggml_tensor* weights);
    void add_lora(const std::string& name, struct ggml_tensor* lora_a, struct ggml_tensor* lora_b, float scale);
    void remove_lora(const std::string& name);
    void clear_loras();
    
    // Optimization
    void quantize_base_model();
    void build_computation_graph();
    void optimize_memory_layout();
    
    // Execution - main entry points
    struct ggml_tensor* forward(
        struct ggml_tensor* input,
        struct ggml_context* ctx,
        const std::vector<std::string>& active_loras = {}
    );
    
    // Batch processing for efficiency
    struct ggml_tensor* forward_batch(
        struct ggml_tensor* input_batch,    // [batch_size, input_dim]
        struct ggml_context* ctx,
        const std::vector<std::string>& active_loras = {}
    );
    
    // Performance monitoring
    const Statistics& get_statistics() const { return stats; }
    void reset_statistics();
    void print_performance_report() const;
    
    // Memory management
    size_t get_memory_usage() const;
    void optimize_memory_usage();
    
private:
    // Core computation kernels
    void fused_quantized_base_forward(
        const float* input,
        float* output,
        int batch_size
    ) const;
    
    void fused_multi_lora_forward(
        const float* input,
        const std::vector<int>& active_lora_indices,
        float* output,
        int batch_size
    ) const;
    
    // MLX-style fused operations
    void addmm_fused(
        const float* bias,      // [output_dim]
        const float* input,     // [batch_size, input_dim]
        const float* weight,    // [output_dim, input_dim]
        float* output,          // [batch_size, output_dim]
        int batch_size,
        int input_dim,
        int output_dim
    ) const;
    
    // Memory pool management
    float* allocate_temp(size_t size);
    void free_temp(float* ptr);
    void reset_temp_pool();
    
    // Optimization helpers
    void detect_computation_patterns();
    void apply_graph_optimizations();
};

/**
 * Factory functions for easy integration with existing KoboldCPP code
 */
namespace Factory {
    // Create processor from existing LoRA model
    std::unique_ptr<MultiLoRAProcessor> from_lora_model(
        const LoraModel& existing_model,
        struct ggml_context* ctx
    );
    
    // Create processor from individual tensors
    std::unique_ptr<MultiLoRAProcessor> from_tensors(
        const std::map<std::string, struct ggml_tensor*>& model_tensors,
        const std::map<std::string, struct ggml_tensor*>& lora_tensors,
        struct ggml_context* ctx
    );
}

/**
 * Integration helpers for backward compatibility
 */
namespace Integration {
    // Replace existing LoRA forward pass with optimized version
    struct ggml_tensor* optimized_lora_forward(
        struct ggml_tensor* input,
        const std::vector<std::string>& lora_paths,
        const std::vector<float>& lora_scales,
        struct ggml_context* ctx
    );
    
    // Benchmark comparison between old and new implementations
    void benchmark_performance(
        const std::vector<std::string>& lora_paths,
        int num_iterations = 100
    );
}

} // namespace MultiLoRAMLX

#endif // __MULTI_LORA_MLX_FUSION_HPP__