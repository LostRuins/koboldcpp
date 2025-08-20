#ifndef __MEMORY_OPTIMIZATION_MLX_HPP__
#define __MEMORY_OPTIMIZATION_MLX_HPP__

#include "ggml_extend.hpp"
#include "multi_lora_mlx_fusion.hpp"
#include "advanced_quantization_mlx.hpp"
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * Memory Layout Optimization System - MLX Inspired
 * 
 * Implements advanced memory management patterns from MLX:
 * 1. Unified memory architecture optimization
 * 2. Cache-friendly tensor layouts 
 * 3. Memory pool management with arena allocation
 * 4. Lazy allocation and memory mapping
 * 5. Hardware-aware memory alignment
 * 6. Memory bandwidth optimization for multi-LoRA
 */

namespace MemoryOptimization {

// ============================================================================
// Memory Layout Strategies
// ============================================================================

enum class MemoryLayout {
    ROW_MAJOR,          // Standard row-major (C-style)
    COLUMN_MAJOR,       // Column-major (Fortran-style)
    BLOCKED_LAYOUT,     // Cache-friendly blocked layout
    INTERLEAVED,        // Interleaved for SIMD efficiency
    COMPRESSED_SPARSE,  // Compressed sparse representation
    ADAPTIVE_HYBRID     // Dynamically choose best layout
};

enum class AlignmentStrategy {
    NO_ALIGNMENT,       // No special alignment
    CACHE_LINE_ALIGNED, // 64-byte cache line alignment
    SIMD_ALIGNED,       // 256-bit AVX/512-bit AVX512 alignment
    PAGE_ALIGNED,       // 4KB page alignment
    OPTIMAL_HARDWARE    // Hardware-specific optimal alignment
};

enum class MemoryTier {
    L1_CACHE,          // L1 cache (32-64KB, fastest)
    L2_CACHE,          // L2 cache (256KB-1MB)
    L3_CACHE,          // L3 cache (8-32MB)  
    MAIN_MEMORY,       // Main RAM (GBs, slower)
    UNIFIED_MEMORY,    // Apple Silicon unified memory
    DISK_CACHE         // Disk-based cache (slowest)
};

// ============================================================================
// Tensor Memory Descriptor
// ============================================================================

struct TensorMemoryInfo {
    std::string name;
    void* data_ptr;
    size_t size_bytes;
    MemoryLayout layout;
    AlignmentStrategy alignment;
    MemoryTier tier;
    
    // Access pattern analysis
    size_t access_frequency;
    bool is_read_only;
    bool is_temporary;
    float cache_hit_ratio;
    
    // Hardware optimization
    int numa_node = -1;
    bool is_pinned = false;
    bool is_mapped = false;
    
    // Performance metrics
    double avg_access_time_ns = 0.0;
    size_t total_bytes_transferred = 0;
};

// ============================================================================
// Advanced Memory Pool System
// ============================================================================

class MemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        size_t alignment;
        bool is_free;
        MemoryTier tier;
        std::chrono::steady_clock::time_point last_access;
    };
    
    std::vector<MemoryBlock> memory_blocks;
    std::unordered_map<MemoryTier, std::vector<size_t>> free_blocks_by_tier;
    
    // Pool configuration
    size_t total_pool_size;
    size_t current_allocated;
    size_t peak_allocated;
    
    // Arena allocator for small allocations
    struct Arena {
        void* memory;
        size_t size;
        size_t offset;
        AlignmentStrategy alignment;
    };
    std::vector<Arena> arenas;

public:
    MemoryPool(size_t initial_size_mb = 1024);
    ~MemoryPool();
    
    // Core allocation functions
    void* allocate(size_t size, 
                   AlignmentStrategy alignment = AlignmentStrategy::CACHE_LINE_ALIGNED,
                   MemoryTier preferred_tier = MemoryTier::MAIN_MEMORY);
    
    void* allocate_aligned(size_t size, size_t alignment);
    void* allocate_temporary(size_t size, double lifetime_seconds = 60.0);
    
    void deallocate(void* ptr);
    void deallocate_temporary_expired();
    
    // Pool management
    void expand_pool(size_t additional_mb);
    void compact_memory();
    void defragment();
    
    // Memory tier management
    void promote_to_faster_tier(void* ptr);
    void demote_to_slower_tier(void* ptr);
    void analyze_access_patterns();
    
    // Statistics and monitoring
    size_t get_total_allocated() const { return current_allocated; }
    size_t get_peak_allocated() const { return peak_allocated; }
    double get_fragmentation_ratio() const;
    void print_memory_stats() const;
    
private:
    void* allocate_from_tier(size_t size, size_t alignment, MemoryTier tier);
    void coalesce_free_blocks();
    size_t find_best_fit_block(size_t size, size_t alignment, MemoryTier tier);
};

// ============================================================================
// Cache-Optimized Tensor Layout Manager
// ============================================================================

class TensorLayoutOptimizer {
private:
    struct LayoutConfig {
        MemoryLayout layout;
        int block_size_rows;
        int block_size_cols;
        int padding_bytes;
        bool use_prefetching;
    };
    
    std::unordered_map<std::string, LayoutConfig> tensor_configs;
    std::unordered_map<std::string, TensorMemoryInfo> tensor_info;
    
    // Hardware characteristics
    struct HardwareProfile {
        int cache_line_size = 64;
        int l1_cache_size = 32 * 1024;
        int l2_cache_size = 256 * 1024;
        int l3_cache_size = 8 * 1024 * 1024;
        int memory_bandwidth_gb_s = 100;
        bool supports_prefetch = true;
        bool has_unified_memory = false; // Apple Silicon
    } hardware;

public:
    TensorLayoutOptimizer();
    
    // Layout analysis and optimization
    MemoryLayout analyze_optimal_layout(
        const std::string& tensor_name,
        int rows, int cols,
        const std::vector<std::string>& access_patterns
    );
    
    void optimize_tensor_layout(
        struct ggml_tensor* tensor,
        const std::string& access_pattern = "sequential"
    );
    
    // Multi-tensor layout optimization for LoRA
    void optimize_multi_lora_layout(
        const std::vector<struct ggml_tensor*>& base_tensors,
        const std::vector<struct ggml_tensor*>& lora_tensors
    );
    
    // Cache blocking optimization
    void apply_cache_blocking(
        struct ggml_tensor* tensor,
        int block_rows, int block_cols
    );
    
    // Memory bandwidth optimization
    void optimize_memory_bandwidth(
        const std::vector<struct ggml_tensor*>& tensors,
        const std::string& access_pattern
    );
    
    // Hardware-specific optimizations
    void detect_hardware_capabilities();
    void apply_hardware_optimizations();
    
    // Performance monitoring
    void track_tensor_access(const std::string& name, size_t bytes_accessed);
    void analyze_cache_performance();
    void print_layout_report() const;

private:
    void convert_layout(struct ggml_tensor* tensor, MemoryLayout from, MemoryLayout to);
    void add_padding_for_alignment(struct ggml_tensor* tensor, int alignment);
    int calculate_optimal_block_size(int dimension, int cache_size);
};

// ============================================================================
// Memory Bandwidth Optimizer
// ============================================================================

class MemoryBandwidthOptimizer {
private:
    struct AccessPattern {
        std::string operation_name;
        std::vector<TensorMemoryInfo*> tensors;
        double bandwidth_requirement_gb_s;
        int memory_access_intensity; // Flops per byte
        bool is_bandwidth_bound;
    };
    
    std::vector<AccessPattern> access_patterns;
    double available_bandwidth_gb_s;
    
    // Prefetching configuration
    struct PrefetchConfig {
        int prefetch_distance;
        bool use_software_prefetch;
        bool use_hardware_prefetch;
        int prefetch_threads;
    } prefetch_config;

public:
    MemoryBandwidthOptimizer(double bandwidth_gb_s = 100.0);
    
    // Bandwidth analysis
    void analyze_memory_bottlenecks(
        const std::vector<struct ggml_tensor*>& tensors,
        const std::string& operation_sequence
    );
    
    void optimize_data_movement(
        const std::vector<struct ggml_tensor*>& tensors
    );
    
    // Multi-LoRA specific optimizations
    void optimize_lora_memory_access(
        struct ggml_tensor* base_weights,
        const std::vector<struct ggml_tensor*>& lora_weights,
        int batch_size
    );
    
    // Prefetching optimization
    void configure_prefetching(const std::string& tensor_name, 
                              const std::string& access_pattern);
    void enable_software_prefetching();
    void enable_hardware_prefetching();
    
    // Memory streaming
    void setup_memory_streaming(
        const std::vector<struct ggml_tensor*>& tensors,
        int stream_buffer_size_mb = 64
    );
    
    // Performance measurement
    double measure_actual_bandwidth();
    void benchmark_memory_operations();
    void print_bandwidth_report() const;

private:
    void prefetch_tensor_data(const struct ggml_tensor* tensor, int ahead_distance);
    void schedule_memory_transfers();
    double estimate_bandwidth_requirement(const struct ggml_tensor* tensor, 
                                         const std::string& access_pattern);
};

// ============================================================================
// Unified Memory Manager (Apple Silicon Optimized)
// ============================================================================

class UnifiedMemoryManager {
private:
    bool is_apple_silicon;
    size_t unified_memory_size;
    size_t memory_pressure_threshold;
    
    // Memory zones for different purposes
    struct MemoryZone {
        void* base_address;
        size_t size;
        std::string purpose; // "base_weights", "lora_weights", "activations", etc.
        bool is_read_only;
        size_t current_usage;
    };
    
    std::vector<MemoryZone> memory_zones;
    
    // Memory mapping for large models
    struct MappedRegion {
        void* mapped_address;
        size_t size;
        int file_descriptor;
        bool is_populated; // For lazy loading
    };
    
    std::unordered_map<std::string, MappedRegion> mapped_regions;

public:
    UnifiedMemoryManager();
    ~UnifiedMemoryManager();
    
    // Unified memory operations
    void* allocate_unified(size_t size, const std::string& purpose = "general");
    void setup_memory_zones(
        size_t base_weights_mb,
        size_t lora_weights_mb,
        size_t activation_mb,
        size_t temp_buffer_mb
    );
    
    // Memory mapping for large models
    void* map_model_file(const std::string& filepath, size_t offset = 0, size_t size = 0);
    void setup_lazy_loading(const std::string& tensor_name, 
                           const std::string& filepath,
                           size_t offset);
    
    // Apple Silicon specific optimizations
    void optimize_for_neural_engine();
    void configure_memory_compression();
    void enable_metal_buffer_sharing();
    
    // Memory pressure management
    void monitor_memory_pressure();
    void handle_memory_pressure();
    void evict_unused_tensors();
    
    // Performance optimization
    void prefault_memory_pages();
    void configure_numa_affinity();
    void optimize_tlb_usage();
    
    // Statistics and diagnostics
    size_t get_available_unified_memory() const;
    void print_memory_zones() const;
    void print_unified_memory_stats() const;

private:
    void detect_apple_silicon();
    void setup_default_zones();
    void* allocate_from_zone(const std::string& purpose, size_t size);
};

// ============================================================================
// Integrated Memory Optimization System
// ============================================================================

class IntegratedMemoryOptimizer {
private:
    std::unique_ptr<MemoryPool> memory_pool;
    std::unique_ptr<TensorLayoutOptimizer> layout_optimizer;
    std::unique_ptr<MemoryBandwidthOptimizer> bandwidth_optimizer;
    std::unique_ptr<UnifiedMemoryManager> unified_manager;
    
    // Integration state
    bool is_initialized = false;
    bool apple_silicon_optimized = false;
    bool prefetching_enabled = false;

public:
    IntegratedMemoryOptimizer(size_t initial_pool_mb = 2048);
    ~IntegratedMemoryOptimizer();
    
    // Initialization and configuration
    void initialize_for_hardware();
    void configure_for_multi_lora(
        int num_base_tensors,
        int num_lora_tensors,
        int max_batch_size
    );
    
    // Comprehensive optimization
    void optimize_full_model(
        const std::map<std::string, struct ggml_tensor*>& base_tensors,
        const std::map<std::string, struct ggml_tensor*>& lora_tensors
    );
    
    // Runtime optimization
    void optimize_for_inference_batch(
        const std::vector<std::string>& active_tensors,
        int batch_size
    );
    
    void optimize_memory_during_generation(
        const std::vector<std::string>& current_active_loras
    );
    
    // Memory management interface  
    void* allocate_optimized(
        size_t size,
        const std::string& tensor_name,
        const std::string& usage_pattern = "sequential"
    );
    
    void deallocate_optimized(void* ptr, const std::string& tensor_name);
    
    // Performance monitoring and tuning
    void start_performance_monitoring();
    void analyze_runtime_performance();
    void auto_tune_parameters();
    
    // Integration with existing systems
    void integrate_with_lora_processor(MultiLoRAMLX::MultiLoRAProcessor* processor);
    void integrate_with_quantization(AdvancedQuantization::QuantizationEngine* engine);
    
    // Comprehensive reporting
    void print_comprehensive_report() const;
    void export_optimization_profile(const std::string& filename) const;

private:
    void detect_and_configure_hardware();
    void setup_memory_hierarchy();
    void configure_bandwidth_optimization();
    void apply_apple_silicon_optimizations();
};

// ============================================================================
// Utility Functions and Helpers
// ============================================================================

namespace Utils {

// Memory alignment utilities
void* align_pointer(void* ptr, size_t alignment);
size_t calculate_aligned_size(size_t size, size_t alignment);
bool is_properly_aligned(const void* ptr, size_t alignment);

// Cache optimization utilities
int get_optimal_tile_size(int dimension, int cache_size);
void optimize_loop_tiling(int& tile_rows, int& tile_cols, 
                         int matrix_rows, int matrix_cols,
                         int cache_size);

// Hardware detection utilities
bool detect_apple_silicon();
bool detect_unified_memory();
size_t get_cache_line_size();
size_t get_page_size();
int get_numa_nodes();

// Performance measurement utilities
double measure_memory_latency(void* ptr, size_t size);
double measure_memory_bandwidth(void* src, void* dst, size_t size);
double measure_cache_miss_rate(void* ptr, size_t size, const std::string& access_pattern);

// Memory layout conversion utilities
void convert_row_major_to_blocked(const float* src, float* dst, 
                                 int rows, int cols, 
                                 int block_rows, int block_cols);
void convert_blocked_to_row_major(const float* src, float* dst,
                                 int rows, int cols,
                                 int block_rows, int block_cols);

} // namespace Utils

} // namespace MemoryOptimization

#endif // __MEMORY_OPTIMIZATION_MLX_HPP__