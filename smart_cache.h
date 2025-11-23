#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

// Forward declaration: compute_purge_parameters is defined in gpttype_adapter.cpp
// Computes reusable tokens (prefix + LCS) using exact logic from concedo_experimental
extern bool compute_purge_parameters(
    const std::vector<int>& current_context_tokens,
    const std::vector<int>& new_context_tokens,
    const int genamt,
    const int nctx,
    int* out_trimstart,
    int* out_purge_offset,
    int* out_purge_length);

// Smart Cache: Intelligent context switching for KV cache management
// Saves/loads KV cache snapshots to/from RAM for fast context switching
// Implements LRU eviction based on timestamp

namespace SmartCache {

// Metadata for a single RAM cache slot
struct CacheSlotMetadata {
    int slot_id;
    std::vector<int> tokens;        // Full context tokens
    uint64_t timestamp;             // Unix timestamp (for LRU)
    size_t ram_size_bytes;          // KV cache size in RAM

    CacheSlotMetadata() : slot_id(-1), timestamp(0), ram_size_bytes(0) {}
    explicit CacheSlotMetadata(int id) : slot_id(id), timestamp(0), ram_size_bytes(0) {}
};

// Metrics tracking
struct SmartCacheMetrics {
    uint64_t total_requests;          // Total valid requests (prompt >= MIN_TOKENS)
    uint64_t requests_skipped;        // Requests skipped (prompt < MIN_TOKENS)
    uint64_t vram_reuse;              // VRAM prefix >= MIN_TOKENS (normal behavior)
    uint64_t ram_hits;                // Loaded from RAM (true smart cache benefit)
    uint64_t ram_misses;              // No RAM slot found, cold prefill
    uint64_t context_switches;        // Total context switches
    uint64_t saves_to_ram;            // Successful saves to RAM
    uint64_t total_saved_prefill_tokens;

    std::vector<float> similarity_on_hit;
    std::vector<float> similarity_on_miss;

    SmartCacheMetrics();
    void reset();
    void record_vram_reuse(size_t tokens_saved);
    void record_ram_hit(float similarity, size_t tokens_saved);
    void record_ram_miss(float similarity);
    void record_context_switch();
    void record_save_to_ram();
    void record_skip();

    float get_ram_hit_rate() const;   // Only RAM hits vs (RAM hits + misses)
    float get_avg_similarity_hit() const;
    float get_avg_similarity_miss() const;
};

// Main Smart Cache Manager
class SmartCacheManager {
private:
    std::unordered_map<int, CacheSlotMetadata> slots_;
    int vram_slot_id_;              // Current active slot in VRAM (-1 if none)
    int next_slot_id_;              // Simple counter for slot IDs
    size_t max_ram_bytes_;          // Maximum RAM for cache
    bool enabled_;

public:
    SmartCacheManager(double max_ram_gb);
    ~SmartCacheManager();

    // Slot management
    int allocate_slot();            // Returns new slot_id
    void set_active_slot(int slot_id);
    void invalidate_slot(int slot_id);
    void invalidate_all();

    // LRU eviction
    bool evict_one_lru_slot();      // Evicts oldest slot (by timestamp)
    void evict_lru_slots_to_fit(size_t required_bytes);

    // Slot operations
    void save_to_slot(int slot_id, const std::vector<int>& tokens, size_t ram_size_bytes);
    const std::vector<int>* get_slot_tokens(int slot_id) const;

    // Search for best slot based on reusable tokens (prefix + LCS via compute_purge_parameters)
    // Returns slot_id with most reusable tokens >= min_tokens, or -1 if none found
    int find_best_match(const std::vector<int>& prompt_tokens, int min_tokens, int genamt, int nctx);

    // Query methods
    size_t get_total_ram_usage() const;
    int get_vram_slot_id() const { return vram_slot_id_; }
    bool is_enabled() const { return enabled_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    // Debugging
    size_t get_slot_count() const { return slots_.size(); }
};

} // namespace SmartCache
