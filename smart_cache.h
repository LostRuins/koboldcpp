#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

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
    uint64_t total_requests;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t context_switches;
    uint64_t saves_to_ram;
    uint64_t total_saved_prefill_tokens;

    std::vector<float> similarity_on_hit;
    std::vector<float> similarity_on_miss;

    SmartCacheMetrics();
    void reset();
    void record_hit(float similarity, size_t tokens_saved);
    void record_miss(float similarity);
    void record_context_switch();
    void record_save_to_ram();

    float get_hit_rate() const;
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

    // Similarity search - returns slot_id or -1
    int find_best_match(const std::vector<int>& prompt_tokens, int min_tokens);

    // Query methods
    size_t get_total_ram_usage() const;
    int get_vram_slot_id() const { return vram_slot_id_; }
    bool is_enabled() const { return enabled_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    // Debugging
    size_t get_slot_count() const { return slots_.size(); }
};

} // namespace SmartCache
