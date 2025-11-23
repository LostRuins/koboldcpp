#include "smart_cache.h"
#include <algorithm>
#include <chrono>
#include <ctime>

namespace SmartCache {

// ============================================================================
// SmartCacheMetrics
// ============================================================================

SmartCacheMetrics::SmartCacheMetrics()
    : total_requests(0)
    , cache_hits(0)
    , cache_misses(0)
    , context_switches(0)
    , saves_to_ram(0)
    , total_saved_prefill_tokens(0)
{
}

void SmartCacheMetrics::reset() {
    total_requests = 0;
    cache_hits = 0;
    cache_misses = 0;
    context_switches = 0;
    saves_to_ram = 0;
    total_saved_prefill_tokens = 0;
    similarity_on_hit.clear();
    similarity_on_miss.clear();
}

void SmartCacheMetrics::record_hit(float similarity, size_t tokens_saved) {
    total_requests++;
    cache_hits++;
    total_saved_prefill_tokens += tokens_saved;
    similarity_on_hit.push_back(similarity);
}

void SmartCacheMetrics::record_miss(float similarity) {
    total_requests++;
    cache_misses++;
    similarity_on_miss.push_back(similarity);
}

void SmartCacheMetrics::record_context_switch() {
    context_switches++;
}

void SmartCacheMetrics::record_save_to_ram() {
    saves_to_ram++;
}

float SmartCacheMetrics::get_hit_rate() const {
    uint64_t total = cache_hits + cache_misses;
    return total > 0 ? static_cast<float>(cache_hits) / total : 0.0f;
}

float SmartCacheMetrics::get_avg_similarity_hit() const {
    if (similarity_on_hit.empty()) return 0.0f;
    float sum = 0.0f;
    for (float s : similarity_on_hit) sum += s;
    return sum / similarity_on_hit.size();
}

float SmartCacheMetrics::get_avg_similarity_miss() const {
    if (similarity_on_miss.empty()) return 0.0f;
    float sum = 0.0f;
    for (float s : similarity_on_miss) sum += s;
    return sum / similarity_on_miss.size();
}

// ============================================================================
// SmartCacheManager
// ============================================================================

SmartCacheManager::SmartCacheManager(double max_ram_gb)
    : vram_slot_id_(-1)
    , next_slot_id_(0)
    , max_ram_bytes_(static_cast<size_t>(max_ram_gb * 1024.0 * 1024.0 * 1024.0))
    , enabled_(false)
{
}

SmartCacheManager::~SmartCacheManager() {
    invalidate_all();
}

int SmartCacheManager::allocate_slot() {
    int new_id = next_slot_id_++;
    slots_[new_id] = CacheSlotMetadata(new_id);
    return new_id;
}

void SmartCacheManager::set_active_slot(int slot_id) {
    vram_slot_id_ = slot_id;
}

void SmartCacheManager::invalidate_slot(int slot_id) {
    auto it = slots_.find(slot_id);
    if (it != slots_.end()) {
        slots_.erase(it);
    }
    if (vram_slot_id_ == slot_id) {
        vram_slot_id_ = -1;
    }
}

void SmartCacheManager::invalidate_all() {
    slots_.clear();
    vram_slot_id_ = -1;
}

bool SmartCacheManager::evict_one_lru_slot() {
    if (slots_.empty()) {
        return false;
    }

    // Find oldest slot (excluding VRAM slot)
    int oldest_slot = -1;
    uint64_t oldest_timestamp = UINT64_MAX;

    for (const auto& pair : slots_) {
        int slot_id = pair.first;
        const CacheSlotMetadata& meta = pair.second;

        // Skip VRAM slot
        if (slot_id == vram_slot_id_) {
            continue;
        }

        if (meta.timestamp < oldest_timestamp) {
            oldest_timestamp = meta.timestamp;
            oldest_slot = slot_id;
        }
    }

    if (oldest_slot == -1) {
        return false; // No evictable slot found
    }

    invalidate_slot(oldest_slot);
    return true;
}

void SmartCacheManager::evict_lru_slots_to_fit(size_t required_bytes) {
    while (get_total_ram_usage() + required_bytes > max_ram_bytes_) {
        if (!evict_one_lru_slot()) {
            break; // No more slots to evict
        }
    }
}

void SmartCacheManager::save_to_slot(int slot_id, const std::vector<int>& tokens, size_t ram_size_bytes) {
    auto it = slots_.find(slot_id);
    if (it == slots_.end()) {
        return; // Slot doesn't exist
    }

    CacheSlotMetadata& meta = it->second;
    meta.tokens = tokens;
    meta.ram_size_bytes = ram_size_bytes;
    meta.timestamp = static_cast<uint64_t>(std::time(nullptr));
}

const std::vector<int>* SmartCacheManager::get_slot_tokens(int slot_id) const {
    auto it = slots_.find(slot_id);
    if (it == slots_.end()) {
        return nullptr;
    }
    return &(it->second.tokens);
}

int SmartCacheManager::find_best_match(const std::vector<int>& prompt_tokens, int min_tokens) {
    if (!enabled_ || slots_.empty()) {
        return -1;
    }

    int best_slot = -1;
    int best_prefix_count = 0;

    for (const auto& pair : slots_) {
        int slot_id = pair.first;
        const CacheSlotMetadata& meta = pair.second;

        // Skip VRAM slot
        if (slot_id == vram_slot_id_) {
            continue;
        }

        // Skip empty slots
        if (meta.tokens.empty()) {
            continue;
        }

        // Compute prefix match (simple token-by-token comparison)
        int prefix_count = 0;
        size_t min_len = std::min(meta.tokens.size(), prompt_tokens.size());

        for (size_t i = 0; i < min_len; ++i) {
            if (meta.tokens[i] == prompt_tokens[i]) {
                prefix_count++;
            } else {
                break; // Stop at first mismatch
            }
        }

        // Update best match
        if (prefix_count > best_prefix_count) {
            best_prefix_count = prefix_count;
            best_slot = slot_id;
        }
    }

    // Only return slot if it meets minimum threshold
    if (best_prefix_count >= min_tokens) {
        return best_slot;
    }

    return -1; // No match found
}

size_t SmartCacheManager::get_total_ram_usage() const {
    size_t total = 0;
    for (const auto& pair : slots_) {
        total += pair.second.ram_size_bytes;
    }
    return total;
}

} // namespace SmartCache
