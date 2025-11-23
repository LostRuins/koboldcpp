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
    , requests_skipped(0)
    , vram_reuse(0)
    , ram_hits(0)
    , ram_misses(0)
    , context_switches(0)
    , saves_to_ram(0)
    , total_saved_prefill_tokens(0)
{
}

void SmartCacheMetrics::reset() {
    total_requests = 0;
    requests_skipped = 0;
    vram_reuse = 0;
    ram_hits = 0;
    ram_misses = 0;
    context_switches = 0;
    saves_to_ram = 0;
    total_saved_prefill_tokens = 0;
    similarity_on_hit.clear();
    similarity_on_miss.clear();
}

void SmartCacheMetrics::record_vram_reuse(size_t tokens_saved) {
    total_requests++;
    vram_reuse++;
    total_saved_prefill_tokens += tokens_saved;
}

void SmartCacheMetrics::record_ram_hit(float similarity, size_t tokens_saved) {
    total_requests++;
    ram_hits++;
    total_saved_prefill_tokens += tokens_saved;
    similarity_on_hit.push_back(similarity);
}

void SmartCacheMetrics::record_ram_miss(float similarity) {
    total_requests++;
    ram_misses++;
    similarity_on_miss.push_back(similarity);
}

void SmartCacheMetrics::record_context_switch() {
    context_switches++;
}

void SmartCacheMetrics::record_save_to_ram() {
    saves_to_ram++;
}

void SmartCacheMetrics::record_skip() {
    requests_skipped++;
}

float SmartCacheMetrics::get_ram_hit_rate() const {
    uint64_t total = ram_hits + ram_misses;
    return total > 0 ? static_cast<float>(ram_hits) / total : 0.0f;
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
        printf("\n[Smart Cache DEBUG] save_to_slot: slot %d doesn't exist!", slot_id);
        return; // Slot doesn't exist
    }

    CacheSlotMetadata& meta = it->second;
    meta.tokens = tokens;
    meta.ram_size_bytes = ram_size_bytes;
    meta.timestamp = static_cast<uint64_t>(std::time(nullptr));

    printf("\n[Smart Cache DEBUG] save_to_slot: slot %d saved %zu tokens (%zu MB)",
           slot_id, tokens.size(), ram_size_bytes / (1024*1024));
}

const std::vector<int>* SmartCacheManager::get_slot_tokens(int slot_id) const {
    auto it = slots_.find(slot_id);
    if (it == slots_.end()) {
        return nullptr;
    }
    return &(it->second.tokens);
}

// Find best matching RAM slot based on reusable tokens (prefix + LCS)
// Uses compute_purge_parameters - SAME logic as VRAM check
// Returns slot_id with most reusable tokens >= min_tokens, or -1 if none found
int SmartCacheManager::find_best_match(const std::vector<int>& prompt_tokens, int min_tokens, int genamt, int nctx) {
    if (!enabled_) {
        printf("\n[Smart Cache DEBUG] find_best_match: disabled");
        return -1;
    }

    if (slots_.empty()) {
        printf("\n[Smart Cache DEBUG] find_best_match: no slots");
        return -1;
    }

    printf("\n[Smart Cache DEBUG] find_best_match: searching %zu slots, min_tokens=%d, prompt_len=%zu",
           slots_.size(), min_tokens, prompt_tokens.size());

    int best_slot = -1;
    int best_reusable_tokens = 0;

    for (const auto& pair : slots_) {
        int slot_id = pair.first;
        const CacheSlotMetadata& meta = pair.second;

        // Skip VRAM slot
        if (slot_id == vram_slot_id_) {
            printf("\n[Smart Cache DEBUG]   Slot %d: SKIP (is VRAM slot)", slot_id);
            continue;
        }

        // Skip empty slots
        if (meta.tokens.empty()) {
            printf("\n[Smart Cache DEBUG]   Slot %d: SKIP (empty)", slot_id);
            continue;
        }

        printf("\n[Smart Cache DEBUG]   Slot %d: %zu tokens saved", slot_id, meta.tokens.size());

        // Use compute_purge_parameters to check reusable tokens (prefix + LCS)
        // SAME logic as VRAM check in gpttype_adapter.cpp
        int trimstart, purge_offset, purge_length;
        bool can_reuse = compute_purge_parameters(
            meta.tokens, prompt_tokens,
            genamt, nctx,
            &trimstart, &purge_offset, &purge_length
        );

        // Calculate total reusable tokens (prefix + LCS after purge)
        int reusable_tokens = trimstart;
        if (can_reuse && purge_length > 0) {
            // Can purge gap → reusable = total - gap
            reusable_tokens = meta.tokens.size() - purge_length;
        }

        printf("\n[Smart Cache DEBUG]     prefix=%d, can_purge=%s, purge_len=%d, reusable=%d",
               trimstart, can_reuse ? "YES" : "NO", purge_length, reusable_tokens);

        // Update best match based on most reusable tokens
        if (reusable_tokens > best_reusable_tokens) {
            best_reusable_tokens = reusable_tokens;
            best_slot = slot_id;
        }
    }

    printf("\n[Smart Cache DEBUG] find_best_match: best_slot=%d, best_reusable=%d (need >= %d)",
           best_slot, best_reusable_tokens, min_tokens);

    // Only return slot if reusable tokens meet minimum threshold
    if (best_reusable_tokens >= min_tokens) {
        return best_slot;
    }

    return -1; // No slot with sufficient reusable tokens
}

size_t SmartCacheManager::get_total_ram_usage() const {
    size_t total = 0;
    for (const auto& pair : slots_) {
        total += pair.second.ram_size_bytes;
    }
    return total;
}

} // namespace SmartCache
