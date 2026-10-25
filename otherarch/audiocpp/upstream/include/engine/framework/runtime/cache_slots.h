#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace engine::runtime {

template <typename Key, typename Value, typename Equal = std::equal_to<Key>>
class CacheSlots {
public:
    explicit CacheSlots(std::size_t capacity = 1)
        : capacity_(capacity) {}

    std::size_t capacity() const noexcept {
        return capacity_;
    }

    std::size_t size() const noexcept {
        return entries_.size();
    }

    void set_capacity(std::size_t capacity) {
        capacity_ = capacity;
        evict_to_capacity();
    }

    Value * find(const Key & key) {
        for (auto & entry : entries_) {
            if (equal_(entry.key, key)) {
                entry.last_used = next_tick();
                return &entry.value;
            }
        }
        return nullptr;
    }

    const Value * find(const Key & key) const {
        for (const auto & entry : entries_) {
            if (equal_(entry.key, key)) {
                return &entry.value;
            }
        }
        return nullptr;
    }

    void put(Key key, Value value) {
        if (capacity_ == 0) {
            entries_.clear();
            return;
        }
        for (auto & entry : entries_) {
            if (equal_(entry.key, key)) {
                entry.key = std::move(key);
                entry.value = std::move(value);
                entry.last_used = next_tick();
                return;
            }
        }
        if (entries_.size() >= capacity_) {
            erase_lru();
        }
        entries_.push_back(Entry{
            std::move(key),
            std::move(value),
            next_tick(),
        });
    }

    void clear() {
        entries_.clear();
    }

private:
    struct Entry {
        Key key;
        Value value;
        std::uint64_t last_used = 0;
    };

    std::uint64_t next_tick() noexcept {
        return ++tick_;
    }

    void evict_to_capacity() {
        while (entries_.size() > capacity_) {
            erase_lru();
        }
    }

    void erase_lru() {
        if (entries_.empty()) {
            return;
        }
        std::size_t oldest = 0;
        for (std::size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].last_used < entries_[oldest].last_used) {
                oldest = i;
            }
        }
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(oldest));
    }

    std::size_t capacity_ = 1;
    std::uint64_t tick_ = 0;
    std::vector<Entry> entries_;
    Equal equal_;
};

}  // namespace engine::runtime
