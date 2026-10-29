#pragma once

#include <any>
#include <string>
#include <unordered_map>

namespace engine::runtime {

class RuntimeCache {
public:
    bool contains(const std::string & key) const;

    template <typename T>
    void put(std::string key, T value) {
        cache_[std::move(key)] = std::move(value);
    }

    template <typename T>
    T * find(const std::string & key) {
        const auto it = cache_.find(key);
        if (it == cache_.end()) {
            return nullptr;
        }
        return std::any_cast<T>(&it->second);
    }

    template <typename T>
    const T * find(const std::string & key) const {
        const auto it = cache_.find(key);
        if (it == cache_.end()) {
            return nullptr;
        }
        return std::any_cast<T>(&it->second);
    }

    void erase(const std::string & key);
    void clear();

private:
    std::unordered_map<std::string, std::any> cache_;
};

}  // namespace engine::runtime
