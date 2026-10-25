#pragma once

#include <any>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::core {

// Owned by one loaded package. Callers opt in only for immutable weights and
// include component, device and storage type in the key. Graphs/caches stay local.
class SharedWeightCache {
public:
    template <typename T, typename Factory>
    std::shared_ptr<const T> get_or_load(const std::string & key, Factory && factory) const {
        return get_or_load_shared<T>(key, [&] {
            return std::make_shared<T>(std::forward<Factory>(factory)());
        });
    }

    // Some model loaders already return shared immutable weights. Retain their
    // ownership directly rather than copying their tensor/weight wrappers.
    template <typename T, typename Factory>
    std::shared_ptr<const T> get_or_load_shared(const std::string & key, Factory && factory) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto found = weights_.find(key);
        if (found != weights_.end()) {
            return std::any_cast<std::shared_ptr<const T>>(found->second);
        }
        std::shared_ptr<const T> value = std::forward<Factory>(factory)();
        if (value == nullptr) {
            throw std::runtime_error("shared weight loader returned null for " + key);
        }
        weights_.emplace(key, value);
        return value;
    }

private:
    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, std::any> weights_;
};

} // namespace engine::core
