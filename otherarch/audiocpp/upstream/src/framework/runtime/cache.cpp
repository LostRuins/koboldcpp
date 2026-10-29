#include "engine/framework/runtime/cache.h"

namespace engine::runtime {

bool RuntimeCache::contains(const std::string & key) const {
    return cache_.find(key) != cache_.end();
}

void RuntimeCache::erase(const std::string & key) {
    cache_.erase(key);
}

void RuntimeCache::clear() {
    cache_.clear();
}

}  // namespace engine::runtime
