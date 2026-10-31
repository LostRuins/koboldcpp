#include "core/pe_cache.h"

namespace sd {
    PECacheKey& PECacheKey::add_shape(const ggml_tensor* tensor) {
        add(tensor != nullptr);
        if (tensor != nullptr) {
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                add(tensor->ne[i]);
            }
        }
        return *this;
    }

    PECacheKey& PECacheKey::add_shapes(const std::vector<ggml_tensor*>& tensors) {
        add(tensors.size());
        for (const auto* tensor : tensors) {
            add_shape(tensor);
        }
        return *this;
    }

    const Tensor<float>& PECache::get_or_create(const PECacheKey& key, const std::function<Tensor<float>()>& build) {
        auto entry = entries_.find(key.name());
        if (entry != entries_.end() && entry->second.key == key) {
            return entry->second.value;
        }
        auto value = build();
        return entries_.insert_or_assign(key.name(), Entry{key, std::move(value)}).first->second.value;
    }
}
