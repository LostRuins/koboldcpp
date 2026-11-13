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
        auto& entries = entries_[key.name()];
        for (auto entry = entries.begin(); entry != entries.end(); ++entry) {
            if (entry->key == key) {
                entries.splice(entries.begin(), entries, entry);
                return entries.front().value;
            }
        }
        entries.push_front({key, build()});
        if (entries.size() > entries_per_name_) {
            entries.pop_back();
        }
        return entries.front().value;
    }
}
