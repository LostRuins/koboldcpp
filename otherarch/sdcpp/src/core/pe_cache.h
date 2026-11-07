#ifndef __SD_CORE_PE_CACHE_H__
#define __SD_CORE_PE_CACHE_H__

#include <functional>
#include <map>
#include <set>
#include <type_traits>

#include "core/tensor.hpp"
#include "ggml.h"

namespace sd {
    class PECacheKey {
        std::string name_;
        std::string fields_;

    public:
        explicit PECacheKey(std::string name)
            : name_(std::move(name)) {}
        const std::string& name() const { return name_; }

        template <typename T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
        PECacheKey& add(T value) {
            if constexpr (std::is_floating_point_v<T>) {
                const double field = static_cast<double>(value);
                fields_.push_back('f');
                fields_.append(reinterpret_cast<const char*>(&field), sizeof(field));
            } else {
                const uint64_t field = static_cast<uint64_t>(value);
                fields_.push_back('i');
                fields_.append(reinterpret_cast<const char*>(&field), sizeof(field));
            }
            return *this;
        }

        template <typename T>
        PECacheKey& add(const std::vector<T>& values) {
            fields_.push_back('v');
            add(values.size());
            for (const auto& value : values) {
                add(value);
            }
            return *this;
        }

        template <typename T>
        PECacheKey& add(const std::set<T>& values) {
            return add(std::vector<T>(values.begin(), values.end()));
        }

        template <typename T>
        PECacheKey& add(const Tensor<T>& values) {
            add(values.shape());
            for (int64_t i = 0; i < values.numel(); ++i) {
                add(values.data()[i]);
            }
            return *this;
        }

        PECacheKey& add_shape(const ggml_tensor* tensor);
        PECacheKey& add_shapes(const std::vector<ggml_tensor*>& tensors);

        bool operator==(const PECacheKey& other) const {
            return name_ == other.name_ && fields_ == other.fields_;
        }
    };

    class PECache {
        struct Entry {
            PECacheKey key;
            Tensor<float> value;
        };

        std::map<std::string, Entry> entries_;

    public:
        const Tensor<float>& get_or_create(const PECacheKey& key, const std::function<Tensor<float>()>& build);
    };

}

#endif  // __SD_CORE_PE_CACHE_H__
