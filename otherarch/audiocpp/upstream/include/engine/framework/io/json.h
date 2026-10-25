#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

namespace engine::io::json {

class Value {
public:
    enum class Kind {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    using Array = std::vector<Value>;
    using Object = std::unordered_map<std::string, Value>;

    Value() = default;
    Value(const Value & other);
    Value & operator=(const Value & other);
    Value(Value && other) noexcept;
    Value & operator=(Value && other) noexcept;
    ~Value();

    static Value make_null();
    static Value make_bool(bool value);
    static Value make_number(double value);
    static Value make_string(std::string value);
    static Value make_array(Array value);
    static Value make_object(Object value);

    Kind kind() const noexcept;
    bool is_null() const noexcept;
    bool is_bool() const noexcept;
    bool is_number() const noexcept;
    bool is_string() const noexcept;
    bool is_array() const noexcept;
    bool is_object() const noexcept;

    bool as_bool() const;
    double as_number() const;
    int64_t as_i64() const;
    float as_f32() const;
    const std::string & as_string() const;
    const Array & as_array() const;
    const Object & as_object() const;

    const Value & require(const std::string & key) const;
    const Value * find(const std::string & key) const noexcept;

private:
    Kind kind_ = Kind::Null;
    bool bool_value_ = false;
    double number_value_ = 0.0;
    std::string string_value_;
    std::unique_ptr<Array> array_value_;
    std::unique_ptr<Object> object_value_;
};

// Startup-only opt-in. Call before starting concurrent parser users; no toggle back.
void enable_serialized_json_parsing();
Value parse(std::string_view text);
Value parse_jsonc(std::string_view text);
Value parse_file(const std::filesystem::path & path);
Value parse_jsonc_file(const std::filesystem::path & path);
std::string stringify(const Value & value);
std::string stringify_number(double value);
std::string stringify_string(std::string_view value);

inline int64_t require_i64(const Value & object, const std::string & key) {
    return object.require(key).as_i64();
}

inline int require_i32(const Value & object, const std::string & key) {
    return static_cast<int>(require_i64(object, key));
}

inline float require_f32(const Value & object, const std::string & key) {
    return object.require(key).as_f32();
}

inline bool require_bool(const Value & object, const std::string & key) {
    return object.require(key).as_bool();
}

inline std::string require_string(const Value & object, const std::string & key) {
    return object.require(key).as_string();
}

inline int64_t optional_i64(const Value & object, const std::string & key, int64_t default_value) {
    const auto * value = object.find(key);
    return value != nullptr && value->is_number() ? value->as_i64() : default_value;
}

inline int64_t optional_nullable_i64(const Value & object, const std::string & key, int64_t default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    return value->as_i64();
}

inline int optional_i32(const Value & object, const std::string & key, int default_value) {
    return static_cast<int>(optional_i64(object, key, default_value));
}

inline float optional_f32(const Value & object, const std::string & key, float default_value) {
    const auto * value = object.find(key);
    return value != nullptr && value->is_number() ? value->as_f32() : default_value;
}

inline float optional_nullable_f32(const Value & object, const std::string & key, float default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    return value->as_f32();
}

inline bool optional_bool(const Value & object, const std::string & key, bool default_value) {
    const auto * value = object.find(key);
    return value != nullptr && value->is_bool() ? value->as_bool() : default_value;
}

inline bool optional_nullable_bool(const Value & object, const std::string & key, bool default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    return value->as_bool();
}

inline std::string optional_string(const Value & object, const std::string & key, std::string default_value) {
    const auto * value = object.find(key);
    return value != nullptr && value->is_string() ? value->as_string() : std::move(default_value);
}

inline std::string optional_nullable_string(const Value & object, const std::string & key, std::string default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    return value->as_string();
}

template <typename T>
std::vector<T> number_array_as(const Value & value) {
    std::vector<T> out;
    out.reserve(value.as_array().size());
    for (const auto & item : value.as_array()) {
        if constexpr (std::is_same_v<T, int32_t>) {
            out.push_back(static_cast<int32_t>(item.as_i64()));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            out.push_back(item.as_i64());
        } else {
            out.push_back(static_cast<T>(item.as_number()));
        }
    }
    return out;
}

inline std::vector<int64_t> require_i64_array_or_scalar(const Value & object, const std::string & key) {
    const auto & value = object.require(key);
    if (value.is_number()) {
        return {value.as_i64()};
    }
    return number_array_as<int64_t>(value);
}

inline std::vector<int64_t> require_i64_array(const Value & object, const std::string & key) {
    return number_array_as<int64_t>(object.require(key));
}

inline std::unordered_map<std::string, int64_t> require_i64_object(const Value & object, const std::string & key) {
    std::unordered_map<std::string, int64_t> values;
    const auto & map = object.require(key).as_object();
    values.reserve(map.size());
    for (const auto & [name, value] : map) {
        values.emplace(name, value.as_i64());
    }
    return values;
}

inline std::vector<int64_t> optional_i64_array(const Value & object, const std::string & key) {
    std::vector<int64_t> values;
    const auto * value = object.find(key);
    if (value == nullptr || !value->is_array()) {
        return values;
    }
    return number_array_as<int64_t>(*value);
}

inline std::vector<int64_t> optional_i64_array(
    const Value & object,
    const std::string & key,
    std::vector<int64_t> default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    return number_array_as<int64_t>(*value);
}

inline std::vector<int64_t> optional_i64_array_or_scalar(
    const Value & object,
    const std::string & key,
    std::vector<int64_t> default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    if (value->is_number()) {
        return {value->as_i64()};
    }
    return number_array_as<int64_t>(*value);
}

inline std::vector<float> optional_f32_array(const Value & object, const std::string & key) {
    std::vector<float> values;
    const auto * value = object.find(key);
    if (value == nullptr || !value->is_array()) {
        return values;
    }
    return number_array_as<float>(*value);
}

inline std::vector<std::string> require_string_array(const Value & object, const std::string & key) {
    std::vector<std::string> values;
    const auto & array = object.require(key).as_array();
    values.reserve(array.size());
    for (const auto & item : array) {
        values.push_back(item.as_string());
    }
    return values;
}

inline std::vector<std::string> optional_string_array(const Value & object, const std::string & key) {
    std::vector<std::string> values;
    const auto * value = object.find(key);
    if (value == nullptr || !value->is_array()) {
        return values;
    }
    for (const auto & item : value->as_array()) {
        values.push_back(item.as_string());
    }
    return values;
}

}  // namespace engine::io::json
