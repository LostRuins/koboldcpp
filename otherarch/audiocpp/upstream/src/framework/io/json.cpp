#include "engine/framework/io/json.h"

#include "engine/framework/io/filesystem.h"

#include <cctype>
#include <cmath>
#include <mutex>
#include <string_view>
#include <stdexcept>

#include "cJSON.h"

namespace engine::io::json {

namespace {
// Set once at parallel-server startup, before any concurrent parser callers.
bool serialized_json_parsing = false;

cJSON * make_cjson(const Value & value) {
    switch (value.kind()) {
    case Value::Kind::Null:
        return cJSON_CreateNull();
    case Value::Kind::Bool:
        return cJSON_CreateBool(value.as_bool());
    case Value::Kind::Number:
        return cJSON_CreateNumber(value.as_number());
    case Value::Kind::String:
        return cJSON_CreateString(value.as_string().c_str());
    case Value::Kind::Array: {
        cJSON * array_json = cJSON_CreateArray();
        if (array_json == nullptr) {
            throw std::runtime_error("failed to create json array");
        }
        const auto & array_values = value.as_array();
        for (const auto & child_value : array_values) {
            cJSON * child = make_cjson(child_value);
            if (child == nullptr) {
                cJSON_Delete(array_json);
                throw std::runtime_error("failed to create json array element");
            }
            cJSON_AddItemToArray(array_json, child);
        }
        return array_json;
    }
    case Value::Kind::Object: {
        cJSON * object = cJSON_CreateObject();
        if (object == nullptr) {
            throw std::runtime_error("failed to create json object");
        }
        for (const auto & [key, child] : value.as_object()) {
            cJSON * child_json = make_cjson(child);
            if (child_json == nullptr) {
                cJSON_Delete(object);
                throw std::runtime_error("failed to create json object field");
            }
            if (!cJSON_AddItemToObject(object, key.c_str(), child_json)) {
                cJSON_Delete(child_json);
                cJSON_Delete(object);
                throw std::runtime_error("failed to add json object field");
            }
        }
        return object;
    }
    }
    throw std::runtime_error("unsupported json value kind");
}

std::string print_cjson(cJSON * root) {
    if (root == nullptr) {
        throw std::runtime_error("cannot print null cJSON root");
    }
    char * text = cJSON_PrintUnformatted(root);
    if (text == nullptr) {
        throw std::runtime_error("failed to serialize json");
    }
    std::string out(text);
    cJSON_free(text);
    return out;
}

std::string strip_jsonc_comments(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_string = false;
    bool escaped = false;
    for (size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        if (in_string) {
            out.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            out.push_back(ch);
            continue;
        }
        if (ch == '/' && index + 1 < text.size() && text[index + 1] == '/') {
            while (index < text.size() && text[index] != '\n') {
                ++index;
            }
            if (index < text.size()) {
                out.push_back('\n');
            }
            continue;
        }
        out.push_back(ch);
    }
    return out;
}

std::string strip_jsonc_trailing_commas(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_string = false;
    bool escaped = false;
    for (size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        if (in_string) {
            out.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            out.push_back(ch);
            continue;
        }
        if (ch == ',') {
            size_t lookahead = index + 1;
            while (lookahead < text.size() && std::isspace(static_cast<unsigned char>(text[lookahead])) != 0) {
                ++lookahead;
            }
            if (lookahead < text.size() && (text[lookahead] == '}' || text[lookahead] == ']')) {
                continue;
            }
        }
        out.push_back(ch);
    }
    return out;
}

Value convert_cjson(const cJSON * node) {
    if (node == nullptr) {
        throw std::runtime_error("json conversion received null node");
    }
    if (cJSON_IsNull(node) != 0) {
        return Value::make_null();
    }
    if (cJSON_IsBool(node) != 0) {
        return Value::make_bool(cJSON_IsTrue(node) != 0);
    }
    if (cJSON_IsNumber(node) != 0) {
        return Value::make_number(node->valuedouble);
    }
    if (cJSON_IsString(node) != 0) {
        return Value::make_string(node->valuestring != nullptr ? node->valuestring : "");
    }
    if (cJSON_IsArray(node) != 0) {
        Value::Array out;
        const int size = cJSON_GetArraySize(node);
        if (size > 0) {
            out.reserve(static_cast<size_t>(size));
        }
        for (const cJSON * child = node->child; child != nullptr; child = child->next) {
            out.push_back(convert_cjson(child));
        }
        return Value::make_array(std::move(out));
    }
    if (cJSON_IsObject(node) != 0) {
        Value::Object out;
        for (const cJSON * child = node->child; child != nullptr; child = child->next) {
            if (child->string == nullptr) {
                throw std::runtime_error("json object key is missing");
            }
            if (out.find(child->string) != out.end()) {
                throw std::runtime_error("duplicate json object key: " + std::string(child->string));
            }
            out.emplace(child->string, convert_cjson(child));
        }
        return Value::make_object(std::move(out));
    }
    throw std::runtime_error("unsupported json node type");
}

}  // namespace

Value Value::make_null() {
    return {};
}

Value::Value(const Value & other)
    : kind_(other.kind_),
      bool_value_(other.bool_value_),
      number_value_(other.number_value_),
      string_value_(other.string_value_) {
    if (other.array_value_) {
        array_value_ = std::make_unique<Array>(*other.array_value_);
    }
    if (other.object_value_) {
        object_value_ = std::make_unique<Object>(*other.object_value_);
    }
}

Value & Value::operator=(const Value & other) {
    if (this == &other) {
        return *this;
    }
    kind_ = other.kind_;
    bool_value_ = other.bool_value_;
    number_value_ = other.number_value_;
    string_value_ = other.string_value_;
    array_value_ = other.array_value_ ? std::make_unique<Array>(*other.array_value_) : nullptr;
    object_value_ = other.object_value_ ? std::make_unique<Object>(*other.object_value_) : nullptr;
    return *this;
}

Value::Value(Value && other) noexcept = default;

Value & Value::operator=(Value && other) noexcept = default;

Value::~Value() = default;

Value Value::make_bool(bool value) {
    Value out;
    out.kind_ = Kind::Bool;
    out.bool_value_ = value;
    return out;
}

Value Value::make_number(double value) {
    Value out;
    out.kind_ = Kind::Number;
    out.number_value_ = value;
    return out;
}

Value Value::make_string(std::string value) {
    Value out;
    out.kind_ = Kind::String;
    out.string_value_ = std::move(value);
    return out;
}

Value Value::make_array(Array value) {
    Value out;
    out.kind_ = Kind::Array;
    out.array_value_ = std::make_unique<Array>(std::move(value));
    return out;
}

Value Value::make_object(Object value) {
    Value out;
    out.kind_ = Kind::Object;
    out.object_value_ = std::make_unique<Object>(std::move(value));
    return out;
}

Value::Kind Value::kind() const noexcept {
    return kind_;
}

bool Value::is_null() const noexcept {
    return kind_ == Kind::Null;
}

bool Value::is_bool() const noexcept {
    return kind_ == Kind::Bool;
}

bool Value::is_number() const noexcept {
    return kind_ == Kind::Number;
}

bool Value::is_string() const noexcept {
    return kind_ == Kind::String;
}

bool Value::is_array() const noexcept {
    return kind_ == Kind::Array;
}

bool Value::is_object() const noexcept {
    return kind_ == Kind::Object;
}

bool Value::as_bool() const {
    if (!is_bool()) {
        throw std::runtime_error("json value is not a bool");
    }
    return bool_value_;
}

double Value::as_number() const {
    if (!is_number()) {
        throw std::runtime_error("json value is not a number");
    }
    return number_value_;
}

int64_t Value::as_i64() const {
    const double value = as_number();
    if (!std::isfinite(value) || std::floor(value) != value) {
        throw std::runtime_error("json number is not an integer");
    }
    return static_cast<int64_t>(value);
}

float Value::as_f32() const {
    return static_cast<float>(as_number());
}

const std::string & Value::as_string() const {
    if (!is_string()) {
        throw std::runtime_error("json value is not a string");
    }
    return string_value_;
}

const Value::Array & Value::as_array() const {
    if (!is_array()) {
        throw std::runtime_error("json value is not an array");
    }
    return *array_value_;
}

const Value::Object & Value::as_object() const {
    if (!is_object()) {
        throw std::runtime_error("json value is not an object");
    }
    return *object_value_;
}

const Value & Value::require(const std::string & key) const {
    const Value * value = find(key);
    if (value == nullptr) {
        throw std::runtime_error("missing required json key: " + key);
    }
    return *value;
}

const Value * Value::find(const std::string & key) const noexcept {
    if (!is_object()) {
        return nullptr;
    }
    const auto it = object_value_->find(key);
    return it == object_value_->end() ? nullptr : &it->second;
}

void enable_serialized_json_parsing() {
    serialized_json_parsing = true;
}

Value parse(std::string_view text) {
    const char * parse_end = nullptr;
    cJSON * root = nullptr;
    if (serialized_json_parsing) {
        // cJSON writes its global error position even when return_parse_end is
        // supplied. Serialize that call; conversion and deletion are per-tree.
        static std::mutex parse_mutex;
        std::lock_guard<std::mutex> lock(parse_mutex);
        root = cJSON_ParseWithLengthOpts(text.data(), text.size(), &parse_end, 0);
    } else {
        root = cJSON_ParseWithLengthOpts(text.data(), text.size(), &parse_end, 0);
    }
    if (root == nullptr) {
        const ptrdiff_t error_offset = parse_end == nullptr ? -1 : parse_end - text.data();
        throw std::runtime_error("failed to parse json at byte " + std::to_string(error_offset));
    }
    try {
        if (parse_end != nullptr) {
            for (const char * cursor = parse_end; cursor < text.data() + text.size(); ++cursor) {
                if (std::isspace(static_cast<unsigned char>(*cursor)) == 0) {
                    throw std::runtime_error("unexpected trailing json content");
                }
            }
        }
        Value out = convert_cjson(root);
        cJSON_Delete(root);
        return out;
    } catch (...) {
        cJSON_Delete(root);
        throw;
    }
}

Value parse_jsonc(std::string_view text) {
    return parse(strip_jsonc_trailing_commas(strip_jsonc_comments(text)));
}

Value parse_file(const std::filesystem::path & path) {
    return parse(engine::io::read_text_file(path));
}

Value parse_jsonc_file(const std::filesystem::path & path) {
    return parse_jsonc(engine::io::read_text_file(path));
}

std::string stringify(const Value & value) {
    cJSON * root = make_cjson(value);
    if (root == nullptr) {
        throw std::runtime_error("failed to convert value to json");
    }
    try {
        std::string out = print_cjson(root);
        cJSON_Delete(root);
        return out;
    } catch (...) {
        cJSON_Delete(root);
        throw;
    }
}

std::string stringify_number(double value) {
    return stringify(Value::make_number(value));
}

std::string stringify_string(std::string_view value) {
    return stringify(Value::make_string(std::string(value)));
}

}  // namespace engine::io::json
