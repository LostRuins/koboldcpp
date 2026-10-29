#pragma once

#include "engine/framework/io/config.h"

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::io::yaml {

struct FlattenedDocument {
    ConfigMap scalars;
    std::unordered_map<std::string, std::vector<std::string>> lists;
};

ConfigMap parse_scalar_mapping(std::string_view text);
FlattenedDocument parse_flattened_document(std::string_view text);
bool parse_bool_scalar(const std::string & value, const std::string & key);
std::string require_string(const FlattenedDocument & document, const std::string & key);
std::string optional_string(const FlattenedDocument & document, const std::string & key, std::string default_value);
int require_int(const FlattenedDocument & document, const std::string & key);
int64_t require_i64(const FlattenedDocument & document, const std::string & key);
float require_float(const FlattenedDocument & document, const std::string & key);
bool require_bool(const FlattenedDocument & document, const std::string & key);
bool optional_bool(const FlattenedDocument & document, const std::string & key, bool default_value);
std::optional<int> optional_int(const FlattenedDocument & document, const std::string & key);
std::optional<float> optional_float(const FlattenedDocument & document, const std::string & key);
float optional_f32(const FlattenedDocument & document, const std::string & key, float default_value);
std::optional<float> optional_nullable_f32(const FlattenedDocument & document, const std::string & key);
std::vector<std::string> require_list_strings(const FlattenedDocument & document, const std::string & key);
std::vector<int> require_list_int(const FlattenedDocument & document, const std::string & key);
std::vector<int64_t> require_list_i64(const FlattenedDocument & document, const std::string & key);

}  // namespace engine::io::yaml
