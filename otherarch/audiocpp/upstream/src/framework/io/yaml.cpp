#include "engine/framework/io/yaml.h"

#include <yaml.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::io::yaml {
namespace {

std::string parser_error_message(const yaml_parser_t & parser) {
    std::ostringstream message;
    message << "yaml parse error";
    if (parser.problem != nullptr) {
        message << ": " << parser.problem;
    }
    if (parser.problem_mark.line != 0 || parser.problem_mark.column != 0 || parser.problem_mark.index != 0) {
        message << " at line " << (parser.problem_mark.line + 1)
                << ", column " << (parser.problem_mark.column + 1);
    }
    return message.str();
}

const yaml_node_t & require_node(yaml_document_t & document, int index, const std::string & context) {
    yaml_node_t * node = yaml_document_get_node(&document, index);
    if (node == nullptr) {
        throw std::runtime_error("yaml document is missing node for " + context);
    }
    return *node;
}

std::string require_scalar_value(yaml_document_t & document, int index, const std::string & context) {
    const yaml_node_t & node = require_node(document, index, context);
    if (node.type != YAML_SCALAR_NODE) {
        throw std::runtime_error("yaml " + context + " must be a scalar");
    }
    return std::string(reinterpret_cast<const char *>(node.data.scalar.value), node.data.scalar.length);
}

std::string join_path(std::string_view prefix, std::string_view key) {
    if (prefix.empty()) {
        return std::string(key);
    }
    std::string path(prefix);
    path += '.';
    path += key;
    return path;
}

void flatten_node(
    yaml_document_t & document,
    int index,
    std::string_view path,
    FlattenedDocument & flattened) {
    const std::string context(path);
    const yaml_node_t & node = require_node(document, index, context);
    switch (node.type) {
        case YAML_SCALAR_NODE: {
            if (path.empty()) {
                throw std::runtime_error("yaml document root must be a mapping");
            }
            flattened.scalars[std::string(path)] = std::string(
                reinterpret_cast<const char *>(node.data.scalar.value),
                node.data.scalar.length);
            return;
        }
        case YAML_MAPPING_NODE: {
            for (yaml_node_pair_t * pair = node.data.mapping.pairs.start;
                 pair != node.data.mapping.pairs.top;
                 ++pair) {
                const std::string key = require_scalar_value(document, pair->key, "mapping key");
                if (key.empty()) {
                    continue;
                }
                flatten_node(document, pair->value, join_path(path, key), flattened);
            }
            return;
        }
        case YAML_SEQUENCE_NODE: {
            if (path.empty()) {
                throw std::runtime_error("yaml document root sequence is not supported");
            }
            auto & values = flattened.lists[std::string(path)];
            for (yaml_node_item_t * item = node.data.sequence.items.start;
                 item != node.data.sequence.items.top;
                 ++item) {
                values.push_back(require_scalar_value(document, *item, "sequence item for key '" + std::string(path) + "'"));
            }
            return;
        }
        case YAML_NO_NODE:
            throw std::runtime_error("yaml document is missing node data");
        default:
            throw std::runtime_error("yaml alias nodes are not supported");
    }
}

class Parser {
public:
    explicit Parser(std::string_view text) {
        if (!yaml_parser_initialize(&parser_)) {
            throw std::runtime_error("failed to initialize yaml parser");
        }
        initialized_ = true;
        yaml_parser_set_input_string(
            &parser_,
            reinterpret_cast<const unsigned char *>(text.data()),
            text.size());
    }

    ~Parser() {
        if (primary_loaded_) {
            yaml_document_delete(&primary_document_);
        }
        if (initialized_) {
            yaml_parser_delete(&parser_);
        }
    }

    yaml_document_t & load_primary() {
        load_document(primary_document_, primary_loaded_);
        return primary_document_;
    }

private:
    void load_document(yaml_document_t & document, bool & loaded_flag) {
        if (loaded_flag) {
            return;
        }
        if (!yaml_parser_load(&parser_, &document)) {
            throw std::runtime_error(parser_error_message(parser_));
        }
        loaded_flag = true;
    }

    yaml_parser_t parser_{};
    bool initialized_ = false;
    yaml_document_t primary_document_{};
    bool primary_loaded_ = false;
};

class EventParser {
public:
    explicit EventParser(std::string_view text) {
        if (!yaml_parser_initialize(&parser_)) {
            throw std::runtime_error("failed to initialize yaml parser");
        }
        initialized_ = true;
        yaml_parser_set_input_string(
            &parser_,
            reinterpret_cast<const unsigned char *>(text.data()),
            text.size());
    }

    ~EventParser() {
        if (initialized_) {
            yaml_parser_delete(&parser_);
        }
    }

    yaml_parser_t & parser() noexcept {
        return parser_;
    }

private:
    yaml_parser_t parser_{};
    bool initialized_ = false;
};

void require_single_document(std::string_view text) {
    EventParser parser(text);
    int documents = 0;
    bool done = false;
    while (!done) {
        yaml_event_t event{};
        if (!yaml_parser_parse(&parser.parser(), &event)) {
            throw std::runtime_error(parser_error_message(parser.parser()));
        }
        const auto event_type = event.type;
        yaml_event_delete(&event);
        if (event_type == YAML_DOCUMENT_START_EVENT) {
            ++documents;
            if (documents > 1) {
                throw std::runtime_error("yaml config must contain exactly one document");
            }
        } else if (event_type == YAML_STREAM_END_EVENT) {
            done = true;
        }
    }
}

}  // namespace

ConfigMap parse_scalar_mapping(std::string_view text) {
    require_single_document(text);
    Parser parser(text);
    yaml_document_t & document = parser.load_primary();

    yaml_node_t * root = yaml_document_get_root_node(&document);
    if (root == nullptr) {
        return {};
    }
    if (root->type != YAML_MAPPING_NODE) {
        throw std::runtime_error("yaml config root must be a mapping");
    }

    ConfigMap config;
    const auto pair_count = static_cast<size_t>(root->data.mapping.pairs.top - root->data.mapping.pairs.start);
    config.reserve(pair_count);

    for (yaml_node_pair_t * pair = root->data.mapping.pairs.start;
         pair != root->data.mapping.pairs.top;
         ++pair) {
        const std::string key = require_scalar_value(document, pair->key, "mapping key");
        const std::string value = require_scalar_value(document, pair->value, "mapping value for key '" + key + "'");
        if (!key.empty()) {
            config[key] = value;
        }
    }

    return config;
}

FlattenedDocument parse_flattened_document(std::string_view text) {
    require_single_document(text);
    Parser parser(text);
    yaml_document_t & document = parser.load_primary();

    yaml_node_t * root = yaml_document_get_root_node(&document);
    if (root == nullptr) {
        return {};
    }
    if (root->type != YAML_MAPPING_NODE) {
        throw std::runtime_error("yaml config root must be a mapping");
    }

    FlattenedDocument flattened;
    for (yaml_node_pair_t * pair = root->data.mapping.pairs.start;
         pair != root->data.mapping.pairs.top;
         ++pair) {
        const std::string key = require_scalar_value(document, pair->key, "mapping key");
        if (key.empty()) {
            continue;
        }
        flatten_node(document, pair->value, key, flattened);
    }
    return flattened;
}

bool parse_bool_scalar(const std::string & value, const std::string & key) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error("yaml boolean key has invalid value: " + key);
}

std::string require_string(const FlattenedDocument & document, const std::string & key) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end()) {
        throw std::runtime_error("Missing yaml scalar key: " + key);
    }
    return it->second;
}

std::string optional_string(const FlattenedDocument & document, const std::string & key, std::string default_value) {
    const auto it = document.scalars.find(key);
    return it == document.scalars.end() ? std::move(default_value) : it->second;
}

int require_int(const FlattenedDocument & document, const std::string & key) {
    return std::stoi(require_string(document, key));
}

int64_t require_i64(const FlattenedDocument & document, const std::string & key) {
    return std::stoll(require_string(document, key));
}

float require_float(const FlattenedDocument & document, const std::string & key) {
    return std::stof(require_string(document, key));
}

bool require_bool(const FlattenedDocument & document, const std::string & key) {
    return parse_bool_scalar(require_string(document, key), key);
}

bool optional_bool(const FlattenedDocument & document, const std::string & key, bool default_value) {
    const auto it = document.scalars.find(key);
    return it == document.scalars.end() ? default_value : parse_bool_scalar(it->second, key);
}

std::optional<int> optional_int(const FlattenedDocument & document, const std::string & key) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end()) {
        return std::nullopt;
    }
    return std::stoi(it->second);
}

std::optional<float> optional_float(const FlattenedDocument & document, const std::string & key) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end()) {
        return std::nullopt;
    }
    return std::stof(it->second);
}

float optional_f32(const FlattenedDocument & document, const std::string & key, float default_value) {
    const auto value = optional_float(document, key);
    return value.value_or(default_value);
}

std::optional<float> optional_nullable_f32(const FlattenedDocument & document, const std::string & key) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end() || it->second == "None" || it->second == "null") {
        return std::nullopt;
    }
    return std::stof(it->second);
}

std::vector<std::string> require_list_strings(const FlattenedDocument & document, const std::string & key) {
    const auto it = document.lists.find(key);
    if (it == document.lists.end() || it->second.empty()) {
        throw std::runtime_error("Missing yaml list key: " + key);
    }
    return it->second;
}

std::vector<int> require_list_int(const FlattenedDocument & document, const std::string & key) {
    const auto values = require_list_strings(document, key);
    std::vector<int> out;
    out.reserve(values.size());
    for (const auto & value : values) {
        out.push_back(std::stoi(value));
    }
    return out;
}

std::vector<int64_t> require_list_i64(const FlattenedDocument & document, const std::string & key) {
    const auto values = require_list_strings(document, key);
    std::vector<int64_t> out;
    out.reserve(values.size());
    for (const auto & value : values) {
        out.push_back(std::stoll(value));
    }
    return out;
}

}  // namespace engine::io::yaml
