#include "engine/framework/model_spec/schema.h"
#include "engine/framework/runtime/task_vocabulary.h"
#include "engine/framework/model_spec/options.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace engine::model_spec {
namespace {

namespace json = engine::io::json;

[[noreturn]] void fail(std::string_view path, const std::string & message) {
    throw std::runtime_error(std::string(path) + ": " + message);
}

const json::Value::Object & require_spec_object(const json::Value & value, std::string_view path) {
    if (!value.is_object()) {
        fail(path, "expected object");
    }
    return value.as_object();
}

const json::Value::Array & require_spec_array(const json::Value & value, std::string_view path) {
    if (!value.is_array()) {
        fail(path, "expected array");
    }
    return value.as_array();
}

const json::Value & require_spec_field(const json::Value & object, std::string_view key, std::string_view path) {
    const std::string key_string(key);
    const auto * value = object.find(key_string);
    if (value == nullptr) {
        fail(path, "missing required field '" + key_string + "'");
    }
    return *value;
}

bool has_spec_field(const json::Value & object, std::string_view key) {
    return object.find(std::string(key)) != nullptr;
}

std::string require_spec_string(const json::Value & value, std::string_view path) {
    if (!value.is_string() || value.as_string().empty()) {
        fail(path, "expected non-empty string");
    }
    return value.as_string();
}

bool require_spec_bool(const json::Value & value, std::string_view path) {
    if (!value.is_bool()) {
        fail(path, "expected bool");
    }
    return value.as_bool();
}

void require_spec_number(const json::Value & value, std::string_view path) {
    if (!value.is_number()) {
        fail(path, "expected number");
    }
}

const std::unordered_set<std::string> & tasks() {
    // Built from the one vocabulary rather than typed out again. The hand-kept
    // copy had drifted both ways: it allowed "codec", which no parser maps to a
    // task kind, and omitted "audio_generation", which the parser accepts.
    static const std::unordered_set<std::string> values = [] {
        std::unordered_set<std::string> names;
        std::size_t count = 0;
        const auto * entries = engine::runtime::task_vocabulary(count);
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t alias = 0; alias < entries[i].alias_count; ++alias) {
                names.emplace(entries[i].aliases[alias]);
            }
        }
        return names;
    }();
    return values;
}

const std::unordered_set<std::string> & modes() {
    static const std::unordered_set<std::string> values = {"offline", "streaming"};
    return values;
}

const std::unordered_set<std::string> & categories() {
    static const std::unordered_set<std::string> values = {
        "asr", "tts", "voice_conversion", "audio_generation", "audio_tools", "speech_analysis", "community",
    };
    return values;
}

const std::unordered_set<std::string> & statuses() {
    static const std::unordered_set<std::string> values = {"supported", "community", "experimental", "wip", "unsupported"};
    return values;
}

const std::unordered_set<std::string> & source_formats() {
    // `rknn` is the compiled RKNPU2 engine image: a per-SoC, per-stage artefact the RKNN sidecar
    // paths load next to the GGUF weights they accelerate (docs/build/rockchip-npu.md).
    static const std::unordered_set<std::string> values = {"safetensors", "gguf", "onnx", "ggml", "pytorch", "rknn"};
    return values;
}

const std::unordered_set<std::string> & precisions() {
    static const std::unordered_set<std::string> values = {
        "native", "orig", "f32", "f16", "bf16",
        "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_k", "q4_0", "q3_k_m", "q3_k_s", "q2_k",
        "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs",
    };
    return values;
}

const std::unordered_set<std::string> & download_kinds() {
    static const std::unordered_set<std::string> values = {
        "huggingface_snapshot", "modelscope_snapshot", "local_snapshot", "converter", "unsupported",
    };
    return values;
}

const std::unordered_set<std::string> & dependency_kinds() {
    static const std::unordered_set<std::string> values = {"model", "bundled_model"};
    return values;
}

const std::unordered_set<std::string> & dependency_scopes() {
    static const std::unordered_set<std::string> values = {"load", "session", "request"};
    return values;
}

const std::unordered_set<std::string> & option_types() {
    static const std::unordered_set<std::string> values = {
        "bool", "int", "float", "string", "enum", "path", "audio_path",
        "string_list", "float_list", "path_list", "audio_path_list",
    };
    return values;
}

bool is_numeric_option_type(const std::string & type) {
    return type == "int" || type == "float" || type == "float_list";
}

const std::unordered_set<std::string> & runtime_tags() {
    static const std::unordered_set<std::string> values = {"gguf", "stream", "server", "cuda", "vulkan", "metal", "cpu"};
    return values;
}

const std::unordered_set<std::string> & ui_tags() {
    static const std::unordered_set<std::string> values = {
        "ASR", "TTS", "Clone", "VC", "Align", "VAD", "Diar", "Codec", "Sep", "Music", "SFX",
        "Edit", "Design", "MIDI", "Wake Word", "Classify", "GGUF", "Stream",
    };
    return values;
}

const std::unordered_set<std::string> & capabilities_for_task(const std::string & task) {
    static const std::unordered_set<std::string> empty;
    static const std::unordered_map<std::string, std::unordered_set<std::string>> values = {
        {"asr", {"word_timestamps", "segments", "speaker_turns", "vad_chunking", "partial_results"}},
        {"tts", {"speaker_reference", "voice_design", "emotion_control", "style_control",
                 "multi_speaker", "long_form", "built_in_voices"}},
        {"clone", {"speaker_reference", "emotion_control", "style_control", "multi_speaker", "long_form"}},
        {"vc", {"speaker_reference", "singing"}},
        {"s2s", {"speaker_reference", "audio_enhancement"}},
        {"svc", {"speaker_reference", "singing"}},
        {"align", {"word_timestamps"}},
        {"vad", {"speech_segments", "chunk_planning"}},
        {"wake_word", {"speech_segments", "stream_events"}},
        {"classify", {"scores"}},
        {"turn", {"turn_decision"}},
        {"diar", {"speaker_turns"}},
        {"sep", {"stems"}},
        {"midi", {"note_events", "midi_artifact"}},
        {"music", {"lyrics", "instrumental", "continuation", "style_control"}},
        {"sfx", {"prompt_generation"}},
        {"edit", {"prompt_editing", "inpaint"}},
        {"design", {"voice_design"}},
        {"codec", {"encode_decode"}},
    };
    const auto it = values.find(task);
    return it == values.end() ? empty : it->second;
}

void validate_enum(const std::string & value,
                   const std::unordered_set<std::string> & allowed,
                   std::string_view path,
                   std::string_view label) {
    if (allowed.find(value) == allowed.end()) {
        fail(path, "unknown " + std::string(label) + " '" + value + "'");
    }
}

void validate_string_array(const json::Value & value,
                           const std::unordered_set<std::string> * allowed,
                           std::string_view path,
                           std::string_view label) {
    const auto & array = require_spec_array(value, path);
    for (size_t index = 0; index < array.size(); ++index) {
        const auto item_path = std::string(path) + "[" + std::to_string(index) + "]";
        const auto value_string = require_spec_string(array[index], item_path);
        if (allowed != nullptr) {
            validate_enum(value_string, *allowed, item_path, label);
        }
    }
}

std::unordered_set<std::string> validate_nonempty_string_set(
    const json::Value & value,
    const std::unordered_set<std::string> * allowed,
    std::string_view path,
    std::string_view label) {
    validate_string_array(value, allowed, path, label);
    const auto & array = value.as_array();
    if (array.empty()) {
        fail(path, std::string(label) + " array must not be empty");
    }
    std::unordered_set<std::string> values;
    for (size_t index = 0; index < array.size(); ++index) {
        const auto item = array[index].as_string();
        if (!values.insert(item).second) {
            fail(std::string(path) + "[" + std::to_string(index) + "]", "duplicate " + std::string(label) + " '" + item + "'");
        }
    }
    return values;
}

void validate_request_option_name(const std::string & name, const std::string & family, std::string_view path) {
    if (name.find('.') == std::string::npos) {
        return;
    }
    const std::string prefix = family + ".";
    if (name.rfind(prefix, 0) == 0 && name.size() > prefix.size()) {
        return;
    }
    fail(path, "dotted request option '" + name + "' must use namespace " + family + ".<name>");
}

void validate_local_option_name(
    const std::string & name,
    const std::string & family,
    std::string_view scope,
    std::string_view path) {
    if (name.find('.') != std::string::npos) {
        fail(path, std::string(scope) + " option must be local; the public key is derived as " + family + ".<name>");
    }
}

void validate_option(
    const json::Value & value,
    const std::string & family,
    std::string_view scope,
    std::string_view path) {
    require_spec_object(value, path);
    const auto name = require_spec_string(require_spec_field(value, "name", path), std::string(path) + ".name");
    if (scope == "request") {
        validate_request_option_name(name, family, std::string(path) + ".name");
    } else {
        validate_local_option_name(name, family, scope, std::string(path) + ".name");
    }
    const auto type = require_spec_string(require_spec_field(value, "type", path), std::string(path) + ".type");
    validate_enum(type, option_types(), std::string(path) + ".type", "option type");
    if (scope == "session" && name == "exec_plan" && type != "string") {
        fail(std::string(path) + ".type", "exec_plan must be a JSON object encoded as a string");
    }
    if (const auto it = shared_option_contracts().find(name); it != shared_option_contracts().end()) {
        if (it->second.find(type) == it->second.end()) {
            fail(std::string(path) + ".type", "shared option '" + name + "' must use its registered type");
        }
    }
    const auto * preset = value.find("preset");
    const auto * values = value.find("values");
    if (preset != nullptr) {
        const auto preset_name = require_spec_string(*preset, std::string(path) + ".preset");
        if (type != "enum") {
            fail(std::string(path) + ".preset", "presets are allowed only for enum options");
        }
        if (option_presets().find(preset_name) == option_presets().end()) {
            fail(std::string(path) + ".preset", "unknown option preset '" + preset_name + "'");
        }
        if (values != nullptr) {
            fail(std::string(path) + ".values", "enum options must use either preset or values, not both");
        }
    }
    if (type == "enum") {
        if (preset == nullptr && values == nullptr) {
            fail(std::string(path) + ".values", "enum option requires values or preset");
        }
        if (values != nullptr) {
            validate_string_array(*values, nullptr, std::string(path) + ".values", "enum value");
            if (values->as_array().empty()) {
                fail(std::string(path) + ".values", "enum option values must not be empty");
            }
        }
    } else if (values != nullptr) {
        fail(std::string(path) + ".values", "values are allowed only for enum options");
    }
    (void) require_spec_bool(require_spec_field(value, "required", path), std::string(path) + ".required");
    const auto * default_value = value.find("default");
    if (default_value != nullptr) {
        const auto default_path = std::string(path) + ".default";
        if (default_value->is_null()) {
            fail(default_path, "use absent default instead of null");
        }
        if (type == "bool" && !default_value->is_bool()) {
            fail(default_path, "expected bool default");
        }
        if ((type == "int" || type == "float") && !default_value->is_number()) {
            fail(default_path, "expected numeric default");
        }
        if ((type == "string" || type == "path" || type == "audio_path" || type == "enum") &&
            !default_value->is_string()) {
            fail(default_path, "expected string default");
        }
        if ((type == "string_list" || type == "float_list" || type == "path_list" || type == "audio_path_list") &&
            !default_value->is_array()) {
            fail(default_path, "expected array default");
        }
        if (type == "enum") {
            if (preset != nullptr) {
                const auto & allowed = require_option_preset(preset->as_string());
                if (std::find(allowed.begin(), allowed.end(), default_value->as_string()) == allowed.end()) {
                    fail(default_path, "default is not in preset values");
                }
            } else if (values != nullptr) {
                bool found = false;
                for (const auto & item : values->as_array()) {
                    found = found || item.as_string() == default_value->as_string();
                }
                if (!found) {
                    fail(default_path, "default is not in enum values");
                }
            }
        }
    }
    const auto * min_value = value.find("min");
    const auto * max_value = value.find("max");
    if ((min_value != nullptr || max_value != nullptr) && !is_numeric_option_type(type)) {
        fail(std::string(path), "min/max are allowed only for int or float options");
    }
    if (min_value != nullptr && !min_value->is_number()) {
        fail(std::string(path) + ".min", "expected numeric min");
    }
    if (max_value != nullptr && !max_value->is_number()) {
        fail(std::string(path) + ".max", "expected numeric max");
    }
    if (min_value != nullptr && max_value != nullptr && min_value->as_number() > max_value->as_number()) {
        fail(std::string(path) + ".min", "min must not exceed max");
    }
    if (default_value != nullptr && default_value->is_number()) {
        if (min_value != nullptr && default_value->as_number() < min_value->as_number()) {
            fail(std::string(path) + ".default", "default is below min");
        }
        if (max_value != nullptr && default_value->as_number() > max_value->as_number()) {
            fail(std::string(path) + ".default", "default exceeds max");
        }
    }
    (void) require_spec_string(require_spec_field(value, "description", path), std::string(path) + ".description");
}

struct DeclaredOptions {
    // Local option names keyed by scope (dependencies[].option looks here).
    std::unordered_map<std::string, std::unordered_set<std::string>> local_by_scope;
    // Public option keys keyed by scope (required_when.option_key looks here).
    std::unordered_map<std::string, std::unordered_set<std::string>> public_by_scope;
};

DeclaredOptions collect_declared_options(const json::Value & value, const std::string & family, std::string_view path) {
    DeclaredOptions declared;
    require_spec_object(value, path);
    for (const std::string scope : {"request", "session", "load"}) {
        const auto child_path = std::string(path) + "." + scope;
        const auto & rows = require_spec_array(require_spec_field(value, scope, path), child_path);
        auto & locals = declared.local_by_scope[scope];
        auto & publics = declared.public_by_scope[scope];
        for (size_t index = 0; index < rows.size(); ++index) {
            const auto & row = rows[index];
            require_spec_object(row, child_path + "[" + std::to_string(index) + "]");
            const auto name = require_spec_string(
                require_spec_field(row, "name", child_path + "[" + std::to_string(index) + "]"),
                child_path + "[" + std::to_string(index) + "].name");
            if (scope == "request") {
                locals.insert(name);
                publics.insert(name);
            } else {
                locals.insert(name);
                publics.insert(family + "." + name);
            }
        }
    }
    return declared;
}

void validate_options(const json::Value & value, const std::string & family, std::string_view path) {
    require_spec_object(value, path);
    for (const std::string scope : {"request", "session", "load"}) {
        const auto child_path = std::string(path) + "." + scope;
        const auto & rows = require_spec_array(require_spec_field(value, scope, path), child_path);
        for (size_t index = 0; index < rows.size(); ++index) {
            validate_option(rows[index], family, scope, child_path + "[" + std::to_string(index) + "]");
        }
    }
}

void validate_capabilities(const json::Value & value,
                           const std::unordered_set<std::string> & task_ids,
                           std::string_view path) {
    const auto & capabilities = require_spec_object(value, path);
    for (const auto & [task, rows] : capabilities) {
        const auto task_path = std::string(path) + "." + task;
        if (task_ids.find(task) == task_ids.end()) {
            fail(task_path, "capability key must be one of this model's tasks");
        }
        const auto & allowed = capabilities_for_task(task);
        if (allowed.empty()) {
            fail(task_path, "task does not define typed capabilities");
        }
        validate_nonempty_string_set(rows, &allowed, task_path, "capability");
    }
}

void validate_runtime(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    validate_string_array(require_spec_field(value, "tags", path), &runtime_tags(), std::string(path) + ".tags", "runtime tag");
}

void validate_snapshot_download(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    (void) require_spec_string(require_spec_field(value, "repo", path), std::string(path) + ".repo");
    if (const auto * revision = value.find("revision")) {
        (void) require_spec_string(*revision, std::string(path) + ".revision");
    }
    for (const std::string key : {"include", "exclude"}) {
        if (const auto * array = value.find(key)) {
            validate_string_array(*array, nullptr, std::string(path) + "." + key, key);
        }
    }
    if (const auto * strip_prefix = value.find("strip_prefix")) {
        (void) require_spec_string(*strip_prefix, std::string(path) + ".strip_prefix");
    }
    if (const auto * gated = value.find("gated")) {
        (void) require_spec_bool(*gated, std::string(path) + ".gated");
    }
}

void validate_download(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    const auto kind = require_spec_string(require_spec_field(value, "kind", path), std::string(path) + ".kind");
    validate_enum(kind, download_kinds(), std::string(path) + ".kind", "download kind");
    if (kind == "huggingface_snapshot" || kind == "modelscope_snapshot") {
        validate_snapshot_download(value, path);
    } else if (kind == "local_snapshot") {
        (void) require_spec_string(require_spec_field(value, "path", path), std::string(path) + ".path");
        if (const auto * array = value.find("include")) {
            validate_string_array(*array, nullptr, std::string(path) + ".include", "include");
        }
    } else if (kind == "converter") {
        (void) require_spec_string(require_spec_field(value, "converter", path), std::string(path) + ".converter");
        (void) require_spec_string(require_spec_field(value, "description", path), std::string(path) + ".description");
    } else if (kind == "unsupported") {
        (void) require_spec_string(require_spec_field(value, "reason", path), std::string(path) + ".reason");
    }
}

void validate_ref(const json::Value & value, const json::Value::Object & roots, std::string_view path) {
    std::string root;
    if (value.is_string()) {
        const auto ref = value.as_string();
        const auto split = ref.find(':');
        if (split == std::string::npos || split == 0) {
            fail(path, "invalid resource reference '" + ref + "'");
        }
        root = ref.substr(0, split);
    } else {
        require_spec_object(value, path);
        const auto source = require_spec_string(require_spec_field(value, "source", path), std::string(path) + ".source");
        const auto split = source.find(':');
        if (split == std::string::npos || split == 0) {
            fail(std::string(path) + ".source", "invalid resource reference '" + source + "'");
        }
        root = source.substr(0, split);
        if (const auto * prefix = value.find("prefix")) {
            (void) require_spec_string(*prefix, std::string(path) + ".prefix");
        }
    }
    if (roots.find(root) == roots.end()) {
        fail(path, "unknown root '" + root + "'");
    }
}

void validate_layout(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    const auto format = require_spec_string(require_spec_field(value, "format", path), std::string(path) + ".format");
    validate_enum(format, source_formats(), std::string(path) + ".format", "format");
    const auto roots_path = std::string(path) + ".roots";
    const auto & roots_field = require_spec_field(value, "roots", path);
    const auto & roots = require_spec_object(roots_field, roots_path);
    for (const auto & [root_id, root_value] : roots) {
        if (root_id.empty()) {
            fail(std::string(path) + ".roots", "root id must not be empty");
        }
        (void) require_spec_string(root_value, std::string(path) + ".roots." + root_id);
    }
    for (const std::string map_name : {"files", "optional_files", "tensors", "optional_tensors"}) {
        const auto * map_value = value.find(map_name);
        if (map_value == nullptr) {
            continue;
        }
        const auto & map = require_spec_object(*map_value, std::string(path) + "." + map_name);
        for (const auto & [id, ref] : map) {
            if (id.empty()) {
                fail(std::string(path) + "." + map_name, "resource id must not be empty");
            }
            validate_ref(ref, roots, std::string(path) + "." + map_name + "." + id);
        }
    }
}

void validate_package_defaults(const json::Value & value, std::string_view path) {
    require_spec_object(value, path);
    if (const auto * download = value.find("download")) {
        validate_download(*download, std::string(path) + ".download");
    }
}

struct ValidatedPackage {
    std::string id;
    bool is_default = false;
};

ValidatedPackage validate_package(const json::Value & value, std::string_view path, bool has_default_download) {
    require_spec_object(value, path);
    ValidatedPackage result;
    result.id = require_spec_string(require_spec_field(value, "id", path), std::string(path) + ".id");
    (void) require_spec_string(require_spec_field(value, "display_name", path), std::string(path) + ".display_name");
    const auto format = require_spec_string(require_spec_field(value, "format", path), std::string(path) + ".format");
    validate_enum(format, source_formats(), std::string(path) + ".format", "format");
    const auto precision = require_spec_string(require_spec_field(value, "precision", path), std::string(path) + ".precision");
    validate_enum(precision, precisions(), std::string(path) + ".precision", "precision");
    (void) require_spec_string(require_spec_field(value, "target_directory", path), std::string(path) + ".target_directory");
    validate_string_array(require_spec_field(value, "files", path), nullptr, std::string(path) + ".files", "file");
    if (value.require("files").as_array().empty()) {
        fail(std::string(path) + ".files", "files must not be empty");
    }
    if (const auto * default_package = value.find("default")) {
        result.is_default = require_spec_bool(*default_package, std::string(path) + ".default");
    }
    if (const auto * download = value.find("download")) {
        validate_download(*download, std::string(path) + ".download");
    } else if (!has_default_download) {
        fail(std::string(path) + ".download", "missing package download and package_defaults.download");
    }
    if (const auto * strip_prefix = value.find("strip_prefix")) {
        (void) require_spec_string(*strip_prefix, std::string(path) + ".strip_prefix");
    }
    if (const auto * description = value.find("description")) {
        (void) require_spec_string(*description, std::string(path) + ".description");
    }
    return result;
}

std::unordered_set<std::string> validate_packages(
    const json::Value & value,
    std::string_view path,
    bool has_default_download,
    bool allow_empty) {
    const auto & packages = require_spec_array(value, path);
    if (packages.empty()) {
        if (allow_empty) {
            return {};
        }
        fail(path, "packages must not be empty unless status is experimental");
    }
    bool has_default = false;
    std::unordered_set<std::string> package_ids;
    for (size_t index = 0; index < packages.size(); ++index) {
        const auto package_path = std::string(path) + "[" + std::to_string(index) + "]";
        const auto package = validate_package(packages[index], package_path, has_default_download);
        if (!package_ids.insert(package.id).second) {
            fail(package_path + ".id", "duplicate package id '" + package.id + "'");
        }
        if (package.is_default) {
            if (has_default) {
                fail(package_path + ".default", "only one package can be default");
            }
            has_default = true;
        }
    }
    if (!has_default) {
        fail(path, "one package must be marked default");
    }
    return package_ids;
}

void validate_dependencies(
    const json::Value & value,
    const std::string & family,
    const DeclaredOptions & declared,
    std::string_view path) {
    const auto & dependencies = require_spec_array(value, path);
    for (size_t index = 0; index < dependencies.size(); ++index) {
        const auto item_path = std::string(path) + "[" + std::to_string(index) + "]";
        const auto & dependency = dependencies[index];
        require_spec_object(dependency, item_path);
        const auto kind = require_spec_string(require_spec_field(dependency, "kind", item_path), item_path + ".kind");
        validate_enum(kind, dependency_kinds(), item_path + ".kind", "dependency kind");
        (void) require_spec_string(require_spec_field(dependency, "family", item_path), item_path + ".family");
        const auto scope = require_spec_string(require_spec_field(dependency, "scope", item_path), item_path + ".scope");
        validate_enum(scope, dependency_scopes(), item_path + ".scope", "dependency scope");
        const auto option = require_spec_string(require_spec_field(dependency, "option", item_path), item_path + ".option");
        if (option.find('.') != std::string::npos) {
            fail(item_path + ".option", "dependency option must be local; the public key is derived as " + family + ".<option>");
        }
        validate_request_option_name(family + "." + option, family, item_path + ".option");
        const auto local_it = declared.local_by_scope.find(scope);
        if (local_it == declared.local_by_scope.end() || local_it->second.find(option) == local_it->second.end()) {
            fail(item_path + ".option",
                 "dependency option '" + option + "' is not declared in options." + scope);
        }
        const auto required = require_spec_bool(require_spec_field(dependency, "required", item_path), item_path + ".required");
        if (dependency.find("required_for") != nullptr) {
            fail(item_path + ".required_for", "use typed required_when conditions");
        }
        if (dependency.find("package") != nullptr) {
            fail(item_path + ".package", "dependency package hints are not part of the core contract");
        }
        if (const auto * required_when = dependency.find("required_when")) {
            if (required) {
                fail(item_path + ".required_when", "required dependencies must not have conditions");
            }
            const auto & conditions = require_spec_array(*required_when, item_path + ".required_when");
            if (conditions.empty()) {
                fail(item_path + ".required_when", "required_when must not be empty");
            }
            for (size_t condition_index = 0; condition_index < conditions.size(); ++condition_index) {
                const auto condition_path =
                    item_path + ".required_when[" + std::to_string(condition_index) + "]";
                const auto & condition = conditions[condition_index];
                require_spec_object(condition, condition_path);
                const auto condition_scope = require_spec_string(
                    require_spec_field(condition, "scope", condition_path),
                    condition_path + ".scope");
                validate_enum(condition_scope, dependency_scopes(), condition_path + ".scope", "dependency scope");
                const auto option_key = require_spec_string(
                    require_spec_field(condition, "option_key", condition_path),
                    condition_path + ".option_key");
                validate_request_option_name(option_key, family, condition_path + ".option_key");
                const auto public_it = declared.public_by_scope.find(condition_scope);
                if (public_it == declared.public_by_scope.end() ||
                    public_it->second.find(option_key) == public_it->second.end()) {
                    fail(condition_path + ".option_key",
                         "required_when option_key '" + option_key +
                             "' is not declared in options." + condition_scope);
                }
                const auto & equals = require_spec_field(condition, "equals", condition_path);
                if (!equals.is_bool() && !equals.is_number() && !equals.is_string()) {
                    fail(condition_path + ".equals", "expected bool, number, or string");
                }
                if (equals.is_string() && equals.as_string().empty()) {
                    fail(condition_path + ".equals", "expected non-empty string");
                }
            }
        } else if (!required) {
            fail(item_path + ".required_when", "optional dependencies require typed conditions");
        }
        if (kind == "bundled_model") {
            (void) require_spec_string(require_spec_field(dependency, "path", item_path), item_path + ".path");
        }
    }
}

void validate_ui(const json::Value & value, const std::unordered_set<std::string> & package_ids, std::string_view path) {
    require_spec_object(value, path);
    if (const auto * recommended_value = value.find("recommended_package")) {
        const auto recommended = require_spec_string(
            *recommended_value, std::string(path) + ".recommended_package");
        if (package_ids.find(recommended) == package_ids.end()) {
            fail(std::string(path) + ".recommended_package", "unknown package '" + recommended + "'");
        }
    } else if (!package_ids.empty()) {
        fail(std::string(path) + ".recommended_package", "missing required field");
    }
    if (const auto * min_vram = value.find("min_vram_gb")) {
        require_spec_number(*min_vram, std::string(path) + ".min_vram_gb");
    }
    validate_string_array(require_spec_field(value, "tags", path), &ui_tags(), std::string(path) + ".tags", "UI tag");
    validate_string_array(require_spec_field(value, "docs", path), nullptr, std::string(path) + ".docs", "doc path");
    if (const auto * summary = value.find("summary")) {
        (void) require_spec_string(*summary, std::string(path) + ".summary");
    }
}

void validate_legacy_source(const json::Value & value, std::string_view path) {
    validate_layout(value, path);
}

void validate_v1(const json::Value & spec, std::string_view source_name) {
    const auto family = require_spec_string(require_spec_field(spec, "family", source_name), std::string(source_name) + ".family");
    (void) require_spec_string(require_spec_field(spec, "display_name", source_name), std::string(source_name) + ".display_name");
    validate_enum(require_spec_string(require_spec_field(spec, "category", source_name), std::string(source_name) + ".category"),
                  categories(), std::string(source_name) + ".category", "category");
    const auto status = require_spec_string(
        require_spec_field(spec, "status", source_name), std::string(source_name) + ".status");
    validate_enum(status, statuses(), std::string(source_name) + ".status", "status");
    const auto task_ids = validate_nonempty_string_set(
        require_spec_field(spec, "tasks", source_name), &tasks(), std::string(source_name) + ".tasks", "task");
    validate_nonempty_string_set(
        require_spec_field(spec, "modes", source_name), &modes(), std::string(source_name) + ".modes", "mode");
    validate_nonempty_string_set(
        require_spec_field(spec, "languages", source_name), nullptr, std::string(source_name) + ".languages", "language");
    validate_runtime(require_spec_field(spec, "runtime", source_name), std::string(source_name) + ".runtime");
    validate_capabilities(require_spec_field(spec, "capabilities", source_name), task_ids, std::string(source_name) + ".capabilities");
    const auto & options_field = require_spec_field(spec, "options", source_name);
    validate_options(options_field, family, std::string(source_name) + ".options");
    const auto declared_options =
        collect_declared_options(options_field, family, std::string(source_name) + ".options");

    const bool has_default_download =
        has_spec_field(spec, "package_defaults") && has_spec_field(*spec.find("package_defaults"), "download");
    if (const auto * package_defaults = spec.find("package_defaults")) {
        validate_package_defaults(*package_defaults, std::string(source_name) + ".package_defaults");
    }

    const auto packages_path = std::string(source_name) + ".packages";
    const auto & packages_field = require_spec_field(spec, "packages", source_name);
    const auto package_ids = validate_packages(
        packages_field, packages_path, has_default_download, status == "experimental");
    validate_dependencies(
        require_spec_field(spec, "dependencies", source_name),
        family,
        declared_options,
        std::string(source_name) + ".dependencies");
    validate_ui(require_spec_field(spec, "ui", source_name), package_ids, std::string(source_name) + ".ui");

    const auto sources_path = std::string(source_name) + ".sources";
    const auto & sources_field = require_spec_field(spec, "sources", source_name);
    const auto & sources = require_spec_array(sources_field, sources_path);
    for (size_t index = 0; index < sources.size(); ++index) {
        validate_legacy_source(sources[index], std::string(source_name) + ".sources[" + std::to_string(index) + "]");
    }
}

void validate_legacy(const json::Value & spec, std::string_view source_name) {
    (void) require_spec_string(require_spec_field(spec, "family", source_name), std::string(source_name) + ".family");
    const auto sources_path = std::string(source_name) + ".sources";
    const auto & sources_field = require_spec_field(spec, "sources", source_name);
    const auto & sources = require_spec_array(sources_field, sources_path);
    for (size_t index = 0; index < sources.size(); ++index) {
        validate_legacy_source(sources[index], std::string(source_name) + ".sources[" + std::to_string(index) + "]");
    }
}

}  // namespace

void validate_spec(const json::Value & spec, std::string_view source_name) {
    require_spec_object(spec, source_name);
    const auto * version = spec.find("schema_version");
    if (version == nullptr) {
        validate_legacy(spec, source_name);
        return;
    }
    if (!version->is_number() || version->as_i64() != kModelSpecSchemaVersion) {
        fail(std::string(source_name) + ".schema_version",
             "expected " + std::to_string(kModelSpecSchemaVersion));
    }
    validate_v1(spec, source_name);
}

}  // namespace engine::model_spec
