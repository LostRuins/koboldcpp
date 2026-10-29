#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/task_vocabulary.h"

#include "engine/framework/model_spec/options.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/model_spec/schema.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::model_spec {
namespace {

namespace json = engine::io::json;

runtime::VoiceTaskKind parse_task_kind(const std::string & value) {
    // Spec names, not ABI tokens: "music" here is "gen" there. The mapping is
    // in runtime::task_vocabulary, which the schema's allowed set reads too, so
    // a name the schema accepts cannot be one this rejects.
    const auto token = runtime::task_token_for_spec_name(value);
    if (!token.empty()) {
        return runtime::parse_voice_task_kind(std::string(token));
    }
    std::string expected;
    std::size_t count = 0;
    const auto * entries = runtime::task_vocabulary(count);
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t alias = 0; alias < entries[i].alias_count; ++alias) {
            if (!expected.empty()) {
                expected += ", ";
            }
            expected.append(entries[i].aliases[alias]);
        }
    }
    throw std::runtime_error("unknown model spec task: " + value + " (expected one of " + expected + ")");
}

runtime::RunMode parse_run_mode(const std::string & value) {
    if (value == "offline") {
        return runtime::RunMode::Offline;
    }
    if (value == "streaming") {
        return runtime::RunMode::Streaming;
    }
    throw std::runtime_error("unknown model spec run mode: " + value);
}

std::vector<runtime::RunMode> parse_run_modes(const json::Value & value) {
    std::vector<runtime::RunMode> modes;
    for (const auto & item : value.as_array()) {
        modes.push_back(parse_run_mode(item.as_string()));
    }
    return modes;
}

std::vector<runtime::TaskCapability> parse_tasks(const json::Value & tasks_value, const json::Value & modes_value) {
    const auto modes = parse_run_modes(modes_value);
    std::vector<runtime::TaskCapability> tasks;
    for (const auto & item : tasks_value.as_array()) {
        runtime::TaskCapability task;
        task.task = parse_task_kind(item.as_string());
        task.modes = modes;
        tasks.push_back(std::move(task));
    }
    return tasks;
}

json::Value load_spec_for_family(std::string_view family) {
    return engine::model_spec::load_spec(engine::model_spec::default_contract_spec_path(family));
}

json::Value load_contract_spec_for_family(std::string_view family) {
    return engine::model_spec::load_contract_spec(engine::model_spec::default_contract_spec_path(family));
}

std::string ref_candidate_path(const std::string & ref) {
    const auto split = ref.find(':');
    if (split == std::string::npos) {
        return ref;
    }
    return ref.substr(split + 1);
}

std::string resource_candidate_path(const json::Value & ref) {
    if (ref.is_string()) {
        return ref_candidate_path(ref.as_string());
    }
    if (const auto * source = ref.find("source")) {
        return ref_candidate_path(source->as_string());
    }
    return {};
}

void push_unique_candidate(std::vector<std::string> & out, const std::string & candidate) {
    if (candidate.empty()) {
        return;
    }
    if (std::find(out.begin(), out.end(), candidate) == out.end()) {
        out.push_back(candidate);
    }
}

void append_resource_candidates(
    std::vector<std::string> & out,
    const json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    std::vector<std::string> candidates;
    candidates.reserve(map_value->as_object().size());
    for (const auto & [_, ref] : map_value->as_object()) {
        candidates.push_back(resource_candidate_path(ref));
    }
    std::sort(candidates.begin(), candidates.end());
    for (const auto & candidate : candidates) {
        push_unique_candidate(out, candidate);
    }
}

void append_source_candidates(
    std::vector<std::string> & config_candidates,
    std::vector<std::string> & weight_candidates,
    const json::Value & source) {
    append_resource_candidates(config_candidates, source.find("files"));
    append_resource_candidates(config_candidates, source.find("optional_files"));
    append_resource_candidates(weight_candidates, source.find("tensors"));
}

void append_package_candidates(std::vector<std::string> & weight_candidates, const json::Value * packages) {
    if (packages == nullptr || packages->is_null()) {
        return;
    }
    for (const auto & package : packages->as_array()) {
        for (const auto & file : package.require("files").as_array()) {
            push_unique_candidate(weight_candidates, file.as_string());
        }
    }
}

bool has_capability(const json::Value & capabilities, std::string_view capability) {
    const std::string capability_string(capability);
    for (const auto & [_, task_capabilities] : capabilities.as_object()) {
        for (const auto & item : task_capabilities.as_array()) {
            if (item.as_string() == capability_string) {
                return true;
            }
        }
    }
    return false;
}

std::string join_values(const std::vector<std::string> & values) {
    std::string out;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out += "|";
        }
        out += values[index];
    }
    return out;
}

std::vector<runtime::CliOptionInfo> parse_cli_options(const json::Value * value, std::string_view family_prefix = {}) {
    std::vector<runtime::CliOptionInfo> options;
    if (value == nullptr || value->is_null()) {
        return options;
    }
    for (const auto & item : value->as_array()) {
        runtime::CliOptionInfo option;
        option.name = json::require_string(item, "name");
        if (!family_prefix.empty()) {
            option.name = std::string(family_prefix) + "." + option.name;
        }
        const auto option_type = json::require_string(item, "type");
        if (option_type == "enum") {
            if (const auto * preset = item.find("preset")) {
                option.value_name = join_values(require_option_preset(preset->as_string()));
            } else {
                option.value_name = join_values(json::require_string_array(item, "values"));
            }
        } else {
            option.value_name = option_type;
        }
        option.description = json::require_string(item, "description");
        option.required = json::require_bool(item, "required");
        if (const auto * default_value = item.find("default")) {
            option.default_value = json::stringify(*default_value);
        }
        if (const auto * min_value = item.find("min")) {
            option.min_value = json::stringify(*min_value);
        }
        if (const auto * max_value = item.find("max")) {
            option.max_value = json::stringify(*max_value);
        }
        options.push_back(std::move(option));
    }
    return options;
}

std::vector<ModelDependencyCondition> parse_dependency_conditions(const json::Value * value) {
    std::vector<ModelDependencyCondition> out;
    if (value == nullptr || value->is_null()) {
        return out;
    }
    for (const auto & item : value->as_array()) {
        ModelDependencyCondition condition;
        condition.scope = json::require_string(item, "scope");
        condition.option_key = json::require_string(item, "option_key");
        const auto & equals = item.require("equals");
        if (equals.is_bool()) {
            condition.equals_type = ModelSpecValueType::Bool;
            condition.equals_bool = equals.as_bool();
        } else if (equals.is_number()) {
            condition.equals_type = ModelSpecValueType::Number;
            condition.equals_number = equals.as_number();
        } else {
            condition.equals_type = ModelSpecValueType::String;
            condition.equals_string = equals.as_string();
        }
        out.push_back(std::move(condition));
    }
    return out;
}

runtime::CapabilitySet capabilities_from_spec(const json::Value & spec) {
    runtime::CapabilitySet out;
    const auto * capabilities = spec.find("capabilities");
    if (capabilities == nullptr || capabilities->is_null()) {
        throw std::runtime_error("model spec has no capabilities contract");
    }
    out.supported_tasks = parse_tasks(spec.require("tasks"), spec.require("modes"));
    out.languages = json::optional_string_array(spec, "languages");
    out.supports_speaker_reference = has_capability(*capabilities, "speaker_reference");
    out.supports_style_condition =
        has_capability(*capabilities, "style_control") || has_capability(*capabilities, "emotion_control");
    out.supports_timestamps =
        has_capability(*capabilities, "word_timestamps") || has_capability(*capabilities, "segments") ||
        has_capability(*capabilities, "speech_segments");
    return out;
}

runtime::ModelMetadata metadata_from_spec(const json::Value & spec) {
    runtime::ModelMetadata out;
    out.family = json::require_string(spec, "family");
    out.variant = json::require_string(spec, "display_name");
    out.description = json::require_string(spec, "description");
    append_package_candidates(out.weight_candidates, spec.find("packages"));
    if (const auto * sources = spec.find("sources")) {
        for (const auto & source : sources->as_array()) {
            append_source_candidates(out.config_candidates, out.weight_candidates, source);
        }
    }
    return out;
}

runtime::ModelCliInterface cli_from_spec(const json::Value & spec) {
    const auto * options = spec.find("options");
    if (options == nullptr || options->is_null()) {
        throw std::runtime_error("model spec has no options contract");
    }
    runtime::ModelCliInterface out;
    const auto family = json::require_string(spec, "family");
    out.request_options = parse_cli_options(options->find("request"));
    out.session_options = parse_cli_options(options->find("session"), family);
    out.load_options = parse_cli_options(options->find("load"), family);
    return out;
}

std::unordered_set<std::string> option_keys(const std::vector<runtime::CliOptionInfo> & options) {
    std::unordered_set<std::string> keys;
    keys.reserve(options.size());
    for (const auto & option : options) {
        keys.insert(option.name);
    }
    return keys;
}

ModelContract contract_from_spec(const json::Value & spec) {
    ModelContract out;
    out.metadata.family = json::require_string(spec, "family");
    out.metadata.variant = json::require_string(spec, "display_name");
    out.metadata.description = json::require_string(spec, "description");
    out.capabilities = capabilities_from_spec(spec);
    out.cli = cli_from_spec(spec);
    out.request_option_keys = option_keys(out.cli.request_options);
    out.session_option_keys = option_keys(out.cli.session_options);
    out.load_option_keys = option_keys(out.cli.load_options);
    return out;
}

}  // namespace

std::optional<ModelContract> model_contract(std::string_view family) {
    const auto spec = load_contract_spec_for_family(family);
    const auto * version = spec.find("schema_version");
    if (version == nullptr) {
        return std::nullopt;
    }
    if (!version->is_number() || version->as_i64() != kModelSpecSchemaVersion) {
        throw std::runtime_error(
            "model spec schema_version: expected " + std::to_string(kModelSpecSchemaVersion));
    }
    return contract_from_spec(spec);
}

std::optional<ModelContract> find_model_contract(std::string_view family) {
    if (!engine::model_spec::find_contract_spec_path(family).has_value()) {
        return std::nullopt;
    }
    return model_contract(family);
}

std::optional<runtime::CapabilitySet> advertised_capabilities(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    if (spec.find("schema_version") == nullptr || spec.find("capabilities") == nullptr) {
        return std::nullopt;
    }
    return capabilities_from_spec(spec);
}

std::optional<runtime::ModelMetadata> model_metadata(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    if (spec.find("schema_version") == nullptr) {
        return std::nullopt;
    }
    return metadata_from_spec(spec);
}

std::optional<runtime::ModelCliInterface> cli_interface(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    if (spec.find("schema_version") == nullptr || spec.find("options") == nullptr) {
        return std::nullopt;
    }
    return cli_from_spec(spec);
}

std::vector<ModelDependency> dependencies(std::string_view family) {
    const auto family_string = std::string(family);
    const auto spec = load_spec_for_family(family);
    const auto * rows = spec.find("dependencies");
    if (rows == nullptr || rows->is_null()) {
        return {};
    }
    std::vector<ModelDependency> out;
    for (const auto & item : rows->as_array()) {
        ModelDependency dependency;
        dependency.kind = json::require_string(item, "kind");
        dependency.family = json::require_string(item, "family");
        dependency.scope = json::require_string(item, "scope");
        dependency.option = json::require_string(item, "option");
        dependency.option_key = family_string + "." + dependency.option;
        dependency.required = json::require_bool(item, "required");
        dependency.required_when = parse_dependency_conditions(item.find("required_when"));
        if (const auto * path = item.find("path")) {
            dependency.path = path->as_string();
        }
        out.push_back(std::move(dependency));
    }
    return out;
}

}  // namespace engine::model_spec
