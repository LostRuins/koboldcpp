#pragma once

#include "engine/framework/runtime/model.h"

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace engine::model_spec {

enum class ModelSpecValueType {
    Bool,
    Number,
    String,
};

struct ModelDependencyCondition {
    std::string scope;
    std::string option_key;
    ModelSpecValueType equals_type = ModelSpecValueType::String;
    bool equals_bool = false;
    double equals_number = 0.0;
    std::string equals_string;
};

struct ModelDependency {
    std::string kind;
    std::string family;
    std::string scope;
    std::string option;
    std::string option_key;
    bool required = false;
    std::vector<ModelDependencyCondition> required_when;
    std::optional<std::string> path;
};

struct ModelContract {
    runtime::ModelMetadata metadata;
    runtime::CapabilitySet capabilities;
    runtime::ModelCliInterface cli;
    std::unordered_set<std::string> request_option_keys;
    std::unordered_set<std::string> session_option_keys;
    std::unordered_set<std::string> load_option_keys;
};

[[nodiscard]] std::optional<ModelContract> model_contract(std::string_view family);
// nullopt when only a legacy embedded spec exists, so a family migrating to spec-backed
// options can keep its pre-migration validation for older GGUFs.
[[nodiscard]] std::optional<ModelContract> find_model_contract(std::string_view family);
[[nodiscard]] std::optional<runtime::CapabilitySet> advertised_capabilities(std::string_view family);
[[nodiscard]] std::optional<runtime::ModelMetadata> model_metadata(std::string_view family);
[[nodiscard]] std::optional<runtime::ModelCliInterface> cli_interface(std::string_view family);
[[nodiscard]] std::vector<ModelDependency> dependencies(std::string_view family);

}  // namespace engine::model_spec
