#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/model.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::runtime {

struct OptionV1CompatibilityAlias {
    std::string legacy_key;
    std::string current_key;
};

template <typename Assets>
struct SpecBackedVoiceModelConfig {
    std::string family;
    std::vector<std::string> aliases;  // accepted family hints (spec resolution still uses family)
    std::function<std::shared_ptr<const Assets>(const std::filesystem::path &)> load_assets;
    std::function<std::unique_ptr<IVoiceTaskSession>(
        const TaskSpec &,
        const SessionOptions &,
        std::shared_ptr<const Assets>,
        std::shared_ptr<const engine::model_spec::ModelContract>)> create_session;
    std::function<ModelResourcePlan(
        const std::filesystem::path &,
        const std::unordered_map<std::string, std::string> &)> model_resources;
};

inline std::shared_ptr<const engine::model_spec::ModelContract> require_model_contract(std::string family) {
    auto out = engine::model_spec::model_contract(family);
    if (!out.has_value()) {
        throw std::runtime_error(family + " requires a schema v1 model contract");
    }
    return std::make_shared<engine::model_spec::ModelContract>(std::move(*out));
}

inline void validate_spec_backed_session_options(
    const SessionOptions & options,
    const engine::model_spec::ModelContract & contract,
    std::string_view family,
    std::string_view model_name) {
    const std::string family_prefix = std::string(family) + ".";
    for (const auto & [key, _] : options.options) {
        if (key.rfind(family_prefix, 0) == 0 &&
            contract.session_option_keys.find(key) == contract.session_option_keys.end()) {
            throw std::runtime_error("unknown " + std::string(model_name) + " session option: " + key);
        }
    }
}

inline void validate_spec_backed_request_options(
    const std::unordered_map<std::string, std::string> & options,
    const engine::model_spec::ModelContract & contract,
    std::string_view model_name) {
    for (const auto & [key, _] : options) {
        if (contract.request_option_keys.find(key) == contract.request_option_keys.end()) {
            throw std::runtime_error(
                "unknown " + std::string(model_name) + " request option: " + key +
                ". The option may be misspelled or require a newer audio.cpp runtime. "
                "Supported options: https://github.com/0xShug0/audio.cpp/blob/main/model_specs/" +
                contract.metadata.family + ".json. "
                "Update audio.cpp: https://github.com/0xShug0/audio.cpp/releases");
        }
    }
}

/// The same check for list-valued options. An overload rather than a second
/// name: a family that gained a list option should not have to remember to call
/// something different, and every existing call site keeps working untouched.
inline void validate_spec_backed_request_options(
    const std::unordered_map<std::string, std::string> & options,
    const std::unordered_map<std::string, std::vector<std::string>> & option_arrays,
    const engine::model_spec::ModelContract & contract,
    std::string_view model_name) {
    validate_spec_backed_request_options(options, contract, model_name);
    for (const auto & [key, _] : option_arrays) {
        if (contract.request_option_keys.find(key) == contract.request_option_keys.end()) {
            throw std::runtime_error(
                "unknown " + std::string(model_name) + " request option: " + key +
                ". The option may be misspelled or require a newer audio.cpp runtime. "
                "Supported options: https://github.com/0xShug0/audio.cpp/blob/main/model_specs/" +
                contract.metadata.family + ".json. "
                "Update audio.cpp: https://github.com/0xShug0/audio.cpp/releases");
        }
    }
}

inline std::unordered_map<std::string, std::string> apply_option_v1_compatibility(
    std::unordered_map<std::string, std::string> options,
    std::initializer_list<OptionV1CompatibilityAlias> aliases,
    std::string_view model_name,
    std::string_view scope_name) {
    for (const auto & alias : aliases) {
        auto legacy = options.find(alias.legacy_key);
        if (legacy == options.end()) {
            continue;
        }
        auto current = options.find(alias.current_key);
        if (current != options.end()) {
            throw std::runtime_error(
                std::string(model_name) + " " + std::string(scope_name) +
                " options contain both " + alias.legacy_key + " and " +
                alias.current_key + "; use " + alias.current_key);
        }
        std::string value = std::move(legacy->second);
        options.erase(legacy);
        options.emplace(alias.current_key, std::move(value));
    }
    return options;
}

inline SessionOptions apply_option_v1_compatibility(
    SessionOptions options,
    std::initializer_list<OptionV1CompatibilityAlias> aliases,
    std::string_view model_name) {
    options.options = apply_option_v1_compatibility(
        std::move(options.options),
        aliases,
        model_name,
        "session");
    return options;
}

template <typename Assets>
class SpecBackedLoadedVoiceModel final : public ILoadedVoiceModel {
public:
    explicit SpecBackedLoadedVoiceModel(
        SpecBackedVoiceModelConfig<Assets> config,
        std::shared_ptr<const engine::model_spec::ModelContract> contract,
        std::shared_ptr<const Assets> assets)
        : config_(std::move(config)),
          contract_(std::move(contract)),
          assets_(std::move(assets)) {
        if (contract_ == nullptr) {
            throw std::invalid_argument(config_.family + " loaded model requires a model contract");
        }
        if (assets_ == nullptr) {
            throw std::invalid_argument(config_.family + " loaded model requires assets");
        }
    }

    const ModelMetadata & metadata() const noexcept override {
        return contract_->metadata;
    }

    const CapabilitySet & capabilities() const noexcept override {
        return contract_->capabilities;
    }

    std::unique_ptr<IVoiceTaskSession> create_task_session(
        const TaskSpec & task,
        const SessionOptions & options) const override {
        return config_.create_session(task, options, assets_, contract_);
    }

private:
    SpecBackedVoiceModelConfig<Assets> config_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<const Assets> assets_;
};

template <typename Assets>
class SpecBackedVoiceModelLoader final : public IVoiceModelLoader {
public:
    explicit SpecBackedVoiceModelLoader(SpecBackedVoiceModelConfig<Assets> config)
        : config_(std::move(config)) {
        if (config_.family.empty()) {
            throw std::invalid_argument("spec-backed model loader requires a family");
        }
        if (!config_.load_assets) {
            throw std::invalid_argument(config_.family + " spec-backed model loader requires an asset loader");
        }
        if (!config_.create_session) {
            throw std::invalid_argument(config_.family + " spec-backed model loader requires a session factory");
        }
    }

    std::string family() const override {
        return config_.family;
    }

    std::vector<std::string> family_aliases() const override {
        return config_.aliases;
    }

    bool family_matches(const std::string & hint) const {
        if (hint == config_.family) return true;
        return std::find(config_.aliases.begin(), config_.aliases.end(), hint) != config_.aliases.end();
    }

    CapabilitySet advertised_capabilities() const override {
        return require_model_contract(config_.family)->capabilities;
    }

    bool can_load(const ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && !family_matches(*request.family_hint)) {
            return false;
        }
        try {
            (void) engine::model_spec::load_resource_bundle_for_family(request.model_path, config_.family);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    ModelInspection inspect(const ModelLoadRequest & request) const override {
        const auto assets = config_.load_assets(request.model_path);
        const auto contract = require_model_contract(config_.family);
        const auto package_spec = engine::model_spec::default_package_spec_path(config_.family);

        ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = contract->metadata;
        inspection.capabilities = contract->capabilities;
        inspection.cli = contract->cli;
        inspection.discovered_configs = discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const override {
        auto assets = config_.load_assets(request.model_path);
        auto contract = require_model_contract(config_.family);
        return std::make_unique<SpecBackedLoadedVoiceModel<Assets>>(
            config_,
            std::move(contract),
            std::move(assets));
    }

    ModelResourcePlan model_resources(
        const std::filesystem::path & model_path,
        const std::unordered_map<std::string, std::string> & session_options) const override {
        return config_.model_resources
            ? config_.model_resources(model_path, session_options)
            : ModelResourcePlan{};
    }

private:
    SpecBackedVoiceModelConfig<Assets> config_;
};

template <typename Assets>
std::shared_ptr<IVoiceModelLoader> make_spec_backed_voice_loader(SpecBackedVoiceModelConfig<Assets> config) {
    return std::make_shared<SpecBackedVoiceModelLoader<Assets>>(std::move(config));
}

}  // namespace engine::runtime
