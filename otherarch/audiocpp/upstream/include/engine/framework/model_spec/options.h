#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace engine::model_spec {

[[nodiscard]] const std::unordered_map<std::string, std::vector<std::string>> & option_presets();
[[nodiscard]] const std::unordered_map<std::string, std::unordered_set<std::string>> & shared_option_contracts();
[[nodiscard]] const std::vector<std::string> & require_option_preset(std::string_view preset);

}  // namespace engine::model_spec
