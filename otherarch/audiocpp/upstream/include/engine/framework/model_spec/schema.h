#pragma once

#include "engine/framework/io/json.h"

#include <string_view>

namespace engine::model_spec {

inline constexpr int kModelSpecSchemaVersion = 1;

void validate_spec(const engine::io::json::Value & spec, std::string_view source_name);

}  // namespace engine::model_spec
