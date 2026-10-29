#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/yue2/types.h"

namespace engine::models::yue2 {

Yue2Request parse_yue2_request(const runtime::TaskRequest & request, const Yue2GenerationConfig & defaults);
Yue2Request parse_yue2_preparation_request(
    const runtime::SessionPreparationRequest & request,
    const Yue2GenerationConfig & defaults);

}  // namespace engine::models::yue2
