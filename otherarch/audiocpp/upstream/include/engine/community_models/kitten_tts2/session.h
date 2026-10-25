#pragma once

#include "engine/framework/runtime/model.h"
#include <memory>

namespace engine::community_models::kitten_tts2 {
std::shared_ptr<runtime::IVoiceModelLoader> make_kitten_tts2_loader();
}
