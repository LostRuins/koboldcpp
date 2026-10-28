#pragma once

#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/pocket_tts/types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;

struct VoiceAttentionCache {
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t cached_steps = 0;
    int64_t offset = 0;
    std::vector<float> key;
    std::vector<float> value;
};

struct VoiceStateAssets {
    std::vector<VoiceAttentionCache> transformer_layers;
};

VoiceStateAssets load_voice_state_assets(const std::filesystem::path & source);
void save_voice_state_assets(
    const std::filesystem::path & destination,
    const runtime::TransformerKVState & state,
    int64_t heads,
    int64_t head_dim);
std::filesystem::path preset_embedding_path(const std::filesystem::path & model_root, const std::string & preset_name);
VoiceStateAssets load_voice_assets_for_plan(const VoiceConditioningPlan & plan, const PocketTTSAssets & manifest);

}  // namespace engine::models::pocket_tts
