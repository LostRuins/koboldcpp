#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/models/qwen3_asr/assets.h"

#include <filesystem>
#include <memory>

#include "engine/community_models/audio8_asr/types.h"

namespace engine::community_models::audio8_asr {

struct Audio8ASRAssets {
    // Bundle and config for this family (arkasr layout).
    assets::ResourceBundle resources;
    Audio8ASRConfig config;

    // Qwen2 BPE tokenizer shared with the Qwen3-ASR family tooling.
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;

    // A Qwen3-ASR view over the same bundle so the audio encoder and Whisper
    // frontend implementations can be reused unchanged. The weights source in
    // this view renames Audio8 tensors to the prefixes those implementations
    // expect (`audio_encoder.*` -> `model.audio_tower.*` etc.).
    std::shared_ptr<const qwen3_asr::Qwen3ASRAssets> encoder_assets;
};

std::shared_ptr<const Audio8ASRAssets> load_audio8_asr_assets(const std::filesystem::path & model_path);

}  // namespace engine::community_models::audio8_asr
