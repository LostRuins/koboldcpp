#include "engine/models/silero_vad/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::silero_vad {
namespace {

std::filesystem::path default_weight_candidate(const std::filesystem::path & model_root) {
    return model_root / "silero_vad_16k.safetensors";
}

SileroWeights load_silero_weights(const std::filesystem::path & checkpoint_path) {
    assets::ResourceBundle resources(checkpoint_path.parent_path());
    resources.add_file("weights", checkpoint_path);
    SileroWeights weights;
    weights.source = resources.open_tensor_source("weights");
    return weights;
}

std::string checkpoint_cache_key(const std::filesystem::path & checkpoint_path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(checkpoint_path, ec);
    return ec ? checkpoint_path.lexically_normal().string() : canonical.string();
}

}  // namespace

SileroAssetPaths resolve_silero_assets(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        const auto checkpoint_path = std::filesystem::weakly_canonical(model_path);
        SileroAssetPaths paths;
        paths.model_root = checkpoint_path.parent_path();
        paths.checkpoint_path = checkpoint_path;
        return paths;
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("Silero VAD model path does not exist: " + model_path.string());
    }
    const auto checkpoint_path = default_weight_candidate(model_path);
    if (!engine::io::is_existing_file(checkpoint_path)) {
        throw std::runtime_error("Silero VAD weights not found: " + checkpoint_path.string());
    }
    SileroAssetPaths paths;
    paths.model_root = std::filesystem::weakly_canonical(model_path);
    paths.checkpoint_path = std::filesystem::weakly_canonical(checkpoint_path);
    return paths;
}

std::shared_ptr<const SileroWeights> load_silero_weights_cached(const std::filesystem::path & checkpoint_path) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<const SileroWeights>> cache;
    const auto key = checkpoint_cache_key(checkpoint_path);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(key); it != cache.end()) {
            if (auto existing = it->second.lock()) {
                return existing;
            }
        }
    }
    auto loaded = std::make_shared<const SileroWeights>(load_silero_weights(checkpoint_path));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[key] = loaded;
    }
    return loaded;
}

}  // namespace engine::models::silero_vad
