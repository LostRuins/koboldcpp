#include "engine/framework/runtime/artifacts.h"

namespace engine::runtime {

bool ArtifactStore::contains(const std::string & key) const {
    return artifacts_.find(key) != artifacts_.end();
}

void ArtifactStore::put(std::string key, StoredArtifact artifact) {
    artifacts_[std::move(key)] = std::move(artifact);
}

StoredArtifact * ArtifactStore::find(const std::string & key) {
    const auto it = artifacts_.find(key);
    if (it == artifacts_.end()) {
        return nullptr;
    }
    return &it->second;
}

const StoredArtifact * ArtifactStore::find(const std::string & key) const {
    const auto it = artifacts_.find(key);
    if (it == artifacts_.end()) {
        return nullptr;
    }
    return &it->second;
}

void ArtifactStore::erase(const std::string & key) {
    artifacts_.erase(key);
}

void ArtifactStore::clear() {
    artifacts_.clear();
}

}  // namespace engine::runtime
