#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace engine::runtime {

enum class StoredArtifactKind {
    TokenIds,
    FloatTensor,
    Bytes,
    Text,
};

struct StoredArtifact {
    StoredArtifactKind kind = StoredArtifactKind::Bytes;
    std::vector<int64_t> shape;
    std::variant<std::vector<int32_t>, std::vector<float>, std::vector<std::byte>, std::string> payload;
};

class ArtifactStore {
public:
    bool contains(const std::string & key) const;
    void put(std::string key, StoredArtifact artifact);
    StoredArtifact * find(const std::string & key);
    const StoredArtifact * find(const std::string & key) const;
    void erase(const std::string & key);
    void clear();

private:
    std::unordered_map<std::string, StoredArtifact> artifacts_;
};

}  // namespace engine::runtime
