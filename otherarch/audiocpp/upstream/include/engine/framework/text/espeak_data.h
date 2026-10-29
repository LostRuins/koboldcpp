#pragma once
#include <filesystem>

namespace engine::audio {
// A data-only GGUF, using audio.cpp's existing embedded binary file layout.
void pack_espeak_data(const std::filesystem::path & directory,
                      const std::filesystem::path & output);
// Validates and atomically publishes an immutable, content-checked cache entry.
std::filesystem::path materialize_espeak_data(const std::filesystem::path & package);
}
