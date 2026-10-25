#pragma once

#include "engine/framework/core/backend.h"

#include <filesystem>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::audio {

enum class BuiltinAudioUtilityKind {
    Denoise,
    SuperResolve,
};

struct BuiltinAudioUtilityInfo {
    std::string_view id;
    BuiltinAudioUtilityKind kind = BuiltinAudioUtilityKind::Denoise;
    int input_sample_rate = 0;
    int output_sample_rate = 0;
};

struct AudioUtilityPaths {
    AudioUtilityPaths() = default;
    explicit AudioUtilityPaths(std::filesystem::path assets_root_path)
        : assets_root(std::move(assets_root_path)) {}
    AudioUtilityPaths(std::filesystem::path assets_root_path, core::BackendConfig backend_config)
        : assets_root(std::move(assets_root_path)),
          backend(backend_config) {}

    std::filesystem::path assets_root = "assets/framework/audio_utilities";
    core::BackendConfig backend;
};

struct AudioUtilityBatchResult {
    std::vector<std::filesystem::path> outputs;
};

std::filesystem::path default_audio_utility_assets_root();
std::vector<BuiltinAudioUtilityInfo> list_builtin_audio_utilities();
std::optional<BuiltinAudioUtilityInfo> find_builtin_audio_utility(std::string_view model);
BuiltinAudioUtilityInfo require_builtin_audio_utility(std::string_view model);
std::filesystem::path resolve_builtin_audio_utility_asset(
    const AudioUtilityPaths & paths,
    std::string_view model);

void denoise_file(
    const std::filesystem::path & input_wav,
    const std::filesystem::path & output_wav,
    std::string_view model = "deepfilternet2",
    const AudioUtilityPaths & paths = {});

AudioUtilityBatchResult denoise_directory(
    const std::filesystem::path & input_dir,
    const std::filesystem::path & output_dir,
    std::string_view model = "deepfilternet2",
    const AudioUtilityPaths & paths = {});

void super_resolve_file(
    const std::filesystem::path & input_wav,
    const std::filesystem::path & output_wav,
    std::string_view model = "flashsr",
    const AudioUtilityPaths & paths = {});

AudioUtilityBatchResult super_resolve_directory(
    const std::filesystem::path & input_dir,
    const std::filesystem::path & output_dir,
    std::string_view model = "flashsr",
    const AudioUtilityPaths & paths = {});

}  // namespace engine::audio
