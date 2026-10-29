#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace engine::package_manager {

struct PackageProgress {
    uint64_t downloaded_bytes = 0;
    uint64_t total_bytes = 0;
    std::string message;
};

class PackageManager {
public:
    PackageManager(std::filesystem::path repository_root, std::filesystem::path models_root);
    ~PackageManager();

    PackageManager(const PackageManager &) = delete;
    PackageManager & operator=(const PackageManager &) = delete;

    std::string install(
        const std::string & package_id,
        bool overwrite,
        const std::shared_ptr<std::atomic_bool> & cancelled,
        const std::function<void(const PackageProgress &)> & progress);
    std::string clean_partial(const std::string & package_id);
    std::string remove(const std::string & package_id);

    // JSON inventory consumed by the native WebUI for local presence, remote
    // sizes, and per-file version state.
    std::string inventory(bool query_remote, const std::string & package_id = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::package_manager
