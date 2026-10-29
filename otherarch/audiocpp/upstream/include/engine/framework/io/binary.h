#pragma once

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>

namespace engine::io {

class BinaryBlob {
public:
    BinaryBlob() = default;
    ~BinaryBlob();

    BinaryBlob(const BinaryBlob &) = delete;
    BinaryBlob & operator=(const BinaryBlob &) = delete;
    BinaryBlob(BinaryBlob && other) noexcept;
    BinaryBlob & operator=(BinaryBlob && other) noexcept;

    [[nodiscard]] const std::byte * data() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    void discard_range(size_t offset, size_t size) const noexcept;

private:
    friend BinaryBlob read_binary_blob(const std::filesystem::path & path, bool require_mapping);

    explicit BinaryBlob(std::vector<std::byte> owned) noexcept;
    BinaryBlob(const std::byte * mapped, size_t size) noexcept;
    void reset() noexcept;

    std::vector<std::byte> owned_;
    const std::byte * mapped_ = nullptr;
    size_t size_ = 0;
};

// require_mapping forbids an owned full-file copy when mapping is unavailable.
BinaryBlob read_binary_blob(const std::filesystem::path & path, bool require_mapping = false);
std::vector<std::byte> read_binary_file(const std::filesystem::path & path);
std::vector<float> read_f32_file(const std::filesystem::path & path);
std::vector<int32_t> read_i32_file(const std::filesystem::path & path);

}  // namespace engine::io
