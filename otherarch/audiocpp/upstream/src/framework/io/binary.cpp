#include "engine/framework/io/binary.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace engine::io {
namespace {

template <typename T>
std::vector<T> read_typed_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    if (size % sizeof(T) != 0) {
        throw std::runtime_error("binary file has invalid size for requested element type: " + path.string());
    }
    std::vector<T> values(size / sizeof(T));
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("failed to read binary file: " + path.string());
    }
    return values;
}

}  // namespace

BinaryBlob::~BinaryBlob() {
    reset();
}

BinaryBlob::BinaryBlob(std::vector<std::byte> owned) noexcept
    : owned_(std::move(owned)) {}

BinaryBlob::BinaryBlob(const std::byte * mapped, size_t size) noexcept
    : mapped_(mapped),
      size_(size) {}

BinaryBlob::BinaryBlob(BinaryBlob && other) noexcept
    : owned_(std::move(other.owned_)),
      mapped_(other.mapped_),
      size_(other.size_) {
    other.mapped_ = nullptr;
    other.size_ = 0;
}

BinaryBlob & BinaryBlob::operator=(BinaryBlob && other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();
    owned_ = std::move(other.owned_);
    mapped_ = other.mapped_;
    size_ = other.size_;
    other.mapped_ = nullptr;
    other.size_ = 0;
    return *this;
}

const std::byte * BinaryBlob::data() const noexcept {
    return mapped_ != nullptr ? mapped_ : owned_.data();
}

size_t BinaryBlob::size() const noexcept {
    return mapped_ != nullptr ? size_ : owned_.size();
}

bool BinaryBlob::empty() const noexcept {
    return size() == 0;
}

void BinaryBlob::discard_range(size_t offset, size_t size) const noexcept {
#if defined(__unix__) || defined(__APPLE__)
    if (mapped_ == nullptr || size == 0 || offset >= size_) {
        return;
    }
    const size_t end = std::min(offset + size, size_);
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return;
    }
    const size_t page_size = static_cast<size_t>(page_size_long);
    const size_t aligned_begin = offset - (offset % page_size);
    const size_t aligned_end = ((end + page_size - 1) / page_size) * page_size;
    const size_t clamped_end = std::min(aligned_end, size_);
    if (clamped_end > aligned_begin) {
        void * address = const_cast<std::byte *>(mapped_) + static_cast<std::ptrdiff_t>(aligned_begin);
        const size_t length = clamped_end - aligned_begin;
#if defined(MADV_DONTNEED)
        (void) madvise(address, length, MADV_DONTNEED);
#elif defined(POSIX_MADV_DONTNEED)
        (void) posix_madvise(address, length, POSIX_MADV_DONTNEED);
#endif
    }
#else
    (void) offset;
    (void) size;
#endif
}

void BinaryBlob::reset() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    if (mapped_ != nullptr) {
        munmap(const_cast<std::byte *>(mapped_), size_);
        mapped_ = nullptr;
        size_ = 0;
    }
#elif defined(_WIN32)
    if (mapped_ != nullptr) {
        UnmapViewOfFile(mapped_);
        mapped_ = nullptr;
        size_ = 0;
    }
#else
    mapped_ = nullptr;
    size_ = 0;
#endif
    owned_.clear();
    owned_.shrink_to_fit();
}

BinaryBlob read_binary_blob(const std::filesystem::path & path, bool require_mapping) {
#if defined(__unix__) || defined(__APPLE__)
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        struct stat st {};
        if (fstat(fd, &st) == 0 && st.st_size >= 0) {
            const size_t size = static_cast<size_t>(st.st_size);
            if (size == 0) {
                close(fd);
                return BinaryBlob();
            }
            void * mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (mapped != MAP_FAILED) {
                return BinaryBlob(static_cast<const std::byte *>(mapped), size);
            }
        } else {
            close(fd);
        }
    }
#elif defined(_WIN32)
    const HANDLE file = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER file_size {};
        if (GetFileSizeEx(file, &file_size) != 0 &&
            file_size.QuadPart >= 0 &&
            static_cast<unsigned long long>(file_size.QuadPart) <=
                static_cast<unsigned long long>(
                    std::numeric_limits<size_t>::max())) {
            const size_t size = static_cast<size_t>(file_size.QuadPart);
            if (size == 0) {
                CloseHandle(file);
                return BinaryBlob();
            }
            const HANDLE mapping = CreateFileMappingW(
                file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping != nullptr) {
                const void * mapped = MapViewOfFile(
                    mapping, FILE_MAP_READ, 0, 0, 0);
                CloseHandle(mapping);
                CloseHandle(file);
                if (mapped != nullptr) {
                    return BinaryBlob(
                        static_cast<const std::byte *>(mapped), size);
                }
            } else {
                CloseHandle(file);
            }
        } else {
            CloseHandle(file);
        }
    }
#endif
    if (require_mapping) {
        throw std::runtime_error("failed to memory-map binary file: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(size);
    if (size > 0) {
        input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
        if (!input) {
            throw std::runtime_error("failed to read binary file: " + path.string());
        }
    }
    return BinaryBlob(std::move(bytes));
}

std::vector<std::byte> read_binary_file(const std::filesystem::path & path) {
    auto blob = read_binary_blob(path);
    std::vector<std::byte> bytes(blob.size());
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), blob.data(), blob.size());
    }
    return bytes;
}

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    return read_typed_file<float>(path);
}

std::vector<int32_t> read_i32_file(const std::filesystem::path & path) {
    return read_typed_file<int32_t>(path);
}

}  // namespace engine::io
