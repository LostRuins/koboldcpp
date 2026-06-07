#ifndef __SD_MODEL_IO_BINARY_IO_H__
#define __SD_MODEL_IO_BINARY_IO_H__

#include <cstdint>
#include <ostream>

namespace model_io {

    inline int32_t read_int(const uint8_t* buffer) {
        uint32_t value = 0;
        value |= static_cast<uint32_t>(buffer[3]) << 24;
        value |= static_cast<uint32_t>(buffer[2]) << 16;
        value |= static_cast<uint32_t>(buffer[1]) << 8;
        value |= static_cast<uint32_t>(buffer[0]);
        return static_cast<int32_t>(value);
    }

    inline uint16_t read_short(const uint8_t* buffer) {
        uint16_t value = 0;
        value |= static_cast<uint16_t>(buffer[1]) << 8;
        value |= static_cast<uint16_t>(buffer[0]);
        return value;
    }

    inline uint64_t read_u64(const uint8_t* buffer) {
        uint64_t value = 0;
        value |= static_cast<uint64_t>(buffer[7]) << 56;
        value |= static_cast<uint64_t>(buffer[6]) << 48;
        value |= static_cast<uint64_t>(buffer[5]) << 40;
        value |= static_cast<uint64_t>(buffer[4]) << 32;
        value |= static_cast<uint64_t>(buffer[3]) << 24;
        value |= static_cast<uint64_t>(buffer[2]) << 16;
        value |= static_cast<uint64_t>(buffer[1]) << 8;
        value |= static_cast<uint64_t>(buffer[0]);
        return value;
    }

    inline void write_u64(std::ostream& stream, uint64_t value) {
        uint8_t buffer[8];
        for (int i = 0; i < 8; ++i) {
            buffer[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xFF);
        }
        stream.write((const char*)buffer, sizeof(buffer));
    }

    inline int find_char(const uint8_t* buffer, int len, char c) {
        for (int pos = 0; pos < len; pos++) {
            if (buffer[pos] == (uint8_t)c) {
                return pos;
            }
        }
        return -1;
    }

}  // namespace model_io

#endif  // __SD_MODEL_IO_BINARY_IO_H__
