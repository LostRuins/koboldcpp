#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <streambuf>

namespace engine::audio {
namespace {

class ReadOnlyMemoryStreamBuffer final : public std::streambuf {
public:
    explicit ReadOnlyMemoryStreamBuffer(std::string_view data) {
        // std::streambuf::setg takes char*, but this input stream only reads from the buffer.
        auto * begin = const_cast<char *>(data.data());
        setg(begin, begin, begin + data.size());
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if ((which & std::ios_base::in) == 0) {
            return pos_type(off_type(-1));
        }
        char * base = eback();
        char * next = gptr();
        char * end = egptr();
        char * target = nullptr;
        if (dir == std::ios_base::beg) {
            target = base + offset;
        } else if (dir == std::ios_base::cur) {
            target = next + offset;
        } else if (dir == std::ios_base::end) {
            target = end + offset;
        }
        if (target == nullptr || target < base || target > end) {
            return pos_type(off_type(-1));
        }
        setg(base, target, end);
        return pos_type(target - base);
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode which) override {
        return seekoff(off_type(position), std::ios_base::beg, which);
    }
};

template <typename T>
T read_scalar(std::istream & input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("failed to read WAV scalar");
    }
    return value;
}

void skip_bytes(std::istream & input, std::streamoff count) {
    input.seekg(count, std::ios::cur);
    if (!input) {
        throw std::runtime_error("failed to seek inside WAV file");
    }
}

// WAVE format tags. EXTENSIBLE is the one that matters in practice: many
// encoders emit it for ordinary PCM16 whenever there are more than two channels
// or a channel mask is set, and the real codec then lives in a SubFormat GUID
// rather than in the format tag itself.
constexpr uint16_t kFormatPcm = 0x0001;
constexpr uint16_t kFormatFloat = 0x0003;
constexpr uint16_t kFormatALaw = 0x0006;
constexpr uint16_t kFormatMuLaw = 0x0007;
constexpr uint16_t kFormatExtensible = 0xFFFE;

// Bytes 2..15 of every KSDATAFORMAT_SUBTYPE_* GUID:
// XXXXXXXX-0000-0010-8000-00aa00389b71.
constexpr std::array<char, 14> kKsDataFormatSubtypeTail = {
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, static_cast<char>(0x80),
    0x00, 0x00, static_cast<char>(0xAA), 0x00, 0x38, static_cast<char>(0x9B), 0x71,
};

// Names a container we can recognise but not decode, so the error can say what
// the file actually is instead of "invalid WAV RIFF header".
const char * identify_foreign_container(const std::array<char, 12> & header) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(header.data());
    if (std::memcmp(header.data(), "fLaC", 4) == 0) {
        return "FLAC";
    }
    if (std::memcmp(header.data(), "OggS", 4) == 0) {
        return "Ogg (Vorbis/Opus)";
    }
    if (std::memcmp(header.data(), "ID3", 3) == 0) {
        return "MP3";
    }
    // MPEG audio frame sync: 11 set bits.
    if (bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0) {
        return "MP3";
    }
    if (std::memcmp(header.data() + 4, "ftyp", 4) == 0) {
        return "MP4/M4A (AAC or ALAC)";
    }
    if (std::memcmp(header.data(), "FORM", 4) == 0) {
        return "AIFF";
    }
    if (std::memcmp(header.data(), "RF64", 4) == 0) {
        return "RF64";
    }
    if (std::memcmp(header.data(), "caff", 4) == 0) {
        return "CAF";
    }
    if (bytes[0] == 0x1A && bytes[1] == 0x45 && bytes[2] == 0xDF && bytes[3] == 0xA3) {
        return "Matroska/WebM";
    }
    return nullptr;
}

// G.711 expansion. Both are 8-bit logarithmic codings still common in
// telephony recordings and in WAVs produced by conferencing tools.
float decode_mu_law(uint8_t value) {
    value = static_cast<uint8_t>(~value);
    const int sign = (value & 0x80) != 0 ? -1 : 1;
    const int exponent = (value >> 4) & 0x07;
    const int mantissa = value & 0x0F;
    const int magnitude = ((mantissa << 3) + 0x84) << exponent;
    return static_cast<float>(sign * (magnitude - 0x84)) / 32768.0F;
}

float decode_a_law(uint8_t value) {
    value ^= 0x55;
    // Note the inversion relative to mu-law above: in A-law the sign bit marks a
    // POSITIVE sample. Getting this backwards is silent -- the audio decodes at
    // the right amplitude, just phase-inverted -- so it is pinned by an
    // exhaustive 256-code table in the tests rather than by spot checks.
    const int sign = (value & 0x80) != 0 ? 1 : -1;
    const int exponent = (value >> 4) & 0x07;
    const int mantissa = value & 0x0F;
    int magnitude = 0;
    if (exponent == 0) {
        magnitude = (mantissa << 4) + 8;
    } else {
        magnitude = ((mantissa << 4) + 0x108) << (exponent - 1);
    }
    return static_cast<float>(sign * magnitude) / 32768.0F;
}

std::string describe_encoding(uint16_t format, uint16_t bits) {
    std::string name;
    switch (format) {
        case kFormatPcm: name = "PCM"; break;
        case kFormatFloat: name = "IEEE float"; break;
        case kFormatALaw: name = "A-law"; break;
        case kFormatMuLaw: name = "mu-law"; break;
        case kFormatExtensible: name = "extensible"; break;
        default: name = "format tag " + std::to_string(format); break;
    }
    return name + ", " + std::to_string(bits) + "-bit";
}

}  // namespace

WavData read_wav_f32(std::istream & input) {
    if (!input) {
        throw std::runtime_error("could not open WAV input");
    }

    // Consume the 12-byte RIFF header once and keep it: it doubles as the magic
    // for naming a non-WAV container below. Deliberately no rewind afterwards --
    // these bytes are spent, and an absolute seek back to 12 would be wrong for
    // an istream that did not begin at offset 0 and impossible for one that
    // cannot seek at all, such as a pipe.
    std::array<char, 12> header{};
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    const auto header_read = static_cast<size_t>(input.gcount());
    input.clear();

    if (header_read < 12 || std::memcmp(header.data(), "RIFF", 4) != 0 ||
        std::memcmp(header.data() + 8, "WAVE", 4) != 0) {
        if (const char * container = identify_foreign_container(header)) {
            throw std::runtime_error(
                std::string("input is ") + container +
                ", not WAV; convert it first, e.g. "
                "`ffmpeg -i input -ac 1 -ar 44100 -c:a pcm_s16le output.wav`");
        }
        throw std::runtime_error("invalid WAV RIFF header");
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> data;

    while (input) {
        char chunk_id[4];
        input.read(chunk_id, 4);
        if (!input) {
            break;
        }
        const uint32_t chunk_size = read_scalar<uint32_t>(input);
        const std::string id(chunk_id, 4);
        if (id == "fmt ") {
            if (chunk_size < 16) {
                throw std::runtime_error(
                    "malformed WAV fmt chunk (needs 16 bytes, got " +
                    std::to_string(chunk_size) + ")");
            }
            audio_format = read_scalar<uint16_t>(input);
            channels = read_scalar<uint16_t>(input);
            sample_rate = read_scalar<uint32_t>(input);
            skip_bytes(input, 6);
            bits_per_sample = read_scalar<uint16_t>(input);
            std::streamoff consumed = 16;
            if (audio_format == kFormatExtensible) {
                if (chunk_size < 40) {
                    throw std::runtime_error(
                        "malformed WAVEFORMATEXTENSIBLE fmt chunk (needs 40 bytes, got " +
                        std::to_string(chunk_size) + ")");
                }
                const uint16_t cb_size = read_scalar<uint16_t>(input);
                if (cb_size < 22) {
                    throw std::runtime_error(
                        "malformed WAVEFORMATEXTENSIBLE fmt chunk (cbSize " +
                        std::to_string(cb_size) + ", needs at least 22)");
                }
                skip_bytes(input, 2);   // wValidBitsPerSample
                skip_bytes(input, 4);   // dwChannelMask
                // Only the first two bytes of the SubFormat GUID carry the real
                // format tag. The remaining fourteen are a fixed suffix shared by
                // every KSDATAFORMAT_SUBTYPE_*; checking them is what separates a
                // genuine format tag from an unrelated codec whose GUID merely
                // happens to start with the same two bytes.
                const uint16_t sub_format = read_scalar<uint16_t>(input);
                std::array<char, 14> guid_tail{};
                input.read(guid_tail.data(), static_cast<std::streamsize>(guid_tail.size()));
                if (!input) {
                    throw std::runtime_error("truncated WAVEFORMATEXTENSIBLE SubFormat GUID");
                }
                if (std::memcmp(guid_tail.data(), kKsDataFormatSubtypeTail.data(),
                                kKsDataFormatSubtypeTail.size()) != 0) {
                    throw std::runtime_error(
                        "unsupported WAV encoding (extensible SubFormat is not a "
                        "KSDATAFORMAT_SUBTYPE_* GUID); convert with "
                        "`ffmpeg -i input -ac 1 -ar 44100 -c:a pcm_s16le output.wav`");
                }
                audio_format = sub_format;
                consumed = 40;
            }
            if (chunk_size > consumed) {
                skip_bytes(input, static_cast<std::streamoff>(chunk_size) - consumed);
            }
        } else if (id == "data") {
            // chunk_size is a 32-bit field read straight from the file, so a
            // few-byte WAV can claim up to 4 GiB. resize() commits that whole
            // allocation before a single byte of it is read, which turns a
            // truncated or hostile header into an out-of-memory condition
            // instead of a parse error.
            //
            // Grow only as fast as data actually arrives: a claim the file
            // cannot back now fails after one block rather than one allocation.
            constexpr size_t kReadBlock = 1u << 20;  // 1 MiB
            data.clear();
            size_t remaining = chunk_size;
            while (remaining > 0) {
                const size_t step = std::min(remaining, kReadBlock);
                const size_t filled = data.size();
                data.resize(filled + step);
                input.read(data.data() + filled, static_cast<std::streamsize>(step));
                if (!input) {
                    throw std::runtime_error("failed to read WAV data chunk");
                }
                remaining -= step;
            }
        } else {
            skip_bytes(input, chunk_size);
        }
        if (chunk_size % 2 == 1) {
            // RIFF pads an odd-sized chunk to an even boundary, but plenty of
            // writers omit that byte when the chunk is the last thing in the
            // file. It carries no data, so a missing one at EOF is not an error.
            // This matters more than it used to: PCM8, A-law and mu-law are one
            // byte per sample, so odd data chunks are now common.
            input.seekg(1, std::ios::cur);
            if (!input) {
                input.clear();
                break;
            }
        }
    }

    if (channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data.empty()) {
        throw std::runtime_error("incomplete WAV file");
    }

    WavData wav;
    wav.sample_rate = static_cast<int>(sample_rate);
    wav.channels = static_cast<int>(channels);

    if (audio_format == kFormatPcm && bits_per_sample == 8) {
        // 8-bit PCM in WAV is unsigned, offset by 128.
        wav.samples.resize(data.size());
        const auto * pcm = reinterpret_cast<const uint8_t *>(data.data());
        for (size_t i = 0; i < data.size(); ++i) {
            wav.samples[i] = (static_cast<float>(pcm[i]) - 128.0F) / 128.0F;
        }
        return wav;
    }

    if (audio_format == kFormatMuLaw && bits_per_sample == 8) {
        wav.samples.resize(data.size());
        const auto * pcm = reinterpret_cast<const uint8_t *>(data.data());
        for (size_t i = 0; i < data.size(); ++i) {
            wav.samples[i] = decode_mu_law(pcm[i]);
        }
        return wav;
    }

    if (audio_format == kFormatALaw && bits_per_sample == 8) {
        wav.samples.resize(data.size());
        const auto * pcm = reinterpret_cast<const uint8_t *>(data.data());
        for (size_t i = 0; i < data.size(); ++i) {
            wav.samples[i] = decode_a_law(pcm[i]);
        }
        return wav;
    }

    if (audio_format == kFormatPcm && bits_per_sample == 32) {
        if (data.size() % sizeof(int32_t) != 0) {
            throw std::runtime_error("malformed PCM32 WAV data chunk");
        }
        const size_t sample_count = data.size() / sizeof(int32_t);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const int32_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 2147483648.0F;
        }
        return wav;
    }

    if (audio_format == kFormatFloat && bits_per_sample == 64) {
        if (data.size() % sizeof(double) != 0) {
            throw std::runtime_error("malformed float64 WAV data chunk");
        }
        const size_t sample_count = data.size() / sizeof(double);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const double *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]);
        }
        return wav;
    }

    if (audio_format == 1 && bits_per_sample == 16) {
        const size_t sample_count = data.size() / sizeof(int16_t);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const int16_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 32768.0F;
        }
        return wav;
    }

    if (audio_format == 1 && bits_per_sample == 24) {
        if (data.size() % 3 != 0) {
            throw std::runtime_error("malformed PCM24 WAV data chunk");
        }
        const size_t sample_count = data.size() / 3;
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const uint8_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            const size_t offset = i * 3;
            int32_t value =
                static_cast<int32_t>(pcm[offset]) |
                (static_cast<int32_t>(pcm[offset + 1]) << 8) |
                (static_cast<int32_t>(pcm[offset + 2]) << 16);
            if ((value & 0x00800000) != 0) {
                value |= ~0x00FFFFFF;
            }
            wav.samples[i] = static_cast<float>(value) / 8388608.0F;
        }
        return wav;
    }

    if (audio_format == 3 && bits_per_sample == 32) {
        const size_t sample_count = data.size() / sizeof(float);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const float *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = pcm[i];
        }
        return wav;
    }

    throw std::runtime_error(
        "unsupported WAV encoding (" + describe_encoding(audio_format, bits_per_sample) +
        "); supported: PCM 8/16/24/32-bit, float 32/64-bit, A-law and mu-law. "
        "Convert with `ffmpeg -i input -ac 1 -ar 44100 -c:a pcm_s16le output.wav`");
}

WavData read_wav_f32(std::string_view input) {
    ReadOnlyMemoryStreamBuffer buffer(input);
    std::istream stream(&buffer);
    return read_wav_f32(stream);
}

WavData read_wav_f32(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open WAV input: " + path.string());
    }

    return read_wav_f32(input);
}

}  // namespace engine::audio
