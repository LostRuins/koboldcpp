#include "regex.h"

#include <cstdint>
#include <limits>
#include <mutex>

#define ONIG_ESCAPE_UCHAR_COLLISION
#define ONIG_ESCAPE_REGEX_T_COLLISION
#include <oniguruma.h>

namespace sd {

    struct Regex::Impl {
        OnigRegex regex = nullptr;

        ~Impl() {
            if (regex) {
                onig_free(regex);
            }
        }
    };

    struct RegexRegionDeleter {
        void operator()(OnigRegion* region) const {
            onig_region_free(region, 1);
        }
    };

    static bool regex_error(std::string* error, const std::string& message) {
        if (error) {
            *error = message;
        }
        return false;
    }

    static bool regex_onig_error(std::string* error, int code, OnigErrorInfo* info = nullptr) {
        OnigUChar buffer[ONIG_MAX_ERROR_MESSAGE_LEN];
        onig_error_code_to_str(buffer, code, info);
        return regex_error(error, reinterpret_cast<const char*>(buffer));
    }

    static int regex_initialize() {
        static std::once_flag once;
        static int result = ONIG_NORMAL;
        std::call_once(once, [] {
            OnigEncoding encodings[] = {ONIG_ENCODING_UTF8};
            result                   = onig_initialize(encodings, 1);
        });
        // onig_end() would invalidate expressions held by other Regex instances.
        return result;
    }

    // Rust str excludes overlong encodings, surrogates and extended UTF-8 accepted by Oniguruma.
    static bool regex_valid_utf8(const std::string& text) {
        size_t position = 0;
        while (position < text.size()) {
            const auto lead = static_cast<unsigned char>(text[position++]);
            if (lead < 0x80) {
                continue;
            }
            int count = 0;
            if (lead >= 0xC2 && lead <= 0xDF) {
                count = 1;
            } else if (lead >= 0xE0 && lead <= 0xEF) {
                count = 2;
            } else if (lead >= 0xF0 && lead <= 0xF4) {
                count = 3;
            }
            if (count == 0 || text.size() - position < static_cast<size_t>(count)) {
                return false;
            }
            uint32_t codepoint = lead & (0x7F >> count);
            for (int i = 0; i < count; ++i) {
                const auto byte = static_cast<unsigned char>(text[position++]);
                if ((byte & 0xC0) != 0x80) {
                    return false;
                }
                codepoint = (codepoint << 6) | (byte & 0x3F);
            }
            constexpr uint32_t minimum[] = {0, 0x80, 0x800, 0x10000};
            if (codepoint < minimum[count] || codepoint > 0x10FFFF || (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
                return false;
            }
        }
        return true;
    }

    Regex::Regex()                            = default;
    Regex::~Regex()                           = default;
    Regex::Regex(Regex&&) noexcept            = default;
    Regex& Regex::operator=(Regex&&) noexcept = default;

    bool Regex::compile(const std::string& pattern, std::string* error) {
        if (error) {
            error->clear();
        }
        const int initialized = regex_initialize();
        if (initialized != ONIG_NORMAL) {
            return regex_onig_error(error, initialized);
        }
        if (pattern.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return regex_error(error, "regex pattern exceeds Oniguruma's offset range");
        }
        const auto* begin = reinterpret_cast<const OnigUChar*>(pattern.data());
        const auto* end   = begin + pattern.size();
        if (!regex_valid_utf8(pattern)) {
            return regex_error(error, "regex pattern is not valid UTF-8");
        }

        auto next = std::make_unique<Impl>();
        OnigErrorInfo info{};
        static std::mutex compile_mutex;
        std::lock_guard<std::mutex> lock(compile_mutex);
        const int result = onig_new(&next->regex, begin, end, ONIG_OPTION_NONE,
                                    ONIG_ENCODING_UTF8, ONIG_SYNTAX_ONIGURUMA, &info);
        if (result != ONIG_NORMAL) {
            return regex_onig_error(error, result, &info);
        }
        impl_ = std::move(next);
        return true;
    }

    bool Regex::find_matches(const std::string& text, std::vector<Match>& matches, std::string* error) const {
        matches.clear();
        if (error) {
            error->clear();
        }
        if (!impl_) {
            return regex_error(error, "regex has not been compiled");
        }
        if (text.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return regex_error(error, "regex input exceeds Oniguruma's offset range");
        }
        const auto* begin = reinterpret_cast<const OnigUChar*>(text.data());
        const auto* end   = begin + text.size();
        if (!regex_valid_utf8(text)) {
            return regex_error(error, "regex input is not valid UTF-8");
        }
        std::unique_ptr<OnigRegion, RegexRegionDeleter> region(onig_region_new());
        if (!region) {
            return regex_error(error, "failed to allocate regex match region");
        }

        size_t position = 0;
        while (position <= text.size()) {
            const int result = onig_search(impl_->regex, begin, end, begin + position, end,
                                           region.get(), ONIG_OPTION_NONE);
            if (result == ONIG_MISMATCH) {
                break;
            }
            if (result < 0) {
                matches.clear();
                return regex_onig_error(error, result);
            }
            const size_t match_begin = static_cast<size_t>(region->beg[0]);
            const size_t match_end   = static_cast<size_t>(region->end[0]);
            // Match rust-onig's find_iter: suppress an empty match at the previous match's end.
            if (match_begin == match_end && !matches.empty() && matches.back().second == match_end) {
                if (position == text.size()) {
                    break;
                }
                position += static_cast<size_t>(ONIGENC_MBC_ENC_LEN(ONIG_ENCODING_UTF8, begin + position));
                continue;
            }
            matches.emplace_back(match_begin, match_end);
            position = match_end;
        }
        return true;
    }

}  // namespace sd
