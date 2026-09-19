#ifndef __SD_CORE_REGEX_H__
#define __SD_CORE_REGEX_H__

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sd {

    class Regex {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        using Match = std::pair<size_t, size_t>;

        Regex();
        ~Regex();
        Regex(Regex&&) noexcept;
        Regex& operator=(Regex&&) noexcept;

        // Failed compilation leaves the previous expression intact.
        bool compile(const std::string& pattern, std::string* error = nullptr);
        // Matches are non-overlapping UTF-8 byte ranges; each call owns its search state.
        bool find_matches(const std::string& text, std::vector<Match>& matches, std::string* error = nullptr) const;
    };

}  // namespace sd

#endif  // __SD_CORE_REGEX_H__
