#pragma once

#include "engine/models/supertonic/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::supertonic {

struct SupertonicTextInputs {
    std::vector<int64_t> ids;
    std::vector<float> mask;
    int64_t length = 0;
};

class SupertonicTextTokenizer {
public:
    explicit SupertonicTextTokenizer(std::shared_ptr<const SupertonicAssets> assets);

    SupertonicTextInputs encode(const std::string & text, const std::string & language) const;

private:
    std::string preprocess(const std::string & text, const std::string & language) const;
    std::vector<uint32_t> utf8_to_codepoints(const std::string & text) const;

    std::shared_ptr<const SupertonicAssets> assets_;
};

}  // namespace engine::models::supertonic
