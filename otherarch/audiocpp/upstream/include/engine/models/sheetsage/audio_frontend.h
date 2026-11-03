#pragma once

#include <map>
#include <utility>
#include <vector>

namespace engine::models::sheetsage {

class SheetSage2AudioFrontend {
public:
    std::vector<float> prepare(
        const std::vector<float> & interleaved,
        int source_rate,
        int channels,
        int target_rate,
        int threads);

private:
    std::map<std::pair<int, int>, std::vector<float>> filters_;
};

}  // namespace engine::models::sheetsage
