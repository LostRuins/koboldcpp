#include "engine/models/yue2/types.h"

#include <stdexcept>

namespace engine::models::yue2 {

const char * cot_mode_name(Yue2CotMode mode) noexcept {
    switch (mode) {
        case Yue2CotMode::Off:
            return "off";
        case Yue2CotMode::Melody:
            return "melody";
        case Yue2CotMode::Full:
            return "full";
    }
    return "full";
}

Yue2CotMode parse_cot_mode(const std::string & value) {
    if (value == "off") {
        return Yue2CotMode::Off;
    }
    if (value == "melody") {
        return Yue2CotMode::Melody;
    }
    if (value == "full") {
        return Yue2CotMode::Full;
    }
    throw std::runtime_error("yue2.cot must be one of off, melody, or full");
}

const char * cot_instruction(Yue2CotMode mode) noexcept {
    switch (mode) {
        case Yue2CotMode::Off:
            return "Generate music with codec tokens from the given conditions.";
        case Yue2CotMode::Melody:
            return "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.";
        case Yue2CotMode::Full:
            return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";
    }
    return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";
}

const char * stop_after_name(Yue2StopAfter stage) noexcept {
    switch (stage) {
        case Yue2StopAfter::Abc:
            return "abc";
        case Yue2StopAfter::Semantic:
            return "semantic";
        case Yue2StopAfter::Audio:
            return "audio";
    }
    return "audio";
}

Yue2StopAfter parse_stop_after(const std::string & value) {
    if (value == "abc") {
        return Yue2StopAfter::Abc;
    }
    if (value == "semantic") {
        return Yue2StopAfter::Semantic;
    }
    if (value == "audio") {
        return Yue2StopAfter::Audio;
    }
    throw std::runtime_error("yue2.stop_after must be one of abc, semantic, or audio");
}

float request_guidance_scale(const Yue2Request & request) noexcept {
    if (request.cfg_scale >= 0.0F) {
        return request.cfg_scale;
    }
    return request.cot == Yue2CotMode::Off ? 1.01F : 1.0F;
}

std::string semantic_codes_to_json(const std::vector<int32_t> & codes) {
    std::string out;
    out.reserve(codes.size() * 6 + 2);
    out.push_back('[');
    for (size_t i = 0; i < codes.size(); ++i) {
        if (i != 0) {
            out.push_back(',');
        }
        out += std::to_string(codes[i]);
    }
    out.push_back(']');
    return out;
}

}  // namespace engine::models::yue2
