#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::text {

struct SubtitleFormatOptions {
    int sample_rate = 0;
    int max_chars_per_line = 42;
    int max_lines = 2;
    double max_block_seconds = 6.0;
    double max_gap_seconds = 1.0;
};

struct SubtitleBlock {
    int index = 0;
    engine::runtime::TimeSpan span;
    std::vector<std::string> lines;
};

std::vector<SubtitleBlock> build_subtitle_blocks(
    const std::vector<engine::runtime::WordTimestamp> & words,
    const SubtitleFormatOptions & options);

std::string format_srt(
    const std::vector<engine::runtime::WordTimestamp> & words,
    const SubtitleFormatOptions & options);

}  // namespace engine::text
