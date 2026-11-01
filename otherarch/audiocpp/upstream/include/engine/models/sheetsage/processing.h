#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::sheetsage {

struct SheetSage2Note {
    int pitch = 0;
    int track = 0;
    int duration_bin = 0;
    int duration_steps = 0;
};

struct SheetSage2Event {
    int64_t subbeat = 0;
    int64_t source_subbeat = 0;
    int64_t global_subbeat = 0;
    int window_index = 0;
    float window_start = 0.0F;
    float time = 0.0F;
    std::optional<float> timestamp;
    std::optional<std::pair<int, int>> meter;
    std::optional<int64_t> eighth_position;
    std::optional<std::string> structure;
    std::optional<std::string> key;
    std::optional<std::string> chord;
    std::vector<SheetSage2Note> notes;
    std::vector<float> note_end_times;
    std::vector<int32_t> timestamp_tokens;
    std::vector<int32_t> rhythm_tokens;
    std::vector<int32_t> structure_tokens;
    std::vector<int32_t> key_tokens;
    std::vector<int32_t> chord_tokens;
    std::vector<int32_t> melody_tokens;
};

int64_t sheetsage2_structure_label_count();
int64_t sheetsage2_duration_bin_count();
std::string sheetsage2_structure_label(int64_t index);
std::string sheetsage2_key_label(int64_t index);
std::string sheetsage2_chord_label(bool full_chord, int64_t index);
SheetSage2Note sheetsage2_note_from_pitch_duration(int pitch_id, int64_t duration_bin);

std::string events_to_abc(const std::vector<SheetSage2Event> & events, double duration);
std::string events_json(const std::vector<SheetSage2Event> & events);

}  // namespace engine::models::sheetsage
