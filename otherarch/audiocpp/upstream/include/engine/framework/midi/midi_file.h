#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace engine::midi {

constexpr int kDrumProgram = 128;

struct MidiNote {
    bool is_drum = false;
    int program = 0;
    double onset_seconds = 0.0;
    double offset_seconds = 0.0;
    int pitch = 0;
    std::string track_name;
};

struct MidiFileOptions {
    int ticks_per_beat = 480;
    int tempo_us_per_beat = 500000;
    int velocity = 100;
};

std::vector<MidiNote> clean_midi_notes(std::vector<MidiNote> notes);
std::vector<std::byte> write_standard_midi_file(
    const std::vector<MidiNote> & notes,
    const MidiFileOptions & options = {});

}  // namespace engine::midi
