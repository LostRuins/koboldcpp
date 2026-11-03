#include "engine/models/sheetsage/processing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::sheetsage {
namespace {

constexpr std::array<const char *, 23> kStructureLabels = {
    "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus",
    "interlude", "fade-out", "loop", "rap", "preshot", "irregular", "instrumental",
    "intro and verse", "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme",
    "development", "variation", "pre-outro",
};

constexpr std::array<int, 24> kDurationTemplates = {
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
};

constexpr std::array<const char *, 12> kChromaticSharps = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
};

const std::vector<std::string> & full_chord_labels() {
    static const std::vector<std::string> labels = [] {
        constexpr std::array<const char *, 15> qualities = {
            "maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7",
            "dim7", "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6",
        };
        const std::map<std::string, std::vector<const char *>> inversions = {
            {"maj", {"/2", "/3", "/5"}},
            {"min", {"/2", "/b3", "/5"}},
            {"maj7", {"/3", "/5", "/7"}},
            {"min7", {"/b3", "/5", "/b7"}},
            {"7", {"/3", "/5", "/b7"}},
        };
        std::vector<std::string> out;
        out.reserve(361);
        out.push_back("N");
        for (const auto * quality : qualities) {
            const auto it = inversions.find(quality);
            std::vector<const char *> suffixes;
            if (it != inversions.end()) {
                suffixes = it->second;
            }
            suffixes.push_back("");
            for (const auto * root : kChromaticSharps) {
                for (const auto * inversion : suffixes) {
                    out.push_back(std::string(root) + ":" + quality + inversion);
                }
            }
        }
        return out;
    }();
    return labels;
}

int key_accidental_count(const std::string & key) {
    static const std::map<std::string, int> values = {
        {"C", 0}, {"G", 1}, {"D", 2}, {"A", 3}, {"E", 4}, {"B", 5}, {"F#", 6}, {"C#", 7},
        {"F", -1}, {"Bb", -2}, {"Eb", -3}, {"Ab", -4}, {"Db", -5}, {"Gb", -6}, {"Cb", -7},
        {"Am", 0}, {"Em", 1}, {"Bm", 2}, {"F#m", 3}, {"C#m", 4}, {"G#m", 5}, {"D#m", 6}, {"A#m", 7},
        {"Dm", -1}, {"Gm", -2}, {"Cm", -3}, {"Fm", -4}, {"Bbm", -5}, {"Ebm", -6}, {"Abm", -7},
    };
    const auto it = values.find(key);
    return it == values.end() ? 0 : it->second;
}

std::array<int, 7> key_accidentals(const std::string & key) {
    std::array<int, 7> accidentals{};
    const int count = key_accidental_count(key);
    const std::string order = count > 0 ? "FCGDAEB" : "BEADGCF";
    for (int i = 0; i < std::abs(count); ++i) {
        const auto index = std::string("CDEFGAB").find(order[static_cast<size_t>(i)]);
        accidentals[index] = count > 0 ? 1 : -1;
    }
    return accidentals;
}

std::string key_relative_pitch_name(int accidental_count, int pitch_class) {
    static const std::map<int, std::array<const char *, 12>> names = {
        {7, {"B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"}},
        {6, {"B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"}},
        {5, {"B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"}},
        {4, {"B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"}},
        {3, {"B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"}},
        {2, {"C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"}},
        {1, {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}},
        {0, {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"}},
        {-1, {"C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"}},
        {-2, {"C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"}},
        {-3, {"C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"}},
        {-4, {"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"}},
        {-5, {"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"}},
        {-6, {"C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"}},
        {-7, {"C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"}},
    };
    return names.at(accidental_count)[static_cast<size_t>(pitch_class)];
}

std::string note_to_abc(
    int midi,
    const std::array<int, 7> & key_accidentals,
    std::map<int, int> & measure_accidentals) {
    const int accidental_count = std::accumulate(key_accidentals.begin(), key_accidentals.end(), 0);
    const int pitch_class = ((midi % 12) + 12) % 12;
    const std::string pitch_name = key_relative_pitch_name(accidental_count, pitch_class);
    const char letter = pitch_name[0];
    const std::string accidental = pitch_name.substr(1);
    const int accidental_number =
        accidental == "bb" ? -2 : accidental == "b" ? -1 : accidental == "#" ? 1 : accidental == "##" ? 2 : 0;
    int octave = (midi - 60) / 12;
    if (pitch_class == 11 && accidental_number == -1) {
        ++octave;
    } else if (pitch_class == 0 && accidental_number == 1) {
        --octave;
    }
    const int scale_index = static_cast<int>(std::string("CDEFGAB").find(letter));
    const auto it = measure_accidentals.find(scale_index);
    const int current = it == measure_accidentals.end() ? key_accidentals[static_cast<size_t>(scale_index)] : it->second;
    std::string prefix;
    if (current != accidental_number) {
        measure_accidentals[scale_index] = accidental_number;
        prefix = accidental_number == -2 ? "__" :
                 accidental_number == -1 ? "_" :
                 accidental_number == 0 ? "=" :
                 accidental_number == 1 ? "^" : "^^";
    }
    std::string note(1, letter);
    if (octave > 0) {
        note[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(note[0])));
        note.append(static_cast<size_t>(std::max(0, octave - 1)), '\'');
    } else if (octave < 0) {
        note.append(static_cast<size_t>(-octave), ',');
    }
    return prefix + note;
}

bool same_note_segment(int value, int next_value) {
    if (value == 0) {
        return next_value == 0;
    }
    const int pitch = value / 2 - 1;
    return next_value == pitch * 2 + 2;
}

bool continues_pitch(int value, int next_value) {
    if (value <= 0) {
        return false;
    }
    const int pitch = value / 2 - 1;
    return next_value == pitch * 2 + 2;
}

std::string abc_duration(int units) {
    return units == 1 ? std::string{} : std::to_string(units);
}

std::string key_symbol_to_abc(std::string key) {
    const auto colon = key.find(':');
    if (colon != std::string::npos) {
        auto root = key.substr(0, colon);
        const auto mode = key.substr(colon + 1);
        // The tokenizer uses sharp roots, including nonportable major keys.
        if (mode == "major") {
            static const std::map<std::string, std::string> portable = {
                {"A#", "Bb"}, {"D#", "Eb"}, {"G#", "Ab"},
            };
            const auto it = portable.find(root);
            if (it != portable.end()) root = it->second;
        }
        return root + (mode == "minor" ? "m" : "");
    }
    return key;
}

std::string chord_symbol_to_abc(const std::string & chord) {
    if (chord.empty() || chord == "N" || chord == "X" || chord == "?") {
        return {};
    }
    const auto colon = chord.find(':');
    if (colon == std::string::npos) {
        return {};
    }
    const auto root = chord.substr(0, colon);
    auto quality = chord.substr(colon + 1);
    const auto slash = quality.find('/');
    std::string bass;
    if (slash != std::string::npos) {
        bass = quality.substr(slash + 1);
        quality = quality.substr(0, slash);
        constexpr std::array<int, 7> natural = {0, 2, 4, 5, 7, 9, 11};
        const std::string letters = "CDEFGAB";
        const int root_letter = static_cast<int>(letters.find(root[0]));
        const int root_pitch = natural[static_cast<size_t>(root_letter)] +
            static_cast<int>(std::count(root.begin(), root.end(), '#')) -
            static_cast<int>(std::count(root.begin(), root.end(), 'b'));
        const auto degree_start = bass.find_first_of("123456789");
        const int degree = std::stoi(bass.substr(degree_start));
        const int alteration = static_cast<int>(std::count(bass.begin(), bass.end(), '#')) -
            static_cast<int>(std::count(bass.begin(), bass.end(), 'b'));
        const int letter = (root_letter + degree - 1) % 7;
        const int pitch = (root_pitch + natural[static_cast<size_t>((degree - 1) % 7)] + alteration + 12) % 12;
        const int accidental = (pitch - natural[static_cast<size_t>(letter)] + 18) % 12 - 6;
        bass = std::string(1, letters[static_cast<size_t>(letter)]) +
            std::string(static_cast<size_t>(std::abs(accidental)), accidental < 0 ? 'b' : '#');
    }
    static const std::map<std::string, std::string> quality_map = {
        {"maj", ""}, {"min", "m"}, {"dim", "dim"}, {"aug", "aug"}, {"7", "7"},
        {"maj7", "maj7"}, {"min7", "m7"}, {"dim7", "dim7"}, {"hdim7", "m7b5"},
        {"sus4", "sus4"}, {"sus2", "sus2"}, {"maj6", "6"}, {"min6", "m6"},
        {"sus4(b7)", "7sus4"}, {"minmaj7", "m(maj7)"},
    };
    const auto it = quality_map.find(quality);
    if (it == quality_map.end()) {
        return {};
    }
    return bass.empty() ? root + it->second : root + it->second + "/" + bass;
}

struct BeatRow {
    double time = 0.0;
    int beat = 1;
    int numerator = 4;
    int denominator = 4;
};

struct NotationMeasure {
    int start_beat = 0;
    int end_beat = 0;
    int numerator = 4;
    int denominator = 4;
    int abc_numerator = 4;
    int abc_denominator = 4;
    bool pad_before = false;
};

struct TextInterval {
    double start = 0.0;
    double end = 0.0;
    std::string value;
};

std::vector<BeatRow> rhythm_rows(const std::vector<SheetSage2Event> & events) {
    std::vector<BeatRow> rows;
    std::optional<std::pair<int, int>> meter;
    for (const auto & event : events) {
        if (event.meter.has_value()) {
            meter = event.meter;
        }
        if (!event.eighth_position.has_value() || !meter.has_value()) {
            continue;
        }
        const int64_t scaled = *event.eighth_position * meter->second;
        if (scaled % 8 != 0) {
            continue;
        }
        const int beat = static_cast<int>(scaled / 8) + 1;
        if (beat < 1 || beat > meter->first) {
            continue;
        }
        rows.push_back({event.time, beat, meter->first, meter->second});
    }
    return rows;
}

std::vector<TextInterval> interval_rows(
    const std::vector<SheetSage2Event> & events,
    const char * field,
    double duration) {
    std::vector<TextInterval> rows;
    for (const auto & event : events) {
        const std::optional<std::string> * value = nullptr;
        if (std::string(field) == "key") {
            value = &event.key;
        } else if (std::string(field) == "chord") {
            value = &event.chord;
        } else if (std::string(field) == "structure") {
            value = &event.structure;
        }
        if (value != nullptr && value->has_value()) {
            rows.push_back({event.time, duration, **value});
        }
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i].end = i + 1 < rows.size() ? rows[i + 1].start : duration;
    }
    rows.erase(
        std::remove_if(rows.begin(), rows.end(), [](const auto & row) { return row.end <= row.start; }),
        rows.end());
    return rows;
}

std::vector<NotationMeasure> infer_measures(const std::vector<BeatRow> & beats) {
    std::vector<int> downbeats;
    for (size_t i = 0; i < beats.size(); ++i) {
        if (beats[i].beat == 1) {
            downbeats.push_back(static_cast<int>(i));
        }
    }
    if (downbeats.empty()) {
        return {};
    }
    std::vector<NotationMeasure> measures;
    const auto add_span = [&](int start, int end) {
        if (end <= start) {
            return;
        }
        std::map<int, int> denom_counts;
        std::map<int, int> numerator_counts;
        for (int i = start; i < end; ++i) {
            ++denom_counts[beats[static_cast<size_t>(i)].denominator];
            ++numerator_counts[beats[static_cast<size_t>(i)].numerator];
        }
        const auto best = [](const std::map<int, int> & counts, int fallback) {
            int value = fallback;
            int count = -1;
            for (const auto & [candidate, observed] : counts) {
                if (observed > count) {
                    value = candidate;
                    count = observed;
                }
            }
            return value;
        };
        const int beat_count = end - start;
        const int denominator = best(denom_counts, 4);
        const int declared = best(numerator_counts, beat_count);
        NotationMeasure measure;
        measure.start_beat = start;
        measure.end_beat = end;
        measure.numerator = beat_count;
        measure.denominator = denominator;
        measure.abc_numerator = (end == static_cast<int>(beats.size()) - 1 && declared >= beat_count) ? declared : beat_count;
        measure.abc_denominator = denominator;
        measures.push_back(measure);
    };
    if (downbeats.front() > 0) {
        add_span(0, downbeats.front());
    }
    for (size_t i = 1; i < downbeats.size(); ++i) {
        add_span(downbeats[i - 1], downbeats[i]);
    }
    if (downbeats.back() < static_cast<int>(beats.size()) - 1) {
        add_span(downbeats.back(), static_cast<int>(beats.size()) - 1);
    }
    if (measures.size() >= 2) {
        const double first = static_cast<double>(measures.front().numerator) / measures.front().denominator;
        const double following = static_cast<double>(measures[1].abc_numerator) / measures[1].abc_denominator;
        if (first < following) {
            measures.front().abc_numerator = measures[1].abc_numerator;
            measures.front().abc_denominator = measures[1].abc_denominator;
            measures.front().pad_before = true;
        }
    }
    return measures;
}

std::vector<double> subbeat_times_from_beats(const std::vector<BeatRow> & beats) {
    std::vector<double> times;
    if (beats.size() < 2) {
        return times;
    }
    for (size_t i = 0; i + 1 < beats.size(); ++i) {
        const double start = beats[i].time;
        const double end = beats[i + 1].time;
        for (int j = 0; j < 4; ++j) {
            times.push_back(start + (end - start) * static_cast<double>(j) / 4.0);
        }
    }
    times.push_back(beats.back().time);
    return times;
}

int quantize_time(double time, const std::vector<double> & subbeat_times) {
    if (subbeat_times.size() < 2) {
        return 0;
    }
    std::vector<double> boundaries;
    boundaries.reserve(subbeat_times.size() - 1);
    for (size_t i = 0; i + 1 < subbeat_times.size(); ++i) {
        boundaries.push_back((subbeat_times[i] + subbeat_times[i + 1]) * 0.5);
    }
    return static_cast<int>(std::distance(
        boundaries.begin(),
        std::lower_bound(boundaries.begin(), boundaries.end(), time)));
}

std::vector<std::string> fill_text_intervals(
    const std::vector<TextInterval> & rows,
    const std::vector<double> & subbeat_times,
    const std::string & fallback) {
    std::vector<std::string> out(subbeat_times.size(), fallback);
    for (const auto & row : rows) {
        int start = std::clamp(quantize_time(row.start, subbeat_times), 0, static_cast<int>(out.size()) - 1);
        int end = std::clamp(quantize_time(row.end, subbeat_times), 0, static_cast<int>(out.size()) - 1);
        if (end <= start) {
            continue;
        }
        std::fill(out.begin() + start, out.begin() + end, row.value);
    }
    if (out.size() > 1) {
        out.back() = out[out.size() - 2];
    }
    return out;
}

std::array<std::vector<int>, 2> build_voice_arrays(
    const std::vector<SheetSage2Event> & events,
    const std::vector<double> & subbeat_times,
    double duration) {
    std::array<std::vector<int>, 2> voices = {
        std::vector<int>(subbeat_times.size(), 0),
        std::vector<int>(subbeat_times.size(), 0),
    };
    struct TimedNote {
        double start = 0.0;
        double end = 0.0;
        int pitch = 0;
        int track = 0;
    };
    std::vector<TimedNote> notes;
    for (const auto & event : events) {
        for (size_t i = 0; i < event.notes.size(); ++i) {
            const double end = i < event.note_end_times.size()
                                 ? event.note_end_times[i]
                                 : std::min(duration, static_cast<double>(event.time) + 0.04);
            notes.push_back({event.time, std::min(duration, end), event.notes[i].pitch, event.notes[i].track});
        }
    }
    std::sort(notes.begin(), notes.end(), [](const auto & a, const auto & b) {
        if (a.track != b.track) {
            return a.track < b.track;
        }
        if (a.start != b.start) {
            return a.start < b.start;
        }
        if (a.pitch != b.pitch) {
            return a.pitch < b.pitch;
        }
        return a.end < b.end;
    });
    for (size_t i = 0; i < notes.size(); ++i) {
        if (i + 1 < notes.size() && notes[i].track == notes[i + 1].track && notes[i].end > notes[i + 1].start) {
            notes[i].end = notes[i + 1].start;
        }
        if (notes[i].end <= notes[i].start) {
            continue;
        }
        const int track = std::clamp(notes[i].track, 0, 1);
        int start = std::clamp(quantize_time(notes[i].start, subbeat_times), 0, static_cast<int>(subbeat_times.size()) - 1);
        int end = std::clamp(quantize_time(notes[i].end, subbeat_times), 0, static_cast<int>(subbeat_times.size()) - 1);
        if (end <= start) {
            end = std::min<int>(static_cast<int>(subbeat_times.size()) - 1, start + 1);
        }
        const int sustain = notes[i].pitch * 2 + 2;
        for (int t = start; t < end; ++t) {
            if (voices[static_cast<size_t>(track)][static_cast<size_t>(t)] == 0) {
                voices[static_cast<size_t>(track)][static_cast<size_t>(t)] = sustain;
            }
        }
        voices[static_cast<size_t>(track)][static_cast<size_t>(start)] = sustain + 1;
    }
    return voices;
}

std::vector<int> subbeat_denominators_from_measures(
    const std::vector<BeatRow> & beats,
    const std::vector<NotationMeasure> & measures,
    size_t subbeat_count) {
    std::vector<int> out(subbeat_count, 4);
    for (const auto & measure : measures) {
        for (int beat = measure.start_beat; beat < measure.end_beat; ++beat) {
            for (int j = 0; j < 4; ++j) {
                const size_t index = static_cast<size_t>(beat * 4 + j);
                if (index < out.size()) {
                    out[index] = measure.denominator;
                }
            }
        }
    }
    if (out.size() > 1) {
        out.back() = out[out.size() - 2];
    }
    (void)beats;
    return out;
}

int duration_units(
    const std::vector<int> & denominators,
    int start_t,
    int end_t,
    int unit_denominator) {
    int units = 0;
    for (int t = start_t; t < end_t; ++t) {
        const int divisor = denominators[static_cast<size_t>(t)] * 4;
        units += unit_denominator / divisor;
    }
    return std::max(1, units);
}

std::vector<int> split_duration_units(int duration) {
    static constexpr std::array<int, 11> supported = {48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1};
    std::vector<int> out;
    int remaining = std::max(1, duration);
    while (remaining > 0) {
        int chunk = 1;
        for (const int candidate : supported) {
            if (candidate <= remaining) {
                chunk = candidate;
                break;
            }
        }
        out.push_back(chunk);
        remaining -= chunk;
    }
    return out;
}

std::vector<std::string> render_duration_tokens(
    const std::string & prefix,
    const std::string & note,
    int duration,
    bool tie_out) {
    const auto chunks = split_duration_units(duration);
    std::vector<std::string> out;
    out.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        const bool tie = note != "z" && (i + 1 < chunks.size() || tie_out);
        out.push_back((i == 0 ? prefix : std::string{}) + note + abc_duration(chunks[i]) + (tie ? "-" : ""));
    }
    return out;
}

std::string render_voice_measure(
    const std::vector<int> & voice,
    const std::vector<std::string> & keys,
    const std::vector<std::string> & chords,
    const std::vector<int> & denominators,
    const NotationMeasure & measure,
    int unit_denominator,
    bool show_chords) {
    std::ostringstream out;
    int t = measure.start_beat * 4;
    const int end_t = measure.end_beat * 4;
    std::string current_key = keys[static_cast<size_t>(t)];
    auto active_key_accidentals = key_accidentals(current_key);
    std::map<int, int> measure_accidentals;
    int padding = measure.abc_numerator * unit_denominator / measure.abc_denominator -
                  measure.numerator * unit_denominator / measure.denominator;
    int leading_padding = measure.pad_before ? padding : 0;
    if (measure.pad_before) {
        padding = 0;
    }
    while (t < end_t && t < static_cast<int>(voice.size())) {
        int next_t = std::min(end_t, static_cast<int>(voice.size()));
        for (int probe = t + 1; probe < next_t; ++probe) {
            const bool note_change = !same_note_segment(
                voice[static_cast<size_t>(probe - 1)],
                voice[static_cast<size_t>(probe)]);
            const bool attack = voice[static_cast<size_t>(probe)] > 0 && voice[static_cast<size_t>(probe)] % 2 == 1;
            const bool key_change = keys[static_cast<size_t>(probe)] != keys[static_cast<size_t>(probe - 1)];
            const bool chord_change = show_chords && chords[static_cast<size_t>(probe)] != chords[static_cast<size_t>(probe - 1)];
            if (note_change || attack || key_change || chord_change) {
                next_t = probe;
                break;
            }
        }
        std::string prefix;
        if (t > measure.start_beat * 4 && keys[static_cast<size_t>(t)] != keys[static_cast<size_t>(t - 1)]) {
            current_key = keys[static_cast<size_t>(t)];
            active_key_accidentals = key_accidentals(current_key);
            measure_accidentals.clear();
            prefix += "[K:" + current_key + "]";
        }
        if (show_chords && (t == measure.start_beat * 4 ||
                            chords[static_cast<size_t>(t)] != chords[static_cast<size_t>(t - 1)])) {
            const auto chord = chord_symbol_to_abc(chords[static_cast<size_t>(t)]);
            if (!chord.empty()) {
                prefix += "\"" + chord + "\"";
            }
        }
        const int value = voice[static_cast<size_t>(t)];
        const std::string note = value == 0 ? "z" : note_to_abc(value / 2 - 1, active_key_accidentals, measure_accidentals);
        int units = duration_units(denominators, t, next_t, unit_denominator);
        if (t == measure.start_beat * 4 && leading_padding > 0) {
            if (value == 0 && prefix.empty()) {
                units += leading_padding;
            } else {
                for (const auto & token : render_duration_tokens("", "z", leading_padding, false)) {
                    out << token;
                }
            }
            leading_padding = 0;
        }
        if (value == 0 && next_t == end_t && padding > 0) {
            units += padding;
            padding = 0;
        }
        const bool tie_out = value > 0 && next_t < static_cast<int>(voice.size()) &&
                             continues_pitch(value, voice[static_cast<size_t>(next_t)]);
        for (const auto & token : render_duration_tokens(prefix, note, units, tie_out)) {
            out << token;
        }
        t = next_t;
    }
    if (padding > 0) {
        for (const auto & token : render_duration_tokens("", "z", padding, false)) {
            out << token;
        }
    }
    return out.str();
}

bool is_full_rest_measure(const std::string & text) {
    return !text.empty() && text.find_first_not_of("z0123456789") == std::string::npos;
}

std::string render_voice_group(
    const std::vector<int> & voice,
    const std::vector<std::string> & keys,
    const std::vector<std::string> & chords,
    const std::vector<int> & denominators,
    const std::vector<NotationMeasure> & measures,
    int unit_denominator,
    bool show_chords) {
    std::ostringstream out;
    std::vector<std::string> rendered;
    rendered.reserve(measures.size());
    for (const auto & measure : measures) {
        rendered.push_back(render_voice_measure(
            voice,
            keys,
            chords,
            denominators,
            measure,
            unit_denominator,
            show_chords));
    }
    size_t index = 0;
    while (index < rendered.size()) {
        if (!is_full_rest_measure(rendered[index])) {
            out << rendered[index] << "|";
            ++index;
            continue;
        }
        size_t end = index + 1;
        while (end < rendered.size() && is_full_rest_measure(rendered[end])) {
            ++end;
        }
        const size_t count = end - index;
        out << "Z";
        if (count > 1) {
            out << count;
        }
        out << "|";
        index = end;
    }
    return out.str();
}

struct MeasureGroup {
    std::vector<NotationMeasure> measures;
    std::vector<std::string> structure_labels;
    bool meter_changed = false;
    bool key_changed = false;
};

std::string sanitize_structure_label(const std::string & value) {
    std::istringstream in(value);
    std::ostringstream out;
    std::string part;
    while (in >> part) {
        if (out.tellp() > 0) {
            out << " ";
        }
        out << part;
    }
    return out.str();
}

std::vector<MeasureGroup> measure_groups(
    const std::vector<NotationMeasure> & measures,
    const std::vector<std::string> & keys,
    const std::vector<std::pair<int, std::string>> & structure_events) {
    if (measures.empty()) {
        return {};
    }
    std::pair<int, int> active_meter{measures.front().abc_numerator, measures.front().abc_denominator};
    std::string active_key = keys[static_cast<size_t>(measures.front().start_beat * 4)];
    std::string active_structure;
    std::vector<MeasureGroup> groups;
    for (const auto & measure : measures) {
        const auto meter = std::make_pair(measure.abc_numerator, measure.abc_denominator);
        const std::string key = keys[static_cast<size_t>(measure.start_beat * 4)];
        std::vector<std::string> labels;
        for (const auto & [t, label] : structure_events) {
            if (t < measure.start_beat * 4 || t >= measure.end_beat * 4) {
                continue;
            }
            const auto clean = sanitize_structure_label(label);
            if (!clean.empty() && clean != active_structure) {
                labels.push_back(clean);
                active_structure = clean;
            }
        }
        const bool meter_changed = meter != active_meter;
        const bool key_changed = key != active_key;
        const bool start_group =
            groups.empty() ||
            groups.back().measures.size() >= 4 ||
            meter_changed ||
            key_changed ||
            !labels.empty();
        if (start_group) {
            groups.push_back({{measure}, labels, meter_changed, key_changed});
        } else {
            groups.back().measures.push_back(measure);
        }
        active_meter = meter;
        active_key = keys[static_cast<size_t>(std::max(0, measure.end_beat * 4 - 1))];
    }
    return groups;
}

}  // namespace

int64_t sheetsage2_structure_label_count() {
    return static_cast<int64_t>(kStructureLabels.size());
}

int64_t sheetsage2_duration_bin_count() {
    return static_cast<int64_t>(kDurationTemplates.size());
}

std::string sheetsage2_structure_label(int64_t index) {
    index = std::clamp<int64_t>(index, 0, static_cast<int64_t>(kStructureLabels.size() - 1));
    return kStructureLabels[static_cast<size_t>(index)];
}

std::string sheetsage2_key_label(int64_t index) {
    return std::string(kChromaticSharps[static_cast<size_t>(index % 12)]) +
           (index >= 12 ? ":minor" : ":major");
}

std::string sheetsage2_chord_label(bool full_chord, int64_t index) {
    if (full_chord) {
        const auto & labels = full_chord_labels();
        index = std::clamp<int64_t>(index, 0, static_cast<int64_t>(labels.size() - 1));
        return labels[static_cast<size_t>(index)];
    }
    if (index == 0) {
        return "N";
    }
    const int64_t root = (index - 1) % 12;
    const bool minor = (index - 1) >= 12;
    return std::string(kChromaticSharps[static_cast<size_t>(root)]) + (minor ? ":min" : ":maj");
}

SheetSage2Note sheetsage2_note_from_pitch_duration(int pitch_id, int64_t duration_bin) {
    duration_bin = std::clamp<int64_t>(duration_bin, 0, static_cast<int64_t>(kDurationTemplates.size() - 1));
    return SheetSage2Note{
        pitch_id % 128,
        pitch_id >= 128 ? 1 : 0,
        static_cast<int>(duration_bin),
        kDurationTemplates[static_cast<size_t>(duration_bin)],
    };
}

std::string events_to_abc(const std::vector<SheetSage2Event> & events, double duration) {
    auto beats = rhythm_rows(events);
    if (beats.size() < 2) {
        return "X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:C\nV: Vocal\nZ|\nV: Ins\nZ|\n";
    }
    std::vector<double> final_periods;
    for (size_t i = beats.size() > 9 ? beats.size() - 8 : 1; i < beats.size(); ++i) {
        final_periods.push_back(beats[i].time - beats[i - 1].time);
    }
    std::sort(final_periods.begin(), final_periods.end());
    const size_t middle = final_periods.size() / 2;
    const double period = final_periods.size() % 2 == 0
        ? (final_periods[middle - 1] + final_periods[middle]) * 0.5
        : final_periods[middle];
    if (period <= 0.0) {
        throw std::runtime_error("SheetSage2 decoded beats must increase in time");
    }
    const double note_end = [&] {
        double end = duration;
        for (const auto & event : events) {
            for (const auto value : event.note_end_times) {
                end = std::max(end, static_cast<double>(value));
            }
        }
        return end;
    }();
    while (beats.back().time < note_end - 1.0e-6) {
        auto next = beats.back();
        next.time += period;
        next.beat = next.beat % next.numerator + 1;
        beats.push_back(next);
    }
    auto measures = infer_measures(beats);
    if (measures.empty()) {
        return {};
    }
    const auto subbeat_times = subbeat_times_from_beats(beats);
    const auto denominators = subbeat_denominators_from_measures(beats, measures, subbeat_times.size());
    int unit_denominator = 1;
    for (const auto & measure : measures) {
        unit_denominator = std::lcm(unit_denominator, measure.denominator * 4);
        unit_denominator = std::lcm(unit_denominator, measure.abc_denominator * 4);
    }
    const auto key_rows = interval_rows(events, "key", duration);
    const auto chord_rows = interval_rows(events, "chord", duration);
    const auto structure_rows = interval_rows(events, "structure", duration);
    auto keys = fill_text_intervals(
        key_rows,
        subbeat_times,
        key_rows.empty() ? std::string("C") : key_symbol_to_abc(key_rows.front().value));
    auto chords = fill_text_intervals(chord_rows, subbeat_times, "N");
    for (auto & key : keys) {
        key = key_symbol_to_abc(key);
    }
    const auto voices = build_voice_arrays(events, subbeat_times, std::max(duration, note_end));
    const double seconds = subbeat_times.back() - subbeat_times.front();
    const double quarters = std::accumulate(denominators.begin(), denominators.end() - 1, 0.0, [](double sum, int denominator) {
        return sum + 4.0 / static_cast<double>(denominator) / 4.0;
    });
    const int tempo = seconds > 0.0 ? static_cast<int>(std::llround(quarters / seconds * 60.0)) : 120;
    std::ostringstream out;
    out << "X:1\n"
        << "T:\n"
        << "M:" << measures.front().abc_numerator << "/" << measures.front().abc_denominator << "\n"
        << "L:1/" << unit_denominator << "\n"
        << "Q:1/4=" << tempo << "\n"
        << "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        << "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        << "K:" << keys[static_cast<size_t>(measures.front().start_beat * 4)] << "\n";
    std::vector<std::pair<int, std::string>> structure_events;
    structure_events.reserve(structure_rows.size());
    for (const auto & row : structure_rows) {
        structure_events.push_back({quantize_time(row.start, subbeat_times), row.value});
    }
    for (const auto & group : measure_groups(measures, keys, structure_events)) {
        for (const auto & label : group.structure_labels) {
            out << "% " << label << "\n";
        }
        const auto & first = group.measures.front();
        out << "V: Vocal\n";
        if (group.meter_changed) {
            out << "M:" << first.abc_numerator << "/" << first.abc_denominator << "\n";
        }
        if (group.key_changed) {
            out << "K:" << keys[static_cast<size_t>(first.start_beat * 4)] << "\n";
        }
        out << render_voice_group(voices[0], keys, chords, denominators, group.measures, unit_denominator, true) << "\n"
            << "V: Ins\n";
        if (group.meter_changed) {
            out << "M:" << first.abc_numerator << "/" << first.abc_denominator << "\n";
        }
        if (group.key_changed) {
            out << "K:" << keys[static_cast<size_t>(first.start_beat * 4)] << "\n";
        }
        out << render_voice_group(voices[1], keys, chords, denominators, group.measures, unit_denominator, false) << "\n";
    }
    return out.str();
}

std::string events_json(const std::vector<SheetSage2Event> & events) {
    const auto write_i32_array = [](std::ostringstream & out, const std::vector<int32_t> & values) {
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << values[i];
        }
        out << "]";
    };
    std::ostringstream out;
    out << "{\"events\":[";
    for (size_t i = 0; i < events.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & event = events[i];
        out << "{\"subbeat\":" << event.subbeat
            << ",\"time\":" << event.time
            << ",\"window_index\":" << event.window_index
            << ",\"source_subbeat\":" << event.source_subbeat
            << ",\"global_subbeat\":" << event.global_subbeat
            << ",\"tokens_by_field\":{";
        bool wrote = false;
        const auto write_field = [&](const char * name, const std::vector<int32_t> & values) {
            if (values.empty()) {
                return;
            }
            if (wrote) {
                out << ",";
            }
            wrote = true;
            out << "\"" << name << "\":";
            write_i32_array(out, values);
        };
        write_field("timestamp", event.timestamp_tokens);
        write_field("rhythm", event.rhythm_tokens);
        write_field("structure", event.structure_tokens);
        write_field("key", event.key_tokens);
        write_field("chord", event.chord_tokens);
        write_field("melody", event.melody_tokens);
        out << "},\"notes\":[";
        for (size_t n = 0; n < event.notes.size(); ++n) {
            if (n != 0) {
                out << ",";
            }
            out << "{\"pitch\":" << event.notes[n].pitch
                << ",\"track\":" << event.notes[n].track
                << ",\"duration_bin\":" << event.notes[n].duration_bin
                << ",\"duration_steps\":" << event.notes[n].duration_steps;
            if (n < event.note_end_times.size()) {
                out << ",\"end_time\":" << event.note_end_times[n];
            }
            out << "}";
        }
        out << "]}";
    }
    out << "]}";
    return out.str();
}

}  // namespace engine::models::sheetsage
