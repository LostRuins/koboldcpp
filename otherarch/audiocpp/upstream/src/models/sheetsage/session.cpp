#include "engine/models/sheetsage/session.h"
#include "engine/models/sheetsage/processing.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::sheetsage {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "sheetsage2";
constexpr int64_t kTimeHz = 100;
constexpr int64_t kPromptCapacity = 256;
constexpr int64_t kMaxSubbeatShift = 256;
constexpr double kDefaultOverlapSeconds = 200.0;
constexpr double kDefaultLookaheadSeconds = 100.0;
constexpr double kDefaultStepSeconds = 0.125;

constexpr std::array<const char *, 8> kPromptNames = {
    "timestamp",
    "downbeat_meter",
    "structure",
    "key",
    "chord_majmin",
    "chord_full",
    "melody_vocal",
    "melody_full",
};

struct SheetSage2Tokenizer {
    int64_t audio_seconds = 300;
    int64_t sos = 1;
    int64_t eos = 2;
    int64_t out = 3;
    int64_t prompt_start = 4;
    int64_t prompt_end = prompt_start + kPromptCapacity;
    int64_t subbeat_start = prompt_end;
    int64_t subbeat_end = subbeat_start + kMaxSubbeatShift + 1;
    int64_t time_start = subbeat_end;
    int64_t time_end = time_start + audio_seconds * kTimeHz;
    int64_t meter_start = time_end;
    int64_t meter_end = meter_start + 32 * 6;
    int64_t eighth_start = meter_end;
    int64_t eighth_end = eighth_start + 256;
    int64_t structure_start = eighth_end;
    int64_t structure_end = structure_start + sheetsage2_structure_label_count();
    int64_t key_start = structure_end;
    int64_t key_end = key_start + 24;
    int64_t majmin_chord_start = key_end;
    int64_t majmin_chord_end = majmin_chord_start + 25;
    int64_t full_chord_start = majmin_chord_end;
    int64_t full_chord_end = full_chord_start + 361;
    int64_t pitch_start = full_chord_end;
    int64_t pitch_end = pitch_start + 256;
    int64_t duration_start = pitch_end;
    int64_t duration_end = duration_start + sheetsage2_duration_bin_count();
    int64_t vocab_size = duration_end;

    explicit SheetSage2Tokenizer(int64_t seconds) : audio_seconds(std::max<int64_t>(1, seconds)) {
        time_end = time_start + audio_seconds * kTimeHz;
        meter_start = time_end;
        meter_end = meter_start + 32 * 6;
        eighth_start = meter_end;
        eighth_end = eighth_start + 256;
        structure_start = eighth_end;
        structure_end = structure_start + sheetsage2_structure_label_count();
        key_start = structure_end;
        key_end = key_start + 24;
        majmin_chord_start = key_end;
        majmin_chord_end = majmin_chord_start + 25;
        full_chord_start = majmin_chord_end;
        full_chord_end = full_chord_start + 361;
        pitch_start = full_chord_end;
        pitch_end = pitch_start + 256;
        duration_start = pitch_end;
        duration_end = duration_start + sheetsage2_duration_bin_count();
        vocab_size = duration_end;
    }
};

enum class SheetSage2TokenType {
    Special,
    Subbeat,
    Time,
    Meter,
    Eighth,
    Structure,
    Key,
    Chord,
    Pitch,
    Duration,
};

struct SheetSage2GrammarState {
    int payload_count = 0;
    bool in_shift = true;
    int shift_run = 0;
    int last_field_index = -1;
    enum class Incomplete {
        None,
        RhythmAfterMeter,
        MelodyAfterPitch,
    } incomplete = Incomplete::None;
};

struct SheetSage2Window {
    int64_t index = 0;
    int64_t start_sample = 0;
    int64_t end_sample = 0;
    double start = 0.0;
    double end = 0.0;
    double accept_start = 0.0;
    double accept_end = 0.0;
    double prefix_end = 0.0;
    std::optional<double> generation_stop;
};

std::shared_ptr<const SheetSage2Assets> require_assets(std::shared_ptr<const SheetSage2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("SheetSage2 session requires assets");
    }
    return assets;
}

std::shared_ptr<const model_spec::ModelContract> require_contract(
    std::shared_ptr<const model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("SheetSage2 session requires a model contract");
    }
    return contract;
}

SheetSage2DecoderConfig parse_decoder_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    SheetSage2DecoderConfig config;
    config.vocab_size = io::json::optional_i64(root, "vocab_size", config.vocab_size);
    config.hidden_size = io::json::optional_i64(root, "hidden_size", config.hidden_size);
    config.intermediate_size = io::json::optional_i64(root, "intermediate_size", config.intermediate_size);
    config.decoder_layers = io::json::optional_i64(root, "decoder_layers", config.decoder_layers);
    config.num_attention_heads = io::json::optional_i64(root, "num_attention_heads", config.num_attention_heads);
    config.max_position_embeddings = io::json::optional_i64(root, "max_output_seq_len", config.max_position_embeddings);
    config.pad_token_id = io::json::optional_i64(root, "pad_token_id", config.pad_token_id);
    config.layer_norm_eps = io::json::optional_f32(root, "layer_norm_eps", config.layer_norm_eps);
    if (const auto * backbone = root.find("backbone_config")) {
        config.encoder_hidden_size = io::json::optional_i64(*backbone, "hidden_size", config.encoder_hidden_size);
        config.encoder_intermediate_size = io::json::optional_i64(*backbone, "intermediate_size", config.encoder_intermediate_size);
        config.encoder_layers = io::json::optional_i64(*backbone, "num_hidden_layers", config.encoder_layers);
        config.encoder_attention_heads = io::json::optional_i64(*backbone, "num_attention_heads", config.encoder_attention_heads);
        config.sampling_rate = io::json::optional_i64(*backbone, "sampling_rate", config.sampling_rate);
        config.n_fft = io::json::optional_i64(*backbone, "n_fft", config.n_fft);
        config.win_length = io::json::optional_i64(*backbone, "win_length", config.win_length);
        config.hop_length = io::json::optional_i64(*backbone, "hop_length", config.hop_length);
        config.mel_bins = io::json::optional_i64(*backbone, "num_mel_bins", config.mel_bins);
        config.input_audio_length_samples =
            io::json::optional_i64(*backbone, "input_audio_length", config.input_audio_length_samples);
        config.encoder_layer_norm_eps = io::json::optional_f32(*backbone, "layer_norm_eps", config.encoder_layer_norm_eps);
        config.subsampling_layer_norm_eps =
            io::json::optional_f32(*backbone, "subsampling_layer_norm_eps", config.subsampling_layer_norm_eps);
        config.rotary_embedding_base =
            io::json::optional_f32(*backbone, "rotary_embedding_base", config.rotary_embedding_base);
        config.conformer_conv_kernel_size =
            io::json::optional_i64(*backbone, "conv_depthwise_kernel_size", config.conformer_conv_kernel_size);
    }
    return config;
}

SheetSage2DecoderRuntimeOptions decoder_options_from_session_options(const runtime::SessionOptions & options) {
    SheetSage2DecoderRuntimeOptions out;
    out.weight_storage_type = runtime::parse_tensor_storage_option(
        options.options,
        "sheetsage2.weight_type",
        assets::TensorStorageType::Native,
        {
            assets::TensorStorageType::Native,
            assets::TensorStorageType::F32,
            assets::TensorStorageType::F16,
            assets::TensorStorageType::BF16,
            assets::TensorStorageType::Q4_0,
            assets::TensorStorageType::Q4_K,
        });
    out.weight_context_bytes =
        runtime::parse_size_mb_option(options.options, {"sheetsage2.weight_context_mb"}, out.weight_context_bytes);
    out.graph_arena_bytes =
        runtime::parse_size_mb_option(options.options, {"sheetsage2.decoder_graph_arena_mb"}, out.graph_arena_bytes);
    out.attention_tile_rows = runtime::parse_i64_option(options.options, {"sheetsage2.attention_tile_rows"}).value_or(0);
    return out;
}

std::vector<int32_t> prompt_prefix(const SheetSage2Tokenizer & tokenizer) {
    return {
        static_cast<int32_t>(tokenizer.sos),
        static_cast<int32_t>(tokenizer.prompt_start),
        static_cast<int32_t>(tokenizer.prompt_start + 1),
        static_cast<int32_t>(tokenizer.prompt_start + 2),
        static_cast<int32_t>(tokenizer.prompt_start + 3),
        static_cast<int32_t>(tokenizer.prompt_start + 5),
        static_cast<int32_t>(tokenizer.prompt_start + 7),
        static_cast<int32_t>(tokenizer.out),
    };
}

SheetSage2TokenType token_type(const SheetSage2Tokenizer & tokenizer, int64_t token) {
    if (token == tokenizer.sos || token == tokenizer.eos || token == tokenizer.out ||
        (token >= tokenizer.prompt_start && token < tokenizer.prompt_start + static_cast<int64_t>(kPromptNames.size()))) {
        return SheetSage2TokenType::Special;
    }
    if (token >= tokenizer.subbeat_start && token < tokenizer.subbeat_end) {
        return SheetSage2TokenType::Subbeat;
    }
    if (token >= tokenizer.time_start && token < tokenizer.time_end) {
        return SheetSage2TokenType::Time;
    }
    if (token >= tokenizer.meter_start && token < tokenizer.meter_end) {
        return SheetSage2TokenType::Meter;
    }
    if (token >= tokenizer.eighth_start && token < tokenizer.eighth_end) {
        return SheetSage2TokenType::Eighth;
    }
    if (token >= tokenizer.structure_start && token < tokenizer.structure_end) {
        return SheetSage2TokenType::Structure;
    }
    if (token >= tokenizer.key_start && token < tokenizer.key_end) {
        return SheetSage2TokenType::Key;
    }
    if ((token >= tokenizer.majmin_chord_start && token < tokenizer.majmin_chord_end) ||
        (token >= tokenizer.full_chord_start && token < tokenizer.full_chord_end)) {
        return SheetSage2TokenType::Chord;
    }
    if (token >= tokenizer.pitch_start && token < tokenizer.pitch_end) {
        return SheetSage2TokenType::Pitch;
    }
    if (token >= tokenizer.duration_start && token < tokenizer.duration_end) {
        return SheetSage2TokenType::Duration;
    }
    return SheetSage2TokenType::Special;
}

bool update_grammar(const SheetSage2Tokenizer & tokenizer, SheetSage2GrammarState & state, int64_t token) {
    if (token == tokenizer.eos) {
        return true;
    }
    const auto type = token_type(tokenizer, token);
    if (type == SheetSage2TokenType::Subbeat) {
        if (!state.in_shift && state.payload_count > 0) {
            state.payload_count = 0;
            state.last_field_index = -1;
            state.incomplete = SheetSage2GrammarState::Incomplete::None;
        }
        state.in_shift = true;
        ++state.shift_run;
        return false;
    }
    state.in_shift = false;
    state.shift_run = 0;
    ++state.payload_count;
    switch (type) {
    case SheetSage2TokenType::Time:
        state.last_field_index = 0;
        state.incomplete = SheetSage2GrammarState::Incomplete::None;
        break;
    case SheetSage2TokenType::Meter:
        state.last_field_index = 1;
        state.incomplete = SheetSage2GrammarState::Incomplete::RhythmAfterMeter;
        break;
    case SheetSage2TokenType::Eighth:
        state.last_field_index = 1;
        state.incomplete = SheetSage2GrammarState::Incomplete::None;
        break;
    case SheetSage2TokenType::Structure:
        state.last_field_index = 2;
        state.incomplete = SheetSage2GrammarState::Incomplete::None;
        break;
    case SheetSage2TokenType::Key:
        state.last_field_index = 3;
        state.incomplete = SheetSage2GrammarState::Incomplete::None;
        break;
    case SheetSage2TokenType::Chord:
        state.last_field_index = 4;
        state.incomplete = SheetSage2GrammarState::Incomplete::None;
        break;
    case SheetSage2TokenType::Pitch:
        state.last_field_index = 5;
        state.incomplete = SheetSage2GrammarState::Incomplete::MelodyAfterPitch;
        break;
    case SheetSage2TokenType::Duration:
        state.last_field_index = 5;
        state.incomplete = SheetSage2GrammarState::Incomplete::None;
        break;
    case SheetSage2TokenType::Special:
    case SheetSage2TokenType::Subbeat:
        throw std::runtime_error("SheetSage2 generated invalid grammar token");
    }
    return false;
}

int32_t select_next_token(
    const std::vector<float> & logits,
    const SheetSage2Tokenizer & tokenizer,
    const SheetSage2GrammarState & state) {
    int32_t best = static_cast<int32_t>(tokenizer.eos);
    float best_value = -std::numeric_limits<float>::infinity();
    const int64_t start = static_cast<int64_t>(logits.size()) - tokenizer.vocab_size;
    if (start < 0) {
        throw std::runtime_error("SheetSage2 decoder logits are smaller than vocabulary");
    }
    const auto scan_token = [&](int64_t token) {
        const float value = logits[static_cast<size_t>(start + token)];
        if (value > best_value) {
            best_value = value;
            best = static_cast<int32_t>(token);
        }
    };
    const auto scan_range = [&](int64_t begin, int64_t end) {
        for (int64_t token = begin; token < end; ++token) {
            scan_token(token);
        }
    };
    if (state.payload_count > 0) {
        scan_token(tokenizer.eos);
    }
    if ((state.payload_count > 0 || state.in_shift) && state.shift_run < 4) {
        scan_range(tokenizer.subbeat_start, tokenizer.subbeat_end);
    }
    if (state.incomplete == SheetSage2GrammarState::Incomplete::RhythmAfterMeter) {
        scan_range(tokenizer.eighth_start, tokenizer.eighth_end);
        return best;
    }
    if (state.incomplete == SheetSage2GrammarState::Incomplete::MelodyAfterPitch) {
        scan_range(tokenizer.pitch_start, tokenizer.pitch_end);
        scan_range(tokenizer.duration_start, tokenizer.duration_end);
        return best;
    }
    if (state.last_field_index < 0) {
        scan_range(tokenizer.time_start, tokenizer.time_end);
    }
    if (state.last_field_index < 1) {
        scan_range(tokenizer.meter_start, tokenizer.meter_end);
        scan_range(tokenizer.eighth_start, tokenizer.eighth_end);
    }
    if (state.last_field_index < 2) {
        scan_range(tokenizer.structure_start, tokenizer.structure_end);
    }
    if (state.last_field_index < 3) {
        scan_range(tokenizer.key_start, tokenizer.key_end);
    }
    if (state.last_field_index < 4) {
        scan_range(tokenizer.full_chord_start, tokenizer.full_chord_end);
    }
    if (state.last_field_index <= 5) {
        scan_range(tokenizer.pitch_start, tokenizer.pitch_end);
    }
    return best;
}

std::vector<SheetSage2Event> decode_events(
    const SheetSage2Tokenizer & tokenizer,
    const std::vector<int32_t> & tokens) {
    std::vector<SheetSage2Event> events;
    auto out_it = std::find(tokens.begin(), tokens.end(), static_cast<int32_t>(tokenizer.out));
    if (out_it == tokens.end()) {
        return events;
    }
    int64_t current_step = 0;
    size_t pos = static_cast<size_t>(std::distance(tokens.begin(), out_it)) + 1;
    while (pos < tokens.size()) {
        int64_t token = tokens[pos];
        if (token == tokenizer.eos) {
            break;
        }
        if (token_type(tokenizer, token) != SheetSage2TokenType::Subbeat) {
            ++pos;
            continue;
        }
        int64_t shift = 0;
        while (pos < tokens.size() && token_type(tokenizer, tokens[pos]) == SheetSage2TokenType::Subbeat) {
            shift += tokens[pos] - tokenizer.subbeat_start;
            ++pos;
        }
        current_step += shift;
        SheetSage2Event event;
        event.subbeat = current_step;
        while (pos < tokens.size()) {
            token = tokens[pos];
            const auto type = token_type(tokenizer, token);
            if (token == tokenizer.eos || type == SheetSage2TokenType::Subbeat) {
                break;
            }
            if (type == SheetSage2TokenType::Time) {
                event.timestamp = static_cast<float>(token - tokenizer.time_start) / static_cast<float>(kTimeHz);
                event.timestamp_tokens.push_back(static_cast<int32_t>(token));
            } else if (type == SheetSage2TokenType::Meter) {
                const int64_t index = token - tokenizer.meter_start;
                static constexpr std::array<int, 6> denominators = {1, 2, 4, 8, 16, 32};
                event.meter = {static_cast<int>(index / 6 + 1), denominators[static_cast<size_t>(index % 6)]};
                event.rhythm_tokens.push_back(static_cast<int32_t>(token));
            } else if (type == SheetSage2TokenType::Eighth) {
                event.eighth_position = token - tokenizer.eighth_start;
                event.rhythm_tokens.push_back(static_cast<int32_t>(token));
            } else if (type == SheetSage2TokenType::Structure) {
                const int64_t index = token - tokenizer.structure_start;
                event.structure = sheetsage2_structure_label(index);
                event.structure_tokens.push_back(static_cast<int32_t>(token));
            } else if (type == SheetSage2TokenType::Key) {
                const int64_t index = token - tokenizer.key_start;
                event.key = sheetsage2_key_label(index);
                event.key_tokens.push_back(static_cast<int32_t>(token));
            } else if (type == SheetSage2TokenType::Chord) {
                if (token >= tokenizer.full_chord_start && token < tokenizer.full_chord_end) {
                    const int64_t index = token - tokenizer.full_chord_start;
                    event.chord = sheetsage2_chord_label(true, index);
                } else {
                    const int64_t index = token - tokenizer.majmin_chord_start;
                    event.chord = sheetsage2_chord_label(false, index);
                }
                event.chord_tokens.push_back(static_cast<int32_t>(token));
            } else if (type == SheetSage2TokenType::Pitch) {
                const int pitch_id = static_cast<int>(token - tokenizer.pitch_start);
                int duration_bin = 0;
                event.melody_tokens.push_back(static_cast<int32_t>(token));
                if (pos + 1 < tokens.size() && token_type(tokenizer, tokens[pos + 1]) == SheetSage2TokenType::Duration) {
                    const int64_t bin = tokens[pos + 1] - tokenizer.duration_start;
                    duration_bin = static_cast<int>(std::clamp<int64_t>(
                        bin,
                        0,
                        sheetsage2_duration_bin_count() - 1));
                    event.melody_tokens.push_back(tokens[pos + 1]);
                    ++pos;
                }
                event.notes.push_back(sheetsage2_note_from_pitch_duration(pitch_id, duration_bin));
            }
            ++pos;
        }
        events.push_back(std::move(event));
    }
    return events;
}

std::vector<int32_t> subbeat_shift_tokens(const SheetSage2Tokenizer & tokenizer, int64_t shift) {
    if (shift < 0) {
        throw std::runtime_error("SheetSage2 prefix events are not sorted by subbeat");
    }
    std::vector<int32_t> tokens;
    while (shift > kMaxSubbeatShift) {
        tokens.push_back(static_cast<int32_t>(tokenizer.subbeat_start + kMaxSubbeatShift));
        shift -= kMaxSubbeatShift;
    }
    tokens.push_back(static_cast<int32_t>(tokenizer.subbeat_start + shift));
    return tokens;
}

std::vector<int32_t> encode_events(
    const SheetSage2Tokenizer & tokenizer,
    const std::vector<SheetSage2Event> & events,
    bool include_eos) {
    auto ids = prompt_prefix(tokenizer);
    int64_t previous_step = 0;
    for (const auto & event : events) {
        auto shift = subbeat_shift_tokens(tokenizer, event.subbeat - previous_step);
        ids.insert(ids.end(), shift.begin(), shift.end());
        previous_step = event.subbeat;
        ids.insert(ids.end(), event.timestamp_tokens.begin(), event.timestamp_tokens.end());
        ids.insert(ids.end(), event.rhythm_tokens.begin(), event.rhythm_tokens.end());
        ids.insert(ids.end(), event.structure_tokens.begin(), event.structure_tokens.end());
        ids.insert(ids.end(), event.key_tokens.begin(), event.key_tokens.end());
        ids.insert(ids.end(), event.chord_tokens.begin(), event.chord_tokens.end());
        ids.insert(ids.end(), event.melody_tokens.begin(), event.melody_tokens.end());
    }
    if (include_eos) {
        ids.push_back(static_cast<int32_t>(tokenizer.eos));
    }
    return ids;
}

float event_time_value(const SheetSage2Event & event) {
    return event.time;
}

std::function<double(int64_t)> make_event_time_lookup(
    const std::vector<SheetSage2Event> & events,
    double target_seconds) {
    std::vector<std::pair<int64_t, double>> anchors;
    for (const auto & event : events) {
        if (event.timestamp.has_value()) {
            anchors.push_back({event.subbeat, *event.timestamp});
        }
    }
    if (anchors.empty()) {
        return [target_seconds](int64_t step) {
            return std::min(target_seconds, std::max(0.0, static_cast<double>(step) * kDefaultStepSeconds));
        };
    }
    std::sort(anchors.begin(), anchors.end());
    anchors.erase(
        std::unique(
            anchors.begin(),
            anchors.end(),
            [](const auto & a, const auto & b) { return a.first == b.first; }),
        anchors.end());
    double step_seconds = kDefaultStepSeconds;
    if (anchors.size() >= 2) {
        std::vector<double> slopes;
        slopes.reserve(anchors.size() - 1);
        for (size_t i = 1; i < anchors.size(); ++i) {
            const double step_delta = static_cast<double>(std::max<int64_t>(1, anchors[i].first - anchors[i - 1].first));
            slopes.push_back((anchors[i].second - anchors[i - 1].second) / step_delta);
        }
        std::sort(slopes.begin(), slopes.end());
        step_seconds = slopes[slopes.size() / 2];
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
            step_seconds = kDefaultStepSeconds;
        }
    }
    return [anchors, target_seconds, step_seconds](int64_t step) {
        const double value = static_cast<double>(step);
        if (value <= static_cast<double>(anchors.front().first)) {
            return std::clamp(
                anchors.front().second + (value - static_cast<double>(anchors.front().first)) * step_seconds,
                0.0,
                target_seconds);
        }
        if (value >= static_cast<double>(anchors.back().first)) {
            return std::clamp(
                anchors.back().second + (value - static_cast<double>(anchors.back().first)) * step_seconds,
                0.0,
                target_seconds);
        }
        for (size_t i = 1; i < anchors.size(); ++i) {
            if (value <= static_cast<double>(anchors[i].first)) {
                const double left_step = static_cast<double>(anchors[i - 1].first);
                const double right_step = static_cast<double>(anchors[i].first);
                const double alpha = (value - left_step) / std::max(1.0, right_step - left_step);
                return anchors[i - 1].second + alpha * (anchors[i].second - anchors[i - 1].second);
            }
        }
        return anchors.back().second;
    };
}

std::vector<SheetSage2Window> sliding_window_plan(
    int64_t audio_samples,
    int64_t sample_rate,
    int64_t window_samples) {
    if (audio_samples <= 0 || sample_rate <= 0 || window_samples <= 0) {
        throw std::runtime_error("SheetSage2 sliding window requires positive audio shape");
    }
    const double duration = static_cast<double>(audio_samples) / static_cast<double>(sample_rate);
    const double window_seconds = static_cast<double>(window_samples) / static_cast<double>(sample_rate);
    const double overlap_seconds = std::min(kDefaultOverlapSeconds, std::max(0.0, window_seconds - 1.0));
    const double lookahead_seconds = std::min(kDefaultLookaheadSeconds, overlap_seconds);
    if (!(0.0 <= lookahead_seconds && lookahead_seconds <= overlap_seconds && overlap_seconds < window_seconds)) {
        throw std::runtime_error("SheetSage2 invalid sliding window overlap");
    }
    const double hop = window_seconds - overlap_seconds;
    double start = 0.0;
    double accepted = 0.0;
    int64_t index = 0;
    std::vector<SheetSage2Window> out;
    while (true) {
        const bool last = start + window_seconds >= duration - 1.0e-6;
        const double accept_end = last ? duration : start + window_seconds - lookahead_seconds;
        SheetSage2Window window;
        window.index = index++;
        window.start = start;
        window.end = std::min(duration, start + window_seconds);
        window.accept_start = accepted;
        window.accept_end = accept_end;
        window.prefix_end = accepted;
        window.generation_stop = last ? std::nullopt : std::optional<double>(window_seconds - lookahead_seconds);
        window.start_sample = static_cast<int64_t>(std::llround(start * static_cast<double>(sample_rate)));
        window.end_sample = std::min<int64_t>(
            audio_samples,
            window.start_sample + window_samples);
        out.push_back(window);
        if (last) {
            return out;
        }
        accepted = accept_end;
        start = std::min(start + hop, duration - window_seconds);
    }
}

std::vector<float> slice_window_audio(
    const std::vector<float> & audio,
    int64_t start_sample,
    int64_t window_samples) {
    std::vector<float> out(static_cast<size_t>(window_samples), 0.0F);
    if (start_sample < 0 || start_sample > static_cast<int64_t>(audio.size())) {
        throw std::runtime_error("SheetSage2 sliding window start is outside audio");
    }
    const int64_t count = std::min<int64_t>(window_samples, static_cast<int64_t>(audio.size()) - start_sample);
    if (count > 0) {
        std::copy_n(
            audio.begin() + static_cast<std::ptrdiff_t>(start_sample),
            static_cast<size_t>(count),
            out.begin());
    }
    return out;
}

void set_event_local_timestamp(
    SheetSage2Event & event,
    const SheetSage2Tokenizer & tokenizer,
    double local_time) {
    int64_t time_id = static_cast<int64_t>(std::llround(local_time * static_cast<double>(kTimeHz)));
    time_id = std::clamp<int64_t>(time_id, 0, tokenizer.time_end - tokenizer.time_start - 1);
    event.timestamp_tokens.clear();
    event.timestamp_tokens.push_back(static_cast<int32_t>(tokenizer.time_start + time_id));
    event.timestamp = static_cast<float>(time_id) / static_cast<float>(kTimeHz);
}

std::vector<SheetSage2Event> build_overlap_prefix_events(
    const std::vector<SheetSage2Event> & stitched_events,
    const SheetSage2Tokenizer & tokenizer,
    const SheetSage2Window & window,
    int64_t & base_subbeat) {
    constexpr double eps = 1.0e-4;
    std::vector<SheetSage2Event> source_events;
    for (const auto & event : stitched_events) {
        const double t = event_time_value(event);
        if (window.start - eps <= t && t < window.prefix_end - eps) {
            source_events.push_back(event);
        }
    }
    std::sort(source_events.begin(), source_events.end(), [](const auto & a, const auto & b) {
        if (a.global_subbeat != b.global_subbeat) {
            return a.global_subbeat < b.global_subbeat;
        }
        return a.time < b.time;
    });
    auto first_beat = std::find_if(source_events.begin(), source_events.end(), [](const auto & event) {
        return event.timestamp.has_value() || !event.rhythm_tokens.empty();
    });
    if (first_beat == source_events.end()) {
        base_subbeat = 0;
        return {};
    }
    source_events.erase(source_events.begin(), first_beat);
    base_subbeat = source_events.front().global_subbeat;

    std::optional<std::vector<int32_t>> context_structure;
    std::optional<std::vector<int32_t>> context_key;
    std::optional<std::vector<int32_t>> context_chord;
    std::optional<std::vector<int32_t>> context_meter;
    for (const auto & event : stitched_events) {
        if (event.time > source_events.front().time + eps) {
            continue;
        }
        if (!event.structure_tokens.empty()) {
            context_structure = event.structure_tokens;
        }
        if (!event.key_tokens.empty()) {
            context_key = event.key_tokens;
        }
        if (!event.chord_tokens.empty()) {
            context_chord = event.chord_tokens;
        }
        for (const auto token : event.rhythm_tokens) {
            if (token_type(tokenizer, token) == SheetSage2TokenType::Meter) {
                context_meter = std::vector<int32_t>{token};
                break;
            }
        }
    }

    std::vector<SheetSage2Event> prefix_events;
    prefix_events.reserve(source_events.size());
    for (auto event : source_events) {
        event.subbeat = std::max<int64_t>(0, event.global_subbeat - base_subbeat);
        if (!event.timestamp_tokens.empty()) {
            set_event_local_timestamp(event, tokenizer, static_cast<double>(event.time) - window.start);
        }
        prefix_events.push_back(std::move(event));
    }
    if (!prefix_events.empty()) {
        auto & first = prefix_events.front();
        if (first.structure_tokens.empty() && context_structure.has_value()) {
            first.structure_tokens = *context_structure;
        }
        if (first.key_tokens.empty() && context_key.has_value()) {
            first.key_tokens = *context_key;
        }
        if (first.chord_tokens.empty() && context_chord.has_value()) {
            first.chord_tokens = *context_chord;
        }
        const bool has_meter = std::any_of(first.rhythm_tokens.begin(), first.rhythm_tokens.end(), [&](int32_t token) {
            return token_type(tokenizer, token) == SheetSage2TokenType::Meter;
        });
        const bool has_eighth = std::any_of(first.rhythm_tokens.begin(), first.rhythm_tokens.end(), [&](int32_t token) {
            return token_type(tokenizer, token) == SheetSage2TokenType::Eighth;
        });
        if (!has_meter && has_eighth && context_meter.has_value()) {
            first.rhythm_tokens.insert(first.rhythm_tokens.begin(), context_meter->begin(), context_meter->end());
        }
    }
    return prefix_events;
}

std::vector<SheetSage2Event> stitched_window_events(
    const std::vector<SheetSage2Event> & decoded_events,
    const std::function<double(int64_t)> & time_lookup,
    const SheetSage2Window & window,
    double song_duration,
    int64_t global_subbeat_base) {
    constexpr double eps = 1.0e-4;
    std::vector<SheetSage2Event> accepted;
    for (auto event : decoded_events) {
        const double local_time = time_lookup(event.subbeat);
        const double abs_time = window.start + local_time;
        if (abs_time < window.accept_start - eps ||
            abs_time >= window.accept_end - eps ||
            abs_time >= song_duration - eps) {
            continue;
        }
        event.time = static_cast<float>(std::clamp(abs_time, 0.0, song_duration));
        event.window_index = static_cast<int>(window.index);
        event.window_start = static_cast<float>(window.start);
        event.source_subbeat = event.subbeat;
        event.global_subbeat = global_subbeat_base + event.subbeat;
        if (event.timestamp.has_value()) {
            event.timestamp = event.time;
        }
        event.note_end_times.clear();
        event.note_end_times.reserve(event.notes.size());
        for (const auto & note : event.notes) {
            const double local_end = time_lookup(event.subbeat + note.duration_steps);
            const double end_time = std::min(song_duration, std::max(abs_time + 0.04, window.start + local_end));
            event.note_end_times.push_back(static_cast<float>(end_time));
        }
        accepted.push_back(std::move(event));
    }
    return accepted;
}

std::vector<float> prepare_normalized_mel(
    const std::vector<float> & waveform,
    const SheetSage2Assets & assets,
    size_t threads,
    int64_t & mel_frames_out) {
    if (static_cast<int64_t>(waveform.size()) <= assets.config.n_fft) {
        throw std::runtime_error("SheetSage2 window audio is too short for the audio frontend");
    }
    const int64_t target_samples = static_cast<int64_t>(waveform.size());
    const engine::audio::STFTConfig stft_config{
        assets.config.n_fft,
        assets.config.hop_length,
        assets.config.win_length,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Default,
    };
    auto magnitude = engine::audio::STFT().compute_magnitude(
        waveform,
        assets.stft_window,
        1,
        target_samples,
        stft_config,
        threads);
    const int64_t freq_bins = assets.config.n_fft / 2 + 1;
    const int64_t stft_frames = magnitude.shape.at(2);
    const int64_t mel_frames = stft_frames - 1;
    auto mel = engine::audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        freq_bins,
        stft_frames,
        mel_frames,
        assets.mel_filterbank);
    if (mel.shape.size() != 3 || mel.shape[1] != assets.config.mel_bins || mel.shape[2] != mel_frames) {
        throw std::runtime_error("SheetSage2 mel frontend produced invalid shape");
    }
    std::vector<float> normalized(static_cast<size_t>(mel_frames * assets.config.mel_bins), 0.0F);
    for (int64_t t = 0; t < mel_frames; ++t) {
        for (int64_t m = 0; m < assets.config.mel_bins; ++m) {
            const float power = std::max(mel.values[static_cast<size_t>(m * mel.shape[2] + t)], 1.0e-10F);
            const float db = 10.0F * std::log10(power);
            const float stdv = std::max(assets.mel_std[static_cast<size_t>(m)], 1.0e-5F);
            normalized[static_cast<size_t>(t * assets.config.mel_bins + m)] =
                (db - assets.mel_mean[static_cast<size_t>(m)]) / stdv;
        }
    }
    mel_frames_out = mel_frames;
    return normalized;
}

std::vector<int32_t> generate_tokens(
    SheetSage2DecoderRuntime & decoder,
    const std::vector<float> & memory,
    int64_t memory_steps,
    const SheetSage2Tokenizer & tokenizer,
    int64_t max_sequence_length,
    const std::vector<int32_t> * prefix_tokens,
    std::optional<double> stop_time_seconds) {
    auto ids = prefix_tokens == nullptr ? prompt_prefix(tokenizer) : *prefix_tokens;
    if (ids.empty() || ids.front() != tokenizer.sos) {
        throw std::runtime_error("SheetSage2 generation prefix must begin with sos");
    }
    if (ids.back() == tokenizer.eos) {
        ids.pop_back();
    }
    const auto out_it = std::find(ids.begin(), ids.end(), static_cast<int32_t>(tokenizer.out));
    if (out_it == ids.end()) {
        throw std::runtime_error("SheetSage2 generation prefix is missing out token");
    }
    SheetSage2GrammarState state;
    for (auto it = out_it + 1; it != ids.end(); ++it) {
        update_grammar(tokenizer, state, *it);
    }
    if (max_sequence_length <= static_cast<int64_t>(ids.size())) {
        throw std::runtime_error("SheetSage2 max_tokens must exceed the generation prefix length");
    }
    decoder.reset_cached_decode(memory, memory_steps, max_sequence_length);
    std::vector<float> logits = decoder.prefill_cached_decode(ids);
    for (int64_t step = static_cast<int64_t>(ids.size()); step < max_sequence_length; ++step) {
        const int32_t token = select_next_token(logits, tokenizer, state);
        ids.push_back(token);
        bool finished = update_grammar(tokenizer, state, token);
        if (!finished && stop_time_seconds.has_value() &&
            token >= tokenizer.time_start && token < tokenizer.time_end &&
            static_cast<double>(token - tokenizer.time_start) / static_cast<double>(kTimeHz) >= *stop_time_seconds) {
            ids.push_back(static_cast<int32_t>(tokenizer.eos));
            finished = true;
        }
        if (finished) {
            break;
        }
        logits = decoder.decode_cached_step(token);
    }
    if (ids.back() != tokenizer.eos) {
        ids.push_back(static_cast<int32_t>(tokenizer.eos));
    }
    return ids;
}

engine::audio::AudioTensor load_mel_filterbank(const assets::TensorSource & source, int64_t n_fft, int64_t mel_bins) {
    const int64_t freq_bins = n_fft / 2 + 1;
    const auto values = source.require_f32("feature_extractor.mel_scale.fb", {freq_bins, mel_bins});
    engine::audio::AudioTensor filterbank;
    filterbank.shape = {mel_bins, freq_bins};
    filterbank.values.assign(static_cast<size_t>(mel_bins * freq_bins), 0.0F);
    for (int64_t f = 0; f < freq_bins; ++f) {
        for (int64_t m = 0; m < mel_bins; ++m) {
            filterbank.values[static_cast<size_t>(m * freq_bins + f)] =
                values[static_cast<size_t>(f * mel_bins + m)];
        }
    }
    return filterbank;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_sheetsage2_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const SheetSage2Assets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract) {
    return std::make_unique<SheetSage2Session>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

std::shared_ptr<const SheetSage2Assets> load_sheetsage2_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<SheetSage2Assets>();
    assets->resources = model_spec::load_resource_bundle_for_family(model_path, kFamily);
    assets->config = parse_decoder_config(assets->resources);
    assets->weights = assets->resources.open_tensor_source("weights");
    assets::require_tensor_shape(*assets->weights, "token_embedding.weight", {assets->config.vocab_size, assets->config.hidden_size});
    assets::require_tensor_shape(
        *assets->weights,
        "encoder_projection.weight",
        {assets->config.hidden_size, assets->config.encoder_hidden_size});
    assets::require_tensor_shape(*assets->weights, "feature_extractor.mel_mean", {assets->config.mel_bins});
    assets::require_tensor_shape(*assets->weights, "feature_extractor.mel_std", {assets->config.mel_bins});
    assets::require_tensor_shape(
        *assets->weights,
        "feature_extractor.mel_scale.fb",
        {assets->config.n_fft / 2 + 1, assets->config.mel_bins});
    assets::require_tensor_shape(*assets->weights, "subsampling_module.0.convnext_layers.0.depthwise_block.1.weight", {128, 1, 7});
    assets::require_tensor_shape(
        *assets->weights,
        "layers.0.attn.query_proj.weight",
        {assets->config.encoder_hidden_size, assets->config.encoder_hidden_size});
    assets->mel_mean = assets->weights->require_f32("feature_extractor.mel_mean", {assets->config.mel_bins});
    assets->mel_std = assets->weights->require_f32("feature_extractor.mel_std", {assets->config.mel_bins});
    assets->stft_window = assets->weights->require_f32("feature_extractor.spectrogram.window", {assets->config.win_length});
    assets->mel_filterbank = engine::audio::MelFilterbank().prepare_sparse(
        load_mel_filterbank(*assets->weights, assets->config.n_fft, assets->config.mel_bins));
    return assets;
}

SheetSage2Session::SheetSage2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SheetSage2Assets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      encoder_(
          assets_->weights,
          execution_context(),
          assets_->config,
          decoder_options_from_session_options(RuntimeSessionBase::options())),
      decoder_(
          assets_->weights,
          execution_context(),
          assets_->config,
          decoder_options_from_session_options(RuntimeSessionBase::options())) {
    if (task_.task != runtime::VoiceTaskKind::Midi || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("SheetSage2 supports only offline midi");
    }
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, "SheetSage2");
}

SheetSage2Session::~SheetSage2Session() = default;

std::string SheetSage2Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind SheetSage2Session::task_kind() const {
    return task_.task;
}

runtime::RunMode SheetSage2Session::run_mode() const {
    return task_.mode;
}

void SheetSage2Session::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "SheetSage2");
    mark_prepared();
}

runtime::TaskResult SheetSage2Session::run(const runtime::TaskRequest & request) {
    require_prepared("SheetSage2 run");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "SheetSage2");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("SheetSage2 run() requires audio_input");
    }
    const auto total_start = Clock::now();
    const int64_t max_tokens = runtime::parse_positive_i64_option(
        request.options,
        {"max_tokens", "sheetsage2.max_tokens"},
        assets_->config.max_position_embeddings);
    if (max_tokens > assets_->config.max_position_embeddings) {
        throw std::runtime_error("SheetSage2 max_tokens exceeds the decoder context");
    }
    const SheetSage2Tokenizer tokenizer(assets_->config.input_audio_length_samples / assets_->config.sampling_rate);
    if (tokenizer.vocab_size != assets_->config.vocab_size) {
        throw std::runtime_error("SheetSage2 tokenizer vocabulary does not match the loaded model");
    }

    const auto frontend_start = Clock::now();
    const auto waveform = audio_frontend_.prepare(
        request.audio_input->samples,
        request.audio_input->sample_rate,
        request.audio_input->channels,
        static_cast<int>(assets_->config.sampling_rate),
        std::max(1, RuntimeSessionBase::options().backend.threads));
    if (static_cast<int64_t>(waveform.size()) <= assets_->config.n_fft) {
        throw std::runtime_error("SheetSage2 input audio is too short for the audio frontend");
    }
    engine::debug::timing_log_scalar("sheetsage2.frontend_ms", engine::debug::elapsed_ms(frontend_start, Clock::now()));

    const auto windows = sliding_window_plan(
        static_cast<int64_t>(waveform.size()),
        assets_->config.sampling_rate,
        assets_->config.input_audio_length_samples);
    engine::debug::trace_log_scalar("sheetsage2.windows", static_cast<int64_t>(windows.size()));
    engine::debug::trace_log_scalar(
        "sheetsage2.duration_seconds",
        static_cast<double>(waveform.size()) / static_cast<double>(assets_->config.sampling_rate));

    double frontend_windows_ms = 0.0;
    double encoder_ms = 0.0;
    double generation_ms = 0.0;
    std::vector<SheetSage2Event> stitched_events;
    std::vector<int32_t> last_ids;
    int64_t last_memory_steps = 0;
    const double window_seconds =
        static_cast<double>(assets_->config.input_audio_length_samples) / static_cast<double>(assets_->config.sampling_rate);

    for (const auto & window : windows) {
        const auto window_frontend_start = Clock::now();
        const auto window_audio = slice_window_audio(waveform, window.start_sample, assets_->config.input_audio_length_samples);
        int64_t mel_frames = 0;
        const auto normalized_mel = prepare_normalized_mel(
            window_audio,
            *assets_,
            static_cast<size_t>(std::max(1, RuntimeSessionBase::options().backend.threads)),
            mel_frames);
        frontend_windows_ms += engine::debug::elapsed_ms(window_frontend_start, Clock::now());

        const auto encoder_start = Clock::now();
        const auto mixed = encoder_.encode_mel(normalized_mel, mel_frames);
        const int64_t memory_steps = static_cast<int64_t>(mixed.size()) / assets_->config.encoder_hidden_size;
        encoder_ms += engine::debug::elapsed_ms(encoder_start, Clock::now());
        last_memory_steps = memory_steps;

        int64_t base_subbeat = 0;
        std::vector<int32_t> prefix_tokens;
        if (window.index > 0) {
            const auto prefix_events = build_overlap_prefix_events(stitched_events, tokenizer, window, base_subbeat);
            if (!prefix_events.empty()) {
                prefix_tokens = encode_events(tokenizer, prefix_events, false);
            }
        }

        const auto generation_start = Clock::now();
        last_ids = generate_tokens(
            decoder_,
            mixed,
            memory_steps,
            tokenizer,
            max_tokens,
            prefix_tokens.empty() ? nullptr : &prefix_tokens,
            window.generation_stop);
        generation_ms += engine::debug::elapsed_ms(generation_start, Clock::now());

        const auto decoded = decode_events(tokenizer, last_ids);
        const auto time_lookup = make_event_time_lookup(decoded, window_seconds);
        auto accepted = stitched_window_events(
            decoded,
            time_lookup,
            window,
            static_cast<double>(waveform.size()) / static_cast<double>(assets_->config.sampling_rate),
            base_subbeat);
        stitched_events.insert(
            stitched_events.end(),
            std::make_move_iterator(accepted.begin()),
            std::make_move_iterator(accepted.end()));
        engine::debug::trace_log_scalar("sheetsage2.window.index", window.index);
        engine::debug::trace_log_scalar("sheetsage2.window.prefix_tokens", static_cast<int64_t>(prefix_tokens.size()));
        engine::debug::trace_log_scalar("sheetsage2.window.tokens", static_cast<int64_t>(last_ids.size()));
    }
    encoder_.release_runtime_graphs();
    decoder_.release_runtime_graphs();

    std::sort(stitched_events.begin(), stitched_events.end(), [](const auto & a, const auto & b) {
        if (a.time != b.time) {
            return a.time < b.time;
        }
        return a.global_subbeat < b.global_subbeat;
    });
    engine::debug::timing_log_scalar("sheetsage2.frontend_windows_ms", frontend_windows_ms);
    engine::debug::timing_log_scalar("sheetsage2.encoder.total_ms", encoder_ms);
    engine::debug::timing_log_scalar("sheetsage2.decoder.generate_ms", generation_ms);

    const auto post_start = Clock::now();
    const auto abc = events_to_abc(
        stitched_events,
        static_cast<double>(waveform.size()) / static_cast<double>(assets_->config.sampling_rate));
    const auto json = events_json(stitched_events);
    engine::debug::timing_log_scalar("sheetsage2.postprocess_ms", engine::debug::elapsed_ms(post_start, Clock::now()));

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{abc, "abc"};
    result.artifact_output = runtime::make_text_artifact(
        runtime::ArtifactKind::Custom,
        "score",
        abc,
        {
            {"mime", "text/vnd.abc"},
            {"format", "abc"},
            {"extension", "abc"},
            {"tokens", std::to_string(last_ids.size())},
            {"memory_steps", std::to_string(last_memory_steps)},
            {"windows", std::to_string(windows.size())},
        });
    result.output_artifacts.push_back(runtime::make_text_artifact(
        runtime::ArtifactKind::Custom,
        "events",
        json,
        {
            {"mime", "application/json"},
            {"format", "sheetsage2-events-json"},
            {"extension", "json"},
        }));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sheetsage2_loader() {
    runtime::SpecBackedVoiceModelConfig<SheetSage2Assets> config;
    config.family = kFamily;
    config.load_assets = load_sheetsage2_assets;
    config.create_session = create_sheetsage2_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::sheetsage
