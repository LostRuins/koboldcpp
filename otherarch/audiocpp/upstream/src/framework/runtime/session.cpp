#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/task_vocabulary.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace engine::runtime {

namespace {

int64_t require_smallest_fitting_capacity(
    const std::vector<int64_t> & capacities,
    int64_t request_size);

}

MappedGraphCapacityAdapter::MappedGraphCapacityAdapter(
    int64_t base_capacity,
    int64_t fixed_capacity,
    GraphCapacityCanonicalFn canonical_capacity_fn,
    GraphCapacityPreparedFn prepared_capacities_fn,
    GraphCapacityPrepareFn prepare_capacity_fn)
    : base_capacity_(base_capacity),
      fixed_capacity_(fixed_capacity),
      canonical_capacity_fn_(std::move(canonical_capacity_fn)),
      prepared_capacities_fn_(std::move(prepared_capacities_fn)),
      prepare_capacity_fn_(std::move(prepare_capacity_fn)) {}

int64_t MappedGraphCapacityAdapter::base_capacity() const {
    return base_capacity_;
}

int64_t MappedGraphCapacityAdapter::fixed_capacity() const {
    return fixed_capacity_;
}

int64_t MappedGraphCapacityAdapter::canonical_capacity_for_request(int64_t request_size) const {
    return canonical_capacity_fn_(request_size);
}

std::vector<int64_t> MappedGraphCapacityAdapter::prepared_capacities() const {
    return prepared_capacities_fn_();
}

void MappedGraphCapacityAdapter::prepare_capacity(int64_t capacity) {
    prepare_capacity_fn_(capacity);
}

DiscreteGraphCapacityAdapter::DiscreteGraphCapacityAdapter(
    std::vector<int64_t> capacities,
    GraphCapacityPreparedFn prepared_capacities_fn,
    GraphCapacityPrepareFn prepare_capacity_fn)
    : capacities_(std::move(capacities)),
      prepared_capacities_fn_(std::move(prepared_capacities_fn)),
      prepare_capacity_fn_(std::move(prepare_capacity_fn)) {
    std::sort(capacities_.begin(), capacities_.end());
    capacities_.erase(std::unique(capacities_.begin(), capacities_.end()), capacities_.end());
    if (capacities_.empty()) {
        throw std::runtime_error("discrete graph capacity list must not be empty");
    }
    if (capacities_.front() <= 0) {
        throw std::runtime_error("discrete graph capacities must be positive");
    }
}

int64_t DiscreteGraphCapacityAdapter::base_capacity() const {
    return capacities_.front();
}

int64_t DiscreteGraphCapacityAdapter::fixed_capacity() const {
    return capacities_.front();
}

int64_t DiscreteGraphCapacityAdapter::canonical_capacity_for_request(int64_t request_size) const {
    return require_smallest_fitting_capacity(capacities_, request_size);
}

std::vector<int64_t> DiscreteGraphCapacityAdapter::prepared_capacities() const {
    return prepared_capacities_fn_();
}

void DiscreteGraphCapacityAdapter::prepare_capacity(int64_t capacity) {
    if (!std::binary_search(capacities_.begin(), capacities_.end(), capacity)) {
        throw std::runtime_error("selected graph capacity is not configured");
    }
    prepare_capacity_fn_(capacity);
}

const char * to_string(VoiceTaskKind task) noexcept {
    std::size_t count = 0;
    const auto * entries = task_vocabulary(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (entries[i].kind == task) {
            return entries[i].token.data();
        }
    }
    return "unknown";
}

const char * to_string(RunMode mode) noexcept {
    switch (mode) {
    case RunMode::Offline:
        return "offline";
    case RunMode::Streaming:
        return "streaming";
    }
    return "unknown";
}

GraphCapacityController::GraphCapacityController(GraphCapacityMode mode)
    : mode_(mode) {}

GraphCapacityMode GraphCapacityController::mode() const noexcept {
    return mode_;
}

const char * to_string(GraphCapacityMode mode) noexcept {
    switch (mode) {
    case GraphCapacityMode::Fixed:
        return "fixed";
    case GraphCapacityMode::Tiered:
        return "tiered";
    case GraphCapacityMode::Grow:
        return "grow";
    case GraphCapacityMode::Double:
        return "double";
    case GraphCapacityMode::Unsupported:
        return "unsupported";
    }
    return "unknown";
}

VoiceTaskKind parse_voice_task_kind(const std::string & value) {
    std::size_t count = 0;
    const auto * entries = task_vocabulary(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (entries[i].token == value) {
            return entries[i].kind;
        }
    }
    // The list in this message was a fifth copy of the vocabulary; it is built
    // from the table now, so a task kind added later cannot leave behind an
    // error message that says it does not exist.
    std::string expected;
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) {
            expected += ", ";
        }
        if (i + 1 == count) {
            expected += "or ";
        }
        expected.append(entries[i].token);
    }
    throw std::runtime_error("unsupported task: " + value + " (expected " + expected + ")");
}

RunMode parse_run_mode(const std::string & value) {
    if (value == "offline") {
        return RunMode::Offline;
    }
    if (value == "streaming") {
        return RunMode::Streaming;
    }
    throw std::runtime_error("unsupported run mode: " + value);
}

std::vector<TaskRequest> chunk_text_request(const TaskRequest & request, int64_t codepoint_budget) {
    return chunk_text_request(request, codepoint_budget, engine::text::TextChunkMode::Default);
}

std::vector<TaskRequest> chunk_text_request(
    const TaskRequest & request,
    int64_t codepoint_budget,
    engine::text::TextChunkMode mode) {
    if (!request.text_input.has_value()) {
        return {request};
    }
    const auto chunks = engine::text::split_text_chunks(request.text_input->text, codepoint_budget, mode);
    if (chunks.empty()) {
        return {request};
    }
    std::vector<TaskRequest> requests;
    requests.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        TaskRequest item = request;
        item.text_input->text = chunk;
        requests.push_back(std::move(item));
    }
    return requests;
}

void append_audio_buffer(AudioBuffer & dst, const AudioBuffer & src) {
    if (src.sample_rate <= 0 || src.channels <= 0) {
        throw std::runtime_error("audio append requires valid source format");
    }
    if (dst.sample_rate == 0) {
        dst.sample_rate = src.sample_rate;
        dst.channels = src.channels;
    } else if (dst.sample_rate != src.sample_rate || dst.channels != src.channels) {
        throw std::runtime_error("audio append requires matching audio format");
    }
    dst.samples.insert(dst.samples.end(), src.samples.begin(), src.samples.end());
}

std::vector<std::byte> bytes_from_string(std::string_view value) {
    std::vector<std::byte> out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        out.push_back(static_cast<std::byte>(ch));
    }
    return out;
}

VoiceArtifact make_voice_artifact(
    ArtifactKind kind,
    std::string id,
    std::vector<std::byte> payload,
    std::unordered_map<std::string, std::string> meta) {
    VoiceArtifact artifact;
    artifact.kind = kind;
    artifact.id = std::move(id);
    artifact.payload = std::move(payload);
    artifact.meta = std::move(meta);
    return artifact;
}

VoiceArtifact make_text_artifact(
    ArtifactKind kind,
    std::string id,
    std::string_view payload,
    std::unordered_map<std::string, std::string> meta) {
    return make_voice_artifact(kind, std::move(id), bytes_from_string(payload), std::move(meta));
}

GraphCapacityMode parse_graph_capacity_mode(const std::string & value) {
    if (value == "fixed") {
        return GraphCapacityMode::Fixed;
    }
    if (value == "tiered") {
        return GraphCapacityMode::Tiered;
    }
    if (value == "grow") {
        return GraphCapacityMode::Grow;
    }
    if (value == "double") {
        return GraphCapacityMode::Double;
    }
    if (value == "unsupported") {
        return GraphCapacityMode::Unsupported;
    }
    throw std::runtime_error("unsupported graph capacity mode: " + value);
}

GraphCapacityMode resolve_graph_capacity_mode(
    const SessionOptions & options,
    GraphCapacityMode default_mode) {
    return resolve_graph_capacity_mode(options, default_mode, {"graph_capacity_mode"});
}

GraphCapacityMode resolve_graph_capacity_mode(
    const SessionOptions & options,
    GraphCapacityMode default_mode,
    std::initializer_list<const char *> keys) {
    for (const char * key : keys) {
        const auto it = options.options.find(key);
        if (it == options.options.end() || it->second.empty()) {
            continue;
        }
        return parse_graph_capacity_mode(it->second);
    }
    return default_mode;
}

namespace {

int64_t require_smallest_fitting_capacity(
    const std::vector<int64_t> & capacities,
    int64_t request_size) {
    if (request_size <= 0) {
        throw std::runtime_error("graph capacity request size must be positive");
    }
    if (capacities.empty()) {
        throw std::runtime_error("graph capacity list must not be empty");
    }
    auto it = std::lower_bound(capacities.begin(), capacities.end(), request_size);
    if (it == capacities.end()) {
        throw std::runtime_error("request exceeds configured graph capacity");
    }
    return *it;
}

std::vector<int64_t> sorted_prepared_capacities(const IGraphCapacityAdapter & adapter) {
    auto capacities = adapter.prepared_capacities();
    std::sort(capacities.begin(), capacities.end());
    capacities.erase(std::unique(capacities.begin(), capacities.end()), capacities.end());
    return capacities;
}

int64_t select_smallest_prepared_fitting_capacity(
    const std::vector<int64_t> & prepared_capacities,
    int64_t request_size) {
    for (const int64_t capacity : prepared_capacities) {
        if (capacity >= request_size) {
            return capacity;
        }
    }
    return 0;
}

int64_t checked_double_capacity_request(int64_t request_size) {
    if (request_size <= 0) {
        throw std::runtime_error("graph capacity request size must be positive");
    }
    if (request_size > (std::numeric_limits<int64_t>::max() / 2)) {
        throw std::runtime_error("graph capacity request size is too large to double");
    }
    return request_size * 2;
}

}  // namespace

int64_t GraphCapacityController::ensure_prepared(IGraphCapacityAdapter & adapter, int64_t request_size) const {
    const auto prepared_capacities = sorted_prepared_capacities(adapter);
    int64_t selected_capacity = 0;
    switch (mode_) {
    case GraphCapacityMode::Fixed: {
        selected_capacity = adapter.fixed_capacity();
        if (selected_capacity <= 0) {
            throw std::runtime_error("fixed graph capacity must be positive");
        }
        if (request_size > 0 && request_size > selected_capacity) {
            throw std::runtime_error("request exceeds fixed graph capacity");
        }
        break;
    }
    case GraphCapacityMode::Tiered:
        if (request_size <= 0) {
            selected_capacity = adapter.base_capacity();
            break;
        }
        selected_capacity = select_smallest_prepared_fitting_capacity(prepared_capacities, request_size);
        if (selected_capacity == 0) {
            selected_capacity = adapter.canonical_capacity_for_request(request_size);
        }
        break;
    case GraphCapacityMode::Grow:
        if (request_size <= 0) {
            selected_capacity = prepared_capacities.empty() ? adapter.base_capacity() : prepared_capacities.back();
            break;
        }
        selected_capacity = prepared_capacities.empty() ? 0 : prepared_capacities.back();
        if (selected_capacity < request_size) {
            selected_capacity = adapter.canonical_capacity_for_request(request_size);
        }
        break;
    case GraphCapacityMode::Double:
        if (request_size <= 0) {
            selected_capacity = prepared_capacities.empty() ? adapter.base_capacity() : prepared_capacities.back();
            break;
        }
        selected_capacity = prepared_capacities.empty() ? 0 : prepared_capacities.back();
        if (selected_capacity < request_size) {
            selected_capacity = adapter.canonical_capacity_for_request(checked_double_capacity_request(request_size));
        }
        break;
    case GraphCapacityMode::Unsupported:
        throw std::runtime_error("graph capacity mode is unsupported");
    }
    if (selected_capacity <= 0) {
        throw std::runtime_error("graph capacity selection must be positive");
    }
    if (std::find(prepared_capacities.begin(), prepared_capacities.end(), selected_capacity) == prepared_capacities.end()) {
        adapter.prepare_capacity(selected_capacity);
    }
    return selected_capacity;
}

int64_t GraphCapacityController::select_capacity_for_run(
    const IGraphCapacityAdapter & adapter,
    int64_t request_size) const {
    if (request_size <= 0) {
        throw std::runtime_error("graph capacity request size must be positive");
    }
    const auto prepared_capacities = sorted_prepared_capacities(adapter);
    switch (mode_) {
    case GraphCapacityMode::Fixed: {
        const int64_t capacity = adapter.fixed_capacity();
        if (capacity <= 0) {
            throw std::runtime_error("fixed graph capacity must be positive");
        }
        if (request_size > capacity) {
            throw std::runtime_error("request exceeds fixed graph capacity");
        }
        return capacity;
    }
    case GraphCapacityMode::Tiered: {
        const int64_t prepared_capacity = select_smallest_prepared_fitting_capacity(prepared_capacities, request_size);
        if (prepared_capacity > 0) {
            return prepared_capacity;
        }
        return adapter.canonical_capacity_for_request(request_size);
    }
    case GraphCapacityMode::Grow:
        if (!prepared_capacities.empty() && prepared_capacities.back() >= request_size) {
            return prepared_capacities.back();
        }
        return adapter.canonical_capacity_for_request(request_size);
    case GraphCapacityMode::Double:
        if (!prepared_capacities.empty() && prepared_capacities.back() >= request_size) {
            return prepared_capacities.back();
        }
        return adapter.canonical_capacity_for_request(checked_double_capacity_request(request_size));
    case GraphCapacityMode::Unsupported:
        throw std::runtime_error("graph capacity mode is unsupported");
    }
    throw std::runtime_error("unknown graph capacity mode");
}

SessionPreparationRequest build_preparation_request(const AudioBuffer & audio) {
    SessionPreparationRequest request;
    request.audio = AudioPreparationContract{
        audio.sample_rate,
        audio.channels,
        static_cast<int64_t>(audio.samples.size()),
    };
    return request;
}

SessionPreparationRequest build_preparation_request(const TaskRequest & request) {
    SessionPreparationRequest prep;
    prep.options = request.options;
    // ⚠ CARRY THE LIST OPTIONS TOO. Preparation decides how the graphs are sized, and a family
    // whose work depends on a list option would otherwise size itself for a different request
    // than the one run() is handed. For kokoro_tts that was not merely a bad estimate: with the
    // phonemes missing here, preparation fell back to the built-in G2P, and a Japanese request
    // failed for want of UniDic even though the caller had supplied phonemes precisely so that
    // the G2P would never be consulted.
    prep.option_arrays = request.option_arrays;
    prep.text = request.text_input;
    prep.voice = request.voice;
    if (request.audio_input.has_value()) {
        prep.audio = AudioPreparationContract{
            request.audio_input->sample_rate,
            request.audio_input->channels,
            static_cast<int64_t>(request.audio_input->samples.size()),
        };
    }
    return prep;
}

}  // namespace engine::runtime
