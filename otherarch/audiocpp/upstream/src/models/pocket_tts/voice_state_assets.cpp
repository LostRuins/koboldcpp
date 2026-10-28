#include "engine/models/pocket_tts/voice_state_assets.h"

#include "engine/framework/io/binary.h"
#include "engine/framework/io/safetensors.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <regex>
#include <stdexcept>

namespace engine::models::pocket_tts {
namespace {

void append_bytes(std::vector<unsigned char> & out, const void * data, size_t size) {
    if (size == 0) {
        return;
    }
    const auto * bytes = static_cast<const unsigned char *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

std::vector<unsigned char> make_i64_bytes(int64_t value) {
    std::vector<unsigned char> data;
    data.reserve(sizeof(int64_t));
    append_bytes(data, &value, sizeof(int64_t));
    return data;
}

size_t element_count_for_shape(const std::vector<int64_t> & shape, const std::string & name) {
    size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("tensor shape contains a negative dimension: " + name);
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

std::pair<const std::byte *, size_t> require_safetensor_data(
    const io::SafeTensorIndex & index,
    const io::BinaryBlob & bytes,
    const io::SafeTensorInfo & info) {
    const size_t byte_size = info.data_end - info.data_begin;
    // Subtractions against the total, so no term can overflow the comparison
    // and let an out-of-range span through. See the matching check in
    // framework/assets/tensor_source.cpp.
    const size_t total = bytes.size();
    if (index.header_bytes > total || byte_size > total - index.header_bytes ||
        info.data_begin > total - index.header_bytes - byte_size) {
        throw std::runtime_error("tensor data range is out of bounds: " + info.name);
    }
    const size_t data_offset = index.header_bytes + info.data_begin;
    return {bytes.data() + static_cast<std::ptrdiff_t>(data_offset), byte_size};
}

std::vector<float> read_voice_state_f32(
    const io::SafeTensorIndex & index,
    const io::BinaryBlob & bytes,
    const io::SafeTensorInfo & info) {
    if (info.dtype != "F32") {
        throw std::runtime_error("Voice state cache tensor must be F32: " + info.name);
    }
    const size_t count = element_count_for_shape(info.shape, info.name);
    const auto [data, byte_size] = require_safetensor_data(index, bytes, info);
    if (byte_size != count * sizeof(float)) {
        throw std::runtime_error("Voice state cache tensor byte size mismatch: " + info.name);
    }
    std::vector<float> values(count);
    if (!values.empty()) {
        std::memcpy(values.data(), data, byte_size);
    }
    return values;
}

int64_t read_voice_state_i64_scalar(
    const io::SafeTensorIndex & index,
    const io::BinaryBlob & bytes,
    const io::SafeTensorInfo & info) {
    if (info.dtype != "I64" || info.data_end - info.data_begin != sizeof(int64_t)) {
        throw std::runtime_error("Voice state offset tensor must be an I64 scalar: " + info.name);
    }
    const auto [data, byte_size] = require_safetensor_data(index, bytes, info);
    (void) byte_size;
    int64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

}  // namespace

VoiceStateAssets load_voice_state_assets(const std::filesystem::path & source) {
    const auto index = io::load_safetensors_index(source);
    const auto bytes = io::read_binary_blob(index.source_path);
    std::map<int64_t, VoiceAttentionCache> layers;
    const std::regex layer_pattern(R"(^transformer\.layers\.(\d+)\.self_attn$)");

    for (const auto & [name, tensor] : index.tensors) {
        const auto slash = name.rfind('/');
        if (slash == std::string::npos) {
            continue;
        }
        const std::string module_name = name.substr(0, slash);
        const std::string tensor_key = name.substr(slash + 1);
        std::smatch match;
        if (!std::regex_match(module_name, match, layer_pattern)) {
            continue;
        }
        const int64_t layer_index = std::stoll(match[1].str());
        auto & layer = layers[layer_index];

        if (tensor_key == "cache") {
            const auto values = read_voice_state_f32(index, bytes, tensor);
            if (tensor.shape.size() == 5) {
                layer.cached_steps = tensor.shape[2];
                layer.heads = tensor.shape[3];
                layer.head_dim = tensor.shape[4];
            } else if (tensor.shape.size() == 4) {
                layer.cached_steps = tensor.shape[1];
                layer.heads = tensor.shape[2];
                layer.head_dim = tensor.shape[3];
            } else {
                throw std::runtime_error("Voice state cache tensor must have rank 4 or 5: " + tensor.name);
            }
            const size_t single_size = static_cast<size_t>(layer.cached_steps * layer.heads * layer.head_dim);
            if (values.size() != single_size * 2U) {
                throw std::runtime_error("Voice state cache tensor size mismatch: " + tensor.name);
            }
            layer.key.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(single_size));
            layer.value.assign(values.begin() + static_cast<std::ptrdiff_t>(single_size), values.end());
        } else if (tensor_key == "offset") {
            layer.offset = read_voice_state_i64_scalar(index, bytes, tensor);
        } else if (tensor_key == "current_end") {
            if (tensor.shape.empty()) {
                throw std::runtime_error("Voice state current_end tensor must have shape: " + tensor.name);
            }
            layer.offset = tensor.shape[0];
        }
    }

    if (layers.empty()) {
        throw std::runtime_error("No transformer voice cache layers found in " + source.string());
    }

    VoiceStateAssets state;
    state.transformer_layers.reserve(layers.size());
    for (auto & [layer_index, layer] : layers) {
        (void) layer_index;
        if (layer.key.empty() || layer.value.empty()) {
            throw std::runtime_error("Incomplete voice state cache layer in " + source.string());
        }
        state.transformer_layers.push_back(std::move(layer));
    }
    return state;
}

void save_voice_state_assets(
    const std::filesystem::path & destination,
    const runtime::TransformerKVState & state,
    int64_t heads,
    int64_t head_dim) {
    if (state.layers.empty()) {
        throw std::runtime_error("PocketTTS voice state export requires cache layers");
    }
    if (heads <= 0 || head_dim <= 0 || state.current_end < 0) {
        throw std::runtime_error("PocketTTS voice state export received invalid cache shape");
    }
    const int64_t step_elems = heads * head_dim;
    const size_t valid_elems = static_cast<size_t>(state.current_end * step_elems);

    std::vector<io::SafeTensorWriteEntry> entries;
    entries.reserve(state.layers.size() * 2U);
    for (size_t layer_index = 0; layer_index < state.layers.size(); ++layer_index) {
        const auto & layer = state.layers[layer_index];
        if (layer.valid_steps != state.current_end) {
            throw std::runtime_error("PocketTTS voice state export requires aligned layer offsets");
        }
        if (layer.key.size() < valid_elems || layer.value.size() < valid_elems) {
            throw std::runtime_error("PocketTTS voice state export cache is smaller than its offset");
        }

        io::SafeTensorWriteEntry cache;
        cache.name = "transformer.layers." + std::to_string(layer_index) + ".self_attn/cache";
        cache.dtype = "F32";
        cache.shape = {2, 1, state.current_end, heads, head_dim};
        cache.data.reserve(valid_elems * 2U * sizeof(float));
        append_bytes(cache.data, layer.key.data(), valid_elems * sizeof(float));
        append_bytes(cache.data, layer.value.data(), valid_elems * sizeof(float));
        entries.push_back(std::move(cache));

        io::SafeTensorWriteEntry offset;
        offset.name = "transformer.layers." + std::to_string(layer_index) + ".self_attn/offset";
        offset.dtype = "I64";
        offset.shape = {1};
        offset.data = make_i64_bytes(state.current_end);
        entries.push_back(std::move(offset));
    }

    io::write_safetensors_file(destination, entries);
}

std::filesystem::path preset_embedding_path(const std::filesystem::path & model_root, const std::string & preset_name) {
    return model_root / "embeddings" / (preset_name + ".safetensors");
}

VoiceStateAssets load_voice_assets_for_plan(const VoiceConditioningPlan & plan, const PocketTTSAssets &) {
    switch (plan.source) {
        case VoiceSourceKind::NamedPreset:
        case VoiceSourceKind::PreparedEmbedding:
            if (plan.asset_path.empty()) {
                throw std::runtime_error("PocketTTS preset voice conditioning requires a voice state asset path");
            }
            return load_voice_state_assets(plan.asset_path);
        case VoiceSourceKind::CloneAudio:
            throw std::runtime_error("Clone audio does not load precomputed voice state assets");
    }
    throw std::runtime_error("PocketTTS voice assets received unknown voice source");
}

}  // namespace engine::models::pocket_tts
