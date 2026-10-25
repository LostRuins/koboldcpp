#include "engine/models/yue2/request.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::yue2 {
namespace {

std::string request_style(const runtime::TaskRequest & request) {
    if (const auto style = runtime::find_option(request.options, {"style"})) {
        return *style;
    }
    if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto & tags = request.voice->style->tags;
        if (const auto it = tags.find("style"); it != tags.end()) {
            return it->second;
        }
        if (const auto it = tags.find("tags"); it != tags.end()) {
            return it->second;
        }
    }
    return {};
}

std::string request_lyrics(const runtime::TaskRequest & request) {
    if (const auto lyrics = runtime::find_option(request.options, {"lyrics"})) {
        return *lyrics;
    }
    if (request.text_input.has_value()) {
        return request.text_input->text;
    }
    return {};
}

std::string abc_from_options(const std::unordered_map<std::string, std::string> & options) {
    if (const auto abc = runtime::find_option(options, {"abc"})) {
        return *abc;
    }
    if (const auto abc_file = runtime::find_option(options, {"abc_file"})) {
        const std::filesystem::path path(*abc_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 abc_file does not exist: " + path.string());
        }
        return engine::io::read_text_file(path);
    }
    return {};
}

std::vector<int32_t> parse_semantic_prefix(const std::string & text, const std::string & source) {
    engine::io::json::Value root;
    try {
        root = engine::io::json::parse(text);
    } catch (const std::exception & error) {
        throw std::runtime_error("Yue2 " + source + " must be a JSON array of codec indices: " + error.what());
    }
    if (!root.is_array()) {
        throw std::runtime_error("Yue2 " + source + " must be a JSON array of codec indices");
    }
    const auto & items = root.as_array();
    if (items.empty()) {
        throw std::runtime_error("Yue2 " + source + " must contain at least one codec index");
    }
    std::vector<int32_t> out;
    out.reserve(items.size());
    for (size_t index = 0; index < items.size(); ++index) {
        const auto & item = items[index];
        if (!item.is_number()) {
            throw std::runtime_error(
                "Yue2 " + source + " entry " + std::to_string(index) + " is not a codec index");
        }
        int64_t value = 0;
        try {
            value = item.as_i64();
        } catch (const std::exception &) {
            throw std::runtime_error(
                "Yue2 " + source + " entry " + std::to_string(index) + " is not an integer");
        }
        if (value < 0 || value >= kCodecSize) {
            throw std::runtime_error(
                "Yue2 " + source + " entry " + std::to_string(index) + " is out of range [0," +
                std::to_string(kCodecSize) + ")");
        }
        out.push_back(static_cast<int32_t>(value));
    }
    return out;
}

std::vector<int32_t> semantic_prefix_from_options(const std::unordered_map<std::string, std::string> & options) {
    if (const auto prefix = runtime::find_option(options, {"semantic_prefix"})) {
        if (!prefix->empty()) {
            return parse_semantic_prefix(*prefix, "semantic_prefix");
        }
    }
    if (const auto prefix_file = runtime::find_option(options, {"semantic_prefix_file"})) {
        const std::filesystem::path path(*prefix_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 semantic_prefix_file does not exist: " + path.string());
        }
        return parse_semantic_prefix(engine::io::read_text_file(path), "semantic_prefix_file");
    }
    return {};
}

void apply_options(
    Yue2Request & out,
    const std::unordered_map<std::string, std::string> & options) {
    if (const auto cot = runtime::find_option(options, {"cot"})) {
        out.cot = parse_cot_mode(*cot);
    }
    if (const auto export_semantic = runtime::find_option(options, {"export_semantic"})) {
        out.export_semantic = runtime::parse_bool_option(*export_semantic, "export_semantic");
    }
    if (const auto stop_after = runtime::find_option(options, {"stop_after"})) {
        out.stop_after = parse_stop_after(*stop_after);
    }
    if (const auto seed = runtime::parse_u64_option(options, {"seed"})) {
        out.seed = *seed;
    }
    if (out.seed >= (uint64_t{1} << 63U)) {
        throw std::runtime_error("Yue2 seed must be in [0, 2^63)");
    }
    if (const auto value = runtime::parse_finite_float_option(options, {"guidance_scale", "cfg_scale"})) {
        if (*value < 0.0F || *value > 20.0F) {
            throw std::runtime_error("Yue2 guidance_scale must be in [0,20]");
        }
        out.cfg_scale = *value;
    }
    out.generation.ode_steps =
        runtime::parse_positive_i64_option(options, {"num_inference_steps"}, Yue2GenerationConfig{}.ode_steps);
    if (out.generation.ode_steps <= 0) {
        throw std::runtime_error("Yue2 num_inference_steps must be positive");
    }
    auto apply_sampling = [&options](Yue2SamplingConfig & sampling, const std::string & prefix) {
        if (const auto value = runtime::parse_finite_float_option(options, {prefix + "_temperature"})) {
            sampling.temperature = *value;
        }
        if (const auto value = runtime::parse_finite_float_option(options, {prefix + "_top_p"})) {
            sampling.top_p = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_top_k"})) {
            sampling.top_k = *value;
        }
        if (const auto value = runtime::parse_finite_float_option(options, {prefix + "_repetition_penalty"})) {
            sampling.repetition_penalty = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_penalty_window"})) {
            sampling.penalty_window = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_min_tokens"})) {
            sampling.min_tokens = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_max_tokens"})) {
            sampling.max_tokens = *value;
        }
    };
    auto validate_sampling = [](const Yue2SamplingConfig & sampling, const std::string & prefix) {
        if (sampling.temperature < 0.0F || sampling.temperature > 5.0F ||
            sampling.top_p <= 0.0F || sampling.top_p > 1.0F ||
            sampling.top_k < 1 ||
            sampling.repetition_penalty <= 0.0F ||
            sampling.penalty_window < 1 ||
            sampling.min_tokens < 0 ||
            sampling.max_tokens < sampling.min_tokens) {
            throw std::runtime_error("Yue2 " + prefix + " sampling options are invalid");
        }
    };
    apply_sampling(out.generation.abc, "abc");
    apply_sampling(out.generation.semantic, "semantic");
    validate_sampling(out.generation.abc, "abc");
    validate_sampling(out.generation.semantic, "semantic");
    out.abc = abc_from_options(options);
    if (!out.abc.empty() && out.cot == Yue2CotMode::Off) {
        throw std::runtime_error("Yue2 external ABC requires cot=melody or cot=full");
    }
    if (out.stop_after == Yue2StopAfter::Semantic) {
        // The semantic token stream is the only product of this stage.
        out.export_semantic = true;
    }
    if (out.stop_after == Yue2StopAfter::Abc) {
        if (out.cot == Yue2CotMode::Off) {
            throw std::runtime_error("Yue2 stop_after=abc requires cot=melody or cot=full");
        }
        if (!out.abc.empty()) {
            throw std::runtime_error("Yue2 stop_after=abc generates no score when abc or abc_file is supplied");
        }
    }
    out.semantic_prefix = semantic_prefix_from_options(options);
    if (!out.semantic_prefix.empty()) {
        if (static_cast<int64_t>(out.semantic_prefix.size()) > out.generation.semantic.max_tokens) {
            throw std::runtime_error("Yue2 semantic prefix is longer than semantic_max_tokens");
        }
        if (out.cot != Yue2CotMode::Off && out.abc.empty()) {
            throw std::runtime_error("Yue2 semantic prefix requires abc or abc_file unless cot=off");
        }
    }
    if (const auto nar_noise_file = runtime::find_option(options, {"nar_noise_file"})) {
        const std::filesystem::path path(*nar_noise_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 nar_noise_file does not exist: " + path.string());
        }
        out.nar_noise = engine::io::read_f32_file(path);
        if (out.nar_noise.empty()) {
            throw std::runtime_error("Yue2 nar_noise_file is empty: " + path.string());
        }
        if (out.nar_noise.size() % static_cast<size_t>(Yue2ModelConfig{}.latent_dim) != 0) {
            throw std::runtime_error("Yue2 nar_noise_file must contain raw float32 acoustic noise rows with 64 columns");
        }
    }
}

Yue2Request normalize_request(Yue2Request out) {
    if (out.style.empty()) {
        throw std::runtime_error("Yue2 requires non-empty style");
    }
    return out;
}

}  // namespace

Yue2Request parse_yue2_request(const runtime::TaskRequest & request, const Yue2GenerationConfig & defaults) {
    if (request.audio_input.has_value()) {
        throw std::runtime_error("Yue2 does not consume audio_input");
    }
    if (!request.input_artifacts.empty()) {
        throw runtime::InvalidRequestError("Yue2 does not consume input artifacts");
    }
    Yue2Request out;
    out.generation = defaults;
    out.style = request_style(request);
    out.lyrics = request_lyrics(request);
    apply_options(out, request.options);
    return normalize_request(std::move(out));
}

Yue2Request parse_yue2_preparation_request(
    const runtime::SessionPreparationRequest & request,
    const Yue2GenerationConfig & defaults) {
    Yue2Request out;
    out.generation = defaults;
    if (request.text.has_value()) {
        out.lyrics = request.text->text;
    }
    if (const auto style = runtime::find_option(request.options, {"style"})) {
        out.style = *style;
    }
    apply_options(out, request.options);
    return out;
}

}  // namespace engine::models::yue2
