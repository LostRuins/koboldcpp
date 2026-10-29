#include "engine/models/ming3_tts/prompt.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <set>
#include <stdexcept>

namespace engine::models::ming3_tts {

MingPromptBuilder::MingPromptBuilder(tokenizers::LlamaBpeTokenizerSpec tokenizer)
    : tokenizer_(std::move(tokenizer)) {
    const std::array<std::pair<const char *, int32_t>, 8> tokens{{
        {"<|endoftext|>", 151643}, {"<|im_start|>", 151644}, {"<|im_end|>", 151645},
        {"<|vision_start|>", 151652}, {"<|vision_end|>", 151653}, {"<|vision_pad|>", 151654},
        {"<audioPatch>", 151705}, {"<audio>", 151706}}};
    for (const auto & [token, id] : tokens) {
        if (tokenizer_.find_token_id(token) != id || tokenizer_.encode(token, true) != std::vector<int32_t>{id}) {
            throw std::runtime_error("Ming tokenizer does not match the trained token ID for " + std::string(token));
        }
    }
}

MingPrompt MingPromptBuilder::build(const std::string & text, const MingPromptOptions & options,
    size_t speakers, const std::string & reference_text, size_t latent_steps) const {
    std::string task = io::trim_ascii_whitespace(options.template_name);
    std::transform(task.begin(), task.end(), task.begin(), [](unsigned char c) {
        return c == '-' ? '_' : static_cast<char>(std::toupper(c));
    });
    static const std::map<std::string, std::string> aliases{
        {"TTS", "TTS"}, {"SEED", "TTS"}, {"PLAIN", "TTS"}, {"MULTILINGUAL", "MULTILINGUAL"},
        {"MULTILINGUAL_TTS", "MULTILINGUAL"}, {"BASIC", "BASIC"}, {"EMOTION", "EMOTION"},
        {"DIALECT", "DIALECT"}, {"IP", "IP"}, {"VOICE_DESIGN", "VOICE_DESIGN"},
        {"STYLE", "VOICE_DESIGN"}, {"FREEFORM", "FREEFORM"}};
    const auto found = aliases.find(task);
    if (found == aliases.end()) throw std::runtime_error("unsupported Ming template_name: " + options.template_name);
    task = found->second;
    std::string language = io::trim_ascii_whitespace(options.language);
    if (task == "DIALECT" && !language.empty()) {
        debug::log_message(debug::LogLevel::Warning, "ming3_tts", "The dialect template ignores language");
        language.clear();
    }
    if (task == "MULTILINGUAL" && language.empty()) throw std::runtime_error("Ming multilingual requires language");
    std::string instruction = io::trim_ascii_whitespace(options.instruct);
    if (task == "FREEFORM" && instruction.empty()) throw std::runtime_error("Ming freeform requires instruct");
    static const std::map<std::string, std::set<std::string>> allowed{
        {"BASIC", {"speed", "pitch", "volume"}}, {"EMOTION", {"emotion"}}, {"DIALECT", {"dialect"}},
        {"VOICE_DESIGN", {"style"}}, {"IP", {"ip"}}};
    if (const auto fields = allowed.find(task); fields != allowed.end()) {
        if (!instruction.empty()) throw std::runtime_error("Ming " + task + " uses structured controls, not raw instruct");
        if (options.controls.empty()) throw std::runtime_error("Ming " + task + " requires controls");
        for (const auto & [name, value] : options.controls) {
            (void) value;
            if (!fields->second.count(name)) throw std::runtime_error("unsupported Ming " + task + " control: " + name);
        }
        auto schema = io::trim_ascii_whitespace(options.instruction_schema);
        std::transform(schema.begin(), schema.end(), schema.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (schema.empty()) schema = task == "DIALECT" ? "en" : "zh";
        if (schema != "en" && schema != "english" && schema != "zh" && schema != "chinese") {
            throw std::runtime_error("unsupported Ming instruction_schema: " + schema);
        }
        const bool english = schema == "en" || schema == "english";
        const std::array<const char *, 8> names = english
            ? std::array<const char *, 8>{"id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion"}
            : std::array<const char *, 8>{u8"\u5e8f\u53f7", u8"\u8bf4\u8bdd\u4eba", u8"\u65b9\u8a00", u8"\u98ce\u683c",
                u8"\u8bed\u901f", u8"\u57fa\u9891", u8"\u97f3\u91cf", u8"\u60c5\u611f"};
        const std::array<const char *, 6> controls{"dialect", "style", "speed", "pitch", "volume", "emotion"};
        instruction = "{\"audio_sequence\": [{" + io::json::stringify_string(names[0]) + ": 1, " +
            io::json::stringify_string(names[1]) + ": \"speaker_1\"";
        for (size_t i = 0; i < controls.size(); ++i) {
            const auto value = options.controls.find(controls[i]);
            instruction += ", " + io::json::stringify_string(names[i + 2]) + ": " +
                (value == options.controls.end() ? std::string("null") : io::json::stringify_string(value->second));
        }
        instruction += ", \"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, \"ENV\": null, \"SNR\": null}, ";
        instruction += io::json::stringify_string(english ? "ip" : "IP") + ": ";
        const auto ip = options.controls.find("ip");
        instruction += ip == options.controls.end() ? "null" : io::json::stringify_string(ip->second);
        instruction += "}]}";
    } else if (!options.controls.empty()) {
        throw std::runtime_error("Ming " + task + " does not accept structured controls");
    }
    std::string prompt = "Please generate speech based on the following description.\n";
    if (!language.empty()) {
        std::string key = language;
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        static const std::map<std::string, std::string> languages{
            {"zh", "Chinese"}, {"zh-cn", "Chinese"}, {"chinese", "Chinese"}, {"en", "English"}, {"english", "English"},
            {"ja", "Japanese"}, {"japanese", "Japanese"}, {"ko", "Korean"}, {"korean", "Korean"}, {"de", "German"},
            {"german", "German"}, {"fr", "French"}, {"french", "French"}, {"es", "Spanish"}, {"spanish", "Spanish"},
            {"yue", "Chinese"}, {"zh-yue", "Chinese"}, {"cantonese", "Chinese"}};
        const auto canonical = languages.find(key);
        prompt += "Target language: " + (canonical == languages.end() ? language : canonical->second) + ".\n";
    }
    const auto target = io::trim_ascii_whitespace(text);
    if (target.empty()) throw std::runtime_error("Ming text must not be empty");
    std::vector<std::string> pieces{"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "<|im_start|>user\n", prompt};
    for (size_t i = 0; i < speakers; ++i) {
        pieces.push_back("  speaker_" + std::to_string(i + 1) + ":");
        pieces.insert(pieces.end(), {"<|vision_start|>", "<|vision_pad|>", "<|vision_end|>\n"});
    }
    const std::array<const char *, 5> bgm{"Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "};
    if (!std::all_of(bgm.begin(), bgm.end(), [&](const char * name) { return target.find(name) != std::string::npos; })) {
        pieces.emplace_back(" Text input:\n");
    }
    if (latent_steps && !reference_text.empty()) pieces.push_back(reference_text);
    pieces.insert(pieces.end(), {target, "<|im_end|>\n", "<|im_start|>assistant\n"});
    if (!instruction.empty()) pieces.insert(pieces.end(), {instruction, "<|im_end|>"});
    pieces.emplace_back("<audio>");
    MingPrompt result;
    for (const auto & piece : pieces) {
        const auto tokens = tokenizer_.encode(piece, true);
        result.ids.insert(result.ids.end(), tokens.begin(), tokens.end());
    }
    for (size_t i = 0; i < result.ids.size(); ++i) {
        if (result.ids[i] == 151652) {
            if (i + 1 >= result.ids.size() || result.ids[i + 1] != 151654) {
                throw std::runtime_error("Ming speaker slot is not followed by its pad token");
            }
            result.speaker_positions.push_back(i + 1);
        }
    }
    if (result.speaker_positions.size() != speakers) throw std::runtime_error("Ming prompt has an unexpected number of speaker slots");
    if (latent_steps && !reference_text.empty()) {
        result.latent_position = static_cast<int64_t>(result.ids.size());
        result.ids.insert(result.ids.end(), latent_steps, 151705);
    }
    return result;
}

}  // namespace engine::models::ming3_tts
