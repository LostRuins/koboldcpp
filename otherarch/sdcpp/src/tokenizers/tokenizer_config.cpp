#include "tokenizer_config.h"

#include <stdexcept>

#include "core/util.h"
#include "hf_tokenizer.h"

TokenizerConfig::TokenizerConfig(const char* config) {
    if (!config || !*config) {
        return;
    }
    const std::string value = config;
    const bool assignments  = value.find('=') != std::string::npos;
    size_t begin            = 0;
    for (;;) {
        const auto end   = assignments ? value.find(',', begin) : std::string::npos;
        const auto entry = value.substr(begin, end == std::string::npos ? end : end - begin);
        const auto equal = entry.find('=');
        if (assignments && equal == std::string::npos) {
            throw std::runtime_error("invalid tokenizer entry '" + entry + "'; expected main=FILE,clip-l=FILE,clip-g=FILE");
        }
        const auto key  = assignments ? entry.substr(0, equal) : "main";
        const auto path = assignments ? entry.substr(equal + 1) : entry;
        Slot slot       = MAIN;
        if (key == "clip-l") {
            slot = CLIP_L;
        } else if (key == "clip-g") {
            slot = CLIP_G;
        } else if (key != "main") {
            throw std::runtime_error("unknown tokenizer slot '" + key + "'; expected main, clip-l or clip-g");
        }
        if (path.empty()) {
            throw std::runtime_error("tokenizer slot '" + key + "' requires a nonempty path");
        }
        if (!paths_[slot].empty()) {
            throw std::runtime_error("tokenizer slot '" + key + "' is specified more than once");
        }
        paths_[slot] = path;
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
}

bool TokenizerConfig::has(Slot slot) const {
    return !paths_[slot].empty();
}

std::shared_ptr<Tokenizer> TokenizerConfig::create(Slot slot, int64_t embedding_rows, int padding_id, bool pad_left, bool clip) const {
    if (!has(slot)) {
        return nullptr;
    }
    try {
        auto tokenizer = std::make_shared<HFTokenizer>(paths_[slot]);
        tokenizer->set_padding(padding_id, pad_left);
        tokenizer->validate_vocab_size(embedding_rows);
        if (clip && (tokenizer->BOS_TOKEN_ID < 0 || tokenizer->EOS_TOKEN_ID < 0)) {
            throw std::runtime_error("CLIP requires a single BOS + A + EOS template");
        }
        used_[slot]         = true;
        const char* names[] = {"main", "clip-l", "clip-g"};
        LOG_INFO("using external tokenizer (%s): %s", names[slot], paths_[slot].c_str());
        return tokenizer;
    } catch (const std::exception& error) {
        throw std::runtime_error("failed to load tokenizer '" + paths_[slot] + "': " + error.what());
    }
}

void TokenizerConfig::validate_usage() const {
    const char* names[] = {"main", "clip-l", "clip-g"};
    for (size_t i = 0; i < paths_.size(); ++i) {
        if (!paths_[i].empty() && !used_[i]) {
            throw std::runtime_error(std::string("tokenizer slot '") + names[i] + "' does not target an active, supported text encoder; SD3 uses the clip-l and clip-g slots");
        }
    }
}
