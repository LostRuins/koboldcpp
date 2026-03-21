// ace-qwen3.cpp : ACE-Step 5Hz LM inference (GGML)
// Qwen3 causal LM: CoT reasoning + audio code generation
// ace-qwen3: Qwen3 causal LM for ACE-Step music generation (GGML backend)
#include "./qwen3-lm.h"
#include "./bpe.h"
#include "./request.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <map>
#include <unordered_map>
#include <iomanip>
#include <sstream>

// Timer
struct Timer {
    std::chrono::steady_clock::time_point t;
    Timer() : t(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t).count();
    }
    void reset() { t = std::chrono::steady_clock::now(); }
};

// Special token IDs (Qwen3 extended vocab)
#define TOKEN_IM_START      151644
#define TOKEN_IM_END        151645
#define TOKEN_THINK         151667
#define TOKEN_THINK_END     151668
#define AUDIO_CODE_BASE     151669
#define AUDIO_CODE_COUNT    65535

//
// Sampling
//

struct TokenProb {
    int id;
    float prob;
};

// Sampling: temperature -> top_k -> top_p -> softmax -> multinomial
// Matches nano-vLLM Sampler: div_(temperature) -> apply_top_k_top_p -> softmax -> sample
static int sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        // greedy
        return (int)(std::max_element(logits, logits + V) - logits);
    }

    // 1. temperature (matches nano-vLLM: logits.float().div_(temperatures))
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++)
        logits[i] *= inv_temp;

    // 2. top_k: keep top K values, set rest to -inf
    //    nano-vLLM: topk(k) returns k-th largest as threshold, mask < threshold
    if (top_k > 0 && top_k < V) {
        std::vector<float> tmp(logits, logits + V);
        std::nth_element(tmp.begin(), tmp.begin() + (top_k - 1), tmp.end(), std::greater<float>());
        float threshold = tmp[top_k - 1];
        for (int i = 0; i < V; i++)
            if (logits[i] < threshold) logits[i] = -INFINITY;
    }

    // 3. top_p: nucleus filter on temp-scaled logits (matches nano-vLLM: softmax on scaled logits)
    //    nano-vLLM sorts ascending, cumsum, masks cumsum <= (1-p), keeps last element.
    //    Equivalent descending: mask tokens where cumsum_before >= top_p (shift-right).
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<TokenProb> sorted(V);
        for (int i = 0; i < V; i++) sorted[i] = {i, logits[i]};
        std::sort(sorted.begin(), sorted.end(),
                  [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });

        // softmax of temp-scaled logits for cumsum
        float max_val = sorted[0].prob;
        float sum = 0.0f;
        std::vector<float> probs(V);
        for (int i = 0; i < V; i++) {
            probs[i] = expf(sorted[i].prob - max_val);
            sum += probs[i];
        }
        float inv = 1.0f / sum;

        // cumulative sum, test before accumulating (shift-right trick)
        float cum = 0.0f;
        for (int i = 0; i < V; i++) {
            if (i > 0 && cum >= top_p)  // i>0: always keep at least first token
                logits[sorted[i].id] = -INFINITY;
            cum += probs[i] * inv;
        }
    }

    // 4. softmax -> multinomial (temperature already applied)
    float max_val = -INFINITY;
    for (int i = 0; i < V; i++)
        if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) return i;
    }
    return 0;
}

//
// BPE decode (token IDs -> text)
//

static std::string bpe_decode(const BPETokenizer & bpe, const std::vector<int> & ids) {
    static std::unordered_map<int, uint8_t> byte_dec;
    static bool init = false;
    if (!init) {
        for (int b = 0; b < 256; b++) {
            int adv;
            int cp = utf8_codepoint(bpe.byte2str[b].c_str(), &adv);
            byte_dec[cp] = (uint8_t)b;
        }
        init = true;
    }

    std::string result;
    for (int id : ids) {
        if (id == TOKEN_THINK)     { result += "<think>";  continue; }
        if (id == TOKEN_THINK_END) { result += "</think>"; continue; }
        if (id == TOKEN_IM_START || id == TOKEN_IM_END) continue;
        if (id >= AUDIO_CODE_BASE) continue;
        if (id < 0 || id >= (int)bpe.id_to_str.size()) continue;
        const std::string & s = bpe.id_to_str[id];
        if (s.empty()) continue;
        const char * p = s.c_str();
        while (*p) {
            int adv;
            int cp = utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) result += (char)it->second;
            p += adv;
        }
    }
    return result;
}

//
// ACE-Step prompt
//

struct AcePrompt {
    std::string caption;
    std::string lyrics;
    float duration;
    int bpm;
    std::string keyscale;
    std::string timesignature;
    std::string vocal_language;
};

static std::mt19937 acestep_lm_rng;
static bool acestep_lm_dbg = false;
static std::vector<int32_t> forced_tokens;
static std::vector<int> caption_tokens = std::vector<int>(); //will be filled with caption tokens
const int rep_pen_range = 48;

//
// CoT parsing (extract metadata + lyrics from LLM Phase1 output)
//

static bool parse_cot_and_lyrics(const std::string & text, AcePrompt * out) {
    // Extract CoT content between <think>...</think>
    size_t ts = text.find("<think>");
    size_t te = text.find("</think>");

    std::string cot;
    std::string lyrics_after;

    if (ts != std::string::npos && te != std::string::npos) {
        cot = text.substr(ts + 7, te - ts - 7);
        lyrics_after = text.substr(te + 8);
    } else if (te != std::string::npos) {
        cot = text.substr(0, te);
        lyrics_after = text.substr(te + 8);
    } else {
        cot = text;
    }

    // Parse YAML-like fields from CoT
    auto get_field = [&](const std::string & key) -> std::string {
        std::string needle = key + ":";
        size_t p = cot.find(needle);
        if (p == std::string::npos) return "";
        p += needle.size();
        while (p < cot.size() && (cot[p] == ' ' || cot[p] == '\'')) p++;
        size_t end = cot.find('\n', p);
        if (end == std::string::npos) end = cot.size();
        std::string val = cot.substr(p, end - p);
        // Strip trailing whitespace and quotes
        while (!val.empty() && (val.back() == ' ' || val.back() == '\'' || val.back() == '\r'))
            val.pop_back();
        return val;
    };

    std::string bpm_s = get_field("bpm");
    if (!bpm_s.empty()) out->bpm = atoi(bpm_s.c_str());

    std::string dur_s = get_field("duration");
    if (!dur_s.empty()) out->duration = (float)atof(dur_s.c_str());

    std::string ks = get_field("keyscale");
    if (!ks.empty()) out->keyscale = ks;

    std::string ts_s = get_field("timesignature");
    if (!ts_s.empty()) out->timesignature = ts_s;

    std::string lang = get_field("language");
    if (!lang.empty()) out->vocal_language = lang;

    std::string cap = get_field("caption");
    if (!cap.empty()) {
        // Caption may span multiple lines (yaml word-wrap)
        size_t cp = cot.find("caption:");
        if (cp != std::string::npos) {
            cp += 8;
            size_t end = cot.find("\nduration:", cp);
            if (end == std::string::npos) end = cot.find("\nkeyscale:", cp);
            if (end == std::string::npos) end = cot.size();
            std::string full_cap = cot.substr(cp, end - cp);
            // Trim and collapse whitespace
            std::string cleaned;
            bool in_space = true;
            for (char ch : full_cap) {
                if (ch == '\n' || ch == '\r') ch = ' ';
                if (ch == ' ') {
                    if (!in_space) cleaned += ' ';
                    in_space = true;
                } else {
                    cleaned += ch;
                    in_space = false;
                }
            }
            while (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
            while (!cleaned.empty() && cleaned.front() == ' ') cleaned.erase(cleaned.begin());
            if (!cleaned.empty()) out->caption = cleaned;
        }
    }

    // Lyrics after </think>
    if (!lyrics_after.empty()) {
        // Trim leading whitespace
        size_t s = lyrics_after.find_first_not_of(" \t\n\r");
        if (s != std::string::npos)
            lyrics_after = lyrics_after.substr(s);
        // Trim trailing whitespace
        while (!lyrics_after.empty() &&
               (lyrics_after.back() == ' ' || lyrics_after.back() == '\n' || lyrics_after.back() == '\r'))
            lyrics_after.pop_back();
        if (!lyrics_after.empty())
            out->lyrics = lyrics_after;
    }

    return (out->bpm > 0 || out->duration > 0);
}

//
// Prompt building (Qwen3 chat template)
//

static std::vector<int> build_lm_prompt(BPETokenizer & bpe, const AcePrompt & prompt) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

static std::vector<int> build_lm_prompt_uncond(BPETokenizer & bpe, const AcePrompt & prompt,
                                                const char * negative_prompt) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    if (has_neg)
        append("user\n# Caption\n" + std::string(negative_prompt) + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    else
        append("user\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

// Build CoT YAML content (matching Python yaml.dump sort_keys=True)
static std::string build_cot_yaml(const AcePrompt & prompt) {
    auto yaml_wrap = [](const std::string & key, const std::string & val) -> std::string {
        std::string result = key + ":";
        int col = (int)(key.size() + 1);
        size_t i = 0;
        while (i < val.size()) {
            size_t end = val.find(' ', i);
            if (end == std::string::npos) end = val.size();
            std::string word = val.substr(i, end - i);
            if (col > 80) {
                result += "\n  ";
                col = 2;
            } else {
                result += " ";
                col += 1;
            }
            result += word;
            col += (int)word.size();
            i = (end < val.size()) ? end + 1 : val.size();
        }
        result += "\n";
        return result;
    };

    std::string yaml;
    if (prompt.bpm > 0)
        yaml += "bpm: " + std::to_string(prompt.bpm) + "\n";
    // if (!prompt.caption.empty())
    //     yaml += yaml_wrap("caption", prompt.caption);
    if (prompt.duration > 0)
        yaml += "duration: " + std::to_string((int)prompt.duration) + "\n";
    if (!prompt.keyscale.empty())
        yaml += "keyscale: " + prompt.keyscale + "\n";
    if (!prompt.vocal_language.empty())
        yaml += "language: " + prompt.vocal_language + "\n";
    if (!prompt.timesignature.empty())
        yaml += "timesignature: " + prompt.timesignature + "\n";
    return yaml;
}

// Prompt with injected CoT (Phase 2: all metas known)
static std::vector<int> build_lm_prompt_with_cot(BPETokenizer & bpe, const AcePrompt & prompt,
                                                   const std::string & cot_yaml) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n" + cot_yaml);
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    return ids;
}

// Unconditional prompt with empty CoT for CFG (Phase 2)
static std::vector<int> build_lm_prompt_uncond_with_cot(BPETokenizer & bpe, const AcePrompt & prompt,
                                                          const char * negative_prompt) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    std::string cap = has_neg ? std::string(negative_prompt) : prompt.caption;
    append("user\n# Caption\n" + cap + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n\n");
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    return ids;
}

// Build Qwen3 chat prompt: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
static std::vector<int> build_custom_prompt(BPETokenizer & bpe, const char * sys, const char * user) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n" + std::string(sys) + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n" + std::string(user) + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

//
// Prefix tree for FSM constrained decoding
//

struct PrefixTree {
    // Maps prefix (token sequence) to set of valid next tokens
    std::map<std::vector<int>, std::vector<int>> nodes;

    void add(const std::vector<int> & seq) {
        for (size_t i = 0; i < seq.size(); i++) {
            std::vector<int> prefix(seq.begin(), seq.begin() + i);
            int next = seq[i];
            auto & vec = nodes[prefix];
            if (std::find(vec.begin(), vec.end(), next) == vec.end())
                vec.push_back(next);
        }
    }

    const std::vector<int> * get(const std::vector<int> & prefix) const {
        auto it = nodes.find(prefix);
        return it != nodes.end() ? &it->second : nullptr;
    }
};

//
// Metadata FSM (constrained decoding for CoT fields)
//

struct MetadataFSM {
    enum State {
        BPM_NAME, BPM_VALUE,
        CAPTION_NAME, CAPTION_VALUE,
        DURATION_NAME, DURATION_VALUE,
        KEYSCALE_NAME, KEYSCALE_VALUE,
        LANGUAGE_NAME, LANGUAGE_VALUE,
        TIMESIG_NAME, TIMESIG_VALUE,
        THINK_END,
        CODES,
        DISABLED
    };

    State state = DISABLED;
    int caption_newline_counter = 0;
    int name_pos = 0;
    std::vector<int> value_acc;
    bool enabled = false;
    bool rewrite_caption = false;

    std::vector<int> bpm_name, caption_name, duration_name;
    std::vector<int> keyscale_name, language_name, timesig_name;
    PrefixTree bpm_tree, duration_tree, keyscale_tree, language_tree, timesig_tree;
    int newline_tok = -1;
    int think_end_tok = TOKEN_THINK_END;
    int vocab_size = 0;

    static std::vector<int> tokenize_strip(BPETokenizer & bpe,
                                            const std::string & full,
                                            const std::string & prefix) {
        std::vector<int> full_tok = bpe_encode(&bpe, full, false);
        std::vector<int> pre_tok  = bpe_encode(&bpe, prefix, false);
        if (full_tok.size() >= pre_tok.size() &&
            std::equal(pre_tok.begin(), pre_tok.end(), full_tok.begin()))
            return std::vector<int>(full_tok.begin() + pre_tok.size(), full_tok.end());
        return full_tok;
    }

    void build_value_tree(BPETokenizer & bpe, PrefixTree & tree,
                          const std::string & field_prefix,
                          const std::vector<std::string> & values) {
        for (auto & val : values) {
            std::string full = field_prefix + val + "\n";
            std::vector<int> vtok = tokenize_strip(bpe, full, field_prefix);
            tree.add(vtok);
        }
    }

    void init(BPETokenizer & bpe, int vsize, bool rew_caption) {
        vocab_size = vsize;
        auto nl = bpe_encode(&bpe, "\n", false);
        newline_tok = nl.empty() ? -1 : nl[0];

        bpm_name      = bpe_encode(&bpe, "bpm:", false);
        caption_name  = bpe_encode(&bpe, "caption:", false);
        duration_name = bpe_encode(&bpe, "duration:", false);
        keyscale_name = bpe_encode(&bpe, "keyscale:", false);
        language_name = bpe_encode(&bpe, "language:", false);
        timesig_name  = bpe_encode(&bpe, "timesignature:", false);

        // BPM 30-195
        {
            std::vector<std::string> vals;
            for (int v = 30; v <= 195; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, bpm_tree, "bpm:", vals);
        }
        // Duration 40-450
        {
            std::vector<std::string> vals;
            for (int v = 40; v <= 450; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, duration_tree, "duration:", vals);
        }
        // Keyscale
        {
            const char * notes[] = {"A","B","C","D","E","F","G"};
            const char * accs[]  = {"","b","#"};
            const char * modes[] = {
                "major","minor","dorian","phrygian","lydian","mixolydian",
                "aeolian","locrian","chromatic","blues","pentatonic",
                "harmonic minor","melodic minor"
            };
            std::vector<std::string> vals;
            for (auto n : notes)
                for (auto a : accs)
                    for (auto m : modes)
                        vals.push_back(std::string(n) + a + " " + m);
            build_value_tree(bpe, keyscale_tree, "keyscale:", vals);
        }
        // Language
        {
            std::vector<std::string> vals = {
                "en","zh","ja","ko","es","fr","de","uk","ru","pt",
                "it","ar","tr","pl","sv","nl","unknown"
            };
            build_value_tree(bpe, language_tree, "language:", vals);
        }
        // Time signature
        {
            std::vector<std::string> vals = {"2","3","4","6"};
            build_value_tree(bpe, timesig_tree, "timesignature:", vals);
        }

        fprintf(stderr, "[FSM] Prefix trees: bpm=%zu, dur=%zu, key=%zu, lang=%zu, tsig=%zu nodes\n",
                bpm_tree.nodes.size(), duration_tree.nodes.size(),
                keyscale_tree.nodes.size(), language_tree.nodes.size(),
                timesig_tree.nodes.size());
        enabled = true;
        state = BPM_NAME;
        name_pos = 0;
        caption_newline_counter = 0;
        rewrite_caption = rew_caption;
        value_acc.clear();
    }

    void reset() {
        state = BPM_NAME;
        name_pos = 0;
        caption_newline_counter = 0;
        value_acc.clear();
    }

    // Force FSM to only allow a specific language value
    void force_language(BPETokenizer & bpe, const std::string & lang) {
        language_tree = PrefixTree();
        build_value_tree(bpe, language_tree, "language:", {lang});
    }

    const std::vector<int> * current_name_tokens() const {
        switch (state) {
            case BPM_NAME:      return &bpm_name;
            case CAPTION_NAME:  return &caption_name;
            case DURATION_NAME: return &duration_name;
            case KEYSCALE_NAME: return &keyscale_name;
            case LANGUAGE_NAME: return &language_name;
            case TIMESIG_NAME:  return &timesig_name;
            default: return nullptr;
        }
    }

    const PrefixTree * current_value_tree() const {
        switch (state) {
            case BPM_VALUE:      return &bpm_tree;
            case DURATION_VALUE: return &duration_tree;
            case KEYSCALE_VALUE: return &keyscale_tree;
            case LANGUAGE_VALUE: return &language_tree;
            case TIMESIG_VALUE:  return &timesig_tree;
            default: return nullptr;
        }
    }

    State next_name_state() const {
        switch (state) {
            case BPM_NAME:      case BPM_VALUE:      return CAPTION_NAME;
            case CAPTION_NAME:  case CAPTION_VALUE:   return DURATION_NAME;
            case DURATION_NAME: case DURATION_VALUE:  return KEYSCALE_NAME;
            case KEYSCALE_NAME: case KEYSCALE_VALUE:  return LANGUAGE_NAME;
            case LANGUAGE_NAME: case LANGUAGE_VALUE:   return TIMESIG_NAME;
            case TIMESIG_NAME:  case TIMESIG_VALUE:   return THINK_END;
            default: return CODES;
        }
    }

    void apply_mask(float * logits) {
        if (!enabled || state == CODES || state == DISABLED) return;

        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int)name->size()) {
            int forced = (*name)[name_pos];
            for (int v = 0; v < vocab_size; v++)
                if (v != forced) logits[v] = -1e9f;
            return;
        }

        const PrefixTree * tree = current_value_tree();
        if (tree) {
            const std::vector<int> * allowed = tree->get(value_acc);
            if (allowed && !allowed->empty()) {
                std::vector<float> saved(allowed->size());
                for (size_t i = 0; i < allowed->size(); i++)
                    saved[i] = logits[(*allowed)[i]];
                for (int v = 0; v < vocab_size; v++) logits[v] = -1e9f;
                for (size_t i = 0; i < allowed->size(); i++)
                    logits[(*allowed)[i]] = saved[i];
            } else {
                if (newline_tok >= 0) {
                    for (int v = 0; v < vocab_size; v++)
                        if (v != newline_tok) logits[v] = -1e9f;
                }
            }
            return;
        }

        if (state == CAPTION_VALUE) {
            for (int v = AUDIO_CODE_BASE; v < AUDIO_CODE_BASE + AUDIO_CODE_COUNT; v++)
                if (v < vocab_size) logits[v] = -1e9f;
            return;
        }

        if (state == THINK_END) {
            for (int v = 0; v < vocab_size; v++)
                if (v != think_end_tok) logits[v] = -1e9f;
            return;
        }
    }

    void update(int token) {
        if (!enabled || state == CODES || state == DISABLED) return;

        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int)name->size()) {
            name_pos++;
            if (name_pos >= (int)name->size()) {
                switch (state) {
                    case BPM_NAME:      state = BPM_VALUE; break;
                    case CAPTION_NAME:
                        state = CAPTION_VALUE;
                        if(caption_tokens.size()>0)
                        {
                            forced_tokens.clear();
                            forced_tokens = caption_tokens;
                        }
                    break;
                    case DURATION_NAME: state = DURATION_VALUE; break;
                    case KEYSCALE_NAME: state = KEYSCALE_VALUE; break;
                    case LANGUAGE_NAME: state = LANGUAGE_VALUE; break;
                    case TIMESIG_NAME:  state = TIMESIG_VALUE; break;
                    default: break;
                }
                value_acc.clear();
            }
            return;
        }

        if (current_value_tree()) {
            if (token == newline_tok) {
                state = next_name_state();
                name_pos = 0;
                value_acc.clear();
            } else {
                value_acc.push_back(token);
            }
            return;
        }

        if (state == CAPTION_VALUE) {
            if(token==newline_tok && !rewrite_caption)
            {
                //not changing caption, so end it now
                caption_newline_counter = 0;
                state = DURATION_NAME;
                name_pos = 0;
                value_acc.clear();
            }
            else if (token == newline_tok) {
                // caption_newline_counter += 1;
                // if(caption_newline_counter>=3) //ingest 3 newlines for the caption, making it longer
                // {
                    caption_newline_counter = 0;
                    state = DURATION_NAME;
                    name_pos = 0;
                    value_acc.clear();
               // }
            }
            else if(token == 17021) // fallback for duration token, immediately go next
            {
                caption_newline_counter = 0;
                state = DURATION_NAME;
                name_pos = 0;
                value_acc.clear();
            }
            return;
        }

        if (state == THINK_END) {
            state = CODES;
            return;
        }
    }
};

//
// Generation
//


// Text-only generation (Phase 1: no CFG, stops at EOS)
static std::string codes_to_string(const std::vector<int> & codes);

// Convert audio codes vector to comma-separated string (Python-compatible)
static std::string codes_to_string(const std::vector<int> & codes) {
    std::string s;
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) s += ',';
        s += std::to_string(codes[i]);
    }
    return s;
}

// Phase 2: run audio code generation with all metas known
// Returns comma-separated codes string (empty on failure)

// Parse N Phase 1 outputs into N AcePrompts, merging into base.
// merge_lyrics: true for simple mode (Phase 1 generates lyrics),
//               false for partial mode (user provided lyrics).
static void parse_phase1_into_aces(
        const std::vector<std::string> & texts, const AcePrompt & base,
        std::vector<AcePrompt> & aces, long long base_seed,
        const char * label, bool merge_lyrics) {
    int N = (int)texts.size();
    aces.resize(N);
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[%s Batch%d] seed=%lld:\n%s\n", label, i, base_seed + i, texts[i].c_str());
        AcePrompt parsed = {};
        if (!parse_cot_and_lyrics(texts[i], &parsed))
            fprintf(stderr, "WARNING: batch %d CoT parse incomplete\n", i);
        aces[i] = base;
        if (parsed.bpm > 0) aces[i].bpm = parsed.bpm;
        if (parsed.duration > 0) aces[i].duration = parsed.duration;
        if (!parsed.keyscale.empty()) aces[i].keyscale = parsed.keyscale;
        if (!parsed.timesignature.empty()) aces[i].timesignature = parsed.timesignature;
        if (!parsed.vocal_language.empty()) aces[i].vocal_language = parsed.vocal_language;
        if (!parsed.caption.empty()) aces[i].caption = parsed.caption;
        if (merge_lyrics && !parsed.lyrics.empty()) aces[i].lyrics = parsed.lyrics;
        if (aces[i].duration <= 0) aces[i].duration = 120.0f;
        if (aces[i].duration > 600) aces[i].duration = 600.0f;
    }
}

//hack for kcpp: forcing the correct tokens after end of thinking
const std::vector<int> think_chain = {271,2,15953,2216,198}; // "\n# Lyric\n"

// Batched Phase 1: N text generations with shared prompt, different seeds.
// No CFG. Each element gets its own FSM state and RNG.
// Returns N generated text strings.
static std::vector<std::string> generate_phase1_batch(
        Qwen3LM * m, BPETokenizer * bpe,
        const std::vector<int> & prompt_tokens,
        int max_new_tokens, float temperature, float top_p, int top_k, float rep_pen,
        long long base_seed, int N,
        MetadataFSM * fsm_template,
        bool lyrics_mode,
        float cfg_scale = 1.0f,
        const std::vector<int> * uncond_tokens = nullptr,
        bool stop_at_reasoning = false) {

    int V = m->cfg.vocab_size;
    bool use_cfg = cfg_scale > 1.0f && uncond_tokens && !uncond_tokens->empty();
    forced_tokens.clear();

    std::vector<int32_t> quicklastntoks;

    // KV sets: cond [0..N-1], uncond [N..2N-1] if CFG
    for (int i = 0; i < N; i++) qw3lm_reset_kv(m, i);
    if (use_cfg)
        for (int i = 0; i < N; i++) qw3lm_reset_kv(m, N + i);

    // Prefill cond once, set 0, copy to 1..N-1
    Timer t_prefill;
    std::vector<float> prefill_logits(V);
    qw3lm_forward(m, prompt_tokens.data(), (int)prompt_tokens.size(), 0, prefill_logits.data());
    for (int i = 1; i < N; i++)
        qw3lm_copy_kv(m, 0, i);

    // Prefill uncond once, set N, copy to N+1..2N-1
    std::vector<float> prefill_logits_uncond(V);
    if (use_cfg) {
        qw3lm_forward(m, uncond_tokens->data(), (int)uncond_tokens->size(), N, prefill_logits_uncond.data());
        for (int i = 1; i < N; i++)
            qw3lm_copy_kv(m, N, N + i);
    }

    fprintf(stderr, "[Phase1] Prefill %.0fms, %zu tokens, N=%d, CFG=%.2f\n",
            t_prefill.ms(), prompt_tokens.size(), N, cfg_scale);

    if(acestep_lm_dbg)
    {
        std::string tks = bpe_decode(*bpe,prompt_tokens);
        printf("\nN:%d Prompt: %s",prompt_tokens.size(),tks.c_str());
    }

    // Per-element state
    struct P1Seq {
        std::mt19937 rng;
        MetadataFSM fsm;
        std::vector<int> gen_tokens;
        int last_token;
        bool codes_phase;
        bool done;
    };
    std::vector<P1Seq> seqs(N);

    // Sample first token from shared prefill logits
    for (int i = 0; i < N; i++) {
        seqs[i].rng.seed((uint32_t)(base_seed + i));
        if (fsm_template) seqs[i].fsm = *fsm_template;
        seqs[i].codes_phase = false;
        seqs[i].done = false;

        std::vector<float> lg(prefill_logits);
        if (use_cfg) {
            for (int v = 0; v < V; v++)
                lg[v] = prefill_logits_uncond[v] + cfg_scale * (lg[v] - prefill_logits_uncond[v]);
        }
        if (fsm_template && fsm_template->enabled)
            seqs[i].fsm.apply_mask(lg.data());

        int tok = kcpp_quick_sample(lg.data(),V,quicklastntoks,rep_pen,top_p,top_k,temperature,acestep_lm_rng);
        quicklastntoks.push_back(tok);
            if (quicklastntoks.size()>rep_pen_range) {
            quicklastntoks.erase(quicklastntoks.begin());
        }

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else {
            if (fsm_template && fsm_template->enabled)
                seqs[i].fsm.update(tok);
            if (tok == TOKEN_THINK_END) {
                seqs[i].codes_phase = true;
                if (stop_at_reasoning) seqs[i].done = true;
            }
            seqs[i].gen_tokens.push_back(tok);
        }
        seqs[i].last_token = tok;
    }

    // KV set arrays + merged CFG arrays
    std::vector<int> cond_sets(N), uncond_sets(N);
    for (int i = 0; i < N; i++) {
        cond_sets[i] = i;
        uncond_sets[i] = N + i;
    }

    // Batched decode
    Timer t_decode;
    std::vector<float> logits_cond(V * N);
    std::vector<float> logits_uncond(V * N);
    std::vector<int> tokens(N);

    // CFG: single forward with 2*N (cond + uncond)
    int N2 = use_cfg ? 2 * N : N;
    std::vector<int> tokens_2n(N2), sets_2n(N2);
    std::vector<float> logits_2n((size_t)V * N2);
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            sets_2n[i] = cond_sets[i];
            sets_2n[N + i] = uncond_sets[i];
        }
    }

    int n_active = N;
    for (int i = 0; i < N; i++)
        if (seqs[i].done) n_active--;

    for (int step = 0; step < max_new_tokens && n_active > 0; step++) {
        for (int i = 0; i < N; i++)
            tokens[i] = seqs[i].last_token;

        if (use_cfg) {
            // Single batched forward: cond[0..N-1] + uncond[N..2N-1]
            for (int i = 0; i < N; i++) {
                tokens_2n[i] = tokens[i];
                tokens_2n[N + i] = tokens[i];
            }
            qw3lm_forward_batch(m, tokens_2n.data(), sets_2n.data(), N2, logits_2n.data());
            memcpy(logits_cond.data(),   logits_2n.data(),                    (size_t)V * N * sizeof(float));
            memcpy(logits_uncond.data(), logits_2n.data() + (size_t)V * N,    (size_t)V * N * sizeof(float));
        } else {
            qw3lm_forward_batch(m, tokens.data(), cond_sets.data(), N, logits_cond.data());
        }

        for (int i = 0; i < N; i++) {
            if (seqs[i].done) continue;

            float * lc = logits_cond.data() + (size_t)i * V;

            // CFG combine
            if (use_cfg) {
                float * lu = logits_uncond.data() + (size_t)i * V;
                for (int v = 0; v < V; v++)
                    lc[v] = lu[v] + cfg_scale * (lc[v] - lu[v]);
            }

            // FSM mask (before </think>)
            if (fsm_template && seqs[i].fsm.enabled && !seqs[i].codes_phase)
                seqs[i].fsm.apply_mask(lc);

            // After </think>: audio code constraint unless lyrics_mode
            if (seqs[i].codes_phase && !lyrics_mode) {
                for (int v = 0; v < AUDIO_CODE_BASE; v++)
                    if (v != TOKEN_IM_END) lc[v] = -1e9f;
            }

            // kcpp: prevent outputting audio codes during lyrics
            if(lyrics_mode)
            {
                for (int v = AUDIO_CODE_BASE; v < AUDIO_CODE_COUNT+AUDIO_CODE_BASE; v++)
                    if (v != TOKEN_IM_END) lc[v] = -1e9f;
            }

            int tok = kcpp_quick_sample(lc,V,quicklastntoks,rep_pen,top_p,top_k,temperature,acestep_lm_rng);
            quicklastntoks.push_back(tok);
             if (quicklastntoks.size()>rep_pen_range) {
                quicklastntoks.erase(quicklastntoks.begin());
            }

            //kcpp: force lyrics tokens right after think
            if(forced_tokens.size()>0)
            {
                tok = forced_tokens[0];
                forced_tokens.erase(forced_tokens.begin());
            }
            if (tok == TOKEN_THINK_END)
            {
                forced_tokens.clear();
                forced_tokens = think_chain;
            }

            if(acestep_lm_dbg)
            {
                std::string tks = bpe_decode(*bpe,std::vector<int>({tok}));
                printf("\nDebug temp: %f, top_p:%f, tok:%d = %s (%d)",temperature,top_p,tok,tks.c_str(),forced_tokens.size());
            }

            if (tok == TOKEN_IM_END) {
                seqs[i].done = true;
                n_active--;
            } else {
                if (seqs[i].fsm.enabled && !seqs[i].codes_phase)
                    seqs[i].fsm.update(tok);
                if (tok == TOKEN_THINK_END && !seqs[i].codes_phase) {
                    seqs[i].codes_phase = true;
                    if (stop_at_reasoning) {
                        seqs[i].gen_tokens.push_back(tok);
                        seqs[i].done = true;
                        n_active--;
                        continue;
                    }
                }

                // Safety check: if we've reached the token limit, force TOKEN_IM_END
                // to prevent KV cache exhaustion (FATAL: kv_len > max_seq)
                if ((int)seqs[i].gen_tokens.size() >= max_new_tokens - 1 && !seqs[i].done) {
                    forced_tokens.clear();
                    forced_tokens.push_back(TOKEN_IM_END);
                }

                seqs[i].gen_tokens.push_back(tok);
            }
            seqs[i].last_token = tok;
        }

        if ((step + 1) % 100 == 0) {
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[Phase1] step %d, %d active, %.1f tok/s\n",
                    step + 1, n_active, (double)(step + 1) * N / elapsed);
        }
    }

    fprintf(stderr, "[Phase1] Decode %.0fms\n", t_decode.ms());

    // Decode tokens to text
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = bpe_decode(*bpe, seqs[i].gen_tokens);
        fprintf(stderr, "[Phase1 Batch%d] seed=%lld, %zu tokens\n",
                i, base_seed + i, seqs[i].gen_tokens.size());
    }
    return results;
}


// Batched Phase 2: N sequences with potentially different prompts.
// aces.size() == N: each element gets its own lyrics/metadata.
// aces.size() == 1: single prompt replicated for all N (prefill once, copy KV).
// Returns N code strings. Seeds = base_seed + 0, 1, ..., N-1.
static std::vector<std::string> run_phase2_batch(
        Qwen3LM * m, BPETokenizer & bpe, const std::vector<AcePrompt> & aces,
        float temperature, float top_p, int top_k, float rep_pen, long long base_seed, int N,
        float cfg_scale, const char * negative_prompt) {

    int V = m->cfg.vocab_size;
    bool use_cfg = cfg_scale > 1.0f;
    bool shared_prompt = ((int)aces.size() == 1);

    std::vector<int32_t> quicklastntoks;

    // Build per-element prompts
    std::vector<std::vector<int>> prompts(N), unconds(N);
    int max_tokens = 0;
    for (int i = 0; i < N; i++) {
        const AcePrompt & a = shared_prompt ? aces[0] : aces[i];
        std::string cot = build_cot_yaml(a);
        if (i == 0)
            fprintf(stderr, "[Phase2] N=%d, CoT[0]:\n%s", N, cot.c_str());
        prompts[i] = build_lm_prompt_with_cot(bpe, a, cot);
        if (use_cfg)
            unconds[i] = build_lm_prompt_uncond_with_cot(bpe, a, negative_prompt);
        int mt = (int)(a.duration * 5) + 100;
        if (mt > max_tokens) max_tokens = mt;
    }
    fprintf(stderr, "[Phase2] max_tokens: %d, CFG: %.2f, seeds: %lld..%lld\n",
            max_tokens, cfg_scale, base_seed, base_seed + N - 1);

    // Reset all KV sets: cond [0..N-1], uncond [N..2N-1]
    for (int i = 0; i < N; i++) qw3lm_reset_kv(m, i);
    if (use_cfg)
        for (int i = 0; i < N; i++) qw3lm_reset_kv(m, N + i);

    // Prefill: if shared prompt, prefill once + copy KV. Otherwise prefill each.
    Timer t_prefill;
    std::vector<std::vector<float>> prefill_logits_vec(N, std::vector<float>(V));

    if(acestep_lm_dbg)
    {
        std::string tks = bpe_decode(bpe,prompts[0]);
        printf("\nPhase2: UseCFG:%d, Promptsiz:%d, Prompt: %s",use_cfg,prompts[0].size(),tks.c_str());
    }

    if (shared_prompt) {
        qw3lm_forward(m, prompts[0].data(), (int)prompts[0].size(), 0, prefill_logits_vec[0].data());
        for (int i = 1; i < N; i++) {
            qw3lm_copy_kv(m, 0, i);
            prefill_logits_vec[i] = prefill_logits_vec[0];
        }
    } else {
        for (int i = 0; i < N; i++)
            qw3lm_forward(m, prompts[i].data(), (int)prompts[i].size(), i, prefill_logits_vec[i].data());
    }

    // Prefill uncond
    std::vector<std::vector<float>> prefill_logits_uncond_vec(N, std::vector<float>(V));
    if (use_cfg) {
        if (shared_prompt) {
            qw3lm_forward(m, unconds[0].data(), (int)unconds[0].size(), N, prefill_logits_uncond_vec[0].data());
            for (int i = 1; i < N; i++) {
                qw3lm_copy_kv(m, N, N + i);
                prefill_logits_uncond_vec[i] = prefill_logits_uncond_vec[0];
            }
        } else {
            for (int i = 0; i < N; i++)
                qw3lm_forward(m, unconds[i].data(), (int)unconds[i].size(), N + i, prefill_logits_uncond_vec[i].data());
        }
    }

    double prefill_ms = t_prefill.ms();
    fprintf(stderr, "[Phase2] Prefill %.0fms (%s)\n",
            prefill_ms, shared_prompt ? "shared, 1 cond + 1 uncond" : "individual, N cond + N uncond");

    // Per-sequence state
    struct BatchSeq {
        std::mt19937 rng;
        std::vector<int> audio_codes;
        int last_token;
        bool done;
    };
    std::vector<BatchSeq> seqs(N);

    // Sample first token from per-element prefill logits (N different seeds)
    for (int i = 0; i < N; i++) {
        seqs[i].rng.seed((uint32_t)(base_seed + i));
        seqs[i].done = false;

        std::vector<float> lg(prefill_logits_vec[i]);  // copy
        if (use_cfg) {
            float * lu = prefill_logits_uncond_vec[i].data();
            for (int v = 0; v < V; v++)
                lg[v] = lu[v] + cfg_scale * (lg[v] - lu[v]);
        }
        // Only audio codes + EOS (codes_phase = true from start)
        for (int v = 0; v < AUDIO_CODE_BASE; v++)
            if (v != TOKEN_IM_END) lg[v] = -1e9f;

        int tok = kcpp_quick_sample(lg.data(),V,quicklastntoks,rep_pen,top_p,top_k,temperature,acestep_lm_rng);
        quicklastntoks.push_back(tok);
            if (quicklastntoks.size()>rep_pen_range) {
            quicklastntoks.erase(quicklastntoks.begin());
        }

        seqs[i].last_token = tok;

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else if (tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            seqs[i].audio_codes.push_back(tok - AUDIO_CODE_BASE);
        }
    }

    // KV set arrays for batched forward
    std::vector<int> cond_sets(N), uncond_sets(N);
    for (int i = 0; i < N; i++) {
        cond_sets[i] = i;
        uncond_sets[i] = N + i;
    }

    // Batched decode loop
    Timer t_decode;
    std::vector<float> logits_cond(V * N);
    std::vector<float> logits_uncond(V * N);
    std::vector<int> tokens(N);

    // CFG: single forward with 2*N (cond + uncond)
    int N2 = use_cfg ? 2 * N : N;
    std::vector<int> tokens_2n(N2), sets_2n(N2);
    std::vector<float> logits_2n((size_t)V * N2);
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            sets_2n[i] = cond_sets[i];
            sets_2n[N + i] = uncond_sets[i];
        }
    }

    int n_active = N;
    for (int i = 0; i < N; i++)
        if (seqs[i].done) n_active--;

    for (int step = 0; step < max_tokens && n_active > 0; step++) {
        // Collect tokens (done sequences feed their last token, result ignored)
        for (int i = 0; i < N; i++)
            tokens[i] = seqs[i].last_token;

        if (use_cfg) {
            // Single batched forward: cond[0..N-1] + uncond[N..2N-1]
            for (int i = 0; i < N; i++) {
                tokens_2n[i] = tokens[i];
                tokens_2n[N + i] = tokens[i];
            }
            qw3lm_forward_batch(m, tokens_2n.data(), sets_2n.data(), N2, logits_2n.data());
            memcpy(logits_cond.data(),   logits_2n.data(),                    (size_t)V * N * sizeof(float));
            memcpy(logits_uncond.data(), logits_2n.data() + (size_t)V * N,    (size_t)V * N * sizeof(float));
        } else {
            qw3lm_forward_batch(m, tokens.data(), cond_sets.data(), N, logits_cond.data());
        }

        // Per-sequence: CFG combine + sample
        for (int i = 0; i < N; i++) {
            if (seqs[i].done) continue;

            float * lc = logits_cond.data() + (size_t)i * V;
            if (use_cfg) {
                float * lu = logits_uncond.data() + (size_t)i * V;
                for (int v = 0; v < V; v++)
                    lc[v] = lu[v] + cfg_scale * (lc[v] - lu[v]);
            }

            // Only audio codes + EOS
            for (int v = 0; v < AUDIO_CODE_BASE; v++)
                if (v != TOKEN_IM_END) lc[v] = -1e9f;

            int tok = kcpp_quick_sample(lc,V,quicklastntoks,rep_pen,top_p,top_k,temperature,acestep_lm_rng);
            quicklastntoks.push_back(tok);
            if (quicklastntoks.size()>rep_pen_range) {
                quicklastntoks.erase(quicklastntoks.begin());
            }
            seqs[i].last_token = tok;

            if (tok == TOKEN_IM_END) {
                seqs[i].done = true;
                n_active--;
            } else if (tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
                seqs[i].audio_codes.push_back(tok - AUDIO_CODE_BASE);
            }
        }

        int total_codes = 0;
        for (int i = 0; i < N; i++) total_codes += (int)seqs[i].audio_codes.size();

        if ((step + 1) % 50 == 0) {
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[Decode] step %d, %d active, %d total codes, %.1f tok/s\n",
                    step + 1, n_active, total_codes, (double)(step + 1) * N / elapsed);
        }
    }

    double decode_ms = t_decode.ms();
    fprintf(stderr, "[Phase2] Decode %.0fms\n", decode_ms);

    // Build results
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = codes_to_string(seqs[i].audio_codes);
        fprintf(stderr, "[Batch %d] seed=%lld, %zu codes\n",
                i, base_seed + i, seqs[i].audio_codes.size());
    }
    return results;
}

//
// CLI
//

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --request <json> --model <gguf> [options]\n"
        "\n"
        "Required:\n"
        "  --request <json>       Input request JSON\n"
        "  --model <gguf>         Model GGUF file\n"
        "\n"
        "Batch:\n"
        "  --batch <N>            Batch N sequences (default: 1)\n"
        "\n"
        "Output naming: input.json -> input0.json, input1.json, ... (last digit = batch index)\n"
        "\n"
        "Debug:\n"
        "  --max-seq <N>          KV cache size (default: 8192)\n"
        "  --no-fsm               Disable FSM constrained decoding\n"
        "  --dump-logits <path>   Dump prefill logits (binary f32)\n"
        "  --dump-tokens <path>   Dump prompt token IDs (CSV)\n"
        , prog);
}

//kcpp stuff

static Qwen3LM acestep_llm;
static BPETokenizer acestep_bpe;
static bool acestep_lm_loaded = false;
static std::string acestep_lm_path = "";
static bool acestep_lm_lowvram = false;

void unload_acestep_lm()
{
    if(acestep_lm_loaded)
    {
        acestep_lm_loaded = false;
        qw3lm_free(&acestep_llm);
        printf("Unload Music LM model...\n");
    }
}

bool load_acestep_lm(std::string model_path, bool lowvram, bool musicdebugmode)
{
    acestep_lm_dbg = musicdebugmode;
    if(acestep_lm_loaded)
    {
        unload_acestep_lm();
    }

    acestep_lm_lowvram = lowvram;
    acestep_lm_path = model_path;
    acestep_lm_loaded = false;
    int max_seq     = 8192;
    const int batch_size  = 1; //only bs 1 is allowed
    if (!load_bpe_from_gguf(&acestep_bpe, model_path.c_str())) {
        return false;
    }
    // Load model
    int n_kv_sets = 2 * batch_size;
    if (!qw3lm_load(&acestep_llm, model_path.c_str(), max_seq, n_kv_sets)) {
        return false;
    }
    acestep_lm_loaded = true;
    return true;
}

std::string acestep_prepare_request(const music_generation_inputs inputs)
{
    if(!acestep_lm_loaded && acestep_lm_path!="")
    {
        printf("\nRuntime reload Music LM model...\n");
        bool ok = load_acestep_lm(acestep_lm_path, acestep_lm_lowvram, acestep_lm_dbg);
        if(!ok)
        {
            printf("\nERROR: Acestep LM load fail\n");
            return "";
        }
    }

    bool rewrite_caption = inputs.rewrite_caption;

    const int batch_size = 1;
    bool use_fsm = true;
    MetadataFSM fsm;
    if (use_fsm) {
        fsm.init(acestep_bpe, acestep_llm.cfg.vocab_size,rewrite_caption);
    }

    // Read request and set essentials
    AceRequest req;
    std::string injson =  inputs.input_json;
    if (!request_parse_from_str(&req, injson))
    {
        fprintf(stderr, "\nMusic JSON parse error\n");
        return "";
    }

    int seed = req.seed;
    if (seed <= 0 || seed==0xFFFFFFFF)
    {
        seed = (((uint32_t)time(NULL)) % 1000000u);
    }
    req.seed = seed;
    acestep_lm_rng = std::mt19937(seed);

    if (req.caption.empty()) {
        req.caption = "An interesting song";
    }

    // Generation params from request
    float temperature      = req.lm_temperature;
    float top_p            = req.lm_top_p;
    int   top_k            = req.lm_top_k;
    float rep_pen          = req.lm_rep_pen;
    float cfg_scale        = req.lm_cfg_scale;
    float phase1cfg = 1.0f;
    const char * neg_prompt = req.lm_negative_prompt.c_str();
    float codes_temperature      = req.codes_temperature;
    float codes_top_p            = req.codes_top_p;
    int   codes_top_k            = req.codes_top_k;

    // Copy request -> AcePrompt (internal LLM struct)
    AcePrompt ace = {};
    ace.caption        = req.caption;
    ace.lyrics         = req.lyrics;
    ace.duration       = req.duration;
    ace.bpm            = req.bpm;
    ace.keyscale       = req.keyscale;
    ace.timesignature  = req.timesignature;
    ace.vocal_language = req.vocal_language;

    bool user_has_codes = !req.audio_codes.empty();
    bool need_lm_codes  = inputs.gen_codes && !user_has_codes;

    bool is_simple = ace.lyrics.empty();

    std::vector<int> prompt;
    std::vector<AcePrompt> aces;  // populated by Phase 1 (simple or partial)

    caption_tokens.clear();
    if(ace.caption!="" && !rewrite_caption)
    {
        caption_tokens = bpe_encode(&acestep_bpe, ace.caption+"\n", false);
    }

    // Preprocessor: simple mode generates lyrics + metas from caption
    if (is_simple) {
        fprintf(stderr, "[Simple] Inspiration\n");

        const char * sys =
            "# Instruction\n"
            "Expand the user's input into a more detailed"
            " and specific musical description:\n";
        std::string user_msg = "# Caption\n"+ace.caption + "\n\ninstrumental: "
            + std::string(req.instrumental ? "true" : "false");
        prompt = build_custom_prompt(acestep_bpe, sys, user_msg.c_str());

        // FSM: reset then optionally force language (shared for both paths)
        fsm.reset();
        if (use_fsm && ace.vocal_language != "unknown" && !ace.vocal_language.empty())
            fsm.force_language(acestep_bpe, ace.vocal_language);

        // Phase 1: N lyrics + metadata generations (always batched, N=batch_size)
        fprintf(stderr, "[Simple] %zu tokens, N=%d, seeds: %lld..%lld\n",
                prompt.size(), batch_size, seed, seed + batch_size - 1);

        auto phase1_texts = generate_phase1_batch(
            &acestep_llm, &acestep_bpe, prompt, 2048, temperature, top_p, top_k, rep_pen,
            seed, batch_size, use_fsm ? &fsm : nullptr, true);

        parse_phase1_into_aces(phase1_texts, ace, aces, seed, "Simple", true);

        for (int i = 0; i < batch_size; i++) qw3lm_reset_kv(&acestep_llm, i);
    }

    // Re-evaluate after possible simple enrichment
    const AcePrompt & ace_ref = aces.empty() ? ace : aces[0];
    bool has_all_metas = (ace_ref.bpm > 0 && ace_ref.duration > 0 &&
                          !ace_ref.keyscale.empty() && !ace_ref.timesignature.empty());

    if (!has_all_metas) {
        // Partial-metas: Phase 1 with CFG to fill missing fields
        prompt = build_lm_prompt(acestep_bpe, ace);
        std::vector<int> uncond;
        if (phase1cfg > 1.0f)
            uncond = build_lm_prompt_uncond(acestep_bpe, ace, neg_prompt);

        fprintf(stderr, "[Partial] %zu tokens, CFG: %.2f, N=%d, seeds: %lld..%lld\n",
                prompt.size(), phase1cfg, batch_size, seed, seed + batch_size - 1);

        fsm.reset();
        auto phase1_texts = generate_phase1_batch(
            &acestep_llm, &acestep_bpe, prompt, 2048, temperature, top_p, top_k, rep_pen,
            seed, batch_size, use_fsm ? &fsm : nullptr, false,
            phase1cfg, uncond.empty() ? nullptr : &uncond, true);

        parse_phase1_into_aces(phase1_texts, ace, aces, seed, "Partial", false);

        for (int i = 0; i < 2 * batch_size; i++) qw3lm_reset_kv(&acestep_llm, i);
    }
    fsm.reset();

    // Guarantee aces is populated (all-metas: single shared ace for prefill optimization)
    if (aces.empty()) {
        aces = { ace };
    }

    // Phase 2: generate audio codes (always batched, N=batch_size)
    std::vector<std::string> batch_codes(batch_size);
    if (need_lm_codes) {
        batch_codes = run_phase2_batch(&acestep_llm, acestep_bpe, aces,
            temperature, top_p, top_k, rep_pen, seed, batch_size, cfg_scale, neg_prompt);
    } else {
        fprintf(stderr, "[Skip] %s, no code generation\n",
                user_has_codes ? "user codes present" : "thinking=false");
    }

    // only batch size 1 is allowed
    AceRequest rr = req;
    const AcePrompt & a = aces[0];
    rr.caption        = a.caption;
    rr.lyrics         = a.lyrics;
    rr.bpm            = a.bpm;
    rr.duration       = a.duration;
    rr.keyscale       = a.keyscale;
    rr.timesignature  = a.timesignature;
    rr.vocal_language = a.vocal_language;
    if (!batch_codes[0].empty()) rr.audio_codes = batch_codes[0];

    std::string prefix_erase = "# Lyric";
    // Check if the string is long enough and starts with the prefix
    if (rr.lyrics.size() >= prefix_erase.size() && rr.lyrics.compare(0, prefix_erase.size(), prefix_erase) == 0) {
        rr.lyrics = rr.lyrics.substr(prefix_erase.size()); // Returns a new string starting after the prefix
    }

    prefix_erase = "keyscale:";
    // Check if the string is long enough and starts with the prefix
    if (rr.keyscale.size() >= prefix_erase.size() && rr.keyscale.compare(0, prefix_erase.size(), prefix_erase) == 0) {
        rr.keyscale = rr.keyscale.substr(prefix_erase.size()); // Returns a new string starting after the prefix
    }

    std::string delimiter = "durationduration";
    size_t ddpos = rr.caption.find(delimiter);
    rr.caption = ((ddpos != std::string::npos) ? rr.caption.substr(0, ddpos) : rr.caption);

    //now convert to string
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"caption\": \"" << json_escape(rr.caption) << "\",\n";
    oss << "  \"lyrics\": \"" << json_escape(rr.lyrics) << "\",\n";
    if (rr.instrumental) {
        oss << "  \"instrumental\": true,\n";
    }
    oss << "  \"bpm\": " << rr.bpm << ",\n";
    oss << "  \"duration\": " << std::fixed << std::setprecision(1) << rr.duration << ",\n";
    oss << "  \"keyscale\": \"" << json_escape(rr.keyscale) << "\",\n";
    oss << "  \"timesignature\": \"" << json_escape(rr.timesignature) << "\",\n";
    oss << "  \"vocal_language\": \"" << json_escape(rr.vocal_language) << "\",\n";
    oss << "  \"task_type\": \"" << json_escape(rr.task_type) << "\",\n";
    oss << "  \"seed\": " << rr.seed << ",\n";
    oss << "  \"thinking\": " << (rr.thinking ? "true" : "false") << ",\n";
    oss << "  \"lm_temperature\": " << std::fixed << std::setprecision(2) << rr.lm_temperature << ",\n";
    oss << "  \"lm_cfg_scale\": " << std::fixed << std::setprecision(1) << rr.lm_cfg_scale << ",\n";
    oss << "  \"lm_top_p\": " << std::fixed << std::setprecision(2) << rr.lm_top_p << ",\n";
    oss << "  \"lm_top_k\": " << std::fixed << std::setprecision(1) << rr.lm_top_k << ",\n";
    oss << "  \"codes_temperature\": " << std::fixed << std::setprecision(2) << rr.codes_temperature << ",\n";
    oss << "  \"codes_top_p\": " << std::fixed << std::setprecision(2) << rr.codes_top_p << ",\n";
    oss << "  \"codes_top_k\": " << std::fixed << std::setprecision(1) << rr.codes_top_k << ",\n";
    oss << "  \"lm_rep_pen\": " << std::fixed << std::setprecision(2) << rr.lm_rep_pen << ",\n";
    oss << "  \"lm_negative_prompt\": \"" << json_escape(rr.lm_negative_prompt) << "\",\n";
    oss << "  \"inference_steps\": " << rr.inference_steps << ",\n";
    oss << "  \"guidance_scale\": " << std::fixed << std::setprecision(1) << rr.guidance_scale << ",\n";
    oss << "  \"shift\": " << std::fixed << std::setprecision(1) << rr.shift << ",\n";
    oss << "  \"audio_codes\": \"" << json_escape(rr.audio_codes) << "\"\n";
    oss << "}\n";
    std::string output_json = oss.str();

    if(acestep_lm_lowvram)
    {
        unload_acestep_lm();
    }

    return output_json;
}

