#pragma once

#include <string>
#include <vector>
#include <deque>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <regex>
#include <memory>

struct lookahead_grammar_state; // grammar parse state at a sentence start (defined in gpttype_adapter.cpp)

struct lookahead_config
{
    bool enabled = false;
    int retries = 0;            // extra attempts per segment (0 = off)
    int sentences = 2;          // sentences held back = how far back a failed sentence can backtrack
    float threshold = 0.35f;    // accept a segment when score <= threshold
    int min_len = 6;            // a sentence may end at punctuation only after this many tokens
    int max_len = 48;           // hard sentence length
    float w_rep = 1.0f;         // repetition weight
    float w_slop = 1.0f;        // slop phrase weight
    float w_struct = 0.6f;      // structural pattern weight
    float w_lp = 0.5f;          // fluency (logprob drop) weight
    std::vector<std::string> slop_phrases; // lowercase
    std::vector<float> slop_weights;
    std::vector<char> slop_whole_word;     // match only as a whole word
    std::vector<char> slop_exempt_seen;    // ignore if it already appears in the context (names)
    std::vector<char> slop_is_name;        // also match by sound (single-word names)
    float fuzzy_cutoff = 0.6f;  // minimum sound similarity for a near-slop match (>=1 disables)
    float fuzzy_base = 0.3f;    // near-slop weight with no suppression (only when the first sound matches)
    float fuzzy_boost = 1.5f;   // extra near-slop weight per unit of suppressed mass
    bool track = false;         // count stand-ins across generations
    int track_min = 3;          // stand-in count at which it becomes a soft slop entry
    float track_weight = 0.5f;  // weight of tracked entries
    float retry_temp = 0.2f;    // temperature added per retry of a sentence, so a retry does not just repeat the rejected one
    std::vector<std::string> allow; // lowercase words that are never penalized (e.g. the names of established characters)
};

struct lookahead_candidate
{
    std::vector<int> tokens;
    std::vector<std::string> strs;
    std::vector<float> lps;
    std::vector<float> supp;    // suppressed mass at each token position
    std::vector<TopPicksData> picks;
    std::vector<std::string> scored_strs; // token strings as scored (JSON-masked in JSON mode)
    float score = 0.0f;
};

// one sentence in the held-back window
struct lookahead_slot
{
    int start = 0;              // index into current_context_tokens where the sentence begins
    int picks_start = 0;        // top_picks_history size at sentence begin
    std::vector<lookahead_candidate> attempts;
    std::vector<int> bans;      // tokens that started a slop phrase in a rejected attempt
    bool accepted = false;
    int accepted_idx = -1;
    std::shared_ptr<lookahead_grammar_state> grammar_snap; // grammar state at the sentence start, if a grammar is active
};

// incremental JSON lexer state, kept per generated token so any rewind can restore it
struct la_json_lex
{
    bool in_str = false;
    bool esc = false;
    int u_skip = 0;        // hex digits of a \uXXXX escape still to skip
    bool after_str = false; // a string just closed; the next structural char tells key from value
};

struct lookahead_state
{
    bool active = false;
    std::vector<lookahead_slot> slots; // window: accepted sentences, last one is in progress
    std::vector<float> seg_lps; // raw logprob of each token of the in-progress sentence
    std::vector<float> seg_supp; // suppressed mass at each token of the in-progress sentence
    std::deque<std::string> recent_openers; // openers of released sentences
    std::string prompt_tail;    // detokenized tail of the prompt, for cross-turn repetition
    float base_lp = 0.0f;
    bool base_lp_init = false;
    int attempts_since_release = 0;
    int total_retries = 0;
    int total_backtracks = 0;
    int total_replays = 0;
    // output mode, decided from the first characters of the reply: -1 undecided, 0 text, 1 JSON
    int gen_start = 0;                  // context size when generation started
    int loop_fixes = 0;                 // token loops cut in this generation
    bool stop_requested = false;        // the model kept looping: end the generation
    int mode = -1;
    int mode_at = 0;                    // number of generated tokens when the mode was decided
    std::vector<std::string> raw_tok;   // generated token strings while the mode is undecided
    std::vector<std::string> scored_tok; // JSON mode: masked string of every generated token
    std::vector<la_json_lex> lex_tok;   // JSON mode: lexer state after every generated token

    void reset()
    {
        active = false;
        slots.clear();
        seg_lps.clear();
        seg_supp.clear();
        recent_openers.clear();
        prompt_tail.clear();
        base_lp = 0.0f;
        base_lp_init = false;
        attempts_since_release = 0;
        total_retries = 0;
        total_backtracks = 0;
        total_replays = 0;
        gen_start = 0;
        loop_fixes = 0;
        stop_requested = false;
        mode = -1;
        mode_at = 0;
        raw_tok.clear();
        scored_tok.clear();
        lex_tok.clear();
    }
};

struct lookahead_slop_hit
{
    size_t pos;        // byte offset in the lowercased segment
    int phrase;        // index into cfg.slop_phrases
    std::string text;  // the matched text (the phrase itself, or the sound-alike word)
    float sim = 1.0f;  // sound similarity for near-slop hits
    float supp = 0.0f; // suppressed mass at the hit
};

// a word of a segment, with the info near-slop matching needs
struct lookahead_word
{
    std::string lower;
    size_t pos = 0;           // byte offset in the lowercased segment
    bool capitalized = false; // first letter was upper case in the original text
    bool sentence_start = false;
    float supp = 0.0f;        // max suppressed mass over the tokens of the word
};

struct lookahead_score_detail
{
    float rep = 0, slop = 0, structure = 0, lp = 0, total = 0;
    std::vector<lookahead_slop_hit> slop_hits;
};

// ---------- text helpers ----------

static inline std::string la_lower(const std::string & s)
{
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i)
    {
        unsigned char c = (unsigned char)s[i];
        // normalize curly apostrophes (U+2019 / U+2018) to '
        if (c == 0xE2 && i + 2 < s.size() && (unsigned char)s[i+1] == 0x80 &&
            ((unsigned char)s[i+2] == 0x99 || (unsigned char)s[i+2] == 0x98))
        {
            out.push_back('\'');
            i += 2;
            continue;
        }
        out.push_back((c < 128) ? (char)std::tolower(c) : (char)c);
    }
    return out;
}

static inline bool la_is_word_char(unsigned char c)
{
    return std::isalnum(c) || c == '\'' || c >= 128;
}

static inline std::vector<std::string> la_words(const std::string & lower)
{
    std::vector<std::string> words;
    std::string cur;
    for (unsigned char c : lower)
    {
        if (la_is_word_char(c))
        {
            cur.push_back((char)c);
        }
        else if (!cur.empty())
        {
            words.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty())
    {
        words.push_back(cur);
    }
    return words;
}

static inline std::string la_join(const std::vector<std::string> & w, size_t from, size_t n)
{
    std::string s;
    for (size_t i = from; i < from + n && i < w.size(); ++i)
    {
        if (i > from) s.push_back(' ');
        s += w[i];
    }
    return s;
}

static inline size_t la_count(const std::string & hay, const std::string & needle)
{
    if (needle.empty()) return 0;
    size_t n = 0, pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos)
    {
        ++n;
        pos += needle.size();
    }
    return n;
}

// Find `needle` in `hay` starting at `from`. A match must begin at a word start (so "aria"
// does not hit "malaria"); with whole_word it must also end at a word end ("aria" vs "arias").
static inline size_t la_find_phrase(const std::string & hay, const std::string & needle, bool whole_word, size_t from = 0)
{
    if (needle.empty()) return std::string::npos;
    const bool needle_starts_word = la_is_word_char((unsigned char)needle.front());
    const bool needle_ends_word = la_is_word_char((unsigned char)needle.back());
    size_t pos = from;
    while ((pos = hay.find(needle, pos)) != std::string::npos)
    {
        bool ok = true;
        if (needle_starts_word && pos > 0 && la_is_word_char((unsigned char)hay[pos-1])) ok = false;
        size_t end = pos + needle.size();
        if (ok && whole_word && needle_ends_word && end < hay.size() && la_is_word_char((unsigned char)hay[end])) ok = false;
        if (ok) return pos;
        pos += 1;
    }
    return std::string::npos;
}

static inline bool la_is_segment_end(const std::string & tokstr)
{
    if (tokstr.find_first_of(".!?\n") != std::string::npos) return true;
    // CJK full-width terminators
    return tokstr.find("\xE3\x80\x82") != std::string::npos ||   // 。
           tokstr.find("\xEF\xBC\x81") != std::string::npos ||   // ！
           tokstr.find("\xEF\xBC\x9F") != std::string::npos;     // ？
}

// number of completed sentences in a token string list, using the same rule as segment ends
static inline int la_count_sentences(const std::vector<std::string> & strs, int min_len)
{
    int count = 0, since = 0;
    for (const auto & s : strs)
    {
        ++since;
        if (since >= min_len && la_is_segment_end(s))
        {
            ++count;
            since = 0;
        }
    }
    return count;
}

// split text into sentences on . ! ? and newlines
static inline std::vector<std::string> la_split_sentences(const std::string & text)
{
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < text.size(); ++i)
    {
        char c = text[i];
        cur.push_back(c);
        if (c == '.' || c == '!' || c == '?' || c == '\n')
        {
            // keep runs like "..." or "?!" together
            while (i + 1 < text.size() && (text[i+1] == '.' || text[i+1] == '!' || text[i+1] == '?' || text[i+1] == '"' || text[i+1] == '\''))
            {
                cur.push_back(text[++i]);
            }
            if (cur.find_first_not_of(" \t\r\n.!?\"'") != std::string::npos) out.push_back(cur);
            cur.clear();
        }
    }
    if (cur.find_first_not_of(" \t\r\n.!?\"'") != std::string::npos) out.push_back(cur);
    return out;
}

static inline std::string la_opener(const std::string & seg_lower)
{
    std::vector<std::string> w = la_words(seg_lower);
    if (w.size() < 2) return "";
    return la_join(w, 0, 2);
}

// common capitalized words that are never treated as names
static const char * lookahead_stopwords[] = {
    "the","a","an","and","but","or","so","then","when","while","as","if","though","although","because",
    "he","she","it","they","we","you","i","his","her","their","our","your","my","its","him","them","us","me",
    "this","that","these","those","there","here","what","who","where","why","how","which",
    "yes","no","not","now","just","still","yet","after","before","with","without","for","from","into","onto",
    "in","on","at","by","of","to","up","down","out","over","under","again","once","all","some","every","each",
    "one","two","three","maybe","perhaps","suddenly","finally","slowly","quickly","instead","meanwhile","even",
    "oh","ah","hey","well","okay","ok","sir","madam","lord","lady","mr","mrs","ms","dr",
};

static inline bool la_is_stopword(const std::string & w)
{
    for (const char * sw : lookahead_stopwords) { if (w == sw) return true; }
    return false;
}

// Lowercase like la_lower, and also report for each output byte whether it was upper case.
static inline void la_lower_flags(const std::string & s, std::string & out, std::vector<char> & upper)
{
    for (size_t i = 0; i < s.size(); ++i)
    {
        unsigned char c = (unsigned char)s[i];
        if (c == 0xE2 && i + 2 < s.size() && (unsigned char)s[i+1] == 0x80 &&
            ((unsigned char)s[i+2] == 0x99 || (unsigned char)s[i+2] == 0x98))
        {
            out.push_back('\'');
            upper.push_back(0);
            i += 2;
            continue;
        }
        out.push_back((c < 128) ? (char)std::tolower(c) : (char)c);
        upper.push_back((c < 128 && std::isupper(c)) ? 1 : 0);
    }
}

// Build the lowercased segment, token byte offsets and the word list from token strings.
static inline std::vector<lookahead_word> la_extract_words(const std::vector<std::string> & strs, const std::vector<float> & supp,
    std::string & seg_lower, std::vector<size_t> & tok_offsets)
{
    std::vector<char> upper;
    seg_lower.clear();
    tok_offsets.clear();
    for (const auto & t : strs)
    {
        tok_offsets.push_back(seg_lower.size());
        la_lower_flags(t, seg_lower, upper);
    }
    std::vector<lookahead_word> words;
    bool at_sentence_start = true;
    size_t i = 0;
    while (i < seg_lower.size())
    {
        unsigned char c = (unsigned char)seg_lower[i];
        if (!la_is_word_char(c))
        {
            if (c == '.' || c == '!' || c == '?' || c == '\n' || c == '"' || c == ':') at_sentence_start = true;
            ++i;
            continue;
        }
        size_t start = i;
        while (i < seg_lower.size() && la_is_word_char((unsigned char)seg_lower[i])) ++i;
        lookahead_word w;
        w.lower = seg_lower.substr(start, i - start);
        // strip possessive 's
        if (w.lower.size() > 2 && w.lower.compare(w.lower.size()-2, 2, "'s") == 0) w.lower.resize(w.lower.size()-2);
        w.pos = start;
        w.capitalized = upper[start] != 0;
        w.sentence_start = at_sentence_start;
        at_sentence_start = false;
        // tokens covering [start, i)
        for (size_t ti = 0; ti < tok_offsets.size(); ++ti)
        {
            size_t tb = tok_offsets[ti];
            size_t te = (ti + 1 < tok_offsets.size()) ? tok_offsets[ti+1] : seg_lower.size();
            if (te <= start) continue;
            if (tb >= i) break;
            if (ti < supp.size()) w.supp = std::max(w.supp, supp[ti]);
        }
        words.push_back(w);
    }
    return words;
}

// Fold spellings that sound alike, so the edit distance below measures sound, not spelling.
static inline std::string la_phonetic_fold(const std::string & w)
{
    std::string s;
    for (size_t i = 0; i < w.size(); ++i)
    {
        char c = w[i];
        char n = (i + 1 < w.size()) ? w[i+1] : 0;
        if (!std::isalpha((unsigned char)c)) continue;
        if (c == 'p' && n == 'h') { s.push_back('f'); ++i; continue; }
        if (c == 'c' && n == 'k') { s.push_back('k'); ++i; continue; }
        if (c == 'a' && n == 'e') { s.push_back('a'); ++i; continue; }
        if (c == 't' && n == 'h') { s.push_back('t'); ++i; continue; }
        if (c == 'y') c = 'i';
        else if (c == 'c') c = 'k';
        else if (c == 'q') c = 'k';
        else if (c == 'z') c = 's';
        else if (c == 'v') c = 'f';
        if (!s.empty() && s.back() == c) continue; // collapse double letters
        s.push_back(c);
    }
    while (s.size() > 3 && (s.back() == 'e' || s.back() == 'h')) s.pop_back(); // silent endings
    return s;
}

// Optimal string alignment distance (Levenshtein + adjacent transposition)
static inline int la_osa_distance(const std::string & a, const std::string & b)
{
    const size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> d(n + 1, std::vector<int>(m + 1));
    for (size_t i = 0; i <= n; ++i) d[i][0] = (int)i;
    for (size_t j = 0; j <= m; ++j) d[0][j] = (int)j;
    for (size_t i = 1; i <= n; ++i)
    {
        for (size_t j = 1; j <= m; ++j)
        {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            d[i][j] = std::min({d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost});
            if (i > 1 && j > 1 && a[i-1] == b[j-2] && a[i-2] == b[j-1])
                d[i][j] = std::min(d[i][j], d[i-2][j-2] + 1);
        }
    }
    return d[n][m];
}

// 0..1 sound similarity of two lowercase words. Edit distance on the folded spelling, but only
// when the words are anchored the way sound-alike names are: same opening (Elena ~ Elara) or
// a rhyme (Clara ~ Elara). Without the anchor, short names two letters apart (Marta, Aria)
// would count as similar.
static inline float la_sound_similarity(const std::string & a, const std::string & b)
{
    std::string fa = la_phonetic_fold(a), fb = la_phonetic_fold(b);
    if (fa.empty() || fb.empty()) return 0.0f;
    if (fa == fb) return 1.0f;
    size_t mx = std::max(fa.size(), fb.size()), mn = std::min(fa.size(), fb.size());
    if (mx - mn > 3) return 0.0f;
    size_t prefix = 0, suffix = 0;
    while (prefix < mn && fa[prefix] == fb[prefix]) ++prefix;
    while (suffix < mn && fa[fa.size()-1-suffix] == fb[fb.size()-1-suffix]) ++suffix;
    if (prefix < 2 && suffix < 3) return 0.0f;
    return 1.0f - (float)la_osa_distance(fa, fb) / (float)mx;
}

static inline bool la_contains_word(const std::string & hay_lower, const std::string & w)
{
    return la_find_phrase(hay_lower, w, true, 0) != std::string::npos;
}

// ---------- JSON output ----------

// Decide from the start of a reply whether it is JSON: the first '{' (or a '[' that opens an array
// of values) after only whitespace, <tags> or a ``` fence. Returns -1 while undecided, 0 for text,
// 1 for JSON with json_start set.
static inline int la_json_detect(const std::string & text, size_t & json_start)
{
    size_t i = 0;
    while (i < text.size())
    {
        unsigned char c = (unsigned char)text[i];
        if (std::isspace(c)) { ++i; continue; }
        if (c == '{') { json_start = i; return 1; }
        if (c == '[')
        {
            // "[OOC: ...]" or "[Scene]" in prose is not JSON
            size_t j = i + 1;
            while (j < text.size() && std::isspace((unsigned char)text[j])) ++j;
            if (j >= text.size()) return -1;
            char n = text[j];
            if (n == '{' || n == '[' || n == '"' || n == ']' || std::isdigit((unsigned char)n) || n == '-') { json_start = i; return 1; }
            return 0;
        }
        if (c == '<')
        {
            size_t e = text.find('>', i);
            if (e == std::string::npos) return (text.size() - i > 64) ? 0 : -1;
            i = e + 1;
            continue;
        }
        if (c == '`')
        {
            if (text.size() - i < 3) return -1;
            if (text.compare(i, 3, "```") != 0) return 0;
            size_t e = text.find('\n', i);
            if (e == std::string::npos) return -1;
            i = e + 1;
            continue;
        }
        return 0;
    }
    return -1;
}

// Lex one token of JSON output and return its mask (same length): string contents are kept, all
// structure becomes spaces, an escaped \n becomes a newline, and the structural character after a
// string value (',' '}' ']') becomes '\n', so each value ends a sentence. Keys are followed by ':'
// and so never end one. Only this token's bytes are touched, so the per-token cost is tiny.
static inline std::string la_json_lex_token(const std::string & tok, la_json_lex & lx)
{
    std::string m(tok.size(), ' ');
    for (size_t i = 0; i < tok.size(); ++i)
    {
        char c = tok[i];
        if (lx.in_str)
        {
            if (lx.u_skip > 0) { --lx.u_skip; continue; }
            if (lx.esc)
            {
                lx.esc = false;
                m[i] = (c == 'n') ? '\n' : ((c == 't' || c == 'r' || c == 'u') ? ' ' : c);
                if (c == 'u') lx.u_skip = 4;
                continue;
            }
            if (c == '\\') { lx.esc = true; continue; }
            if (c == '"') { lx.in_str = false; lx.after_str = true; continue; }
            m[i] = c;
            continue;
        }
        if (c == '"') { lx.in_str = true; lx.after_str = false; continue; }
        if (std::isspace((unsigned char)c)) continue;
        if (lx.after_str && (c == ',' || c == '}' || c == ']')) m[i] = '\n';
        lx.after_str = false;
    }
    return m;
}

static inline bool la_is_allowed(const lookahead_config & cfg, const std::string & w)
{
    return std::find(cfg.allow.begin(), cfg.allow.end(), w) != cfg.allow.end();
}

// A name counts as established when it appears at least twice in the context. A single mention is
// usually a "don't use these names" list in the prompt, which must not exempt the name.
static inline bool la_is_established(const std::string & hist_lower, const std::string & w, bool whole)
{
    size_t pos = la_find_phrase(hist_lower, w, whole, 0);
    if (pos == std::string::npos) return false;
    return la_find_phrase(hist_lower, w, whole, pos + w.size()) != std::string::npos;
}

// ---------- token loops ----------

// Degenerate loops ("med med med ...") never reach a sentence end, so they are caught per token on
// token ids: the tail of toks[from..] repeating with a period of 1..8 tokens (6 copies for a single
// token, 4 for longer periods, so "no, no, no" in prose is left alone). Returns the index where the
// first repeat starts (everything from there on is the loop), or -1.
static inline int la_find_loop(const std::vector<int> & toks, int from, int & period)
{
    const int end = (int)toks.size();
    for (int p = 1; p <= 8; ++p)
    {
        const int need = (p == 1) ? 6 : 4;
        if (end - from < p * need) continue;
        bool ok = true;
        for (int i = end - p * need; i < end - p && ok; ++i)
        {
            if (toks[i] != toks[i + p]) ok = false;
        }
        if (!ok) continue;
        int s = end - p * need;
        while (s - 1 >= from && toks[s - 1] == toks[s - 1 + p]) --s; // the run may have started earlier
        period = p;
        return s + p;
    }
    return -1;
}

// ---------- critics ----------

// 0..1. Exact sentence repeats, 4-gram overlap with history, and internal trigram looping.
static inline float la_score_repetition(const std::string & seg_lower, const std::string & hist_lower)
{
    std::vector<std::string> sw = la_words(seg_lower);
    if (sw.size() < 3) return 0.0f;
    std::vector<std::string> hw = la_words(hist_lower);
    std::string hjoined = la_join(hw, 0, hw.size());

    // exact repeat of any sentence of the segment
    for (const auto & sent : la_split_sentences(seg_lower))
    {
        std::vector<std::string> ww = la_words(sent);
        std::string trimmed = la_join(ww, 0, ww.size());
        if (trimmed.size() >= 16 && hjoined.find(trimmed) != std::string::npos) return 1.0f;
    }

    float overlap = 0.0f;
    if (sw.size() >= 4)
    {
        std::unordered_set<std::string> hist4;
        for (size_t i = 0; i + 4 <= hw.size(); ++i) hist4.insert(la_join(hw, i, 4));
        int total = 0, hit = 0;
        for (size_t i = 0; i + 4 <= sw.size(); ++i)
        {
            ++total;
            if (hist4.count(la_join(sw, i, 4))) ++hit;
        }
        if (total > 0) overlap = (float)hit / (float)total;
    }

    float internal = 0.0f;
    {
        std::unordered_set<std::string> seen;
        int total = 0, dup = 0;
        for (size_t i = 0; i + 3 <= sw.size(); ++i)
        {
            ++total;
            if (!seen.insert(la_join(sw, i, 3)).second) ++dup;
        }
        if (total > 0) internal = (float)dup / (float)total;
    }

    return std::max(overlap, internal);
}

// 0..1. Sum of weights of slop phrases found in the segment, plus near-slop (sound-alike
// names). Records every hit position.
static inline float la_score_slop(const std::string & seg_lower, const std::vector<lookahead_word> & words,
    const std::string & hist_lower, const lookahead_config & cfg, std::vector<lookahead_slop_hit> & hits)
{
    float s = 0.0f;
    std::vector<char> exact_word(words.size(), 0);
    for (size_t i = 0; i < cfg.slop_phrases.size(); ++i)
    {
        const std::string & ph = cfg.slop_phrases[i];
        const bool whole = cfg.slop_whole_word[i];
        size_t pos = la_find_phrase(seg_lower, ph, whole, 0);
        if (pos == std::string::npos) continue;
        if (la_is_allowed(cfg, ph)) continue;
        if (cfg.slop_exempt_seen[i] && la_is_established(hist_lower, ph, whole)) continue;
        while (pos != std::string::npos)
        {
            s += cfg.slop_weights[i];
            lookahead_slop_hit h;
            h.pos = pos; h.phrase = (int)i; h.text = ph;
            for (size_t w = 0; w < words.size(); ++w)
            {
                if (words[w].pos == pos) { exact_word[w] = 1; h.supp = words[w].supp; }
            }
            hits.push_back(h);
            pos = la_find_phrase(seg_lower, ph, whole, pos + ph.size());
        }
    }

    // near-slop: words that sound like a listed name
    if (cfg.fuzzy_cutoff > 0.0f && cfg.fuzzy_cutoff < 1.0f)
    {
        for (size_t w = 0; w < words.size(); ++w)
        {
            const lookahead_word & wd = words[w];
            if (exact_word[w] || wd.lower.size() < 3 || la_is_stopword(wd.lower)) continue;
            if (!wd.capitalized && wd.supp < 0.05f) continue; // plain lowercase word, nothing banned here
            if (la_is_allowed(cfg, wd.lower)) continue;
            if (la_is_established(hist_lower, wd.lower, true)) continue; // already established in the text
            float best = 0.0f; int best_i = -1; float best_sim = 0.0f;
            for (size_t i = 0; i < cfg.slop_phrases.size(); ++i)
            {
                if (!cfg.slop_is_name[i]) continue;
                const std::string & ph = cfg.slop_phrases[i];
                if (ph == wd.lower) continue;
                float sim = la_sound_similarity(wd.lower, ph);
                if (sim < cfg.fuzzy_cutoff) continue;
                float c;
                if (sim >= 0.95f)
                {
                    c = cfg.slop_weights[i]; // sounds identical: same as the listed name
                }
                else
                {
                    std::string fa = la_phonetic_fold(wd.lower), fb = la_phonetic_fold(ph);
                    bool same_first = !fa.empty() && !fb.empty() && fa[0] == fb[0];
                    c = sim * ((same_first ? cfg.fuzzy_base : 0.0f) + cfg.fuzzy_boost * wd.supp) * cfg.slop_weights[i];
                }
                if (c > best) { best = c; best_i = (int)i; best_sim = sim; }
            }
            if (best_i >= 0 && best > 0.0f)
            {
                s += best;
                lookahead_slop_hit h;
                h.pos = wd.pos; h.phrase = best_i; h.text = wd.lower; h.sim = best_sim; h.supp = wd.supp;
                hits.push_back(h);
            }
        }
    }
    return std::min(1.0f, s);
}

static inline float la_score_sentence_structure(const std::string & sent_lower)
{
    static const std::regex re_notbut(
        R"((\bnot (just|only|merely|simply)\b[^.!?]{1,80}\bbut\b)|(\b(it|this|that)('s| is| was) not\b[^.!?]{1,60}[,;]\s*(it|this|that)('s| is| was)\b)|(\b(isn't|wasn't|aren't) (just|only|merely)\b))",
        std::regex::ECMAScript | std::regex::optimize);
    static const std::regex re_tricolon(
        R"(\b[a-z']+( [a-z']+)?, [a-z']+( [a-z']+)?,? and [a-z']+\b)",
        std::regex::ECMAScript | std::regex::optimize);

    float s = 0.0f;
    if (std::regex_search(sent_lower, re_notbut)) s += 0.6f;
    size_t dashes = la_count(sent_lower, "\xE2\x80\x94"); // em dash
    if (dashes >= 2) s += 0.5f;
    else if (dashes == 1) s += 0.15f;
    if (std::regex_search(sent_lower, re_tricolon)) s += 0.25f;
    return s;
}

// 0..1. Structural tics per sentence ("not X, but Y", em-dash spam, tricolons) plus sentence
// openers repeated from recent committed text or within the segment. Worst sentence wins.
static inline float la_score_structure(const std::string & seg_lower, const std::deque<std::string> & recent_openers)
{
    float worst = 0.0f;
    std::vector<std::string> openers(recent_openers.begin(), recent_openers.end());
    for (const auto & sent : la_split_sentences(seg_lower))
    {
        float s = la_score_sentence_structure(sent);
        std::string op = la_opener(sent);
        if (!op.empty())
        {
            if (std::find(openers.begin(), openers.end(), op) != openers.end()) s += 0.4f;
            openers.push_back(op);
        }
        worst = std::max(worst, s);
    }
    return std::min(1.0f, worst);
}

static inline lookahead_score_detail la_score_segment(const std::string & seg_lower, const std::vector<lookahead_word> & words, const std::string & hist_text,
    float seg_mean_lp, const std::deque<std::string> & openers, const lookahead_state & st, const lookahead_config & cfg, bool json)
{
    lookahead_score_detail d;
    std::string hist_lower = la_lower(hist_text);
    if (cfg.w_rep > 0 && !json) d.rep = la_score_repetition(seg_lower, hist_lower); // JSON repeats its own structure
    if (cfg.w_slop > 0) d.slop = la_score_slop(seg_lower, words, hist_lower, cfg, d.slop_hits);
    if (cfg.w_struct > 0) d.structure = la_score_structure(seg_lower, openers);
    if (cfg.w_lp > 0 && st.base_lp_init)
    {
        float drop = st.base_lp - seg_mean_lp; // positive when the segment is less likely than usual
        d.lp = std::max(0.0f, std::min(1.0f, drop / 2.0f));
    }
    d.total = cfg.w_rep * d.rep + cfg.w_slop * d.slop + cfg.w_struct * d.structure + cfg.w_lp * d.lp;
    return d;
}

// log-sum-exp of the raw logits
static inline float la_logsumexp(const float * logits, int n_vocab)
{
    float mx = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) mx = std::max(mx, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double)(logits[i] - mx));
    return (float)((double)mx + std::log(sum));
}