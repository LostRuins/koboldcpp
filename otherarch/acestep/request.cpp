//
// request.cpp - AceStep request JSON read/write
//
// Flat JSON only (no nested objects). Handles string escaping for lyrics etc.
//

#include "./request.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Defaults (aligned with Python GenerationParams)
void request_init(AceRequest * r) {
    r->caption            = "";
    r->lyrics             = "";
    r->instrumental       = false;
    r->bpm                = 0;
    r->duration           = -1.0f;
    r->keyscale           = "";
    r->timesignature      = "";
    r->vocal_language     = "en";
    r->task_type          = "text2music";
    r->seed               = -1;
    r->thinking           = false;
    r->lm_temperature     = 0.9f;
    r->lm_cfg_scale       = 3.0f;
    r->lm_top_p           = 0.9f;
    r->lm_top_k           = 50;
    r->codes_temperature  = 1.0f;
    r->codes_top_p        = 0.99f;
    r->codes_top_k        = 1000;
    r->lm_rep_pen         = 1.03f;
    r->lm_negative_prompt = "";
    r->audio_codes        = "";
    r->inference_steps    = 8;
    r->guidance_scale     = 1.0f;
    r->shift              = 3.0f;
    r->audio_cover_strength = 0.5f;
}

// JSON string escape / unescape
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string json_unescape(const char * s, size_t len) {
    std::string out;
    out.reserve(len);
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '\\' && i + 1 < len) {
            switch (s[++i]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u':
                    // \\uXXXX: parse 4 hex digits, emit as UTF-8 (ASCII subset only)
                    if (i + 4 < len) {
                        char hex[5] = {s[i+1], s[i+2], s[i+3], s[i+4], 0};
                        unsigned cp = (unsigned)strtoul(hex, nullptr, 16);
                        i += 4;
                        if (cp < 0x80) {
                            out += (char)cp;
                        } else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                default: out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

// Minimal flat JSON parser
struct JsonPair {
    std::string key;
    std::string value;  // raw value (unquoted strings are unescaped, numbers/bools as-is)
    bool is_string;
};

static const char * skip_ws(const char * p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

// Parse a JSON string starting at p (must point to opening '"').
// Returns pointer past closing '"', or nullptr on error.
static const char * parse_string(const char * p, std::string * out) {
    if (*p != '"') return nullptr;
    p++;
    const char * start = p;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (!*p) return nullptr;
        }
        p++;
    }
    if (*p != '"') return nullptr;
    *out = json_unescape(start, (size_t)(p - start));
    return p + 1;
}

// Parse a JSON value (string, number, bool, null).
// Skips arrays/objects by bracket matching (for forward compat).
static const char * parse_value(const char * p, std::string * out, bool * is_str) {
    *is_str = false;
    if (*p == '"') {
        *is_str = true;
        return parse_string(p, out);
    }
    if (*p == '[' || *p == '{') {
        // skip nested structure (not used, but don't choke)
        char open = *p, close = (*p == '[') ? ']' : '}';
        int depth = 1;
        const char * start = p;
        p++;
        while (*p && depth > 0) {
            if (*p == open) depth++;
            else if (*p == close) depth--;
            else if (*p == '"') {
                // skip strings inside nested structure
                std::string dummy;
                p = parse_string(p, &dummy);
                if (!p) return nullptr;
                continue;
            }
            p++;
        }
        *out = std::string(start, (size_t)(p - start));
        return p;
    }
    // number, bool, null
    const char * start = p;
    while (*p && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    *out = std::string(start, (size_t)(p - start));
    return p;
}

static bool parse_json_flat(const char * json, std::vector<JsonPair> * pairs) {
    const char * p = skip_ws(json);
    if (*p != '{') return false;
    p = skip_ws(p + 1);

    while (*p && *p != '}') {
        JsonPair kv;
        p = parse_string(p, &kv.key);
        if (!p) return false;
        p = skip_ws(p);
        if (*p != ':') return false;
        p = skip_ws(p + 1);
        p = parse_value(p, &kv.value, &kv.is_string);
        if (!p) return false;
        pairs->push_back(kv);
        p = skip_ws(p);
        if (*p == ',') p = skip_ws(p + 1);
    }
    return true;
}

// File I/O helpers
static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf((size_t)sz, '\0');
    size_t nr = fread(&buf[0], 1, (size_t)sz, f);
    fclose(f);
    if ((long)nr != sz) buf.resize(nr);
    return buf;
}

bool request_parse(AceRequest * r, const char * path)
{
    std::string json = read_file(path);
    if (json.empty()) {
        fprintf(stderr, "[Request] ERROR: cannot read %s\n", path);
        return false;
    }
    return request_parse_from_str(r, json);
}

// Public API
bool request_parse_from_str(AceRequest * r, std::string json) {
    request_init(r);
    std::vector<JsonPair> pairs;
    if (!parse_json_flat(json.c_str(), &pairs)) {
        fprintf(stderr, "[Request] ERROR: malformed JSON\n");
        return false;
    }

    for (const auto & kv : pairs) {
        const std::string & k = kv.key;
        const std::string & v = kv.value;

        // strings
        if      (k == "caption")            r->caption            = v;
        else if (k == "lyrics")             r->lyrics             = v;
        else if (k == "keyscale")           r->keyscale           = v;
        else if (k == "timesignature")      r->timesignature      = v;
        else if (k == "vocal_language")     r->vocal_language     = v;
        else if (k == "task_type")          r->task_type          = v;
        else if (k == "audio_codes")        r->audio_codes        = v;
        else if (k == "lm_negative_prompt") r->lm_negative_prompt = v;

        // ints
        else if (k == "bpm")                r->bpm                = atoi(v.c_str());
        else if (k == "seed")               r->seed               = strtoll(v.c_str(), nullptr, 10);

        // floats
        else if (k == "duration")           r->duration           = (float)atof(v.c_str());
        else if (k == "lm_temperature")     r->lm_temperature     = (float)atof(v.c_str());
        else if (k == "lm_cfg_scale")       r->lm_cfg_scale       = (float)atof(v.c_str());
        else if (k == "lm_top_p")           r->lm_top_p           = (float)atof(v.c_str());
        else if (k == "lm_rep_pen")         r->lm_rep_pen         = (float)atof(v.c_str());
        else if (k == "lm_top_k")           r->lm_top_k           = atoi(v.c_str());
        else if (k == "inference_steps")    r->inference_steps    = atoi(v.c_str());
        else if (k == "guidance_scale")     r->guidance_scale     = (float)atof(v.c_str());
        else if (k == "shift")              r->shift              = (float)atof(v.c_str());
        else if (k == "audio_cover_strength") r->audio_cover_strength = (float)atof(v.c_str());

        else if (k == "codes_temperature")     r->codes_temperature     = (float)atof(v.c_str());
        else if (k == "codes_top_p")           r->codes_top_p           = (float)atof(v.c_str());
        else if (k == "codes_top_k")           r->codes_top_k           = atoi(v.c_str());

        // bools
        else if (k == "thinking")           r->thinking           = (v == "true");
        else if (k == "instrumental")       r->instrumental       = (v == "true");
        // unknown keys: silently ignored (forward compat)
    }

    fprintf(stderr, "[Request] parsed json (%zu fields)\n", pairs.size());
    return true;
}

bool request_write(const AceRequest * r, const char * path) {
    FILE * f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[Request] ERROR: cannot write %s\n", path);
        return false;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"caption\": \"%s\",\n",            json_escape(r->caption).c_str());
    fprintf(f, "  \"lyrics\": \"%s\",\n",             json_escape(r->lyrics).c_str());
    if (r->instrumental)
        fprintf(f, "  \"instrumental\": true,\n");
    fprintf(f, "  \"bpm\": %d,\n",                    r->bpm);
    fprintf(f, "  \"duration\": %.1f,\n",             r->duration);
    fprintf(f, "  \"keyscale\": \"%s\",\n",           json_escape(r->keyscale).c_str());
    fprintf(f, "  \"timesignature\": \"%s\",\n",      json_escape(r->timesignature).c_str());
    fprintf(f, "  \"vocal_language\": \"%s\",\n",     json_escape(r->vocal_language).c_str());
    fprintf(f, "  \"task_type\": \"%s\",\n",          json_escape(r->task_type).c_str());
    fprintf(f, "  \"seed\": %lld,\n",                 (long long)r->seed);
    fprintf(f, "  \"thinking\": %s,\n",               r->thinking ? "true" : "false");
    fprintf(f, "  \"lm_temperature\": %.2f,\n",       r->lm_temperature);
    fprintf(f, "  \"lm_cfg_scale\": %.1f,\n",         r->lm_cfg_scale);
    fprintf(f, "  \"lm_top_p\": %.2f,\n",             r->lm_top_p);
    fprintf(f, "  \"lm_rep_pen\": %.2f,\n",           r->lm_rep_pen);
    fprintf(f, "  \"lm_top_k\": %d,\n",               r->lm_top_k);
    fprintf(f, "  \"codes_temperature\": %.2f,\n",    r->codes_temperature);
    fprintf(f, "  \"codes_top_p\": %.2f,\n",          r->codes_top_p);
    fprintf(f, "  \"codes_top_k\": %d,\n",            r->codes_top_k);
    fprintf(f, "  \"lm_negative_prompt\": \"%s\",\n", json_escape(r->lm_negative_prompt).c_str());
    fprintf(f, "  \"inference_steps\": %d,\n",        r->inference_steps);
    fprintf(f, "  \"guidance_scale\": %.1f,\n",       r->guidance_scale);
    fprintf(f, "  \"shift\": %.1f,\n",                r->shift);
    fprintf(f, "  \"audio_cover_strength\": %.2f,\n", r->audio_cover_strength);
    // audio_codes last (no trailing comma)
    fprintf(f, "  \"audio_codes\": \"%s\"\n",         json_escape(r->audio_codes).c_str());
    fprintf(f, "}\n");

    fclose(f);
    fprintf(stderr, "[Request] wrote %s\n", path);
    return true;
}

void request_dump(const AceRequest * r, FILE * f) {
    fprintf(f, "[Request] task=%s thinking=%s seed=%lld\n",
            r->task_type.c_str(), r->thinking ? "true" : "false", (long long)r->seed);
    fprintf(f, "  caption:    %.60s%s\n",
            r->caption.c_str(), r->caption.size() > 60 ? "..." : "");
    fprintf(f, "  lyrics:     %zu bytes\n", r->lyrics.size());
    fprintf(f, "  bpm=%d dur=%.0f key=%s ts=%s lang=%s\n",
            r->bpm, r->duration, r->keyscale.c_str(),
            r->timesignature.c_str(), r->vocal_language.c_str());
    fprintf(f, "  lm: temp=%.2f cfg=%.1f top_p=%.2f top_k=%d rep_pen=%d\n",
            r->lm_temperature, r->lm_cfg_scale, r->lm_top_p, r->lm_top_k, r->lm_rep_pen);
    fprintf(f, "  codes: temp=%.2f top_p=%.2f top_k=%d \n",
            r->lm_temperature, r->lm_top_p, r->lm_top_k);
    fprintf(f, "  dit: steps=%d guidance=%.1f shift=%.1f\n",
            r->inference_steps, r->guidance_scale, r->shift);
    if (r->audio_cover_strength != 0.5f)
        fprintf(f, "  cover: strength=%.2f\n", r->audio_cover_strength);
    fprintf(f, "  audio_codes: %s\n",
            r->audio_codes.empty() ? "(none)" : "(present)");
}
