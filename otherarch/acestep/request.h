#pragma once
//
// request.h - AceStep generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.
// Aligned with Python GenerationParams (inference.py:39) and API /release_task.
//

#include <string>
#include <cstdio>

struct AceRequest {
    // text content
    std::string caption;            // ""
    std::string lyrics;             // ""
    bool        instrumental;       // false

    // metadata (user-provided or LLM-enriched)
    int         bpm;                // 0 = unset
    float       duration;           // -1 = unset
    std::string keyscale;           // "" = unset
    std::string timesignature;      // "" = unset
    std::string vocal_language;     // "unknown"

    // generation
    std::string task_type;          // "text2music"
    int64_t     seed;               // -1 = random

    // LM control
    bool        thinking;           // true
    float       lm_temperature;     // 0.85
    float       lm_cfg_scale;       // 2.0
    float       lm_top_p;           // 0.9
    int         lm_top_k;           // 0 = disabled (matches Python None)
    float       lm_rep_pen;         // 1.03
    std::string lm_negative_prompt; // "NO USER INPUT"

    int         codes_top_k;
    float       codes_top_p;
    float       codes_temperature;

    // codes (Python-compatible string: "3101,11837,27514,...")
    // empty = text2music (silence context), non-empty = cover mode
    std::string audio_codes;        // ""

    // DiT control (Python: inference_steps, guidance_scale, shift)
    int         inference_steps;    // 8
    float       guidance_scale;     // 7.0
    float       shift;              // 1.0

    // cover mode (active when --src-audio is provided on CLI)
    float       audio_cover_strength; // 0.5 (0-1, fraction of DiT steps using source context)
};

// Initialize all fields to defaults (matches Python GenerationParams defaults)
void request_init(AceRequest * r);

// Parse JSON file into struct. Missing fields keep their defaults.
// Returns false on file error or malformed JSON.
bool request_parse(AceRequest * r, const char * path);

bool request_parse_from_str(AceRequest * r, std::string json);

// Write struct to JSON file (overwrites). Returns false on file error.
bool request_write(const AceRequest * r, const char * path);

// Dump human-readable summary to stream (debug)
void request_dump(const AceRequest * r, FILE * f);
