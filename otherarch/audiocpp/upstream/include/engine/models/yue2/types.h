#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::yue2 {

constexpr int32_t kEodToken = 151643;
constexpr int32_t kAbcStartToken = 151847;
constexpr int32_t kAbcEndToken = 151848;
constexpr int32_t kMusicStartToken = 151851;
constexpr int32_t kMusicEndToken = 151852;
constexpr int32_t kCodecOffset = 151853;
constexpr int32_t kCodecSize = 32768;
constexpr int32_t kVocabSize = 184704;
constexpr int64_t kContextTokens = 24576;

struct Yue2ModelConfig {
    int64_t hidden_size = 2048;
    int64_t layers = 28;
    int64_t attention_heads = 16;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t intermediate_size = 6144;
    int64_t vocab_size = kVocabSize;
    int64_t max_position_embeddings = kContextTokens;
    int64_t latent_dim = 64;
    int64_t max_latent_frames = kContextTokens;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    float timestep_shift = 1.0F;
};

struct Yue2VAEConfig {
    int sample_rate = 48000;
    int64_t channels = 2;
    int64_t latent_dim = 64;
    int64_t encoder_latent_dim = 128;
    int64_t downsampling_ratio = 1920;
    int64_t decode_core_frames = 1024;
    int64_t decode_halo_frames = 16;
};

struct Yue2SamplingConfig {
    float temperature = 1.0F;
    float top_p = 0.95F;
    int64_t top_k = 100;
    float repetition_penalty = 1.2F;
    int64_t penalty_window = 50;
    int64_t min_tokens = 200;
    int64_t max_tokens = 9000;
};

struct Yue2GenerationConfig {
    Yue2SamplingConfig abc;
    Yue2SamplingConfig semantic;
    int64_t ode_steps = 8;
    int64_t context = kContextTokens;
};

struct Yue2Config {
    Yue2ModelConfig model;
    Yue2VAEConfig vae;
    Yue2GenerationConfig generation;
};

enum class Yue2CotMode {
    Off,
    Melody,
    Full,
};

enum class Yue2StopAfter {
    Abc,
    Semantic,
    Audio,
};

struct Yue2Request {
    std::string style;
    std::string lyrics;
    Yue2CotMode cot = Yue2CotMode::Full;
    Yue2StopAfter stop_after = Yue2StopAfter::Audio;
    std::string abc;
    std::vector<int32_t> semantic_prefix;
    std::vector<float> nar_noise;
    bool export_semantic = false;
    uint64_t seed = 1234;
    float cfg_scale = -1.0F;
    Yue2GenerationConfig generation;
};

struct Yue2Plan {
    Yue2CotMode cot = Yue2CotMode::Full;
    std::string abc;
    std::vector<int32_t> abc_ids;
    std::vector<int32_t> prefix;
    bool truncated = false;
};

struct Yue2SemanticResult {
    Yue2Plan plan;
    std::vector<int32_t> tokens;
    bool truncated = false;
};

const char * cot_mode_name(Yue2CotMode mode) noexcept;
Yue2CotMode parse_cot_mode(const std::string & value);
const char * cot_instruction(Yue2CotMode mode) noexcept;
const char * stop_after_name(Yue2StopAfter stage) noexcept;
Yue2StopAfter parse_stop_after(const std::string & value);
float request_guidance_scale(const Yue2Request & request) noexcept;
std::string semantic_codes_to_json(const std::vector<int32_t> & codes);

}  // namespace engine::models::yue2
