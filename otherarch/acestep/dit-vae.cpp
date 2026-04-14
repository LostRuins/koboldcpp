// dit.cpp: ACEStep music generation via ggml (dit-vae binary)
//
// Usage: ./dit-vae [options]
// See --help for full option list.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "./dit.h"
#include "./vae.h"
#include "./qwen3.h"
#include "./tokenizer.h"
#include "./cond.h"
#include "./bpe.h"
#include "./debug.h"
#include "./request.h"
#include "./vae-enc.h"
#include "otherarch/utils.h"

// Parse comma-separated codes string into vector
static std::vector<int> parse_codes_string(const std::string & s) {
    std::vector<int> codes;
    if (s.empty()) return codes;
    const char * p = s.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        codes.push_back(atoi(p));
        while (*p && *p != ',') p++;
    }
    return codes;
}

//kcpp stuff
static DiTGGML acestep_dit = {};
static bool acestep_dit_others_loaded = false;
static bool acestep_dit_core_loaded = false;
static bool acestep_vae_enc_loaded = false;
static bool acestep_vae_dec_loaded = false;
static DiTGGMLConfig music_dit_cfg;
static Timer music_dit_timer;
static bool is_turbo = false;
static VAEGGML vae = {};
static VAEEncoder vae_enc = {};
static BPETokenizer music_tok;
static Qwen3GGML music_text_enc = {};
static GGUFModel gf_te = {};
static const void * musice_te_embed_data = nullptr;
static CondGGML music_cond = {};
static std::vector<float> silence_full;  // [15000, 64] f32
static DetokGGML detok = {};

static bool acestep_dit_lowvram = false;
static std::string acestep_music_embd_path = "";
static std::string acestep_music_dit_path = "";
static std::string acestep_music_vae_path = "";

void unload_acestep_dit_core()
{
    if(acestep_dit_core_loaded)
    {
        acestep_dit_core_loaded = false;
        dit_ggml_free(&acestep_dit);
        printf("Unload music diffusion model...\n");
    }
}
void unload_acestep_dit_others()
{
    if(acestep_dit_others_loaded)
    {
        acestep_dit_others_loaded = false;
        gf_close(&gf_te);
        cond_ggml_free(&music_cond);
        detok_ggml_free(&detok);
        qwen3_free(&music_text_enc);
        printf("Unload music tokenizer and conditioner model...\n");
    }
}
void unload_acestep_vae_enc()
{
    if(acestep_vae_enc_loaded)
    {
        acestep_vae_enc_loaded = false;
        vae_enc_free(&vae_enc);
        printf("Unload music VAE enc model...\n");
    }
}
void unload_acestep_vae_dec()
{
    if(acestep_vae_dec_loaded)
    {
        acestep_vae_dec_loaded = false;
        vae_ggml_free(&vae);
        printf("Unload music VAE dec model...\n");
    }
}

bool load_acestep_vae_dec(std::string music_vae_path, bool lowvram)
{
    if(acestep_vae_dec_loaded)
    {
        unload_acestep_vae_dec();
    }

    const char * vae_gguf       = music_vae_path.c_str();
    acestep_dit_lowvram = lowvram;
    acestep_music_vae_path = music_vae_path;

     // Load VAE model (once for all requests)
    music_dit_timer.reset();
    vae_ggml_load(&vae, vae_gguf);
    fprintf(stderr, "[Load] VAE weights: %.1f ms\n", music_dit_timer.ms());

    acestep_vae_dec_loaded = true;
    return true;
}
bool load_acestep_vae_enc(std::string music_vae_path, bool lowvram)
{
    if(acestep_vae_enc_loaded)
    {
        unload_acestep_vae_enc();
    }

    const char * vae_gguf       = music_vae_path.c_str();
    acestep_dit_lowvram = lowvram;
    acestep_music_vae_path = music_vae_path;

    music_dit_timer.reset();
    vae_enc_load(&vae_enc, vae_gguf);
    fprintf(stderr, "[Load] VAE Enc weights: %.1f ms\n", music_dit_timer.ms());

    acestep_vae_enc_loaded = true;
    return true;
}

bool load_acestep_dit(std::string music_embd_path, std::string music_dit_path, bool lowvram)
{
    if(acestep_dit_others_loaded || acestep_dit_core_loaded)
    {
        unload_acestep_dit_core();
        unload_acestep_dit_others();
    }

    acestep_dit_others_loaded = false;
    acestep_dit_core_loaded = false;
    const char * text_enc_gguf = music_embd_path.c_str();
    const char * dit_gguf      = music_dit_path.c_str();

    acestep_dit_lowvram = lowvram;
    acestep_music_embd_path = music_embd_path;
    acestep_music_dit_path = music_dit_path;

    // Load DiT model (once for all requests)
    dit_ggml_init_backend(&acestep_dit);
    fprintf(stderr, "[Load] Backend init: %.1f ms\n", music_dit_timer.ms());

    music_dit_timer.reset();
    if (!dit_ggml_load(&acestep_dit, dit_gguf, music_dit_cfg)) {
        fprintf(stderr, "FATAL: failed to load DiT model\n");
        return false;
    }
    fprintf(stderr, "[Load] DiT weight load: %.1f ms\n", music_dit_timer.ms());

    // Read DiT GGUF metadata + silence_latent tensor (once)
    is_turbo = false;
    {
        GGUFModel gf = {};
        if (gf_load(&gf, dit_gguf)) {
            is_turbo = gf_get_bool(gf, "acestep.is_turbo");
            const void * sl_data = gf_get_data(gf, "silence_latent");
            if (sl_data) {
                silence_full.resize(15000 * 64);
                memcpy(silence_full.data(), sl_data, 15000 * 64 * sizeof(float));
                fprintf(stderr, "[Load] silence_latent: [15000, 64] from GGUF\n");
            } else {
                fprintf(stderr, "FATAL: silence_latent tensor not found in %s\n", dit_gguf);
                return false;
            }
            gf_close(&gf);
        } else {
            fprintf(stderr, "FATAL: cannot reopen %s for metadata\n", dit_gguf);
            return false;
        }
    }

    music_dit_timer.reset();
    if (!load_bpe_from_gguf(&music_tok, text_enc_gguf)) {
        fprintf(stderr, "FATAL: failed to load music tokenizer from %s\n", text_enc_gguf);
        return false;
    }
    fprintf(stderr, "[Load] BPE tokenizer: %.1f ms\n", music_dit_timer.ms());

    // Text encoder forward (caption only)
    music_dit_timer.reset();
    qwen3_init_backend(&music_text_enc);
    if (!qwen3_load_text_encoder(&music_text_enc, text_enc_gguf)) {
        fprintf(stderr, "FATAL: failed to load text encoder\n");
        return false;
    }
    fprintf(stderr, "[Load] TextEncoder: %.1f ms\n", music_dit_timer.ms());

    if (!gf_load(&gf_te, text_enc_gguf)) {
        fprintf(stderr, "FATAL: cannot reopen text encoder GGUF for lyric embed\n");
        return false;
    }
    musice_te_embed_data = gf_get_data(gf_te, "embed_tokens.weight");
    if (!musice_te_embed_data) {
        fprintf(stderr, "FATAL: embed_tokens.weight not found\n");
        return false;
    }

    // Condition encoder forward
    music_dit_timer.reset();
    cond_ggml_init_backend(&music_cond);
    if (!cond_ggml_load(&music_cond, dit_gguf)) {
        fprintf(stderr, "FATAL: failed to load condition encoder\n");
        return false;
    }
    fprintf(stderr, "[Load] ConditionEncoder: %.1f ms\n", music_dit_timer.ms());

    music_dit_timer.reset();

    if (!detok_ggml_load(&detok, dit_gguf, acestep_dit.backend, acestep_dit.cpu_backend)) {
        fprintf(stderr, "FATAL: failed to load detokenizer\n");
        return false;
    }
    fprintf(stderr, "[Load] Detokenizer: %.1f ms\n", music_dit_timer.ms());

    acestep_dit_others_loaded = true;
    acestep_dit_core_loaded = true;
    return true;
}

std::string acestep_generate_audio(const music_generation_inputs inputs)
{
    if(acestep_music_dit_path=="" || acestep_music_vae_path=="" || acestep_music_embd_path=="")
    {
        printf("\nERROR: Acestep DiT, VAE or Embd path is empty!\n");
        return "";
    }

    const int FRAMES_PER_SECOND = 25;
    int Oc = music_dit_cfg.out_channels;          // 64
    int ctx_ch = music_dit_cfg.in_channels - Oc;  // 128
    int batch_n                = 1;
    int vae_chunk              = 256;
    int vae_overlap            = 64;

    // Parse request JSON
    AceRequest req;
    std::string injson =  inputs.input_json;
    request_init(&req);
    if (!request_parse_from_str(&req, injson)) {
        fprintf(stderr, "ERROR: failed to parse music gen request\n");
        return "";
    }
    if (req.caption.empty()) {
        req.caption = "An interesting song";
    }
    req.thinking = false;
    req.inference_steps = (req.inference_steps>100?100:req.inference_steps); //clamp to 100
    req.duration = (req.duration>420?420:req.duration); //clamp to 7 min

    // Cover mode: load VAE encoder and encode source audio
    bool have_cover = false;
    std::vector<float> cover_latents;  // [T_cover, 64] time-major
    int T_cover = 0;
    std::string custom_reference_audio_str = inputs.music_reference_audio_data;
    if (custom_reference_audio_str!="" && req.audio_cover_strength>0)
    {
        if(!acestep_vae_enc_loaded)
        {
            printf("\nRuntime reload Music VAE enc model...\n");
            load_acestep_vae_enc(acestep_music_vae_path,acestep_dit_lowvram);
        }

        music_dit_timer.reset();
        int T_audio = 0;
        int wav_sr = 48000;

        std::vector<uint8_t> media_data_buffer = kcpp_base64_decode(custom_reference_audio_str);
        std::vector<float> pcm;
        bool ok = kcpp_decode_audio_to_f32_stereo_48k(media_data_buffer.data(), media_data_buffer.size(), pcm, T_audio);
        if (!ok) {
            printf("\nError: Cannot decode audio\n");
            return "";
        }
        float *wav_data = pcm.data();

        fprintf(stderr, "[Cover] Source audio: %.2fs, SR:%d, WavDataSize:%zu\n", (float)T_audio / (float)(wav_sr > 0 ? wav_sr : 48000),wav_sr,T_audio);
        int max_T_lat = (T_audio / 1920) + 64;
        cover_latents.resize(max_T_lat * 64);
        T_cover = vae_enc_encode_tiled(&vae_enc, wav_data, T_audio,
                                        cover_latents.data(), max_T_lat,
                                        vae_chunk, vae_overlap);
        if (T_cover < 0) {
            fprintf(stderr, "FATAL: VAE encode of src_audio failed\n");
            return "";
        }
        cover_latents.resize(T_cover * 64);
        fprintf(stderr, "[Cover] Encoded: T_cover=%d (%.2fs), %.1f ms\n",
                T_cover, (float)T_cover * 1920.0f / 48000.0f, music_dit_timer.ms());
        have_cover = true;

        if(acestep_dit_lowvram)
        {
            unload_acestep_vae_enc();
        }
    }


    if(!acestep_dit_others_loaded && !acestep_dit_core_loaded)
    {
        printf("\nRuntime reload Music DiT model...\n");
        bool ok = load_acestep_dit(acestep_music_embd_path, acestep_music_dit_path, acestep_dit_lowvram);
        if(!ok)
        {
            printf("\nERROR: Acestep DiT load fail\n");
            return "";
        }
    }

    // Extract params
    const char * caption  = req.caption.c_str();
    const char * lyrics   = req.lyrics.empty() ? "[Instrumental]" : req.lyrics.c_str();
    char bpm_str[16] = "N/A";
    if (req.bpm > 0) snprintf(bpm_str, sizeof(bpm_str), "%d", req.bpm);
    const char * bpm      = bpm_str;
    const char * keyscale = req.keyscale.empty() ? "N/A" : req.keyscale.c_str();
    const char * timesig  = req.timesignature.empty() ? "N/A" : req.timesignature.c_str();
    const char * language = req.vocal_language.empty() ? "en" : req.vocal_language.c_str();
    float duration        = req.duration > 0 ? req.duration : 120.0f;
    long long seed        = req.seed;
    int num_steps         = req.inference_steps > 0 ? req.inference_steps : 8;
    float guidance_scale  = req.guidance_scale > 0 ? req.guidance_scale : 1.0f;
    float shift           = req.shift > 0 ? req.shift : 1.0f;

    if (is_turbo && guidance_scale > 1.0f) {
        fprintf(stderr, "[Pipeline] WARNING: turbo model, forcing guidance_scale=1.0 (was %.1f)\n",
                guidance_scale);
        guidance_scale = 1.0f;
    }

    if (seed <= 0 || seed==0xFFFFFFFF)
    {
        seed = (((uint32_t)time(NULL)) % 1000000u);
    }

    // Parse audio codes from request
    std::vector<int> codes_vec = parse_codes_string(req.audio_codes);
    if (!codes_vec.empty())
        fprintf(stderr, "[Pipeline] %zu audio codes (%.1fs @ 5Hz)\n",
                codes_vec.size(), (float)codes_vec.size() / 5.0f);

    // Build schedule: t_i = shift * t / (1 + (shift-1)*t) where t = 1 - i/steps
    std::vector<float> schedule(num_steps);
    for (int i = 0; i < num_steps; i++) {
        float t = 1.0f - (float)i / (float)num_steps;
        schedule[i] = shift * t / (1.0f + (shift - 1.0f) * t);
    }

    // T = number of 25Hz latent frames for DiT
    // Cover: from source audio. Codes: from code count. Else: from duration.
    int T;
    if (have_cover) {
        T = T_cover;
        // duration in metas must match actual source length, not JSON default
        duration = (float)T_cover / (float)FRAMES_PER_SECOND;
    } else if (!codes_vec.empty()) {
        T = (int)codes_vec.size() * 5;
    } else {
        T = (int)(duration * FRAMES_PER_SECOND);
    }
    T = ((T + music_dit_cfg.patch_size - 1) / music_dit_cfg.patch_size) * music_dit_cfg.patch_size;
    int S = T / music_dit_cfg.patch_size;
    int enc_S = 0;

    fprintf(stderr, "[Pipeline] T=%d, S=%d\n", T, S);
    fprintf(stderr, "[Pipeline] seed=%lld, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n",
                seed, num_steps, guidance_scale, shift, duration);

    if (T > 15000) {
        fprintf(stderr, "ERROR: T=%d exceeds silence_latent max 15000, skipping\n", T);
        return "";
    }

    // Text encoding
    // 1. Load BPE tokenizer
    music_dit_timer.reset();

    // 2. Build formatted prompts
    std::string instruction = "Generate audio semantic tokens based on the given conditions:";
    if(have_cover)
    {
        instruction = "Fill the audio semantic mask based on the given conditions:";
    }
    char metas[512];
    snprintf(metas, sizeof(metas),
                "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n",
                bpm, timesig, keyscale, (int)duration);
    std::string text_str = std::string("# Instruction\n")
        + instruction + "\n\n"
        + "# Caption\n" + caption + "\n\n"
        + "# Metas\n" + metas + "<|endoftext|>\n";

    bool instrumental = (strcmp(lyrics, "[Instrumental]") == 0 || strcmp(lyrics, "[instrumental]") == 0);
    std::string lyric_str = std::string("# Languages\n") + language + "\n\n# Lyric\n"
        + (instrumental ? "[Instrumental]" : lyrics) + "<|endoftext|>";

    // 3. Tokenize
    auto text_ids  = bpe_encode(&music_tok, text_str.c_str(), true);
    auto lyric_ids = bpe_encode(&music_tok, lyric_str.c_str(), true);
    int S_text  = (int)text_ids.size();
    int S_lyric = (int)lyric_ids.size();
    fprintf(stderr, "[Pipeline] caption: %d tokens, lyrics: %d tokens\n", S_text, S_lyric);

    int H_text = music_text_enc.cfg.hidden_size;  // 1024
    std::vector<float> text_hidden(H_text * S_text);

    music_dit_timer.reset();
    qwen3_forward(&music_text_enc, text_ids.data(), S_text, text_hidden.data());
    fprintf(stderr, "[Encode] TextEncoder (%d tokens): %.1f ms\n", S_text, music_dit_timer.ms());

    // 5. Lyric embedding (CPU vocab lookup from text encoder embed table)
    music_dit_timer.reset();
    std::vector<float> lyric_embed(H_text * S_lyric);
    qwen3_cpu_embed_lookup(musice_te_embed_data, H_text,
                                lyric_ids.data(), S_lyric,
                                lyric_embed.data());

    fprintf(stderr, "[Encode] Lyric vocab lookup (%d tokens): %.1f ms\n", S_lyric, music_dit_timer.ms());

    // Silence feats for timbre input: first 750 frames (30s @ 25Hz)
    const int S_ref = 750;
    std::vector<float> silence_feats(S_ref * 64);
    memcpy(silence_feats.data(), silence_full.data(), S_ref * 64 * sizeof(float));

    music_dit_timer.reset();
    std::vector<float> enc_hidden;
    cond_ggml_forward(&music_cond, text_hidden.data(), S_text,
                        lyric_embed.data(), S_lyric,
                        silence_feats.data(), S_ref,
                        enc_hidden, &enc_S);
    fprintf(stderr, "[Encode] ConditionEncoder: %.1f ms, enc_S=%d\n", music_dit_timer.ms(), enc_S);

    // Context building
    // Silence latent for this T
    // std::vector<float> silence(Oc * T);
    // memcpy(silence.data(), silence_full.data(), (size_t)(Oc * T) * sizeof(float));

    // Decode audio codes if provided (passthrough mode only, NOT cover)
    int decoded_T = 0;
    std::vector<float> decoded_latents;
    if (!have_cover && !codes_vec.empty()) {
        int T_5Hz = (int)codes_vec.size();
        int T_25Hz_codes = T_5Hz * 5;
        decoded_latents.resize(T_25Hz_codes * Oc);

        music_dit_timer.reset();
        int ret = detok_ggml_decode(&detok, codes_vec.data(), T_5Hz, decoded_latents.data());
        if (ret < 0) {
            fprintf(stderr, "FATAL: music detokenizer decode failed\n");
            return "";
        }
        fprintf(stderr, "[Context] Detokenizer: %.1f ms\n", music_dit_timer.ms());

        decoded_T = T_25Hz_codes < T ? T_25Hz_codes : T;
    }

    // Build context: [T, ctx_ch] = src_latents[64] + mask_ones[64]
    // Cover: VAE latents directly (matching Python: is_covers=False, raw latents as context)
    // Passthrough: detokenized FSQ codes + silence padding
    // Text2music: silence only
    std::vector<float> context_single(T * ctx_ch);
    if (have_cover) {
        for (int t = 0; t < T; t++) {
            const float * src = (t < T_cover)
                ? cover_latents.data() + t * Oc
                : silence_full.data() + t * Oc;
            for (int c = 0; c < Oc; c++)
                context_single[t * ctx_ch + c] = src[c];
            for (int c = 0; c < Oc; c++)
                context_single[t * ctx_ch + Oc + c] = 1.0f;
        }
    } else {
        for (int t = 0; t < T; t++) {
            const float * src = (t < decoded_T)
                ? decoded_latents.data() + t * Oc
                : silence_full.data() + (t - decoded_T) * Oc;
            for (int c = 0; c < Oc; c++)
                context_single[t * ctx_ch + c] = src[c];
            for (int c = 0; c < Oc; c++)
                context_single[t * ctx_ch + Oc + c] = 1.0f;
        }
    }

    // Replicate context for N batch samples (all identical)
    std::vector<float> context(batch_n * T * ctx_ch);
    for (int b = 0; b < batch_n; b++)
    {
        memcpy(context.data() + b * T * ctx_ch, context_single.data(), T * ctx_ch * sizeof(float));
    }

    // Cover mode: build silence context for audio_cover_strength switching
    // When step >= cover_steps, DiT switches from cover context to silence context
    std::vector<float> context_silence;
    int cover_steps = -1;
    if (have_cover) {
        float cover_strength = req.audio_cover_strength;
        if (cover_strength < 1.0f) {
            // Build silence context: all frames use silence_latent
            std::vector<float> silence_single(T * ctx_ch);
            for (int t = 0; t < T; t++) {
                const float * src = silence_full.data() + t * Oc;
                for (int c = 0; c < Oc; c++)
                    silence_single[t * ctx_ch + c] = src[c];
                for (int c = 0; c < Oc; c++)
                    silence_single[t * ctx_ch + Oc + c] = 1.0f;
            }
            context_silence.resize(batch_n * T * ctx_ch);
            for (int b = 0; b < batch_n; b++)
                memcpy(context_silence.data() + b * T * ctx_ch,
                        silence_single.data(), T * ctx_ch * sizeof(float));
            cover_steps = (int)((float)num_steps * cover_strength);
            fprintf(stderr, "[Cover] audio_cover_strength=%.2f -> switch at step %d/%d\n",
                    cover_strength, cover_steps, num_steps);
        }
    }

    // Generate N noise samples
    std::vector<float> noise(batch_n * Oc * T);

    {
        // Generate N noise samples with seeds: seed, seed+1, ..., seed+N-1
        for (int b = 0; b < batch_n; b++) {
            std::mt19937 rng((uint32_t)(seed + b));
            std::normal_distribution<float> normal(0.0f, 1.0f);
            float * dst = noise.data() + b * Oc * T;
            for (int i = 0; i < Oc * T; i++)
                dst[i] = normal(rng);
            fprintf(stderr, "[Context Batch%d] noise seed=%lld\n", b, seed + b);
        }
    }

    // DiT Generate
    std::vector<float> output(batch_n * Oc * T);

    fprintf(stderr, "[DiT] Starting: T=%d, S=%d, enc_S=%d, steps=%d, batch=%d%s\n",
            T, S, enc_S, num_steps, batch_n,
            have_cover ? " (cover)" : "");

    music_dit_timer.reset();
    dit_ggml_generate(&acestep_dit, noise.data(), context.data(), enc_hidden.data(),
                        enc_S, T, batch_n, num_steps, schedule.data(), output.data(),
                        guidance_scale, nullptr,
                        context_silence.empty() ? nullptr : context_silence.data(),
                        cover_steps);
    fprintf(stderr, "[DiT] Total generation: %.1f ms (%.1f ms/sample)\n",
            music_dit_timer.ms(), music_dit_timer.ms() / batch_n);

    // VAE Decode + Write WAVs
    int T_latent = T;
    int T_audio_max = T_latent * 1920;
    std::vector<float> audio(2 * T_audio_max);

    int b = 0;
    float * dit_out = output.data() + b * Oc * T;

    if(acestep_dit_lowvram)
    {
        unload_acestep_dit_core();
    }

    if(!acestep_vae_dec_loaded)
    {
        printf("\nRuntime reload Music VAE dec model...\n");
        load_acestep_vae_dec(acestep_music_vae_path,acestep_dit_lowvram);
    }

    music_dit_timer.reset();
    int T_audio = vae_ggml_decode_tiled(&vae, dit_out, T_latent, audio.data(), T_audio_max, vae_chunk, vae_overlap);
    if (T_audio < 0) {
        fprintf(stderr, "[VAE] ERROR: decode failed\n");
        return "";
    }
    fprintf(stderr, "[VAE] Decode: %.1f ms\n", music_dit_timer.ms());

    // Peak normalization to -1.0 dB
    {
        float peak = 0.0f;
        int n_samples = 2 * T_audio;
        for (int i = 0; i < n_samples; i++) {
            float a = audio[i] < 0 ? -audio[i] : audio[i];
            if (a > peak) peak = a;
        }
        if (peak > 1e-6f) {
            const float target_amp = powf(10.0f, -1.0f / 20.0f);
            float gain = target_amp / peak;
            for (int i = 0; i < n_samples; i++)
                audio[i] *= gain;
        }
    }

    // output wav
    float muslen = (float)T_audio / 48000.0f;
    std::string finalb64;
    if (inputs.use_mp3) {
        fprintf(stderr, "[Save Audio] Converting to Mp3 (CPU based, may be slow, use .wav if too slow)...\n",muslen);
        finalb64 = save_stereo_mp3_base64(audio, T_audio, 48000);
    } else if (inputs.stereo) {
         fprintf(stderr, "[Save Audio] Save as Stereo WAV...\n",muslen);
        finalb64 = save_stereo_wav16_base64(audio, T_audio, 48000);
    } else {
         fprintf(stderr, "[Save Audio] Save as Mono WAV...\n",muslen);
        std::vector<float> mono          = mix_planar_stereo_to_mono(audio.data(), T_audio);
        std::vector<float> resampled_buf = resample_wav(1, mono, 48000, 32000);
        finalb64                         = save_wav16_base64(resampled_buf, 32000);
    }

    if(acestep_dit_lowvram)
    {
        unload_acestep_dit_others();
        unload_acestep_vae_dec();
    }

    fprintf(stderr, "[Request Done: Music Length %.2fs]\n",muslen);
    return finalb64;
}