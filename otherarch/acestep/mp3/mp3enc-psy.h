#pragma once

// mp3enc-psy.h
// Psychoacoustic model for MP3 encoding.
// Computes masking thresholds per scalefactor band from MDCT coefficients.
//
// Based on ISO 11172-3 Annex D principles:
//   - Absolute threshold of hearing (ATH)
//   - Bark-domain asymmetric spreading function (Schroeder)
//   - Tonal vs noise masking detection (spectral flatness)
//   - Variable masking offset: tonal -14.5 dB, noise -5.5 dB
//
// Constants from ISO 11172-3 Annex D and Zwicker (1961).
// MIT license.

#include <cmath>

// Number of scalefactor bands for long blocks
#define MP3ENC_PSY_SFB_MAX 21

// Absolute threshold of hearing in dB SPL.
// ISO formula: ATH(f) = 3.64*(f/1000)^-0.8 - 6.5*exp(-0.6*(f/1000-3.3)^2) + 1e-3*(f/1000)^4
// Minimum clamped at -20 dB to avoid numerical issues.
static inline float mp3enc_ath_db(float freq_hz) {
    if (freq_hz < 10.0f) {
        freq_hz = 10.0f;
    }
    float fk  = freq_hz * 0.001f;
    float ath = 3.64f * powf(fk, -0.8f) - 6.5f * expf(-0.6f * (fk - 3.3f) * (fk - 3.3f)) + 0.001f * fk * fk * fk * fk;
    if (ath < -20.0f) {
        ath = -20.0f;
    }
    return ath;
}

// Convert frequency in Hz to Bark scale.
// Traunmuller (1990) approximation, accurate to ~0.05 Bark.
static inline float mp3enc_hz_to_bark(float f) {
    if (f < 1.0f) {
        f = 1.0f;
    }
    return 13.0f * atanf(0.00076f * f) + 3.5f * atanf((f / 7500.0f) * (f / 7500.0f));
}

// Schroeder spreading function in dB.
// dz = bark distance from masker to maskee (positive = maskee above masker).
// This models the asymmetric excitation pattern of the basilar membrane:
// steep below the masker (~27 dB/Bark), shallow above (~10-25 dB/Bark).
// From Schroeder, Atal, Hall (1979).
static inline float mp3enc_spreading_db(float dz) {
    float t = dz + 0.474f;
    return 15.81f + 7.5f * t - 17.5f * sqrtf(1.0f + t * t);
}

// Psychoacoustic model state.
struct mp3enc_psy {
    // Output: allowed distortion energy per SFB
    float xmin[MP3ENC_PSY_SFB_MAX];

    // Output: perceptual entropy for this granule/channel (ISO 11172-3 Annex D).
    // Higher PE = more complex signal = needs more bits.
    float pe;

    // Forward masking: previous granule's masking energy (per channel)
    float prev_mask[2][MP3ENC_PSY_SFB_MAX];

    // Pre-echo control: 2 previous granules of spread energy (per channel).
    // Used to prevent masking threshold from rising too fast on transients
    // (ISO 11172-3 Annex D, l3psy.c lines 610-616).
    float nb_1[2][MP3ENC_PSY_SFB_MAX];  // previous granule spread energy
    float nb_2[2][MP3ENC_PSY_SFB_MAX];  // 2 granules ago spread energy

    // Precomputed per-SFB data (set once per sample rate)
    float ath_energy[3][MP3ENC_PSY_SFB_MAX];  // [sr_index][sfb]: ATH in linear power
    float sfb_bark[3][MP3ENC_PSY_SFB_MAX];    // [sr_index][sfb]: center freq in Bark
    bool  ath_valid;

    void init() {
        ath_valid = false;
        pe        = 0.0f;
        memset(xmin, 0, sizeof(xmin));
        memset(prev_mask, 0, sizeof(prev_mask));
        memset(nb_1, 0, sizeof(nb_1));
        memset(nb_2, 0, sizeof(nb_2));
    }

    // Precompute ATH energy and Bark positions per SFB.
    // Must be called once before compute().
    void init_ath(int sr_index, const uint8_t * sfb_table, int sample_rate) {
        // MDCT has 576 lines covering 0 to samplerate/2.
        // Each line represents a frequency bin of width samplerate / (2*576).
        float freq_per_line = (float) sample_rate / (2.0f * 576.0f);

        int pos = 0;
        for (int sfb = 0; sfb < MP3ENC_PSY_SFB_MAX; sfb++) {
            int width = sfb_table[sfb];
            if (width == 0) {
                ath_energy[sr_index][sfb] = 1e-20f;
                sfb_bark[sr_index][sfb]   = 0.0f;
                continue;
            }

            // Center frequency of this SFB
            float center_freq       = ((float) pos + (float) width * 0.5f) * freq_per_line;
            sfb_bark[sr_index][sfb] = mp3enc_hz_to_bark(center_freq);

            // ATH: minimum of all lines in the band (most permissive)
            float ath_min_db = 200.0f;
            for (int j = 0; j < width; j++) {
                float freq = ((float) (pos + j) + 0.5f) * freq_per_line;
                float db   = mp3enc_ath_db(freq);
                if (db < ath_min_db) {
                    ath_min_db = db;
                }
            }

            // Convert dB SPL to linear power, scaled by band width.
            // Reference at 120 dB SPL (tuned empirically). A higher reference
            // makes the ATH floor less dominant relative to the spreading
            // function, so bits are spent on perceptual masking rather than
            // fighting the absolute hearing threshold in quiet passages.
            float ath_linear          = powf(10.0f, (ath_min_db - 120.0f) * 0.1f) * (float) width;
            ath_energy[sr_index][sfb] = ath_linear;

            pos += width;
        }
        ath_valid = true;
    }

    // Compute masking thresholds for one granule/channel.
    //
    // Algorithm:
    //   1. Compute energy per SFB from MDCT coefficients
    //   2. Estimate tonality per SFB (spectral flatness measure)
    //   3. Compute masking offset per SFB based on tonality
    //   4. Apply Bark-domain spreading function
    //   5. Combine spread masking with ATH
    //
    // mdct: 576 MDCT coefficients (after MS stereo if applicable)
    // sfb_table: SFB widths for this sample rate
    // sr_index: sample rate index
    void compute(const float * mdct, const uint8_t * sfb_table, int sr_index, int ch = 0) {
        float energy[MP3ENC_PSY_SFB_MAX];
        float tonality[MP3ENC_PSY_SFB_MAX];

        // Step 1: compute energy per SFB
        int pos = 0;
        for (int sfb = 0; sfb < MP3ENC_PSY_SFB_MAX; sfb++) {
            int   width = sfb_table[sfb];
            float e     = 0.0f;
            for (int j = 0; j < width; j++) {
                float x = mdct[pos + j];
                e += x * x;
            }
            energy[sfb] = e;
            pos += width;
        }

        // Step 2: estimate tonality per SFB using spectral flatness measure (SFM).
        // SFM = geometric_mean(power) / arithmetic_mean(power)
        // SFM = 1.0 for flat noise, SFM -> 0 for a single tone.
        // We use log domain to avoid overflow: log(geometric_mean) = mean(log(power)).
        pos = 0;
        for (int sfb = 0; sfb < MP3ENC_PSY_SFB_MAX; sfb++) {
            int width = sfb_table[sfb];
            if (width == 0 || energy[sfb] < 1e-20f) {
                tonality[sfb] = 0.5f;  // default: assume mixed
                pos += width;
                continue;
            }

            float log_sum  = 0.0f;
            float arith    = 0.0f;
            int   n_active = 0;
            for (int j = 0; j < width; j++) {
                float p = mdct[pos + j] * mdct[pos + j];
                if (p > 1e-20f) {
                    log_sum += logf(p);
                    n_active++;
                }
                arith += p;
            }

            if (n_active < 2) {
                // Single line or silence: treat as tonal
                tonality[sfb] = 0.0f;
            } else {
                float geom_log   = log_sum / (float) n_active;
                float arith_mean = arith / (float) n_active;
                // SFM in log domain: log(geom/arith) = geom_log - log(arith)
                float sfm_log    = geom_log - logf(arith_mean);
                // sfm_log is <= 0. For flat spectrum sfm_log ~ 0, for tonal sfm_log << 0.
                // Map to tonality index alpha in [0,1]:
                //   alpha = min(sfm_log / log(0.01), 1.0)
                // log(0.01) = -4.605; so if sfm_log < -4.6 we consider it fully tonal.
                float alpha      = sfm_log / -4.605f;
                if (alpha < 0.0f) {
                    alpha = 0.0f;
                }
                if (alpha > 1.0f) {
                    alpha = 1.0f;
                }
                tonality[sfb] = alpha;  // 0 = noise, 1 = tonal
            }
            pos += width;
        }

        // Step 3: compute masking offset per SFB.
        // TMN/NMT from ISO 11172-3, relaxed empirically for 128kbps.
        // TMN=13.0 (tonal masking noise), NMT=4.5 (noise masking tone).
        float offset_linear[MP3ENC_PSY_SFB_MAX];
        for (int sfb = 0; sfb < MP3ENC_PSY_SFB_MAX; sfb++) {
            float alpha        = tonality[sfb];
            float offset_db    = alpha * 13.0f + (1.0f - alpha) * 4.5f;
            offset_linear[sfb] = powf(10.0f, -offset_db * 0.1f);
        }

        // Step 4: Bark-domain spreading function.
        // For each target SFB, sum the spread contributions from all source SFBs.
        // The spreading function is asymmetric: steep below, shallow above.
        float spread_energy[MP3ENC_PSY_SFB_MAX];
        for (int j = 0; j < MP3ENC_PSY_SFB_MAX; j++) {
            float sum    = 0.0f;
            float bark_j = sfb_bark[sr_index][j];

            for (int i = 0; i < MP3ENC_PSY_SFB_MAX; i++) {
                if (energy[i] < 1e-20f) {
                    continue;
                }

                float bark_i = sfb_bark[sr_index][i];
                float dz_raw = bark_j - bark_i;

                // Asymmetric spreading (ISO 11172-3 psy model 2, L3para_read).
                // Upward masking (dz > 0): gentle slope, low freqs mask highs well.
                // Downward masking (dz < 0): steep slope, highs mask lows poorly.
                float dz = (dz_raw >= 0.0f) ? dz_raw * 1.35f : dz_raw * 2.7f;

                // Spreading function value in dB
                float sf_db = mp3enc_spreading_db(dz);

                // Only apply if spreading is above -60 dB (optimization)
                if (sf_db < -60.0f) {
                    continue;
                }

                float sf_linear = powf(10.0f, sf_db * 0.1f);
                sum += energy[i] * sf_linear;
            }
            spread_energy[j] = sum;
        }

        // Step 5: combine spreading + offset, then apply pre-echo control.
        // Pre-echo (ISO 11172-3 Annex D): prevent threshold from rising too fast
        // on transients. Clamp current nb by 2x previous and 16x two-back.
        // xmin = max(ath, min(nb, 2*nb_1, 16*nb_2))
        for (int sfb = 0; sfb < MP3ENC_PSY_SFB_MAX; sfb++) {
            float nb = spread_energy[sfb] * offset_linear[sfb];

            // Pre-echo clamp: use history to limit sudden threshold rises
            float clamped = nb;
            float lim1    = 2.0f * nb_1[ch][sfb];
            float lim2    = 16.0f * nb_2[ch][sfb];
            if (lim1 > 0.0f && lim1 < clamped) {
                clamped = lim1;
            }
            if (lim2 > 0.0f && lim2 < clamped) {
                clamped = lim2;
            }

            // Update history (store UN-clamped nb for future reference)
            nb_2[ch][sfb] = nb_1[ch][sfb];
            nb_1[ch][sfb] = nb;

            // ATH floor
            float ath = ath_energy[sr_index][sfb];
            xmin[sfb] = (clamped > ath) ? clamped : ath;
        }

        // Step 6: forward masking (temporal).
        // A loud granule raises the masking threshold for the next granule.
        // At 44.1kHz one granule = 13ms. Forward masking decays ~12dB over 13ms.
        // Decay factor: 10^(-12/10) = ~0.063.
        static const float fwd_decay = 0.063f;
        for (int sfb = 0; sfb < MP3ENC_PSY_SFB_MAX; sfb++) {
            float fwd = prev_mask[ch][sfb] * fwd_decay;
            if (fwd > xmin[sfb]) {
                xmin[sfb] = fwd;
            }
            // Save current mask (spread_energy, not xmin) for next granule
            prev_mask[ch][sfb] = spread_energy[sfb];
        }

        // Step 7: perceptual entropy (ISO 11172-3 Annex D).
        // PE = sum of width * log(energy/threshold) for bands where energy > threshold.
        // Used by the bit reservoir to give more bits to complex granules.
        pe = 0.0f;
        for (int sfb = 0; sfb < MP3ENC_PSY_SFB_MAX; sfb++) {
            if (energy[sfb] > xmin[sfb] && xmin[sfb] > 1e-20f) {
                pe += (float) sfb_table[sfb] * logf(energy[sfb] / xmin[sfb]);
            }
        }
    }
};
