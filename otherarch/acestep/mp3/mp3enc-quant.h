#pragma once
// mp3enc-quant.h
// Quantization: inner loop (global_gain search) + outer loop (scalefactor iteration).
// ISO 11172-3 Annex C, encoding process.
// Part of mp3enc. MIT license.

#include <cmath>
#include <cstring>

// Quantize one MDCT coefficient using the MP3 power law quantizer.
// xr = input MDCT value (float)
// istep = 2^(-3/16 * (global_gain - 210))
// Returns quantized integer (always >= 0; sign stored separately).
static inline int mp3enc_quantize_value(float xr, float istep) {
    float ax  = fabsf(xr);
    // ix = nint(ax^0.75 * istep)
    float val = sqrtf(ax * sqrtf(ax)) * istep;
    if (val > 8191.0f) {
        return 8191;
    }
    int ix = (int) (val + 0.36f);  // rounding bias for better SNR
    return ix;
}

// Quantize 576 MDCT coefficients with per-band scalefactors.
// The scalefactor amplifies each band's coefficients before quantization,
// giving finer resolution to bands that need it.
//
// Decoder (minimp3) dequantization:
//   scf_shift = scalefac_scale + 1
//   band_gain = 2^(-(sf << scf_shift) / 4)
//   which gives 2^(-sf/2) for scalefac_scale=0, 2^(-sf) for scalefac_scale=1
//
// Encoder compensates: sfb_amp = 2^(ss * sf) where ss = 0.5 or 1.0
static void mp3enc_quantize_sfb(const float *   xr,
                                int *           ix,
                                int             global_gain,
                                const int *     scalefac,
                                int             scalefac_scale,
                                int             preflag,
                                const uint8_t * sfb_table) {
    float istep = powf(2.0f, -0.1875f * (float) (global_gain - 210));
    float ss    = scalefac_scale ? 1.0f : 0.5f;

    int pos = 0;
    for (int sfb = 0; sfb_table[sfb] != 0 && pos < 576; sfb++) {
        int width = sfb_table[sfb];

        int   sf      = scalefac[sfb] + (preflag ? mp3enc_pretab[sfb] : 0);
        float sfb_amp = (sf > 0) ? powf(2.0f, ss * (float) sf) : 1.0f;

        for (int j = 0; j < width && pos < 576; j++, pos++) {
            float xr_adj = fabsf(xr[pos]) * sfb_amp;
            float val    = sqrtf(xr_adj * sqrtf(xr_adj)) * istep;
            int   q;
            if (val > 8191.0f) {
                q = 8191;
            } else {
                q = (int) (val + 0.36f);
            }
            ix[pos] = (xr[pos] >= 0.0f) ? q : -q;
        }
    }
    while (pos < 576) {
        ix[pos++] = 0;
    }
}

// Simple quantize without scalefactors (Phase 1 compatible).
// Kept for the initial global_gain search before outer loop kicks in.
static void mp3enc_quantize(const float * xr, int * ix, int global_gain) {
    float istep = powf(2.0f, -0.1875f * (float) (global_gain - 210));
    for (int i = 0; i < 576; i++) {
        int q = mp3enc_quantize_value(xr[i], istep);
        ix[i] = (xr[i] >= 0.0f) ? q : -q;
    }
}

// Compute quantization noise energy per SFB.
// Noise = sum((xr[i] - dequant(ix[i]))^2) for each band.
// Dequant matches minimp3: xr = |ix|^(4/3) * 2^((gg-210)/4) * 2^(-ss*sf)
// where ss = 0.5 (scalefac_scale=0) or 1.0 (scalefac_scale=1).
static void mp3enc_calc_noise(const float *   xr,
                              const int *     ix,
                              int             global_gain,
                              const int *     scalefac,
                              int             scalefac_scale,
                              int             preflag,
                              const uint8_t * sfb_table,
                              float *         noise) {
    float step = powf(2.0f, 0.25f * (float) (global_gain - 210));
    float ss   = scalefac_scale ? 1.0f : 0.5f;

    int pos = 0;
    for (int sfb = 0; sfb_table[sfb] != 0 && pos < 576; sfb++) {
        int   width    = sfb_table[sfb];
        int   sf       = scalefac[sfb] + (preflag ? mp3enc_pretab[sfb] : 0);
        float sfb_gain = (sf > 0) ? powf(2.0f, -ss * (float) sf) : 1.0f;

        float n = 0.0f;
        for (int j = 0; j < width && pos < 576; j++, pos++) {
            // Dequantize
            int   aix     = abs(ix[pos]);
            float dequant = (float) aix;
            // |ix|^(4/3): use pow for accuracy
            if (aix > 0) {
                dequant = powf((float) aix, 4.0f / 3.0f) * step * sfb_gain;
            } else {
                dequant = 0.0f;
            }
            if (ix[pos] < 0) {
                dequant = -dequant;
            }
            float diff = xr[pos] - dequant;
            n += diff * diff;
        }
        noise[sfb] = n;
    }
}

// Count total Huffman bits for 576 quantized values.
// Also fills out granule info: big_values, table_select, count1, etc.
// Returns total bits for Huffman data (not including scalefactors).
static int mp3enc_count_bits(const int * ix, mp3enc_granule_info & gi, const uint8_t * sfb_table, int sr_index) {
    (void) sr_index;

    // Find the three regions: big_values, count1, rzero
    int rzero_pairs = mp3enc_count_rzero(ix, 576);
    int nz_end      = 576 - rzero_pairs * 2;

    // count1: quadruples with |val| <= 1, scanning from end of nonzero region
    int c1_start = nz_end;
    int c1_count = 0;
    {
        int i = nz_end - 4;
        while (i >= 0 && abs(ix[i]) <= 1 && abs(ix[i + 1]) <= 1 && abs(ix[i + 2]) <= 1 && abs(ix[i + 3]) <= 1) {
            c1_start = i;
            c1_count++;
            i -= 4;
        }
    }

    gi.big_values = c1_start / 2;

    int bv_end = c1_start;  // end of big_values region (pair aligned)

    // Region boundaries from SFB table.
    int region_end[3] = { 0, 0, bv_end };

    if (gi.block_type == 0) {
        // Try a few region0_count values and pick the best.
        int total_sfb   = 0;
        int sfb_acc[22] = {};
        {
            int acc = 0;
            for (int sfb = 0; sfb < 22; sfb++) {
                acc += sfb_table[sfb];
                sfb_acc[sfb] = acc;
                if (acc <= bv_end) {
                    total_sfb = sfb + 1;
                }
            }
        }

        int best_r0 = 7, best_r1 = 0, best_rbits = 999999;
        for (int r0t = 5; r0t < 11 && r0t < total_sfb; r0t++) {
            int r1t = total_sfb - r0t - 1;
            if (r1t < 0) {
                r1t = 0;
            }
            if (r1t > 7) {
                r1t = 7;
            }
            int re0  = (sfb_acc[r0t] < bv_end) ? sfb_acc[r0t] : bv_end;
            int sfb1 = r0t + r1t + 1;
            if (sfb1 > 21) {
                sfb1 = 21;
            }
            int re1 = (sfb_acc[sfb1] < bv_end) ? sfb_acc[sfb1] : bv_end;

            int t0    = mp3enc_choose_table(ix, 0, re0 / 2);
            int t1    = mp3enc_choose_table(ix, re0, (re1 - re0) / 2);
            int t2    = mp3enc_choose_table(ix, re1, (bv_end - re1) / 2);
            int rbits = 0;
            for (int i = 0; i < re0; i += 2) {
                rbits += mp3enc_pair_bits(t0, ix[i], ix[i + 1]);
            }
            for (int i = re0; i < re1; i += 2) {
                rbits += mp3enc_pair_bits(t1, ix[i], ix[i + 1]);
            }
            for (int i = re1; i < bv_end; i += 2) {
                rbits += mp3enc_pair_bits(t2, ix[i], ix[i + 1]);
            }
            if (rbits < best_rbits) {
                best_rbits = rbits;
                best_r0    = r0t;
                best_r1    = r1t;
            }
        }
        gi.region0_count = best_r0;
        gi.region1_count = best_r1;

        // Compute region end positions
        {
            int acc = 0;
            for (int sfb = 0; sfb <= gi.region0_count; sfb++) {
                acc += sfb_table[sfb];
            }
            region_end[0] = (acc < bv_end) ? acc : bv_end;
        }
        {
            int acc = 0;
            for (int sfb = 0; sfb <= gi.region0_count + gi.region1_count + 1; sfb++) {
                acc += sfb_table[sfb];
            }
            region_end[1] = (acc < bv_end) ? acc : bv_end;
        }
        region_end[2] = bv_end;
    }

    // Choose Huffman tables for each region (3 regions for long blocks)
    int n_regions  = 3;
    int total_bits = 0;
    int prev_end   = 0;

    for (int r = 0; r < n_regions; r++) {
        int pairs          = (region_end[r] - prev_end) / 2;
        gi.table_select[r] = mp3enc_choose_table(ix, prev_end, pairs);
        for (int p = 0; p < pairs; p++) {
            int i = prev_end + p * 2;
            total_bits += mp3enc_pair_bits(gi.table_select[r], ix[i], ix[i + 1]);
        }
        prev_end = region_end[r];
    }

    // Count1 region: try both tables, pick the smaller
    int c1_bits_a = 0, c1_bits_b = 0;
    for (int q = 0; q < c1_count; q++) {
        int i = c1_start + q * 4;
        int v = abs(ix[i]), w = abs(ix[i + 1]), x = abs(ix[i + 2]), y = abs(ix[i + 3]);
        int idx   = v * 8 + w * 4 + x * 2 + y;
        int signs = (v > 0) + (w > 0) + (x > 0) + (y > 0);
        c1_bits_a += mp3enc_count1a_len[idx] + signs;
        c1_bits_b += mp3enc_count1b_len[idx] + signs;
    }

    if (c1_bits_a <= c1_bits_b) {
        gi.count1table_select = 0;
        total_bits += c1_bits_a;
    } else {
        gi.count1table_select = 1;
        total_bits += c1_bits_b;
    }

    return total_bits;
}

// Compute part2_length: number of bits for scalefactors.
// Depends on scalefac_compress and which bands are transmitted.
static int mp3enc_part2_length(const mp3enc_granule_info & gi, int gr, const int scfsi[4]) {
    int slen1 = mp3enc_slen[0][gi.scalefac_compress];
    int slen2 = mp3enc_slen[1][gi.scalefac_compress];
    int bits  = 0;

    // Long blocks: 4 scfsi groups
    static const int band_start[4] = { 0, 6, 11, 16 };
    static const int band_end[4]   = { 6, 11, 16, 21 };

    for (int b = 0; b < 4; b++) {
        if (gr == 0 || scfsi[b] == 0) {
            int slen  = (b < 2) ? slen1 : slen2;
            int count = band_end[b] - band_start[b];
            bits += count * slen;
        }
    }
    return bits;
}

// Find the best scalefac_compress for the current scalefactors.
// Returns the compress index (0..15) that can represent all scalefactors
// with the fewest total bits.
static int mp3enc_best_scalefac_compress(const int * scalefac_l) {
    // Find max scalefactor in each group
    int max1 = 0;  // bands 0..10 (slen1)
    int max2 = 0;  // bands 11..20 (slen2)
    for (int sfb = 0; sfb < 11; sfb++) {
        if (scalefac_l[sfb] > max1) {
            max1 = scalefac_l[sfb];
        }
    }
    for (int sfb = 11; sfb < 21; sfb++) {
        if (scalefac_l[sfb] > max2) {
            max2 = scalefac_l[sfb];
        }
    }

    // Try all 16 compress values, pick the one with fewest bits
    // that can represent the max values
    int best_compress = 0;
    int best_bits     = 999;

    for (int c = 0; c < 16; c++) {
        int s1       = mp3enc_slen[0][c];
        int s2       = mp3enc_slen[1][c];
        int max_val1 = (s1 > 0) ? ((1 << s1) - 1) : 0;
        int max_val2 = (s2 > 0) ? ((1 << s2) - 1) : 0;

        // Can this compress value represent our scalefactors?
        if (max1 > max_val1 || max2 > max_val2) {
            continue;
        }

        // Total bits for scalefactors (granule 0, no scfsi)
        int bits = 11 * s1 + 10 * s2;
        if (bits < best_bits) {
            best_bits     = bits;
            best_compress = c;
        }
    }
    return best_compress;
}

// Inner loop: find minimum global_gain where Huffman bits fit the budget.
// Bit count is monotonically decreasing with global_gain (higher gain = coarser
// quantization = fewer bits). We want the smallest gain where bits <= budget.
//
// When hint_gain >= 0 (from a previous inner_loop call in the same outer loop),
// the optimal gain is typically within a few steps. We scan linearly from the
// hint instead of doing a full binary search on [0, 255]. This cuts the typical
// iteration count from 8 to 3-4.
//
// scalefac: per-band scalefactors (NULL for initial call before outer loop).
// hint_gain: previous global_gain from last inner_loop call, or -1 for full search.
static int mp3enc_inner_loop(const float *         xr,
                             int *                 ix,
                             mp3enc_granule_info & gi,
                             int                   available_bits,
                             const uint8_t *       sfb_table,
                             int                   sr_index,
                             const int *           scalefac  = nullptr,
                             int                   hint_gain = -1) {
    // quantize + count_bits helper (avoids repeating the branch 5 times)
    auto try_gain = [&](int g) -> int {
        if (!scalefac || gi.scalefac_compress == 0) {
            mp3enc_quantize(xr, ix, g);
        } else {
            mp3enc_quantize_sfb(xr, ix, g, scalefac, gi.scalefac_scale, gi.preflag, sfb_table);
        }
        for (int i = 0; i < 576; i++) {
            if (abs(ix[i]) >= 8191) {
                return available_bits + 1;  // saturated
            }
        }
        return mp3enc_count_bits(ix, gi, sfb_table, sr_index);
    };

    int best_gain = 210;
    int best_bits = available_bits + 1;

    if (hint_gain >= 0) {
        // Linear scan from hint. Typical cost: 3-4 try_gain calls.
        int bits = try_gain(hint_gain);

        if (bits <= available_bits) {
            // Hint fits. Scan downward to find the minimum valid gain.
            best_gain = hint_gain;
            best_bits = bits;
            for (int g = hint_gain - 1; g >= 0 && g >= hint_gain - 10; g--) {
                bits = try_gain(g);
                if (bits > available_bits) {
                    break;
                }
                best_gain = g;
                best_bits = bits;
            }
        } else {
            // Hint doesn't fit. Scan upward to find the first valid gain.
            bool found = false;
            for (int g = hint_gain + 1; g <= 255 && g <= hint_gain + 20; g++) {
                bits = try_gain(g);
                if (bits <= available_bits) {
                    best_gain = g;
                    best_bits = bits;
                    found     = true;
                    break;
                }
            }
            // Fallback: if scan didn't find it (rare, e.g. scalefac_scale toggle),
            // binary search on the remaining range.
            if (!found) {
                int lo = hint_gain + 21;
                int hi = 255;
                while (lo <= hi) {
                    int mid = (lo + hi) / 2;
                    bits    = try_gain(mid);
                    if (bits <= available_bits) {
                        best_gain = mid;
                        best_bits = bits;
                        hi        = mid - 1;
                    } else {
                        lo = mid + 1;
                    }
                }
            }
        }
    } else {
        // No hint: full binary search on [0, 255].
        int lo = 0, hi = 255;
        while (lo <= hi) {
            int mid  = (lo + hi) / 2;
            int bits = try_gain(mid);
            if (bits <= available_bits) {
                best_gain = mid;
                best_bits = bits;
                hi        = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
    }

    // Final quantization with the best gain
    gi.global_gain = best_gain;
    if (!scalefac || gi.scalefac_compress == 0) {
        mp3enc_quantize(xr, ix, best_gain);
    } else {
        mp3enc_quantize_sfb(xr, ix, best_gain, scalefac, gi.scalefac_scale, gi.preflag, sfb_table);
    }
    best_bits = mp3enc_count_bits(ix, gi, sfb_table, sr_index);
    return best_bits;
}

// Outer loop: iteratively adjust scalefactors to push quantization noise
// below the masking thresholds computed by the psy model.
//
// Algorithm:
//   1. Start with all scalefactors = 0, run inner loop
//   2. Compute noise per SFB
//   3. For each SFB where noise > xmin, bump its scalefactor
//   4. Update scalefac_compress, recompute bit budget, re-run inner loop
//   5. Repeat until noise is under control or we run out of iterations/bits
//
// xr: 576 MDCT coefficients
// ix: 576 quantized output
// gi: granule info (filled on return)
// xmin: masking thresholds per SFB from psy model
// available_bits: total bits for part2_3 (scalefactors + Huffman)
// sfb_table: SFB widths
// sr_index: sample rate index
// gr: granule number (0 or 1)
// scfsi: scfsi flags (for part2_length calculation)
// Returns part2_3_length (scalefactor bits + Huffman bits).
static int mp3enc_outer_loop(const float *         xr,
                             int *                 ix,
                             mp3enc_granule_info & gi,
                             const float *         xmin,
                             int                   available_bits,
                             const uint8_t *       sfb_table,
                             int                   sr_index,
                             int                   gr,
                             const int             scfsi[4]) {
    // Initialize: no scalefactors
    memset(&gi, 0, sizeof(gi));
    gi.block_type = 0;

    // Initial inner loop with flat quantization
    int huff_bits     = mp3enc_inner_loop(xr, ix, gi, available_bits, sfb_table, sr_index);
    gi.part2_3_length = huff_bits;

    // If no psy thresholds (all zero), skip outer loop
    bool have_psy = false;
    for (int sfb = 0; sfb < 21; sfb++) {
        if (xmin[sfb] > 0.0f) {
            have_psy = true;
            break;
        }
    }
    if (!have_psy) {
        return gi.part2_3_length;
    }

    // Outer iteration loop (ISO 11172-3 Annex C.1.5.4.3).
    // For each iteration:
    //   - compute distortion per SFB
    //   - bump scalefactor for EVERY band where noise > xmin
    //   - re-run inner loop with updated scalefactors
    //   - stop when all bands are under threshold or no bits left
    //
    // Max 25 passes: enough for convergence at all bitrates.
    float               noise[22];  // 22 SFB bands before table terminator
    int                 best_ix[576];
    mp3enc_granule_info best_gi     = gi;
    int                 best_total  = gi.part2_3_length;
    int                 best_over   = 21;      // start pessimistic
    float               best_max_db = 999.0f;  // worst-band noise in dB over threshold
    float               best_tot_db = 999.0f;  // total over-threshold noise in dB
    memcpy(best_ix, ix, sizeof(best_ix));

    for (int iter = 0; iter < 25; iter++) {
        // Compute noise per SFB with current quantization
        mp3enc_calc_noise(xr, ix, gi.global_gain, gi.scalefac_l, gi.scalefac_scale, gi.preflag, sfb_table, noise);

        // Compute noise metrics for 3-axis comparison (GPSYCHO approach).
        // Instead of just counting bands over threshold, track:
        //   - max_over_db: worst violation in dB (peak distortion)
        //   - tot_over_db: sum of violations in dB (for average)
        //   - over_count:  number of distorted bands
        // This prefers solutions that minimize peak distortion and spread
        // remaining noise evenly, rather than concentrating it in one band.
        int   over_count  = 0;
        float max_over_db = 0.0f;
        float tot_over_db = 0.0f;

        for (int sfb = 0; sfb < 21; sfb++) {
            if (xmin[sfb] > 0.0f && noise[sfb] > xmin[sfb]) {
                over_count++;
                float over_db = 10.0f * log10f(noise[sfb] / xmin[sfb]);
                tot_over_db += over_db;
                if (over_db > max_over_db) {
                    max_over_db = over_db;
                }
            }
        }

        // 3-axis quant_compare (inspired by LAME GPSYCHO outer_loop):
        //   1. Clean (over=0) always beats dirty (over>0)
        //   2. Among clean solutions: prefer fewer bits
        //   3. Among dirty solutions: minimize peak, then average, then count
        bool is_better = false;
        if (over_count == 0 && best_over > 0) {
            is_better = true;
        } else if (over_count == 0 && best_over == 0) {
            is_better = (gi.part2_3_length < best_total);
        } else if (over_count > 0 && best_over > 0) {
            // both dirty: compare peak distortion first
            if (max_over_db < best_max_db - 0.5f) {
                // significantly lower peak -> better
                is_better = true;
            } else if (max_over_db < best_max_db + 0.5f) {
                // similar peak: compare average violation
                float avg      = tot_over_db / (float) over_count;
                float best_avg = (best_over > 0) ? best_tot_db / (float) best_over : 0.0f;
                if (avg < best_avg - 0.3f) {
                    is_better = true;
                } else if (avg < best_avg + 0.3f) {
                    // similar average: prefer fewer violated bands
                    is_better = (over_count < best_over);
                }
            }
        }

        if (is_better) {
            best_gi     = gi;
            best_total  = gi.part2_3_length;
            best_over   = over_count;
            best_max_db = max_over_db;
            best_tot_db = tot_over_db;
            memcpy(best_ix, ix, sizeof(best_ix));
        }

        // If all bands are under threshold, we are done
        if (over_count == 0) {
            break;
        }

        // Bump scalefactor for EVERY band where noise > threshold.
        // ISO outer loop: amplify all distorted bands by 1 step per iteration.
        bool any_changed = false;
        for (int sfb = 0; sfb < 21; sfb++) {
            if (xmin[sfb] > 0.0f && noise[sfb] > xmin[sfb]) {
                gi.scalefac_l[sfb]++;
                any_changed = true;
            }
        }
        if (!any_changed) {
            break;
        }

        // Preflag: if HF bands need large scalefactors, enable preflag
        // to get free amplification from the pretab table (ISO Table B.6).
        // This saves bits: pretab adds 0-3 to HF scalefactors for free
        // (encoded in a single bit rather than per-band bits).
        if (!gi.preflag) {
            int hf_need = 0;
            for (int sfb = 11; sfb < 21; sfb++) {
                if (gi.scalefac_l[sfb] >= 2 && mp3enc_pretab[sfb] > 0) {
                    hf_need++;
                }
            }
            // Enable if at least 3 HF bands need boosting
            if (hf_need >= 3) {
                gi.preflag = 1;
                for (int sfb = 0; sfb < 21; sfb++) {
                    gi.scalefac_l[sfb] -= mp3enc_pretab[sfb];
                    if (gi.scalefac_l[sfb] < 0) {
                        gi.scalefac_l[sfb] = 0;
                    }
                }
            }
        }

        // scalefac_scale: if any scalefactor exceeds 15 (4 bit max),
        // double the step size. This halves all scalefactors but each
        // step now represents sqrt(2) instead of 2^(1/4).
        int max_sf = 0;
        for (int sfb = 0; sfb < 21; sfb++) {
            if (gi.scalefac_l[sfb] > max_sf) {
                max_sf = gi.scalefac_l[sfb];
            }
        }
        if (max_sf > 15 && !gi.scalefac_scale) {
            gi.scalefac_scale = 1;
            for (int sfb = 0; sfb < 21; sfb++) {
                gi.scalefac_l[sfb] = (gi.scalefac_l[sfb] + 1) / 2;
            }
        }

        // Clamp to 15
        for (int sfb = 0; sfb < 21; sfb++) {
            if (gi.scalefac_l[sfb] > 15) {
                gi.scalefac_l[sfb] = 15;
            }
        }

        // Update scalefac_compress and compute part2 bits
        gi.scalefac_compress = mp3enc_best_scalefac_compress(gi.scalefac_l);
        int part2            = mp3enc_part2_length(gi, gr, scfsi);
        int huff_budget      = available_bits - part2;
        if (huff_budget < 0) {
            break;
        }

        // Re-run inner loop
        huff_bits = mp3enc_inner_loop(xr, ix, gi, huff_budget, sfb_table, sr_index, gi.scalefac_l, gi.global_gain);
        int total = part2 + huff_bits;

        if (total <= available_bits) {
            gi.part2_3_length = total;
        } else {
            break;
        }
    }

    // Restore best result
    gi = best_gi;
    memcpy(ix, best_ix, sizeof(best_ix));
    gi.part2_3_length = best_total;

    return gi.part2_3_length;
}
