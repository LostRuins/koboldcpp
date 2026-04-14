#pragma once
// mp3enc-mdct.h
// Forward MDCT for the MP3 encoder: 36 subband samples -> 18 frequency lines.
// Window and MDCT are combined into a single step (ISO 11172-3 Annex C).
// Part of mp3enc. MIT license.

#include <cmath>
#include <cstring>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
// Forward MDCT-36 for long blocks.
// in[36] = prev[18] + cur[18] (raw subband samples)
// out[18] = MDCT frequency coefficients
//
// Formula (ISO 11172-3 Annex C):
//   out[k] = (1/9) * sum(n=0..35) in[n] * sin(pi/36*(n+0.5)) * cos(pi/72*(2n+19)*(2k+1))
static void mp3enc_mdct36(const float * in, float * out) {
    for (int k = 0; k < 18; k++) {
        float sum = 0.0f;
        for (int n = 0; n < 36; n++) {
            float w = sinf((float) M_PI / 36.0f * ((float) n + 0.5f));
            float c = cosf((float) M_PI / 72.0f * (float) (2 * n + 19) * (float) (2 * k + 1));
            sum += in[n] * w * c;
        }
        out[k] = sum * (1.0f / 9.0f);
    }
}

// Alias reduction butterfly between adjacent subbands.
// Applied after MDCT, before quantization.
// ISO 11172-3 Table B.9 coefficients (from minimp3 CC0).
//
// For each pair of adjacent bands (band, band+1):
//   mdct[band][17-i] and mdct[band+1][i] are butterflied with cs/ca.
static void mp3enc_alias_reduce(float * mdct_out) {
    for (int band = 1; band < 32; band++) {
        float * a = mdct_out + (band - 1) * 18;  // previous band
        float * b = mdct_out + band * 18;        // current band
        for (int i = 0; i < 8; i++) {
            float u   = a[17 - i];
            float d   = b[i];
            a[17 - i] = u * mp3enc_cs[i] - d * mp3enc_ca[i];
            b[i]      = d * mp3enc_cs[i] + u * mp3enc_ca[i];
        }
    }
}

// Process all 32 subbands for one granule (long blocks only).
// sb_samples layout: prev_gr[32][18] and cur_gr[32][18] (band major).
// mdct_out[576]: output frequency lines (32 subbands * 18 lines)
static void mp3enc_mdct_granule(const float sb_prev[32][18], const float sb_cur[32][18], float * mdct_out) {
    for (int band = 0; band < 32; band++) {
        float mdct_in[36];
        for (int k = 0; k < 18; k++) {
            mdct_in[k]      = sb_prev[band][k];
            mdct_in[k + 18] = sb_cur[band][k];
        }
        mp3enc_mdct36(mdct_in, mdct_out + band * 18);
    }

    mp3enc_alias_reduce(mdct_out);
}
