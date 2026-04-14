#pragma once
// mp3enc-filter.h
// Polyphase analysis filterbank: 32 new PCM samples in, 32 subband samples out.
// ISO 11172-3 clause 2.4.3.2 (encoder side, Annex C).
// Part of mp3enc. MIT license.

#include <cmath>
#include <cstring>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
// The analysis filterbank state for one channel.
// Holds the 512 sample FIFO and produces 32 subband outputs per call.
struct mp3enc_filter {
    float buf[512];  // circular buffer of past PCM samples
    int   off;       // current write offset into buf

    void init() {
        memset(buf, 0, sizeof(buf));
        off = 0;
    }

    // Feed 32 new PCM samples, produce 32 subband samples.
    // pcm: pointer to 32 float samples (mono, this channel only)
    // sb_out: output array of 32 subband values
    //
    // Algorithm (ISO 11172-3 Annex C, encoding process):
    // 1. Shift 32 new samples into the 512 sample FIFO
    // 2. Window: multiply by 512 Ci coefficients (mp3enc_enwindow)
    // 3. Partial sum: 512 windowed values -> 64 values
    // 4. Matrixing: 64 values -> 32 subband samples via DCT
    void process(const float * pcm, float * sb_out) {
        // Step 1: shift new samples into the buffer.
        // The newest sample goes at buf[off], oldest at buf[off+31].
        // We store them in reverse order so windowing is a straight multiply.
        off = (off - 32) & 511;
        for (int i = 0; i < 32; i++) {
            buf[(off + i) & 511] = pcm[31 - i];
        }

        // Step 2 + 3: window and partial sum.
        // z[i] = sum over j=0..7 of: buf[(off + i + 64*j) & 511] * Ci[i + 64*j]
        // This produces 64 values from the 512 windowed samples.
        float z[64];
        for (int i = 0; i < 64; i++) {
            float sum = 0.0f;
            for (int j = 0; j < 8; j++) {
                int buf_idx = (off + i + 64 * j) & 511;
                int win_idx = i + 64 * j;
                sum += buf[buf_idx] * mp3enc_enwindow[win_idx];
            }
            z[i] = sum;
        }

        // Step 4: matrixing (ISO 11172-3, Annex C, analysis filter).
        // sb_out[k] = sum over i=0..63 of: z[i] * cos((2*k + 1) * (16 - i) * PI / 64)
        // for k = 0..31 (subbands)
        // Formula verified against ISO 11172-3 Table C.1
        for (int k = 0; k < 32; k++) {
            float sum = 0.0f;
            for (int i = 0; i < 64; i++) {
                float angle = (float) M_PI * (float) (2 * k + 1) * (float) (16 - i) / 64.0f;
                sum += z[i] * cosf(angle);
            }
            sb_out[k] = sum;
        }
    }
};
