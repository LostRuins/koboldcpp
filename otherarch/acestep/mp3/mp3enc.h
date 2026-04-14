#pragma once
// mp3enc.h
// MPEG1 Layer III MP3 encoder, CBR, 32/44.1/48 kHz, mono/stereo.
// MIT license.
//
// Usage:
//   mp3enc_t * enc = mp3enc_init(44100, 2, 128);
//   while (have_audio) {
//       const uint8_t * mp3 = mp3enc_encode(enc, pcm, samples, &size);
//       fwrite(mp3, 1, size, fp);
//   }
//   const uint8_t * mp3 = mp3enc_flush(enc, &size);
//   fwrite(mp3, 1, size, fp);
//   mp3enc_free(enc);
//
// PCM input: planar float [ch0: N samples] [ch1: N samples]
// Output: raw MP3 frames (no ID3 tags)
//
// MIT license.

// clang-format off
// Order matters: tables.h defines constants used by all other headers.
#include "mp3enc-tables.h"
#include "mp3enc-bits.h"
#include "mp3enc-filter.h"
#include "mp3enc-huff.h"
#include "mp3enc-mdct.h"
#include "mp3enc-psy.h"
#include "mp3enc-quant.h"
// clang-format on

#include <cstdlib>
#include <cstring>

// Encoder state.
struct mp3enc_t {
    // Config
    int sample_rate;
    int channels;
    int bitrate_kbps;
    int sr_index;  // 0=44100, 1=48000, 2=32000

    // Frame geometry
    int frame_samples;    // always 1152 for MPEG1 Layer III
    int slots_per_frame;  // frame size in bytes (varies with padding)

    // Per channel state
    mp3enc_filter filter[2];           // analysis filterbank
    float         sb_prev[2][32][18];  // subband overlap memory
    float         sb_cur[2][32][18];   // current granule subbands
    mp3enc_psy    psy;                 // psychoacoustic model

    // Bit reservoir (ISO 11172-3, clause 2.4.2.7).
    // Tracks unused bits from previous frames that can be borrowed.
    int resv_size;  // current reservoir size in bits
    int resv_max;   // max: 511 bytes * 8 = 4088 bits

    // Adaptive lowpass: MDCT line index above which coefficients are zeroed.
    // Saves bits at low bitrates by not encoding inaudible HF content.
    int lowpass_line;

    // PCM input buffer (accumulates until we have 1152 samples)
    float * pcm_buf;   // interleaved buffer: [ch0_1152][ch1_1152]
    int     pcm_fill;  // samples accumulated per channel

    // Output buffer (holds accumulated MP3 frames for one encode call)
    uint8_t * out_buf;
    int       out_capacity;
    int       out_written;

    // Scratch buffer for one frame (max ~1441 bytes at 320kbps/32kHz)
    uint8_t frame_buf[2048];

    // Main data scratch: written separately from header+sideinfo for reservoir
    uint8_t md_scratch[2048];

    // Pending frame: we delay output by one frame so we can write the next
    // frame's main_data overflow into the current frame's unused tail.
    // This is how the bit reservoir works (ISO 11172-3 clause 2.4.2.7).
    uint8_t pending_frame[2048];
    int     pending_bytes;   // size of pending frame (0 = no pending frame)
    int     pending_md_end;  // byte offset where main_data ends in pending_frame

    // Padding state (for 44100 Hz which needs alternating padding)
    int pad_remainder;
};

// Initialize encoder.
static mp3enc_t * mp3enc_init(int sample_rate, int channels, int bitrate_kbps) {
    mp3enc_t * enc = (mp3enc_t *) calloc(1, sizeof(mp3enc_t));
    if (!enc) {
        return nullptr;
    }

    enc->sample_rate   = sample_rate;
    enc->channels      = channels;
    enc->bitrate_kbps  = bitrate_kbps;
    enc->frame_samples = 1152;

    // Determine sample rate index
    if (sample_rate == 44100) {
        enc->sr_index = 0;
    } else if (sample_rate == 48000) {
        enc->sr_index = 1;
    } else if (sample_rate == 32000) {
        enc->sr_index = 2;
    } else {
        free(enc);
        return nullptr;
    }

    // Init subsystems
    for (int ch = 0; ch < channels; ch++) {
        enc->filter[ch].init();
    }
    memset(enc->sb_prev, 0, sizeof(enc->sb_prev));
    enc->psy.init();
    enc->psy.init_ath(enc->sr_index, mp3enc_sfb_long[enc->sr_index], sample_rate);

    // Bit reservoir: max 511 bytes (9 bit field main_data_begin)
    enc->resv_size = 0;
    enc->resv_max  = 511 * 8;

    // Adaptive lowpass: cut HF that wastes bits at low bitrates.
    // Cutoff per total bitrate in Hz. Higher bitrates preserve more bandwidth.
    {
        static const struct {
            int kbps;
            int hz;
        } lp_table[] = {
            { 8,   2000  },
            { 16,  3700  },
            { 24,  3900  },
            { 32,  5500  },
            { 40,  7000  },
            { 48,  7500  },
            { 56,  10000 },
            { 64,  11000 },
            { 80,  13500 },
            { 96,  15100 },
            { 112, 15600 },
            { 128, 17000 },
            { 160, 17500 },
            { 192, 18600 },
            { 224, 19400 },
            { 256, 19700 },
            { 320, 20500 }
        };

        static const int lp_count = (int) (sizeof(lp_table) / sizeof(lp_table[0]));

        // Find nearest bitrate in table
        int best      = 0;
        int best_dist = 999;
        for (int i = 0; i < lp_count; i++) {
            int dist = abs(bitrate_kbps - lp_table[i].kbps);
            if (dist < best_dist) {
                best_dist = dist;
                best      = i;
            }
        }
        float cutoff_hz = (float) lp_table[best].hz;

        // MDCT line corresponding to cutoff: 576 lines cover 0..samplerate/2
        float freq_per_line = (float) sample_rate / (2.0f * 576.0f);
        enc->lowpass_line   = (int) (cutoff_hz / freq_per_line);
        if (enc->lowpass_line > 576) {
            enc->lowpass_line = 576;
        }
    }

    // Allocate buffers
    enc->pcm_buf  = (float *) calloc(1152 * channels, sizeof(float));
    enc->pcm_fill = 0;

    enc->out_capacity = 2048;
    enc->out_buf      = (uint8_t *) malloc(enc->out_capacity);
    enc->out_written  = 0;

    enc->pad_remainder = 0;

    // No pending frame at start
    enc->pending_bytes  = 0;
    enc->pending_md_end = 0;

    return enc;
}

// Free encoder.
static void mp3enc_free(mp3enc_t * enc) {
    if (!enc) {
        return;
    }
    free(enc->pcm_buf);
    free(enc->out_buf);
    free(enc);
}

// Compute padding for this frame (needed at 44100 Hz).
static int mp3enc_get_padding(mp3enc_t * enc) {
    int dif = (144 * enc->bitrate_kbps * 1000) % enc->sample_rate;
    enc->pad_remainder -= dif;
    if (enc->pad_remainder < 0) {
        enc->pad_remainder += enc->sample_rate;
        return 1;
    }
    return 0;
}

// Encode one complete frame (1152 samples per channel).
// pcm: planar float [ch0: 1152 samples][ch1: 1152 samples]
// Returns number of bytes written to enc->frame_buf.
static int mp3enc_encode_frame(mp3enc_t * enc, const float * pcm) {
    int nch     = enc->channels;
    int padding = mp3enc_get_padding(enc);

    // MS stereo decision is deferred until after MDCT energy analysis.
    // mode=1 (joint) allows switching M/S on or off per frame.
    // mode=0 (stereo) would force L/R always. mode=3 (mono).
    int mode     = (nch == 1) ? 3 : 1;
    int mode_ext = 0;  // set after MDCT energy analysis

    // Setup header
    mp3enc_header hdr;
    hdr.bitrate_kbps = enc->bitrate_kbps;
    hdr.samplerate   = enc->sample_rate;
    hdr.mode         = mode;
    hdr.mode_ext     = mode_ext;
    hdr.padding      = padding;

    int frame_bytes     = hdr.frame_bytes();
    int side_info_bytes = (nch == 1) ? 17 : 32;
    int main_data_bytes = frame_bytes - 4 - side_info_bytes;
    int main_data_bits  = main_data_bytes * 8;

    // SFB tables for this sample rate
    const uint8_t * sfb_long = mp3enc_sfb_long[enc->sr_index];

    // Process 2 granules (each is 576 samples = 18 subband slots of 32 samples)
    mp3enc_side_info si;
    memset(&si, 0, sizeof(si));

    // Cross-frame bit reservoir (ISO 11172-3 clause 2.4.2.7).
    // Unused bytes at the end of the previous frame can be borrowed by this frame.
    // The pending_frame holds the previous frame; its unused tail is the reservoir.
    int resv_bytes = 0;
    if (enc->pending_bytes > 0) {
        resv_bytes = enc->pending_bytes - enc->pending_md_end;
        if (resv_bytes < 0) {
            resv_bytes = 0;
        }
        if (resv_bytes > 511) {
            resv_bytes = 511;
        }
    }
    si.main_data_begin = resv_bytes;

    // Total main_data bits: this frame's area + reservoir from previous frame
    int total_md_bits = main_data_bits + resv_bytes * 8;

    // Phase 1: compute MDCT + psy for ALL granules before bit allocation.
    // We need PE from both granules to weight the bit budget.
    float mdct_all[2][2][576];  // [granule][channel][576 MDCT lines]
    float saved_xmin[2][2][MP3ENC_PSY_SFB_MAX];
    float saved_pe[2] = { 0.0f, 0.0f };

    // Accumulators for MS stereo decision
    float energy_mid  = 0.0f;
    float energy_side = 0.0f;

    for (int gr = 0; gr < 2; gr++) {
        int pcm_offset = gr * 576;

        // filterbank + MDCT for all channels
        for (int ch = 0; ch < nch; ch++) {
            const float * ch_pcm = pcm + ch * 1152 + pcm_offset;
            float         sb_out[32];
            for (int slot = 0; slot < 18; slot++) {
                enc->filter[ch].process(ch_pcm + slot * 32, sb_out);
                for (int sb = 0; sb < 32; sb++) {
                    enc->sb_cur[ch][sb][slot] = sb_out[sb];
                }
                // frequency inversion: negate odd subbands at odd time slots
                if (slot & 1) {
                    for (int sb = 1; sb < 32; sb += 2) {
                        enc->sb_cur[ch][sb][slot] = -enc->sb_cur[ch][sb][slot];
                    }
                }
            }
            mp3enc_mdct_granule(enc->sb_prev[ch], enc->sb_cur[ch], mdct_all[gr][ch]);
            memcpy(enc->sb_prev[ch], enc->sb_cur[ch], sizeof(enc->sb_cur[ch]));
        }

        // MS stereo energy analysis (before transform, on L/R data).
        // Accumulate mid and side energy across both granules to decide
        // whether M/S coding is beneficial for this frame.
        // M/S wins when L and R are correlated (side energy is small).
        if (nch == 2) {
            for (int i = 0; i < enc->lowpass_line && i < 576; i++) {
                float l = mdct_all[gr][0][i];
                float r = mdct_all[gr][1][i];
                float m = l + r;
                float s = l - r;
                energy_mid += m * m;
                energy_side += s * s;
            }
        }
    }

    // MS stereo decision: use M/S when channels are correlated enough
    // that the side channel is cheap to encode. FhG "almost always uses
    // ms_stereo" (GPSYCHO docs). We use it unless side energy dominates,
    // which means the channels are very different (rare for music).
    bool use_ms = false;
    if (nch == 2) {
        // Use M/S unless side channel has more energy than mid.
        // This is generous -- almost always enables M/S (like FhG).
        use_ms       = (energy_side < energy_mid * 1.2f) || (energy_mid < 1e-20f);
        mode_ext     = use_ms ? 2 : 0;
        hdr.mode_ext = mode_ext;
    }

    // Now apply M/S transform + lowpass + psy for both granules
    for (int gr = 0; gr < 2; gr++) {
        // MS stereo transform
        if (use_ms) {
            static const float ms_scale = 0.7071067811865476f;  // 1/sqrt(2)
            for (int i = 0; i < 576; i++) {
                float l            = mdct_all[gr][0][i];
                float r            = mdct_all[gr][1][i];
                mdct_all[gr][0][i] = (l + r) * ms_scale;
                mdct_all[gr][1][i] = (l - r) * ms_scale;
            }
        }

        // adaptive lowpass
        for (int ch = 0; ch < nch; ch++) {
            for (int i = enc->lowpass_line; i < 576; i++) {
                mdct_all[gr][ch][i] = 0.0f;
            }
        }

        // psy: compute masking thresholds and perceptual entropy
        for (int ch = 0; ch < nch; ch++) {
            enc->psy.compute(mdct_all[gr][ch], sfb_long, enc->sr_index, ch);
            memcpy(saved_xmin[gr][ch], enc->psy.xmin, sizeof(enc->psy.xmin));
            saved_pe[gr] += enc->psy.pe;
        }
    }

    // Phase 2: PE-weighted bit allocation + quantization.
    // Give more bits to granules with higher perceptual entropy (more complex
    // signal). ISO 11172-3 computes PE but then ignores it for allocation;
    // LAME uses it to drive the bit reservoir. We weight the per-granule
    // budget proportionally to PE, clamped to avoid starving either granule.
    int   gr_budget[2];
    float pe_sum = saved_pe[0] + saved_pe[1];
    if (pe_sum > 1e-6f) {
        float frac0  = saved_pe[0] / pe_sum;
        gr_budget[0] = (int) ((float) total_md_bits * frac0);
        gr_budget[1] = total_md_bits - gr_budget[0];
        // clamp: neither granule gets less than 20% or more than 80%
        int lo       = total_md_bits / 5;
        int hi       = total_md_bits * 4 / 5;
        for (int gr = 0; gr < 2; gr++) {
            if (gr_budget[gr] < lo) {
                gr_budget[gr] = lo;
            }
            if (gr_budget[gr] > hi) {
                gr_budget[gr] = hi;
            }
        }
        // re-normalize after clamp
        int clamped_sum    = gr_budget[0] + gr_budget[1];
        int bits_to_spread = total_md_bits - clamped_sum;
        gr_budget[0] += bits_to_spread / 2;
        gr_budget[1] += bits_to_spread - bits_to_spread / 2;
    } else {
        gr_budget[0] = total_md_bits / 2;
        gr_budget[1] = total_md_bits - gr_budget[0];
    }

    int ix[2][2][576];
    int total_bits_used = 0;
    int intra_resv      = 0;

    for (int gr = 0; gr < 2; gr++) {
        int max_bits = gr_budget[gr] + intra_resv;

        // don't exceed remaining budget
        int remaining_bits = total_md_bits - total_bits_used;
        if (max_bits > remaining_bits) {
            max_bits = remaining_bits;
        }
        if (max_bits > 4095 * nch) {
            max_bits = 4095 * nch;
        }

        int bits_per_ch = max_bits / nch;

        // quantize each channel with psy thresholds
        int gr_bits_used = 0;
        for (int ch = 0; ch < nch; ch++) {
            int bits = mp3enc_outer_loop(mdct_all[gr][ch], ix[gr][ch], si.gr[gr][ch], saved_xmin[gr][ch], bits_per_ch,
                                         sfb_long, enc->sr_index, gr, si.scfsi[ch]);
            gr_bits_used += bits;
        }

        // intra-frame savings: unused bits carry to next granule
        intra_resv += gr_budget[gr] - gr_bits_used;
        if (intra_resv < 0) {
            intra_resv = 0;
        }
        total_bits_used += gr_bits_used;
    }

    // Write header + side_info to frame_buf
    mp3enc_bs hdr_bs;
    hdr_bs.init(enc->frame_buf, sizeof(enc->frame_buf));
    hdr.write(hdr_bs);
    si.write(hdr_bs, nch);
    int hdr_si_bytes = (hdr_bs.total_bits() + 7) / 8;

    // Write main_data to md_scratch (separate buffer for reservoir assembly)
    mp3enc_bs md_bs;
    md_bs.init(enc->md_scratch, sizeof(enc->md_scratch));

    for (int gr = 0; gr < 2; gr++) {
        for (int ch = 0; ch < nch; ch++) {
            const mp3enc_granule_info & gi = si.gr[gr][ch];

            // Scalefactors
            mp3enc_write_scalefactors(md_bs, gi, gr, nch, si.scfsi[ch]);

            // Huffman data: big_values region (3 regions for long blocks)
            int prev_end = 0;
            for (int r = 0; r < 3; r++) {
                int reg_end;
                if (r == 0) {
                    int acc = 0;
                    for (int s = 0; s <= gi.region0_count; s++) {
                        acc += sfb_long[s];
                    }
                    reg_end = (acc < gi.big_values * 2) ? acc : gi.big_values * 2;
                } else if (r == 1) {
                    int acc = 0;
                    for (int s = 0; s <= gi.region0_count + gi.region1_count + 1; s++) {
                        acc += sfb_long[s];
                    }
                    reg_end = (acc < gi.big_values * 2) ? acc : gi.big_values * 2;
                } else {
                    reg_end = gi.big_values * 2;
                }

                int pairs = (reg_end - prev_end) / 2;
                for (int p = 0; p < pairs; p++) {
                    int i = prev_end + p * 2;
                    mp3enc_write_pair(md_bs, gi.table_select[r], ix[gr][ch][i], ix[gr][ch][i + 1]);
                }
                prev_end = reg_end;
            }

            // Count1 region
            int c1_start = gi.big_values * 2;
            int nz_end   = 576 - mp3enc_count_rzero(ix[gr][ch], 576) * 2;
            int c1_count = (nz_end - c1_start) / 4;
            mp3enc_write_count1(md_bs, ix[gr][ch], c1_start, c1_count, gi.count1table_select);
        }
    }
    int md_bytes = (md_bs.total_bits() + 7) / 8;

    // Assemble output: write overflow to pending frame, output it, save current as pending.
    //
    // The first resv_bytes of main_data go into the previous frame's unused tail.
    // The rest goes into the current frame's main data area.
    // This is how the decoder's main_data_begin backpointer works.
    int output_bytes = 0;

    if (enc->pending_bytes > 0) {
        // Write main_data overflow into pending frame's unused tail
        int overflow = (md_bytes < resv_bytes) ? md_bytes : resv_bytes;
        if (overflow > 0) {
            memcpy(enc->pending_frame + enc->pending_md_end, enc->md_scratch, overflow);
        }

        // Grow out_buf if needed
        int need = enc->out_written + enc->pending_bytes + frame_bytes;
        if (need > enc->out_capacity) {
            enc->out_capacity = need * 2;
            enc->out_buf      = (uint8_t *) realloc(enc->out_buf, enc->out_capacity);
        }

        // Output the pending frame (now complete with overflow data)
        memcpy(enc->out_buf + enc->out_written, enc->pending_frame, enc->pending_bytes);
        enc->out_written += enc->pending_bytes;
        output_bytes = enc->pending_bytes;
    }

    // Build current frame: header+sideinfo already in frame_buf, add main_data + padding
    int md_in_frame = md_bytes - resv_bytes;
    if (md_in_frame < 0) {
        md_in_frame = 0;
    }
    int md_area = frame_bytes - hdr_si_bytes;

    // Copy the portion of main_data that goes in this frame
    if (md_in_frame > 0) {
        memcpy(enc->frame_buf + hdr_si_bytes, enc->md_scratch + resv_bytes, md_in_frame);
    }
    // Zero-pad the unused tail (this becomes reservoir for the next frame)
    if (md_in_frame < md_area) {
        memset(enc->frame_buf + hdr_si_bytes + md_in_frame, 0, md_area - md_in_frame);
    }

    // Save current frame as pending (will be output when next frame is encoded)
    memcpy(enc->pending_frame, enc->frame_buf, frame_bytes);
    enc->pending_bytes  = frame_bytes;
    enc->pending_md_end = hdr_si_bytes + md_in_frame;

    return output_bytes;
}

// Public API: encode PCM samples.
// audio: planar float [ch0: n_samples][ch1: n_samples]
// n_samples: number of samples per channel
// out_size: receives the total MP3 bytes produced
// Returns pointer to MP3 data (internal buffer, valid until next call).
static const uint8_t * mp3enc_encode(mp3enc_t * enc, const float * audio, int n_samples, int * out_size) {
    *out_size    = 0;
    int nch      = enc->channels;
    int consumed = 0;

    // Reset output write position
    enc->out_written = 0;

    while (consumed < n_samples) {
        // Fill PCM buffer up to 1152 samples per channel
        int space = 1152 - enc->pcm_fill;
        int avail = n_samples - consumed;
        int copy  = (avail < space) ? avail : space;

        for (int ch = 0; ch < nch; ch++) {
            memcpy(enc->pcm_buf + ch * 1152 + enc->pcm_fill, audio + ch * n_samples + consumed, copy * sizeof(float));
        }
        enc->pcm_fill += copy;
        consumed += copy;

        // When we have 1152 samples, encode a frame.
        // encode_frame handles writing to out_buf (including pending frame output).
        if (enc->pcm_fill == 1152) {
            mp3enc_encode_frame(enc, enc->pcm_buf);
            enc->pcm_fill = 0;
        }
    }

    *out_size = enc->out_written;
    return enc->out_buf;
}

// Flush remaining samples (zero pad to 1152, encode last frame).
// Also outputs the final pending frame from the bit reservoir delay.
static const uint8_t * mp3enc_flush(mp3enc_t * enc, int * out_size) {
    *out_size        = 0;
    enc->out_written = 0;

    if (enc->pcm_fill > 0) {
        // Zero pad to 1152
        int nch = enc->channels;
        for (int ch = 0; ch < nch; ch++) {
            memset(enc->pcm_buf + ch * 1152 + enc->pcm_fill, 0, (1152 - enc->pcm_fill) * sizeof(float));
        }
        enc->pcm_fill = 1152;
        mp3enc_encode_frame(enc, enc->pcm_buf);
        enc->pcm_fill = 0;
    }

    // Output the final pending frame (delayed by one frame for reservoir)
    if (enc->pending_bytes > 0) {
        int need = enc->out_written + enc->pending_bytes;
        if (need > enc->out_capacity) {
            enc->out_capacity = need * 2;
            enc->out_buf      = (uint8_t *) realloc(enc->out_buf, enc->out_capacity);
        }
        memcpy(enc->out_buf + enc->out_written, enc->pending_frame, enc->pending_bytes);
        enc->out_written += enc->pending_bytes;
        enc->pending_bytes = 0;
    }

    *out_size = enc->out_written;
    return enc->out_buf;
}
