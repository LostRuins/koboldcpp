#pragma once
// mp3enc-bits.h
// Bitstream writer, frame header, side information packer.
// Part of mp3enc. MIT license.

#include <cstdint>
#include <cstring>

// Bitstream writer: accumulates bits MSB first into a byte buffer.
// The encoder writes frame header, side info, and Huffman data through this.
struct mp3enc_bs {
    uint8_t * buf;       // output buffer (caller owned)
    int       capacity;  // total bytes available
    int       byte_pos;  // current byte offset
    int       bit_pos;   // bits used in current byte (0..7, 0 = empty)

    void init(uint8_t * dst, int cap) {
        buf      = dst;
        capacity = cap;
        byte_pos = 0;
        bit_pos  = 0;
        memset(dst, 0, cap);
    }

    // Write n bits (1..32) from val, MSB first.
    void put(uint32_t val, int n) {
        for (int i = n - 1; i >= 0; i--) {
            buf[byte_pos] |= (uint8_t) (((val >> i) & 1) << (7 - bit_pos));
            bit_pos++;
            if (bit_pos == 8) {
                bit_pos = 0;
                byte_pos++;
            }
        }
    }

    // Total bits written so far
    int total_bits() const { return byte_pos * 8 + bit_pos; }

    // Byte align (pad with zeros)
    void align() {
        if (bit_pos > 0) {
            byte_pos++;
            bit_pos = 0;
        }
    }
};

// Frame header: 4 bytes, fixed format for MPEG1 Layer III.
// ISO 11172-3, clause 2.4.2.3
struct mp3enc_header {
    int bitrate_kbps;  // from mp3enc_bitrate_kbps[]
    int samplerate;    // 44100, 48000, or 32000
    int mode;          // 0=stereo, 1=joint, 2=dual, 3=mono
    int mode_ext;      // for joint stereo: bit0=intensity, bit1=ms
    int padding;       // 0 or 1

    // Compute bitrate_index from kbps
    int bitrate_index() const {
        // Match against the Layer III table
        static const int br[] = { 0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 };
        for (int i = 1; i < 15; i++) {
            if (br[i] == bitrate_kbps) {
                return i;
            }
        }
        return 0;  // free format
    }

    // Compute sampling_frequency field
    int sr_index() const {
        if (samplerate == 44100) {
            return 0;
        }
        if (samplerate == 48000) {
            return 1;
        }
        if (samplerate == 32000) {
            return 2;
        }
        return 0;
    }

    // Frame size in bytes (including header)
    int frame_bytes() const { return 144 * bitrate_kbps * 1000 / samplerate + padding; }

    // Write 4 byte header to bitstream
    void write(mp3enc_bs & bs) const {
        bs.put(0xFFF, 12);           // syncword
        bs.put(1, 1);                // ID = MPEG1
        bs.put(1, 2);                // layer = III (01)
        bs.put(1, 1);                // protection_bit = 1 (no CRC)
        bs.put(bitrate_index(), 4);  // bitrate_index
        bs.put(sr_index(), 2);       // sampling_frequency
        bs.put(padding, 1);          // padding_bit
        bs.put(0, 1);                // private_bit
        bs.put(mode, 2);             // mode
        bs.put(mode_ext, 2);         // mode_extension
        bs.put(0, 1);                // copyright
        bs.put(1, 1);                // original
        bs.put(0, 2);                // emphasis = none
    }
};

// Granule side information for one channel.
// ISO 11172-3, clause 2.4.1.7
struct mp3enc_granule_info {
    int part2_3_length;      // total bits: scalefactors + Huffman data
    int big_values;          // number of pairs in big_values region
    int global_gain;         // quantizer step size (0..255)
    int scalefac_compress;   // index into slen table (0..15)
    int block_type;          // 0=normal, 1=start, 2=short, 3=stop
    int mixed_block_flag;    // 1 if lower bands use long windows
    int table_select[3];     // Huffman table for each region
    int subblock_gain[3];    // gain offset per short window
    int region0_count;       // sfb count in region 0
    int region1_count;       // sfb count in region 1
    int preflag;             // high frequency boost
    int scalefac_scale;      // 0 = sqrt(2) step, 1 = 2 step
    int count1table_select;  // 0 = table A, 1 = table B

    // Scale factors (filled by quantization loop)
    int scalefac_l[21];     // long block scalefactors
    int scalefac_s[12][3];  // short block scalefactors
};

// Side information for one frame.
// Stereo: 32 bytes, mono: 17 bytes.
struct mp3enc_side_info {
    int                 main_data_begin;  // bit reservoir backpointer (bytes)
    int                 scfsi[2][4];      // scalefactor selection info per channel
    mp3enc_granule_info gr[2][2];         // [granule][channel]

    // Write side info to bitstream (after header).
    // nch = 1 for mono, 2 for stereo/joint/dual
    void write(mp3enc_bs & bs, int nch) const {
        bs.put(main_data_begin, 9);

        // private bits
        if (nch == 1) {
            bs.put(0, 5);
        } else {
            bs.put(0, 3);
        }

        // scfsi
        for (int ch = 0; ch < nch; ch++) {
            for (int band = 0; band < 4; band++) {
                bs.put(scfsi[ch][band], 1);
            }
        }

        // per granule, per channel
        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < nch; ch++) {
                const mp3enc_granule_info & gi = gr[g][ch];
                bs.put(gi.part2_3_length, 12);
                bs.put(gi.big_values, 9);
                bs.put(gi.global_gain, 8);
                bs.put(gi.scalefac_compress, 4);

                int window_switching = (gi.block_type != 0) ? 1 : 0;
                bs.put(window_switching, 1);

                if (window_switching) {
                    bs.put(gi.block_type, 2);
                    bs.put(gi.mixed_block_flag, 1);
                    for (int r = 0; r < 2; r++) {
                        bs.put(gi.table_select[r], 5);
                    }
                    for (int w = 0; w < 3; w++) {
                        bs.put(gi.subblock_gain[w], 3);
                    }
                } else {
                    for (int r = 0; r < 3; r++) {
                        bs.put(gi.table_select[r], 5);
                    }
                    bs.put(gi.region0_count, 4);
                    bs.put(gi.region1_count, 3);
                }

                bs.put(gi.preflag, 1);
                bs.put(gi.scalefac_scale, 1);
                bs.put(gi.count1table_select, 1);
            }
        }
    }
};

// Write scalefactors for one granule/channel into main_data.
// Returns number of bits written.
static int mp3enc_write_scalefactors(mp3enc_bs &                 bs,
                                     const mp3enc_granule_info & gi,
                                     int                         gr,
                                     int                         nch_unused,
                                     const int                   scfsi[4]) {
    (void) nch_unused;
    int slen1 = mp3enc_slen[0][gi.scalefac_compress];
    int slen2 = mp3enc_slen[1][gi.scalefac_compress];
    int bits  = 0;

    // Long blocks: 21 scalefactor bands, split by scfsi
    // bands 0..5 (scfsi band 0)
    // bands 6..10 (scfsi band 1)
    // bands 11..15 (scfsi band 2)
    // bands 16..20 (scfsi band 3)
    static const int band_start[4] = { 0, 6, 11, 16 };
    static const int band_end[4]   = { 6, 11, 16, 21 };

    for (int b = 0; b < 4; b++) {
        if (gr == 0 || scfsi[b] == 0) {
            int slen = (b < 2) ? slen1 : slen2;
            for (int sfb = band_start[b]; sfb < band_end[b]; sfb++) {
                bs.put(gi.scalefac_l[sfb], slen);
                bits += slen;
            }
        }
    }
    return bits;
}
