#pragma once
// mp3enc-huff.h
// Huffman encoding of quantized spectral values.
// ISO 11172-3 clause 2.4.2.7.
// Part of mp3enc. MIT license.

#include <cstdlib>

// Count trailing zero pairs in quantized spectrum.
// Returns number of pairs of zeros from the end.
static int mp3enc_count_rzero(const int * ix, int n) {
    int i = n - 2;
    while (i >= 0 && ix[i] == 0 && ix[i + 1] == 0) {
        i -= 2;
    }
    return (n - i - 2) / 2;
}

// Count quadruples with |val| <= 1 after big_values region.
// start: index right after big_values*2
// end: index right before rzero
// Returns number of quadruples.
static int mp3enc_count_count1(const int * ix, int start, int end) {
    int count = 0;
    int i     = end - 4;
    while (i >= start) {
        if (abs(ix[i]) <= 1 && abs(ix[i + 1]) <= 1 && abs(ix[i + 2]) <= 1 && abs(ix[i + 3]) <= 1) {
            count++;
            i -= 4;
        } else {
            break;
        }
    }
    return count;
}

// Compute bits needed to encode a pair (x, y) with a given table.
// Returns total bits (Huffman code + linbits + sign bits).
static int mp3enc_pair_bits(int table, int x, int y) {
    if (table == 0) {
        return 0;  // table 0 = all zeros, no bits emitted
    }
    int ax   = abs(x);
    int ay   = abs(y);
    int bits = 0;

    // Clamp to 15 for Huffman lookup; excess goes to linbits
    int cx = (ax < 15) ? ax : 15;
    int cy = (ay < 15) ? ay : 15;
    int lb = mp3enc_linbits[table];

    // Get Huffman code length.
    // Tables share trees; we select the right code/len arrays based on table number.
    // For phase 1, we use tables 1, 2, 5, 6, 7, 10, 13, 15 (the common choices).
    int hlen = 0;
    switch (table) {
        case 1:
            hlen = mp3enc_hlen_1[cy][cx];
            break;
        case 2:
            hlen = mp3enc_hlen_2[cy][cx];
            break;
        case 3:
            hlen = mp3enc_hlen_3[cy][cx];
            break;
        case 5:
            hlen = mp3enc_hlen_5[cy][cx];
            break;
        case 6:
            hlen = mp3enc_hlen_6[cy][cx];
            break;
        case 7:
            hlen = mp3enc_hlen_7[cy][cx];
            break;
        case 8:
            hlen = mp3enc_hlen_8[cy][cx];
            break;
        case 9:
            hlen = mp3enc_hlen_9[cy][cx];
            break;
        case 10:
            hlen = mp3enc_hlen_10[cy][cx];
            break;
        case 11:
            hlen = mp3enc_hlen_11[cy][cx];
            break;
        case 12:
            hlen = mp3enc_hlen_12[cy][cx];
            break;
        case 13:
            hlen = mp3enc_hlen_13[cy][cx];
            break;
        case 15:
            hlen = mp3enc_hlen_15[cy][cx];
            break;
        // 16..23 use table 16 tree; 24..31 use table 24 tree
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
            hlen = mp3enc_hlen_16[cy][cx];
            break;
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
            hlen = mp3enc_hlen_24[cy][cx];
            break;
        default:
            return 9999;  // invalid table
    }

    if (hlen == 0 && (cx || cy)) {
        return 9999;  // no valid code
    }

    bits = hlen;
    if (ax >= 15) {
        bits += lb;  // linbits for x
    }
    if (ay >= 15) {
        bits += lb;  // linbits for y
    }
    if (ax > 0) {
        bits += 1;  // sign bit for x
    }
    if (ay > 0) {
        bits += 1;  // sign bit for y
    }
    return bits;
}

// Write a Huffman coded pair to the bitstream.
static void mp3enc_write_pair(mp3enc_bs & bs, int table, int x, int y) {
    if (table == 0) {
        return;  // table 0 = all zeros, nothing to write
    }
    int ax = abs(x);
    int ay = abs(y);
    int cx = (ax < 15) ? ax : 15;
    int cy = (ay < 15) ? ay : 15;
    int lb = mp3enc_linbits[table];

    // Get Huffman code and length
    uint16_t code = 0;
    int      len  = 0;
    switch (table) {
        case 1:
            code = mp3enc_hcode_1[cy][cx];
            len  = mp3enc_hlen_1[cy][cx];
            break;
        case 2:
            code = mp3enc_hcode_2[cy][cx];
            len  = mp3enc_hlen_2[cy][cx];
            break;
        case 3:
            code = mp3enc_hcode_3[cy][cx];
            len  = mp3enc_hlen_3[cy][cx];
            break;
        case 5:
            code = mp3enc_hcode_5[cy][cx];
            len  = mp3enc_hlen_5[cy][cx];
            break;
        case 6:
            code = mp3enc_hcode_6[cy][cx];
            len  = mp3enc_hlen_6[cy][cx];
            break;
        case 7:
            code = mp3enc_hcode_7[cy][cx];
            len  = mp3enc_hlen_7[cy][cx];
            break;
        case 8:
            code = mp3enc_hcode_8[cy][cx];
            len  = mp3enc_hlen_8[cy][cx];
            break;
        case 9:
            code = mp3enc_hcode_9[cy][cx];
            len  = mp3enc_hlen_9[cy][cx];
            break;
        case 10:
            code = mp3enc_hcode_10[cy][cx];
            len  = mp3enc_hlen_10[cy][cx];
            break;
        case 11:
            code = mp3enc_hcode_11[cy][cx];
            len  = mp3enc_hlen_11[cy][cx];
            break;
        case 12:
            code = mp3enc_hcode_12[cy][cx];
            len  = mp3enc_hlen_12[cy][cx];
            break;
        case 13:
            code = mp3enc_hcode_13[cy][cx];
            len  = mp3enc_hlen_13[cy][cx];
            break;
        case 15:
            code = mp3enc_hcode_15[cy][cx];
            len  = mp3enc_hlen_15[cy][cx];
            break;
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
            code = mp3enc_hcode_16[cy][cx];
            len  = mp3enc_hlen_16[cy][cx];
            break;
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
            code = mp3enc_hcode_24[cy][cx];
            len  = mp3enc_hlen_24[cy][cx];
            break;
    }

    bs.put(code, len);
    if (cx == 15 && lb > 0) {
        bs.put(ax - 15, lb);
    }
    if (ax > 0) {
        bs.put(x < 0 ? 1 : 0, 1);
    }
    if (cy == 15 && lb > 0) {
        bs.put(ay - 15, lb);
    }
    if (ay > 0) {
        bs.put(y < 0 ? 1 : 0, 1);
    }
}

// Write count1 region (quadruples of {-1, 0, 1}).
static void mp3enc_write_count1(mp3enc_bs & bs, const int * ix, int start, int count, int table_sel) {
    const uint8_t * ctab_code = table_sel ? mp3enc_count1b_code : mp3enc_count1a_code;
    const uint8_t * ctab_len  = table_sel ? mp3enc_count1b_len : mp3enc_count1a_len;

    for (int q = 0; q < count; q++) {
        int i   = start + q * 4;
        int v   = abs(ix[i]);
        int w   = abs(ix[i + 1]);
        int x   = abs(ix[i + 2]);
        int y   = abs(ix[i + 3]);
        int idx = v * 8 + w * 4 + x * 2 + y;

        bs.put(ctab_code[idx], ctab_len[idx]);
        if (v) {
            bs.put(ix[i] < 0 ? 1 : 0, 1);
        }
        if (w) {
            bs.put(ix[i + 1] < 0 ? 1 : 0, 1);
        }
        if (x) {
            bs.put(ix[i + 2] < 0 ? 1 : 0, 1);
        }
        if (y) {
            bs.put(ix[i + 3] < 0 ? 1 : 0, 1);
        }
    }
}

// Choose the best Huffman table for a region of pairs.
// Tries all candidate tables and returns the one with fewest bits.
static int mp3enc_choose_table(const int * ix, int start, int count) {
    if (count <= 0) {
        return 0;
    }

    // Find max absolute value in this region
    int maxval = 0;
    for (int i = start; i < start + count * 2; i++) {
        int a = abs(ix[i]);
        if (a > maxval) {
            maxval = a;
        }
    }

    if (maxval == 0) {
        return 0;
    }

    // Candidate tables grouped by max entry:
    // max 1:  tables 1
    // max 2:  tables 2, 3
    // max 3:  tables 5, 6
    // max 5:  tables 7, 8, 9
    // max 7:  tables 10, 11, 12
    // max 15: tables 13, 15
    // max >15: tables 16..31 (with linbits)
    static const int candidates[][8] = {
        { 1, 0 }, // maxval = 1
        { 2, 3, 0 }, // maxval = 2
        { 5, 6, 0 }, // maxval = 3
        { 7, 8, 9, 0 }, // maxval = 4..5
        { 7, 8, 9, 0 }, // (same)
        { 10, 11, 12, 0 }, // maxval = 6..7
        { 10, 11, 12, 0 }, // (same)
        { 13, 15, 0 }, // maxval = 8..15
    };
    // For values > 15, need linbits tables
    static const int linbit_tables[] = { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0 };

    const int * cands;
    if (maxval <= 1) {
        cands = candidates[0];
    } else if (maxval <= 2) {
        cands = candidates[1];
    } else if (maxval <= 3) {
        cands = candidates[2];
    } else if (maxval <= 5) {
        cands = candidates[3];
    } else if (maxval <= 7) {
        cands = candidates[5];
    } else if (maxval <= 15) {
        cands = candidates[7];
    } else {
        cands = linbit_tables;
    }

    int best_table = cands[0];
    int best_bits  = 999999;

    for (int c = 0; cands[c] != 0; c++) {
        int t = cands[c];
        // For linbits tables, skip if linbits too small for our max value
        if (maxval > 15 && mp3enc_linbits[t] > 0) {
            int max_encodable = 15 + (1 << mp3enc_linbits[t]) - 1;
            if (maxval > max_encodable) {
                continue;
            }
        }

        int total = 0;
        for (int p = 0; p < count; p++) {
            total += mp3enc_pair_bits(t, ix[start + p * 2], ix[start + p * 2 + 1]);
        }
        if (total < best_bits) {
            best_bits  = total;
            best_table = t;
        }
    }
    return best_table;
}
