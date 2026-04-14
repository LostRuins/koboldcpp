// gen_final.c: correct Huffman encoder table generator
// The key insight: leaf>>8 is only the FINAL flush.
// Total bits = sum of all intermediate flushes + leaf>>8.
// Only the first peek (5 bits) is NOT flushed if we get a direct hit.

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "tabs_data.h"

static const int tmax[] = {
    0, 1, 2, 2, 0, 3, 3, 5, 5, 5, 7, 7, 7, 15, 0, 15,
    15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15
};

typedef struct { uint32_t code; uint8_t len; } henc_t;
static henc_t enc[16][16];

// Walk the minimp3 tree and return {leaf, total_bits_consumed}
// Returns -1 if cannot decode
static int walk_tree(const int16_t *codebook, uint32_t bits, int nbits,
                     int *total_consumed) {
    uint32_t cache = (nbits < 32) ? (bits << (32 - nbits)) : bits;
    int w = 5;
    int leaf = codebook[cache >> (32 - w)];
    int flushed = 0; // bits flushed so far

    if (leaf >= 0) {
        // Direct hit: only flush leaf>>8 bits (subset of the initial 5 peeked)
        *total_consumed = leaf >> 8;
        return leaf;
    }

    // Tree traversal: flush the initial 5 bits, then follow nodes
    cache <<= w;
    flushed = w;

    while (leaf < 0) {
        w = leaf & 7;
        if (w == 0) return -1;
        int peek = (int)(cache >> (32 - w));
        int idx = peek - (leaf >> 3);
        leaf = codebook[idx];

        if (leaf < 0) {
            // Not a leaf yet, flush these w bits and continue
            cache <<= w;
            flushed += w;
            if (flushed > 30) return -1;
        }
    }

    // leaf is positive: flush leaf>>8 more bits
    *total_consumed = flushed + (leaf >> 8);
    return leaf;
}

static void build_pair_table(int tab_num) {
    int mx = tmax[tab_num];
    memset(enc, 0, sizeof(enc));
    const int16_t *codebook = tabs + tabindex[tab_num];

    for (int len = 1; len <= 19; len++) {
        for (uint32_t code = 0; code < (1u << len); code++) {
            int total = 0;
            int leaf = walk_tree(codebook, code, len, &total);
            if (leaf < 0) continue;
            if (total != len) continue;
            int x = (leaf >> 4) & 0xF;
            int y = leaf & 0xF;
            if (x > mx || y > mx) continue;
            if (enc[x][y].len == 0) {
                enc[x][y].code = code;
                enc[x][y].len = (uint8_t)len;
            }
        }
    }
}

int main(void) {
    // Verify completeness first
    int ok = 1;
    for (int t = 1; t < 32; t++) {
        if (t == 0 || t == 4 || t == 14) continue;
        int mx = tmax[t]; if (mx == 0) continue;
        build_pair_table(t);
        int dim = mx + 1, found = 0;
        for (int x = 0; x < dim; x++)
            for (int y = 0; y < dim; y++)
                if (enc[x][y].len > 0) found++;
        int expected = dim * dim;
        if (found != expected) {
            fprintf(stderr, "Table %2d: %d/%d INCOMPLETE\n", t, found, expected);
            ok = 0;
        } else {
            fprintf(stderr, "Table %2d: %d/%d OK\n", t, found, expected);
        }
    }
    if (!ok) {
        fprintf(stderr, "WARNING: some tables incomplete\n");
    }

    // Generate output
    printf("// mp3 Huffman encoder lookup tables\n");
    printf("// Generated from minimp3 (CC0). ISO 11172-3 Table B.7.\n");
    printf("// code = MSB first bit pattern. len = number of bits.\n");
    printf("// Append sign bits for nonzero |x| and |y|.\n");
    printf("// Values >= 15 use linbits extension.\n\n");

    printf("static const uint8_t mp3enc_linbits[32] = {\n");
    printf("    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n");
    printf("    1,2,3,4,6,8,10,13,4,5,6,7,8,9,11,13\n};\n\n");

    for (int t = 1; t < 32; t++) {
        if (t == 0 || t == 4 || t == 14) continue;
        int mx = tmax[t]; if (mx == 0) continue;
        // Skip duplicate trees
        int skip = 0;
        for (int j = 1; j < t; j++) {
            if (j == 4 || j == 14) continue;
            if (tabindex[t] == tabindex[j] && tmax[t] == tmax[j]) {
                printf("// table %d: same tree as table %d (linbits=%d)\n\n", t, j, g_linbits[t]);
                skip = 1; break;
            }
        }
        if (skip) continue;

        build_pair_table(t);
        int dim = mx + 1;
        printf("static const uint16_t mp3enc_hcode_%d[%d][%d] = {\n", t, dim, dim);
        for (int x = 0; x < dim; x++) {
            printf("    {");
            for (int y = 0; y < dim; y++) {
                printf("0x%x", enc[x][y].code);
                if (y < dim-1) printf(",");
            }
            printf("}%s\n", x < dim-1 ? "," : "");
        }
        printf("};\nstatic const uint8_t mp3enc_hlen_%d[%d][%d] = {\n", t, dim, dim);
        for (int x = 0; x < dim; x++) {
            printf("    {");
            for (int y = 0; y < dim; y++) {
                printf("%d", enc[x][y].len);
                if (y < dim-1) printf(",");
            }
            printf("}%s\n", x < dim-1 ? "," : "");
        }
        printf("};\n\n");
    }

    // Count1 tables
    for (int tbl = 0; tbl < 2; tbl++) {
        const uint8_t *cb = tbl ? tab33 : tab32;
        printf("// count1 table %c (count1table_select=%d)\n", 'A'+tbl, tbl);
        printf("// index = v*8 + w*4 + x*2 + y\n");
        uint32_t codes[16]={0}; uint8_t lens[16]={0};
        for (int len = 1; len <= 10; len++) {
            for (uint32_t code = 0; code < (1u<<len); code++) {
                uint32_t cache = (len<32) ? (code<<(32-len)) : code;
                int peek4 = (int)(cache>>28);
                int leaf = cb[peek4];
                int consumed;
                if (leaf & 8) { consumed = leaf & 7; }
                else {
                    int extra = leaf & 3, offset = leaf >> 3;
                    cache <<= 4;
                    int peek2 = (int)(cache >> (32-extra));
                    leaf = cb[offset + peek2];
                    consumed = leaf & 7;
                }
                if (consumed != len) continue;
                int v=(leaf>>7)&1, w=(leaf>>6)&1, x=(leaf>>5)&1, y=(leaf>>4)&1;
                int idx = v*8+w*4+x*2+y;
                if (lens[idx]==0) { codes[idx]=code; lens[idx]=(uint8_t)consumed; }
            }
        }
        printf("static const uint8_t mp3enc_count1%c_code[16] = {\n    ", 'a'+tbl);
        for (int i=0;i<16;i++) { printf("%d",codes[i]); if(i<15) printf(","); if(i==7) printf("\n    "); }
        printf("\n};\nstatic const uint8_t mp3enc_count1%c_len[16] = {\n    ", 'a'+tbl);
        for (int i=0;i<16;i++) { printf("%d",lens[i]); if(i<15) printf(","); if(i==7) printf("\n    "); }
        printf("\n};\n\n");
    }
    return 0;
}
