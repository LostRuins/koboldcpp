//
// gif_memory.h
// Modified from charlietangora/gif-h to write GIFs entirely to memory (no disk IO)
// Public domain adaptation
//
// Changes:
// - All file I/O replaced with an in-memory writer (growable buffer).
// - GifBegin now initializes an in-memory GIF writer.
// - Provided helper create_gif_buf_from_sd_images(...) that builds a GIF from an
//   array of sd_image_t frames and returns an allocated buffer (caller must free).
// - Added simple heuristics for `quality` -> bitDepth & dithering, and fps -> delay
//
#ifndef GIF_MEMORY_H
#define GIF_MEMORY_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

// Memory writer structure (growable buffer)
typedef struct MemFile
{
    uint8_t* data;    // allocated buffer
    size_t len;       // number of bytes written
    size_t cap;       // buffer capacity
} MemFile;

static bool memfile_init(MemFile* mf, size_t initial_cap)
{
    mf->data = (uint8_t*)malloc(initial_cap);
    if(!mf->data) return false;
    mf->len = 0;
    mf->cap = initial_cap;
    return true;
}

static void memfile_free(MemFile* mf)
{
    if(!mf) return;
    if(mf->data) free(mf->data);
    mf->data = NULL;
    mf->len = 0;
    mf->cap = 0;
}

static bool memfile_ensure(MemFile* mf, size_t extra)
{
    if(mf->len + extra <= mf->cap) return true;
    size_t newcap = mf->cap ? mf->cap : 1024;
    while(newcap < mf->len + extra) newcap *= 2;
    uint8_t* newdata = (uint8_t*)realloc(mf->data, newcap);
    if(!newdata) return false;
    mf->data = newdata;
    mf->cap = newcap;
    return true;
}

static bool memfile_write(MemFile* mf, const void* src, size_t bytes)
{
    if(!memfile_ensure(mf, bytes)) return false;
    memcpy(mf->data + mf->len, src, bytes);
    mf->len += bytes;
    return true;
}

static bool memfile_putc(MemFile* mf, int c)
{
    uint8_t b = (uint8_t)c;
    return memfile_write(mf, &b, 1);
}

static uint8_t* memfile_detach(MemFile* mf, size_t* out_len)
{
    // Transfer ownership of the exact-sized buffer to the caller
    if(!mf->data)
    {
        if(out_len) *out_len = 0;
        return NULL;
    }
    uint8_t* exact = (uint8_t*)malloc(mf->len);
    if(!exact) return NULL;
    memcpy(exact, mf->data, mf->len);
    if(out_len) *out_len = mf->len;
    // free internal buffer
    free(mf->data);
    mf->data = NULL;
    mf->len = 0;
    mf->cap = 0;
    return exact;
}

// allow custom allocators like original file
#ifndef GIF_TEMP_MALLOC
#define GIF_TEMP_MALLOC malloc
#endif
#ifndef GIF_TEMP_FREE
#define GIF_TEMP_FREE free
#endif
#ifndef GIF_MALLOC
#define GIF_MALLOC malloc
#endif
#ifndef GIF_FREE
#define GIF_FREE free
#endif

const int kGifTransIndex = 0;

typedef struct
{
    int bitDepth;

    uint8_t r[256];
    uint8_t g[256];
    uint8_t b[256];

    uint8_t treeSplitElt[256];
    uint8_t treeSplit[256];
} GifPalette;

// max/min/abs
static int GifIMax(int l, int r) { return l>r?l:r; }
static int GifIMin(int l, int r) { return l<r?l:r; }
static int GifIAbs(int i) { return i<0?-i:i; }

// forward declare

typedef struct GifWriter GifWriter;

// LZW writing state
typedef struct
{
    uint32_t chunkIndex;
    uint8_t chunk[256];

    uint8_t bitIndex;
    uint8_t byte;

    uint8_t padding[2];
} GifBitStatus;

// Gif writer contains an in-memory memfile rather than FILE*
struct GifWriter
{
    MemFile mem;
    uint8_t* oldImage; // palettized image used for delta encoding
    bool firstFrame;
    uint8_t padding[7];
};

// --- Palette / quantization / dithering code (almost unchanged) ---

void GifGetClosestPaletteColor( GifPalette* pPal, int r, int g, int b, int* bestInd, int* bestDiff, int treeRoot )
{
    if(treeRoot > (1<<pPal->bitDepth)-1)
    {
        int ind = treeRoot-(1<<pPal->bitDepth);
        if(ind == kGifTransIndex) return;

        int r_err = r - ((int32_t)pPal->r[ind]);
        int g_err = g - ((int32_t)pPal->g[ind]);
        int b_err = b - ((int32_t)pPal->b[ind]);
        int diff = GifIAbs(r_err)+GifIAbs(g_err)+GifIAbs(b_err);

        if(diff < *bestDiff)
        {
            *bestInd = ind;
            *bestDiff = diff;
        }

        return;
    }

    int comps[3]; comps[0] = r; comps[1] = g; comps[2] = b;
    int splitComp = comps[pPal->treeSplitElt[treeRoot]];

    int splitPos = pPal->treeSplit[treeRoot];
    if(splitPos > splitComp)
    {
        GifGetClosestPaletteColor(pPal, r, g, b, bestInd, bestDiff, treeRoot*2);
        if( *bestDiff > splitPos - splitComp )
        {
            GifGetClosestPaletteColor(pPal, r, g, b, bestInd, bestDiff, treeRoot*2+1);
        }
    }
    else
    {
        GifGetClosestPaletteColor(pPal, r, g, b, bestInd, bestDiff, treeRoot*2+1);
        if( *bestDiff > splitComp - splitPos )
        {
            GifGetClosestPaletteColor(pPal, r, g, b, bestInd, bestDiff, treeRoot*2);
        }
    }
}

void GifSwapPixels(uint8_t* image, int pixA, int pixB)
{
    uint8_t rA = image[pixA*4];
    uint8_t gA = image[pixA*4+1];
    uint8_t bA = image[pixA*4+2];
    uint8_t aA = image[pixA*4+3];

    uint8_t rB = image[pixB*4];
    uint8_t gB = image[pixB*4+1];
    uint8_t bB = image[pixB*4+2];
    uint8_t aB = image[pixB*4+3];

    image[pixA*4] = rB;
    image[pixA*4+1] = gB;
    image[pixA*4+2] = bB;
    image[pixA*4+3] = aB;

    image[pixB*4] = rA;
    image[pixB*4+1] = gA;
    image[pixB*4+2] = bA;
    image[pixB*4+3] = aA;
}

int GifPartition(uint8_t* image, const int left, const int right, const int elt, int pivotValue)
{
    int storeIndex = left;
    bool split = 0;
    for(int ii=left; ii<right; ++ii)
    {
        int arrayVal = image[ii*4+elt];
        if( arrayVal < pivotValue )
        {
            GifSwapPixels(image, ii, storeIndex);
            ++storeIndex;
        }
        else if( arrayVal == pivotValue )
        {
            if(split)
            {
                GifSwapPixels(image, ii, storeIndex);
                ++storeIndex;
            }
            split = !split;
        }
    }
    return storeIndex;
}

void GifPartitionByMedian(uint8_t* image, int left, int right, int com, int neededCenter)
{
    if(left < right-1)
    {
        int pivotValue = image[(neededCenter)*4+com];
        GifSwapPixels(image, neededCenter, right-1);
        int pivotIndex = GifPartition(image, left, right-1, com, pivotValue);
        GifSwapPixels(image, pivotIndex, right-1);

        if(pivotIndex > neededCenter)
            GifPartitionByMedian(image, left, pivotIndex, com, neededCenter);

        if(pivotIndex < neededCenter)
            GifPartitionByMedian(image, pivotIndex+1, right, com, neededCenter);
    }
}

int GifPartitionByMean(uint8_t* image, int left, int right, int com, int neededMean)
{
    if(left < right-1)
    {
        return GifPartition(image, left, right-1, com, neededMean);
    }
    return left;
}

void GifSplitPalette(uint8_t* image, int numPixels, int treeNode, int treeLevel, bool buildForDither, GifPalette* pal)
{
    if(numPixels == 0)
        return;

    int numColors = (1 << pal->bitDepth);

    if(treeNode >= numColors)
    {
        int entry = treeNode - numColors;

        if(buildForDither)
        {
            if( entry == 1 )
            {
                uint32_t r=255, g=255, b=255;
                for(int ii=0; ii<numPixels; ++ii)
                {
                    r = (uint32_t)GifIMin((int32_t)r, image[ii * 4 + 0]);
                    g = (uint32_t)GifIMin((int32_t)g, image[ii * 4 + 1]);
                    b = (uint32_t)GifIMin((int32_t)b, image[ii * 4 + 2]);
                }

                pal->r[entry] = (uint8_t)r;
                pal->g[entry] = (uint8_t)g;
                pal->b[entry] = (uint8_t)b;

                return;
            }

            if( entry == numColors-1 )
            {
                uint32_t r=0, g=0, b=0;
                for(int ii=0; ii<numPixels; ++ii)
                {
                    r = (uint32_t)GifIMax((int32_t)r, image[ii * 4 + 0]);
                    g = (uint32_t)GifIMax((int32_t)g, image[ii * 4 + 1]);
                    b = (uint32_t)GifIMax((int32_t)b, image[ii * 4 + 2]);
                }

                pal->r[entry] = (uint8_t)r;
                pal->g[entry] = (uint8_t)g;
                pal->b[entry] = (uint8_t)b;

                return;
            }
        }

        uint64_t r=0, g=0, b=0;
        for(int ii=0; ii<numPixels; ++ii)
        {
            r += image[ii*4+0];
            g += image[ii*4+1];
            b += image[ii*4+2];
        }

        r += (uint64_t)numPixels / 2;
        g += (uint64_t)numPixels / 2;
        b += (uint64_t)numPixels / 2;

        r /= (uint64_t)numPixels;
        g /= (uint64_t)numPixels;
        b /= (uint64_t)numPixels;

        pal->r[entry] = (uint8_t)r;
        pal->g[entry] = (uint8_t)g;
        pal->b[entry] = (uint8_t)b;

        return;
    }

    int minR = 255, maxR = 0;
    int minG = 255, maxG = 0;
    int minB = 255, maxB = 0;
    for(int ii=0; ii<numPixels; ++ii)
    {
        int r = image[ii*4+0];
        int g = image[ii*4+1];
        int b = image[ii*4+2];

        if(r > maxR) maxR = r;
        if(r < minR) minR = r;

        if(g > maxG) maxG = g;
        if(g < minG) minG = g;

        if(b > maxB) maxB = b;
        if(b < minB) minB = b;
    }

    int rRange = maxR - minR;
    int gRange = maxG - minG;
    int bRange = maxB - minB;

    int splitCom = 1; int rangeMin = minG; int rangeMax = maxG;
    if(bRange > gRange) { splitCom = 2; rangeMin = minB; rangeMax = maxB; }
    if(rRange > bRange && rRange > gRange) { splitCom = 0; rangeMin = minR; rangeMax = maxR; }

    int subPixelsA = numPixels / 2;

    GifPartitionByMedian(image, 0, numPixels, splitCom, subPixelsA);
    int splitValue = image[subPixelsA*4+splitCom];

    int splitUnbalance = GifIAbs( (splitValue - rangeMin) - (rangeMax - splitValue) );
    if( splitUnbalance > (1536 >> treeLevel) )
    {
        splitValue = rangeMin + (rangeMax-rangeMin) / 2;
        subPixelsA = GifPartitionByMean(image, 0, numPixels, splitCom, splitValue);
    }

    if( treeNode == numColors/2 )
    {
        subPixelsA = 0;
        splitValue = 0;
    }

    int subPixelsB = numPixels-subPixelsA;
    pal->treeSplitElt[treeNode] = (uint8_t)splitCom;
    pal->treeSplit[treeNode] = (uint8_t)splitValue;

    GifSplitPalette(image,              subPixelsA, treeNode*2,   treeLevel+1, buildForDither, pal);
    GifSplitPalette(image+subPixelsA*4, subPixelsB, treeNode*2+1, treeLevel+1, buildForDither, pal);
}

int GifPickChangedPixels( const uint8_t* lastFrame, uint8_t* frame, int numPixels )
{
    int numChanged = 0;
    uint8_t* writeIter = frame;

    for (int ii=0; ii<numPixels; ++ii)
    {
        if(lastFrame[0] != frame[0] ||
           lastFrame[1] != frame[1] ||
           lastFrame[2] != frame[2])
        {
            writeIter[0] = frame[0];
            writeIter[1] = frame[1];
            writeIter[2] = frame[2];
            ++numChanged;
            writeIter += 4;
        }
        lastFrame += 4;
        frame += 4;
    }

    return numChanged;
}

void GifMakePalette( const uint8_t* lastFrame, const uint8_t* nextFrame, uint32_t width, uint32_t height, int bitDepth, bool buildForDither, GifPalette* pPal )
{
    pPal->bitDepth = bitDepth;

    size_t imageSize = (size_t)(width * height * 4 * sizeof(uint8_t));
    uint8_t* destroyableImage = (uint8_t*)GIF_TEMP_MALLOC(imageSize);
    memcpy(destroyableImage, nextFrame, imageSize);

    int numPixels = (int)(width * height);
    if(lastFrame)
        numPixels = GifPickChangedPixels(lastFrame, destroyableImage, numPixels);

    GifSplitPalette(destroyableImage, numPixels, 1, 0, buildForDither, pPal);

    GIF_TEMP_FREE(destroyableImage);

    pPal->treeSplit[1 << (bitDepth-1)] = 0;
    pPal->treeSplitElt[1 << (bitDepth-1)] = 0;

    pPal->r[0] = pPal->g[0] = pPal->b[0] = 0;
}

void GifDitherImage( const uint8_t* lastFrame, const uint8_t* nextFrame, uint8_t* outFrame, uint32_t width, uint32_t height, GifPalette* pPal )
{
    int numPixels = (int)(width * height);

    int32_t *quantPixels = (int32_t *)GIF_TEMP_MALLOC(sizeof(int32_t) * (size_t)numPixels * 4);

    for( int ii=0; ii<numPixels*4; ++ii )
    {
        uint8_t pix = nextFrame[ii];
        int32_t pix16 = (int32_t)(pix) * 256;
        quantPixels[ii] = pix16;
    }

    for( uint32_t yy=0; yy<height; ++yy )
    {
        for( uint32_t xx=0; xx<width; ++xx )
        {
            int32_t* nextPix = quantPixels + 4*(yy*width+xx);
            const uint8_t* lastPix = lastFrame? lastFrame + 4*(yy*width+xx) : NULL;

            int32_t rr = (nextPix[0] + 127) / 256;
            int32_t gg = (nextPix[1] + 127) / 256;
            int32_t bb = (nextPix[2] + 127) / 256;

            if( lastFrame &&
               lastPix[0] == rr &&
               lastPix[1] == gg &&
               lastPix[2] == bb )
            {
                nextPix[0] = rr;
                nextPix[1] = gg;
                nextPix[2] = bb;
                nextPix[3] = kGifTransIndex;
                continue;
            }

            int32_t bestDiff = 1000000;
            int32_t bestInd = kGifTransIndex;

            GifGetClosestPaletteColor(pPal, rr, gg, bb, &bestInd, &bestDiff, 1);

            int32_t r_err = nextPix[0] - (int32_t)(pPal->r[bestInd]) * 256;
            int32_t g_err = nextPix[1] - (int32_t)(pPal->g[bestInd]) * 256;
            int32_t b_err = nextPix[2] - (int32_t)(pPal->b[bestInd]) * 256;

            nextPix[0] = pPal->r[bestInd];
            nextPix[1] = pPal->g[bestInd];
            nextPix[2] = pPal->b[bestInd];
            nextPix[3] = bestInd;

            int quantloc_7 = (int)(yy * width + xx + 1);
            int quantloc_3 = (int)(yy * width + width + xx - 1);
            int quantloc_5 = (int)(yy * width + width + xx);
            int quantloc_1 = (int)(yy * width + width + xx + 1);

            if(quantloc_7 < numPixels)
            {
                int32_t* pix7 = quantPixels+4*quantloc_7;
                pix7[0] += GifIMax( -pix7[0], r_err * 7 / 16 );
                pix7[1] += GifIMax( -pix7[1], g_err * 7 / 16 );
                pix7[2] += GifIMax( -pix7[2], b_err * 7 / 16 );
            }

            if(quantloc_3 < numPixels)
            {
                int32_t* pix3 = quantPixels+4*quantloc_3;
                pix3[0] += GifIMax( -pix3[0], r_err * 3 / 16 );
                pix3[1] += GifIMax( -pix3[1], g_err * 3 / 16 );
                pix3[2] += GifIMax( -pix3[2], b_err * 3 / 16 );
            }

            if(quantloc_5 < numPixels)
            {
                int32_t* pix5 = quantPixels+4*quantloc_5;
                pix5[0] += GifIMax( -pix5[0], r_err * 5 / 16 );
                pix5[1] += GifIMax( -pix5[1], g_err * 5 / 16 );
                pix5[2] += GifIMax( -pix5[2], b_err * 5 / 16 );
            }

            if(quantloc_1 < numPixels)
            {
                int32_t* pix1 = quantPixels+4*quantloc_1;
                pix1[0] += GifIMax( -pix1[0], r_err / 16 );
                pix1[1] += GifIMax( -pix1[1], g_err / 16 );
                pix1[2] += GifIMax( -pix1[2], b_err / 16 );
            }
        }
    }

    for( int ii=0; ii<numPixels*4; ++ii )
    {
        outFrame[ii] = (uint8_t)quantPixels[ii];
    }

    GIF_TEMP_FREE(quantPixels);
}

void GifThresholdImage( const uint8_t* lastFrame, const uint8_t* nextFrame, uint8_t* outFrame, uint32_t width, uint32_t height, GifPalette* pPal )
{
    uint32_t numPixels = width*height;
    for( uint32_t ii=0; ii<numPixels; ++ii )
    {
        if(lastFrame &&
           lastFrame[0] == nextFrame[0] &&
           lastFrame[1] == nextFrame[1] &&
           lastFrame[2] == nextFrame[2])
        {
            outFrame[0] = lastFrame[0];
            outFrame[1] = lastFrame[1];
            outFrame[2] = lastFrame[2];
            outFrame[3] = kGifTransIndex;
        }
        else
        {
            int32_t bestDiff = 1000000;
            int32_t bestInd = 1;
            GifGetClosestPaletteColor(pPal, nextFrame[0], nextFrame[1], nextFrame[2], &bestInd, &bestDiff, 1);

            outFrame[0] = pPal->r[bestInd];
            outFrame[1] = pPal->g[bestInd];
            outFrame[2] = pPal->b[bestInd];
            outFrame[3] = (uint8_t)bestInd;
        }

        if(lastFrame) lastFrame += 4;
        outFrame += 4;
        nextFrame += 4;
    }
}

// --- LZW and writing to MemFile ---

void GifWriteBit( GifBitStatus* stat, uint32_t bit )
{
    bit = bit & 1;
    bit = bit << stat->bitIndex;
    stat->byte |= bit;

    ++stat->bitIndex;
    if( stat->bitIndex > 7 )
    {
        stat->chunk[stat->chunkIndex++] = stat->byte;
        stat->bitIndex = 0;
        stat->byte = 0;
    }
}

static bool GifWriteChunkToMem(MemFile* mf, GifBitStatus* stat)
{
    // write chunk length and bytes
    if(!memfile_putc(mf, (int)stat->chunkIndex)) return false;
    if(!memfile_write(mf, stat->chunk, stat->chunkIndex)) return false;
    stat->bitIndex = 0;
    stat->byte = 0;
    stat->chunkIndex = 0;
    return true;
}

void GifWriteCodeMem( MemFile* mf, GifBitStatus* stat, uint32_t code, uint32_t length )
{
    for( uint32_t ii=0; ii<length; ++ii )
    {
        GifWriteBit(stat, code);
        code = code >> 1;

        if( stat->chunkIndex == 255 )
        {
            // flush chunk to mem
            GifWriteChunkToMem(mf, stat);
        }
    }
}

typedef struct
{
    uint16_t m_next[256];
} GifLzwNode;

static bool GifWritePaletteMem( const GifPalette* pPal, MemFile* mf )
{
    // transparency color
    if(!memfile_putc(mf, 0)) return false;
    if(!memfile_putc(mf, 0)) return false;
    if(!memfile_putc(mf, 0)) return false;

    for(int ii=1; ii<(1 << pPal->bitDepth); ++ii)
    {
        if(!memfile_putc(mf, pPal->r[ii])) return false;
        if(!memfile_putc(mf, pPal->g[ii])) return false;
        if(!memfile_putc(mf, pPal->b[ii])) return false;
    }

    return true;
}

static bool GifWriteLzwImageMem(MemFile* mf, uint8_t* image, uint32_t left, uint32_t top,  uint32_t width, uint32_t height, uint32_t delay, GifPalette* pPal)
{
    // graphics control extension
    if(!memfile_putc(mf, 0x21)) return false;
    if(!memfile_putc(mf, 0xf9)) return false;
    if(!memfile_putc(mf, 0x04)) return false;
    if(!memfile_putc(mf, 0x05)) return false; // leave prev frame in place, has transparency
    if(!memfile_putc(mf, delay & 0xff)) return false;
    if(!memfile_putc(mf, (delay >> 8) & 0xff)) return false;
    if(!memfile_putc(mf, kGifTransIndex)) return false; // transparent index
    if(!memfile_putc(mf, 0)) return false;

    if(!memfile_putc(mf, 0x2c)) return false; // image descriptor block

    if(!memfile_putc(mf, left & 0xff)) return false;
    if(!memfile_putc(mf, (left >> 8) & 0xff)) return false;
    if(!memfile_putc(mf, top & 0xff)) return false;
    if(!memfile_putc(mf, (top >> 8) & 0xff)) return false;

    if(!memfile_putc(mf, width & 0xff)) return false;
    if(!memfile_putc(mf, (width >> 8) & 0xff)) return false;
    if(!memfile_putc(mf, height & 0xff)) return false;
    if(!memfile_putc(mf, (height >> 8) & 0xff)) return false;

    // local color table present, size = 2^bitDepth
    if(!memfile_putc(mf, 0x80 + pPal->bitDepth-1)) return false;

    if(!GifWritePaletteMem(pPal, mf)) return false;

    const int minCodeSize = pPal->bitDepth;
    const uint32_t clearCode = 1 << pPal->bitDepth;

    if(!memfile_putc(mf, minCodeSize)) return false; // min code size

    GifLzwNode* codetree = (GifLzwNode*)GIF_TEMP_MALLOC(sizeof(GifLzwNode)*4096);
    if(!codetree) return false;
    memset(codetree, 0, sizeof(GifLzwNode)*4096);

    int32_t curCode = -1;
    uint32_t codeSize = (uint32_t)minCodeSize + 1;
    uint32_t maxCode = clearCode+1;

    GifBitStatus stat;
    stat.byte = 0;
    stat.bitIndex = 0;
    stat.chunkIndex = 0;

    // write initial clear code
    GifWriteCodeMem(mf, &stat, clearCode, codeSize);

    for(uint32_t yy=0; yy<height; ++yy)
    {
        for(uint32_t xx=0; xx<width; ++xx)
        {
#ifdef GIF_FLIP_VERT
            uint8_t nextValue = image[((height-1-yy)*width+xx)*4+3];
#else
            uint8_t nextValue = image[(yy*width+xx)*4+3];
#endif
            if( curCode < 0 )
            {
                curCode = nextValue;
            }
            else if( codetree[curCode].m_next[nextValue] )
            {
                curCode = codetree[curCode].m_next[nextValue];
            }
            else
            {
                GifWriteCodeMem(mf, &stat, (uint32_t)curCode, codeSize);

                codetree[curCode].m_next[nextValue] = (uint16_t)++maxCode;

                if( maxCode >= (1ul << codeSize) )
                {
                    codeSize++;
                }
                if( maxCode == 4095 )
                {
                    GifWriteCodeMem(mf, &stat, clearCode, codeSize);

                    memset(codetree, 0, sizeof(GifLzwNode)*4096);
                    codeSize = (uint32_t)(minCodeSize + 1);
                    maxCode = clearCode+1;
                }

                curCode = nextValue;
            }
        }
    }

    GifWriteCodeMem(mf, &stat, (uint32_t)curCode, codeSize);
    GifWriteCodeMem(mf, &stat, clearCode, codeSize);
    GifWriteCodeMem(mf, &stat, clearCode + 1, (uint32_t)minCodeSize + 1);

    while( stat.bitIndex ) GifWriteBit(&stat, 0);
    if( stat.chunkIndex ) GifWriteChunkToMem(mf, &stat);

    if(!memfile_putc(mf, 0)) { GIF_TEMP_FREE(codetree); return false; } // image block terminator

    GIF_TEMP_FREE(codetree);
    return true;
}

// --- Public API: GifBegin / GifWriteFrame / GifEnd (memory-based) ---

// Initialize a GifWriter that writes to memory. The memory buffer is owned by the writer
// until you call memfile_detach on writer->mem (or use GifEnd + memfile_detach inside helper)
static bool GifBegin( GifWriter* writer, uint32_t width, uint32_t height, uint32_t delay, int32_t bitDepth, bool dither )
{
    (void)bitDepth; (void)dither;
    if(!writer) return false;
    size_t approx = (size_t)width * (size_t)height * 1; // rough initial guess
    if(approx < 16384) approx = 16384;
    if(!memfile_init(&writer->mem, approx)) return false;

    writer->firstFrame = true;
    writer->oldImage = (uint8_t*)GIF_MALLOC(width*height*4);
    if(!writer->oldImage) { memfile_free(&writer->mem); return false; }

    // header
    if(!memfile_write(&writer->mem, "GIF89a", 6)) return false;

    // screen descriptor (width, height)
    if(!memfile_putc(&writer->mem, width & 0xff)) return false;
    if(!memfile_putc(&writer->mem, (width >> 8) & 0xff)) return false;
    if(!memfile_putc(&writer->mem, height & 0xff)) return false;
    if(!memfile_putc(&writer->mem, (height >> 8) & 0xff)) return false;

    if(!memfile_putc(&writer->mem, 0xf0)) return false;  // global color table flag (dummy)
    if(!memfile_putc(&writer->mem, 0)) return false;     // background
    if(!memfile_putc(&writer->mem, 0)) return false;     // aspect

    // global palette (dummy two black entries)
    if(!memfile_putc(&writer->mem, 0)) return false;
    if(!memfile_putc(&writer->mem, 0)) return false;
    if(!memfile_putc(&writer->mem, 0)) return false;
    if(!memfile_putc(&writer->mem, 0)) return false;
    if(!memfile_putc(&writer->mem, 0)) return false;
    if(!memfile_putc(&writer->mem, 0)) return false;

    if( delay != 0 )
    {
        // NETSCAPE2.0 application extension for looping
        if(!memfile_putc(&writer->mem, 0x21)) return false; // extension
        if(!memfile_putc(&writer->mem, 0xff)) return false; // application
        if(!memfile_putc(&writer->mem, 11)) return false; // length
        if(!memfile_write(&writer->mem, "NETSCAPE2.0", 11)) return false;
        if(!memfile_putc(&writer->mem, 3)) return false; // data length
        if(!memfile_putc(&writer->mem, 1)) return false; // sub-block ID
        if(!memfile_putc(&writer->mem, 0)) return false; // loop count low (0 = infinite)
        if(!memfile_putc(&writer->mem, 0)) return false; // loop count high
        if(!memfile_putc(&writer->mem, 0)) return false; // terminator
    }

    return true;
}

static bool GifWriteFrame( GifWriter* writer, const uint8_t* image, uint32_t width, uint32_t height, uint32_t delay, int bitDepth, bool dither )
{
    if(!writer) return false;

    const uint8_t* oldImage = writer->firstFrame? NULL : writer->oldImage;
    writer->firstFrame = false;

    GifPalette pal;
    GifMakePalette((dither? NULL : oldImage), image, width, height, bitDepth, dither, &pal);

    if(dither)
        GifDitherImage(oldImage, image, writer->oldImage, width, height, &pal);
    else
        GifThresholdImage(oldImage, image, writer->oldImage, width, height, &pal);

    if(!GifWriteLzwImageMem(&writer->mem, writer->oldImage, 0, 0, width, height, delay, &pal)) return false;

    return true;
}

static bool GifEnd( GifWriter* writer )
{
    if(!writer) return false;
    if(!memfile_putc(&writer->mem, 0x3b)) return false; // EOF

    if(writer->oldImage) GIF_FREE(writer->oldImage);
    writer->oldImage = NULL;
    writer->firstFrame = true;

    // keep mem buffer available for detaching by caller
    return true;
}


#endif // GIF_MEMORY_H
