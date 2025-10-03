#ifndef __AVI_WRITER_H__
#define __AVI_WRITER_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stable-diffusion.h"
#include "./gif.h"   // charlietangora/gif-h

#ifndef MSF_GIF_IMPL
#define MSF_GIF_IMPL
#endif
#include "./msf_gif.h" //notnullnotvoid/msf_gif


#ifndef INCLUDE_STB_IMAGE_WRITE_H
#include "stb_image_write.h"
#endif

typedef struct {
    uint32_t offset;
    uint32_t size;
} avi_index_entry;

// Write 32-bit little-endian integer
void write_u32_le(FILE* f, uint32_t val) {
    fwrite(&val, 4, 1, f);
}

// Write 16-bit little-endian integer
void write_u16_le(FILE* f, uint16_t val) {
    fwrite(&val, 2, 1, f);
}

/**
 * Create an MJPG AVI file from an array of sd_image_t images.
 * Images are encoded to JPEG using stb_image_write.
 *
 * @param filename Output AVI file name.
 * @param images Array of input images.
 * @param num_images Number of images in the array.
 * @param fps Frames per second for the video.
 * @param quality JPEG quality (0-100).
 * @return 0 on success, -1 on failure.
 */
int create_mjpg_avi_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality = 90) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return -1;
    }

    FILE* f = fopen(filename, "wb");
    if (!f) {
        perror("Error opening file for writing");
        return -1;
    }

    uint32_t width    = images[0].width;
    uint32_t height   = images[0].height;
    uint32_t channels = images[0].channel;
    if (channels != 3 && channels != 4) {
        fprintf(stderr, "Error: Unsupported channel count: %u\n", channels);
        fclose(f);
        return -1;
    }

    // --- RIFF AVI Header ---
    fwrite("RIFF", 4, 1, f);
    long riff_size_pos = ftell(f);
    write_u32_le(f, 0);  // Placeholder for file size
    fwrite("AVI ", 4, 1, f);

    // 'hdrl' LIST (header list)
    fwrite("LIST", 4, 1, f);
    write_u32_le(f, 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40);
    fwrite("hdrl", 4, 1, f);

    // 'avih' chunk (AVI main header)
    fwrite("avih", 4, 1, f);
    write_u32_le(f, 56);
    write_u32_le(f, 1000000 / fps);       // Microseconds per frame
    write_u32_le(f, 0);                   // Max bytes per second
    write_u32_le(f, 0);                   // Padding granularity
    write_u32_le(f, 0x110);               // Flags (HASINDEX | ISINTERLEAVED)
    write_u32_le(f, num_images);          // Total frames
    write_u32_le(f, 0);                   // Initial frames
    write_u32_le(f, 1);                   // Number of streams
    write_u32_le(f, width * height * 3);  // Suggested buffer size
    write_u32_le(f, width);
    write_u32_le(f, height);
    write_u32_le(f, 0);  // Reserved
    write_u32_le(f, 0);  // Reserved
    write_u32_le(f, 0);  // Reserved
    write_u32_le(f, 0);  // Reserved

    // 'strl' LIST (stream list)
    fwrite("LIST", 4, 1, f);
    write_u32_le(f, 4 + 8 + 56 + 8 + 40);
    fwrite("strl", 4, 1, f);

    // 'strh' chunk (stream header)
    fwrite("strh", 4, 1, f);
    write_u32_le(f, 56);
    fwrite("vids", 4, 1, f);              // Stream type: video
    fwrite("MJPG", 4, 1, f);              // Codec: Motion JPEG
    write_u32_le(f, 0);                   // Flags
    write_u16_le(f, 0);                   // Priority
    write_u16_le(f, 0);                   // Language
    write_u32_le(f, 0);                   // Initial frames
    write_u32_le(f, 1);                   // Scale
    write_u32_le(f, fps);                 // Rate
    write_u32_le(f, 0);                   // Start
    write_u32_le(f, num_images);          // Length
    write_u32_le(f, width * height * 3);  // Suggested buffer size
    write_u32_le(f, (uint32_t)-1);        // Quality
    write_u32_le(f, 0);                   // Sample size
    write_u16_le(f, 0);                   // rcFrame.left
    write_u16_le(f, 0);                   // rcFrame.top
    write_u16_le(f, 0);                   // rcFrame.right
    write_u16_le(f, 0);                   // rcFrame.bottom

    // 'strf' chunk (stream format: BITMAPINFOHEADER)
    fwrite("strf", 4, 1, f);
    write_u32_le(f, 40);
    write_u32_le(f, 40);  // biSize
    write_u32_le(f, width);
    write_u32_le(f, height);
    write_u16_le(f, 1);                   // biPlanes
    write_u16_le(f, 24);                  // biBitCount
    fwrite("MJPG", 4, 1, f);              // biCompression (FOURCC)
    write_u32_le(f, width * height * 3);  // biSizeImage
    write_u32_le(f, 0);                   // XPelsPerMeter
    write_u32_le(f, 0);                   // YPelsPerMeter
    write_u32_le(f, 0);                   // Colors used
    write_u32_le(f, 0);                   // Colors important

    // 'movi' LIST (video frames)
    long movi_list_pos = ftell(f);
    fwrite("LIST", 4, 1, f);
    long movi_size_pos = ftell(f);
    write_u32_le(f, 0);  // Placeholder for movi size
    fwrite("movi", 4, 1, f);

    avi_index_entry* index = (avi_index_entry*)malloc(sizeof(avi_index_entry) * num_images);
    if (!index) {
        fclose(f);
        return -1;
    }

    // Encode and write each frame as JPEG
    struct {
        uint8_t* buf;
        size_t size;
    } jpeg_data;

    for (int i = 0; i < num_images; i++) {
        jpeg_data.buf  = NULL;
        jpeg_data.size = 0;

        // Callback function to collect JPEG data into memory
        auto write_to_buf = [](void* context, void* data, int size) {
            auto jd = (decltype(jpeg_data)*)context;
            jd->buf = (uint8_t*)realloc(jd->buf, jd->size + size);
            memcpy(jd->buf + jd->size, data, size);
            jd->size += size;
        };

        // Encode to JPEG in memory
        stbi_write_jpg_to_func(
            write_to_buf,
            &jpeg_data,
            images[i].width,
            images[i].height,
            channels,
            images[i].data,
            quality);

        // Write '00dc' chunk (video frame)
        fwrite("00dc", 4, 1, f);
        write_u32_le(f, jpeg_data.size);
        index[i].offset = ftell(f) - 8;
        index[i].size   = jpeg_data.size;
        fwrite(jpeg_data.buf, 1, jpeg_data.size, f);

        // Align to even byte size
        if (jpeg_data.size % 2)
            fputc(0, f);

        free(jpeg_data.buf);
    }

    // Finalize 'movi' size
    long cur_pos   = ftell(f);
    long movi_size = cur_pos - movi_size_pos - 4;
    fseek(f, movi_size_pos, SEEK_SET);
    write_u32_le(f, movi_size);
    fseek(f, cur_pos, SEEK_SET);

    // Write 'idx1' index
    fwrite("idx1", 4, 1, f);
    write_u32_le(f, num_images * 16);
    for (int i = 0; i < num_images; i++) {
        fwrite("00dc", 4, 1, f);
        write_u32_le(f, 0x10);
        write_u32_le(f, index[i].offset);
        write_u32_le(f, index[i].size);
    }

    // Finalize RIFF size
    cur_pos        = ftell(f);
    long file_size = cur_pos - riff_size_pos - 4;
    fseek(f, riff_size_pos, SEEK_SET);
    write_u32_le(f, file_size);
    fseek(f, cur_pos, SEEK_SET);

    fclose(f);
    free(index);

    return 0;
}


//// KCPP PART FOR WRITING AVI TO MEMORY

typedef struct {
    uint8_t* data;
    size_t size;
} mem_buffer_t;

// Append raw bytes
static void mem_write(mem_buffer_t* buf, const void* data, size_t size) {
    buf->data = (uint8_t*)realloc(buf->data, buf->size + size);
    memcpy(buf->data + buf->size, data, size);
    buf->size += size;
}

// Write 32-bit LE
static void mem_write_u32_le(mem_buffer_t* buf, uint32_t val) {
    mem_write(buf, &val, 4);
}

// Write 16-bit LE
static void mem_write_u16_le(mem_buffer_t* buf, uint16_t val) {
    mem_write(buf, &val, 2);
}

/**
 * Create MJPG AVI file in memory and return as base64 string.
 * Returns 0 on success, -1 on failure
 * must be freed by caller after use
 */
int create_mjpg_avi_membuf_from_sd_images(sd_image_t* images, int num_images, int fps, int quality,  uint8_t** out_data, size_t *out_len)
{
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return -1;
    }

    mem_buffer_t buf = {NULL, 0};
    uint32_t width    = images[0].width;
    uint32_t height   = images[0].height;
    uint32_t channels = images[0].channel;

    if (channels != 3 && channels != 4) {
        fprintf(stderr, "Error: Unsupported channel count: %u\n", channels);
        return -1;
    }

    // --- RIFF AVI Header ---
    mem_write(&buf, "RIFF", 4);
    size_t riff_size_pos = buf.size;
    mem_write_u32_le(&buf, 0);  // placeholder
    mem_write(&buf, "AVI ", 4);

    // 'hdrl' LIST
    mem_write(&buf, "LIST", 4);
    mem_write_u32_le(&buf, 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40);
    mem_write(&buf, "hdrl", 4);

    // 'avih'
    mem_write(&buf, "avih", 4);
    mem_write_u32_le(&buf, 56);
    mem_write_u32_le(&buf, 1000000 / fps);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 0x110);
    mem_write_u32_le(&buf, num_images);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 1);
    mem_write_u32_le(&buf, width * height * 3);
    mem_write_u32_le(&buf, width);
    mem_write_u32_le(&buf, height);
    mem_write_u32_le(&buf, 0); mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 0); mem_write_u32_le(&buf, 0);

    // 'strl' LIST
    mem_write(&buf, "LIST", 4);
    mem_write_u32_le(&buf, 4 + 8 + 56 + 8 + 40);
    mem_write(&buf, "strl", 4);

    // 'strh'
    mem_write(&buf, "strh", 4);
    mem_write_u32_le(&buf, 56);
    mem_write(&buf, "vids", 4);
    mem_write(&buf, "MJPG", 4);
    mem_write_u32_le(&buf, 0);
    mem_write_u16_le(&buf, 0);
    mem_write_u16_le(&buf, 0);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 1);
    mem_write_u32_le(&buf, fps);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, num_images);
    mem_write_u32_le(&buf, width * height * 3);
    mem_write_u32_le(&buf, (uint32_t)-1);
    mem_write_u32_le(&buf, 0);
    mem_write_u16_le(&buf, 0); mem_write_u16_le(&buf, 0);
    mem_write_u16_le(&buf, 0); mem_write_u16_le(&buf, 0);

    // 'strf'
    mem_write(&buf, "strf", 4);
    mem_write_u32_le(&buf, 40);
    mem_write_u32_le(&buf, 40);
    mem_write_u32_le(&buf, width);
    mem_write_u32_le(&buf, height);
    mem_write_u16_le(&buf, 1);
    mem_write_u16_le(&buf, 24);
    mem_write(&buf, "MJPG", 4);
    mem_write_u32_le(&buf, width * height * 3);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 0);
    mem_write_u32_le(&buf, 0);

    // 'movi' LIST
    mem_write(&buf, "LIST", 4);
    size_t movi_size_pos = buf.size;
    mem_write_u32_le(&buf, 0);
    mem_write(&buf, "movi", 4);

    avi_index_entry* index = (avi_index_entry*)malloc(sizeof(avi_index_entry) * num_images);

    // Encode and write each frame
    for (int i = 0; i < num_images; i++) {
        struct {
            uint8_t* buf;
            size_t size;
        } jpeg_data = {NULL, 0};

        auto write_to_buf = [](void* context, void* data, int size) {
            auto jd = (decltype(jpeg_data)*)context;
            jd->buf = (uint8_t*)realloc(jd->buf, jd->size + size);
            memcpy(jd->buf + jd->size, data, size);
            jd->size += size;
        };

        stbi_write_jpg_to_func(
            write_to_buf, &jpeg_data,
            images[i].width, images[i].height,
            channels, images[i].data, quality
        );

        mem_write(&buf, "00dc", 4);
        mem_write_u32_le(&buf, jpeg_data.size);
        index[i].offset = buf.size - 8;
        index[i].size   = jpeg_data.size;
        mem_write(&buf, jpeg_data.buf, jpeg_data.size);
        if (jpeg_data.size % 2) mem_write(&buf, "\0", 1);

        free(jpeg_data.buf);
    }

    // finalize movi size
    uint32_t movi_size = buf.size - movi_size_pos - 4;
    memcpy(buf.data + movi_size_pos, &movi_size, 4);

    // write idx1
    mem_write(&buf, "idx1", 4);
    mem_write_u32_le(&buf, num_images * 16);
    for (int i = 0; i < num_images; i++) {
        mem_write(&buf, "00dc", 4);
        mem_write_u32_le(&buf, 0x10);
        mem_write_u32_le(&buf, index[i].offset);
        mem_write_u32_le(&buf, index[i].size);
    }

    // finalize RIFF size
    uint32_t riff_size = buf.size - riff_size_pos - 4;
    memcpy(buf.data + riff_size_pos, &riff_size, 4);

    free(index);

    *out_data = buf.data;
    *out_len = buf.size;
    return 0;
}

/// kcpp gif writer

// ---------------- Helper: create_gif_buf_from_sd_images ----------------
// Builds a GIF in memory from an array of sd_image_t. Returns 0 on success, -1 on failure.
// Caller must free(*out_data) when done.
int create_gif_buf_from_sd_images_msf(sd_image_t* images, int num_images, int fps,  uint8_t** out_data, size_t *out_len)
{
    if(!images || num_images <= 0 || !out_data || !out_len) return -1;

    // basic parameter heuristics
    if(fps <= 0) fps = 16;
    uint32_t delay = (uint32_t)(100 / fps); // hundredths of a second per frame

    int bitDepth = 8;
    bool dither = false;

    // assume all images same size; use first
    uint32_t width = images[0].width;
    uint32_t height = images[0].height;

    int centisecondsPerFrame = delay;
    int quality = 16;
    MsfGifState gifState = {};
    msf_gif_begin(&gifState, width, height);

    // Feed frames
    for (int i = 0; i < num_images; i++)
    {
        sd_image_t* img = &images[i];

        if (img->width != width || img->height != height) {
            fprintf(stderr, "Frame %d has mismatched dimensions.\n", i);
            return -1;
        }

        // gif-h expects 4 channels (RGBA) or 3 channels (RGB). It quantizes internally.
        // If your images have 3 channels, that’s fine. If 4 channels, it also works.
        int channels = img->channel;
        if (channels != 3 && channels != 4) {
            fprintf(stderr, "Unsupported channel count: %d\n", channels);
            return -1;
        }

        // gif-h requires 4 channels (RGBA). If you only have RGB, add opaque alpha.
        uint8_t* frame_rgba = NULL;
        if (channels == 3) {
            frame_rgba = (uint8_t*)malloc(width * height * 4);
            for (int p = 0; p < width * height; p++) {
                frame_rgba[p*4+0] = img->data[p*3+0];
                frame_rgba[p*4+1] = img->data[p*3+1];
                frame_rgba[p*4+2] = img->data[p*3+2];
                frame_rgba[p*4+3] = 255;
            }
        } else {
            frame_rgba = img->data; // already RGBA
        }

        int res = msf_gif_frame(&gifState, frame_rgba, centisecondsPerFrame, quality, width * 4); //frame
        if(!res)
        {
            fprintf(stderr, "GIF Write Failed\n");
            return -1;
        }

        if (channels == 3) {
            free(frame_rgba);
        }
    }

    MsfGifResult result = msf_gif_end(&gifState);
    *out_data = (uint8_t*)result.data;
    *out_len = result.dataSize;
    return 0;
}

int create_gif_buf_from_sd_images_gifh(sd_image_t* images, int num_images, int fps,  uint8_t** out_data, size_t *out_len)
{
    if(!images || num_images <= 0 || !out_data || !out_len) return -1;

    // basic parameter heuristics
    if(fps <= 0) fps = 16;
    uint32_t delay = (uint32_t)(100 / fps); // hundredths of a second per frame

    int bitDepth = 8;
    bool dither = false;

    // assume all images same size; use first
    uint32_t width = images[0].width;
    uint32_t height = images[0].height;

    GifWriter gw;
    memset(&gw, 0, sizeof(gw));

    if(!GifBegin(&gw, width, height, delay, bitDepth, dither))
    {
        if(gw.oldImage) GIF_FREE(gw.oldImage);

        fprintf(stderr, "Error: GifBegin failed.\n");
        return -1;
    }

    // Feed frames
    for (int i = 0; i < num_images; i++)
    {
        sd_image_t* img = &images[i];

        if (img->width != width || img->height != height) {
            fprintf(stderr, "Frame %d has mismatched dimensions.\n", i);
            GifEnd(&gw);
            memfile_free(&gw.mem);
            return -1;
        }

        // gif-h expects 4 channels (RGBA) or 3 channels (RGB). It quantizes internally.
        // If your images have 3 channels, that’s fine. If 4 channels, it also works.
        int channels = img->channel;
        if (channels != 3 && channels != 4) {
            fprintf(stderr, "Unsupported channel count: %d\n", channels);
            GifEnd(&gw);
            memfile_free(&gw.mem);
            return -1;
        }

        // gif-h requires 4 channels (RGBA). If you only have RGB, add opaque alpha.
        uint8_t* frame_rgba = NULL;
        if (channels == 3) {
            frame_rgba = (uint8_t*)malloc(width * height * 4);
            for (int p = 0; p < width * height; p++) {
                frame_rgba[p*4+0] = img->data[p*3+0];
                frame_rgba[p*4+1] = img->data[p*3+1];
                frame_rgba[p*4+2] = img->data[p*3+2];
                frame_rgba[p*4+3] = 255;
            }
        } else {
            frame_rgba = img->data; // already RGBA
        }

        if(!GifWriteFrame(&gw, frame_rgba, width, height, delay, bitDepth, dither))
        {
            fprintf(stderr, "GIF Write Failed\n");
            GifEnd(&gw);
            memfile_free(&gw.mem);
            return -1;
        }

        if (channels == 3) {
            free(frame_rgba);
        }
    }

    if(!GifEnd(&gw))
    {
        memfile_free(&gw.mem);
        return -1;
    }

    uint8_t* buf = memfile_detach(&gw.mem, out_len);
    if(!buf)
    {
        memfile_free(&gw.mem);
        return -1;
    }

    *out_data = buf;
    return 0;
}

#endif  // __AVI_WRITER_H__