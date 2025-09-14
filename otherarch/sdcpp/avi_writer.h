#ifndef __AVI_WRITER_H__
#define __AVI_WRITER_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stable-diffusion.h"

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

#endif  // __AVI_WRITER_H__