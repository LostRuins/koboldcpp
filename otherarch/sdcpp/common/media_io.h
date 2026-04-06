#ifndef __MEDIA_IO_H__
#define __MEDIA_IO_H__

#include <cstdint>
#include <string>
#include <vector>

#include "stable-diffusion.h"

enum class EncodedImageFormat {
    JPEG,
    PNG,
    WEBP,
    UNKNOWN,
};

EncodedImageFormat encoded_image_format_from_path(const std::string& path);

std::vector<uint8_t> encode_image_to_vector(EncodedImageFormat format,
                                            const uint8_t* image,
                                            int width,
                                            int height,
                                            int channels,
                                            const std::string& parameters = "",
                                            int quality                   = 90);

bool write_image_to_file(const std::string& path,
                         const uint8_t* image,
                         int width,
                         int height,
                         int channels,
                         const std::string& parameters = "",
                         int quality                   = 90);

uint8_t* load_image_from_file(const char* image_path,
                              int& width,
                              int& height,
                              int expected_width   = 0,
                              int expected_height  = 0,
                              int expected_channel = 3);

bool load_sd_image_from_file(sd_image_t* image,
                             const char* image_path,
                             int expected_width   = 0,
                             int expected_height  = 0,
                             int expected_channel = 3);

uint8_t* load_image_from_memory(const char* image_bytes,
                                int len,
                                int& width,
                                int& height,
                                int expected_width   = 0,
                                int expected_height  = 0,
                                int expected_channel = 3);

int create_mjpg_avi_from_sd_images(const char* filename,
                                   sd_image_t* images,
                                   int num_images,
                                   int fps,
                                   int quality = 90);

#ifdef SD_USE_WEBP
int create_animated_webp_from_sd_images(const char* filename,
                                        sd_image_t* images,
                                        int num_images,
                                        int fps,
                                        int quality = 90);
#endif

#ifdef SD_USE_WEBM
int create_webm_from_sd_images(const char* filename,
                               sd_image_t* images,
                               int num_images,
                               int fps,
                               int quality = 90);
#endif

int create_video_from_sd_images(const char* filename,
                                sd_image_t* images,
                                int num_images,
                                int fps,
                                int quality = 90);

#endif  // __MEDIA_IO_H__
