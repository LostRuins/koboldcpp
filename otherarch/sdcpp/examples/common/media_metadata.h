#ifndef __EXAMPLES_COMMON_MEDIA_METADATA_H__
#define __EXAMPLES_COMMON_MEDIA_METADATA_H__

#include <cstdint>
#include <string>
#include <vector>

std::string generation_xmp(const std::string& parameters);
bool embed_jpeg_metadata(std::vector<uint8_t>& image, const std::string& parameters);

#endif  // __EXAMPLES_COMMON_MEDIA_METADATA_H__
