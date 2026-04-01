#pragma once

#include <iosfwd>
#include <string>

enum class MetadataOutputFormat {
    TEXT,
    JSON,
};

struct MetadataReadOptions {
    MetadataOutputFormat output_format = MetadataOutputFormat::TEXT;
    bool include_raw                   = false;
    bool brief                         = false;
    bool include_structural            = false;
};

bool print_image_metadata(const std::string& image_path,
                          const MetadataReadOptions& options,
                          std::ostream& out,
                          std::string& error);
