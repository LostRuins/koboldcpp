#include "media_metadata.h"

std::string generation_xmp(const std::string& info) {
    std::string escaped;
    for (char c : info) {
        switch (c) {
            case '&':
                escaped += "&amp;";
                break;
            case '<':
                escaped += "&lt;";
                break;
            case '>':
                escaped += "&gt;";
                break;
            case '"':
                escaped += "&quot;";
                break;
            case '\'':
                escaped += "&apos;";
                break;
            default:
                escaped += c;
                break;
        }
    }
    return "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
           "<rdf:Description rdf:about=\"\" xmlns:sdcpp=\"https://github.com/leejet/stable-diffusion.cpp/ns/1.0/\">"
           "<sdcpp:parameters>" +
           escaped + "</sdcpp:parameters></rdf:Description></rdf:RDF></x:xmpmeta>";
}

bool embed_jpeg_metadata(std::vector<uint8_t>& image, const std::string& parameters) {
    if (parameters.empty())
        return true;
    const char label[]    = "parameters";
    const size_t overhead = sizeof(label) + 1;
    if (image.size() < 2 || parameters.size() > 65533 - overhead)
        return false;
    const size_t length          = overhead + parameters.size() + 2;
    std::vector<uint8_t> segment = {0xff, 0xfe, static_cast<uint8_t>(length >> 8), static_cast<uint8_t>(length)};
    segment.insert(segment.end(), label, label + sizeof(label));
    segment.insert(segment.end(), parameters.begin(), parameters.end());
    segment.push_back(0);
    image.insert(image.begin() + 2, segment.begin(), segment.end());
    return true;
}
