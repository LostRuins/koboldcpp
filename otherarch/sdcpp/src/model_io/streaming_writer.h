#ifndef __SD_MODEL_IO_STREAMING_WRITER_H__
#define __SD_MODEL_IO_STREAMING_WRITER_H__

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "tensor_storage.h"

class StreamingModelWriter {
public:
    virtual ~StreamingModelWriter() = default;

    virtual bool write_metadata(const std::string& file_path,
                                const std::vector<TensorWritePlan>& tensors,
                                std::string* error = nullptr)     = 0;
    virtual bool write_tensor(std::ostream& output,
                              size_t tensor_index,
                              const uint8_t* data,
                              size_t size,
                              std::string* error = nullptr) const = 0;
    virtual uint64_t file_size() const                            = 0;
};

#endif  // __SD_MODEL_IO_STREAMING_WRITER_H__
