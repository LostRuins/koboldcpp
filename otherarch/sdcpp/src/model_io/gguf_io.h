#ifndef __SD_MODEL_IO_GGUF_IO_H__
#define __SD_MODEL_IO_GGUF_IO_H__

#include <string>
#include <vector>

#include "streaming_writer.h"
#include "tensor_storage.h"

struct ggml_context;
struct gguf_context;

bool is_gguf_file(const std::string& file_path);
bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error = nullptr);
bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error = nullptr);

class GGUFStreamingWriter : public StreamingModelWriter {
public:
    GGUFStreamingWriter() = default;
    ~GGUFStreamingWriter() override;

    bool write_metadata(const std::string& file_path,
                        const std::vector<TensorWritePlan>& tensors,
                        std::string* error = nullptr) override;
    bool write_tensor(std::ostream& output,
                      size_t tensor_index,
                      const uint8_t* data,
                      size_t size,
                      std::string* error = nullptr) const override;
    uint64_t file_size() const override;
    void close();

private:
    std::vector<TensorWritePlan> tensors_;
    std::vector<uint64_t> tensor_offsets_;
    uint64_t file_size_     = 0;
    ggml_context* meta_ctx_ = nullptr;
    gguf_context* gguf_ctx_ = nullptr;
};

#endif  // __SD_MODEL_IO_GGUF_IO_H__
