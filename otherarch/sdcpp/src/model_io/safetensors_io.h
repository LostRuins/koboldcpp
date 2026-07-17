#ifndef __SD_MODEL_IO_SAFETENSORS_IO_H__
#define __SD_MODEL_IO_SAFETENSORS_IO_H__

#include <map>
#include <string>
#include <vector>

#include "streaming_writer.h"
#include "tensor_storage.h"

bool is_safetensors_file(const std::string& file_path);
bool read_safetensors_file(const std::string& file_path,
                           std::vector<TensorStorage>& tensor_storages,
                           std::string* error                           = nullptr,
                           std::map<std::string, std::string>* metadata = nullptr);
bool read_safetensors_index_file(const std::string& file_path,
                                 std::vector<std::string>& shard_paths,
                                 std::string* error = nullptr);
bool write_safetensors_file(const std::string& file_path,
                            const std::vector<TensorWriteInfo>& tensors,
                            std::string* error = nullptr);

class SafetensorsStreamingWriter : public StreamingModelWriter {
public:
    SafetensorsStreamingWriter() = default;

    bool write_metadata(const std::string& file_path,
                        const std::vector<TensorWritePlan>& tensors,
                        std::string* error = nullptr) override;
    bool write_tensor(std::ostream& output,
                      size_t tensor_index,
                      const uint8_t* data,
                      size_t size,
                      std::string* error = nullptr) const override;
    uint64_t file_size() const override;

private:
    std::string file_path_;
    std::vector<TensorWritePlan> tensors_;
    std::vector<uint64_t> tensor_offsets_;
    uint64_t data_start_ = 0;
    uint64_t file_size_  = 0;
};

#endif  // __SD_MODEL_IO_SAFETENSORS_IO_H__
