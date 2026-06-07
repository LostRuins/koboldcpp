#ifndef __SD_MODEL_IO_SAFETENSORS_IO_H__
#define __SD_MODEL_IO_SAFETENSORS_IO_H__

#include <string>
#include <vector>

#include "tensor_storage.h"

bool is_safetensors_file(const std::string& file_path);
bool read_safetensors_file(const std::string& file_path,
                           std::vector<TensorStorage>& tensor_storages,
                           std::string* error = nullptr);
bool write_safetensors_file(const std::string& file_path,
                            const std::vector<TensorWriteInfo>& tensors,
                            std::string* error = nullptr);

#endif  // __SD_MODEL_IO_SAFETENSORS_IO_H__
