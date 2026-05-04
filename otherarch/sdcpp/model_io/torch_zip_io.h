#ifndef __SD_MODEL_IO_TORCH_ZIP_IO_H__
#define __SD_MODEL_IO_TORCH_ZIP_IO_H__

#include <string>
#include <vector>

#include "tensor_storage.h"

bool is_torch_zip_file(const std::string& file_path);
bool read_torch_zip_file(const std::string& file_path,
                         std::vector<TensorStorage>& tensor_storages,
                         std::string* error = nullptr);

#endif  // __SD_MODEL_IO_TORCH_ZIP_IO_H__
