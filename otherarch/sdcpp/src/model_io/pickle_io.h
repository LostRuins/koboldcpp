#ifndef __SD_MODEL_IO_PICKLE_IO_H__
#define __SD_MODEL_IO_PICKLE_IO_H__

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor_storage.h"

bool skip_pickle_object(const uint8_t* buffer, size_t buffer_size, size_t* object_size);
bool pickle_object_is_torch_magic_number(const uint8_t* buffer, size_t buffer_size);
bool parse_pickle_uint32_object(const uint8_t* buffer, size_t buffer_size, uint32_t* value);
bool parse_torch_state_dict_pickle(const uint8_t* buffer,
                                   size_t buffer_size,
                                   std::vector<TensorStorage>& tensor_storages,
                                   std::unordered_map<std::string, uint64_t>& storage_nbytes,
                                   std::string* error = nullptr);

#endif  // __SD_MODEL_IO_PICKLE_IO_H__
