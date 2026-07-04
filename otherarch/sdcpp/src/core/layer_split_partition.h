#ifndef __SD_CORE_LAYER_SPLIT_PARTITION_H__
#define __SD_CORE_LAYER_SPLIT_PARTITION_H__

#include <map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace sd {

    std::string layer_split_backend_device_display_name(ggml_backend_t backend);
    int layer_split_tensor_block_index(const std::string& name);

    std::vector<std::map<std::string, ggml_tensor*>> partition_layer_split_tensors(
        const std::string& desc,
        const std::map<std::string, ggml_tensor*>& tensors,
        const std::map<std::string, ggml_tensor*>& split_tensors,
        const std::vector<ggml_backend_t>& backends);

}  // namespace sd

#endif  // __SD_CORE_LAYER_SPLIT_PARTITION_H__
