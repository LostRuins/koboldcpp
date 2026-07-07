#ifndef __SD_CORE_LAYER_SPLIT_PARTITION_H__
#define __SD_CORE_LAYER_SPLIT_PARTITION_H__

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "core/ggml_graph_cut.h"

namespace sd {

    struct GraphCutLayerSplitAssignment {
        std::vector<std::vector<ggml_tensor*>> tensors_by_backend;
        std::vector<int64_t> bytes_by_backend;
        std::vector<size_t> first_segment_by_backend;
        std::vector<size_t> last_segment_by_backend;
        std::unordered_map<const ggml_tensor*, ggml_backend_t> node_assignments;
        size_t segment_count          = 0;
        bool has_new_param_assignment = false;
    };

    std::string layer_split_backend_device_display_name(ggml_backend_t backend);
    int layer_split_tensor_block_index(const std::string& name);
    bool partition_graph_cut_layer_split(const char* desc,
                                         ggml_cgraph* gf,
                                         const sd::ggml_graph_cut::Plan& plan,
                                         const std::vector<ggml_backend_t>& split_backends,
                                         const std::vector<size_t>& backend_vram_limits,
                                         size_t primary_backend_vram_limit,
                                         std::unordered_map<const ggml_tensor*, ggml_backend_t>& param_assignments,
                                         const std::function<ggml_tensor*(ggml_tensor*)>& canonical_param_tensor,
                                         GraphCutLayerSplitAssignment* assignment_out);
    void log_graph_cut_layer_split_assignment(const char* desc,
                                              const std::vector<ggml_backend_t>& split_backends,
                                              const GraphCutLayerSplitAssignment& assignment);

}  // namespace sd

#endif  // __SD_CORE_LAYER_SPLIT_PARTITION_H__
