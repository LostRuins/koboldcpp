#ifndef __SD_GGML_GRAPH_CUT_H__
#define __SD_GGML_GRAPH_CUT_H__

#include <array>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace sd::ggml_graph_cut {

    struct Segment {
        enum InputType {
            INPUT_EXTERNAL = 0,
            INPUT_PREVIOUS_CUT,
            INPUT_PARAM,
        };

        struct InputRef {
            InputType type = INPUT_EXTERNAL;
            std::string display_name;
            int leaf_index = -1;
            int node_index = -1;
        };

        size_t compute_buffer_size      = 0;
        size_t output_bytes             = 0;
        size_t input_external_bytes     = 0;
        size_t input_previous_cut_bytes = 0;
        size_t input_param_bytes        = 0;
        std::string group_name;
        std::vector<int> internal_node_indices;
        std::vector<int> output_node_indices;
        std::vector<InputRef> input_refs;
    };

    struct Plan {
        struct InputShape {
            int leaf_index                        = -1;
            ggml_type type                        = GGML_TYPE_COUNT;
            std::array<int64_t, GGML_MAX_DIMS> ne = {0, 0, 0, 0};
        };

        bool available = false;
        bool has_cuts  = false;
        bool valid     = true;
        int n_nodes    = 0;
        int n_leafs    = 0;
        std::vector<InputShape> input_shapes;
        std::vector<Segment> segments;
    };

    struct PlanCache {
        Plan graph_cut_plan;
        Plan budgeted_graph_cut_plan;
        size_t budgeted_graph_cut_plan_max_vram_bytes = 0;
    };

    static constexpr const char* GGML_RUNNER_CUT_PREFIX = "ggml_runner_cut:";

    bool is_graph_cut_tensor(const ggml_tensor* tensor);
    std::string make_graph_cut_name(const std::string& group, const std::string& output);
    void mark_graph_cut(ggml_tensor* tensor, const std::string& group, const std::string& output);
    int leaf_count(ggml_cgraph* gf);
    ggml_tensor* leaf_tensor(ggml_cgraph* gf, int leaf_index);
    ggml_backend_buffer_t tensor_buffer(const ggml_tensor* tensor);
    ggml_tensor* cache_source_tensor(ggml_tensor* tensor);
    size_t cache_tensor_bytes(const ggml_tensor* tensor);
    bool plan_matches_graph(ggml_cgraph* gf, const Plan& plan);
    ggml_tensor* output_tensor(ggml_cgraph* gf, const Segment& segment, size_t output_index);
    ggml_tensor* input_tensor(ggml_cgraph* gf, const Segment::InputRef& input_ref);
    std::vector<ggml_tensor*> param_tensors(ggml_cgraph* gf, const Segment& segment);
    std::vector<ggml_tensor*> runtime_param_tensors(ggml_cgraph* gf, const Segment& segment, const char* log_desc);
    std::unordered_set<std::string> collect_future_input_names(ggml_cgraph* gf,
                                                               const Plan& plan,
                                                               size_t current_segment_index);
    ggml_cgraph* build_segment_graph(ggml_cgraph* gf,
                                     const Segment& segment,
                                     ggml_context** graph_ctx_out);
    size_t measure_segment_compute_buffer(ggml_backend_t backend,
                                          ggml_cgraph* gf,
                                          const Segment& segment,
                                          const char* log_desc);
    Plan build_plan(ggml_backend_t backend,
                    ggml_cgraph* gf,
                    const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                    const char* log_desc);
    Plan apply_max_vram_budget(ggml_cgraph* gf,
                               const Plan& base_plan,
                               size_t max_graph_vram_bytes,
                               ggml_backend_t backend,
                               const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                               const char* log_desc);
    Plan resolve_plan(ggml_backend_t backend,
                      ggml_cgraph* gf,
                      PlanCache* cache,
                      size_t max_graph_vram_bytes,
                      const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                      const char* log_desc);
}  // namespace sd::ggml_graph_cut

#endif
