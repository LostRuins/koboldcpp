#ifndef __SD_CORE_GGML_GRAPH_CUT_H__
#define __SD_CORE_GGML_GRAPH_CUT_H__

#include <array>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

        size_t compute_buffer_size = 0;
        std::string group_name;
        std::vector<int> internal_node_indices;
        std::vector<int> output_node_indices;
        std::vector<InputRef> input_refs;
        std::unordered_set<std::string> future_cut_names;
        std::unordered_set<std::string> live_cut_names;
    };

    struct Plan {
        bool available             = false;
        bool has_cuts              = false;
        bool valid                 = true;
        size_t compute_buffer_size = 0;
        std::vector<uint64_t> layout;
        std::vector<std::string> leaf_names;
        std::vector<std::pair<int, std::string>> cut_markers;
        std::vector<Segment> segments;
    };

    struct PlanCache {
        static constexpr size_t MAX_PLANS = 4;
        std::list<Plan> graph_cut_plans;
    };

    static constexpr const char* GGML_RUNNER_CUT_PREFIX = "ggml_runner_cut:";
    static constexpr const char* GGML_RUNNER_CUT_SUFFIX = "|";

    struct MaxVramAssignment {
        float default_gib = 0.f;
        std::unordered_map<std::string, float> backend_gib;
        std::unordered_map<std::string, size_t> resolved_backend_bytes;

        void reset(float fallback_gib);
        bool parse(const std::string& raw_spec, std::string* error);
        bool canonicalize_backend_keys(std::string* error);
        size_t bytes_for_backend(ggml_backend_t backend);
    };

    bool is_graph_cut_tensor(const ggml_tensor* tensor);
    std::string make_graph_cut_name(const std::string& group, const std::string& output);
    void mark_graph_cut(ggml_tensor* tensor, const std::string& group, const std::string& output);
    int leaf_count(ggml_cgraph* gf);
    ggml_tensor* leaf_tensor(ggml_cgraph* gf, int leaf_index);
    ggml_backend_buffer_t tensor_buffer(const ggml_tensor* tensor);
    ggml_tensor* cache_source_tensor(ggml_tensor* tensor);
    size_t cache_tensor_bytes(const ggml_tensor* tensor);
    // Plans ignore runtime bindings; allocator reservations must include them.
    std::vector<uint64_t> graph_layout(ggml_cgraph* graph, bool include_bindings);
    bool plan_matches_graph(ggml_cgraph* gf, const Plan& plan);
    ggml_tensor* output_tensor(ggml_cgraph* gf, const Segment& segment, size_t output_index);
    ggml_tensor* input_tensor(ggml_cgraph* gf, const Segment::InputRef& input_ref);
    std::vector<ggml_tensor*> param_tensors(ggml_cgraph* gf, const Segment& segment);
    ggml_cgraph* build_segment_graph(ggml_cgraph* gf,
                                     const Segment& segment,
                                     ggml_context** graph_ctx_out);
    size_t measure_segment_compute_buffer(ggml_backend_t backend,
                                          ggml_cgraph* gf,
                                          const Segment& segment,
                                          const char* log_desc);
    size_t max_vram_gib_to_bytes(float max_vram);
    float resolve_max_vram_gib(float max_vram, ggml_backend_t backend);
    Plan build_plan(ggml_backend_t backend,
                    ggml_cgraph* gf,
                    const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                    const char* log_desc);
    Plan resolve_plan(ggml_backend_t backend,
                      ggml_cgraph* gf,
                      PlanCache* cache,
                      const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                      const char* log_desc);

}  // namespace sd::ggml_graph_cut

#endif  // __SD_CORE_GGML_GRAPH_CUT_H__
