#ifndef __SD_CORE_SEGMENT_GRAPH_BINDINGS_H__
#define __SD_CORE_SEGMENT_GRAPH_BINDINGS_H__

#include <array>
#include <unordered_map>
#include "ggml.h"

struct ggml_backend_buffer;
struct ggml_cgraph;
struct ggml_tensor;

namespace sd {
    class GraphCutTensorCache;

    namespace ggml_graph_cut {
        struct Plan;
        struct Segment;
    }

    class SegmentGraphBindings {
    public:
        SegmentGraphBindings(GraphCutTensorCache& tensor_cache,
                             const ggml_graph_cut::Plan& plan,
                             ggml_cgraph* graph);

        void reset(const ggml_graph_cut::Segment& segment);
        void restore();
        ~SegmentGraphBindings() { restore(); }
        bool bind_cached_inputs(const ggml_graph_cut::Segment& segment,
                                const char* log_desc);

    private:
        struct ExternalBinding {
            ggml_backend_buffer* buffer = nullptr;
            void* data                  = nullptr;
            void* extra                 = nullptr;
        };

        GraphCutTensorCache& tensor_cache_;
        ggml_cgraph* graph_ = nullptr;
        std::unordered_map<ggml_tensor*, ExternalBinding> external_bindings_;
        struct Topology {
            ggml_op op;
            std::array<ggml_tensor*, GGML_MAX_SRC> sources;
            ggml_tensor* view_source;
            int flags;
        };
        std::unordered_map<ggml_tensor*, Topology> topology_;
    };
}

#endif  // __SD_CORE_SEGMENT_GRAPH_BINDINGS_H__
