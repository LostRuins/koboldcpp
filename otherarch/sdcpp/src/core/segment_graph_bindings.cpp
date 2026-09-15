#include "core/segment_graph_bindings.h"

#include <algorithm>
#include <iterator>

#include "core/ggml_graph_cut.h"
#include "core/runner_cache.h"
#include "core/util.h"
#include "ggml.h"

namespace sd {
    SegmentGraphBindings::SegmentGraphBindings(GraphCutTensorCache& tensor_cache,
                                               const ggml_graph_cut::Plan& plan,
                                               ggml_cgraph* graph)
        : tensor_cache_(tensor_cache),
          graph_(graph) {
        GGML_ASSERT(graph_ != nullptr);
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            ggml_tensor* tensor = ggml_graph_node(graph_, i);
            Topology topology{tensor->op, {}, tensor->view_src, tensor->flags};
            std::copy(std::begin(tensor->src), std::end(tensor->src), topology.sources.begin());
            topology_[tensor] = topology;
        }
        for (const auto& segment : plan.segments) {
            for (const auto& input : segment.input_refs) {
                if (input.type != ggml_graph_cut::Segment::INPUT_EXTERNAL) {
                    continue;
                }
                ggml_tensor* tensor = ggml_graph_cut::input_tensor(graph_, input);
                if (tensor == nullptr || tensor->buffer == nullptr) {
                    continue;
                }
                external_bindings_[tensor] = {tensor->buffer, tensor->data, tensor->extra};
            }
        }
    }

    void SegmentGraphBindings::reset(const ggml_graph_cut::Segment& segment) {
        restore();
        for (const auto& input : segment.input_refs) {
            ggml_tensor* tensor = ggml_graph_cut::input_tensor(graph_, input);
            if (tensor == nullptr) {
                continue;
            }
            switch (input.type) {
                case ggml_graph_cut::Segment::INPUT_PREVIOUS_CUT:
                    tensor->buffer = nullptr;
                    tensor->data   = nullptr;
                    tensor->extra  = nullptr;
                    break;
                case ggml_graph_cut::Segment::INPUT_EXTERNAL: {
                    auto binding = external_bindings_.find(tensor);
                    if (binding != external_bindings_.end()) {
                        tensor->buffer = binding->second.buffer;
                        tensor->data   = binding->second.data;
                        tensor->extra  = binding->second.extra;
                    } else {
                        tensor->buffer = nullptr;
                        tensor->data   = nullptr;
                        tensor->extra  = nullptr;
                    }
                    break;
                }
                case ggml_graph_cut::Segment::INPUT_PARAM:
                    break;
            }
        }

        for (int node_index : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(graph_, node_index);
            if (node == nullptr) {
                continue;
            }
            node->buffer = nullptr;
            node->data   = nullptr;
            node->extra  = nullptr;
        }
    }

    void SegmentGraphBindings::restore() {
        for (const auto& entry : topology_) {
            entry.first->op       = entry.second.op;
            entry.first->view_src = entry.second.view_source;
            entry.first->flags    = entry.second.flags;
            std::copy(entry.second.sources.begin(), entry.second.sources.end(), std::begin(entry.first->src));
        }
    }

    bool SegmentGraphBindings::bind_cached_inputs(
        const ggml_graph_cut::Segment& segment,
        const char* log_desc) {
        std::unordered_map<ggml_tensor*, ggml_tensor*> cached_view_sources;
        for (const auto& input : segment.input_refs) {
            if (input.type != ggml_graph_cut::Segment::INPUT_PREVIOUS_CUT) {
                continue;
            }
            ggml_tensor* input_tensor = ggml_graph_cut::input_tensor(graph_, input);
            if (input_tensor == nullptr) {
                continue;
            }
            ggml_tensor* cached_tensor = tensor_cache_.get(input.display_name);
            if (cached_tensor == nullptr) {
                LOG_ERROR("%s missing graph cut cache tensor: %s",
                          log_desc,
                          input.display_name.c_str());
                return false;
            }
            if (input_tensor->view_src != nullptr) {
                cached_view_sources[topology_.at(input_tensor).view_source] = cached_tensor;
                input_tensor->view_src                                      = cached_tensor;
                input_tensor->buffer                                        = nullptr;
                input_tensor->data                                          = cached_tensor->data == nullptr
                                                                                  ? nullptr
                                                                                  : static_cast<void*>(static_cast<char*>(cached_tensor->data) +
                                                              input_tensor->view_offs);
                input_tensor->extra                                         = cached_tensor->extra;
            } else {
                input_tensor->buffer = cached_tensor->buffer;
                input_tensor->data   = cached_tensor->data;
                input_tensor->extra  = cached_tensor->extra;
            }
            for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
                input_tensor->src[source_index] = nullptr;
            }
            input_tensor->op = GGML_OP_NONE;
        }
        // ggml flattens view chains, so descendants also need the cached root.
        for (int node_index : segment.internal_node_indices) {
            ggml_tensor* node  = ggml_graph_node(graph_, node_index);
            auto cached_source = cached_view_sources.find(topology_.at(node).view_source);
            if (cached_source != cached_view_sources.end()) {
                node->view_src = cached_source->second;
            }
        }
        return true;
    }
}
