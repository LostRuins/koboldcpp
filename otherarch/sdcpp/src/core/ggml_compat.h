#ifndef __SD_CORE_GGML_COMPAT_H__
#define __SD_CORE_GGML_COMPAT_H__

#include "ggml-backend.h"

#if __has_include("ggml-graph.h")
#include "ggml-graph.h"
#elif defined(SD_USE_UPSTREAM_GGML)
#define SD_GGML_GRAPH_COMPAT

int ggml_graph_n_leafs(const struct ggml_cgraph* graph);
struct ggml_tensor* ggml_graph_leaf(const struct ggml_cgraph* graph, int index);
// Like ggml_graph_add_node, appends metadata without traversing tensor dependencies.
void ggml_graph_add_leaf(struct ggml_cgraph* graph, struct ggml_tensor* tensor);
// Returns zero when the tensor has no recorded use count. Includes uses outside graph views.
int32_t ggml_graph_use_count(const struct ggml_cgraph* graph, const struct ggml_tensor* tensor);

typedef struct ggml_tensor* (*ggml_graph_tensor_map_fn)(const struct ggml_tensor* tensor, void* user_data);

// Copies forward node/leaf order and recorded use counts, including external consumers.
// The callback must be injective and may return NULL for dependencies outside the
// node/leaf lists. It is called once per tensor in their dependency closure.
// Tensor storage and edges are mapped by the caller.
// The destination must have no gradients and enough capacity. Gradients are not copied.
// Invalid mappings leave the destination unchanged. src and dst must be distinct.
bool ggml_graph_cpy_mapped(const struct ggml_cgraph* src, struct ggml_cgraph* dst, ggml_graph_tensor_map_fn map, void* user_data);

// Copies recorded counts for tensors in dst's node/leaf lists. Other metadata is unchanged.
// This preserves external consumers when constructing an inference graph segment.
bool ggml_graph_cpy_use_counts(const struct ggml_cgraph* src, struct ggml_cgraph* dst);

// Executes nodes [first, last) with the original graph's use counts. No graph ownership
// is transferred. Tensor storage must already be allocated; invalid bounds fail.
enum ggml_status ggml_backend_graph_compute_async_range(ggml_backend_t backend,
                                                        struct ggml_cgraph* graph,
                                                        int first,
                                                        int last);

#else
#error "Patched ggml requires ggml-graph.h; select SD_USE_UPSTREAM_GGML for upstream compatibility"
#endif

#endif
