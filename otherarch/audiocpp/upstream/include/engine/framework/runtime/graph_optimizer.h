#pragma once

#include "ggml.h"

#include <cstddef>

namespace engine::runtime {

enum class GraphOptimizationBackend {
    Cpu,
    Gpu,
    Other,
};

enum class GraphOptimizationRewriteKind {
    BroadcastRepeat,
    CommutativeLhsRepeat,
    TwoSidedBroadcastRepeat,
    UnaryBroadcastRepeat,
    IdentityMaterialization,
    NoopNode,
    MetadataOnlyNode,
};

struct GraphOptimizationRewriteContext {
    GraphOptimizationRewriteKind kind = GraphOptimizationRewriteKind::BroadcastRepeat;
    GraphOptimizationBackend backend = GraphOptimizationBackend::Other;
    int nodes_before = 0;
    const ggml_tensor * node = nullptr;
    const ggml_tensor * consumer = nullptr;
    int consumer_src_index = -1;
};

using GraphOptimizationExclusionFn = bool (*)(
    const GraphOptimizationRewriteContext & context,
    const void * user_data);

bool graph_optimization_enabled_from_env();

struct GraphOptimizationOptions {
    GraphOptimizationBackend backend = GraphOptimizationBackend::Other;
    bool enabled = true;
    bool fold_broadcast_repeats = true;
    bool fold_commutative_lhs_repeats = true;
    bool fold_two_sided_broadcast_repeats = true;
    bool fold_unary_broadcast_repeats = true;
    bool fold_identity_materializations = true;
    bool elide_noop_nodes = true;
    bool elide_metadata_only_ops = true;
    GraphOptimizationExclusionFn exclude_rewrite = nullptr;
    const void * exclude_rewrite_user_data = nullptr;
};

struct GraphOptimizationReport {
    int nodes_before = 0;
    int nodes_after = 0;
    int broadcast_repeats_folded = 0;
    int commutative_lhs_repeats_folded = 0;
    int two_sided_broadcast_repeats_folded = 0;
    int unary_broadcast_repeats_folded = 0;
    int candidate_two_sided_broadcast_repeats = 0;
    int identity_materializations_elided = 0;
    int noop_nodes_elided = 0;
    int metadata_only_nodes_elided = 0;
};

GraphOptimizationOptions graph_optimization_options_for_backend(
    GraphOptimizationBackend backend,
    bool enabled = graph_optimization_enabled_from_env());

GraphOptimizationReport optimize_graph(
    ggml_cgraph & graph,
    const GraphOptimizationOptions & options = {});

GraphOptimizationReport optimize_graph(
    ggml_cgraph & graph,
    GraphOptimizationBackend backend,
    bool enabled = graph_optimization_enabled_from_env());

}  // namespace engine::runtime
