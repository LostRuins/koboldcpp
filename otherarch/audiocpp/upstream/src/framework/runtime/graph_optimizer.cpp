#include "engine/framework/runtime/graph_optimizer.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace engine::runtime {

namespace {

bool same_type_shape_and_layout(const ggml_tensor & lhs, const ggml_tensor & rhs) {
    if (lhs.type != rhs.type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i] || lhs.nb[i] != rhs.nb[i]) {
            return false;
        }
    }
    return true;
}

bool is_metadata_only_op(ggml_op op) {
    switch (op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

bool can_elide_from_execution_schedule(const ggml_tensor & node) {
    // Keep explicit outputs addressable in graph dumps and backend tooling.
    if ((node.flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    return true;
}

bool is_identity_materialization(const ggml_tensor & node) {
    if (node.src[0] == nullptr || !same_type_shape_and_layout(node, *node.src[0])) {
        return false;
    }

    switch (node.op) {
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
            return true;
        default:
            return false;
    }
}

bool supports_implicit_broadcast_rhs(ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return true;
        default:
            return false;
    }
}

bool is_commutative_elementwise_op(ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return true;
        default:
            return false;
    }
}

ggml_tensor * resolve_alias(ggml_tensor * node, ggml_tensor * const * nodes, ggml_tensor * const * aliases, int nodes_before) {
    for (int i = 0; i < nodes_before; ++i) {
        bool changed = false;
        for (int j = 0; j < nodes_before; ++j) {
            if (aliases[j] != nullptr && node == nodes[j]) {
                node = aliases[j];
                changed = true;
                break;
            }
        }
        if (!changed) {
            return node;
        }
    }
    return node;
}

bool can_fold_repeat_use(const ggml_tensor & repeat_node, const ggml_tensor & consumer, int src_index) {
    return src_index == 1 &&
        supports_implicit_broadcast_rhs(consumer.op) &&
        repeat_node.src[0] != nullptr &&
        consumer.src[0] != nullptr &&
        ggml_can_repeat(repeat_node.src[0], consumer.src[0]);
}

bool can_fold_lhs_repeat_use(const ggml_tensor & repeat_node, const ggml_tensor & consumer, int src_index) {
    return src_index == 0 &&
        is_commutative_elementwise_op(consumer.op) &&
        repeat_node.src[0] != nullptr &&
        consumer.src[1] != nullptr &&
        ggml_are_same_shape(consumer.src[1], &consumer) &&
        ggml_can_repeat(repeat_node.src[0], consumer.src[1]);
}

bool is_candidate_two_sided_broadcast_repeat(const ggml_tensor & repeat_node, const ggml_tensor & consumer, int src_index) {
    if (src_index != 0 ||
        !is_commutative_elementwise_op(consumer.op) ||
        repeat_node.src[0] == nullptr ||
        consumer.src[1] == nullptr ||
        !ggml_are_same_shape(&repeat_node, &consumer)) {
        return false;
    }

    return ggml_can_repeat(repeat_node.src[0], &consumer) &&
        ggml_can_repeat(consumer.src[1], &consumer) &&
        !ggml_are_same_shape(repeat_node.src[0], &consumer);
}

bool can_fold_two_sided_broadcast_repeat_use(const ggml_tensor & repeat_node, const ggml_tensor & consumer, int src_index) {
    return src_index == 0 &&
        is_commutative_elementwise_op(consumer.op) &&
        repeat_node.src[0] != nullptr &&
        consumer.src[1] != nullptr &&
        ggml_are_same_shape(&repeat_node, &consumer) &&
        ggml_can_repeat(repeat_node.src[0], &consumer) &&
        ggml_can_repeat(consumer.src[1], &consumer);
}

bool supports_implicit_broadcast_src0(ggml_op op) {
    switch (op) {
        case GGML_OP_SCALE:
            return true;
        default:
            return false;
    }
}

bool can_fold_unary_broadcast_repeat_use(const ggml_tensor & repeat_node, const ggml_tensor & consumer, int src_index) {
    return src_index == 0 &&
        supports_implicit_broadcast_src0(consumer.op) &&
        repeat_node.src[0] != nullptr &&
        ggml_are_same_shape(&repeat_node, &consumer) &&
        ggml_can_repeat(repeat_node.src[0], &consumer);
}

bool rewrite_excluded(
    const GraphOptimizationOptions & options,
    GraphOptimizationRewriteKind kind,
    const ggml_tensor * node,
    const ggml_tensor * consumer,
    int consumer_src_index,
    int nodes_before) {
    if (options.exclude_rewrite == nullptr) {
        return false;
    }
    return options.exclude_rewrite(
        GraphOptimizationRewriteContext{
            kind,
            options.backend,
            nodes_before,
            node,
            consumer,
            consumer_src_index,
        },
        options.exclude_rewrite_user_data);
}

}  // namespace

bool graph_optimization_enabled_from_env() {
    const char * value = std::getenv("ENGINE_GRAPH_OPTIMIZER");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }

    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    return normalized != "0" &&
        normalized != "false" &&
        normalized != "off" &&
        normalized != "no";
}

GraphOptimizationOptions graph_optimization_options_for_backend(
    GraphOptimizationBackend backend,
    bool enabled) {
    GraphOptimizationOptions options;
    options.backend = backend;
    options.enabled = enabled;

    switch (backend) {
        case GraphOptimizationBackend::Cpu:
            // Pure graph rewrites plus CPU direct-execution schedule pruning.
            options.fold_broadcast_repeats = true;
            options.fold_commutative_lhs_repeats = true;
            options.fold_two_sided_broadcast_repeats = true;
            options.fold_unary_broadcast_repeats = true;
            options.fold_identity_materializations = true;
            options.elide_noop_nodes = true;
            options.elide_metadata_only_ops = true;
            break;

        case GraphOptimizationBackend::Gpu:
            // Keep only semantic graph rewrites on scheduler-based GPU backends.
            options.fold_broadcast_repeats = true;
            options.fold_commutative_lhs_repeats = true;
            options.fold_two_sided_broadcast_repeats = true;
            options.fold_unary_broadcast_repeats = true;
            options.fold_identity_materializations = true;
            options.elide_noop_nodes = false;
            options.elide_metadata_only_ops = false;
            break;

        case GraphOptimizationBackend::Other:
            options.fold_broadcast_repeats = false;
            options.fold_commutative_lhs_repeats = false;
            options.fold_two_sided_broadcast_repeats = false;
            options.fold_unary_broadcast_repeats = false;
            options.fold_identity_materializations = false;
            options.elide_noop_nodes = false;
            options.elide_metadata_only_ops = false;
            break;
    }

    return options;
}

GraphOptimizationReport optimize_graph(
    ggml_cgraph & graph,
    const GraphOptimizationOptions & options) {
    GraphOptimizationReport report;
    report.nodes_before = ggml_graph_n_nodes(&graph);
    report.nodes_after = report.nodes_before;

    // kcpp: Audio.cpp mutates ggml_cgraph's private node count after rewriting
    // it. KoboldCpp intentionally follows upstream GGML's opaque graph API, so
    // keep this optional optimizer as a measured no-op.
    (void) options;
    return report;

#if 0

    if (!options.enabled || report.nodes_before <= 0) {
        return report;
    }

    ggml_tensor ** nodes = ggml_graph_nodes(&graph);
    std::vector<bool> folded_broadcast_repeats(report.nodes_before, false);
    std::vector<bool> folded_commutative_lhs_repeats(report.nodes_before, false);
    std::vector<bool> folded_two_sided_broadcast_repeats(report.nodes_before, false);
    std::vector<bool> folded_unary_broadcast_repeats(report.nodes_before, false);
    if (options.fold_broadcast_repeats) {
        for (int repeat_index = 0; repeat_index < report.nodes_before - 1; ++repeat_index) {
            ggml_tensor * repeat_node = nodes[repeat_index];
            if (repeat_node == nullptr ||
                repeat_node->op != GGML_OP_REPEAT ||
                !can_elide_from_execution_schedule(*repeat_node)) {
                continue;
            }

            bool has_use = false;
            bool all_uses_foldable = true;
            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer == nullptr) {
                    continue;
                }
                for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
                    if (consumer->src[src_index] != repeat_node) {
                        continue;
                    }
                    has_use = true;
                    if (!can_fold_repeat_use(*repeat_node, *consumer, src_index) ||
                        rewrite_excluded(
                            options,
                            GraphOptimizationRewriteKind::BroadcastRepeat,
                            repeat_node,
                            consumer,
                            src_index,
                            report.nodes_before)) {
                        all_uses_foldable = false;
                    }
                }
            }

            if (!has_use || !all_uses_foldable) {
                continue;
            }

            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer != nullptr && consumer->src[1] == repeat_node) {
                    consumer->src[1] = repeat_node->src[0];
                }
            }
            folded_broadcast_repeats[repeat_index] = true;
        }
    }

    if (options.fold_commutative_lhs_repeats) {
        for (int repeat_index = 0; repeat_index < report.nodes_before - 1; ++repeat_index) {
            ggml_tensor * repeat_node = nodes[repeat_index];
            if (repeat_node == nullptr ||
                repeat_node->op != GGML_OP_REPEAT ||
                folded_broadcast_repeats[repeat_index] ||
                !can_elide_from_execution_schedule(*repeat_node)) {
                continue;
            }

            bool has_use = false;
            bool all_uses_foldable = true;
            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer == nullptr) {
                    continue;
                }
                for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
                    if (consumer->src[src_index] != repeat_node) {
                        continue;
                    }
                    has_use = true;
                    if (!can_fold_lhs_repeat_use(*repeat_node, *consumer, src_index) ||
                        rewrite_excluded(
                            options,
                            GraphOptimizationRewriteKind::CommutativeLhsRepeat,
                            repeat_node,
                            consumer,
                            src_index,
                            report.nodes_before)) {
                        all_uses_foldable = false;
                    }
                }
            }

            if (!has_use || !all_uses_foldable) {
                continue;
            }

            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer != nullptr && consumer->src[0] == repeat_node) {
                    consumer->src[0] = consumer->src[1];
                    consumer->src[1] = repeat_node->src[0];
                }
            }
            folded_commutative_lhs_repeats[repeat_index] = true;
        }
    }

    if (options.fold_two_sided_broadcast_repeats) {
        for (int repeat_index = 0; repeat_index < report.nodes_before - 1; ++repeat_index) {
            ggml_tensor * repeat_node = nodes[repeat_index];
            if (repeat_node == nullptr ||
                repeat_node->op != GGML_OP_REPEAT ||
                folded_broadcast_repeats[repeat_index] ||
                folded_commutative_lhs_repeats[repeat_index] ||
                !can_elide_from_execution_schedule(*repeat_node)) {
                continue;
            }

            bool has_use = false;
            bool all_uses_foldable = true;
            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer == nullptr) {
                    continue;
                }
                for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
                    if (consumer->src[src_index] != repeat_node) {
                        continue;
                    }
                    has_use = true;
                    if (!can_fold_two_sided_broadcast_repeat_use(*repeat_node, *consumer, src_index) ||
                        rewrite_excluded(
                            options,
                            GraphOptimizationRewriteKind::TwoSidedBroadcastRepeat,
                            repeat_node,
                            consumer,
                            src_index,
                            report.nodes_before)) {
                        all_uses_foldable = false;
                    }
                }
            }

            if (!has_use || !all_uses_foldable) {
                continue;
            }

            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer != nullptr && consumer->src[0] == repeat_node) {
                    consumer->src[0] = repeat_node->src[0];
                }
            }
            folded_two_sided_broadcast_repeats[repeat_index] = true;
        }
    }

    if (options.fold_unary_broadcast_repeats) {
        for (int repeat_index = 0; repeat_index < report.nodes_before - 1; ++repeat_index) {
            ggml_tensor * repeat_node = nodes[repeat_index];
            if (repeat_node == nullptr ||
                repeat_node->op != GGML_OP_REPEAT ||
                folded_broadcast_repeats[repeat_index] ||
                folded_commutative_lhs_repeats[repeat_index] ||
                folded_two_sided_broadcast_repeats[repeat_index] ||
                !can_elide_from_execution_schedule(*repeat_node)) {
                continue;
            }

            bool has_use = false;
            bool all_uses_foldable = true;
            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer == nullptr) {
                    continue;
                }
                for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
                    if (consumer->src[src_index] != repeat_node) {
                        continue;
                    }
                    has_use = true;
                    if (!can_fold_unary_broadcast_repeat_use(*repeat_node, *consumer, src_index) ||
                        rewrite_excluded(
                            options,
                            GraphOptimizationRewriteKind::UnaryBroadcastRepeat,
                            repeat_node,
                            consumer,
                            src_index,
                            report.nodes_before)) {
                        all_uses_foldable = false;
                    }
                }
            }

            if (!has_use || !all_uses_foldable) {
                continue;
            }

            for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
                ggml_tensor * consumer = nodes[consumer_index];
                if (consumer != nullptr && consumer->src[0] == repeat_node) {
                    consumer->src[0] = repeat_node->src[0];
                }
            }
            folded_unary_broadcast_repeats[repeat_index] = true;
        }
    }

    for (int repeat_index = 0; repeat_index < report.nodes_before - 1; ++repeat_index) {
        ggml_tensor * repeat_node = nodes[repeat_index];
        if (repeat_node == nullptr ||
            repeat_node->op != GGML_OP_REPEAT ||
            folded_broadcast_repeats[repeat_index] ||
            folded_commutative_lhs_repeats[repeat_index] ||
            folded_two_sided_broadcast_repeats[repeat_index] ||
            folded_unary_broadcast_repeats[repeat_index]) {
            continue;
        }
        bool has_candidate_use = false;
        bool all_uses_candidate = true;
        for (int consumer_index = repeat_index + 1; consumer_index < report.nodes_before; ++consumer_index) {
            ggml_tensor * consumer = nodes[consumer_index];
            if (consumer == nullptr) {
                continue;
            }
            for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
                if (consumer->src[src_index] != repeat_node) {
                    continue;
                }
                has_candidate_use = true;
                if (!is_candidate_two_sided_broadcast_repeat(*repeat_node, *consumer, src_index)) {
                    all_uses_candidate = false;
                }
            }
        }
        if (has_candidate_use && all_uses_candidate) {
            ++report.candidate_two_sided_broadcast_repeats;
        }
    }

    std::vector<ggml_tensor *> aliases(report.nodes_before, nullptr);
    if (options.fold_identity_materializations) {
        for (int i = 0; i < report.nodes_before - 1; ++i) {
            ggml_tensor * node = nodes[i];
            if (node != nullptr &&
                can_elide_from_execution_schedule(*node) &&
                is_identity_materialization(*node) &&
                !rewrite_excluded(
                    options,
                    GraphOptimizationRewriteKind::IdentityMaterialization,
                    node,
                    nullptr,
                    -1,
                    report.nodes_before)) {
                aliases[i] = node->src[0];
            }
        }

        for (int i = 0; i < report.nodes_before; ++i) {
            ggml_tensor * node = nodes[i];
            if (node == nullptr) {
                continue;
            }
            for (ggml_tensor *& src : node->src) {
                if (src != nullptr) {
                    src = resolve_alias(src, nodes, aliases.data(), report.nodes_before);
                }
            }
        }
    }

    int write_index = 0;
    for (int read_index = 0; read_index < report.nodes_before; ++read_index) {
        ggml_tensor * node = nodes[read_index];
        if (node != nullptr && can_elide_from_execution_schedule(*node)) {
            if (folded_broadcast_repeats[read_index]) {
                ++report.broadcast_repeats_folded;
                continue;
            }
            if (folded_commutative_lhs_repeats[read_index]) {
                ++report.commutative_lhs_repeats_folded;
                continue;
            }
            if (folded_two_sided_broadcast_repeats[read_index]) {
                ++report.two_sided_broadcast_repeats_folded;
                continue;
            }
            if (folded_unary_broadcast_repeats[read_index]) {
                ++report.unary_broadcast_repeats_folded;
                continue;
            }
            if (aliases[read_index] != nullptr) {
                ++report.identity_materializations_elided;
                continue;
            }
            if (options.elide_noop_nodes &&
                node->op == GGML_OP_NONE &&
                !rewrite_excluded(
                    options,
                    GraphOptimizationRewriteKind::NoopNode,
                    node,
                    nullptr,
                    -1,
                    report.nodes_before)) {
                ++report.noop_nodes_elided;
                continue;
            }
            if (options.elide_metadata_only_ops &&
                is_metadata_only_op(node->op) &&
                !rewrite_excluded(
                    options,
                    GraphOptimizationRewriteKind::MetadataOnlyNode,
                    node,
                    nullptr,
                    -1,
                    report.nodes_before)) {
                ++report.metadata_only_nodes_elided;
                continue;
            }
        }
        nodes[write_index++] = node;
    }

    for (int i = write_index; i < report.nodes_before; ++i) {
        nodes[i] = nullptr;
    }

    ggml_graph_set_n_nodes(&graph, write_index);
    report.nodes_after = ggml_graph_n_nodes(&graph);
#endif
    return report;
}

GraphOptimizationReport optimize_graph(
    ggml_cgraph & graph,
    GraphOptimizationBackend backend,
    bool enabled) {
    return optimize_graph(graph, graph_optimization_options_for_backend(backend, enabled));
}

}  // namespace engine::runtime
