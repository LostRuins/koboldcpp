#include <algorithm>
#include <map>
#include <utility>

#include "core/ggml_extend_backend.h"
#include "core/ggml_runner.h"
#include "core/ggml_tensor_utils.h"
#include "core/layer_split_partition.h"
#include "core/segment_graph_bindings.h"
#include "core/segment_weight_pipeline.h"

using namespace sd;

void GGMLRunner::alloc_params_ctx() {
    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    params_ctx = ggml_init(params);
    GGML_ASSERT(params_ctx != nullptr);
    params_tensor_set_.clear();
    params_tensor_set_dirty_ = true;
}

void GGMLRunner::free_params_ctx() {
    if (params_ctx != nullptr) {
        ggml_free(params_ctx);
        params_ctx = nullptr;
    }
    params_tensor_set_.clear();
    params_tensor_set_dirty_ = true;
}

void GGMLRunner::alloc_compute_ctx() {
    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    compute_ctx = ggml_init(params);
    GGML_ASSERT(compute_ctx != nullptr);
}

void GGMLRunner::free_compute_ctx() {
    debug_tensors.clear();
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
    backend_tensor_data_map.clear();
}

void GGMLRunner::rebuild_params_tensor_set() {
    if (!params_tensor_set_dirty_) {
        return;
    }
    params_tensor_set_.clear();
    if (params_ctx == nullptr) {
        return;
    }
    for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
        params_tensor_set_.insert(t);
    }
    params_tensor_set_dirty_ = false;
}

ggml_tensor* GGMLRunner::canonical_param_tensor(ggml_tensor* tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }
    if (params_tensor_set_.find(tensor) != params_tensor_set_.end()) {
        return tensor;
    }
    if (tensor->view_src != nullptr &&
        params_tensor_set_.find(tensor->view_src) != params_tensor_set_.end()) {
        return tensor->view_src;
    }
    return nullptr;
}

std::vector<ggml_tensor*> GGMLRunner::collect_used_param_tensors(ggml_cgraph* gf) {
    std::vector<ggml_tensor*> used_params;
    rebuild_params_tensor_set();
    if (gf == nullptr || params_tensor_set_.empty()) {
        return used_params;
    }

    std::unordered_set<const ggml_tensor*> seen_params;
    const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
    seen_params.reserve(static_cast<size_t>(n_leafs));
    for (int i = 0; i < n_leafs; ++i) {
        ggml_tensor* leaf       = sd::ggml_graph_cut::leaf_tensor(gf, i);
        ggml_tensor* param_leaf = canonical_param_tensor(leaf);
        if (param_leaf != nullptr &&
            seen_params.insert(param_leaf).second) {
            used_params.push_back(param_leaf);
        }
    }
    return used_params;
}

void GGMLRunner::evict_compute_backend_param_tensors(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    auto manager = residency_manager.lock();
    if (manager != nullptr) {
        manager->evict_compute_backend_params(tensors);
    }
}

void GGMLRunner::prepare_build_in_tensor_before() {
    one_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
    ggml_set_name(one_tensor, "ggml_runner_build_in_tensor:one");
    set_backend_tensor_data(one_tensor, one_vec.data());

    zero_int_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
    ggml_set_name(zero_int_tensor, "ggml_runner_build_in_tensor:zero_int");
    set_backend_tensor_data(zero_int_tensor, zero_int_vec.data());
}

void GGMLRunner::prepare_build_in_tensor_after(ggml_cgraph* gf) {
    ggml_build_forward_expand(gf, one_tensor);
    ggml_build_forward_expand(gf, zero_int_tensor);
}

ggml_cgraph* GGMLRunner::new_graph_custom(size_t graph_size) {
    if (weight_adapter) {
        graph_size += weight_adapter->get_extra_graph_size();
    }
    return ggml_new_graph_custom(compute_ctx, graph_size, false);
}

ggml_cgraph* GGMLRunner::get_compute_graph(get_graph_cb_t get_graph) {
    prepare_build_in_tensor_before();
    ggml_cgraph* gf = get_graph();
    if (gf == nullptr) {
        return nullptr;
    }
    if (ggml_graph_n_nodes(gf) > 0) {
        auto result = ggml_graph_node(gf, -1);
        ggml_set_name(result, final_result_name.c_str());
    }
    for (const auto& entry : debug_tensors) {
        if (entry.first != nullptr) {
            ggml_build_forward_expand(gf, entry.first);
        }
    }
    for (const auto& entry : cache_.outputs()) {
        if (entry.second != nullptr) {
            ggml_build_forward_expand(gf, entry.second);
        }
    }
    prepare_build_in_tensor_after(gf);
    return gf;
}

bool GGMLRunner::prepare_compute_graph(get_graph_cb_t get_graph,
                                       ggml_cgraph** gf_out) {
    GGML_ASSERT(gf_out != nullptr);

    reset_compute_ctx();
    ggml_cgraph* gf = get_compute_graph(get_graph);
    if (gf == nullptr) {
        free_compute_ctx();
        return false;
    }

    *gf_out = gf;
    return true;
}

ggml_backend_t GGMLRunner::backend_for_weight(const ggml_tensor* tensor) const {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }
    if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        ggml_backend_buffer_is_host(tensor->buffer)) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    if (dev == nullptr) {
        return nullptr;
    }
    if (ggml_backend_get_device(runtime_backend) == dev) {
        return runtime_backend;
    }
    for (ggml_backend_t backend : extra_runtime_backends) {
        if (ggml_backend_get_device(backend) == dev) {
            return backend;
        }
    }
    return nullptr;
}

void GGMLRunner::pin_multi_device_nodes(ggml_backend_sched_t sched, ggml_cgraph* gf, ggml_cgraph* original_graph) {
    if (sched == nullptr || gf == nullptr) {
        return;
    }
    ggml_backend_t current = runtime_backend;
    const int n_nodes      = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor* node    = ggml_graph_node(gf, i);
        auto node_assignment = graph_cut_layer_split_node_assignments_.find(original_graph == nullptr ? node : ggml_graph_node(original_graph, i));
        if (node_assignment != graph_cut_layer_split_node_assignments_.end()) {
            current = node_assignment->second;
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_backend_t weight_backend = backend_for_weight(node->src[s]);
            if (weight_backend != nullptr) {
                if (node_assignment == graph_cut_layer_split_node_assignments_.end()) {
                    current = weight_backend;
                }
            }
        }
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (ggml_backend_supports_op(current, node)) {
            ggml_backend_sched_set_tensor_backend(sched, node, current);
        }
    }
}

size_t GGMLRunner::retained_runtime_buffer_bytes(ggml_backend_t backend) const {
    backend      = backend == nullptr ? runtime_backend : backend;
    size_t bytes = workspace_.bytes(backend);
    if (backend == runtime_backend) {
        const size_t cache_bytes = cache_.resident_bytes(ggml_backend_get_device(backend));
        bytes                    = cache_bytes > SIZE_MAX - bytes ? SIZE_MAX : bytes + cache_bytes;
        const size_t cut_bytes   = cut_cache_.resident_bytes(ggml_backend_get_device(backend));
        bytes                    = cut_bytes > SIZE_MAX - bytes ? SIZE_MAX : bytes + cut_bytes;
    }
    return bytes;
}

void GGMLRunner::sync_runtime_residency() {
    if (auto manager = residency_manager.lock()) {
        manager->update_runtime_residency(reinterpret_cast<uintptr_t>(this),
                                          runtime_backend, retained_runtime_buffer_bytes());
        for (auto backend : extra_runtime_backends) {
            manager->update_runtime_residency(reinterpret_cast<uintptr_t>(this),
                                              backend, retained_runtime_buffer_bytes(backend));
        }
    }
}

std::optional<sd::Tensor<float>> GGMLRunner::read_graph_tensor(ggml_tensor* tensor, const char* label) {
    if (tensor == nullptr) {
        LOG_ERROR("%s %s tensor is null", get_desc().c_str(), label);
        return std::nullopt;
    }
    if (tensor->type != GGML_TYPE_F32) {
        LOG_ERROR("%s %s tensor type mismatch: got %s",
                  get_desc().c_str(),
                  label,
                  ggml_type_name(tensor->type));
        return std::nullopt;
    }
    ggml_backend_buffer_t buf = sd::ggml_graph_cut::tensor_buffer(tensor);
    if (buf == nullptr) {
        LOG_ERROR("%s %s tensor buffer missing: name=%s op=%s buffer=%p view_src=%p view_src_buffer=%p data=%p",
                  get_desc().c_str(),
                  label,
                  tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                  ggml_op_name(tensor->op),
                  tensor->buffer,
                  tensor->view_src,
                  tensor->view_src ? tensor->view_src->buffer : nullptr,
                  tensor->data);
        return std::nullopt;
    }

    return sd::make_sd_tensor_from_ggml<float>(tensor);
}

void GGMLRunner::copy_data_to_backend_tensor(ggml_cgraph* gf, bool clear_after_copy) {
    GGML_ASSERT(gf != nullptr);
    std::unordered_set<const ggml_tensor*> graph_tensor_set;
    const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
    const int n_nodes = ggml_graph_n_nodes(gf);
    graph_tensor_set.reserve(static_cast<size_t>(n_leafs + n_nodes));
    for (int i = 0; i < n_leafs; ++i) {
        graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
    }
    for (int i = 0; i < n_nodes; ++i) {
        graph_tensor_set.insert(ggml_graph_node(gf, i));
    }

    for (auto& kv : backend_tensor_data_map) {
        auto tensor = kv.first;
        auto data   = kv.second;
        if (tensor == nullptr || data == nullptr) {
            continue;
        }
        const char* name = ggml_get_name(tensor);
        if (graph_tensor_set.find(tensor) == graph_tensor_set.end()) {
            continue;
        }
        if (tensor->buffer == nullptr) {
            LOG_WARN("%s skip backend tensor copy: tensor buffer not set, name='%s', ne=[%lld,%lld,%lld,%lld], type=%s",
                     get_desc().c_str(),
                     name != nullptr ? name : "",
                     (long long)tensor->ne[0],
                     (long long)tensor->ne[1],
                     (long long)tensor->ne[2],
                     (long long)tensor->ne[3],
                     ggml_type_name(tensor->type));
            continue;
        }

        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (buf == nullptr) {
            LOG_WARN("%s graph exec skip tensor copy: name=%s op=%s reason=buffer_not_set data=%p view_src=%p view_src_buffer=%p",
                     get_desc().c_str(),
                     tensor && tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                     tensor ? ggml_op_name(tensor->op) : "<null>",
                     data,
                     tensor ? tensor->view_src : nullptr,
                     (tensor && tensor->view_src) ? tensor->view_src->buffer : nullptr);
            continue;
        }

        ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
    }

    if (clear_after_copy) {
        backend_tensor_data_map.clear();
    }
}

bool GGMLRunner::resolve_graph_cut_plan(ggml_cgraph* gf,
                                        GraphCutPlan* plan_out) {
    GGML_ASSERT(plan_out != nullptr);
    GGML_ASSERT(gf != nullptr);
    *plan_out = sd::ggml_graph_cut::resolve_plan(runtime_backend,
                                                 gf,
                                                 &graph_cut_plan_cache_,
                                                 params_tensor_set_,
                                                 get_desc().c_str());
    return true;
}

bool GGMLRunner::resolve_graph_cut_layer_split_plan(ggml_cgraph* gf,
                                                    GraphCutPlan* plan_out) {
    return resolve_graph_cut_plan(gf, plan_out);
}

bool GGMLRunner::assign_graph_cut_layer_split_backends(ggml_cgraph* gf) {
    graph_cut_layer_split_node_assignments_.clear();
    if (!graph_cut_layer_split_enabled) {
        return true;
    }
    if (!is_multi_device()) {
        LOG_ERROR("%s graph-cut layer split requires multiple runtime backends", get_desc().c_str());
        return false;
    }

    GraphCutPlan plan;
    if (!resolve_graph_cut_layer_split_plan(gf, &plan)) {
        return false;
    }
    if (!plan.valid || !plan.has_cuts || plan.segments.size() <= 1) {
        auto manager = residency_manager.lock();
        if (manager == nullptr) {
            LOG_ERROR("%s weight manager is not set for graph-cut layer split", get_desc().c_str());
            return false;
        }
        std::vector<ggml_tensor*> graph_params = collect_used_param_tensors(gf);
        if (!graph_params.empty() &&
            !manager->assign_compute_backend(graph_params, runtime_backend)) {
            LOG_ERROR("%s graph-cut layer split failed to assign unmarked graph params to %s",
                      get_desc().c_str(),
                      sd::layer_split_backend_device_display_name(runtime_backend).c_str());
            return false;
        }
        for (ggml_tensor* param : graph_params) {
            if (param != nullptr) {
                graph_cut_layer_split_assignments_[param] = runtime_backend;
            }
        }
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (node != nullptr) {
                graph_cut_layer_split_node_assignments_[node] = runtime_backend;
            }
        }
        if (!graph_cut_layer_split_primary_notice_logged_) {
            LOG_WARN("%s graph-cut layer split: graph has no mark_graph_cut segments; using primary backend %s for %zu graph params",
                     get_desc().c_str(),
                     sd::layer_split_backend_device_display_name(runtime_backend).c_str(),
                     graph_params.size());
            graph_cut_layer_split_primary_notice_logged_ = true;
        } else {
            LOG_VERBOSE("%s graph-cut layer split: graph has no mark_graph_cut segments; using primary backend %s for %zu graph params",
                        get_desc().c_str(),
                        sd::layer_split_backend_device_display_name(runtime_backend).c_str(),
                        graph_params.size());
        }
        return true;
    }

    std::vector<ggml_backend_t> split_backends;
    split_backends.reserve(extra_runtime_backends.size() + 1);
    split_backends.push_back(runtime_backend);
    for (ggml_backend_t backend : extra_runtime_backends) {
        if (backend != nullptr) {
            split_backends.push_back(backend);
        }
    }

    auto manager = residency_manager.lock();
    if (manager == nullptr) {
        LOG_ERROR("%s weight manager is not set for graph-cut layer split", get_desc().c_str());
        return false;
    }

    sd::GraphCutLayerSplitAssignment assignment;
    auto canonicalize_param = [this](ggml_tensor* tensor) {
        return canonical_param_tensor(tensor);
    };
    if (!sd::partition_graph_cut_layer_split(get_desc().c_str(),
                                             gf,
                                             plan,
                                             split_backends,
                                             graph_cut_layer_split_backend_vram_limits_,
                                             max_graph_vram_bytes,
                                             graph_cut_layer_split_assignments_,
                                             canonicalize_param,
                                             &assignment)) {
        return false;
    }

    for (size_t i = 0; i < split_backends.size(); i++) {
        if (assignment.tensors_by_backend[i].empty()) {
            continue;
        }
        if (!manager->assign_compute_backend(assignment.tensors_by_backend[i], split_backends[i])) {
            LOG_ERROR("%s graph-cut layer split failed to assign params to %s",
                      get_desc().c_str(),
                      sd::layer_split_backend_device_display_name(split_backends[i]).c_str());
            return false;
        }
    }

    graph_cut_layer_split_node_assignments_ = std::move(assignment.node_assignments);
    sd::log_graph_cut_layer_split_assignment(get_desc().c_str(), split_backends, assignment);

    return true;
}

bool GGMLRunner::runner_start() {
    if (runner_started_) {
        return true;
    }
    cache_.clear();
    workspace_.set_extra_backends(extra_runtime_backends);
    if (auto manager = residency_manager.lock()) {
        manager->set_workspace_reclaimer(reinterpret_cast<uintptr_t>(this), [this]() {
            if (!workspace_.release()) {
                return false;
            }
            sync_runtime_residency();
            return true;
        });
    }
    runner_started_ = true;
    return true;
}

void GGMLRunner::runner_end() {
    GGML_ASSERT(!graph_active_);
    if (!runner_started_) {
        return;
    }
    workspace_.release();
    cache_.clear();
    logged_compute_bytes_.clear();
    logged_segment_count_ = 0;
    if (auto manager = residency_manager.lock()) {
        manager->clear_prefetched_params(reinterpret_cast<uintptr_t>(this));
        std::vector<ggml_tensor*> tensors;
        for (auto tensor = ggml_get_first_tensor(params_ctx); tensor != nullptr;
             tensor      = ggml_get_next_tensor(params_ctx, tensor)) {
            tensors.push_back(tensor);
        }
        manager->evict_compute_backend_params(tensors);
        manager->remove_runtime_owner(reinterpret_cast<uintptr_t>(this));
    }
    runner_started_ = false;
}

GGMLRunner::GGMLRunner(ggml_backend_t backend,
                       std::shared_ptr<DeviceResidencyManager> manager)
    : runtime_backend(backend),
      cache_(backend),
      cut_cache_(backend),
      workspace_(backend),
      residency_manager(manager) {
    GGML_ASSERT(runtime_backend != nullptr);
    alloc_params_ctx();
}

GGMLRunner::~GGMLRunner() {
    runner_end();
    free_compute_ctx();
    free_params_ctx();
}

GGMLRunnerContext GGMLRunner::get_context() {
    GGMLRunnerContext runner_ctx;
    runner_ctx.ggml_ctx              = compute_ctx;
    runner_ctx.backend               = runtime_backend;
    runner_ctx.flash_attn_enabled    = flash_attn_enabled;
    runner_ctx.conv2d_direct_enabled = conv2d_direct_enabled;
    runner_ctx.circular_x_enabled    = circular_x_enabled;
    runner_ctx.circular_y_enabled    = circular_y_enabled;
    runner_ctx.weight_adapter        = weight_adapter;
    runner_ctx.debug_tensors         = &debug_tensors;
    runner_ctx.get_cache_tensor      = [this](const std::string& name) {
        return this->get_cache_tensor_by_name(name);
    };
    runner_ctx.cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
        this->cache(name, tensor);
    };
    runner_ctx.set_backend_tensor_data = [this](ggml_tensor* tensor, const void* data) {
        this->set_backend_tensor_data(tensor, data);
    };
    return runner_ctx;
}

void GGMLRunner::reset_compute_ctx() {
    free_compute_ctx();
    alloc_compute_ctx();
}

void GGMLRunner::free_cache_ctx_and_buffer() {
    cache_.clear();
    sync_runtime_residency();
}

void GGMLRunner::set_backend_tensor_data(ggml_tensor* tensor, const void* data) {
    // The scheduler only allocates standalone data tensors when they are
    // marked as graph inputs. The flag is harmless for single-backend graphs.
    ggml_set_input(tensor);
    backend_tensor_data_map[tensor] = data;
}

ggml_tensor* GGMLRunner::to_backend(ggml_tensor* tensor) {
    GGML_ASSERT(compute_ctx != nullptr);
    if (tensor == nullptr) {
        return nullptr;
    }
    // it's performing a compute, check if backend isn't cpu
    if (!sd_backend_is_cpu(runtime_backend) && (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer))) {
        // pass input tensors to gpu memory
        auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

        set_backend_tensor_data(backend_tensor, tensor->data);
        return backend_tensor;
    } else {
        return tensor;
    }
}

void GGMLRunner::cache(const std::string name, ggml_tensor* tensor) {
    if (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = ggml_cont(compute_ctx, tensor);
    }
    if (tensor != nullptr) {
        ggml_set_output(tensor);
    }
    cache_.stage(name, tensor);
}

std::optional<sd::Tensor<float>> GGMLRunner::compute(get_graph_cb_t get_graph,
                                                     int n_threads,
                                                     bool auto_runner_end,
                                                     bool no_return,
                                                     const std::function<bool()>& read_outputs) {
    if (graph_active_) {
        LOG_ERROR("%s does not support reentrant graph execution", get_desc().c_str());
        return std::nullopt;
    }
    if (!runner_start()) {
        runner_end();
        return std::nullopt;
    }
    struct RunnerEndGuard {
        GGMLRunner& runner;
        bool enabled;
        ~RunnerEndGuard() {
            if (enabled) {
                runner.runner_end();
            }
        }
    } runner_guard{*this, auto_runner_end};
    graph_active_ = true;
    bool success  = false;
    struct GraphEndGuard {
        GGMLRunner& runner;
        const bool& success;
        ~GraphEndGuard() {
            runner.workspace_.segment_end();
            runner.cache_.graph_end(false);
            runner.cut_cache_.clear();
            runner.free_compute_ctx();
            runner.graph_active_ = false;
            if (!success) {
                runner.workspace_.release();
            }
            runner.sync_runtime_residency();
        }
    } graph_guard{*this, success};

    ggml_cgraph* graph = nullptr;
    if (!prepare_compute_graph(get_graph, &graph)) {
        return std::nullopt;
    }
    rebuild_params_tensor_set();
    auto output = execute_graph(graph, n_threads, no_return, read_outputs);
    success     = output.has_value();
    if (success) {
        cache_.graph_end(true);
    }
    return output;
}

void GGMLRunner::set_graph_cut_layer_split_enabled(bool enabled) {
    graph_cut_layer_split_enabled = enabled;
    if (!enabled) {
        graph_cut_layer_split_assignments_.clear();
        graph_cut_layer_split_node_assignments_.clear();
        graph_cut_layer_split_primary_notice_logged_ = false;
    }
}

void GGMLRunner::set_graph_cut_layer_split_backend_vram_limits(const std::vector<size_t>& limits) {
    graph_cut_layer_split_backend_vram_limits_ = limits;
    graph_cut_layer_split_assignments_.clear();
    graph_cut_layer_split_node_assignments_.clear();
    graph_cut_layer_split_primary_notice_logged_ = false;
}

void GGMLRunner::set_runtime_backends(const std::vector<ggml_backend_t>& backends) {
    extra_runtime_backends.clear();
    for (ggml_backend_t backend : backends) {
        if (backend == nullptr || backend == runtime_backend) {
            continue;
        }
        if (std::find(extra_runtime_backends.begin(), extra_runtime_backends.end(), backend) ==
            extra_runtime_backends.end()) {
            extra_runtime_backends.push_back(backend);
        }
    }
    workspace_.set_extra_backends(extra_runtime_backends);
    graph_cut_layer_split_assignments_.clear();
    graph_cut_layer_split_node_assignments_.clear();
    graph_cut_layer_split_primary_notice_logged_ = false;
}

static size_t add_bytes(size_t a, size_t b) {
    return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

ComputeWorkspace::Measurement GGMLRunner::measure(ggml_cgraph* graph, size_t direct_bytes) {
    auto external_backend = [&](const ggml_tensor* tensor) -> ggml_backend_t {
        if (!params_tensor_set_.count(tensor)) {
            return nullptr;
        }
        auto placement = graph_cut_layer_split_assignments_.find(tensor);
        return placement == graph_cut_layer_split_assignments_.end() ? runtime_backend : placement->second;
    };
    auto assign_nodes = [&](ggml_backend_sched_t scheduler, ggml_cgraph* copy) {
        pin_multi_device_nodes(scheduler, copy, graph);
    };
    return workspace_.measure(graph, direct_bytes, external_backend, assign_nodes);
}

std::vector<DeviceMemoryRequest> GGMLRunner::memory_requests(
    const std::vector<BackendBufferSize>& sizes,
    size_t pending_cache_bytes) const {
    std::vector<DeviceMemoryRequest> requests;
    for (const auto& size : sizes) {
        const size_t retained    = retained_runtime_buffer_bytes(size.backend);
        const size_t reusable    = workspace_.bytes(size.backend);
        const size_t cache_bytes = size.backend == runtime_backend ? pending_cache_bytes : 0;
        const size_t pending     = add_bytes(size.bytes > reusable ? size.bytes - reusable : 0, cache_bytes);
        size_t limit             = max_graph_vram_bytes;
        if (is_multi_device()) {
            size_t index = 0;
            if (size.backend != runtime_backend) {
                auto position = std::find(extra_runtime_backends.begin(), extra_runtime_backends.end(), size.backend);
                index         = static_cast<size_t>(position - extra_runtime_backends.begin()) + 1;
            }
            if (index < graph_cut_layer_split_backend_vram_limits_.size()) {
                limit = graph_cut_layer_split_backend_vram_limits_[index];
            }
        }
        requests.push_back({size.backend, reinterpret_cast<uintptr_t>(this), pending,
                            retained, limit});
    }
    return requests;
}

bool GGMLRunner::fits(const std::vector<DeviceMemoryRequest>& requests,
                      const std::vector<ggml_tensor*>& params) const {
    auto manager = residency_manager.lock();
    if (manager == nullptr) {
        return params.empty();
    }
    for (const auto& request : requests) {
        if (!manager->fits_compute_backend_capacity(request, params)) {
            return false;
        }
    }
    return true;
}

bool GGMLRunner::execute_segment(ggml_cgraph* graph, int n_threads) {
    if (sd_backend_is_cpu(runtime_backend)) {
        sd_backend_cpu_set_n_threads(runtime_backend, n_threads);
    }
    if (workspace_.cpu_backend() != nullptr) {
        sd_backend_cpu_set_n_threads(workspace_.cpu_backend(), n_threads);
    }
    auto scheduler = workspace_.scheduler();
    ggml_status status;
    if (scheduler != nullptr) {
        if (sd_get_backend_eval_callback() != nullptr && !multi_device_eval_callback_warned) {
            LOG_WARN("%s: eval callback is not supported with the backend scheduler; ignoring", get_desc().c_str());
            multi_device_eval_callback_warned = true;
        }
        status = ggml_backend_sched_graph_compute(scheduler, graph);
    } else {
        status = sd_backend_graph_compute_with_eval_callback(runtime_backend, graph,
                                                             sd_get_backend_eval_callback(),
                                                             sd_get_backend_eval_callback_data());
    }
    workspace_.synchronize();
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
        return false;
    }
    const std::string description = get_desc();
    if (!debug_tensors.empty()) {
        std::unordered_set<const ggml_tensor*> graph_tensors;
        const int leaf_count = ggml_graph_cut::leaf_count(graph);
        const int node_count = ggml_graph_n_nodes(graph);
        graph_tensors.reserve(static_cast<size_t>(leaf_count + node_count));
        for (int index = 0; index < leaf_count; ++index) {
            graph_tensors.insert(ggml_graph_cut::leaf_tensor(graph, index));
        }
        for (int index = 0; index < node_count; ++index) {
            graph_tensors.insert(ggml_graph_node(graph, index));
        }

        for (const auto& entry : debug_tensors) {
            ggml_tensor* tensor = entry.first;
            if (tensor == nullptr || graph_tensors.find(tensor) == graph_tensors.end()) {
                continue;
            }
            ggml_backend_buffer_t buffer =
                tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
            if (buffer == nullptr) {
                LOG_WARN("%s skip debug tensor '%s': tensor buffer not set",
                         description.c_str(),
                         entry.second.c_str());
                continue;
            }
            if (tensor->type != GGML_TYPE_F32) {
                LOG_WARN("%s skip debug tensor '%s': only GGML_TYPE_F32 is supported, got %s",
                         description.c_str(),
                         entry.second.c_str(),
                         ggml_type_name(tensor->type));
                continue;
            }
            auto debug_tensor = make_sd_tensor_from_ggml<float>(tensor);
            print_sd_tensor(debug_tensor, false, entry.second.c_str());
        }
    }

    return true;
}

std::optional<Tensor<float>> GGMLRunner::execute_graph(ggml_cgraph* graph, int n_threads, bool no_return, const std::function<bool()>& read_outputs) {
    if (!assign_graph_cut_layer_split_backends(graph)) {
        return std::nullopt;
    }
    const auto params = collect_used_param_tensors(graph);
    ggml_graph_cut::Plan plan;
    if (!resolve_graph_cut_plan(graph, &plan)) {
        return std::nullopt;
    }
    const auto full_measurement = measure(graph, plan.compute_buffer_size);
    if (full_measurement.buffers.empty()) {
        return std::nullopt;
    }
    auto manager         = residency_manager.lock();
    const bool segmented = !is_multi_device() && !sd_backend_is_cpu(runtime_backend) &&
                           manager != nullptr && manager->segmented_compute_enabled() &&
                           plan.valid && plan.has_cuts && plan.segments.size() > 1 &&
                           !fits(memory_requests(full_measurement.buffers, cache_.pending_bytes(graph)), params);
    if (!segmented) {
        ggml_graph_cut::Segment segment;
        segment.group_name          = "graph";
        segment.compute_buffer_size = plan.compute_buffer_size;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            segment.internal_node_indices.push_back(i);
        }
        for (int i = 0; i < ggml_graph_cut::leaf_count(graph); ++i) {
            auto tensor = ggml_graph_cut::leaf_tensor(graph, i);
            ggml_graph_cut::Segment::InputRef input;
            input.leaf_index = i;
            input.type       = canonical_param_tensor(tensor) != nullptr
                                   ? ggml_graph_cut::Segment::INPUT_PARAM
                                   : ggml_graph_cut::Segment::INPUT_EXTERNAL;
            segment.input_refs.push_back(input);
        }
        plan.segments = {std::move(segment)};
    }
    const bool segments_changed = plan.segments.size() != logged_segment_count_;
    if (segments_changed && (segmented || logged_segment_count_ > 1)) {
        LOG_VERBOSE("%s using %zu segment%s", get_desc().c_str(),
                    plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
    }
    SegmentGraphBindings bindings(cut_cache_, plan, graph);
    SegmentWeightPipeline weights(manager, runtime_backend, reinterpret_cast<uintptr_t>(this),
                                  graph, plan, params_tensor_set_,
                                  segmented && manager != nullptr && manager->prefetch_enabled());

    std::map<ggml_backend_t, size_t> peak_compute_bytes;
    auto track_compute_buffer = [&](ggml_backend_t backend) {
        if (backend != nullptr) {
            auto& peak = peak_compute_bytes[backend];
            peak       = std::max(peak, workspace_.bytes(backend));
        }
    };
    std::optional<Tensor<float>> output = Tensor<float>();
    for (size_t index = 0; index < plan.segments.size(); ++index) {
        const auto& segment = plan.segments[index];
        const bool last     = index + 1 == plan.segments.size();
        auto fail_segment   = [&](const char* phase) {
            LOG_ERROR("%s segment %zu/%zu (%s) failed during %s", get_desc().c_str(),
                        index + 1, plan.segments.size(), segment.group_name.c_str(), phase);
            return std::nullopt;
        };
        cut_cache_.prune(segment.live_cut_names);
        bindings.reset(segment);
        if (!bindings.bind_cached_inputs(segment, get_desc().c_str())) {
            return fail_segment("input binding");
        }
        ggml_context* segment_context = nullptr;
        auto segment_graph            = segmented
                                            ? ggml_graph_cut::build_segment_graph(graph, segment, &segment_context)
                                            : graph;
        struct SegmentCleanup {
            GGMLRunner& runner;
            SegmentWeightPipeline& weights;
            SegmentGraphBindings& bindings;
            ggml_context* context;
            ~SegmentCleanup() {
                runner.workspace_.segment_end();
                bindings.restore();
                weights.segment_end();
                ggml_free(context);
                runner.sync_runtime_residency();
            }
        } segment_cleanup{*this, weights, bindings, segment_context};

        auto measurement = segmented ? measure(segment_graph, segment.compute_buffer_size) : full_measurement;
        if (!workspace_.prepare(measurement)) {
            return fail_segment("workspace preparation");
        }
        const size_t cut_bytes       = last ? 0 : cut_cache_.estimate_output_bytes(graph, segment);
        const size_t new_cache_bytes = add_bytes(cut_bytes, cache_.pending_bytes(segment_graph));
        auto ensure_capacity         = [&]() {
            sync_runtime_residency();
            auto requests = memory_requests(measurement.buffers, new_cache_bytes);
            if (!fits(requests, weights.params(index)) && workspace_.release_excess(measurement)) {
                sync_runtime_residency();
                requests = memory_requests(measurement.buffers, new_cache_bytes);
            }
            return weights.ensure_segment_capacity(index, requests);
        };
        if (!weights.segment_start(index, ensure_capacity)) {
            return fail_segment("weight preparation");
        }
        // Preparing weights can execute LoRA graphs and reclaim an idle workspace.
        if (!workspace_.measurement_matches(segment_graph, measurement)) {
            measurement = measure(segment_graph, segment.compute_buffer_size);
        }
        if (!workspace_.prepare(measurement) || !ensure_capacity()) {
            return fail_segment("workspace capacity check");
        }
        if (!workspace_.allocate(segment_graph, [&](ggml_backend_sched_t scheduler, ggml_cgraph* current) {
                pin_multi_device_nodes(scheduler, current);
            })) {
            return fail_segment("workspace allocation");
        }
        for (const auto& size : measurement.buffers) {
            track_compute_buffer(size.backend);
        }
        if (workspace_.scheduler() != nullptr) {
            track_compute_buffer(workspace_.cpu_backend());
        }
        if (!ensure_capacity()) {
            return fail_segment("allocated capacity check");
        }
        copy_data_to_backend_tensor(segment_graph, false);
        auto prefetch_requests = memory_requests(measurement.buffers, new_cache_bytes);
        if (!prefetch_requests.empty()) {
            weights.enqueue_next(index, prefetch_requests.front());
        }
        LOG_DEBUG("%s executing segment %zu/%zu: %s", get_desc().c_str(),
                  index + 1, plan.segments.size(), segment.group_name.c_str());
        if (!execute_segment(segment_graph, n_threads) ||
            !cache_.capture(segment_graph) ||
            !cut_cache_.capture(graph, segment, get_desc().c_str())) {
            return fail_segment("execution or output caching");
        }
        sync_runtime_residency();
        if (last) {
            if (read_outputs && !read_outputs()) {
                return fail_segment("output finalization");
            }
            if (!no_return) {
                auto result = ggml_get_tensor(compute_ctx, final_result_name.c_str());
                output      = read_graph_tensor(result, "output");
                if (!output.has_value()) {
                    return fail_segment("output readback");
                }
            }
        }
        // Final outputs and their callbacks may still be views of consumed cuts.
        cut_cache_.prune(segment.future_cut_names);
    }
    if (segments_changed || peak_compute_bytes != logged_compute_bytes_) {
        for (const auto& entry : peak_compute_bytes) {
            LOG_VERBOSE("%s compute buffer size: %.2f MB(%s) on %s (peak across %zu segment%s)",
                        get_desc().c_str(), entry.second / (1024.0 * 1024.0),
                        sd_backend_is_cpu(entry.first) ? "RAM" : "VRAM", ggml_backend_name(entry.first),
                        plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
        }
        logged_compute_bytes_ = std::move(peak_compute_bytes);
        logged_segment_count_ = plan.segments.size();
    }
    return output;
}
