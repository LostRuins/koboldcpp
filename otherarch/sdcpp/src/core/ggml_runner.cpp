#include <algorithm>
#include <exception>
#include <utility>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"
#include "core/ggml_runner.h"
#include "core/ggml_tensor_utils.h"

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
    params.mem_size = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
    if (compute_ctx_buffer_ == nullptr) {
        const size_t count = (params.mem_size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
        compute_ctx_buffer_.reset(new std::max_align_t[count]);
    }
    // Keep tensor addresses stable between steps so CUDA graphs can finish warming up.
    params.mem_buffer = compute_ctx_buffer_.get();
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
    for (auto* current = tensor; current != nullptr; current = current->view_src) {
        if (params_tensor_set_.count(current) != 0)
            return current;
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
    for (const auto& entry : execution_session_.cache().outputs()) {
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

bool GGMLRunner::runner_start() {
    return execution_engine_->start(execution_session_);
}

void GGMLRunner::runner_end() {
    GGML_ASSERT(!graph_active_);
    execution_engine_->end(execution_session_);
    free_compute_ctx();
    compute_ctx_buffer_.reset();
}

GGMLRunner::GGMLRunner(ggml_backend_t backend,
                       std::shared_ptr<DeviceResidencyManager> manager)
    : execution_engine_(manager != nullptr ? manager->execution_engine() : std::make_shared<sd::ExecutionEngine>()),
      execution_session_(execution_engine_->create_session(backend, manager)),
      runtime_backend(backend),
      residency_manager(manager) {
    GGML_ASSERT(runtime_backend != nullptr);
    alloc_params_ctx();
}

GGMLRunner::~GGMLRunner() {
    runner_end();
    execution_engine_->release_session(execution_session_);
    free_compute_ctx();
    free_params_ctx();
}

GGMLRunnerContext GGMLRunner::get_context(ggml_cgraph* graph) {
    GGMLRunnerContext runner_ctx;
    runner_ctx.ggml_ctx              = compute_ctx;
    runner_ctx.graph                 = graph;
    runner_ctx.backend               = runtime_backend;
    runner_ctx.flash_attn_enabled    = flash_attn_enabled;
    runner_ctx.sage_attn_enabled     = sage_attn_enabled;
    runner_ctx.linear_scale          = linear_scale;
    runner_ctx.attn_scale            = attn_scale;
    runner_ctx.conv2d_direct_enabled = conv2d_direct_enabled;
    runner_ctx.conv3d_direct_enabled = conv3d_direct_enabled;
    runner_ctx.circular_x_enabled    = circular_x_enabled;
    runner_ctx.circular_y_enabled    = circular_y_enabled;
    runner_ctx.weight_adapter        = weight_adapter;
    runner_ctx.debug_tensors         = &debug_tensors;
    runner_ctx.get_cache_tensor      = [this](const std::string& name) {
        return this->get_cache_tensor_by_name(name);
    };
    runner_ctx.cache_tensor = [this, graph](const std::string& name, ggml_tensor* tensor) {
        this->cache(name, tensor, graph);
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
    execution_engine_->clear_cache(execution_session_);
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

void GGMLRunner::cache(const std::string name, ggml_tensor* tensor, ggml_cgraph* graph) {
    if (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = ggml_cont(compute_ctx, tensor);
    }
    if (tensor != nullptr) {
        ggml_set_output(tensor);
    }
    execution_session_.cache().stage(name, tensor);
    if (graph != nullptr && tensor != nullptr) {
        // Schedule the cache output here so its source can be reused before graph end.
        ggml_build_forward_expand(graph, tensor);
    }
}

std::optional<sd::Tensor<float>> GGMLRunner::compute(get_graph_cb_t get_graph,
                                                     int n_threads,
                                                     bool auto_runner_end,
                                                     bool no_return,
                                                     const std::function<bool()>& read_outputs) {
    last_compute_status_ = GGML_STATUS_FAILED;
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
            if (!success) {
                runner.execution_engine_->discard_graph(runner.execution_session_);
            }
            runner.free_compute_ctx();
            runner.graph_active_ = false;
        }
    } graph_guard{*this, success};

    try {
        ggml_cgraph* graph = nullptr;
        if (!prepare_compute_graph(get_graph, &graph)) {
            return std::nullopt;
        }
        params_tensor_set_dirty_ = true;
        rebuild_params_tensor_set();
        sd::ComputeTask task;
        task.description     = get_desc();
        task.graph           = graph;
        task.output          = no_return ? nullptr : ggml_get_tensor(compute_ctx, final_result_name.c_str());
        task.weights         = collect_used_param_tensors(graph);
        task.inputs          = backend_tensor_data_map;
        task.debug_tensors   = debug_tensors;
        task.n_threads       = n_threads;
        task.no_return       = no_return;
        task.read_outputs    = read_outputs;
        auto result          = execution_engine_->execute(execution_session_, task);
        last_compute_status_ = result.status;
        success              = result.status == GGML_STATUS_SUCCESS;
        if (success) {
            return std::move(result.output);
        }
    } catch (const std::bad_alloc&) {
        last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
        LOG_ERROR("%s graph allocation failed", get_desc().c_str());
    } catch (const std::exception& error) {
        LOG_ERROR("%s graph submission failed: %s", get_desc().c_str(), error.what());
    }
    return std::nullopt;
}

void GGMLRunner::set_graph_cut_layer_split_enabled(bool enabled) {
    execution_session_.set_layer_split_enabled(enabled);
}

void GGMLRunner::set_graph_cut_layer_split_backend_vram_limits(const std::vector<size_t>& limits) {
    execution_session_.set_backend_limits(limits);
}

void GGMLRunner::set_runtime_backends(const std::vector<ggml_backend_t>& backends) {
    execution_session_.set_runtime_backends(backends);
}
