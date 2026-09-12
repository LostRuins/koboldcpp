#ifndef __SD_CORE_GGML_RUNNER_H__
#define __SD_CORE_GGML_RUNNER_H__

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/compute_workspace.h"
#include "core/ggml_graph_cut.h"
#include "core/runner_cache.h"
#include "core/tensor_ggml.hpp"
#include "core/util.h"
#include "device_residency_manager.h"

/* SDXL with LoRA requires more space */
#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE 327680

struct WeightAdapter {
    struct ForwardParams {
        enum class op_type_t {
            OP_LINEAR,
            OP_CONV2D,
        } op_type;
        struct {
            bool force_prec_f32 = false;
            float scale         = 1.f;
        } linear;
        struct conv2d_params_t {
            int s0          = 1;
            int s1          = 1;
            int p0          = 0;
            int p1          = 0;
            int d0          = 1;
            int d1          = 1;
            bool direct     = false;
            bool circular_x = false;
            bool circular_y = false;
            float scale     = 1.f;
        } conv2d;
    };
    virtual ggml_tensor* patch_weight(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* weight, const std::string& weight_name) = 0;
    virtual ggml_tensor* forward_with_lora(ggml_context* ctx,
                                           ggml_backend_t backend,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* b,
                                           const std::string& prefix,
                                           ForwardParams forward_params)                                                              = 0;
    virtual ggml_tensor* add_lora_to_output(ggml_context* ctx,
                                            ggml_backend_t backend,
                                            ggml_tensor* x,
                                            ggml_tensor* w,
                                            ggml_tensor* output,
                                            const std::string& prefix,
                                            ForwardParams forward_params)                                                             = 0;
    virtual size_t get_extra_graph_size()                                                                                             = 0;
};

struct GGMLRunnerContext {
    ggml_backend_t backend                                           = nullptr;
    ggml_context* ggml_ctx                                           = nullptr;
    bool flash_attn_enabled                                          = false;
    bool conv2d_direct_enabled                                       = false;
    bool circular_x_enabled                                          = false;
    bool circular_y_enabled                                          = false;
    ggml_tensor* ip_context                                          = nullptr;
    float ip_scale                                                   = 1.0f;
    std::shared_ptr<WeightAdapter> weight_adapter                    = nullptr;
    std::vector<std::pair<ggml_tensor*, std::string>>* debug_tensors = nullptr;
    std::function<ggml_tensor*(const std::string&)> get_cache_tensor;
    std::function<void(const std::string&, ggml_tensor*)> cache_tensor;
    std::function<void(ggml_tensor*, const void*)> set_backend_tensor_data;
    std::map<std::pair<ggml_tensor*, int>, ggml_tensor*> int8_convrot_cache;

    void capture_tensor(const std::string& name, ggml_tensor* tensor) {
        if (debug_tensors == nullptr || tensor == nullptr) {
            return;
        }
        ggml_tensor* snapshot = ggml_cont(ggml_ctx, tensor);
        ggml_tensor* dst      = ggml_dup_tensor(ggml_ctx, snapshot);
        snapshot              = ggml_cpy(ggml_ctx, snapshot, dst);
        ggml_set_output(snapshot);
        debug_tensors->push_back({snapshot, name});
    }

    ggml_tensor* load_cache_tensor(const std::string& name) const {
        if (!get_cache_tensor) {
            return nullptr;
        }
        return get_cache_tensor(name);
    }

    void persist_cache_tensor(const std::string& name, ggml_tensor* tensor) const {
        if (!cache_tensor || tensor == nullptr) {
            return;
        }
        cache_tensor(name, tensor);
    }

    void bind_backend_tensor_data(ggml_tensor* tensor, const void* data) const {
        if (!set_backend_tensor_data || tensor == nullptr || data == nullptr) {
            return;
        }
        set_backend_tensor_data(tensor, data);
    }
};

struct GGMLRunner {
private:
    std::map<ggml_backend_t, size_t> logged_compute_bytes_;
    size_t logged_segment_count_ = 0;

    sd::ComputeWorkspace::Measurement measure(ggml_cgraph* graph, size_t direct_bytes);
    std::vector<DeviceMemoryRequest> memory_requests(const std::vector<sd::BackendBufferSize>& sizes,
                                                     size_t pending_cache_bytes) const;
    bool fits(const std::vector<DeviceMemoryRequest>& requests,
              const std::vector<ggml_tensor*>& params) const;
    bool execute_segment(ggml_cgraph* graph, int n_threads);
    std::optional<sd::Tensor<float>> execute_graph(ggml_cgraph* graph, int n_threads, bool no_return, const std::function<bool()>& read_outputs);

protected:
    typedef std::function<ggml_cgraph*()> get_graph_cb_t;
    using GraphCutPlan = sd::ggml_graph_cut::Plan;

    ggml_backend_t runtime_backend = nullptr;

    ggml_context* params_ctx = nullptr;

    sd::RunnerCache cache_;
    sd::GraphCutTensorCache cut_cache_;
    sd::ComputeWorkspace workspace_;
    ggml_context* compute_ctx = nullptr;
    bool runner_started_      = false;
    bool graph_active_        = false;

    size_t max_graph_vram_bytes        = 0;
    bool graph_cut_layer_split_enabled = false;
    std::vector<size_t> graph_cut_layer_split_backend_vram_limits_;

    std::vector<ggml_backend_t> extra_runtime_backends;  // borrowed (SDBackendManager-owned)
    bool multi_device_eval_callback_warned = false;

    std::shared_ptr<WeightAdapter> weight_adapter = nullptr;
    std::weak_ptr<DeviceResidencyManager> residency_manager;
    bool params_tensor_set_dirty_ = true;

    std::vector<float> one_vec = {1.f};
    ggml_tensor* one_tensor    = nullptr;

    std::vector<int> zero_int_vec = {0};
    ggml_tensor* zero_int_tensor  = nullptr;

    std::map<ggml_tensor*, const void*> backend_tensor_data_map;
    std::vector<std::pair<ggml_tensor*, std::string>> debug_tensors;
    const std::string final_result_name = "ggml_runner_final_result_tensor";

    bool flash_attn_enabled    = false;
    bool conv2d_direct_enabled = false;
    bool circular_x_enabled    = false;
    bool circular_y_enabled    = false;

    sd::ggml_graph_cut::PlanCache graph_cut_plan_cache_;
    std::unordered_set<const ggml_tensor*> params_tensor_set_;
    std::unordered_map<const ggml_tensor*, ggml_backend_t> graph_cut_layer_split_assignments_;
    std::unordered_map<const ggml_tensor*, ggml_backend_t> graph_cut_layer_split_node_assignments_;
    bool graph_cut_layer_split_primary_notice_logged_ = false;

    template <typename T>
    static sd::Tensor<T> take_or_empty(std::optional<sd::Tensor<T>> tensor) {
        if (!tensor.has_value()) {
            return {};
        }
        return std::move(*tensor);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(std::optional<sd::Tensor<T>> tensor,
                                                         size_t expected_dim) {
        return restore_trailing_singleton_dims(take_or_empty(std::move(tensor)), expected_dim);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(sd::Tensor<T> tensor,
                                                         size_t expected_dim) {
        if (tensor.empty()) {
            return tensor;
        }
        while (static_cast<size_t>(tensor.dim()) < expected_dim) {
            tensor.unsqueeze_(tensor.dim());
        }
        return tensor;
    }

    void alloc_params_ctx();

    void free_params_ctx();

    void alloc_compute_ctx();

    void free_compute_ctx();

    void rebuild_params_tensor_set();

    ggml_tensor* canonical_param_tensor(ggml_tensor* tensor);

    std::vector<ggml_tensor*> collect_used_param_tensors(ggml_cgraph* gf);

    void evict_compute_backend_param_tensors(const std::vector<ggml_tensor*>& tensors);

    void prepare_build_in_tensor_before();

    void prepare_build_in_tensor_after(ggml_cgraph* gf);

    ggml_cgraph* new_graph_custom(size_t graph_size);

    ggml_cgraph* get_compute_graph(get_graph_cb_t get_graph);

    bool prepare_compute_graph(get_graph_cb_t get_graph,
                               ggml_cgraph** gf_out);

    ggml_backend_t backend_for_weight(const ggml_tensor* tensor) const;

    // Weightless ops have no scheduler anchor, so pin them to the most recent
    // weight device. Views must stay unpinned or cross-device copies can be
    // skipped for their consumers.
    void pin_multi_device_nodes(ggml_backend_sched_t sched, ggml_cgraph* gf, ggml_cgraph* original_graph = nullptr);

    bool is_multi_device() const {
        return !extra_runtime_backends.empty();
    }

    size_t reusable_compute_buffer_bytes() const {
        return workspace_.bytes(runtime_backend);
    }

    size_t retained_runtime_buffer_bytes(ggml_backend_t backend = nullptr) const;

    void sync_runtime_residency();

    std::optional<sd::Tensor<float>> read_graph_tensor(ggml_tensor* tensor, const char* label);

    void copy_data_to_backend_tensor(ggml_cgraph* gf, bool clear_after_copy = true);

    bool resolve_graph_cut_plan(ggml_cgraph* gf,
                                GraphCutPlan* plan_out);

    bool resolve_graph_cut_layer_split_plan(ggml_cgraph* gf,
                                            GraphCutPlan* plan_out);

    bool assign_graph_cut_layer_split_backends(ggml_cgraph* gf);

public:
    bool runner_start();

    bool runner_started() const { return runner_started_; }

    void runner_end();

public:
    virtual std::string get_desc() = 0;

    GGMLRunner(ggml_backend_t backend,
               std::shared_ptr<DeviceResidencyManager> manager = nullptr);

    virtual ~GGMLRunner();

    virtual GGMLRunnerContext get_context();

    void reset_compute_ctx();

public:
    void free_cache_ctx_and_buffer();

    // do copy after alloc graph
    void set_backend_tensor_data(ggml_tensor* tensor, const void* data);

    template <typename T>
    ggml_tensor* make_input(const sd::Tensor<T>& tensor) {
        ggml_tensor* input = sd::make_ggml_tensor(compute_ctx, tensor, false);
        set_backend_tensor_data(input, tensor.data());
        return input;
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>& tensor) {
        if (tensor.empty()) {
            return nullptr;
        }
        return make_input(tensor);
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>* tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return make_input(*tensor);
    }

    ggml_tensor* to_backend(ggml_tensor* tensor);

    void cache(const std::string name, ggml_tensor* tensor);

    ggml_tensor* get_cache_tensor_by_name(const std::string& name) {
        return cache_.get(name);
    }

    std::optional<sd::Tensor<float>> compute(get_graph_cb_t get_graph,
                                             int n_threads,
                                             bool auto_runner_end                      = true,
                                             bool no_return                            = false,
                                             const std::function<bool()>& read_outputs = {});

    void set_flash_attention_enabled(bool enabled) {
        flash_attn_enabled = enabled;
    }

    void set_conv2d_direct_enabled(bool enabled) {
        conv2d_direct_enabled = enabled;
    }

    void set_circular_axes(bool circular_x, bool circular_y) {
        circular_x_enabled = circular_x;
        circular_y_enabled = circular_y;
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        weight_adapter = adapter;
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) {
        max_graph_vram_bytes = max_vram_bytes;
    }

    void set_graph_cut_layer_split_enabled(bool enabled);

    void set_graph_cut_layer_split_backend_vram_limits(const std::vector<size_t>& limits);

    void set_runtime_backends(const std::vector<ggml_backend_t>& backends);
};

#endif  // __SD_CORE_GGML_RUNNER_H__
