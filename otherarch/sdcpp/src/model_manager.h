#ifndef __MODEL_MANAGER_H__
#define __MODEL_MANAGER_H__

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "device_residency_manager.h"
#include "model_loader.h"

class ModelManager : public DeviceResidencyManager {
public:
    enum class ResidencyMode {
        Disk,
        ParamBackend,
    };

    struct LoraSpec {
        std::string path;
        float multiplier   = 1.0f;
        bool is_high_noise = false;
        std::string tensor_name_prefix_filter;
        bool required = false;
    };

private:
    static constexpr size_t MAX_RESIDENCY_BLOCK_BYTES = 64ULL * 1024ULL * 1024ULL;

    struct TensorState {
        std::string name;
        ggml_tensor* tensor = nullptr;
        std::string desc;

        ResidencyMode residency_mode                 = ResidencyMode::ParamBackend;
        ggml_backend_t compute_backend               = nullptr;
        ggml_backend_t params_backend                = nullptr;
        ggml_backend_buffer_type_t split_buffer_type = nullptr;
        bool params_follow_compute_backend           = false;
        bool metadata_validated                      = false;
        enum ggml_op usage_op                        = GGML_OP_NONE;

        int pin_count = 0;

        bool loaded_to_params_backend  = false;
        bool staged_to_compute_backend = false;
        uint64_t applied_lora_epoch    = UINT64_MAX;
        uint64_t last_use_epoch        = 0;
    };

    struct ParamsStorageBlock {
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<MmapTensorStore> mmap_tensor_stores;
        std::vector<TensorState*> states;
    };

    struct ComputeStagingBlock {
        ggml_backend_t compute_backend = nullptr;
        ggml_backend_buffer_t buffer   = nullptr;
        ggml_context* staging_ctx      = nullptr;
        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
    };

    struct PrefetchBlock {
        std::vector<TensorState*> states;
        ggml_backend_t compute_backend  = nullptr;
        ggml_backend_t transfer_backend = nullptr;
        ggml_backend_event_t event      = nullptr;
        std::vector<std::unique_ptr<ComputeStagingBlock>> staging_blocks;
    };

    struct RuntimeResidency {
        ggml_backend_t compute_backend = nullptr;
        size_t resident_bytes          = 0;
    };

    ModelLoader model_loader_;
    std::vector<std::unique_ptr<TensorState>> tensor_states_;
    std::map<std::string, TensorState*> tensor_states_by_name_;
    std::vector<std::unique_ptr<ParamsStorageBlock>> params_storage_blocks_;
    std::vector<std::unique_ptr<ComputeStagingBlock>> compute_staging_blocks_;
    std::map<ggml_backend_t, ggml_backend_buffer_type_t> split_buffer_types_;
    std::map<ggml_backend_buffer_type_t, std::vector<std::pair<ggml_backend_t, size_t>>> split_buffer_devices_;
    std::map<uintptr_t, std::unique_ptr<PrefetchBlock>> prefetch_blocks_;
    std::map<ggml_backend_t, ggml_backend_t> prefetch_backends_;
    std::map<std::pair<uintptr_t, ggml_backend_t>, RuntimeResidency> runtime_residencies_;
    std::map<uintptr_t, std::function<bool()>> workspace_reclaimers_;
    bool warned_split_lora_skip_ = false;
    std::set<std::string> common_ignore_tensors_;
    std::vector<LoraSpec> loras_;
    SDVersion lora_version_          = VERSION_COUNT;
    uint64_t current_lora_epoch_     = 0;
    uint64_t residency_epoch_        = 0;
    int n_threads_                   = 0;
    bool enable_mmap_                = false;
    bool writable_mmap_              = false;
    bool segmented_compute_disabled_ = false;
    bool prefetch_disabled_          = false;

    void finish_compute_backend_usage(const std::vector<TensorState*>& states);
    void release_all();

    ggml_backend_t prefetch_backend_for(ggml_backend_t compute_backend);
    bool populate_prefetch_block(PrefetchBlock& block);
    void synchronize_prefetch_block(PrefetchBlock& block);
    void free_prefetch_block(PrefetchBlock& block);
    void clear_all_prefetched_params();
    void release_prefetch();

    bool resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                        std::vector<TensorState*>& required_states,
                                        ggml_backend_t compute_backend = nullptr) const;
    bool should_ignore(const TensorState& state) const;
    bool is_optional_missing_tensor(const std::string& name) const;
    bool validate_tensor(const TensorState& state) const;

    bool load_tensors_to_params_backend(const std::vector<TensorState*>& states);
    bool apply_loras_to_params(const std::vector<TensorState*>& states);
    bool mmap_params(const std::vector<TensorState*>& states,
                     std::vector<ParamsStorageBlock*>& created_storage_blocks);
    bool can_mmap_storage(const TensorState& state) const;
    bool alloc_params_buffers(const std::vector<TensorState*>& states,
                              std::vector<ParamsStorageBlock*>& created_storage_blocks);
    bool load_tensors(const std::vector<TensorState*>& states);
    bool stage_tensors_to_compute_backend(const std::vector<TensorState*>& states);
    size_t compute_backend_alloc_size(const std::vector<TensorState*>& states,
                                      bool missing_only) const;
    size_t compute_backend_resident_bytes(ggml_backend_t compute_backend) const;
    struct CapacityCheck {
        size_t required_device_bytes  = 0;
        size_t required_budget_bytes  = 0;
        size_t available_device_bytes = SIZE_MAX;
        size_t available_budget_bytes = SIZE_MAX;
        bool fits() const {
            return required_device_bytes <= available_device_bytes &&
                   required_budget_bytes <= available_budget_bytes;
        }
    };
    CapacityCheck check_capacity(const DeviceMemoryRequest& request,
                                 const std::vector<TensorState*>& states) const;

    ggml_backend_buffer_type_t params_buffer_type_for(const TensorState& state) const;
    ggml_backend_buffer_type_t split_buffer_type_for(const TensorState& state) const;
    void release_compute_staging_blocks(bool force                                            = false,
                                        const std::unordered_set<TensorState*>* target_states = nullptr);
    void release_params_storage_blocks(bool force                                            = false,
                                       const std::unordered_set<TensorState*>* target_states = nullptr);
    void free_compute_staging_block(ComputeStagingBlock& block);
    void free_params_storage_block(ParamsStorageBlock& block);
    void erase_params_storage_block(ParamsStorageBlock* block);
    void reset_lora_applied_params();
    size_t other_runtime_resident_bytes(uintptr_t owner_id,
                                        ggml_backend_t compute_backend) const;

public:
    ~ModelManager() override;

    ModelLoader& loader() { return model_loader_; }
    const ModelLoader& loader() const { return model_loader_; }

    void set_n_threads(int n_threads) {
        n_threads_ = n_threads;
        model_loader_.set_n_threads(n_threads);
    }
    void set_segmented_compute_disabled(bool disabled) {
        segmented_compute_disabled_ = disabled;
    }
    void set_prefetch_disabled(bool disabled) { prefetch_disabled_ = disabled; }
    void set_enable_mmap(bool enable_mmap) { enable_mmap_ = enable_mmap; }
    void set_writable_mmap(bool writable_mmap) { writable_mmap_ = writable_mmap; }
    void set_common_ignore_tensors(std::set<std::string> ignore_tensors);
    void set_loras(std::vector<LoraSpec> loras, SDVersion version);
    void set_split_buffer_type(ggml_backend_t compute_backend, ggml_backend_buffer_type_t split_buft, const std::vector<std::pair<ggml_backend_t, size_t>>& device_limits);

    static bool tensor_shape_supports_split_buffer(const ggml_tensor* tensor);

    std::set<std::string> tensor_names() const;

    bool register_param_tensors(const std::string& desc,
                                std::map<std::string, ggml_tensor*> tensors,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size                         = nullptr,
                                bool allow_split_buffer                                = false,
                                bool params_follow_compute_backend                     = false,
                                const std::map<ggml_tensor*, enum ggml_op>* tensor_ops = nullptr);

    bool unregister_param_tensors(const std::string& desc,
                                  size_t* registered_tensor_size = nullptr);

    template <typename Runner>
    bool register_runner_params(const std::string& desc,
                                Runner& runner,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors);
        return register_param_tensors(desc,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    template <typename Runner>
    bool register_runner_params(const std::string& desc,
                                Runner& runner,
                                const std::string& prefix,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors, prefix);
        return register_param_tensors(desc,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    bool validate_registered_tensors();
    bool load_all_params_eagerly();

    bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                ggml_backend_t compute_backend) override;
    bool prepare_params(const std::vector<ggml_tensor*>& tensors) override;
    void set_workspace_reclaimer(uintptr_t owner_id, std::function<bool()> reclaim) override;
    void remove_runtime_owner(uintptr_t owner_id) override;
    bool fits_compute_backend_capacity(const DeviceMemoryRequest& request,
                                       const std::vector<ggml_tensor*>& required_params) const override;
    bool segmented_compute_enabled() const override { return !segmented_compute_disabled_; }
    bool prefetch_enabled() const override { return !prefetch_disabled_; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    void evict_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    WeightResidencyInfo inspect_compute_backend_params(
        const std::vector<ggml_tensor*>& tensors) const override;
    void update_runtime_residency(uintptr_t owner_id,
                                  ggml_backend_t compute_backend,
                                  size_t resident_bytes) override;
    bool ensure_compute_backend_capacity(
        const DeviceMemoryRequest& request,
        const std::vector<ggml_tensor*>& required_params,
        const std::vector<std::vector<ggml_tensor*>>& preferred_eviction_order,
        const std::vector<ggml_tensor*>& protected_params) override;
    WeightPrefetchResult prefetch_params(
        uintptr_t owner_id,
        const std::vector<ggml_tensor*>& tensors) override;
    bool activate_prefetched_params(uintptr_t owner_id,
                                    const std::vector<ggml_tensor*>& tensors) override;
    void clear_prefetched_params(uintptr_t owner_id) override;
};

#endif  // __MODEL_MANAGER_H__
