#ifndef __MODEL_MANAGER_H__
#define __MODEL_MANAGER_H__

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "model_loader.h"
#include "weight_manager.h"

class ModelManager : public RunnerWeightManager {
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
    struct TensorState {
        std::string name;
        ggml_tensor* tensor = nullptr;
        std::string desc;

        ResidencyMode residency_mode   = ResidencyMode::ParamBackend;
        ggml_backend_t compute_backend = nullptr;
        ggml_backend_t params_backend  = nullptr;
        bool metadata_validated        = false;

        int active_prepare_count = 0;

        bool loaded_to_params_backend  = false;
        bool staged_to_compute_backend = false;
        uint64_t applied_lora_epoch    = UINT64_MAX;
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

    ModelLoader model_loader_;
    std::vector<std::unique_ptr<TensorState>> tensor_states_;
    std::map<std::string, TensorState*> tensor_states_by_name_;
    std::vector<std::unique_ptr<ParamsStorageBlock>> params_storage_blocks_;
    std::vector<std::unique_ptr<ComputeStagingBlock>> compute_staging_blocks_;
    std::set<std::string> common_ignore_tensors_;
    std::vector<LoraSpec> loras_;
    SDVersion lora_version_      = VERSION_COUNT;
    uint64_t current_lora_epoch_ = 0;
    int n_threads_               = 0;
    bool enable_mmap_            = false;

    void finish_compute_backend_usage(const std::vector<TensorState*>& states);
    void release_all();

    bool resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                        std::vector<TensorState*>& required_states) const;
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

    ggml_backend_buffer_type_t params_buffer_type_for(const TensorState& state) const;
    void release_compute_staging_blocks(bool force                                            = false,
                                        const std::unordered_set<TensorState*>* target_states = nullptr);
    void release_params_storage_blocks(bool force                                            = false,
                                       const std::unordered_set<TensorState*>* target_states = nullptr);
    void free_compute_staging_block(ComputeStagingBlock& block);
    void free_params_storage_block(ParamsStorageBlock& block);
    void erase_params_storage_block(ParamsStorageBlock* block);
    void reset_lora_applied_params();

public:
    ~ModelManager() override;

    ModelLoader& loader() { return model_loader_; }
    const ModelLoader& loader() const { return model_loader_; }

    void set_n_threads(int n_threads) {
        n_threads_ = n_threads;
        model_loader_.set_n_threads(n_threads);
    }
    void set_enable_mmap(bool enable_mmap) { enable_mmap_ = enable_mmap; }
    void set_common_ignore_tensors(std::set<std::string> ignore_tensors);
    void set_loras(std::vector<LoraSpec> loras, SDVersion version);

    std::set<std::string> tensor_names() const;

    bool register_param_tensors(const std::string& desc,
                                std::map<std::string, ggml_tensor*> tensors,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
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

    bool prepare_params(const std::vector<ggml_tensor*>& tensors) override;
    void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    void release_params_backend_params(const std::vector<ggml_tensor*>& tensors) override;
};

#endif  // __MODEL_MANAGER_H__
