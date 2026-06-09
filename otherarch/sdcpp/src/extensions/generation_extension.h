#ifndef __SD_EXTENSIONS_GENERATION_EXTENSION_H__
#define __SD_EXTENSIONS_GENERATION_EXTENSION_H__

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "conditioning/conditioner.hpp"
#include "core/ggml_extend_backend.h"
#include "model_loader.h"
#include "stable-diffusion.h"

struct GenerationExtensionInitContext {
    const sd_ctx_params_t* params;
    SDVersion version;
    const String2TensorStorage& tensor_storage_map;
    ModelLoader& model_loader;
    int n_threads;
    std::function<bool(SDBackendModule)> ensure_backend_pair;
    std::function<ggml_backend_t(SDBackendModule)> backend_for;
    std::function<ggml_backend_t(SDBackendModule)> params_backend_for;
};

struct GenerationExtensionTensorContext {
    std::map<std::string, ggml_tensor*>& tensors;
    std::map<std::string, ggml_tensor*>& mmap_able_tensors;
    std::function<bool(SDBackendModule)> module_can_mmap;
};

struct GenerationExtensionConditionContext {
    Conditioner* conditioner;
    ConditionerParams& condition_params;
    const sd_pm_params_t& pm_params;
    std::map<std::string, ggml_tensor*>& tensors;
    SDVersion version;
    int n_threads;
    int total_steps;
    bool free_params_immediately;
};

struct GenerationExtension {
    virtual ~GenerationExtension() = default;

    virtual const char* name() const = 0;
    virtual bool is_enabled() const {
        return false;
    }
    virtual bool init(const GenerationExtensionInitContext&) {
        return true;
    }
    virtual void collect_param_tensors(GenerationExtensionTensorContext&) {}
    virtual void add_ignore_tensors(std::set<std::string>&) const {}
    virtual bool alloc_params_buffer() {
        return true;
    }
    virtual size_t get_params_buffer_size() const {
        return 0;
    }
    virtual void reset_runtime_condition() {}
    virtual bool prepare_condition(GenerationExtensionConditionContext&) {
        return false;
    }
    virtual const SDCondition& before_condition(int step,
                                                const SDCondition& condition) const {
        return condition;
    }
};

std::shared_ptr<GenerationExtension> create_photomaker_extension();

#endif
