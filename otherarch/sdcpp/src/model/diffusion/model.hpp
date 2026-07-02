#ifndef __SD_MODEL_DIFFUSION_MODEL_HPP__
#define __SD_MODEL_DIFFUSION_MODEL_HPP__

#include <string>
#include <utility>
#include <variant>

#include "core/ggml_extend.hpp"
#include "core/tensor_ggml.hpp"
#include "model/common/rope.hpp"
#include "model_manager.h"

struct UNetDiffusionExtra {
    int num_video_frames                           = -1;
    const std::vector<sd::Tensor<float>>* controls = nullptr;
    float control_strength                         = 0.f;
};

struct SkipLayerDiffusionExtra {
    const std::vector<int>* skip_layers = nullptr;
};

struct FluxDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const std::vector<int>* skip_layers = nullptr;
    const sd::Tensor<float>* pulid_id   = nullptr;
    float pulid_id_weight               = 1.0f;
};

struct AnimaDiffusionExtra {
    const sd::Tensor<int32_t>* t5_ids   = nullptr;
    const sd::Tensor<float>* t5_weights = nullptr;
};

struct WanDiffusionExtra {
    const sd::Tensor<float>* vace_context = nullptr;
    float vace_strength                   = 1.f;
};

struct HiDreamO1DiffusionExtra {
    const sd::Tensor<int32_t>* input_ids                               = nullptr;
    const sd::Tensor<int32_t>* input_pos                               = nullptr;
    const sd::Tensor<int32_t>* token_types                             = nullptr;
    const sd::Tensor<int32_t>* vinput_mask                             = nullptr;
    const std::vector<std::pair<int, sd::Tensor<float>>>* image_embeds = nullptr;
};

struct LTXAVDiffusionExtra {
    const sd::Tensor<float>* audio_x         = nullptr;
    const sd::Tensor<float>* audio_timesteps = nullptr;
    int audio_length                         = 0;
    float frame_rate                         = 24.f;
    const sd::Tensor<float>* video_positions = nullptr;
};

struct MiniT2IDiffusionExtra {
    const sd::Tensor<float>* mask = nullptr;
};

using DiffusionExtraParams = std::variant<std::monostate,
                                          UNetDiffusionExtra,
                                          SkipLayerDiffusionExtra,
                                          FluxDiffusionExtra,
                                          AnimaDiffusionExtra,
                                          WanDiffusionExtra,
                                          HiDreamO1DiffusionExtra,
                                          LTXAVDiffusionExtra,
                                          MiniT2IDiffusionExtra>;

struct DiffusionParams {
    const sd::Tensor<float>* x                        = nullptr;
    const sd::Tensor<float>* timesteps                = nullptr;
    const sd::Tensor<float>* context                  = nullptr;
    const sd::Tensor<float>* c_concat                 = nullptr;
    const sd::Tensor<float>* y                        = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents = nullptr;
    Rope::RefIndexMode ref_index_mode                 = Rope::RefIndexMode::FIXED;
    DiffusionExtraParams extra                        = std::monostate{};
};

template <typename T>
static inline const T* diffusion_extra_as(const DiffusionParams& params) {
    const auto* extra = std::get_if<T>(&params.extra);
    GGML_ASSERT(extra != nullptr);
    return extra;
}

template <typename T>
static inline const sd::Tensor<T>& tensor_or_empty(const sd::Tensor<T>* tensor) {
    static const sd::Tensor<T> kEmpty;
    return tensor != nullptr ? *tensor : kEmpty;
}

struct DiffusionModelRunner : public GGMLRunner {
protected:
    std::string prefix;

public:
    DiffusionModelRunner(ggml_backend_t backend,
                         const std::string& prefix,
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          prefix(prefix) {}

    virtual sd::Tensor<float> compute(int n_threads,
                                      const DiffusionParams& diffusion_params) = 0;

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        get_param_tensors(tensors, prefix);
    }

    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                   const std::string& prefix) = 0;
};

#endif  // __SD_MODEL_DIFFUSION_MODEL_HPP__
