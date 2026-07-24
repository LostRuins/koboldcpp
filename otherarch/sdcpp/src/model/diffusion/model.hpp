#ifndef __SD_MODEL_DIFFUSION_MODEL_HPP__
#define __SD_MODEL_DIFFUSION_MODEL_HPP__

#include <string>
#include <utility>
#include <variant>

#include "core/ggml_extend.hpp"
#include "core/tensor_ggml.hpp"
#include "model/common/rope.hpp"
#include "model_manager.h"

enum class RefImageResizeMode {
    NONE,
    LONGEST_SIDE,
    AREA,
};

struct RefImageParams {
    bool pass_to_vlm                   = false;
    bool pass_to_dit                   = true;
    Rope::RefIndexMode ref_index_mode  = Rope::RefIndexMode::FIXED;
    bool force_ref_timestep_zero       = false;
    bool resize_before_vae             = true;
    int vae_input_max_pixels           = -1;
    RefImageResizeMode vlm_resize_mode = RefImageResizeMode::AREA;
    int vlm_min_size                   = -1;
    int vlm_max_size                   = -1;
    bool resize_vae_to_target          = false;
};

const std::unordered_map<std::string, RefImageParams> REF_IMAGE_PRESETS = {
    {"flux_kontext", {false, true, Rope::RefIndexMode::FIXED, false, true, -1, RefImageResizeMode::NONE, -1, -1}},
    {"longcat", {true, true, Rope::RefIndexMode::FIXED, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"flux2", {false, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::NONE, -1, -1}},
    {"qwen", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"qwen_layered", {true, true, Rope::RefIndexMode::DECREASE, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"mage_flow", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::LONGEST_SIDE, -1, 384, true}},
    {"z_image_omni", {true, true, Rope::RefIndexMode::FIXED, false, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"krea2_ostris_edit", {true, true, Rope::RefIndexMode::INCREASE, true, true, -1, RefImageResizeMode::AREA, -1, -1}},
    {"krea2_edit", {true, true, Rope::RefIndexMode::INCREASE, false, true, -1, RefImageResizeMode::LONGEST_SIDE, 768, 768}},
    {"cosmos_reference", {false, true, Rope::RefIndexMode::INCREASE, false, false, -1, RefImageResizeMode::NONE, -1, -1}},
};

struct UNetDiffusionExtra {
    int num_video_frames                           = -1;
    const std::vector<sd::Tensor<float>>* controls = nullptr;
    float control_strength                         = 0.f;
    const sd::Tensor<float>* ip_context            = nullptr;
    float ip_scale                                 = 1.f;
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

struct HunyuanVideoDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const sd::Tensor<float>* byt5       = nullptr;
    const sd::Tensor<float>* vision     = nullptr;
    const sd::Tensor<float>* timestep_r = nullptr;
};

using DiffusionExtraParams = std::variant<std::monostate,
                                          UNetDiffusionExtra,
                                          SkipLayerDiffusionExtra,
                                          FluxDiffusionExtra,
                                          AnimaDiffusionExtra,
                                          WanDiffusionExtra,
                                          HiDreamO1DiffusionExtra,
                                          LTXAVDiffusionExtra,
                                          MiniT2IDiffusionExtra,
                                          HunyuanVideoDiffusionExtra>;

struct DiffusionParams {
    const sd::Tensor<float>* x                        = nullptr;
    const sd::Tensor<float>* timesteps                = nullptr;
    const sd::Tensor<float>* context                  = nullptr;
    const sd::Tensor<float>* c_concat                 = nullptr;
    const sd::Tensor<float>* y                        = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents = nullptr;
    RefImageParams ref_image_params                   = {false, false, Rope::RefIndexMode::FIXED, false};
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
