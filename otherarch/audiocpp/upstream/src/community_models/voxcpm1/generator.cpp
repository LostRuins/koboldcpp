#include "engine/community_models/voxcpm1/generator.h"

#include "minicpm_blocks.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/community_models/voxcpm1/assets.h"
#include "engine/community_models/voxcpm1/minicpm.h"
#include "engine/community_models/voxcpm1/tokenizer_text.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::voxcpm1 {
namespace {

namespace binding = engine::modules::binding;

using Clock = std::chrono::steady_clock;

constexpr int32_t kRefAudioStartToken = 103;
constexpr int32_t kRefAudioEndToken = 104;
constexpr int64_t kStreamingPrefixLen = 4;

struct PrefillRow {
  int32_t token = 0;
  std::vector<float> feature;
  std::vector<float> embedding;
  bool text_mask = false;
  bool audio_mask = false;
};

struct PrefillSequence {
  std::vector<PrefillRow> rows;
  int64_t target_text_tokens = 0;
};

std::shared_ptr<const VoxCPM1Assets>
require_assets(std::shared_ptr<const VoxCPM1Assets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("VoxCPM1 feature generator requires assets");
  }
  return assets;
}

void validate_generation_options(const VoxCPM1GenerationOptions &options) {
  if (options.min_tokens < 0) {
    throw std::runtime_error("VoxCPM1 min_tokens must be non-negative");
  }
  if (options.max_tokens < 0) {
    throw std::runtime_error("VoxCPM1 max_tokens must be non-negative");
  }
  if (options.num_inference_steps <= 0) {
    throw std::runtime_error(
        "VoxCPM1 num_inference_steps must be positive");
  }
  if (!std::isfinite(options.guidance_scale)) {
    throw std::runtime_error("VoxCPM1 guidance_scale must be finite");
  }
  if (options.retry_badcase_max_times <= 0) {
    throw std::runtime_error(
        "VoxCPM1 retry_badcase_max_times must be positive");
  }
  if (!std::isfinite(options.retry_badcase_ratio_threshold) ||
      options.retry_badcase_ratio_threshold <= 0.0F) {
    throw std::runtime_error(
        "VoxCPM1 retry_badcase_ratio_threshold must be positive and finite");
  }
}

int64_t effective_max_tokens(const VoxCPM1GenerationOptions &options,
                             int64_t target_text_tokens) {
  const auto ratio_bound =
      static_cast<int64_t>(static_cast<float>(target_text_tokens) *
                               options.retry_badcase_ratio_threshold +
                           10.0F);
  return std::min<int64_t>(ratio_bound, options.max_tokens);
}

int stop_class(const std::vector<float> &logits) {
  if (logits.size() != 2) {
    throw std::runtime_error("VoxCPM1 stop logits must have two classes");
  }
  return logits[1] > logits[0] ? 1 : 0;
}

std::vector<float> add_dit_mu(const std::vector<float> &lm,
                              const std::vector<float> &residual) {
  if (lm.size() != residual.size()) {
    throw std::runtime_error("VoxCPM1 dit mu inputs must have equal size");
  }
  std::vector<float> out(lm.size(), 0.0F);
  for (size_t i = 0; i < lm.size(); ++i) {
    out[i] = lm[i] + residual[i];
  }
  return out;
}

void append_patch(std::vector<float> &features, const std::vector<float> &patch,
                  int64_t expected_size) {
  if (static_cast<int64_t>(patch.size()) != expected_size) {
    throw std::runtime_error("VoxCPM1 generated patch size mismatch");
  }
  features.insert(features.end(), patch.begin(), patch.end());
}

void validate_feature_block(const std::vector<float> &features, int64_t patches,
                            int64_t patch_elems, const char *label) {
  if (patches < 0) {
    throw std::runtime_error(std::string("VoxCPM1 ") + label +
                             " patch count is negative");
  }
  if (static_cast<int64_t>(features.size()) != patches * patch_elems) {
    throw std::runtime_error(std::string("VoxCPM1 ") + label +
                             " feature size mismatch");
  }
}

std::vector<float> feature_patch(const std::vector<float> &features,
                                 int64_t index, int64_t patch_elems) {
  const auto begin =
      features.begin() + static_cast<std::ptrdiff_t>(index * patch_elems);
  return std::vector<float>(begin,
                            begin + static_cast<std::ptrdiff_t>(patch_elems));
}

bool has_prompt_audio(const VoxCPM1EncodedPrompt *prompt) {
  return prompt != nullptr && prompt->prompt_patches > 0;
}

bool has_reference_audio(const VoxCPM1EncodedPrompt *prompt) {
  return prompt != nullptr && prompt->reference_patches > 0;
}

std::string normalize_wrapper_text(const std::string &text) {
  std::string out;
  out.reserve(text.size());
  bool in_space = false;
  for (const unsigned char ch : text) {
    if (std::isspace(ch)) {
      if (!in_space) {
        out.push_back(' ');
        in_space = true;
      }
      continue;
    }
    out.push_back(static_cast<char>(ch));
    in_space = false;
  }
  return out;
}

} // namespace

struct VoxCPM1StepProjectionOutput {
  std::vector<float> fsq_hidden;
  std::vector<float> current_residual_input;
  std::vector<float> residual_input;
  std::vector<float> current_lm_dit_hidden;
  std::vector<float> fsq_lm_dit_hidden;
  std::vector<float> residual_dit_hidden;
  std::vector<float> current_stop_logits;
  std::vector<float> fsq_stop_logits;
};

class VoxCPM1StepProjectionRuntime final {
public:
  VoxCPM1StepProjectionRuntime(
      std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
      size_t graph_context_bytes, bool mem_saver = false);
  ~VoxCPM1StepProjectionRuntime();

  VoxCPM1StepProjectionOutput run(const std::vector<float> &lm_hidden,
                                  const std::vector<float> &residual_hidden,
                                  const std::vector<float> &current_embed);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM1LocalEncoderRuntime final {
public:
  VoxCPM1LocalEncoderRuntime(
      std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
      size_t graph_context_bytes, bool mem_saver = false);
  ~VoxCPM1LocalEncoderRuntime();

  std::vector<float>
  encode_patch(const std::vector<float> &patch_features) const;
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM1DiTEstimatorRuntime final {
public:
  VoxCPM1DiTEstimatorRuntime(
      std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
      size_t graph_context_bytes, bool mem_saver = false);
  ~VoxCPM1DiTEstimatorRuntime();

  std::vector<float> run(const std::vector<float> &x,
                         const std::vector<float> &mu,
                         const std::vector<float> &cond,
                         const std::vector<float> &time_embedding,
                         const std::vector<float> &delta_time_embedding);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM1CFMRuntime final {
public:
  VoxCPM1CFMRuntime(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
                    size_t estimator_graph_context_bytes,
                    bool mem_saver = false);
  ~VoxCPM1CFMRuntime();

  std::vector<float> generate_patch(const std::vector<float> &mu,
                                    const std::vector<float> &cond_patch,
                                    int64_t timesteps, float cfg_value,
                                    uint64_t seed,
                                    uint64_t noise_start_index = 0,
                                    const std::string &noise_file = {},
                                    float temperature = 1.0F);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM1StepProjectionRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
       size_t graph_context_bytes, bool mem_saver)
      : weights_(std::move(weights)), mem_saver_(mem_saver) {
    if (weights_ == nullptr) {
      throw std::runtime_error(
          "VoxCPM1 step projection runtime requires weights");
    }
    build(graph_context_bytes);
  }

  ~Impl() { release_graph(); }

  void release_runtime_memory() { release_graph(); }

  VoxCPM1StepProjectionOutput run(const std::vector<float> &lm_hidden,
                                  const std::vector<float> &residual_hidden,
                                  const std::vector<float> &current_embed) {
    const auto &config = weights_->assets().config;
    if (static_cast<int64_t>(lm_hidden.size()) != config.lm.hidden_size) {
      throw std::runtime_error(
          "VoxCPM1 step projection lm_hidden size mismatch");
    }
    if (static_cast<int64_t>(residual_hidden.size()) != config.lm.hidden_size) {
      throw std::runtime_error(
          "VoxCPM1 step projection residual_hidden size mismatch");
    }
    if (static_cast<int64_t>(current_embed.size()) != config.lm.hidden_size) {
      throw std::runtime_error(
          "VoxCPM1 step projection current_embed size mismatch");
    }
    ggml_backend_tensor_set(lm_hidden_, lm_hidden.data(), 0,
                            lm_hidden.size() * sizeof(float));
    ggml_backend_tensor_set(residual_hidden_, residual_hidden.data(), 0,
                            residual_hidden.size() * sizeof(float));
    ggml_backend_tensor_set(current_embed_, current_embed.data(), 0,
                            current_embed.size() * sizeof(float));
    engine::core::set_backend_threads(weights_->backend(), weights_->threads());
    const ggml_status status =
        engine::core::compute_backend_graph(weights_->backend(), graph_);
    ggml_backend_synchronize(weights_->backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM1 step projection graph compute failed");
    }
    VoxCPM1StepProjectionOutput output;
    output.fsq_hidden.resize(static_cast<size_t>(config.lm.hidden_size), 0.0F);
    output.current_residual_input.resize(
        static_cast<size_t>(config.lm.hidden_size), 0.0F);
    output.residual_input.resize(static_cast<size_t>(config.lm.hidden_size),
                                 0.0F);
    output.current_lm_dit_hidden.resize(
        static_cast<size_t>(config.dit.hidden_dim), 0.0F);
    output.fsq_lm_dit_hidden.resize(static_cast<size_t>(config.dit.hidden_dim),
                                    0.0F);
    output.residual_dit_hidden.resize(
        static_cast<size_t>(config.dit.hidden_dim), 0.0F);
    output.current_stop_logits.resize(2, 0.0F);
    output.fsq_stop_logits.resize(2, 0.0F);
    ggml_backend_tensor_get(fsq_hidden_output_, output.fsq_hidden.data(), 0,
                            output.fsq_hidden.size() * sizeof(float));
    ggml_backend_tensor_get(
        current_residual_input_output_, output.current_residual_input.data(), 0,
        output.current_residual_input.size() * sizeof(float));
    ggml_backend_tensor_get(residual_input_output_,
                            output.residual_input.data(), 0,
                            output.residual_input.size() * sizeof(float));
    ggml_backend_tensor_get(
        current_lm_dit_output_, output.current_lm_dit_hidden.data(), 0,
        output.current_lm_dit_hidden.size() * sizeof(float));
    ggml_backend_tensor_get(fsq_lm_dit_output_, output.fsq_lm_dit_hidden.data(),
                            0, output.fsq_lm_dit_hidden.size() * sizeof(float));
    ggml_backend_tensor_get(residual_dit_output_,
                            output.residual_dit_hidden.data(), 0,
                            output.residual_dit_hidden.size() * sizeof(float));
    ggml_backend_tensor_get(current_stop_logits_output_,
                            output.current_stop_logits.data(), 0,
                            output.current_stop_logits.size() * sizeof(float));
    ggml_backend_tensor_get(fsq_stop_logits_output_,
                            output.fsq_stop_logits.data(), 0,
                            output.fsq_stop_logits.size() * sizeof(float));
    return output;
  }

private:
  void release_graph() {
    if (graph_ != nullptr) {
      engine::core::release_backend_graph_resources(weights_->backend(), graph_);
    }
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
      buffer_ = nullptr;
    }
    if (gallocr_ != nullptr) {
      ggml_gallocr_free(gallocr_);
      gallocr_ = nullptr;
    }
    graph_ = nullptr;
    lm_hidden_ = nullptr;
    residual_hidden_ = nullptr;
    current_embed_ = nullptr;
    fsq_hidden_output_ = nullptr;
    current_residual_input_output_ = nullptr;
    residual_input_output_ = nullptr;
    current_lm_dit_output_ = nullptr;
    fsq_lm_dit_output_ = nullptr;
    residual_dit_output_ = nullptr;
    current_stop_logits_output_ = nullptr;
    fsq_stop_logits_output_ = nullptr;
    ctx_.reset();
  }

  void build(size_t graph_context_bytes) {
    const auto &config = weights_->assets().config;
    if (graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM1 step projection graph context bytes must be non-zero");
    }
    ggml_init_params params{graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 step projection graph context");
    }
    engine::core::ModuleBuildContext ctx{ctx_.get(), "voxcpm1.step_projection"};
    const auto &proj = weights_->weights().projections;
    auto lm_hidden = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, config.lm.hidden_size}));
    lm_hidden_ = lm_hidden.tensor;
    if (mem_saver_) {
      ggml_set_input(lm_hidden_);
    }
    auto residual_hidden = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, config.lm.hidden_size}));
    residual_hidden_ = residual_hidden.tensor;
    if (mem_saver_) {
      ggml_set_input(residual_hidden_);
    }
    auto current_embed = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, config.lm.hidden_size}));
    current_embed_ = current_embed.tensor;
    if (mem_saver_) {
      ggml_set_input(current_embed_);
    }

    auto fsq =
        engine::modules::LinearModule(
            binding::linear_config(config.lm.hidden_size,
                                   config.scalar_quantization_latent_dim, true))
            .build(ctx, lm_hidden, proj.fsq_in_proj);
    fsq = engine::core::wrap_tensor(ggml_tanh(ctx.ggml, fsq.tensor), fsq.shape,
                                    GGML_TYPE_F32);
    fsq = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, fsq.tensor,
                   static_cast<float>(config.scalar_quantization_scale)),
        fsq.shape, GGML_TYPE_F32);
    fsq = engine::core::wrap_tensor(ggml_round(ctx.ggml, fsq.tensor), fsq.shape,
                                    GGML_TYPE_F32);
    fsq = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, fsq.tensor,
                   1.0F / static_cast<float>(config.scalar_quantization_scale)),
        fsq.shape, GGML_TYPE_F32);
    fsq = engine::modules::LinearModule(
              binding::linear_config(config.scalar_quantization_latent_dim,
                                     config.lm.hidden_size, true))
              .build(ctx, fsq, proj.fsq_out_proj);
    fsq_hidden_output_ = fsq.tensor;

    // V1 path: simple ADD (no fusion_concat_proj projection).
    current_residual_input_output_ =
        engine::modules::AddModule()
            .build(ctx, lm_hidden, current_embed)
            .tensor;
    residual_input_output_ =
        engine::modules::AddModule().build(ctx, fsq, current_embed).tensor;

    auto current_lm_dit =
        engine::modules::LinearModule(
            binding::linear_config(config.lm.hidden_size, config.dit.hidden_dim,
                                   true))
            .build(ctx, lm_hidden, proj.lm_to_dit_proj);
    current_lm_dit_output_ = current_lm_dit.tensor;

    auto fsq_lm_dit = engine::modules::LinearModule(
                          binding::linear_config(config.lm.hidden_size,
                                                 config.dit.hidden_dim, true))
                          .build(ctx, fsq, proj.lm_to_dit_proj);
    fsq_lm_dit_output_ = fsq_lm_dit.tensor;

    auto residual_dit = engine::modules::LinearModule(
                            binding::linear_config(config.lm.hidden_size,
                                                   config.dit.hidden_dim, true))
                            .build(ctx, residual_hidden, proj.res_to_dit_proj);
    residual_dit_output_ = residual_dit.tensor;

    auto current_stop = engine::modules::LinearModule(
                            binding::linear_config(config.lm.hidden_size,
                                                   config.lm.hidden_size, true))
                            .build(ctx, lm_hidden, proj.stop_proj);
    current_stop = engine::modules::SiluModule{}.build(ctx, current_stop);
    current_stop = engine::modules::LinearModule(
                       binding::linear_config(config.lm.hidden_size, 2, false))
                       .build(ctx, current_stop, proj.stop_head);
    current_stop_logits_output_ = current_stop.tensor;

    auto fsq_stop = engine::modules::LinearModule(
                        binding::linear_config(config.lm.hidden_size,
                                               config.lm.hidden_size, true))
                        .build(ctx, fsq, proj.stop_proj);
    fsq_stop = engine::modules::SiluModule{}.build(ctx, fsq_stop);
    fsq_stop = engine::modules::LinearModule(
                   binding::linear_config(config.lm.hidden_size, 2, false))
                   .build(ctx, fsq_stop, proj.stop_head);
    fsq_stop_logits_output_ = fsq_stop.tensor;

    graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
    ggml_set_output(fsq_hidden_output_);
    if (mem_saver_ && fsq_hidden_output_->view_src != nullptr) {
      ggml_set_output(fsq_hidden_output_->view_src);
    }
    ggml_set_output(current_residual_input_output_);
    if (mem_saver_ && current_residual_input_output_->view_src != nullptr) {
      ggml_set_output(current_residual_input_output_->view_src);
    }
    ggml_set_output(residual_input_output_);
    if (mem_saver_ && residual_input_output_->view_src != nullptr) {
      ggml_set_output(residual_input_output_->view_src);
    }
    ggml_set_output(current_lm_dit_output_);
    if (mem_saver_ && current_lm_dit_output_->view_src != nullptr) {
      ggml_set_output(current_lm_dit_output_->view_src);
    }
    ggml_set_output(fsq_lm_dit_output_);
    if (mem_saver_ && fsq_lm_dit_output_->view_src != nullptr) {
      ggml_set_output(fsq_lm_dit_output_->view_src);
    }
    ggml_set_output(residual_dit_output_);
    if (mem_saver_ && residual_dit_output_->view_src != nullptr) {
      ggml_set_output(residual_dit_output_->view_src);
    }
    ggml_set_output(current_stop_logits_output_);
    if (mem_saver_ && current_stop_logits_output_->view_src != nullptr) {
      ggml_set_output(current_stop_logits_output_->view_src);
    }
    ggml_set_output(fsq_stop_logits_output_);
    if (mem_saver_ && fsq_stop_logits_output_->view_src != nullptr) {
      ggml_set_output(fsq_stop_logits_output_->view_src);
    }
    ggml_build_forward_expand(graph_, fsq_hidden_output_);
    ggml_build_forward_expand(graph_, current_residual_input_output_);
    ggml_build_forward_expand(graph_, residual_input_output_);
    ggml_build_forward_expand(graph_, current_lm_dit_output_);
    ggml_build_forward_expand(graph_, fsq_lm_dit_output_);
    ggml_build_forward_expand(graph_, residual_dit_output_);
    ggml_build_forward_expand(graph_, current_stop_logits_output_);
    ggml_build_forward_expand(graph_, fsq_stop_logits_output_);
    if (mem_saver_) {
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(weights_->backend()));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        if (gallocr_ != nullptr) {
          ggml_gallocr_free(gallocr_);
          gallocr_ = nullptr;
        }
        throw std::runtime_error(
            "failed to allocate VoxCPM1 step projection graph");
      }
      return;
    }
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
    if (buffer_ == nullptr) {
      throw std::runtime_error(
          "failed to allocate VoxCPM1 step projection graph");
    }
  }

  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  bool mem_saver_ = false;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_tensor *lm_hidden_ = nullptr;
  ggml_tensor *residual_hidden_ = nullptr;
  ggml_tensor *current_embed_ = nullptr;
  ggml_tensor *fsq_hidden_output_ = nullptr;
  ggml_tensor *current_residual_input_output_ = nullptr;
  ggml_tensor *residual_input_output_ = nullptr;
  ggml_tensor *current_lm_dit_output_ = nullptr;
  ggml_tensor *fsq_lm_dit_output_ = nullptr;
  ggml_tensor *residual_dit_output_ = nullptr;
  ggml_tensor *current_stop_logits_output_ = nullptr;
  ggml_tensor *fsq_stop_logits_output_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_gallocr_t gallocr_ = nullptr;
};

VoxCPM1StepProjectionRuntime::VoxCPM1StepProjectionRuntime(
    std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
    size_t graph_context_bytes, bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes,
                                   mem_saver)) {}

VoxCPM1StepProjectionRuntime::~VoxCPM1StepProjectionRuntime() = default;

void VoxCPM1StepProjectionRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

VoxCPM1StepProjectionOutput
VoxCPM1StepProjectionRuntime::run(const std::vector<float> &lm_hidden,
                                  const std::vector<float> &residual_hidden,
                                  const std::vector<float> &current_embed) {
  return impl_->run(lm_hidden, residual_hidden, current_embed);
}

class VoxCPM1LocalEncoderRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
       size_t graph_context_bytes, bool mem_saver)
      : weights_(std::move(weights)), mem_saver_(mem_saver) {
    if (weights_ == nullptr) {
      throw std::runtime_error(
          "VoxCPM1 local encoder runtime requires weights");
    }
    build(graph_context_bytes);
  }

  ~Impl() { release_graph(); }

  void release_runtime_memory() { release_graph(); }

  std::vector<float>
  encode_patch(const std::vector<float> &patch_features) const {
    const auto &config = weights_->assets().config;
    const int64_t expected = config.patch_size * config.feat_dim;
    if (static_cast<int64_t>(patch_features.size()) != expected) {
      throw std::runtime_error(
          "VoxCPM1 local encoder patch feature size mismatch");
    }
    ggml_backend_tensor_set(input_, patch_features.data(), 0,
                            patch_features.size() * sizeof(float));
    engine::core::set_backend_threads(weights_->backend(), weights_->threads());
    const ggml_status status =
        engine::core::compute_backend_graph(weights_->backend(), graph_);
    ggml_backend_synchronize(weights_->backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM1 local encoder graph compute failed");
    }
    std::vector<float> output(static_cast<size_t>(config.lm.hidden_size), 0.0F);
    ggml_backend_tensor_get(output_, output.data(), 0,
                            output.size() * sizeof(float));
    return output;
  }

private:
  void release_graph() {
    if (graph_ != nullptr) {
      engine::core::release_backend_graph_resources(weights_->backend(), graph_);
    }
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
      buffer_ = nullptr;
    }
    if (gallocr_ != nullptr) {
      ggml_gallocr_free(gallocr_);
      gallocr_ = nullptr;
    }
    graph_ = nullptr;
    input_ = nullptr;
    positions_ = nullptr;
    output_ = nullptr;
    ctx_.reset();
  }

  void build(size_t graph_context_bytes) {
    const auto &root_config = weights_->assets().config;
    if (graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM1 local encoder graph context bytes must be non-zero");
    }
    ggml_init_params params{graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 local encoder graph context");
    }
    engine::core::ModuleBuildContext ctx{ctx_.get(), "voxcpm1.local_encoder"};
    const auto &feat_weights = weights_->weights().feat_encoder;
    auto x = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims(
            {1, root_config.patch_size, root_config.feat_dim}));
    input_ = x.tensor;
    if (mem_saver_) {
      ggml_set_input(input_);
    }
    x = engine::modules::LinearModule(
            binding::linear_config(root_config.feat_dim,
                                   root_config.encoder.hidden_dim, true))
            .build(ctx, x, feat_weights.in_proj);
    auto special = engine::core::reshape_tensor(
        ctx, feat_weights.special_token,
        engine::core::TensorShape::from_dims(
            {1, 1, root_config.encoder.hidden_dim}));
    special = engine::core::wrap_tensor(
        ggml_cast(ctx.ggml, special.tensor, GGML_TYPE_F32), special.shape,
        GGML_TYPE_F32);
    x = engine::modules::ConcatModule({1}).build(ctx, special, x);
    positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32,
                                    root_config.patch_size + 1);
    if (mem_saver_) {
      ggml_set_input(positions_);
      ggml_set_output(positions_);
    }
    auto positions = engine::core::wrap_tensor(
        positions_,
        engine::core::TensorShape::from_dims({root_config.patch_size + 1}),
        GGML_TYPE_I32);
    x = minicpm_transformer(ctx, x, positions, feat_weights.encoder, false);
    x = engine::modules::SliceModule({1, 0, 1}).build(ctx, x);
    x = engine::modules::LinearModule(
            binding::linear_config(root_config.encoder.hidden_dim,
                                   root_config.lm.hidden_size, true))
            .build(ctx, x, weights_->weights().projections.enc_to_lm_proj);
    output_ = x.tensor;
    ggml_set_output(output_);
    if (mem_saver_ && output_->view_src != nullptr) {
      ggml_set_output(output_->view_src);
    }
    graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
    ggml_build_forward_expand(graph_, output_);
    if (mem_saver_) {
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(weights_->backend()));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        if (gallocr_ != nullptr) {
          ggml_gallocr_free(gallocr_);
          gallocr_ = nullptr;
        }
        throw std::runtime_error(
            "failed to allocate VoxCPM1 local encoder graph");
      }
    } else {
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
    }
    if (!mem_saver_ && buffer_ == nullptr) {
      throw std::runtime_error(
          "failed to allocate VoxCPM1 local encoder graph");
    }
    std::vector<int32_t> position_ids(
        static_cast<size_t>(root_config.patch_size + 1), 0);
    for (int64_t i = 0; i < root_config.patch_size + 1; ++i) {
      position_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(positions_, position_ids.data(), 0,
                            position_ids.size() * sizeof(int32_t));
  }

  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  bool mem_saver_ = false;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_tensor *input_ = nullptr;
  ggml_tensor *positions_ = nullptr;
  ggml_tensor *output_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_gallocr_t gallocr_ = nullptr;
};

VoxCPM1LocalEncoderRuntime::VoxCPM1LocalEncoderRuntime(
    std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
    size_t graph_context_bytes, bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes,
                                   mem_saver)) {}

VoxCPM1LocalEncoderRuntime::~VoxCPM1LocalEncoderRuntime() = default;

void VoxCPM1LocalEncoderRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

std::vector<float> VoxCPM1LocalEncoderRuntime::encode_patch(
    const std::vector<float> &patch_features) const {
  return impl_->encode_patch(patch_features);
}

class VoxCPM1DiTEstimatorRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
       size_t graph_context_bytes, bool mem_saver)
      : weights_(std::move(weights)), mem_saver_(mem_saver) {
    if (weights_ == nullptr) {
      throw std::runtime_error(
          "VoxCPM1 DiT estimator runtime requires weights");
    }
    build(graph_context_bytes);
  }

  ~Impl() { release_graph(); }

  void release_runtime_memory() { release_graph(); }

  std::vector<float> run(const std::vector<float> &x,
                         const std::vector<float> &mu,
                         const std::vector<float> &cond,
                         const std::vector<float> &time_embedding,
                         const std::vector<float> &delta_time_embedding) {
    const auto &config = weights_->assets().config;
    const int64_t patch_elems = 2 * config.feat_dim * config.patch_size;
    if (static_cast<int64_t>(x.size()) != patch_elems) {
      throw std::runtime_error("VoxCPM1 DiT estimator x size mismatch");
    }
    if (static_cast<int64_t>(cond.size()) != patch_elems) {
      throw std::runtime_error("VoxCPM1 DiT estimator cond size mismatch");
    }
    const int64_t expected_mu = 2 * config.dit.hidden_dim;
    if (static_cast<int64_t>(mu.size()) != expected_mu) {
      throw std::runtime_error("VoxCPM1 DiT estimator mu size mismatch");
    }
    if (static_cast<int64_t>(time_embedding.size()) !=
        2 * config.dit.hidden_dim) {
      throw std::runtime_error(
          "VoxCPM1 DiT estimator time embedding size mismatch");
    }
    if (static_cast<int64_t>(delta_time_embedding.size()) !=
        2 * config.dit.hidden_dim) {
      throw std::runtime_error(
          "VoxCPM1 DiT estimator delta-time embedding size mismatch");
    }
    ggml_backend_tensor_set(x_, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(cond_, cond.data(), 0, cond.size() * sizeof(float));
    ggml_backend_tensor_set(mu_, mu.data(), 0, mu.size() * sizeof(float));
    ggml_backend_tensor_set(time_embedding_, time_embedding.data(), 0,
                            time_embedding.size() * sizeof(float));
    ggml_backend_tensor_set(delta_time_embedding_, delta_time_embedding.data(),
                            0, delta_time_embedding.size() * sizeof(float));
    engine::core::set_backend_threads(weights_->backend(), weights_->threads());
    const ggml_status status =
        engine::core::compute_backend_graph(weights_->backend(), graph_);
    ggml_backend_synchronize(weights_->backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM1 DiT estimator graph compute failed");
    }
    std::vector<float> output(static_cast<size_t>(patch_elems), 0.0F);
    ggml_backend_tensor_get(output_, output.data(), 0,
                            output.size() * sizeof(float));
    return output;
  }

private:
  void release_graph() {
    if (graph_ != nullptr) {
      engine::core::release_backend_graph_resources(weights_->backend(), graph_);
    }
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
      buffer_ = nullptr;
    }
    if (gallocr_ != nullptr) {
      ggml_gallocr_free(gallocr_);
      gallocr_ = nullptr;
    }
    graph_ = nullptr;
    x_ = nullptr;
    mu_ = nullptr;
    cond_ = nullptr;
    time_embedding_ = nullptr;
    delta_time_embedding_ = nullptr;
    positions_ = nullptr;
    output_ = nullptr;
    ctx_.reset();
  }

  void build(size_t graph_context_bytes) {
    const auto &root_config = weights_->assets().config;
    const auto &config = root_config.dit;
    if (graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM1 DiT estimator graph context bytes must be non-zero");
    }
    ggml_init_params params{graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 DiT estimator graph context");
    }
    engine::core::ModuleBuildContext ctx{ctx_.get(), "voxcpm1.dit.estimator"};
    const auto &weights = weights_->weights().dit;
    x_ = engine::core::make_tensor(
             ctx, GGML_TYPE_F32,
             engine::core::TensorShape::from_dims(
                 {2, root_config.feat_dim, root_config.patch_size}))
             .tensor;
    if (mem_saver_) {
      ggml_set_input(x_);
    }
    cond_ = engine::core::make_tensor(
                ctx, GGML_TYPE_F32,
                engine::core::TensorShape::from_dims(
                    {2, root_config.feat_dim, root_config.patch_size}))
                .tensor;
    if (mem_saver_) {
      ggml_set_input(cond_);
    }
    mu_ = engine::core::make_tensor(
              ctx, GGML_TYPE_F32,
              engine::core::TensorShape::from_dims(
                  {2, config.hidden_dim}))
              .tensor;
    if (mem_saver_) {
      ggml_set_input(mu_);
    }
    time_embedding_ =
        engine::core::make_tensor(
            ctx, GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({2, config.hidden_dim}))
            .tensor;
    if (mem_saver_) {
      ggml_set_input(time_embedding_);
    }
    delta_time_embedding_ =
        engine::core::make_tensor(
            ctx, GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({2, config.hidden_dim}))
            .tensor;
    if (mem_saver_) {
      ggml_set_input(delta_time_embedding_);
    }

    auto x = engine::core::wrap_tensor(
        x_,
        engine::core::TensorShape::from_dims(
            {2, root_config.feat_dim, root_config.patch_size}),
        GGML_TYPE_F32);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = engine::modules::LinearModule(
            binding::linear_config(root_config.feat_dim, config.hidden_dim,
                                   true))
            .build(ctx, x, weights.in_proj);

    auto cond = engine::core::wrap_tensor(
        cond_,
        engine::core::TensorShape::from_dims(
            {2, root_config.feat_dim, root_config.patch_size}),
        GGML_TYPE_F32);
    cond = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, cond);
    cond = engine::modules::LinearModule(
               binding::linear_config(root_config.feat_dim, config.hidden_dim,
                                      true))
               .build(ctx, cond, weights.cond_proj);

    auto time = engine::core::wrap_tensor(
        time_embedding_,
        engine::core::TensorShape::from_dims({2, config.hidden_dim}),
        GGML_TYPE_F32);
    time =
        engine::modules::LinearModule(
            binding::linear_config(config.hidden_dim, config.hidden_dim, true))
            .build(ctx, time, weights.time_mlp_1);
    time = engine::modules::SiluModule{}.build(ctx, time);
    time =
        engine::modules::LinearModule(
            binding::linear_config(config.hidden_dim, config.hidden_dim, true))
            .build(ctx, time, weights.time_mlp_2);

    auto dt = engine::core::wrap_tensor(
        delta_time_embedding_,
        engine::core::TensorShape::from_dims({2, config.hidden_dim}),
        GGML_TYPE_F32);
    dt = engine::modules::LinearModule(
             binding::linear_config(config.hidden_dim, config.hidden_dim, true))
             .build(ctx, dt, weights.delta_time_mlp_1);
    dt = engine::modules::SiluModule{}.build(ctx, dt);
    dt = engine::modules::LinearModule(
             binding::linear_config(config.hidden_dim, config.hidden_dim, true))
             .build(ctx, dt, weights.delta_time_mlp_2);
    time = engine::modules::AddModule{}.build(ctx, time, dt);

    // V1 path: the DiT conditioning mu is a single hidden vector that is
    // ADDED into the timestep token. Batch 0 carries mu (conditioned branch);
    // batch 1 carries zeros (unconditioned branch), mirroring
    // LocDiTModel::forward_cfg_pair_projected with mu_tokens == 1.
    const int64_t prefix_token_count = 1;
    auto hidden = time;
    {
      auto mu = engine::core::wrap_tensor(
          mu_, engine::core::TensorShape::from_dims({2, config.hidden_dim}),
          GGML_TYPE_F32);
      hidden = engine::modules::AddModule{}.build(ctx, hidden, mu);
    }
    hidden = engine::core::reshape_tensor(
        ctx, hidden,
        engine::core::TensorShape::from_dims({2, 1, config.hidden_dim}));
    hidden = engine::modules::ConcatModule({1}).build(ctx, hidden, cond);
    hidden = engine::modules::ConcatModule({1}).build(ctx, hidden, x);

    positions_ = ggml_new_tensor_1d(
        ctx_.get(), GGML_TYPE_I32,
        prefix_token_count + root_config.patch_size * 2);
    if (mem_saver_) {
      ggml_set_input(positions_);
      ggml_set_output(positions_);
    }
    auto positions =
        engine::core::wrap_tensor(positions_,
                                  engine::core::TensorShape::from_dims(
                                      {prefix_token_count +
                                       root_config.patch_size * 2}),
                                  GGML_TYPE_I32);
    hidden =
        minicpm_transformer(ctx, hidden, positions, weights.decoder, false);
    hidden = engine::modules::SliceModule(
                 {1, prefix_token_count + root_config.patch_size,
                  root_config.patch_size})
                 .build(ctx, hidden);
    hidden = engine::modules::LinearModule(
                 binding::linear_config(config.hidden_dim, root_config.feat_dim,
                                        true))
                 .build(ctx, hidden, weights.out_proj);
    hidden =
        engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = ensure_contiguous(ctx, hidden);
    output_ = hidden.tensor;
    ggml_set_output(output_);
    if (mem_saver_ && output_->view_src != nullptr) {
      ggml_set_output(output_->view_src);
    }
    graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
    ggml_build_forward_expand(graph_, output_);
    if (mem_saver_) {
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(weights_->backend()));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        if (gallocr_ != nullptr) {
          ggml_gallocr_free(gallocr_);
          gallocr_ = nullptr;
        }
        throw std::runtime_error(
            "failed to allocate VoxCPM1 DiT estimator graph");
      }
    } else {
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
    }
    if (!mem_saver_ && buffer_ == nullptr) {
      throw std::runtime_error(
          "failed to allocate VoxCPM1 DiT estimator graph");
    }
    std::vector<int32_t> positions_data(
        static_cast<size_t>(prefix_token_count + root_config.patch_size * 2),
        0);
    for (int64_t i = 0; i < static_cast<int64_t>(positions_data.size()); ++i) {
      positions_data[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(positions_, positions_data.data(), 0,
                            positions_data.size() * sizeof(int32_t));
  }

  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  bool mem_saver_ = false;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_tensor *x_ = nullptr;
  ggml_tensor *mu_ = nullptr;
  ggml_tensor *cond_ = nullptr;
  ggml_tensor *time_embedding_ = nullptr;
  ggml_tensor *delta_time_embedding_ = nullptr;
  ggml_tensor *positions_ = nullptr;
  ggml_tensor *output_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_gallocr_t gallocr_ = nullptr;
};

VoxCPM1DiTEstimatorRuntime::VoxCPM1DiTEstimatorRuntime(
    std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
    size_t graph_context_bytes, bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes,
                                   mem_saver)) {}

VoxCPM1DiTEstimatorRuntime::~VoxCPM1DiTEstimatorRuntime() = default;

void VoxCPM1DiTEstimatorRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

std::vector<float> VoxCPM1DiTEstimatorRuntime::run(
    const std::vector<float> &x, const std::vector<float> &mu,
    const std::vector<float> &cond, const std::vector<float> &time_embedding,
    const std::vector<float> &delta_time_embedding) {
  return impl_->run(x, mu, cond, time_embedding, delta_time_embedding);
}


std::vector<float> sinusoidal_time_embedding(float timestep,
                                             int64_t hidden_size) {
  if (hidden_size <= 0 || hidden_size % 2 != 0) {
    throw std::runtime_error(
        "VoxCPM1 sinusoidal time embedding requires even hidden size");
  }
  const int64_t half = hidden_size / 2;
  std::vector<float> out(static_cast<size_t>(hidden_size), 0.0F);
  const double emb_scale = std::log(10000.0) / static_cast<double>(half - 1);
  for (int64_t index = 0; index < half; ++index) {
    const double freq = std::exp(static_cast<double>(index) * -emb_scale);
    const double arg = 1000.0 * static_cast<double>(timestep) * freq;
    out[static_cast<size_t>(index)] = static_cast<float>(std::sin(arg));
    out[static_cast<size_t>(half + index)] = static_cast<float>(std::cos(arg));
  }
  return out;
}

class VoxCPM1CFMRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
       size_t estimator_graph_context_bytes, bool mem_saver)
      : weights_(std::move(weights)),
        estimator_(weights_, estimator_graph_context_bytes, mem_saver) {
    if (weights_ == nullptr) {
      throw std::runtime_error("VoxCPM1 CFM runtime requires weights");
    }
  }

  void release_runtime_memory() { estimator_.release_runtime_memory(); }

  std::vector<float> generate_patch(const std::vector<float> &mu,
                                    const std::vector<float> &cond_patch,
                                    int64_t timesteps, float cfg_value,
                                    uint64_t seed, uint64_t noise_start_index,
                                    const std::string &noise_file,
                                    float temperature) {
    const auto &config = weights_->assets().config;
    if (timesteps <= 0) {
      throw std::runtime_error("VoxCPM1 CFM requires positive timesteps");
    }
    if (!std::isfinite(cfg_value) || !std::isfinite(temperature)) {
      throw std::runtime_error("VoxCPM1 CFM received non-finite scalar input");
    }
    const int64_t patch_elems = config.feat_dim * config.patch_size;
    const int64_t mu_dim = config.dit.hidden_dim;
    if (static_cast<int64_t>(mu.size()) != mu_dim) {
      throw std::runtime_error("VoxCPM1 CFM mu size mismatch");
    }
    if (static_cast<int64_t>(cond_patch.size()) != patch_elems) {
      throw std::runtime_error("VoxCPM1 CFM conditioning patch size mismatch");
    }

    std::vector<float> x;
    if (noise_file.empty()) {
      x = engine::sampling::generate_torch_cuda_randn(
          static_cast<size_t>(patch_elems), seed,
          engine::sampling::TorchRandnPrecision::Float32, noise_start_index);
    } else {
      if (noise_file_ != noise_file) {
        noise_values_ = engine::io::read_f32_file(noise_file);
        noise_file_ = noise_file;
      }
      const auto start = static_cast<size_t>(noise_start_index);
      const auto count = static_cast<size_t>(patch_elems);
      if (noise_values_.size() < start + count) {
        throw std::runtime_error(
            "VoxCPM1 CFM noise file is too short: expected at least " +
            std::to_string(start + count) + " floats, got " +
            std::to_string(noise_values_.size()));
      }
      x.assign(noise_values_.begin() + static_cast<std::ptrdiff_t>(start),
               noise_values_.begin() + static_cast<std::ptrdiff_t>(start + count));
    }
    for (float &value : x) {
      value *= temperature;
    }
    x = patch_major_to_channel_major(x);
    const std::vector<float> cond = patch_major_to_channel_major(cond_patch);
    std::vector<float> x_in(static_cast<size_t>(2 * patch_elems), 0.0F);
    std::vector<float> cond_in(static_cast<size_t>(2 * patch_elems), 0.0F);
    const int64_t mu_elements = 2 * config.dit.hidden_dim;
    std::vector<float> mu_in(static_cast<size_t>(mu_elements), 0.0F);
    std::copy(mu.begin(), mu.end(), mu_in.begin());
    std::copy(cond.begin(), cond.end(), cond_in.begin());
    std::copy(cond.begin(), cond.end(),
              cond_in.begin() + static_cast<std::ptrdiff_t>(patch_elems));

    std::vector<float> t_span(static_cast<size_t>(timesteps + 1), 0.0F);
    constexpr double kHalfPi = 1.57079632679489661923;
    for (int64_t i = 0; i <= timesteps; ++i) {
      const double base =
          1.0 - static_cast<double>(i) / static_cast<double>(timesteps);
      t_span[static_cast<size_t>(i)] =
          static_cast<float>(base + (std::cos(kHalfPi * base) - 1.0 + base));
    }

    float t = t_span.front();
    float dt = t_span[0] - t_span[1];
    const int64_t zero_init_steps =
        std::max<int64_t>(1, static_cast<int64_t>(t_span.size() * 0.04));
    for (int64_t step = 1; step < static_cast<int64_t>(t_span.size()); ++step) {
      std::vector<float> dphi(static_cast<size_t>(patch_elems), 0.0F);
      if (step > zero_init_steps) {
        std::copy(x.begin(), x.end(), x_in.begin());
        std::copy(x.begin(), x.end(),
                  x_in.begin() + static_cast<std::ptrdiff_t>(patch_elems));
        const auto time_one =
            sinusoidal_time_embedding(t, config.dit.hidden_dim);
        const float dt_value = config.dit.mean_mode ? dt : 0.0F;
        const auto dt_one =
            sinusoidal_time_embedding(dt_value, config.dit.hidden_dim);
        std::vector<float> time_embedding(
            static_cast<size_t>(2 * config.dit.hidden_dim), 0.0F);
        std::vector<float> delta_embedding(
            static_cast<size_t>(2 * config.dit.hidden_dim), 0.0F);
        std::copy(time_one.begin(), time_one.end(), time_embedding.begin());
        std::copy(time_one.begin(), time_one.end(),
                  time_embedding.begin() +
                      static_cast<std::ptrdiff_t>(config.dit.hidden_dim));
        std::copy(dt_one.begin(), dt_one.end(), delta_embedding.begin());
        std::copy(dt_one.begin(), dt_one.end(),
                  delta_embedding.begin() +
                      static_cast<std::ptrdiff_t>(config.dit.hidden_dim));

        const auto estimator = estimator_.run(x_in, mu_in, cond_in,
                                              time_embedding, delta_embedding);
        const float scale = optimized_cfg_scale(estimator, patch_elems);
        for (int64_t i = 0; i < patch_elems; ++i) {
          const size_t index = static_cast<size_t>(i);
          const float positive = estimator[index];
          const float negative =
              estimator[static_cast<size_t>(patch_elems + i)];
          dphi[index] =
              negative * scale + cfg_value * (positive - negative * scale);
        }
      }
      for (int64_t i = 0; i < patch_elems; ++i) {
        x[static_cast<size_t>(i)] -= dt * dphi[static_cast<size_t>(i)];
      }
      t -= dt;
      if (step < static_cast<int64_t>(t_span.size()) - 1) {
        dt = t - t_span[static_cast<size_t>(step + 1)];
      }
    }
    return channel_major_to_patch_major(x);
  }

private:
  std::vector<float>
  patch_major_to_channel_major(const std::vector<float> &patch) const {
    const auto &config = weights_->assets().config;
    std::vector<float> out(patch.size(), 0.0F);
    for (int64_t p = 0; p < config.patch_size; ++p) {
      for (int64_t d = 0; d < config.feat_dim; ++d) {
        out[static_cast<size_t>(d * config.patch_size + p)] =
            patch[static_cast<size_t>(p * config.feat_dim + d)];
      }
    }
    return out;
  }

  std::vector<float>
  channel_major_to_patch_major(const std::vector<float> &channel) const {
    const auto &config = weights_->assets().config;
    std::vector<float> out(channel.size(), 0.0F);
    for (int64_t p = 0; p < config.patch_size; ++p) {
      for (int64_t d = 0; d < config.feat_dim; ++d) {
        out[static_cast<size_t>(p * config.feat_dim + d)] =
            channel[static_cast<size_t>(d * config.patch_size + p)];
      }
    }
    return out;
  }

  float optimized_cfg_scale(const std::vector<float> &estimator,
                            int64_t patch_elems) const {
    double dot = 0.0;
    double norm = 1.0e-8;
    for (int64_t i = 0; i < patch_elems; ++i) {
      const double positive = estimator[static_cast<size_t>(i)];
      const double negative = estimator[static_cast<size_t>(patch_elems + i)];
      dot += positive * negative;
      norm += negative * negative;
    }
    return static_cast<float>(dot / norm);
  }

  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  VoxCPM1DiTEstimatorRuntime estimator_;
  std::string noise_file_;
  std::vector<float> noise_values_;
};

VoxCPM1CFMRuntime::VoxCPM1CFMRuntime(
    std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
    size_t estimator_graph_context_bytes,
    bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(weights),
                                   estimator_graph_context_bytes,
                                   mem_saver)) {}

VoxCPM1CFMRuntime::~VoxCPM1CFMRuntime() = default;

void VoxCPM1CFMRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

std::vector<float> VoxCPM1CFMRuntime::generate_patch(
    const std::vector<float> &mu, const std::vector<float> &cond_patch,
    int64_t timesteps, float cfg_value, uint64_t seed,
    uint64_t noise_start_index, const std::string &noise_file,
    float temperature) {
  return impl_->generate_patch(mu, cond_patch, timesteps, cfg_value, seed,
                               noise_start_index, noise_file, temperature);
}

class VoxCPM1FeatureGeneratorRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1Assets> assets,
       engine::core::ExecutionContext &execution_context,
       VoxCPM1FeatureGeneratorConfig config)
      : assets_(require_assets(std::move(assets))),
        weights_(std::make_shared<VoxCPM1WeightsRuntime>(
            assets_, execution_context, config.weight_context_bytes,
            config.weight_storage_type)),
        tokenizer_(assets_),
        text_embedding_(weights_, config.text_embedding_graph_context_bytes,
                        config.mem_saver),
        prefill_(weights_, config.lm_step_graph_context_bytes,
                 config.mem_saver),
        base_lm_(weights_, VoxCPM1MiniCPMKind::BaseLM,
                 assets_->config.max_length,
                 config.lm_step_graph_context_bytes),
        residual_lm_(weights_, VoxCPM1MiniCPMKind::ResidualLM,
                     assets_->config.max_length,
                     config.lm_step_graph_context_bytes),
        projection_(weights_, config.projection_graph_context_bytes,
                    config.mem_saver),
        cfm_(weights_, config.dit_graph_context_bytes, config.mem_saver),
        local_encoder_(weights_, config.local_encoder_graph_context_bytes,
                       config.mem_saver),
        prompt_audio_embedding_cache_(config.prompt_cache_slots) {}

  VoxCPM1Result generate_zero_shot(const std::string &text,
                                   const VoxCPM1GenerationOptions &options) {
    return generate(text, nullptr, options);
  }

  VoxCPM1Result generate(const std::string &text,
                         const VoxCPM1EncodedPrompt *prompt,
                         const VoxCPM1GenerationOptions &options) {
    validate_generation_options(options);
    const auto prefill = build_prefill_sequence(text, prompt);

    const int64_t max_tokens =
        effective_max_tokens(options, prefill.target_text_tokens);
    VoxCPM1Result last_result;
    uint64_t retry_noise_start = 0;
    for (int64_t attempt = 0; attempt < options.retry_badcase_max_times;
         ++attempt) {
      last_result =
          generate_once(prefill, max_tokens, options, retry_noise_start);
      retry_noise_start += static_cast<uint64_t>(last_result.generated_patches *
                                                 assets_->config.patch_size *
                                                 assets_->config.feat_dim);
      if (!options.retry_badcase ||
          static_cast<float>(last_result.generated_patches) <
              static_cast<float>(prefill.target_text_tokens) *
                  options.retry_badcase_ratio_threshold) {
        break;
      }
    }
    return last_result;
  }

  VoxCPM1StreamingResult
  generate_streaming(const std::string &text,
                     const VoxCPM1EncodedPrompt *prompt,
                     const VoxCPM1GenerationOptions &options,
                     const std::function<void(const VoxCPM1StreamingChunk &)>
                         &chunk_callback) {
    validate_generation_options(options);
    if (options.retry_badcase) {
      fprintf(stderr,
              "[VoxCPM1] warning: retry_badcase ignored in streaming "
              "generation\n");
    }
    const auto prefill = build_prefill_sequence(text, prompt);
    const int64_t max_tokens =
        effective_max_tokens(options, prefill.target_text_tokens);
    VoxCPM1StreamingResult streaming;
    auto *streaming_chunks =
        chunk_callback ? nullptr : &streaming.chunks;
    const auto result = generate_once(prefill, max_tokens, options, 0,
                                      streaming_chunks, chunk_callback);
    streaming.generated_patches = result.generated_patches;
    return streaming;
  }

  void release_runtime_memory() {
    // Release every staged graph so a session can idle at weight-only
    // VRAM. Each runtime lazily rebuilds its graph on the next use.
    text_embedding_.release_runtime_memory();
    prefill_.release_runtime_memory();
    base_lm_.release_runtime_memory();
    residual_lm_.release_runtime_memory();
    projection_.release_runtime_memory();
    cfm_.release_runtime_memory();
    local_encoder_.release_runtime_memory();
  }

  void release_text_length_memory() {
    // Only the prompt-prefill graph is sized by the request text/prompt
    // length; the other generator graphs have fixed-size workspaces. Drop it
    // after every request so a long-lived session does not retain buffers
    // that scale with text length; the next request rebuilds it fresh.
    prefill_.release_runtime_memory();
  }

private:
  struct PromptAudioEmbeddingCacheKey {
    std::vector<float> prompt_features;
    int64_t prompt_patches = 0;
    std::vector<float> reference_features;
    int64_t reference_patches = 0;
  };

  struct PromptAudioEmbeddingCacheKeyEqual {
    bool operator()(const PromptAudioEmbeddingCacheKey &lhs,
                    const PromptAudioEmbeddingCacheKey &rhs) const {
      return lhs.prompt_patches == rhs.prompt_patches &&
             lhs.reference_patches == rhs.reference_patches &&
             lhs.prompt_features == rhs.prompt_features &&
             lhs.reference_features == rhs.reference_features;
    }
  };

  struct PromptAudioEmbeddingCacheEntry {
    std::vector<float> prompt_embeddings;
    std::vector<float> reference_embeddings;
  };

  PrefillSequence build_prefill_sequence(const std::string &target_text,
                                         const VoxCPM1EncodedPrompt *prompt) {
    const auto &config = assets_->config;
    const int64_t patch_elems = config.patch_size * config.feat_dim;
    const std::string normalized_target_text =
        normalize_wrapper_text(target_text);
    if (prompt != nullptr) {
      validate_feature_block(prompt->prompt_features, prompt->prompt_patches,
                             patch_elems, "prompt");
      validate_feature_block(prompt->reference_features,
                             prompt->reference_patches, patch_elems,
                             "reference");
      if (prompt->prompt_patches > 0 && prompt->prompt_text.empty()) {
        throw std::runtime_error(
            "VoxCPM1 continuation prompt requires prompt text");
      }
    }

    const bool use_prompt = has_prompt_audio(prompt);
    const bool use_reference = has_reference_audio(prompt);
    const PromptAudioEmbeddingCacheEntry *embedding_cache =
        prompt != nullptr ? &cached_prompt_audio_embeddings(*prompt) : nullptr;
    const std::string combined_text =
        use_prompt ? prompt->prompt_text + normalized_target_text
                   : normalized_target_text;
    const VoxCPM1TextPrompt text_prompt =
        tokenizer_.build_prompt(combined_text);
    const VoxCPM1TextPrompt target_prompt =
        tokenizer_.build_prompt(normalized_target_text);
    std::vector<float> zero_patch(static_cast<size_t>(patch_elems), 0.0F);
    PrefillSequence sequence;
    sequence.target_text_tokens =
        static_cast<int64_t>(target_prompt.input_ids.size());

    auto append_text = [&](int32_t token) {
      PrefillRow row;
      row.token = token;
      row.feature = zero_patch;
      row.text_mask = true;
      sequence.rows.push_back(std::move(row));
    };
    auto append_audio = [&](const std::vector<float> &features,
                            const std::vector<float> &embeddings,
                            int64_t patch_index) {
      PrefillRow row;
      row.feature = feature_patch(features, patch_index, patch_elems);
      row.embedding = hidden_patch(embeddings, patch_index,
                                   config.lm.hidden_size);
      row.audio_mask = true;
      sequence.rows.push_back(std::move(row));
    };

    if (use_reference) {
      append_text(kRefAudioStartToken);
      for (int64_t i = 0; i < prompt->reference_patches; ++i) {
        append_audio(prompt->reference_features,
                     embedding_cache->reference_embeddings, i);
      }
      append_text(kRefAudioEndToken);
    }
    for (const int32_t token : text_prompt.input_ids) {
      append_text(token);
    }
    append_text(tokenizer_.audio_start_token_id());
    if (use_prompt) {
      for (int64_t i = 0; i < prompt->prompt_patches; ++i) {
        append_audio(prompt->prompt_features, embedding_cache->prompt_embeddings,
                     i);
      }
    }

    if (sequence.rows.empty() ||
        static_cast<int64_t>(sequence.rows.size()) >= config.max_length) {
      // Caller-controlled: the prompt audio/text decides how many rows this
      // is. Report the numbers so the remedy is arithmetic, not guesswork.
      throw engine::runtime::CapacityError(
          "VoxCPM1 prompt exceeds the model cache length ("
          + std::to_string(sequence.rows.size()) + " rows, limit "
          + std::to_string(config.max_length) + "); shorten the prompt");
    }
    return sequence;
  }

  std::vector<float> hidden_patch(const std::vector<float> &embeddings,
                                  int64_t index, int64_t hidden_size) const {
    if (index < 0 || hidden_size <= 0 ||
        static_cast<int64_t>(embeddings.size()) < (index + 1) * hidden_size) {
      throw std::runtime_error(
          "VoxCPM1 prompt audio embedding cache size mismatch");
    }
    const auto begin =
        embeddings.begin() + static_cast<std::ptrdiff_t>(index * hidden_size);
    return std::vector<float>(begin,
                              begin + static_cast<std::ptrdiff_t>(hidden_size));
  }

  std::vector<float>
  encode_feature_embeddings(const std::vector<float> &features, int64_t patches,
                            int64_t patch_elems) const {
    std::vector<float> embeddings;
    embeddings.reserve(static_cast<size_t>(patches *
                                           assets_->config.lm.hidden_size));
    for (int64_t i = 0; i < patches; ++i) {
      auto embedding =
          local_encoder_.encode_patch(feature_patch(features, i, patch_elems));
      embeddings.insert(embeddings.end(), embedding.begin(), embedding.end());
    }
    return embeddings;
  }

  const PromptAudioEmbeddingCacheEntry &
  cached_prompt_audio_embeddings(const VoxCPM1EncodedPrompt &prompt) {
    const int64_t patch_elems =
        assets_->config.patch_size * assets_->config.feat_dim;
    PromptAudioEmbeddingCacheKey key;
    key.prompt_features = prompt.prompt_features;
    key.prompt_patches = prompt.prompt_patches;
    key.reference_features = prompt.reference_features;
    key.reference_patches = prompt.reference_patches;
    if (auto *cached = prompt_audio_embedding_cache_.find(key)) {
      debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.hit", 1);
      debug::trace_log_scalar(
          "voxcpm1.prompt_audio_embedding_cache.slots",
          static_cast<int64_t>(prompt_audio_embedding_cache_.capacity()));
      debug::trace_log_scalar(
          "voxcpm1.prompt_audio_embedding_cache.entries",
          static_cast<int64_t>(prompt_audio_embedding_cache_.size()));
      debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.evicted",
                              0);
      debug::timing_log_scalar("voxcpm1.prompt_audio_embedding_ms", 0.0);
      return *cached;
    }

    const auto embedding_start = Clock::now();
    PromptAudioEmbeddingCacheEntry entry;
    entry.prompt_embeddings = encode_feature_embeddings(
        prompt.prompt_features, prompt.prompt_patches, patch_elems);
    entry.reference_embeddings = encode_feature_embeddings(
        prompt.reference_features, prompt.reference_patches, patch_elems);
    const double embedding_ms = engine::debug::elapsed_ms(embedding_start);
    if (prompt_audio_embedding_cache_.capacity() == 0) {
      uncached_prompt_audio_embedding_ = std::move(entry);
      debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.hit", 0);
      debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.slots", 0);
      debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.entries",
                              0);
      debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.evicted",
                              0);
      debug::timing_log_scalar("voxcpm1.prompt_audio_embedding_ms",
                               embedding_ms);
      return *uncached_prompt_audio_embedding_;
    }
    const bool will_evict = prompt_audio_embedding_cache_.size() >=
                            prompt_audio_embedding_cache_.capacity();
    prompt_audio_embedding_cache_.put(std::move(key), std::move(entry));
    PromptAudioEmbeddingCacheKey lookup;
    lookup.prompt_features = prompt.prompt_features;
    lookup.prompt_patches = prompt.prompt_patches;
    lookup.reference_features = prompt.reference_features;
    lookup.reference_patches = prompt.reference_patches;
    auto *cached = prompt_audio_embedding_cache_.find(lookup);
    if (cached == nullptr) {
      throw std::runtime_error(
          "VoxCPM1 prompt audio embedding cache insert failed");
    }
    debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.hit", 0);
    debug::trace_log_scalar(
        "voxcpm1.prompt_audio_embedding_cache.slots",
        static_cast<int64_t>(prompt_audio_embedding_cache_.capacity()));
    debug::trace_log_scalar(
        "voxcpm1.prompt_audio_embedding_cache.entries",
        static_cast<int64_t>(prompt_audio_embedding_cache_.size()));
    debug::trace_log_scalar("voxcpm1.prompt_audio_embedding_cache.evicted",
                            will_evict ? 1 : 0);
    debug::timing_log_scalar("voxcpm1.prompt_audio_embedding_ms",
                             embedding_ms);
    return *cached;
  }

  VoxCPM1Result
  generate_once(const PrefillSequence &prefill, int64_t max_tokens,
                const VoxCPM1GenerationOptions &options,
                uint64_t noise_start_index,
                std::vector<VoxCPM1StreamingChunk> *streaming_chunks =
                    nullptr,
                const std::function<void(const VoxCPM1StreamingChunk &)>
                    &streaming_chunk_callback = nullptr) {
    const auto &config = assets_->config;
    const int64_t hidden_size = config.lm.hidden_size;
    const int64_t patch_elems = config.patch_size * config.feat_dim;
    if (static_cast<int64_t>(prefill.rows.size()) + max_tokens >
        config.max_length) {
      // Same: prefill rows come from the input text, max_tokens from the
      // request. Both are the caller's to reduce.
      throw engine::runtime::CapacityError(
          "VoxCPM1 generation exceeds the model cache length ("
          + std::to_string(prefill.rows.size()) + " prefill rows + "
          + std::to_string(max_tokens) + " requested tokens, limit "
          + std::to_string(config.max_length) + "); shorten the input text");
    }
    base_lm_.reset();
    residual_lm_.reset();

    std::vector<float> zero_hidden(static_cast<size_t>(hidden_size), 0.0F);
    std::vector<float> zero_patch(static_cast<size_t>(patch_elems), 0.0F);
    VoxCPM1PromptPrefillInput prefill_input;
    prefill_input.steps = static_cast<int64_t>(prefill.rows.size());
    prefill_input.input_embeddings.reserve(
        static_cast<size_t>(prefill_input.steps * hidden_size));
    prefill_input.current_embeddings.reserve(
        static_cast<size_t>(prefill_input.steps * hidden_size));
    prefill_input.text_mask.reserve(static_cast<size_t>(prefill_input.steps));
    prefill_input.audio_mask.reserve(static_cast<size_t>(prefill_input.steps));
    std::vector<float> prefix_cond = zero_patch;
    for (const auto &row : prefill.rows) {
      if (row.text_mask == row.audio_mask) {
        throw std::runtime_error("VoxCPM1 prefill row mask is invalid");
      }
      std::vector<float> input_embedding;
      std::vector<float> current_embed = zero_hidden;
      if (row.text_mask) {
        input_embedding = text_embedding_.embed_token(row.token);
      } else {
        current_embed = row.embedding;
        if (static_cast<int64_t>(current_embed.size()) != hidden_size) {
          throw std::runtime_error(
              "VoxCPM1 prompt audio embedding size mismatch");
        }
        input_embedding = current_embed;
      }
      if (row.audio_mask) {
        prefix_cond = row.feature;
      }
      prefill_input.input_embeddings.insert(prefill_input.input_embeddings.end(),
                                            input_embedding.begin(),
                                            input_embedding.end());
      prefill_input.current_embeddings.insert(
          prefill_input.current_embeddings.end(), current_embed.begin(),
          current_embed.end());
      prefill_input.text_mask.push_back(row.text_mask ? 1.0F : 0.0F);
      prefill_input.audio_mask.push_back(row.audio_mask ? 1.0F : 0.0F);
    }
    const auto prefill_output = prefill_.run(prefill_input);
    base_lm_.import_state(prefill_output.base_state);
    residual_lm_.import_state(prefill_output.residual_state);
    std::vector<float> lm_hidden = prefill_output.lm_hidden;
    std::vector<float> residual_hidden = prefill_output.residual_hidden;
    // The prefill graph holds the largest sequence-shaped workspace; its
    // outputs have been copied to host hiddens and its KV state imported
    // into the step runtimes, so nothing below references it. Drop it now
    // so the token loop runs against the much smaller step graphs.
    prefill_.release_runtime_memory();

    VoxCPM1Result result;
    std::vector<const PrefillRow *> context_rows;
    for (auto it = prefill.rows.rbegin(); it != prefill.rows.rend(); ++it) {
      if (!it->audio_mask) {
        break;
      }
      if (static_cast<int64_t>(context_rows.size()) >=
          kStreamingPrefixLen - 1) {
        break;
      }
      context_rows.push_back(&*it);
    }
    result.decode_trim_patches = static_cast<int64_t>(context_rows.size());
    for (auto it = context_rows.rbegin(); it != context_rows.rend(); ++it) {
      append_patch(result.decode_features, (*it)->feature, patch_elems);
      ++result.decode_patches;
    }
    uint64_t patch_noise_start = noise_start_index;
    for (int64_t index = 0; index < max_tokens; ++index) {
      const auto projected =
          projection_.run(lm_hidden, residual_hidden, zero_hidden);
      const auto mu = add_dit_mu(projected.current_lm_dit_hidden,
                                 projected.residual_dit_hidden);
      const auto patch = cfm_.generate_patch(
          mu, prefix_cond, options.num_inference_steps, options.guidance_scale,
          options.seed, patch_noise_start, options.cfm_noise_file);
      patch_noise_start += static_cast<uint64_t>(patch_elems);
      append_patch(result.generated_features, patch, patch_elems);
      ++result.generated_patches;
      append_patch(result.decode_features, patch, patch_elems);
      ++result.decode_patches;
      if (streaming_chunks != nullptr || streaming_chunk_callback) {
        VoxCPM1StreamingChunk chunk;
        chunk.decode_features = patch;
        chunk.decode_patches = 1;
        chunk.generated_patches = result.generated_patches;
        if (streaming_chunk_callback) {
          streaming_chunk_callback(chunk);
        }
        if (streaming_chunks != nullptr) {
          streaming_chunks->push_back(std::move(chunk));
        }
      }
      prefix_cond = patch;

      if (index > options.min_tokens &&
          stop_class(projected.current_stop_logits) == 1) {
        break;
      }

      const auto curr_embed = local_encoder_.encode_patch(patch);
      const auto next_lm = base_lm_.run_step(curr_embed).hidden;
      const auto next_projected =
          projection_.run(next_lm, residual_hidden, curr_embed);
      lm_hidden = next_projected.fsq_hidden;
      residual_hidden =
          residual_lm_.run_step(next_projected.residual_input).hidden;
    }
    return result;
  }

  std::shared_ptr<const VoxCPM1Assets> assets_;
  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  VoxCPM1TextTokenizer tokenizer_;
  VoxCPM1TextEmbeddingRuntime text_embedding_;
  VoxCPM1PromptPrefillRuntime prefill_;
  VoxCPM1MiniCPMStepRuntime base_lm_;
  VoxCPM1MiniCPMStepRuntime residual_lm_;
  VoxCPM1StepProjectionRuntime projection_;
  VoxCPM1CFMRuntime cfm_;
  VoxCPM1LocalEncoderRuntime local_encoder_;
  engine::runtime::CacheSlots<PromptAudioEmbeddingCacheKey,
                              PromptAudioEmbeddingCacheEntry,
                              PromptAudioEmbeddingCacheKeyEqual>
      prompt_audio_embedding_cache_;
  std::optional<PromptAudioEmbeddingCacheEntry>
      uncached_prompt_audio_embedding_;
};

VoxCPM1FeatureGeneratorRuntime::VoxCPM1FeatureGeneratorRuntime(
    std::shared_ptr<const VoxCPM1Assets> assets,
    engine::core::ExecutionContext &execution_context,
    VoxCPM1FeatureGeneratorConfig config)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   std::move(config))) {}

VoxCPM1FeatureGeneratorRuntime::~VoxCPM1FeatureGeneratorRuntime() = default;

VoxCPM1Result VoxCPM1FeatureGeneratorRuntime::generate_zero_shot(
    const std::string &text, const VoxCPM1GenerationOptions &options) {
  return impl_->generate_zero_shot(text, options);
}

VoxCPM1Result VoxCPM1FeatureGeneratorRuntime::generate(
    const std::string &text, const VoxCPM1EncodedPrompt *prompt,
    const VoxCPM1GenerationOptions &options) {
  return impl_->generate(text, prompt, options);
}

VoxCPM1StreamingResult VoxCPM1FeatureGeneratorRuntime::generate_streaming(
    const std::string &text, const VoxCPM1EncodedPrompt *prompt,
    const VoxCPM1GenerationOptions &options,
    const std::function<void(const VoxCPM1StreamingChunk &)> &chunk_callback) {
  return impl_->generate_streaming(text, prompt, options, chunk_callback);
}

void VoxCPM1FeatureGeneratorRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

void VoxCPM1FeatureGeneratorRuntime::release_text_length_memory() {
  impl_->release_text_length_memory();
}

} // namespace engine::community_models::voxcpm1

