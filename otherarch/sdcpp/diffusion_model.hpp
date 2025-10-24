#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "flux.hpp"
#include "mmdit.hpp"
#include "qwen_image.hpp"
#include "unet.hpp"
#include "wan.hpp"

struct DiffusionParams {
    struct ggml_tensor* x                     = nullptr;
    struct ggml_tensor* timesteps             = nullptr;
    struct ggml_tensor* context               = nullptr;
    struct ggml_tensor* c_concat              = nullptr;
    struct ggml_tensor* y                     = nullptr;
    struct ggml_tensor* guidance              = nullptr;
    std::vector<ggml_tensor*> ref_latents     = {};
    bool increase_ref_index                   = false;
    int num_video_frames                      = -1;
    std::vector<struct ggml_tensor*> controls = {};
    float control_strength                    = 0.f;
    struct ggml_tensor* vace_context          = nullptr;
    float vace_strength                       = 1.f;
    std::vector<int> skip_layers              = {};
};

struct DiffusionModel {
    virtual std::string get_desc()                                                      = 0;
    virtual void compute(int n_threads,
                         DiffusionParams diffusion_params,
                         struct ggml_tensor** output     = nullptr,
                         struct ggml_context* output_ctx = nullptr)                     = 0;
    virtual void alloc_params_buffer()                                                  = 0;
    virtual void free_params_buffer()                                                   = 0;
    virtual void free_compute_buffer()                                                  = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                             = 0;
    virtual int64_t get_adm_in_channels()                                               = 0;
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2GGMLType& tensor_types = {},
              SDVersion version                   = VERSION_SD1,
              bool flash_attn                     = false)
        : unet(backend, offload_params_to_cpu, tensor_types, "model.diffusion_model", version, flash_attn) {
    }

    std::string get_desc() override {
        return unet.get_desc();
    }

    void alloc_params_buffer() override {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() override {
        unet.free_params_buffer();
    }

    void free_compute_buffer() override {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return unet.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() override {
        return unet.unet.adm_in_channels;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return unet.compute(n_threads,
                            diffusion_params.x,
                            diffusion_params.timesteps,
                            diffusion_params.context,
                            diffusion_params.c_concat,
                            diffusion_params.y,
                            diffusion_params.num_video_frames,
                            diffusion_params.controls,
                            diffusion_params.control_strength, output, output_ctx);
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               bool flash_attn                     = false,
               const String2GGMLType& tensor_types = {})
        : mmdit(backend, offload_params_to_cpu, flash_attn, tensor_types, "model.diffusion_model") {
    }

    std::string get_desc() override {
        return mmdit.get_desc();
    }

    void alloc_params_buffer() override {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() override {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() override {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return mmdit.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() override {
        return 768 + 1280;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return mmdit.compute(n_threads,
                             diffusion_params.x,
                             diffusion_params.timesteps,
                             diffusion_params.context,
                             diffusion_params.y,
                             output,
                             output_ctx,
                             diffusion_params.skip_layers);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;

    FluxModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2GGMLType& tensor_types = {},
              SDVersion version                   = VERSION_FLUX,
              bool flash_attn                     = false,
              bool use_mask                       = false)
        : flux(backend, offload_params_to_cpu, tensor_types, "model.diffusion_model", version, flash_attn, use_mask) {
    }

    std::string get_desc() override {
        return flux.get_desc();
    }

    void alloc_params_buffer() override {
        flux.alloc_params_buffer();
    }

    void free_params_buffer() override {
        flux.free_params_buffer();
    }

    void free_compute_buffer() override {
        flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return flux.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return flux.compute(n_threads,
                            diffusion_params.x,
                            diffusion_params.timesteps,
                            diffusion_params.context,
                            diffusion_params.c_concat,
                            diffusion_params.y,
                            diffusion_params.guidance,
                            diffusion_params.ref_latents,
                            diffusion_params.increase_ref_index,
                            output,
                            output_ctx,
                            diffusion_params.skip_layers);
    }
};

struct WanModel : public DiffusionModel {
    std::string prefix;
    WAN::WanRunner wan;

    WanModel(ggml_backend_t backend,
             bool offload_params_to_cpu,
             const String2GGMLType& tensor_types = {},
             const std::string prefix            = "model.diffusion_model",
             SDVersion version                   = VERSION_WAN2,
             bool flash_attn                     = false)
        : prefix(prefix), wan(backend, offload_params_to_cpu, tensor_types, prefix, version, flash_attn) {
    }

    std::string get_desc() override {
        return wan.get_desc();
    }

    void alloc_params_buffer() override {
        wan.alloc_params_buffer();
    }

    void free_params_buffer() override {
        wan.free_params_buffer();
    }

    void free_compute_buffer() override {
        wan.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        wan.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return wan.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return wan.compute(n_threads,
                           diffusion_params.x,
                           diffusion_params.timesteps,
                           diffusion_params.context,
                           diffusion_params.y,
                           diffusion_params.c_concat,
                           nullptr,
                           diffusion_params.vace_context,
                           diffusion_params.vace_strength,
                           output,
                           output_ctx);
    }
};

struct QwenImageModel : public DiffusionModel {
    std::string prefix;
    Qwen::QwenImageRunner qwen_image;

    QwenImageModel(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2GGMLType& tensor_types = {},
                   const std::string prefix            = "model.diffusion_model",
                   SDVersion version                   = VERSION_QWEN_IMAGE,
                   bool flash_attn                     = false)
        : prefix(prefix), qwen_image(backend, offload_params_to_cpu, tensor_types, prefix, version, flash_attn) {
    }

    std::string get_desc() override {
        return qwen_image.get_desc();
    }

    void alloc_params_buffer() override {
        qwen_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        qwen_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        qwen_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        qwen_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return qwen_image.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr) override {
        return qwen_image.compute(n_threads,
                                  diffusion_params.x,
                                  diffusion_params.timesteps,
                                  diffusion_params.context,
                                  diffusion_params.ref_latents,
                                  true,  // increase_ref_index
                                  output,
                                  output_ctx);
    }
};

#endif
