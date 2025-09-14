#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "flux.hpp"
#include "mmdit.hpp"
#include "unet.hpp"
#include "wan.hpp"

struct DiffusionParams {
    struct ggml_tensor* x                     = NULL;
    struct ggml_tensor* timesteps             = NULL;
    struct ggml_tensor* context               = NULL;
    struct ggml_tensor* c_concat              = NULL;
    struct ggml_tensor* y                     = NULL;
    struct ggml_tensor* guidance              = NULL;
    std::vector<ggml_tensor*> ref_latents     = {};
    bool increase_ref_index                   = false;
    int num_video_frames                      = -1;
    std::vector<struct ggml_tensor*> controls = {};
    float control_strength                    = 0.f;
    struct ggml_tensor* vace_context          = NULL;
    float vace_strength                       = 1.f;
    std::vector<int> skip_layers              = {};
};

struct DiffusionModel {
    virtual std::string get_desc()                                                      = 0;
    virtual void compute(int n_threads,
                         DiffusionParams diffusion_params,
                         struct ggml_tensor** output     = NULL,
                         struct ggml_context* output_ctx = NULL)                        = 0;
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

    std::string get_desc() {
        return unet.get_desc();
    }

    void alloc_params_buffer() {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() {
        unet.free_params_buffer();
    }

    void free_compute_buffer() {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return unet.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return unet.unet.adm_in_channels;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL) {
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

    std::string get_desc() {
        return mmdit.get_desc();
    }

    void alloc_params_buffer() {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return mmdit.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768 + 1280;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL) {
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

    std::string get_desc() {
        return flux.get_desc();
    }

    void alloc_params_buffer() {
        flux.alloc_params_buffer();
    }

    void free_params_buffer() {
        flux.free_params_buffer();
    }

    void free_compute_buffer() {
        flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return flux.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL) {
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

    std::string get_desc() {
        return wan.get_desc();
    }

    void alloc_params_buffer() {
        wan.alloc_params_buffer();
    }

    void free_params_buffer() {
        wan.free_params_buffer();
    }

    void free_compute_buffer() {
        wan.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        wan.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() {
        return wan.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768;
    }

    void compute(int n_threads,
                 DiffusionParams diffusion_params,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL) {
        return wan.compute(n_threads,
                           diffusion_params.x,
                           diffusion_params.timesteps,
                           diffusion_params.context,
                           diffusion_params.y,
                           diffusion_params.c_concat,
                           NULL,
                           diffusion_params.vace_context,
                           diffusion_params.vace_strength,
                           output,
                           output_ctx);
    }
};

#endif
