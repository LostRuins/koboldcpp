#ifndef __SD_PIPELINE_MODEL_BUILDERS_H__
#define __SD_PIPELINE_MODEL_BUILDERS_H__

#include <memory>
#include <vector>

#include "model.h"
#include "stable-diffusion.h"

class SDBackendManager;
struct DeviceResidencyManager;
struct Conditioner;
struct FrozenCLIPVisionEmbedder;
struct DiffusionModelRunner;
struct VAE;
struct AudioVAERunner;
struct ControlNet;
struct GenerationExtension;
struct GenerationExtensionInitContext;
namespace IPAdapter {
    struct IPAdapterRunner;
}

namespace sd::model_builders {

    struct Context {
        const sd_ctx_params_t& params;
        SDVersion version;
        const String2TensorStorage& tensor_storage_map;
        SDBackendManager& backends;
        std::shared_ptr<DeviceResidencyManager> weight_manager;
    };

    struct CoreRunners {
        std::shared_ptr<Conditioner> conditioner;
        std::shared_ptr<DiffusionModelRunner> diffusion;
        std::shared_ptr<DiffusionModelRunner> high_noise_diffusion;
        std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;
        std::shared_ptr<IPAdapter::IPAdapterRunner> ip_adapter;
    };

    struct VAEOptions {
        bool use_tae                 = false;
        bool tae_preview_only        = false;
        bool use_audio_vae           = false;
        bool external_vae_is_invalid = false;
    };

    struct VAERunners {
        std::shared_ptr<VAE> vae;
        std::shared_ptr<VAE> preview;
        std::shared_ptr<AudioVAERunner> audio;
    };

    bool build_core_runners(const Context& ctx, CoreRunners& runners);
    bool build_vae_runners(const Context& ctx, const VAEOptions& options, VAERunners& runners);
    bool build_control_net_runner(const Context& ctx, std::shared_ptr<ControlNet>& runner);
    bool build_extension_runners(const GenerationExtensionInitContext& ctx,
                                 std::vector<std::shared_ptr<GenerationExtension>>& extensions);

}  // namespace sd::model_builders

#endif  // __SD_PIPELINE_MODEL_BUILDERS_H__
