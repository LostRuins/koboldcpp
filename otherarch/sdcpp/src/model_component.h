#ifndef __SD_MODEL_COMPONENT_H__
#define __SD_MODEL_COMPONENT_H__

enum class ModelComponent {
    Conditioner,
    Diffusion,
    HighNoiseDiffusion,
    CLIPVision,
    IPAdapter,
    VAE,
    PreviewVAE,
    AudioVAE,
    ControlNet,
    PhotoMaker,
    PuLID,
    LoRA,
    Upscaler,
    Detector,
    LatentUpsampler,
    Count,
};

inline const char* model_component_name(ModelComponent component) {
    switch (component) {
        case ModelComponent::Conditioner:
            return "Conditioner model";
        case ModelComponent::Diffusion:
            return "Diffusion model";
        case ModelComponent::HighNoiseDiffusion:
            return "High noise diffusion model";
        case ModelComponent::CLIPVision:
            return "CLIP vision";
        case ModelComponent::IPAdapter:
            return "IP-Adapter";
        case ModelComponent::VAE:
            return "VAE";
        case ModelComponent::PreviewVAE:
            return "preview VAE";
        case ModelComponent::AudioVAE:
            return "audio VAE";
        case ModelComponent::ControlNet:
            return "ControlNet";
        case ModelComponent::PhotoMaker:
            return "photomaker";
        case ModelComponent::PuLID:
            return "pulid";
        case ModelComponent::LoRA:
            return "LoRA";
        case ModelComponent::Upscaler:
            return "ESRGAN";
        case ModelComponent::Detector:
            return "YOLOv8";
        case ModelComponent::LatentUpsampler:
            return "LTX latent upsampler";
        case ModelComponent::Count:
            break;
    }
    return "unknown";
}

#endif  // __SD_MODEL_COMPONENT_H__
