#include "generation_metadata.h"

#include <filesystem>
#include <map>

#include "json.hpp"

static void relativize_lora_paths(nlohmann::json& info, const std::filesystem::path& root) {
    if (info.contains("loras")) {
        for (auto& lora : info["loras"]) {
            const auto path = lora.value("path", "");
            if (path.empty())
                continue;
            std::error_code error;
            const auto absolute = std::filesystem::absolute(std::filesystem::u8path(path), error);
            if (error)
                continue;
            const auto relative = absolute.lexically_normal().lexically_relative(root);
            if (!relative.empty())
                lora["path"] = relative.generic_u8string();
        }
    }
    if (info.contains("inpaint_params"))
        relativize_lora_paths(info["inpaint_params"], root);
    if (info.contains("postprocess")) {
        for (auto& operation : info["postprocess"])
            relativize_lora_paths(operation, root);
    }
}

std::string generation_info_with_relative_lora_paths(const std::string& info, const std::string& lora_model_dir) {
    if (info.empty() || lora_model_dir.empty())
        return info;
    std::error_code error;
    const auto root = std::filesystem::absolute(std::filesystem::u8path(lora_model_dir), error);
    if (error)
        return info;
    auto result = nlohmann::json::parse(info);
    relativize_lora_paths(result, root.lexically_normal());
    return result.dump();
}

std::string merge_generation_info(const std::string& source, const std::string& operation) {
    if (source.empty())
        return operation;
    auto result     = nlohmann::json::parse(source);
    const auto next = nlohmann::json::parse(operation);
    result["postprocess"].push_back(next);
    return result.dump();
}

int64_t generation_seed(const std::string& info, int64_t fallback) {
    if (info.empty())
        return fallback;
    const auto value   = nlohmann::json::parse(info);
    const auto& params = value.contains("inpaint_params") ? value.at("inpaint_params") : value;
    return params.contains("seed") ? std::stoll(params.at("seed").get<std::string>()) : fallback;
}

std::string generation_info_to_webui(const std::string& text) {
    using nlohmann::json;
    if (text.empty())
        return "";
    const auto info = json::parse(text);
    if (info.value("schema", "") != "sdcpp.generation/v1")
        return "";
    // Standalone postprocessing has no source diffusion settings to import.
    if (!info.contains("sample_params"))
        return "";
    const auto& sampling = info.at("sample_params");
    std::string result   = info.value("prompt", "");
    if (info.contains("loras")) {
        for (const auto& lora : info.at("loras")) {
            // WebUI tags cannot restrict a LoRA to the high-noise model.
            if (lora.value("is_high_noise", false))
                continue;
            auto name            = lora.at("path").get<std::string>();
            const auto separator = name.find_last_of("/\\");
            if (separator != std::string::npos)
                name.erase(0, separator + 1);
            const auto extension = name.find_last_of('.');
            if (extension != std::string::npos && extension > 0)
                name.erase(extension);
            if (name.empty() || name.find_first_of(":<>\r\n") != std::string::npos)
                continue;
            if (!result.empty())
                result += " ";
            result += "<lora:" + name + ":" + lora.at("multiplier").dump() + ">";
        }
    }
    const auto negative = info.value("negative_prompt", "");
    if (!negative.empty())
        result += "\nNegative prompt: " + negative;
    result += "\n";
    bool first       = true;
    const auto field = [&](const char* key, const json& value) {
        std::string v = value.is_string() ? value.get<std::string>() : value.dump();
        if (v.find_first_of(",\n\r:\"") != std::string::npos)
            v = json(v).dump(-1, ' ', false);
        if (!first)
            result += ", ";
        first = false;
        result += std::string(key) + ": " + v;
    };
    field("Steps", sampling.contains("custom_sigmas") ? json(sampling.at("custom_sigmas").size() - 1) : sampling.at("sample_steps"));
    static const std::map<std::string, std::string> samplers = {
        {"euler", "Euler"}, {"euler_a", "Euler a"}, {"heun", "Heun"}, {"dpm2", "DPM2"}, {"dpm++2s_a", "DPM++ 2S a"}, {"dpm++2m", "DPM++ 2M"}, {"dpm++2m_sde", "DPM++ 2M SDE"}, {"lcm", "LCM"}};
    const auto method = sampling.at("sample_method").get<std::string>();
    auto mapped       = samplers.find(method);
    field(mapped != samplers.end() ? "Sampler" : "SDCPP Sampler", mapped != samplers.end() ? mapped->second : method);
    static const std::map<std::string, std::string> schedules = {
        {"discrete", "Normal"}, {"karras", "Karras"}, {"exponential", "Exponential"}, {"sgm_uniform", "SGM Uniform"}, {"simple", "Simple"}, {"ddim_uniform", "DDIM"}, {"beta", "Beta"}};
    if (sampling.contains("scheduler")) {
        const auto scheduler = sampling.at("scheduler").get<std::string>();
        auto mapped_schedule = schedules.find(scheduler);
        field(mapped_schedule != schedules.end() ? "Schedule type" : "SDCPP Schedule", mapped_schedule != schedules.end() ? mapped_schedule->second : scheduler);
    }
    field("CFG scale", sampling.at("guidance").at("txt_cfg"));
    field("Seed", info.at("seed"));
    field("Size", std::to_string(info.at("width").get<int>()) + "x" + std::to_string(info.at("height").get<int>()));
    if (info.contains("models")) {
        const auto& models   = info.at("models");
        auto model           = models.value("diffusion_model", models.value("model", ""));
        const auto extension = model.find_last_of('.');
        if (extension != std::string::npos && extension > 0)
            model.erase(extension);
        if (!model.empty())
            field("Model", model);
        if (models.contains("vae"))
            field("VAE", models.at("vae"));
    }
    if (sampling.contains("eta"))
        field("Eta", sampling.at("eta"));
    if (info.value("clip_skip", 0) > 0)
        field("Clip skip", info.at("clip_skip"));
    if (info.contains("strength"))
        field("Denoising strength", info.at("strength"));
    if (info.contains("hires") && info.at("hires").value("enabled", false)) {
        const auto& hires = info.at("hires");
        field("Hires resize", std::to_string(hires.at("target_width").get<int>()) + "x" + std::to_string(hires.at("target_height").get<int>()));
        if (hires.value("steps", 0) > 0)
            field("Hires steps", hires.at("steps"));
        field("Denoising strength", hires.at("denoising_strength"));
        field("Hires upscaler", hires.at("upscaler"));
    }
    field("Version", "stable-diffusion.cpp");
    return result;
}

std::string generation_info_to_image_metadata(const std::string& info) {
    if (info.empty())
        return "";
    const auto parameters = generation_info_to_webui(info);
    return parameters.empty() ? info : parameters + ", SDCPP: " + info;
}
