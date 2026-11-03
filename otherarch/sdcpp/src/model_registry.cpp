#include <array>
#include <stdexcept>
#include "model_definition.h"

const std::vector<ModelDefinition>& fibo2_definitions();
const std::vector<ModelDefinition>& sd3_definitions();
const std::vector<ModelDefinition>& pid_definitions();
const std::vector<ModelDefinition>& ideogram4_definitions();
const std::vector<ModelDefinition>& krea2_definitions();
const std::vector<ModelDefinition>& flux_definitions();
const std::vector<ModelDefinition>& flux3_definitions();
const std::vector<ModelDefinition>& flux2_definitions();
const std::vector<ModelDefinition>& ltxav_definitions();
const std::vector<ModelDefinition>& minimax_h3_definitions();
const std::vector<ModelDefinition>& hunyuan_video_definitions();
const std::vector<ModelDefinition>& wan_definitions();
const std::vector<ModelDefinition>& lingbot_video_definitions();
const std::vector<ModelDefinition>& qwen_image_definitions();
const std::vector<ModelDefinition>& pixart_definitions();
const std::vector<ModelDefinition>& mage_flow_definitions();
const std::vector<ModelDefinition>& longcat_definitions();
const std::vector<ModelDefinition>& hidream_o1_definitions();
const std::vector<ModelDefinition>& minit2i_definitions();
const std::vector<ModelDefinition>& sensenova_u1_definitions();
const std::vector<ModelDefinition>& anima_definitions();
const std::vector<ModelDefinition>& ming_image_definitions();
const std::vector<ModelDefinition>& z_image_definitions();
const std::vector<ModelDefinition>& z_image_l2p_definitions();
const std::vector<ModelDefinition>& llada_image_definitions();
const std::vector<ModelDefinition>& boogu_image_definitions();
const std::vector<ModelDefinition>& ernie_image_definitions();
const std::vector<ModelDefinition>& lens_definitions();
const std::vector<ModelDefinition>& iris_definitions();
const std::vector<ModelDefinition>& sd_definitions();
const std::vector<ModelDefinition>& moss_image_definitions();

static const std::array<const ModelDefinition*, VERSION_COUNT>& model_definitions() {
    static const auto definitions = [] {
        std::array<const ModelDefinition*, VERSION_COUNT> result{};
        auto append = [&](const std::vector<ModelDefinition>& family) {
            for (const auto& definition : family) {
                if (definition.version < 0 || definition.version >= VERSION_COUNT || result[definition.version]) {
                    throw std::logic_error("duplicate or invalid model registration");
                }
                result[definition.version] = &definition;
            }
        };
        append(fibo2_definitions());
        append(sd3_definitions());
        append(pid_definitions());
        append(ideogram4_definitions());
        append(krea2_definitions());
        append(flux_definitions());
        append(flux3_definitions());
        append(flux2_definitions());
        append(ltxav_definitions());
        append(minimax_h3_definitions());
        append(hunyuan_video_definitions());
        append(wan_definitions());
        append(lingbot_video_definitions());
        append(qwen_image_definitions());
        append(pixart_definitions());
        append(mage_flow_definitions());
        append(longcat_definitions());
        append(hidream_o1_definitions());
        append(minit2i_definitions());
        append(sensenova_u1_definitions());
        append(anima_definitions());
        append(ming_image_definitions());
        append(z_image_definitions());
        append(z_image_l2p_definitions());
        append(llada_image_definitions());
        append(boogu_image_definitions());
        append(ernie_image_definitions());
        append(lens_definitions());
        append(iris_definitions());
        append(sd_definitions());
        append(moss_image_definitions());
        for (const auto* definition : result) {
            if (!definition)
                throw std::logic_error("missing model registration");
        }
        return result;
    }();
    return definitions;
}

const ModelDefinition* find_model_definition(SDVersion version) {
    if (version < 0 || version >= VERSION_COUNT)
        return nullptr;
    return model_definitions()[version];
}

const ModelTraits* find_model_traits(SDVersion version) {
    const auto* definition = find_model_definition(version);
    return definition ? &definition->traits : nullptr;
}

const char* sd_version_name(SDVersion version) {
    const auto* definition = find_model_definition(version);
    return definition ? definition->name : "Unknown";
}
