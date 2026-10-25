#include "engine/models/yue2/assets.h"

#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/debug/trace.h"

#include <cmath>
#include <utility>

namespace engine::models::yue2 {

std::shared_ptr<const assets::TensorSource> make_yue2_lora_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path, float scale, int64_t layer_count,
    const std::filesystem::path & nar_adapter_path, float nar_scale) {
    assets::WeightAdapter merged;
    for (const bool nar : {false, true}) {
        const auto & path = nar ? nar_adapter_path : adapter_path;
        if (path.empty()) continue;
        const float strength = nar ? nar_scale : scale;
        const std::string option = nar ? "yue2.nar_lora" : "yue2.ar_lora";
        if (!std::isfinite(strength)) throw std::runtime_error(option + "_scale must be finite");
        assets::WeightAdapterBindings bindings;
        for (int64_t layer = 0; layer < layer_count; ++layer) {
            for (const auto * projection : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
                const auto prefix = "layers." + std::to_string(layer) + "." + (nar ? "nar_" : "") + projection;
                const auto name = "model." + prefix + ".weight";
                bindings.add(prefix, name);
                bindings.add("model." + prefix, name);
            }
            const auto root = "layers." + std::to_string(layer) + "." + (nar ? "nar_" : "");
            for (const auto & fused : {
                     std::make_pair(std::string("self_attn.qkv_proj"), std::vector<std::string>{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}),
                     std::make_pair(std::string("mlp.gate_up_proj"), std::vector<std::string>{"mlp.gate_proj", "mlp.up_proj"})}) {
                std::vector<assets::WeightAdapterTarget> targets;
                int64_t offset = 0;
                for (const auto & projection : fused.second) {
                    const auto name = "model." + root + projection + ".weight";
                    const auto rows = base->require_metadata(name).shape.at(0);
                    targets.push_back({name, offset, rows});
                    offset += rows;
                }
                bindings.add(root + fused.first, targets);
                bindings.add("model." + root + fused.first, std::move(targets));
            }
        }
        if (nar) {
            for (const auto * name : {"vae2llm.weight", "vae2llm.bias", "llm2vae.weight", "llm2vae.bias"}) {
                bindings.add(name, name);
            }
        }
        assets::WeightAdapterReadOptions read_options;
        read_options.strength = strength;
        auto adapter = assets::bind_weight_adapter(*base, assets::read_weight_adapter(path, read_options),
            bindings, assets::WeightMergeMode::RoundedBF16Delta);
        const auto projections = adapter.updates.size();
        if (projections == 0) throw std::runtime_error(option + " contains no matching weight updates");
        if (strength != 0.0F) {
            merged.updates.merge(adapter.updates);
            merged.replacements.merge(adapter.replacements);
        }
        engine::debug::log_message(engine::debug::LogLevel::Info, "yue2",
            option + ": " + path.string() + ", projections=" + std::to_string(projections) +
            ", scale=" + std::to_string(strength));
    }
    return assets::make_weight_adapter_source(std::move(base), std::move(merged), "yue2.lora", true);
}

}  // namespace engine::models::yue2
