#include "backend_fit.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "ggml-backend.h"

namespace sd::backend_fit {
    namespace {

        constexpr int64_t MiB = 1024ll * 1024;

        enum class ComponentKind {
            DIT         = 0,
            VAE         = 1,
            CONDITIONER = 2,
        };

        struct Component {
            ComponentKind kind;
            const char* name;
            int64_t params_bytes  = 0;
            int64_t reserve_bytes = 0;
            bool splittable       = false;
        };

        struct Device {
            ggml_backend_dev_t dev = nullptr;
            std::string name;
            std::string description;
            int64_t free_bytes   = 0;
            int64_t total_bytes  = 0;
            int64_t budget_bytes = 0;
        };

        struct Decision {
            ComponentKind kind;
            bool on_cpu = false;
            std::vector<size_t> device_idxs;
        };

        struct Plan {
            bool valid      = false;
            bool time_share = false;
            std::vector<Decision> decisions;
        };

        bool classify_tensor(const std::string& name, ComponentKind& out) {
            auto contains = [&](const char* s) { return name.find(s) != std::string::npos; };

            if (contains("model.diffusion_model.") || contains("unet.")) {
                out = ComponentKind::DIT;
                return true;
            }
            if (contains("first_stage_model.") ||
                name.rfind("vae.", 0) == 0 ||
                name.rfind("tae.", 0) == 0) {
                out = ComponentKind::VAE;
                return true;
            }
            if (contains("text_encoders") ||
                contains("cond_stage_model") ||
                contains("te.text_model.") ||
                contains("conditioner") ||
                name.rfind("text_encoder.", 0) == 0 ||
                name.rfind("text_embedding_projection.", 0) == 0 ||
                contains(".aggregate_embed.")) {
                out = ComponentKind::CONDITIONER;
                return true;
            }
            return false;
        }

        std::vector<Component> estimate_components(ModelLoader& loader, ggml_type override_wtype) {
            const auto& storage = loader.get_tensor_storage_map();

            int64_t bytes[3] = {0, 0, 0};
            for (const auto& [name, ts_const] : storage) {
                TensorStorage ts = ts_const;
                if (is_unused_tensor(ts.name)) {
                    continue;
                }
                ComponentKind kind;
                if (!classify_tensor(ts.name, kind)) {
                    continue;
                }
                if (override_wtype != GGML_TYPE_COUNT &&
                    loader.tensor_should_be_converted(ts, override_wtype)) {
                    ts.type = override_wtype;
                } else if (ts.expected_type != GGML_TYPE_COUNT && ts.expected_type != ts.type) {
                    ts.type = ts.expected_type;
                }
                bytes[int(kind)] += (int64_t)ts.nbytes() + 64;
            }

            std::vector<Component> out;
            out.push_back({ComponentKind::DIT, "DiT", bytes[int(ComponentKind::DIT)], 2048 * MiB, true});
            out.push_back({ComponentKind::VAE, "VAE", bytes[int(ComponentKind::VAE)], 1024 * MiB, false});
            out.push_back({ComponentKind::CONDITIONER, "Conditioner", bytes[int(ComponentKind::CONDITIONER)], 2048 * MiB, true});
            return out;
        }

        std::vector<Device> enumerate_gpu_devices(const sd::ggml_graph_cut::MaxVramAssignment& budgets) {
            std::vector<Device> out;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                    continue;
                }
                Device d;
                d.dev             = dev;
                d.name            = ggml_backend_dev_name(dev);
                d.description     = ggml_backend_dev_description(dev);
                size_t free_bytes = 0, total_bytes = 0;
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
                d.free_bytes  = (int64_t)free_bytes;
                d.total_bytes = (int64_t)total_bytes;

                std::string budget_key = d.name;
                std::transform(budget_key.begin(), budget_key.end(), budget_key.begin(),
                               [](unsigned char c) { return (char)std::tolower(c); });
                float gib = budgets.default_gib;
                auto it   = budgets.backend_gib.find(budget_key);
                if (it != budgets.backend_gib.end()) {
                    gib = it->second;
                }
                if (gib > 0.f) {
                    d.budget_bytes = std::min<int64_t>((int64_t)(gib * 1024.0 * 1024.0 * 1024.0), d.free_bytes);
                } else if (gib < 0.f) {
                    d.budget_bytes = d.free_bytes + (int64_t)(gib * 1024.0 * 1024.0 * 1024.0);
                } else {
                    d.budget_bytes = d.free_bytes - 512 * MiB;
                }
                d.budget_bytes = std::max<int64_t>(d.budget_bytes, 0);
                out.push_back(d);
            }
            return out;
        }

        Plan compute_plan(const std::vector<Component>& components, const std::vector<Device>& devices) {
            Plan plan;
            if (devices.empty()) {
                return plan;
            }

            std::vector<size_t> order(components.size());
            for (size_t i = 0; i < order.size(); i++) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return components[a].params_bytes > components[b].params_bytes;
            });

            {
                std::vector<int64_t> params_sum(devices.size(), 0);
                std::vector<int64_t> max_reserve(devices.size(), 0);
                std::vector<Decision> decisions(components.size());
                bool ok = true;
                for (size_t ci : order) {
                    const Component& comp = components[ci];
                    decisions[ci].kind    = comp.kind;
                    if (comp.params_bytes == 0) {
                        continue;
                    }
                    int best = -1;
                    for (size_t di = 0; di < devices.size(); di++) {
                        int64_t need = params_sum[di] + comp.params_bytes + std::max(max_reserve[di], comp.reserve_bytes);
                        if (need <= devices[di].budget_bytes &&
                            (best < 0 || devices[di].budget_bytes - params_sum[di] > devices[best].budget_bytes - params_sum[best])) {
                            best = (int)di;
                        }
                    }
                    if (best < 0) {
                        ok = false;
                        break;
                    }
                    params_sum[best] += comp.params_bytes;
                    max_reserve[best] = std::max(max_reserve[best], comp.reserve_bytes);
                    decisions[ci].device_idxs.push_back((size_t)best);
                }
                if (ok) {
                    plan.valid      = true;
                    plan.time_share = false;
                    plan.decisions  = std::move(decisions);
                    return plan;
                }
            }

            plan.decisions.assign(components.size(), {});
            for (size_t ci : order) {
                const Component& comp = components[ci];
                Decision& decision    = plan.decisions[ci];
                decision.kind         = comp.kind;
                if (comp.params_bytes == 0) {
                    continue;
                }
                int best = -1;
                for (size_t di = 0; di < devices.size(); di++) {
                    if (comp.params_bytes + comp.reserve_bytes <= devices[di].budget_bytes &&
                        (best < 0 || devices[di].budget_bytes > devices[best].budget_bytes)) {
                        best = (int)di;
                    }
                }
                if (best >= 0) {
                    decision.device_idxs.push_back((size_t)best);
                    continue;
                }
                if (comp.splittable && devices.size() > 1) {
                    int64_t capacity = 0;
                    for (const Device& d : devices) {
                        capacity += std::max<int64_t>(d.budget_bytes - comp.reserve_bytes, 0);
                    }
                    if (comp.params_bytes <= capacity) {
                        std::vector<size_t> idxs(devices.size());
                        for (size_t i = 0; i < idxs.size(); i++) {
                            idxs[i] = i;
                        }
                        std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
                            return devices[a].budget_bytes > devices[b].budget_bytes;
                        });
                        decision.device_idxs = std::move(idxs);
                        continue;
                    }
                }
                decision.on_cpu = true;
            }
            plan.valid      = true;
            plan.time_share = true;
            return plan;
        }

        void print_plan(const Plan& plan,
                        const std::vector<Component>& components,
                        const std::vector<Device>& devices) {
            LOG_INFO("auto-fit plan%s:", plan.time_share ? " (time-share: params load per phase and free after)" : "");
            LOG_INFO("  devices:");
            for (const Device& d : devices) {
                LOG_INFO("    %-12s %-32s free %6lld MiB, budget %6lld MiB",
                         d.name.c_str(), d.description.c_str(),
                         (long long)(d.free_bytes / MiB), (long long)(d.budget_bytes / MiB));
            }
            LOG_INFO("  components:");
            for (size_t ci = 0; ci < components.size(); ci++) {
                const Component& comp    = components[ci];
                const Decision& decision = plan.decisions[ci];
                std::string target;
                if (comp.params_bytes == 0) {
                    target = "(not present)";
                } else if (decision.on_cpu) {
                    target = "CPU";
                } else {
                    for (size_t k = 0; k < decision.device_idxs.size(); k++) {
                        if (k > 0) {
                            target += " & ";
                        }
                        target += devices[decision.device_idxs[k]].name;
                    }
                    if (decision.device_idxs.size() > 1) {
                        target += " (split)";
                    }
                }
                LOG_INFO("    %-12s params %6lld MiB, compute reserve %5lld MiB -> %s",
                         comp.name,
                         (long long)(comp.params_bytes / MiB),
                         (long long)(comp.reserve_bytes / MiB),
                         target.c_str());
            }
        }

        void append_assignment(std::string& spec, const char* key, const std::string& value) {
            if (!spec.empty()) {
                spec += ",";
            }
            spec += key;
            spec += "=";
            spec += value;
        }

        void append_component_decision(const std::vector<Component>& components,
                                       const std::vector<Device>& devices,
                                       const Plan& plan,
                                       ComponentKind kind,
                                       const char* module_key,
                                       std::string& runtime_spec,
                                       std::string& params_spec) {
            for (size_t ci = 0; ci < components.size(); ci++) {
                if (components[ci].kind != kind || components[ci].params_bytes == 0) {
                    continue;
                }
                const Decision& decision = plan.decisions[ci];
                if (decision.on_cpu) {
                    append_assignment(runtime_spec, module_key, "cpu");
                    return;
                }
                if (decision.device_idxs.empty()) {
                    return;
                }
                std::string device_list;
                for (size_t k = 0; k < decision.device_idxs.size(); k++) {
                    if (k > 0) {
                        device_list += "&";
                    }
                    device_list += devices[decision.device_idxs[k]].name;
                }
                append_assignment(runtime_spec, module_key, device_list);
                if (plan.time_share) {
                    append_assignment(params_spec, module_key, "disk");
                }
                return;
            }
        }

    }  // namespace

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec) {
        if (!runtime_spec.empty() || !params_spec.empty()) {
            LOG_WARN("--auto-fit is enabled; ignoring --backend / --params-backend");
        }

        {
            std::string error;
            if (!budgets.canonicalize_backend_keys(&error)) {
                LOG_ERROR("%s", error.c_str());
                return false;
            }
        }

        auto components = estimate_components(loader, override_wtype);
        auto devices    = enumerate_gpu_devices(budgets);
        auto plan       = compute_plan(components, devices);
        if (!plan.valid) {
            LOG_WARN("auto-fit: no usable GPU devices; using the default backend");
            runtime_spec.clear();
            params_spec.clear();
            return true;
        }

        print_plan(plan, components, devices);

        std::string derived_runtime_spec;
        std::string derived_params_spec;
        append_component_decision(components, devices, plan, ComponentKind::DIT, "diffusion", derived_runtime_spec, derived_params_spec);
        append_component_decision(components, devices, plan, ComponentKind::CONDITIONER, "te", derived_runtime_spec, derived_params_spec);
        append_component_decision(components, devices, plan, ComponentKind::VAE, "vae", derived_runtime_spec, derived_params_spec);

        runtime_spec = std::move(derived_runtime_spec);
        params_spec  = std::move(derived_params_spec);

        LOG_INFO("auto-fit: --backend \"%s\"%s%s%s",
                 runtime_spec.empty() ? "(default)" : runtime_spec.c_str(),
                 params_spec.empty() ? "" : " --params-backend \"",
                 params_spec.c_str(),
                 params_spec.empty() ? "" : "\"");
        return true;
    }

    bool prepare_vae_decode_retry_tiling(sd_tiling_params_t& tiling_params, bool prefer_temporal_tiling) {
        if (prefer_temporal_tiling) {
            if (tiling_params.temporal_tiling) {
                return false;
            }
            tiling_params.temporal_tiling = true;
        } else {
            if (tiling_params.enabled) {
                return false;
            }
            tiling_params.enabled = true;
            if (tiling_params.tile_size_x <= 0) {
                tiling_params.tile_size_x = 256;
            }
            if (tiling_params.tile_size_y <= 0) {
                tiling_params.tile_size_y = 256;
            }
        }

        LOG_WARN("auto-fit: VAE decode failed (likely out of memory); retrying with %s tiling",
                 tiling_params.temporal_tiling ? "temporal" : "spatial");
        return true;
    }

}  // namespace sd::backend_fit
