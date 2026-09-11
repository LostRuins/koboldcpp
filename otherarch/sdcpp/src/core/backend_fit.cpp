#include "backend_fit.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "ggml-backend.h"

namespace sd::backend_fit {

    static constexpr int64_t MiB = 1024ll * 1024;

    enum class ComponentKind {
        DIT,
        CONDITIONER,
        VAE,
    };

    struct Component {
        ComponentKind kind;
        const char* name;
        int64_t params_bytes  = 0;
        int64_t reserve_bytes = 0;
        int64_t staging_bytes = 0;
    };

    struct Device {
        std::string name;
        std::string description;
        int64_t free_bytes   = 0;
        int64_t budget_bytes = 0;
    };

    enum class ParamsLocation {
        MAIN_GPU,
        CPU,
        OTHER_GPU,
        DISK,
    };

    struct Decision {
        ParamsLocation params_location = ParamsLocation::DISK;
        size_t params_device           = SIZE_MAX;
    };

    struct Plan {
        bool valid         = false;
        size_t main_device = SIZE_MAX;
        std::vector<Decision> decisions;
    };

    static bool classify_tensor(const std::string& name, ComponentKind& out) {
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

    static std::vector<Component> estimate_components(ModelLoader& loader, ggml_type override_wtype) {
        int64_t bytes[3]          = {0, 0, 0};
        int64_t largest_tensor[3] = {0, 0, 0};
        for (const auto& [name, stored_tensor] : loader.get_tensor_storage_map()) {
            TensorStorage ts = stored_tensor;
            ComponentKind kind;
            if (is_unused_tensor(ts.name) || !classify_tensor(ts.name, kind)) {
                continue;
            }
            if (ts.expected_type != GGML_TYPE_COUNT) {
                ts.type = ts.expected_type;
            } else if (override_wtype != GGML_TYPE_COUNT && loader.tensor_should_be_converted(ts, override_wtype)) {
                ts.type = override_wtype;
            }
            const int64_t tensor_bytes = (int64_t)ts.nbytes() + 64;
            bytes[int(kind)] += tensor_bytes;
            largest_tensor[int(kind)] = std::max(largest_tensor[int(kind)], tensor_bytes);
        }

        return {
            {ComponentKind::DIT, "DiT", bytes[int(ComponentKind::DIT)], 2048 * MiB, largest_tensor[int(ComponentKind::DIT)]},
            {ComponentKind::CONDITIONER, "Conditioner", bytes[int(ComponentKind::CONDITIONER)], 2048 * MiB, largest_tensor[int(ComponentKind::CONDITIONER)]},
            {ComponentKind::VAE, "VAE", bytes[int(ComponentKind::VAE)], 1024 * MiB, largest_tensor[int(ComponentKind::VAE)]},
        };
    }

    static std::string budget_key(std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return name;
    }

    static std::vector<Device> enumerate_gpu_devices(const sd::ggml_graph_cut::MaxVramAssignment& budgets) {
        std::vector<Device> out;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            Device device;
            device.name        = ggml_backend_dev_name(dev);
            device.description = ggml_backend_dev_description(dev);
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            device.free_bytes = (int64_t)free_bytes;

            float gib = budgets.default_gib;
            auto it   = budgets.backend_gib.find(budget_key(device.name));
            if (it != budgets.backend_gib.end()) {
                gib = it->second;
            }
            if (gib > 0.f) {
                device.budget_bytes = (int64_t)std::min(gib * 1024.0 * MiB, (double)device.free_bytes);
            } else if (gib < 0.f) {
                device.budget_bytes = (int64_t)std::max<double>(device.free_bytes + gib * 1024.0 * MiB, 0);
            } else {
                device.budget_bytes = std::max<int64_t>(device.free_bytes - 512 * MiB, 0);
            }
            out.push_back(std::move(device));
        }
        return out;
    }

    static int64_t available_ram_bytes() {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            return (int64_t)status.ullAvailPhys;
        }
#elif defined(__linux__)
        std::ifstream meminfo("/proc/meminfo");
        std::string key, unit;
        int64_t kib = 0;
        while (meminfo >> key >> kib >> unit) {
            if (key == "MemAvailable:" && unit == "kB" && kib >= 0) {
                return kib * 1024;
            }
        }
#elif defined(__APPLE__)
        const mach_port_t host = mach_host_self();
        vm_size_t page_size    = 0;
        vm_statistics64_data_t stats{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        const bool ok                = host_page_size(host, &page_size) == KERN_SUCCESS &&
                        host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&stats, &count) == KERN_SUCCESS;
        mach_port_deallocate(mach_task_self(), host);
        if (ok) {
            return ((int64_t)stats.free_count + stats.inactive_count) * page_size;
        }
#endif
        return -1;
    }

    static Plan compute_plan(const std::vector<Component>& components,
                             const std::vector<Device>& devices,
                             int64_t ram_budget_bytes) {
        Plan plan;
        for (size_t di = 0; di < devices.size(); ++di) {
            if (devices[di].budget_bytes > 0 &&
                (plan.main_device == SIZE_MAX || devices[di].budget_bytes > devices[plan.main_device].budget_bytes)) {
                plan.main_device = di;
            }
        }
        if (plan.main_device == SIZE_MAX) {
            return plan;
        }

        std::vector<size_t> order(components.size());
        for (size_t ci = 0; ci < components.size(); ++ci) {
            order[ci] = ci;
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return components[a].kind < components[b].kind;
        });

        std::vector<int64_t> remaining;
        for (const Device& device : devices) {
            remaining.push_back(std::max<int64_t>(device.budget_bytes, 0));
        }
        ram_budget_bytes = std::max<int64_t>(ram_budget_bytes, 0);
        plan.decisions.resize(components.size());

        for (size_t ci : order) {
            const Component& comp = components[ci];
            Decision& decision    = plan.decisions[ci];
            if (comp.params_bytes == 0) {
                continue;
            }

            // Higher-priority offloaded weights need GPU cache space across graph runs.
            int64_t headroom = 0;
            for (size_t other = 0; other < components.size(); ++other) {
                if (components[other].params_bytes == 0) {
                    continue;
                }
                const bool resident          = other == ci || plan.decisions[other].params_location == ParamsLocation::MAIN_GPU;
                const int64_t cached_weights = components[other].kind < comp.kind
                                                   ? components[other].params_bytes
                                                   : components[other].staging_bytes;
                headroom                     = std::max(headroom, components[other].reserve_bytes +
                                                                      (resident ? 0 : cached_weights));
            }
            int64_t& main_remaining = remaining[plan.main_device];
            if (headroom <= main_remaining && comp.params_bytes <= main_remaining - headroom) {
                decision.params_location = ParamsLocation::MAIN_GPU;
                decision.params_device   = plan.main_device;
                main_remaining -= comp.params_bytes;
                continue;
            }
            if (comp.params_bytes <= ram_budget_bytes) {
                decision.params_location = ParamsLocation::CPU;
                ram_budget_bytes -= comp.params_bytes;
                continue;
            }

            size_t best = SIZE_MAX;
            for (size_t di = 0; di < devices.size(); ++di) {
                if (di != plan.main_device && comp.params_bytes <= remaining[di] &&
                    (best == SIZE_MAX || remaining[di] > remaining[best])) {
                    best = di;
                }
            }
            if (best != SIZE_MAX) {
                decision.params_location = ParamsLocation::OTHER_GPU;
                decision.params_device   = best;
                remaining[best] -= comp.params_bytes;
            }
        }
        plan.valid = true;
        return plan;
    }

    static std::string params_backend_name(const Decision& decision, const std::vector<Device>& devices) {
        switch (decision.params_location) {
            case ParamsLocation::MAIN_GPU:
            case ParamsLocation::OTHER_GPU:
                return devices[decision.params_device].name;
            case ParamsLocation::CPU:
                return "cpu";
            case ParamsLocation::DISK:
                return "disk";
        }
        return "disk";
    }

    static void print_plan(const Plan& plan,
                           const std::vector<Component>& components,
                           const std::vector<Device>& devices,
                           int64_t free_ram,
                           int64_t ram_budget) {
        LOG_INFO("auto-fit plan (single-GPU compute on %s):", devices[plan.main_device].name.c_str());
        LOG_INFO("  devices:");
        for (const Device& device : devices) {
            LOG_INFO("    %-12s %-32s free %6lld MiB, budget %6lld MiB",
                     device.name.c_str(), device.description.c_str(),
                     (long long)(device.free_bytes / MiB), (long long)(device.budget_bytes / MiB));
        }
        if (free_ram < 0) {
            LOG_WARN("auto-fit: available RAM is unknown; skipping CPU parameter residency");
        } else {
            LOG_INFO("    RAM          free %6lld MiB, params budget %6lld MiB",
                     (long long)(free_ram / MiB), (long long)(ram_budget / MiB));
        }
        LOG_INFO("  main-GPU weight cache priority: diffusion > te > vae");
        LOG_INFO("  components (params: main GPU -> RAM -> other GPU -> disk):");
        for (size_t ci = 0; ci < components.size(); ++ci) {
            const Component& comp = components[ci];
            if (comp.params_bytes == 0) {
                continue;
            }
            const std::string params = params_backend_name(plan.decisions[ci], devices);
            LOG_INFO("    %-12s params %6lld MiB, compute reserve %5lld MiB -> compute %s, params %s",
                     comp.name, (long long)(comp.params_bytes / MiB), (long long)(comp.reserve_bytes / MiB),
                     devices[plan.main_device].name.c_str(), params.c_str());
        }
    }

    static void append_assignment(std::string& spec, const char* key, const std::string& value) {
        if (!spec.empty()) {
            spec += ",";
        }
        spec += key;
        spec += "=";
        spec += value;
    }

    static const char* module_key(ComponentKind kind) {
        switch (kind) {
            case ComponentKind::DIT:
                return "diffusion";
            case ComponentKind::CONDITIONER:
                return "te";
            case ComponentKind::VAE:
                return "vae";
        }
        return "";
    }

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec) {
        std::string error;
        if (!budgets.canonicalize_backend_keys(&error)) {
            LOG_ERROR("%s", error.c_str());
            return false;
        }

        const auto components    = estimate_components(loader, override_wtype);
        const auto devices       = enumerate_gpu_devices(budgets);
        const int64_t free_ram   = available_ram_bytes();
        const int64_t ram_budget = std::max<int64_t>(free_ram - std::max<int64_t>(2048 * MiB, free_ram / 10), 0);
        const auto plan          = compute_plan(components, devices, ram_budget);
        runtime_spec.clear();
        params_spec.clear();
        if (!plan.valid) {
            if (devices.empty()) {
                LOG_WARN("auto-fit: no GPU devices; using the default backend");
            } else {
                LOG_WARN("auto-fit: no GPU memory budget available; using CPU");
                runtime_spec = "cpu";
            }
            return true;
        }

        print_plan(plan, components, devices, free_ram, ram_budget);
        for (size_t ci = 0; ci < components.size(); ++ci) {
            if (components[ci].params_bytes == 0) {
                continue;
            }
            const char* key = module_key(components[ci].kind);
            append_assignment(runtime_spec, key, devices[plan.main_device].name);
            if (plan.decisions[ci].params_location != ParamsLocation::MAIN_GPU) {
                append_assignment(params_spec, key, params_backend_name(plan.decisions[ci], devices));
            }
        }

        // Keep the planner's safety margin when the runner resolves its device limits.
        for (const Device& device : devices) {
            if (device.budget_bytes > 0) {
                budgets.backend_gib[budget_key(device.name)] = (float)(device.budget_bytes / (1024.0 * MiB));
            }
        }
        budgets.resolved_backend_bytes.clear();

        LOG_INFO("auto-fit: --backend \"%s\"%s%s%s",
                 runtime_spec.empty() ? "(default)" : runtime_spec.c_str(),
                 params_spec.empty() ? "" : " --params-backend \"",
                 params_spec.c_str(), params_spec.empty() ? "" : "\"");
        return true;
    }

    bool prepare_vae_decode_retry_tiling(sd_tiling_params_t& tiling_params, bool prefer_temporal_tiling) {
        const char* retry_mode = nullptr;
        if (prefer_temporal_tiling && !tiling_params.temporal_tiling) {
            tiling_params.temporal_tiling = true;
            retry_mode                    = tiling_params.enabled ? "spatial+temporal" : "temporal";
        } else if (!tiling_params.enabled) {
            tiling_params.enabled = true;
            if (tiling_params.tile_size_x <= 0) {
                tiling_params.tile_size_x = 256;
            }
            if (tiling_params.tile_size_y <= 0) {
                tiling_params.tile_size_y = 256;
            }
            retry_mode = tiling_params.temporal_tiling ? "spatial+temporal" : "spatial";
        } else {
            return false;
        }

        LOG_WARN("auto-fit: VAE decode failed (likely out of memory); retrying with %s tiling",
                 retry_mode);
        return true;
    }

}  // namespace sd::backend_fit
