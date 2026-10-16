#include "core/probe/flash_attn.h"

#include <array>
#include <cstring>
#include <exception>
#include <sstream>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"
#include "core/probe/backend_probe.h"
#include "core/util.h"

static bool is_vulkan_nvidia_rtx(ggml_backend_t backend) {
    auto device = ggml_backend_get_device(backend);
    if (device == nullptr) {
        return false;
    }
    auto reg = ggml_backend_dev_backend_reg(device);
    if (reg == nullptr || std::strcmp(ggml_backend_reg_name(reg), "Vulkan") != 0) {
        return false;
    }
    const char* description = ggml_backend_dev_description(device);
    if (description == nullptr) {
        return false;
    }
    std::istringstream stream(description);
    std::string token;
    bool nvidia = false, rtx = false;
    while (stream >> token) {
        nvidia = nvidia || token == "NVIDIA";
        rtx    = rtx || token == "RTX";
    }
    return nvidia && rtx;
}

static void build_flash_attention_probe(BackendProbeGraph& graph, ggml_backend_t backend, int length, int head_size, bool flash) {
    constexpr int heads = 4;
    auto q              = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F32, head_size * heads, length, 1);
    auto k              = ggml_dup_tensor(graph.ctx, q);
    auto v              = ggml_dup_tensor(graph.ctx, q);
    graph.inputs        = {q, k, v};
    bool used_flash     = false;
    auto output         = ggml_ext_attention_ext(graph.ctx, backend, q, k, v, heads, nullptr, false, flash, 1.f, false, &used_flash);
    graph.output        = flash && !used_flash ? nullptr : output;
}

bool sd_probe_flash_attention(ggml_backend_t backend, int n_threads) {
    if (backend == nullptr) {
        return false;
    }
    const char* name = ggml_backend_name(backend);
    try {
        BackendProbeGraph capability(backend);
        if (capability.ctx == nullptr) {
            LOG_INFO("flash attention auto: %s -> off (probe graph unavailable)", name);
            return false;
        }
        build_flash_attention_probe(capability, backend, 512, 64, true);
        if (!capability.supported()) {
            LOG_INFO("flash attention auto: %s -> off (unsupported probe graph)", name);
            return false;
        }
        if (sd_backend_is_cpu(backend)) {
            LOG_INFO("flash attention auto: %s -> on (CPU)", name);
            return true;
        }
        if (sd_backend_supports_cuda_mma(backend)) {
            LOG_INFO("flash attention auto: %s -> on (CUDA MMA)", name);
            return true;
        }
        if (sd_backend_supports_hip_mma(backend)) {
            LOG_INFO("flash attention auto: %s -> on (HIP MMA)", name);
            return true;
        }
        if (is_vulkan_nvidia_rtx(backend)) {
            LOG_INFO("flash attention auto: %s -> on (Vulkan NVIDIA RTX)", name);
            return true;
        }
        std::vector<BackendProbeCase> cases;
        for (const auto& shape : {std::array<int, 2>{512, 64}, std::array<int, 2>{1024, 128}}) {
            BackendProbeCase test;
            test.build = [backend, shape](BackendProbeGraph& graph, bool candidate) {
                build_flash_attention_probe(graph, backend, shape[0], shape[1], candidate);
            };
            cases.push_back(std::move(test));
        }
        const auto result = sd_run_backend_probe(backend, n_threads, cases);
        if (!result.completed) {
            LOG_WARN("flash attention auto: %s -> off (%s)", name, result.error.c_str());
            return false;
        }
        LOG_INFO("flash attention auto: %s -> %s (probe: normal %.3f ms, flash %.3f ms)",
                 name, result.candidate_faster ? "on" : "off", result.reference_ms, result.candidate_ms);
        return result.candidate_faster;
    } catch (const std::exception& error) {
        LOG_WARN("flash attention auto: %s -> off (probe failed: %s)", name, error.what());
        return false;
    }
}
