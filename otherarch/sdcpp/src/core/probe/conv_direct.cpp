#include "core/probe/conv_direct.h"

#include <cmath>
#include <exception>
#include <utility>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/probe/backend_probe.h"
#include "core/util.h"

static void build_conv_direct_probe(BackendProbeGraph& graph, bool conv3d, bool direct, int size = 32, int channels = 64) {
    if (conv3d) {
        auto input   = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, size / 2, size / 2, 4, channels);
        auto weight  = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F16, 3, 3, 3, channels * channels);
        graph.inputs = {input, weight};
        graph.output = direct ? ggml_conv_3d_direct(graph.ctx, weight, input, 1, 1, 1, 1, 1, 1, 1, 1, 1, channels, 1, channels)
                              : ggml_conv_3d(graph.ctx, weight, input, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    } else {
        auto input   = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, size, size, channels, 1);
        auto weight  = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        graph.inputs = {input, weight};
        graph.output = direct ? ggml_conv_2d_direct(graph.ctx, weight, input, 1, 1, 1, 1, 1, 1)
                              : ggml_conv_2d(graph.ctx, weight, input, 1, 1, 1, 1, 1, 1);
    }
}

static bool initialize_conv_probe(BackendProbeGraph& graph, bool conv3d) {
    uint32_t state = 42;
    for (auto* tensor : graph.inputs) {
        const size_t count = static_cast<size_t>(ggml_nelements(tensor));
        const float scale  = tensor->type == GGML_TYPE_F16 ? 1.f / std::sqrt(float(graph.inputs[0]->ne[conv3d ? 3 : 2] * (conv3d ? 27 : 9))) : 1.f;
        std::vector<ggml_fp16_t> values(count);
        for (auto& value : values) {
            state = state * 1664525u + 1013904223u;
            value = ggml_fp32_to_fp16((static_cast<float>(state >> 8) / 16777216.f - 0.5f) * scale);
        }
        if (tensor->type == GGML_TYPE_F16) {
            ggml_backend_tensor_set(tensor, values.data(), 0, count * sizeof(ggml_fp16_t));
        } else {
            // im2col rounds inputs to F16; use the same input precision for both paths.
            std::vector<float> input(count);
            ggml_fp16_to_fp32_row(values.data(), input.data(), count);
            ggml_backend_tensor_set(tensor, input.data(), 0, count * sizeof(float));
        }
    }
    return true;
}

bool sd_probe_conv_direct(ggml_backend_t backend, bool conv3d) {
    if (backend == nullptr) {
        return false;
    }
    const char* name = ggml_backend_name(backend);
    const int dims   = conv3d ? 3 : 2;
    try {
        BackendProbeGraph capability(backend);
        if (capability.ctx == nullptr) {
            LOG_INFO("conv%dd direct auto: %s -> off (probe graph unavailable)", dims, name);
            return false;
        }
        build_conv_direct_probe(capability, conv3d, true);
        if (!capability.supported()) {
            LOG_INFO("conv%dd direct auto: %s -> off (unsupported probe graph)", dims, name);
            return false;
        }
        if (sd_backend_is(backend, "Vulkan")) {
            std::vector<BackendProbeCase> cases;
            for (const auto& shape : {std::pair<int, int>{32, 64}, {16, 128}}) {
                BackendProbeCase test;
                test.build = [conv3d, shape](BackendProbeGraph& graph, bool direct) {
                    build_conv_direct_probe(graph, conv3d, direct, shape.first, shape.second);
                };
                test.initialize = [conv3d](BackendProbeGraph& graph) { return initialize_conv_probe(graph, conv3d); };
                cases.push_back(std::move(test));
            }
            const auto result = sd_run_backend_probe(backend, 0, cases);
            if (!result.completed) {
                LOG_WARN("conv%dd direct auto: %s -> off (%s)", dims, name, result.error.c_str());
                return false;
            }
            LOG_INFO("conv%dd direct auto: %s -> %s (probe: im2col %.3f ms, direct %.3f ms)",
                     dims, name, result.candidate_faster ? "on" : "off", result.reference_ms, result.candidate_ms);
            return result.candidate_faster;
        }
        if (sd_backend_supports_conv_implicit_gemm(backend, conv3d)) {
            LOG_INFO("conv%dd direct auto: %s -> on (implicit GEMM)", dims, name);
            return true;
        }
        LOG_INFO("conv%dd direct auto: %s -> off (no implicit GEMM path)", dims, name);
        return false;
    } catch (const std::exception& error) {
        LOG_WARN("conv%dd direct auto: %s -> off (probe failed: %s)", dims, name, error.what());
        return false;
    }
}
