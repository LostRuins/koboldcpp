#include "core/probe/backend_probe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "ggml-alloc.h"

BackendProbeGraph::BackendProbeGraph(ggml_backend_t backend, size_t graph_size)
    : backend_(backend), graph_size_(graph_size) {
    ctx = ggml_init({graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true});
}

BackendProbeGraph::~BackendProbeGraph() {
    ggml_backend_buffer_free(buffer_);
    ggml_free(ctx);
}

bool BackendProbeGraph::supported() {
    if (backend_ == nullptr || ctx == nullptr || output == nullptr) {
        return false;
    }
    if (graph_ == nullptr) {
        graph_ = ggml_new_graph_custom(ctx, graph_size_, false);
        ggml_build_forward_expand(graph_, output);
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
        if (!ggml_backend_supports_op(backend_, ggml_graph_node(graph_, i))) {
            return false;
        }
    }
    return true;
}

bool BackendProbeGraph::allocate() {
    if (ctx == nullptr || backend_ == nullptr) {
        return false;
    }
    if (buffer_ == nullptr) {
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx, backend_);
    }
    return buffer_ != nullptr;
}

bool BackendProbeGraph::initialize_f32_inputs() {
    uint32_t state = 42;
    for (auto* tensor : inputs) {
        if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
            return false;
        }
        std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
        for (float& value : values) {
            state = state * 1664525u + 1013904223u;
            value = static_cast<float>(state >> 8) / 16777216.f - 0.5f;
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }
    return true;
}

double BackendProbeGraph::run() {
    if (graph_ == nullptr || buffer_ == nullptr) {
        return -1.;
    }
    ggml_backend_synchronize(backend_);
    const auto start  = std::chrono::steady_clock::now();
    const auto status = ggml_backend_graph_compute(backend_, graph_);
    ggml_backend_synchronize(backend_);
    if (status != GGML_STATUS_SUCCESS) {
        return -1.;
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static bool backend_probe_outputs_match(const BackendProbeGraph& reference,
                                        const BackendProbeGraph& candidate,
                                        const BackendProbeOptions& options) {
    if (!ggml_are_same_shape(reference.output, candidate.output) ||
        reference.output->type != GGML_TYPE_F32 || candidate.output->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(reference.output) || !ggml_is_contiguous(candidate.output)) {
        return false;
    }
    const size_t count = static_cast<size_t>(ggml_nelements(reference.output));
    std::vector<float> expected(count), actual(count);
    ggml_backend_tensor_get(reference.output, expected.data(), 0, count * sizeof(float));
    ggml_backend_tensor_get(candidate.output, actual.data(), 0, count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
            std::abs(expected[i] - actual[i]) > options.absolute_tolerance + options.relative_tolerance * std::abs(expected[i])) {
            return false;
        }
    }
    return true;
}

BackendProbeResult sd_run_backend_probe(ggml_backend_t backend,
                                        int n_threads,
                                        const std::vector<BackendProbeCase>& cases,
                                        const BackendProbeOptions& options) {
    BackendProbeResult result;
    if (backend == nullptr || cases.empty() || options.graph_size == 0 || options.warmup_runs < 1 || options.measurement_runs < 1 ||
        !std::isfinite(options.time_budget_ms) || options.time_budget_ms < 0. ||
        !std::isfinite(options.min_improvement) || options.min_improvement < 0. || options.min_improvement >= 1. ||
        !std::isfinite(options.absolute_tolerance) || options.absolute_tolerance < 0.f ||
        !std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0.f) {
        result.error = "invalid probe configuration";
        return result;
    }
    try {
        if (sd_backend_is_cpu(backend)) {
            sd_backend_cpu_set_n_threads(backend, n_threads > 0 ? n_threads : sd_get_num_physical_cores());
        }
        auto dev = ggml_backend_get_device(backend);
        if (dev != nullptr && !sd_backend_is_cpu(backend)) {
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            if (free_bytes > total_bytes || free_bytes < options.min_free_device_bytes) {
                result.error = "insufficient probe memory";
                return result;
            }
        }
        double timed_ms = 0.;
        for (const auto& test : cases) {
            BackendProbeGraph reference(backend, options.graph_size), candidate(backend, options.graph_size);
            if (!test.build || reference.ctx == nullptr || candidate.ctx == nullptr) {
                result.error = "probe graph unavailable";
                return result;
            }
            test.build(reference, false);
            test.build(candidate, true);
            if (!reference.supported() || !candidate.supported()) {
                result.error = "unsupported probe graph";
                return result;
            }
            if (!reference.allocate() || !candidate.allocate()) {
                result.error = "probe allocation failed";
                return result;
            }
            auto initialize = [&](BackendProbeGraph& graph) {
                return test.initialize ? test.initialize(graph) : graph.initialize_f32_inputs();
            };
            if (!initialize(reference) || !initialize(candidate)) {
                result.error = "probe input initialization failed";
                return result;
            }
            for (int i = 0; i < options.warmup_runs; ++i) {
                if (reference.run() < 0. || candidate.run() < 0.) {
                    result.error = "probe warmup failed";
                    return result;
                }
            }
            if (!(test.compare ? test.compare(reference, candidate) : backend_probe_outputs_match(reference, candidate, options))) {
                result.error = "probe output check failed";
                return result;
            }
            std::vector<double> reference_times(options.measurement_runs), candidate_times(options.measurement_runs);
            for (int i = 0; i < options.measurement_runs; ++i) {
                if (timed_ms >= options.time_budget_ms) {
                    result.error = "probe time budget exhausted";
                    return result;
                }
                // Alternate order so clock ramp-up does not consistently favor one path.
                if (i % 2 == 0) {
                    reference_times[i] = reference.run();
                    candidate_times[i] = candidate.run();
                } else {
                    candidate_times[i] = candidate.run();
                    reference_times[i] = reference.run();
                }
                if (reference_times[i] < 0. || candidate_times[i] < 0.) {
                    result.error = "probe execution failed";
                    return result;
                }
                timed_ms += reference_times[i] + candidate_times[i];
            }
            auto median = [](std::vector<double>& values) {
                std::sort(values.begin(), values.end());
                const size_t mid = values.size() / 2;
                return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
            };
            result.reference_ms += median(reference_times);
            result.candidate_ms += median(candidate_times);
        }
        result.completed        = true;
        result.candidate_faster = result.candidate_ms < result.reference_ms * (1. - options.min_improvement);
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}
