#ifndef __SD_CORE_PROBE_BACKEND_PROBE_H__
#define __SD_CORE_PROBE_BACKEND_PROBE_H__

#include <functional>
#include <string>
#include <vector>

#include "ggml-backend.h"

struct BackendProbeOptions {
    size_t graph_size            = 128;
    int warmup_runs              = 1;
    int measurement_runs         = 5;
    double time_budget_ms        = 1000.;
    size_t min_free_device_bytes = 128ULL * 1024 * 1024;
    double min_improvement       = 0.05;
    float absolute_tolerance     = 0.001f;
    float relative_tolerance     = 0.02f;
};

class BackendProbeGraph {
    ggml_backend_t backend_;
    size_t graph_size_;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph* graph_           = nullptr;

public:
    ggml_context* ctx = nullptr;
    std::vector<ggml_tensor*> inputs;
    ggml_tensor* output = nullptr;

    explicit BackendProbeGraph(ggml_backend_t backend, size_t graph_size = 128);
    ~BackendProbeGraph();
    BackendProbeGraph(const BackendProbeGraph&)            = delete;
    BackendProbeGraph& operator=(const BackendProbeGraph&) = delete;

    bool supported();
    bool allocate();
    bool initialize_f32_inputs();
    double run();
};

struct BackendProbeCase {
    // Variants must be repeatable without mutating inputs; leave output null if unavailable.
    std::function<void(BackendProbeGraph&, bool candidate)> build;
    // Optional overrides for non-F32 inputs/outputs or operation-specific data.
    std::function<bool(BackendProbeGraph&)> initialize;
    std::function<bool(const BackendProbeGraph&, const BackendProbeGraph&)> compare;
};

struct BackendProbeResult {
    bool completed        = false;
    bool candidate_faster = false;
    double reference_ms   = 0.;
    double candidate_ms   = 0.;
    std::string error;
};

BackendProbeResult sd_run_backend_probe(ggml_backend_t backend,
                                        int n_threads,
                                        const std::vector<BackendProbeCase>& cases,
                                        const BackendProbeOptions& options = {});

#endif  // __SD_CORE_PROBE_BACKEND_PROBE_H__
