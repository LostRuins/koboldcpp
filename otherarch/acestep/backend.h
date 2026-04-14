#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <thread>

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
};

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
static BackendPair backend_init(const char * label) {
    ggml_backend_load_all();
    BackendPair bp = {};
    bp.backend = ggml_backend_init_best();
    int n_threads = (int)std::thread::hardware_concurrency() / 2;
    if (n_threads < 1) n_threads = 1;
    // [GGML] If best backend is already CPU, reuse it (avoid 2 CPU instances
    // where only one gets the thread count)
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    if (best_is_cpu) {
        bp.cpu_backend = bp.backend;
        ggml_backend_cpu_set_n_threads(bp.backend, n_threads);
    } else {
        bp.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        ggml_backend_cpu_set_n_threads(bp.cpu_backend, n_threads);
    }
    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\n",
            label, ggml_backend_name(bp.backend), n_threads);
    return bp;
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t backends[2] = { bp.backend, bp.cpu_backend };
    int n = (bp.backend == bp.cpu_backend) ? 1 : 2;
    return ggml_backend_sched_new(backends, NULL, n, max_nodes, false, true);
}
