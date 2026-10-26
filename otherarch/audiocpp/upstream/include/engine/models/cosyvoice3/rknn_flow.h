#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::cosyvoice3 {

// Rockchip NPU (RKNN) backend for the flow-matching DiT estimator: the ODE step that the ggml
// DiT graph in flow.cpp computes. The estimator is the flow decoder's velocity field, and it is
// the stage the reference RKNN port keeps on the NPU (the AR LLM and the HiFT vocoder stay on
// CPU there, for the same reasons this repository keeps its sidecars on the decoder half).
//
// RKNN has no dynamic shapes, so each engine is compiled offline for one fixed sequence length
// (tools/rknn/prepare_cosyvoice3_rknn_flow.py) and a ladder of them is registered here. An utterance of
// `frames` mel frames runs on the smallest engine that fits it: `x`/`mu`/`cond` zero-padded on
// the right, `mask` 1 on the real frames and 0 in the pad, output cropped back. Measured on the
// host, a padded run is bit-identical to an exact run at the real length (355 dB, max_abs 0.0),
// so the ladder is a time trade, not a quality one: the estimator costs ~S^2, and every frame of
// pad is paid for ten times, once per ODE step.
//
// The engine takes the raw timestep `t` (its time-embedding MLP is inside the graph) and a frame
// `mask`; the ggml graph takes the timestep embedding and has no mask. Both are supplied here
// from the same ODE loop, so the two backends see the same step.
class CosyVoice3RknnFlow {
public:
    // Registers the ladder without loading any of it: one engine asks the driver for 0.7-3.3 GB
    // of NPU memory (measured, seq200 to seq1100), so a ladder cannot be resident at once. The
    // length an engine covers is read from its file name (`flow_estimator_seq900.rknn`, the
    // producer's convention) and validated against the engine's own attributes when it loads.
    // `mel_channels` is the checkpoint's, checked the same way.
    CosyVoice3RknnFlow(const std::vector<std::string> & engine_paths, int64_t mel_channels);
    ~CosyVoice3RknnFlow();

    CosyVoice3RknnFlow(const CosyVoice3RknnFlow &) = delete;
    CosyVoice3RknnFlow & operator=(const CosyVoice3RknnFlow &) = delete;

    // Longest utterance the ladder covers, in mel frames. A longer one has no engine and has to
    // run on the ggml graph, which is why the caller keeps it.
    int64_t max_frames() const;
    int64_t engine_count() const;
    // Mel channels the ladder was built for.
    int64_t mel_channels() const;

    // One ODE step. `x`, `mu` and `cond` are channel-major [2, mel_channels, frames] and `spks`
    // is [2, mel_channels], with the classifier-free-guidance uncond branch zeroed in `mu`,
    // `cond` and `spks` exactly as the ggml path builds it. `t` is the raw timestep. Returns
    // dphi_dt as [2, mel_channels, frames]: the cond branch first, then uncond.
    std::vector<float> run(
        const float * x,
        const float * mu,
        const float * cond,
        const float * spks,
        float t,
        int64_t frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The resident-set ceiling this sidecar enforces on its ladder, in MiB: the value
// `AUDIOCPP_RKNN_APERTURE_MIB` sets, else this board's measured ceiling, and 0 when the accounting
// is off and every declared rung is pinned. A host consumer (the server's load guard) asks for it
// through the loader's resource plan, which is where a family's residency policy belongs.
int64_t aperture_budget_mib();

}  // namespace engine::models::cosyvoice3
