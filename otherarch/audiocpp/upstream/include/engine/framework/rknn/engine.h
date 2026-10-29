#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#if defined(ENGINE_HAVE_RKNN)
#include <rknn_api.h>
#endif

namespace engine::rknn {

// Shared plumbing for the model-local Rockchip NPU sidecars (docs/build/rockchip-npu.md).
// A sidecar is not a ggml backend: RKNN has no runtime graph builder, so each
// graph-isolated stage is compiled offline for one fixed shape and voiced through
// `librknnrt` by a module in the model's own directory. What every one of those modules
// needs from the driver is the same, and it lives here: read the engine image, init the
// context, pick the NPU cores, read the tensor attributes, and run one step. What stays
// model-local is the tensor geometry — which named input is the latent, how the window
// walk slices the source, how the seams are stitched.

// Reads a compiled `.rknn` image. `label` prefixes every message so a failure names the
// stage that failed ("Qwen3 speech decoder RKNN: ...").
std::vector<char> read_engine(const std::string & path, const std::string & label);

// Formats one `RKNN_*` return code the same way every sidecar does. Where the driver has answered in
// this process, the message names it (`... failed with code -6 (driver 0.9.8, api 2.3.2 ...)`),
// because the code alone says what was refused and not who refused it: a `.rknn` is compiled per SoC
// and per SDK generation, and the runtime library and the kernel driver are versioned apart (a
// working RK3588 reports `api 2.3.2` beside `driver 0.9.8`), with the toolkit that built the image
// as the third version an operator has to line up. See `driver_report()`.
std::string status(const std::string & label, const char * what, int code);

// What the driver said it is, as one phrase, once any context in this process has reported it;
// empty where no driver can answer (a build without an RKNN runtime, an engine that never
// initialized). Asked of the first context the driver builds -- `rknn_query(RKNN_QUERY_SDK_VERSION)`
// is per-context -- and cached, so a later failure carries it at no round trip.
std::string driver_report();

// Aperture accounting for a sidecar that keeps more than one compiled engine resident: the resident
// set is the sum of the residency each loaded engine reports, and a new engine is pinned only while
// the whole set stays inside the budget; a unit that cannot share the aperture with the set evicts it
// and is itself evicted by the next change of shape. What an engine costs and what the board fits are
// both measurements of one engine on one board, so both are passed in by the model: this type is the
// accounting, not the policy, and it holds no ceiling of its own. It counts MiB and nothing else --
// the model derives that number from the driver's own report (`engine_memory_mib`) or from a proxy it
// owns, and the accounting stays free of any assumption about how a cost is estimated.
class ApertureBudget {
public:
    explicit ApertureBudget(int64_t budget_mib) : budget_mib_(budget_mib) {}

    // Would adding an engine of `mib` resident megabytes keep the set inside the aperture?
    bool fits(int64_t mib) const { return resident_mib_ + mib <= budget_mib_; }
    void reserve(int64_t mib) { resident_mib_ += mib; }
    void clear() { resident_mib_ = 0; }
    int64_t resident() const { return resident_mib_; }

private:
    int64_t resident_mib_ = 0;
    int64_t budget_mib_ = 0;
};

// Resolves one `exec_plan` artifact path against the directory the model was loaded from. Every
// sidecar takes its engine as a path the operator types, and the convention everywhere else in the
// runtime is that a model's files are named relative to the model (a `ResourceBundle` reference, a
// `miotts` codec, a `chatterbox` sidecar). A relative name is therefore joined onto `model_root`
// when that file exists; an absolute path, an empty root, and a name that is not under the root are
// returned unchanged, so a failed load quotes exactly what was asked for.
std::string resolve_engine_path(const std::string & artifact, const std::filesystem::path & model_root);

// What a compiled engine will occupy once it is resident, in MiB, read from the driver itself:
// `rknn_init` with `RKNN_FLAG_COLLECT_MODEL_INFO_ONLY` parses the image without building the network
// (~0.5 s per engine on RK3588, no NPU memory allocated) and `rknn_query(RKNN_QUERY_MEM_SIZE)` then
// reports `total_weight_size + total_internal_size` -- the bytes the real init goes on to allocate,
// less a per-context slack of 30-50 MiB. Returns 0 when the driver cannot say, so a caller can keep
// a proxy for that case; a build without an RKNN runtime is that case for every engine, so the
// declaration is not conditional and a consumer (a sidecar, or the server's load guard) needs no
// preprocessor test of its own to ask.
int64_t engine_memory_mib(const std::string & path, const std::string & label);

#if defined(ENGINE_HAVE_RKNN)

// One graph input for `Engine::run`. `size` is the size of the buffer in bytes: with
// `pass_through = 0` the runtime wants the width of the type it is handed, not
// `rknn_tensor_attr.size`, which reports the engine's native (often fp16) width.
struct Input {
    uint32_t index = 0;
    const void * buf = nullptr;
    uint32_t size = 0;
    rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
    uint32_t pass_through = 0;
};

// One compiled engine: a `rknn_context` plus its input/output counts.
class Engine {
public:
    Engine() = default;
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    ~Engine();

    // Loads `path` and asks for all three NPU cores by name: RKNN_NPU_CORE_AUTO lets the
    // driver pick, and for these graphs it picks one core. `AUDIOCPP_RKNN_CORE_MASK`
    // (1/2/4 = core 0/1/2, 7 = all) overrides it, for measuring whether an fp16 engine is
    // core-invariant.
    void load(const std::string & path, const std::string & label);
    void release();
    bool loaded() const { return ctx_ != 0; }

    const std::string & label() const { return label_; }
    uint32_t input_count() const { return n_input_; }
    uint32_t output_count() const { return n_output_; }
    // Attributes are read once at load and cached, so the model can validate names and
    // shapes without a driver round trip per call.
    const rknn_tensor_attr & input_attr(uint32_t index) const;
    const rknn_tensor_attr & output_attr(uint32_t index) const;

    // One step: sets every input, runs, and copies out every output as fp32.
    void run(const std::vector<Input> & inputs, std::vector<std::vector<float>> & outputs);

private:
    std::string label_;
    rknn_context ctx_ = 0;
    std::vector<rknn_tensor_attr> inputs_;
    std::vector<rknn_tensor_attr> outputs_;
    uint32_t n_input_ = 0;
    uint32_t n_output_ = 0;
};

#endif  // ENGINE_HAVE_RKNN

}  // namespace engine::rknn
