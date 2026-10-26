#include "engine/models/cosyvoice3/rknn_flow.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/rknn/engine.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::cosyvoice3 {

// 3900 MiB is where this board's driver puts the per-process ceiling, measured by initializing the
// sets in one process: 110+200+250 (3341 MiB), 110x4 (3144), 200x3 (3387), 250+250+110 (3638),
// 110+400 (3401) and 200+400 (3744) all initialize together, while 250+400 (4041), 110+500 (4538)
// and every set holding an S>=900 engine fail at `rknn_init` with -4. The proxy refused 200+400,
// which this board admits; the measured cost reproduces all twelve outcomes. Measured on this board,
// not a board property: the accounting is shared (rknn::ApertureBudget), the numbers are this
// engine's.
constexpr int64_t kAperturePinBudgetMiB = 3900;

namespace {

#if defined(ENGINE_HAVE_RKNN)
using Clock = std::chrono::steady_clock;

// Prefix of every message this sidecar raises, shared with the plumbing in engine::rknn so a
// failure names the stage that failed.
const char * const kLabel = "CosyVoice3 flow estimator RKNN";

// The classifier-free-guidance pair: branch 0 is the conditioned estimate, branch 1 the
// unconditioned one, and the ODE loop mixes them outside the engine.
constexpr int64_t kCfgBatch = 2;

// The aperture accounting of this sidecar, in the driver's own units: each engine's residency is
// read from the runtime (`rknn::engine_memory_mib`, a model-info-only init plus
// `RKNN_QUERY_MEM_SIZE`), which on this board reports 786 MiB at S=110, 1129 at 200, 1426 at 250,
// 2615 at 400, 3752 at 500 and 1039 at 900 -- all of them 661-714 MB files, so a context costs its
// graph and not its image, and the cost is not a function of the compiled length (S=900 costs less
// than S=500). The linear proxy this replaces charged 5 MiB per frame, which over-pins the short
// rungs by up to 11x and under-pins S=500.
constexpr int64_t kApertureMiBPerFrame = 5;  // only if the driver cannot say

// Graph order, as tools/rknn/prepare_cosyvoice3_rknn_flow.py compiles it.
const char * const kInputNames[] = {"x", "mask", "mu", "t", "spks", "cond"};
constexpr uint32_t kInputCount = 6;
constexpr uint32_t kOutputCount = 1;
constexpr uint32_t kInputX = 0;
constexpr uint32_t kInputMask = 1;
constexpr uint32_t kInputMu = 2;
constexpr uint32_t kInputT = 3;
constexpr uint32_t kInputSpks = 4;
constexpr uint32_t kInputCond = 5;

// The sequence length an engine covers, read out of the producer's file name
// (`flow_estimator_seq900.rknn`). Reading it without loading is what keeps the ladder from being
// resident: an engine asks the driver for 0.7-3.3 GB of NPU memory, so a ladder of them does not
// fit, and the length is all the sidecar needs in order to choose.
int64_t name_seq_len(const std::string & path) {
    const std::string stem = std::filesystem::path(path).stem().string();
    const auto at = stem.find("seq");
    if (at == std::string::npos) {
        return -1;
    }
    int64_t value = 0;
    size_t index = at + 3;
    for (; index < stem.size() && stem[index] >= '0' && stem[index] <= '9'; ++index) {
        value = value * 10 + (stem[index] - '0');
    }
    return index == at + 3 ? -1 : value;
}
#endif

}  // namespace

struct CosyVoice3RknnFlow::Impl {
    // One registered engine of the ladder, plus the scratch the padding needs. The scratch is
    // sized on load and reused across calls because the ODE loop runs the same shape ten times.
    struct Slot {
        std::string path;
        int64_t frames = 0;    // the sequence length this engine covers
        int64_t channels = 0;
#if defined(ENGINE_HAVE_RKNN)
        // What this engine will occupy once resident, read from the driver at registration; 0 means
        // the driver could not say, and the linear proxy is used for it.
        int64_t aperture_mib = 0;
        engine::rknn::Engine engine;
        bool loaded = false;
        std::vector<float> x;
        std::vector<float> mask;
        std::vector<float> mu;
        std::vector<float> cond;
        std::vector<float> t;
        std::vector<float> spks;

        void load(int64_t expected_channels) {
            const auto start = Clock::now();
            engine.load(path, kLabel);
            if (engine.input_count() != kInputCount || engine.output_count() != kOutputCount) {
                throw std::runtime_error(
                    std::string(kLabel) + ": expected the 6-input/1-output estimator export, got " +
                    std::to_string(engine.input_count()) + "/" + std::to_string(engine.output_count()));
            }
            for (uint32_t index = 0; index < engine.input_count(); ++index) {
                const rknn_tensor_attr & attr = engine.input_attr(index);
                const std::string name(attr.name);
                debug::timing_log_scalar(
                    "cosyvoice3.flow.dit.rknn.input_attr" + std::to_string(index),
                    name + " type=" + std::to_string(attr.type) + " size=" + std::to_string(attr.size) +
                        " elems=" + std::to_string(attr.n_elems) + " fmt=" + std::to_string(attr.fmt) +
                        " dims=" + std::to_string(attr.n_dims));
                if (name == "x" && attr.n_dims == 3) {
                    if (attr.dims[0] != kCfgBatch) {
                        throw std::runtime_error(
                            std::string(kLabel) + ": " + path + " has batch " + std::to_string(attr.dims[0]) +
                            ", the classifier-free-guidance pair is " + std::to_string(kCfgBatch));
                    }
                    channels = attr.dims[1];
                    if (channels * attr.dims[2] != frames * channels) {
                        throw std::runtime_error(
                            std::string(kLabel) + ": " + path + " covers " + std::to_string(attr.dims[2]) +
                            " frames, its name promises " + std::to_string(frames));
                    }
                }
            }
            if (channels <= 0) {
                throw std::runtime_error(std::string(kLabel) + ": " + path + " is not the estimator export");
            }
            if (channels != expected_channels) {
                throw std::runtime_error(
                    std::string(kLabel) + ": " + path + " carries " + std::to_string(channels) +
                    " mel channels, the checkpoint has " + std::to_string(expected_channels) +
                    "; compile the engines with tools/rknn/prepare_cosyvoice3_rknn_flow.py from this checkpoint");
            }
            const rknn_tensor_attr & out = engine.output_attr(0);
            if (out.n_elems != kCfgBatch * channels * frames) {
                throw std::runtime_error(
                    std::string(kLabel) + ": " + path + " returns " + std::to_string(out.n_elems) +
                    " values, expected " + std::to_string(kCfgBatch * channels * frames));
            }
            // The length-carrying inputs are [2, channels, frames] — the guidance pair is two
            // copies of the plane — while `mask` is [2, 1, frames], two rows of `frames`.
            const size_t plane = static_cast<size_t>(kCfgBatch) * channels * static_cast<size_t>(frames);
            x.resize(plane);
            mu.resize(plane);
            cond.resize(plane);
            mask.resize(static_cast<size_t>(kCfgBatch) * static_cast<size_t>(frames));
            t.resize(kCfgBatch);
            spks.resize(static_cast<size_t>(kCfgBatch) * static_cast<size_t>(channels));
            debug::timing_log_scalar("cosyvoice3.flow.dit.rknn.load_ms", debug::elapsed_ms(start));
            debug::trace_log_scalar("cosyvoice3.flow.dit.rknn.loaded_frames", frames);
            // Only once the engine is validated and its scratch exists: a slot that threw halfway
            // through would otherwise be marked resident, and every later request would be handed
            // an engine with no input buffers instead of the same clear failure.
            loaded = true;
        }

        // One ODE step at the engine's own length. The five host-side tensors arrive at the real
        // length `frames_in` and are widened to `frames` here; the result is cut back by the caller.
        void run(
            const float * x_in,
            const float * mu_in,
            const float * cond_in,
            const float * spks_in,
            float t_in,
            int64_t frames_in,
            std::vector<float> & out) {
            // Row strides, not plane sizes: row `b*channels + ch` starts at row*frames_in on the
            // host and at row*frames in the engine, because each row is one (branch, channel) strip
            // of the frame axis.
            const int64_t rows = kCfgBatch * channels;
            const auto copy_padded = [this, rows, frames_in](
                                         std::vector<float> & dst, const float * src) {
                std::fill(dst.begin(), dst.end(), 0.0F);
                for (int64_t row = 0; row < rows; ++row) {
                    std::memcpy(
                        dst.data() + row * frames,
                        src + row * frames_in,
                        static_cast<size_t>(frames_in) * sizeof(float));
                }
            };
            copy_padded(x, x_in);
            copy_padded(mu, mu_in);
            copy_padded(cond, cond_in);
            // mask: 1 where there is a real frame, 0 in the pad, which is what switches the pad
            // off inside the graph (measured: a padded run is bit-identical to an exact one).
            std::fill(mask.begin(), mask.end(), 0.0F);
            for (int64_t row = 0; row < kCfgBatch; ++row) {
                std::fill(mask.begin() + row * frames, mask.begin() + row * frames + frames_in, 1.0F);
            }
            std::fill(t.begin(), t.end(), t_in);
            std::memcpy(spks.data(), spks_in, spks.size() * sizeof(float));

            std::vector<engine::rknn::Input> inputs(kInputCount);
            for (uint32_t index = 0; index < inputs.size(); ++index) {
                inputs[index].index = index;
                inputs[index].type = RKNN_TENSOR_FLOAT32;
            }
            // With pass_through = 0 the runtime wants the width of the declared type, not
            // attr.size, which reports the engine's native fp16 width.
            inputs[kInputX].buf = x.data();
            inputs[kInputX].size = static_cast<uint32_t>(x.size() * sizeof(float));
            inputs[kInputMask].buf = mask.data();
            inputs[kInputMask].size = static_cast<uint32_t>(mask.size() * sizeof(float));
            inputs[kInputMu].buf = mu.data();
            inputs[kInputMu].size = static_cast<uint32_t>(mu.size() * sizeof(float));
            inputs[kInputT].buf = t.data();
            inputs[kInputT].size = static_cast<uint32_t>(t.size() * sizeof(float));
            inputs[kInputSpks].buf = spks.data();
            inputs[kInputSpks].size = static_cast<uint32_t>(spks.size() * sizeof(float));
            inputs[kInputCond].buf = cond.data();
            inputs[kInputCond].size = static_cast<uint32_t>(cond.size() * sizeof(float));
            std::vector<std::vector<float>> outputs;
            engine.run(inputs, outputs);
            out.swap(outputs[0]);
        }
#endif
    };

    // Held by pointer: Engine owns a rknn_context and is neither copyable nor movable, so the
    // ladder has to be a vector of slots, not of slots by value.
    std::vector<std::unique_ptr<Slot>> slots;
    int64_t channels = 0;
#if defined(ENGINE_HAVE_RKNN)
    // Resident-set accounting for the ladder, in the driver's own units (see kAperturePinBudgetMiB).
    engine::rknn::ApertureBudget budget_{kAperturePinBudgetMiB};
#endif

#if defined(ENGINE_HAVE_RKNN)
    void register_ladder(const std::vector<std::string> & engine_paths, int64_t expected_channels) {
        channels = expected_channels;
        for (const auto & path : engine_paths) {
            if (path.empty()) {
                continue;
            }
            const int64_t frames = name_seq_len(path);
            if (frames <= 0) {
                throw std::runtime_error(
                    std::string(kLabel) + ": " + path +
                    " does not name the length it covers; the producer names engines "
                    "flow_estimator_seq<frames>.rknn and the length is read from the name so a ladder "
                    "can be chosen without loading all of it");
            }
            // Built in place: Slot owns an Engine, so it is neither copyable nor movable.
            auto slot = std::make_unique<Slot>();
            slot->path = path;
            slot->frames = frames;
            slot->channels = expected_channels;
            // Ask the driver what this engine will cost before deciding to keep it: a model-info-only
            // init parses the image without allocating the network (~0.5 s per engine, once at
            // startup), which is what makes the per-engine table a measurement instead of a guess.
            slot->aperture_mib = engine::rknn::engine_memory_mib(path, kLabel);
            debug::trace_log_scalar(
                "cosyvoice3.flow.dit.rknn.aperture_mib", slot->aperture_mib);
            slots.push_back(std::move(slot));
        }
        std::sort(slots.begin(), slots.end(), [](const auto & a, const auto & b) {
            return a->frames < b->frames;
        });
        if (slots.empty()) {
            throw std::runtime_error(std::string(kLabel) + ": no engine paths given");
        }
        // The board states its own ceiling; 0 means pin every rung the deployment declares.
        const int64_t budget_mib = aperture_budget_mib();
        budget_ = engine::rknn::ApertureBudget(
            budget_mib > 0 ? budget_mib : std::numeric_limits<int64_t>::max());
        debug::trace_log_scalar("cosyvoice3.flow.dit.rknn.aperture_budget_mib", budget_mib);
    }

    // The smallest engine that fits `frames`. The long-sequence engines are loaded on demand and
    // evicted on the next length change, because the driver allocates the whole engine (weights and
    // workspace) per context and the long rungs need gigabytes. The short rungs stay resident: they
    // fit together in the NPU aperture (see kAperturePinBudgetMiB), they are the rungs the served
    // requests ask for, and each of them costs ~9 s of `rknn_init` per utterance when it has to be
    // re-loaded.
    Slot & pick(int64_t frames) {
        for (auto & holder : slots) {
            Slot & slot = *holder;
            if (slot.frames < frames) {
                continue;
            }
            if (!slot.loaded) {
                // The rung is kept (and the new one pinned) only while the whole resident set stays
                // inside the aperture; a rung above the budget can never share it with anything, so
                // it evicts everything and is itself evicted by the next length change.
                const int64_t cost_mib = slot.aperture_mib > 0
                    ? slot.aperture_mib
                    : slot.frames * kApertureMiBPerFrame;
                const bool fits = budget_.fits(cost_mib);
                if (!fits) {
                    for (auto & other : slots) {
                        if (other->loaded) {
                            other->engine.release();
                            other->loaded = false;
                        }
                    }
                    budget_.clear();
                }
                slot.load(channels);
                budget_.reserve(cost_mib);
                debug::trace_log_scalar(
                    "cosyvoice3.flow.dit.rknn.resident_mib", budget_.resident());
            }
            return slot;
        }
        throw std::runtime_error(
            std::string(kLabel) + ": no engine covers " + std::to_string(frames) + " mel frames, the largest is " +
            std::to_string(slots.back()->frames));
    }
#endif
};

CosyVoice3RknnFlow::CosyVoice3RknnFlow(
    const std::vector<std::string> & engine_paths,
    int64_t mel_channels)
    : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_HAVE_RKNN)
    impl_->register_ladder(engine_paths, mel_channels);
#else
    (void)engine_paths;
    (void)mel_channels;
    throw std::runtime_error("CosyVoice3 flow estimator RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

CosyVoice3RknnFlow::~CosyVoice3RknnFlow() = default;

int64_t CosyVoice3RknnFlow::max_frames() const {
    return impl_->slots.empty() ? 0 : impl_->slots.back()->frames;
}

int64_t CosyVoice3RknnFlow::engine_count() const {
    return static_cast<int64_t>(impl_->slots.size());
}

int64_t CosyVoice3RknnFlow::mel_channels() const {
    return impl_->channels;
}

std::vector<float> CosyVoice3RknnFlow::run(
    const float * x,
    const float * mu,
    const float * cond,
    const float * spks,
    float t,
    int64_t frames) {
#if defined(ENGINE_HAVE_RKNN)
    const auto start = Clock::now();
    auto & slot = impl_->pick(frames);
    std::vector<float> padded;
    slot.run(x, mu, cond, spks, t, frames, padded);
    // Cut the pad back out: [2, channels, frames] out of [2, channels, engine_frames]. Same row
    // strides as the padding above: `frames` on the host, `engine_frames` in the engine.
    std::vector<float> out(static_cast<size_t>(kCfgBatch * slot.channels * frames), 0.0F);
    for (int64_t row = 0; row < kCfgBatch * slot.channels; ++row) {
        std::memcpy(
            out.data() + row * frames,
            padded.data() + row * slot.frames,
            static_cast<size_t>(frames) * sizeof(float));
    }
    debug::timing_log_scalar("cosyvoice3.flow.dit.rknn.step_ms", debug::elapsed_ms(start));
    debug::trace_log_scalar("cosyvoice3.flow.dit.rknn.engine_frames", slot.frames);
    return out;
#else
    (void)x;
    (void)mu;
    (void)cond;
    (void)spks;
    (void)t;
    (void)frames;
    throw std::runtime_error("CosyVoice3 flow estimator RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

// The ceiling the ladder evicts to, read once per call: a board states its own aperture with
// `AUDIOCPP_RKNN_APERTURE_MIB` in the spirit of `AUDIOCPP_RKNN_CORE_MASK`, and 0 (also the reading of
// a malformed value, so a typo cannot silently mean one megabyte) turns the accounting off and pins
// every rung the deployment declares.
int64_t aperture_budget_mib() {
    const char * const env = std::getenv("AUDIOCPP_RKNN_APERTURE_MIB");
    if (env == nullptr) {
        return kAperturePinBudgetMiB;
    }
    char * end = nullptr;
    const long long value = std::strtoll(env, &end, 10);
    return value > 0 && *end == '\0' ? static_cast<int64_t>(value) : 0;
}

}  // namespace engine::models::cosyvoice3
