#include "engine/framework/rknn/engine.h"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::rknn {

std::vector<char> read_engine(const std::string & path, const std::string & label) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error(label + ": cannot open model " + path);
    }
    const auto size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error(label + ": model " + path + " is empty");
    }
    std::vector<char> data(static_cast<size_t>(size));
    in.seekg(0);
    if (!in.read(data.data(), size)) {
        throw std::runtime_error(label + ": short read on " + path);
    }
    return data;
}

std::string status(const std::string & label, const char * what, int code) {
    std::ostringstream out;
    out << label << ": " << what << " failed with code " << code;
    // The versions are the diagnosis a `RKNN_ERR_*` code invites: -13 says the image targets a
    // platform this SoC is not, -6 and -4 say the driver refused, and none of them says who refused:
    // the runtime library and the kernel driver are versioned apart, and the toolkit that built the
    // image is a third version an operator has to line up against them.
    const auto report = driver_report();
    if (!report.empty()) {
        out << " (" << report << ")";
    }
    return out.str();
}

std::string resolve_engine_path(const std::string & artifact, const std::filesystem::path & model_root) {
    if (artifact.empty() || model_root.empty()) {
        return artifact;
    }
    const std::filesystem::path here(artifact);
    if (here.is_absolute()) {
        return artifact;
    }
    std::error_code ec;
    const auto joined = model_root / here;
    return std::filesystem::exists(joined, ec) ? joined.string() : artifact;
}

#if defined(ENGINE_HAVE_RKNN)
namespace {

// The driver's self-report, cached the first time any context answers: `RKNN_QUERY_SDK_VERSION` is
// per-context, so it is taken from a context that exists and kept for the ones that fail to build.
std::mutex g_report_mutex;
std::string g_driver_report;

void note_driver_versions(rknn_context ctx) {
    rknn_sdk_version sdk{};
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk, sizeof(sdk)) != 0) {
        return;
    }
    const std::string api = sdk.api_version;
    const std::string driver = sdk.drv_version;
    if (api.empty() && driver.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_report_mutex);
    g_driver_report = "driver " + driver + ", api " + api;
}

}  // namespace

std::string driver_report() {
    std::lock_guard<std::mutex> lock(g_report_mutex);
    return g_driver_report;
}

int64_t engine_memory_mib(const std::string & path, const std::string & label) {
    (void)label;
    std::vector<char> model;
    try {
        model = read_engine(path, label);
    } catch (const std::exception &) {
        return 0;
    }
    rknn_context ctx = 0;
    if (rknn_init(
            &ctx, model.data(), static_cast<uint32_t>(model.size()),
            RKNN_FLAG_COLLECT_MODEL_INFO_ONLY, nullptr) != 0) {
        return 0;
    }
    rknn_mem_size mem{};
    const int ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem, sizeof(mem));
    note_driver_versions(ctx);
    rknn_destroy(ctx);
    if (ret != 0) {
        return 0;
    }
    // `rknn_mem_size` carries the two byte counts as 32-bit, so widen before adding: a 1 GiB weight
    // set beside a 3.5 GiB workspace is 4.5 GiB, and a 32-bit sum reports 512 MiB.
    const uint64_t bytes = static_cast<uint64_t>(mem.total_weight_size) + mem.total_internal_size;
    // Round up: the budget is a ceiling, and a resident set measured at its sum should not be
    // admitted by fractions of a megabyte.
    return static_cast<int64_t>((bytes + (1ull << 20) - 1) >> 20);
}

Engine::~Engine() {
    release();
}

void Engine::load(const std::string & path, const std::string & label) {
    label_ = label;
    auto model = read_engine(path, label_);
    if (const int ret = rknn_init(
            &ctx_, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
        ret != 0) {
        ctx_ = 0;
        throw std::runtime_error(status(label_, "rknn_init", ret));
    }
    note_driver_versions(ctx_);
    if (const char * mask = std::getenv("AUDIOCPP_RKNN_CORE_MASK"); mask != nullptr) {
        if (rknn_set_core_mask(ctx_, static_cast<rknn_core_mask>(std::atoi(mask))) != 0) {
            rknn_set_core_mask(ctx_, RKNN_NPU_CORE_AUTO);
        }
    } else if (rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0_1_2) != 0) {
        rknn_set_core_mask(ctx_, RKNN_NPU_CORE_AUTO);
    }

    rknn_input_output_num io_num{};
    if (const int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        ret != 0) {
        throw std::runtime_error(status(label_, "rknn_query(io_num)", ret));
    }
    n_input_ = io_num.n_input;
    n_output_ = io_num.n_output;
    inputs_.resize(n_input_);
    for (uint32_t index = 0; index < n_input_; ++index) {
        inputs_[index] = {};
        inputs_[index].index = index;
        if (const int ret =
                rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inputs_[index], sizeof(rknn_tensor_attr));
            ret != 0) {
            throw std::runtime_error(status(label_, "rknn_query(input_attr)", ret));
        }
    }
    outputs_.resize(n_output_);
    for (uint32_t index = 0; index < n_output_; ++index) {
        outputs_[index] = {};
        outputs_[index].index = index;
        if (const int ret = rknn_query(
                ctx_, RKNN_QUERY_OUTPUT_ATTR, &outputs_[index], sizeof(rknn_tensor_attr));
            ret != 0) {
            throw std::runtime_error(status(label_, "rknn_query(output_attr)", ret));
        }
    }
}

void Engine::release() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    inputs_.clear();
    outputs_.clear();
    n_input_ = 0;
    n_output_ = 0;
}

const rknn_tensor_attr & Engine::input_attr(uint32_t index) const {
    if (index >= n_input_) {
        throw std::runtime_error(status(label_, "input index out of range", static_cast<int>(index)));
    }
    return inputs_[index];
}

const rknn_tensor_attr & Engine::output_attr(uint32_t index) const {
    if (index >= n_output_) {
        throw std::runtime_error(
            status(label_, "output index out of range", static_cast<int>(index)));
    }
    return outputs_[index];
}

void Engine::run(const std::vector<Input> & inputs, std::vector<std::vector<float>> & outputs) {
    if (ctx_ == 0) {
        throw std::runtime_error(label_ + ": engine not loaded");
    }
    std::vector<rknn_input> driver_inputs(inputs.size());
    for (size_t index = 0; index < inputs.size(); ++index) {
        driver_inputs[index] = {};
        driver_inputs[index].index = inputs[index].index;
        driver_inputs[index].buf = const_cast<void *>(inputs[index].buf);
        driver_inputs[index].size = inputs[index].size;
        driver_inputs[index].type = inputs[index].type;
        driver_inputs[index].fmt = RKNN_TENSOR_UNDEFINED;
        driver_inputs[index].pass_through = inputs[index].pass_through;
    }
    if (const int ret =
            rknn_inputs_set(ctx_, static_cast<uint32_t>(driver_inputs.size()), driver_inputs.data());
        ret != 0) {
        throw std::runtime_error(status(label_, "rknn_inputs_set", ret));
    }
    if (const int ret = rknn_run(ctx_, nullptr); ret != 0) {
        throw std::runtime_error(status(label_, "rknn_run", ret));
    }
    std::vector<rknn_output> driver_outputs(n_output_);
    for (uint32_t index = 0; index < n_output_; ++index) {
        driver_outputs[index] = {};
        driver_outputs[index].index = index;
        driver_outputs[index].want_float = 1;
        driver_outputs[index].is_prealloc = 0;
    }
    if (const int ret = rknn_outputs_get(ctx_, n_output_, driver_outputs.data(), nullptr);
        ret != 0) {
        throw std::runtime_error(status(label_, "rknn_outputs_get", ret));
    }
    outputs.clear();
    outputs.resize(n_output_);
    for (uint32_t index = 0; index < n_output_; ++index) {
        const auto * begin = static_cast<const float *>(driver_outputs[index].buf);
        const auto count = driver_outputs[index].size / sizeof(float);
        outputs[index].assign(begin, begin + count);
    }
    rknn_outputs_release(ctx_, n_output_, driver_outputs.data());
}

#else

std::string driver_report() {
    // Nothing asked, nothing answered: `status()` keeps the exact shape it always had.
    return {};
}

int64_t engine_memory_mib(const std::string & /* path */, const std::string & /* label */) {
    // No driver to ask, so every engine reports "unknown" -- the answer a caller's proxy is written
    // to fall back on.
    return 0;
}

#endif  // ENGINE_HAVE_RKNN

}  // namespace engine::rknn
