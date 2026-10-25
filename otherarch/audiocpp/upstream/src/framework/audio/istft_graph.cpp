#include "engine/framework/audio/istft_graph.h"

#include "engine/framework/audio/fft.h"
#ifdef ENGINE_HAS_CUDA_ISTFT
#include "istft_cuda_runtime.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace engine::audio {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Workspace {
    int64_t frames = 0;
    int64_t freq_bins = 0;
    int64_t n_fft = 0;
    int64_t output_size = 0;
    std::vector<std::complex<float>> spectrum;
    std::vector<float> framed;
    std::vector<float> folded;
    std::vector<float> envelope;
    std::vector<float> audio;
    std::vector<float> window;
};

struct WorkspaceConfig {
    int64_t frames = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
};

void ensure_workspace(
    Workspace & workspace,
    const WorkspaceConfig & config,
    int64_t freq_bins,
    const std::vector<float> & window,
    const char * label) {
    const int64_t pad = (config.n_fft - config.hop_length) / 2;
    const int64_t output_size = (config.frames - 1) * config.hop_length + config.n_fft;
    const int64_t samples = output_size - 2 * pad;
    require(samples > 0, label);
    if (workspace.frames == config.frames &&
        workspace.freq_bins == freq_bins &&
        workspace.n_fft == config.n_fft &&
        workspace.output_size == output_size &&
        workspace.window == window) {
        return;
    }

    workspace.frames = config.frames;
    workspace.freq_bins = freq_bins;
    workspace.n_fft = config.n_fft;
    workspace.output_size = output_size;
    workspace.window = window;
    workspace.spectrum.resize(static_cast<size_t>(config.frames * freq_bins));
    workspace.framed.resize(static_cast<size_t>(config.frames * config.n_fft));
    workspace.folded.resize(static_cast<size_t>(output_size));
    workspace.envelope.assign(static_cast<size_t>(output_size), 0.0F);
    workspace.audio.resize(static_cast<size_t>(samples));
    for (int64_t frame = 0; frame < config.frames; ++frame) {
        const int64_t start = frame * config.hop_length;
        for (int64_t i = 0; i < config.n_fft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            workspace.envelope[static_cast<size_t>(start + i)] += w * w;
        }
    }
}

template <typename Timing>
void finish_istft_from_spectrum(
    Workspace & workspace,
    const WorkspaceConfig & config,
    int64_t freq_bins,
    const std::vector<float> & window,
    size_t threads,
    Timing & timing,
    const char * label) {
    ensure_workspace(workspace, config, freq_bins, window, label);

    auto timing_start = Clock::now();
    std::fill(workspace.framed.begin(), workspace.framed.end(), 0.0F);
    timing.framed_clear_ms = elapsed_ms(timing_start, Clock::now());

    timing_start = Clock::now();
    real_fft_inverse(
        {static_cast<size_t>(config.frames), static_cast<size_t>(config.n_fft)},
        {
            static_cast<std::ptrdiff_t>(freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
        },
        {
            static_cast<std::ptrdiff_t>(config.n_fft * static_cast<int64_t>(sizeof(float))),
            static_cast<std::ptrdiff_t>(sizeof(float)),
        },
        1,
        workspace.spectrum.data(),
        workspace.framed.data(),
        1.0F / static_cast<float>(config.n_fft),
        threads);
    timing.fft_inverse_ms = elapsed_ms(timing_start, Clock::now());

    timing_start = Clock::now();
    std::fill(workspace.folded.begin(), workspace.folded.end(), 0.0F);
    timing.fold_clear_ms = elapsed_ms(timing_start, Clock::now());

    timing_start = Clock::now();
    for (int64_t frame = 0; frame < config.frames; ++frame) {
        const int64_t start = frame * config.hop_length;
        const float * src = workspace.framed.data() + static_cast<size_t>(frame * config.n_fft);
        for (int64_t i = 0; i < config.n_fft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            workspace.folded[static_cast<size_t>(start + i)] += src[i] * w;
        }
    }
    timing.overlap_add_ms = elapsed_ms(timing_start, Clock::now());

    const int64_t pad = (config.n_fft - config.hop_length) / 2;
    const int64_t samples = workspace.output_size - 2 * pad;
    timing_start = Clock::now();
    for (int64_t i = 0; i < samples; ++i) {
        const int64_t src = i + pad;
        const float denom = workspace.envelope[static_cast<size_t>(src)];
        if (denom <= 1.0e-11F) {
            throw std::runtime_error("ISTFT window envelope underflow");
        }
        workspace.audio[static_cast<size_t>(i)] = workspace.folded[static_cast<size_t>(src)] / denom;
    }
    timing.normalize_ms = elapsed_ms(timing_start, Clock::now());
}

}  // namespace

class HostLogMagnitudePhaseISTFT::Impl {
public:
    explicit Impl(const HostLogMagnitudePhaseISTFTConfig & config)
        : config_(config),
          freq_bins_(config.n_fft / 2 + 1) {
        require(config_.frames > 0, "host ISTFT requires positive frames");
        require(config_.n_fft > 0, "host ISTFT requires positive n_fft");
        require(config_.hop_length > 0, "host ISTFT requires positive hop length");
        require(config_.out_dim == freq_bins_ * 2, "host ISTFT expects log-magnitude and phase halves");
    }

    HostLogMagnitudePhaseISTFTResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window) {
        require(
            static_cast<int64_t>(log_magnitude_phase.size()) == config_.frames * config_.out_dim,
            "host ISTFT input size mismatch");
        require(
            static_cast<int64_t>(window.size()) == config_.n_fft,
            "host ISTFT window size mismatch");

        HostLogMagnitudePhaseISTFTResult result;
        auto & timing = result.timing;
        const auto total_start = Clock::now();
        const WorkspaceConfig workspace_config{config_.frames, config_.n_fft, config_.hop_length};
        ensure_workspace(workspace_, workspace_config, freq_bins_, window, "host ISTFT produced non-positive samples");

        auto timing_start = Clock::now();
#ifdef _OPENMP
        const int omp_threads = static_cast<int>(std::max<size_t>(1, config_.threads));
#pragma omp parallel for num_threads(omp_threads) if (config_.frames >= 8)
#endif
        for (int64_t frame = 0; frame < config_.frames; ++frame) {
            const float * row = log_magnitude_phase.data() + static_cast<size_t>(frame * config_.out_dim);
            auto * spectrum_row = workspace_.spectrum.data() + static_cast<size_t>(frame * freq_bins_);
            for (int64_t freq = 0; freq < freq_bins_; ++freq) {
                const float mag = std::min(std::exp(row[freq]), 100.0F);
                const float phase = row[freq_bins_ + freq];
                spectrum_row[static_cast<size_t>(freq)] = {
                    mag * std::cos(phase),
                    mag * std::sin(phase),
                };
            }
        }
        timing.spectrum_ms = elapsed_ms(timing_start, Clock::now());

        finish_istft_from_spectrum(
            workspace_,
            workspace_config,
            freq_bins_,
            window,
            config_.threads,
            timing,
            "host ISTFT produced non-positive samples");
        result.audio = workspace_.audio;
        timing.total_ms = elapsed_ms(total_start, Clock::now());
        return result;
    }

private:
    HostLogMagnitudePhaseISTFTConfig config_;
    int64_t freq_bins_ = 0;
    Workspace workspace_;
};

class CudaLogMagnitudePhaseISTFT::Impl {
public:
    explicit Impl(const CudaLogMagnitudePhaseISTFTConfig & config)
        : config_(config) {
        require(config_.frames > 0, "CUDA ISTFT requires positive frames");
        require(config_.n_fft > 0, "CUDA ISTFT requires positive n_fft");
        require(config_.hop_length > 0, "CUDA ISTFT requires positive hop length");
        require(config_.out_dim == (config_.n_fft / 2 + 1) * 2, "CUDA ISTFT expects log-magnitude and phase halves");
#ifdef ENGINE_HAS_CUDA_ISTFT
        runtime_ = std::make_unique<detail::CudaIstftRuntime>(
            detail::CudaIstftRuntimeConfig{
                config_.frames,
                config_.n_fft,
                config_.hop_length,
                config_.out_dim,
                config_.device,
            });
#else
        throw std::runtime_error("CUDA ISTFT runtime was not built");
#endif
    }

    CudaLogMagnitudePhaseISTFTResult compute(
        const std::vector<float> & log_magnitude_phase,
        const std::vector<float> & window) {
        require(
            static_cast<int64_t>(log_magnitude_phase.size()) == config_.frames * config_.out_dim,
            "CUDA ISTFT input size mismatch");
        require(
            static_cast<int64_t>(window.size()) == config_.n_fft,
            "CUDA ISTFT window size mismatch");

#ifdef ENGINE_HAS_CUDA_ISTFT
        auto runtime_result = runtime_->compute(log_magnitude_phase, window);
        CudaLogMagnitudePhaseISTFTResult result;
        result.audio = std::move(runtime_result.audio);
        result.timing.input_upload_ms = runtime_result.timing.input_upload_ms;
        result.timing.spectrum_kernel_ms = runtime_result.timing.spectrum_kernel_ms;
        result.timing.fft_inverse_ms = runtime_result.timing.fft_inverse_ms;
        result.timing.overlap_add_ms = runtime_result.timing.overlap_add_ms;
        result.timing.normalize_ms = runtime_result.timing.normalize_ms;
        result.timing.audio_read_ms = runtime_result.timing.audio_read_ms;
        result.timing.total_ms = runtime_result.timing.total_ms;
        return result;
#else
        throw std::runtime_error("CUDA ISTFT runtime was not built");
#endif
    }

private:
    CudaLogMagnitudePhaseISTFTConfig config_;
#ifdef ENGINE_HAS_CUDA_ISTFT
    std::unique_ptr<detail::CudaIstftRuntime> runtime_;
#endif
};

HostLogMagnitudePhaseISTFT::HostLogMagnitudePhaseISTFT(const HostLogMagnitudePhaseISTFTConfig & config)
    : impl_(std::make_unique<Impl>(config)) {}

HostLogMagnitudePhaseISTFT::~HostLogMagnitudePhaseISTFT() = default;

HostLogMagnitudePhaseISTFT::HostLogMagnitudePhaseISTFT(HostLogMagnitudePhaseISTFT &&) noexcept = default;

HostLogMagnitudePhaseISTFT & HostLogMagnitudePhaseISTFT::operator=(
    HostLogMagnitudePhaseISTFT &&) noexcept = default;

HostLogMagnitudePhaseISTFTResult HostLogMagnitudePhaseISTFT::compute(
    const std::vector<float> & log_magnitude_phase,
    const std::vector<float> & window) {
    return impl_->compute(log_magnitude_phase, window);
}

CudaLogMagnitudePhaseISTFT::CudaLogMagnitudePhaseISTFT(
    const CudaLogMagnitudePhaseISTFTConfig & config)
    : impl_(std::make_unique<Impl>(config)) {}

CudaLogMagnitudePhaseISTFT::~CudaLogMagnitudePhaseISTFT() = default;

CudaLogMagnitudePhaseISTFT::CudaLogMagnitudePhaseISTFT(CudaLogMagnitudePhaseISTFT &&) noexcept = default;

CudaLogMagnitudePhaseISTFT & CudaLogMagnitudePhaseISTFT::operator=(
    CudaLogMagnitudePhaseISTFT &&) noexcept = default;

CudaLogMagnitudePhaseISTFTResult CudaLogMagnitudePhaseISTFT::compute(
    const std::vector<float> & log_magnitude_phase,
    const std::vector<float> & window) {
    return impl_->compute(log_magnitude_phase, window);
}

}  // namespace engine::audio
