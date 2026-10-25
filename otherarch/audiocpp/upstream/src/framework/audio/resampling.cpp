#include "engine/framework/audio/resampling.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/io/dynamic_library.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::audio {
namespace {

constexpr long double kPi = 3.14159265358979323846264338327950288L;

enum SoxrDataType {
    kSoxrFloat32 = 0,
    kSoxrFloat32Interleaved = kSoxrFloat32,
};

struct SoxrIoSpec {
    SoxrDataType input_type = kSoxrFloat32Interleaved;
    SoxrDataType output_type = kSoxrFloat32Interleaved;
    double scale = 1.0;
    void * reserved = nullptr;
    unsigned long flags = 0;
};

struct SoxrQualitySpec {
    double precision = 0.0;
    double phase_response = 0.0;
    double passband_end = 0.0;
    double stopband_begin = 0.0;
    void * reserved = nullptr;
    unsigned long flags = 0;
};

struct SoxrRuntimeSpec {
    unsigned log2_min_dft_size = 0;
    unsigned log2_large_dft_size = 0;
    unsigned coef_size_kbytes = 0;
    unsigned num_threads = 0;
    void * reserved = nullptr;
    unsigned long flags = 0;
};

class SoxrApi {
public:
    using SoxrError = const char *;
    using OneShotFn = SoxrError (*)(
        double,
        double,
        unsigned,
        const void *,
        size_t,
        size_t *,
        void *,
        size_t,
        size_t *,
        const SoxrIoSpec *,
        const SoxrQualitySpec *,
        const SoxrRuntimeSpec *);
    using IoSpecFn = SoxrIoSpec (*)(SoxrDataType, SoxrDataType);
    using QualitySpecFn = SoxrQualitySpec (*)(unsigned long, unsigned long);
    using RuntimeSpecFn = SoxrRuntimeSpec (*)(unsigned);

    SoxrApi() {
        handle_ = io::open_dynamic_library(
            {"libsoxr.so.0", "libsoxr.so", "libsoxr.dylib", "soxr.dll", "libsoxr.dll",
#if defined(__APPLE__)
             "/opt/homebrew/lib/libsoxr.dylib", "/usr/local/lib/libsoxr.dylib",
#endif
            });
        if (handle_ == nullptr) {
            return;
        }
        oneshot_ = load_symbol<OneShotFn>("soxr_oneshot");
        io_spec_ = load_symbol<IoSpecFn>("soxr_io_spec");
        quality_spec_ = load_symbol<QualitySpecFn>("soxr_quality_spec");
        runtime_spec_ = load_symbol<RuntimeSpecFn>("soxr_runtime_spec");
        if (oneshot_ == nullptr || quality_spec_ == nullptr) {
            io::close_dynamic_library(handle_);
            handle_ = nullptr;
            oneshot_ = nullptr;
            io_spec_ = nullptr;
            quality_spec_ = nullptr;
            runtime_spec_ = nullptr;
        }
    }

    ~SoxrApi() {
        if (handle_ != nullptr) {
            io::close_dynamic_library(handle_);
        }
    }

    SoxrApi(const SoxrApi &) = delete;
    SoxrApi & operator=(const SoxrApi &) = delete;

    bool available() const noexcept {
        return handle_ != nullptr;
    }

    bool supports_profile(SoxrResampleProfile profile) const noexcept {
        return profile == SoxrResampleProfile::QualityOnly ||
               (io_spec_ != nullptr && runtime_spec_ != nullptr);
    }

    OneShotFn oneshot() const noexcept {
        return oneshot_;
    }

    IoSpecFn io_spec() const noexcept {
        return io_spec_;
    }

    QualitySpecFn quality_spec() const noexcept {
        return quality_spec_;
    }

    RuntimeSpecFn runtime_spec() const noexcept {
        return runtime_spec_;
    }

private:
    template <typename Fn>
    Fn load_symbol(const char * name) {
        void * symbol = io::dynamic_library_symbol(handle_, name);
        return reinterpret_cast<Fn>(symbol);
    }

    io::DynamicLibraryHandle handle_ = nullptr;
    OneShotFn oneshot_ = nullptr;
    IoSpecFn io_spec_ = nullptr;
    QualitySpecFn quality_spec_ = nullptr;
    RuntimeSpecFn runtime_spec_ = nullptr;
};

const SoxrApi & get_soxr_api() {
    static const SoxrApi api;
    return api;
}

size_t expected_resample_output_count(
    size_t input_count,
    int source_sample_rate_hz,
    int target_sample_rate_hz) {
    const double exact_output =
        static_cast<double>(input_count) * static_cast<double>(target_sample_rate_hz) /
        static_cast<double>(source_sample_rate_hz);
    return static_cast<size_t>(std::ceil(exact_output));
}

void log_soxr_fallback(const SoxrResampleOptions & options, const std::string & reason) {
    std::string message = options.warning_context;
    message += " libsoxr resampling unavailable; falling back to ";
    message += options.fallback_description;
    message += ": ";
    message += reason;
    debug::log_message(
        debug::LogLevel::Warning,
        "audio.resample.soxr",
        message);
}

struct TorchaudioSincHannResampleKey {
    int source_sample_rate_hz = 0;
    int target_sample_rate_hz = 0;
    int64_t lowpass_filter_width = 0;
    double rolloff = 0.0;
    TorchaudioSincHannKernelMode kernel_mode = TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat32;

    bool operator==(const TorchaudioSincHannResampleKey & other) const noexcept {
        return source_sample_rate_hz == other.source_sample_rate_hz &&
               target_sample_rate_hz == other.target_sample_rate_hz &&
               lowpass_filter_width == other.lowpass_filter_width &&
               rolloff == other.rolloff &&
               kernel_mode == other.kernel_mode;
    }
};

struct TorchaudioSincHannResampleKeyHash {
    size_t operator()(const TorchaudioSincHannResampleKey & key) const noexcept {
        size_t seed = std::hash<int>{}(key.source_sample_rate_hz);
        seed ^= std::hash<int>{}(key.target_sample_rate_hz) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.lowpass_filter_width) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<double>{}(key.rolloff) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(static_cast<int>(key.kernel_mode)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct TorchaudioSincHannResampleKernel {
    int64_t orig_freq = 0;
    int64_t new_freq = 0;
    int64_t width = 0;
    int64_t kernel_size = 0;
    std::vector<double> values;
};

const TorchaudioSincHannResampleKernel & get_cached_torchaudio_sinc_hann_resample_kernel(
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const TorchaudioSincHannResampleOptions & options) {
    if (options.lowpass_filter_width <= 0) {
        throw std::runtime_error("torchaudio Hann resampling requires positive lowpass filter width");
    }
    if (options.rolloff <= 0.0 || options.rolloff > 1.0) {
        throw std::runtime_error("torchaudio Hann resampling requires rolloff in (0, 1]");
    }

    static std::mutex mutex;
    static std::unordered_map<
        TorchaudioSincHannResampleKey,
        TorchaudioSincHannResampleKernel,
        TorchaudioSincHannResampleKeyHash>
        cache;

    const TorchaudioSincHannResampleKey key{
        source_sample_rate_hz,
        target_sample_rate_hz,
        options.lowpass_filter_width,
        options.rolloff,
        options.kernel_mode,
    };
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    const int64_t rate_gcd = std::gcd(source_sample_rate_hz, target_sample_rate_hz);
    const int64_t orig_freq = source_sample_rate_hz / rate_gcd;
    const int64_t new_freq = target_sample_rate_hz / rate_gcd;
    const double base_freq = static_cast<double>(std::min(orig_freq, new_freq)) * options.rolloff;
    const int64_t width = static_cast<int64_t>(
        std::ceil(static_cast<double>(options.lowpass_filter_width * orig_freq) / base_freq));
    const int64_t kernel_size = width * 2 + orig_freq;
    const double scale = base_freq / static_cast<double>(orig_freq);

    TorchaudioSincHannResampleKernel kernel;
    kernel.orig_freq = orig_freq;
    kernel.new_freq = new_freq;
    kernel.width = width;
    kernel.kernel_size = kernel_size;
    kernel.values.assign(static_cast<size_t>(new_freq * kernel_size), 0.0);
    for (int64_t phase = 0; phase < new_freq; ++phase) {
        for (int64_t k = 0; k < kernel_size; ++k) {
            const int64_t idx = -width + k;
            if (options.kernel_mode == TorchaudioSincHannKernelMode::Float32ComputationStoredAsFloat32) {
                const float base_freq_f = static_cast<float>(base_freq);
                const float lowpass_f = static_cast<float>(options.lowpass_filter_width);
                const float pi_f = 3.14159265358979323846F;
                float t = (static_cast<float>(-phase) / static_cast<float>(new_freq) +
                           static_cast<float>(idx) / static_cast<float>(orig_freq)) *
                    base_freq_f;
                t = std::clamp(t, -lowpass_f, lowpass_f);
                const float window = std::cos(t * pi_f / lowpass_f / 2.0F);
                const float angle = t * pi_f;
                const float sinc = angle == 0.0F ? 1.0F : std::sin(angle) / angle;
                kernel.values[static_cast<size_t>(phase * kernel_size + k)] =
                    static_cast<double>(sinc * window * window * static_cast<float>(scale));
                continue;
            }
            double t = static_cast<double>(idx) / static_cast<double>(orig_freq) -
                static_cast<double>(phase) / static_cast<double>(new_freq);
            t *= base_freq;
            t = std::clamp(t, -static_cast<double>(options.lowpass_filter_width), static_cast<double>(options.lowpass_filter_width));
            const double window =
                std::pow(std::cos(t * kPi / static_cast<double>(options.lowpass_filter_width) * 0.5), 2.0);
            const double angle = t * kPi;
            const double sinc = angle == 0.0 ? 1.0 : std::sin(angle) / angle;
            double value = sinc * window * scale;
            if (options.kernel_mode == TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat32) {
                value = static_cast<double>(static_cast<float>(value));
            }
            kernel.values[static_cast<size_t>(phase * kernel_size + k)] = value;
        }
    }

    std::lock_guard<std::mutex> lock(mutex);
    const auto [it, inserted] = cache.emplace(key, std::move(kernel));
    (void) inserted;
    return it->second;
}

}  // namespace

std::optional<std::vector<float>> try_resample_mono_soxr(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const SoxrResampleOptions & options) {
    if (source_sample_rate_hz <= 0 || target_sample_rate_hz <= 0) {
        throw std::runtime_error("soxr resampling requires positive sample rates");
    }
    if (source_sample_rate_hz == target_sample_rate_hz || mono_samples.empty()) {
        return mono_samples;
    }
    const SoxrApi & soxr = get_soxr_api();
    if (!soxr.available()) {
        log_soxr_fallback(options, "library was not found");
        return std::nullopt;
    }
    if (!soxr.supports_profile(options.profile)) {
        log_soxr_fallback(options, "required symbols were not resolved");
        return std::nullopt;
    }

    const size_t expected = expected_resample_output_count(
        mono_samples.size(),
        source_sample_rate_hz,
        target_sample_rate_hz);
    std::vector<float> output(expected + options.output_padding, 0.0F);
    size_t input_done = 0;
    size_t output_done = 0;
    constexpr unsigned long kSoxrHq = 4;
    auto quality_spec = soxr.quality_spec()(kSoxrHq, 0);
    SoxrIoSpec io_spec;
    SoxrRuntimeSpec runtime_spec;
    const SoxrIoSpec * io_spec_ptr = nullptr;
    const SoxrRuntimeSpec * runtime_spec_ptr = nullptr;
    if (options.profile == SoxrResampleProfile::ExplicitFloat32Runtime) {
        io_spec = soxr.io_spec()(kSoxrFloat32Interleaved, kSoxrFloat32Interleaved);
        runtime_spec = soxr.runtime_spec()(1);
        io_spec_ptr = &io_spec;
        runtime_spec_ptr = &runtime_spec;
    }
    const SoxrApi::SoxrError error = soxr.oneshot()(
        static_cast<double>(source_sample_rate_hz),
        static_cast<double>(target_sample_rate_hz),
        1,
        mono_samples.data(),
        mono_samples.size(),
        &input_done,
        output.data(),
        output.size(),
        &output_done,
        io_spec_ptr,
        &quality_spec,
        runtime_spec_ptr);
    if (error != nullptr) {
        log_soxr_fallback(options, error);
        return std::nullopt;
    }
    if (options.require_full_input && input_done != mono_samples.size()) {
        log_soxr_fallback(options, "input was not fully consumed");
        return std::nullopt;
    }
    if (options.reject_empty_output && output_done == 0) {
        log_soxr_fallback(options, "no output samples were produced");
        return std::nullopt;
    }

    output.resize(output_done);
    if (options.output_length_policy == SoxrOutputLengthPolicy::ClampToExpected && output.size() > expected) {
        output.resize(expected);
    } else if (options.output_length_policy == SoxrOutputLengthPolicy::ExactExpected) {
        if (output.size() < expected) {
            output.resize(expected, 0.0F);
        } else if (output.size() > expected) {
            output.resize(expected);
        }
    }
    return output;
}

std::vector<float> resample_mono_soxr_or_linear(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const SoxrResampleOptions & options) {
    if (auto output = try_resample_mono_soxr(
            mono_samples,
            source_sample_rate_hz,
            target_sample_rate_hz,
            options)) {
        return *output;
    }
    return resample_mono_linear(mono_samples, source_sample_rate_hz, target_sample_rate_hz);
}

std::vector<float> resample_mono_linear(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz) {
    if (source_sample_rate_hz <= 0 || target_sample_rate_hz <= 0) {
        throw std::runtime_error("linear mono resampling requires positive sample rates");
    }
    if (source_sample_rate_hz == target_sample_rate_hz || mono_samples.empty()) {
        return mono_samples;
    }
    const double scale = static_cast<double>(target_sample_rate_hz) / static_cast<double>(source_sample_rate_hz);
    const size_t output_samples = static_cast<size_t>(std::llround(static_cast<double>(mono_samples.size()) * scale));
    std::vector<float> output(output_samples, 0.0F);
    for (size_t i = 0; i < output_samples; ++i) {
        const double src_pos = static_cast<double>(i) / scale;
        const size_t left = static_cast<size_t>(std::floor(src_pos));
        const size_t right = std::min(left + 1, mono_samples.size() - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(left));
        output[i] = mono_samples[left] * (1.0F - frac) + mono_samples[right] * frac;
    }
    return output;
}

TorchaudioSincHannResampleOptions torchaudio_sinc_hann_float32_options() {
    TorchaudioSincHannResampleOptions options;
    options.kernel_mode = TorchaudioSincHannKernelMode::Float32ComputationStoredAsFloat32;
    options.accumulation = TorchaudioSincHannAccumulation::Float32;
    return options;
}

std::vector<float> resample_mono_torchaudio_sinc_hann(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const TorchaudioSincHannResampleOptions & options) {
    if (source_sample_rate_hz <= 0 || target_sample_rate_hz <= 0) {
        throw std::runtime_error("torchaudio Hann resampling requires positive sample rates");
    }
    if (source_sample_rate_hz == target_sample_rate_hz || mono_samples.empty()) {
        return mono_samples;
    }

    const auto & kernel = get_cached_torchaudio_sinc_hann_resample_kernel(
        source_sample_rate_hz,
        target_sample_rate_hz,
        options);
    const int64_t input_length = static_cast<int64_t>(mono_samples.size());
    const int64_t target_length = static_cast<int64_t>(
        std::ceil(static_cast<double>(kernel.new_freq * input_length) /
            static_cast<double>(kernel.orig_freq)));
    const int64_t blocks = (target_length + kernel.new_freq - 1) / kernel.new_freq;
    std::vector<int64_t> active_begin(static_cast<size_t>(kernel.new_freq), 0);
    std::vector<int64_t> active_end(static_cast<size_t>(kernel.new_freq), kernel.kernel_size);
    for (int64_t phase = 0; phase < kernel.new_freq; ++phase) {
        const auto * row = kernel.values.data() + static_cast<size_t>(phase * kernel.kernel_size);
        int64_t begin = 0;
        while (begin < kernel.kernel_size && row[begin] == 0.0) {
            ++begin;
        }
        int64_t end = kernel.kernel_size;
        while (end > begin && row[end - 1] == 0.0) {
            --end;
        }
        active_begin[static_cast<size_t>(phase)] = begin;
        active_end[static_cast<size_t>(phase)] = end;
    }
    std::vector<float> out(static_cast<size_t>(target_length), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for if (target_length >= 4096)
#endif
    for (int64_t block = 0; block < blocks; ++block) {
        const int64_t start = block * kernel.orig_freq;
        for (int64_t phase = 0; phase < kernel.new_freq; ++phase) {
            const int64_t out_index = block * kernel.new_freq + phase;
            if (out_index >= target_length) {
                break;
            }
            const int64_t phase_active_begin = active_begin[static_cast<size_t>(phase)];
            const int64_t phase_active_end = active_end[static_cast<size_t>(phase)];
            const auto * values = kernel.values.data() + static_cast<size_t>(phase * kernel.kernel_size + phase_active_begin);
            const int64_t active_size = phase_active_end - phase_active_begin;
            const int64_t input_start = start - kernel.width + phase_active_begin;
            if (input_start >= 0 && input_start + active_size <= input_length) {
                const auto * samples = mono_samples.data() + input_start;
                if (options.accumulation == TorchaudioSincHannAccumulation::Float32) {
                    float sum = 0.0F;
                    for (int64_t k = 0; k < active_size; ++k) {
                        sum += samples[k] * static_cast<float>(values[k]);
                    }
                    out[static_cast<size_t>(out_index)] = sum;
                } else {
                    double sum = 0.0;
                    for (int64_t k = 0; k < active_size; ++k) {
                        sum += static_cast<double>(samples[k]) * values[k];
                    }
                    out[static_cast<size_t>(out_index)] = static_cast<float>(sum);
                }
                continue;
            }
            if (options.accumulation == TorchaudioSincHannAccumulation::Float32) {
                float sum = 0.0F;
                for (int64_t k = 0; k < active_size; ++k) {
                    const int64_t input_index = input_start + k;
                    const float sample =
                        input_index >= 0 && input_index < input_length
                            ? mono_samples[static_cast<size_t>(input_index)]
                            : 0.0F;
                    sum += sample * static_cast<float>(values[k]);
                }
                out[static_cast<size_t>(out_index)] = sum;
            } else {
                double sum = 0.0;
                for (int64_t k = 0; k < active_size; ++k) {
                    const int64_t input_index = input_start + k;
                    const float sample =
                        input_index >= 0 && input_index < input_length
                            ? mono_samples[static_cast<size_t>(input_index)]
                            : 0.0F;
                    sum += static_cast<double>(sample) * values[k];
                }
                out[static_cast<size_t>(out_index)] = static_cast<float>(sum);
            }
        }
    }
    return out;
}

}  // namespace engine::audio
