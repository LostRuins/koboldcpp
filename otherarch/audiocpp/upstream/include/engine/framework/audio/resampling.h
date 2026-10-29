#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace engine::audio {

enum class SoxrResampleProfile {
    QualityOnly,
    ExplicitFloat32Runtime,
};

enum class SoxrOutputLengthPolicy {
    ActualOutput,
    ClampToExpected,
    ExactExpected,
};

struct SoxrResampleOptions {
    SoxrResampleProfile profile = SoxrResampleProfile::QualityOnly;
    SoxrOutputLengthPolicy output_length_policy = SoxrOutputLengthPolicy::ActualOutput;
    size_t output_padding = 0;
    bool require_full_input = false;
    bool reject_empty_output = false;
    const char * warning_context = "audio";
    const char * fallback_description = "fallback resampling";
};

std::optional<std::vector<float>> try_resample_mono_soxr(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const SoxrResampleOptions & options);

std::vector<float> resample_mono_soxr_or_linear(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const SoxrResampleOptions & options);

std::vector<float> resample_mono_linear(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz);

enum class TorchaudioSincHannKernelMode {
    Float64ComputationStoredAsFloat32,
    Float32ComputationStoredAsFloat32,
    Float64ComputationStoredAsFloat64,
};

enum class TorchaudioSincHannAccumulation {
    Float32,
    Float64,
};

struct TorchaudioSincHannResampleOptions {
    int64_t lowpass_filter_width = 6;
    double rolloff = 0.99;
    TorchaudioSincHannKernelMode kernel_mode = TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat32;
    TorchaudioSincHannAccumulation accumulation = TorchaudioSincHannAccumulation::Float64;
};

TorchaudioSincHannResampleOptions torchaudio_sinc_hann_float32_options();

std::vector<float> resample_mono_torchaudio_sinc_hann(
    const std::vector<float> & mono_samples,
    int source_sample_rate_hz,
    int target_sample_rate_hz,
    const TorchaudioSincHannResampleOptions & options = {});

}  // namespace engine::audio
