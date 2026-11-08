#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/silero_vad/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::models::silero_vad {

struct SileroBackendWeights;

struct SileroSpeechTimestamp {
    int64_t start = 0;
    int64_t end = 0;
};

struct SileroVADConfig {
    float threshold = 0.5f;
    int min_speech_duration_ms = 250;
    int min_silence_duration_ms = 100;
    int speech_pad_ms = 30;
    float max_speech_duration_s = 1.0e9f;
    float neg_threshold = -1.0f;
    int min_silence_at_max_speech_ms = 98;
    bool use_max_poss_sil_at_max_speech = true;
};

class SileroRuntime {
public:
    SileroRuntime(
        std::shared_ptr<const SileroWeights> weights,
        core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type);
    ~SileroRuntime();

    void prepare(int sample_rate);
    void reset(int64_t batch_size = 1);
    runtime::TaskResult run_offline(const runtime::AudioBuffer & audio, const SileroVADConfig & config);
    runtime::StreamEvent process_chunk(const runtime::AudioChunk & chunk, const SileroVADConfig & config);
    runtime::TaskResult finalize_stream(const SileroVADConfig & config) const;

private:
    struct ChunkGraph;
    struct ChunkResult {
        std::vector<float> probability;
    };

    ChunkResult infer_chunk(const std::vector<float> & chunk, int64_t batch, int sample_rate);
    void reset_states(int64_t batch_size = 1);
    void ensure_state(int64_t batch_size, int sample_rate);
    void ensure_chunk_graph(int64_t batch_size);
    std::vector<float> audio_forward(
        const std::vector<float> & waveform,
        int64_t batch,
        int64_t samples,
        int sample_rate);
    std::vector<SileroSpeechTimestamp> decode_speech_timestamps(
        const std::vector<float> & probabilities,
        int64_t samples,
        int sample_rate,
        const SileroVADConfig & config) const;
    runtime::TaskResult build_offline_result(const std::vector<float> & waveform, int sample_rate, const SileroVADConfig & config);
    runtime::StreamEvent build_stream_event(float probability, const SileroVADConfig & config, int64_t chunk_start_sample);

    core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<const SileroWeights> weights_;
    std::shared_ptr<const SileroBackendWeights> backend_weights_;
    std::unique_ptr<ChunkGraph> chunk_graph_;
    std::vector<float> hidden_;
    std::vector<float> cell_;
    std::vector<float> context_;
    int64_t last_batch_size_ = 0;
    int last_sample_rate_ = 0;
    bool triggered_ = false;
    int64_t temp_end_ = 0;
    int64_t current_sample_ = 0;
    int64_t current_speech_start_ = 0;
    std::vector<SileroSpeechTimestamp> streamed_segments_;
};

}  // namespace engine::models::silero_vad
