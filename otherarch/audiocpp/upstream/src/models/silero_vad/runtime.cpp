#include "engine/models/silero_vad/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::silero_vad {

struct SileroBackendWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue stft_conv_weight;
    core::TensorValue conv1_weight;
    core::TensorValue conv1_bias;
    core::TensorValue conv2_weight;
    core::TensorValue conv2_bias;
    core::TensorValue conv3_weight;
    core::TensorValue conv3_bias;
    core::TensorValue conv4_weight;
    core::TensorValue conv4_bias;
    core::TensorValue lstm_weight_ih;
    core::TensorValue lstm_weight_hh;
    core::TensorValue lstm_bias_ih;
    core::TensorValue lstm_bias_hh;
    core::TensorValue final_conv_weight;
    core::TensorValue final_conv_bias;
};

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kSampleRate = 16000;
constexpr int64_t kChunkSamples = 512;
constexpr int64_t kContextSamples = 64;
constexpr int64_t kInputSamples = kChunkSamples + kContextSamples;
constexpr int64_t kCutoff = 129;
constexpr int64_t kHiddenSize = 128;

core::TensorValue load_f32_bias(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const char * name,
    std::initializer_list<int64_t> shape) {
    return store.load_f32_tensor(source, name, shape);
}

std::shared_ptr<const SileroBackendWeights> load_backend_weights(
    const SileroWeights & weights,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type) {
    auto out = std::make_shared<SileroBackendWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "silero_vad.weights", 16ull * 1024ull * 1024ull);
    auto & store = *out->store;
    if (weights.source == nullptr) {
        throw std::runtime_error("Silero VAD weights require a tensor source");
    }
    const auto & source = *weights.source;
    out->stft_conv_weight = store.load_tensor(source, "stft_conv.weight", storage_type, {258, 1, 256});
    out->conv1_weight = store.load_tensor(source, "conv1.weight", storage_type, {128, 129, 3});
    out->conv1_bias = load_f32_bias(store, source, "conv1.bias", {128});
    out->conv2_weight = store.load_tensor(source, "conv2.weight", storage_type, {64, 128, 3});
    out->conv2_bias = load_f32_bias(store, source, "conv2.bias", {64});
    out->conv3_weight = store.load_tensor(source, "conv3.weight", storage_type, {64, 64, 3});
    out->conv3_bias = load_f32_bias(store, source, "conv3.bias", {64});
    out->conv4_weight = store.load_tensor(source, "conv4.weight", storage_type, {128, 64, 3});
    out->conv4_bias = load_f32_bias(store, source, "conv4.bias", {128});
    out->lstm_weight_ih = store.load_tensor(source, "lstm_cell.weight_ih", storage_type, {4 * kHiddenSize, kHiddenSize});
    out->lstm_weight_hh = store.load_tensor(source, "lstm_cell.weight_hh", storage_type, {4 * kHiddenSize, kHiddenSize});
    out->lstm_bias_ih = load_f32_bias(store, source, "lstm_cell.bias_ih", {4 * kHiddenSize});
    out->lstm_bias_hh = load_f32_bias(store, source, "lstm_cell.bias_hh", {4 * kHiddenSize});
    out->final_conv_weight = store.load_tensor(source, "final_conv.weight", storage_type, {1, kHiddenSize, 1});
    out->final_conv_bias = load_f32_bias(store, source, "final_conv.bias", {1});
    store.upload();
    weights.source->release_storage();
    return out;
}

std::shared_ptr<const SileroWeights> require_weights(std::shared_ptr<const SileroWeights> weights) {
    if (!weights) {
        throw std::runtime_error("Silero VAD runtime requires weights");
    }
    return weights;
}

std::vector<float> require_mono_audio(const runtime::AudioBuffer & audio, const char * context) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error(std::string(context) + " requires a positive sample rate");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error(std::string(context) + " requires a positive channel count");
    }
    if (audio.channels == 1) {
        return audio.samples;
    }
    return engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
}

}  // namespace

struct SileroRuntime::ChunkGraph {
    ggml_context * ggml = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    int64_t batch = 0;

    core::TensorValue input;
    core::TensorValue hidden_in;
    core::TensorValue cell_in;

    core::TensorValue stft_magnitude;
    core::TensorValue conv1;
    core::TensorValue conv2;
    core::TensorValue conv3;
    core::TensorValue conv4;
    core::TensorValue hidden_out;
    core::TensorValue cell_out;
    core::TensorValue probability;

    ~ChunkGraph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
            ggml = nullptr;
        }
    }
};

SileroRuntime::SileroRuntime(
    std::shared_ptr<const SileroWeights> weights,
    core::ExecutionContext & execution_context,
    assets::TensorStorageType weight_storage_type)
    : execution_context_(&execution_context),
      weights_(require_weights(std::move(weights))),
      backend_weights_(load_backend_weights(*weights_, execution_context.backend(), execution_context.backend_type(), weight_storage_type)) {
    reset(1);
}

SileroRuntime::~SileroRuntime() = default;

void SileroRuntime::prepare(int sample_rate) {
    const auto start = Clock::now();
    ensure_state(1, sample_rate);
    ensure_chunk_graph(1);
    engine::debug::timing_log_scalar("silero_vad.prepare.total_ms", engine::debug::elapsed_ms(start));
}

void SileroRuntime::reset(int64_t batch_size) {
    reset_states(batch_size);
    triggered_ = false;
    temp_end_ = 0;
    current_sample_ = 0;
    current_speech_start_ = 0;
    streamed_segments_.clear();
}

void SileroRuntime::reset_states(int64_t batch_size) {
    hidden_.assign(static_cast<size_t>(batch_size * kHiddenSize), 0.0f);
    cell_.assign(static_cast<size_t>(batch_size * kHiddenSize), 0.0f);
    context_.assign(static_cast<size_t>(batch_size * kContextSamples), 0.0f);
    last_batch_size_ = batch_size;
    last_sample_rate_ = 0;
}

void SileroRuntime::ensure_state(int64_t batch_size, int sample_rate) {
    if (sample_rate != kSampleRate) {
        throw std::runtime_error("Silero VAD 16k model only supports sample_rate=16000");
    }
    if (last_batch_size_ != batch_size || last_sample_rate_ != sample_rate || hidden_.empty()) {
        reset_states(batch_size);
    }
    last_sample_rate_ = sample_rate;
}

void SileroRuntime::ensure_chunk_graph(int64_t batch_size) {
    if (batch_size <= 0) {
        throw std::runtime_error("Silero VAD chunk graph requires a positive batch size");
    }
    if (chunk_graph_ != nullptr && chunk_graph_->batch == batch_size) {
        engine::debug::timing_log_scalar("silero_vad.chunk.graph.reused", true);
        return;
    }

    const auto build_start = Clock::now();
    auto graph = std::make_unique<ChunkGraph>();
    graph->batch = batch_size;

    ggml_init_params params = {};
    params.mem_size = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Silero VAD ggml context");
    }

    core::ModuleBuildContext ctx = {};
    ctx.ggml = graph->ggml;
    ctx.module_instance_name = "silero_vad";

    graph->input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size, 1, kInputSamples}));
    graph->hidden_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size, kHiddenSize}));
    graph->cell_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size, kHiddenSize}));

    const auto & weights = *backend_weights_;

    core::TensorValue padded = modules::ReflectPad1dModule({0, kContextSamples}).build(ctx, graph->input);
    const auto stft_conv = modules::Conv1dModule({1, kCutoff * 2, 256, 128, 0, 1, false}).build(
        ctx,
        padded,
        {weights.stft_conv_weight, std::nullopt});
    const auto stft_real = modules::SliceModule({1, 0, kCutoff}).build(ctx, stft_conv);
    const auto stft_imag = modules::SliceModule({1, kCutoff, kCutoff}).build(ctx, stft_conv);
    const auto stft_real_sq = modules::MulModule().build(ctx, stft_real, stft_real);
    const auto stft_imag_sq = modules::MulModule().build(ctx, stft_imag, stft_imag);
    graph->stft_magnitude = modules::SqrtModule().build(ctx, modules::AddModule().build(ctx, stft_real_sq, stft_imag_sq));

    graph->conv1 = modules::ReluModule().build(
        ctx,
        modules::Conv1dModule({129, 128, 3, 1, 1, 1, true}).build(
            ctx,
            graph->stft_magnitude,
            {weights.conv1_weight, weights.conv1_bias}));
    graph->conv2 = modules::ReluModule().build(
        ctx,
        modules::Conv1dModule({128, 64, 3, 2, 1, 1, true}).build(
            ctx,
            graph->conv1,
            {weights.conv2_weight, weights.conv2_bias}));
    graph->conv3 = modules::ReluModule().build(
        ctx,
        modules::Conv1dModule({64, 64, 3, 2, 1, 1, true}).build(
            ctx,
            graph->conv2,
            {weights.conv3_weight, weights.conv3_bias}));
    const auto conv4_bct = modules::ReluModule().build(
        ctx,
        modules::Conv1dModule({64, 128, 3, 1, 1, 1, true}).build(
            ctx,
            graph->conv3,
            {weights.conv4_weight, weights.conv4_bias}));
    graph->conv4 = modules::ReshapeModule({core::TensorShape::from_dims({batch_size, kHiddenSize})}).build(
        ctx,
        modules::SliceModule({2, 0, 1}).build(ctx, conv4_bct));

    const auto lstm_outputs = modules::LSTMCellModule({kHiddenSize, kHiddenSize}).build(
        ctx,
        graph->conv4,
        graph->hidden_in,
        graph->cell_in,
        {
            weights.lstm_weight_ih,
            weights.lstm_weight_hh,
            weights.lstm_bias_ih,
            weights.lstm_bias_hh,
        });
    graph->hidden_out = lstm_outputs.hidden;
    graph->cell_out = lstm_outputs.cell;
    const auto hidden_relu = modules::ReluModule().build(ctx, graph->hidden_out);
    const auto hidden_bct = modules::ReshapeModule({core::TensorShape::from_dims({batch_size, kHiddenSize, 1})}).build(ctx, hidden_relu);
    const auto probability_bct = modules::SigmoidModule().build(
        ctx,
        modules::Conv1dModule({kHiddenSize, 1, 1, 1, 0, 1, true}).build(
            ctx,
            hidden_bct,
            {weights.final_conv_weight, weights.final_conv_bias}));
    graph->probability = modules::ReshapeModule({core::TensorShape::from_dims({batch_size, 1})}).build(ctx, probability_bct);

    graph->graph = ggml_new_graph_custom(graph->ggml, 4096, false);
    ggml_build_forward_expand(graph->graph, graph->probability.tensor);

    graph->buffer = ggml_backend_alloc_ctx_tensors(graph->ggml, execution_context_->backend());
    if (graph->buffer == nullptr) {
        throw std::runtime_error("Failed to allocate Silero VAD backend tensors");
    }

    chunk_graph_ = std::move(graph);
    engine::debug::timing_log_scalar("silero_vad.chunk.graph.build_ms", engine::debug::elapsed_ms(build_start));
    engine::debug::timing_log_scalar("silero_vad.chunk.graph.rebuilt", true);
}

SileroRuntime::ChunkResult SileroRuntime::infer_chunk(
    const std::vector<float> & chunk,
    int64_t batch,
    int sample_rate) {
    if (static_cast<int64_t>(chunk.size()) != batch * kChunkSamples) {
        throw std::runtime_error("Silero VAD chunk input size mismatch");
    }
    const auto total_start = Clock::now();
    const auto state_start = Clock::now();
    ensure_state(batch, sample_rate);
    engine::debug::timing_log_scalar("silero_vad.chunk.state_ms", engine::debug::elapsed_ms(state_start));

    const auto input_prepare_start = Clock::now();
    std::vector<float> chunk_input(static_cast<size_t>(batch * kInputSamples), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        std::copy(
            context_.begin() + static_cast<ptrdiff_t>(b * kContextSamples),
            context_.begin() + static_cast<ptrdiff_t>((b + 1) * kContextSamples),
            chunk_input.begin() + static_cast<ptrdiff_t>(b * kInputSamples));
        std::copy(
            chunk.begin() + static_cast<ptrdiff_t>(b * kChunkSamples),
            chunk.begin() + static_cast<ptrdiff_t>((b + 1) * kChunkSamples),
            chunk_input.begin() + static_cast<ptrdiff_t>(b * kInputSamples + kContextSamples));
    }
    engine::debug::timing_log_scalar("silero_vad.chunk.input_prepare_ms", engine::debug::elapsed_ms(input_prepare_start));
    ensure_chunk_graph(batch);
    const auto upload_start = Clock::now();
    core::write_tensor_f32(chunk_graph_->input, chunk_input);
    core::write_tensor_f32(chunk_graph_->hidden_in, hidden_);
    core::write_tensor_f32(chunk_graph_->cell_in, cell_);
    engine::debug::timing_log_scalar("silero_vad.chunk.input_upload_ms", engine::debug::elapsed_ms(upload_start));
    const auto compute_start = Clock::now();
    engine::core::compute_backend_graph(execution_context_->backend(), chunk_graph_->graph);
    engine::debug::timing_log_scalar("silero_vad.chunk.graph.compute_ms", engine::debug::elapsed_ms(compute_start));

    const auto read_start = Clock::now();
    std::vector<float> probability(static_cast<size_t>(batch), 0.0f);
    core::read_tensor_f32_into(chunk_graph_->probability.tensor, probability);
    core::read_tensor_f32_into(chunk_graph_->hidden_out.tensor, hidden_);
    core::read_tensor_f32_into(chunk_graph_->cell_out.tensor, cell_);
    engine::debug::timing_log_scalar("silero_vad.chunk.output_read_ms", engine::debug::elapsed_ms(read_start));

    const auto context_start = Clock::now();
    for (int64_t b = 0; b < batch; ++b) {
        std::copy(
            chunk.begin() + static_cast<ptrdiff_t>((b + 1) * kChunkSamples - kContextSamples),
            chunk.begin() + static_cast<ptrdiff_t>((b + 1) * kChunkSamples),
            context_.begin() + static_cast<ptrdiff_t>(b * kContextSamples));
    }
    engine::debug::timing_log_scalar("silero_vad.chunk.context_update_ms", engine::debug::elapsed_ms(context_start));

    ChunkResult result;
    result.probability = probability;
    engine::debug::timing_log_scalar("silero_vad.chunk.total_ms", engine::debug::elapsed_ms(total_start));
    return result;
}

std::vector<float> SileroRuntime::audio_forward(
    const std::vector<float> & waveform,
    int64_t batch,
    int64_t samples,
    int sample_rate) {
    if (sample_rate != kSampleRate) {
        throw std::runtime_error("Silero VAD 16k model only supports sample_rate=16000");
    }
    if (static_cast<int64_t>(waveform.size()) != batch * samples) {
        throw std::runtime_error("Silero VAD waveform size mismatch");
    }

    reset_states(batch);
    last_sample_rate_ = sample_rate;

    const int64_t chunks = (samples + kChunkSamples - 1) / kChunkSamples;
    const int64_t padded_samples = chunks * kChunkSamples;
    std::vector<float> padded(static_cast<size_t>(batch * padded_samples), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        std::copy(
            waveform.begin() + static_cast<ptrdiff_t>(b * samples),
            waveform.begin() + static_cast<ptrdiff_t>((b + 1) * samples),
            padded.begin() + static_cast<ptrdiff_t>(b * padded_samples));
    }

    std::vector<float> probabilities(static_cast<size_t>(batch * chunks), 0.0f);
    for (int64_t chunk_index = 0; chunk_index < chunks; ++chunk_index) {
        std::vector<float> chunk(static_cast<size_t>(batch * kChunkSamples), 0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            const auto src = padded.begin() + static_cast<ptrdiff_t>(b * padded_samples + chunk_index * kChunkSamples);
            std::copy(src, src + kChunkSamples, chunk.begin() + static_cast<ptrdiff_t>(b * kChunkSamples));
        }
        const auto chunk_result = infer_chunk(chunk, batch, sample_rate);
        for (int64_t b = 0; b < batch; ++b) {
            probabilities[static_cast<size_t>(b * chunks + chunk_index)] = chunk_result.probability[static_cast<size_t>(b)];
        }
    }
    return probabilities;
}

std::vector<SileroSpeechTimestamp> SileroRuntime::decode_speech_timestamps(
    const std::vector<float> & probs,
    int64_t samples,
    int sample_rate,
    const SileroVADConfig & config) const {
    float neg_threshold = config.neg_threshold;
    if (neg_threshold < 0.0f) {
        neg_threshold = std::max(config.threshold - 0.15f, 0.01f);
    }
    const int64_t min_speech_samples = sample_rate * config.min_speech_duration_ms / 1000;
    const int64_t speech_pad_samples = sample_rate * config.speech_pad_ms / 1000;
    const int64_t max_speech_samples = static_cast<int64_t>(sample_rate * config.max_speech_duration_s) - kChunkSamples - 2 * speech_pad_samples;
    const int64_t min_silence_samples = sample_rate * config.min_silence_duration_ms / 1000;
    const int64_t min_silence_samples_at_max_speech = sample_rate * config.min_silence_at_max_speech_ms / 1000;

    bool triggered = false;
    std::vector<SileroSpeechTimestamp> speeches;
    SileroSpeechTimestamp current = {};
    int64_t temp_end = 0;
    int64_t prev_end = 0;
    int64_t next_start = 0;
    std::vector<std::pair<int64_t, int64_t>> possible_ends;

    for (size_t i = 0; i < probs.size(); ++i) {
        const float speech_prob = probs[i];
        const int64_t cur_sample = static_cast<int64_t>(i) * kChunkSamples;

        if (speech_prob >= config.threshold && temp_end) {
            const int64_t sil_dur = cur_sample - temp_end;
            if (sil_dur > min_silence_samples_at_max_speech) {
                possible_ends.push_back({temp_end, sil_dur});
            }
            temp_end = 0;
            if (next_start < prev_end) {
                next_start = cur_sample;
            }
        }

        if (speech_prob >= config.threshold && !triggered) {
            triggered = true;
            current.start = cur_sample;
            continue;
        }

        if (triggered && cur_sample - current.start > max_speech_samples) {
            if (config.use_max_poss_sil_at_max_speech && !possible_ends.empty()) {
                auto best = *std::max_element(possible_ends.begin(), possible_ends.end(), [](const auto & lhs, const auto & rhs) {
                    return lhs.second < rhs.second;
                });
                prev_end = best.first;
                current.end = prev_end;
                speeches.push_back(current);
                current = {};
                next_start = prev_end + best.second;
                if (next_start < prev_end + cur_sample) {
                    current.start = next_start;
                } else {
                    triggered = false;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else if (prev_end) {
                current.end = prev_end;
                speeches.push_back(current);
                current = {};
                if (next_start < prev_end) {
                    triggered = false;
                } else {
                    current.start = next_start;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else {
                current.end = cur_sample;
                speeches.push_back(current);
                current = {};
                prev_end = next_start = temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }

        if (speech_prob < neg_threshold && triggered) {
            if (!temp_end) {
                temp_end = cur_sample;
            }
            const int64_t sil_dur_now = cur_sample - temp_end;
            if (!config.use_max_poss_sil_at_max_speech && sil_dur_now > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }
            if (sil_dur_now < min_silence_samples) {
                continue;
            }
            current.end = temp_end;
            if (current.end - current.start > min_speech_samples) {
                speeches.push_back(current);
            }
            current = {};
            prev_end = next_start = temp_end = 0;
            triggered = false;
            possible_ends.clear();
        }
    }

    if (triggered && (samples - current.start) > min_speech_samples) {
        current.end = samples;
        speeches.push_back(current);
    }

    for (size_t i = 0; i < speeches.size(); ++i) {
        if (i == 0) {
            speeches[i].start = std::max<int64_t>(0, speeches[i].start - speech_pad_samples);
        }
        if (i + 1 < speeches.size()) {
            const int64_t silence = speeches[i + 1].start - speeches[i].end;
            if (silence < 2 * speech_pad_samples) {
                speeches[i].end += silence / 2;
                speeches[i + 1].start = std::max<int64_t>(0, speeches[i + 1].start - silence / 2);
            } else {
                speeches[i].end = std::min<int64_t>(samples, speeches[i].end + speech_pad_samples);
                speeches[i + 1].start = std::max<int64_t>(0, speeches[i + 1].start - speech_pad_samples);
            }
        } else {
            speeches[i].end = std::min<int64_t>(samples, speeches[i].end + speech_pad_samples);
        }
    }
    return speeches;
}

runtime::TaskResult SileroRuntime::build_offline_result(
    const std::vector<float> & waveform,
    int sample_rate,
    const SileroVADConfig & config) {
    const auto probabilities = audio_forward(
        waveform,
        1,
        static_cast<int64_t>(waveform.size()),
        sample_rate);
    const auto timestamps = decode_speech_timestamps(
        probabilities,
        static_cast<int64_t>(waveform.size()),
        sample_rate,
        config);

    runtime::TaskResult result;
    for (const auto & segment : timestamps) {
        result.speech_segments.push_back({
            {segment.start, segment.end},
            1.0f,
            {},
        });
    }
    return result;
}

runtime::StreamEvent SileroRuntime::build_stream_event(
    float probability,
    const SileroVADConfig & config,
    int64_t chunk_start_sample) {
    runtime::StreamEvent event;
    const int64_t window_size = kChunkSamples;
    const int64_t speech_pad_samples = kSampleRate * config.speech_pad_ms / 1000;
    const int64_t min_silence_samples = kSampleRate * config.min_silence_duration_ms / 1000;
    const float neg_threshold = config.neg_threshold < 0.0f
        ? std::max(config.threshold - 0.15f, 0.01f)
        : config.neg_threshold;

    if (probability >= config.threshold && temp_end_) {
        temp_end_ = 0;
    }

    if (probability >= config.threshold && !triggered_) {
        triggered_ = true;
        const int64_t speech_start = std::max<int64_t>(0, chunk_start_sample - speech_pad_samples);
        current_speech_start_ = speech_start;
        event.voice_activity.push_back({
            runtime::VoiceActivityEvent::Kind::SpeechStart,
            speech_start,
            probability,
            std::nullopt,
        });
        return event;
    }

    if (probability < neg_threshold && triggered_) {
        if (!temp_end_) {
            temp_end_ = chunk_start_sample + window_size;
        }
        if ((chunk_start_sample + window_size) - temp_end_ < min_silence_samples) {
            return event;
        }
        const int64_t speech_end = temp_end_ + speech_pad_samples - window_size;
        triggered_ = false;
        temp_end_ = 0;
        SileroSpeechTimestamp segment{
            current_speech_start_,
            std::max(current_speech_start_, speech_end),
        };
        current_speech_start_ = 0;
        streamed_segments_.push_back(segment);
        event.voice_activity.push_back({
            runtime::VoiceActivityEvent::Kind::SpeechEnd,
            speech_end,
            probability,
            runtime::SpeechSegment{{segment.start, segment.end}, probability, {}},
        });
    }
    return event;
}

runtime::TaskResult SileroRuntime::run_offline(const runtime::AudioBuffer & audio, const SileroVADConfig & config) {
    const auto total_start = Clock::now();
    const auto mono = require_mono_audio(audio, "Silero VAD offline run");
    engine::debug::timing_log_scalar("silero_vad.offline.audio_prepare_ms", engine::debug::elapsed_ms(total_start));
    const auto infer_start = Clock::now();
    auto result = build_offline_result(mono, audio.sample_rate, config);
    engine::debug::timing_log_scalar("silero_vad.offline.infer_total_ms", engine::debug::elapsed_ms(infer_start));
    engine::debug::timing_log_scalar("silero_vad.offline.total_ms", engine::debug::elapsed_ms(total_start));
    return result;
}

runtime::StreamEvent SileroRuntime::process_chunk(const runtime::AudioChunk & chunk, const SileroVADConfig & config) {
    const auto total_start = Clock::now();
    if (chunk.channels != 1) {
        throw std::runtime_error("Silero VAD streaming currently requires mono audio");
    }
    if (chunk.sample_rate <= 0) {
        throw std::runtime_error("Silero VAD streaming requires a positive sample rate");
    }
    if (static_cast<int64_t>(chunk.samples.size()) != kChunkSamples) {
        throw std::runtime_error("Silero VAD streaming chunk must contain exactly 512 samples");
    }
    if (current_sample_ != 0 && chunk.start_sample != current_sample_) {
        throw std::runtime_error("Silero VAD streaming chunks must be contiguous");
    }
    const auto chunk_result = infer_chunk(chunk.samples, 1, chunk.sample_rate);
    const float probability = chunk_result.probability[0];
    const auto postprocess_start = Clock::now();
    auto event = build_stream_event(probability, config, chunk.start_sample);
    current_sample_ = chunk.start_sample + static_cast<int64_t>(chunk.samples.size());
    engine::debug::timing_log_scalar("silero_vad.streaming.postprocess_ms", engine::debug::elapsed_ms(postprocess_start));
    engine::debug::timing_log_scalar("silero_vad.streaming.chunk.total_ms", engine::debug::elapsed_ms(total_start));
    return event;
}

runtime::TaskResult SileroRuntime::finalize_stream(const SileroVADConfig & config) const {
    const auto total_start = Clock::now();
    runtime::TaskResult result;
    for (const auto & segment : streamed_segments_) {
        result.speech_segments.push_back({
            {segment.start, segment.end},
            config.threshold,
            {},
        });
    }
    const int64_t min_speech_samples = static_cast<int64_t>(kSampleRate) * config.min_speech_duration_ms / 1000;
    if (triggered_ && (current_sample_ - current_speech_start_) > min_speech_samples) {
        result.speech_segments.push_back({
            {current_speech_start_, current_sample_},
            config.threshold,
            {},
        });
    }
    engine::debug::timing_log_scalar("silero_vad.streaming.finalize_ms", engine::debug::elapsed_ms(total_start));
    return result;
}

}  // namespace engine::models::silero_vad
