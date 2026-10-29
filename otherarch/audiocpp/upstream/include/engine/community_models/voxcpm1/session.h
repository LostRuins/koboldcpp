#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/voxcpm1/assets.h"
#include "engine/community_models/voxcpm1/audiovae.h"
#include "engine/community_models/voxcpm1/generator.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace engine::community_models::voxcpm1 {

class VoxCPM1SessionBase : public runtime::RuntimeSessionBase {
public:
  VoxCPM1SessionBase(runtime::TaskSpec task, runtime::SessionOptions options,
                     std::shared_ptr<const VoxCPM1Assets> assets);
  ~VoxCPM1SessionBase() override;

protected:
  std::string family_impl() const;
  runtime::VoiceTaskKind task_kind_impl() const;
  runtime::RunMode run_mode_impl() const;
  void prepare_impl(const runtime::SessionPreparationRequest &request);

  struct EncodedPromptCacheKey {
    std::string prompt_text;
    std::optional<runtime::AudioBuffer> prompt_audio;
    std::optional<runtime::AudioBuffer> reference_audio;
  };

  struct EncodedPromptCacheKeyEqual {
    bool operator()(const EncodedPromptCacheKey &lhs,
                    const EncodedPromptCacheKey &rhs) const;
  };

  struct EncodedPromptCacheEntry {
    VoxCPM1EncodedPrompt encoded;
  };

  VoxCPM1GenerationOptions
  generation_options_from_request(const runtime::TaskRequest &request) const;
  void validate_request(const runtime::TaskRequest &request) const;
  const VoxCPM1EncodedPrompt *encoded_prompt_for_request(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio);

  runtime::TaskResult run_offline_request(const runtime::TaskRequest &request);
  runtime::TaskResult run_streaming_request(
      const runtime::TaskRequest &request,
      const runtime::StreamEventCallback &stream_event_sink = nullptr);
  void release_request_runtime_memory();

  runtime::TaskSpec task_;
  std::shared_ptr<const VoxCPM1Assets> assets_;
  VoxCPM1FeatureGeneratorConfig generator_config_;
  VoxCPM1AudioVAEDecoderConfig decoder_config_;
  std::unique_ptr<VoxCPM1FeatureGeneratorRuntime> generator_;
  std::unique_ptr<VoxCPM1AudioVAEDecoderRuntime> decoder_;
  runtime::CacheSlots<EncodedPromptCacheKey, EncodedPromptCacheEntry,
                      EncodedPromptCacheKeyEqual>
      encoded_prompt_cache_;
  std::optional<EncodedPromptCacheEntry> uncached_encoded_prompt_;
};

class VoxCPM1OfflineSession final : public VoxCPM1SessionBase,
                                    public runtime::IOfflineVoiceTaskSession {
public:
  VoxCPM1OfflineSession(runtime::TaskSpec task, runtime::SessionOptions options,
                        std::shared_ptr<const VoxCPM1Assets> assets);

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;
};

class VoxCPM1StreamingSession final : public VoxCPM1SessionBase,
                                      public runtime::IStreamingVoiceTaskSession {
public:
  VoxCPM1StreamingSession(runtime::TaskSpec task, runtime::SessionOptions options,
                          std::shared_ptr<const VoxCPM1Assets> assets);

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::StreamingPolicy streaming_policy() const override;
  void start_stream(const runtime::TaskRequest &request) override;
  std::optional<runtime::StreamEvent> next_stream_event() override;
  void set_stream_event_sink(runtime::StreamEventCallback sink) override;
  runtime::TaskResult finish_stream() override;
  void reset() override;
  runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &chunk) override;
  runtime::TaskResult finalize() override;

private:
  runtime::TaskResult result_;
  size_t next_chunk_index_ = 0;
  bool started_ = false;
  runtime::StreamEventCallback stream_event_sink_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_voxcpm1_loader();

} // namespace engine::community_models::voxcpm1

