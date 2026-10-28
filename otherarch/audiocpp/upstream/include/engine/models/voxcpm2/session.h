#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/models/voxcpm2/assets.h"
#include "engine/models/voxcpm2/audiovae.h"
#include "engine/models/voxcpm2/generator.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::voxcpm2 {

class VoxCPM2SessionBase : public runtime::RuntimeSessionBase {
public:
  VoxCPM2SessionBase(runtime::TaskSpec task, runtime::SessionOptions options,
                     std::shared_ptr<const VoxCPM2Assets> assets);
  ~VoxCPM2SessionBase() override;

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
    VoxCPM2EncodedPrompt encoded;
  };

  VoxCPM2GenerationOptions
  generation_options_from_request(const runtime::TaskRequest &request) const;
  void validate_request(const runtime::TaskRequest &request) const;
  const VoxCPM2EncodedPrompt *encoded_prompt_for_request(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio);

  runtime::TaskResult run_offline_request(const runtime::TaskRequest &request);
  runtime::TaskResult run_streaming_request(
      const runtime::TaskRequest &request,
      const runtime::StreamEventCallback &stream_event_sink = nullptr);
  void release_request_runtime_memory();

  runtime::TaskSpec task_;
  std::shared_ptr<const VoxCPM2Assets> assets_;
  VoxCPM2FeatureGeneratorConfig generator_config_;
  VoxCPM2AudioVAEDecoderConfig decoder_config_;
  // Set by voxcpm2.exec_plan.audiovae.params.threads: the AudioVAE runs on its own CPU backend
  // with its own threads, so streaming can decode a patch window while the
  // generator works on the next patch. Declared before decoder_ so the decoder
  // is destroyed first.
  std::unique_ptr<engine::core::ExecutionContext> decoder_execution_context_;
  std::unique_ptr<VoxCPM2FeatureGeneratorRuntime> generator_;
  std::unique_ptr<VoxCPM2AudioVAEDecoderRuntime> decoder_;
  runtime::CacheSlots<EncodedPromptCacheKey, EncodedPromptCacheEntry,
                      EncodedPromptCacheKeyEqual>
      encoded_prompt_cache_;
  std::optional<EncodedPromptCacheEntry> uncached_encoded_prompt_;
  runtime::StreamingAudioController<float> streaming_audio_;
};

class VoxCPM2OfflineSession final : public VoxCPM2SessionBase,
                                    public runtime::IOfflineVoiceTaskSession {
public:
  VoxCPM2OfflineSession(runtime::TaskSpec task, runtime::SessionOptions options,
                        std::shared_ptr<const VoxCPM2Assets> assets);

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;
};

class VoxCPM2StreamingSession final : public VoxCPM2SessionBase,
                                      public runtime::StreamingTtsSessionBase {
public:
  VoxCPM2StreamingSession(runtime::TaskSpec task, runtime::SessionOptions options,
                          std::shared_ptr<const VoxCPM2Assets> assets);

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
private:
  runtime::TaskResult generate_stream(const runtime::TaskRequest &request) override;
  void reset_stream_state() override { streaming_audio_.reset(); }
};

} // namespace engine::models::voxcpm2

