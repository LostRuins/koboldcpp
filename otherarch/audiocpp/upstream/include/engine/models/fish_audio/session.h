#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/generator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::fish_audio {

class FishAudioSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession,
                               public runtime::StreamingTtsSessionBase {
public:
    FishAudioSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const FishAudioAssets> assets);
    ~FishAudioSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override {}
    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming);

    struct ReferenceCacheKey {
        std::string source_id;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
    };

    struct ReferenceCacheKeyEqual {
        bool operator()(const ReferenceCacheKey & lhs, const ReferenceCacheKey & rhs) const;
    };

    struct ReferenceCacheEntry {
        engine::codecs::FishDacCodes codes;
    };

    FishAudioRequest make_request(const runtime::TaskRequest & request) const;
    const engine::codecs::FishDacCodes & resolve_reference_codes(const FishAudioReference & reference);

    runtime::TaskSpec task_;
    std::shared_ptr<const FishAudioAssets> assets_;
    std::unique_ptr<FishAudioGenerator> generator_;
    std::optional<FishAudioRequest> defaults_;
    runtime::CacheSlots<ReferenceCacheKey, ReferenceCacheEntry, ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;
};

}  // namespace engine::models::fish_audio
