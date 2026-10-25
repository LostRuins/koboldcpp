#include "components/component_weights.h"

#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"

#include <memory>
#include <utility>

namespace engine::models::chatterbox {
namespace {

SpeakerEncoderOutputs to_chatterbox_outputs(engine::modules::CampplusEncoderOutputs outputs) {
    SpeakerEncoderOutputs result;
    result.embedding = std::move(outputs.embedding);
    result.embedding_size = outputs.embedding_size;
    return result;
}

}  // namespace

struct CAMPPlusEncoderComponent::State {
    engine::modules::CampplusEncoderComponent component;
};

CAMPPlusEncoderComponent CAMPPlusEncoderComponent::load_from_source(
    std::shared_ptr<const engine::assets::TensorSource> source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    engine::modules::CampplusEncoderConfig config;
    config.feat_dim = 80;
    config.embedding_size = 192;
    config.weight_storage_type = weight_storage_type;

    auto state = std::make_shared<State>();
    state->component = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
        std::move(source),
        execution_context.config(),
        config);
    return CAMPPlusEncoderComponent(std::move(state), execution_context);
}

CAMPPlusEncoderComponent::CAMPPlusEncoderComponent(
    std::shared_ptr<State> state,
    const engine::core::ExecutionContext & execution_context)
    : execution_context_(&execution_context),
      state_(std::move(state)) {}

const engine::core::BackendConfig & CAMPPlusEncoderComponent::backend() const noexcept {
    return execution_context_->config();
}

SpeakerEncoderOutputs CAMPPlusEncoderComponent::embed_from_audio(const runtime::AudioBuffer & audio) const {
    const auto fbank = components::compute_campplus_fbank(audio);
    return embed_from_features(fbank.features, fbank.frames, fbank.dims);
}

SpeakerEncoderOutputs CAMPPlusEncoderComponent::embed_from_features(
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims) const {
    return to_chatterbox_outputs(state_->component.embed_from_features(features, frames, dims));
}

}  // namespace engine::models::chatterbox
