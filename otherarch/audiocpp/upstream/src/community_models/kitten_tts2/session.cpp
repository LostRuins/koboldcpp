#include "engine/community_models/kitten_tts2/session.h"
#include "engine/community_models/kitten_tts2/decoder.h"
#include "engine/community_models/kitten_tts2/speaker.h"
#include "engine/community_models/kitten_tts2/reference.h"
#include "engine/framework/audio/conversion.h"
#include "engine/models/chatterbox/components.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/tokenizers/qwen_bpe_bundle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <regex>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace engine::community_models::kitten_tts2 {
namespace {
namespace json = engine::io::json;
namespace binding = engine::modules::binding;
constexpr const char * family_name = "kitten_tts2";

struct TokenMap {
    int32_t audio_base, audio_count, speech_start, speech_end, text_start, start, stop, final_seg;
    int32_t ref_text_start, ref_text_end, ref_speech_start, ref_speech_end;
};
struct Voice {
    std::string transcript;
    std::vector<int32_t> reference_codes;
    std::vector<float> speaker, mel, embedding;
    std::vector<int64_t> prompt_tokens;
};
template <typename T>
std::vector<T> numbers(const json::Value & value) {
    std::vector<T> result;
    for (const auto & item : value.as_array()) {
        const double number = item.as_number();
        if (!std::isfinite(number) || number < static_cast<double>(std::numeric_limits<T>::lowest()) ||
            number >= static_cast<double>(std::numeric_limits<T>::max()))
            throw std::runtime_error("invalid Kitten voice value");
        if constexpr (std::is_integral_v<T>) {
            if (std::floor(number) != number) throw std::runtime_error("Kitten voice token must be an integer");
        }
        result.push_back(static_cast<T>(number));
    }
    return result;
}
struct Assets {
    assets::ResourceBundle resources;
    json::Value config, lm_config;
    TokenMap tokens{};
    std::shared_ptr<const assets::TensorSource> lm;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
};
std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<Assets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path(family_name));
    out->config = out->resources.parse_json("config");
    out->lm_config = out->resources.parse_json("lm_config");
    if (json::require_string(out->config, "type") != "KITTEN2" ||
        json::require_string(out->lm_config, "model_type") != "qwen3")
        throw std::runtime_error("Kitten TTS 2 requires a KITTEN2 / Qwen3 checkpoint");
    const auto & tm = out->config.require("token_map");
    out->tokens = {json::require_i32(tm,"audio_id_base"), json::require_i32(tm,"num_audio_tokens"),
        json::require_i32(tm,"speech_start_id"), json::require_i32(tm,"speech_end_id"),
        json::require_i32(tm,"text_start_id"), json::require_i32(tm,"start_id"),
        json::require_i32(tm,"stop_id"), json::require_i32(tm,"final_seg_id"),
        json::require_i32(tm,"reference_text_start_id"), json::require_i32(tm,"reference_text_end_id"),
        json::require_i32(tm,"reference_speech_start_id"), json::require_i32(tm,"reference_speech_end_id")};
    const auto & t = out->tokens;
    if (t.audio_count != 6561 || t.audio_base < 0 || t.speech_end < 0 || t.speech_end >= t.audio_base ||
        t.stop < t.speech_end || t.stop >= t.audio_base ||
        t.audio_base + static_cast<int64_t>(t.audio_count) > json::require_i64(out->lm_config,"vocab_size"))
        throw std::runtime_error("unsupported Kitten TTS 2 token map");
    out->lm = out->resources.open_tensor_source("language_model");
    out->tokenizer = tokenizers::load_qwen_bpe_tokenizer(out->resources);
    return out;
}

core::TensorValue load_projection(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & name, assets::TensorStorageType type, int64_t rows, int64_t columns) {
    if (type != assets::TensorStorageType::Q4_0)
        return store.load_tensor(source, name, type, {rows, columns});
    // Kitten's folded ternary matrices fit standard Q4_0 exactly when the scale
    // is gamma/4 and the three signed codewords are -4, 0, +4. Small scales
    // use +/-2 or +/-1 to avoid rounding an FP16 subnormal. Generic Q4_0
    // quantization instead uses asymmetric endpoints and changes these weights.
    // This is model-local packing, with no custom GGML type or backend changes.
    if (columns % 32) throw std::runtime_error("Kitten ternary Q4 rows must be divisible by 32");
    const auto values=source.require_f32(name,{rows,columns});
    const size_t blocks=values.size()/32;
    std::vector<std::byte> bytes(blocks*18);
    for (size_t block=0;block<blocks;++block) {
        const auto * input=values.data()+block*32;
        float gamma=0;
        for (size_t i=0;i<32;++i) gamma=std::max(gamma,std::abs(input[i]));
        uint8_t magnitude=4;
        auto scale=ggml_fp32_to_fp16(gamma/magnitude);
        while (magnitude>1 && ggml_fp16_to_fp32(scale)*magnitude!=gamma) {
            magnitude/=2;
            scale=ggml_fp32_to_fp16(gamma/magnitude);
        }
        if (ggml_fp16_to_fp32(scale)*magnitude!=gamma)
            throw std::runtime_error("Kitten exact Q4 packing requires original ternary weights; use q8_0 for this source: "+name);
        std::memcpy(bytes.data()+block*18,&scale,sizeof(scale));
        auto code=[gamma,magnitude,&name](float value)->uint8_t {
            if (value==0) return 8;
            if (value==gamma) return 8+magnitude;
            if (value==-gamma) return 8-magnitude;
            throw std::runtime_error("non-ternary Kitten projection: "+name);
        };
        for (size_t i=0;i<16;++i)
            bytes[block*18+2+i]=static_cast<std::byte>(code(input[i]) | (code(input[i+16])<<4));
    }
    return store.make_tensor(core::TensorShape::from_dims({rows,columns}),GGML_TYPE_Q4_0,bytes.data(),bytes.size());
}

class Session final : public runtime::IOfflineVoiceTaskSession, public runtime::RuntimeSessionBase {
public:
    Session(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(std::move(task)), assets_(std::move(assets)), contract_(std::move(contract)) {
        if ((task_.task != runtime::VoiceTaskKind::Tts && task_.task != runtime::VoiceTaskKind::VoiceCloning) ||
            task_.mode != runtime::RunMode::Offline)
            throw std::runtime_error("Kitten TTS 2 supports offline TTS and voice cloning sessions");
        runtime::validate_spec_backed_session_options(options, *contract_, family_name, "Kitten TTS 2");
        load_voices(assets_->resources.require_file("default_voices"));
        waveform_ = std::make_unique<WaveformDecoder>(assets_->resources.open_tensor_source("s3gen"), execution_context());
        const auto type = runtime::parse_tensor_storage_option(options.options, "kitten_tts2.weight_type",
            assets::TensorStorageType::Native,
            {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
             assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
             assets::TensorStorageType::Q4_0, assets::TensorStorageType::Q8_0});
        load_weights(type);
    }
    std::string family() const override { return family_name; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        // Include reference conditioning and lazy cloning encoder loads.
        const auto start = std::chrono::steady_clock::now();
        require_prepared("Kitten TTS 2 run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Kitten TTS 2");
        if (!request.text_input || request.text_input->text.empty())
            throw std::runtime_error("Kitten TTS 2 requires non-empty text_input");
        std::string name = json::optional_string(assets_->config, "default_voice", "Bruno");
        const runtime::AudioBuffer * reference = request.audio_input ? &*request.audio_input : nullptr;
        if (request.voice && request.voice->speaker) {
            if (request.voice->speaker->audio) reference = &*request.voice->speaker->audio;
            if (request.voice->speaker->cached_voice_id) name = *request.voice->speaker->cached_voice_id;
        }
        if (const auto id = request.options.find("voice_id"); id != request.options.end()) name = id->second;
        const Voice * selected = nullptr;
        if (reference) {
            const auto transcript = request.options.find("reference_text");
            if (transcript == request.options.end() || transcript->second.find_first_not_of(" \r\n\t") == std::string::npos)
                throw std::runtime_error("Kitten voice cloning requires reference_text matching the reference audio");
            selected = &clone_voice(*reference, transcript->second);
        } else {
            if (task_.task == runtime::VoiceTaskKind::VoiceCloning)
                throw std::runtime_error("Kitten voice cloning requires reference audio");
            const auto found = voices_.find(name);
            if (found == voices_.end()) throw std::runtime_error("unknown Kitten TTS 2 voice: " + name);
            selected = &found->second;
        }
        const auto & voice = *selected;
        const auto parsed_seed = runtime::parse_i64_option(request.options, {"seed"});
        if (parsed_seed && *parsed_seed < 0) throw std::runtime_error("Kitten seed must be non-negative");
        if (!request.option_arrays.empty()) throw std::runtime_error("Kitten TTS 2 does not accept list-valued options");
        const uint64_t seed = static_cast<uint64_t>(parsed_seed
            .value_or(static_cast<int64_t>(runtime::random_u64_seed() & 0x7fffffff)));
        auto size = text::parse_text_chunk_size_override(request.options).value_or(380);
        if (size < 32 || size > 1000) throw std::runtime_error("Kitten text_chunk_size must be between 32 and 1000");
        auto mode = text::parse_text_chunk_mode_override(request.options).value_or(text::TextChunkMode::TagAware);
        const auto chunks = runtime::chunk_text_request(request, size, mode);
        runtime::AudioBuffer audio{24000, 1, {}};
        std::mt19937 rng(static_cast<uint32_t>(seed));
        for (size_t i = 0; i < chunks.size(); ++i) {
            auto codes = generate(chunks[i].text_input->text, voice, request.options, rng);
            const auto waveform_start = std::chrono::steady_clock::now();
            auto samples = waveform_->decode(codes, voice.prompt_tokens, voice.mel, voice.embedding,
                seed + i);
            debug::timing_log_scalar("kitten_tts2.waveform_ms", debug::elapsed_ms(waveform_start));
            if (chunks.size() > 1) {
                // Match the Python joiner's RMS edge trim and eight-millisecond fades.
                size_t first = samples.size(), last = 0;
                for (size_t pos = 0; pos + 240 <= samples.size(); pos += 240) {
                    double power = 1e-12;
                    for (size_t n = pos; n < pos + 240; ++n) power += samples[n] * samples[n] / 240.0;
                    if (std::sqrt(power) > std::pow(10.0, -45.0 / 20.0)) {
                        first = std::min(first, pos); last = pos + 240;
                    }
                }
                if (first < last) {
                    first = first > 480 ? first - 480 : 0;
                    last = std::min(samples.size(), last + 480);
                    samples = std::vector<float>(samples.begin() + first, samples.begin() + last);
                }
                if (samples.size() > 384) for (size_t n = 0; n < 192; ++n) {
                    samples[n] *= static_cast<float>(n) / 191;
                    samples[samples.size() - 1 - n] *= static_cast<float>(n) / 191;
                }
            }
            if (i) audio.samples.insert(audio.samples.end(), 3840, 0.0f);
            audio.samples.insert(audio.samples.end(), samples.begin(), samples.end());
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        runtime::TaskResult result;
        result.audio_output = std::move(audio);
        return result;
    }
private:
    const Voice & clone_voice(const runtime::AudioBuffer & input, const std::string & transcript) {
        if (input.sample_rate <= 0 || input.channels <= 0 || input.samples.empty() ||
            input.samples.size() % input.channels != 0)
            throw std::runtime_error("invalid Kitten reference audio format");
        const double seconds = static_cast<double>(input.samples.size()) / input.channels / input.sample_rate;
        if (seconds < 1 || seconds > 30)
            throw std::runtime_error("Kitten voice reference must contain between 1 and 30 seconds of audio");
        for (float value : input.samples)
            if (!std::isfinite(value)) throw std::runtime_error("Kitten reference audio contains non-finite samples");
        if (clone_ && clone_->transcript == transcript && reference_cache_.sample_rate == input.sample_rate &&
            reference_cache_.channels == input.channels && reference_cache_.samples == input.samples) return *clone_;
        if (!speaker_) {
            auto source = assets_->resources.open_tensor_source("s3gen");
            auto speaker = std::make_unique<SpeakerEncoder>(*assets_->resources.open_tensor_source("speaker"), *assets_->lm, execution_context());
            auto tokenizer = std::make_unique<models::chatterbox::S3TokenizerComponent>(
                models::chatterbox::S3TokenizerComponent::load_from_source(*source, execution_context(), assets::TensorStorageType::F32));
            auto camp = std::make_unique<models::chatterbox::CAMPPlusEncoderComponent>(
                models::chatterbox::CAMPPlusEncoderComponent::load_from_source(source, execution_context(), assets::TensorStorageType::F32));
            speaker_ = std::move(speaker);
            tokenizer_ = std::move(tokenizer);
            camp_ = std::move(camp);
        }
        const auto started = std::chrono::steady_clock::now();
        auto mono = audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
        const auto resample = [&](int rate) {
            return audio::resample_mono_torchaudio_sinc_hann(mono, input.sample_rate, rate,
                audio::torchaudio_sinc_hann_float32_options());
        };
        runtime::AudioBuffer wav16{16000,1,resample(16000)};
        auto voice = std::make_unique<Voice>();
        voice->transcript = transcript;
        const auto identity = speaker_->embed(wav16.samples);
        voice->speaker = speaker_->project(identity);
        voice->reference_codes = tokenizer_->tokenize(wav16, std::nullopt).tokens;
        runtime::AudioBuffer wav24{24000,1,resample(24000)};
        auto conditioning = prepare_decoder_reference(std::move(wav24), *tokenizer_, *camp_);
        voice->prompt_tokens.assign(conditioning.prompt_tokens.begin(), conditioning.prompt_tokens.end());
        voice->mel = std::move(conditioning.prompt_feat);
        voice->embedding = std::move(conditioning.embedding);
        if (voice->reference_codes.empty() || voice->prompt_tokens.empty())
            throw std::runtime_error("Kitten reference audio produced no codec tokens");
        // A changed reference invalidates the LM prefix even when its allocation
        // happens to reuse the same address as the previous cached voice.
        last_voice_ = nullptr;
        reference_cache_ = input;
        clone_ = std::move(voice);
        debug::timing_log_scalar("kitten_tts2.reference_ms", debug::elapsed_ms(started));
        return *clone_;
    }
    void load_voices(const std::filesystem::path & path) {
        const int64_t hidden = json::require_i64(assets_->lm_config, "hidden_size");
        const auto all = json::parse_file(path);
        for (const auto & [name, item] : all.as_object()) {
            Voice voice;
            voice.transcript = json::require_string(item, "transcript");
            voice.reference_codes = numbers<int32_t>(item.require("reference_tokens"));
            voice.speaker = numbers<float>(item.require("speaker"));
            voice.prompt_tokens = numbers<int64_t>(item.require("prompt_token").as_array().at(0));
            for (const auto & row : item.require("prompt_feat").as_array().at(0).as_array()) {
                auto values = numbers<float>(row);
                if (values.size() != 80) throw std::runtime_error("Kitten voice mel rows must have 80 channels");
                voice.mel.insert(voice.mel.end(), values.begin(), values.end());
            }
            voice.embedding = numbers<float>(item.require("embedding").as_array().at(0));
            if (voice.speaker.size() != static_cast<size_t>(hidden) || voice.embedding.size() != 192 ||
                voice.prompt_tokens.empty() || voice.mel.empty() || voice.reference_codes.empty())
                throw std::runtime_error("invalid prepared Kitten voice: " + name);
            for (auto code : voice.reference_codes) if (code < 0 || code >= 6561) throw std::runtime_error("invalid reference codec token");
            for (auto code : voice.prompt_tokens) if (code < 0 || code >= 6561) throw std::runtime_error("invalid decoder prompt token");
            voices_.emplace(name, std::move(voice));
        }
    }
    void load_weights(assets::TensorStorageType type) {
        const auto & c = assets_->lm_config;
        const auto & source = *assets_->lm;
        const auto hidden = json::require_i64(c,"hidden_size");
        const auto intermediate = json::require_i64(c,"intermediate_size");
        const auto heads = json::require_i64(c,"num_attention_heads");
        const auto kv_heads = json::require_i64(c,"num_key_value_heads");
        const auto dim = json::require_i64(c,"head_dim");
        const auto layers = json::require_i64(c,"num_hidden_layers");
        const auto vocab = json::require_i64(c,"vocab_size");
        store_ = std::make_unique<core::BackendWeightStore>(execution_context().backend(), execution_context().backend_type(),
            "kitten_tts2.lm.weights", 4 * 1024 * 1024);
        modules::CausalDecoderRuntimeWeights weights;
        weights.token_embedding = store_->load_tensor(source, "model.embed_tokens.weight",
            assets::TensorStorageType::Native, {vocab,hidden});
        for (int64_t i = 0; i < layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(*store_,source,p+".input_layernorm",hidden);
            layer.post_norm = binding::norm_weight_from_source(*store_,source,p+".post_attention_layernorm",hidden);
            layer.self_attention.q_weight = load_projection(*store_,source,p+".self_attn.q_proj.weight",type,heads*dim,hidden);
            layer.self_attention.k_weight = load_projection(*store_,source,p+".self_attn.k_proj.weight",type,kv_heads*dim,hidden);
            layer.self_attention.v_weight = load_projection(*store_,source,p+".self_attn.v_proj.weight",type,kv_heads*dim,hidden);
            layer.self_attention.out_weight = load_projection(*store_,source,p+".self_attn.o_proj.weight",type,hidden,heads*dim);
            layer.q_norm = binding::norm_weight_from_source(*store_,source,p+".self_attn.q_norm",dim);
            layer.k_norm = binding::norm_weight_from_source(*store_,source,p+".self_attn.k_norm",dim);
            layer.mlp.gate_proj = {load_projection(*store_,source,p+".mlp.gate_proj.weight",type,intermediate,hidden),std::nullopt};
            layer.mlp.up_proj = {load_projection(*store_,source,p+".mlp.up_proj.weight",type,intermediate,hidden),std::nullopt};
            const auto down_name = p + ".mlp.down_proj.weight";
            // The folded model's large SwiGLU activations can overflow CUDA's
            // Q4 activation-sum intermediates. Keep only these down projections
            // in BF16 on CUDA; their ternary values remain exact. CPU retains Q4.
            const bool q4_down = type == assets::TensorStorageType::Q4_0 ||
                (type == assets::TensorStorageType::Native &&
                 assets::tensor_storage_type_for_dtype(source.require_metadata(down_name).dtype) == assets::TensorStorageType::Q4_0);
            const auto down_type = q4_down && execution_context().backend_type() == core::BackendType::Cuda
                ? assets::TensorStorageType::BF16 : type;
            layer.mlp.down_proj = {load_projection(*store_,source,down_name,down_type,hidden,intermediate),std::nullopt};
            weights.stack.layers.push_back(std::move(layer));
        }
        weights.final_norm = binding::norm_weight_from_source(*store_,source,"model.norm",hidden);
        store_->upload();
        embedding_ = weights.token_embedding;
        // Restrict projection to the speech vocabulary. Preserve global token IDs
        // for embedding/KV; compact readback indexes are only used by the sampler.
        const auto & t = assets_->tokens;
        head_context_ = std::shared_ptr<ggml_context>(ggml_init({2 * ggml_tensor_overhead(),nullptr,true}),ggml_free);
        if (!head_context_) throw std::runtime_error("failed to allocate Kitten head metadata");
        const int64_t head_rows = t.audio_base + t.audio_count - t.speech_end;
        core::ModuleBuildContext head_build{head_context_.get(), "kitten_tts2.speech_head", execution_context().backend_type()};
        const auto head = modules::SliceModule({0, t.speech_end, head_rows}).build(head_build, embedding_);
        if (ggml_backend_view_init(head.tensor) != GGML_STATUS_SUCCESS) throw std::runtime_error("failed to create Kitten speech head");
        weights.lm_head = modules::LinearWeights{head,std::nullopt};
        modules::CausalDecoderRuntimeConfig runtime_config;
        runtime_config.trace_name = "kitten_tts2.lm";
        runtime_config.prefill_graph_arena_bytes = 16 * 1024 * 1024;
        runtime_config.decode_graph_arena_bytes = 16 * 1024 * 1024;
        auto & stack = runtime_config.decoder.stack;
        stack.hidden_size=hidden; stack.intermediate_size=intermediate; stack.layers=layers;
        stack.num_attention_heads=heads; stack.num_key_value_heads=kv_heads; stack.head_dim=dim;
        stack.rms_norm_eps=json::require_f32(c,"rms_norm_eps"); stack.rope_theta=json::require_f32(c,"rope_theta");
        stack.use_qk_norm=true;
        stack.runtime.attention.prefill_mode=modules::DecoderAttentionMode::FlashGrouped;
        stack.runtime.attention.static_mode=modules::DecoderAttentionMode::FlashGrouped;
        stack.runtime.static_cache.update_mode=modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        runtime_config.decoder.logits_size=head_rows;
        lm_=std::make_unique<modules::CausalDecoderRuntime>(execution_context(),runtime_config,std::move(weights));
        source.release_storage();
    }
    std::vector<float> embed(const std::vector<int32_t> & ids, const Voice & voice) {
        const auto hidden=embedding_.tensor->ne[0];
        const auto row_bytes=ggml_row_size(embedding_.tensor->type,hidden);
        assets::TensorData row;
        row.type=embedding_.tensor->type;
        row.shape=core::TensorShape::from_dims({1,hidden});
        row.bytes.resize(row_bytes);
        std::vector<float> result;
        result.reserve(ids.size()*static_cast<size_t>(hidden));
        for (size_t i=0;i<ids.size();++i) {
            if (ids[i]<0 || ids[i]>=embedding_.tensor->ne[1]) throw std::runtime_error("Kitten prompt token outside vocabulary");
            if (i==0) result.insert(result.end(),voice.speaker.begin(),voice.speaker.end());
            else {
                ggml_backend_tensor_get(embedding_.tensor,row.bytes.data(),ids[i]*embedding_.tensor->nb[1],row_bytes);
                auto values=assets::tensor_data_to_f32("kitten_tts2.embedding",row);
                result.insert(result.end(),values.begin(),values.end());
            }
        }
        return result;
    }
    std::vector<int32_t> generate(const std::string & text_value, const Voice & voice,
        const std::unordered_map<std::string,std::string> & options, std::mt19937 & rng) {
        const auto & t=assets_->tokens;
        auto encode=[this](const std::string & text){return assets_->tokenizer->tokenize(text).token_ids;};
        std::vector<int32_t> prompt{t.start,t.ref_text_start};
        auto transcript=encode(voice.transcript);
        prompt.insert(prompt.end(),transcript.begin(),transcript.end());
        prompt.insert(prompt.end(),{t.ref_text_end,t.ref_speech_start});
        for (auto code:voice.reference_codes) prompt.push_back(t.audio_base+code);
        prompt.push_back(t.ref_speech_end);
        const int64_t reference_steps=static_cast<int64_t>(prompt.size());
        static const std::regex expression(R"(\[(mundane|nervous|tender|angry|excited|stern|sad|contemplative|surprised|joyful)\]|<(pause|sigh|gasp|laugh|giggle|sob|scoff|growl|um|gulp)>|\(\(\()",std::regex::icase);
        if (std::regex_search(text_value,expression)) {
            auto emotion=encode("{emo: 1}");
            prompt.insert(prompt.end(),emotion.begin(),emotion.end());
        }
        prompt.push_back(t.text_start);
        auto target=encode(text_value);
        prompt.insert(prompt.end(),target.begin(),target.end());
        prompt.insert(prompt.end(),{t.final_seg,t.speech_start});
        const int64_t budget=runtime::parse_i64_option(options,{"max_tokens"}).value_or(
            std::max<int64_t>(200,static_cast<int64_t>(text_value.size()*25.0/20.0*1.8)));
        const auto context=json::require_i64(assets_->lm_config,"max_position_embeddings");
        if (budget<=0 || budget>8192 || budget>context-static_cast<int64_t>(prompt.size())) throw std::runtime_error("Kitten token budget exceeds context");
        const auto prefill_start=std::chrono::steady_clock::now();
        const auto required=static_cast<int64_t>(prompt.size())+std::max<int64_t>(budget,128);
        const int64_t cache_steps=std::max(lm_->decode_cache_steps(),((required+255)/256)*256);
        const int64_t keep_steps=last_voice_==&voice?reference_steps:0;
        const auto retained=lm_->retainable_prefix_steps(prompt.size(),cache_steps,128,keep_steps);
        auto embeddings=embed(prompt,voice);
        if (retained==0) {
            // Compute the reference separately even on the first request. The
            // target then uses the same graph shape on cold and warm requests,
            // avoiding quantized-kernel rounding differences for a fixed seed.
            std::vector<float> prefix(embeddings.begin(),
                embeddings.begin()+reference_steps*embedding_.tensor->ne[0]);
            lm_->prefill_embeddings_into_cache(prefix,reference_steps,cache_steps,128);
        }
        auto prefill=lm_->prefill_embeddings_into_cache(embeddings,prompt.size(),cache_steps,128,reference_steps);
        last_voice_=&voice;
        debug::timing_log_scalar("kitten_tts2.retained_prefix_steps",static_cast<double>(retained));
        debug::timing_log_scalar("kitten_tts2.prefill_ms",debug::elapsed_ms(prefill_start));
        std::string preset="stable";
        if (auto p=options.find("preset");p!=options.end()) preset=p->second;
        const auto & defaults=assets_->config.require("decode_presets").require(preset);
        sampling::HfSamplingOptions sampling_options;
        sampling_options.temperature=runtime::parse_float_option(options,{"temperature"}).value_or(json::require_f32(defaults,"temperature"));
        sampling_options.top_p=runtime::parse_float_option(options,{"top_p"}).value_or(json::require_f32(defaults,"top_p"));
        sampling_options.top_k=runtime::parse_i64_option(options,{"top_k"}).value_or(json::require_i64(defaults,"top_k"));
        sampling_options.min_p=runtime::parse_float_option(options,{"min_p"}).value_or(json::require_f32(defaults,"min_p"));
        if (!std::isfinite(sampling_options.temperature) || sampling_options.temperature < 0 ||
            !std::isfinite(sampling_options.top_p) || sampling_options.top_p <= 0 || sampling_options.top_p > 1 ||
            !std::isfinite(sampling_options.min_p) || sampling_options.min_p < 0 || sampling_options.min_p > 1 || sampling_options.top_k < 0)
            throw std::runtime_error("invalid Kitten TTS 2 sampling options");
        sampling_options.do_sample=sampling_options.temperature>0;
        if (!sampling_options.do_sample) sampling_options.temperature=1;
        const float repetition=runtime::parse_float_option(options,{"repetition_penalty"}).value_or(1.1f);
        if (!std::isfinite(repetition) || repetition < 1) throw std::runtime_error("invalid Kitten repetition_penalty");
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        std::vector<int32_t> codes;
        auto compact=[&t](const std::vector<float> & scores) {
            const auto offset=t.audio_base-t.speech_end;
            if (scores.size()!=static_cast<size_t>(offset+t.audio_count))
                throw std::runtime_error("invalid Kitten speech-head output size");
            std::vector<float> out(scores.begin()+offset,scores.end());
            out.push_back(scores[0]);
            out.push_back(scores[t.stop-t.speech_end]);
            return out;
        };
        auto logits=compact(prefill.logits);
        const auto decode_start=std::chrono::steady_clock::now();
        for (int64_t step=0;step<budget;++step) {
            if (!codes.empty()) {
                const auto last=codes.back();
                int64_t run=0;
                for (auto i=codes.rbegin();i!=codes.rend() && *i==last;++i) ++run;
                if (run>=10) logits[last]-=static_cast<float>(run-9)*1.3f;
            }
            std::unordered_set<int32_t> seen;
            for (size_t i=codes.size()>50?codes.size()-50:0;i<codes.size();++i) {
                auto code=codes[i];
                if (code==4299 || !seen.insert(code).second) continue;
                logits[code]=logits[code]<0?logits[code]*repetition:logits[code]/repetition;
            }
            auto token=sampler.sample(logits,{},sampling_options,scratch,rng,nullptr,"Kitten TTS 2");
            if (token>=t.audio_count) break;
            codes.push_back(token);
            if (step+1<budget) logits=compact(lm_->decode_token(t.audio_base+token).logits);
        }
        debug::timing_log_scalar("kitten_tts2.decode_ms",debug::elapsed_ms(decode_start));
        debug::timing_log_scalar("kitten_tts2.audio_tokens",static_cast<double>(codes.size()));
        return codes;
    }
    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unordered_map<std::string,Voice> voices_;
    std::unique_ptr<WaveformDecoder> waveform_;
    std::unique_ptr<SpeakerEncoder> speaker_;
    std::unique_ptr<models::chatterbox::S3TokenizerComponent> tokenizer_;
    std::unique_ptr<models::chatterbox::CAMPPlusEncoderComponent> camp_;
    std::unique_ptr<Voice> clone_;
    runtime::AudioBuffer reference_cache_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::shared_ptr<ggml_context> head_context_;
    core::TensorValue embedding_;
    std::unique_ptr<modules::CausalDecoderRuntime> lm_;
    const Voice * last_voice_=nullptr;
};
}
std::shared_ptr<runtime::IVoiceModelLoader> make_kitten_tts2_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family=family_name;
    config.load_assets=load_assets;
    config.create_session=[](const runtime::TaskSpec & task,const runtime::SessionOptions & options,
        std::shared_ptr<const Assets> assets,std::shared_ptr<const model_spec::ModelContract> contract){
        return std::make_unique<Session>(task,options,std::move(assets),std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}
