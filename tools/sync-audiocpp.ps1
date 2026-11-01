param(
    [Parameter(Mandatory = $true)]
    [string] $SourceRoot,

    [ValidateSet("supertonic", "yue2", "fish_audio", "maya1", "chatterbox", "chatterbox_turbo", "kitten_tts2", "soprano_tts", "cosyvoice3", "pocket_tts", "vibevoice", "voxcpm2", "voxcpm1", "omnivoice", "breeze_tts", "ming3_tts", "neutts", "qwen3_tts", "all")]
    [string[]] $Component = @("all"),

    [switch] $ModelSourcesOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$vendorRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "otherarch/audiocpp/upstream"))
$sourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "CMakeLists.txt") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $sourceRoot "include/engine/framework") -PathType Container)) {
    throw "SourceRoot does not look like an Audio.cpp checkout: $sourceRoot"
}

function Copy-AudioCppFile([string] $relativePath) {
    $source = Join-Path $sourceRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Audio.cpp source file is missing: $relativePath"
    }

    $destination = Join-Path $vendorRoot $relativePath
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

function Copy-AudioCppTree([string] $relativePath) {
    $source = Join-Path $sourceRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Audio.cpp source directory is missing: $relativePath"
    }

    Get-ChildItem -LiteralPath $source -Recurse -File | ForEach-Object {
        $sourcePrefix = $sourceRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
        $relativeFile = $_.FullName.Substring($sourcePrefix.Length)
        Copy-AudioCppFile $relativeFile
    }
}

$foundationFiles = @(
    "LICENSE",
    "external/cJSON/LICENSE",
    "external/cJSON/cJSON.c",
    "external/cJSON/cJSON.h",
    "src/framework/assets/resource_bundle.cpp",
    "src/framework/assets/tensor_source.cpp",
    "src/framework/assets/weight_adapter.cpp",
    "src/framework/assets/weight_adapter_source.cpp",
    "src/framework/assets/weight_norm_folding.cpp",
    "src/framework/model_spec/package.cpp",
    "src/framework/model_spec/options.cpp",
    "src/framework/model_spec/schema.cpp",
    "src/framework/core/backend.cpp",
    "src/framework/core/deferred_tensor_writer.cpp",
    "src/framework/core/execution_context.cpp",
    "src/framework/core/load_progress.cpp",
    "src/framework/core/module.cpp",
    "src/framework/runtime/artifacts.cpp",
    "src/framework/runtime/cache.cpp",
    "src/framework/runtime/graph_executor.cpp",
    "src/framework/runtime/model.cpp",
    "src/framework/runtime/options.cpp",
    "src/framework/runtime/session.cpp",
    "src/framework/runtime/session_base.cpp",
    "src/framework/runtime/task_vocabulary.cpp",
    "src/framework/runtime/workspace.cpp",
    "src/framework/io/binary.cpp",
    "src/framework/io/config.cpp",
    "src/framework/io/filesystem.cpp",
    "src/framework/io/json.cpp",
    "src/framework/io/safetensors.cpp",
    "src/framework/io/text.cpp",
    "src/framework/debug/profiler.cpp",
    "src/framework/debug/trace.cpp",
    "src/framework/text/chunking.cpp",
    "src/framework/text/unicode_normalization.cpp",
    "src/framework/text/unicode_nfkd_data.inc",
    "src/framework/modules/activation_modules.cpp",
    "src/framework/modules/conv_modules.cpp",
    "src/framework/modules/linear_module.cpp",
    "src/framework/modules/lookup_modules.cpp",
    "src/framework/modules/norm_modules.cpp",
    "src/framework/modules/packed_linear_weights.cpp",
    "src/framework/modules/primitive_modules.cpp",
    "src/framework/modules/streaming_conv_modules.cpp",
    "src/framework/modules/structural_modules.cpp",
    "src/framework/modules/tensor_layout_utils.h"
)

$supertonicFiles = @(
    "model_specs/supertonic.json",
    "src/models/supertonic/assets.cpp",
    "src/models/supertonic/loader.cpp",
    "src/models/supertonic/runtime.cpp",
    "src/models/supertonic/session.cpp",
    "src/models/supertonic/tokenizer_text.cpp"
)

$fishAudioFiles = @(
    "model_specs/fish_audio.json",
    "src/framework/audio/conversion.cpp",
    "src/framework/audio/resampling.cpp",
    "src/framework/audio/wav_reader.cpp",
    "src/framework/codecs/fish_dac_codec_runtime.cpp",
    "src/framework/core/host_memory.cpp",
    "src/framework/modules/attention/streaming_frame_attention_cache.cpp",
    "src/framework/modules/conditioning_modules.cpp",
    "src/models/fish_audio/ar.cpp",
    "src/models/fish_audio/assets.cpp",
    "src/models/fish_audio/generator.cpp",
    "src/models/fish_audio/loader.cpp",
    "src/models/fish_audio/prompt_builder.cpp",
    "src/models/fish_audio/session.cpp",
    "src/models/fish_audio/tokenizer_text.cpp"
)

$maya1Files = @(
    "model_specs/maya1.json",
    "src/framework/codecs/snac_decoder_runtime.cpp",
    "src/framework/model_spec/metadata.cpp",
    "src/models/maya1/assets.cpp",
    "src/models/maya1/generator.cpp",
    "src/models/maya1/session.cpp",
    "src/models/maya1/tokenizer.cpp"
)

# Chatterbox, Chatterbox Turbo, and Kitten TTS 2 share the S3Gen decoder,
# vocoder, reference-audio frontend, and most of the autoregressive runtime.
# Keep that closure explicit so each family can still be refreshed separately.
$s3genClusterFiles = @(
    "src/framework/audio/conversion.cpp",
    "src/framework/audio/dsp.cpp",
    "src/framework/audio/detail/speech_fft_internal.h",
    "src/framework/audio/fft.cpp",
    "src/framework/audio/istft_graph.cpp",
    "src/framework/audio/kaldi_fbank.cpp",
    "src/framework/audio/mel_spectrogram_frontend.cpp",
    "src/framework/audio/reference_audio_frontend.cpp",
    "src/framework/audio/resampling.cpp",
    "src/framework/audio/waveform_ops.cpp",
    "src/framework/audio/wav_reader.cpp",
    "src/framework/codecs/s3gen_runtime.cpp",
    "src/framework/core/attention_fallback.cpp",
    "src/framework/modules/attention/grouped_query_attention.cpp",
    "src/framework/modules/attention/attention_internal.h",
    "src/framework/modules/attention/positional_encoding.cpp",
    "src/framework/modules/attention/scaled_dot_product_attention.cpp",
    "src/framework/modules/module_internal.h",
    "src/framework/modules/optimizations/fast_kv_modules.cpp",
    "src/framework/modules/positional_modules.cpp",
    "src/framework/modules/speaker_encoders/campplus_encoder.cpp",
    "src/framework/modules/transformers/causal_decoder.cpp",
    "src/framework/modules/transformers/causal_decoder_runtime.cpp",
    "src/framework/modules/vocoders/hift_vocoder.cpp",
    "src/framework/runtime/bounded_static_kv_decode.cpp",
    "src/framework/runtime/graph_executor.cpp",
    "src/framework/runtime/host_ops.cpp",
    "src/framework/runtime/kv_cache.cpp",
    "src/framework/sampling/hf_sampler.cpp",
    "src/framework/sampling/torch_random.cpp",
    "src/framework/tokenizers/llama_bpe.cpp",
    "src/framework/tokenizers/qwen_bpe_bundle.cpp"
)

$chatterboxFiles = @(
    "model_specs/chatterbox.json",
    "src/models/chatterbox/assets.cpp",
    "src/models/chatterbox/audio_features.cpp",
    "src/models/chatterbox/audio_processing.cpp",
    "src/models/chatterbox/campplus_encoder_impl.cpp",
    "src/models/chatterbox/component_weights.cpp",
    "src/models/chatterbox/conditionals.cpp",
    "src/models/chatterbox/loader.cpp",
    "src/models/chatterbox/s3_tokenizer_impl.cpp",
    "src/models/chatterbox/session.cpp",
    "src/models/chatterbox/t3_component.cpp",
    "src/models/chatterbox/t3_weights.cpp",
    "src/models/chatterbox/text_tokenizer.cpp",
    "src/models/chatterbox/tts.cpp",
    "src/models/chatterbox/vc.cpp",
    "src/models/chatterbox/voice_encoder_impl.cpp",
    "src/models/chatterbox/components/audio_processing.h",
    "src/models/chatterbox/components/component_weights.h",
    "src/models/chatterbox/components/t3_runtime.h"
)

$chatterboxTurboFiles = @(
    "model_specs/chatterbox_turbo.json",
    "src/community_models/chatterbox_turbo/assets.cpp",
    "src/community_models/chatterbox_turbo/loader.cpp",
    "src/community_models/chatterbox_turbo/s3gen_turbo.cpp",
    "src/community_models/chatterbox_turbo/session.cpp",
    "src/community_models/chatterbox_turbo/t3_turbo_component.cpp",
    "src/community_models/chatterbox_turbo/t3_turbo_weights.cpp",
    "src/community_models/chatterbox_turbo/text_tokenizer_turbo.cpp",
    "src/community_models/chatterbox_turbo/tts.cpp",
    "src/community_models/chatterbox_turbo/components/t3_turbo_runtime.h"
)

$kittenTts2Files = @(
    "model_specs/kitten_tts2.json",
    "src/community_models/kitten_tts2/decoder.cpp",
    "src/community_models/kitten_tts2/session.cpp",
    "src/community_models/kitten_tts2/speaker.cpp"
)

$sopranoTtsFiles = @(
    "model_specs/soprano_tts.json",
    "src/framework/text/text_normalization.cpp",
    "src/community_models/soprano_tts/assets.cpp",
    "src/community_models/soprano_tts/generator.cpp",
    "src/community_models/soprano_tts/session.cpp",
    "src/community_models/soprano_tts/soprano_unidecode_table.inc",
    "src/community_models/soprano_tts/text_normalizer.cpp",
    "src/community_models/soprano_tts/tokenizer_text.cpp",
    "src/community_models/soprano_tts/vocoder.cpp"
)

# CosyVoice3 shares Audio.cpp's causal decoder, CAMPPlus, HiFT, FFT, and
# tokenizer infrastructure with the S3Gen cluster. The RKNN units below are
# dependency-free stubs unless ENGINE_HAVE_RKNN is explicitly enabled; KCPP
# does not enable or link Rockchip's runtime.
$cosyVoice3Files = @(
    "model_specs/cosyvoice3.json",
    "src/framework/audio/conversion.cpp",
    "src/framework/audio/dsp.cpp",
    "src/framework/audio/detail/speech_fft_internal.h",
    "src/framework/audio/fft.cpp",
    "src/framework/audio/istft_graph.cpp",
    "src/framework/audio/kaldi_fbank.cpp",
    "src/framework/audio/resampling.cpp",
    "src/framework/core/attention_fallback.cpp",
    "src/framework/modules/attention/grouped_query_attention.cpp",
    "src/framework/modules/attention/attention_internal.h",
    "src/framework/modules/attention/positional_encoding.cpp",
    "src/framework/modules/attention/projected_grouped_self_attention.cpp",
    "src/framework/modules/attention/scaled_dot_product_attention.cpp",
    "src/framework/modules/feed_forward_modules.cpp",
    "src/framework/modules/module_internal.h",
    "src/framework/modules/optimizations/fast_kv_modules.cpp",
    "src/framework/modules/positional_modules.cpp",
    "src/framework/modules/speaker_encoders/campplus_encoder.cpp",
    "src/framework/modules/speech_encoders/s3_tokenizer.cpp",
    "src/framework/modules/transformers/causal_decoder.cpp",
    "src/framework/modules/transformers/causal_decoder_runtime.cpp",
    "src/framework/modules/vocoders/hift_vocoder.cpp",
    "src/framework/rknn/engine.cpp",
    "src/framework/rknn/engine_resources.cpp",
    "src/framework/runtime/bounded_static_kv_decode.cpp",
    "src/framework/runtime/graph_executor.cpp",
    "src/framework/runtime/host_ops.cpp",
    "src/framework/runtime/kv_cache.cpp",
    "src/framework/sampling/hf_sampler.cpp",
    "src/framework/sampling/torch_random.cpp",
    "src/framework/tokenizers/llama_bpe.cpp",
    "src/models/cosyvoice3/ar.cpp",
    "src/models/cosyvoice3/assets.cpp",
    "src/models/cosyvoice3/flow.cpp",
    "src/models/cosyvoice3/frontend.cpp",
    "src/models/cosyvoice3/hift.cpp",
    "src/models/cosyvoice3/rknn_flow.cpp",
    "src/models/cosyvoice3/session.cpp",
    "src/models/cosyvoice3/tokenizer_text.cpp"
)

# PocketTTS uses SentencePiece's inference API and libyaml. Keep the imported
# dependency slices explicit: in particular, do not synchronize SentencePiece's
# 40 MB data directory, trainer, command-line tools, tests, or benchmarks.
$sentencePieceInferenceFiles = @(
    "external/sentencepiece/LICENSE",
    "external/sentencepiece/src/bpe_model.h",
    "external/sentencepiece/src/common.h",
    "external/sentencepiece/src/normalizer.h",
    "external/sentencepiece/src/util.h",
    "external/sentencepiece/src/freelist.h",
    "external/sentencepiece/src/filesystem.h",
    "external/sentencepiece/src/init.h",
    "external/sentencepiece/src/sentencepiece_processor.h",
    "external/sentencepiece/src/word_model.h",
    "external/sentencepiece/src/model_factory.h",
    "external/sentencepiece/src/char_model.h",
    "external/sentencepiece/src/model_interface.h",
    "external/sentencepiece/src/unigram_model.h",
    "external/sentencepiece/src/bpe_model.cc",
    "external/sentencepiece/src/char_model.cc",
    "external/sentencepiece/src/error.cc",
    "external/sentencepiece/src/filesystem.cc",
    "external/sentencepiece/src/model_factory.cc",
    "external/sentencepiece/src/model_interface.cc",
    "external/sentencepiece/src/normalizer.cc",
    "external/sentencepiece/src/sentencepiece_processor.cc",
    "external/sentencepiece/src/unigram_model.cc",
    "external/sentencepiece/src/util.cc",
    "external/sentencepiece/src/word_model.cc",
    "external/sentencepiece/src/builtin_pb/sentencepiece.pb.h",
    "external/sentencepiece/src/builtin_pb/sentencepiece.pb.cc",
    "external/sentencepiece/src/builtin_pb/sentencepiece_model.pb.h",
    "external/sentencepiece/src/builtin_pb/sentencepiece_model.pb.cc"
)

$pocketTtsFiles = @(
    "model_specs/pocket_tts.json",
    "src/framework/io/yaml.cpp",
    "src/framework/modules/attention/cross_attention.cpp",
    "src/framework/modules/attention/self_attention.cpp",
    "src/framework/modules/transformers/transformer_blocks.cpp",
    "src/framework/tokenizers/sentencepiece.cpp",
    "src/models/pocket_tts/assets.cpp",
    "src/models/pocket_tts/backend_weights.cpp",
    "src/models/pocket_tts/acoustic_model.cpp",
    "src/models/pocket_tts/audio_decoder.cpp",
    "src/models/pocket_tts/flow_lm.cpp",
    "src/models/pocket_tts/graph_common.h",
    "src/models/pocket_tts/mimi_decoder.cpp",
    "src/models/pocket_tts/mimi_encoder.cpp",
    "src/models/pocket_tts/text_conditioner.cpp",
    "src/models/pocket_tts/voice_state_assets.cpp",
    "src/models/pocket_tts/voice_conditioner.cpp",
    "src/models/pocket_tts/loader.cpp",
    "src/models/pocket_tts/session.cpp"
)

$vibeVoiceFiles = @(
    "model_specs/vibevoice.json",
    "src/models/vibevoice/assets.cpp",
    "src/models/vibevoice/connector.cpp",
    "src/models/vibevoice/decoder.cpp",
    "src/models/vibevoice/diffusion_head.cpp",
    "src/models/vibevoice/diffusion_sampler.cpp",
    "src/models/vibevoice/generator.cpp",
    "src/models/vibevoice/loader.cpp",
    "src/models/vibevoice/lora.cpp",
    "src/models/vibevoice/scheduler.cpp",
    "src/models/vibevoice/session.cpp",
    "src/models/vibevoice/tokenizer_audio.cpp",
    "src/models/vibevoice/tokenizer_text.cpp"
)

# BreezeTTS 2's framework closure is already shared by the selected TTS
# models. Keep the model implementation intact, including its offline-only
# use through KoboldCpp and the upstream streaming implementation.
$breezeTtsFiles = @(
    "model_specs/breeze_tts.json",
    "src/framework/modules/text_encoders/t5_gemma_encoder.cpp",
    "src/models/breeze_tts/assets.cpp",
    "src/models/breeze_tts/generator.cpp",
    "src/models/breeze_tts/session.cpp",
    "src/models/breeze_tts/speech_decoder.cpp",
    "src/models/breeze_tts/speech_encoder.cpp",
    "src/models/breeze_tts/text_encoder.cpp",
    "src/models/breeze_tts/tokenizer_text.cpp"
)

# Ming 3.0 TTS 0.5B adds its AudioVAE and flow decoder runtimes. CAMPPlus,
# reference-audio preparation, the Qwen decoder, and the remaining framework
# closure are already synchronized for other selected models.
$ming3TtsFiles = @(
    "model_specs/ming3_tts.json",
    "src/framework/codecs/ming_audio_vae_runtime.cpp",
    "src/framework/modules/speech_decoders/ming_flow_runtime.cpp",
    "src/models/ming3_tts/assets.cpp",
    "src/models/ming3_tts/generator.cpp",
    "src/models/ming3_tts/prompt.cpp",
    "src/models/ming3_tts/session.cpp"
)

# NeuTTS 2E adds a compact Qwen3-style AR model and NeuCodec decoder. The
# decoder's reusable FSQ runtime is the only shared implementation not already
# synchronized by the selected TTS foundation.
$neuTtsFiles = @(
    "model_specs/neutts.json",
    "src/framework/codecs/fsq_audio_codec_runtime.cpp",
    "src/models/neutts/ar.cpp",
    "src/models/neutts/assets.cpp",
    "src/models/neutts/backbone.cpp",
    "src/models/neutts/codec.cpp",
    "src/models/neutts/prompt.cpp",
    "src/models/neutts/session.cpp"
)

# Qwen3-TTS uses the causal decoder, attention, audio frontend, sampling, and
# tokenizer infrastructure already synchronized by the selected TTS models.
# One loader handles Base, CustomVoice, and VoiceDesign packages by reading the
# checkpoint's embedded tts_model_type.
$qwen3TtsFiles = @(
    "model_specs/qwen3_tts.json",
    "src/models/qwen3_tts/assets.cpp",
    "src/models/qwen3_tts/tokenizer_text.cpp",
    "src/models/qwen3_tts/tokenizer_speech_encoder.cpp",
    "src/models/qwen3_tts/tokenizer_speech_decoder.cpp",
    "src/models/qwen3_tts/rknn_speech_decoder.cpp",
    "src/models/qwen3_tts/speaker_encoder.cpp",
    "src/models/qwen3_tts/talker.cpp",
    "src/models/qwen3_tts/prompt_tts_voice_clone.cpp",
    "src/models/qwen3_tts/prompt_tts_voice_design.cpp",
    "src/models/qwen3_tts/prompt_tts_custom_voice.cpp",
    "src/models/qwen3_tts/session.cpp",
    "src/models/qwen3_tts/loader.cpp"
)

# VoxCPM2 and the legacy VoxCPM1 implementation share the MiniCPM/AudioVAE
# shape, but remain separate upstream families and model formats.
$voxCpmSharedFiles = @(
    "src/framework/audio/conversion.cpp",
    "src/framework/audio/resampling.cpp",
    "src/framework/audio/waveform_ops.cpp",
    "src/framework/modules/optimizations/fast_kv_modules.cpp",
    "src/framework/runtime/kv_cache.cpp",
    "src/framework/sampling/torch_random.cpp"
)

$voxCpm2Files = @(
    "model_specs/voxcpm2.json",
    "src/models/voxcpm2/assets.cpp",
    "src/models/voxcpm2/audiovae.cpp",
    "src/models/voxcpm2/generator.cpp",
    "src/models/voxcpm2/loader.cpp",
    "src/models/voxcpm2/minicpm.cpp",
    "src/models/voxcpm2/minicpm_blocks.h",
    "src/models/voxcpm2/session.cpp",
    "src/models/voxcpm2/tokenizer_text.cpp"
)

$voxCpm1Files = @(
    "model_specs/voxcpm1.json",
    "src/framework/model_spec/metadata.cpp",
    "src/framework/text/chinese_variant.cpp",
    "src/framework/text/chinese_variant_data.inc",
    "src/community_models/voxcpm1/assets.cpp",
    "src/community_models/voxcpm1/audiovae.cpp",
    "src/community_models/voxcpm1/generator.cpp",
    "src/community_models/voxcpm1/minicpm.cpp",
    "src/community_models/voxcpm1/minicpm_blocks.h",
    "src/community_models/voxcpm1/session.cpp",
    "src/community_models/voxcpm1/tokenizer_text.cpp"
)

# OmniVoice's Higgs-derived audio tokenizer and diffusion generator are
# self-contained model code. Their shared framework closure is already kept by
# the selected TTS models, so do not refresh KoboldCpp-patched foundation files
# when synchronizing this component alone.
$omniVoiceFiles = @(
    "model_specs/omnivoice.json",
    "src/models/omnivoice/assets.cpp",
    "src/models/omnivoice/audio_tokenizer.cpp",
    "src/models/omnivoice/generator.cpp",
    "src/models/omnivoice/generator_layerwise.inc",
    "src/models/omnivoice/language_map.inc",
    "src/models/omnivoice/loader.cpp",
    "src/models/omnivoice/postprocess.cpp",
    "src/models/omnivoice/prompt_builder.cpp",
    "src/models/omnivoice/session.cpp",
    "src/models/omnivoice/tokenizer_text.cpp"
)

# YuE2 is intentionally a separate import tier. Its framework closure is larger
# than Supertonic's, but it still avoids Audio.cpp's server, CLI, converters,
# unrelated models, bundled GGML, sentencepiece, and YAML dependencies.
$yue2Files = @(
    "model_specs/yue2.json",
    "src/framework/core/attention_fallback.cpp",
    "src/framework/runtime/bounded_static_kv_decode.cpp",
    "src/framework/runtime/graph_executor.cpp",
    "src/framework/runtime/graph_optimizer.cpp",
    "src/framework/runtime/greedy_causal_decoder.cpp",
    "src/framework/runtime/host_ops.cpp",
    "src/framework/runtime/kv_cache.cpp",
    "src/framework/modules/attention/grouped_query_attention.cpp",
    "src/framework/modules/attention/attention_internal.h",
    "src/framework/modules/module_internal.h",
    "src/framework/modules/attention/positional_encoding.cpp",
    "src/framework/modules/attention/scaled_dot_product_attention.cpp",
    "src/framework/modules/positional_modules.cpp",
    "src/framework/modules/optimizations/fast_kv_modules.cpp",
    "src/framework/modules/transformers/causal_decoder.cpp",
    "src/framework/modules/transformers/causal_decoder_runtime.cpp",
    "src/framework/modules/transformers/decoder.cpp",
    "src/framework/tokenizers/llama_bpe.cpp",
    "src/framework/codecs/oobleck_audio_vae_runtime.cpp",
    "src/framework/sampling/greedy_decode.cpp",
    "src/framework/sampling/hf_sampler.cpp",
    "src/framework/sampling/torch_random.cpp",
    "src/models/yue2/ar_runtime.cpp",
    "src/models/yue2/assets.cpp",
    "src/models/yue2/lora.cpp",
    "src/models/yue2/nar_runtime.cpp",
    "src/models/yue2/pipeline.cpp",
    "src/models/yue2/request.cpp",
    "src/models/yue2/session.cpp",
    "src/models/yue2/tokenizer_text.cpp",
    "src/models/yue2/types.cpp"
)

$selected = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($name in $Component) {
    if ($name -eq "all") {
        [void] $selected.Add("supertonic")
        [void] $selected.Add("yue2")
        [void] $selected.Add("fish_audio")
        [void] $selected.Add("maya1")
        [void] $selected.Add("chatterbox")
        [void] $selected.Add("chatterbox_turbo")
        [void] $selected.Add("kitten_tts2")
        [void] $selected.Add("soprano_tts")
        [void] $selected.Add("cosyvoice3")
        [void] $selected.Add("pocket_tts")
        [void] $selected.Add("vibevoice")
        [void] $selected.Add("voxcpm2")
        [void] $selected.Add("voxcpm1")
        [void] $selected.Add("omnivoice")
        [void] $selected.Add("breeze_tts")
        [void] $selected.Add("ming3_tts")
        [void] $selected.Add("neutts")
        [void] $selected.Add("qwen3_tts")
    } else {
        [void] $selected.Add($name)
    }
}

# Upstream Kitten TTS 2 reuses Chatterbox's S3 tokenizer, CAMPPlus encoder,
# and reference-conditioning implementation rather than duplicating them.
if ($selected.Contains("kitten_tts2")) {
    [void] $selected.Add("chatterbox")
}

if ($selected.Contains("pocket_tts")) {
    Copy-AudioCppTree "include/engine/models/pocket_tts"
    Copy-AudioCppTree "external/libyaml/include"
    Copy-AudioCppTree "external/libyaml/src"
    Copy-AudioCppFile "external/libyaml/LICENSE"
    Copy-AudioCppTree "external/sentencepiece/third_party/absl"
    Copy-AudioCppTree "external/sentencepiece/third_party/darts_clone"
    Copy-AudioCppTree "external/sentencepiece/third_party/protobuf-lite"
    foreach ($relativePath in $sentencePieceInferenceFiles) {
        Copy-AudioCppFile $relativePath
    }
    $sentencePieceConfig = @"
#ifndef CONFIG_H_
#define CONFIG_H_
#define VERSION "0.2.1"
#define PACKAGE "sentencepiece"
#define PACKAGE_STRING "sentencepiece"
#define INSTALL_DATADIR ""
#endif
"@
    [System.IO.File]::WriteAllText(
        (Join-Path $vendorRoot "external/sentencepiece/config.h"),
        $sentencePieceConfig,
        (New-Object System.Text.UTF8Encoding($false)))
    foreach ($relativePath in $pocketTtsFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("vibevoice")) {
    Copy-AudioCppTree "include/engine/models/vibevoice"
    foreach ($relativePath in $vibeVoiceFiles) {
        Copy-AudioCppFile $relativePath
    }
}

$hasVoxCpm = $selected.Contains("voxcpm2") -or $selected.Contains("voxcpm1")
if ($hasVoxCpm) {
    foreach ($relativePath in $voxCpmSharedFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("voxcpm2")) {
    Copy-AudioCppTree "include/engine/models/voxcpm2"
    foreach ($relativePath in $voxCpm2Files) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("voxcpm1")) {
    Copy-AudioCppTree "include/engine/community_models/voxcpm1"
    foreach ($relativePath in $voxCpm1Files) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("omnivoice")) {
    Copy-AudioCppTree "include/engine/models/omnivoice"
    foreach ($relativePath in $omniVoiceFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("breeze_tts")) {
    Copy-AudioCppTree "include/engine/models/breeze_tts"
    foreach ($relativePath in $breezeTtsFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("ming3_tts")) {
    Copy-AudioCppTree "include/engine/models/ming3_tts"
    foreach ($relativePath in $ming3TtsFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("neutts")) {
    Copy-AudioCppTree "include/engine/models/neutts"
    foreach ($relativePath in $neuTtsFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("qwen3_tts")) {
    Copy-AudioCppTree "include/engine/models/qwen3_tts"
    foreach ($relativePath in $qwen3TtsFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("soprano_tts")) {
    Copy-AudioCppTree "include/engine/community_models/soprano_tts"
    foreach ($relativePath in $sopranoTtsFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("cosyvoice3")) {
    Copy-AudioCppTree "include/engine/models/cosyvoice3"
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.cpp"
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.h"
    Copy-AudioCppFile "external/llama_tokenizer/unicode.h"
    foreach ($relativePath in $cosyVoice3Files) {
        Copy-AudioCppFile $relativePath
    }
}

# Headers are inexpensive and are kept as one upstream-owned API surface. The
# model-only mode is useful after local compatibility patches have been applied
# to an already synchronized foundation.
if (-not $ModelSourcesOnly) {
    Copy-AudioCppTree "include/engine/framework"
    foreach ($relativePath in $foundationFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("supertonic")) {
    Copy-AudioCppTree "include/engine/models/supertonic"
    foreach ($relativePath in $supertonicFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("yue2")) {
    Copy-AudioCppTree "include/engine/models/yue2"
    # Reuse KoboldCpp's newer unicode implementation. Audio.cpp's tokenizer
    # only contributes the BPE core and its matching public declarations.
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.cpp"
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.h"
    Copy-AudioCppFile "external/llama_tokenizer/unicode.h"
    foreach ($relativePath in $yue2Files) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("fish_audio")) {
    Copy-AudioCppTree "include/engine/models/fish_audio"
    foreach ($relativePath in $fishAudioFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("maya1")) {
    Copy-AudioCppTree "include/engine/models/maya1"
    foreach ($relativePath in $maya1Files) {
        Copy-AudioCppFile $relativePath
    }
}

$hasS3genCluster = $selected.Contains("chatterbox") -or
    $selected.Contains("chatterbox_turbo") -or
    $selected.Contains("kitten_tts2")
if ($hasS3genCluster) {
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.cpp"
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.h"
    Copy-AudioCppFile "external/llama_tokenizer/unicode.h"
    foreach ($relativePath in $s3genClusterFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("chatterbox")) {
    Copy-AudioCppTree "include/engine/models/chatterbox"
    foreach ($relativePath in $chatterboxFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("chatterbox_turbo")) {
    Copy-AudioCppTree "include/engine/community_models/chatterbox_turbo"
    foreach ($relativePath in $chatterboxTurboFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("kitten_tts2")) {
    Copy-AudioCppTree "include/engine/community_models/kitten_tts2"
    foreach ($relativePath in $kittenTts2Files) {
        Copy-AudioCppFile $relativePath
    }
}

$generatedDirectory = Join-Path $vendorRoot "generated"
New-Item -ItemType Directory -Force -Path $generatedDirectory | Out-Null

function ConvertTo-AudioCppSpecLiterals {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Json
    )

    # MSVC limits each string literal to 16,380 single-byte characters. Keep
    # each adjacent raw-string chunk comfortably below that limit even after a
    # Windows checkout expands LF to CRLF or a future spec gains UTF-8 text.
    $maxChunkBytes = 12000
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    $remaining = $Json
    $literals = New-Object System.Collections.Generic.List[string]
    while ($utf8.GetByteCount($remaining) -gt $maxChunkBytes) {
        $cut = [Math]::Min($remaining.Length, $maxChunkBytes)
        while ($cut -gt 0 -and $utf8.GetByteCount($remaining.Substring(0, $cut)) -gt $maxChunkBytes) {
            $cut--
        }
        if ($cut -le 0) {
            throw "Unable to split model spec into an MSVC-safe string literal"
        }
        $newline = $remaining.LastIndexOf("`n", $cut - 1)
        if ($newline -ge 0) {
            $cut = $newline + 1
        }
        $chunk = $remaining.Substring(0, $cut)
        [void] $literals.Add("R`"AUDIOCPP_SPEC($chunk)AUDIOCPP_SPEC`"")
        $remaining = $remaining.Substring($cut)
    }
    [void] $literals.Add("R`"AUDIOCPP_SPEC($remaining)AUDIOCPP_SPEC`"")
    return [string]::Join(" ", $literals)
}

$modelSpecEntries = ""
Get-ChildItem -LiteralPath (Join-Path $vendorRoot "model_specs") -Filter "*.json" -File |
    Sort-Object Name |
    ForEach-Object {
        $family = $_.BaseName
        $json = [System.IO.File]::ReadAllText($_.FullName)
        if ($json.Contains(")AUDIOCPP_SPEC`"")) {
            throw "Model spec contains the reserved C++ raw-string delimiter: $($_.FullName)"
        }
        $literals = ConvertTo-AudioCppSpecLiterals -Json $json
        $modelSpecEntries += "    {`"$family`", $literals},`n"
    }
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $generatedDirectory "model_specs.inc"),
    $modelSpecEntries,
    $utf8WithoutBom)

# KoboldCpp uses a small host-owned registry rather than Audio.cpp's full
# server registry. Generate it from the model slices that are actually present
# so tts_adapter.cpp remains family-agnostic as more selected models are added.
$ttsLoaders = @(
    @{
        Family = "cosyvoice3"
        Header = "engine/models/cosyvoice3/session.h"
        Factory = "engine::models::cosyvoice3::make_cosyvoice3_loader"
    },
    @{
        Family = "chatterbox"
        Header = "engine/models/chatterbox/loader.h"
        Factory = "engine::models::chatterbox::make_chatterbox_loader"
    },
    @{
        Family = "chatterbox_turbo"
        Header = "engine/community_models/chatterbox_turbo/loader.h"
        Factory = "engine::community_models::chatterbox_turbo::make_chatterbox_turbo_loader"
    },
    @{
        Family = "fish_audio"
        Header = "engine/models/fish_audio/loader.h"
        Factory = "engine::models::fish_audio::make_fish_audio_loader"
    },
    @{
        Family = "kitten_tts2"
        Header = "engine/community_models/kitten_tts2/session.h"
        Factory = "engine::community_models::kitten_tts2::make_kitten_tts2_loader"
    },
    @{
        Family = "maya1"
        Header = "engine/models/maya1/session.h"
        Factory = "engine::models::maya1::make_maya1_loader"
    },
    @{
        Family = "pocket_tts"
        Header = "engine/models/pocket_tts/loader.h"
        Factory = "engine::models::pocket_tts::make_pocket_tts_loader"
    },
    @{
        Family = "soprano_tts"
        Header = "engine/community_models/soprano_tts/session.h"
        Factory = "engine::community_models::soprano_tts::make_soprano_tts_loader"
    },
    @{
        Family = "supertonic"
        Header = "engine/models/supertonic/loader.h"
        Factory = "engine::models::supertonic::make_supertonic_loader"
    },
    @{
        Family = "vibevoice"
        Header = "engine/models/vibevoice/loader.h"
        Factory = "engine::models::vibevoice::make_vibevoice_loader"
    },
    @{
        Family = "voxcpm2"
        Header = "engine/models/voxcpm2/loader.h"
        Factory = "engine::models::voxcpm2::make_voxcpm2_loader"
    },
    @{
        Family = "voxcpm1"
        Header = "engine/community_models/voxcpm1/session.h"
        Factory = "engine::community_models::voxcpm1::make_voxcpm1_loader"
    },
    @{
        Family = "omnivoice"
        Header = "engine/models/omnivoice/loader.h"
        Factory = "engine::models::omnivoice::make_omnivoice_loader"
    },
    @{
        Family = "breeze_tts"
        Header = "engine/models/breeze_tts/session.h"
        Factory = "engine::models::breeze_tts::make_breeze_tts_loader"
    },
    @{
        Family = "ming3_tts"
        Header = "engine/models/ming3_tts/session.h"
        Factory = "engine::models::ming3_tts::make_ming3_tts_loader"
    },
    @{
        Family = "neutts"
        Header = "engine/models/neutts/session.h"
        Factory = "engine::models::neutts::make_neutts_loader"
    },
    @{
        Family = "qwen3_tts"
        Header = "engine/models/qwen3_tts/loader.h"
        Factory = "engine::models::qwen3_tts::make_qwen3_tts_loader"
    }
)
$ttsRegistryHeaders = ""
$ttsRegistryLoaders = ""
foreach ($loader in $ttsLoaders) {
    $headerPath = Join-Path $vendorRoot ("include/" + $loader.Header)
    $specPath = Join-Path $vendorRoot ("model_specs/" + $loader.Family + ".json")
    if ((Test-Path -LiteralPath $headerPath -PathType Leaf) -and
        (Test-Path -LiteralPath $specPath -PathType Leaf)) {
        $ttsRegistryHeaders += "#include `"$($loader.Header)`"`n"
        $ttsRegistryLoaders += "    {`"$($loader.Family)`", $($loader.Factory)()},`n"
    }
}
[System.IO.File]::WriteAllText(
    (Join-Path $generatedDirectory "kcpp_tts_registry_headers.inc"),
    $ttsRegistryHeaders,
    $utf8WithoutBom)
[System.IO.File]::WriteAllText(
    (Join-Path $generatedDirectory "kcpp_tts_registry_loaders.inc"),
    $ttsRegistryLoaders,
    $utf8WithoutBom)

Write-Host "Synchronized Audio.cpp components [$($selected -join ', ')] into $vendorRoot"
Write-Host "Reapply and review files in otherarch/audiocpp/patches after every synchronization."
