# Audio.cpp selective integration

This directory vendors a deliberately small source slice from
[Audio.cpp](https://github.com/0xShug0/audio.cpp). It is not a nested copy of
Audio.cpp and must not import Audio.cpp's GGML, CLI/server, model manager,
converters, tests, or unrelated model implementations.

## Sync boundary

Run the repository script against an Audio.cpp checkout or source snapshot:

```powershell
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component supertonic
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component yue2
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component fish_audio
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component maya1
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component chatterbox
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component chatterbox_turbo
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component kitten_tts2
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component soprano_tts
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component cosyvoice3
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component voxcpm2
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component voxcpm1
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component omnivoice
./scripts/sync-audiocpp.ps1 -SourceRoot C:\path\to\audio.cpp -Component yue2 -ModelSourcesOnly
```

The script is intentionally additive: it overwrites files in its explicit
manifest but does not delete anything. This makes an upstream refresh safe to
review in Git and prevents a malformed source path from deleting local adapter
or patch files. It also regenerates the small built-in model-spec catalog used
by the selected models; model specs are metadata, not weights.

`-ModelSourcesOnly` skips the full shared-foundation refresh and copies only
the requested model plus the explicit dependency closure in that component's
manifest. Some of those dependencies are shared framework files and may carry
KoboldCpp compatibility patches, so always review and reapply the patches
listed below after a component refresh.

The `upstream` subtree retains Audio.cpp's Apache-2.0 license. KoboldCpp-owned
adapters and compatibility code live outside that subtree. The imported local
snapshot reports `AUDIOCPP_VERSION=dev` and contains no `.git` metadata, so an
exact upstream commit cannot be recorded until a real checkout is supplied.

## Selected scope

- `supertonic`: offline generation through `otherarch/tts_adapter.cpp`, with
  GGUF and safetensors-package discovery, all ten built-in voices, language,
  seed, and the existing WAV/MP3 output path.
- `yue2`: generation through `otherarch/acestep/music_adapter.cpp`. The
  existing planner call stops after ABC/semantic generation and returns a
  self-contained continuation JSON; the render call consumes it and runs NAR
  plus the Oobleck VAE to WAV or MP3. This covers text-to-music and
  score/semantic-prefix continuation, which are the edit controls exposed by
  upstream YuE2. YuE2 does not accept an input waveform.
- `fish_audio`: Fish Audio S2 Pro offline TTS and voice cloning through the
  generic Audio.cpp path in `otherarch/tts_adapter.cpp`. Text, language, seed,
  inline Fish prosody tags, reference audio, and an optional reference
  transcript are supported.
- `maya1`: Maya1 offline TTS/voice design through the same generic adapter.
  KoboldCpp's `instruction` field supplies Maya1's required natural-language
  voice description.
- `chatterbox`: multilingual Chatterbox TTS and zero-shot voice cloning. A
  selected voice-directory WAV/MP3 supplies the mandatory speaker reference;
  exaggeration, guidance, and sampling controls are forwarded to Audio.cpp.
- `chatterbox_turbo`: fast English TTS using its built-in voice and distilled
  two-step S3Gen decoder. Upstream Turbo does not support custom voice cloning.
- `kitten_tts2`: multilingual Kitten TTS 2 with prepared built-in voices and
  optional reference-audio cloning. Cloning requires `reference_text`.
- `soprano_tts`: ultra-lightweight, English-only offline TTS with a fixed
  model voice. It has no voice cloning or instruction-following interface.
- `cosyvoice3`: multilingual zero-shot, cross-lingual, and
  instruction-conditioned TTS. Every request uses a reference recording;
  zero-shot mode additionally uses its transcript.
- `voxcpm2`: multilingual offline TTS, short-reference voice cloning, and
  voice design. KoboldCpp maps its instruction field to VoxCPM2's native
  parenthesized text prefix and supports reference transcripts for ultimate
  cloning.
- `voxcpm1`: the legacy 0.5B multilingual VoxCPM runtime, with offline TTS
  and short-reference cloning. It remains a separate community family because
  its GGUF tensor layout and 16 kHz output differ from VoxCPM2.
- `omnivoice`: massively multilingual offline TTS, optional short-reference
  voice cloning, and instruction-driven voice design. Reference transcripts
  are optional, and VoiceTut-TTS packages use the same runtime.
- Later candidates should be added only when their framework closure is small
  and measurable. Existing KoboldCpp Qwen3-TTS, MiniMax H3, ACE-Step, Kokoro,
  OuteTTS, Parler, Dia, and Orpheus implementations are not imported.

The MinGW default and Vulkan DLL targets compile and link with all selected
models. Runtime output still needs validation with real Soprano and
CosyVoice3 packages; weights are intentionally not vendored here.

## Adapter contract

Pass a Supertonic GGUF file, or the root of its safetensors package, as the
normal TTS model path. `custom_speaker_voice` accepts `M1` through `M5` and `F1`
through `F5`; speaker seeds 1 through 10 map to the same presets when no custom
voice is supplied.

Pass the YuE2 model GGUF as `musicllm_filename` (`--musicllm`) and its VAE GGUF
as `musicvae_filename` (`--musicvae`). KoboldCpp supplies the Qwen tokenizer
from `embd_res/yue2_qwen_tiktoken.embd`; no model-directory sidecars are
required. Music request JSON
accepts `style` (or KoboldCpp's `caption` alias), `lyrics`, `cot`, `abc`,
`semantic_prefix`, `seed`, `guidance_scale`, `num_inference_steps`, and the
`abc_*` / `semantic_*` sampling controls. Arrays such as `semantic_prefix` may
be supplied directly as JSON arrays.

Pass `fish-audio-s2-pro-q8_0.gguf` as the normal TTS model. Plain TTS requires
no other file. For cloning, select a WAV or MP3 from KoboldCpp's voice
directory. Send `reference_text` containing its transcript when available;
otherwise KoboldCpp supplies the adapter's neutral fallback transcript. The
shared `instruction` field is prepended using Fish S2 Pro's native free-form
bracket syntax, so `professional broadcast tone` becomes
`[professional broadcast tone]` before the text.

Pass `maya1-q8_0.gguf` (or `maya1-orig.gguf`) as the normal TTS model. Supply a
voice description with the existing `instruction` request field; when omitted,
KoboldCpp uses a neutral expressive-voice description. KoboldCpp terminates
unpunctuated Maya descriptions with a period because the model can otherwise
leak the description into spoken output, particularly for short utterances.

Pass `chatterbox-q8_0.gguf` as the normal TTS model and start KoboldCpp with a
voice directory. Every Chatterbox TTS request must select a WAV/MP3 reference
using the `voice` field. The model supports `language`, `exaggeration`,
`guidance_scale`, `temperature`, `top_p`, `min_p`, `repetition_penalty`, and
the common seed/output controls.

Pass `chatterbox-turbo-q8_0.gguf` as the normal TTS model. Turbo uses its
built-in English voice, so no voice directory or reference transcript is
needed. Its sampling controls include `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `max_tokens`, and `seed`.

Pass `kitten-tts2-20261021-q8_0.gguf` as the normal TTS model. `voice` selects
a case-sensitive prepared voice such as `Bruno`, `Bella`, or a language voice.
KoboldCpp's standard aliases map `kobo`, `cheery`, `sleepy`, `shouty`, and
`chatty` to Kitten voices `Frank`, `Rosie`, `Willow`, `Otis`, and `Iris`,
respectively.
Alternatively select a WAV/MP3 from the voice directory and provide its exact
transcript in `reference_text` for zero-shot cloning.

Pass `soprano-1.1-80m-q8_0.gguf` (or the BF16 variant) as the normal TTS
model. The GGUF package is self-contained. Soprano produces English speech in
one fixed voice at 32 kHz and accepts `temperature`, `top_p`,
`repetition_penalty`, `eos_bias`, `max_tokens`, and `seed`. Voice selection,
reference audio, language selection, and instructions do not alter its voice.

Pass `cosyvoice3-q8_0.gguf` (or `cosyvoice3-f32.gguf`) as the normal TTS model
and select a WAV/MP3 reference through KoboldCpp's `voice` field. No separate
RKNN files or runtime are needed. `template_name` selects `zero_shot`,
`cross_lingual`, or `instruct`. KoboldCpp selects `zero_shot` when
`reference_text` is present and `cross_lingual` when it is absent. Explicit
zero-shot requests require the exact reference transcript; cross-lingual mode
infers the language from the target text. KoboldCpp's existing `instruction`
field selects `instruct` unless an explicit template was provided. Sampling
controls include `top_k`,
`min_tokens`, `max_tokens`, `num_inference_steps`, and `seed`. Supported target
languages are Chinese, English, Japanese, Korean, German, Spanish, French,
Italian, Russian, and Cantonese.

Pass a VoxCPM2 GGUF such as `voxcpm2-q8_0.gguf` as the normal TTS model.
Plain TTS needs no reference. The existing `instruction` field enables voice
design and is wrapped in VoxCPM2's native parenthesized prefix. Selecting a
WAV/MP3 enables voice cloning; supplying its exact `reference_text` enables
the model's transcript-conditioned ultimate-clone path. Common generation and
memory controls can be sent through `audiocpp_extra_params_json`.

Pass `voxcpm-0.5b-q8_0-audiovae-f16.gguf` for VoxCPM1. It supports plain
offline TTS and reference-audio cloning; provide `reference_text` when
available. VoxCPM1 produces 16 kHz mono output and does not support VoxCPM2
voice-design instructions.

Pass an OmniVoice GGUF package as the normal TTS model. Plain synthesis needs
no reference. Selecting a WAV/MP3 enables 24 kHz voice cloning, with an
optional exact transcript in `reference_text`. KoboldCpp maps its existing
`instruction` field to OmniVoice's native `instruct` voice-design option.
Language selection and advanced diffusion, duration, chunking, and sampling
controls remain available through `audiocpp_extra_params_json`.
Streaming is intentionally not exposed because upstream implements it as
text-chunked event delivery rather than a distinct streaming model runtime.

## Intentional compatibility changes

- `otherarch/tts_adapter.cpp`: Kitten TTS 2 sessions fall back to CPU when the
  selected KoboldCpp build exposes Vulkan. Its hybrid-Q4 language model produces
  non-finite logits with ggml-vulkan. CUDA remains enabled, and explicit CPU
  operation is unchanged.
- `src/framework/modules/attention/attention_internal.h`: KoboldCpp lowers
  Audio.cpp's optional private CUDA `ggml_mul_mat_pack4` projection to the
  standard portable linear operation. This keeps upstream GGML unchanged.

CosyVoice3's RKNN source files are present only because its loader references
the optional execution-plan plumbing. `ENGINE_HAVE_RKNN` is not defined and
KoboldCpp neither includes nor links Rockchip libraries; normal CPU, Vulkan,
and CUDA operation uses the GGML flow implementation.

KoboldCpp owns GGML. Audio.cpp's bundled GGML is never synchronized.

Local changes are marked `// kcpp:` and are summarized in `patches/README.md`.
The integration disables YAML configuration in favor of JSON model packages.
YuE2's unofficial operations are expressed with standard GGML graph operations;
no Audio.cpp GGML patch is carried in KoboldCpp.
