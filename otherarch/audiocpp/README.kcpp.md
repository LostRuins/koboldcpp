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
  inline Fish prosody tags, reference audio, and its required reference
  transcript are supported.
- `maya1`: Maya1 offline TTS/voice design through the same generic adapter.
  KoboldCpp's `instruction` field supplies Maya1's required natural-language
  voice description.
- Later candidates should be added only when their framework closure is small
  and measurable. Existing KoboldCpp Qwen3-TTS, MiniMax H3, ACE-Step, Kokoro,
  OuteTTS, Parler, Dia, and Orpheus implementations are not imported.

The MinGW default and Vulkan DLL targets compile and link with all selected
models. Runtime output still needs validation with real Fish Audio and Maya1
packages; weights are intentionally not vendored here.

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
directory and send `reference_text` containing its transcript.

Pass `maya1-q8_0.gguf` (or `maya1-orig.gguf`) as the normal TTS model. Supply a
voice description with the existing `instruction` request field; when omitted,
KoboldCpp uses a neutral expressive-voice description.

## Intentional compatibility changes

KoboldCpp owns GGML. Audio.cpp's bundled GGML is never synchronized.

Local changes are marked `// kcpp:` and are summarized in `patches/README.md`.
The integration disables YAML configuration in favor of JSON model packages.
YuE2's unofficial operations are expressed with standard GGML graph operations;
no Audio.cpp GGML patch is carried in KoboldCpp.
