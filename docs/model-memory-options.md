# Lazy tensor loading and MoE expert cache

KoboldCpp exposes two existing llama.cpp controls through the GUI, command line, and
`.kcpps` configuration files. Defaults preserve existing behavior.

| Option | Config key | Default | Meaning |
| --- | --- | --- | --- |
| `--lazy-mode` / `--lazymode` | `lazy_mode` | `"auto"` | `auto`, `on`, or `off` for architecture-marked lazy tensors. |
| `--moe-cache-mib` / `--moecache` | `moe_cache_mib` | `0` | Experimental GPU cache budget in MiB for CPU-resident MoE experts. Zero disables it. |

## Usage

For example, keep experts on the CPU and reserve a 4096 MiB GPU expert cache:

```powershell
python koboldcpp.py --model MODEL.gguf --usecuda --gpulayers 200 --moecpu --lazy-mode auto --moe-cache-mib 4096
```

To compare resident PLE embeddings, replace `--lazy-mode auto` with
`--lazy-mode off`. This can substantially increase resident memory requirements.
The cache example is a starting configuration, not a measured optimal size.

The equivalent additional settings in a `.kcpps` JSON object are:

```json
{
  "lazy_mode": "auto",
  "moe_cache_mib": 4096
}
```

Existing config save/load paths preserve these argument fields. Older configs
receive the default values. In the GUI, open the Context tab and use **Lazy Tensors** and
**MoE Cache MiB** below Override KV / Override Tensors. Both fields are included
when saving or loading a config. Loading an older config resets them to auto/0. Explicit CLI options follow the existing config override
rules. Cache sizes must be whole numbers from 0 through 2147483647; config values
are also checked before constructing native model inputs.

`auto` enables lazy reading for architecture-marked tensors larger than 4 GiB;
`on` requests lazy reading regardless of size; `off` keeps them resident. This
controls tensors such as Qwen3.8-Flash-Next's PLE/ngram embedding table, not ngram
speculative decoding. Lazy reads use the existing upstream mmap path and its
backend restrictions; this does not add direct-read or SSD streaming code.

Expert caching uses spare VRAM to retain experts whose backing weights are in
host memory. Use CPU expert placement (`--moecpu`) or `--autofit` to leave experts
in host memory. With manual placement, leave VRAM for the cache as well as model,
KV, and compute buffers. Autofit receives the cache budget during estimation.
Multiple GPUs divide the budget according to upstream layer placement. The
cache is experimental and inherits upstream backend restrictions; in particular,
avoid combining it with `--lowvram` / disabled KV offload pending upstream support.
No change is made to routing, the number of active experts, or model weights.

## Implementation and review notes

- Python `load_model_inputs` and C++ `load_model_inputs` carry matching integer
  fields for lazy mode and cache MiB. Lazy modes map to upstream OFF=0, AUTO=1,
  ON=2; MiB are converted to bytes using `size_t` in the GGUF adapter.
- Settings reach `llama_model_params.lazy_mode` and
  `llama_context_params.moe_cache_size` before autofit and model/context creation.
- Separate speculative draft models and their memory estimates inherit lazy mode.
  Draft contexts keep their default zero expert cache. Built-in MTP explicitly
  clears the cache budget copied from the target context, matching upstream's
  target-only cache policy.
- Legacy model adapters do not implement these controls. No vendored llama.cpp
  source or build-system changes are required.
- **Rebuild and distribute Python and native libraries together.** The internal
  load-model structure layout changes; an older DLL is incompatible with the new
  launcher. The local CUDA build produces `build/bin/koboldcpp_cublas.dll` and the
  existing build rule copies it to the repository root.

Upstream references:

- [MoE GPU cache, merged PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
- [Multi-GPU cache support, PR #30112](https://github.com/ggml-org/llama.cpp/pull/30112)
- Lazy mode definitions: `include/llama.h`, `common/arg.cpp`.
- Draft cache policy: `common/speculative.cpp` (`result.moe_cache_size = 0`).

## Validation

Executed locally on Windows:

- CUDA CMake build: successful, including the shared DLL link; existing compiler
  warnings remain.
- `python -m unittest discover -s tests -p test_model_memory_options.py`: five tests
  pass (defaults/aliases, CLI rejection, config rejection, native input mapping,
  and GUI config round trips including zero and old-config defaults).
- `python -m unittest discover -s tests -p test_numeric_parsing.py`: six tests pass.
- Python and C++ input declarations use the same field order and integer types.
- `git diff --check` passes.

The owner successfully rebuilt the updated native library and ran Qwen inference
with lazy mode off and expert caching disabled on two NVIDIA GPUs. This verifies
the resident/default-disabled path, not enabled expert caching or lazy reads.

Not yet measured: enabled cache hit rate or speed, lazy-on/auto PLE memory usage,
and MTP inference with these options. Before release, compare the
same model/prompt/context/seed at cache budgets 0 and 4096 MiB, then compare lazy
mode auto versus off. Include a no-flags baseline and an MTP smoke test. Record
hardware, quant, startup memory, prompt processing speed, and generation speed.
