# KoboldCpp compatibility patches

Files under `upstream` are synchronized from Audio.cpp, then receive the small
patches listed here. Recheck these after every run of `sync-audiocpp.ps1`.

1. `src/framework/io/config.cpp`: compile without libyaml. JSON and simple
   key/value configs remain supported; YAML produces an explicit error.
2. `src/framework/modules/activation_modules.cpp`: replace Audio.cpp's
   unofficial `ggml_round_bf16` operation with a standard F32 -> BF16 -> F32
   cast sequence.
3. `src/framework/modules/conv_modules.cpp`: use standard cast, multiply, and
   add graphs instead of Audio.cpp's private BF16-round, fused matmul-accumulate,
   and cuDNN annotation helpers. These are correctness-preserving fallbacks;
   backend-specific performance can be recovered later without forking GGML.
4. `src/framework/modules/linear_module.cpp`: lower Audio.cpp's private
   F16-output matmul to a normal matmul plus cast, and omit backend-specific
   lowering hints. First-wave models do not request those hints.
5. `src/framework/modules/norm_modules.cpp`: lower private CUDA F16/fused
   affine hints and BF16 rounding to standard normalization, affine, and cast
   graph operations.
6. `src/framework/modules/primitive_modules.cpp`: expand the private fused
   RoPE interleaved-pairs operation into multiply/add/subtract/concat nodes.
7. `src/framework/modules/structural_modules.cpp`: omit Audio.cpp's private
   CUDA concat-lowering hint and retain the ordinary concat graph.
8. `src/framework/assets/resource_bundle.cpp` and `tensor_source.cpp`: reject
   YAML resources and PyTorch pickle/bin tensors. Selected runtime packages use
   JSON with safetensors or GGUF; the omitted formats mainly serve converters.
9. `src/framework/modules/attention/scaled_dot_product_attention.cpp`: lower
   Audio.cpp's CUDA-only head-40 flash kernel to standard flash attention plus
   an output cast.
10. YuE2 decoder/positional sources: replace all remaining private BF16-round
    calls with F32 -> BF16 -> F32 casts.
11. `src/framework/runtime/graph_optimizer.cpp`: disable optional graph
    rewriting because Audio.cpp mutates GGML's private graph node count.
12. The build reuses KoboldCpp's newer `src/unicode.cpp` implementation for the
    imported llama BPE core. Audio.cpp's duplicate unicode translation units
    are deliberately neither synchronized nor compiled.
13. YuE2 asset/session loading accepts explicit absolute component paths and a
    tokenizer override. Its fixed 3B and VAE configurations fall back to the
    compiled defaults, allowing KoboldCpp to use `--musicllm`, `--musicvae`,
    and the bundled tokenizer without upstream package sidecars.
14. `src/framework/codecs/fish_dac_codec_runtime.cpp`: provide the same
    portable F32 -> BF16 -> F32 compatibility lowering for Fish Audio S2's
    remaining private `ggml_round_bf16` calls.
15. Framework filesystem/package sources route path normalization through
    `engine::io::normalize_path`. On Windows this avoids MinGW's broken
    `weakly_canonical` traversal for otherwise accessible special-folder paths;
    other platforms retain upstream canonicalization.
16. `src/framework/debug/trace.cpp`: use a locale-independent stream formatter
    for scalar doubles because older Linux libstdc++ versions do not provide
    the floating-point `std::to_chars` overload.

Do not add KCPP adapter behavior to upstream-owned files. Put it in the parent
directory so upstream refreshes remain reviewable.
