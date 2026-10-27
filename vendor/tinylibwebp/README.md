# tinylibwebp

This is a source-only, scalar subset of Google's libwebp 1.6.0, vendored for
KoboldCpp's future animated WebP output. It provides the WebP encoder, decoder
support required by the muxer, and the animation mux/encoder API. It requires no
installed library and ships no precompiled binaries.

Upstream archive:
`https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz`

SHA-256:
`e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564`

## Deliberate build choices

- Only portable C implementations are compiled.
- SSE, SSE4, AVX, NEON, MIPS and MSA source files are omitted.
- CPU-specific flags inherited from KoboldCpp are filtered from these objects.
- Internal libwebp threading is disabled; KoboldCpp can parallelize above the
  encoder if desired.
- The source objects are collected into a generated static archive. No archive
  or other binary is checked in, and unused WebP code is omitted at final link.
- Demux-only APIs, tools, examples, tests and upstream build generators are
  omitted.

`src/dsp/cpu.h` has one small KoboldCpp patch guarded by `WEBP_DISABLE_SIMD`.
It disables automatic SIMD selection so generic dispatch code cannot refer to
the intentionally omitted platform objects. `src/webp/config.h` is the local,
minimal configuration header. All other C and header files are unmodified
upstream 1.6.0 files.

The upstream BSD-style license is in `LICENSE`/`COPYING`; Google's additional
patent grant is in `PATENTS`.

## Updating

Verify the new release archive checksum, refresh only the files listed in
`sources.mk` and `CMakeLists.txt` plus their headers, reapply the small
`WEBP_DISABLE_SIMD` guard, and run the scalar animation smoke test described by
the KoboldCpp build.
