# utf8proc source subset

This directory contains the utf8proc sources used by sd.cpp's
[JSON tokenizer](../../src/tokenizers/json_tokenizer.cpp) for NFC normalization,
lowercase mapping, and UTF-8 decoding and encoding.

## Upstream source

- Repository: [JuliaStrings/utf8proc](https://github.com/JuliaStrings/utf8proc)
- Release: `v2.10.0`
- Commit: [`a1b99daa2a3393884220264c927a48ba1251a9c6`](https://github.com/JuliaStrings/utf8proc/tree/a1b99daa2a3393884220264c927a48ba1251a9c6)
- Unicode version: `16.0.0`
- License: [MIT and Unicode data licenses](LICENSE.md)

The following four files were copied from the upstream repository root without
local modifications. Their original filenames are preserved. This README is
maintained by sd.cpp and replaces the upstream README.

| File | Purpose |
| --- | --- |
| `utf8proc.c` | Library implementation; the only separately compiled C file. |
| `utf8proc.h` | Public declarations, types and version definitions. |
| `utf8proc_data.c` | Unicode data tables included by `utf8proc.c`. |
| `LICENSE.md` | Library and Unicode data license notices. |

The library implementation and Unicode tables are retained in full.
`utf8proc_data.c` is copied as supplied by upstream; sd.cpp does not regenerate
it or compile it as a separate translation unit. Upstream tests, benchmarks,
documentation, data-generation tools and build/packaging files are omitted.

## Build integration

[The parent CMake file](../CMakeLists.txt) compiles `utf8proc.c` as the
`sd-utf8proc` OBJECT target with `UTF8PROC_STATIC` and position-independent code
enabled. The object is included directly in the sd.cpp static or shared library,
with no separate utf8proc library required by consumers. The license is installed
alongside sd.cpp.

All four upstream files belong under version control. Build artifacts belong
in the build directory; this subset requires no generated configuration header.

When refreshing this subset, copy all four files from the same upstream revision
and retain `LICENSE.md`. Update the release, commit and Unicode version recorded
here. Recheck source dependencies and validate tokenizer normalization, lowercase
conversion and token IDs on supported platforms.
