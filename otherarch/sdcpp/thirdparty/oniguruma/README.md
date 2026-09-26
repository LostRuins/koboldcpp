# Oniguruma source subset

This directory contains the Oniguruma sources needed by sd.cpp's
[UTF-8 regex wrapper](../../src/core/regex.cpp).

## Upstream source

- Repository: [kkos/oniguruma](https://github.com/kkos/oniguruma)
- Release: `v6.9.10`
- Commit: [`4ef89209a239c1aea328cf13c05a2807e5c146d1`](https://github.com/kkos/oniguruma/tree/4ef89209a239c1aea328cf13c05a2807e5c146d1)
- License: [BSD-2-Clause](COPYING)

The 24 upstream files below were copied without local modifications. `COPYING`
comes from the upstream repository root; all other files come from upstream
`src/` and retain their original filenames, flattened into this directory.
This README is maintained by sd.cpp and is not part of the upstream copy.

## File selection

The following 13 C files are compiled separately:

```text
ascii.c
regcomp.c
regenc.c
regerror.c
regexec.c
regparse.c
st.c
unicode.c
unicode_fold1_key.c
unicode_fold2_key.c
unicode_fold3_key.c
unicode_unfold_key.c
utf8.c
```

Four Unicode data tables are included by `unicode.c` and must not be compiled
as separate translation units. They were copied from upstream as supplied;
sd.cpp does not regenerate them:

```text
unicode_fold_data.c
unicode_property_data.c
unicode_egcb_data.c
unicode_wb_data.c
```

The remaining files are five headers, the platform configuration template,
and the license:

```text
oniguruma.h
regint.h
regenc.h
regparse.h
st.h
config.h.cmake.in
COPYING
```

The wrapper uses `ONIG_ENCODING_UTF8` and `ONIG_SYNTAX_ONIGURUMA`. ASCII support
is also retained because the engine requires it for initialization and error
handling. Unicode property, case-folding, grapheme-cluster and word-boundary
support remain enabled as in upstream.

Other encodings, GNU/POSIX compatibility APIs, unused API implementations,
upstream tests, examples, build scripts and packaging files are omitted.
This subset supports sd.cpp's internal wrapper; it does not provide the full
Oniguruma API declared in `oniguruma.h`.

## Build integration

[The parent CMake file](../CMakeLists.txt) detects platform headers and type
sizes, then generates `config.h` from `config.h.cmake.in` in the build directory.
It builds the 13 C files as the `onig` OBJECT target with `ONIG_STATIC` and
position-independent code enabled. The resulting objects are included directly
in the sd.cpp static or shared library, with no separate Oniguruma library
required by consumers.

Keep all 24 upstream files under version control, including the Unicode tables,
configuration template and license. Generated `config.h` and build artifacts
belong in the build directory.

When refreshing this subset, copy the listed files from the selected upstream
revision, retain `COPYING`, and update the revision recorded here.
Recheck source dependencies and the CMake configuration, then validate the
regex wrapper and tokenizer output on supported platforms.
