# KoboldCpp native sources

This directory contains KoboldCpp-owned implementation and build-support code.
Upstream-synced llama.cpp and ggml sources remain in their original directories
so upstream updates can continue to be merged without path-only conflicts.

Prebuilt helper programs and link inputs used by KoboldCpp builds live under
`bin/` and `lib/`, respectively. Release version metadata and its generation
scripts live under `packaging/`; Conda environment definitions are kept in
`packaging/environments/`.
