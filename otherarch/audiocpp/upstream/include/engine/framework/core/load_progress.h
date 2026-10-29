#pragma once

#include <cstdint>
#include <string>

namespace engine::core {

// Model-load progress accounting for hosts that stream the trace log
// (audiocpp_server --log-file, audiocpp_cli --log). Mirrors the way
// llama.cpp reports model loading: a bytes-done / bytes-total curve fed
// once per tensor while weights upload, so load UIs get a continuous,
// honest signal instead of a single "loaded" milestone at the end.
//
// Strictly opt-in via --load-progress 1 (default 0): with the flag off,
// no load-progress lines are emitted at all and the log is byte-identical
// to a build without this feature. When enabled alongside --log /
// --log-file, one curve per ModelRegistry::load call; the registry
// brackets it with runtime.load.phase lines and BackendWeightStore::upload
// feeds it:
//
//   [TRACE ...] runtime.load.phase inspect
//   [TRACE ...] runtime.load.phase load
//   [TRACE ...] <store>.weights.upload_progress 0.002
//   ...
//   [TRACE ...] <store>.weights.upload_progress 0.99
//   [TRACE ...] runtime.load.phase loaded
//   [TRACE ...] <store>.weights.upload_progress 1
//
// Thread-safe: a lazy first-use load can overlap an eager load.

/// Enable or disable load-progress reporting (--load-progress 0|1).
/// Default: disabled. Must be called before the first model load.
void set_load_progress_enabled(bool enabled);

/// Whether load-progress reporting is enabled.
bool load_progress_enabled();

/// Begin tracking a new model load. total_bytes seeds the denominator from
/// the model's on-disk weight files; stores raise it if they end up
/// uploading more (dequantized expansion, derived tensors, weights the
/// inspection could not see). No-op unless reporting is enabled and the
/// trace log is on.
void begin_model_load(uint64_t total_bytes);

/// A weight store is about to upload `bytes` (BackendWeightStore::upload).
void register_weight_bytes(uint64_t bytes);

/// One tensor finished uploading.
void add_uploaded_weight_bytes(uint64_t bytes);

/// Trace `<store>.weights.upload_progress <0..1>` when the fraction has
/// moved enough since the last emitted line. The reported fraction is
/// always done/total over all work known so far: when a later store raises
/// the total, the next emission corrects downward instead of clamping to
/// the previous high-water mark, so the curve keeps reflecting the
/// remaining work. The per-tensor path never reports 1.0 - the final
/// curve point belongs to end_model_load, after the last store uploaded.
void emit_weight_upload_progress(const std::string & store_name);

/// The model load finished: emit the final curve point at 1.0.
void end_model_load();

}  // namespace engine::core
