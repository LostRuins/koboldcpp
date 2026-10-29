#pragma once

#include <cstdint>
#include <vector>

namespace engine::modules {

// True when a cached encoder graph built for `capacity_frames` may be reused
// for a request of `request_frames`.
//
// An ASR encoder graph runs at its built capacity no matter how short the real
// audio is -- encode() zero-pads up to it and masks the padding out of the
// result, not out of the arithmetic. So an oversized cached graph is paid for
// in full on every call, and with self-attention that cost is quadratic in
// frames. Reusing one indefinitely turns the largest request the process has
// ever seen into a floor under every later request.
//
// Rebuilding is a one-off cost of a few hundred ms, dominated by the positional
// projections, so it wins outright once the mismatch is more than a few
// percent. The tolerance keeps a stream of clips whose lengths wobble slightly
// from rebuilding on every call, while capping the wasted compute at roughly
// the same fraction.
bool asr_graph_capacity_usable(int64_t capacity_frames, int64_t request_frames);

std::vector<int32_t> make_asr_keep_mask(int64_t frames, int64_t valid_frames);

void fill_asr_keep_mask(std::vector<int32_t> & out, int64_t frames, int64_t valid_frames);

std::vector<float> make_asr_full_attention_bias(int64_t frames, int64_t valid_frames);

void fill_asr_full_attention_bias(std::vector<float> & out, int64_t frames, int64_t valid_frames);

void fill_asr_chunked_attention_bias(
    std::vector<float> & out,
    int64_t frames,
    int64_t valid_frames,
    int64_t left_context,
    int64_t right_context);

void fill_asr_stream_attention_bias(
    std::vector<float> & out,
    int64_t current_frames,
    int64_t key_frames,
    int64_t valid_cached_frames,
    int64_t q_offset,
    int64_t kv_offset,
    int64_t left_context,
    int64_t right_context);

}  // namespace engine::modules
