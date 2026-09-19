#include "wan_audio.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sd::wan_audio {

    static BucketPlan plan_buckets(int audio_frames, int batch_frames, int video_rate, int fps) {
        BucketPlan plan;
        plan.audio_frames  = audio_frames;
        plan.batch_frames  = batch_frames;
        plan.video_rate    = video_rate;
        plan.fps           = fps;
        const double scale = static_cast<double>(video_rate) / fps;
        // Keep a trailing chunk even when audio ends on a chunk boundary.
        plan.num_chunks          = static_cast<int>(audio_frames / (batch_frames * scale)) + 1;
        plan.bucket_frames       = plan.num_chunks * batch_frames;
        plan.padded_audio_frames = static_cast<int>(
            std::ceil(plan.bucket_frames / static_cast<double>(fps) * video_rate));
        return plan;
    }

    // Match NumPy's round-half-even sampling.
    static int bucket_source_frame(int bucket_frame, int video_rate, int fps) {
        return static_cast<int>(std::nearbyint(static_cast<double>(bucket_frame) * video_rate / fps));
    }

    static int interpolated_frame_count(int in_frames, int input_fps, int output_fps) {
        return static_cast<int>(in_frames / static_cast<double>(input_fps) * output_fps);
    }

    // Match PyTorch linear interpolation with align_corners=True.
    static std::vector<float> linear_interpolate_frames(const std::vector<float>& in,
                                                        int num_layers,
                                                        int in_frames,
                                                        int dim,
                                                        int out_frames) {
        std::vector<float> out(static_cast<size_t>(num_layers) * out_frames * dim, 0.0f);
        if (in.empty() || in_frames <= 0 || out_frames <= 0 || num_layers <= 0 || dim <= 0) {
            return out;
        }
        const double scale = out_frames > 1 ? static_cast<double>(in_frames - 1) / (out_frames - 1) : 0.0;
        for (int layer = 0; layer < num_layers; ++layer) {
            for (int out_i = 0; out_i < out_frames; ++out_i) {
                const double pos     = out_i * scale;
                const int src0       = static_cast<int>(pos);
                const int src1       = std::min(src0 + 1, in_frames - 1);
                const float frac     = static_cast<float>(pos - src0);
                const float* in_row  = &in[(static_cast<size_t>(layer) * in_frames + src0) * dim];
                const float* in_next = &in[(static_cast<size_t>(layer) * in_frames + src1) * dim];
                float* out_row       = &out[(static_cast<size_t>(layer) * out_frames + out_i) * dim];
                for (int d = 0; d < dim; ++d) {
                    out_row[d] = in_row[d] * (1.0f - frac) + in_next[d] * frac;
                }
            }
        }
        return out;
    }

    std::vector<float> build_audio_buckets(const float* stacked_states,
                                           int num_layers,
                                           int in_frames,
                                           int dim,
                                           int batch_frames,
                                           BucketPlan* plan_out,
                                           int input_fps,
                                           int video_rate,
                                           int fps) {
        if (stacked_states == nullptr || num_layers <= 0 || in_frames <= 0 || dim <= 0 || batch_frames <= 0) {
            return {};
        }
        const int audio_frames = interpolated_frame_count(in_frames, input_fps, video_rate);
        if (audio_frames <= 0) {
            return {};
        }
        const std::vector<float> interpolated =
            linear_interpolate_frames(std::vector<float>(stacked_states,
                                                         stacked_states + static_cast<size_t>(num_layers) * in_frames * dim),
                                      num_layers,
                                      in_frames,
                                      dim,
                                      audio_frames);
        const BucketPlan plan = plan_buckets(audio_frames, batch_frames, video_rate, fps);
        if (plan_out != nullptr) {
            *plan_out = plan;
        }
        std::vector<float> buckets(static_cast<size_t>(plan.bucket_frames) * num_layers * dim, 0.0f);
        for (int frame = 0; frame < plan.bucket_frames; ++frame) {
            const int src = bucket_source_frame(frame, video_rate, fps);
            if (src >= plan.audio_frames) {
                continue;
            }
            for (int layer = 0; layer < num_layers; ++layer) {
                std::copy_n(interpolated.data() + (static_cast<size_t>(layer) * audio_frames + src) * dim,
                            static_cast<size_t>(dim),
                            buckets.data() + (static_cast<size_t>(frame) * num_layers + layer) * dim);
            }
        }
        return buckets;
    }

}  // namespace sd::wan_audio
