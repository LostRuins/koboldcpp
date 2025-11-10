#ifndef __LTXV_HPP__
#define __LTXV_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"

namespace LTXV {

    class CausalConv3d : public GGMLBlock {
    protected:
        int time_kernel_size;

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     int kernel_size                  = 3,
                     std::tuple<int, int, int> stride = {1, 1, 1},
                     int dilation                     = 1,
                     bool bias                        = true) {
            time_kernel_size = kernel_size / 2;
            blocks["conv"]   = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                     out_channels,
                                                                     {kernel_size, kernel_size, kernel_size},
                                                                     stride,
                                                                     {0, kernel_size / 2, kernel_size / 2},
                                                                     {dilation, 1, 1},
                                                                     bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    bool causal = true) {
            // x: [N*IC, ID, IH, IW]
            // result: [N*OC, OD, OH, OW]
            auto conv = std::dynamic_pointer_cast<Conv3d>(blocks["conv"]);
            if (causal) {
                auto h               = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));                                                  // [ID, N*IC, IH, IW]
                auto first_frame     = ggml_view_3d(ctx, h, h->ne[0], h->ne[1], h->ne[2], h->nb[1], h->nb[2], 0);                         // [N*IC, IH, IW]
                first_frame          = ggml_reshape_4d(ctx, first_frame, first_frame->ne[0], first_frame->ne[1], 1, first_frame->ne[2]);  // [N*IC, 1, IH, IW]
                auto first_frame_pad = first_frame;
                for (int i = 1; i < time_kernel_size - 1; i++) {
                    first_frame_pad = ggml_concat(ctx, first_frame_pad, first_frame, 2);
                }
                x = ggml_concat(ctx, first_frame_pad, x, 2);
            } else {
                auto h         = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));  // [ID, N*IC, IH, IW]
                int64_t offset = h->nb[2] * h->ne[2];

                auto first_frame     = ggml_view_3d(ctx, h, h->ne[0], h->ne[1], h->ne[2], h->nb[1], h->nb[2], 0);                         // [N*IC, IH, IW]
                first_frame          = ggml_reshape_4d(ctx, first_frame, first_frame->ne[0], first_frame->ne[1], 1, first_frame->ne[2]);  // [N*IC, 1, IH, IW]
                auto first_frame_pad = first_frame;
                for (int i = 1; i < (time_kernel_size - 1) / 2; i++) {
                    first_frame_pad = ggml_concat(ctx, first_frame_pad, first_frame, 2);
                }

                auto last_frame     = ggml_view_3d(ctx, h, h->ne[0], h->ne[1], h->ne[2], h->nb[1], h->nb[2], offset * (h->ne[3] - 1));  // [N*IC, IH, IW]
                last_frame          = ggml_reshape_4d(ctx, last_frame, last_frame->ne[0], last_frame->ne[1], 1, last_frame->ne[2]);     // [N*IC, 1, IH, IW]
                auto last_frame_pad = last_frame;
                for (int i = 1; i < (time_kernel_size - 1) / 2; i++) {
                    last_frame_pad = ggml_concat(ctx, last_frame_pad, last_frame, 2);
                }

                x = ggml_concat(ctx, first_frame_pad, x, 2);
                x = ggml_concat(ctx, x, last_frame_pad, 2);
            }

            x = conv->forward(ctx, x);
            return x;
        }
    };

};

#endif