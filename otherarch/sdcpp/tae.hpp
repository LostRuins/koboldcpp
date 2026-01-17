#ifndef __TAE_HPP__
#define __TAE_HPP__

#include "ggml_extend.hpp"

#include "model.h"

/*
    ===================================    TinyAutoEncoder  ===================================
    References:
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py
    https://github.com/madebyollin/taesd/blob/main/taesd.py

*/

class TAEBlock : public UnaryBlock {
protected:
    int n_in;
    int n_out;

public:
    TAEBlock(int n_in, int n_out)
        : n_in(n_in), n_out(n_out) {
        blocks["conv.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_in, n_out, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_out, n_out, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_out, n_out, {3, 3}, {1, 1}, {1, 1}));
        if (n_in != n_out) {
            blocks["skip"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_in, n_out, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
        }
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        // x: [n, n_in, h, w]
        // return: [n, n_out, h, w]

        auto conv_0 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.0"]);
        auto conv_2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.2"]);
        auto conv_4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.4"]);

        auto h = conv_0->forward(ctx, x);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv_2->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv_4->forward(ctx, h);

        if (n_in != n_out) {
            auto skip = std::dynamic_pointer_cast<Conv2d>(blocks["skip"]);
            LOG_DEBUG("skip");
            x = skip->forward(ctx, x);
        }

        h = ggml_add(ctx->ggml_ctx, h, x);
        h = ggml_relu_inplace(ctx->ggml_ctx, h);
        return h;
    }
};

class TinyEncoder : public UnaryBlock {
    int in_channels = 3;
    int channels    = 64;
    int z_channels  = 4;
    int num_blocks  = 3;

public:
    TinyEncoder(int z_channels = 4)
        : z_channels(z_channels) {
        int index                       = 0;
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, channels, {3, 3}, {1, 1}, {1, 1}));
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, z_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        // x: [n, in_channels, h, w]
        // return: [n, z_channels, h/8, w/8]

        for (int i = 0; i < num_blocks * 3 + 6; i++) {
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(i)]);

            x = block->forward(ctx, x);
        }

        return x;
    }
};

class TinyDecoder : public UnaryBlock {
    int z_channels   = 4;
    int channels     = 64;
    int out_channels = 3;
    int num_blocks   = 3;

public:
    TinyDecoder(int z_channels = 4)
        : z_channels(z_channels) {
        int index = 0;

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, channels, {3, 3}, {1, 1}, {1, 1}));
        index++;  // nn.ReLU()

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* z) override {
        // z: [n, z_channels, h, w]
        // return: [n, out_channels, h*8, w*8]

        auto h = ggml_scale(ctx->ggml_ctx, z, 1.0f / 3.0f);
        h      = ggml_tanh_inplace(ctx->ggml_ctx, h);
        h      = ggml_scale(ctx->ggml_ctx, h, 3.0f);

        for (int i = 0; i < num_blocks * 3 + 10; i++) {
            if (blocks.find(std::to_string(i)) == blocks.end()) {
                if (i == 1) {
                    h = ggml_relu_inplace(ctx->ggml_ctx, h);
                } else {
                    h = ggml_upscale(ctx->ggml_ctx, h, 2, GGML_SCALE_MODE_NEAREST);
                }
                continue;
            }
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(i)]);

            h = block->forward(ctx, h);
        }

        return h;
    }
};

class TPool : public UnaryBlock {
    int stride;

public:
    TPool(int channels, int stride)
        : stride(stride) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels * stride, channels, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        auto conv = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv"]);
        auto h    = x;
        if (stride != 1) {
            h = ggml_reshape_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], h->ne[2] * stride, h->ne[3] / stride);
        }
        h = conv->forward(ctx, h);
        return h;
    }
};

class TGrow : public UnaryBlock {
    int stride;

public:
    TGrow(int channels, int stride)
        : stride(stride) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels * stride, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        auto conv = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv"]);
        auto h    = conv->forward(ctx, x);
        if (stride != 1) {
            h = ggml_reshape_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], h->ne[2] / stride, h->ne[3] * stride);
        }
        return h;
    }
};

class MemBlock : public GGMLBlock {
    bool has_skip_conv = false;

public:
    MemBlock(int channels, int out_channels)
        : has_skip_conv(channels != out_channels) {
        blocks["conv.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels * 2, out_channels, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
        if (has_skip_conv) {
            blocks["skip"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
        }
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x, struct ggml_tensor* past) {
        // x: [n, channels, h, w]
        auto conv0 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.0"]);
        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.2"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.4"]);

        auto h = ggml_concat(ctx->ggml_ctx, x, past, 2);
        h      = conv0->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv1->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv2->forward(ctx, h);

        auto skip = x;
        if (has_skip_conv) {
            auto skip_conv = std::dynamic_pointer_cast<Conv2d>(blocks["skip"]);
            skip           = skip_conv->forward(ctx, x);
        }
        h = ggml_add_inplace(ctx->ggml_ctx, h, skip);
        h = ggml_relu_inplace(ctx->ggml_ctx, h);
        return h;
    }
};

struct ggml_tensor* patchify(struct ggml_context* ctx,
                             struct ggml_tensor* x,
                             int64_t patch_size,
                             int64_t b = 1) {
    // x: [f, b*c, h*q, w*r]
    // return: [f, b*c*r*q, h, w]
    if (patch_size == 1) {
        return x;
    }
    int64_t r = patch_size;
    int64_t q = patch_size;

    int64_t W = x->ne[0];
    int64_t H = x->ne[1];
    int64_t C = x->ne[2];
    int64_t f = x->ne[3];

    int64_t w = W / r;
    int64_t h = H / q;

    x = ggml_reshape_4d(ctx, x, W, q, h, C * f);                         // [W, q, h, C*f]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [W, h, q, C*f]
    x = ggml_reshape_4d(ctx, x, r, w, h, q * C * f);                     // [r, w, h, q*C*f]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));  // [w, h, r, q*C*f]
    x = ggml_reshape_4d(ctx, x, w, h, r * q * C, f);                     // [f, b*c*r*q, h, w]

    return x;
}

struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                               struct ggml_tensor* x,
                               int64_t patch_size,
                               int64_t b = 1) {
    // x: [f, b*c*r*q, h, w]
    // return: [f, b*c, h*q, w*r]
    if (patch_size == 1) {
        return x;
    }
    int64_t r = patch_size;
    int64_t q = patch_size;
    int64_t c = x->ne[2] / b / q / r;
    int64_t f = x->ne[3];
    int64_t h = x->ne[1];
    int64_t w = x->ne[0];

    x = ggml_reshape_4d(ctx, x, w, h, r, q * c * b * f);                 // [q*c*b*f, r, h, w]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [r, w, h, q*c*b*f]
    x = ggml_reshape_4d(ctx, x, r * w, h, q, c * b * f);                 // [c*b*f, q, h, r*w]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [r*w, q, h, c*b*f]
    x = ggml_reshape_4d(ctx, x, r * w, q * h, c * b, f);

    return x;
}

class TinyVideoEncoder : public UnaryBlock {
    int in_channels = 3;
    int hidden      = 64;
    int z_channels  = 4;
    int num_blocks  = 3;
    int num_layers  = 3;
    int patch_size  = 1;

public:
    TinyVideoEncoder(int z_channels = 4, int patch_size = 1)
        : z_channels(z_channels), patch_size(patch_size) {
        int index                       = 0;
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels * patch_size * patch_size, hidden, {3, 3}, {1, 1}, {1, 1}));
        index++;  // nn.ReLU()
        for (int i = 0; i < num_layers; i++) {
            int stride                      = i == num_layers - 1 ? 1 : 2;
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TPool(hidden, stride));
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(hidden, hidden, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
            for (int j = 0; j < num_blocks; j++) {
                blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new MemBlock(hidden, hidden));
            }
        }
        blocks[std::to_string(index)] = std::shared_ptr<GGMLBlock>(new Conv2d(hidden, z_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* z) override {
        auto first_conv = std::dynamic_pointer_cast<Conv2d>(blocks["0"]);

        if (patch_size > 1) {
            z = patchify(ctx->ggml_ctx, z, patch_size, 1);
        }

        auto h = first_conv->forward(ctx, z);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);

        int index = 2;
        for (int i = 0; i < num_layers; i++) {
            auto pool = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);
            auto conv = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);

            h = pool->forward(ctx, h);
            h = conv->forward(ctx, h);
            for (int j = 0; j < num_blocks; j++) {
                auto block = std::dynamic_pointer_cast<MemBlock>(blocks[std::to_string(index++)]);
                auto mem   = ggml_pad_ext(ctx->ggml_ctx, h, 0, 0, 0, 0, 0, 0, 1, 0);
                mem        = ggml_view_4d(ctx->ggml_ctx, mem, h->ne[0], h->ne[1], h->ne[2], h->ne[3], h->nb[1], h->nb[2], h->nb[3], 0);
                h          = block->forward(ctx, h, mem);
            }
        }
        auto last_conv = std::dynamic_pointer_cast<Conv2d>(blocks[std::to_string(index)]);
        h              = last_conv->forward(ctx, h);
        return h;
    }
};

class TinyVideoDecoder : public UnaryBlock {
    int z_channels               = 4;
    int out_channels             = 3;
    int num_blocks               = 3;
    static const int num_layers  = 3;
    int channels[num_layers + 1] = {256, 128, 64, 64};
    int patch_size               = 1;

public:
    TinyVideoDecoder(int z_channels = 4, int patch_size = 1)
        : z_channels(z_channels), patch_size(patch_size) {
        int index                       = 1;  // Clamp()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, channels[0], {3, 3}, {1, 1}, {1, 1}));
        index++;  // nn.ReLU()
        for (int i = 0; i < num_layers; i++) {
            int stride = i == 0 ? 1 : 2;
            for (int j = 0; j < num_blocks; j++) {
                blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new MemBlock(channels[i], channels[i]));
            }
            index++;  // nn.Upsample()
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TGrow(channels[i], stride));
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels[i], channels[i + 1], {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));
        }
        index++;  // nn.ReLU()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels[num_layers], out_channels * patch_size * patch_size, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* z) override {
        auto first_conv = std::dynamic_pointer_cast<Conv2d>(blocks["1"]);

        // Clamp()
        auto h = ggml_scale_inplace(ctx->ggml_ctx,
                                    ggml_tanh_inplace(ctx->ggml_ctx,
                                                      ggml_scale(ctx->ggml_ctx, z, 1.0f / 3.0f)),
                                    3.0f);

        h         = first_conv->forward(ctx, h);
        h         = ggml_relu_inplace(ctx->ggml_ctx, h);
        int index = 3;
        for (int i = 0; i < num_layers; i++) {
            for (int j = 0; j < num_blocks; j++) {
                auto block = std::dynamic_pointer_cast<MemBlock>(blocks[std::to_string(index++)]);
                auto mem   = ggml_pad_ext(ctx->ggml_ctx, h, 0, 0, 0, 0, 0, 0, 1, 0);
                mem        = ggml_view_4d(ctx->ggml_ctx, mem, h->ne[0], h->ne[1], h->ne[2], h->ne[3], h->nb[1], h->nb[2], h->nb[3], 0);
                h          = block->forward(ctx, h, mem);
            }
            // upsample
            index++;
            h          = ggml_upscale(ctx->ggml_ctx, h, 2, GGML_SCALE_MODE_NEAREST);
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);
            h          = block->forward(ctx, h);
            block      = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);
            h          = block->forward(ctx, h);
        }
        h = ggml_relu_inplace(ctx->ggml_ctx, h);

        auto last_conv = std::dynamic_pointer_cast<Conv2d>(blocks[std::to_string(++index)]);
        h              = last_conv->forward(ctx, h);
        if (patch_size > 1) {
            h = unpatchify(ctx->ggml_ctx, h, patch_size, 1);
        }
        // shape(W, H, 3, 3 + T) => shape(W, H, 3, T)
        h = ggml_view_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], h->ne[2], h->ne[3] - 3, h->nb[1], h->nb[2], h->nb[3], 3 * h->nb[3]);
        return h;
    }
};

class TAEHV : public GGMLBlock {
protected:
    bool decode_only;
    SDVersion version;

public:
    TAEHV(bool decode_only = true, SDVersion version = VERSION_WAN2)
        : decode_only(decode_only), version(version) {
        int z_channels = 16;
        int patch      = 1;
        if (version == VERSION_WAN2_2_TI2V) {
            z_channels = 48;
            patch      = 2;
        }
        blocks["decoder"] = std::shared_ptr<GGMLBlock>(new TinyVideoDecoder(z_channels, patch));
        if (!decode_only) {
            blocks["encoder"] = std::shared_ptr<GGMLBlock>(new TinyVideoEncoder(z_channels, patch));
        }
    }

    struct ggml_tensor* decode(GGMLRunnerContext* ctx, struct ggml_tensor* z) {
        auto decoder = std::dynamic_pointer_cast<TinyVideoDecoder>(blocks["decoder"]);
        if (sd_version_is_wan(version)) {
            // (W, H, C, T) -> (W, H, T, C)
            z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 0, 1, 3, 2));
        }
        auto result = decoder->forward(ctx, z);
        if (sd_version_is_wan(version)) {
            // (W, H, C, T) -> (W, H, T, C)
            result = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, result, 0, 1, 3, 2));
        }
        return result;
    }

    struct ggml_tensor* encode(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        auto encoder = std::dynamic_pointer_cast<TinyVideoEncoder>(blocks["encoder"]);
        // (W, H, T, C) -> (W, H, C, T)
        x                  = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
        int64_t num_frames = x->ne[3];
        if (num_frames % 4) {
            // pad to multiple of 4 at the end
            auto last_frame = ggml_view_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1], x->ne[2], 1, x->nb[1], x->nb[2], x->nb[3], (num_frames - 1) * x->nb[3]);
            for (int i = 0; i < 4 - num_frames % 4; i++) {
                x = ggml_concat(ctx->ggml_ctx, x, last_frame, 3);
            }
        }
        x = encoder->forward(ctx, x);
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
        return x;
    }
};

class TAESD : public GGMLBlock {
protected:
    bool decode_only;

public:
    TAESD(bool decode_only = true, SDVersion version = VERSION_SD1)
        : decode_only(decode_only) {
        int z_channels = 4;
        if (sd_version_is_dit(version)) {
            z_channels = 16;
        }
        blocks["decoder.layers"] = std::shared_ptr<GGMLBlock>(new TinyDecoder(z_channels));

        if (!decode_only) {
            blocks["encoder.layers"] = std::shared_ptr<GGMLBlock>(new TinyEncoder(z_channels));
        }
    }

    struct ggml_tensor* decode(GGMLRunnerContext* ctx, struct ggml_tensor* z) {
        auto decoder = std::dynamic_pointer_cast<TinyDecoder>(blocks["decoder.layers"]);
        return decoder->forward(ctx, z);
    }

    struct ggml_tensor* encode(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        auto encoder = std::dynamic_pointer_cast<TinyEncoder>(blocks["encoder.layers"]);
        return encoder->forward(ctx, x);
    }
};

struct TinyAutoEncoder : public GGMLRunner {
    TinyAutoEncoder(ggml_backend_t backend, bool offload_params_to_cpu)
        : GGMLRunner(backend, offload_params_to_cpu) {}
    virtual bool compute(const int n_threads,
                         struct ggml_tensor* z,
                         bool decode_graph,
                         struct ggml_tensor** output,
                         struct ggml_context* output_ctx = nullptr) = 0;

    virtual bool load_from_file(const std::string& file_path, int n_threads)                                      = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) = 0;
};

struct TinyImageAutoEncoder : public TinyAutoEncoder {
    TAESD taesd;
    bool decode_only = false;

    TinyImageAutoEncoder(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix,
                         bool decoder_only = true,
                         SDVersion version = VERSION_SD1)
        : decode_only(decoder_only),
          taesd(decoder_only, version),
          TinyAutoEncoder(backend, offload_params_to_cpu) {
        taesd.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "taesd";
    }

    bool load_from_file(const std::string& file_path, int n_threads) {
        LOG_INFO("loading taesd from '%s', decode_only = %s", file_path.c_str(), decode_only ? "true" : "false");
        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> taesd_tensors;
        taesd.get_param_tensors(taesd_tensors);
        std::set<std::string> ignore_tensors;
        if (decode_only) {
            ignore_tensors.insert("encoder.");
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(file_path)) {
            LOG_ERROR("init taesd model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool success = model_loader.load_tensors(taesd_tensors, ignore_tensors, n_threads);

        if (!success) {
            LOG_ERROR("load tae tensors from model loader failed");
            return false;
        }

        LOG_INFO("taesd model loaded");
        return success;
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        taesd.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        struct ggml_cgraph* gf  = ggml_new_graph(compute_ctx);
        z                       = to_backend(z);
        auto runner_ctx         = get_context();
        struct ggml_tensor* out = decode_graph ? taesd.decode(&runner_ctx, z) : taesd.encode(&runner_ctx, z);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool compute(const int n_threads,
                 struct ggml_tensor* z,
                 bool decode_graph,
                 struct ggml_tensor** output,
                 struct ggml_context* output_ctx = nullptr) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };

        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

struct TinyVideoAutoEncoder : public TinyAutoEncoder {
    TAEHV taehv;
    bool decode_only = false;

    TinyVideoAutoEncoder(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix,
                         bool decoder_only = true,
                         SDVersion version = VERSION_WAN2)
        : decode_only(decoder_only),
          taehv(decoder_only, version),
          TinyAutoEncoder(backend, offload_params_to_cpu) {
        taehv.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "taehv";
    }

    bool load_from_file(const std::string& file_path, int n_threads) {
        LOG_INFO("loading taehv from '%s', decode_only = %s", file_path.c_str(), decode_only ? "true" : "false");
        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> taehv_tensors;
        taehv.get_param_tensors(taehv_tensors);
        std::set<std::string> ignore_tensors;
        if (decode_only) {
            ignore_tensors.insert("encoder.");
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init taehv model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool success = model_loader.load_tensors(taehv_tensors, ignore_tensors, n_threads);

        if (!success) {
            LOG_ERROR("load tae tensors from model loader failed");
            return false;
        }

        LOG_INFO("taehv model loaded");
        return success;
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        taehv.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        struct ggml_cgraph* gf  = ggml_new_graph(compute_ctx);
        z                       = to_backend(z);
        auto runner_ctx         = get_context();
        struct ggml_tensor* out = decode_graph ? taehv.decode(&runner_ctx, z) : taehv.encode(&runner_ctx, z);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool compute(const int n_threads,
                 struct ggml_tensor* z,
                 bool decode_graph,
                 struct ggml_tensor** output,
                 struct ggml_context* output_ctx = nullptr) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };

        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif  // __TAE_HPP__