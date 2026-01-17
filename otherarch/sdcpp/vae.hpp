#ifndef __VAE_HPP__
#define __VAE_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"

/*================================================== AutoEncoderKL ===================================================*/

#define VAE_GRAPH_SIZE 20480

class ResnetBlock : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;

public:
    ResnetBlock(int64_t in_channels,
                int64_t out_channels)
        : in_channels(in_channels),
          out_channels(out_channels) {
        // temb_channels is always 0
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));

        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(out_channels));
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));

        if (out_channels != in_channels) {
            blocks["nin_shortcut"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, {1, 1}));
        }
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        // x: [N, in_channels, h, w]
        // t_emb is always None
        auto norm1 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm1"]);
        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
        auto norm2 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm2"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);

        auto h = x;
        h      = norm1->forward(ctx, h);
        h      = ggml_silu_inplace(ctx->ggml_ctx, h);  // swish
        h      = conv1->forward(ctx, h);
        // return h;

        h = norm2->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);  // swish
        // dropout, skip for inference
        h = conv2->forward(ctx, h);

        // skip connection
        if (out_channels != in_channels) {
            auto nin_shortcut = std::dynamic_pointer_cast<Conv2d>(blocks["nin_shortcut"]);

            x = nin_shortcut->forward(ctx, x);  // [N, out_channels, h, w]
        }

        h = ggml_add(ctx->ggml_ctx, h, x);
        return h;  // [N, out_channels, h, w]
    }
};

class AttnBlock : public UnaryBlock {
protected:
    int64_t in_channels;
    bool use_linear;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
        auto iter = tensor_storage_map.find(prefix + "proj_out.weight");
        if (iter != tensor_storage_map.end()) {
            if (iter->second.n_dims == 4 && use_linear) {
                use_linear         = false;
                blocks["q"]        = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
                blocks["k"]        = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
                blocks["v"]        = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
                blocks["proj_out"] = std::make_shared<Conv2d>(in_channels, in_channels, std::pair{1, 1});
            } else if (iter->second.n_dims == 2 && !use_linear) {
                use_linear         = true;
                blocks["q"]        = std::make_shared<Linear>(in_channels, in_channels);
                blocks["k"]        = std::make_shared<Linear>(in_channels, in_channels);
                blocks["v"]        = std::make_shared<Linear>(in_channels, in_channels);
                blocks["proj_out"] = std::make_shared<Linear>(in_channels, in_channels);
            }
        }
    }

public:
    AttnBlock(int64_t in_channels, bool use_linear)
        : in_channels(in_channels), use_linear(use_linear) {
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        if (use_linear) {
            blocks["q"]        = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
            blocks["k"]        = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
            blocks["v"]        = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, in_channels));
        } else {
            blocks["q"]        = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
            blocks["k"]        = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
            blocks["v"]        = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
        }
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        // x: [N, in_channels, h, w]
        auto norm     = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto q_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["q"]);
        auto k_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["k"]);
        auto v_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["v"]);
        auto proj_out = std::dynamic_pointer_cast<UnaryBlock>(blocks["proj_out"]);

        auto h_ = norm->forward(ctx, x);

        const int64_t n = h_->ne[3];
        const int64_t c = h_->ne[2];
        const int64_t h = h_->ne[1];
        const int64_t w = h_->ne[0];

        ggml_tensor* q;
        ggml_tensor* k;
        ggml_tensor* v;
        if (use_linear) {
            h_ = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 2, 0, 3));  // [N, h, w, in_channels]
            h_ = ggml_reshape_3d(ctx->ggml_ctx, h_, c, h * w, n);                        // [N, h * w, in_channels]

            q = q_proj->forward(ctx, h_);  // [N, h * w, in_channels]
            k = k_proj->forward(ctx, h_);  // [N, h * w, in_channels]
            v = v_proj->forward(ctx, h_);  // [N, h * w, in_channels]
        } else {
            q = q_proj->forward(ctx, h_);                                              // [N, in_channels, h, w]
            q = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, q, 1, 2, 0, 3));  // [N, h, w, in_channels]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, c, h * w, n);                        // [N, h * w, in_channels]

            k = k_proj->forward(ctx, h_);                                              // [N, in_channels, h, w]
            k = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, k, 1, 2, 0, 3));  // [N, h, w, in_channels]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, c, h * w, n);                        // [N, h * w, in_channels]

            v = v_proj->forward(ctx, h_);                                              // [N, in_channels, h, w]
            v = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, v, 1, 2, 0, 3));  // [N, h, w, in_channels]
            v = ggml_reshape_3d(ctx->ggml_ctx, v, c, h * w, n);                        // [N, h * w, in_channels]
        }

        h_ = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, true, false);

        if (use_linear) {
            h_ = proj_out->forward(ctx, h_);  // [N, h * w, in_channels]

            h_ = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
            h_ = ggml_reshape_4d(ctx->ggml_ctx, h_, w, h, c, n);                         // [N, in_channels, h, w]
        } else {
            h_ = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
            h_ = ggml_reshape_4d(ctx->ggml_ctx, h_, w, h, c, n);                         // [N, in_channels, h, w]

            h_ = proj_out->forward(ctx, h_);  // [N, in_channels, h, w]
        }

        h_ = ggml_add(ctx->ggml_ctx, h_, x);
        return h_;
    }
};

class AE3DConv : public Conv2d {
public:
    AE3DConv(int64_t in_channels,
             int64_t out_channels,
             std::pair<int, int> kernel_size,
             int video_kernel_size        = 3,
             std::pair<int, int> stride   = {1, 1},
             std::pair<int, int> padding  = {0, 0},
             std::pair<int, int> dilation = {1, 1},
             bool bias                    = true)
        : Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias) {
        int kernel_padding      = video_kernel_size / 2;
        blocks["time_mix_conv"] = std::shared_ptr<GGMLBlock>(new Conv3d(out_channels,
                                                                        out_channels,
                                                                        {video_kernel_size, 1, 1},
                                                                        {1, 1, 1},
                                                                        {kernel_padding, 0, 0}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                struct ggml_tensor* x) override {
        // timesteps always None
        // skip_video always False
        // x: [N, IC, IH, IW]
        // result: [N, OC, OH, OW]
        auto time_mix_conv = std::dynamic_pointer_cast<Conv3d>(blocks["time_mix_conv"]);

        x = Conv2d::forward(ctx, x);
        // timesteps = x.shape[0]
        // x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
        // x = conv3d(x)
        // return rearrange(x, "b c t h w -> (b t) c h w")
        int64_t T = x->ne[3];
        int64_t B = x->ne[3] / T;
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];

        x = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, C, T, B);                     // (b t) c h w -> b t c (h w)
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
        x = time_mix_conv->forward(ctx, x);                                        // [B, OC, T, OH * OW]
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
        x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H, C, T * B);                     // b t c (h w) -> (b t) c h w
        return x;                                                                  // [B*T, OC, OH, OW]
    }
};

class VideoResnetBlock : public ResnetBlock {
protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "mix_factor", tensor_storage_map, GGML_TYPE_F32);
        params["mix_factor"] = ggml_new_tensor_1d(ctx, wtype, 1);
    }

    float get_alpha() {
        float alpha = ggml_ext_backend_tensor_get_f32(params["mix_factor"]);
        return sigmoid(alpha);
    }

public:
    VideoResnetBlock(int64_t in_channels,
                     int64_t out_channels,
                     int video_kernel_size = 3)
        : ResnetBlock(in_channels, out_channels) {
        // merge_strategy is always learned
        blocks["time_stack"] = std::shared_ptr<GGMLBlock>(new ResBlock(out_channels, 0, out_channels, {video_kernel_size, 1}, 3, false, true));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        // x: [N, in_channels, h, w] aka [b*t, in_channels, h, w]
        // return: [N, out_channels, h, w] aka [b*t, out_channels, h, w]
        // t_emb is always None
        // skip_video is always False
        // timesteps is always None
        auto time_stack = std::dynamic_pointer_cast<ResBlock>(blocks["time_stack"]);

        x = ResnetBlock::forward(ctx, x);  // [N, out_channels, h, w]
        // return x;

        int64_t T = x->ne[3];
        int64_t B = x->ne[3] / T;
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];

        x          = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, C, T, B);                     // (b t) c h w -> b t c (h w)
        x          = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
        auto x_mix = x;

        x = time_stack->forward(ctx, x);  // b t c (h w)

        float alpha = get_alpha();
        x           = ggml_add(ctx->ggml_ctx,
                               ggml_scale(ctx->ggml_ctx, x, alpha),
                               ggml_scale(ctx->ggml_ctx, x_mix, 1.0f - alpha));

        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
        x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H, C, T * B);                     // b t c (h w) -> (b t) c h w

        return x;
    }
};

// ldm.modules.diffusionmodules.model.Encoder
class Encoder : public GGMLBlock {
protected:
    int ch                   = 128;
    std::vector<int> ch_mult = {1, 2, 4, 4};
    int num_res_blocks       = 2;
    int in_channels          = 3;
    int z_channels           = 4;
    bool double_z            = true;

public:
    Encoder(int ch,
            std::vector<int> ch_mult,
            int num_res_blocks,
            int in_channels,
            int z_channels,
            bool double_z              = true,
            bool use_linear_projection = false)
        : ch(ch),
          ch_mult(ch_mult),
          num_res_blocks(num_res_blocks),
          in_channels(in_channels),
          z_channels(z_channels),
          double_z(double_z) {
        blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, ch, {3, 3}, {1, 1}, {1, 1}));

        size_t num_resolutions = ch_mult.size();

        int block_in = 1;
        for (int i = 0; i < num_resolutions; i++) {
            if (i == 0) {
                block_in = ch;
            } else {
                block_in = ch * ch_mult[i - 1];
            }
            int block_out = ch * ch_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                blocks[name]     = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_out));
                block_in         = block_out;
            }
            if (i != num_resolutions - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(block_in, block_in, true));
            }
        }

        blocks["mid.block_1"] = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_in));
        blocks["mid.attn_1"]  = std::shared_ptr<GGMLBlock>(new AttnBlock(block_in, use_linear_projection));
        blocks["mid.block_2"] = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_in));

        blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(block_in));
        blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(block_in, double_z ? z_channels * 2 : z_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        auto h = conv_in->forward(ctx, x);  // [N, ch, h, w]

        // downsampling
        size_t num_resolutions = ch_mult.size();
        for (int i = 0; i < num_resolutions; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                auto down_block  = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = down_block->forward(ctx, h);
            }
            if (i != num_resolutions - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                auto down_sample = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = down_sample->forward(ctx, h);
            }
        }

        // middle
        h = mid_block_1->forward(ctx, h);
        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]

        // end
        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);  // nonlinearity/swish
        h = conv_out->forward(ctx, h);            // [N, z_channels*2, h, w]
        return h;
    }
};

// ldm.modules.diffusionmodules.model.Decoder
class Decoder : public GGMLBlock {
protected:
    int ch                   = 128;
    int out_ch               = 3;
    std::vector<int> ch_mult = {1, 2, 4, 4};
    int num_res_blocks       = 2;
    int z_channels           = 4;
    bool video_decoder       = false;
    int video_kernel_size    = 3;

    virtual std::shared_ptr<GGMLBlock> get_conv_out(int64_t in_channels,
                                                    int64_t out_channels,
                                                    std::pair<int, int> kernel_size,
                                                    std::pair<int, int> stride  = {1, 1},
                                                    std::pair<int, int> padding = {0, 0}) {
        if (video_decoder) {
            return std::shared_ptr<GGMLBlock>(new AE3DConv(in_channels, out_channels, kernel_size, video_kernel_size, stride, padding));
        } else {
            return std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, kernel_size, stride, padding));
        }
    }

    virtual std::shared_ptr<GGMLBlock> get_resnet_block(int64_t in_channels,
                                                        int64_t out_channels) {
        if (video_decoder) {
            return std::shared_ptr<GGMLBlock>(new VideoResnetBlock(in_channels, out_channels, video_kernel_size));
        } else {
            return std::shared_ptr<GGMLBlock>(new ResnetBlock(in_channels, out_channels));
        }
    }

public:
    Decoder(int ch,
            int out_ch,
            std::vector<int> ch_mult,
            int num_res_blocks,
            int z_channels,
            bool use_linear_projection = false,
            bool video_decoder         = false,
            int video_kernel_size      = 3)
        : ch(ch),
          out_ch(out_ch),
          ch_mult(ch_mult),
          num_res_blocks(num_res_blocks),
          z_channels(z_channels),
          video_decoder(video_decoder),
          video_kernel_size(video_kernel_size) {
        int num_resolutions = static_cast<int>(ch_mult.size());
        int block_in        = ch * ch_mult[num_resolutions - 1];

        blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, block_in, {3, 3}, {1, 1}, {1, 1}));

        blocks["mid.block_1"] = get_resnet_block(block_in, block_in);
        blocks["mid.attn_1"]  = std::shared_ptr<GGMLBlock>(new AttnBlock(block_in, use_linear_projection));
        blocks["mid.block_2"] = get_resnet_block(block_in, block_in);

        for (int i = num_resolutions - 1; i >= 0; i--) {
            int mult      = ch_mult[i];
            int block_out = ch * mult;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                blocks[name]     = get_resnet_block(block_in, block_out);

                block_in = block_out;
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(block_in, block_in));
            }
        }

        blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(block_in));
        blocks["conv_out"] = get_conv_out(block_in, out_ch, {3, 3}, {1, 1}, {1, 1});
    }

    virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        // alpha is always 0
        // merge_strategy is always learned
        // time_mode is always conv-only, so we need to replace conv_out_op/resnet_op to AE3DConv/VideoResBlock
        // AttnVideoBlock will not be used
        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        // conv_in
        auto h = conv_in->forward(ctx, z);  // [N, block_in, h, w]

        // middle
        h = mid_block_1->forward(ctx, h);
        // return h;

        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]

        // upsampling
        int num_resolutions = static_cast<int>(ch_mult.size());
        for (int i = num_resolutions - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                auto up_block    = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = up_block->forward(ctx, h);
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                auto up_sample   = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                h = up_sample->forward(ctx, h);
            }
        }

        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);  // nonlinearity/swish
        h = conv_out->forward(ctx, h);            // [N, out_ch, h*8, w*8]
        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
class AutoencodingEngine : public GGMLBlock {
protected:
    SDVersion version;
    bool decode_only       = true;
    bool use_video_decoder = false;
    bool use_quant         = true;
    int embed_dim          = 4;
    struct {
        int z_channels           = 4;
        int resolution           = 256;
        int in_channels          = 3;
        int out_ch               = 3;
        int ch                   = 128;
        std::vector<int> ch_mult = {1, 2, 4, 4};
        int num_res_blocks       = 2;
        bool double_z            = true;
    } dd_config;

public:
    AutoencodingEngine(SDVersion version          = VERSION_SD1,
                       bool decode_only           = true,
                       bool use_linear_projection = false,
                       bool use_video_decoder     = false)
        : version(version), decode_only(decode_only), use_video_decoder(use_video_decoder) {
        if (sd_version_is_dit(version)) {
            if (sd_version_is_flux2(version)) {
                dd_config.z_channels = 32;
                embed_dim            = 32;
            } else {
                use_quant            = false;
                dd_config.z_channels = 16;
            }
        }
        if (use_video_decoder) {
            use_quant = false;
        }
        blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder(dd_config.ch,
                                                                   dd_config.out_ch,
                                                                   dd_config.ch_mult,
                                                                   dd_config.num_res_blocks,
                                                                   dd_config.z_channels,
                                                                   use_linear_projection,
                                                                   use_video_decoder));
        if (use_quant) {
            blocks["post_quant_conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dd_config.z_channels,
                                                                              embed_dim,
                                                                              {1, 1}));
        }
        if (!decode_only) {
            blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder(dd_config.ch,
                                                                       dd_config.ch_mult,
                                                                       dd_config.num_res_blocks,
                                                                       dd_config.in_channels,
                                                                       dd_config.z_channels,
                                                                       dd_config.double_z,
                                                                       use_linear_projection));
            if (use_quant) {
                int factor = dd_config.double_z ? 2 : 1;

                blocks["quant_conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(embed_dim * factor,
                                                                             dd_config.z_channels * factor,
                                                                             {1, 1}));
            }
        }
    }

    struct ggml_tensor* decode(GGMLRunnerContext* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        if (sd_version_is_flux2(version)) {
            // [N, C*p*p, h, w] -> [N, C, h*p, w*p]
            int64_t p = 2;

            int64_t N = z->ne[3];
            int64_t C = z->ne[2] / p / p;
            int64_t h = z->ne[1];
            int64_t w = z->ne[0];
            int64_t H = h * p;
            int64_t W = w * p;

            z = ggml_reshape_4d(ctx->ggml_ctx, z, w * h, p * p, C, N);                           // [N, C, p*p, h*w]
            z = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, z, 1, 0, 2, 3));  // [N, C, h*w, p*p]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, p, p, w, h * C * N);                           // [N*C*h, w, p, p]
            z = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, z, 0, 2, 1, 3));  // [N*C*h, p, w, p]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, W, H, C, N);                                   // [N, C, h*p, w*p]
        }

        if (use_quant) {
            auto post_quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["post_quant_conv"]);
            z                    = post_quant_conv->forward(ctx, z);  // [N, z_channels, h, w]
        }
        auto decoder = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);

        ggml_set_name(z, "bench-start");
        auto h = decoder->forward(ctx, z);
        ggml_set_name(h, "bench-end");
        return h;
    }

    struct ggml_tensor* encode(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);

        auto z = encoder->forward(ctx, x);  // [N, 2*z_channels, h/8, w/8]
        if (use_quant) {
            auto quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["quant_conv"]);
            z               = quant_conv->forward(ctx, z);  // [N, 2*embed_dim, h/8, w/8]
        }
        if (sd_version_is_flux2(version)) {
            z = ggml_ext_chunk(ctx->ggml_ctx, z, 2, 2)[0];

            // [N, C, H, W] -> [N, C*p*p, H/p, W/p]
            int64_t p = 2;
            int64_t N = z->ne[3];
            int64_t C = z->ne[2];
            int64_t H = z->ne[1];
            int64_t W = z->ne[0];
            int64_t h = H / p;
            int64_t w = W / p;

            z = ggml_reshape_4d(ctx->ggml_ctx, z, p, w, p, h * C * N);                 // [N*C*h, p, w, p]
            z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 0, 2, 1, 3));  // [N*C*h, w, p, p]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, p * p, w * h, C, N);                 // [N, C, h*w, p*p]
            z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 1, 0, 2, 3));  // [N, C, p*p, h*w]
            z = ggml_reshape_4d(ctx->ggml_ctx, z, w, h, p * p * C, N);                 // [N, C*p*p, h*w]
        }
        return z;
    }
};

struct VAE : public GGMLRunner {
    VAE(ggml_backend_t backend, bool offload_params_to_cpu)
        : GGMLRunner(backend, offload_params_to_cpu) {}
    virtual bool compute(const int n_threads,
                         struct ggml_tensor* z,
                         bool decode_graph,
                         struct ggml_tensor** output,
                         struct ggml_context* output_ctx)                                                         = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) = 0;
    virtual void set_conv2d_scale(float scale) { SD_UNUSED(scale); };
};

struct FakeVAE : public VAE {
    FakeVAE(ggml_backend_t backend, bool offload_params_to_cpu)
        : VAE(backend, offload_params_to_cpu) {}
    bool compute(const int n_threads,
                 struct ggml_tensor* z,
                 bool decode_graph,
                 struct ggml_tensor** output,
                 struct ggml_context* output_ctx) override {
        if (*output == nullptr && output_ctx != nullptr) {
            *output = ggml_dup_tensor(output_ctx, z);
        }
        ggml_ext_tensor_iter(z, [&](ggml_tensor* z, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = ggml_ext_tensor_get_f32(z, i0, i1, i2, i3);
            ggml_ext_tensor_set_f32(*output, value, i0, i1, i2, i3);
        });
        return true;
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {}

    std::string get_desc() override {
        return "fake_vae";
    }
};

struct AutoEncoderKL : public VAE {
    bool decode_only = true;
    AutoencodingEngine ae;

    AutoEncoderKL(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix,
                  bool decode_only       = false,
                  bool use_video_decoder = false,
                  SDVersion version      = VERSION_SD1)
        : decode_only(decode_only), VAE(backend, offload_params_to_cpu) {
        bool use_linear_projection = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }
            if (ends_with(name, "attn_1.proj_out.weight")) {
                if (tensor_storage.n_dims == 2) {
                    use_linear_projection = true;
                }
                break;
            }
        }
        ae = AutoencodingEngine(version, decode_only, use_linear_projection, use_video_decoder);
        ae.init(params_ctx, tensor_storage_map, prefix);
    }

    void set_conv2d_scale(float scale) override {
        std::vector<GGMLBlock*> blocks;
        ae.get_all_blocks(blocks);
        for (auto block : blocks) {
            if (block->get_desc() == "Conv2d") {
                auto conv_block = (Conv2d*)block;
                conv_block->set_scale(scale);
            }
        }
    }

    std::string get_desc() override {
        return "vae";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {
        ae.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        z = to_backend(z);

        auto runner_ctx = get_context();

        struct ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    bool compute(const int n_threads,
                 struct ggml_tensor* z,
                 bool decode_graph,
                 struct ggml_tensor** output,
                 struct ggml_context* output_ctx = nullptr) override {
        GGML_ASSERT(!decode_only || decode_graph);
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };
        // ggml_set_f32(z, 0.5f);
        // print_ggml_tensor(z);
        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }

    void test() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != nullptr);

        {
            // CPU, x{1, 3, 64, 64}: Pass
            // CUDA, x{1, 3, 64, 64}: Pass, but sill get wrong result for some image, may be due to interlnal nan
            // CPU, x{2, 3, 64, 64}: Wrong result
            // CUDA, x{2, 3, 64, 64}: Wrong result, and different from CPU result
            auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 64, 64, 3, 2);
            ggml_set_f32(x, 0.5f);
            print_ggml_tensor(x);
            struct ggml_tensor* out = nullptr;

            int64_t t0 = ggml_time_ms();
            compute(8, x, false, &out, work_ctx);
            int64_t t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("encode test done in %lldms", t1 - t0);
        }

        if (false) {
            // CPU, z{1, 4, 8, 8}: Pass
            // CUDA, z{1, 4, 8, 8}: Pass
            // CPU, z{3, 4, 8, 8}: Wrong result
            // CUDA, z{3, 4, 8, 8}: Wrong result, and different from CPU result
            auto z = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
            ggml_set_f32(z, 0.5f);
            print_ggml_tensor(z);
            struct ggml_tensor* out = nullptr;

            int64_t t0 = ggml_time_ms();
            compute(8, z, true, &out, work_ctx);
            int64_t t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("decode test done in %lldms", t1 - t0);
        }
    };
};

#endif
