#ifndef __AUTO_ENCODER_KL_HPP__
#define __AUTO_ENCODER_KL_HPP__

#include "vae.hpp"

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

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
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

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
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

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
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

        h_ = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);

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

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x) override {
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
    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
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

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
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
                               ggml_ext_scale(ctx->ggml_ctx, x, alpha),
                               ggml_ext_scale(ctx->ggml_ctx, x_mix, 1.0f - alpha));

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

    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        auto h = conv_in->forward(ctx, x);  // [N, ch, h, w]
        // sd::ggml_graph_cut::mark_graph_cut(h, "vae.encoder.prelude", "h");

        // downsampling
        size_t num_resolutions = ch_mult.size();
        for (int i = 0; i < num_resolutions; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                auto down_block  = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = down_block->forward(ctx, h);
                // sd::ggml_graph_cut::mark_graph_cut(h, "vae.encoder.down." + std::to_string(i) + ".block." + std::to_string(j), "h");
            }
            if (i != num_resolutions - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                auto down_sample = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = down_sample->forward(ctx, h);
                // sd::ggml_graph_cut::mark_graph_cut(h, "vae.encoder.down." + std::to_string(i) + ".downsample", "h");
            }
        }

        // middle
        h = mid_block_1->forward(ctx, h);
        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]
        // sd::ggml_graph_cut::mark_graph_cut(h, "vae.encoder.mid", "h");

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

    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z) {
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
        // sd::ggml_graph_cut::mark_graph_cut(h, "vae.decoder.prelude", "h");

        // middle
        h = mid_block_1->forward(ctx, h);
        // return h;

        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]
        // sd::ggml_graph_cut::mark_graph_cut(h, "vae.decoder.mid", "h");

        // upsampling
        int num_resolutions = static_cast<int>(ch_mult.size());
        for (int i = num_resolutions - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                auto up_block    = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = up_block->forward(ctx, h);
                // sd::ggml_graph_cut::mark_graph_cut(h, "vae.decoder.up." + std::to_string(i) + ".block." + std::to_string(j), "h");
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                auto up_sample   = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                h = up_sample->forward(ctx, h);
                // sd::ggml_graph_cut::mark_graph_cut(h, "vae.decoder.up." + std::to_string(i) + ".upsample", "h");
            }
        }

        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);  // nonlinearity/swish
        h = conv_out->forward(ctx, h);            // [N, out_ch, h*8, w*8]
        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
class AutoEncoderKLModel : public GGMLBlock {
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

    static std::string get_tensor_name(const std::string& prefix, const std::string& name) {
        return prefix.empty() ? name : prefix + "." + name;
    }

    void detect_decoder_ch(const String2TensorStorage& tensor_storage_map,
                           const std::string& prefix,
                           int& decoder_ch) {
        auto conv_in_iter = tensor_storage_map.find(get_tensor_name(prefix, "decoder.conv_in.weight"));
        if (conv_in_iter != tensor_storage_map.end() && conv_in_iter->second.n_dims >= 4 && conv_in_iter->second.ne[3] > 0) {
            int last_ch_mult             = dd_config.ch_mult.back();
            int64_t conv_in_out_channels = conv_in_iter->second.ne[3];
            if (last_ch_mult > 0 && conv_in_out_channels % last_ch_mult == 0) {
                decoder_ch = static_cast<int>(conv_in_out_channels / last_ch_mult);
                LOG_INFO("vae decoder: ch = %d", decoder_ch);
            } else {
                LOG_WARN("vae decoder: failed to infer ch from %s (%" PRId64 " / %d)",
                         get_tensor_name(prefix, "decoder.conv_in.weight").c_str(),
                         conv_in_out_channels,
                         last_ch_mult);
            }
        }
    }

public:
    AutoEncoderKLModel(SDVersion version                              = VERSION_SD1,
                       bool decode_only                               = true,
                       bool use_linear_projection                     = false,
                       bool use_video_decoder                         = false,
                       const String2TensorStorage& tensor_storage_map = {},
                       const std::string& prefix                      = "")
        : version(version), decode_only(decode_only), use_video_decoder(use_video_decoder) {
        if (sd_version_is_dit(version)) {
            if (sd_version_uses_flux2_vae(version)) {
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
        int decoder_ch = dd_config.ch;
        detect_decoder_ch(tensor_storage_map, prefix, decoder_ch);
        blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder(decoder_ch,
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

    ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        if (sd_version_uses_flux2_vae(version)) {
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
            // sd::ggml_graph_cut::mark_graph_cut(z, "vae.decode.prelude", "z");
        }
        auto decoder = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);

        ggml_set_name(z, "bench-start");
        auto h = decoder->forward(ctx, z);
        ggml_set_name(h, "bench-end");
        return h;
    }

    ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);

        auto z = encoder->forward(ctx, x);  // [N, 2*z_channels, h/8, w/8]
        if (use_quant) {
            auto quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["quant_conv"]);
            z               = quant_conv->forward(ctx, z);  // [N, 2*embed_dim, h/8, w/8]
            // sd::ggml_graph_cut::mark_graph_cut(z, "vae.encode.final", "z");
        }
        if (sd_version_uses_flux2_vae(version)) {
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

    int get_encoder_output_channels() {
        int factor = dd_config.double_z ? 2 : 1;
        if (sd_version_uses_flux2_vae(version)) {
            return dd_config.z_channels * 4;
        }
        return dd_config.z_channels * factor;
    }
};

struct AutoEncoderKL : public VAE {
    float scale_factor = 1.f;
    float shift_factor = 0.f;
    bool decode_only   = true;
    AutoEncoderKLModel ae;

    AutoEncoderKL(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix,
                  bool decode_only       = false,
                  bool use_video_decoder = false,
                  SDVersion version      = VERSION_SD1)
        : decode_only(decode_only), VAE(version, backend, offload_params_to_cpu) {
        if (sd_version_is_sd1(version) || sd_version_is_sd2(version)) {
            scale_factor = 0.18215f;
            shift_factor = 0.f;
        } else if (sd_version_is_sdxl(version)) {
            scale_factor = 0.13025f;
            shift_factor = 0.f;
        } else if (sd_version_is_sd3(version)) {
            scale_factor = 1.5305f;
            shift_factor = 0.0609f;
        } else if (sd_version_is_flux(version) || sd_version_is_z_image(version)) {
            scale_factor = 0.3611f;
            shift_factor = 0.1159f;
        } else if (sd_version_uses_flux2_vae(version)) {
            scale_factor = 1.0f;
            shift_factor = 0.f;
        }
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
        ae = AutoEncoderKLModel(version, decode_only, use_linear_projection, use_video_decoder, tensor_storage_map, prefix);
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

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
        ae.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);
        ggml_tensor* z  = make_input(z_tensor);

        auto runner_ctx = get_context();

        ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z,
                               bool decode_graph) override {
        GGML_ASSERT(!decode_only || decode_graph);
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(z, decode_graph);
        };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), z.dim());
    }

    sd::Tensor<float> gaussian_latent_sample(const sd::Tensor<float>& moments, std::shared_ptr<RNG> rng) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        auto chunks               = sd::ops::chunk(moments, 2, 2);
        const auto& mean          = chunks[0];
        const auto& logvar        = chunks[1];
        sd::Tensor<float> stddev  = sd::ops::exp(0.5f * sd::ops::clamp(logvar, -30.0f, 20.0f));
        sd::Tensor<float> noise   = sd::Tensor<float>::randn_like(mean, rng);
        sd::Tensor<float> latents = mean + stddev * noise;
        return latents;
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        if (sd_version_uses_flux2_vae(version)) {
            return vae_output;
        } else if (version == VERSION_SD1_PIX2PIX) {
            return sd::ops::chunk(vae_output, 2, 2)[0];
        } else {
            return gaussian_latent_sample(vae_output, rng);
        }
    }

    std::pair<sd::Tensor<float>, sd::Tensor<float>> get_latents_mean_std(const sd::Tensor<float>& latents, int channel_dim) {
        GGML_ASSERT(channel_dim >= 0 && static_cast<size_t>(channel_dim) < static_cast<size_t>(latents.dim()));
        if (sd_version_uses_flux2_vae(version)) {
            GGML_ASSERT(latents.shape()[channel_dim] == 128);
            std::vector<int64_t> stats_shape(static_cast<size_t>(latents.dim()), 1);
            stats_shape[static_cast<size_t>(channel_dim)] = latents.shape()[channel_dim];

            auto mean_tensor = sd::Tensor<float>::from_vector({-0.0676f, -0.0715f, -0.0753f, -0.0745f, 0.0223f, 0.0180f, 0.0142f, 0.0184f,
                                                               -0.0001f, -0.0063f, -0.0002f, -0.0031f, -0.0272f, -0.0281f, -0.0276f, -0.0290f,
                                                               -0.0769f, -0.0672f, -0.0902f, -0.0892f, 0.0168f, 0.0152f, 0.0079f, 0.0086f,
                                                               0.0083f, 0.0015f, 0.0003f, -0.0043f, -0.0439f, -0.0419f, -0.0438f, -0.0431f,
                                                               -0.0102f, -0.0132f, -0.0066f, -0.0048f, -0.0311f, -0.0306f, -0.0279f, -0.0180f,
                                                               0.0030f, 0.0015f, 0.0126f, 0.0145f, 0.0347f, 0.0338f, 0.0337f, 0.0283f,
                                                               0.0020f, 0.0047f, 0.0047f, 0.0050f, 0.0123f, 0.0081f, 0.0081f, 0.0146f,
                                                               0.0681f, 0.0679f, 0.0767f, 0.0732f, -0.0462f, -0.0474f, -0.0392f, -0.0511f,
                                                               -0.0528f, -0.0477f, -0.0470f, -0.0517f, -0.0317f, -0.0316f, -0.0345f, -0.0283f,
                                                               0.0510f, 0.0445f, 0.0578f, 0.0458f, -0.0412f, -0.0458f, -0.0487f, -0.0467f,
                                                               -0.0088f, -0.0106f, -0.0088f, -0.0046f, -0.0376f, -0.0432f, -0.0436f, -0.0499f,
                                                               0.0118f, 0.0166f, 0.0203f, 0.0279f, 0.0113f, 0.0129f, 0.0016f, 0.0072f,
                                                               -0.0118f, -0.0018f, -0.0141f, -0.0054f, -0.0091f, -0.0138f, -0.0145f, -0.0187f,
                                                               0.0323f, 0.0305f, 0.0259f, 0.0300f, 0.0540f, 0.0614f, 0.0495f, 0.0590f,
                                                               -0.0511f, -0.0603f, -0.0478f, -0.0524f, -0.0227f, -0.0274f, -0.0154f, -0.0255f,
                                                               -0.0572f, -0.0565f, -0.0518f, -0.0496f, 0.0116f, 0.0054f, 0.0163f, 0.0104f});
            mean_tensor.reshape_(stats_shape);
            auto std_tensor = sd::Tensor<float>::from_vector({1.8029f, 1.7786f, 1.7868f, 1.7837f, 1.7717f, 1.7590f, 1.7610f, 1.7479f,
                                                              1.7336f, 1.7373f, 1.7340f, 1.7343f, 1.8626f, 1.8527f, 1.8629f, 1.8589f,
                                                              1.7593f, 1.7526f, 1.7556f, 1.7583f, 1.7363f, 1.7400f, 1.7355f, 1.7394f,
                                                              1.7342f, 1.7246f, 1.7392f, 1.7304f, 1.7551f, 1.7513f, 1.7559f, 1.7488f,
                                                              1.8449f, 1.8454f, 1.8550f, 1.8535f, 1.8240f, 1.7813f, 1.7854f, 1.7945f,
                                                              1.8047f, 1.7876f, 1.7695f, 1.7676f, 1.7782f, 1.7667f, 1.7925f, 1.7848f,
                                                              1.7579f, 1.7407f, 1.7483f, 1.7368f, 1.7961f, 1.7998f, 1.7920f, 1.7925f,
                                                              1.7780f, 1.7747f, 1.7727f, 1.7749f, 1.7526f, 1.7447f, 1.7657f, 1.7495f,
                                                              1.7775f, 1.7720f, 1.7813f, 1.7813f, 1.8162f, 1.8013f, 1.8023f, 1.8033f,
                                                              1.7527f, 1.7331f, 1.7563f, 1.7482f, 1.7610f, 1.7507f, 1.7681f, 1.7613f,
                                                              1.7665f, 1.7545f, 1.7828f, 1.7726f, 1.7896f, 1.7999f, 1.7864f, 1.7760f,
                                                              1.7613f, 1.7625f, 1.7560f, 1.7577f, 1.7783f, 1.7671f, 1.7810f, 1.7799f,
                                                              1.7201f, 1.7068f, 1.7265f, 1.7091f, 1.7793f, 1.7578f, 1.7502f, 1.7455f,
                                                              1.7587f, 1.7500f, 1.7525f, 1.7362f, 1.7616f, 1.7572f, 1.7444f, 1.7430f,
                                                              1.7509f, 1.7610f, 1.7634f, 1.7612f, 1.7254f, 1.7135f, 1.7321f, 1.7226f,
                                                              1.7664f, 1.7624f, 1.7718f, 1.7664f, 1.7457f, 1.7441f, 1.7569f, 1.7530f});
            std_tensor.reshape_(stats_shape);
            return {std::move(mean_tensor), std::move(std_tensor)};
        } else {
            GGML_ABORT("unknown version %d", version);
        }
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        if (sd_version_uses_flux2_vae(version)) {
            int channel_dim                = 2;
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents, channel_dim);
            return (latents * std_tensor) / scale_factor + mean_tensor;
        }
        return (latents / scale_factor) + shift_factor;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        if (sd_version_uses_flux2_vae(version)) {
            int channel_dim                = 2;
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents, channel_dim);
            return ((latents - mean_tensor) * scale_factor) / std_tensor;
        }
        return (latents - shift_factor) * scale_factor;
    }

    int get_encoder_output_channels(int input_channels) {
        return ae.get_encoder_output_channels();
    }

    void test() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        {
            // CPU, x{1, 3, 64, 64}: Pass
            // CUDA, x{1, 3, 64, 64}: Pass, but sill get wrong result for some image, may be due to interlnal nan
            // CPU, x{2, 3, 64, 64}: Wrong result
            // CUDA, x{2, 3, 64, 64}: Wrong result, and different from CPU result
            sd::Tensor<float> x({64, 64, 3, 2});
            x.fill_(0.5f);
            print_sd_tensor(x);
            sd::Tensor<float> out;

            int64_t t0   = ggml_time_ms();
            auto out_opt = _compute(8, x, false);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
            LOG_DEBUG("encode test done in %lldms", t1 - t0);
        }

        if (false) {
            // CPU, z{1, 4, 8, 8}: Pass
            // CUDA, z{1, 4, 8, 8}: Pass
            // CPU, z{3, 4, 8, 8}: Wrong result
            // CUDA, z{3, 4, 8, 8}: Wrong result, and different from CPU result
            sd::Tensor<float> z({8, 8, 4, 1});
            z.fill_(0.5f);
            print_sd_tensor(z);
            sd::Tensor<float> out;

            int64_t t0   = ggml_time_ms();
            auto out_opt = _compute(8, z, true);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
            LOG_DEBUG("decode test done in %lldms", t1 - t0);
        }
    };
};

#endif  // __AUTO_ENCODER_KL_HPP__
