#include "generation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>

#include "core/rng.hpp"
#include "core/rng_philox.hpp"
#include "diffusion_engine.h"
#include "model/diffusion/minimax_h3.hpp"
#include "model/upscaler/ltx_latent_upscaler.hpp"
#include "model/vae/audio_vae.hpp"
#include "model/vae/vae.hpp"
#include "request.h"
#include "runtime/denoiser.hpp"

// kcpp
#include "model/vae/ltx_audio_vae.hpp"

namespace sd::pipeline {

    static sd_audio_t* waveform_to_sd_audio(const StableDiffusionGGML* sd,
                                            const sd::Tensor<float>& waveform) {
        if (sd == nullptr || waveform.empty()) {
            return nullptr;
        }

        int64_t sample_count = waveform.shape()[0];
        int64_t channels     = waveform.shape().size() > 1 ? waveform.shape()[1] : 1;
        if (sample_count <= 0 || channels <= 0) {
            return nullptr;
        }

        sd_audio_t* audio = (sd_audio_t*)malloc(sizeof(sd_audio_t));
        if (audio == nullptr) {
            return nullptr;
        }

        audio->sample_rate  = static_cast<uint32_t>(sd->audio_vae_model != nullptr ? sd->audio_vae_model->output_sample_rate() : 0);
        audio->channels     = static_cast<uint32_t>(channels);
        audio->sample_count = static_cast<uint64_t>(sample_count);
        size_t sample_bytes = waveform.numel() * sizeof(float);
        audio->data         = (float*)malloc(sample_bytes);
        if (audio->data == nullptr) {
            free(audio);
            return nullptr;
        }

        auto wavaform_t = waveform.permute({1, 0, 2, 3});
        std::memcpy(audio->data, wavaform_t.data(), sample_bytes);

        return audio;
    }

    static float ltxv_latent_corner_to_pixel_frame(int64_t corner_index,
                                                   int temporal_scale,
                                                   bool causal_temporal_positioning) {
        float pixel_t = static_cast<float>(corner_index * temporal_scale);
        if (causal_temporal_positioning) {
            pixel_t = std::max(0.f, pixel_t + 1.f - static_cast<float>(temporal_scale));
        }
        return pixel_t;
    }

    static void set_ltxv_video_position(sd::Tensor<float>* positions,
                                        int64_t token,
                                        float t_start,
                                        float t_end,
                                        float h_start,
                                        float h_end,
                                        float w_start,
                                        float w_end) {
        positions->index(0, 0, token, 0) = t_start;
        positions->index(1, 0, token, 0) = t_end;
        positions->index(0, 1, token, 0) = h_start;
        positions->index(1, 1, token, 0) = h_end;
        positions->index(0, 2, token, 0) = w_start;
        positions->index(1, 2, token, 0) = w_end;
    }

    static sd::Tensor<float> build_ltxv_video_positions(int64_t width,
                                                        int64_t height,
                                                        int64_t target_latent_frames,
                                                        int64_t keyframe_latent_frames,
                                                        int keyframe_frame_idx,
                                                        int keyframe_pixel_frames,
                                                        int fps,
                                                        int spatial_scale,
                                                        int temporal_scale,
                                                        bool causal_temporal_positioning) {
        GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0);
        GGML_ASSERT(keyframe_latent_frames > 0);
        GGML_ASSERT(fps > 0);

        int64_t total_tokens = width * height * (target_latent_frames + keyframe_latent_frames);
        sd::Tensor<float> positions({2, 3, total_tokens, 1});
        int64_t token = 0;

        for (int64_t t = 0; t < target_latent_frames; t++) {
            float t_start = ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
            float t_end   = ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
            for (int64_t h = 0; h < height; h++) {
                float h_start = static_cast<float>(h * spatial_scale);
                float h_end   = static_cast<float>((h + 1) * spatial_scale);
                for (int64_t w = 0; w < width; w++) {
                    float w_start = static_cast<float>(w * spatial_scale);
                    float w_end   = static_cast<float>((w + 1) * spatial_scale);
                    set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
                }
            }
        }

        for (int64_t t = 0; t < keyframe_latent_frames; t++) {
            float t_start = static_cast<float>(keyframe_frame_idx + t * temporal_scale);
            float t_end   = static_cast<float>(keyframe_frame_idx + (t + 1) * temporal_scale);
            if (keyframe_pixel_frames == 1) {
                t_end = t_start + 1.f;
            }
            t_start /= static_cast<float>(fps);
            t_end /= static_cast<float>(fps);
            for (int64_t h = 0; h < height; h++) {
                float h_start = static_cast<float>(h * spatial_scale);
                float h_end   = static_cast<float>((h + 1) * spatial_scale);
                for (int64_t w = 0; w < width; w++) {
                    float w_start = static_cast<float>(w * spatial_scale);
                    float w_end   = static_cast<float>((w + 1) * spatial_scale);
                    set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
                }
            }
        }

        return positions;
    }

    static sd::Tensor<float> pack_ltxav_audio_and_video_latents(const sd::Tensor<float>& video_latent,
                                                                const sd::Tensor<float>& audio_latent) {
        if (audio_latent.empty()) {
            return video_latent;
        }

        GGML_ASSERT(video_latent.dim() == 4 || video_latent.dim() == 5);
        GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
        if (video_latent.dim() == 5) {
            GGML_ASSERT(video_latent.shape()[4] == 1);
        }
        if (audio_latent.dim() == 4) {
            GGML_ASSERT(audio_latent.shape()[3] == 1);
        }

        int64_t width        = video_latent.shape()[0];
        int64_t height       = video_latent.shape()[1];
        int64_t frames       = video_latent.shape()[2];
        int64_t video_ch     = video_latent.shape()[3];
        int64_t spatial_size = width * height * frames;
        int64_t audio_values = audio_latent.numel();
        int64_t extra_ch     = (audio_values + spatial_size - 1) / spatial_size;

        std::vector<int64_t> packed_shape = video_latent.shape();
        packed_shape[3]                   = video_ch + extra_ch;
        sd::Tensor<float> packed          = sd::zeros<float>(packed_shape);

        std::copy_n(video_latent.data(), video_latent.numel(), packed.data());
        std::copy_n(audio_latent.data(), audio_latent.numel(), packed.data() + video_latent.numel());
        return packed;
    }

    // kcpp {
    static sd_audio_t* clone_sd_audio(const sd_audio_t* src) {
        if (src == nullptr || src->data == nullptr || src->sample_rate == 0 || src->channels == 0 || src->sample_count == 0) {
            return nullptr;
        }

        sd_audio_t* audio = (sd_audio_t*)malloc(sizeof(sd_audio_t));
        if (audio == nullptr) {
            return nullptr;
        }

        audio->sample_rate  = src->sample_rate;
        audio->channels     = src->channels;
        audio->sample_count = src->sample_count;
        size_t sample_bytes = static_cast<size_t>(src->sample_count) * static_cast<size_t>(src->channels) * sizeof(float);
        audio->data         = (float*)malloc(sample_bytes);
        if (audio->data == nullptr) {
            free(audio);
            return nullptr;
        }

        std::memcpy(audio->data, src->data, sample_bytes);
        return audio;
    }

    static sd::Tensor<float> sd_audio_to_ltx_waveform_tensor(const sd_audio_t* audio,
            int target_sample_rate,
            int target_channels) {
        if (audio == nullptr || audio->data == nullptr || audio->sample_rate == 0 ||
                audio->channels == 0 || audio->sample_count == 0 || target_sample_rate <= 0 ||
                target_channels <= 0) {
            return {};
        }

        uint64_t out_samples_u64 = (audio->sample_count * static_cast<uint64_t>(target_sample_rate) +
                static_cast<uint64_t>(audio->sample_rate) - 1) /
            static_cast<uint64_t>(audio->sample_rate);
        if (out_samples_u64 == 0 || out_samples_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return {};
        }

        int64_t out_samples = static_cast<int64_t>(out_samples_u64);
        sd::Tensor<float> waveform({out_samples, target_channels, 1, 1});
        const double src_rate  = static_cast<double>(audio->sample_rate);
        const double dst_rate  = static_cast<double>(target_sample_rate);
        const int src_channels = static_cast<int>(audio->channels);

        auto src_value = [&](uint64_t sample, int channel) -> float {
            int src_channel = channel;
            if (src_channels == 1) {
                src_channel = 0;
            } else if (channel >= src_channels) {
                src_channel = src_channels - 1;
            }
            return audio->data[static_cast<size_t>(sample) * static_cast<size_t>(src_channels) + static_cast<size_t>(src_channel)];
        };

        for (int64_t t = 0; t < out_samples; ++t) {
            double src_pos = static_cast<double>(t) * src_rate / dst_rate;
            uint64_t i0    = static_cast<uint64_t>(std::floor(src_pos));
            uint64_t i1    = std::min<uint64_t>(i0 + 1, audio->sample_count - 1);
            float frac     = static_cast<float>(src_pos - static_cast<double>(i0));
            for (int ch = 0; ch < target_channels; ++ch) {
                float v0                    = src_value(i0, ch);
                float v1                    = src_value(i1, ch);
                waveform.index(t, ch, 0, 0) = v0 + (v1 - v0) * frac;
            }
        }

        return waveform;
    }
    // kcpp }

    static sd::Tensor<float> pack_ltxav_audio_and_video_denoise_mask(const sd::Tensor<float>& video_mask,
                                                                     const sd::Tensor<float>& video_latent,
                                                                     const sd::Tensor<float>& audio_latent, float audio_mask_value = 1.f) { //kcpp
        if (video_mask.empty() || audio_latent.empty()) {
            return video_mask;
        }

        GGML_ASSERT(video_latent.dim() == 4 || video_latent.dim() == 5);
        GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
        if (video_latent.dim() == 5) {
            GGML_ASSERT(video_latent.shape()[4] == 1);
        }
        if (audio_latent.dim() == 4) {
            GGML_ASSERT(audio_latent.shape()[3] == 1);
        }

        int64_t width        = video_latent.shape()[0];
        int64_t height       = video_latent.shape()[1];
        int64_t frames       = video_latent.shape()[2];
        int64_t video_ch     = video_latent.shape()[3];
        int64_t spatial_size = width * height * frames;
        int64_t audio_values = audio_latent.numel();
        int64_t extra_ch     = (audio_values + spatial_size - 1) / spatial_size;

        GGML_ASSERT(video_mask.dim() == video_latent.dim());
        GGML_ASSERT(video_mask.shape()[0] == width);
        GGML_ASSERT(video_mask.shape()[1] == height);
        GGML_ASSERT(video_mask.shape()[2] == frames);
        if (video_mask.dim() == 5) {
            GGML_ASSERT(video_mask.shape()[4] == video_latent.shape()[4]);
        }

        int64_t mask_ch = video_mask.shape()[3];
        if (mask_ch == video_ch + extra_ch) {
            return video_mask;
        }
        GGML_ASSERT(mask_ch == 1 || mask_ch == video_ch);

        sd::Tensor<float> video_mask_full = video_mask;
        if (mask_ch == 1 && video_ch != 1) {
            video_mask_full = video_mask * sd::Tensor<float>::ones(video_latent.shape());
        }

        std::vector<int64_t> audio_mask_shape = video_latent.shape();
        audio_mask_shape[3]                   = extra_ch;
        auto audio_mask                       = sd::full<float>(audio_mask_shape, audio_mask_value); //kcpp
        return sd::ops::concat(video_mask_full, audio_mask, 3);
    }

    static sd::Tensor<float> make_ltxav_video_denoise_mask(const sd::Tensor<float>& video_latent, float value = 1.f) {
        if (video_latent.empty()) {
            return {};
        }
        return sd::full<float>({video_latent.shape()[0],
                                video_latent.shape()[1],
                                video_latent.shape()[2],
                                1,
                                1},
                               value);
    }

    static sd::Tensor<float> encode_ltxav_condition_image(StableDiffusionGGML* sd,
                                                          const sd::Tensor<float>& image,
                                                          const char* name) {
        if (sd == nullptr || image.empty()) {
            return {};
        }
        auto condition_image  = image.reshape({image.shape()[0],
                                               image.shape()[1],
                                               1,
                                               image.shape()[2],
                                               image.shape()[3]});
        auto condition_latent = sd->encode_first_stage(condition_image);
        if (condition_latent.empty()) {
            LOG_ERROR("failed to encode LTXAV %s image", name);
        }
        return condition_latent;
    }

    static bool apply_ltxav_condition_by_latent_index(sd::Tensor<float>* video_latent,
                                                      sd::Tensor<float>* video_mask,
                                                      const sd::Tensor<float>& condition_latent,
                                                      int64_t latent_idx,
                                                      const char* name,
                                                      float conditioned_mask) {
        if (video_latent == nullptr || video_mask == nullptr || video_latent->empty() || video_mask->empty()) {
            return false;
        }
        if (condition_latent.empty() ||
            condition_latent.shape()[0] != video_latent->shape()[0] ||
            condition_latent.shape()[1] != video_latent->shape()[1] ||
            condition_latent.shape()[3] != video_latent->shape()[3]) {
            LOG_ERROR("invalid LTXAV %s condition latent shape", name);
            return false;
        }
        int64_t latent_frames    = video_latent->shape()[2];
        int64_t condition_frames = condition_latent.shape()[2];
        if (latent_idx < 0 || condition_frames <= 0 || latent_idx + condition_frames > latent_frames) {
            LOG_ERROR("invalid LTXAV %s image latent range: start=%" PRId64 ", length=%" PRId64 ", latent_frames=%" PRId64,
                      name,
                      latent_idx,
                      condition_frames,
                      latent_frames);
            return false;
        }

        sd::ops::slice_assign(video_latent, 2, latent_idx, latent_idx + condition_frames, condition_latent);
        sd::ops::fill_slice(video_mask, 2, latent_idx, latent_idx + condition_frames, conditioned_mask);
        return true;
    }

    static bool apply_ltxav_condition_image_by_latent_index(StableDiffusionGGML* sd,
                                                            const sd::Tensor<float>& image,
                                                            sd::Tensor<float>* video_latent,
                                                            sd::Tensor<float>* video_mask,
                                                            int64_t latent_idx,
                                                            const char* name,
                                                            float strength) {
        auto condition_latent = encode_ltxav_condition_image(sd, image, name);
        return !condition_latent.empty() &&
               apply_ltxav_condition_by_latent_index(video_latent,
                                                     video_mask,
                                                     condition_latent,
                                                     latent_idx,
                                                     name,
                                                     1.0f - std::clamp(strength, 0.f, 1.f));
    }

    static sd::Tensor<float> unpack_ltxav_audio_latent(const sd::Tensor<float>& packed_latent,
                                                       int audio_length,
                                                       int video_channels) {
        if (packed_latent.empty() || audio_length <= 0) {
            return {};
        }

        GGML_ASSERT(packed_latent.dim() == 4 || packed_latent.dim() == 5);
        int64_t width          = packed_latent.shape()[0];
        int64_t height         = packed_latent.shape()[1];
        int64_t frames         = packed_latent.shape()[2];
        int64_t total_channels = packed_latent.shape()[3];
        int64_t spatial_size   = width * height * frames;
        if (total_channels <= video_channels) {
            return {};
        }

        constexpr int kLtxavAudioFrequencyBins = 16;
        constexpr int kLtxavAudioChannels      = 8;
        int64_t required_values                = static_cast<int64_t>(audio_length) * kLtxavAudioFrequencyBins * kLtxavAudioChannels;
        int64_t packed_values                  = (total_channels - video_channels) * spatial_size;
        if (packed_values < required_values) {
            return {};
        }

        sd::Tensor<float> audio_latent({kLtxavAudioFrequencyBins, audio_length, kLtxavAudioChannels, 1});
        const float* audio_src = packed_latent.data() + static_cast<size_t>(video_channels) * static_cast<size_t>(spatial_size);
        std::copy_n(audio_src, static_cast<size_t>(required_values), audio_latent.data());
        return audio_latent;
    }

    static sd::Tensor<float> make_ltxav_empty_audio_latent(int audio_length) {
        if (audio_length <= 0) {
            return {};
        }
        constexpr int kLtxavAudioFrequencyBins = 16;
        constexpr int kLtxavAudioChannels      = 8;
        return sd::zeros<float>({kLtxavAudioFrequencyBins, audio_length, kLtxavAudioChannels, 1});
    }

    static sd::Tensor<float> resize_ltxav_audio_latent(const sd::Tensor<float>& audio_latent,
                                                       int target_audio_length) {
        auto resized = make_ltxav_empty_audio_latent(target_audio_length);
        if (resized.empty() || audio_latent.empty()) {
            return resized;
        }
        GGML_ASSERT(audio_latent.dim() == 3 || audio_latent.dim() == 4);
        int copy_length = std::min(static_cast<int>(audio_latent.shape()[1]), target_audio_length);
        if (copy_length > 0) {
            auto copied = sd::ops::slice(audio_latent, 1, 0, copy_length);
            sd::ops::slice_assign(&resized, 1, 0, copy_length, copied);
        }
        return resized;
    }

    static int get_ltxav_num_audio_latents(int frames, int fps) {
        GGML_ASSERT(frames > 0);
        GGML_ASSERT(fps > 0);
        constexpr float kSampleRate            = 16000.0f;
        constexpr float kMelHopLength          = 160.0f;
        constexpr float kAudioLatentDownsample = 4.0f;
        constexpr float kLatentsPerSecond      = kSampleRate / kMelHopLength / kAudioLatentDownsample;
        return static_cast<int>(std::ceil((static_cast<float>(frames) / static_cast<float>(fps)) * kLatentsPerSecond));
    }

    static int get_minimax_h3_num_audio_latents(int frames, int fps) {
        GGML_ASSERT(frames > 0 && fps > 0);
        return std::max(1,
                        static_cast<int>(std::lround(
                            static_cast<double>(frames) * 40.0 / fps)));
    }

    static sd::Tensor<float> make_minimax_h3_empty_audio_latent(int audio_length) {
        if (audio_length <= 0) {
            return {};
        }
        return sd::zeros<float>({audio_length, 2, 32, 1});
    }

    static sd::Tensor<float> prepare_minimax_h3_reference_waveform(const sd_audio_t& audio,
                                                                   int target_sample_rate = 32000) {
        if (audio.data == nullptr || audio.sample_count == 0 || audio.channels == 0 || audio.sample_rate == 0) {
            return {};
        }
        uint64_t output_samples = static_cast<uint64_t>(std::llround(
            static_cast<long double>(audio.sample_count) * target_sample_rate / audio.sample_rate));
        output_samples          = std::max<uint64_t>(1, output_samples);
        uint64_t padded_samples = (output_samples + 799) / 800 * 800;
        // Keep stereo streams planar for the mono-per-stream audio encoder:
        // [samples, 1, stereo, batch]. This avoids flattening interleaved L/R
        // storage into alternating samples when the encoder folds streams into
        // its batch dimension.
        sd::Tensor<float> waveform({static_cast<int64_t>(padded_samples), 1, 2, 1});

        for (uint64_t i = 0; i < output_samples; ++i) {
            long double source_pos = static_cast<long double>(i) * audio.sample_rate / target_sample_rate;
            uint64_t source0       = std::min<uint64_t>(static_cast<uint64_t>(source_pos), audio.sample_count - 1);
            uint64_t source1       = std::min<uint64_t>(source0 + 1, audio.sample_count - 1);
            float fraction         = static_cast<float>(source_pos - source0);
            for (uint32_t channel = 0; channel < 2; ++channel) {
                uint32_t source_channel = audio.channels == 1 ? 0 : std::min<uint32_t>(channel, audio.channels - 1);
                float a                 = audio.data[source0 * audio.channels + source_channel];
                float b                 = audio.data[source1 * audio.channels + source_channel];
                waveform.index(static_cast<int64_t>(i), 0, channel, 0) =
                    std::clamp(a + (b - a) * fraction, -1.f, 1.f);
            }
        }
        return waveform;
    }

    static sd::Tensor<float> unpack_minimax_h3_audio_latent(const sd::Tensor<float>& packed_latent,
                                                            int audio_length,
                                                            int video_channels) {
        if (packed_latent.empty() || audio_length <= 0) {
            return {};
        }
        GGML_ASSERT(packed_latent.dim() == 4 || packed_latent.dim() == 5);
        int64_t spatial_size = packed_latent.shape()[0] * packed_latent.shape()[1] * packed_latent.shape()[2];
        int64_t required     = static_cast<int64_t>(audio_length) * 2 * 32;
        int64_t available    = (packed_latent.shape()[3] - video_channels) * spatial_size;
        if (available < required) {
            return {};
        }
        sd::Tensor<float> audio({audio_length, 2, 32, 1});
        const float* source = packed_latent.data() +
                              static_cast<size_t>(video_channels) * static_cast<size_t>(spatial_size);
        std::copy_n(source, static_cast<size_t>(required), audio.data());
        return audio;
    }

    static std::optional<ImageGenerationLatents> prepare_video_generation_latents(StableDiffusionGGML* sd,
                                                                                  const sd_vid_gen_params_t* sd_vid_gen_params,
                                                                                  GenerationRequest* request) {
        ImageGenerationLatents latents;
        int64_t prepare_start_ms = ggml_time_ms();

        sd::Tensor<float> start_image;
        sd::Tensor<float> end_image;

        struct RunnerEndGuard {
            GGMLRunner* runner = nullptr;
            ~RunnerEndGuard() {
                if (runner) {
                    runner->runner_end();
                }
            }
        } runner_guard{sd->audio_vae_model.get()};

        if (sd_vid_gen_params->init_image.data) {
            start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, request->width, request->height);
        }

        if (sd_vid_gen_params->end_image.data) {
            end_image = sd_image_to_tensor(sd_vid_gen_params->end_image, request->width, request->height);
        }

        if (sd_version_is_minimax_h3(sd->version)) {
            if (sd_vid_gen_params->ref_images_count < 0 || sd_vid_gen_params->ref_videos_count < 0 ||
                sd_vid_gen_params->ref_audios_count < 0 ||
                (sd_vid_gen_params->ref_images_count > 0 && sd_vid_gen_params->ref_images == nullptr) ||
                (sd_vid_gen_params->ref_videos_count > 0 && sd_vid_gen_params->ref_videos == nullptr) ||
                (sd_vid_gen_params->ref_audios_count > 0 && sd_vid_gen_params->ref_audios == nullptr)) {
                LOG_ERROR("invalid MiniMax-H3 Ref2VA input arrays");
                return std::nullopt;
            }

            latents.audio_length = get_minimax_h3_num_audio_latents(request->frames,
                                                                    request->fps);
            latents.audio_latent = make_minimax_h3_empty_audio_latent(latents.audio_length);

            bool has_references = sd_vid_gen_params->ref_images_count > 0 ||
                                  sd_vid_gen_params->ref_videos_count > 0 ||
                                  sd_vid_gen_params->ref_audios_count > 0;
            if (has_references && (!start_image.empty() || !end_image.empty())) {
                LOG_ERROR("MiniMax-H3 keyframes and Ref2VA references cannot be used together");
                return std::nullopt;
            }

            if (sd_vid_gen_params->control_frames_size > 0) {
                LOG_ERROR("MiniMax-H3 control_frames are not implemented");
                return std::nullopt;
            }

            auto add_visual_noise = [&](sd::Tensor<float> latent) {
                auto condition_rng = std::make_shared<PhiloxRNG>();
                condition_rng->manual_seed(static_cast<uint64_t>(request->seed));
                return latent * MiniMaxH3::VISUAL_COND_TIMESTEP +
                       sd::Tensor<float>::randn_like(latent, condition_rng) *
                           (1.f - MiniMaxH3::VISUAL_COND_TIMESTEP);
            };

            auto add_keyframe = [&](const sd::Tensor<float>& image,
                                    int32_t frame_index,
                                    const char* name) -> bool {
                if (image.empty()) {
                    return true;
                }
                auto video_image = image.reshape({image.shape()[0],
                                                  image.shape()[1],
                                                  1,
                                                  image.shape()[2],
                                                  image.shape()[3]});
                auto latent      = sd->encode_first_stage(video_image);
                if (latent.empty()) {
                    LOG_ERROR("failed to encode MiniMax-H3 %s keyframe", name);
                    return false;
                }
                latents.ref_images.push_back(image);
                latents.ref_latents.push_back(add_visual_noise(std::move(latent)));
                latents.keyframe_indices.push_back(frame_index);
                return true;
            };

            auto resize_reference = [&](const sd::Tensor<float>& image,
                                        int width,
                                        int height) {
                return sd::ops::interpolate(
                    image,
                    std::vector<int64_t>{width, height, image.shape()[2], image.shape()[3]});
            };

            auto encode_reference_audio = [&](const sd_audio_t& audio,
                                              int32_t* audio_index) -> bool {
                if (sd->audio_vae_model == nullptr) {
                    LOG_ERROR("MiniMax-H3 Ref2VA audio requires --audio-vae with encoder weights");
                    return false;
                }
                auto waveform = prepare_minimax_h3_reference_waveform(
                    audio,
                    sd->audio_vae_model->input_sample_rate());
                if (waveform.empty()) {
                    LOG_ERROR("invalid MiniMax-H3 reference audio");
                    return false;
                }
                auto encoded = sd->audio_vae_model->encode(sd->n_threads, waveform);
                if (encoded.empty()) {
                    LOG_ERROR("failed to encode MiniMax-H3 reference audio");
                    return false;
                }
                *audio_index = static_cast<int32_t>(latents.reference_audio_latents.size());
                latents.reference_audio_latents.push_back(std::move(encoded));
                return true;
            };

            if (has_references) {
                LOG_INFO("MiniMax-H3 Ref2VA: %d image(s), %d video(s), %d audio clip(s)",
                         sd_vid_gen_params->ref_images_count,
                         sd_vid_gen_params->ref_videos_count,
                         sd_vid_gen_params->ref_audios_count);

                for (int i = 0; i < sd_vid_gen_params->ref_images_count; ++i) {
                    auto image = ensure_image_tensor_channels(
                        sd_image_to_tensor(sd_vid_gen_params->ref_images[i]),
                        3);
                    if (image.empty()) {
                        LOG_ERROR("failed to load MiniMax-H3 reference image %d", i + 1);
                        return std::nullopt;
                    }
                    int source_w       = static_cast<int>(image.shape()[0]);
                    int source_h       = static_cast<int>(image.shape()[1]);
                    double source_area = static_cast<double>(source_w) * source_h;
                    double target_area = static_cast<double>(request->width) * request->height;
                    double scale       = std::min(1.0, std::sqrt(target_area / source_area));
                    int width          = std::max(32, static_cast<int>(std::round(source_w * scale / 32.f)) * 32);
                    int height         = std::max(32, static_cast<int>(std::round(source_h * scale / 32.f)) * 32);
                    image              = resize_reference(image, width, height);
                    auto latent        = sd->encode_first_stage(image);
                    if (latent.empty()) {
                        LOG_ERROR("failed to encode MiniMax-H3 reference image %d", i + 1);
                        return std::nullopt;
                    }
                    int32_t video_index = static_cast<int32_t>(latents.ref_latents.size());
                    latents.ref_latents.push_back(add_visual_noise(std::move(latent)));
                    latents.minimax_reference_blocks.push_back({MiniMaxH3ReferenceKind::IMAGE,
                                                                video_index,
                                                                -1});
                    MiniMaxH3PresentationItem item;
                    item.kind = MiniMaxH3PresentationKind::IMAGE;
                    item.frames.push_back(std::move(image));
                    latents.minimax_presentation_refs.push_back(std::move(item));
                }

                for (int video_idx = 0; video_idx < sd_vid_gen_params->ref_videos_count; ++video_idx) {
                    const auto& reference = sd_vid_gen_params->ref_videos[video_idx];
                    if (reference.frames == nullptr || reference.frame_count < 1) {
                        LOG_ERROR("invalid MiniMax-H3 reference video %d", video_idx + 1);
                        return std::nullopt;
                    }
                    int source_fps        = reference.fps > 0 ? reference.fps : 24;
                    int normalized_frames = static_cast<int>(std::lround(
                        static_cast<double>(reference.frame_count) * 24.0 / source_fps));
                    normalized_frames     = std::min(normalized_frames, request->frames);
                    if (normalized_frames < 5) {
                        LOG_ERROR("MiniMax-H3 reference video %d needs at least 5 frames at 24 fps",
                                  video_idx + 1);
                        return std::nullopt;
                    }
                    while (normalized_frames % 17 != 5) {
                        --normalized_frames;
                    }

                    auto first = ensure_image_tensor_channels(sd_image_to_tensor(reference.frames[0]), 3);
                    if (first.empty()) {
                        LOG_ERROR("invalid first frame in MiniMax-H3 reference video %d", video_idx + 1);
                        return std::nullopt;
                    }
                    int source_w     = static_cast<int>(first.shape()[0]);
                    int source_h     = static_cast<int>(first.shape()[1]);
                    double ratio     = static_cast<double>(source_w) / source_h;
                    double nominal_w = ratio >= 1.0 ? 768.0 * ratio : 768.0;
                    double nominal_h = ratio >= 1.0 ? 768.0 : 768.0 / ratio;
                    if (nominal_w * nominal_h > 768.0 * 1344.0) {
                        double scale = std::sqrt((768.0 * 1344.0) / (nominal_w * nominal_h));
                        nominal_w *= scale;
                        nominal_h *= scale;
                    }
                    int width  = std::max(32, static_cast<int>(std::round(nominal_w / 32.0)) * 32);
                    int height = std::max(32, static_cast<int>(std::round(nominal_h / 32.0)) * 32);
                    if (source_w * source_h < width * height) {
                        width  = std::max(32, static_cast<int>(std::round(source_w / 32.0)) * 32);
                        height = std::max(32, static_cast<int>(std::round(source_h / 32.0)) * 32);
                    }

                    sd::Tensor<float> video({width, height, normalized_frames, 3, 1});
                    for (int frame = 0; frame < normalized_frames; ++frame) {
                        int source_index = std::min(reference.frame_count - 1,
                                                    static_cast<int>(std::floor(frame * source_fps / 24.0)));
                        auto source      = ensure_image_tensor_channels(
                                 sd_image_to_tensor(reference.frames[source_index]),
                                 3);
                        if (source.empty()) {
                            LOG_ERROR("invalid frame %d in MiniMax-H3 reference video %d",
                                      source_index + 1,
                                      video_idx + 1);
                            return std::nullopt;
                        }
                        source = resize_reference(source, width, height);
                        sd::ops::slice_assign(&video, 2, frame, frame + 1, source.unsqueeze(2));
                    }
                    auto video_latent = sd->encode_first_stage(video);
                    if (video_latent.empty()) {
                        LOG_ERROR("failed to encode MiniMax-H3 reference video %d", video_idx + 1);
                        return std::nullopt;
                    }
                    int32_t audio_index = -1;
                    bool has_audio      = reference.audio.data != nullptr && reference.audio.sample_count > 0;
                    if (has_audio) {
                        if (!encode_reference_audio(reference.audio, &audio_index)) {
                            return std::nullopt;
                        }
                        MiniMaxH3PresentationItem audio_item;
                        audio_item.kind = MiniMaxH3PresentationKind::AUDIO;
                        latents.minimax_presentation_refs.push_back(std::move(audio_item));
                    }

                    MiniMaxH3PresentationItem video_item;
                    video_item.kind = MiniMaxH3PresentationKind::VIDEO;
                    for (int frame = 0; frame < normalized_frames; frame += 12) {
                        auto sampled = sd::ops::slice(video, 2, frame, frame + 1)
                                           .reshape({width, height, 3, 1});
                        video_item.frames.push_back(std::move(sampled));
                        video_item.timestamps.push_back(frame / 24.f);
                    }
                    latents.minimax_presentation_refs.push_back(std::move(video_item));

                    int32_t video_index = static_cast<int32_t>(latents.ref_latents.size());
                    latents.ref_latents.push_back(add_visual_noise(std::move(video_latent)));
                    latents.minimax_reference_blocks.push_back({has_audio ? MiniMaxH3ReferenceKind::VIDEO_AUDIO
                                                                          : MiniMaxH3ReferenceKind::VIDEO,
                                                                video_index,
                                                                audio_index});
                }

                for (int audio_idx = 0; audio_idx < sd_vid_gen_params->ref_audios_count; ++audio_idx) {
                    int32_t encoded_index = -1;
                    if (!encode_reference_audio(sd_vid_gen_params->ref_audios[audio_idx], &encoded_index)) {
                        return std::nullopt;
                    }
                    MiniMaxH3PresentationItem item;
                    item.kind = MiniMaxH3PresentationKind::AUDIO;
                    latents.minimax_presentation_refs.push_back(std::move(item));
                    latents.minimax_reference_blocks.push_back({MiniMaxH3ReferenceKind::AUDIO,
                                                                -1,
                                                                encoded_index});
                }
            }

            if (!has_references && (!start_image.empty() || !end_image.empty())) {
                LOG_INFO(!start_image.empty() && !end_image.empty() ? "MiniMax-H3 FL2VA" : !start_image.empty() ? "MiniMax-H3 I2VA"
                                                                                                                : "MiniMax-H3 end-frame conditioning");
            }
            if (!has_references &&
                (!add_keyframe(start_image, 0, "start") ||
                 !add_keyframe(end_image, request->frames - 1, "end"))) {
                return std::nullopt;
            }
        }

        if (sd_version_is_ltxav(sd->version)) {
            latents.audio_length = get_ltxav_num_audio_latents(request->frames, request->fps);
            latents.audio_latent = make_ltxav_empty_audio_latent(latents.audio_length);
            // kcpp
            auto ltx_audio_vae_model = std::dynamic_pointer_cast<LTXV::LTXAudioVAERunner>(sd->audio_vae_model);
            if (sd_vid_gen_params->input_audio != nullptr &&
                    sd_vid_gen_params->input_audio->data != nullptr &&
                    sd_vid_gen_params->input_audio->sample_count > 0) {
                if (ltx_audio_vae_model == nullptr || !ltx_audio_vae_model->config.has_encoder) {
                    LOG_ERROR("LTX A2V requires an audio VAE with encoder weights");
                    return std::nullopt;
                }

                int64_t audio_encode_start = ggml_time_ms();
                auto waveform              = sd_audio_to_ltx_waveform_tensor(sd_vid_gen_params->input_audio,
                        ltx_audio_vae_model->config.sample_rate,
                        ltx_audio_vae_model->config.audio_channels);
                if (waveform.empty()) {
                    LOG_ERROR("failed to convert source audio for LTX A2V encoding");
                    return std::nullopt;
                }

                auto encoded_audio_latent = ltx_audio_vae_model->encode(sd->n_threads, waveform);
                if (encoded_audio_latent.empty()) {
                    LOG_ERROR("LTX A2V audio latent encoding failed");
                    return std::nullopt;
                }

                latents.audio_latent = resize_ltxav_audio_latent(encoded_audio_latent, latents.audio_length);
                if (latents.audio_latent.empty()) {
                    LOG_ERROR("failed to resize encoded LTX A2V audio latent");
                    return std::nullopt;
                }

                int64_t audio_encode_end = ggml_time_ms();
                LOG_INFO("encoded LTX A2V source audio latent %dx%dx%dx%d -> length %d, taking %.2fs",
                        (int)encoded_audio_latent.shape()[0],
                        (int)encoded_audio_latent.shape()[1],
                        (int)encoded_audio_latent.shape()[2],
                        (int)encoded_audio_latent.shape()[3],
                        latents.audio_length,
                        (audio_encode_end - audio_encode_start) * 1.0f / 1000);
            }
        }

        if (sd_version_is_ltxav(sd->version)) {
            if (sd_vid_gen_params->control_frames_size > 0) {
                LOG_ERROR("LTXAV control_frames are not implemented");
                return std::nullopt;
            }

            if (!start_image.empty() || !end_image.empty()) {
                if (!start_image.empty() && !end_image.empty()) {
                    LOG_INFO("FLF2V");
                } else if (!start_image.empty()) {
                    LOG_INFO("IMG2VID");
                } else {
                    LOG_INFO("END2VID");
                }

                int64_t t1          = ggml_time_ms();
                latents.init_latent = sd->generate_init_latent(request->width, request->height, request->frames, true);

                float conditioning_strength = std::clamp(request->strength, 0.f, 1.f);
                float conditioned_mask      = 1.0f - conditioning_strength;
                latents.denoise_mask        = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);

                auto apply_video_condition_by_keyframe_index = [&](const sd::Tensor<float>& keyframes,
                                                                   int frame_idx,
                                                                   const char* name) -> bool {
                    int64_t keyframe_frames = keyframes.shape()[2];
                    if (keyframe_frames <= 0 || keyframes.shape()[0] != latents.init_latent.shape()[0] ||
                        keyframes.shape()[1] != latents.init_latent.shape()[1] ||
                        keyframes.shape()[3] != latents.init_latent.shape()[3]) {
                        LOG_ERROR("invalid LTXAV %s keyframe latent shape", name);
                        return false;
                    }

                    latents.video_target_frame_count       = latents.init_latent.shape()[2];
                    latents.video_conditioning_frame_count = keyframe_frames;
                    latents.init_latent                    = sd::ops::concat(latents.init_latent, keyframes, 2);

                    auto keyframe_mask      = sd::full<float>({keyframes.shape()[0],
                                                               keyframes.shape()[1],
                                                               keyframes.shape()[2],
                                                               1,
                                                               1},
                                                         conditioned_mask);
                    latents.denoise_mask    = sd::ops::concat(latents.denoise_mask, keyframe_mask, 2);
                    latents.video_positions = build_ltxv_video_positions(latents.init_latent.shape()[0],
                                                                         latents.init_latent.shape()[1],
                                                                         latents.video_target_frame_count,
                                                                         keyframe_frames,
                                                                         frame_idx,
                                                                         1,
                                                                         request->fps,
                                                                         request->vae_scale_factor,
                                                                         8,
                                                                         true);
                    return true;
                };

                if (!start_image.empty()) {
                    if (!apply_ltxav_condition_image_by_latent_index(sd,
                                                                     start_image,
                                                                     &latents.init_latent,
                                                                     &latents.denoise_mask,
                                                                     0,
                                                                     "init",
                                                                     conditioning_strength)) {
                        return std::nullopt;
                    }
                }

                if (!end_image.empty()) {
                    auto end_image_latent = encode_ltxav_condition_image(sd, end_image, "end");
                    if (end_image_latent.empty()) {
                        return std::nullopt;
                    }

                    int frame_idx = request->frames - 1;
                    bool ok       = frame_idx == 0 ? apply_ltxav_condition_by_latent_index(&latents.init_latent,
                                                                                           &latents.denoise_mask,
                                                                                           end_image_latent,
                                                                                           0,
                                                                                           "end",
                                                                                           conditioned_mask)
                                                   : apply_video_condition_by_keyframe_index(end_image_latent, frame_idx, "end");
                    if (!ok) {
                        return std::nullopt;
                    }
                }

                int64_t t2 = ggml_time_ms();
                LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
            }
        }

        if (sd_version_is_hunyuan_video(sd->version) &&
            (!start_image.empty() || !end_image.empty())) {
            LOG_INFO("Hunyuan Video IMG2VID");

            int64_t t1                  = ggml_time_ms();
            auto concat_latent          = sd->generate_init_latent(request->width,
                                                                   request->height,
                                                                   request->frames,
                                                                   true);
            auto encode_condition_frame = [&](const sd::Tensor<float>& image,
                                              int64_t latent_frame,
                                              const char* name) -> bool {
                auto encoded = sd->encode_first_stage(image.unsqueeze(2));
                if (encoded.empty()) {
                    LOG_ERROR("failed to encode Hunyuan Video %s conditioning frame", name);
                    return false;
                }
                if (encoded.dim() == 4) {
                    encoded.unsqueeze_(2);
                }
                if (encoded.dim() != 5 ||
                    encoded.shape()[0] != concat_latent.shape()[0] ||
                    encoded.shape()[1] != concat_latent.shape()[1] ||
                    encoded.shape()[3] != concat_latent.shape()[3]) {
                    LOG_ERROR("invalid Hunyuan Video %s conditioning latent shape", name);
                    return false;
                }
                sd::ops::slice_assign(&concat_latent,
                                      2,
                                      latent_frame,
                                      latent_frame + 1,
                                      sd::ops::slice(encoded, 2, 0, 1));
                return true;
            };

            if (!start_image.empty() && !encode_condition_frame(start_image, 0, "start")) {
                return std::nullopt;
            }
            if (!end_image.empty() &&
                !encode_condition_frame(end_image, concat_latent.shape()[2] - 1, "end")) {
                return std::nullopt;
            }

            sd::Tensor<float> concat_mask = sd::zeros<float>({concat_latent.shape()[0],
                                                              concat_latent.shape()[1],
                                                              concat_latent.shape()[2],
                                                              1,
                                                              1});
            if (!start_image.empty()) {
                sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
            }
            if (!end_image.empty()) {
                sd::ops::fill_slice(&concat_mask, 2, concat_mask.shape()[2] - 1, concat_mask.shape()[2], 1.0f);
            }
            latents.concat_latent = sd::ops::concat(concat_latent, concat_mask, 3);

            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
        }

        if (sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
            sd->diffusion_model->get_desc() == "Wan2.2-I2V-14B" ||
            sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
            sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
            LOG_INFO("IMG2VID");

            if (sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
                sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
                if (!start_image.empty()) {
                    auto clip_vision_output = sd->get_clip_vision_output(start_image, false, -2);
                    if (clip_vision_output.empty()) {
                        LOG_ERROR("failed to compute clip vision output for init image");
                        return std::nullopt;
                    }
                    latents.clip_vision_output = std::move(clip_vision_output);
                } else {
                    latents.clip_vision_output = sd->get_clip_vision_output(start_image, false, -2, true);
                }

                if (sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
                    sd::Tensor<float> end_image_clip_vision_output;
                    if (!end_image.empty()) {
                        end_image_clip_vision_output = sd->get_clip_vision_output(end_image, false, -2);
                        if (end_image_clip_vision_output.empty()) {
                            LOG_ERROR("failed to compute clip vision output for end image");
                            return std::nullopt;
                        }
                    } else {
                        end_image_clip_vision_output = sd->get_clip_vision_output(end_image, false, -2, true);
                    }
                    latents.clip_vision_output = sd::ops::concat(latents.clip_vision_output, end_image_clip_vision_output, 1);
                }

                int64_t t1 = ggml_time_ms();
                LOG_INFO("get_clip_vision_output completed, taking %" PRId64 " ms", t1 - prepare_start_ms);
            }

            int64_t t1              = ggml_time_ms();
            sd::Tensor<float> image = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
            if (!start_image.empty()) {
                sd::ops::slice_assign(&image, 2, 0, 1, start_image.unsqueeze(2));
            }
            if (!end_image.empty()) {
                sd::ops::slice_assign(&image, 2, request->frames - 1, request->frames, end_image.unsqueeze(2));
            }

            auto concat_latent = sd->encode_first_stage(image);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
            if (concat_latent.empty()) {
                LOG_ERROR("failed to encode video conditioning frames");
                return std::nullopt;
            }
            latents.concat_latent = std::move(concat_latent);

            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);

            sd::Tensor<float> concat_mask = sd::zeros<float>({latents.concat_latent.shape()[0],
                                                              latents.concat_latent.shape()[1],
                                                              latents.concat_latent.shape()[2],
                                                              4,
                                                              1});  // [b, 4, t, h/vae_scale_factor, w/vae_scale_factor]
            if (!start_image.empty()) {
                sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
            }
            if (!end_image.empty()) {
                auto last_channel = sd::ops::slice(concat_mask, 3, 3, 4);
                sd::ops::fill_slice(&last_channel, 2, last_channel.shape()[2] - 1, last_channel.shape()[2], 1.0f);
                sd::ops::slice_assign(&concat_mask, 3, 3, 4, last_channel);
            }
            latents.concat_latent = sd::ops::concat(concat_mask, latents.concat_latent, 3);  // [b, 4+c, t, h/vae_scale_factor, w/vae_scale_factor]
        } else if (sd->diffusion_model->get_desc() == "Wan2.2-TI2V-5B" && !start_image.empty()) {
            LOG_INFO("IMG2VID");

            int64_t t1             = ggml_time_ms();
            auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
            auto init_image_latent = sd->encode_first_stage(init_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
            if (init_image_latent.empty()) {
                LOG_ERROR("failed to encode init video frame");
                return std::nullopt;
            }

            latents.init_latent = sd->generate_init_latent(request->width, request->height, request->frames, true);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
            sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

            latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
            sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

            if (!end_image.empty()) {
                auto end_img          = end_image.reshape({end_image.shape()[0], end_image.shape()[1], 1, end_image.shape()[2], 1});
                auto end_image_latent = sd->encode_first_stage(end_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
                if (end_image_latent.empty()) {
                    LOG_ERROR("failed to encode end video frame");
                    return std::nullopt;
                }
                sd::ops::slice_assign(&latents.init_latent, 2, latents.init_latent.shape()[2] - 1, latents.init_latent.shape()[2], end_image_latent);
                sd::ops::fill_slice(&latents.denoise_mask, 2, latents.init_latent.shape()[2] - 1, latents.init_latent.shape()[2], 0.0f);
            }

            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
        } else if (sd_version_is_lingbot_video(sd->version) && !start_image.empty()) {
            LOG_INFO("LingBot Video IMG2VID");

            int64_t t1             = ggml_time_ms();
            auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
            auto init_image_latent = sd->encode_first_stage(init_img);
            if (init_image_latent.empty()) {
                LOG_ERROR("failed to encode init video frame");
                return std::nullopt;
            }

            latents.init_latent = sd->generate_init_latent(request->width, request->height, request->frames, true);
            sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

            latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
            sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

            latents.ref_images.push_back(start_image);

            int64_t t2 = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
        } else if (sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
                   sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B") {
            LOG_INFO("VACE");
            int64_t t1 = ggml_time_ms();
            sd::Tensor<float> ref_image_latent;
            if (!start_image.empty()) {
                auto ref_img     = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
                auto encoded_ref = sd->encode_first_stage(ref_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
                if (encoded_ref.empty()) {
                    LOG_ERROR("failed to encode VACE reference image");
                    return std::nullopt;
                }
                ref_image_latent = sd::ops::concat(encoded_ref, sd::zeros<float>(encoded_ref.shape()), 3);  // [b, 2*c, 1, h/vae_scale_factor, w/vae_scale_factor]
            }

            sd::Tensor<float> control_video = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
            int64_t control_frame_count     = std::min<int64_t>(request->frames, sd_vid_gen_params->control_frames_size);
            for (int64_t i = 0; i < control_frame_count; ++i) {
                auto control_frame = sd_image_to_tensor(sd_vid_gen_params->control_frames[i], request->width, request->height);
                sd::ops::slice_assign(&control_video, 2, i, i + 1, control_frame.unsqueeze(2));
            }

            sd::Tensor<float> mask = sd::full<float>({request->width, request->height, request->frames, 1, 1}, 1.0f);

            control_video              = control_video - 0.5f;
            sd::Tensor<float> inactive = control_video * (1.0f - mask) + 0.5f;
            sd::Tensor<float> reactive = control_video * mask + 0.5f;

            inactive = sd->encode_first_stage(inactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
            if (inactive.empty()) {
                LOG_ERROR("failed to encode VACE inactive context");
                return std::nullopt;
            }

            reactive = sd->encode_first_stage(reactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
            if (reactive.empty()) {
                LOG_ERROR("failed to encode VACE reactive context");
                return std::nullopt;
            }

            int64_t length = inactive.shape()[2];
            if (!ref_image_latent.empty()) {
                length += 1;
                request->frames       = static_cast<int>((length - 1) * 4 + 1);
                latents.ref_image_num = 1;
            }
            auto vace_context = sd::ops::concat(inactive, reactive, 3);  // [b, 2*c, t, h/vae_scale_factor, w/vae_scale_factor]

            mask              = sd::full<float>({request->width, request->height, inactive.shape()[2], 1, 1}, 1.0f);
            auto mask_context = mask.reshape({request->vae_scale_factor,
                                              inactive.shape()[0],
                                              request->vae_scale_factor,
                                              inactive.shape()[1],
                                              inactive.shape()[2]});   // [t, h/vae_scale_factor, vae_scale_factor, w/vae_scale_factor, vae_scale_factor]
            mask_context      = mask_context.permute({1, 3, 4, 0, 2})  // [vae_scale_factor, vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]
                               .reshape({inactive.shape()[0],
                                         inactive.shape()[1],
                                         inactive.shape()[2],
                                         request->vae_scale_factor * request->vae_scale_factor});  // [vae_scale_factor*vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]

            if (!ref_image_latent.empty()) {
                vace_context  = sd::ops::concat(ref_image_latent, vace_context, 2);  // [b, 2*c, t+1, h/vae_scale_factor, w/vae_scale_factor]
                auto mask_pad = sd::zeros<float>({mask_context.shape()[0],
                                                  mask_context.shape()[1],
                                                  1,
                                                  mask_context.shape()[3]});  // [vae_scale_factor*vae_scale_factor, 1, h/vae_scale_factor, w/vae_scale_factor]
                mask_context  = sd::ops::concat(mask_pad, mask_context, 2);   // [vae_scale_factor*vae_scale_factor, t + 1, h/vae_scale_factor, w/vae_scale_factor]
            }

            mask_context.unsqueeze_(mask_context.dim());  // [b, vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]

            latents.vace_context = sd::ops::concat(vace_context, mask_context, 3);  // [b, 2*c + vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]
            int64_t t2           = ggml_time_ms();
            LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
        }

        if (latents.init_latent.empty()) {
            latents.init_latent = sd->generate_init_latent(request->width, request->height, request->frames, true);
        }

        if ((sd_version_is_ltxav(sd->version) || sd_version_is_minimax_h3(sd->version)) &&
            !latents.audio_latent.empty()) {
            //kcpp
            bool has_input_audio = sd_vid_gen_params->input_audio != nullptr &&
                sd_vid_gen_params->input_audio->data != nullptr &&
                sd_vid_gen_params->input_audio->sample_count > 0;
            if (has_input_audio && latents.denoise_mask.empty()) {
                latents.denoise_mask = make_ltxav_video_denoise_mask(latents.init_latent, 1.f);
            }
            if (!latents.denoise_mask.empty()) {
                latents.denoise_mask = pack_ltxav_audio_and_video_denoise_mask(latents.denoise_mask,
                                                                               latents.init_latent,
                                                                               latents.audio_latent, has_input_audio ? 0.f : 1.f);
            }
            latents.init_latent = pack_ltxav_audio_and_video_latents(latents.init_latent, latents.audio_latent);
        }

        return latents;
    }

    static ImageGenerationEmbeds prepare_video_generation_embeds(StableDiffusionGGML* sd,
                                                                 const sd_vid_gen_params_t* sd_vid_gen_params,
                                                                 const GenerationRequest& request,
                                                                 const ImageGenerationLatents& latents) {
        ConditionerRunnerEndOnExit conditioner_runner_end{sd->cond_stage_model.get()};

        ImageGenerationEmbeds embeds;
        ConditionerParams condition_params;
        condition_params.clip_skip             = request.clip_skip;
        condition_params.text                  = request.prompt;
        condition_params.zero_out_masked       = true;
        condition_params.ref_images            = &latents.ref_images;
        condition_params.minimax_h3_references = &latents.minimax_presentation_refs;
        if (sd_version_is_lingbot_video(sd->version) || sd_version_is_minimax_h3(sd->version)) {
            condition_params.ref_image_params.vlm_resize_mode = RefImageResizeMode::AREA;
        }

        int64_t prepare_start_ms = ggml_time_ms();
        embeds.cond              = sd->cond_stage_model->get_learned_condition(sd->n_threads,
                                                                               condition_params);
        embeds.cond.c_concat     = latents.concat_latent;
        embeds.cond.c_vector     = latents.clip_vision_output;
        if (sd_version_is_minimax_h3(sd->version)) {
            embeds.cond.c_ref_images       = latents.ref_latents;
            embeds.cond.c_ref_audios       = latents.reference_audio_latents;
            embeds.cond.c_reference_blocks = latents.minimax_reference_blocks;
            if (!latents.keyframe_indices.empty()) {
                embeds.cond.c_position_ids = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(latents.keyframe_indices.size())},
                    latents.keyframe_indices);
            }
        }
        if (request.use_uncond) {
            condition_params.text  = request.negative_prompt;
            embeds.uncond          = sd->cond_stage_model->get_learned_condition(sd->n_threads,
                                                                                 condition_params);
            embeds.uncond.c_concat = latents.concat_latent;
            embeds.uncond.c_vector = latents.clip_vision_output;
            if (sd_version_is_minimax_h3(sd->version)) {
                embeds.uncond.c_ref_images       = latents.ref_latents;
                embeds.uncond.c_ref_audios       = latents.reference_audio_latents;
                embeds.uncond.c_reference_blocks = latents.minimax_reference_blocks;
                embeds.uncond.c_position_ids     = embeds.cond.c_position_ids;
            }
        }

        int64_t t1 = ggml_time_ms();
        LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

        return embeds;
    }

    static sd_image_t* decode_video_outputs(StableDiffusionGGML* sd,
                                            const GenerationRequest& request,
                                            const sd::Tensor<float>& final_latent,
                                            int* num_frames_out) {
        if (final_latent.empty()) {
            LOG_ERROR("no latent video to decode");
            return nullptr;
        }
        if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling video decode");
            return nullptr;
        }
        sd::Tensor<float> video_latent = final_latent;
        if ((sd_version_is_ltxav(sd->version) || sd_version_is_minimax_h3(sd->version)) &&
            video_latent.shape()[3] > sd->get_latent_channel()) {
            video_latent = sd::ops::slice(video_latent, 3, 0, sd->get_latent_channel());
        }
        LOG_VERBOSE("decode_video_outputs latent %dx%dx%dx%d",
                    (int)video_latent.shape()[0],
                    (int)video_latent.shape()[1],
                    (int)video_latent.shape()[2],
                    (int)video_latent.shape()[3]);
        // auto z = sd::load_tensor_from_file_as_tensor<float>("ltx_vae_z.bin");
        int64_t t4            = ggml_time_ms();
        sd::Tensor<float> vid = sd->decode_first_stage(video_latent, true);
        int64_t t5            = ggml_time_ms();
        LOG_INFO("decode_first_stage completed, taking %.2fs", (t5 - t4) * 1.0f / 1000);
        if (vid.empty()) {
            LOG_ERROR("decode_first_stage failed for video");
            return nullptr;
        }
        LOG_VERBOSE("decode_video_outputs decoded %dx%dx%dx%d",
                    (int)vid.shape()[0],
                    (int)vid.shape()[1],
                    (int)vid.shape()[2],
                    (int)vid.shape()[3]);
        if (request.frames > 0 &&
            vid.shape()[2] > request.frames) {
            vid = sd::ops::slice(vid, 2, 0, request.frames);
        }

        sd_image_t* result_images = (sd_image_t*)calloc(vid.shape()[2], sizeof(sd_image_t));
        if (result_images == nullptr) {
            return nullptr;
        }
        if (num_frames_out != nullptr) {
            *num_frames_out = static_cast<int>(vid.shape()[2]);
        }

        for (int64_t i = 0; i < vid.shape()[2]; i++) {
            result_images[i] = tensor_to_sd_image(vid, static_cast<int>(i));
        }

        return result_images;
    }

    sd::Tensor<float> upscale_ltx_spatial_video_latent(StableDiffusionGGML* sd,
                                                       const char* model_path,
                                                       const sd::Tensor<float>& packed_latent,
                                                       int audio_length) {
        if (sd == nullptr || sd->model_manager == nullptr || packed_latent.empty()) {
            return {};
        }
        if (strlen(SAFE_STR(model_path)) == 0) {
            LOG_ERROR("LTX latent spatial upscale requires a model path");
            return {};
        }
        if (!sd->ensure_backend_pair(SDBackendModule::UPSCALER)) {
            return {};
        }

        int latent_channels            = sd->get_latent_channel();
        sd::Tensor<float> video_latent = packed_latent;
        sd::Tensor<float> audio_latent;
        if (packed_latent.shape()[3] > latent_channels) {
            video_latent = sd::ops::slice(packed_latent, 3, 0, latent_channels);
            audio_latent = unpack_ltxav_audio_latent(packed_latent, audio_length, latent_channels);
        }

        LOG_INFO("LTX latent spatial upscale: latent %dx%dx%dx%d -> model output",
                 (int)video_latent.shape()[0],
                 (int)video_latent.shape()[1],
                 (int)video_latent.shape()[2],
                 (int)video_latent.shape()[3]);

        sd::Tensor<float> unnormalized = sd->un_normalize_ltx_video_latents(video_latent);
        if (sd->first_stage_model) {
            sd->first_stage_model->runner_end();
        }
        if (unnormalized.empty()) {
            LOG_ERROR("LTX latent un-normalization failed before spatial upscale");
            return {};
        }

        auto model_manager = sd->model_manager;
        struct UpsamplerScope {
            ModelManager& manager;
            ModelLoader::FileId owned_source = 0;
            std::unique_ptr<LTXVUpsampler::LatentUpsamplerRunner> runner;
            std::vector<ggml_tensor*> params;

            ~UpsamplerScope() {
                if (runner) {
                    runner->runner_end();
                }
                GGML_ASSERT(manager.unregister_param_tensors(params));
                if (owned_source != 0) {
                    GGML_ASSERT(manager.del_file(owned_source));
                }
            }
        } scope{*model_manager};

        const std::string prefix        = "ltx_latent_upsampler";
        ModelLoader candidate           = model_manager->loader();
        ModelLoader::FileId source_file = 0;
        if (!candidate.add_file(model_path, prefix + ".", &source_file)) {
            LOG_ERROR("init LTX latent upsampler model loader from file failed: '%s'", model_path);
            return {};
        }
        const bool owns_source = model_manager->loader().file_revision(source_file) == 0;
        if (!model_manager->set_loader(std::move(candidate))) {
            return {};
        }
        scope.owned_source = owns_source ? source_file : 0;

        auto& upsampler                   = scope.runner;
        upsampler                         = std::make_unique<LTXVUpsampler::LatentUpsamplerRunner>(sd->backend_for(SDBackendModule::UPSCALER),
                                                                           model_manager->loader().get_tensor_storage_map(),
                                                                           prefix,
                                                                           model_manager);
        const size_t max_graph_vram_bytes = sd->max_graph_vram_bytes_for_module(SDBackendModule::UPSCALER);
        upsampler->set_max_graph_vram_bytes(max_graph_vram_bytes);
        if (upsampler->model == nullptr) {
            LOG_ERROR("init LTX latent upsampler from metadata failed");
            return {};
        }

        std::map<std::string, ggml_tensor*> tensors;
        upsampler->get_param_tensors(tensors);
        for (const auto& entry : tensors) {
            scope.params.push_back(entry.second);
        }
        if (!model_manager->register_param_tensors(ModelComponent::LatentUpsampler,
                                                   std::move(tensors),
                                                   ModelManager::ResidencyMode::ParamBackend,
                                                   sd->backend_for(SDBackendModule::UPSCALER),
                                                   sd->params_backend_for(SDBackendModule::UPSCALER)) ||
            !model_manager->validate_registered_tensors()) {
            LOG_ERROR("register LTX latent upsampler tensors with model manager failed");
            return {};
        }

        sd::Tensor<float> upscaled = upsampler->compute(sd->n_threads, unnormalized);
        upsampler->runner_end();
        if (upscaled.empty()) {
            LOG_ERROR("LTX latent spatial upscale failed");
            return {};
        }

        upscaled = sd->normalize_ltx_video_latents(upscaled);
        sd->first_stage_model->runner_end();
        if (upscaled.empty()) {
            LOG_ERROR("LTX latent normalization failed after spatial upscale");
            return {};
        }

        if (!audio_latent.empty()) {
            upscaled = pack_ltxav_audio_and_video_latents(upscaled, audio_latent);
        }
        return upscaled;
    }

    static bool apply_ltxv_refine_image_conditioning(StableDiffusionGGML* sd,
                                                     const sd_vid_gen_params_t* sd_vid_gen_params,
                                                     const GenerationRequest& request,
                                                     const ImageGenerationLatents& latents,
                                                     sd::Tensor<float>* latent,
                                                     sd::Tensor<float>* denoise_mask,
                                                     sd::Tensor<float>* video_positions) {
        if (sd == nullptr || sd_vid_gen_params == nullptr ||
            latent == nullptr || latent->empty() || denoise_mask == nullptr || video_positions == nullptr) {
            return true;
        }
        if (sd_vid_gen_params->init_image.data == nullptr &&
            sd_vid_gen_params->end_image.data == nullptr) {
            return true;
        }
        constexpr float conditioning_strength = 1.f;
        int latent_channels                   = sd->get_latent_channel();
        sd::Tensor<float> video_latent        = *latent;
        sd::Tensor<float> audio_latent;
        if (latent->shape()[3] > latent_channels) {
            video_latent = sd::ops::slice(*latent, 3, 0, latent_channels);
            audio_latent = unpack_ltxav_audio_latent(*latent, latents.audio_length, latent_channels);
            if (audio_latent.empty()) {
                LOG_ERROR("failed to unpack LTXAV audio latent before image-to-video inplace conditioning");
                return false;
            }
        }

        int image_width              = static_cast<int>(video_latent.shape()[0]) * request.vae_scale_factor;
        int image_height             = static_cast<int>(video_latent.shape()[1]) * request.vae_scale_factor;
        sd::Tensor<float> video_mask = make_ltxav_video_denoise_mask(video_latent, 1.f);

        if (sd_vid_gen_params->init_image.data != nullptr) {
            sd::Tensor<float> start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, image_width, image_height);
            if (!apply_ltxav_condition_image_by_latent_index(sd,
                                                             start_image,
                                                             &video_latent,
                                                             &video_mask,
                                                             0,
                                                             "init",
                                                             conditioning_strength)) {
                return false;
            }
        }

        if (sd_vid_gen_params->end_image.data != nullptr) {
            sd::Tensor<float> end_image        = sd_image_to_tensor(sd_vid_gen_params->end_image, image_width, image_height);
            sd::Tensor<float> end_image_latent = encode_ltxav_condition_image(sd, end_image, "end");
            if (end_image_latent.empty()) {
                return false;
            }

            int frame_idx = request.frames - 1;
            if (frame_idx == 0) {
                if (!apply_ltxav_condition_by_latent_index(&video_latent,
                                                           &video_mask,
                                                           end_image_latent,
                                                           0,
                                                           "end",
                                                           1.f - conditioning_strength)) {
                    return false;
                }
            } else {
                if (latents.video_conditioning_frame_count <= 0 || latents.video_target_frame_count <= 0) {
                    LOG_ERROR("LTXV FLF2V refine conditioning requires low-resolution keyframe conditioning metadata");
                    return false;
                }
                int64_t target_latent_frames = latents.video_target_frame_count;
                if (!apply_ltxav_condition_by_latent_index(&video_latent,
                                                           &video_mask,
                                                           end_image_latent,
                                                           target_latent_frames,
                                                           "end",
                                                           1.f - conditioning_strength)) {
                    return false;
                }
                *video_positions = build_ltxv_video_positions(video_latent.shape()[0],
                                                              video_latent.shape()[1],
                                                              target_latent_frames,
                                                              end_image_latent.shape()[2],
                                                              frame_idx,
                                                              1,
                                                              request.fps,
                                                              request.vae_scale_factor,
                                                              8,
                                                              true);
            }
        }

        if (!audio_latent.empty()) {
            //kcpp
            bool has_input_audio = sd_vid_gen_params->input_audio != nullptr &&
                sd_vid_gen_params->input_audio->data != nullptr &&
                sd_vid_gen_params->input_audio->sample_count > 0;
            *latent       = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
            *denoise_mask = pack_ltxav_audio_and_video_denoise_mask(video_mask, video_latent, audio_latent, has_input_audio ? 0.f : 1.f);
        } else {
            *latent       = std::move(video_latent);
            *denoise_mask = std::move(video_mask);
        }
        LOG_INFO("LTXV refine image conditioning applied at %dx%d", image_width, image_height);
        return true;
    }

    static bool generate_animatediff_video(StableDiffusionGGML* sd,
                                           const sd_vid_gen_params_t* sd_vid_gen_params,
                                           sd_image_t** frames_out,
                                           int* num_frames_out) {
        int n_frames = sd_vid_gen_params->video_frames;
        if (n_frames < 1) {
            LOG_ERROR("AnimateDiff: --video-frames must be >= 1");
            return false;
        }
        if (n_frames > 32) {
            LOG_WARN("AnimateDiff motion modules have a 32-frame positional-encoding context; capping to 32");
            n_frames = 32;
        }

        sd_img_gen_params_t img_gen_params;
        sd_img_gen_params_init(&img_gen_params);
        img_gen_params.loras             = sd_vid_gen_params->loras;
        img_gen_params.lora_count        = sd_vid_gen_params->lora_count;
        img_gen_params.prompt            = sd_vid_gen_params->prompt;
        img_gen_params.negative_prompt   = sd_vid_gen_params->negative_prompt;
        img_gen_params.clip_skip         = sd_vid_gen_params->clip_skip;
        img_gen_params.width             = sd_vid_gen_params->width;
        img_gen_params.height            = sd_vid_gen_params->height;
        img_gen_params.sample_params     = sd_vid_gen_params->sample_params;
        img_gen_params.strength          = sd_vid_gen_params->strength;
        img_gen_params.init_image        = sd_vid_gen_params->init_image;
        img_gen_params.seed              = sd_vid_gen_params->seed;
        img_gen_params.batch_count       = 1;
        img_gen_params.control_strength  = 1.0f;
        img_gen_params.vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
        img_gen_params.cache             = sd_vid_gen_params->cache;
        img_gen_params.hires             = sd_vid_gen_params->hires;
        img_gen_params.qwen_image_layers = 0;
        img_gen_params.circular_x        = sd_vid_gen_params->circular_x;
        img_gen_params.circular_y        = sd_vid_gen_params->circular_y;

        sd->animatediff_num_frames = n_frames;
        bool ok                    = generate_image(sd, &img_gen_params, frames_out, num_frames_out);
        sd->animatediff_num_frames = 0;
        return ok;
    }

    bool generate_video(StableDiffusionGGML* sd,
                        const sd_vid_gen_params_t* sd_vid_gen_params,
                        sd_image_t** frames_out,
                        int* num_frames_out,
                        sd_audio_t** audio_out) {
        if (sd->config_->animatediff_loaded && sd_version_supports_animatediff(sd->version)) {
            LOG_INFO("AnimateDiff dispatch: %d frames, %dx%d",
                     sd_vid_gen_params->video_frames, sd_vid_gen_params->width, sd_vid_gen_params->height);
            return generate_animatediff_video(sd, sd_vid_gen_params, frames_out, num_frames_out);
        }

        sd->reset_cancel_flag();

        const RefImageParams ref_image_params;

        int64_t t0            = ggml_time_ms();
        sd->vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
        sd->apply_circular_axes(sd_vid_gen_params->circular_x, sd_vid_gen_params->circular_y);
        GenerationRequest request(sd, sd_vid_gen_params);
        bool has_input_audio = sd_vid_gen_params->input_audio != nullptr && //kcpp
            sd_vid_gen_params->input_audio->data != nullptr &&
            sd_vid_gen_params->input_audio->sample_count > 0;
        bool latent_upscale_enabled     = request.hires.enabled;
        GenerationRequest hires_request = request;
        if (latent_upscale_enabled) {
            if (!sd_version_is_ltxav(sd->version)) {
                LOG_ERROR("LTX latent spatial upscale is only supported for LTX video models");
                return false;
            }
            if (request.hires.upscaler != SD_HIRES_UPSCALER_MODEL) {
                LOG_ERROR("LTX latent spatial upscale currently requires hires upscaler MODEL");
                return false;
            }
            if (strlen(SAFE_STR(request.hires.model_path)) == 0) {
                LOG_ERROR("LTX latent spatial upscale is enabled but hires model path was not provided");
                return false;
            }
        }

        sd->rng->manual_seed(request.seed);
        sd->sampler_rng->manual_seed(request.seed);
        sd->set_flow_shift(sd_vid_gen_params->sample_params.flow_shift);
        if (!sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count))
            return false;
        sd->reset_generation_extensions();

        SamplePlan plan(sd, sd_vid_gen_params, request);
        auto latent_inputs_opt = prepare_video_generation_latents(sd, sd_vid_gen_params, &request);
        if (!latent_inputs_opt.has_value()) {
            return false;
        }
        ImageGenerationLatents latents = std::move(*latent_inputs_opt);

        ImageGenerationEmbeds embeds = prepare_video_generation_embeds(sd,
                                                                       sd_vid_gen_params,
                                                                       request,
                                                                       latents);
        if (latent_upscale_enabled) {
            LOG_INFO("generate_video %dx%dx%d -> LTX latent spatial upscale",
                     request.width,
                     request.height,
                     request.frames);
        } else {
            LOG_INFO("generate_video %dx%dx%d",
                     request.width,
                     request.height,
                     request.frames);
        }

        int64_t latent_start = ggml_time_ms();
        int W                = request.width / request.vae_scale_factor;
        int H                = request.height / request.vae_scale_factor;
        int T                = static_cast<int>(latents.init_latent.shape()[2]);

        sd::Tensor<float> x_t   = latents.init_latent;
        sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd->rng);

        if (plan.high_noise_sample_steps > 0) {
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation before high-noise sampling");
                return false;
            }
            LOG_VERBOSE("sample(high noise) %dx%dx%d", W, H, T);

            int64_t sampling_start = ggml_time_ms();
            std::vector<float> high_noise_sigmas(plan.sigmas.begin(), plan.sigmas.begin() + plan.high_noise_sample_steps + 1);
            plan.sigmas = std::vector<float>(plan.sigmas.begin() + plan.high_noise_sample_steps, plan.sigmas.end());

            sd::Tensor<float> x_t_sampled = sd->sample(sd->high_noise_diffusion_model,
                                                       false,
                                                       x_t,
                                                       std::move(noise),
                                                       embeds.cond,
                                                       request.use_high_noise_uncond ? embeds.uncond : SDCondition(),
                                                       embeds.img_uncond,
                                                       sd::Tensor<float>(),
                                                       0.f,
                                                       request.high_noise_guidance,
                                                       plan.high_noise_eta,
                                                       request.shifted_timestep,
                                                       plan.high_noise_sample_method,
                                                       sd->is_flow_denoiser(),
                                                       plan.high_noise_extra_sample_args,
                                                       high_noise_sigmas,
                                                       std::vector<sd::Tensor<float>>{},
                                                       ref_image_params,
                                                       latents.denoise_mask,
                                                       latents.vace_context,
                                                       request.vace_strength,
                                                       latents.audio_length,
                                                       static_cast<float>(request.fps),
                                                       request.cache_params,
                                                       true,
                                                       latents.video_positions);
            int64_t sampling_end          = ggml_time_ms();
            if (x_t_sampled.empty()) {
                LOG_ERROR("sampling(high noise) failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
                return false;
            }

            x_t   = std::move(x_t_sampled);
            noise = {};
            LOG_INFO("sampling(high noise) completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        }

        if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation before sampling");
            return false;
        }
        LOG_VERBOSE("sample %dx%dx%d", W, H, T);
        int64_t sampling_start         = ggml_time_ms();
        sd::Tensor<float> final_latent = sd->sample(sd->diffusion_model,
                                                    true,
                                                    x_t,
                                                    std::move(noise),
                                                    embeds.cond,
                                                    request.use_uncond ? embeds.uncond : SDCondition(),
                                                    embeds.img_uncond,
                                                    sd::Tensor<float>(),
                                                    0.f,
                                                    sd_vid_gen_params->sample_params.guidance,
                                                    plan.eta,
                                                    sd_vid_gen_params->sample_params.shifted_timestep,
                                                    plan.sample_method,
                                                    sd->is_flow_denoiser(),
                                                    plan.extra_sample_args,
                                                    plan.sigmas,
                                                    std::vector<sd::Tensor<float>>{},
                                                    ref_image_params,
                                                    latents.denoise_mask,
                                                    latents.vace_context,
                                                    request.vace_strength,
                                                    latents.audio_length,
                                                    static_cast<float>(request.fps),
                                                    request.cache_params,
                                                    plan.high_noise_sample_steps <= 0,
                                                    latents.video_positions);

        int64_t sampling_end = ggml_time_ms();
        if (final_latent.empty()) {
            LOG_ERROR("sampling failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            return false;
        }
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);

        if (latent_upscale_enabled) {
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation before latent upscale");
                return false;
            }
            int64_t upscale_start             = ggml_time_ms();
            sd::Tensor<float> upscaled_latent = upscale_ltx_spatial_video_latent(sd,
                                                                                 request.hires.model_path,
                                                                                 final_latent,
                                                                                 latents.audio_length);
            int64_t upscale_end               = ggml_time_ms();
            if (upscaled_latent.empty()) {
                return false;
            }
            LOG_INFO("LTX latent spatial upscale completed, taking %.2fs",
                     (upscale_end - upscale_start) * 1.0f / 1000);

            x_t                        = std::move(upscaled_latent);
            hires_request.width        = static_cast<int>(x_t.shape()[0]) * hires_request.vae_scale_factor;
            hires_request.height       = static_cast<int>(x_t.shape()[1]) * hires_request.vae_scale_factor;
            int upscaled_latent_frames = static_cast<int>(x_t.shape()[2]);
            int upscaled_frames        = sd->latent_frames_to_video_frames(upscaled_latent_frames);
            if (upscaled_frames != hires_request.frames) {
                LOG_INFO("LTX latent upsampler output latent frames %d, frames %d -> %d",
                         upscaled_latent_frames,
                         hires_request.frames,
                         upscaled_frames);
                hires_request.frames = upscaled_frames;
            }
            if (sd_version_is_ltxav(sd->version) && latents.audio_length > 0) {
                int target_audio_length = get_ltxav_num_audio_latents(hires_request.frames, hires_request.fps);
                if (target_audio_length != latents.audio_length) {
                    int latent_channels            = sd->get_latent_channel();
                    sd::Tensor<float> video_latent = x_t;
                    sd::Tensor<float> audio_latent = latents.audio_latent;
                    if (x_t.shape()[3] > latent_channels) {
                        video_latent = sd::ops::slice(x_t, 3, 0, latent_channels);
                        audio_latent = unpack_ltxav_audio_latent(x_t, latents.audio_length, latent_channels);
                    }
                    audio_latent = resize_ltxav_audio_latent(audio_latent, target_audio_length);
                    if (audio_latent.empty()) {
                        LOG_ERROR("failed to resize LTX audio latent for latent upscale: %d -> %d",
                                  latents.audio_length,
                                  target_audio_length);
                        return false;
                    }
                    x_t                  = pack_ltxav_audio_and_video_latents(video_latent, audio_latent);
                    latents.audio_latent = std::move(audio_latent);
                    LOG_INFO("LTX audio latent length adjusted for latent upscale: %d -> %d",
                             latents.audio_length,
                             target_audio_length);
                    latents.audio_length = target_audio_length;
                }
            }
            if ((request.hires.target_width > 0 || request.hires.target_height > 0) &&
                (request.hires.target_width != hires_request.width || request.hires.target_height != hires_request.height)) {
                LOG_WARN("LTX latent spatial upsampler output is %dx%d; ignoring hires target %dx%d",
                         hires_request.width,
                         hires_request.height,
                         request.hires.target_width,
                         request.hires.target_height);
            }
            sd::Tensor<float> hires_denoise_mask;
            sd::Tensor<float> hires_video_positions;
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation before latent upscale refine");
                return false;
            }
            if (!apply_ltxv_refine_image_conditioning(sd,
                                                      sd_vid_gen_params,
                                                      hires_request,
                                                      latents,
                                                      &x_t,
                                                      &hires_denoise_mask,
                                                      &hires_video_positions)) {
                return false;
            }
            if (has_input_audio && hires_denoise_mask.empty() && x_t.shape()[3] > sd->get_latent_channel()) { //kcpp
                int latent_channels = sd->get_latent_channel();
                auto video_latent   = sd::ops::slice(x_t, 3, 0, latent_channels);
                auto audio_latent   = unpack_ltxav_audio_latent(x_t, latents.audio_length, latent_channels);
                if (!audio_latent.empty()) {
                    hires_denoise_mask = pack_ltxav_audio_and_video_denoise_mask(make_ltxav_video_denoise_mask(video_latent, 1.f),
                            video_latent,
                            audio_latent,
                            0.f);
                }
            }
            noise = sd::Tensor<float>::randn_like(x_t, sd->rng);

            W                                   = hires_request.width / hires_request.vae_scale_factor;
            H                                   = hires_request.height / hires_request.vae_scale_factor;
            T                                   = static_cast<int>(x_t.shape()[2]);
            sample_method_t hires_sample_method = plan.sample_method;
            int hires_scheduler_steps           = 0;
            std::vector<float> hires_sigma_sched =
                make_hires_sigma_schedule(sd,
                                          request.hires,
                                          sd_vid_gen_params->sample_params,
                                          hires_sample_method,
                                          plan.sample_steps,
                                          sd->get_image_seq_len(hires_request.height, hires_request.width) * T,
                                          &hires_scheduler_steps);
            float hires_eta = resolve_eta(sd,
                                          sd_vid_gen_params->sample_params.eta,
                                          hires_sample_method);

            LOG_VERBOSE("sample(latent upscale) %dx%dx%d", W, H, T);
            LOG_INFO("LTX latent spatial upscale refine: scheduler_steps=%d, denoising_strength=%.2f, sampler=%s, sigma_sched_size=%zu%s",
                     hires_scheduler_steps,
                     request.hires.denoising_strength,
                     sampling_methods_str[hires_sample_method],
                     hires_sigma_sched.size(),
                     request.hires.custom_sigmas_count > 0 ? ", custom_sigmas=true" : "");

            sampling_start = ggml_time_ms();
            final_latent   = sd->sample(sd->diffusion_model,
                                        true,
                                        x_t,
                                        std::move(noise),
                                        embeds.cond,
                                      hires_request.use_uncond ? embeds.uncond : SDCondition(),
                                        embeds.img_uncond,
                                        sd::Tensor<float>(),
                                        0.f,
                                        sd_vid_gen_params->sample_params.guidance,
                                        hires_eta,
                                        sd_vid_gen_params->sample_params.shifted_timestep,
                                        hires_sample_method,
                                        sd->is_flow_denoiser(),
                                        plan.extra_sample_args,
                                        hires_sigma_sched,
                                        std::vector<sd::Tensor<float>>{},
                                        ref_image_params,
                                        hires_denoise_mask,
                                        sd::Tensor<float>(),
                                        hires_request.vace_strength,
                                        latents.audio_length,
                                        static_cast<float>(hires_request.fps),
                                        hires_request.cache_params,
                                        false,
                                        hires_video_positions);
            sampling_end   = ggml_time_ms();
            if (final_latent.empty()) {
                LOG_ERROR("sampling(latent upscale) failed after %.2fs",
                          (sampling_end - sampling_start) * 1.0f / 1000);
                return false;
            }
            LOG_INFO("sampling(latent upscale) completed, taking %.2fs",
                     (sampling_end - sampling_start) * 1.0f / 1000);
        }

        int64_t latent_end = ggml_time_ms();
        LOG_INFO("generating latent video completed, taking %.2fs", (latent_end - latent_start) * 1.0f / 1000);

        struct RunnerEndGuard {
            GGMLRunner* runner = nullptr;
            ~RunnerEndGuard() {
                if (runner) {
                    runner->runner_end();
                }
            }
        } runner_guard{sd->audio_vae_model.get()};

        sd_audio_t* generated_audio = nullptr;
        if ((sd_version_is_ltxav(sd->version) || sd_version_is_minimax_h3(sd->version)) &&
            has_input_audio) { //kcpp
            generated_audio = clone_sd_audio(sd_vid_gen_params->input_audio);
        } else if ((sd_version_is_ltxav(sd->version) || sd_version_is_minimax_h3(sd->version)) &&
            latents.audio_length > 0 &&
            sd->audio_vae_model != nullptr) {
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cancelling generation before audio decode");
                return false;
            }
            int64_t audio_latent_decode_start = ggml_time_ms();

            auto audio_latent = sd_version_is_minimax_h3(sd->version)
                                    ? unpack_minimax_h3_audio_latent(final_latent,
                                                                     latents.audio_length,
                                                                     sd->get_latent_channel())
                                    : unpack_ltxav_audio_latent(final_latent,
                                                                latents.audio_length,
                                                                sd->get_latent_channel());
            if (!audio_latent.empty()) {
                LOG_VERBOSE("decode audio latent %dx%dx%dx%d",
                            (int)audio_latent.shape()[0],
                            (int)audio_latent.shape()[1],
                            (int)audio_latent.shape()[2],
                            (int)audio_latent.shape()[3]);
                auto waveform = sd->decode_ltx_audio_latent(audio_latent);
                if (!waveform.empty()) {
                    generated_audio = waveform_to_sd_audio(sd, waveform);
                } else {
                    LOG_WARN("audio latent decode failed; continuing with silent video output");
                }
            }
            int64_t audio_latent_decode_end = ggml_time_ms();
            LOG_INFO("decoding audio latent completed, taking %.2fs", (audio_latent_decode_end - audio_latent_decode_start) * 1.0f / 1000);
        }

        if (latents.video_conditioning_frame_count > 0) {
            int64_t target_frames = latents.video_target_frame_count > 0 ? latents.video_target_frame_count
                                                                         : final_latent.shape()[2] - latents.video_conditioning_frame_count;
            final_latent          = sd::ops::slice(final_latent, 2, 0, target_frames);
        }

        if (latents.ref_image_num > 0) {
            final_latent = sd::ops::slice(final_latent, 2, latents.ref_image_num, final_latent.shape()[2]);
        }

        if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cancelling generation before video decode");
            free_sd_audio(generated_audio);
            return false;
        }
        auto result = decode_video_outputs(sd, latent_upscale_enabled ? hires_request : request, final_latent, num_frames_out);
        if (result == nullptr) {
            free_sd_audio(generated_audio);
            return false;
        }

        sd->lora_stat();

        int64_t t1 = ggml_time_ms();
        LOG_INFO("generate_video completed in %.2fs", (t1 - t0) * 1.0f / 1000);
        if (frames_out != nullptr) {
            *frames_out = result;
        }
        if (audio_out != nullptr) {
            *audio_out = generated_audio;
        } else {
            free_sd_audio(generated_audio);
        }
        return true;
    }

}  // namespace sd::pipeline
