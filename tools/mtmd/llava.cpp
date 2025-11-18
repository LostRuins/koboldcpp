#include "clip.h"
#include "clip-impl.h"
#include "llava.h"
#include "mtmd-audio.h"

#include "llama.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <memory>


// convenience cpp wrapper
struct clip_image_f32_batch_deleter {
    void operator()(clip_image_f32_batch * val) { clip_image_f32_batch_free(val); }
};
typedef std::unique_ptr<clip_image_f32_batch, clip_image_f32_batch_deleter> clip_image_f32_batch_ptr;


static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, struct clip_image_f32_batch * preprocessed_img, float * image_embd, int * n_img_pos) {

    const int64_t t_img_enc_start_us = ggml_time_us();

    const char * mm_patch_merge_type = clip_patch_merge_type(ctx_clip);

    const size_t n_imgs = clip_image_f32_batch_n_images(preprocessed_img);

    clip_image_f32 * img_res = clip_image_f32_get_img(preprocessed_img, 0);
    *n_img_pos = clip_n_output_tokens(ctx_clip, img_res);
    bool encoded = clip_image_encode(ctx_clip, n_threads, img_res, image_embd); // image_embd shape is 576 x 4096
    if (!encoded) {
        LOG_ERR("Unable to encode image\n");
        return false;
    }

    LOG_INF("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    LOG_INF("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}

bool llava_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {
    // Granite vision uses up to 10 patches + base patch
    int num_max_patches = 11;
    if (clip_is_minicpmv(ctx_clip)) {
        num_max_patches = 10;
    }
    if (clip_is_glm(ctx_clip)) {
        num_max_patches = 1;
    }
    float * image_embd;
    clip_image_f32_batch_ptr preprocessed_img(clip_image_f32_batch_init());
    if (!clip_image_preprocess(ctx_clip, img, preprocessed_img.get())) {
        LOG_ERR("%s: unable to preprocess image\n", __func__);
        return false;
    }

    if (clip_is_qwen2vl(ctx_clip)) {
        // qwen2vl don't split image into chunks, so `num_max_patches` is not needed.
        //sometimes they resize the image LARGER than before (padding up), so we must account for that
        int max_nx = img->nx;
        int max_ny = img->ny;
        for(int i=0;i<preprocessed_img->entries.size();++i)
        {
            int a = preprocessed_img->entries[i].get()->nx;
            int b = preprocessed_img->entries[i].get()->ny;
            max_nx = std::max(max_nx,a);
            max_ny = std::max(max_ny,b);
        }
        image_embd = (float *)malloc(clip_embd_nbytes_by_img(ctx_clip, max_nx, max_ny));
    } else {
        image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*num_max_patches); // TODO: base on gridsize/llava model
    }
    if (!image_embd) {
        LOG_ERR("Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, preprocessed_img.get(), image_embd, &n_img_pos)) {
        LOG_ERR("%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

struct llava_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    llava_embd_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};


//kcpp helper function
bool audio_embd_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const whisper_preprocessor::whisper_mel & mel_spec, float ** image_embd_out, int * n_img_pos_out)
{
    clip_image_f32_ptr mel_f32(clip_image_f32_init());
    mel_f32->nx  = mel_spec.n_len;
    mel_f32->ny  = mel_spec.n_mel;
    mel_f32->buf = std::move(mel_spec.data);
    size_t n_tokens = clip_n_output_tokens(ctx_clip, mel_f32.get());

    clip_image_f32_batch batch_f32;
    batch_f32.is_audio = true;
    batch_f32.entries.push_back(std::move(mel_f32));

    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    float * audio_embd = (float *)malloc(n_tokens * n_mmproj_embd * sizeof(float));
    bool ok = clip_image_batch_encode(
        ctx_clip,
        n_threads,
        &batch_f32,
        audio_embd);
    *image_embd_out = audio_embd;
    *n_img_pos_out = n_tokens;
    return ok;
}