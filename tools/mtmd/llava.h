#ifndef LLAVA_H
#define LLAVA_H

#include "ggml.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAVA_API __declspec(dllexport)
#        else
#            define LLAVA_API __declspec(dllimport)
#        endif
#    else
#        define LLAVA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAVA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct clip_ctx;
struct llava_image_embed {
    float * embed;
    int n_image_pos;
};

struct mtmd_audio_mel;


LLAVA_API bool llava_image_embed_make_with_clip_img(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out, int * nx_out, int * ny_out);

LLAVA_API bool audio_embd_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const mtmd_audio_mel & mel_spec, float ** image_embd_out, int * n_img_pos_out);


#ifdef __cplusplus
}
#endif

#endif
