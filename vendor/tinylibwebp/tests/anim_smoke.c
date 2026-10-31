#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "webp/decode.h"
#include "webp/encode.h"
#include "webp/mux.h"

enum { kWidth = 32, kHeight = 32 };

static int AddFrame(WebPAnimEncoder* encoder, const uint8_t* rgba,
                    int timestamp_ms, const WebPConfig* config) {
  WebPPicture picture;
  int ok;
  if (!WebPPictureInit(&picture)) return 0;
  picture.use_argb = 1;
  picture.width = kWidth;
  picture.height = kHeight;
  ok = WebPPictureImportRGBA(&picture, rgba, kWidth * 4) &&
       WebPAnimEncoderAdd(encoder, &picture, timestamp_ms, config);
  WebPPictureFree(&picture);
  return ok;
}

int main(void) {
  uint8_t red[kWidth * kHeight * 4];
  uint8_t blue[kWidth * kHeight * 4];
  WebPAnimEncoderOptions options;
  WebPConfig config;
  WebPData output;
  WebPBitstreamFeatures features;
  WebPAnimEncoder* encoder = NULL;
  int i;
  int ok = 0;

  for (i = 0; i < kWidth * kHeight; ++i) {
    red[4 * i + 0] = 255;
    red[4 * i + 1] = 0;
    red[4 * i + 2] = 0;
    red[4 * i + 3] = 255;
    blue[4 * i + 0] = 0;
    blue[4 * i + 1] = 0;
    blue[4 * i + 2] = 255;
    blue[4 * i + 3] = 255;
  }

  WebPDataInit(&output);
  if (!WebPAnimEncoderOptionsInit(&options) || !WebPConfigInit(&config)) {
    fprintf(stderr, "failed to initialize libwebp\n");
    goto done;
  }
  options.anim_params.loop_count = 0;
  config.lossless = 0;
  config.method = 3;
  config.thread_level = 1;
  encoder = WebPAnimEncoderNew(kWidth, kHeight, &options);
  if (encoder == NULL || !AddFrame(encoder, red, 0, &config) ||
      !AddFrame(encoder, blue, 100, &config) ||
      !WebPAnimEncoderAdd(encoder, NULL, 200, NULL) ||
      !WebPAnimEncoderAssemble(encoder, &output)) {
    fprintf(stderr, "animation encoding failed\n");
    goto done;
  }
  if (output.size < 12 || memcmp(output.bytes, "RIFF", 4) != 0 ||
      memcmp(output.bytes + 8, "WEBP", 4) != 0 ||
      WebPGetFeatures(output.bytes, output.size, &features) != VP8_STATUS_OK ||
      features.width != kWidth || features.height != kHeight ||
      !features.has_animation) {
    fprintf(stderr, "encoded animation failed validation\n");
    goto done;
  }
  printf("animated WebP smoke test passed (%zu bytes)\n", output.size);
  ok = 1;

done:
  WebPDataClear(&output);
  WebPAnimEncoderDelete(encoder);
  return ok ? 0 : 1;
}
