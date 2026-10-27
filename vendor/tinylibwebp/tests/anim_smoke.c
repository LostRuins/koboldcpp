#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "webp/decode.h"
#include "webp/encode.h"
#include "webp/mux.h"

static int AddFrame(WebPAnimEncoder* encoder, const uint8_t rgba[16],
                    int timestamp_ms, const WebPConfig* config) {
  WebPPicture picture;
  int ok;
  if (!WebPPictureInit(&picture)) return 0;
  picture.use_argb = 1;
  picture.width = 2;
  picture.height = 2;
  ok = WebPPictureImportRGBA(&picture, rgba, 2 * 4) &&
       WebPAnimEncoderAdd(encoder, &picture, timestamp_ms, config);
  WebPPictureFree(&picture);
  return ok;
}

int main(void) {
  static const uint8_t kRed[16] = {
      255, 0, 0, 255, 255, 0, 0, 255,
      255, 0, 0, 255, 255, 0, 0, 255,
  };
  static const uint8_t kBlue[16] = {
      0, 0, 255, 255, 0, 0, 255, 255,
      0, 0, 255, 255, 0, 0, 255, 255,
  };
  WebPAnimEncoderOptions options;
  WebPConfig config;
  WebPData output;
  WebPBitstreamFeatures features;
  WebPAnimEncoder* encoder = NULL;
  int ok = 0;

  WebPDataInit(&output);
  if (!WebPAnimEncoderOptionsInit(&options) || !WebPConfigInit(&config)) {
    fprintf(stderr, "failed to initialize libwebp\n");
    goto done;
  }
  options.anim_params.loop_count = 0;
  config.lossless = 1;
  config.method = 3;
  encoder = WebPAnimEncoderNew(2, 2, &options);
  if (encoder == NULL || !AddFrame(encoder, kRed, 0, &config) ||
      !AddFrame(encoder, kBlue, 100, &config) ||
      !WebPAnimEncoderAdd(encoder, NULL, 200, NULL) ||
      !WebPAnimEncoderAssemble(encoder, &output)) {
    fprintf(stderr, "animation encoding failed\n");
    goto done;
  }
  if (output.size < 12 || memcmp(output.bytes, "RIFF", 4) != 0 ||
      memcmp(output.bytes + 8, "WEBP", 4) != 0 ||
      WebPGetFeatures(output.bytes, output.size, &features) != VP8_STATUS_OK ||
      features.width != 2 || features.height != 2 || !features.has_animation) {
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
