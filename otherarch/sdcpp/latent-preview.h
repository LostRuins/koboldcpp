#include <cstddef>
#include <cstdint>
#include "ggml.h"

const float wan_21_latent_rgb_proj[16][3] = {
    {0.015123f, -0.148418f, 0.479828f},
    {0.003652f, -0.010680f, -0.037142f},
    {0.212264f, 0.063033f, 0.016779f},
    {0.232999f, 0.406476f, 0.220125f},
    {-0.051864f, -0.082384f, -0.069396f},
    {0.085005f, -0.161492f, 0.010689f},
    {-0.245369f, -0.506846f, -0.117010f},
    {-0.151145f, 0.017721f, 0.007207f},
    {-0.293239f, -0.207936f, -0.421135f},
    {-0.187721f, 0.050783f, 0.177649f},
    {-0.013067f, 0.265964f, 0.166578f},
    {0.028327f, 0.109329f, 0.108642f},
    {-0.205343f, 0.043991f, 0.148914f},
    {0.014307f, -0.048647f, -0.007219f},
    {0.217150f, 0.053074f, 0.319923f},
    {0.155357f, 0.083156f, 0.064780f}};
float wan_21_latent_rgb_bias[3] = {-0.270270f, -0.234976f, -0.456853f};

const float wan_22_latent_rgb_proj[48][3] = {
    {0.017126f, -0.027230f, -0.019257f},
    {-0.113739f, -0.028715f, -0.022885f},
    {-0.000106f, 0.021494f, 0.004629f},
    {-0.013273f, -0.107137f, -0.033638f},
    {-0.000381f, 0.000279f, 0.025877f},
    {-0.014216f, -0.003975f, 0.040528f},
    {0.001638f, -0.000748f, 0.011022f},
    {0.029238f, -0.006697f, 0.035933f},
    {0.021641f, -0.015874f, 0.040531f},
    {-0.101984f, -0.070160f, -0.028855f},
    {0.033207f, -0.021068f, 0.002663f},
    {-0.104711f, 0.121673f, 0.102981f},
    {0.082647f, -0.004991f, 0.057237f},
    {-0.027375f, 0.031581f, 0.006868f},
    {-0.045434f, 0.029444f, 0.019287f},
    {-0.046572f, -0.012537f, 0.006675f},
    {0.074709f, 0.033690f, 0.025289f},
    {-0.008251f, -0.002745f, -0.006999f},
    {0.012685f, -0.061856f, -0.048658f},
    {0.042304f, -0.007039f, 0.000295f},
    {-0.007644f, -0.060843f, -0.033142f},
    {0.159909f, 0.045628f, 0.367541f},
    {0.095171f, 0.086438f, 0.010271f},
    {0.006812f, 0.019643f, 0.029637f},
    {0.003467f, -0.010705f, 0.014252f},
    {-0.099681f, -0.066272f, -0.006243f},
    {0.047357f, 0.037040f, 0.000185f},
    {-0.041797f, -0.089225f, -0.032257f},
    {0.008928f, 0.017028f, 0.018684f},
    {-0.042255f, 0.016045f, 0.006849f},
    {0.011268f, 0.036462f, 0.037387f},
    {0.011553f, -0.016375f, -0.048589f},
    {0.046266f, -0.027189f, 0.056979f},
    {0.009640f, -0.017576f, 0.030324f},
    {-0.045794f, -0.036083f, -0.010616f},
    {0.022418f, 0.039783f, -0.032939f},
    {-0.052714f, -0.015525f, 0.007438f},
    {0.193004f, 0.223541f, 0.264175f},
    {-0.059406f, -0.008188f, 0.022867f},
    {-0.156742f, -0.263791f, -0.007385f},
    {-0.015717f, 0.016570f, 0.033969f},
    {0.037969f, 0.109835f, 0.200449f},
    {-0.000782f, -0.009566f, -0.008058f},
    {0.010709f, 0.052960f, -0.044195f},
    {0.017271f, 0.045839f, 0.034569f},
    {0.009424f, 0.013088f, -0.001714f},
    {-0.024805f, -0.059378f, -0.033756f},
    {-0.078293f, 0.029070f, 0.026129f}};
float wan_22_latent_rgb_bias[3] = {0.013160f, -0.096492f, -0.071323f};

const float flux_latent_rgb_proj[16][3] = {
    {-0.041168f, 0.019917f, 0.097253f},
    {0.028096f, 0.026730f, 0.129576f},
    {0.065618f, -0.067950f, -0.014651f},
    {-0.012998f, -0.014762f, 0.081251f},
    {0.078567f, 0.059296f, -0.024687f},
    {-0.015987f, -0.003697f, 0.005012f},
    {0.033605f, 0.138999f, 0.068517f},
    {-0.024450f, -0.063567f, -0.030101f},
    {-0.040194f, -0.016710f, 0.127185f},
    {0.112681f, 0.088764f, -0.041940f},
    {-0.023498f, 0.093664f, 0.025543f},
    {0.082899f, 0.048320f, 0.007491f},
    {0.075712f, 0.074139f, 0.081965f},
    {-0.143501f, 0.018263f, -0.136138f},
    {-0.025767f, -0.082035f, -0.040023f},
    {-0.111849f, -0.055589f, -0.032361f}};
float flux_latent_rgb_bias[3] = {0.024600f, -0.006937f, -0.008089f};

const float flux2_latent_rgb_proj[32][3] = {
    {0.000736f, -0.008385f, -0.019710f},
    {-0.001352f, -0.016392f, 0.020693f},
    {-0.006376f, 0.002428f, 0.036736f},
    {0.039384f, 0.074167f, 0.119789f},
    {0.007464f, -0.005705f, -0.004734f},
    {-0.004086f, 0.005287f, -0.000409f},
    {-0.032835f, 0.050802f, -0.028120f},
    {-0.003158f, -0.000835f, 0.000406f},
    {-0.112840f, -0.084337f, -0.023083f},
    {0.001462f, -0.006656f, 0.000549f},
    {-0.009980f, -0.007480f, 0.009702f},
    {0.032540f, 0.000214f, -0.061388f},
    {0.011023f, 0.000694f, 0.007143f},
    {-0.001468f, -0.006723f, -0.001678f},
    {-0.005921f, -0.010320f, -0.003907f},
    {-0.028434f, 0.027584f, 0.018457f},
    {0.014349f, 0.011523f, 0.000441f},
    {0.009874f, 0.003081f, 0.001507f},
    {0.002218f, 0.005712f, 0.001563f},
    {0.053010f, -0.019844f, 0.008683f},
    {-0.002507f, 0.005384f, 0.000938f},
    {-0.002177f, -0.011366f, 0.003559f},
    {-0.000261f, 0.015121f, -0.003240f},
    {-0.003944f, -0.002083f, 0.005043f},
    {-0.009138f, 0.011336f, 0.003781f},
    {0.011429f, 0.003985f, -0.003855f},
    {0.010518f, -0.005586f, 0.010131f},
    {0.007883f, 0.002912f, -0.001473f},
    {-0.003318f, -0.003160f, 0.003684f},
    {-0.034560f, -0.008740f, 0.012996f},
    {0.000166f, 0.001079f, -0.012153f},
    {0.017772f, 0.000937f, -0.011953f}};
float flux2_latent_rgb_bias[3] = {-0.028738f, -0.098463f, -0.107619f};

// This one was taken straight from
// https://github.com/Stability-AI/sd3.5/blob/8565799a3b41eb0c7ba976d18375f0f753f56402/sd3_impls.py#L288-L303
// (MiT Licence)
const float sd3_latent_rgb_proj[16][3] = {
    {-0.0645f, 0.0177f, 0.1052f},
    {0.0028f, 0.0312f, 0.0650f},
    {0.1848f, 0.0762f, 0.0360f},
    {0.0944f, 0.0360f, 0.0889f},
    {0.0897f, 0.0506f, -0.0364f},
    {-0.0020f, 0.1203f, 0.0284f},
    {0.0855f, 0.0118f, 0.0283f},
    {-0.0539f, 0.0658f, 0.1047f},
    {-0.0057f, 0.0116f, 0.0700f},
    {-0.0412f, 0.0281f, -0.0039f},
    {0.1106f, 0.1171f, 0.1220f},
    {-0.0248f, 0.0682f, -0.0481f},
    {0.0815f, 0.0846f, 0.1207f},
    {-0.0120f, -0.0055f, -0.0867f},
    {-0.0749f, -0.0634f, -0.0456f},
    {-0.1418f, -0.1457f, -0.1259f},
};
float sd3_latent_rgb_bias[3] = {0, 0, 0};

const float sdxl_latent_rgb_proj[4][3] = {
    {0.258303f, 0.277640f, 0.329699f},
    {-0.299701f, 0.105446f, 0.014194f},
    {0.050522f, 0.186163f, -0.143257f},
    {-0.211938f, -0.149892f, -0.080036f}};
float sdxl_latent_rgb_bias[3] = {0.144381f, -0.033313f, 0.007061f};

const float sd_latent_rgb_proj[4][3] = {
    {0.337366f, 0.216344f, 0.257386f},
    {0.165636f, 0.386828f, 0.046994f},
    {-0.267803f, 0.237036f, 0.223517f},
    {-0.178022f, -0.200862f, -0.678514f}};
float sd_latent_rgb_bias[3] = {-0.017478f, -0.055834f, -0.105825f};

void preview_latent_video(uint8_t* buffer, struct ggml_tensor* latents, const float (*latent_rgb_proj)[3], const float latent_rgb_bias[3], int patch_size) {
    size_t buffer_head = 0;

    uint32_t latent_width  = static_cast<uint32_t>(latents->ne[0]);
    uint32_t latent_height = static_cast<uint32_t>(latents->ne[1]);
    uint32_t dim           = static_cast<uint32_t>(latents->ne[ggml_n_dims(latents) - 1]);
    uint32_t frames        = 1;
    if (ggml_n_dims(latents) == 4) {
        frames = static_cast<uint32_t>(latents->ne[2]);
    }

    uint32_t rgb_width  = latent_width * patch_size;
    uint32_t rgb_height = latent_height * patch_size;

    uint32_t unpatched_dim = dim / (patch_size * patch_size);

    for (uint32_t k = 0; k < frames; k++) {
        for (uint32_t rgb_x = 0; rgb_x < rgb_width; rgb_x++) {
            for (uint32_t rgb_y = 0; rgb_y < rgb_height; rgb_y++) {
                int latent_x = rgb_x / patch_size;
                int latent_y = rgb_y / patch_size;

                int channel_offset = 0;
                if (patch_size > 1) {
                    channel_offset = ((rgb_y % patch_size) * patch_size + (rgb_x % patch_size));
                }

                size_t latent_id = (latent_x * latents->nb[0] + latent_y * latents->nb[1] + k * latents->nb[2]);

                // should be incremented by 1 for each pixel
                size_t pixel_id = k * rgb_width * rgb_height + rgb_y * rgb_width + rgb_x;

                float r = 0, g = 0, b = 0;
                if (latent_rgb_proj != nullptr) {
                    for (uint32_t d = 0; d < unpatched_dim; d++) {
                        float value = *(float*)((char*)latents->data + latent_id + (d * patch_size * patch_size + channel_offset) * latents->nb[ggml_n_dims(latents) - 1]);
                        r += value * latent_rgb_proj[d][0];
                        g += value * latent_rgb_proj[d][1];
                        b += value * latent_rgb_proj[d][2];
                    }
                } else {
                    // interpret first 3 channels as RGB
                    r = *(float*)((char*)latents->data + latent_id + 0 * latents->nb[ggml_n_dims(latents) - 1]);
                    g = *(float*)((char*)latents->data + latent_id + 1 * latents->nb[ggml_n_dims(latents) - 1]);
                    b = *(float*)((char*)latents->data + latent_id + 2 * latents->nb[ggml_n_dims(latents) - 1]);
                }
                if (latent_rgb_bias != nullptr) {
                    // bias
                    r += latent_rgb_bias[0];
                    g += latent_rgb_bias[1];
                    b += latent_rgb_bias[2];
                }
                // change range
                r = r * .5f + .5f;
                g = g * .5f + .5f;
                b = b * .5f + .5f;

                // clamp rgb values to [0,1] range
                r = r >= 0 ? r <= 1 ? r : 1 : 0;
                g = g >= 0 ? g <= 1 ? g : 1 : 0;
                b = b >= 0 ? b <= 1 ? b : 1 : 0;

                buffer[pixel_id * 3 + 0] = (uint8_t)(r * 255);
                buffer[pixel_id * 3 + 1] = (uint8_t)(g * 255);
                buffer[pixel_id * 3 + 2] = (uint8_t)(b * 255);
            }
        }
    }
}
