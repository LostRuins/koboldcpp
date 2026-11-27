#include "options.h"
#include "core/option_types.h"
#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

enum class ParamType { Bool,
                       Int,
                       Float,
                       String,
                       Tensor,
                       Tensors,
                       Floats,
                       Ints,
                       Loras,
                       Embeddings,
                       Audios,
                       Videos };

struct LoraValues {
    std::vector<std::string> paths;
    std::vector<sd_lora_t> items;
};

struct EmbeddingValues {
    std::vector<std::string> strings;
    std::vector<sd_embedding_t> items;
};

struct TensorValue {
    sd::api::TensorRef tensor;
    std::vector<sd::Image> frames;
};

struct ImageValues {
    std::vector<sd::api::TensorRef> tensors;
    std::vector<sd::Image> items;
};

struct AudioValues {
    std::vector<sd::api::TensorRef> tensors;
    std::vector<sd::Audio> items;
};

struct VideoValues {
    std::vector<sd::api::TensorRef> tensors;
    std::vector<std::vector<sd::Image>> frames;
    std::vector<sd::ReferenceVideo> items;
};

using ParamValue = std::variant<bool, int64_t, double, std::string, TensorValue, ImageValues, std::vector<float>, std::vector<int>, LoraValues, EmbeddingValues, AudioValues, VideoValues>;

struct sd_opt {
    sd::api::Operation operation;
    std::map<std::string, ParamValue> values;
};

struct ParamSpec {
    const char* key;
    ParamType type;
    int operations;
    std::function<void(const ParamValue&, sd::OptConfig&)> apply;
    int64_t int_min  = std::numeric_limits<int64_t>::min();
    int64_t int_max  = std::numeric_limits<int64_t>::max();
    double float_min = -std::numeric_limits<float>::max();
    double float_max = std::numeric_limits<float>::max();
    std::function<bool(const ParamValue&)> valid;
    const char* invalid_reason = "invalid parameter value";
    bool video_tensor          = false;
};

static sd_cache_mode_t parse_cache(const char* text) {
    static const char* names[] = {"disabled", "easycache", "ucache", "dbcache", "taylorseer", "cache_dit", "spectrum"};
    for (int i = 0; i < 7; ++i)
        if (strcmp(text, names[i]) == 0)
            return static_cast<sd_cache_mode_t>(i);
    return static_cast<sd_cache_mode_t>(-1);
}

static sd_vae_format_t parse_vae_format(const char* text) {
    static const char* names[] = {"auto", "flux", "sd3", "flux2", "wan"};
    for (int i = 0; i < 5; ++i)
        if (strcmp(text, names[i]) == 0)
            return static_cast<sd_vae_format_t>(i - 1);
    return SD_VAE_FORMAT_COUNT;
}

static sd_hires_upscaler_t parse_hires_upscaler(const char* text) {
    static const char* names[] = {"none", "latent", "latent_nearest", "latent_nearest_exact",
                                  "latent_antialiased", "latent_bicubic", "latent_bicubic_antialiased",
                                  "lanczos", "nearest", "model"};
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; ++i)
        if (strcmp(text, names[i]) == 0)
            return static_cast<sd_hires_upscaler_t>(i);
    return SD_HIRES_UPSCALER_COUNT;
}

static bool valid_sigmas(const ParamValue& value) {
    const auto& sigmas = std::get<std::vector<float>>(value);
    if (sigmas.empty())
        return true;
    if (sigmas.size() < 2 || sigmas.back() != 0.f)
        return false;
    for (size_t i = 0; i < sigmas.size(); ++i)
        if (sigmas[i] < 0.f || (i && sigmas[i] >= sigmas[i - 1]))
            return false;
    return true;
}

static bool valid_layers(const ParamValue& value) {
    const auto& layers = std::get<std::vector<int>>(value);
    return std::all_of(layers.begin(), layers.end(), [](int index) { return index >= 0; });
}

template <typename Value, typename Apply>
static ParamSpec parameter(const char* key, ParamType type, int operations, Apply apply) {
    return {key, type, operations, [apply](const ParamValue& value, sd::OptConfig& config) {
                apply(std::get<Value>(value), config);
            }};
}

template <typename Apply>
static ParamSpec integer_parameter(const char* key, int operations, Apply apply, int64_t min = INT64_MIN, int64_t max = INT64_MAX) {
    auto spec    = parameter<int64_t>(key, ParamType::Int, operations, apply);
    spec.int_min = min;
    spec.int_max = max;
    return spec;
}

template <typename Apply>
static ParamSpec float_parameter(const char* key, int operations, Apply apply, double min = -std::numeric_limits<float>::max(), double max = std::numeric_limits<float>::max()) {
    auto spec      = parameter<double>(key, ParamType::Float, operations, apply);
    spec.float_min = min;
    spec.float_max = max;
    return spec;
}

template <typename T>
struct ArrayStorage;
template <>
struct ArrayStorage<float> {
    using Value                = std::vector<float>;
    static constexpr auto type = ParamType::Floats;
};
template <>
struct ArrayStorage<int> {
    using Value                = std::vector<int>;
    static constexpr auto type = ParamType::Ints;
};
template <>
struct ArrayStorage<sd::Image> {
    using Value                = ImageValues;
    static constexpr auto type = ParamType::Tensors;
};
template <>
struct ArrayStorage<sd::Audio> {
    using Value                = AudioValues;
    static constexpr auto type = ParamType::Audios;
};
template <>
struct ArrayStorage<sd::ReferenceVideo> {
    using Value                = VideoValues;
    static constexpr auto type = ParamType::Videos;
};
template <>
struct ArrayStorage<sd_lora_t> {
    using Value                = LoraValues;
    static constexpr auto type = ParamType::Loras;
};
template <>
struct ArrayStorage<sd_embedding_t> {
    using Value                = EmbeddingValues;
    static constexpr auto type = ParamType::Embeddings;
};

template <typename T>
static const auto& array_items(const std::vector<T>& value) {
    return value;
}
static const auto& array_items(const ImageValues& value) {
    return value.items;
}
static const auto& array_items(const AudioValues& value) {
    return value.items;
}
static const auto& array_items(const VideoValues& value) {
    return value.items;
}
static const auto& array_items(const LoraValues& value) {
    return value.items;
}
static const auto& array_items(const EmbeddingValues& value) {
    return value.items;
}

template <typename Select>
class ParamGroup {
    std::vector<ParamSpec>& specs;
    int operations;
    Select select;

    template <typename Value, typename Object, typename Field, typename Convert>
    ParamSpec binding(const char* key, ParamType type, Field Object::* field, Convert convert) const {
        return parameter<Value>(key, type, operations, [select = select, field, convert](const Value& value, sd::OptConfig& config) {
            const auto decoded = convert(value);
            std::apply([&](auto&... target) { ((target.*field = decoded), ...); }, select(config));
        });
    }

public:
    ParamGroup(std::vector<ParamSpec>& specs, int operations, Select select)
        : specs(specs), operations(operations), select(select) {}

    ParamGroup for_operations(int mask) const { return {specs, mask, select}; }

    template <typename Object>
    void boolean(const char* key, bool Object::* field) {
        specs.push_back(binding<bool>(key, ParamType::Bool, field, [](bool value) { return value; }));
    }

    template <typename Object, typename Field>
    void integer(const char* key, Field Object::* field, int64_t min = INT64_MIN, int64_t max = INT64_MAX) {
        auto spec    = binding<int64_t>(key, ParamType::Int, field, [](int64_t value) { return static_cast<Field>(value); });
        spec.int_min = min;
        spec.int_max = max;
        specs.push_back(std::move(spec));
    }

    template <typename Object>
    void number(const char* key, float Object::* field, double min = -std::numeric_limits<float>::max(), double max = std::numeric_limits<float>::max()) {
        auto spec      = binding<double>(key, ParamType::Float, field, [](double value) { return static_cast<float>(value); });
        spec.float_min = min;
        spec.float_max = max;
        specs.push_back(std::move(spec));
    }

    template <typename Object>
    void string(const char* key, const char* Object::* field) {
        specs.push_back(binding<std::string>(key, ParamType::String, field, [](const std::string& value) { return value.c_str(); }));
    }

    template <typename Object>
    void image(const char* key, sd::Image Object::* field) {
        specs.push_back(binding<TensorValue>(key, ParamType::Tensor, field, [](const TensorValue& value) { return sd::api::image_view(*value.tensor.get()); }));
    }

    template <typename Object, typename Enum>
    void choice(const char* key, Enum Object::* field, Enum (*parse)(const char*), Enum invalid, bool allow_auto = false) {
        auto spec  = binding<std::string>(key, ParamType::String, field, [=](const std::string& value) {
            return allow_auto && value == "auto" ? invalid : parse(value.c_str());
        });
        spec.valid = [=](const ParamValue& value) {
            const auto& text = std::get<std::string>(value);
            return (allow_auto && text == "auto") || parse(text.c_str()) != invalid;
        };
        specs.push_back(std::move(spec));
    }

    template <typename Object, typename Element, typename Count>
    void frames(const char* key, Element* Object::* data, Count Object::* count) {
        auto spec         = parameter<TensorValue>(key, ParamType::Tensor, operations, [select = select, data, count](const TensorValue& value, sd::OptConfig& config) {
            std::apply([&](auto&... target) {
                ((target.*data = const_cast<Element*>(value.frames.data()), target.*count = static_cast<Count>(value.frames.size())), ...);
            },
                       select(config));
        });
        spec.video_tensor = true;
        specs.push_back(std::move(spec));
    }

    template <typename Object, typename Element, typename Count>
    void array(const char* key, Element* Object::* data, Count Object::* count, bool (*valid)(const ParamValue&) = nullptr, const char* reason = "invalid parameter value") {
        using Storage       = ArrayStorage<std::remove_const_t<Element>>;
        using Value         = typename Storage::Value;
        auto spec           = parameter<Value>(key, Storage::type, operations, [select = select, data, count](const Value& value, sd::OptConfig& config) {
            const auto& items = array_items(value);
            std::apply([&](auto&... target) {
                ((target.*data = const_cast<Element*>(items.data()), target.*count = static_cast<Count>(items.size())), ...);
            },
                       select(config));
        });
        spec.valid          = valid;
        spec.invalid_reason = reason;
        specs.push_back(std::move(spec));
    }
};

static void add_context_specs(std::vector<ParamSpec>& specs) {
    using namespace sd::api;
    ParamGroup ctx(specs, Context, [](sd::OptConfig& config) { return std::tie(config.context); });

    ctx.number("attn_scale", &sd::ContextConfig::attn_scale, 0, std::numeric_limits<float>::max());
    ctx.string("audio_encoder_path", &sd::ContextConfig::audio_encoder_path);
    ctx.string("audio_vae_path", &sd::ContextConfig::audio_vae_path);
    ctx.boolean("auto_fit", &sd::ContextConfig::auto_fit);
    ctx.boolean("batched_cfg", &sd::ContextConfig::batched_cfg);
    ctx.string("clip_g_path", &sd::ContextConfig::clip_g_path);
    ctx.string("clip_l_path", &sd::ContextConfig::clip_l_path);
    ctx.string("clip_vision_path", &sd::ContextConfig::clip_vision_path);
    ctx.integer("conditioning_cache_size", &sd::ContextConfig::conditioning_cache_size, 0, INT32_MAX);
    ctx.string("control_net_path", &sd::ContextConfig::control_net_path);
    ctx.boolean("diffusion_conv_direct", &sd::ContextConfig::diffusion_conv_direct);
    ctx.boolean("diffusion_flash_attn", &sd::ContextConfig::diffusion_flash_attn);
    ctx.boolean("disable_prefetch", &sd::ContextConfig::disable_prefetch);
    ctx.boolean("disable_segmented_compute", &sd::ContextConfig::disable_segmented_compute);
    ctx.boolean("eager_load", &sd::ContextConfig::eager_load);
    ctx.array("embeddings", &sd::ContextConfig::embeddings, &sd::ContextConfig::embedding_count);
    ctx.string("embeddings_connectors_path", &sd::ContextConfig::embeddings_connectors_path);
    ctx.boolean("enable_mmap", &sd::ContextConfig::enable_mmap);
    ctx.boolean("flash_attn", &sd::ContextConfig::flash_attn);
    ctx.boolean("force_sdxl_vae_conv_scale", &sd::ContextConfig::force_sdxl_vae_conv_scale);
    ctx.string("high_noise_diffusion_model_path", &sd::ContextConfig::high_noise_diffusion_model_path);
    ctx.string("ip_adapter_path", &sd::ContextConfig::ip_adapter_path);
    ctx.number("linear_scale", &sd::ContextConfig::linear_scale, 0, std::numeric_limits<float>::max());
    ctx.string("llm_path", &sd::ContextConfig::llm_path);
    ctx.string("llm_vision_path", &sd::ContextConfig::llm_vision_path);
    ctx.choice("lora_apply_mode", &sd::ContextConfig::lora_apply_mode, str_to_lora_apply_mode, LORA_APPLY_MODE_COUNT);
    ctx.string("max_vram", &sd::ContextConfig::max_vram);
    ctx.string("model_args", &sd::ContextConfig::model_args);
    ctx.for_operations(Context | Upscaler | ADetailer).string("model_path", &sd::ContextConfig::model_path);
    ctx.string("motion_module_path", &sd::ContextConfig::motion_module_path);
    ctx.string("photo_maker_path", &sd::ContextConfig::photo_maker_path);
    ctx.choice("prediction", &sd::ContextConfig::prediction, str_to_prediction, PREDICTION_COUNT, true);
    ctx.string("pulid_weights_path", &sd::ContextConfig::pulid_weights_path);
    ctx.choice("rng_type", &sd::ContextConfig::rng_type, str_to_rng_type, RNG_TYPE_COUNT, true);
    ctx.string("rpc_servers", &sd::ContextConfig::rpc_servers);
    ctx.boolean("sage_attn", &sd::ContextConfig::sage_attn);
    ctx.choice("sampler_rng_type", &sd::ContextConfig::sampler_rng_type, str_to_rng_type, RNG_TYPE_COUNT, true);
    ctx.string("split_mode", &sd::ContextConfig::split_mode);
    ctx.string("t5xxl_path", &sd::ContextConfig::t5xxl_path);
    ctx.boolean("tae_preview_only", &sd::ContextConfig::tae_preview_only);
    ctx.string("taesd_path", &sd::ContextConfig::taesd_path);
    ctx.string("tensor_type_rules", &sd::ContextConfig::tensor_type_rules);
    ctx.string("tokenizer", &sd::ContextConfig::tokenizer);
    ctx.string("uncond_diffusion_model_path", &sd::ContextConfig::uncond_diffusion_model_path);
    ctx.boolean("vae_conv_direct", &sd::ContextConfig::vae_conv_direct);
    ctx.choice("vae_format", &sd::ContextConfig::vae_format, parse_vae_format, SD_VAE_FORMAT_COUNT);
    ctx.choice("wtype", &sd::ContextConfig::wtype, str_to_sd_type, SD_TYPE_COUNT, true);

    specs.push_back(parameter<std::string>("backend", ParamType::String, Context | Upscaler | VidUpscaler | ADetailer, [](const std::string& value, sd::OptConfig& config) { config.context.backend = value.c_str(); config.post.video_upscaler.backend = value.c_str(); }));
    specs.push_back(parameter<std::string>("conv_direct_mode", ParamType::String, Context | VidUpscaler, [](const std::string& value, sd::OptConfig& config) { config.context.conv_direct_mode = value.c_str(); config.post.video_upscaler.conv_direct_mode = value.c_str(); }));
    specs.push_back(parameter<std::string>("diffusion_model_path", ParamType::String, Context | VidUpscaler, [](const std::string& value, sd::OptConfig& config) { config.context.diffusion_model_path = value.c_str(); config.post.video_upscaler.diffusion_model_path = value.c_str(); }));
    specs.push_back(parameter<std::string>("flash_attn_mode", ParamType::String, Context | VidUpscaler, [](const std::string& value, sd::OptConfig& config) { config.context.flash_attn_mode = value.c_str(); config.post.video_upscaler.flash_attn_mode = value.c_str(); }));
    specs.push_back(integer_parameter(
        "n_threads", Context | Upscaler | VidUpscaler | ADetailer, [](const int64_t& value, sd::OptConfig& config) { config.context.n_threads = static_cast<int>(value); config.post.video_upscaler.n_threads = static_cast<int>(value); }, 1, INT32_MAX));
    specs.push_back(parameter<std::string>("params_backend", ParamType::String, Context | Upscaler | VidUpscaler | ADetailer, [](const std::string& value, sd::OptConfig& config) { config.context.params_backend = value.c_str(); config.post.video_upscaler.params_backend = value.c_str(); }));
    specs.push_back(parameter<std::string>("vae_path", ParamType::String, Context | VidUpscaler, [](const std::string& value, sd::OptConfig& config) { config.context.vae_path = value.c_str(); config.post.video_upscaler.vae_path = value.c_str(); }));
}

static void add_generation_specs(std::vector<ParamSpec>& specs) {
    using namespace sd::api;
    ParamGroup img(specs, Image, [](sd::OptConfig& config) { return std::tie(config.image); });
    ParamGroup vid(specs, Video, [](sd::OptConfig& config) { return std::tie(config.video); });
    ParamGroup preprocess(specs, Image | Video, [](sd::OptConfig& config) { return std::tie(config.image.image_preprocess, config.video.image_preprocess); });
    ParamGroup photomaker(specs, Image, [](sd::OptConfig& config) { return std::tie(config.image.pm_params); });
    ParamGroup pulid(specs, Image, [](sd::OptConfig& config) { return std::tie(config.image.pulid_params); });

    img.integer("batch_count", &sd::ImageConfig::batch_count, 1, INT32_MAX);
    img.image("control_image", &sd::ImageConfig::control_image);
    img.number("control_strength", &sd::ImageConfig::control_strength, 0, std::numeric_limits<float>::max());
    img.choice("control_type", &sd::ImageConfig::control_type, str_to_sd_control_type, SD_CONTROL_TYPE_COUNT);
    img.image("ip_adapter_image", &sd::ImageConfig::ip_adapter_image);
    img.number("ip_adapter_strength", &sd::ImageConfig::ip_adapter_strength, 0, std::numeric_limits<float>::max());
    img.image("mask_image", &sd::ImageConfig::mask_image);
    img.integer("qwen_image_layers", &sd::ImageConfig::qwen_image_layers, 0, INT32_MAX);

    vid.frames("control_frames", &sd::VideoConfig::control_frames, &sd::VideoConfig::control_frames_size);
    vid.image("end_image", &sd::VideoConfig::end_image);
    vid.number("moe_boundary", &sd::VideoConfig::moe_boundary, 0, 1);
    vid.array("ref_audios", &sd::VideoConfig::ref_audios, &sd::VideoConfig::ref_audios_count);
    vid.array("ref_videos", &sd::VideoConfig::ref_videos, &sd::VideoConfig::ref_videos_count);
    vid.number("vace_strength", &sd::VideoConfig::vace_strength, 0, std::numeric_limits<float>::max());
    vid.integer("video_frames", &sd::VideoConfig::video_frames, 1, INT32_MAX);

    preprocess.string("image_preprocess.rules", &sd::ImagePreprocessConfig::rules);

    photomaker.string("photomaker.id_embed_path", &sd::PhotoMakerConfig::id_embed_path);
    photomaker.array("photomaker.id_images", &sd::PhotoMakerConfig::id_images, &sd::PhotoMakerConfig::id_images_count);
    photomaker.number("photomaker.style_strength", &sd::PhotoMakerConfig::style_strength, 0, 100);

    pulid.string("pulid.id_embedding_path", &sd::PuLIDConfig::id_embedding_path);
    pulid.number("pulid.id_weight", &sd::PuLIDConfig::id_weight, 0, std::numeric_limits<float>::max());

    specs.push_back(parameter<bool>("circular_x", ParamType::Bool, Image | Video | ADetail, [](const bool& value, sd::OptConfig& config) { config.image.circular_x = value; config.video.circular_x = value; }));
    specs.push_back(parameter<bool>("circular_y", ParamType::Bool, Image | Video | ADetail, [](const bool& value, sd::OptConfig& config) { config.image.circular_y = value; config.video.circular_y = value; }));
    specs.push_back(integer_parameter(
        "clip_skip", Image | Video | ADetail, [](const int64_t& value, sd::OptConfig& config) { config.image.clip_skip = static_cast<int>(value); config.video.clip_skip = static_cast<int>(value); }, INT32_MIN, INT32_MAX));
    specs.push_back(integer_parameter(
        "fps", Video | VidUpscale, [](const int64_t& value, sd::OptConfig& config) { config.video.fps = static_cast<int>(value); config.post.video_upscale.fps = static_cast<int>(value); }, 1, INT32_MAX));
    specs.push_back(integer_parameter(
        "height", Image | Video | ADetail, [](const int64_t& value, sd::OptConfig& config) { config.image.height = static_cast<int>(value); config.video.height = static_cast<int>(value); }, 1, INT32_MAX));
    specs.push_back(parameter<TensorValue>("init_image", ParamType::Tensor, Image | Video, [](const TensorValue& value, sd::OptConfig& config) { config.image.init_image = sd::api::image_view(*value.tensor.get()); config.video.init_image = config.image.init_image; }));
    specs.push_back(parameter<LoraValues>("loras", ParamType::Loras, Image | Video | ADetail, [](const LoraValues& value, sd::OptConfig& config) { config.image.loras = value.items.data(); config.image.lora_count = static_cast<decltype(config.image.lora_count)>(value.items.size()); config.video.loras = value.items.data(); config.video.lora_count = static_cast<decltype(config.video.lora_count)>(value.items.size()); }));
    specs.push_back(parameter<std::string>("negative_prompt", ParamType::String, Image | Video | ADetail, [](const std::string& value, sd::OptConfig& config) { config.image.negative_prompt = value.c_str(); config.video.negative_prompt = value.c_str(); }));
    specs.push_back(parameter<std::string>("prompt", ParamType::String, Image | Video | ADetail, [](const std::string& value, sd::OptConfig& config) { config.image.prompt = value.c_str(); config.video.prompt = value.c_str(); }));
    specs.push_back(parameter<std::string>("ref_image_args", ParamType::String, Image | Video, [](const std::string& value, sd::OptConfig& config) { config.image.ref_image_args = value.c_str(); config.video.ref_image_args = value.c_str(); }));
    specs.push_back(parameter<ImageValues>("ref_images", ParamType::Tensors, Image | Video, [](const ImageValues& value, sd::OptConfig& config) { config.image.ref_images = const_cast<sd::Image*>(value.items.data()); config.image.ref_images_count = static_cast<int>(value.items.size()); config.video.ref_images = config.image.ref_images; config.video.ref_images_count = config.image.ref_images_count; }));
    specs.push_back(integer_parameter("seed", Image | Video | VidUpscale | ADetail, [](const int64_t& value, sd::OptConfig& config) { config.image.seed = static_cast<int64_t>(value); config.video.seed = static_cast<int64_t>(value); config.post.video_upscale.seed = value; }));
    specs.push_back(float_parameter(
        "strength", Image | Video | ADetail, [](const double& value, sd::OptConfig& config) { config.image.strength = static_cast<float>(value); config.video.strength = static_cast<float>(value); }, 0, 1));
    specs.push_back(integer_parameter(
        "width", Image | Video | ADetail, [](const int64_t& value, sd::OptConfig& config) { config.image.width = static_cast<int>(value); config.video.width = static_cast<int>(value); }, 1, INT32_MAX));
}

static void add_sampling_specs(std::vector<ParamSpec>& specs) {
    using namespace sd::api;
    ParamGroup high_noise(specs, Video, [](sd::OptConfig& config) { return std::tie(config.video.high_noise_sample_params); });
    ParamGroup high_noise_guidance(specs, Video, [](sd::OptConfig& config) { return std::tie(config.video.high_noise_sample_params.guidance); });
    ParamGroup high_noise_slg(specs, Video, [](sd::OptConfig& config) { return std::tie(config.video.high_noise_sample_params.guidance.slg); });
    ParamGroup sample(specs, Image | Video | ADetail, [](sd::OptConfig& config) { return std::tie(config.image.sample_params, config.video.sample_params); });
    ParamGroup guidance(specs, Image | Video | ADetail, [](sd::OptConfig& config) { return std::tie(config.image.sample_params.guidance, config.video.sample_params.guidance); });
    ParamGroup slg(specs, Image | Video | ADetail, [](sd::OptConfig& config) { return std::tie(config.image.sample_params.guidance.slg, config.video.sample_params.guidance.slg); });

    high_noise.string("high_noise_sample.args", &sd::SampleConfig::extra_sample_args);
    high_noise.number("high_noise_sample.eta", &sd::SampleConfig::eta, 0, std::numeric_limits<float>::max());
    high_noise.choice("high_noise_sample.method", &sd::SampleConfig::sample_method, str_to_sample_method, SAMPLE_METHOD_COUNT, true);
    high_noise.integer("high_noise_sample.steps", &sd::SampleConfig::sample_steps, -1, INT32_MAX);

    high_noise_guidance.number("high_noise_sample.guidance.distilled_guidance", &sd::GuidanceConfig::distilled_guidance);
    high_noise_guidance.number("high_noise_sample.guidance.img_cfg", &sd::GuidanceConfig::img_cfg);
    high_noise_guidance.number("high_noise_sample.guidance.txt_cfg", &sd::GuidanceConfig::txt_cfg);

    high_noise_slg.number("high_noise_sample.guidance.slg.layer_end", &sd::SLGConfig::layer_end, 0, 1);
    high_noise_slg.number("high_noise_sample.guidance.slg.layer_start", &sd::SLGConfig::layer_start, 0, 1);
    high_noise_slg.array("high_noise_sample.guidance.slg.layers", &sd::SLGConfig::layers, &sd::SLGConfig::layer_count, valid_layers, "layer indices must be nonnegative");
    high_noise_slg.number("high_noise_sample.guidance.slg.scale", &sd::SLGConfig::scale, 0, std::numeric_limits<float>::max());

    sample.string("sample.args", &sd::SampleConfig::extra_sample_args);
    sample.number("sample.eta", &sd::SampleConfig::eta, 0, std::numeric_limits<float>::max());
    sample.number("sample.flow_shift", &sd::SampleConfig::flow_shift, 0, std::numeric_limits<float>::max());
    sample.choice("sample.method", &sd::SampleConfig::sample_method, str_to_sample_method, SAMPLE_METHOD_COUNT, true);
    sample.choice("sample.scheduler", &sd::SampleConfig::scheduler, str_to_scheduler, SCHEDULER_COUNT, true);
    sample.integer("sample.shifted_timestep", &sd::SampleConfig::shifted_timestep, 0, INT32_MAX);
    sample.array("sample.sigmas", &sd::SampleConfig::custom_sigmas, &sd::SampleConfig::custom_sigmas_count, valid_sigmas, "sigmas require at least two strictly decreasing nonnegative entries ending in zero, or an empty array");
    sample.integer("sample.steps", &sd::SampleConfig::sample_steps, 1, INT32_MAX);

    guidance.number("sample.guidance.distilled_guidance", &sd::GuidanceConfig::distilled_guidance);
    guidance.number("sample.guidance.img_cfg", &sd::GuidanceConfig::img_cfg);
    guidance.number("sample.guidance.txt_cfg", &sd::GuidanceConfig::txt_cfg);

    slg.number("sample.guidance.slg.layer_end", &sd::SLGConfig::layer_end, 0, 1);
    slg.number("sample.guidance.slg.layer_start", &sd::SLGConfig::layer_start, 0, 1);
    slg.array("sample.guidance.slg.layers", &sd::SLGConfig::layers, &sd::SLGConfig::layer_count, valid_layers, "layer indices must be nonnegative");
    slg.number("sample.guidance.slg.scale", &sd::SLGConfig::scale, 0, std::numeric_limits<float>::max());
}

static void add_cache_specs(std::vector<ParamSpec>& specs) {
    using namespace sd::api;
    ParamGroup cache(specs, Image | Video | ADetail, [](sd::OptConfig& config) { return std::tie(config.image.cache, config.video.cache); });

    cache.integer("cache.bn_compute_blocks", &sd::CacheConfig::Bn_compute_blocks, 0, INT32_MAX);
    cache.number("cache.end_percent", &sd::CacheConfig::end_percent, 0, 1);
    cache.number("cache.error_decay_rate", &sd::CacheConfig::error_decay_rate, 0, 1);
    cache.integer("cache.fn_compute_blocks", &sd::CacheConfig::Fn_compute_blocks, 0, INT32_MAX);
    cache.integer("cache.max_cached_steps", &sd::CacheConfig::max_cached_steps, INT32_MIN, INT32_MAX);
    cache.integer("cache.max_continuous_cached_steps", &sd::CacheConfig::max_continuous_cached_steps, INT32_MIN, INT32_MAX);
    cache.integer("cache.max_warmup_steps", &sd::CacheConfig::max_warmup_steps, 0, INT32_MAX);
    cache.choice("cache.mode", &sd::CacheConfig::mode, parse_cache, static_cast<sd_cache_mode_t>(-1));
    cache.boolean("cache.reset_error_on_compute", &sd::CacheConfig::reset_error_on_compute);
    cache.number("cache.residual_diff_threshold", &sd::CacheConfig::residual_diff_threshold, 0, std::numeric_limits<float>::max());
    cache.number("cache.reuse_threshold", &sd::CacheConfig::reuse_threshold, 0, std::numeric_limits<float>::max());
    cache.string("cache.scm_mask", &sd::CacheConfig::scm_mask);
    cache.boolean("cache.scm_policy_dynamic", &sd::CacheConfig::scm_policy_dynamic);
    cache.number("cache.spectrum_flex_window", &sd::CacheConfig::spectrum_flex_window, 0, std::numeric_limits<float>::max());
    cache.number("cache.spectrum_lam", &sd::CacheConfig::spectrum_lam, 0, std::numeric_limits<float>::max());
    cache.integer("cache.spectrum_m", &sd::CacheConfig::spectrum_m, 1, INT32_MAX);
    cache.number("cache.spectrum_stop_percent", &sd::CacheConfig::spectrum_stop_percent, 0, 1);
    cache.number("cache.spectrum_w", &sd::CacheConfig::spectrum_w, 0, std::numeric_limits<float>::max());
    cache.integer("cache.spectrum_warmup_steps", &sd::CacheConfig::spectrum_warmup_steps, 0, INT32_MAX);
    cache.integer("cache.spectrum_window_size", &sd::CacheConfig::spectrum_window_size, 1, INT32_MAX);
    cache.number("cache.start_percent", &sd::CacheConfig::start_percent, 0, 1);
    cache.integer("cache.taylorseer_n_derivatives", &sd::CacheConfig::taylorseer_n_derivatives, 0, INT32_MAX);
    cache.integer("cache.taylorseer_skip_interval", &sd::CacheConfig::taylorseer_skip_interval, 1, INT32_MAX);
    cache.boolean("cache.use_relative_threshold", &sd::CacheConfig::use_relative_threshold);
}

static void add_hires_specs(std::vector<ParamSpec>& specs) {
    using namespace sd::api;
    ParamGroup hires(specs, Image | Video, [](sd::OptConfig& config) { return std::tie(config.image.hires, config.video.hires); });

    hires.number("hires.denoising_strength", &sd::HiresConfig::denoising_strength, 0, 1);
    hires.boolean("hires.enabled", &sd::HiresConfig::enabled);
    hires.string("hires.model_path", &sd::HiresConfig::model_path);
    hires.number("hires.scale", &sd::HiresConfig::scale, 0, std::numeric_limits<float>::max());
    hires.array("hires.sigmas", &sd::HiresConfig::custom_sigmas, &sd::HiresConfig::custom_sigmas_count, valid_sigmas, "sigmas require at least two strictly decreasing nonnegative entries ending in zero, or an empty array");
    hires.integer("hires.steps", &sd::HiresConfig::steps, 0, INT32_MAX);
    hires.integer("hires.target_height", &sd::HiresConfig::target_height, 0, INT32_MAX);
    hires.integer("hires.target_width", &sd::HiresConfig::target_width, 0, INT32_MAX);
    hires.integer("hires.upscale_tile_size", &sd::HiresConfig::upscale_tile_size, 1, INT32_MAX);
    hires.choice("hires.upscaler", &sd::HiresConfig::upscaler, parse_hires_upscaler, SD_HIRES_UPSCALER_COUNT);
}

static void add_tiling_specs(std::vector<ParamSpec>& specs) {
    using namespace sd::api;
    ParamGroup tiling(specs, Image | Video | ADetail, [](sd::OptConfig& config) { return std::tie(config.image.vae_tiling_params, config.video.vae_tiling_params); });

    tiling.boolean("vae_tiling.enabled", &sd::TilingConfig::enabled);
    tiling.string("vae_tiling.extra_tiling_args", &sd::TilingConfig::extra_tiling_args);
    tiling.number("vae_tiling.rel_size_h", &sd::TilingConfig::rel_size_h, 0, std::numeric_limits<float>::max());
    tiling.number("vae_tiling.rel_size_w", &sd::TilingConfig::rel_size_w, 0, std::numeric_limits<float>::max());
    tiling.number("vae_tiling.target_overlap", &sd::TilingConfig::target_overlap, 0, 0.99);
    tiling.boolean("vae_tiling.temporal_tiling", &sd::TilingConfig::temporal_tiling);
    tiling.integer("vae_tiling.tile_size_h", &sd::TilingConfig::tile_size_h, 0, INT32_MAX);
    tiling.integer("vae_tiling.tile_size_w", &sd::TilingConfig::tile_size_w, 0, INT32_MAX);
}

static void add_postprocess_specs(std::vector<ParamSpec>& specs) {
    using namespace sd::api;
    ParamGroup input(specs, Upscale | ADetail, [](sd::OptConfig& config) { return std::tie(config.post); });
    ParamGroup upscaler(specs, Upscaler, [](sd::OptConfig& config) { return std::tie(config.post); });
    ParamGroup video_upscaler(specs, VidUpscaler, [](sd::OptConfig& config) { return std::tie(config.post.video_upscaler); });
    ParamGroup video_upscale(specs, VidUpscale, [](sd::OptConfig& config) { return std::tie(config.post.video_upscale); });
    ParamGroup adetail(specs, ADetail, [](sd::OptConfig& config) { return std::tie(config.post.adetail); });

    input.image("image", &sd::PostprocessConfig::image);

    upscaler.boolean("conv_direct", &sd::PostprocessConfig::conv_direct);
    upscaler.integer("tile_size", &sd::PostprocessConfig::tile_size, 1, INT32_MAX);

    video_upscaler.integer("upscale_factor", &sd::VideoUpscalerConfig::upscale_factor, 1, INT32_MAX);

    video_upscale.frames("frames", &sd::VideoUpscaleConfig::frames, &sd::VideoUpscaleConfig::frame_count);
    video_upscale.integer("window_size", &sd::VideoUpscaleConfig::window_size, 1, INT32_MAX);
    video_upscale.integer("temporal_chunk_size", &sd::VideoUpscaleConfig::temporal_chunk_size, 1, INT32_MAX);

    adetail.string("adetail.prompt", &sd::ADetailConfig::prompt);
    adetail.string("adetail.negative_prompt", &sd::ADetailConfig::negative_prompt);
    adetail.string("adetail.args", &sd::ADetailConfig::extra_ad_args);
}

static std::vector<ParamSpec> make_specs() {
    std::vector<ParamSpec> specs;
    add_context_specs(specs);
    add_generation_specs(specs);
    add_sampling_specs(specs);
    add_cache_specs(specs);
    add_hires_specs(specs);
    add_tiling_specs(specs);
    add_postprocess_specs(specs);
    return specs;
}

static const ParamSpec* find_spec(const char* key) {
    static const auto specs = make_specs();
    if (!key)
        return nullptr;
    for (const auto& spec : specs)
        if (strcmp(key, spec.key) == 0)
            return &spec;
    return nullptr;
}

static sd_status_t check_key(sd_opt_t* params, const char* key, ParamType type, const ParamSpec** out) {
    if (!params || !key)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "params and key are required");
    *out = find_spec(key);
    if (!*out)
        return sd::api::error(SD_STATUS_UNKNOWN_KEY, key, "unknown parameter");
    if (!((*out)->operations & params->operation))
        return sd::api::error(SD_STATUS_UNSUPPORTED, key, "parameter does not apply to this option object");
    if ((*out)->type != type)
        return sd::api::error(SD_STATUS_TYPE_MISMATCH, key, "incorrect setter type");
    return SD_STATUS_OK;
}

static sd_status_t store_value(sd_opt_t* params, const ParamSpec& spec, ParamValue value) {
    if (spec.type == ParamType::Int) {
        const auto number = std::get<int64_t>(value);
        if (number < spec.int_min || number > spec.int_max)
            return sd::api::error(SD_STATUS_OUT_OF_RANGE, spec.key, "integer outside supported range");
    }
    if (spec.type == ParamType::Float) {
        const auto number = std::get<double>(value);
        if (!std::isfinite(number) || number < spec.float_min || number > spec.float_max)
            return sd::api::error(SD_STATUS_OUT_OF_RANGE, spec.key, "float must be finite and within supported range");
    }
    if (spec.valid && !spec.valid(value))
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, spec.key, spec.invalid_reason);
    params->values.insert_or_assign(spec.key, std::move(value));
    return SD_STATUS_OK;
}

static sd_opt_t* create_opt(sd::api::Operation operation) {
    sd_opt_t* params = nullptr;
    sd::api::call([&] { params = new sd_opt_t{operation, {}}; return SD_STATUS_OK; });
    return params;
}

sd_ctx_opt_t* sd_ctx_opt_create(void) {
    return create_opt(sd::api::Context);
}
sd_img_gen_opt_t* sd_img_gen_opt_create(void) {
    return create_opt(sd::api::Image);
}
sd_vid_gen_opt_t* sd_vid_gen_opt_create(void) {
    return create_opt(sd::api::Video);
}
sd_upscaler_opt_t* sd_upscaler_opt_create(void) {
    return create_opt(sd::api::Upscaler);
}

sd_upscale_opt_t* sd_upscale_opt_create(void) {
    return create_opt(sd::api::Upscale);
}

sd_vid_upscaler_opt_t* sd_vid_upscaler_opt_create(void) {
    return create_opt(sd::api::VidUpscaler);
}

sd_vid_upscale_opt_t* sd_vid_upscale_opt_create(void) {
    return create_opt(sd::api::VidUpscale);
}

sd_adetailer_opt_t* sd_adetailer_opt_create(void) {
    return create_opt(sd::api::ADetailer);
}

sd_adetail_opt_t* sd_adetail_opt_create(void) {
    return create_opt(sd::api::ADetail);
}

void sd_opt_free(sd_opt_t* params) {
    delete params;
}

sd_status_t sd_opt_set_bool(sd_opt_t* params, const char* key, bool value) {
    return sd::api::call([&] {
        const ParamSpec* spec = nullptr;
        auto status           = check_key(params, key, ParamType::Bool, &spec);
        if (status != SD_STATUS_OK)
            return status;
        return store_value(params, *spec, ParamValue(value));
    });
}

sd_status_t sd_opt_set_int(sd_opt_t* params, const char* key, int64_t value) {
    return sd::api::call([&] {
        const ParamSpec* spec = nullptr;
        auto status           = check_key(params, key, ParamType::Int, &spec);
        if (status != SD_STATUS_OK)
            return status;
        return store_value(params, *spec, ParamValue(value));
    });
}

sd_status_t sd_opt_set_float(sd_opt_t* params, const char* key, double value) {
    return sd::api::call([&] {
        const ParamSpec* spec = nullptr;
        auto status           = check_key(params, key, ParamType::Float, &spec);
        if (status != SD_STATUS_OK)
            return status;
        return store_value(params, *spec, ParamValue(value));
    });
}

sd_status_t sd_opt_set_string(sd_opt_t* params, const char* key, const char* value) {
    return sd::api::call([&] {
        const ParamSpec* spec = nullptr;
        auto status           = check_key(params, key, ParamType::String, &spec);
        if (status != SD_STATUS_OK)
            return status;
        if (!value)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "string is NULL; use unset to restore defaults");
        return store_value(params, *spec, std::string(value));
    });
}

sd_status_t sd_opt_set_tensor(sd_opt_t* params, const char* key, const sd_tensor_t* tensor) {
    return sd::api::call([&] {
        const ParamSpec* spec = nullptr;
        auto status           = check_key(params, key, ParamType::Tensor, &spec);
        if (status != SD_STATUS_OK)
            return status;
        if (!tensor)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "tensor is required");
        TensorValue value;
        value.tensor = sd::api::TensorRef(tensor);
        try {
            if (spec->video_tensor)
                value.frames = sd::api::video_frames(*tensor);
            else
                sd::api::image_view(*tensor);
        } catch (const std::invalid_argument& error) {
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, error.what());
        }
        return store_value(params, *spec, std::move(value));
    });
}

template <typename T, typename Function>
static sd_status_t set_array(sd_opt_t* params, const char* key, const T* values, size_t count, ParamType type, Function build) {
    return sd::api::call([&] {
        const ParamSpec* spec = nullptr;
        auto status           = check_key(params, key, type, &spec);
        if (status != SD_STATUS_OK)
            return status;
        if (count > INT32_MAX)
            return sd::api::error(SD_STATUS_OUT_OF_RANGE, key, "too many array elements");
        if (!values && count)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "array is NULL with nonzero count");
        return build(*spec);
    });
}

sd_status_t sd_opt_set_tensors(sd_opt_t* params, const char* key, const sd_tensor_t* const* values, size_t count) {
    return set_array(params, key, values, count, ParamType::Tensors, [&](const ParamSpec& spec) {
        ImageValues copy;
        copy.tensors.reserve(count);
        copy.items.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (!values[i])
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "tensor is required");
            copy.items.push_back(sd::api::image_view(*values[i]));
            copy.tensors.emplace_back(values[i]);
        }
        return store_value(params, spec, std::move(copy));
    });
}

sd_status_t sd_opt_set_floats(sd_opt_t* params, const char* key, const float* values, size_t count) {
    return set_array(params, key, values, count, ParamType::Floats, [&](const ParamSpec& spec) {
        std::vector<float> copy;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i]))
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "array values must be finite");
            copy.push_back(values[i]);
        }
        return store_value(params, spec, std::move(copy));
    });
}

sd_status_t sd_opt_set_ints(sd_opt_t* params, const char* key, const int32_t* values, size_t count) {
    return set_array(params, key, values, count, ParamType::Ints, [&](const ParamSpec& spec) {
        std::vector<int> copy;
        for (size_t i = 0; i < count; ++i) {
            copy.push_back(values[i]);
        }
        return store_value(params, spec, std::move(copy));
    });
}

sd_status_t sd_opt_set_loras(sd_opt_t* params, const char* key, const sd_lora_t* values, size_t count) {
    return set_array(params, key, values, count, ParamType::Loras, [&](const ParamSpec& spec) {
        LoraValues copy;
        copy.paths.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (!values[i].path || !*values[i].path || !std::isfinite(values[i].multiplier))
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "LoRA requires a path and finite multiplier");
            copy.paths.emplace_back(values[i].path);
            copy.items.push_back({values[i].is_high_noise, values[i].multiplier, copy.paths.back().c_str()});
        }
        return store_value(params, spec, std::move(copy));
    });
}

sd_status_t sd_opt_set_embeddings(sd_opt_t* params, const char* key, const sd_embedding_t* values, size_t count) {
    return set_array(params, key, values, count, ParamType::Embeddings, [&](const ParamSpec& spec) {
        EmbeddingValues copy;
        copy.strings.reserve(count * 2);
        for (size_t i = 0; i < count; ++i) {
            if (!values[i].name || !*values[i].name || !values[i].path || !*values[i].path)
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "embedding requires a name and path");
            copy.strings.emplace_back(values[i].name);
            const char* name = copy.strings.back().c_str();
            copy.strings.emplace_back(values[i].path);
            copy.items.push_back({name, copy.strings.back().c_str()});
        }
        return store_value(params, spec, std::move(copy));
    });
}

sd_status_t sd_opt_set_audios(sd_opt_t* params, const char* key, const sd_audio_t* values, size_t count) {
    return set_array(params, key, values, count, ParamType::Audios, [&](const ParamSpec& spec) {
        AudioValues copy;
        copy.tensors.reserve(count);
        copy.items.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            copy.items.push_back(sd::api::audio_view(values[i]));
            copy.tensors.emplace_back(values[i].data);
        }
        return store_value(params, spec, std::move(copy));
    });
}

sd_status_t sd_opt_set_videos(sd_opt_t* params, const char* key, const sd_video_t* values, size_t count) {
    return set_array(params, key, values, count, ParamType::Videos, [&](const ParamSpec& spec) {
        VideoValues copy;
        copy.frames.reserve(count);
        copy.items.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& video = values[i];
            if (!video.data || video.fps <= 0)
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "video data and positive fps are required");
            copy.frames.push_back(sd::api::video_frames(*video.data));
            copy.tensors.emplace_back(video.data);
            sd::Audio audio{};
            if (video.audio.data) {
                audio = sd::api::audio_view(video.audio);
                copy.tensors.emplace_back(video.audio.data);
            }
            copy.items.push_back({copy.frames.back().data(), static_cast<int>(copy.frames.back().size()), video.fps, audio});
        }
        return store_value(params, spec, std::move(copy));
    });
}

sd_status_t sd_opt_unset(sd_opt_t* params, const char* key) {
    return sd::api::call([&] {
        if (!params || !key)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, key, "params and key are required");
        const auto* spec = find_spec(key);
        if (!spec)
            return sd::api::error(SD_STATUS_UNKNOWN_KEY, key, "unknown parameter");
        if (!(spec->operations & params->operation))
            return sd::api::error(SD_STATUS_UNSUPPORTED, key, "parameter does not apply to this option object");
        params->values.erase(key);
        return SD_STATUS_OK;
    });
}

bool sd::api::is_set(const sd_opt_t* params, const char* key) {
    return params && params->values.count(key) != 0;
}

bool sd::api::has_prefix(const sd_opt_t* params, const char* prefix) {
    if (!params)
        return false;
    const auto it = params->values.lower_bound(prefix);
    return it != params->values.end() && it->first.compare(0, strlen(prefix), prefix) == 0;
}

sd_status_t sd::api::resolve(const sd_opt_t* params, Operation operation, OptConfig& config) {
    if (!params)
        return error(SD_STATUS_INVALID_ARGUMENT, nullptr, "params is required");
    if (params->operation != operation)
        return error(SD_STATUS_UNSUPPORTED, nullptr, "option object has the wrong purpose for this operation");
    config = OptConfig{};
    if (operation == ADetail)
        config.image.strength = 0.4f;
    for (const auto& entry : params->values) {
        const auto* spec = find_spec(entry.first.c_str());
        if (!(spec->operations & operation))
            return error(SD_STATUS_UNSUPPORTED, spec->key, "parameter does not apply to this operation");
        spec->apply(entry.second, config);
    }
    const auto& image   = config.image;
    const auto& video   = config.video;
    const bool sampling = operation == Image || operation == Video || operation == ADetail;
    const auto& cache   = operation == Video ? video.cache : image.cache;
    if (sampling && cache.mode != SD_CACHE_DISABLED) {
        if (cache.start_percent >= cache.end_percent)
            return error(SD_STATUS_INVALID_ARGUMENT, "cache", "start_percent must precede end_percent");
    }
    if (sampling) {
        const auto& sample = operation == Video ? video.sample_params : image.sample_params;
        if (sample.guidance.slg.layer_start > sample.guidance.slg.layer_end)
            return error(SD_STATUS_INVALID_ARGUMENT, "sample.guidance.slg", "layer_start must not exceed layer_end");
        const auto& hires = operation == Video ? video.hires : image.hires;
        if (hires.enabled && hires.upscaler != SD_HIRES_UPSCALER_NONE &&
            hires.scale <= 0.f && hires.target_width <= 0 && hires.target_height <= 0)
            return error(SD_STATUS_INVALID_ARGUMENT, "hires", "enabled hires requires a positive scale or target size");
    }
    if (operation == Video && video.high_noise_sample_params.guidance.slg.layer_start > video.high_noise_sample_params.guidance.slg.layer_end)
        return error(SD_STATUS_INVALID_ARGUMENT, "high_noise_sample.guidance.slg", "layer_start must not exceed layer_end");
    if ((operation == Upscaler || operation == ADetailer) && (!config.context.model_path || !*config.context.model_path))
        return error(SD_STATUS_INVALID_ARGUMENT, "model_path", "model path is required");
    if (operation == VidUpscaler) {
        const auto& load = config.post.video_upscaler;
        if (!load.diffusion_model_path || !*load.diffusion_model_path || !load.vae_path || !*load.vae_path || load.upscale_factor <= 0)
            return error(SD_STATUS_INVALID_ARGUMENT, nullptr, "diffusion_model_path, vae_path and upscale_factor are required");
    }
    if ((operation == Upscale || operation == ADetail) && !config.post.image.data)
        return error(SD_STATUS_INVALID_ARGUMENT, "image", "input image is required");
    if (operation == Upscale && config.post.image.channel != 3)
        return error(SD_STATUS_INVALID_ARGUMENT, "image", "upscaling requires an RGB image");
    if (operation == VidUpscale) {
        const auto& input = config.post.video_upscale;
        if (!input.frames || input.frame_count <= 0)
            return error(SD_STATUS_INVALID_ARGUMENT, "frames", "video upscaling requires nonempty frames");
        for (int i = 0; i < input.frame_count; ++i)
            if (input.frames[i].channel != 3 || input.frames[i].width != input.frames[0].width || input.frames[i].height != input.frames[0].height)
                return error(SD_STATUS_INVALID_ARGUMENT, "frames", "frames must be RGB and have equal dimensions");
    }
    return SD_STATUS_OK;
}
