#include "ttscpp.h"
#include "ttsargs.h"
#include "ttscommon.h"
#include "playback.h"
#include "vad.h"
#include "write_file.h"
#include <thread>

class tts_timing_printer {
    const int64_t start_us{[] {
        ggml_time_init();
        return ggml_time_us();
    }()};
public:
    ~tts_timing_printer() {
        const int64_t end_us{ggml_time_us()};
        // Just a simple "total time" for now before adding "load" / "prompt eval" / "eval" from llama_print_timings
        printf("total time = %.2f ms\n", (end_us - start_us) / 1000.0f);
    }
};

int main(int argc, const char ** argv) {
    const tts_timing_printer _{};
    float default_temperature = 1.0f;
    int default_n_threads = std::max((int)std::thread::hardware_concurrency(), 1);
    int default_top_k = 50;
    int default_max_tokens = 0;
    float default_repetition_penalty = 1.0f;
    float default_top_p = 1.0f;
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1, Dia, or Kokoro.", "-mp", true));
    args.add_argument(string_arg("--prompt", "(REQUIRED) The text prompt for which to generate audio in quotation markers.", "-p", true));
    args.add_argument(string_arg("--save-path", "(OPTIONAL) The path to save the audio output to in a .wav format. Defaults to TTS.cpp.wav", "-sp", false, "TTS.cpp.wav"));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs. Defaults to 1.0.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to hardware concurrency. If hardware concurrency cannot be determined then it defaults to 1.", "-nt", false, &default_n_threads));
    args.add_argument(int_arg("--topk", "(OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.0.", "-r", false, &default_repetition_penalty));
    args.add_argument(bool_arg("--use-metal", "(OPTIONAL) Whether to use metal acceleration", "-m"));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.add_argument(string_arg("--conditional-prompt", "(OPTIONAL) A distinct conditional prompt to use for generating. If none is provided the preencoded prompt is used. '--text-encoder-path' must be set to use conditional generation.", "-cp", false));
    args.add_argument(string_arg("--text-encoder-path", "(OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.", "-tep", false));
    args.add_argument(string_arg("--voice", "(OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.", "-v", false, "af_heart"));
    args.add_argument(bool_arg("--vad", "(OPTIONAL) whether to apply voice inactivity detection (VAD) and strip silence form the end of the output (particularly useful for Parler TTS). By default, no VAD is applied.", "-va"));
    args.add_argument(string_arg("--espeak-voice-id", "(OPTIONAL) The espeak voice id to use for phonemization. This should only be specified when the correct espeak voice cannot be inferred from the kokoro voice ( see MultiLanguage Configuration in the README for more info).", "-eid", false));
    args.add_argument(int_arg("--max-tokens", "(OPTIONAL) The max audio tokens or token batches to generate where each represents approximates 11 ms of audio. Only applied to Dia generation. If set to zero as is its default then the default max generation size. Warning values under 15 are not supported.", "-mt", false, &default_max_tokens));
    args.add_argument(float_arg("--top-p", "(OPTIONAL) the sum of probabilities to sample over. Must be a value between 0.0 and 1.0. Defaults to 1.0.", "-tp", false, &default_top_p));
    register_play_tts_response_args(args);
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        exit(0);
    }
    args.validate();

    std::string conditional_prompt = args.get_string_param("--conditional-prompt");
    std::string text_encoder_path = args.get_string_param("--text-encoder-path");
    if (conditional_prompt.size() > 0 && text_encoder_path.size() <= 0) {
        fprintf(stderr, "The '--text-encoder-path' must be specified when '--condtional-prompt' is passed.\n");
        exit(1);
    }

    if (*args.get_float_param("--top-p") > 1.0f || *args.get_float_param("--top-p") <= 0.0f) {
        fprintf(stderr, "The '--top-p' value must be between 0.0 and 1.0. It was set to '%.6f'.\n", *args.get_float_param("--top-p"));
        exit(1);
    }

    generation_configuration * config = new generation_configuration(
        args.get_string_param("--voice"),
        *args.get_int_param("--topk"),
        *args.get_float_param("--temperature"),
        *args.get_float_param("--repetition-penalty"),
        !args.get_bool_param("--no-cross-attn"),
        args.get_string_param("--espeak-voice-id"),
        *args.get_int_param("--max-tokens"),
        *args.get_float_param("--top-p"));

    struct tts_runner * runner = runner_from_file(args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), config, !args.get_bool_param("--use-metal"));

    if (conditional_prompt.size() > 0) {
        update_conditional_prompt(runner, text_encoder_path, conditional_prompt, true);
    }
    tts_response data;

    generate(runner, args.get_string_param("--prompt"), &data, config);
    if (data.n_outputs == 0) {
        fprintf(stderr, "Got empty response for prompt, '%s'.\n", args.get_string_param("--prompt").c_str());
        exit(1);
    }
    if (args.get_bool_param("--vad")) {
        apply_energy_voice_inactivity_detection(data, runner->sampling_rate);
    }
    if (!play_tts_response(args, data, runner->sampling_rate)) {
        write_audio_file(data, args.get_string_param("--save-path"), runner->sampling_rate);
    }
    return 0;
}
