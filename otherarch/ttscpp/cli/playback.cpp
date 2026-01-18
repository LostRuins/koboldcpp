#include <cstdint>
#include "playback.h"

#ifndef SDL2_INSTALL
void register_play_tts_response_args(arg_list & args) {
    // Hide --play
}

bool play_tts_response(arg_list & args, const tts_response & data, float sample_rate) {
    return false;
}
#else
#include "SDL.h"
void register_play_tts_response_args(arg_list & args) {
    args.add_argument(bool_arg("--play", "(OPTIONAL) Whether to play back the audio immediately instead of saving it to file."));
}

bool play_tts_response(arg_list & args, const tts_response & data, float sample_rate) {
    if (!args.get_bool_param("--play")) {
        return false;
    }

    if (SDL_Init(SDL_INIT_AUDIO)) {
        fprintf(stderr, "SDL_INIT failed\n");
        exit(1);
    }

    const SDL_AudioSpec desired{
        .freq = static_cast<int>(sample_rate),
        .format = AUDIO_F32,
        .channels = 1,
        .silence = 0,
        .padding = 0,
        .size = static_cast<unsigned>(data.n_outputs),
        .callback = nullptr,
        .userdata = nullptr,
    };
    const SDL_AudioDeviceID dev = SDL_OpenAudioDevice(nullptr, false, &desired, nullptr, 0);
    if (!dev) {
        fprintf(stderr, "SDL_OpenAudioDevice failed\n");
        exit(1);
    }

    SDL_PauseAudioDevice(dev, false);
    fprintf(stdout, "Playing %ld samples of audio\n", data.n_outputs);
    if (SDL_QueueAudio(dev, data.data, data.n_outputs * sizeof(data.data[0]))) {
        fprintf(stderr, "SDL_QueueAudio failed\n");
        exit(1);
    }

    SDL_Event event;
    while (SDL_GetQueuedAudioSize(dev)) {
        if (SDL_PollEvent(&event) && event.type == SDL_QUIT) break;
        SDL_Delay(100);
    }

    SDL_CloseAudioDevice(dev);
    SDL_Quit();

    return true;
}
#endif
