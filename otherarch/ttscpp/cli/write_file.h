#pragma once

#include "ttscommon.h"

void write_audio_file(const tts_response & data, std::string path = "TTS.cpp.wav", float sample_rate = 44100.0f);
