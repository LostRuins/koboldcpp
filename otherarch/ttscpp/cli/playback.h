#pragma once

#include "ttsargs.h"
#include "ttscommon.h"

void register_play_tts_response_args(arg_list & args);
bool play_tts_response(arg_list & args, const tts_response & data, float sample_rate);
