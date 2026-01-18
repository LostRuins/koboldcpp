#pragma once

#include <math.h>
#include "ttscommon.h"

float energy(float * chunk, int count);

/*
 * This function is used to trim trailing silence at the end of audio data within the tts_response struct.
 * It detects silence by min-max normalizing energy and trimming frames which fall under a relative threshold.
 */
void apply_energy_voice_inactivity_detection(
	tts_response & data,
	float sample_rate = 44100.0f, // the sample rate of the audio
	int ms_per_frame = 10, // the audio time per frame
	int frame_threshold = 20, // the number of trailing empty frames upon which silence is clipped.
	float normalized_energy_threshold = 0.01f, // the normalized threshold to determine a silent frame
	int trailing_silent_frames = 5, // the number of frames of silence to allow
	int early_cutoff_seconds_threshold = 3, // the number of seconds of complete silence before terminating and cutting audio early
	float early_cutoff_energy_threshold = 0.1 // the energy threshold for treating a frame as silent for early cutoff
);
