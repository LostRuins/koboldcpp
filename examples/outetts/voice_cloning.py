import outetts
print("Speaker JSON creation for Voice Cloning for OuteTTS...")

model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

speaker = interface.create_speaker(
    audio_path="input_audio.wav",

    # If transcript is not provided, it will be automatically transcribed using Whisper
    transcript=None,            # Set to None to use Whisper for transcription

    whisper_model="turbo",      # Optional: specify Whisper model (default: "turbo")
    whisper_device=None,        # Optional: specify device for Whisper (default: None)
)

interface.save_speaker(speaker, "speaker_output.json")
print("Speaker JSON saved!")