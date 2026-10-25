    {"fish_audio", R"AUDIOCPP_SPEC({
  "family": "fish_audio",
  "display_name": "Fish Audio S2 Pro",
  "description": "Fish Audio S2 Pro text-to-speech model for expressive speech across 80+ languages, with automatic language handling, inline prosody/emotion controls, and rapid voice cloning from short reference samples.",
  "category": "tts",
  "status": "supported",
  "tasks": [
    "tts",
    "clone"
  ],
  "modes": [
    "offline",
    "streaming"
  ],
  "languages": [
    "80+ languages"
  ],
  "capabilities": {
    "clone": [
      "speaker_reference"
    ]
  },
  "runtime": {
    "tags": [
      "gguf"
    ]
  },
  "options": {
    "request": [
      {
        "name": "reference_text",
        "type": "string",
        "description": "Reference transcript; required with inline reference audio.",
        "required": false
      },
      {
        "name": "multi_reference_cond",
        "type": "string",
        "description": "Fish Audio ordered JSON array of reference conditioning pairs for one generated speaker: [{\"audio\":\"ref1.wav\",\"text\":\"Transcript one.\"},{\"audio\":\"ref2.wav\",\"text\":\"Transcript two.\"}]. This is not a multi-speaker dialogue routing option.",
        "required": false
      },
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum Fish Audio semantic tokens to generate; default 1024. Zero selects the default token budget.",
        "required": false,
        "min": 0,
        "default": 1024
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Maximum characters per generated text chunk; default 200.",
        "required": false,
        "min": 1,
        "default": 200
      },
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "description": "Framework text chunking mode; default word_budget.",
        "preset": "text_chunk_mode_full",
        "required": false,
        "default": "word_budget"
      },
      {
        "name": "top_p",
        "type": "float",
        "description": "Top-p sampling value in (0, 1]; default 0.8.",
        "required": false,
        "min": 0.0,
        "max": 1.0,
        "default": 0.8
      },
      {
        "name": "top_k",
        "type": "int",
        "description": "Top-k sampling value; default 30.",
        "required": false,
        "min": 1,
        "default": 30
      },
      {
        "name": "temperature",
        "type": "float",
        "description": "Semantic-token sampling temperature in (0, 2); default 0.8.",
        "required": false,
        "min": 0.0,
        "max": 2.0,
        "default": 0.8
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Sampling seed for reproducible output; omitted uses a random seed.",
        "required": false,
        "min": 0
      }
    ],
    "session": [
      {
        "name": "lora",
        "type": "path",
        "description": "Prepared Fish Speech adapter checkpoint; merged into slow/fast decoder and embedding weights at session creation. Codec weights are unchanged.",
        "required": false
      },
      {
        "name": "lora_scale",
        "type": "float",
        "description": "Multiplier for the adapter's alpha/r scaling; default 1.",
        "required": false,
        "default": 1.0
      },
      {
        "name": "mem_saver",
        "type": "bool",
        "description": "Release cached AR runtime graphs after each request; default false.",
        "required": false,
        "default": false
      },
      {
        "name": "reference_cache_slots",
        "type": "int",
        "description": "Prepared reference-audio cache slots; default 1.",
        "required": false,
        "min": 0,
        "default": 1
      },
      {
        "name": "weight_type",
        "type": "enum",
        "description": "AR matmul weight storage type; default native.",
        "preset": "weight_type_full",
        "required": false,
        "default": "native"
      },
      {
        "name": "codec_weight_type",
        "type": "enum",
        "description": "Codec conv/matmul weight storage type; default native.",
        "preset": "weight_type_codec_q8",
        "required": false,
        "default": "native"
      },
      {
        "name": "codec_encoder_bf16_compute",
        "type": "bool",
        "description": "Use BF16 reference-encoder arithmetic on CUDA with F16/BF16 codec weights; default false. Decoding is unchanged.",
        "required": false,
        "default": false
      },
      {
        "name": "ar_graph_arena_mb",
        "type": "int",
        "description": "AR runtime graph arena size in MiB; default 512.",
        "required": false,
        "min": 1,
        "default": 512
      },
      {
        "name": "ar_weight_context_mb",
        "type": "int",
        "description": "AR weight context size in MiB; default 32.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "codec_graph_arena_mb",
        "type": "int",
        "description": "Codec encode/decode graph arena size in MiB; default 512.",
        "required": false,
        "min": 1,
        "default": 512
      },
      {
        "name": "codec_weight_context_mb",
        "type": "int",
        "description": "Codec weight context size in MiB; default 32.",
        "required": false,
        "min": 1,
        "default": 32
      }
    ],
    "load": []
  },
  "ui": {
    "recommended_package": "fish_audio_s2_pro_q8_0",
    "tags": [
      "TTS",
      "Clone",
      "GGUF"
    ],
    "docs": [
      "docs/tts.md",
      "docs/gguf.md"
    ]
  },
  "package_defaults": {
    "download": {
      "kind": "huggingface_snapshot",
      "repo": "audio-cpp/audio.cpp-gguf",
      "revision": "main",
      "gated": false
    }
  },
  "packages": [
    {
      "id": "fish_audio_s2_pro_q8_0",
      "display_name": "Fish Audio S2 Pro Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Fish-Audio-S2-Pro-GGUF",
      "files": [
        "Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf"
      ],
      "strip_prefix": "Fish-Audio-S2-Pro-GGUF"
    },
    {
      "id": "fish_audio_s2_pro_bf16",
      "display_name": "Fish Audio S2 Pro BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "Fish-Audio-S2-Pro-GGUF",
      "files": [
        "Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-bf16.gguf"
      ],
      "strip_prefix": "Fish-Audio-S2-Pro-GGUF"
    }
  ],
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "config": "model:config.json",
        "tokenizer_config": "model:tokenizer_config.json",
        "tokenizer_json": "model:tokenizer.json"
      },
      "tensors": {
        "model_weights": {
          "source": "weights:",
          "prefix": "model_weights"
        },
        "codec_weights": {
          "source": "weights:",
          "prefix": "codec_weights"
        }
      }
    },
    {
      "format": "safetensors",
      "roots": {
        "model": "."
      },
      "files": {
        "config": "model:config.json",
        "tokenizer_config": "model:tokenizer_config.json",
        "tokenizer_json": "model:tokenizer.json"
      },
      "tensors": {
        "model_weights": "model:model_audio_cpp.safetensors.index.json",
        "codec_weights": "model:codec.safetensors"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"maya1", R"AUDIOCPP_SPEC({
  "schema_version": 1,
  "family": "maya1",
  "display_name": "Maya1",
  "description": "Maya Research's 3B expressive English text-to-speech model with natural-language voice design and inline emotion control.",
  "category": "tts",
  "status": "supported",
  "tasks": [
    "tts",
    "design"
  ],
  "modes": [
    "offline"
  ],
  "languages": [
    "en"
  ],
  "runtime": {
    "tags": [
      "gguf"
    ]
  },
  "capabilities": {
    "tts": [
      "voice_design",
      "emotion_control",
      "long_form"
    ],
    "design": [
      "voice_design"
    ]
  },
  "options": {
    "request": [
      {
        "name": "instruct",
        "type": "string",
        "description": "Natural-language description of the voice, accent, timbre, age, and delivery.",
        "required": true
      },
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum generated audio-token count per text chunk; default 2048.",
        "required": false,
        "min": 1,
        "default": 2048
      },
      {
        "name": "min_tokens",
        "type": "int",
        "description": "Minimum generated audio-token count before EOS may stop generation; default 28.",
        "required": false,
        "min": 0,
        "default": 28
      },
      {
        "name": "temperature",
        "type": "float",
        "description": "Autoregressive sampling temperature; default 0.4. Must be > 0.",
        "required": false,
        "min": 0.0,
        "default": 0.4
      },
      {
        "name": "top_p",
        "type": "float",
        "description": "Autoregressive nucleus sampling probability; must be in (0, 1], default 0.9.",
        "required": false,
        "min": 0.0,
        "max": 1.0,
        "default": 0.9
      },
      {
        "name": "repetition_penalty",
        "type": "float",
        "description": "Autoregressive repetition penalty; default 1.1. Must be > 0.",
        "required": false,
        "min": 0.0,
        "default": 1.1
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Sampling and SNAC decoder noise seed; omitted requests choose a random seed.",
        "required": false,
        "min": 0
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Maximum Unicode codepoints per long-form text chunk; default 300.",
        "required": false,
        "min": 1,
        "default": 300
      },
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "description": "Framework long-form text chunking mode; tag_aware preserves inline emotion tags.",
        "values": [
          "default",
          "tag_aware",
          "japanese",
          "endline"
        ],
        "required": false,
        "default": "tag_aware"
      }
    ],
    "session": [
      {
        "name": "weight_type",
        "type": "enum",
        "description": "Shared AR and SNAC decoder weight storage type; default native.",
        "values": [
          "native",
          "f32",
          "f16",
          "bf16",
          "q4_0",
          "q4_1",
          "q5_0",
          "q5_1",
          "q2_k",
          "q3_k",
          "q4_k",
          "q5_k",
          "q6_k",
          "q8_0",
          "nvfp4"
        ],
        "required": false,
        "default": "native"
      },
      {
        "name": "ar_weight_type",
        "type": "enum",
        "description": "AR backbone weight storage type; defaults to weight_type.",
        "values": [
          "native",
          "f32",
          "f16",
          "bf16",
          "q4_0",
          "q4_1",
          "q5_0",
          "q5_1",
          "q2_k",
          "q3_k",
          "q4_k",
          "q5_k",
          "q6_k",
          "q8_0",
          "nvfp4"
        ],
        "required": false
      },
      {
        "name": "codec_weight_type",
        "type": "enum",
        "description": "SNAC decoder weight storage type; only native or f32 is supported. Inherits weight_type when omitted; set codec_weight_type=f32 explicitly when using another shared weight_type.",
        "values": [
          "native",
          "f32"
        ],
        "required": false
      }
    ],
    "load": []
  },
  "package_defaults": {
    "download": {
      "kind": "huggingface_snapshot",
      "repo": "audio-cpp/audio.cpp-gguf",
      "revision": "main",
      "gated": false
    }
  },
  "packages": [
    {
      "id": "maya1_orig",
      "display_name": "Maya1 Original-Precision GGUF",
      "default": true,
      "format": "gguf",
      "precision": "orig",
      "target_directory": "Maya1-GGUF",
      "files": [
        "Maya1-GGUF/maya1-orig.gguf"
      ],
      "strip_prefix": "Maya1-GGUF"
    },
    {
      "id": "maya1_q8_0",
      "display_name": "Maya1 Q8_0 GGUF",
      "default": false,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Maya1-GGUF",
      "files": [
        "Maya1-GGUF/maya1-q8_0.gguf"
      ],
      "strip_prefix": "Maya1-GGUF"
    }
  ],
  "dependencies": [],
  "ui": {
    "recommended_package": "maya1_orig",
    "tags": [
      "TTS",
      "Design",
      "GGUF"
    ],
    "docs": [
      "docs/models/maya1.md",
      "docs/gguf.md"
    ]
  },
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "config": "model:config.json",
        "generation_config": "model:generation_config.json",
        "tokenizer_json": "model:tokenizer.json",
        "tokenizer_config": "model:tokenizer_config.json"
      },
      "tensors": {
        "model_weights": "weights:",
        "codec_weights": "weights:"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"supertonic", R"AUDIOCPP_SPEC({
  "family": "supertonic",
  "display_name": "Supertonic 3",
  "description": "Supertone on-device TTS model designed for fast local speech synthesis across 31 languages, with preset voices and compact deployment for browser, mobile, and desktop applications.",
  "category": "tts",
  "status": "supported",
  "tasks": [
    "tts"
  ],
  "modes": [
    "offline",
    "streaming"
  ],
  "languages": [
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi"
  ],
  "capabilities": {
    "tts": [
      "built_in_voices",
      "long_form"
    ]
  },
  "runtime": {
    "tags": [
      "gguf",
      "stream"
    ]
  },
  "options": {
    "request": [
      {
        "name": "voice_id",
        "type": "enum",
        "values": [
          "M1",
          "M2",
          "M3",
          "M4",
          "M5",
          "F1",
          "F2",
          "F3",
          "F4",
          "F5"
        ],
        "default": "M1",
        "description": "Built-in voice preset; --voice-id takes precedence.",
        "required": false
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "min": 1,
        "default": 8,
        "description": "Flow denoising steps.",
        "required": false
      },
      {
        "name": "speed",
        "type": "float",
        "min": 0,
        "default": 1.05,
        "description": "Positive speech speed multiplier; voice.style.speaking_rate takes precedence.",
        "required": false
      },
      {
        "name": "seed",
        "type": "int",
        "min": 0,
        "max": 4294967295,
        "default": 1234,
        "description": "Noise seed.",
        "required": false
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Framework text chunk size; defaults to 120 for Korean/Japanese, otherwise 300.",
        "required": false
      },
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "values": [
          "default",
          "tag_aware",
          "japanese",
          "endline"
        ],
        "default": "default",
        "description": "Framework text chunking mode.",
        "required": false
      }
    ],
    "session": [
      {
        "name": "weight_type",
        "type": "enum",
        "preset": "weight_type_full",
        "default": "native",
        "description": "Weight storage type.",
        "required": false
      },
      {
        "name": "style_cache_slots",
        "type": "int",
        "min": 0,
        "default": 4,
        "description": "Cached preset style slots; zero disables reuse.",
        "required": false
      }
    ],
    "load": []
  },
  "ui": {
    "recommended_package": "supertonic_3_orig",
    "tags": [
      "TTS",
      "GGUF",
      "Stream"
    ],
    "docs": [
      "docs/tts.md",
      "docs/gguf.md"
    ]
  },
  "package_defaults": {
    "download": {
      "kind": "huggingface_snapshot",
      "repo": "audio-cpp/audio.cpp-gguf",
      "revision": "main",
      "gated": false
    }
  },
  "packages": [
    {
      "id": "supertonic_3_q8_0",
      "display_name": "Supertonic 3 Q8_0 GGUF",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Supertonic-3-GGUF",
      "files": [
        "Supertonic-3-GGUF/supertonic-3-q8_0.gguf"
      ],
      "strip_prefix": "Supertonic-3-GGUF"
    },
    {
      "id": "supertonic_3_f16",
      "display_name": "Supertonic 3 F16 GGUF",
      "format": "gguf",
      "precision": "f16",
      "target_directory": "Supertonic-3-GGUF",
      "files": [
        "Supertonic-3-GGUF/supertonic-3-f16.gguf"
      ],
      "strip_prefix": "Supertonic-3-GGUF"
    },
    {
      "id": "supertonic_3_orig",
      "display_name": "Supertonic 3 Original-Dtype GGUF",
      "default": true,
      "format": "gguf",
      "precision": "orig",
      "target_directory": "Supertonic-3-GGUF",
      "files": [
        "Supertonic-3-GGUF/supertonic-3-orig.gguf"
      ],
      "strip_prefix": "Supertonic-3-GGUF"
    },
    {
      "id": "supertonic_3_safetensors",
      "display_name": "Supertonic 3 Safetensors",
      "format": "safetensors",
      "precision": "native",
      "target_directory": "supertonic-3",
      "files": [
        "config/tts.json",
        "config/unicode_indexer.json",
        "ggml/supertonic.safetensors",
        "voice_styles/F1.json",
        "voice_styles/F2.json",
        "voice_styles/F3.json",
        "voice_styles/F4.json",
        "voice_styles/F5.json",
        "voice_styles/M1.json",
        "voice_styles/M2.json",
        "voice_styles/M3.json",
        "voice_styles/M4.json",
        "voice_styles/M5.json"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "mlx-community/supertonic-3-mlx"
      }
    }
  ],
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "tts_config": "model:config/tts.json",
        "unicode_indexer": "model:config/unicode_indexer.json",
        "voice_style_F1": "model:voice_styles/F1.json",
        "voice_style_F2": "model:voice_styles/F2.json",
        "voice_style_F3": "model:voice_styles/F3.json",
        "voice_style_F4": "model:voice_styles/F4.json",
        "voice_style_F5": "model:voice_styles/F5.json",
        "voice_style_M1": "model:voice_styles/M1.json",
        "voice_style_M2": "model:voice_styles/M2.json",
        "voice_style_M3": "model:voice_styles/M3.json",
        "voice_style_M4": "model:voice_styles/M4.json",
        "voice_style_M5": "model:voice_styles/M5.json"
      },
      "tensors": {
        "weights": {
          "source": "weights:",
          "prefix": "weights"
        }
      }
    },
    {
      "format": "safetensors",
      "roots": {
        "model": "."
      },
      "files": {
        "tts_config": "model:config/tts.json",
        "unicode_indexer": "model:config/unicode_indexer.json",
        "voice_style_F1": "model:voice_styles/F1.json",
        "voice_style_F2": "model:voice_styles/F2.json",
        "voice_style_F3": "model:voice_styles/F3.json",
        "voice_style_F4": "model:voice_styles/F4.json",
        "voice_style_F5": "model:voice_styles/F5.json",
        "voice_style_M1": "model:voice_styles/M1.json",
        "voice_style_M2": "model:voice_styles/M2.json",
        "voice_style_M3": "model:voice_styles/M3.json",
        "voice_style_M4": "model:voice_styles/M4.json",
        "voice_style_M5": "model:voice_styles/M5.json"
      },
      "tensors": {
        "weights": "model:ggml/supertonic.safetensors"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"yue2", R"AUDIOCPP_SPEC({
  "schema_version": 1,
  "family": "yue2",
  "display_name": "YuE2",
  "description": "YuE2 music generation model with symbolic ABC planning, semantic codec generation, NAR acoustic flow synthesis, and Oobleck VAE decode.",
  "category": "audio_generation",
  "status": "supported",
  "tasks": [
    "music"
  ],
  "modes": [
    "offline"
  ],
  "languages": [
    "en"
  ],
  "capabilities": {
    "music": [
      "lyrics",
      "style_control"
    ]
  },
  "runtime": {
    "tags": [
      "gguf"
    ]
  },
  "dependencies": [],
  "ui": {
    "recommended_package": "yue2_main_q8_0",
    "tags": [
      "Music",
      "GGUF"
    ],
    "docs": [
      "docs/models/yue2.md",
      "docs/music_generation.md",
      "docs/gguf.md"
    ]
  },
  "options": {
    "request": [
      {
        "name": "style",
        "type": "string",
        "description": "Song style/tags.",
        "required": true
      },
      {
        "name": "lyrics",
        "type": "string",
        "description": "Lyrics text. Leave empty for instrumental generation; if omitted, the CLI text input is used.",
        "required": false
      },
      {
        "name": "cot",
        "type": "enum",
        "values": [
          "off",
          "melody",
          "full"
        ],
        "description": "Symbolic planning route. off skips ABC generation; melody/full generate or consume ABC before music tokens.",
        "required": false,
        "default": "full"
      },
      {
        "name": "stop_after",
        "type": "enum",
        "values": [
          "abc",
          "semantic",
          "audio"
        ],
        "description": "Last stage to run. abc writes the generated score only; semantic also writes the semantic token export; audio renders the full song.",
        "required": false,
        "default": "audio"
      },
      {
        "name": "abc",
        "type": "string",
        "description": "External ABC score text for melody/full routes.",
        "required": false
      },
      {
        "name": "abc_file",
        "type": "string",
        "description": "Path to an external ABC score file for melody/full routes.",
        "required": false
      },
      {
        "name": "semantic_prefix",
        "type": "string",
        "description": "JSON array of semantic codec indices to force as the start of the music stream.",
        "required": false
      },
      {
        "name": "semantic_prefix_file",
        "type": "string",
        "description": "Path to a JSON array of semantic codec indices to force as the start of the music stream.",
        "required": false
      },
      {
        "name": "nar_noise_file",
        "type": "string",
        "description": "Path to a raw float32 noise file for NAR generation, shaped [frames,64].",
        "required": false
      },
      {
        "name": "export_semantic",
        "type": "bool",
        "description": "Attach the semantic token stream as a semantic artifact (JSON array of codec indices); default false.",
        "required": false,
        "default": false
      },
      {
        "name": "guidance_scale",
        "type": "float",
        "description": "Semantic classifier-free guidance scale. Default follows the upstream route defaults. Omitted defaults to 1.01 with cot=off, otherwise 1.0.",
        "required": false,
        "min": 0,
        "max": 20
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Generation seed. Must be nonnegative and less than 2^63.",
        "required": false,
        "min": 0,
        "default": 1234
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "description": "NAR midpoint ODE steps.",
        "required": false,
        "min": 1,
        "default": 8
      },
      {
        "name": "abc_temperature",
        "type": "float",
        "description": "ABC planner sampling temperature.",
        "required": false,
        "min": 0.0,
        "max": 5.0
      },
      {
        "name": "abc_top_p",
        "type": "float",
        "description": "ABC planner nucleus sampling probability. Must be in (0, 1].",
        "required": false,
        "max": 1
      },
      {
        "name": "abc_top_k",
        "type": "int",
        "description": "ABC planner top-k sampling limit.",
        "required": false,
        "min": 1
      },
      {
        "name": "abc_repetition_penalty",
        "type": "float",
        "description": "ABC planner repetition penalty. Must be positive.",
        "required": false
      },
      {
        "name": "abc_penalty_window",
        "type": "int",
        "description": "ABC planner repetition penalty window.",
        "required": false,
        "min": 1
      },
      {
        "name": "abc_min_tokens",
        "type": "int",
        "description": "Minimum ABC planner tokens before EOS is accepted.",
        "required": false,
        "min": 0
      },
      {
        "name": "abc_max_tokens",
        "type": "int",
        "description": "Maximum ABC planner tokens. Zero is allowed when min_tokens is also zero.",
        "required": false,
        "min": 0
      },
      {
        "name": "semantic_temperature",
        "type": "float",
        "description": "Semantic codec sampling temperature.",
        "required": false,
        "min": 0.0,
        "max": 5.0
      },
      {
        "name": "semantic_top_p",
        "type": "float",
        "description": "Semantic codec nucleus sampling probability. Must be in (0, 1].",
        "required": false,
        "max": 1
      },
      {
        "name": "semantic_top_k",
        "type": "int",
        "description": "Semantic codec top-k sampling limit.",
        "required": false,
        "min": 1
      },
      {
        "name": "semantic_repetition_penalty",
        "type": "float",
        "description": "Semantic codec repetition penalty. Must be positive.",
        "required": false
      },
      {
        "name": "semantic_penalty_window",
        "type": "int",
        "description": "Semantic codec repetition penalty window.",
        "required": false,
        "min": 1
      },
      {
        "name": "semantic_min_tokens",
        "type": "int",
        "description": "Minimum semantic tokens before EOS is accepted.",
        "required": false,
        "min": 0
      },
      {
        "name": "semantic_max_tokens",
        "type": "int",
        "description": "Maximum semantic codec tokens. Zero is allowed when min_tokens is also zero.",
        "required": false,
        "min": 0
      }
    ],
    "session": [
      {
        "name": "ar_lora",
        "type": "string",
        "description": "AR adapter safetensors file; native, PEFT, and ComfyUI LoRA layouts. Absolute path or relative to the model root. Requires session reload.",
        "required": false
      },
      {
        "name": "ar_lora_scale",
        "type": "float",
        "description": "AR LoRA delta scale; zero disables the adapter. Requires session reload.",
        "required": false,
        "default": 1.0
      },
      {
        "name": "nar_lora",
        "type": "string",
        "description": "NAR adapter safetensors file. Absolute path or relative to the model root. Requires session reload.",
        "required": false
      },
      {
        "name": "nar_lora_scale",
        "type": "float",
        "description": "NAR LoRA delta scale; full vae2llm/llm2vae replacements are unscaled. Zero disables the entire adapter. Requires session reload.",
        "required": false,
        "default": 1.0
      },
      {
        "name": "weight_type",
        "type": "enum",
        "values": [
          "native",
          "f32",
          "f16",
          "bf16",
          "q8_0",
          "q4_0",
          "q4_k"
        ],
        "description": "Shared weight storage type.",
        "required": false,
        "default": "native"
      },
      {
        "name": "model_weight_type",
        "type": "enum",
        "values": [
          "native",
          "f32",
          "f16",
          "bf16",
          "q8_0",
          "q4_0",
          "q4_k"
        ],
        "description": "YuE2 MoT weight storage type. Omitted inherits weight_type.",
        "required": false
      },
      {
        "name": "model_gguf",
        "type": "string",
        "description": "Yue2 main AR/NAR component GGUF file relative to the model root.",
        "required": false,
        "default": "yue2-3b-q8_0.gguf"
      },
      {
        "name": "vae_gguf",
        "type": "string",
        "description": "Yue2 VAE component GGUF file relative to the model root.",
        "required": false,
        "default": "yue2-vae-f16.gguf"
      },
      {
        "name": "vae_weight_type",
        "type": "enum",
        "values": [
          "native",
          "f32",
          "f16",
          "bf16",
          "q8_0",
          "q4_0",
          "q4_k"
        ],
        "description": "YuE2 VAE weight storage type. Omitted inherits weight_type.",
        "required": false
      },
      {
        "name": "model_weight_context_mb",
        "type": "int",
        "description": "YuE2 MoT weight context size in MiB.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "vae_weight_context_mb",
        "type": "int",
        "description": "YuE2 VAE weight context size in MiB.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "ar_prefill_graph_arena_mb",
        "type": "int",
        "description": "AR prefill graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "ar_decode_graph_arena_mb",
        "type": "int",
        "description": "AR one-token decode graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "nar_graph_arena_mb",
        "type": "int",
        "description": "NAR acoustic flow graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "vae_graph_arena_mb",
        "type": "int",
        "description": "VAE decode graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "attention",
        "type": "enum",
        "description": "NAR acoustic-flow attention lowering; auto selects the backend-appropriate implementation.",
        "required": false,
        "values": [
          "auto",
          "flash",
          "eager"
        ],
        "default": "auto"
      },
      {
        "name": "attention_tile_rows",
        "type": "int",
        "description": "Query rows per eager NAR attention tile; nonpositive values select automatic tiling to keep each score matrix below 3 GiB.",
        "required": false,
        "default": 0
      }
    ],
    "load": []
  },
  "package_defaults": {
    "download": {
      "kind": "huggingface_snapshot",
      "repo": "audio-cpp/Yue2-3B-GGUF",
      "revision": "main",
      "gated": false
    }
  },
  "packages": [
    {
      "id": "yue2_main_q8_0",
      "display_name": "Yue2 3B Main Q8_0",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Yue2-3B-GGUF",
      "files": [
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
        "yue2-3b-q8_0.gguf"
      ]
    },
    {
      "id": "yue2_main_bf16",
      "display_name": "Yue2 3B Main BF16",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "Yue2-3B-GGUF",
      "files": [
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
        "yue2-3b-bf16.gguf"
      ]
    },
    {
      "id": "yue2_main_q4_0",
      "display_name": "Yue2 3B Main Q4_0",
      "format": "gguf",
      "precision": "q4_0",
      "target_directory": "Yue2-3B-GGUF",
      "files": [
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
        "yue2-3b-q4_0.gguf"
      ]
    },
    {
      "id": "yue2_vae_f16",
      "display_name": "Yue2 VAE F16",
      "format": "gguf",
      "precision": "f16",
      "target_directory": "Yue2-3B-GGUF",
      "files": [
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
        "yue2-vae-f16.gguf"
      ]
    },
    {
      "id": "yue2_vae_f32",
      "display_name": "Yue2 VAE F32",
      "format": "gguf",
      "precision": "f32",
      "target_directory": "Yue2-3B-GGUF",
      "files": [
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
        "yue2-vae-f32.gguf"
      ]
    }
  ],
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "model_config": "model:sidecars/yue2-model-config.json",
        "generation_config": "model:sidecars/yue2-generation-config.json",
        "tiktoken": "model:sidecars/yue2-qwen.tiktoken",
        "vae_config": "model:sidecars/yue2-vae-config.json"
      },
      "tensors": {
        "model_weights": {
          "source": "weights:",
          "prefix": "model_weights"
        }
      }
    },
    {
      "format": "safetensors",
      "roots": {
        "model": "YuE2-3B",
        "vae": "YuE2-Vae"
      },
      "files": {
        "model_config": "model:config.json",
        "generation_config": "model:yue2_generation_config.json",
        "tiktoken": "model:qwen.tiktoken",
        "vae_config": "vae:config.json"
      },
      "tensors": {
        "model_weights": "model:model.safetensors",
        "vae_weights": "vae:model.safetensors"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
