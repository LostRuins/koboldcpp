    {"chatterbox", R"AUDIOCPP_SPEC({
  "family": "chatterbox",
  "display_name": "Chatterbox",
  "description": "Open-source Chatterbox family for expressive TTS and voice conversion, with emotion exaggeration control, fast generation, zero-shot voice cloning, and an integrated multilingual TTS path.",
  "category": "tts",
  "status": "supported",
  "tasks": [
    "tts",
    "clone",
    "vc"
  ],
  "modes": [
    "offline"
  ],
  "languages": [
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "hi",
    "it",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "sv",
    "sw",
    "tr"
  ],
  "options": {
    "request": [
      {
        "name": "exaggeration",
        "type": "float",
        "description": "Emotion exaggeration. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": 0.5
      },
      {
        "name": "guidance_scale",
        "type": "float",
        "description": "T3 classifier-free guidance scale. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": 0.5
      },
      {
        "name": "temperature",
        "type": "float",
        "description": "T3 sampling temperature. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": 0.8
      },
      {
        "name": "repetition_penalty",
        "type": "float",
        "description": "T3 repetition penalty. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": 1.2
      },
      {
        "name": "min_p",
        "type": "float",
        "description": "Minimum sampling probability. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": 0.05
      },
      {
        "name": "top_p",
        "type": "float",
        "description": "Nucleus sampling probability. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": 1
      },
      {
        "name": "flow_guidance_scale",
        "type": "float",
        "description": "S3Gen flow guidance scale. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": 0.7
      },
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum generated speech tokens. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "min": 1,
        "default": 384
      },
      {
        "name": "do_sample",
        "type": "bool",
        "description": "Enable sampling. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": true
      },
      {
        "name": "stop_on_eos",
        "type": "bool",
        "description": "Stop token generation on EOS. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "default": true
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "description": "Voice conversion flow steps; max_steps is also accepted.",
        "required": false,
        "min": 1,
        "default": 10
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Sampling seed; omitted chooses a random seed. Captured during clone preparation; changing it on the same prepared session is rejected.",
        "required": false,
        "min": 0,
        "max": 4294967295
      },
      {
        "name": "source_audio",
        "type": "audio_path",
        "description": "Voice conversion source WAV path; overrides audio_input.",
        "required": false
      },
      {
        "name": "target_voice",
        "type": "audio_path",
        "description": "Voice conversion target WAV path; overrides the speaker reference.",
        "required": false
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Framework long-form text chunk size.",
        "required": false,
        "default": 128
      }
    ],
    "session": [
      {
        "name": "weight_type",
        "type": "enum",
        "description": "Weight storage type; defaults to native.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "t3_weight_type",
        "type": "enum",
        "description": "T3 weight storage; inherits weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "multilingual_t3",
        "type": "enum",
        "description": "Multilingual T3 checkpoint selection.",
        "required": false,
        "values": [
          "v2",
          "v3"
        ],
        "default": "v2"
      },
      {
        "name": "conditionals_cache_slots",
        "type": "int",
        "description": "Cached speaker conditionals; zero disables reuse.",
        "required": false,
        "min": 0,
        "default": 1
      },
      {
        "name": "mem_saver",
        "type": "bool",
        "description": "Release staged runtime components.",
        "required": false,
        "default": false
      },
      {
        "name": "encoder_condition_samples",
        "type": "int",
        "description": "Maximum reference samples for the T3 tokenizer; the voice encoder still consumes the full 16 kHz reference.",
        "required": false,
        "default": 96000
      },
      {
        "name": "decoder_condition_samples",
        "type": "int",
        "description": "S3Gen reference sample limit at 24 kHz.",
        "required": false,
        "default": 240000
      },
      {
        "name": "t3_speech_cond_prompt_len",
        "type": "int",
        "description": "T3 reference speech prompt token limit.",
        "required": false,
        "default": 150
      }
    ],
    "load": []
  },
  "capabilities": {
    "clone": [
      "speaker_reference"
    ],
    "vc": [
      "speaker_reference"
    ]
  },
  "runtime": {
    "tags": [
      "gguf"
    ]
  },
  "ui": {
    "recommended_package": "chatterbox_q8_0",
    "tags": [
      "TTS",
      "Clone",
      "VC",
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
      "id": "chatterbox_q8_0",
      "display_name": "Chatterbox Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Chatterbox-GGUF",
      "files": [
        "Chatterbox-GGUF/chatterbox-q8_0.gguf"
      ],
      "strip_prefix": "Chatterbox-GGUF"
    },
    {
      "id": "chatterbox_f16",
      "display_name": "Chatterbox F16 GGUF",
      "format": "gguf",
      "precision": "f16",
      "target_directory": "Chatterbox-GGUF",
      "files": [
        "Chatterbox-GGUF/chatterbox-f16.gguf"
      ],
      "strip_prefix": "Chatterbox-GGUF"
    },
    {
      "id": "chatterbox_safetensors",
      "display_name": "Chatterbox Safetensors",
      "format": "safetensors",
      "precision": "native",
      "target_directory": "chatterbox",
      "files": [
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "grapheme_mtl_merged_expanded_v1.json",
        "Cangjie5_TC.json",
        "conds.pt",
        "t3_mtl23ls_v2.safetensors",
        "t3_mtl23ls_v3.safetensors"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "ResembleAI/chatterbox"
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
        "english_tokenizer": "model:tokenizer.json",
        "multilingual_tokenizer": "model:grapheme_mtl_merged_expanded_v1.json",
        "cangjie_mapping": "model:Cangjie5_TC.json",
        "builtin_conditionals": "model:conds.pt"
      },
      "tensors": {
        "voice_encoder_weights": {
          "source": "weights:",
          "prefix": "voice_encoder"
        },
        "s3gen_weights": {
          "source": "weights:",
          "prefix": "s3gen"
        },
        "t3_english_weights": {
          "source": "weights:",
          "prefix": "t3_english"
        },
        "t3_multilingual_v2_weights": {
          "source": "weights:",
          "prefix": "t3_multilingual_v2"
        },
        "t3_multilingual_v3_weights": {
          "source": "weights:",
          "prefix": "t3_multilingual_v3"
        }
      }
    },
    {
      "format": "safetensors",
      "roots": {
        "model": "."
      },
      "files": {
        "english_tokenizer": "model:tokenizer.json",
        "multilingual_tokenizer": "model:grapheme_mtl_merged_expanded_v1.json",
        "cangjie_mapping": "model:Cangjie5_TC.json",
        "builtin_conditionals": "model:conds.pt"
      },
      "tensors": {
        "voice_encoder_weights": "model:ve.safetensors",
        "s3gen_weights": "model:s3gen.safetensors",
        "t3_english_weights": "model:t3_cfg.safetensors",
        "t3_multilingual_v2_weights": "model:t3_mtl23ls_v2.safetensors",
        "t3_multilingual_v3_weights": "model:t3_mtl23ls_v3.safetensors"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"chatterbox_turbo", R"AUDIOCPP_SPEC({
  "family": "chatterbox_turbo",
  "display_name": "Chatterbox Turbo",
  "description": "Resemble AI's distilled 350M Chatterbox Turbo: GPT2 T3 backbone, GPT2 BPE tokenizer, and a 2-step meanflow S3Gen decoder for fast English TTS with a built-in voice.",
  "category": "tts",
  "status": "testing",
  "tasks": [
    "tts"
  ],
  "modes": [
    "offline"
  ],
  "languages": [
    "en"
  ],
  "capabilities": {},
  "options": {
    "request": [
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum generated tokens; legacy max_new_tokens remains accepted.",
        "required": false,
        "min": 0,
        "default": 1000
      },
      {
        "name": "temperature",
        "type": "float",
        "description": "T3 sampling temperature.",
        "required": false,
        "default": 0.8
      },
      {
        "name": "top_p",
        "type": "float",
        "description": "T3 nucleus sampling cutoff.",
        "required": false,
        "default": 0.95
      },
      {
        "name": "top_k",
        "type": "int",
        "description": "T3 top-k sampling limit.",
        "required": false,
        "min": 0,
        "default": 1000
      },
      {
        "name": "repetition_penalty",
        "type": "float",
        "description": "T3 token repetition penalty.",
        "required": false,
        "default": 1.2
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Generation seed.",
        "required": false,
        "min": 0,
        "max": 4294967295,
        "default": 0
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Maximum characters per text chunk; must be positive.",
        "required": false,
        "min": 1,
        "default": 128
      }
    ],
    "session": [],
    "load": []
  },
  "runtime": {
    "tags": [
      "gguf"
    ]
  },
  "ui": {
    "recommended_package": "chatterbox_turbo_q8_0",
    "tags": [
      "TTS",
      "GGUF"
    ],
    "docs": [
      "docs/models/chatterbox_turbo.md",
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
      "id": "chatterbox_turbo_q8_0",
      "display_name": "Chatterbox Turbo Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Chatterbox-Turbo-GGUF",
      "files": [
        "Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf"
      ],
      "strip_prefix": "Chatterbox-Turbo-GGUF"
    },
    {
      "id": "chatterbox_turbo_f16",
      "display_name": "Chatterbox Turbo F16 GGUF",
      "format": "gguf",
      "precision": "f16",
      "target_directory": "Chatterbox-Turbo-GGUF",
      "files": [
        "Chatterbox-Turbo-GGUF/chatterbox-turbo-f16.gguf"
      ],
      "strip_prefix": "Chatterbox-Turbo-GGUF"
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
        "tokenizer_vocab": "model:chatterbox_turbo_vocab.json",
        "tokenizer_merges": "model:chatterbox_turbo_merges.txt",
        "tokenizer_special_tokens": "model:chatterbox_turbo_special_tokens.json"
      },
      "tensors": {
        "t3_turbo_weights": {
          "source": "weights:",
          "prefix": "t3"
        },
        "builtin_conditionals_turbo": {
          "source": "weights:",
          "prefix": "conds"
        },
        "s3gen_weights": {
          "source": "weights:",
          "prefix": "s3gen"
        }
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"cosyvoice3", R"AUDIOCPP_SPEC({
  "schema_version": 1,
  "family": "cosyvoice3",
  "display_name": "CosyVoice3",
  "description": "Fun-CosyVoice3 native GGUF package for zero-shot, cross-lingual, and instruction-conditioned text-to-speech using CosyVoice3 speech-token AR generation, causal masked flow mel decoding, CAM++ speaker conditioning, and causal HiFT waveform decoding.",
  "category": "tts",
  "status": "supported",
  "tasks": [
    "tts",
    "clone"
  ],
  "modes": [
    "offline"
  ],
  "languages": [
    "zh",
    "en",
    "ja",
    "ko",
    "de",
    "es",
    "fr",
    "it",
    "ru",
    "yue"
  ],
  "runtime": {
    "tags": [
      "gguf"
    ]
  },
  "dependencies": [],
  "capabilities": {
    "tts": [
      "speaker_reference",
      "style_control"
    ],
    "clone": [
      "speaker_reference"
    ]
  },
  "options": {
    "request": [
      {
        "name": "template_name",
        "type": "enum",
        "description": "CosyVoice3 request template.",
        "required": false,
        "values": [
          "zero_shot",
          "cross_lingual",
          "instruct"
        ],
        "default": "zero_shot"
      },
      {
        "name": "reference_text",
        "type": "string",
        "description": "Transcript for the prompt audio.",
        "required": false
      },
      {
        "name": "instruct",
        "type": "string",
        "description": "Instruction text for instruct mode.",
        "required": false
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Maximum Unicode codepoints per long-form text chunk; default 600.",
        "required": false,
        "min": 1,
        "default": 600
      },
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "description": "Framework text chunking mode.",
        "values": [
          "default",
          "tag_aware",
          "japanese",
          "endline"
        ],
        "required": false,
        "default": "default"
      },
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum generated speech tokens; -1 (default) selects 20 times the target text token count. Explicit limits must be positive.",
        "required": false,
        "min": -1,
        "default": -1
      },
      {
        "name": "min_tokens",
        "type": "int",
        "description": "Minimum generated speech tokens before stop is accepted; -1 (default) selects twice the target text token count. Must not exceed the resolved maximum.",
        "required": false,
        "min": -1,
        "default": -1
      },
      {
        "name": "top_k",
        "type": "int",
        "description": "AR speech-token top-k sampling limit; default 25.",
        "required": false,
        "min": 1,
        "default": 25
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "description": "Flow decoder Euler steps; default 10.",
        "required": false,
        "min": 1,
        "default": 10
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Generation seed.",
        "required": false,
        "min": 0,
        "default": 1986
      }
    ],
    "session": [
      {
        "name": "lora",
        "type": "path",
        "description": "PEFT adapter with adapter_config.json, or native lora_weights with config.json and speech_embedding/llm_decoder sidecars. Checkpoints may use safetensors or PyTorch containers.",
        "required": false
      },
      {
        "name": "lora_scale",
        "type": "float",
        "description": "Multiplier for the adapter's alpha/r scaling; default 1. Full-weight replacements are not scaled.",
        "required": false,
        "default": 1.0
      },
      {
        "name": "weight_type",
        "type": "enum",
        "description": "CosyVoice3 matmul weight storage type; default native.",
        "required": false,
        "default": "native",
        "values": [
          "native",
          "f32",
          "f16",
          "bf16",
          "q8_0",
          "q4_0",
          "q4_k"
        ]
      },
      {
        "name": "graph_arena_mb",
        "type": "int",
        "description": "Reusable runtime graph arena size in MiB; default 1024.",
        "required": false,
        "min": 1,
        "default": 1024
      },
      {
        "name": "weight_context_mb",
        "type": "int",
        "description": "Weight loading context size in MiB; default 2048.",
        "required": false,
        "min": 1,
        "default": 2048
      },
      {
        "name": "reference_cache_slots",
        "type": "int",
        "description": "Prepared reference-audio cache slots; set 0 to disable reuse.",
        "required": false,
        "min": 0,
        "default": 4
      },
      {
        "name": "mem_saver",
        "type": "bool",
        "description": "Release cached runtime graphs after each request to reduce peak VRAM; default false.",
        "required": false,
        "default": false
      },
      {
        "name": "exec_plan",
        "type": "string",
        "description": "JSON object of component execution plans. Component: flow. Each plan accepts impl, backend, artifacts, and params. impl is default/ggml/rknn. GGML keeps the session backend. RKNN requires artifacts.main, a comma-separated ladder of estimator engines compiled one per sequence length (tools/rknn/prepare_cosyvoice3_rknn_flow.py), and ENGINE_ENABLE_RKNN; omit backend. An utterance longer than the largest engine runs the CPU graph. Neither implementation takes params.",
        "required": false,
        "default": "{}"
      }
    ],
    "load": []
  },
  "packages": [
    {
      "id": "cosyvoice3_q8_0",
      "display_name": "CosyVoice3 Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "CosyVoice3-GGUF",
      "files": [
        "CosyVoice3-GGUF/cosyvoice3-q8_0.gguf"
      ],
      "strip_prefix": "CosyVoice3-GGUF",
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "audio-cpp/audio.cpp-gguf",
        "revision": "main",
        "gated": false
      }
    },
    {
      "id": "cosyvoice3_f32",
      "display_name": "CosyVoice3 F32 GGUF",
      "default": false,
      "format": "gguf",
      "precision": "f32",
      "target_directory": "CosyVoice3-GGUF",
      "files": [
        "CosyVoice3-GGUF/cosyvoice3-f32.gguf"
      ],
      "strip_prefix": "CosyVoice3-GGUF",
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "audio-cpp/audio.cpp-gguf",
        "revision": "main",
        "gated": false
      }
    }
  ],
  "ui": {
    "recommended_package": "cosyvoice3_q8_0",
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
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "cosyvoice3_yaml": "model:cosyvoice3.yaml",
        "qwen_config": "model:CosyVoice-BlankEN/config.json",
        "tokenizer_config": "model:CosyVoice-BlankEN/tokenizer_config.json",
        "vocab_json": "model:CosyVoice-BlankEN/vocab.json",
        "merges_txt": "model:CosyVoice-BlankEN/merges.txt"
      },
      "tensors": {
        "llm_weights": {
          "source": "weights:",
          "prefix": "llm"
        },
        "flow_weights": {
          "source": "weights:",
          "prefix": "flow"
        },
        "hift_weights": {
          "source": "weights:",
          "prefix": "hift"
        },
        "campplus_weights": {
          "source": "weights:",
          "prefix": "campplus"
        },
        "speech_tokenizer_weights": {
          "source": "weights:",
          "prefix": "speech_tokenizer"
        },
        "blank_en_weights": {
          "source": "weights:",
          "prefix": "blank_en"
        }
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
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
    {"kitten_tts2", R"AUDIOCPP_SPEC({
  "schema_version": 1,
  "family": "kitten_tts2",
  "display_name": "Kitten TTS 2",
  "description": "Native multilingual Kitten TTS 2 Qwen3 speech generation, S3 meanflow decoder, and reference-audio voice cloning.",
  "category": "tts",
  "status": "experimental",
  "tasks": [
    "tts",
    "clone"
  ],
  "modes": [
    "offline"
  ],
  "languages": [
    "en",
    "ar",
    "zh",
    "fr",
    "de",
    "hi",
    "it",
    "pt",
    "ru",
    "es",
    "nl",
    "id",
    "ja",
    "ko",
    "pl",
    "tr"
  ],
  "runtime": {
    "tags": [
      "gguf",
      "cpu"
    ]
  },
  "capabilities": {
    "tts": [
      "built_in_voices",
      "long_form"
    ],
    "clone": [
      "speaker_reference"
    ]
  },
  "options": {
    "request": [
      {
        "name": "voice_id",
        "type": "enum",
        "values": [
          "Bella",
          "Jasper",
          "Luna",
          "Bruno",
          "Rosie",
          "Hugo",
          "Kiki",
          "Leo",
          "Matthew",
          "Elliot",
          "Willow",
          "Dolores",
          "Victor",
          "Dante",
          "Alfred",
          "Saoirse",
          "Claire",
          "Raven",
          "Marcus",
          "Herbert",
          "Diana",
          "Laurence",
          "Maeve",
          "Walter",
          "Edith",
          "Miles",
          "Grace",
          "Reginald",
          "Iris",
          "Frank",
          "Serena",
          "Julian",
          "Eleanor",
          "Otis",
          "Vincent",
          "Martha",
          "Sable",
          "Victoria",
          "PreparedBruno",
          "Arabic",
          "Chinese",
          "French",
          "German",
          "Hindi",
          "Italian",
          "Portuguese",
          "Russian",
          "Spanish",
          "AirportAnnouncer",
          "ArabicChildSchool",
          "ArabicFalconer",
          "ArabicMother",
          "ArabicNews",
          "AudiobookNarratorFemale",
          "BoredTeenager",
          "CalmHostageNegotiator",
          "ChineseKidSpringFestival",
          "ChineseLivestreamSeller",
          "ChineseNewsCctv",
          "ChineseXianxiaMaster",
          "DutchBaker",
          "DutchGrandpaStory",
          "DutchWeatherPresenter",
          "FairytaleNarratorWhimsical",
          "FrenchDocumentary",
          "FrenchGrandmereRecette",
          "FrenchTeacher",
          "FrenchWaiter",
          "GermanKidBirthday",
          "GermanKneipe",
          "GermanProfessor",
          "GermanTrainAnnouncement",
          "GrandmaBedtimeStory",
          "GriefEulogy",
          "HindiBollywoodDrama",
          "HindiCollegeFriends",
          "HindiDadiKahani",
          "HindiYogaTeacher",
          "IndonesianNewsAnchor",
          "IndonesianTourGuide",
          "ItalianChildSchool",
          "ItalianFashionDesigner",
          "ItalianFootballFan",
          "ItalianGondolier",
          "ItalianNonnaKitchen",
          "ItalianOperaTenor",
          "JapaneseAnimeHeroine",
          "JapaneseGrandma",
          "JapaneseSalarymanApology",
          "JapaneseTrainConductor",
          "KoreanGrandpaAdvice",
          "KoreanKdramaConfession",
          "KoreanNewsAnchor",
          "LittleGirlExplaining",
          "NervousFirstDate",
          "NewsAnchorFemale",
          "PilotCabinAnnouncement",
          "PolishBabciaPierogi",
          "PolishMountainGuide",
          "PolishNewsAnchor",
          "PolishPoetryStudent",
          "PortugueseBeach",
          "PortugueseBusiness",
          "PortugueseFado",
          "PortugueseGrandpaFarm",
          "PortugueseKidFutebol",
          "PortugueseNewsAnchor",
          "RussianBabushka",
          "RussianFairyTale",
          "RussianHockeyCheer",
          "RussianLiteratureTeacher",
          "RussianSpaceControl",
          "RussianTaxiDriver",
          "SarcasticCoworker",
          "SchoolboyPresentation",
          "SpanishAbuela",
          "SpanishBreakupSad",
          "SpanishChildCuento",
          "SpanishNews",
          "SpanishTango",
          "TurkishFisherman",
          "TurkishGrandmaCooking",
          "TurkishNewsAnchor",
          "WeatherPresenterCheerful"
        ],
        "required": false,
        "default": "Bruno",
        "description": "Prepared voice name available in the loaded package. A supplied reference clip takes precedence."
      },
      {
        "name": "preset",
        "type": "enum",
        "values": [
          "stable",
          "expressive"
        ],
        "required": false,
        "default": "stable",
        "description": "Upstream sampling preset."
      },
      {
        "name": "temperature",
        "type": "float",
        "min": 0,
        "required": false,
        "description": "Sampling temperature; zero selects greedy decoding. Otherwise defaults to the preset."
      },
      {
        "name": "top_k",
        "type": "int",
        "min": 0,
        "required": false,
        "description": "Top-k sampling cutoff; defaults to the preset."
      },
      {
        "name": "top_p",
        "type": "float",
        "max": 1,
        "required": false,
        "description": "Nucleus sampling cutoff; defaults to the preset. Must be in (0, 1]."
      },
      {
        "name": "min_p",
        "type": "float",
        "min": 0,
        "max": 1,
        "required": false,
        "description": "Minimum relative probability; defaults to the preset."
      },
      {
        "name": "repetition_penalty",
        "type": "float",
        "min": 1,
        "required": false,
        "default": 1.1,
        "description": "Penalty over the last 50 generated tokens; silence and ending tokens are exempt."
      },
      {
        "name": "max_tokens",
        "type": "int",
        "min": 1,
        "max": 8192,
        "required": false,
        "description": "Speech-token budget per text chunk. Omitted budgets scale with text length, with a minimum of 200."
      },
      {
        "name": "seed",
        "type": "int",
        "min": 0,
        "required": false,
        "description": "Seed for sampling and waveform decoding."
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "min": 32,
        "max": 1000,
        "required": false,
        "default": 380,
        "description": "Maximum Unicode codepoints per framework text chunk."
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
        "required": false,
        "default": "tag_aware",
        "description": "Framework long-form text chunking mode."
      },
      {
        "name": "reference_text",
        "type": "string",
        "required": false,
        "description": "Transcript of the reference audio; required when cloning from a 1\u201330 second clip."
      }
    ],
    "load": [],
    "session": [
      {
        "name": "weight_type",
        "type": "enum",
        "values": [
          "native",
          "f32",
          "f16",
          "bf16",
          "q4_0",
          "q8_0"
        ],
        "required": false,
        "default": "native",
        "description": "Qwen projection storage. native preserves the GGUF's intended tensor types. An explicit override changes projection storage only."
      }
    ]
  },
  "package_defaults": {
    "download": {
      "kind": "huggingface_snapshot",
      "repo": "dignome/kitten_tts2",
      "revision": "main",
      "gated": false
    }
  },
  "packages": [
    {
      "id": "kitten_tts2_bf16",
      "display_name": "Kitten TTS 2 BF16 GGUF (Previous Checkpoint)",
      "default": false,
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "kitten-tts2-v2",
      "files": [
        "kitten-tts2-bf16-multilingual.gguf",
        "LICENSE",
        "NOTICE",
        "native/LICENSE",
        "speaker/LICENSE",
        "SHA256SUMS"
      ]
    },
    {
      "id": "kitten_tts2_q8_v2",
      "display_name": "Kitten TTS 2 Q8 GGUF (Previous Checkpoint)",
      "default": false,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "kitten-tts2-v2",
      "files": [
        "kitten-tts2-q8-v2-multilingual.gguf",
        "LICENSE",
        "NOTICE",
        "native/LICENSE",
        "speaker/LICENSE",
        "SHA256SUMS"
      ]
    },
    {
      "id": "kitten_tts2_q4",
      "display_name": "Kitten TTS 2 Hybrid Q4 GGUF (Previous Checkpoint, Experimental)",
      "default": false,
      "format": "gguf",
      "precision": "q4_0",
      "target_directory": "kitten-tts2-v2",
      "files": [
        "kitten-tts2-q4-multilingual.gguf",
        "LICENSE",
        "NOTICE",
        "native/LICENSE",
        "speaker/LICENSE",
        "SHA256SUMS"
      ]
    },
    {
      "id": "kitten_tts2_20261021_q8_0",
      "display_name": "Kitten TTS 2 2026-10-21 Q8 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "KittenTTS2-GGUF",
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "audio-cpp/audio.cpp-gguf",
        "revision": "main",
        "gated": false
      },
      "files": [
        "KittenTTS2-GGUF/kitten-tts2-20261021-q8_0.gguf"
      ],
      "strip_prefix": "KittenTTS2-GGUF"
    },
    {
      "id": "kitten_tts2_20261021_q4_0",
      "display_name": "Kitten TTS 2 2026-10-21 Q4 GGUF (BF16 Embedding)",
      "default": false,
      "format": "gguf",
      "precision": "q4_0",
      "target_directory": "KittenTTS2-GGUF",
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "audio-cpp/audio.cpp-gguf",
        "revision": "main",
        "gated": false
      },
      "files": [
        "KittenTTS2-GGUF/kitten-tts2-20261021-q4_0.gguf"
      ],
      "strip_prefix": "KittenTTS2-GGUF"
    }
  ],
  "dependencies": [],
  "ui": {
    "recommended_package": "kitten_tts2_20261021_q8_0",
    "min_vram_gb": 7,
    "tags": [
      "TTS",
      "GGUF"
    ],
    "builtin_voices": [
      "Bella",
      "Jasper",
      "Luna",
      "Bruno",
      "Rosie",
      "Hugo",
      "Kiki",
      "Leo",
      "Matthew",
      "Elliot",
      "Willow",
      "Dolores",
      "Victor",
      "Dante",
      "Alfred",
      "Saoirse",
      "Claire",
      "Raven",
      "Marcus",
      "Herbert",
      "Diana",
      "Laurence",
      "Maeve",
      "Walter",
      "Edith",
      "Miles",
      "Grace",
      "Reginald",
      "Iris",
      "Frank",
      "Serena",
      "Julian",
      "Eleanor",
      "Otis",
      "Vincent",
      "Martha",
      "Sable",
      "Victoria",
      "Arabic",
      "Chinese",
      "French",
      "German",
      "Hindi",
      "Italian",
      "Portuguese",
      "Russian",
      "Spanish"
    ],
    "default_voice": "Bruno",
    "docs": [
      "docs/models/kitten_tts2.md"
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
        "lm_config": "model:lm/config.json",
        "tokenizer_config": "model:lm/tokenizer_config.json",
        "tokenizer_json": "model:lm/tokenizer.json",
        "default_voices": "model:cpp/default/voices.json"
      },
      "tensors": {
        "language_model": {
          "source": "weights:",
          "prefix": "language_model"
        },
        "s3gen": {
          "source": "weights:",
          "prefix": "s3gen"
        },
        "speaker": {
          "source": "weights:",
          "prefix": "speaker"
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
        "lm_config": "model:lm/config.json",
        "tokenizer_config": "model:lm/tokenizer_config.json",
        "tokenizer_json": "model:lm/tokenizer.json",
)AUDIOCPP_SPEC" R"AUDIOCPP_SPEC(        "default_voices": "model:cpp/default/voices.json"
      },
      "tensors": {
        "language_model": "model:lm/model.safetensors",
        "s3gen": "model:native/s3gen_meanflow.safetensors",
        "speaker": "model:speaker/model.safetensors"
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
    {"omnivoice", R"AUDIOCPP_SPEC({
  "family": "omnivoice",
  "display_name": "OmniVoice",
  "description": "Massively multilingual zero-shot TTS model from k2-fsa for 600+ languages, supporting short-reference voice cloning, attribute-based voice design, pronunciation controls, and nonverbal tags.",
  "category": "tts",
  "status": "supported",
  "tasks": [
    "tts",
    "clone",
    "design"
  ],
  "modes": [
    "offline",
    "streaming"
  ],
  "languages": [
    "600+ languages"
  ],
  "capabilities": {
    "clone": [
      "speaker_reference"
    ],
    "design": [
      "voice_design"
    ]
  },
  "options": {
    "request": [
      {
        "name": "language",
        "type": "string",
        "description": "Synthesis language code or label; omitted uses automatic language selection.",
        "required": false
      },
      {
        "name": "reference_text",
        "type": "string",
        "description": "Transcript for speaker reference audio used in voice cloning.",
        "required": false
      },
      {
        "name": "instruct",
        "type": "string",
        "description": "Voice-design instruction such as speaker attributes, accent, style, or delivery.",
        "required": false
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Generation seed for reproducible output; omitted value keeps the session RNG state.",
        "required": false,
        "min": 0,
        "max": 4294967295
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "description": "Diffusion language-model inference steps; default 32.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "guidance_scale",
        "type": "float",
        "description": "Classifier-free guidance scale; default 2.0.",
        "required": false,
        "default": 2
      },
      {
        "name": "speed",
        "type": "float",
        "description": "Speech speed multiplier used for target-length estimation; default 1.0.",
        "required": false,
        "default": 1
      },
      {
        "name": "duration_sec",
        "type": "float",
        "description": "Target output duration in seconds; omitted uses the model's estimated duration.",
        "required": false,
        "min": 0
      },
      {
        "name": "shift",
        "type": "float",
        "description": "Diffusion timestep schedule shift; default 0.1.",
        "required": false,
        "default": 0.1
      },
      {
        "name": "denoise",
        "type": "bool",
        "description": "Add the denoise control token when reference audio is present; default true.",
        "required": false,
        "default": true
      },
      {
        "name": "preprocess_prompt",
        "type": "bool",
        "description": "Preprocess speaker reference audio before tokenization; default true.",
        "required": false,
        "default": true
      },
      {
        "name": "postprocess_output",
        "type": "bool",
        "description": "Apply output postprocessing and trimming; default true.",
        "required": false,
        "default": true
      },
      {
        "name": "layer_penalty_factor",
        "type": "float",
        "description": "Layer-order penalty used while selecting masked audio-codebook positions; default 5.0.",
        "required": false,
        "default": 5
      },
      {
        "name": "position_temperature",
        "type": "float",
        "description": "Gumbel temperature for masked-position selection; default 5.0.",
        "required": false,
        "default": 5
      },
      {
        "name": "class_temperature",
        "type": "float",
        "description": "Audio-token class sampling temperature; default 0 for greedy class selection.",
        "required": false,
        "min": 0,
        "default": 0
      },
      {
        "name": "audio_chunk_duration_sec",
        "type": "float",
        "description": "Target chunk duration for automatic long-form splitting; default 15 seconds.",
        "required": false,
        "min": 0,
        "default": 15
      },
      {
        "name": "audio_chunk_threshold_sec",
        "type": "float",
        "description": "Estimated duration above which long-form generation is split into chunks; default 30 seconds.",
        "required": false,
        "min": 0,
        "default": 30
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Explicit long-form text chunk size in Unicode codepoints. Omitted value uses duration-based automatic chunking.",
        "required": false,
        "min": 1
      },
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "description": "Text chunking mode; default tag_aware.",
        "preset": "text_chunk_mode_full",
        "required": false,
        "default": "tag_aware"
      }
    ],
    "session": [
      {
        "name": "weight_type",
        "type": "enum",
        "description": "Shared generator weight storage type; default native.",
        "preset": "weight_type_full",
        "required": false,
        "default": "native"
      },
      {
        "name": "generator_weight_type",
        "type": "enum",
        "description": "Generator weight storage type; defaults to weight_type, which defaults to native.",
        "preset": "weight_type_full",
        "required": false
      },
      {
        "name": "audio_tokenizer_weight_type",
        "type": "enum",
        "description": "Audio tokenizer weight storage type; default native.",
        "preset": "weight_type_full",
        "required": false,
        "default": "native"
      },
      {
        "name": "audio_tokenizer_graph_arena_mb",
        "type": "int",
        "description": "Audio tokenizer graph arena size in MiB; default 128.",
        "required": false,
        "min": 1,
        "default": 128
      },
      {
        "name": "generator_prefill_graph_arena_mb",
        "type": "int",
        "description": "Generator prefill graph arena size in MiB; default 256.",
        "required": false,
        "min": 1,
        "default": 256
      },
      {
        "name": "generator_decode_graph_arena_mb",
        "type": "int",
        "description": "Generator decode graph arena size in MiB; default 256.",
        "required": false,
        "min": 1,
        "default": 256
      },
      {
        "name": "audio_tokenizer_weight_context_mb",
        "type": "int",
        "description": "Audio tokenizer weight context size in MiB; default 128.",
        "required": false,
        "min": 1,
        "default": 128
      },
      {
        "name": "generator_weight_context_mb",
        "type": "int",
        "description": "Generator weight context size in MiB; default 256.",
        "required": false,
        "min": 1,
        "default": 256
      },
      {
        "name": "mem_saver",
        "type": "bool",
        "description": "Release staged runtime graphs after request phases; default false. Cannot be combined with perf_mode=flash_attention.",
        "required": false,
        "default": false
      },
      {
        "name": "perf_mode",
        "type": "enum",
        "description": "Generator performance mode; default off keeps the exact-safe attention path. flash_attention requires mem_saver=false.",
        "preset": "perf_mode_flash_attention",
        "required": false,
        "default": "off"
      }
    ],
    "load": []
  },
  "runtime": {
    "tags": [
      "gguf",
      "stream"
    ]
  },
  "ui": {
    "recommended_package": "omnivoice_q8_0",
    "tags": [
      "TTS",
      "Clone",
      "Design",
      "GGUF",
      "Stream"
    ],
    "docs": [
      "docs/models/omnivoice.md",
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
      "id": "omnivoice_q8_0",
      "display_name": "OmniVoice Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "OmniVoice-GGUF",
      "files": [
        "OmniVoice-GGUF/omnivoice-q8_0.gguf"
      ],
      "strip_prefix": "OmniVoice-GGUF"
    },
    {
      "id": "omnivoice_bf16",
      "display_name": "OmniVoice BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "OmniVoice-GGUF",
      "files": [
        "OmniVoice-GGUF/omnivoice-bf16.gguf"
      ],
      "strip_prefix": "OmniVoice-GGUF"
    },
    {
      "id": "omnivoice_f16",
      "display_name": "OmniVoice F16 GGUF",
      "format": "gguf",
      "precision": "f16",
      "target_directory": "OmniVoice-GGUF",
      "files": [
        "OmniVoice-GGUF/omnivoice-f16.gguf"
      ],
      "strip_prefix": "OmniVoice-GGUF"
    },
    {
      "id": "omnivoice_safetensors",
      "display_name": "OmniVoice Safetensors",
      "format": "safetensors",
      "precision": "native",
      "target_directory": "OmniVoice",
      "files": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/preprocessor_config.json",
        "audio_tokenizer/model.safetensors"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "k2-fsa/OmniVoice"
      }
    },
    {
      "id": "voicetut_tts_q8_0",
      "display_name": "VoiceTut-TTS Egyptian Arabic Q8_0 GGUF",
      "description": "Egyptian Arabic and Arabic-English code-switching fine-tune of OmniVoice from mohammedaly22/VoiceTut-TTS (same architecture and audio tokenizer as OmniVoice). Use language=arz.",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "VoiceTut-TTS-GGUF",
      "files": [
        "voicetut-tts-q8_0.gguf"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "mohammedaly22/VoiceTut-TTS-GGUF"
      }
    },
    {
      "id": "voicetut_tts_f16",
      "display_name": "VoiceTut-TTS Egyptian Arabic F16 GGUF",
      "description": "Egyptian Arabic and Arabic-English code-switching fine-tune of OmniVoice from mohammedaly22/VoiceTut-TTS (same architecture and audio tokenizer as OmniVoice). Use language=arz.",
      "format": "gguf",
      "precision": "f16",
      "target_directory": "VoiceTut-TTS-GGUF",
      "files": [
        "voicetut-tts-f16.gguf"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "mohammedaly22/VoiceTut-TTS-GGUF"
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
        "config": "model:config.json",
        "tokenizer_json": "model:tokenizer.json",
        "tokenizer_config": "model:tokenizer_config.json",
        "audio_tokenizer_config": "model:audio_tokenizer/config.json",
        "audio_tokenizer_preprocessor": "model:audio_tokenizer/preprocessor_config.json"
      },
      "optional_files": {
        "chat_template": "model:chat_template.jinja"
      },
      "tensors": {
        "weights": {
          "source": "weights:",
          "prefix": "weights"
        },
        "audio_tokenizer_weights": {
          "source": "weights:",
          "prefix": "audio_tokenizer_weights"
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
        "tokenizer_json": "model:tokenizer.json",
        "tokenizer_config": "model:tokenizer_config.json",
        "audio_tokenizer_config": "model:audio_tokenizer/config.json",
        "audio_tokenizer_preprocessor": "model:audio_tokenizer/preprocessor_config.json"
      },
      "optional_files": {
        "chat_template": "model:chat_template.jinja"
      },
      "tensors": {
)AUDIOCPP_SPEC" R"AUDIOCPP_SPEC(        "weights": "model:model.safetensors",
        "audio_tokenizer_weights": "model:audio_tokenizer/model.safetensors"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"pocket_tts", R"AUDIOCPP_SPEC({
  "family": "pocket_tts",
  "display_name": "PocketTTS",
  "description": "Kyutai 100M-parameter CPU-friendly TTS package set for real-time local synthesis and small-footprint voice cloning in English, French, German, Italian, Portuguese, and Spanish.",
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
    "en",
    "fr",
    "de",
    "it",
    "pt",
    "es"
  ],
  "options": {
    "request": [
      {
        "name": "frames_after_eos",
        "type": "int",
        "description": "Extra frames after EOS; -1 uses the model-derived value.",
        "required": false,
        "min": -1,
        "default": -1
      },
      {
        "name": "max_frames",
        "type": "int",
        "description": "Maximum generated audio frames; zero estimates the cap from text length. Captured during session preparation.",
        "required": false,
        "min": 0,
        "default": 0
      },
      {
        "name": "text_chunk_tokens",
        "type": "int",
        "description": "Tokenizer tokens per text chunk; must be positive, default 50. Set during preparation and overridable per request.",
        "required": false,
        "min": 1,
        "default": 50
      },
      {
        "name": "temperature",
        "type": "float",
        "description": "Positive flow noise temperature; defaults to the model configuration. Captured during session preparation.",
        "required": false
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Noise seed; omitted chooses a random seed. Captured during session preparation.",
        "required": false,
        "min": 0,
        "max": 4294967295
      },
      {
        "name": "noise_clamp",
        "type": "float",
        "description": "Noise clamp; a negative value disables clamping. Captured during session preparation.",
        "required": false,
        "default": -1
      },
      {
        "name": "eos_threshold",
        "type": "float",
        "description": "EOS detection threshold. Captured during session preparation.",
        "required": false,
        "default": -4
      },
      {
        "name": "noise_file",
        "type": "path",
        "description": "Precomputed noise schedule file. Captured during session preparation.",
        "required": false
      },
      {
        "name": "voice_embedding_file",
        "type": "path",
        "description": "Precomputed voice-state embedding path. Captured during session preparation.",
        "required": false
      },
      {
        "name": "truncate_clone_audio",
        "type": "bool",
        "description": "Truncate voice clone audio. Captured during session preparation.",
        "required": false,
        "default": false
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Optional per-request character chunk size; otherwise use sentence/token budgeting.",
        "required": false
      },
      {
        "name": "export_voice_state_file",
        "type": "path",
        "description": "Export the prepared voice state during session preparation.",
        "required": false
      }
    ],
    "session": [
      {
        "name": "voice_state_cache_slots",
        "type": "int",
        "description": "Cached prepared voice states; zero disables reuse.",
        "required": false,
        "min": 0,
        "default": 4
      },
      {
        "name": "offline_graph_capacity_mode",
        "type": "enum",
        "description": "Offline graph capacity policy; CPU/Metal default tiered, other backends double.",
        "required": false,
        "values": [
          "fixed",
          "double",
          "tiered",
          "grow"
        ]
      },
      {
        "name": "prompt_graph_capacity",
        "type": "int",
        "description": "Optional reserved prompt capacity.",
        "required": false,
        "min": 1
      },
      {
        "name": "generation_graph_capacity",
        "type": "int",
        "description": "Optional reserved generation capacity.",
        "required": false,
        "min": 1
      },
      {
        "name": "weight_type",
        "type": "enum",
        "description": "Shared matmul weight storage; component matmul settings inherit it. Convolution storage remains independently native unless overridden.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "matmul_weight_type",
        "type": "enum",
        "description": "Weight storage type; component-specific settings inherit weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "conv_weight_type",
        "type": "enum",
        "description": "Convolution weight storage.",
        "required": false,
        "preset": "weight_type_conv",
        "default": "native"
      },
      {
        "name": "weight_context_mb",
        "type": "int",
        "description": "Context or graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 64
      },
      {
        "name": "flow_weight_context_mb",
        "type": "int",
        "description": "Flow weight context in MiB; inherits pocket_tts.weight_context_mb when omitted.",
        "required": false,
        "min": 1
      },
      {
        "name": "mimi_encoder_weight_context_mb",
        "type": "int",
        "description": "Mimi encoder weight context in MiB; inherits pocket_tts.weight_context_mb when omitted.",
        "required": false,
        "min": 1
      },
      {
        "name": "mimi_decoder_weight_context_mb",
        "type": "int",
        "description": "Mimi decoder weight context in MiB; inherits pocket_tts.weight_context_mb when omitted.",
        "required": false,
        "min": 1
      },
      {
        "name": "flow_weights_view_context_mb",
        "type": "int",
        "description": "Context or graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 256
      },
      {
        "name": "flow_step_graph_arena_mb",
        "type": "int",
        "description": "Flow-step host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 256
      },
      {
        "name": "mimi_encoder_graph_arena_mb",
        "type": "int",
        "description": "Mimi encoder host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 512
      },
      {
        "name": "mimi_conv_graph_arena_mb",
        "type": "int",
        "description": "Mimi convolution host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 32
      },
      {
        "name": "mimi_transformer_graph_arena_mb",
        "type": "int",
        "description": "Mimi transformer host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 96
      },
      {
        "name": "mimi_tail_graph_arena_mb",
        "type": "int",
        "description": "Mimi tail host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 512
      },
      {
        "name": "mimi_full_chunk_frames",
        "type": "int",
        "description": "Full-sequence decoder chunk frames.",
        "required": false,
        "min": 1,
        "default": 90
      },
      {
        "name": "mimi_stage2_chunk_frames",
        "type": "int",
        "description": "Second-stage decoder chunk frames.",
        "required": false,
        "min": 1,
        "default": 900
      },
      {
        "name": "use_full_mimi",
        "type": "bool",
        "description": "Use full-sequence Mimi decoding.",
        "required": false,
        "default": true
      }
    ],
    "load": [
      {
        "name": "language",
        "type": "string",
        "description": "Model language selection.",
        "required": false,
        "default": "english"
      }
    ]
  },
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
  "ui": {
    "recommended_package": "pocket_tts_english_q8_0",
    "default_voice": "alba",
    "builtin_voices": [
      "alba"
    ],
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
      "id": "pocket_tts_english_q8_0",
      "display_name": "PocketTTS English Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "PocketTTS-GGUF/english",
      "files": [
        "PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
        "PocketTTS-GGUF/english/embeddings/alba.safetensors"
      ],
      "strip_prefix": "PocketTTS-GGUF/english"
    },
    {
      "id": "pocket_tts_english_bf16",
      "display_name": "PocketTTS English BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "PocketTTS-GGUF/english",
      "files": [
        "PocketTTS-GGUF/english/pocket-tts-english-bf16.gguf",
        "PocketTTS-GGUF/english/embeddings/alba.safetensors"
      ],
      "strip_prefix": "PocketTTS-GGUF/english"
    },
    {
      "id": "pocket_tts_french_q8_0",
      "display_name": "PocketTTS French Q8_0 GGUF",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "PocketTTS-GGUF/french",
      "files": [
        "PocketTTS-GGUF/french/pocket-tts-french-q8_0.gguf",
        "PocketTTS-GGUF/french/embeddings/cosette.safetensors"
      ],
      "strip_prefix": "PocketTTS-GGUF/french"
    },
    {
      "id": "pocket_tts_french_bf16",
      "display_name": "PocketTTS French BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "PocketTTS-GGUF/french",
      "files": [
        "PocketTTS-GGUF/french/pocket-tts-french-bf16.gguf",
        "PocketTTS-GGUF/french/embeddings/cosette.safetensors"
      ],
      "strip_prefix": "PocketTTS-GGUF/french"
    },
    {
      "id": "pocket_tts_german_q8_0",
      "display_name": "PocketTTS German Q8_0 GGUF",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "PocketTTS-GGUF/german",
      "files": [
        "PocketTTS-GGUF/german/pocket-tts-german-q8_0.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/german"
    },
    {
      "id": "pocket_tts_german_bf16",
      "display_name": "PocketTTS German BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "PocketTTS-GGUF/german",
      "files": [
        "PocketTTS-GGUF/german/pocket-tts-german-bf16.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/german"
    },
    {
      "id": "pocket_tts_italian_q8_0",
      "display_name": "PocketTTS Italian Q8_0 GGUF",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "PocketTTS-GGUF/italian",
      "files": [
        "PocketTTS-GGUF/italian/pocket-tts-italian-q8_0.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/italian"
    },
    {
      "id": "pocket_tts_italian_bf16",
      "display_name": "PocketTTS Italian BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "PocketTTS-GGUF/italian",
      "files": [
        "PocketTTS-GGUF/italian/pocket-tts-italian-bf16.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/italian"
    },
    {
      "id": "pocket_tts_portuguese_q8_0",
      "display_name": "PocketTTS Portuguese Q8_0 GGUF",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "PocketTTS-GGUF/portuguese",
)AUDIOCPP_SPEC" R"AUDIOCPP_SPEC(      "files": [
        "PocketTTS-GGUF/portuguese/pocket-tts-portuguese-q8_0.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/portuguese"
    },
    {
      "id": "pocket_tts_portuguese_bf16",
      "display_name": "PocketTTS Portuguese BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "PocketTTS-GGUF/portuguese",
      "files": [
        "PocketTTS-GGUF/portuguese/pocket-tts-portuguese-bf16.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/portuguese"
    },
    {
      "id": "pocket_tts_spanish_q8_0",
      "display_name": "PocketTTS Spanish Q8_0 GGUF",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "PocketTTS-GGUF/spanish",
      "files": [
        "PocketTTS-GGUF/spanish/pocket-tts-spanish-q8_0.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/spanish"
    },
    {
      "id": "pocket_tts_spanish_bf16",
      "display_name": "PocketTTS Spanish BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "PocketTTS-GGUF/spanish",
      "files": [
        "PocketTTS-GGUF/spanish/pocket-tts-spanish-bf16.gguf"
      ],
      "strip_prefix": "PocketTTS-GGUF/spanish"
    },
    {
      "id": "pocket_tts_english_safetensors",
      "display_name": "PocketTTS English Safetensors",
      "format": "safetensors",
      "precision": "native",
      "target_directory": "pocket-tts",
      "files": [
        "languages/english/embeddings/alba.safetensors",
        "languages/english/embeddings/anna.safetensors",
        "languages/english/embeddings/azelma.safetensors",
        "languages/english/embeddings/bill_boerst.safetensors",
        "languages/english/embeddings/caro_davy.safetensors",
        "languages/english/embeddings/charles.safetensors",
        "languages/english/embeddings/cosette.safetensors",
        "languages/english/embeddings/eponine.safetensors",
        "languages/english/embeddings/estelle.safetensors",
        "languages/english/embeddings/eve.safetensors",
        "languages/english/embeddings/fantine.safetensors",
        "languages/english/embeddings/george.safetensors",
        "languages/english/embeddings/giovanni.safetensors",
        "languages/english/embeddings/jane.safetensors",
        "languages/english/embeddings/javert.safetensors",
        "languages/english/embeddings/jean.safetensors",
        "languages/english/embeddings/juergen.safetensors",
        "languages/english/embeddings/lola.safetensors",
        "languages/english/embeddings/marius.safetensors",
        "languages/english/embeddings/mary.safetensors",
        "languages/english/embeddings/michael.safetensors",
        "languages/english/embeddings/paul.safetensors",
        "languages/english/embeddings/peter_yearsley.safetensors",
        "languages/english/embeddings/rafael.safetensors",
        "languages/english/embeddings/stuart_bell.safetensors",
        "languages/english/embeddings/vera.safetensors",
        "languages/english/model.safetensors",
        "languages/english/tokenizer.model"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "kyutai/pocket-tts",
        "gated": true
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
        "tokenizer": "model:tokenizer.model"
      },
      "optional_files": {
        "config": "model:config.yaml"
      },
      "tensors": {
        "weights": "weights:"
      }
    },
    {
      "format": "safetensors",
      "roots": {
        "language": "languages/english"
      },
      "files": {
        "tokenizer": "language:tokenizer.model"
      },
      "optional_files": {
        "config": "language:config.yaml"
      },
      "tensors": {
        "weights": "language:model.safetensors"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"soprano_tts", R"AUDIOCPP_SPEC({
  "schema_version": 1,
  "family": "soprano_tts",
  "display_name": "Soprano",
  "description": "Soprano is an ultra-lightweight (~80M) English-only text-to-speech model. Syntax uses a 17-layer Qwen3-style causal LM (hidden 512, vocab 8192) that autoregressively emits per-frame 512-dimensional features; a non-iterative Vocos-style decoder (ConvNeXt backbone + single ISTFT head, n_fft 2048 / hop 512) turns those features into 32 kHz audio. No diffusion refinement is performed in the decoder.",
  "category": "tts",
  "status": "community",
  "tasks": [
    "tts"
  ],
  "modes": [
    "offline",
    "streaming"
  ],
  "languages": [
    "en"
  ],
  "runtime": {
    "tags": [
      "gguf",
      "stream"
    ]
  },
  "capabilities": {
    "tts": [
      "long_form"
    ]
  },
  "options": {
    "request": [
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum generated audio frames for the autoregressive LM; default 512.",
        "required": false,
        "min": 1,
        "default": 512
      },
      {
        "name": "temperature",
        "type": "float",
        "description": "Autoregressive sampling temperature; must be > 0, default 0.3.",
        "required": false,
        "min": 0.0,
        "default": 0.3
      },
      {
        "name": "top_p",
        "type": "float",
        "description": "Nucleus sampling probability; default 0.95. Values >=1 disable nucleus filtering.",
        "required": false,
        "min": 0,
        "default": 0.95
      },
      {
        "name": "repetition_penalty",
        "type": "float",
        "description": "Repetition penalty applied to the LM head; default 1.2. Must be positive; values below 1 are accepted.",
        "required": false,
        "default": 1.2
      },
      {
        "name": "eos_bias",
        "type": "float",
        "description": "Additive bias on the EOS token logit during generation. Positive values make the model stop sooner when speech ends (mitigating runaway generations that hit max_tokens); negative values encourage longer utterances. Default 0 disables the adjustment.",
        "required": false,
        "default": 0.0
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Autoregressive sampling seed; omitted requests choose a random seed.",
        "required": false,
        "min": 0
      }
    ],
    "session": [
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Maximum codepoints per sentence chunk before the model generates and decodes separately. Smaller values keep prompts short (more reliable EOS) but increase overhead. Default 200.",
        "required": false,
        "min": 1,
        "default": 200
      },
      {
        "name": "backbone_weight_type",
        "type": "enum",
        "preset": "weight_type_full",
        "required": false,
        "default": "f32",
        "description": "Storage type for the Qwen3 LM backbone weights."
      },
      {
        "name": "decoder_weight_type",
        "type": "enum",
        "preset": "weight_type_conv",
        "required": false,
        "default": "f32",
        "description": "Storage type for the Vocos decoder weights."
      }
    ],
    "load": []
  },
  "package_defaults": {
    "download": {
      "kind": "huggingface_snapshot",
      "repo": "WalkingCat/Soprano-1.1-80M-GGUF",
      "revision": "main",
      "gated": false
    }
  },
  "packages": [
    {
      "id": "soprano_1_1_80m_q8_0",
      "display_name": "Soprano-1.1-80M Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Soprano-1.1-80M-GGUF",
      "files": [
        "Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf"
      ],
      "strip_prefix": "Soprano-1.1-80M-GGUF"
    },
    {
      "id": "soprano_1_1_80m_bf16",
      "display_name": "Soprano-1.1-80M BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "Soprano-1.1-80M-GGUF",
      "files": [
        "Soprano-1.1-80M-GGUF/soprano-1.1-80m-bf16.gguf"
      ],
      "strip_prefix": "Soprano-1.1-80M-GGUF"
    }
  ],
  "dependencies": [],
  "ui": {
    "recommended_package": "soprano_1_1_80m_q8_0",
    "tags": [
      "TTS",
      "Stream"
    ],
    "docs": [
      "docs/models/soprano_tts.md"
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
        "tokenizer_json": "model:tokenizer.json"
      },
      "tensors": {
        "backbone": "weights:",
        "decoder": "weights:"
      }
    },
    {
      "format": "safetensors",
      "roots": {
        "model": "."
      },
      "files": {
        "config": "model:config.json",
        "generation_config": "model:generation_config.json",
        "tokenizer_json": "model:tokenizer.json"
      },
      "tensors": {
        "backbone": "model:combined.safetensors",
        "decoder": "model:combined.safetensors"
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
    {"vibevoice", R"AUDIOCPP_SPEC({
  "family": "vibevoice",
  "display_name": "VibeVoice",
  "description": "Microsoft long-form multi-speaker TTS model for expressive conversational audio such as podcasts, supporting up to 90 minutes of speech with as many as four speakers.",
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
    "zh"
  ],
  "options": {
    "request": [
      {
        "name": "voice_samples",
        "type": "string",
        "description": "Comma-separated reference WAV paths ordered by speaker ID; cannot be combined with voice_ref.",
        "required": false
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "description": "Diffusion steps; defaults to the checkpoint configuration.",
        "required": false,
        "min": 1
      },
      {
        "name": "guidance_scale",
        "type": "float",
        "description": "Classifier-free guidance scale.",
        "required": false,
        "min": 0,
        "default": 1.3
      },
      {
        "name": "max_length_times",
        "type": "float",
        "description": "Positive generation length multiplier.",
        "required": false,
        "default": 2
      },
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum generated tokens; zero uses the model-derived limit.",
        "required": false,
        "min": 0,
        "default": 0
      },
      {
        "name": "do_sample",
        "type": "bool",
        "description": "Enable sampling.",
        "required": false,
        "default": false
      },
      {
        "name": "temperature",
        "type": "float",
        "description": "Positive sampling temperature.",
        "required": false,
        "default": 1
      },
      {
        "name": "top_k",
        "type": "int",
        "description": "Top-k sampling limit; zero disables filtering.",
        "required": false,
        "min": 0,
        "default": 50
      },
      {
        "name": "top_p",
        "type": "float",
        "description": "Nucleus probability in (0, 1].",
        "required": false,
        "default": 1
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Sampling seed.",
        "required": false,
        "min": 0,
        "max": 4294967295,
        "default": 1234
      },
      {
        "name": "prompt_noise_file",
        "type": "path",
        "description": "F32 prompt acoustic noise file.",
        "required": false
      },
      {
        "name": "diffusion_noise_file",
        "type": "path",
        "description": "F32 initial diffusion noise file.",
        "required": false
      }
    ],
    "session": [
      {
        "name": "weight_type",
        "type": "enum",
        "description": "Weight storage type; component-specific settings inherit weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "tokenizer_weight_type",
        "type": "enum",
        "description": "Weight storage type; component-specific settings inherit weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "connector_weight_type",
        "type": "enum",
        "description": "Weight storage type; component-specific settings inherit weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "decoder_weight_type",
        "type": "enum",
        "description": "Weight storage type; component-specific settings inherit weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "diffusion_head_weight_type",
        "type": "enum",
        "description": "Weight storage type; component-specific settings inherit weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "lora",
        "type": "path",
        "description": "Decoder adapter file or PEFT directory applied when creating the session; do not also supply the load option.",
        "required": false
      },
      {
        "name": "lora_scale",
        "type": "float",
        "description": "Session adapter scale override; omitted uses lora_alpha / r. Must be strictly positive when an adapter is supplied.",
        "required": false
      }
    ],
    "load": [
      {
        "name": "lora",
        "type": "path",
        "description": "Decoder adapter file or PEFT directory, merged at load time.",
        "required": false
      },
      {
        "name": "lora_scale",
        "type": "float",
        "description": "LoRA scale override; omitted uses lora_alpha / r. Must be strictly positive when an adapter is supplied.",
        "required": false
      }
    ]
  },
  "capabilities": {
    "tts": [
      "multi_speaker",
      "long_form"
    ]
  },
  "runtime": {
    "tags": [
      "gguf"
    ]
  },
  "ui": {
    "recommended_package": "vibevoice_1_5b_q8_0",
    "tags": [
      "TTS",
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
      "id": "vibevoice_1_5b_q8_0",
      "display_name": "VibeVoice 1.5B Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "VibeVoice-1.5B-GGUF",
      "files": [
        "VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf"
      ],
      "strip_prefix": "VibeVoice-1.5B-GGUF"
    },
    {
      "id": "vibevoice_1_5b_bf16",
      "display_name": "VibeVoice 1.5B BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "VibeVoice-1.5B-GGUF",
      "files": [
        "VibeVoice-1.5B-GGUF/vibevoice-1.5b-bf16.gguf"
      ],
      "strip_prefix": "VibeVoice-1.5B-GGUF"
    },
    {
      "id": "vibevoice_7b_q8_0",
      "display_name": "VibeVoice 7B Q8_0 GGUF",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "VibeVoice-7B-GGUF",
      "files": [
        "vibevoice-7b-q8_0.gguf"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "audio-cpp/VibeVoice-7B-GGUF",
        "revision": "main",
        "gated": false
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
        "config": "model:config.json",
        "preprocessor_config": "model:preprocessor_config.json",
        "tokenizer_config": "model:tokenizer_config.json",
        "tokenizer_json": "model:tokenizer.json",
        "tokenizer_vocab": "model:vocab.json",
        "tokenizer_merges": "model:merges.txt"
      },
      "tensors": {
        "model_weights": "weights:"
      }
    },
    {
      "format": "safetensors",
      "roots": {
        "model": "."
      },
      "files": {
        "config": "model:config.json",
        "preprocessor_config": "model:preprocessor_config.json",
        "tokenizer_config": "model:tokenizer_config.json",
        "tokenizer_json": "model:tokenizer.json",
        "tokenizer_vocab": "model:vocab.json",
        "tokenizer_merges": "model:merges.txt"
      },
      "tensors": {
        "model_weights": "model:model.safetensors.index.json"
      }
    }
  ]
}
)AUDIOCPP_SPEC"},
    {"voxcpm1", R"AUDIOCPP_SPEC({
  "schema_version": 1,
  "family": "voxcpm1",
  "display_name": "VoxCPM1",
  "description": "OpenBMB VoxCPM 0.5B tokenizer-free TTS model supporting short-reference voice cloning and streaming output (16kHz).",
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
    "zh",
    "en",
    "ja",
    "ko"
  ],
  "capabilities": {
    "clone": [
      "speaker_reference"
    ]
  },
  "dependencies": [],
  "options": {
    "request": [
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "description": "Text chunking mode; default tag_aware.",
        "preset": "text_chunk_mode_full",
        "required": false,
        "default": "tag_aware"
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Random seed for MiniCPM and diffusion sampling.",
        "required": false
      },
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum MiniCPM output tokens; default 4096. Zero selects model max_length; negative counts are rejected.",
        "required": false,
        "default": 4096
      },
      {
        "name": "min_tokens",
        "type": "int",
        "description": "Minimum MiniCPM output tokens before an EOS stop is honored; default 2.",
        "required": false,
        "default": 2
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "description": "CFM diffusion sampling steps; default 10. Must be positive.",
        "required": false,
        "default": 10
      },
      {
        "name": "guidance_scale",
        "type": "float",
        "description": "CFM classifier-free guidance rate. Must be nonnegative.",
        "required": false,
        "default": 2
      },
      {
        "name": "retry_badcase",
        "type": "bool",
        "description": "Retry when generation is detected as a bad case; defaults to true offline and false in streaming. Streaming ignores an explicit true value with a warning.",
        "required": false
      },
      {
        "name": "retry_badcase_max_times",
        "type": "int",
        "description": "Maximum total generation attempts when retrying bad cases, including the initial attempt; default 3. Must be positive.",
        "required": false,
        "default": 3
      },
      {
        "name": "retry_badcase_ratio_threshold",
        "type": "float",
        "description": "Bad-case ratio threshold for retry decisions. Must be positive.",
        "required": false
      },
      {
        "name": "prompt_text",
        "type": "string",
        "description": "Text prompt for prompt-continuation voice cloning.",
        "required": false
      },
      {
        "name": "reference_text",
        "type": "string",
        "description": "Alias for prompt_text; text prompt for prompt-continuation voice cloning.",
        "required": false
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Maximum characters per offline text chunk; must be positive. Does not control streaming chunk cadence.",
        "required": false,
        "min": 1,
        "default": 2048
      },
      {
        "name": "language",
        "type": "string",
        "description": "Language hint for Chinese-script normalization; Cantonese/yue preserves Traditional characters. Inherits structured transcript or voice language when provided.",
        "required": false
      }
    ],
    "session": [
      {
        "name": "lora",
        "type": "path",
        "description": "Native LoRA directory or checkpoint file with adjacent lora_config.json; merged into generator weights at session creation.",
        "required": false
      },
      {
        "name": "lora_scale",
        "type": "float",
        "description": "Multiplier for the adapter's configured alpha/r scaling; default 1.",
        "required": false,
        "default": 1.0
      },
      {
        "name": "mem_saver",
        "type": "bool",
        "description": "Use tighter graph workspaces and release request runtime graphs; default false.",
        "required": false,
        "default": false
      },
      {
        "name": "prompt_cache_slots",
        "type": "int",
        "description": "Prompt and prompt-audio embedding cache slots; default 1.",
        "required": false,
        "default": 1
      },
      {
        "name": "weight_type",
        "type": "enum",
        "description": "Model weight storage type.",
        "preset": "weight_type_full",
        "required": false,
        "default": "native"
      },
      {
        "name": "audiovae_weight_type",
        "type": "enum",
        "description": "AudioVAE weight storage type.",
        "preset": "weight_type_full",
        "required": false,
        "default": "native"
      },
      {
        "name": "weight_context_mb",
        "type": "int",
        "description": "Model weight graph context size in MB.",
        "required": false
      },
      {
        "name": "text_embedding_graph_arena_mb",
        "type": "int",
        "description": "Text embedding host graph-metadata arena size in MiB, not device workspace.",
        "required": false
      },
      {
        "name": "lm_step_graph_arena_mb",
        "type": "int",
        "description": "LM step host graph-metadata arena size in MiB, not device workspace.",
        "required": false
      },
      {
        "name": "projection_graph_arena_mb",
        "type": "int",
        "description": "Projection host graph-metadata arena size in MiB, not device workspace.",
        "required": false
      },
      {
        "name": "local_encoder_graph_arena_mb",
        "type": "int",
        "description": "Local encoder host graph-metadata arena size in MiB, not device workspace.",
        "required": false
      },
      {
        "name": "dit_graph_arena_mb",
        "type": "int",
        "description": "DiT estimator host graph-metadata arena size in MiB, not device workspace.",
        "required": false
      },
      {
        "name": "audiovae_weight_context_mb",
        "type": "int",
        "description": "AudioVAE weight graph context size in MB.",
        "required": false
      },
      {
        "name": "audiovae_graph_arena_mb",
        "type": "int",
        "description": "AudioVAE decoder host graph-metadata arena size in MiB, not device workspace.",
        "required": false
      },
      {
        "name": "audiovae_encoder_graph_arena_mb",
        "type": "int",
        "description": "AudioVAE encoder host graph-metadata arena size in MiB, not device workspace.",
        "required": false
      },
      {
        "name": "audiovae_latent_capacity",
        "type": "int",
        "description": "AudioVAE decoder latent frame capacity.",
        "required": false
      },
      {
        "name": "audiovae_encoder_sample_capacity",
        "type": "int",
        "description": "AudioVAE encoder sample capacity.",
        "required": false
      }
    ],
    "load": []
  },
  "runtime": {
    "tags": [
      "gguf",
      "stream"
    ]
  },
  "ui": {
    "recommended_package": "voxcpm1_0_5b_q8_0",
    "tags": [
      "TTS",
      "Clone",
      "GGUF",
      "Stream"
    ],
    "docs": [
      "docs/models/voxcpm1.md",
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
      "id": "voxcpm1_0_5b_q8_0",
      "display_name": "VoxCPM 0.5B Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "VoxCPM1-GGUF",
      "files": [
        "VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf"
      ],
      "strip_prefix": "VoxCPM1-GGUF"
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
        "tokenizer_json": "model:tokenizer.json",
        "tokenizer_config": "model:tokenizer_config.json"
      },
      "optional_files": {
        "special_tokens_map": "model:special_tokens_map.json"
      },
      "tensors": {
        "weights": {
          "source": "weights:"
        },
        "audiovae_weights": {
          "source": "weights:"
        }
      }
    }
  ]
}

)AUDIOCPP_SPEC"},
    {"voxcpm2", R"AUDIOCPP_SPEC({
  "family": "voxcpm2",
  "display_name": "VoxCPM2",
  "description": "OpenBMB tokenizer-free TTS model supporting 30 languages and 9 Chinese dialects, with 48 kHz output, natural-language voice design, controllable short-reference voice cloning, and expressive style guidance.",
  "category": "tts",
  "status": "supported",
  "tasks": [
    "tts",
    "clone",
    "design"
  ],
  "modes": [
    "offline",
    "streaming"
  ],
  "languages": [
    "ar",
    "my",
    "zh",
    "zh dialects",
    "da",
    "nl",
    "en",
    "fi",
    "fr",
    "de",
    "el",
    "he",
    "hi",
    "id",
    "it",
    "ja",
    "km",
    "ko",
    "lo",
    "ms",
    "no",
    "pl",
    "pt",
    "ru",
    "es",
    "sw",
    "sv",
    "tl",
    "th",
    "tr",
    "vi"
  ],
  "options": {
    "request": [
      {
        "name": "min_tokens",
        "type": "int",
        "description": "Minimum generated audio tokens.",
        "required": false,
        "min": 0,
        "default": 2
      },
      {
        "name": "max_tokens",
        "type": "int",
        "description": "Maximum generated audio tokens. Zero resolves to model max_length; requires min_tokens <= resolved max_tokens <= model limit.",
        "required": false,
        "min": 0,
        "default": 4096
      },
      {
        "name": "num_inference_steps",
        "type": "int",
        "description": "Flow sampling steps. Must be positive.",
        "required": false,
        "default": 10
      },
      {
        "name": "guidance_scale",
        "type": "float",
        "description": "Classifier-free guidance scale. Must be nonnegative.",
        "required": false,
        "default": 2
      },
      {
        "name": "retry_badcase",
        "type": "bool",
        "description": "Retry detected bad generations. Offline only; streaming requires false.",
        "required": false,
        "default": true
      },
      {
        "name": "retry_badcase_max_times",
        "type": "int",
        "description": "Total attempts including the initial attempt, not extra retries; must be positive.",
        "required": false,
        "default": 3
      },
      {
        "name": "retry_badcase_ratio_threshold",
        "type": "float",
        "description": "Bad-case audio/text length ratio threshold. Must be positive.",
        "required": false,
        "default": 6
      },
      {
        "name": "seed",
        "type": "int",
        "description": "Sampling seed.",
        "required": false,
        "min": 0,
        "max": 4294967295,
        "default": 1234
      },
      {
        "name": "cfm_noise_file",
        "type": "path",
        "description": "Precomputed flow noise file.",
        "required": false
      },
      {
        "name": "stream_chunk_policy",
        "type": "enum",
        "description": "Streaming decoder cadence: grow emits one patch first, then doubles toward stream_frames_per_event; fixed uses the configured size. Ignored offline.",
        "required": false,
        "values": [
          "grow",
          "fixed"
        ],
        "default": "grow"
      },
      {
        "name": "stream_frames_per_event",
        "type": "int",
        "description": "Maximum generated latent patches per streaming decoder event. Set to 1 for one-patch emission. Ignored offline.",
        "required": false,
        "min": 1,
        "default": 4
      },
      {
        "name": "stream_left_context",
        "type": "int",
        "description": "Previous patches decoded as streaming left context.",
        "required": false,
        "min": 0,
        "max": 8,
        "default": 3
      },
      {
        "name": "prompt_text",
        "type": "string",
        "description": "Transcript for continuation prompt audio, distinct from the base voice reference.",
        "required": false
      },
      {
        "name": "chunk_strategy",
        "type": "enum",
        "description": "Long-form continuation or independent generation per chunk. Offline only.",
        "required": false,
        "values": [
          "continuation",
          "stateless"
        ],
        "default": "continuation"
      },
      {
        "name": "text_chunk_size",
        "type": "int",
        "description": "Framework long-form text chunk size. Offline only.",
        "required": false,
        "default": 2048
      },
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "description": "Framework text chunking mode. Offline only.",
        "required": false,
        "values": [
          "default",
          "tag_aware",
          "japanese",
          "endline"
        ],
        "default": "tag_aware"
      }
    ],
    "session": [
      {
        "name": "audiovae_tile_frames",
        "type": "int",
        "description": "Opt-in causal AudioVAE tiling. Maximum new latent frames per tile; zero preserves full decoding.",
        "required": false,
        "default": 0,
        "min": 0
      },
      {
        "name": "perf_mode",
        "type": "enum",
        "description": "Opt-in packed attention and MLP projections. May change numerical results; off preserves standard execution.",
        "required": false,
        "values": [
          "off",
          "optimized"
        ],
        "default": "off"
      },
      {
        "name": "lora",
        "type": "path",
        "description": "Native LoRA directory or checkpoint file with adjacent lora_config.json; merged into generator weights at session creation.",
        "required": false
      },
      {
        "name": "lora_scale",
        "type": "float",
        "description": "Multiplier for the adapter's configured alpha/r scaling; default 1.",
        "required": false,
        "default": 1.0
      },
      {
        "name": "weight_type",
        "type": "enum",
        "description": "Language-model weight storage; does not override AudioVAE storage.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "audiovae_weight_type",
        "type": "enum",
        "description": "AudioVAE weight storage; defaults independently to native, not the shared weight_type.",
        "required": false,
        "preset": "weight_type_full"
      },
      {
        "name": "prompt_cache_slots",
        "type": "int",
        "description": "Cached encoded prompt slots; zero disables reuse.",
        "required": false,
        "min": 0,
        "default": 1
      },
      {
        "name": "mem_saver",
        "type": "bool",
        "description": "Release staged runtime graphs.",
        "required": false,
        "default": true
      },
      {
        "name": "weight_context_mb",
        "type": "int",
        "description": "Context or graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 3072
      },
      {
        "name": "text_embedding_graph_arena_mb",
        "type": "int",
        "description": "Text embedding host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 64
      },
      {
        "name": "lm_step_graph_arena_mb",
        "type": "int",
        "description": "LM step host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 1024
      },
      {
        "name": "projection_graph_arena_mb",
        "type": "int",
        "description": "Projection host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 256
      },
      {
        "name": "local_encoder_graph_arena_mb",
        "type": "int",
        "description": "Local encoder host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 512
      },
      {
        "name": "dit_graph_arena_mb",
        "type": "int",
        "description": "DiT estimator host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 1024
      },
      {
        "name": "audiovae_weight_context_mb",
        "type": "int",
        "description": "Context or graph arena size in MiB.",
        "required": false,
        "min": 1,
        "default": 768
      },
      {
        "name": "audiovae_graph_arena_mb",
        "type": "int",
        "description": "AudioVAE decoder host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 1024
      },
      {
        "name": "audiovae_encoder_graph_arena_mb",
        "type": "int",
        "description": "AudioVAE encoder host graph-metadata arena size in MiB, not device workspace.",
        "required": false,
        "min": 1,
        "default": 1024
      },
      {
        "name": "audiovae_latent_capacity",
        "type": "int",
        "description": "Optional decoder graph reservation in latent frames, reused by shorter requests and doubled when exceeded; omitted builds the graph at each request's own frame count.",
        "required": false,
        "min": 1
      },
      {
        "name": "exec_plan",
        "type": "string",
        "description": "JSON component execution plans. audiovae: impl=default/ggml; backend must match the session; no artifacts. params.threads is a nonnegative int: positive values require a CPU session and enable a dedicated AudioVAE backend with streaming decode overlap; also controls offline decoding and prompt encoding threads. Omitted/0 shares the session backend.",
        "required": false,
        "default": "{}"
      },
      {
        "name": "audiovae_encoder_sample_capacity",
        "type": "int",
        "description": "Reserved encoder sample capacity.",
        "required": false,
        "min": 1,
        "default": 240000
      }
    ],
    "load": []
  },
  "capabilities": {
    "clone": [
      "speaker_reference"
    ],
    "design": [
      "voice_design"
    ]
  },
  "runtime": {
    "tags": [
      "gguf",
      "stream"
    ]
  },
  "ui": {
    "recommended_package": "voxcpm2_q8_0",
    "tags": [
      "TTS",
      "Clone",
      "Design",
      "GGUF",
      "Stream"
    ],
    "docs": [
      "docs/tts.md",
      "docs/gguf.md",
      "docs/models/vocalrender.md"
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
      "id": "voxcpm2_q8_0",
      "display_name": "VoxCPM2 Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "VoxCPM2-GGUF",
      "files": [
        "VoxCPM2-GGUF/voxcpm2-q8_0.gguf"
      ],
      "strip_prefix": "VoxCPM2-GGUF"
    },
    {
      "id": "voxcpm2_bf16",
      "display_name": "VoxCPM2 BF16 GGUF",
      "format": "gguf",
      "precision": "bf16",
      "target_directory": "VoxCPM2-GGUF",
      "files": [
        "VoxCPM2-GGUF/voxcpm2-bf16.gguf"
      ],
      "strip_prefix": "VoxCPM2-GGUF"
    },
    {
      "id": "voxcpm2_orig",
      "display_name": "VoxCPM2 Original-Dtype GGUF",
      "format": "gguf",
      "precision": "orig",
      "target_directory": "VoxCPM2-GGUF",
      "files": [
        "VoxCPM2-GGUF/voxcpm2-orig.gguf"
      ],
      "strip_prefix": "VoxCPM2-GGUF"
    },
    {
      "id": "voxcpm2_vocalrender_pro",
      "display_name": "VocalRender-Pro (VoxCPM2) Original-Dtype GGUF",
      "format": "gguf",
      "precision": "orig",
      "target_directory": "VoxCPM2-GGUF",
      "files": [
        "VoxCPM2-GGUF/finetune/vocalrender-pro-orig.gguf"
      ],
      "strip_prefix": "VoxCPM2-GGUF"
    },
    {
      "id": "voxcpm2_safetensors",
      "display_name": "VoxCPM2 Safetensors",
      "format": "safetensors",
      "precision": "native",
      "target_directory": "VoxCPM2",
      "files": [
        "config.json",
)AUDIOCPP_SPEC" R"AUDIOCPP_SPEC(        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
      ],
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "OpenBMB/VoxCPM2"
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
        "config": "model:config.json",
        "tokenizer_config": "model:tokenizer_config.json",
        "tokenizer_json": "model:tokenizer.json",
        "special_tokens_map": "model:special_tokens_map.json"
      },
      "tensors": {
        "weights": {
          "source": "weights:",
          "prefix": "weights"
        },
        "audiovae_weights": {
          "source": "weights:",
          "prefix": "audiovae_weights"
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
        "tokenizer_json": "model:tokenizer.json",
        "special_tokens_map": "model:special_tokens_map.json"
      },
      "tensors": {
        "weights": "model:model.safetensors",
        "audiovae_weights": "model:audiovae.safetensors"
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
)AUDIOCPP_SPEC" R"AUDIOCPP_SPEC(      ]
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
