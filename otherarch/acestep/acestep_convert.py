#!/usr/bin/env python3
# convert.py: safetensors to GGUF for ACE-Step (LM, DiT, TextEncoder, VAE)
# Reads from checkpoints/, writes GGUF to models/
# Each GGUF is self-contained: weights + config + tokenizer + silence_latent

import os
import sys
import json
import struct
import zipfile
import numpy as np
import gguf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")

BF16 = gguf.GGMLQuantizationType.BF16

def log(tag, msg):
    print("[%s] %s" % (tag, msg), file=sys.stderr, flush=True)

# Safetensors reader
def read_sf_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return meta, 8 + n

def find_sf_files(model_dir):
    """Return list of safetensors paths (single, sharded, or diffusers VAE)."""
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        return [os.path.join(model_dir, s) for s in shards]
    diffusers = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(diffusers):
        return [diffusers]
    return []

# Model classification
ARCHS = {
    "lm":       "acestep-lm",
    "dit":      "acestep-dit",
    "text-enc": "acestep-text-enc",
    "vae":      "acestep-vae",
}

def classify(name):
    if name.startswith("acestep-5Hz-lm"):
        return "lm"
    if name.startswith("acestep-v15"):
        return "dit"
    if name.startswith("Qwen3-Embedding"):
        return "text-enc"
    if name == "vae":
        return "vae"
    return None

# GGUF metadata from config.json
def add_metadata(w, cfg, model_type):
    if "num_hidden_layers" in cfg:
        w.add_block_count(cfg["num_hidden_layers"])
    if "hidden_size" in cfg:
        w.add_embedding_length(cfg["hidden_size"])
    if "intermediate_size" in cfg:
        w.add_feed_forward_length(cfg["intermediate_size"])
    if "num_attention_heads" in cfg:
        w.add_head_count(cfg["num_attention_heads"])
    if "num_key_value_heads" in cfg:
        w.add_head_count_kv(cfg["num_key_value_heads"])
    if "head_dim" in cfg:
        w.add_key_length(cfg["head_dim"])
    if "vocab_size" in cfg:
        w.add_vocab_size(cfg["vocab_size"])
    if "max_position_embeddings" in cfg:
        w.add_context_length(cfg["max_position_embeddings"])
    if "rms_norm_eps" in cfg:
        w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    rope = cfg.get("rope_theta")
    if rope:
        w.add_rope_freq_base(float(rope))

    if model_type == "lm":
        if cfg.get("tie_word_embeddings"):
            w.add_bool("acestep.tie_word_embeddings", True)

    if model_type == "dit":
        for key in [
            "in_channels", "audio_acoustic_hidden_dim", "patch_size",
            "sliding_window", "fsq_dim", "text_hidden_dim", "timbre_hidden_dim",
            "num_lyric_encoder_hidden_layers", "num_timbre_encoder_hidden_layers",
            "num_audio_decoder_hidden_layers", "num_attention_pooler_hidden_layers",
        ]:
            if key in cfg:
                w.add_uint32("acestep.%s" % key, cfg[key])
        # XL models have separate encoder dimensions (2B models omit these)
        for key in [
            "encoder_hidden_size", "encoder_intermediate_size",
            "encoder_num_attention_heads", "encoder_num_key_value_heads",
        ]:
            if key in cfg:
                w.add_uint32("acestep.%s" % key, cfg[key])
        if cfg.get("is_turbo"):
            w.add_bool("acestep.is_turbo", True)
        levels = cfg.get("fsq_input_levels")
        if levels:
            w.add_array("acestep.fsq_input_levels", levels)

    w.add_string("acestep.config_json", json.dumps(cfg, separators=(",", ":")))

# Tensor packing from safetensors
def add_tensors_from_sf(w, sf_path, tag, model_type):
    meta, hdr_size = read_sf_header(sf_path)
    names = sorted(meta.keys())
    with open(sf_path, "rb") as f:
        count = 0
        total = 0

        for name in names:
            info = meta[name]

            # normalize: some upstream checkpoints omit the "model." prefix
            if model_type == "lm" and not name.startswith("model."):
                name = "model." + name

            dtype_str = info["dtype"]
            shape = info["shape"]
            off0, off1 = info["data_offsets"]
            nbytes = off1 - off0

            f.seek(hdr_size + off0)
            raw = f.read(nbytes)

            if dtype_str == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                w.add_tensor(name, arr, raw_dtype=BF16)
            elif dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
                w.add_tensor(name, arr)
            elif dtype_str == "F32":
                # convert F32 to BF16: truncate lower 16 mantissa bits
                arr = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
                arr = (arr >> 16).astype(np.uint16)
                w.add_tensor(name, arr, raw_dtype=BF16)
                nbytes = nbytes // 2  # actual stored size
            else:
                log(tag, "  skip %s: dtype %s" % (name, dtype_str))
                continue

            count += 1
            total += nbytes

    return count, total

# silence_latent.pt reader (replaces pt2bin C++ tool)
# PyTorch .pt is a ZIP with entry "*/data/0" containing f32 [64, 15000]
# We transpose to [15000, 64] (ggml layout: 64 contiguous per frame)
def read_silence_latent(model_dir):
    pt_path = os.path.join(model_dir, "silence_latent.pt")
    if not os.path.exists(pt_path):
        return None
    with zipfile.ZipFile(pt_path) as z:
        for entry in z.namelist():
            if entry.endswith("/data/0"):
                raw = z.read(entry)
                src = np.frombuffer(raw, dtype=np.float32).reshape(64, 15000)
                return src.T.copy()
    return None

# BPE tokenizer embedding (vocab.json + merges.txt -> GGUF KV)
def add_bpe_tokenizer(w, model_dir, tag):
    vocab_path = os.path.join(model_dir, "vocab.json")
    merges_path = os.path.join(model_dir, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        return False

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokens = [""] * len(vocab)
    for tok_str, tok_id in vocab.items():
        if 0 <= tok_id < len(tokens):
            tokens[tok_id] = tok_str

    with open(merges_path, "r", encoding="utf-8") as f:
        merges = []
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith("#version:"):
                continue
            merges.append(line)

    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)

    log(tag, "  tokenizer: %d vocab, %d merges" % (len(tokens), len(merges)))
    return True

# Main conversion
def convert_model(name, model_dir, output_path, model_type):
    tag = "GGUF"
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        log(tag, "skip %s: no config.json" % name)
        return False

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sf_files = find_sf_files(model_dir)
    if not sf_files:
        log(tag, "skip %s: no safetensors" % name)
        return False

    arch = ARCHS[model_type]
    log(tag, "%s (%s, %d shard%s) -> %s" % (
        name, arch, len(sf_files), "" if len(sf_files) == 1 else "s",
        os.path.basename(output_path)))

    w = gguf.GGUFWriter(output_path, arch, use_temp_file=True)
    w.add_name(name)
    add_metadata(w, cfg, model_type)

    # BPE tokenizer for LM and text encoder
    if model_type in ("lm", "text-enc"):
        add_bpe_tokenizer(w, model_dir, tag)

    # Model weights
    n_tensors = 0
    n_bytes = 0
    for sf in sf_files:
        c, b = add_tensors_from_sf(w, sf, tag, model_type)
        n_tensors += c
        n_bytes += b
        if len(sf_files) > 1:
            log(tag, "  %s: %d tensors" % (os.path.basename(sf), c))

    # silence_latent for DiT (read .pt, transpose, embed as f32 tensor)
    if model_type == "dit":
        sl = read_silence_latent(model_dir)
        if sl is not None:
            w.add_tensor("silence_latent", sl)
            n_tensors += 1
            n_bytes += sl.nbytes
            log(tag, "  silence_latent: [%d, %d] f32 (%.1f MB)" % (
                sl.shape[0], sl.shape[1], sl.nbytes / (1 << 20)))
        else:
            log(tag, "  WARNING: no silence_latent.pt found")

    log(tag, "  total: %d tensors, %.1f GB" % (n_tensors, n_bytes / (1 << 30)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    out_mb = os.path.getsize(output_path) / (1 << 20)
    log(tag, "  wrote %.0f MB -> %s" % (out_mb, output_path))
    return True

def main():
    if not os.path.isdir(CHECKPOINT_DIR):
        log("GGUF", "checkpoints/ not found, run checkpoints.sh first")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entries = sorted(os.listdir(CHECKPOINT_DIR))
    converted = 0
    skipped = []

    for name in entries:
        model_dir = os.path.join(CHECKPOINT_DIR, name)
        if not os.path.isdir(model_dir):
            continue

        model_type = classify(name)
        if model_type is None:
            skipped.append(name)
            continue

        output_path = os.path.join(OUTPUT_DIR, "%s-BF16.gguf" % name)
        if os.path.exists(output_path):
            log("GGUF", "skip %s: %s exists" % (name, os.path.basename(output_path)))
            converted += 1
            continue

        if convert_model(name, model_dir, output_path, model_type):
            converted += 1

    if skipped:
        log("GGUF", "skipped (unknown): %s" % ", ".join(skipped))
    log("GGUF", "done: %d model(s) in %s" % (converted, OUTPUT_DIR))

if __name__ == "__main__":
    main()
