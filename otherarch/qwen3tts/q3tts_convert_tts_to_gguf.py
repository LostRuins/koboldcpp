#!/usr/bin/env python3
"""
Convert HuggingFace Qwen3-TTS model to GGUF format.

Usage:
    python scripts/convert_tts_to_gguf.py \
        --input models/Qwen3-TTS-12Hz-0.6B-Base \
        --output models/qwen3-tts-0.6b-f16.gguf \
        --type f16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

# Add gguf-py to path (if available)
GGUF_PY_PATH = Path(__file__).resolve().parents[1] / "gguf-py"
try:
    if GGUF_PY_PATH.exists():
        sys.path.insert(0, str(GGUF_PY_PATH))
except (PermissionError, OSError):
    pass

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Qwen3TTSConverter:
    """Converter for Qwen3-TTS models to GGUF format."""

    # Direct tensor name mapping from HuggingFace to GGML conventions
    TENSOR_MAP = {
        # Talker - Main embeddings and heads
        "talker.model.codec_embedding.weight": "talker.codec_embd.weight",
        "talker.model.text_embedding.weight": "talker.text_embd.weight",
        "talker.codec_head.weight": "talker.codec_head.weight",
        "talker.model.norm.weight": "talker.output_norm.weight",
        # Talker - Text projection
        "talker.text_projection.linear_fc1.weight": "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias": "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight": "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias": "talker.text_proj.fc2.bias",
        # Code Predictor - Output norm
        "talker.code_predictor.model.norm.weight": "code_pred.output_norm.weight",
        # Code Predictor - MTP projection (talker_hidden -> code_pred_hidden)
        "talker.code_predictor.small_to_mtp_projection.weight": "code_pred.mtp_proj.weight",
        "talker.code_predictor.small_to_mtp_projection.bias": "code_pred.mtp_proj.bias",
        # Speaker Encoder - Initial conv
        "speaker_encoder.blocks.0.conv.weight": "spk_enc.conv0.weight",
        "speaker_encoder.blocks.0.conv.bias": "spk_enc.conv0.bias",
        # Speaker Encoder - ASP (Attentive Statistics Pooling)
        "speaker_encoder.asp.conv.weight": "spk_enc.asp.conv.weight",
        "speaker_encoder.asp.conv.bias": "spk_enc.asp.conv.bias",
        "speaker_encoder.asp.tdnn.conv.weight": "spk_enc.asp.tdnn.weight",
        "speaker_encoder.asp.tdnn.conv.bias": "spk_enc.asp.tdnn.bias",
        # Speaker Encoder - MFA (Multi-layer Feature Aggregation)
        "speaker_encoder.mfa.conv.weight": "spk_enc.mfa.weight",
        "speaker_encoder.mfa.conv.bias": "spk_enc.mfa.bias",
        # Speaker Encoder - Final FC
        "speaker_encoder.fc.weight": "spk_enc.fc.weight",
        "speaker_encoder.fc.bias": "spk_enc.fc.bias",
    }

    # Regex patterns for layer-specific tensors
    TALKER_LAYER_PATTERNS = [
        # Talker transformer layers (28 layers)
        (r"talker\.model\.layers\.(\d+)\.input_layernorm\.weight", "talker.blk.{}.attn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "talker.blk.{}.attn_q.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "talker.blk.{}.attn_k.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "talker.blk.{}.attn_v.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "talker.blk.{}.attn_output.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "talker.blk.{}.attn_q_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "talker.blk.{}.attn_k_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "talker.blk.{}.ffn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "talker.blk.{}.ffn_gate.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "talker.blk.{}.ffn_up.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "talker.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_LAYER_PATTERNS = [
        # Code Predictor transformer layers (5 layers)
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.input_layernorm\.weight", "code_pred.blk.{}.attn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "code_pred.blk.{}.attn_q.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "code_pred.blk.{}.attn_k.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "code_pred.blk.{}.attn_v.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "code_pred.blk.{}.attn_output.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "code_pred.blk.{}.attn_q_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "code_pred.blk.{}.attn_k_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "code_pred.blk.{}.ffn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "code_pred.blk.{}.ffn_gate.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "code_pred.blk.{}.ffn_up.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "code_pred.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_CODEBOOK_PATTERNS = [
        # Code Predictor codebook embeddings (15 codebooks, indices 0-14)
        (r"talker\.code_predictor\.model\.codec_embedding\.(\d+)\.weight", "code_pred.codec_embd.{}.weight"),
        # Code Predictor LM heads (15 heads)
        (r"talker\.code_predictor\.lm_head\.(\d+)\.weight", "code_pred.lm_head.{}.weight"),
    ]

    SPEAKER_ENCODER_PATTERNS = [
        # Speaker Encoder Res2Net blocks (blocks 1-3)
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.weight", "spk_enc.blk.{}.res2net.{}.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.bias", "spk_enc.blk.{}.res2net.{}.bias"),
        # Speaker Encoder SE blocks
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.weight", "spk_enc.blk.{}.se.conv1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.bias", "spk_enc.blk.{}.se.conv1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.weight", "spk_enc.blk.{}.se.conv2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.bias", "spk_enc.blk.{}.se.conv2.bias"),
        # Speaker Encoder TDNN layers
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.weight", "spk_enc.blk.{}.tdnn1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.bias", "spk_enc.blk.{}.tdnn1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.weight", "spk_enc.blk.{}.tdnn2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.bias", "spk_enc.blk.{}.tdnn2.bias"),
    ]

    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        output_type: str = "f16",
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type

        # Load config
        self.config = self._load_config()

        # Extract model parameters
        self._extract_params()

    def _load_config(self) -> dict[str, Any]:
        """Load model configuration from config.json."""
        config_path = self.input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_params(self) -> None:
        """Extract model parameters from config."""
        talker_config = self.config.get("talker_config", {})
        code_predictor_config = talker_config.get("code_predictor_config", {})
        speaker_encoder_config = self.config.get("speaker_encoder_config", {})

        # Talker parameters
        self.hidden_size = talker_config.get("hidden_size", 1024)
        self.intermediate_size = talker_config.get("intermediate_size", 3072)
        self.num_hidden_layers = talker_config.get("num_hidden_layers", 28)
        self.num_attention_heads = talker_config.get("num_attention_heads", 16)
        self.num_kv_heads = talker_config.get("num_key_value_heads", 8)
        self.head_dim = talker_config.get("head_dim", 128)
        self.vocab_size = talker_config.get("vocab_size", 3072)  # codec vocab
        self.text_vocab_size = talker_config.get("text_vocab_size", 151936)
        self.text_hidden_size = talker_config.get("text_hidden_size", 2048)
        self.num_code_groups = talker_config.get("num_code_groups", 16)
        self.rms_norm_eps = talker_config.get("rms_norm_eps", 1e-6)
        self.rope_theta = talker_config.get("rope_theta", 1000000)

        # M-RoPE configuration
        rope_scaling = talker_config.get("rope_scaling", {})
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

        # Code Predictor parameters
        self.code_predictor_num_layers = code_predictor_config.get("num_hidden_layers", 5)
        self.code_predictor_vocab_size = code_predictor_config.get("vocab_size", 2048)
        self.code_pred_hidden_size = code_predictor_config.get("hidden_size", 1024)
        self.code_pred_n_attention_heads = code_predictor_config.get("num_attention_heads", 16)
        self.code_pred_n_key_value_heads = code_predictor_config.get("num_key_value_heads", 8)
        self.code_pred_intermediate_size = code_predictor_config.get("intermediate_size", 3072)
        self.code_pred_head_dim = code_predictor_config.get("head_dim", 128)

        # Speaker Encoder parameters
        self.speaker_enc_dim = speaker_encoder_config.get("enc_dim", 1024)
        self.speaker_sample_rate = speaker_encoder_config.get("sample_rate", 24000)

        # Special codec token IDs
        self.codec_pad_id = talker_config.get("codec_pad_id", 2148)
        self.codec_bos_id = talker_config.get("codec_bos_id", 2149)
        self.codec_eos_id = talker_config.get("codec_eos_token_id", 2150)

        # Model name — derive from config or input directory name
        self.model_name = self.config.get("general_name",
                          self.input_dir.name.replace("-Base", "").replace("-", " ").strip()
                          or "Qwen3-TTS")

    def _map_tensor_name(self, hf_name: str) -> str | None:
        """Map HuggingFace tensor name to GGML convention."""
        # Check direct mapping first
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]

        # Check Talker layer patterns
        for pattern, template in self.TALKER_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)

        # Check Code Predictor layer patterns
        for pattern, template in self.CODE_PREDICTOR_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)

        # Check Code Predictor codebook patterns
        for pattern, template in self.CODE_PREDICTOR_CODEBOOK_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                codebook_idx = match.group(1)
                return template.format(codebook_idx)

        # Check Speaker Encoder patterns
        for pattern, template in self.SPEAKER_ENCODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return template.format(groups[0], groups[1])
                else:
                    return template.format(groups[0])

        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over all tensors from safetensors files."""
        safetensor_files = list(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir}")

        for sf_path in sorted(safetensor_files):
            logger.info(f"Loading tensors from {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def _should_quantize(self, tensor_name: str, n_dims: int) -> bool:
        """Determine if a tensor should be quantized (Q8_0) or kept in F16.

        Tensors to keep in F16:
        - 3D tensors (conv1d weights — Q8_0 doesn't support 3D)
        - Speaker encoder tensors (tiny, not worth quantizing)
        - Embeddings (codec_embd, text_embd, codebook)
        - Layer norms (attn_norm, ffn_norm, output_norm)
        - Biases
        - LM heads
        """
        # 3D tensors (conv1d) can't be quantized
        if n_dims >= 3:
            return False

        # Speaker encoder is tiny, keep in F16
        if "spk_enc" in tensor_name:
            return False

        # Keep embeddings in F16
        if any(x in tensor_name for x in ["_embd", "codebook"]):
            return False

        # Keep layer norms in F16
        if "_norm" in tensor_name:
            return False

        # Keep biases in F16
        if ".bias" in tensor_name:
            return False

        # Keep LM heads in F16
        if "lm_head" in tensor_name or "codec_head" in tensor_name:
            return False

        # Quantize weight matrices
        return True

    def _convert_dtype(self, tensor: torch.Tensor, tensor_name: str = "") -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        """Convert tensor to appropriate dtype for GGUF."""
        # Convert to numpy
        if tensor.dtype == torch.bfloat16:
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()

        n_dims = len(data.shape)

        if n_dims == 3 and "weight" in tensor_name:
            logger.info(f"Conv1d weight {tensor_name}: shape {data.shape} [OC,IC,K] - GGUF will reverse to [K,IC,OC]")

        # 1D tensors (norms, biases) should be F32
        if n_dims <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        # For 2D+ tensors, use the specified output type
        if self.output_type == "f32":
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        elif self.output_type == "f16":
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        elif self.output_type == "q8_0":
            if not self._should_quantize(tensor_name, n_dims):
                logger.debug(f"Keeping {tensor_name} in F16 (not quantizing)")
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16

            data = data.astype(np.float32)
            try:
                quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
                return quantized, gguf.GGMLQuantizationType.Q8_0
            except Exception as e:
                logger.warning(f"Q8_0 quantization failed for {tensor_name}: {e}, falling back to F16")
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        else:
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    def _load_tokenizer(self) -> tuple[list[str], list[int], list[str]]:
        """Load tokenizer vocabulary and merges."""
        vocab_path = self.input_dir / "vocab.json"
        merges_path = self.input_dir / "merges.txt"

        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        # Sort by token ID
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

        tokens = []
        toktypes = []

        for token, token_id in sorted_vocab:
            tokens.append(token)
            # Determine token type
            if token.startswith("<|") and token.endswith("|>"):
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.NORMAL)

        # Pad to text_vocab_size if needed
        while len(tokens) < self.text_vocab_size:
            tokens.append(f"[PAD{len(tokens)}]")
            toktypes.append(gguf.TokenType.UNUSED)

        # Load merges
        merges = []
        if merges_path.exists():
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        merges.append(line)

        return tokens, toktypes, merges

    def convert(self) -> None:
        """Convert the model to GGUF format."""
        logger.info(f"Converting {self.model_name} to GGUF format")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Output type: {self.output_type}")

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize GGUF writer
        arch = "qwen3-tts"
        writer = gguf.GGUFWriter(path=None, arch=arch)

        # Add metadata
        self._add_metadata(writer)

        # Add tokenizer
        self._add_tokenizer(writer)

        # Process tensors
        tensor_count = 0
        skipped_count = 0

        logger.info("Processing tensors...")
        for hf_name, tensor in tqdm(list(self._get_tensors()), desc="Converting"):
            ggml_name = self._map_tensor_name(hf_name)

            if ggml_name is None:
                logger.warning(f"Skipping unmapped tensor: {hf_name}")
                skipped_count += 1
                continue

            # Convert tensor
            data, dtype = self._convert_dtype(tensor, ggml_name)

            # Add tensor to writer
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

            logger.debug(f"  {hf_name} -> {ggml_name} [{dtype.name}] {data.shape}")

        logger.info(f"Converted {tensor_count} tensors, skipped {skipped_count}")

        # Write to file
        logger.info(f"Writing GGUF file to {self.output_path}")
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        logger.info("Conversion complete!")

    def _add_metadata(self, writer: gguf.GGUFWriter) -> None:
        """Add model metadata to GGUF writer."""
        arch = "qwen3-tts"

        # General metadata
        writer.add_name(self.model_name)
        writer.add_type(gguf.GGUFType.MODEL)

        # File type
        if self.output_type == "f32":
            ftype = gguf.LlamaFileType.ALL_F32
        elif self.output_type == "f16":
            ftype = gguf.LlamaFileType.MOSTLY_F16
        elif self.output_type == "q8_0":
            ftype = gguf.LlamaFileType.MOSTLY_Q8_0
        else:
            ftype = gguf.LlamaFileType.MOSTLY_F16
        writer.add_file_type(ftype)

        # Quantization version
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        # Talker (main architecture) parameters
        writer.add_block_count(self.num_hidden_layers)
        writer.add_embedding_length(self.hidden_size)
        writer.add_feed_forward_length(self.intermediate_size)
        writer.add_head_count(self.num_attention_heads)
        writer.add_head_count_kv(self.num_kv_heads)
        writer.add_key_length(self.head_dim)
        writer.add_value_length(self.head_dim)
        writer.add_rope_freq_base(self.rope_theta)
        writer.add_layer_norm_rms_eps(self.rms_norm_eps)
        writer.add_vocab_size(self.vocab_size)

        # TTS-specific parameters
        writer.add_uint32(f"{arch}.text_vocab_size", self.text_vocab_size)
        writer.add_uint32(f"{arch}.text_hidden_size", self.text_hidden_size)
        writer.add_uint32(f"{arch}.num_code_groups", self.num_code_groups)

        # M-RoPE configuration
        writer.add_array(f"{arch}.rope.mrope_section", self.mrope_section)

        # Code Predictor parameters
        writer.add_uint32(f"{arch}.code_predictor.layer_count", self.code_predictor_num_layers)
        writer.add_uint32(f"{arch}.code_predictor.vocab_size", self.code_predictor_vocab_size)
        writer.add_uint32(f"{arch}.code_pred.embedding_length", self.code_pred_hidden_size)
        writer.add_uint32(f"{arch}.code_pred.attention.head_count", self.code_pred_n_attention_heads)
        writer.add_uint32(f"{arch}.code_pred.attention.head_count_kv", self.code_pred_n_key_value_heads)
        writer.add_uint32(f"{arch}.code_pred.feed_forward_length", self.code_pred_intermediate_size)
        writer.add_uint32(f"{arch}.code_pred.attention.key_length", self.code_pred_head_dim)

        # Speaker Encoder parameters
        writer.add_uint32(f"{arch}.speaker_encoder.embedding_length", self.speaker_enc_dim)
        writer.add_uint32(f"{arch}.speaker_encoder.sample_rate", self.speaker_sample_rate)

        # Special codec token IDs
        writer.add_uint32(f"{arch}.codec.pad_id", self.codec_pad_id)
        writer.add_uint32(f"{arch}.codec.bos_id", self.codec_bos_id)
        writer.add_uint32(f"{arch}.codec.eos_id", self.codec_eos_id)

        logger.info("Added model metadata")

    def _add_tokenizer(self, writer: gguf.GGUFWriter) -> None:
        """Add tokenizer to GGUF writer."""
        tokens, toktypes, merges = self._load_tokenizer()

        # Tokenizer model type
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")

        # Token list
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)

        # Merges
        if merges:
            writer.add_token_merges(merges)

        # Special tokens from tokenizer_config.json
        tokenizer_config_path = self.input_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)

            # EOS token
            eos_token = tokenizer_config.get("eos_token")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")
            if eos_token:
                vocab_path = self.input_dir / "vocab.json"
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                if eos_token in vocab:
                    writer.add_eos_token_id(vocab[eos_token])

            # PAD token
            pad_token = tokenizer_config.get("pad_token")
            if isinstance(pad_token, dict):
                pad_token = pad_token.get("content")
            if pad_token:
                vocab_path = self.input_dir / "vocab.json"
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                if pad_token in vocab:
                    writer.add_pad_token_id(vocab[pad_token])

            # Chat template
            chat_template = tokenizer_config.get("chat_template")
            if chat_template:
                writer.add_chat_template(chat_template)

        logger.info(f"Added tokenizer with {len(tokens)} tokens and {len(merges)} merges")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS model to GGUF format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["f16", "f32", "q8_0"],
        default="f16",
        help="Output data type (default: f16). q8_0 provides ~50%% size reduction."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = Qwen3TTSConverter(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
    )
    converter.convert()


if __name__ == "__main__":
    main()