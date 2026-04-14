#!/usr/bin/env python3
"""
Convert Qwen3-TTS-Tokenizer-12Hz model to GGUF format.

Usage:
    python scripts/convert_tokenizer_to_gguf.py \
        --input models/Qwen3-TTS-Tokenizer-12Hz \
        --output models/qwen3-tts-tokenizer-f16.gguf \
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


class Qwen3TTSTokenizerConverter:
    """Converter for Qwen3-TTS-Tokenizer-12Hz model to GGUF format."""

    # Direct tensor name mappings
    TENSOR_MAP = {
        # Encoder - downsample conv
        "encoder.downsample.conv.weight": "tok_enc.downsample.weight",
        
        # Encoder quantizer projections
        "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight": "tok_enc.vq_acoustic.input_proj.weight",
        "encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight": "tok_enc.vq_acoustic.output_proj.weight",
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight": "tok_enc.vq_semantic.input_proj.weight",
        "encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight": "tok_enc.vq_semantic.output_proj.weight",
        
        # Decoder pre-conv and output
        "decoder.pre_conv.conv.bias": "tok_dec.pre_conv.bias",
        "decoder.pre_conv.conv.weight": "tok_dec.pre_conv.weight",
        
        # Decoder pre-transformer projections
        "decoder.pre_transformer.input_proj.bias": "tok_dec.pre_tfm.input_proj.bias",
        "decoder.pre_transformer.input_proj.weight": "tok_dec.pre_tfm.input_proj.weight",
        "decoder.pre_transformer.output_proj.bias": "tok_dec.pre_tfm.output_proj.bias",
        "decoder.pre_transformer.output_proj.weight": "tok_dec.pre_tfm.output_proj.weight",
        "decoder.pre_transformer.norm.weight": "tok_dec.pre_tfm.norm.weight",
        
        # Decoder quantizer projections
        "decoder.quantizer.rvq_first.input_proj.weight": "tok_dec.vq_first.input_proj.weight",
        "decoder.quantizer.rvq_first.output_proj.weight": "tok_dec.vq_first.output_proj.weight",
        "decoder.quantizer.rvq_rest.input_proj.weight": "tok_dec.vq_rest.input_proj.weight",
        "decoder.quantizer.rvq_rest.output_proj.weight": "tok_dec.vq_rest.output_proj.weight",
        
        # Decoder initial conv (index 0)
        "decoder.decoder.0.conv.weight": "tok_dec.dec.0.conv.weight",
        "decoder.decoder.0.conv.bias": "tok_dec.dec.0.conv.bias",
        
        # Decoder final snake activation (index 5) and output conv (index 6)
        "decoder.decoder.5.alpha": "tok_dec.dec.5.snake.alpha",
        "decoder.decoder.5.beta": "tok_dec.dec.5.snake.beta",
        "decoder.decoder.6.conv.weight": "tok_dec.dec.6.conv.weight",
        "decoder.decoder.6.conv.bias": "tok_dec.dec.6.conv.bias",
    }

    # Regex patterns for layer-specific tensors
    ENCODER_PATTERNS = [
        # Encoder conv layers (various indices)
        (r"encoder\.encoder\.layers\.(\d+)\.conv\.weight", "tok_enc.conv.{}.weight"),
        (r"encoder\.encoder\.layers\.(\d+)\.conv\.bias", "tok_enc.conv.{}.bias"),
        
        # Encoder residual blocks
        (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.weight", "tok_enc.res.{}.blk.{}.weight"),
        (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.bias", "tok_enc.res.{}.blk.{}.bias"),
        
        # Encoder transformer layers
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.weight", "tok_enc.blk.{}.attn_norm.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.bias", "tok_enc.blk.{}.attn_norm.bias"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight", "tok_enc.blk.{}.ffn_norm.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.bias", "tok_enc.blk.{}.ffn_norm.bias"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight", "tok_enc.blk.{}.attn_q.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight", "tok_enc.blk.{}.attn_k.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight", "tok_enc.blk.{}.attn_v.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight", "tok_enc.blk.{}.attn_output.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale", "tok_enc.blk.{}.attn_scale"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc1\.weight", "tok_enc.blk.{}.ffn_up.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc2\.weight", "tok_enc.blk.{}.ffn_down.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale", "tok_enc.blk.{}.ffn_scale"),
        
        # Encoder acoustic quantizer codebooks (embed_sum is the actual codebook)
        (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum", "tok_enc.vq_acoustic.{}.codebook"),
        (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.cluster_usage", "tok_enc.vq_acoustic.{}.usage"),
        (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.initialized", "tok_enc.vq_acoustic.{}.initialized"),
        
        # Encoder semantic quantizer codebooks
        (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum", "tok_enc.vq_semantic.{}.codebook"),
        (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.cluster_usage", "tok_enc.vq_semantic.{}.usage"),
        (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.initialized", "tok_enc.vq_semantic.{}.initialized"),
    ]

    DECODER_PATTERNS = [
        # Decoder blocks (1-4 have residual structure)
        (r"decoder\.decoder\.(\d+)\.block\.0\.alpha", "tok_dec.dec.{}.snake.alpha"),
        (r"decoder\.decoder\.(\d+)\.block\.0\.beta", "tok_dec.dec.{}.snake.beta"),
        (r"decoder\.decoder\.(\d+)\.block\.1\.conv\.weight", "tok_dec.dec.{}.conv_t.weight"),
        (r"decoder\.decoder\.(\d+)\.block\.1\.conv\.bias", "tok_dec.dec.{}.conv_t.bias"),
        
        # Decoder residual blocks within each decoder block
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act1\.alpha", "tok_dec.dec.{}.res.{}.act1.alpha"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act1\.beta", "tok_dec.dec.{}.res.{}.act1.beta"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act2\.alpha", "tok_dec.dec.{}.res.{}.act2.alpha"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act2\.beta", "tok_dec.dec.{}.res.{}.act2.beta"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv1\.conv\.weight", "tok_dec.dec.{}.res.{}.conv1.weight"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv1\.conv\.bias", "tok_dec.dec.{}.res.{}.conv1.bias"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv2\.conv\.weight", "tok_dec.dec.{}.res.{}.conv2.weight"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv2\.conv\.bias", "tok_dec.dec.{}.res.{}.conv2.bias"),
        
        # Decoder pre-transformer layers
        (r"decoder\.pre_transformer\.layers\.(\d+)\.input_layernorm\.weight", "tok_dec.pre_tfm.blk.{}.attn_norm.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight", "tok_dec.pre_tfm.blk.{}.ffn_norm.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_q.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_k.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_v.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_output.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale", "tok_dec.pre_tfm.blk.{}.attn_scale"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.gate_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_gate.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.up_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_up.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.down_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_down.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale", "tok_dec.pre_tfm.blk.{}.ffn_scale"),
        
        # Decoder quantizer codebooks (rvq_first and rvq_rest)
        (r"decoder\.quantizer\.rvq_first\.vq\.layers\.(\d+)\._codebook\.embedding_sum", "tok_dec.vq_first.{}.codebook"),
        (r"decoder\.quantizer\.rvq_first\.vq\.layers\.(\d+)\._codebook\.cluster_usage", "tok_dec.vq_first.{}.usage"),
        (r"decoder\.quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.embedding_sum", "tok_dec.vq_rest.{}.codebook"),
        (r"decoder\.quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.cluster_usage", "tok_dec.vq_rest.{}.usage"),
        
        # Decoder upsample layers
        (r"decoder\.upsample\.(\d+)\.0\.conv\.weight", "tok_dec.upsample.{}.conv.weight"),
        (r"decoder\.upsample\.(\d+)\.0\.conv\.bias", "tok_dec.upsample.{}.conv.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.weight", "tok_dec.upsample.{}.dwconv.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.bias", "tok_dec.upsample.{}.dwconv.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.gamma", "tok_dec.upsample.{}.gamma"),
        (r"decoder\.upsample\.(\d+)\.1\.norm\.weight", "tok_dec.upsample.{}.norm.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.norm\.bias", "tok_dec.upsample.{}.norm.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.weight", "tok_dec.upsample.{}.pwconv1.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.bias", "tok_dec.upsample.{}.pwconv1.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.weight", "tok_dec.upsample.{}.pwconv2.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.bias", "tok_dec.upsample.{}.pwconv2.bias"),
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
        encoder_config = self.config.get("encoder_config", {})
        decoder_config = self.config.get("decoder_config", {})

        # Encoder parameters
        self.encoder_hidden_size = encoder_config.get("hidden_size", 512)
        self.encoder_num_layers = encoder_config.get("num_hidden_layers", 8)
        self.encoder_num_heads = encoder_config.get("num_attention_heads", 8)
        self.encoder_num_quantizers = encoder_config.get("num_quantizers", 32)
        self.encoder_valid_quantizers = self.config.get("encoder_valid_num_quantizers", 16)
        self.encoder_codebook_size = encoder_config.get("codebook_size", 2048)
        self.encoder_codebook_dim = encoder_config.get("codebook_dim", 256)

        # Decoder parameters
        self.decoder_hidden_size = decoder_config.get("hidden_size", 512)
        self.decoder_num_layers = decoder_config.get("num_hidden_layers", 8)
        self.decoder_num_heads = decoder_config.get("num_attention_heads", 16)
        self.decoder_latent_dim = decoder_config.get("latent_dim", 1024)
        self.decoder_codebook_size = decoder_config.get("codebook_size", 2048)
        self.decoder_codebook_dim = decoder_config.get("codebook_dim", 512)
        self.decoder_num_quantizers = decoder_config.get("num_quantizers", 16)
        self.decoder_semantic_codebook_size = decoder_config.get("semantic_codebook_size", 4096)

        # Audio parameters
        self.sample_rate = self.config.get("input_sample_rate", 24000)
        self.frame_rate = encoder_config.get("_frame_rate", 12.5)
        self.upsample_rates = decoder_config.get("upsample_rates", [8, 5, 4, 3])

        self.model_name = "Qwen3-TTS-Tokenizer-12Hz"

    def _map_tensor_name(self, hf_name: str) -> str | None:
        """Map HuggingFace tensor name to GGML convention."""
        # Check direct mapping first
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]

        # Check encoder patterns
        for pattern, template in self.ENCODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return template.format(groups[0])
                elif len(groups) == 2:
                    return template.format(groups[0], groups[1])
                return None

        # Check decoder patterns
        for pattern, template in self.DECODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return template.format(groups[0])
                elif len(groups) == 2:
                    return template.format(groups[0], groups[1])
                return None

        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over all tensors from safetensors files."""
        tokenizer_dir = self.input_dir / "speech_tokenizer"
        if tokenizer_dir.exists():
            safetensor_files = list(tokenizer_dir.glob("*.safetensors"))
        else:
            safetensor_files = list(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir} or {tokenizer_dir}")

        for sf_path in sorted(safetensor_files):
            logger.info(f"Loading tensors from {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def _convert_dtype(self, tensor: torch.Tensor, tensor_name: str = "") -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        """Convert tensor to appropriate dtype for GGUF."""
        if tensor.dtype == torch.bfloat16:
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()

        n_dims = len(data.shape)

        # 1D tensors (norms, biases, scales) should be F32
        if n_dims <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        # For 2D+ tensors, use the specified output type
        if self.output_type == "f32":
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        elif self.output_type == "f16":
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        elif self.output_type == "q8_0":
            # Keep some tensors in F16 for quality (codebooks, norms)
            if any(x in tensor_name for x in ["codebook", "_norm", "norm.", "scale", "alpha", "beta"]):
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

    def convert(self) -> None:
        """Convert the model to GGUF format."""
        logger.info(f"Converting {self.model_name} to GGUF format")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Output type: {self.output_type}")

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize GGUF writer
        arch = "qwen3-tts-tokenizer"
        writer = gguf.GGUFWriter(path=None, arch=arch)

        # Add metadata
        self._add_metadata(writer)

        # Process tensors
        tensor_count = 0
        skipped_count = 0
        skipped_tensors = []
        
        # Collect embedding_sum and cluster_usage pairs for codebook computation
        codebook_pairs: dict[str, dict[str, torch.Tensor]] = {}

        logger.info("Processing tensors...")
        all_tensors = list(self._get_tensors())
        
        # First pass: collect codebook pairs
        for hf_name, tensor in all_tensors:
            if "embedding_sum" in hf_name:
                base_name = hf_name.replace("embedding_sum", "")
                if base_name not in codebook_pairs:
                    codebook_pairs[base_name] = {}
                codebook_pairs[base_name]["embedding_sum"] = tensor
            elif "cluster_usage" in hf_name:
                base_name = hf_name.replace("cluster_usage", "")
                if base_name not in codebook_pairs:
                    codebook_pairs[base_name] = {}
                codebook_pairs[base_name]["cluster_usage"] = tensor
        
        for hf_name, tensor in tqdm(all_tensors, desc="Converting"):
            ggml_name = self._map_tensor_name(hf_name)

            if ggml_name is None:
                skipped_tensors.append(hf_name)
                skipped_count += 1
                continue
            
            # Skip cluster_usage tensors (we'll use them to compute codebooks)
            if "cluster_usage" in hf_name:
                skipped_count += 1
                continue
            
            # For embedding_sum, compute actual codebook = embedding_sum / cluster_usage
            if "embedding_sum" in hf_name:
                base_name = hf_name.replace("embedding_sum", "")
                if base_name in codebook_pairs and "cluster_usage" in codebook_pairs[base_name]:
                    embedding_sum = codebook_pairs[base_name]["embedding_sum"]
                    cluster_usage = codebook_pairs[base_name]["cluster_usage"]
                    tensor = embedding_sum / cluster_usage.clamp(min=1e-5).unsqueeze(1)
                    logger.debug(f"  Computing codebook from embedding_sum/cluster_usage for {hf_name}")

            # Convert tensor
            data, dtype = self._convert_dtype(tensor, ggml_name)

            # Add tensor to writer
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

            logger.debug(f"  {hf_name} -> {ggml_name} [{dtype.name}] {data.shape}")

        logger.info(f"Converted {tensor_count} tensors, skipped {skipped_count}")
        if skipped_tensors and logger.level <= logging.DEBUG:
            logger.debug("Skipped tensors:")
            for t in skipped_tensors:
                logger.debug(f"  {t}")

        # Write to file
        logger.info(f"Writing GGUF file to {self.output_path}")
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        logger.info("Conversion complete!")

    def _add_metadata(self, writer: gguf.GGUFWriter) -> None:
        """Add model metadata to GGUF writer."""
        arch = "qwen3-tts-tokenizer"
        
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

        # Tokenizer-specific hyperparameters
        writer.add_uint32(f"{arch}.num_codebooks", self.decoder_num_quantizers)
        writer.add_uint32(f"{arch}.codebook_size", self.encoder_codebook_size)
        writer.add_uint32(f"{arch}.sample_rate", self.sample_rate)
        writer.add_float32(f"{arch}.frame_rate", self.frame_rate)
        
        # Encoder parameters
        writer.add_uint32(f"{arch}.encoder.hidden_size", self.encoder_hidden_size)
        writer.add_uint32(f"{arch}.encoder.num_layers", self.encoder_num_layers)
        writer.add_uint32(f"{arch}.encoder.num_heads", self.encoder_num_heads)
        writer.add_uint32(f"{arch}.encoder.num_quantizers", self.encoder_num_quantizers)
        writer.add_uint32(f"{arch}.encoder.valid_quantizers", self.encoder_valid_quantizers)
        writer.add_uint32(f"{arch}.encoder.codebook_dim", self.encoder_codebook_dim)
        
        # Decoder parameters
        writer.add_uint32(f"{arch}.decoder.hidden_size", self.decoder_hidden_size)
        writer.add_uint32(f"{arch}.decoder.num_layers", self.decoder_num_layers)
        writer.add_uint32(f"{arch}.decoder.num_heads", self.decoder_num_heads)
        writer.add_uint32(f"{arch}.decoder.latent_dim", self.decoder_latent_dim)
        writer.add_uint32(f"{arch}.decoder.codebook_dim", self.decoder_codebook_dim)
        writer.add_uint32(f"{arch}.decoder.semantic_codebook_size", self.decoder_semantic_codebook_size)
        
        # Upsample rates as array
        writer.add_array(f"{arch}.upsample_rates", self.upsample_rates)

        logger.info("Added model metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS-Tokenizer-12Hz model to GGUF format"
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
        help="Output data type (default: f16)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = Qwen3TTSTokenizerConverter(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
    )
    converter.convert()


if __name__ == "__main__":
    main()
