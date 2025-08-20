from pathlib import Path
from typing import TYPE_CHECKING

import mlx
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.dreambooth.lora_layers.linear_lora_layer import LoRALinear
from mflux.dreambooth.state.training_spec import SingleTransformerBlocks, TrainingSpec, TransformerBlocks
from mflux.dreambooth.state.zip_util import ZipUtil
from mflux.models.transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.transformer.single_transformer_block import SingleTransformerBlock
from mflux.utils.version_util import VersionUtil
from mflux.weights.weight_handler import MetaData, WeightHandler

if TYPE_CHECKING:
    from mflux.flux.flux import Flux1


class LoRALayers:
    def __init__(self, weights: "WeightHandler"):
        self.layers = weights

    @staticmethod
    def from_spec(flux: "Flux1", training_spec: TrainingSpec) -> "LoRALayers":
        if training_spec.lora_layers.state_path is not None:
            # Load from state if present in the spec
            from mflux.weights.weight_handler_lora import WeightHandlerLoRA

            weights = ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.lora_layers.state_path,
                loader=lambda x: WeightHandlerLoRA.load_lora_weights(
                    transformer=flux.transformer, lora_files=[x], lora_scales=[1.0]
                ),
            )
            return LoRALayers(weights=weights[0])
        else:
            # Construct the LoRA weights from the spec
            transformer_lora_layers = {}
            single_transformer_lora_layers = {}

            if training_spec.lora_layers.transformer_blocks:
                transformer_lora_layers = LoRALayers._construct_layers(
                    blocks=flux.transformer.transformer_blocks,
                    block_spec=training_spec.lora_layers.transformer_blocks,
                    block_prefix="transformer.transformer_blocks",
                )

            if training_spec.lora_layers.single_transformer_blocks:
                single_transformer_lora_layers = LoRALayers._construct_layers(
                    blocks=flux.transformer.single_transformer_blocks,
                    block_spec=training_spec.lora_layers.single_transformer_blocks,
                    block_prefix="transformer.single_transformer_blocks",
                )

            lora_layers = {**transformer_lora_layers, **single_transformer_lora_layers}

            weights = WeightHandler(
                meta_data=MetaData(mflux_version=VersionUtil.get_mflux_version()),
                transformer=mlx.utils.tree_unflatten(list(lora_layers.items()))["transformer"],
            )

            return LoRALayers(weights=weights)

    @staticmethod
    def _construct_layers(
        block_spec: TransformerBlocks | SingleTransformerBlocks,
        blocks: list[JointTransformerBlock] | list[SingleTransformerBlock],
        block_prefix: str,
    ) -> dict:
        block_indices = block_spec.block_range.get_blocks()
        lora_layers = {}
        for idx in block_indices:
            if idx >= len(blocks):
                raise IndexError(f"Index {idx} over range")

            block = blocks[idx]
            for layer_type in block_spec.layer_types:
                original_layer = LoRALayers._get_nested_attr(block, layer_type)
                is_list = isinstance(original_layer, list)

                lora_layer = LoRALinear.from_linear(
                    linear=original_layer[0] if is_list else original_layer,
                    r=block_spec.lora_rank,
                )
                layer_path = f"{block_prefix}.{idx}.{layer_type}"

                lora_layers[layer_path] = [lora_layer] if is_list else lora_layer

        return lora_layers

    @staticmethod
    def transformer_dict_from_template(weights: dict, transformer: nn.Module, scale: float) -> dict:
        lora_layers = {}
        for key in weights.keys():
            if key.endswith(".lora_A"):
                base_path = key[: -len(".lora_A")]
                parts = base_path.split(".")

                if parts[1] == "transformer_blocks":
                    LoRALayers._handle_transformer_blocks(
                        weights=weights,
                        scale=scale,
                        transformer=transformer,
                        lora_layers=lora_layers,
                        base_path=base_path,
                    )

                elif parts[1] == "single_transformer_blocks":
                    LoRALayers._handle_single_transformer_blocks(
                        weights=weights,
                        scale=scale,
                        transformer=transformer,
                        lora_layers=lora_layers,
                        base_path=base_path,
                    )

                # Handle top-level transformer components like x_embedder, context_embedder, proj_out
                elif len(parts) == 2 and parts[0] == "transformer":
                    LoRALayers._handle_top_level_component(
                        weights=weights,
                        scale=scale,
                        transformer=transformer,
                        lora_layers=lora_layers,
                        base_path=base_path,
                        component_name=parts[1],
                    )

        return lora_layers

    @staticmethod
    def _resolve_legacy_paths(module, module_name: str, attr_name: str, parts: list, block_idx: int) -> tuple:
        # Handle legacy naming: ff.net.2 -> ff.linear2
        if module_name == "ff" and attr_name == "net" and len(parts) >= 6 and parts[5] == "2":
            return getattr(module, "linear2"), f"transformer.transformer_blocks.{block_idx}.ff.linear2"

        # Handle attn.to_out.0 -> attn.to_out[0] (list access)
        if module_name == "attn" and attr_name == "to_out" and len(parts) >= 6 and parts[5] == "0":
            return module.to_out[0], f"transformer.transformer_blocks.{block_idx}.attn.to_out"

        # Default case - get the attribute and check if it's actually a list
        original_layer = getattr(module, attr_name)

        # Some layers (like attn.to_out) are legitimately lists in the transformer architecture
        # In such cases, we need to extract the first element for LoRA creation
        if isinstance(original_layer, list):
            # For list layers, we use the first element for LoRA creation
            # but we need to return the list itself for proper storage
            return original_layer, ".".join(parts)

        return original_layer, ".".join(parts)

    @staticmethod
    def _create_lora_layer(weights: dict, base_path: str, original_layer, scale: float):
        lora_A = weights[f"{base_path}.lora_A"]
        rank = lora_A.shape[1]

        # Handle the case where original_layer is a list (like attn.to_out)
        # In such cases, use the first element for LoRA creation (same as construction code)
        is_list = isinstance(original_layer, list)
        layer_for_lora = original_layer[0] if is_list else original_layer

        lora_layer = LoRALinear.from_linear(linear=layer_for_lora, r=rank, scale=scale)
        lora_layer.lora_A = lora_A
        lora_layer.lora_B = weights[f"{base_path}.lora_B"]
        return lora_layer

    @staticmethod
    def _handle_transformer_blocks(
        weights: dict, scale: float, transformer: nn.Module, lora_layers: dict, base_path: str
    ):
        parts = base_path.split(".")
        block_idx = int(parts[2])
        module_name = parts[3]
        attr_name = parts[4]
        block = transformer.transformer_blocks[block_idx]
        module = getattr(block, module_name)

        # Resolve legacy naming and get the actual layer and storage path
        original_layer, storage_path = LoRALayers._resolve_legacy_paths(
            module=module,
            module_name=module_name,
            attr_name=attr_name,
            parts=parts,
            block_idx=block_idx,
        )

        # Create and store LoRA layer
        lora_layer = LoRALayers._create_lora_layer(weights, base_path, original_layer, scale)

        # Store as list if original layer was a list (matching construction logic)
        is_list = isinstance(original_layer, list)
        lora_layers[storage_path] = [lora_layer] if is_list else lora_layer

    @staticmethod
    def _handle_single_transformer_blocks(
        weights: dict, scale: float, transformer: nn.Module, lora_layers: dict, base_path: str
    ):
        parts = base_path.split(".")
        block_idx = int(parts[2])
        module_name = parts[3]

        if len(parts) == 4:
            original_layer = getattr(transformer.single_transformer_blocks[block_idx], module_name)
        else:
            attr_name = parts[4]
            block = transformer.single_transformer_blocks[block_idx]
            module = getattr(block, module_name)
            original_layer = getattr(module, attr_name)

        # Create and store LoRA layer
        lora_layer = LoRALayers._create_lora_layer(weights, base_path, original_layer, scale)

        # Store as list if original layer was a list (matching construction logic)
        is_list = isinstance(original_layer, list)
        lora_layers[base_path] = [lora_layer] if is_list else lora_layer

    @staticmethod
    def _handle_top_level_component(
        weights: dict, scale: float, transformer: nn.Module, lora_layers: dict, base_path: str, component_name: str
    ):
        original_layer = getattr(transformer, component_name)
        lora_layer = LoRALayers._create_lora_layer(weights, base_path, original_layer, scale)

        # Store as list if original layer was a list (matching construction logic)
        is_list = isinstance(original_layer, list)
        lora_layers[base_path] = [lora_layer] if is_list else lora_layer

    @staticmethod
    def set_transformer_block(transformer_block, dictionary: dict):
        for key, val in dictionary.items():
            if key == "attn":
                LoRALayers._set_attribute(transformer_block, key, val, "to_q")
                LoRALayers._set_attribute(transformer_block, key, val, "to_k")
                LoRALayers._set_attribute(transformer_block, key, val, "to_v")
                LoRALayers._set_attribute(transformer_block, key, val, "to_out")
                LoRALayers._set_attribute(transformer_block, key, val, "add_q_proj")
                LoRALayers._set_attribute(transformer_block, key, val, "add_k_proj")
                LoRALayers._set_attribute(transformer_block, key, val, "add_v_proj")
                LoRALayers._set_attribute(transformer_block, key, val, "to_add_out")
            elif key == "ff" or key == "ff_context":
                LoRALayers._set_attribute(transformer_block, key, val, "linear1")
                LoRALayers._set_attribute(transformer_block, key, val, "linear2")
            elif key == "norm1" or key == "norm1_context":
                LoRALayers._set_attribute(transformer_block, key, val, "linear")
            else:
                raise Exception("Could not set LoRA weights")

    @staticmethod
    def set_single_transformer_block(single_transformer_block, dictionary: dict):
        for key, val in dictionary.items():
            if key == "attn":
                LoRALayers._set_attribute(single_transformer_block, key, val, "to_q")
                LoRALayers._set_attribute(single_transformer_block, key, val, "to_k")
                LoRALayers._set_attribute(single_transformer_block, key, val, "to_v")
            elif key == "norm":
                LoRALayers._set_attribute(single_transformer_block, key, val, "linear")
            elif key == "proj_mlp" or key == "proj_out":
                single_transformer_block[key] = val
            else:
                raise Exception("Could not set LoRA weights")

    @staticmethod
    def _set_attribute(block, key: str, val: dict, name: str):
        if block[key].get(name, False) and val.get(name, False):
            if name == "to_out" and isinstance(block[key][name], list):
                if len(block[key][name]) > 0:
                    lora_layer = val[name][0] if isinstance(val[name], list) else val[name]
                    block[key][name][0] = lora_layer
            else:
                block[key][name] = val[name]

    @staticmethod
    def _get_nested_attr(obj, attr_path):
        attrs = attr_path.split(".")
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj

    def save(self, path: Path, training_spec: TrainingSpec) -> None:
        weights = {}
        for entry in tree_flatten(self.layers.transformer):
            name = entry[0]
            weight = entry[1]
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                weights[name] = weight

        weights = {key: mx.transpose(val) for key, val in weights.items()}
        weights = {"transformer": weights}
        mx.save_safetensors(
            str(path),
            dict(tree_flatten(weights)),
            metadata={
                "mflux_version": VersionUtil.get_mflux_version(),
                "transformer_blocks": str(training_spec.lora_layers.transformer_blocks),
                "single_transformer_blocks": str(training_spec.lora_layers.single_transformer_blocks),
            },
        )
