import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.dreambooth.lora_layers.fused_linear_lora_layer import FusedLoRALinear
from mflux.dreambooth.lora_layers.linear_lora_layer import LoRALinear
from mflux.dreambooth.lora_layers.lora_layers import LoRALayers
from mflux.weights.weight_handler import MetaData, WeightHandler


class WeightHandlerLoRA:
    def __init__(self, weight_handlers: list[WeightHandler]):
        self.weight_handlers = weight_handlers

    @staticmethod
    def load_lora_weights(
        transformer: nn.Module,
        lora_files: list[str],
        lora_scales: list[float] | None = None,
    ) -> list["WeightHandler"]:
        lora_weights = []
        if lora_files:
            lora_scales = WeightHandlerLoRA._validate_lora_scales(lora_files, lora_scales)
            for lora_file, lora_scale in zip(lora_files, lora_scales):
                weights, _, mflux_version = WeightHandler.load_transformer(lora_path=lora_file)
                weights = dict(tree_flatten(weights))
                weights = {key.removesuffix(".weight"): value for key, value in weights.items()}
                weights = {f"transformer.{key}": value for key, value in weights.items()}
                weights = {key: mx.transpose(value) for key, value in weights.items()}
                lora_transformer_dict = LoRALayers.transformer_dict_from_template(weights, transformer, lora_scale)
                transformer_weights = tree_unflatten(list(lora_transformer_dict.items()))["transformer"]
                weights = WeightHandler(
                    clip_encoder=None,
                    t5_encoder=None,
                    vae=None,
                    transformer=transformer_weights,
                    meta_data=MetaData(
                        quantization_level=None,
                        scale=lora_scale,
                        is_lora=True,
                        mflux_version=mflux_version,
                    ),
                )
                lora_weights.append(weights)

        return lora_weights

    @staticmethod
    def set_lora_weights(transformer: nn.Module, loras: list["WeightHandler"]) -> None:
        if loras:
            lora_transformer_weights = [lora.transformer for lora in loras]
            fused_weights = WeightHandlerLoRA._fuse_multiple_lora_dicts(lora_transformer_weights)
            fused_weights = WeightHandler(
                meta_data=MetaData(),
                clip_encoder=None,
                t5_encoder=None,
                vae=None,
                transformer=fused_weights,
            )

            WeightHandlerLoRA.set_lora_layers(
                transformer_module=transformer,
                lora_layers=LoRALayers(weights=fused_weights),
            )

    @staticmethod
    def _fuse_multiple_lora_dicts(dicts: list[dict]) -> dict:
        if not dicts:
            raise ValueError("No dictionaries provided for fusion.")
        if len(dicts) == 1:
            return dicts[0]

        # Collect all unique keys across all dictionaries
        all_keys = set().union(*dicts)
        fused_dict = {}

        for key in all_keys:
            # Get all values for this key, filtering out dictionaries that don't have it
            values = [d[key] for d in dicts if key in d]

            # Skip if no values found (shouldn't happen due to how we collect keys)
            if not values:
                continue

            first_value = values[0]

            # Handle nested dictionaries
            if all(isinstance(v, dict) and not isinstance(v, LoRALinear) for v in values):
                fused_dict[key] = WeightHandlerLoRA._fuse_multiple_lora_dicts(values)

            # Handle LoRALinear layers
            elif all(isinstance(v, LoRALinear) for v in values):
                fused_dict[key] = FusedLoRALinear(base_linear=first_value.linear, loras=values)

            # Handle lists
            elif all(isinstance(v, list) for v in values):
                # Get the maximum length of all lists
                max_length = max(len(v) for v in values)

                # Initialize the fused list
                fused_dict[key] = []

                # Process each index up to the maximum length
                for idx in range(max_length):
                    # Get elements at current index from lists that are long enough
                    elements = [v[idx] for v in values if idx < len(v)]

                    if not elements:
                        continue

                    # If elements are dicts or LoRALinear, recursively fuse them
                    if all(isinstance(e, (dict, LoRALinear)) for e in elements):
                        fused_element = WeightHandlerLoRA._fuse_multiple_lora_dicts([{str(idx): e} for e in elements])[str(idx)]  # fmt:off
                        fused_dict[key].append(fused_element)
                    else:
                        # For non-LoRALinear types, keep the first element
                        fused_dict[key].append(elements[0])

            else:
                types_str = ", ".join(type(v).__name__ for v in values)
                raise ValueError(f"Incompatible types for key {key}: {types_str}")

        return fused_dict

    @staticmethod
    def _validate_lora_scales(lora_files: list[str], lora_scales: list[float]) -> list[float]:
        if len(lora_files) == 1:
            if not lora_scales:
                lora_scales = [1.0]
            if len(lora_scales) > 1:
                raise ValueError("Please provide a single scale for the LoRA, or skip it to default to 1")
        elif len(lora_files) > 1:
            if len(lora_files) != len(lora_scales):
                raise ValueError("When providing multiple LoRAs, be sure to specify a scale for each one respectively")
        return lora_scales

    @staticmethod
    def set_lora_layers(transformer_module: nn.Module, lora_layers: LoRALayers) -> None:
        transformer = lora_layers.layers.transformer

        # Handle top-level transformer components (x_embedder, context_embedder, proj_out, etc.)
        for attr_name in ["x_embedder", "context_embedder", "proj_out"]:
            component = transformer.get(attr_name, None)
            if component is not None:
                setattr(transformer_module, attr_name, component)

        # Handle transformer_blocks
        transformer_blocks = transformer.get("transformer_blocks", [])
        for i, weights in enumerate(transformer_blocks):
            LoRALayers.set_transformer_block(
                transformer_block=transformer_module.transformer_blocks[i],
                dictionary=weights,
            )

        # Handle single_transformer_blocks
        single_transformer_blocks = transformer.get("single_transformer_blocks", [])
        for i, weights in enumerate(single_transformer_blocks):
            LoRALayers.set_single_transformer_block(
                single_transformer_block=transformer_module.single_transformer_blocks[i],
                dictionary=weights,
            )
