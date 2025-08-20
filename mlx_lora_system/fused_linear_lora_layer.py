import mlx.core as mx
from mlx import nn

from mflux.dreambooth.lora_layers.linear_lora_layer import LoRALinear


class FusedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear | nn.QuantizedLinear, loras: list[LoRALinear]):
        super().__init__()
        self.base_linear = base_linear
        self.loras = loras

    def __call__(self, x):
        base_out = self.base_linear(x)

        lora_out = mx.zeros_like(base_out)
        for lora in self.loras:
            lora_out += lora.scale * mx.matmul(mx.matmul(x, lora.lora_A), lora.lora_B)

        return base_out + lora_out
