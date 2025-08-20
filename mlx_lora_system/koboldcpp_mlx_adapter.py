"""
KoboldCpp MLX LoRA Adapter
Integrates mflux LoRA classes with KoboldCpp's safetensors loading
"""

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class KoboldCppMLXLoRAProcessor:
    """
    Processes LoRA files using MLX (copied from mflux) to avoid GGML crashes
    """
    
    def __init__(self):
        self.loaded_loras: List[Dict] = []
        self.base_weights: Optional[Dict] = None
        
    def load_lora_from_safetensors(self, lora_path: str, scale: float = 1.0) -> bool:
        """
        Load LoRA A/B matrices from safetensors file
        Returns True if successful, False if incompatible
        """
        try:
            print(f"🔥 MLX-LORA: Loading {Path(lora_path).name} (scale: {scale})")
            
            lora_tensors = {}
            with safe_open(lora_path, framework='pt', device='cpu') as f:
                keys = list(f.keys())
                
                # Group A/B pairs
                for key in keys:
                    if '.lora_A.weight' in key or '.lora_B.weight' in key:
                        tensor = f.get_tensor(key)
                        # Convert to MLX array
                        lora_tensors[key] = mx.array(tensor.numpy())
                        
            # Validate A/B pairs
            pairs = self._validate_lora_pairs(lora_tensors)
            
            if pairs:
                self.loaded_loras.append({
                    'path': lora_path,
                    'scale': scale,
                    'tensors': lora_tensors,
                    'pairs': pairs
                })
                print(f"✅ MLX-LORA: Loaded {len(pairs)} LoRA pairs")
                return True
            else:
                print(f"❌ MLX-LORA: No valid LoRA pairs found in {lora_path}")
                return False
                
        except Exception as e:
            print(f"❌ MLX-LORA: Failed to load {lora_path}: {e}")
            return False
    
    def _validate_lora_pairs(self, tensors: Dict) -> List[Tuple]:
        """Validate and pair LoRA A/B matrices"""
        pairs = []
        
        # Find A/B pairs
        a_keys = [k for k in tensors.keys() if '.lora_A.weight' in k]
        
        for a_key in a_keys:
            b_key = a_key.replace('.lora_A.weight', '.lora_B.weight')
            
            if b_key in tensors:
                lora_a = tensors[a_key]
                lora_b = tensors[b_key]
                
                # Validate dimensions: B @ A should work
                if lora_a.shape[0] == lora_b.shape[1]:
                    base_key = a_key.replace('.lora_A.weight', '')
                    pairs.append((base_key, lora_a, lora_b))
                    print(f"🔧 MLX-LORA: Pair {base_key}: A{lora_a.shape} @ B{lora_b.shape}")
                else:
                    print(f"⚠️ MLX-LORA: Dimension mismatch for {a_key}")
                    
        return pairs
    
    def compute_fused_weights(self, layer_name: str) -> Optional[mx.array]:
        """
        Compute fused LoRA weights for a specific layer
        This is what replaces the crashing GGML computation
        """
        try:
            fused_delta = None
            
            for lora in self.loaded_loras:
                for base_key, lora_a, lora_b in lora['pairs']:
                    if layer_name in base_key:
                        # Compute LoRA delta: scale * (B @ A)
                        delta = lora['scale'] * (lora_b @ lora_a)
                        
                        if fused_delta is None:
                            fused_delta = delta
                        else:
                            fused_delta = fused_delta + delta
                            
            return fused_delta
            
        except Exception as e:
            print(f"❌ MLX-LORA: Fusion failed for {layer_name}: {e}")
            return None
    
    def is_compatible_with_base_model(self, expected_shape: Tuple) -> bool:
        """Check if loaded LoRAs are compatible with base model shape"""
        if not self.loaded_loras:
            return True
            
        # Check first LoRA's computed shape
        for lora in self.loaded_loras:
            for base_key, lora_a, lora_b in lora['pairs']:
                computed_shape = (lora_b.shape[0], lora_a.shape[1])
                if computed_shape != expected_shape:
                    print(f"⚠️ MLX-LORA: Shape mismatch - Expected {expected_shape}, got {computed_shape}")
                    return False
                break  # Check first pair only
            break  # Check first LoRA only
            
        return True
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        total_pairs = sum(len(lora['pairs']) for lora in self.loaded_loras)
        return {
            'loaded_loras': len(self.loaded_loras),
            'total_pairs': total_pairs,
            'total_files': [lora['path'] for lora in self.loaded_loras]
        }


def test_mlx_lora_processor():
    """Test the MLX LoRA processor with the problematic file"""
    processor = KoboldCppMLXLoRAProcessor()
    
    # Test with the file that crashes KoboldCpp
    lora_path = "/Users/admin/AI/koboldcpp/character_loras/wife_lora_flux_alt.safetensors"
    
    if processor.load_lora_from_safetensors(lora_path, scale=1.0):
        stats = processor.get_stats()
        print(f"🎉 MLX-LORA: Successfully processed {stats}")
        
        # Test computing fused weights
        test_shape = (3072, 3072)
        if processor.is_compatible_with_base_model(test_shape):
            print("✅ MLX-LORA: Compatible with base model!")
        else:
            print("❌ MLX-LORA: Incompatible with base model")
            
        return True
    else:
        print("❌ MLX-LORA: Failed to process LoRA file")
        return False


if __name__ == "__main__":
    test_mlx_lora_processor()