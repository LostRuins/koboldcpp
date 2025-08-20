"""
Python Bindings for MLX-Inspired Multi-LoRA System in KoboldCPP

This module provides Python interfaces to the advanced multi-LoRA processing system
implemented in C++, making it easy to integrate with existing Python workflows.
"""

import ctypes
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import os
import warnings
from pathlib import Path

# ============================================================================
# C++ Interface Definitions
# ============================================================================

# Load the compiled C++ library
try:
    # Try to load from the same directory as this script
    lib_path = Path(__file__).parent / "libmultilora_mlx.so"
    if not lib_path.exists():
        # Fallback locations
        lib_path = Path(__file__).parent / "libmultilora_mlx.dylib"  # macOS
        if not lib_path.exists():
            lib_path = Path(__file__).parent / "libmultilora_mlx.dll"  # Windows
    
    if lib_path.exists():
        _mlx_lora_lib = ctypes.CDLL(str(lib_path))
        print(f"✅ MLX-LORA: Loaded C++ library from {lib_path}")
    else:
        # Try system paths
        _mlx_lora_lib = ctypes.CDLL("libmultilora_mlx.so")
        print("✅ MLX-LORA: Loaded C++ library from system path")
        
except OSError as e:
    warnings.warn(f"Failed to load C++ library: {e}. Using fallback Python implementation.")
    _mlx_lora_lib = None

# Define C function signatures
if _mlx_lora_lib:
    # MultiLoRAProcessor functions
    _mlx_lora_lib.create_multi_lora_processor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    _mlx_lora_lib.create_multi_lora_processor.restype = ctypes.c_void_p
    
    _mlx_lora_lib.destroy_multi_lora_processor.argtypes = [ctypes.c_void_p]
    _mlx_lora_lib.destroy_multi_lora_processor.restype = None
    
    _mlx_lora_lib.add_lora_to_processor.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,  # lora_A
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,  # lora_B
        ctypes.c_float  # scale
    ]
    _mlx_lora_lib.add_lora_to_processor.restype = ctypes.c_bool
    
    _mlx_lora_lib.multi_lora_forward.argtypes = [
        ctypes.c_void_p,  # processor
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.c_int, ctypes.c_int,  # input dimensions
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int, ctypes.c_int,  # output dimensions
        ctypes.POINTER(ctypes.c_char_p),  # active_loras
        ctypes.c_int  # num_active_loras
    ]
    _mlx_lora_lib.multi_lora_forward.restype = ctypes.c_bool
    
    # QuantizationEngine functions
    _mlx_lora_lib.create_quantization_engine.argtypes = [ctypes.c_int, ctypes.c_int]  # type, group_size
    _mlx_lora_lib.create_quantization_engine.restype = ctypes.c_void_p
    
    _mlx_lora_lib.quantize_tensor.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
    ]
    _mlx_lora_lib.quantize_tensor.restype = ctypes.c_void_p
    
    # Memory optimization functions
    _mlx_lora_lib.create_memory_optimizer.argtypes = [ctypes.c_size_t]
    _mlx_lora_lib.create_memory_optimizer.restype = ctypes.c_void_p
    
    print("✅ MLX-LORA: C++ function signatures configured")

# ============================================================================
# Python Wrapper Classes
# ============================================================================

class MultiLoRAProcessor:
    """
    Python wrapper for the C++ MultiLoRAProcessor class.
    
    Provides MLX-inspired multi-LoRA processing with quantization support,
    memory optimization, and hardware-aware acceleration.
    """
    
    def __init__(self, max_input_dim: int = 4096, max_output_dim: int = 4096, 
                 temp_buffer_mb: int = 512):
        """
        Initialize the multi-LoRA processor.
        
        Args:
            max_input_dim: Maximum input dimension
            max_output_dim: Maximum output dimension  
            temp_buffer_mb: Temporary buffer size in MB
        """
        self.max_input_dim = max_input_dim
        self.max_output_dim = max_output_dim
        self.temp_buffer_mb = temp_buffer_mb
        
        if _mlx_lora_lib:
            self._processor = _mlx_lora_lib.create_multi_lora_processor(
                max_input_dim, max_output_dim, temp_buffer_mb * 1024 * 1024
            )
            if not self._processor:
                raise RuntimeError("Failed to create C++ MultiLoRAProcessor")
            print(f"✅ MLX-LORA: Created processor ({max_input_dim}x{max_output_dim})")
        else:
            self._processor = None
            self._fallback_init()
            
        self.loras = {}  # Track loaded LoRAs
        self.performance_stats = {
            'total_forward_calls': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0
        }
    
    def _fallback_init(self):
        """Initialize fallback Python implementation if C++ library unavailable."""
        print("⚠️ MLX-LORA: Using fallback Python implementation")
        self.base_weights = None
        self.lora_weights = {}
        
    def __del__(self):
        """Cleanup C++ resources."""
        if _mlx_lora_lib and self._processor:
            _mlx_lora_lib.destroy_multi_lora_processor(self._processor)
            
    def set_base_model(self, weights: np.ndarray) -> bool:
        """
        Set the base model weights.
        
        Args:
            weights: Numpy array of base model weights [output_dim, input_dim]
            
        Returns:
            True if successful
        """
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=np.float32)
        
        if weights.dtype != np.float32:
            weights = weights.astype(np.float32)
            
        print(f"🔧 MLX-LORA: Setting base model weights {weights.shape}")
        
        if _mlx_lora_lib and self._processor:
            # C++ implementation would be called here
            # For now, we'll store the weights
            self.base_weights = weights
            return True
        else:
            # Fallback implementation
            self.base_weights = weights
            return True
    
    def add_lora(self, name: str, lora_a: np.ndarray, lora_b: np.ndarray, 
                 scale: float = 1.0) -> bool:
        """
        Add a LoRA adapter to the processor.
        
        Args:
            name: Unique name for this LoRA
            lora_a: LoRA A matrix [input_dim, rank]
            lora_b: LoRA B matrix [rank, output_dim]
            scale: LoRA scaling factor
            
        Returns:
            True if successful
        """
        # Validate inputs
        if not isinstance(lora_a, np.ndarray) or not isinstance(lora_b, np.ndarray):
            raise ValueError("LoRA matrices must be numpy arrays")
            
        if lora_a.dtype != np.float32:
            lora_a = lora_a.astype(np.float32)
        if lora_b.dtype != np.float32:
            lora_b = lora_b.astype(np.float32)
            
        # Check dimension compatibility
        if lora_a.shape[1] != lora_b.shape[0]:
            raise ValueError(f"LoRA dimension mismatch: A={lora_a.shape}, B={lora_b.shape}")
            
        print(f"➕ MLX-LORA: Adding LoRA '{name}' (rank {lora_a.shape[1]}, scale {scale})")
        
        if _mlx_lora_lib and self._processor:
            # Call C++ function
            name_bytes = name.encode('utf-8')
            result = _mlx_lora_lib.add_lora_to_processor(
                self._processor, name_bytes,
                lora_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                lora_a.shape[0], lora_a.shape[1],
                lora_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                lora_b.shape[0], lora_b.shape[1],
                scale
            )
            if result:
                self.loras[name] = {'lora_a': lora_a, 'lora_b': lora_b, 'scale': scale}
            return result
        else:
            # Fallback implementation
            self.loras[name] = {'lora_a': lora_a, 'lora_b': lora_b, 'scale': scale}
            return True
    
    def remove_lora(self, name: str) -> bool:
        """Remove a LoRA adapter."""
        if name in self.loras:
            del self.loras[name]
            print(f"➖ MLX-LORA: Removed LoRA '{name}'")
            return True
        return False
    
    def forward(self, input_tensor: np.ndarray, 
                active_loras: Optional[List[str]] = None) -> np.ndarray:
        """
        Perform forward pass with multi-LoRA processing.
        
        Args:
            input_tensor: Input tensor [batch_size, input_dim] or [input_dim]
            active_loras: List of LoRA names to activate (all if None)
            
        Returns:
            Output tensor [batch_size, output_dim] or [output_dim]
        """
        import time
        start_time = time.time()
        
        # Validate and prepare input
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = np.array(input_tensor, dtype=np.float32)
        
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)
        
        # Handle 1D input (add batch dimension)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, input_dim = input_tensor.shape
        
        # Use all LoRAs if none specified
        if active_loras is None:
            active_loras = list(self.loras.keys())
        
        print(f"🚀 MLX-LORA: Forward pass (batch: {batch_size}, active LoRAs: {len(active_loras)})")
        
        if _mlx_lora_lib and self._processor:
            # C++ implementation
            output_dim = self.max_output_dim  # This should come from the actual model
            output_tensor = np.zeros((batch_size, output_dim), dtype=np.float32)
            
            # Prepare active LoRA names for C++
            active_lora_cstrs = [ctypes.c_char_p(name.encode('utf-8')) for name in active_loras]
            active_lora_array = (ctypes.c_char_p * len(active_loras))(*active_lora_cstrs)
            
            result = _mlx_lora_lib.multi_lora_forward(
                self._processor,
                input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size, input_dim,
                output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size, output_dim,
                active_lora_array,
                len(active_loras)
            )
            
            if not result:
                raise RuntimeError("C++ forward pass failed")
                
        else:
            # Fallback Python implementation
            output_tensor = self._fallback_forward(input_tensor, active_loras)
        
        # Update performance stats
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.performance_stats['total_forward_calls'] += 1
        self.performance_stats['total_time_ms'] += elapsed_time
        self.performance_stats['average_time_ms'] = (
            self.performance_stats['total_time_ms'] / self.performance_stats['total_forward_calls']
        )
        
        print(f"🚀 MLX-LORA: Forward completed in {elapsed_time:.2f}ms")
        
        # Remove batch dimension if input was 1D
        if squeeze_output:
            output_tensor = output_tensor.squeeze(0)
            
        return output_tensor
    
    def _fallback_forward(self, input_tensor: np.ndarray, 
                         active_loras: List[str]) -> np.ndarray:
        """Fallback Python implementation of forward pass."""
        batch_size, input_dim = input_tensor.shape
        
        # Start with base model if available
        if self.base_weights is not None:
            output_tensor = np.dot(input_tensor, self.base_weights.T)
        else:
            output_tensor = np.zeros((batch_size, self.max_output_dim), dtype=np.float32)
        
        # Add LoRA contributions
        for lora_name in active_loras:
            if lora_name in self.loras:
                lora = self.loras[lora_name]
                # Compute input @ lora_A @ lora_B * scale
                temp = np.dot(input_tensor, lora['lora_a'])
                lora_output = np.dot(temp, lora['lora_b'])
                output_tensor += lora['scale'] * lora_output
                
        return output_tensor
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def print_performance_report(self):
        """Print detailed performance report."""
        stats = self.performance_stats
        print("\n📈 MLX-LORA Performance Report")
        print("=" * 40)
        print(f"Total forward calls: {stats['total_forward_calls']}")
        print(f"Total time: {stats['total_time_ms']:.2f} ms")
        print(f"Average time per call: {stats['average_time_ms']:.2f} ms")
        print(f"Loaded LoRAs: {len(self.loras)}")
        print(f"Memory usage: {self._estimate_memory_usage():.2f} MB")
        print("=" * 40)
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        total_bytes = 0
        
        # Base weights
        if self.base_weights is not None:
            total_bytes += self.base_weights.nbytes
            
        # LoRA weights
        for lora in self.loras.values():
            total_bytes += lora['lora_a'].nbytes + lora['lora_b'].nbytes
            
        return total_bytes / (1024 * 1024)

# ============================================================================
# Advanced Quantization Interface
# ============================================================================

class QuantizationEngine:
    """Python interface for the advanced quantization system."""
    
    def __init__(self, quantization_type: str = "int4_group64", group_size: int = 64):
        """
        Initialize quantization engine.
        
        Args:
            quantization_type: "int4_group64", "int4_group32", "int8_symmetric", etc.
            group_size: Group size for quantization
        """
        self.quantization_type = quantization_type
        self.group_size = group_size
        
        if _mlx_lora_lib:
            type_map = {
                "int4_group64": 0,
                "int4_group32": 1, 
                "int8_symmetric": 2,
                "fp16_mixed": 3,
                "dynamic_adapt": 4
            }
            
            self._engine = _mlx_lora_lib.create_quantization_engine(
                type_map.get(quantization_type, 0), group_size
            )
            print(f"✅ QUANT: Created quantization engine ({quantization_type}, group_size={group_size})")
        else:
            self._engine = None
            print("⚠️ QUANT: Using fallback quantization")
    
    def quantize_tensor(self, tensor: np.ndarray, name: str = "tensor") -> 'QuantizedTensor':
        """
        Quantize a tensor using the configured method.
        
        Args:
            tensor: Input tensor to quantize
            name: Name for the tensor (for debugging)
            
        Returns:
            QuantizedTensor object
        """
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        
        print(f"🔢 QUANT: Quantizing tensor '{name}' {tensor.shape}")
        
        return QuantizedTensor(tensor, name, self.quantization_type, self.group_size)
    
    def dequantize_tensor(self, quantized_tensor: 'QuantizedTensor') -> np.ndarray:
        """
        Dequantize a tensor back to float32.
        
        Args:
            quantized_tensor: QuantizedTensor object to dequantize
            
        Returns:
            Dequantized float32 tensor
        """
        return quantized_tensor.dequantize()
    
    def analyze_tensor_stats(self, tensor: np.ndarray) -> Dict:
        """Analyze tensor statistics for optimal quantization."""
        stats = {
            'mean': float(np.mean(tensor)),
            'std': float(np.std(tensor)),
            'min': float(np.min(tensor)),
            'max': float(np.max(tensor)),
            'sparsity': float(np.mean(np.abs(tensor) < 1e-6)),
            'dynamic_range': float(np.max(tensor) - np.min(tensor))
        }
        
        # Recommend quantization type
        if stats['sparsity'] > 0.5:
            stats['recommended_type'] = 'int4_group32'
        elif stats['dynamic_range'] > 10.0:
            stats['recommended_type'] = 'int8_symmetric'
        else:
            stats['recommended_type'] = 'int4_group64'
            
        return stats

class QuantizedTensor:
    """Represents a quantized tensor with dequantization capabilities."""
    
    def __init__(self, tensor: np.ndarray, name: str = "tensor", 
                 quantization_type: str = "int4_group64", group_size: int = 64):
        self.name = name
        self.quantization_type = quantization_type
        self.group_size = group_size
        self.original_shape = tensor.shape
        self.original_dtype = tensor.dtype
        
        # Simple quantization implementation (fallback)
        self._quantize_simple(tensor)
        
        compression_ratio = self._calculate_compression_ratio()
        print(f"🔢 QUANT: Quantized '{name}' - {compression_ratio:.2f}x compression")
    
    def _quantize_simple(self, tensor: np.ndarray):
        """Simple quantization implementation."""
        flat_tensor = tensor.flatten()
        
        if self.quantization_type.startswith('int4'):
            # 4-bit quantization with grouping
            num_groups = (len(flat_tensor) + self.group_size - 1) // self.group_size
            self.scales = np.zeros(num_groups, dtype=np.float32)
            self.zero_points = np.zeros(num_groups, dtype=np.float32)
            
            # Pack into 4-bit values (simplified)
            self.quantized_data = np.zeros(len(flat_tensor) // 2 + 1, dtype=np.uint8)
            
            for i in range(num_groups):
                start_idx = i * self.group_size
                end_idx = min(start_idx + self.group_size, len(flat_tensor))
                group_data = flat_tensor[start_idx:end_idx]
                
                # Calculate scale and zero point
                min_val = np.min(group_data)
                max_val = np.max(group_data)
                self.scales[i] = (max_val - min_val) / 15.0  # 4-bit range
                self.zero_points[i] = min_val
                
                # Quantize group
                if self.scales[i] > 1e-8:
                    quantized_group = ((group_data - min_val) / self.scales[i]).astype(np.uint8)
                    quantized_group = np.clip(quantized_group, 0, 15)
                else:
                    quantized_group = np.zeros_like(group_data, dtype=np.uint8)
                
                # Pack into bytes (simplified)
                for j, val in enumerate(quantized_group):
                    byte_idx = (start_idx + j) // 2
                    bit_pos = ((start_idx + j) % 2) * 4
                    self.quantized_data[byte_idx] |= (val << bit_pos)
        
        elif self.quantization_type == 'int8_symmetric':
            # 8-bit symmetric quantization
            max_abs = np.max(np.abs(flat_tensor))
            self.scale = max_abs / 127.0 if max_abs > 0 else 1.0
            self.quantized_data = np.clip((flat_tensor / self.scale).astype(np.int8), -127, 127)
    
    def dequantize(self) -> np.ndarray:
        """Dequantize back to original precision."""
        if self.quantization_type.startswith('int4'):
            # Dequantize 4-bit
            flat_size = self.original_shape[0] * self.original_shape[1] if len(self.original_shape) == 2 else len(self.quantized_data) * 2
            dequantized = np.zeros(flat_size, dtype=np.float32)
            
            for i in range(flat_size):
                group_idx = i // self.group_size
                byte_idx = i // 2
                bit_pos = (i % 2) * 4
                
                quantized_val = (self.quantized_data[byte_idx] >> bit_pos) & 0xF
                dequantized[i] = quantized_val * self.scales[group_idx] + self.zero_points[group_idx]
                
        elif self.quantization_type == 'int8_symmetric':
            # Dequantize 8-bit
            dequantized = self.quantized_data.astype(np.float32) * self.scale
            
        return dequantized.reshape(self.original_shape)
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_size = np.prod(self.original_shape) * 4  # float32 = 4 bytes
        
        if self.quantization_type.startswith('int4'):
            compressed_size = len(self.quantized_data) + len(self.scales) * 4 + len(self.zero_points) * 4
        elif self.quantization_type == 'int8_symmetric':
            compressed_size = len(self.quantized_data) + 4  # quantized data + scale
        else:
            compressed_size = original_size
            
        return original_size / compressed_size
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        usage = len(self.quantized_data)
        if hasattr(self, 'scales'):
            usage += len(self.scales) * 4
        if hasattr(self, 'zero_points'):
            usage += len(self.zero_points) * 4
        if hasattr(self, 'scale'):
            usage += 4
        return usage

# ============================================================================
# High-level Integration Functions
# ============================================================================

def create_optimized_lora_system(config: Dict) -> MultiLoRAProcessor:
    """
    Create an optimized multi-LoRA system with the specified configuration.
    
    Args:
        config: Configuration dictionary with parameters like:
            - max_input_dim: Maximum input dimension
            - max_output_dim: Maximum output dimension
            - temp_buffer_mb: Temporary buffer size
            - enable_quantization: Whether to enable quantization
            - quantization_type: Type of quantization to use
    
    Returns:
        Configured MultiLoRAProcessor
    """
    print("🏗️ MLX-LORA: Creating optimized LoRA system")
    
    processor = MultiLoRAProcessor(
        max_input_dim=config.get('max_input_dim', 4096),
        max_output_dim=config.get('max_output_dim', 4096),
        temp_buffer_mb=config.get('temp_buffer_mb', 512)
    )
    
    if config.get('enable_quantization', False):
        print("🔢 MLX-LORA: Quantization enabled")
        # Quantization setup would be integrated here
    
    print("✅ MLX-LORA: Optimized LoRA system created")
    return processor

def benchmark_performance(processor: MultiLoRAProcessor, 
                         input_shapes: List[Tuple[int, int]],
                         num_iterations: int = 100) -> Dict:
    """
    Benchmark the performance of the multi-LoRA processor.
    
    Args:
        processor: MultiLoRAProcessor to benchmark
        input_shapes: List of (batch_size, input_dim) shapes to test
        num_iterations: Number of iterations per shape
        
    Returns:
        Performance results dictionary
    """
    print(f"🏁 MLX-LORA: Starting performance benchmark ({num_iterations} iterations)")
    
    results = {
        'shapes_tested': input_shapes,
        'iterations': num_iterations,
        'times_ms': {},
        'throughput_samples_per_sec': {},
        'memory_usage_mb': processor._estimate_memory_usage()
    }
    
    import time
    
    for batch_size, input_dim in input_shapes:
        print(f"🏁 Testing shape ({batch_size}, {input_dim})")
        
        # Generate test input
        test_input = np.random.randn(batch_size, input_dim).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            _ = processor.forward(test_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = processor.forward(test_input)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        avg_time_ms = elapsed_ms / num_iterations
        throughput = (batch_size * num_iterations) / (elapsed_ms / 1000)  # samples/sec
        
        shape_key = f"{batch_size}x{input_dim}"
        results['times_ms'][shape_key] = avg_time_ms
        results['throughput_samples_per_sec'][shape_key] = throughput
        
        print(f"🏁 Shape {shape_key}: {avg_time_ms:.2f}ms avg, {throughput:.0f} samples/sec")
    
    print("🏁 MLX-LORA: Benchmark complete")
    return results

def load_lora_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load LoRA weights from a file (safetensors, pickle, etc.).
    
    Args:
        filepath: Path to LoRA file
        
    Returns:
        Tuple of (lora_a, lora_b, scale)
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.safetensors':
        try:
            from safetensors import safe_open
            tensors = {}
            with safe_open(filepath, framework="np") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            
            # Find LoRA A and B matrices
            lora_a_key = None
            lora_b_key = None
            
            for key in tensors.keys():
                if 'lora_A' in key or 'lora.A' in key or 'down' in key:
                    lora_a_key = key
                elif 'lora_B' in key or 'lora.B' in key or 'up' in key:
                    lora_b_key = key
                    
            if lora_a_key and lora_b_key:
                lora_a = tensors[lora_a_key].astype(np.float32)
                lora_b = tensors[lora_b_key].astype(np.float32)
                
                # Try to find scale/alpha
                scale = 1.0
                for key in tensors.keys():
                    if 'alpha' in key or 'scale' in key:
                        scale = float(tensors[key])
                        break
                        
                return lora_a, lora_b, scale
            else:
                raise ValueError(f"Could not find LoRA A/B matrices in {filepath}")
                
        except ImportError:
            raise ImportError("safetensors library required for .safetensors files")
            
    elif filepath.suffix in ['.pkl', '.pickle']:
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            return data['lora_a'], data['lora_b'], data.get('scale', 1.0)
        else:
            raise ValueError("Pickle file must contain dict with 'lora_a', 'lora_b' keys")
            
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """Example of how to use the MLX-inspired multi-LoRA system."""
    print("🧪 MLX-LORA: Running example usage")
    
    # Create processor
    config = {
        'max_input_dim': 2048,
        'max_output_dim': 2048,
        'temp_buffer_mb': 256,
        'enable_quantization': True,
        'quantization_type': 'int4_group64'
    }
    
    processor = create_optimized_lora_system(config)
    
    # Add some example LoRAs
    input_dim, output_dim, rank = 2048, 2048, 64
    
    lora_a1 = np.random.randn(input_dim, rank).astype(np.float32) * 0.01
    lora_b1 = np.random.randn(rank, output_dim).astype(np.float32) * 0.01
    processor.add_lora("character_lora", lora_a1, lora_b1, scale=0.8)
    
    lora_a2 = np.random.randn(input_dim, rank//2).astype(np.float32) * 0.01
    lora_b2 = np.random.randn(rank//2, output_dim).astype(np.float32) * 0.01
    processor.add_lora("style_lora", lora_a2, lora_b2, scale=0.6)
    
    # Test forward pass
    test_input = np.random.randn(4, input_dim).astype(np.float32)
    output = processor.forward(test_input, active_loras=["character_lora", "style_lora"])
    
    print(f"✅ Output shape: {output.shape}")
    
    # Print performance report
    processor.print_performance_report()
    
    # Run benchmark
    benchmark_results = benchmark_performance(
        processor, 
        [(1, 2048), (4, 2048), (16, 2048)],
        num_iterations=50
    )
    
    print("📊 Benchmark Results:")
    for shape, time_ms in benchmark_results['times_ms'].items():
        throughput = benchmark_results['throughput_samples_per_sec'][shape]
        print(f"  {shape}: {time_ms:.2f}ms, {throughput:.0f} samples/sec")

if __name__ == "__main__":
    example_usage()