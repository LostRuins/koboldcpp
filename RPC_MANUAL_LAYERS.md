# Manual RPC Layer Distribution

This feature allows you to manually control how LLM model layers are distributed across multiple RPC servers.

## Overview

When using multiple RPC servers, you can now specify exactly how many layers each server should handle using the `--tensor_split` parameter.

## Usage

### Basic Syntax

```bash
python koboldcpp.py --model <model.gguf> --rpc <server1> <server2> ... --tensor_split <ratio1> <ratio2> ... --gpulayers 999
```

### Example: 4 RPC Servers with Custom Distribution

For a model with 100 layers distributed across 4 servers:

```bash
python koboldcpp.py \
  --model /path/to/Qwen3.5-397B-A17B-K_G_2.93.gguf \
  --rpc 192.168.1.101:50054 192.168.1.16:50054 192.168.1.17:50054 192.168.1.18:50054 \
  --tensor_split 20 40 20 20 \
  --gpulayers 999 \
  --port 5002 \
  --contextsize 262144 \
  --quiet
```

This distributes layers as:
- Server 1 (192.168.1.101): 20% of layers
- Server 2 (192.168.1.16): 40% of layers  
- Server 3 (192.168.1.17): 20% of layers
- Server 4 (192.168.1.18): 20% of layers

### Example: 2 RPC Servers with Equal Distribution

```bash
python koboldcpp.py \
  --model /path/to/model.gguf \
  --rpc 192.168.1.101:50054 192.168.1.16:50054 \
  --tensor_split 50 50 \
  --gpulayers 999
```

### Example: 2 RPC Servers with Unequal Distribution

```bash
python koboldcpp.py \
  --model /path/to/model.gguf \
  --rpc 192.168.1.101:50054 192.168.1.16:50054 \
  --tensor_split 70 30 \
  --gpulayers 999
```

This gives 70% of layers to the first server and 30% to the second.

## Important Notes

1. **Number of Values**: The number of tensor_split values should match the number of RPC servers. If you provide fewer values, the last value will be replicated. If you provide more, excess values will be truncated.

2. **Ratios, Not Absolute Layers**: The values are ratios/proportions, not absolute layer counts. For example, `20 40 20 20` means the same as `2 4 2 2` or `10 20 10 10`.

3. **Full Offload**: Always use `--gpulayers 999` when using RPC to ensure all layers are offloaded to the RPC servers.

4. **Automatic Extension**: If you provide only one tensor_split value for multiple servers, it will be replicated:
   ```bash
   --rpc server1:50054 server2:50054 --tensor_split 50
   # Automatically becomes: --tensor_split 50 50
   ```

## Validation

The system will automatically validate and adjust tensor_split values:
- If tensor_split has fewer values than RPC servers, it extends with the last value
- If tensor_split has more values than RPC servers, it truncates to match
- A warning message will be displayed if adjustment occurs

## Troubleshooting

### Layers Not Distributing Correctly

1. Ensure you're using `--gpulayers 999`
2. Check that the number of tensor_split values matches your RPC server count
3. Look for validation messages in the console output

### Performance Issues

If you experience performance issues with manual distribution:
1. Try equal distribution first (e.g., `50 50` for 2 servers)
2. Adjust based on each server's available memory and compute power
3. More powerful servers should receive higher ratios

## Technical Details

The implementation works by:
1. Parsing RPC endpoints and tensor_split ratios
2. Calculating layer distribution based on ratios
3. Using llama.cpp's tensor_split mechanism to control device assignment
4. Setting n_gpu_layers to maximum to ensure full offload to RPC servers

## Example with Horde

You can combine manual RPC layer distribution with Horde worker:

```bash
python koboldcpp.py \
  --model /path/to/Qwen3.5-397B-A17B-K_G_2.93.gguf \
  --rpc 192.168.1.101:50054 192.168.1.16:50054 \
  --tensor_split 60 40 \
  --gpulayers 999 \
  --port 5002 \
  --contextsize 262144 \
  --quiet \
  --hordemodelname 397B-A17B \
  --hordeworkername Mandurin \
  --hordekey 03cJvq79g-GVANS653cFlw \
  --hordemaxctx 16384 \
  --hordegenlen 512
```

Note: Use `--hordegenlen` (double dash), not `-hordegenlen`.
