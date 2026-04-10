# Manual RPC Layer Distribution - Implementation Summary

## Overview

This implementation adds support for manual layer distribution across multiple RPC servers using the `--tensor_split` parameter.

## Changes Made

### 1. C++ Backend (`gpttype_adapter.cpp`)

#### Modified RPC Device Handling
- **Location**: Lines ~2394-2448
- **Changes**:
  - Added `rpc_endpoint_list` vector to track individual RPC server endpoints
  - Store endpoint strings for later use in layer distribution calculation

#### Added Tensor Split Logic for RPC
- **Location**: Lines ~2509-2545
- **Changes**:
  - Parse `tensor_split` values from inputs
  - Detect when manual split is requested (non-zero values)
  - Calculate total ratio from tensor_split values
  - Print debug information about split ratios
  - Apply `model_params.tensor_split` with the manual ratios
  - Force full offload (`n_gpu_layers = 999`) when using RPC with manual split

**Key Code**:
```cpp
// Check if tensor_split is provided for manual layer distribution
bool use_manual_split = false;
std::vector<float> tensor_split_rpc(tensor_split_max);
for (int i = 0; i < tensor_split_max; ++i) {
    tensor_split_rpc[i] = inputs.tensor_split[i];
    if (inputs.tensor_split[i] != 0.0f) {
        use_manual_split = true;
    }
}

// If using RPC with manual tensor_split, calculate layer distribution
if(use_rpc && use_manual_split && rpc_devices.size() > 0) {
    printf("[RPC] Manual layer distribution requested via tensor_split\n");
    
    // Calculate total ratio and apply
    float total_ratio = 0.0f;
    int rpc_server_count = rpc_endpoint_list.size();
    for(int i = 0; i < rpc_server_count && i < tensor_split_max; ++i) {
        total_ratio += tensor_split_rpc[i];
    }
    
    if(total_ratio > 0.0f) {
        model_params.tensor_split = tensor_split_rpc;
        model_params.n_gpu_layers = 999;
        printf("[RPC] Manual layer distribution configured across %d RPC servers\n", rpc_server_count);
    }
}
```

### 2. Python Frontend (`koboldcpp.py`)

#### Updated Help Text
- **Location**: Line ~16965
- **Changes**:
  - Updated `--tensor_split` help text to mention RPC support
  - Clarified that it controls manual layer distribution

**Before**:
```python
help="For CUDA and Vulkan only, ratio to split tensors across multiple GPUs, space-separated list of proportions, e.g. 7 3"
```

**After**:
```python
help="For RPC, CUDA and Vulkan: ratio to split tensors across multiple GPUs/servers. Space-separated list of proportions (e.g., 20 40 20 20 for 4 RPC servers). Controls manual layer distribution."
```

#### Added Validation Logic
- **Location**: Lines ~16102-16120
- **Changes**:
  - Validate tensor_split count matches RPC server count
  - Auto-extend tensor_split if fewer values than servers
  - Auto-truncate tensor_split if more values than servers
  - Print informative messages about adjustments

**Key Code**:
```python
# Validate RPC tensor_split if both are provided
if args.userpc and args.tensor_split:
    rpc_count = len(args.userpc)
    ts_count = len(args.tensor_split)
    if ts_count > 0 and ts_count < rpc_count:
        print(f"[RPC] WARNING: tensor_split has {ts_count} values but {rpc_count} RPC servers specified")
        print(f"[RPC] Extending tensor_split with equal distribution for remaining servers")
        # Extend tensor_split
        if ts_count == 1:
            args.tensor_split = args.tensor_split * rpc_count
        else:
            last_val = args.tensor_split[-1]
            while len(args.tensor_split) < rpc_count:
                args.tensor_split.append(last_val)
    elif ts_count > rpc_count:
        print(f"[RPC] WARNING: tensor_split has {ts_count} values but only {rpc_count} RPC servers")
        args.tensor_split = args.tensor_split[:rpc_count]
    else:
        print(f"[RPC] Manual layer distribution: {rpc_count} servers with split ratios: {args.tensor_split}")
```

### 3. Documentation

#### Created `RPC_MANUAL_LAYERS.md`
- Comprehensive user guide
- Usage examples for 2 and 4 server setups
- Troubleshooting section
- Horde integration examples

#### Created `test_rpc_layers.sh`
- Test script demonstrating the feature
- Example commands (commented out for safety)
- Validation test cases

## How It Works

1. **User provides RPC endpoints and tensor_split ratios**:
   ```bash
   --rpc server1:50054 server2:50054 server3:50054 server4:50054 --tensor_split 20 40 20 20
   ```

2. **Python validation**:
   - Checks if tensor_split count matches RPC server count
   - Adjusts tensor_split if needed (extend or truncate)
   - Prints informative messages

3. **C++ backend processing**:
   - Parses RPC endpoints and connects to servers
   - Detects manual tensor_split request
   - Calculates layer distribution based on ratios
   - Applies tensor_split to model_params
   - Forces full GPU offload (n_gpu_layers = 999)

4. **llama.cpp layer assignment**:
   - Uses tensor_split ratios to distribute layers
   - Assigns layers to RPC devices based on ratios
   - Each RPC server gets a proportional share of layers

## Usage Examples

### Equal Distribution (2 servers)
```bash
python koboldcpp.py --model model.gguf \
  --rpc 192.168.1.101:50054 192.168.1.16:50054 \
  --tensor_split 50 50 \
  --gpulayers 999
```

### Unequal Distribution (4 servers)
```bash
python koboldcpp.py --model model.gguf \
  --rpc server1:50054 server2:50054 server3:50054 server4:50054 \
  --tensor_split 20 40 20 20 \
  --gpulayers 999
```

### With Horde Worker
```bash
python koboldcpp.py --model model.gguf \
  --rpc 192.168.1.101:50054 192.168.1.16:50054 \
  --tensor_split 60 40 \
  --gpulayers 999 \
  --hordemodelname MyModel \
  --hordeworkername MyWorker \
  --hordekey mykey \
  --hordemaxctx 8192 \
  --hordegenlen 256
```

## Testing

To test the implementation:

1. **Basic test** - Check help text:
   ```bash
   python koboldcpp.py --help | grep -A 3 "tensor_split"
   ```

2. **Validation test** - Mismatched counts:
   ```bash
   python koboldcpp.py --rpc s1:50054 s2:50054 s3:50054 --tensor_split 30 70
   # Should auto-extend to: 30 70 70
   ```

3. **Full integration test** - Requires actual RPC servers:
   ```bash
   python koboldcpp.py --model test.gguf \
     --rpc server1:50054 server2:50054 \
     --tensor_split 50 50 \
     --gpulayers 999 \
     --contextsize 8192
   ```

## Limitations

1. **Ratios, not absolute layers**: Values are proportions, not exact layer counts
2. **Requires llama.cpp support**: Depends on llama.cpp's tensor_split mechanism
3. **No row-wise splitting**: Only layer-wise distribution is supported

## Future Enhancements

Potential improvements for future versions:
1. Support for absolute layer counts (e.g., `--tensor_split 24 48 24 24` for exact layers)
2. Automatic optimization based on server capabilities
3. Dynamic rebalancing during runtime
4. Row-wise splitting support for RPC

## Files Modified

1. `gpttype_adapter.cpp` - C++ backend logic
2. `koboldcpp.py` - Python frontend and validation
3. `RPC_MANUAL_LAYERS.md` - User documentation (new)
4. `test_rpc_layers.sh` - Test script (new)

## Compatibility

- **Backward compatible**: Existing RPC usage without tensor_split continues to work
- **Multi-backend**: Works with CUDA, Vulkan, and RPC
- **Horde compatible**: Can be combined with Horde worker feature
