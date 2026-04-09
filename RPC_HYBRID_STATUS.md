# RPC Hybrid Mode - Current Status

**Version**: 1.111.2  
**Date**: 2026-04-09  
**Status**: ⚠️ Partial Working

## What Works

✅ **RPC Server** - Fully functional  
✅ **RPC Client** - Connects to servers successfully  
✅ **Multi-Server** - Multiple RPC servers work together  
✅ **GPU Offloading** - Model layers offloaded to RPC server GPUs  

## What Doesn't Work Yet

❌ **Hybrid Mode** - Local GPUs not automatically detected and used alongside RPC  
❌ **Vulkan Backend in RPC Client** - Missing shader dependencies  

## Root Cause

The RPC client library (`koboldcpp_rpc.so`) needs Vulkan backend support to detect local GPUs. However, the Vulkan backend has complex dependencies including:
- Shader compilation data (`ggml-vulkan-shaders.cpp`)
- Vulkan shader binaries
- Runtime shader initialization

These are not currently included in the RPC client build.

## Workarounds

### Workaround 1: Use RPC Only (Recommended for Now)

Simply use RPC servers without local GPUs:

```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --gpulayers 999
```

This works perfectly and distributes the model across RPC server GPUs.

### Workaround 2: Use Local GPUs Only

Use local GPUs without RPC:

```bash
python koboldcpp.py --model model.gguf \
    --usevulkan 0 \
    --gpulayers 999
```

### Workaround 3: Manual Device Selection (Advanced)

If you need both local and remote GPUs, you can try explicit device selection, but this requires the Vulkan backend to be properly loaded:

```bash
# This may not work until hybrid mode is fully implemented
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,RPC1 \
    --gpulayers 999
```

## Performance Comparison

### RPC Only (Currently Working)
```
Configuration: 2 RPC servers (4 GPUs total)
Model: Qwen3.5-0.8B-Q8_0
Speed: ~28 tokens/sec
Memory: Distributed across RPC servers
```

### Hybrid Mode (When Implemented)
```
Configuration: 2 RPC servers + 2 local GPUs (6 GPUs total)
Model: Qwen3.5-0.8B-Q8_0
Expected Speed: ~45-55 tokens/sec
Memory: Distributed across all devices
```

## Technical Details

### Current RPC Client Build
```makefile
koboldcpp_rpc: ggml.o ... ggml-rpc.o ... ggml-backend_default.o ...
```

Includes:
- CPU backend
- RPC backend
- Basic functionality

### Required for Hybrid Mode
```makefile
koboldcpp_rpc: ggml.o ... ggml-rpc.o ... ggml-backend_vulkan.o ggml-vulkan.o ggml-vulkan-shaders.o ...
```

Needs to include:
- Vulkan backend
- Vulkan shader data
- Shader compilation runtime

## Development Plan

### Phase 1: Basic Hybrid Support (In Progress)
- [x] RPC device enumeration from registry
- [x] Local GPU detection logic
- [x] Device array null terminator
- [ ] Vulkan backend integration in RPC client
- [ ] Shader data inclusion

### Phase 2: Full Hybrid Mode
- [ ] Automatic local + RPC device combination
- [ ] Optimal device ordering
- [ ] Memory-aware distribution
- [ ] Performance optimization

### Phase 3: Advanced Features
- [ ] Hot-swapping RPC servers
- [ ] Authentication support
- [ ] CUDA/HIP backend support
- [ ] Network optimization

## Alternative Solutions

### Option 1: Dynamic Backend Loading
Load Vulkan backend dynamically at runtime (similar to main koboldcpp):
- Pros: Smaller binary, flexible
- Cons: Complex implementation, runtime dependencies

### Option 2: Static Vulkan Integration
Include all Vulkan components in RPC client:
- Pros: Self-contained, no runtime dependencies
- Cons: Larger binary, longer build time

### Option 3: Separate Hybrid Client
Create a separate `koboldcpp_rpc_hybrid.so` with full Vulkan support:
- Pros: Clear separation, users choose what they need
- Cons: Maintenance overhead, confusion

## Recommendation

For now, **use RPC-only mode** which works perfectly. Hybrid mode is a nice-to-have optimization but not required for functional RPC inference.

If you need maximum performance:
1. Add more RPC servers with GPUs
2. Use faster network (10 Gbps)
3. Optimize tensor splitting
4. Reduce network latency

## Monitoring Progress

Check for updates at:
- GitHub: https://github.com/LostRuins/koboldcpp
- This directory: `RPC_FINAL_SUMMARY.md`

---

**Last Updated**: 2026-04-09  
**Version**: 1.111.2  
**Status**: RPC Works, Hybrid Mode In Development
