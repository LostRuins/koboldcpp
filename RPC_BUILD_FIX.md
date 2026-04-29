# RPC Build Fix Notes

**Date:** 2026-04-29  
**Issue:** Compilation errors fixed

## Problem Encountered

During initial build, the following compilation errors occurred:

```
gpttype_adapter.cpp:2485:17: error: 'devices_override' was not declared in this scope
gpttype_adapter.cpp:2493:12: error: 'dev_override_str' was not declared in this scope
```

## Root Cause

The variables `devices_override` and `dev_override_str` were being used before they were declared in the code.

## Solution

Added variable declarations at the beginning of the RPC initialization section in `gpttype_adapter.cpp`:

```cpp
//set device overrides if needed
std::vector<ggml_backend_dev_t> devices_override;
std::string dev_override_str = inputs.devices_override ? inputs.devices_override : "";
```

This was added BEFORE the RPC backend loading code, ensuring the variables are in scope when used.

## File Modified

- **File:** `gpttype_adapter.cpp`
- **Line:** ~2434
- **Change:** Added variable declarations before RPC initialization

## Successful Build Command

```bash
cd /home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp-1.112.2

# Generate vulkan shaders first (required dependency)
make vulkan-shaders-gen
./vulkan-shaders-gen --glslc ./glslc-linux --input-dir ggml/src/ggml-vulkan/vulkan-shaders --target-hpp ggml/src/ggml-vulkan-shaders.hpp --target-cpp ggml/src/ggml-vulkan-shaders.cpp --output-dir vulkan-spv-tmp

# Build RPC library
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j4
```

## Build Result

✅ **SUCCESS**

```
-rwxr-xr-x 1 lunarbuntu lunarbuntu 72M 29. Apr 21:26 koboldcpp_rpc.so
```

The RPC library was successfully compiled and linked:
- Size: 72 MB
- Type: Shared library (.so)
- Dependencies: Vulkan, RPC backend

## All RPC Build Commands

### Build RPC Library (Vulkan)
```bash
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j4
```

### Build RPC Library (CUDA)
```bash
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j4
```

### Build RPC Library (HIP)
```bash
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j4
```

### Build RPC Servers
```bash
# Vulkan RPC server
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j4

# CUDA RPC server
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j4

# HIP RPC server
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j4
```

### Build Everything RPC
```bash
make rpc-full-all
```

## Prerequisites

Before building RPC support, ensure:

1. **Vulkan SDK installed** (for Vulkan builds)
   ```bash
   sudo apt-get install libvulkan-dev vulkan-tools
   ```

2. **CUDA toolkit** (for CUDA builds)
   ```bash
   # Follow NVIDIA's installation guide
   ```

3. **ROCm/HIP** (for HIP builds)
   ```bash
   sudo apt install rocm-dev
   ```

4. **Vulkan shaders generator** (automatically built by Makefile)
   ```bash
   make vulkan-shaders-gen
   ```

## Verification

After successful build, verify the artifacts:

```bash
# Check RPC library exists
ls -lh koboldcpp_rpc.so

# Check RPC servers exist
ls -lh rpc-server-*

# Verify library symbols (optional)
nm -D koboldcpp_rpc.so | grep rpc
```

## Next Steps

1. ✅ Build RPC library - **COMPLETED**
2. ⏳ Test RPC server startup
3. ⏳ Test RPC client connection
4. ⏳ Test model loading
5. ⏳ Test generation

## Quick Start Testing

```bash
# Start RPC server (on remote GPU machine)
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0

# Connect client (on local machine)
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50053 --gpulayers 99
```

---

**Status:** ✅ Build Successful  
**Fixed Issues:** Variable scope errors  
**Ready for Testing:** YES
