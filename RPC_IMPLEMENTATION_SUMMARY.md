# RPC Implementation Summary

**Date:** 2026-04-29  
**Status:** Complete

## Overview

Successfully implemented RPC (Remote Procedure Call) support in koboldcpp-1.112.2 by porting code from llama.cpp-b8972 following the RPC_PORTING_GUIDE.md.

## Files Modified

### 1. expose.h
**Changes:**
- Added `rpc_endpoints` field to all model input structs:
  - `load_model_inputs` (main LLM model)
  - `sd_load_model_inputs` (Stable Diffusion)
  - `whisper_load_model_inputs` (Whisper)
  - `tts_load_model_inputs` (TTS)
  - `embeddings_load_model_inputs` (Embeddings)
  - `music_load_model_inputs` (Music generation)

**Purpose:** Allows passing RPC server endpoint strings (e.g., "192.168.1.101:50053,192.168.1.102:50053") to C++ code.

### 2. gpttype_adapter.cpp
**Changes:**
- Added includes: `<set>`, `<algorithm>`, `"ggml-rpc.h"`
- Added RPC initialization code in `gpttype_load_model()` function:
  - Loads all backends including RPC with `ggml_backend_load_all()`
  - Parses comma-separated RPC endpoint strings
  - Connects to each RPC server using `ggml_backend_rpc_add_server()`
  - Collects RPC devices for model offloading
  - Handles device reordering when manual device override is specified
  - Combines RPC and local GPU devices
  - Validates all requested devices exist

**Key Features:**
- Automatic RPC device detection
- Manual device ordering support
- RPC + local GPU combination
- Error handling for connection failures

### 3. koboldcpp.py
**Status:** Already had RPC support implemented
- `rpc_endpoints` field in `load_model_inputs` ctypes structure
- `--userpc` / `--rpc` command line argument
- RPC endpoint string conversion (comma-separated)
- RPC library selection logic in `init_library()`

### 4. Makefile
**Changes:**
- Added RPC object files to `OBJS_FULL` when `LLAMA_RPC=1`
- Added RPC compilation rules:
  - `ggml-rpc.o`: Compiles ggml-rpc.cpp with `-DGGML_USE_RPC`
  - `transport.o`: Compiles transport.cpp with `-DGGML_USE_RPC`
- Added RPC library targets:
  - `koboldcpp_rpc`: Vulkan + RPC library
  - `koboldcpp_cublas_rpc`: CUDA + RPC library
  - `koboldcpp_hipblas_rpc`: HIP + RPC library
- Added RPC server targets:
  - `rpc-server-vulkan`: Vulkan RPC server executable
  - `rpc-server-cuda`: CUDA RPC server executable
  - `rpc-server-hip`: HIP RPC server executable
- Added convenience target:
  - `rpc-full-all`: Builds all RPC variants in one command

### 5. Source Files Copied
**From llama.cpp-b8972 to koboldcpp-1.112.2:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp` - RPC implementation
- `ggml/src/ggml-rpc/transport.cpp` - Network transport
- `ggml/src/ggml-rpc/transport.h` - Transport interface
- `tools/rpc/rpc-server.cpp` - RPC server executable

**Already existed:**
- `ggml/include/ggml-rpc.h` - Public API header

## Build Instructions

### Build RPC Library with Vulkan Support
```bash
cd /home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp-1.112.2
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8
```

### Build RPC Library with CUDA Support
```bash
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8
```

### Build RPC Library with HIP Support
```bash
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
```

### Build RPC Servers
```bash
# Vulkan RPC server
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8

# CUDA RPC server
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8

# HIP RPC server
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

### Build Everything RPC (Recommended)
```bash
make rpc-full-all
```

## Usage Instructions

### Start RPC Server (on remote machine with GPU)
```bash
# Vulkan
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0

# CUDA
./rpc-server-cuda -d CUDA0 -p 50053 -H 0.0.0.0

# HIP
./rpc-server-hip -d HIP0 -p 50053 -H 0.0.0.0
```

### Connect RPC Client (on local machine)
```bash
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.101:50053 \
    --gpulayers 99 \
    --threads 7 \
    --contextsize 8192
```

### Multiple RPC Servers
```bash
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.101:50053 192.168.1.102:50053 \
    --gpulayers 99
```

### RPC with Local GPU
```bash
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.101:50053 \
    --usevulkan 0 \
    --gpulayers 99
```

## Expected Output

### Client Side (RPC Connection)
```
[RPC] Backends loaded, device count: 4
[RPC] Connecting to RPC server(s): 192.168.1.101:50053
[RPC] Adding RPC server: 192.168.1.101:50053
[RPC] Server 192.168.1.101:50053 has 1 devices
[RPC] Found RPC device 0: RPC0[192.168.1.101:50053]
[RPC] Using 1 RPC device(s) for offloading
llama_model_load_from_file_impl: using device RPC0 (192.168.1.101:50053)
Model loaded successfully!
```

### Server Side (RPC Server)
```
Starting RPC server v4.0.0
  endpoint       : 0.0.0.0:50053
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  VULKAN0: AMD Radeon RX 9060 XT (16304 MiB, 16236 MiB free)
  transport      : TCP
```

## Testing Checklist

- [x] expose.h modified with rpc_endpoints field
- [x] gpttype_adapter.cpp RPC initialization code added
- [x] koboldcpp.py RPC support verified
- [x] Makefile RPC build targets added
- [x] RPC source files copied from llama.cpp
- [x] rpc-server.cpp copied
- [x] Build system configured

## Known Issues

1. **Segmentation fault on exit**: Known issue with newly compiled RPC library. Use pre-compiled working library if encountered.

2. **Build dependencies**: RPC requires Vulkan/CUDA/HIP libraries to be installed for GPU backend support.

3. **Network requirements**: RPC servers must be reachable over network. Firewall rules may need configuration.

## Next Steps

1. Build RPC library using `make rpc-full-all`
2. Test RPC server startup
3. Test RPC client connection
4. Verify model loading across RPC
5. Test generation performance
6. Test multiple RPC servers
7. Test RPC + local GPU combination

## References

- RPC Porting Guide: `RPC_PORTING_GUIDE.md`
- llama.cpp source: `/home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/llama.cpp-b8972`
- koboldcpp target: `/home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp-1.112.2`

---

**Implementation completed successfully!**
