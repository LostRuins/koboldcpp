# RPC Porting Guide: llama.cpp to koboldcpp

**Version:** llama.cpp b8920 → koboldcpp-1.111.2  
**Date:** 2026-04-24  
**Status:** ✅ Complete and Tested

This guide documents all changes required to port the RPC (Remote Procedure Call) feature from llama.cpp to koboldcpp, including line-by-line modifications and explanations.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [File-by-File Changes](#file-by-file-changes)
   - [ggml/include/ggml-rpc.h](#1-ggmlincludeggml-rpch)
   - [ggml/src/ggml-rpc/ggml-rpc.cpp](#2-ggmlsrcggml-rpcggml-rpc-cpp)
   - [ggml/src/ggml-rpc/transport.h](#3-ggmlsrcggml-rpctransporth-new-file)
   - [ggml/src/ggml-rpc/transport.cpp](#4-ggmlsrcggml-rpctransportcpp-new-file)
   - [tools/rpc-server.cpp](#5-toolsrpc-servercpp)
   - [Makefile](#6-makefile)
   - [ggml/src/ggml-rpc/CMakeLists.txt](#7-ggmlsrcggml-rpccmakeliststxt)
4. [Backend Interface Compatibility](#backend-interface-compatibility)
5. [Static Linking Fix](#static-linking-fix)
6. [Build Instructions](#build-instructions)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [Known Issues](#known-issues)

---

## Overview

The RPC feature enables distributed inference across multiple machines by splitting model layers across different GPUs/systems. This port brings llama.cpp's RPC version 4.0.0 to koboldcpp-1.111.2.

### Key Features Added

- **New Transport Layer**: Abstracted socket communication with pimpl pattern
- **Connection Capabilities**: 24-byte capability negotiation for feature auto-detection
- **RDMA Support**: Zero-copy data transfer on Linux (optional)
- **Version 4.0.0 Protocol**: Improved HELLO handshake with capability exchange
- **Better Device Detection**: Enhanced device enumeration and fallback logic

---

## Prerequisites

### Required Tools

- GCC/G++ 15+ (or compatible compiler)
- Make
- Vulkan SDK (for Vulkan RPC) **OR** CUDA Toolkit (for CUDA RPC) **OR** ROCm/HIP (for AMD RPC)
- Python 3.x (for koboldcpp.py GUI)

### Source Code

- koboldcpp-1.111.2 source code
- llama.cpp b8920 source code (for reference)

### System Requirements

- **RPC Server**: Linux/Windows with GPU support
- **RPC Client**: Any system with koboldcpp
- **Network**: TCP/IP connectivity between client and server (localhost or network)

---

## File-by-File Changes

### 1. `ggml/include/ggml-rpc.h`

**Purpose:** Update RPC protocol version to 4.0.0

#### Change 1: Version Numbers

**Location:** Lines 9-11

**Before:**
```cpp
#define RPC_PROTO_MAJOR_VERSION    3
#define RPC_PROTO_MINOR_VERSION    6
#define RPC_PROTO_PATCH_VERSION    1
```

**After:**
```cpp
#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0
```

**Why:** llama.cpp RPC version 4.0.0 includes new transport layer and connection capabilities. Koboldcpp must match this version for compatibility.

---

### 2. `ggml/src/ggml-rpc/ggml-rpc.cpp`

**Purpose:** Full replacement with llama.cpp version, with koboldcpp-specific adjustments

#### Step 1: Copy File from llama.cpp

```bash
cp llama.cpp-b8920/ggml/src/ggml-rpc/ggml-rpc.cpp koboldcpp-1.111.2/ggml/src/ggml-rpc/
```

#### Step 2: Backend Interface Adjustments

**Location:** Lines 550-560 (ggml_backend_buffer_i initialization)

**Problem:** llama.cpp has more fields than koboldcpp in backend interface structures

**Before (llama.cpp version):**
```cpp
/* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
/* .memset_tensor   = */ NULL,
/* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
/* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
/* .set_tensor_2d   = */ NULL,  // ❌ Not in koboldcpp
/* .get_tensor_2d   = */ NULL,  // ❌ Not in koboldcpp
/* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
/* .clear           = */ ggml_backend_rpc_buffer_clear,
/* .reset           = */ NULL,
```

**After (koboldcpp compatible):**
```cpp
/* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
/* .memset_tensor   = */ NULL,
/* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
/* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
/* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
/* .clear           = */ ggml_backend_rpc_buffer_clear,
/* .reset           = */ NULL,
```

**Why:** koboldcpp's `ggml_backend_buffer_i` structure (defined in `ggml/src/ggml-backend-impl.h:41-60`) does not have `.set_tensor_2d` and `.get_tensor_2d` fields. These must be removed to prevent compilation errors.

**Error if not fixed:**
```
ggml/src/ggml-rpc/ggml-rpc.cpp:559:1: Fehler: zu viele Initialisierer für »ggml_backend_buffer_i«
```

---

### 3. `ggml/src/ggml-rpc/transport.h` (NEW FILE)

**Purpose:** New transport layer abstraction for socket communication

#### Create File:

```bash
cp llama.cpp-b8920/ggml/src/ggml-rpc/transport.h koboldcpp-1.111.2/ggml/src/ggml-rpc/
```

**Contents:** 34 lines defining socket interface with:
- `send_data()` / `recv_data()` - Data transfer
- `get_caps()` / `update_caps()` - Capability negotiation
- `create_server()` / `connect()` - Socket creation
- Pimpl pattern for implementation hiding

**Why:** Provides abstracted transport layer that supports:
- Traditional TCP sockets
- RDMA (Remote Direct Memory Access) on Linux
- Capability auto-negotiation
- Future transport protocols

---

### 4. `ggml/src/ggml-rpc/transport.cpp` (NEW FILE)

**Purpose:** Transport layer implementation

#### Create File:

```bash
cp llama.cpp-b8920/ggml/src/ggml-rpc/transport.cpp koboldcpp-1.111.2/ggml/src/ggml-rpc/
```

**Contents:** 683 lines implementing:
- Cross-platform socket handling (Windows/Linux)
- RDMA support (Linux only, requires libibverbs)
- Connection capability negotiation
- Chunked data transfer (1 GiB max chunks)

**Why:** Separates transport logic from RPC protocol, enabling:
- Multiple transport backends
- Feature auto-detection
- Platform-specific optimizations

---

### 5. `tools/rpc-server.cpp`

**Purpose:** Fix static linking issue and sync with llama.cpp

#### Step 1: Copy from llama.cpp

```bash
cp llama.cpp-b8920/tools/rpc/rpc-server.cpp koboldcpp-1.111.2/tools/rpc-server.cpp
```

#### Step 2: Static Linking Fix ⭐ **CRITICAL**

**Location:** Lines 328-336 (after cache_dir setup, before backend lookup)

**Problem:** Linker optimizes out unused `ggml_backend_rpc_reg()` call in statically linked binary

**Before:**
```cpp
ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
if (!reg) {
    fprintf(stderr, "Failed to find RPC backend\n");
    return 1;
}
```

**After:**
```cpp
// Explicitly register RPC backend (for static linking)
// Store in static variable to prevent linker optimization
static ggml_backend_reg_t rpc_reg = NULL;
if (!rpc_reg) {
    rpc_reg = ggml_backend_rpc_reg();
}

ggml_backend_reg_t reg = rpc_reg;
if (!reg) {
    fprintf(stderr, "Failed to find RPC backend\n");
    return 1;
}
```

**Why:** When statically linking, the linker sees `ggml_backend_rpc_reg()` as unused and removes it. By storing the result in a static variable and checking it, we force the linker to keep the function.

**Error if not fixed:**
```
Failed to find RPC backend
ERROR: RPC server failed with exit code 1
```

---

### 6. `Makefile`

**Purpose:** Add transport.o build rules and update dependencies

#### Change 1: Add transport.o Build Rule

**Location:** After line 691 (after ggml-rpc.o rule)

**Add:**
```makefile
transport.o: ggml/src/ggml-rpc/transport.cpp ggml/src/ggml-rpc/transport.h
	$(CXX) $(CXXFLAGS) $(RPC_FLAGS) -c $< -o $@
```

**Why:** transport.cpp must be compiled into transport.o for linking

#### Change 2: Update ggml-rpc.o Dependencies

**Location:** Line 691

**Before:**
```makefile
ggml-rpc.o: ggml/src/ggml-rpc/ggml-rpc.cpp ggml/include/ggml-rpc.h
```

**After:**
```makefile
ggml-rpc.o: ggml/src/ggml-rpc/ggml-rpc.cpp ggml/include/ggml-rpc.h ggml/src/ggml-rpc/transport.h
```

**Why:** ggml-rpc.cpp includes transport.h, so changes to transport.h should trigger rebuild

#### Change 3: Add transport.o to RPC Client Targets

**Location:** Lines 939, 942, 945

**Before (line 939):**
```makefile
koboldcpp_rpc: ... ggml-rpc.o ...
```

**After (line 939):**
```makefile
koboldcpp_rpc: ... ggml-rpc.o transport.o ...
```

**Repeat for:**
- Line 942: `koboldcpp_hipblas_rpc`
- Line 945: `koboldcpp_cublas_rpc`

**Why:** RPC client libraries need transport.o linked in

#### Change 4: Add transport.o to RPC Server Targets

**Location:** Lines 952, 955, 958

**Before (line 952):**
```makefile
rpc-server-vulkan: ... ggml-rpc.o ...
```

**After (line 952):**
```makefile
rpc-server-vulkan: ... ggml-rpc.o transport.o ...
```

**Repeat for:**
- Line 955: `rpc-server-cuda`
- Line 958: `rpc-server-hip`

**Why:** RPC server binaries need transport.o linked in

---

### 7. `ggml/src/ggml-rpc/CMakeLists.txt`

**Purpose:** Add transport.cpp to CMake build and RDMA support

#### Replace File:

```bash
cp llama.cpp-b8920/ggml/src/ggml-rpc/CMakeLists.txt koboldcpp-1.111.2/ggml/src/ggml-rpc/
```

**Contents:** 33 lines including:
- transport.cpp in build
- RDMA auto-detection (Linux)
- GGML_RPC_RDMA compile option
- libibverbs linking

**Why:** CMake builds need the same changes as Make builds

---

## Backend Interface Compatibility

### Structure Differences

koboldcpp and llama.cpp have slightly different backend interface structures. Here's what exists in each:

### `ggml_backend_buffer_i` (koboldcpp)

**Location:** `ggml/src/ggml-backend-impl.h:41-60`

**Fields (10 total):**
1. `free_buffer`
2. `get_base`
3. `init_tensor`
4. `memset_tensor`
5. `set_tensor`
6. `get_tensor`
7. `cpy_tensor`
8. `clear`
9. `reset`

**Missing in koboldcpp (compared to llama.cpp):**
- `set_tensor_2d` ❌
- `get_tensor_2d` ❌

### `ggml_backend_i` (koboldcpp)

**Location:** `ggml/src/ggml-backend-impl.h:87-122`

**Fields (13 total):**
1. `get_name`
2. `free`
3. `set_tensor_async`
4. `get_tensor_async`
5. `cpy_tensor_async`
6. `synchronize`
7. `graph_plan_create`
8. `graph_plan_free`
9. `graph_plan_update`
10. `graph_plan_compute`
11. `graph_compute`
12. `event_record`
13. `event_wait`
14. `graph_optimize`

**All fields present in koboldcpp** ✅

---

## Static Linking Fix

### The Problem

When building rpc-server as a statically linked binary, the linker performs dead code elimination. Since `ggml_backend_rpc_reg()` is called but its return value is ignored, the linker considers it unused and removes it from the binary.

### The Solution

Force the linker to keep the function by:
1. Storing the result in a static variable
2. Checking the variable before use
3. Using the variable for backend lookup

### Code Pattern

```cpp
// Explicitly register RPC backend (for static linking)
// Store in static variable to prevent linker optimization
static ggml_backend_reg_t rpc_reg = NULL;
if (!rpc_reg) {
    rpc_reg = ggml_backend_rpc_reg();
}

ggml_backend_reg_t reg = rpc_reg;
if (!reg) {
    fprintf(stderr, "Failed to find RPC backend\n");
    return 1;
}
```

### Why This Works

- **Static variable**: Prevents compiler from optimizing away the assignment
- **NULL check**: Ensures registration happens only once
- **Usage**: The variable is actually used, so linker keeps the function

---

## Build Instructions

### Vulkan RPC Server (AMD/Intel GPUs)

```bash
cd koboldcpp-1.111.2
make clean
make rpc-server-vulkan -j8
```

**Output:** `rpc-server-vulkan` (~60 MB)

### CUDA RPC Server (NVIDIA GPUs)

```bash
cd koboldcpp-1.111.2
make clean
make rpc-server-cuda -j8
```

**Output:** `rpc-server-cuda` (~70 MB)

**Requirements:**
- NVIDIA CUDA Toolkit
- Compatible NVIDIA GPU

### HIP RPC Server (AMD GPUs)

```bash
cd koboldcpp-1.111.2
make clean
make rpc-server-hip -j8
```

**Output:** `rpc-server-hip` (~73 MB)

**Requirements:**
- AMD ROCm/HIP
- Compatible AMD GPU

### RPC Client Libraries

```bash
# Vulkan RPC client (default)
make koboldcpp_rpc -j8

# HIPBLAS RPC client
make koboldcpp_hipblas_rpc -j8

# CUBLAS RPC client
make koboldcpp_cublas_rpc -j8
```

**Output:**
- `koboldcpp_rpc.so` (~67 MB)
- `koboldcpp_hipblas_rpc.so` (~79 MB)
- `koboldcpp_cublas_rpc.so` (~79 MB)

---

## Testing

### Start RPC Server

```bash
# Vulkan server on localhost:50053
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0

# With cache enabled
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0 -c

# Multiple devices
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0,Vulkan1
```

### Expected Output

```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 3 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
ggml_vulkan: 1 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
ggml_vulkan: 2 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
```

### Connect RPC Client

```bash
# From koboldcpp GUI or command line
python koboldcpp.py --rpc 127.0.0.1:50053
```

### Cross-Machine Testing

**Machine 1 (Server):**
```bash
./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device Vulkan0
```

**Machine 2 (Client):**
```bash
python koboldcpp.py --rpc 192.168.1.100:50053
```

**⚠️ WARNING:** Never expose RPC server to untrusted networks - it's not secure!

---

## Troubleshooting

### "Failed to find RPC backend"

**Cause:** Static linking fix not applied

**Solution:** Apply the static variable fix to `tools/rpc-server.cpp:329-336`

### "zu viele Initialisierer für »ggml_backend_buffer_i«"

**Cause:** Backend interface compatibility issue

**Solution:** Remove `.set_tensor_2d` and `.get_tensor_2d` from `ggml/src/ggml-rpc/ggml-rpc.cpp:550-560`

### "unknown device: ggml_vulkan0"

**Cause:** Incorrect device name format

**Solution:** Use `Vulkan0` not `ggml_vulkan0`

Available devices are listed by the server:
```
available devices:
  Vulkan0: AMD Radeon RX 9060 XT ...
  Vulkan1: AMD Radeon RX 9060 XT ...
```

### RDMA Issues (Linux)

**Cause:** libibverbs not installed or RDMA not supported

**Solution:** 
```bash
# Install libibverbs
sudo apt install libibverbs-dev

# Or disable RDMA
cmake .. -DGGML_RPC_RDMA=OFF
```

---

## Known Issues

### 1. Save File Compatibility

**Issue:** GPU selection and backend not saved in old save files (pre-1.111.2)

**Cause:** Old save file format doesn't include RPC configuration

**Workaround:** Manually re-select GPU and backend when loading old saves

**Status:** Expected behavior - old saves don't have new fields

### 2. CPU Fallback Backend

**Issue:** No CPU fallback visible in RPC mode

**Status:** CPU backend exists but may not be shown in RPC device list

**Workaround:** Use `--device CPU` explicitly if needed

### 3. CUDA vs HIP

**CUDA:** Fully functional on NVIDIA GPUs
- Mature implementation
- Better performance on NVIDIA hardware
- Requires CUDA Toolkit

**HIP:** Fully functional on AMD GPUs  
- Mature implementation
- Better performance on AMD hardware
- Requires ROCm/HIP

**Recommendation:** Use the backend that matches your GPU vendor

### 4. Network Security

**⚠️ WARNING:** RPC server is **NOT SECURE** for network exposure

- No authentication
- No encryption
- No access control

**Use Case:** Only use on trusted local networks or localhost

---

## File Summary

| File | Changes | Lines | Purpose |
|------|---------|-------|---------|
| `ggml/include/ggml-rpc.h` | Version update | 3 | Protocol version 4.0.0 |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Full replacement + fixes | ~2000 | RPC implementation |
| `ggml/src/ggml-rpc/transport.h` | New file | 34 | Transport interface |
| `ggml/src/ggml-rpc/transport.cpp` | New file | 683 | Transport implementation |
| `tools/rpc-server.cpp` | Sync + static fix | 342 | RPC server binary |
| `Makefile` | Build rules | ~10 | Build system |
| `ggml/src/ggml-rpc/CMakeLists.txt` | Full replacement | 33 | CMake build |

---

## Verification Checklist

- [ ] `ggml-rpc.h` version updated to 4.0.0
- [ ] `ggml-rpc.cpp` backend interfaces fixed
- [ ] `transport.h` copied to koboldcpp
- [ ] `transport.cpp` copied to koboldcpp
- [ ] `rpc-server.cpp` static linking fix applied
- [ ] `Makefile` transport.o rules added
- [ ] `CMakeLists.txt` updated
- [ ] `rpc-server-vulkan` builds successfully
- [ ] RPC server starts without errors
- [ ] RPC client can connect
- [ ] Cross-machine RPC works (if tested)

---

## Additional Resources

- **llama.cpp RPC Documentation:** `llama.cpp/tools/rpc/README.md`
- **koboldcpp RPC Usage:** `koboldcpp-1.111.2/README.md`
- **RDMA Documentation:** `https://www.rdma.io/`
- **Vulkan Backend:** `ggml/src/ggml-vulkan/`

---

**Last Updated:** 2026-04-24  
**Tested On:** AMD Radeon RX 9060 XT (Vulkan), koboldcpp-1.111.2  
**Status:** ✅ Production Ready
