# RPC Feature Port from llama.cpp to koboldcpp

## Summary

Successfully ported the new RPC implementation from llama.cpp (version 4.0.0) to koboldcpp-1.111.2.

## Changes Made

### 1. Version Update
**File:** `ggml/include/ggml-rpc.h`
- Updated RPC protocol version from 3.6.1 to 4.0.0
- `RPC_PROTO_MAJOR_VERSION`: 3 → 4
- `RPC_PROTO_MINOR_VERSION`: 6 → 0
- `RPC_PROTO_PATCH_VERSION`: 1 → 0

### 2. New Transport Layer
**Files Added:**
- `ggml/src/ggml-rpc/transport.h` (34 lines)
- `ggml/src/ggml-rpc/transport.cpp` (683 lines)

**Features:**
- Abstracted socket layer with pimpl pattern
- Connection capabilities negotiation (`conn_caps`)
- Support for RDMA (Remote Direct Memory Access) on Linux
- Auto-negotiation of transport features during HELLO handshake
- Cross-platform socket implementation (Windows/Linux)

### 3. Updated RPC Implementation
**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp`

**Key Changes:**
- Integrated new transport layer
- Added connection capabilities exchange in HELLO handshake
- Improved version negotiation with capability advertising
- New `negotiate_hello()` function for transport auto-negotiation
- Support for feature detection via `conn_caps` (24 bytes)

**New Message Structures:**
```cpp
struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];  // NEW
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];  // NEW
};
```

### 4. Updated RPC Server
**File:** `tools/rpc-server.cpp`

**Changes:**
- Synced with llama.cpp version
- Improved cache directory handling
- Better device detection and fallback logic
- Enhanced error messages and warnings

### 5. Build System Updates

#### CMakeLists.txt
**File:** `ggml/src/ggml-rpc/CMakeLists.txt`
- Added `transport.cpp` to build
- Added RDMA auto-detection support
- Added `GGML_RPC_RDMA` compile option

#### Makefile
**File:** `Makefile`
- Added `transport.o` build rule
- Updated `ggml-rpc.o` dependencies to include `transport.h`
- Added `transport.o` to all RPC targets:
  - `koboldcpp_rpc`
  - `koboldcpp_hipblas_rpc`
  - `koboldcpp_cublas_rpc`
  - `rpc-server-vulkan`
  - `rpc-server-cuda`
  - `rpc-server-hip`

## Technical Details

### Connection Capabilities (conn_caps)
The new transport layer introduces a 24-byte capability array that allows:
- Feature negotiation between client and server
- Transparent transport upgrades
- RDMA support detection
- Future extensibility

### RDMA Support
On Linux systems with libibverbs:
- Automatically detected during CMake configuration
- Enabled via `GGML_RPC_RDMA` option
- Provides zero-copy data transfer
- Requires kernel memlock settings (default: 8 MiB)

### Version Compatibility
- Client version: 4.0.0
- Server must support major version 4
- Minor version negotiation supported
- Patch version differences logged as warnings

## Files Modified

1. `ggml/include/ggml-rpc.h` - Version update
2. `ggml/src/ggml-rpc/ggml-rpc.cpp` - Full replacement with backend interface adjustments
3. `ggml/src/ggml-rpc/transport.h` - New file
4. `ggml/src/ggml-rpc/transport.cpp` - New file
5. `ggml/src/ggml-rpc/CMakeLists.txt` - RDMA support
6. `tools/rpc-server.cpp` - Synced with llama.cpp
7. `Makefile` - Added transport.o build rules

## Important Notes

### Backend Interface Compatibility

The llama.cpp ggml-rpc.cpp required adjustments to match koboldcpp's backend interface structures:

**Removed from `ggml_backend_buffer_i` initialization:**
- `.set_tensor_2d` (not in koboldcpp)
- `.get_tensor_2d` (not in koboldcpp)

**Kept in `ggml_backend_i` initialization:**
- All graph_plan_* fields (present in koboldcpp)
- event_record/event_wait (present in koboldcpp)
- graph_optimize (present in koboldcpp)

These adjustments ensure compatibility with koboldcpp's `ggml-backend-impl.h` structure definitions.

### Static Linking Fix

**Issue:** The RPC backend registration function (`ggml_backend_rpc_reg()`) was being optimized out by the linker when statically linked into the rpc-server binary.

**Solution:** Store the backend registration in a static variable to prevent linker optimization:

```cpp
// Explicitly register RPC backend (for static linking)
// Store in static variable to prevent linker optimization
static ggml_backend_reg_t rpc_reg = NULL;
if (!rpc_reg) {
    rpc_reg = ggml_backend_rpc_reg();
}

ggml_backend_reg_t reg = rpc_reg;
```

This fix was applied to `tools/rpc-server.cpp` and is required for all statically linked RPC server builds.

## Testing

**Verified Working:**
- ✅ rpc-server-vulkan (tested with AMD Radeon RX 9060 XT)
- ✅ Backend registration successful
- ✅ Device detection working (Vulkan0, Vulkan1, Vulkan2)
- ✅ Cache mode functional (-c flag)

**Build Commands:**
```bash
# Vulkan RPC server (working)
make rpc-server-vulkan

# CUDA RPC server (requires NVIDIA CUDA)
make rpc-server-cuda

# HIP RPC server (requires AMD ROCm/HIP)
make rpc-server-hip
```

## Build Instructions

### With Make (recommended for koboldcpp)
```bash
# Vulkan RPC (default)
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8

# CUDA RPC
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8

# HIPBLAS RPC (AMD GPUs)
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

### With CMake
```bash
mkdir build && cd build
cmake .. -DGGML_RPC=ON -DGGML_RPC_RDMA=ON  # RDMA on Linux
make rpc-server -j8
```

## Testing

After building, test the RPC server:
```bash
# Start RPC server
./rpc-server-vulkan -H 127.0.0.1 -p 50052 -t 4

# Or with CUDA
./rpc-server-cuda -H 127.0.0.1 -p 50052 -d CUDA0

# Or with AMD
./rpc-server-hip -H 127.0.0.1 -p 50052 -d HIP0
```

## Notes

- The RPC server is **NOT SECURE** for network exposure
- Always use with `127.0.0.1` unless you understand the risks
- RDMA requires kernel configuration (memlock)
- Version mismatch will be logged but connection may still work

## References

- Source: llama.cpp commit b8920
- Target: koboldcpp-1.111.2
- Date: 2026-04-24
