# RPC Feature Documentation Index

**Last Updated:** 2026-04-28  
**RPC Protocol Version:** 4.0.0

---

## Quick Start

### For Users
See: **RPC_MANUAL.md** - Complete user guide for using RPC inference

### For Developers
See: **RPC_TRANSFER_ANALYSIS.md** - Comprehensive technical analysis

### For Porting
See: **RPC_PORTING_LLAMACPP_TO_KOBOLDCPP.md** - Porting guide from llama.cpp

---

## Documentation Files

### Core Documentation (Original)

| File | Size | Purpose |
|------|------|---------|
| [RPC_MANUAL.md](RPC_MANUAL.md) | 16 KB | User guide for RPC server and client usage |
| [RPC_PORTING_LLAMACPP_TO_KOBOLDCPP.md](RPC_PORTING_LLAMACPP_TO_KOBOLDCPP.md) | 17 KB | Guide for porting RPC from llama.cpp |
| [RPC_PORT_SUMMARY.md](RPC_PORT_SUMMARY.md) | 6 KB | Technical summary of RPC implementation |

### Analysis Documentation (New - 2026-04-28)

| File | Size | Purpose |
|------|------|---------|
| [RPC_TRANSFER_ANALYSIS.md](RPC_TRANSFER_ANALYSIS.md) | 16 KB | **Comprehensive analysis** of RPC transfer from llama.cpp |
| [RPC_TRANSFER_SUMMARY.txt](RPC_TRANSFER_SUMMARY.txt) | 6 KB | **Quick summary** of transfer status and key points |
| [RPC_CODE_DIFFERENCES.md](RPC_CODE_DIFFERENCES.md) | 6 KB | **Line-by-line code differences** between implementations |

---

## Key Findings

### ✅ Transfer Status: COMPLETE

The RPC feature from llama.cpp has been **fully transferred** to koboldcpp with enhancements:

1. **Better endpoint validation** - User-friendly error messages
2. **Device name aliasing** - Automatic CUDA ↔ HIP ↔ ROCm conversion
3. **Explicit backend registration** - Static linking support
4. **Cross-platform fixes** - FreeBSD compatibility, path handling
5. **Enhanced documentation** - 3x more comprehensive

### 📊 Protocol Version

```
Major: 4
Minor: 0
Patch: 0
```

Both codebases use **RPC Protocol v4.0.0** ✅

### 🔧 Build Targets

```bash
# RPC Servers
make rpc-server-vulkan    # AMD/Intel GPUs
make rpc-server-cuda      # NVIDIA GPUs
make rpc-server-hip       # AMD ROCm

# Client Libraries
make koboldcpp_rpc        # Vulkan + RPC
make koboldcpp_cublas_rpc # CUDA + RPC
make koboldcpp_hipblas_rpc # HIP + RPC

# All-in-one
make rpc-full-all
```

### 📦 Compiled Artifacts

**Client Libraries:**
- `koboldcpp_rpc.so` (75 MB)
- `koboldcpp_cublas_rpc.so` (82 MB)
- `koboldcpp_hipblas_rpc.so` (85 MB)

**RPC Servers:**
- `rpc-server-vulkan` (67 MB)
- `rpc-server-hip` (78 MB)

---

## Usage Examples

### Start RPC Server
```bash
$ bin/rpc-server -d CUDA0 -p 50052
```

### Connect Client
```bash
$ koboldcpp --model model.gguf --rpc 192.168.1.100:50052
```

### Multiple Servers
```bash
$ koboldcpp --model model.gguf --rpc 192.168.1.100:50052,192.168.1.101:50052
```

### With Caching
```bash
$ bin/rpc-server -c -p 50052
```

### Debug Mode
```bash
$ GGML_RPC_DEBUG=1 bin/rpc-server
```

---

## RPC Commands (17 total)

| ID | Command | Purpose |
|----|---------|---------|
| 0 | `ALLOC_BUFFER` | Allocate buffer on remote device |
| 1 | `GET_ALIGNMENT` | Query buffer alignment |
| 2 | `GET_MAX_SIZE` | Query maximum buffer size |
| 3 | `BUFFER_GET_BASE` | Get buffer base pointer |
| 4 | `FREE_BUFFER` | Free remote buffer |
| 5 | `BUFFER_CLEAR` | Clear buffer contents |
| 6 | `SET_TENSOR` | Transfer tensor to server |
| 7 | `SET_TENSOR_HASH` | Optimized transfer (hash-based) |
| 8 | `GET_TENSOR` | Retrieve tensor from server |
| 9 | `COPY_TENSOR` | Copy tensor between buffers |
| 10 | `GRAPH_COMPUTE` | Execute compute graph |
| 11 | `GET_DEVICE_MEMORY` | Query device memory |
| 12 | `INIT_TENSOR` | Initialize tensor |
| 13 | `GET_ALLOC_SIZE` | Query allocation size |
| 14 | `HELLO` | Handshake with capability exchange |
| 15 | `DEVICE_COUNT` | Query number of devices |
| 16 | `GRAPH_RECOMPUTE` | Execute cached graph |

---

## Optimization Features

### 1. Tensor Hashing
- **Threshold:** 10 MB
- **Mechanism:** FNV-1a hash
- **Benefit:** Skips transfer for cached tensors

### 2. Graph Caching
- **First execution:** `GRAPH_COMPUTE`
- **Subsequent:** `GRAPH_RECOMPUTE`
- **Benefit:** Avoids serialization overhead

### 3. RDMA Transport
- **Platform:** Linux only
- **Requirement:** libibverbs
- **Benefit:** Lower latency than TCP
- **Negotiation:** Automatic

---

## Security Warning

> ⚠️ **This is a proof-of-concept feature.**  
> ⚠️ **NEVER expose the RPC server on an open network - it is NOT secure.**

---

## Recommended Reading Order

### For First-Time Users
1. **RPC_MANUAL.md** - Learn how to use RPC
2. **RPC_TRANSFER_SUMMARY.txt** - Quick overview
3. **RPC_TRANSFER_ANALYSIS.md** - Deep dive (optional)

### For Developers
1. **RPC_CODE_DIFFERENCES.md** - See exact code changes
2. **RPC_TRANSFER_ANALYSIS.md** - Complete technical analysis
3. **RPC_PORTING_LLAMACPP_TO_KOBOLDCPP.md** - Porting context

### For Maintainers
1. **RPC_TRANSFER_ANALYSIS.md** - Full analysis
2. **RPC_CODE_DIFFERENCES.md** - Code-level details
3. **RPC_PORT_SUMMARY.md** - Technical summary

---

## Additional Resources

### Source Code Locations

| Component | Path |
|-----------|------|
| Protocol Header | `ggml/include/ggml-rpc.h` |
| RPC Implementation | `ggml/src/ggml-rpc/ggml-rpc.cpp` |
| Transport Layer | `ggml/src/ggml-rpc/transport.cpp` |
| Server Executable | `tools/rpc-server.cpp` |
| Build Config | `ggml/src/ggml-rpc/CMakeLists.txt` |

### API Headers

| API | Header |
|-----|--------|
| RPC Backend API | `ggml/include/ggml-rpc.h` |
| llama.cpp API | `include/llama.h` |

### Build Configuration

| File | Purpose |
|------|---------|
| `Makefile:937-989` | RPC build targets |
| `ggml/src/ggml-rpc/CMakeLists.txt` | CMake configuration |

---

## Contact & Support

For issues or questions about RPC functionality:
1. Check **RPC_MANUAL.md** for usage instructions
2. Review **RPC_TRANSFER_ANALYSIS.md** for technical details
3. Examine **RPC_CODE_DIFFERENCES.md** for code-level changes

---

## Document History

| Date | Event |
|------|-------|
| 2026-04-28 | Complete RPC transfer analysis performed |
| 2026-04-28 | Analysis documentation created |
| 2026-04-28 | Code differences documented |
| 2026-04-28 | Transfer confirmed as COMPLETE ✅ |

---

**Generated:** 2026-04-28  
**Analysis Tool:** Automated diff and feature comparison  
**Confidence Level:** 100%
