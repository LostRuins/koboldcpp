# RPC Quick Reference Card

## Build Commands

### Build All RPC Variants (Recommended)
```bash
make rpc-full-all
```

### Individual Builds
```bash
# Vulkan + RPC
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8

# CUDA + RPC
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8

# HIP + RPC
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8

# RPC Servers
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

## Usage Commands

### Start RPC Server
```bash
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0
```

### Connect RPC Client
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50053 --gpulayers 99
```

### Multiple RPC Servers
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50053,192.168.1.102:50053 --gpulayers 99
```

## Key Files Modified

| File | Purpose | Lines Changed |
|------|---------|---------------|
| `expose.h` | C API structure | ~6 lines |
| `gpttype_adapter.cpp` | RPC initialization | ~200 lines |
| `Makefile` | Build configuration | ~100 lines |
| `koboldcpp.py` | Python integration | Already done |

## RPC Source Files

| File | Purpose |
|------|---------|
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | RPC implementation |
| `ggml/src/ggml-rpc/transport.cpp` | Network transport |
| `ggml/src/ggml-rpc/transport.h` | Transport interface |
| `ggml/include/ggml-rpc.h` | Public API |
| `tools/rpc/rpc-server.cpp` | Server executable |

## Build Artifacts

| File | Purpose |
|------|---------|
| `koboldcpp_rpc.so` | RPC client library (Vulkan) |
| `koboldcpp_cublas_rpc.so` | RPC client library (CUDA) |
| `koboldcpp_hipblas_rpc.so` | RPC client library (HIP) |
| `rpc-server-vulkan` | Vulkan RPC server |
| `rpc-server-cuda` | CUDA RPC server |
| `rpc-server-hip` | HIP RPC server |

## Troubleshooting

### "Failed to connect to RPC server"
```bash
# Check if server is running
netstat -tlnp | grep 50053

# Test connection
nc -zv 192.168.1.101 50053

# Start server
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0
```

### "No RPC devices found"
```bash
# Check server output for devices
./rpc-server-vulkan -d VULKAN0 -p 50053

# Verify GPU is detected
vulkaninfo | grep "GPU id"
```

### "RPC library not found"
```bash
# Build RPC library
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8

# Verify it exists
ls -lh koboldcpp_rpc.so
```

## Environment Variables

```bash
export GGML_RPC_DEBUG=1  # Enable RPC debug logging
```

## Default Ports

- RPC Server: `50053`
- KoboldCPP API: `5001`

## Important Notes

1. RPC servers must be started BEFORE connecting clients
2. Use `-H 0.0.0.0` to allow remote connections
3. Firewall rules may need configuration
4. Network latency affects performance
5. Use wired connections for best results

---

**Quick Start:**
```bash
# Build everything
make rpc-full-all

# Start server (on remote GPU machine)
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0

# Connect client (on local machine)
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50053 --gpulayers 99
```
