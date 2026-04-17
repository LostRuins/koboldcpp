# RPC Makefile Guide

**Version**: 1.111.2  
**Last Updated**: 2026-04-17

---

## Quick Build Commands

### Build Everything Automatically (Recommended)

```bash
make rpc-full-all
```

This command:
- ✅ Detects CUDA (`nvcc`) availability
- ✅ Detects HIPBLAS (`hipcc`) availability
- ✅ Always builds Vulkan (universal fallback)
- ✅ Builds **regular backends** (non-RPC)
- ✅ Builds **RPC clients** (all variants)
- ✅ Builds **RPC servers** (all variants)
- ✅ Builds only compatible backends (CUDA/HIPBLAS are mutually exclusive)

### Build Specific Backends

```bash
# Vulkan only (fastest)
make LLAMA_VULKAN=1 rpc-all

# HIPBLAS only (AMD)
make LLAMA_HIPBLAS=1 koboldcpp_hipblas
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip

# CUDA only (NVIDIA)
make LLAMA_CUBLAS=1 koboldcpp_cublas
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda
```

---

## Makefile Targets

### Regular Backend Targets (Non-RPC)

| Target | Description | Flags Required |
|--------|-------------|----------------|
| `koboldcpp_vulkan` | Vulkan backend | `LLAMA_VULKAN=1` |
| `koboldcpp_hipblas` | HIPBLAS backend | `LLAMA_HIPBLAS=1` |
| `koboldcpp_cublas` | CUDA backend | `LLAMA_CUBLAS=1` |

### RPC Client Targets

| Target | Description | Flags Required |
|--------|-------------|----------------|
| `koboldcpp_rpc` | Vulkan + RPC client | `LLAMA_VULKAN=1 LLAMA_RPC=1` |
| `koboldcpp_hipblas_rpc` | HIPBLAS + RPC client | `LLAMA_HIPBLAS=1 LLAMA_RPC=1` |
| `koboldcpp_cublas_rpc` | CUDA + RPC client | `LLAMA_CUBLAS=1 LLAMA_RPC=1` |

### RPC Server Targets

| Target | Description | Flags Required |
|--------|-------------|----------------|
| `rpc-server-vulkan` | Vulkan RPC server | `LLAMA_VULKAN=1 LLAMA_RPC=1` |
| `rpc-server-hip` | HIPBLAS RPC server | `LLAMA_HIPBLAS=1 LLAMA_RPC=1` |
| `rpc-server-cuda` | CUDA RPC server | `LLAMA_CUBLAS=1 LLAMA_RPC=1` |

### Convenience Targets

| Target | Description |
|--------|-------------|
| `rpc-all` | Build all RPC clients + servers (Vulkan) |
| `rpc-clients-all` | Build all RPC clients (Vulkan) |
| `rpc-servers-all` | Build all RPC servers (Vulkan) |
| `rpc-full-all` | **Auto-detect and build ALL backends (regular + RPC)** |

---

## rpc-full-all Build Logic

The `rpc-full-all` target uses this logic:

```makefile
1. Check for nvcc (CUDA)
   - If found: mark CUDA as available

2. Check for hipcc (HIPBLAS)
   - If found: mark HIPBLAS as available

3. Build Regular Backends (Non-RPC)
   - Build Vulkan: koboldcpp_vulkan.so
   - Build HIPBLAS: koboldcpp_hipblas.so (if hipcc found)
   - Build CUDA: koboldcpp_cublas.so (if nvcc found, no hipcc)

4. Build Vulkan + RPC (always)
   - Build koboldcpp_rpc.so
   - Build rpc-server-vulkan

5. Build HIPBLAS + RPC (if hipcc found)
   - Build koboldcpp_hipblas_rpc.so
   - Build rpc-server-hip

6. Build CUDA + RPC (if nvcc found AND hipcc NOT found)
   - Build koboldcpp_cublas_rpc.so
   - Build rpc-server-cuda
```

---

## Important Notes

### CUDA and HIPBLAS Are Mutually Exclusive

**Reason**: Both backends use similar symbol names and cannot be linked together.

**Behavior**:
- If only `nvcc` found → Build CUDA
- If only `hipcc` found → Build HIPBLAS
- If both found → Build HIPBLAS only (AMD precedence)
- If neither found → Build Vulkan only

### HIPBLAS Flags Include CUDA Compatibility

When building with `LLAMA_HIPBLAS=1`, the Makefile sets:
```makefile
HIPFLAGS += -DGGML_USE_HIP -DGGML_USE_CUDA ...
```

This allows HIPBLAS to use CUDA-compatible APIs, but requires proper ROCm installation.

---

## Build Examples

### Example 1: Clean Build of Everything

```bash
make clean
make rpc-full-all -j8
```

**Builds**:
- koboldcpp_vulkan.so (Vulkan)
- koboldcpp_hipblas.so (HIPBLAS)
- koboldcpp_rpc.so (Vulkan + RPC)
- koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)
- rpc-server-vulkan
- rpc-server-hip

### Example 2: Vulkan Only (Fastest)

```bash
make clean
make LLAMA_VULKAN=1 rpc-all -j8
```

**Builds**:
- koboldcpp_rpc.so (Vulkan + RPC)
- rpc-server-vulkan

### Example 3: HIPBLAS Only (AMD)

```bash
make clean
make LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

**Builds**:
- koboldcpp_hipblas.so (HIPBLAS)
- koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)
- rpc-server-hip

### Example 4: CUDA Only (NVIDIA)

```bash
make clean
make LLAMA_CUBLAS=1 koboldcpp_cublas -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
```

**Builds**:
- koboldcpp_cublas.so (CUDA)
- koboldcpp_cublas_rpc.so (CUDA + RPC)
- rpc-server-cuda

---

## Troubleshooting

### Issue: "undefined symbol: cudaHostRegister"

**Cause**: HIPBLAS trying to link CUDA symbols without proper ROCm

**Solution**:
```bash
# Ensure ROCm is installed
rocminfo

# Rebuild HIPBLAS
make clean
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
```

### Issue: "nvcc not found"

**Cause**: CUDA toolkit not installed or not in PATH

**Solution**:
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Or use Vulkan instead
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all
```

### Issue: "hipcc not found"

**Cause**: ROCm not installed or not in PATH

**Solution**:
```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# Or use Vulkan instead
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all
```

### Issue: Vulkan build fails silently

**Cause**: Missing Vulkan dependencies

**Solution**:
```bash
# Install Vulkan dependencies
sudo apt-get install libvulkan-dev vulkan-tools glslc

# Rebuild
make clean
make LLAMA_VULKAN=1 koboldcpp_vulkan -j8
```

---

## Makefile Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_VULKAN` | Enable Vulkan backend | 0 |
| `LLAMA_CUBLAS` | Enable CUDA backend | 0 |
| `LLAMA_HIPBLAS` | Enable HIPBLAS backend | 0 |
| `LLAMA_RPC` | Enable RPC support | 0 |
| `LLAMA_RPC_SERVER` | Build RPC server | 0 |

---

## Build Output Files

After `make rpc-full-all`, expect these files (if backends available):

```
Regular Backends:
  koboldcpp_vulkan.so         # Vulkan (always built)
  koboldcpp_hipblas.so        # HIPBLAS (if hipcc found)
  koboldcpp_cublas.so         # CUDA (if nvcc found, no hipcc)

RPC Clients:
  koboldcpp_rpc.so            # Vulkan + RPC (always built)
  koboldcpp_hipblas_rpc.so    # HIPBLAS + RPC (if hipcc found)
  koboldcpp_cublas_rpc.so     # CUDA + RPC (if nvcc found, no hipcc)

RPC Servers:
  rpc-server-vulkan           # Vulkan server (always built)
  rpc-server-hip              # HIPBLAS server (if hipcc found)
  rpc-server-cuda             # CUDA server (if nvcc found, no hipcc)
```

---

**See Also**:
- `RPC_QUICKSTART.md` - Quick start guide
- `RPC_MANUAL.md` - Complete manual
- `RPC_BACKEND_COMPATIBILITY.md` - Backend compatibility matrix
