# KoboldCPP RPC - Quick Start Guide

**Version**: 1.111.2  
**Last Updated**: 2026-04-17  
**Status**: ✅ Complete - All Features Working

---

## 1-Minute Quick Start

### Step 1: Build All Backends Automatically (Recommended)

This single command detects available backends and builds everything:

```bash
cd koboldcpp_rpc_attempt
make clean
make rpc-full-all -j8
```

**What This Does**:
- ✅ Detects if CUDA (`nvcc`) is available
- ✅ Detects if HIPBLAS (`hipcc`) is available
- ✅ Always builds Vulkan (universal fallback)
- ✅ Builds **regular backends** (non-RPC)
- ✅ Builds **RPC clients** (all variants)
- ✅ Builds **RPC servers** (all variants)
- ✅ Builds only compatible backends (CUDA and HIPBLAS are mutually exclusive)

**What Gets Built**:

#### Regular Backends (Non-RPC)
| File | Purpose | Backend |
|------|---------|---------|
| `koboldcpp_vulkan.so` | Vulkan backend | Vulkan (always) |
| `koboldcpp_hipblas.so` | HIPBLAS backend | HIPBLAS (if hipcc found) |
| `koboldcpp_cublas.so` | CUDA backend | CUDA (if nvcc found, no hipcc) |

#### RPC Clients
| File | Purpose | Backend |
|------|---------|---------|
| `koboldcpp_rpc.so` | RPC client | Vulkan + RPC (always) |
| `koboldcpp_hipblas_rpc.so` | RPC client | HIPBLAS + RPC (if hipcc found) |
| `koboldcpp_cublas_rpc.so` | RPC client | CUDA + RPC (if nvcc found, no hipcc) |

#### RPC Servers
| File | Purpose | Backend |
|------|---------|---------|
| `rpc-server-vulkan` | RPC server | Vulkan (always) |
| `rpc-server-hip` | RPC server | HIPBLAS (if hipcc found) |
| `rpc-server-cuda` | RPC server | CUDA (if nvcc found, no hipcc) |

**Expected Output**:
```
=== Detecting available backends ===
Checking for NVIDIA CUDA (nvcc)...
✗ CUDA: Not available
Checking for AMD HIPBLAS (hipcc)...
✓ HIPBLAS: Available (hipcc found)

=== Building Regular Backends (Non-RPC) ===
Building Vulkan backend...
[builds koboldcpp_vulkan.so]
Building HIPBLAS backend...
[builds koboldcpp_hipblas.so]

=== Building Vulkan + RPC (universal fallback) ===
[builds koboldcpp_rpc.so, rpc-server-vulkan]

=== Building HIPBLAS + RPC (AMD GPUs) ===
HIPBLAS detected, building...
[builds koboldcpp_hipblas_rpc.so, rpc-server-hip]

=== All RPC components built successfully! ===

Built Regular Backends (Non-RPC):
  ✓ koboldcpp_vulkan.so (Vulkan)
  ✓ koboldcpp_hipblas.so (HIPBLAS)

Built RPC Clients:
  ✓ koboldcpp_rpc.so (Vulkan + RPC)
  ✓ koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)

Built RPC Servers:
  ✓ rpc-server-vulkan (Vulkan backend)
  ✓ rpc-server-hip (HIPBLAS backend)
```

### Step 1a: Manual Backend Selection

If you want specific backends only:

```bash
cd koboldcpp_rpc_attempt
make clean

# Vulkan + RPC (works on all systems)
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all -j8

# HIPBLAS + RPC (AMD GPUs only, requires ROCm)
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8

# CUDA + RPC (NVIDIA GPUs only, requires CUDA toolkit)
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
```

**Important**: CUDA and HIPBLAS cannot be built together. If both `nvcc` and `hipcc` are detected, HIPBLAS takes precedence.

### Step 2: Start Server (on GPU machine)

```bash
# Vulkan backend (universal)
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c

# HIPBLAS backend (AMD)
./rpc-server-hip -H 192.168.1.101 --port 50054 --device HIP0,HIP1 -c

# CUDA backend (NVIDIA)
./rpc-server-cuda -H 192.168.1.101 --port 50054 --device CUDA0,CUDA1 -c
```

### Step 3: Start Client (on any machine)

```bash
# Vulkan RPC client
python koboldcpp.py --model /path/to/model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,RPC1,VULKAN1 \
    --gpulayers 999 \
    --port 5001

# HIPBLAS RPC client (AMD)
python koboldcpp.py --model /path/to/model.gguf \
    --rpc 192.168.1.101:50054 \
    --device HIP0,RPC0,RPC1,HIP1 \
    --gpulayers 999 \
    --port 5001

# CUDA RPC client (NVIDIA)
python koboldcpp.py --model /path/to/model.gguf \
    --rpc 192.168.1.101:50054 \
    --device CUDA0,RPC0,RPC1,CUDA1 \
    --gpulayers 999 \
    --port 5001
```

### Step 4: Open Browser

```
http://localhost:5001
```

**Done!**

---

## Device Naming Conventions

### HIPBLAS Backend (Triple Naming Support)

When using HIPBLAS libraries (`koboldcpp_hipblas.so`, `koboldcpp_hipblas_rpc.so`), you can use **any** of these names for the same physical device:

| Physical Device | Valid Names |
|----------------|-------------|
| HIP Device 0 | `HIP0`, `CUDA0`, `ROCm0` |
| HIP Device 1 | `HIP1`, `CUDA1`, `ROCm1` |

**Triple naming**: `HIP0` = `CUDA0` = `ROCm0` (all refer to the same HIP device 0)

**All equivalent**:
```bash
python koboldcpp.py --device HIP0,RPC0,HIP1 --rpc 192.168.1.101:50053
python koboldcpp.py --device CUDA0,RPC0,CUDA1 --rpc 192.168.1.101:50053
python koboldcpp.py --device ROCm0,RPC0,ROCm1 --rpc 192.168.1.101:50053
```

### Vulkan Backend
When using `koboldcpp_rpc.so`:

| Physical Device | Valid Names |
|----------------|-------------|
| Vulkan Device 0 | `VULKAN0` |
| Vulkan Device 1 | `VULKAN1` |

### CUDA Backend (NVIDIA)
When using `koboldcpp_cublas_rpc.so`:

| Physical Device | Valid Names |
|----------------|-------------|
| CUDA Device 0 | `CUDA0`, `HIP0` |
| CUDA Device 1 | `CUDA1`, `HIP1` |

---

## Expected Output

### Server
```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
ggml_vulkan: 1 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
Starting RPC server v3.6.1
  endpoint       : 192.168.1.101:50054
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
```

### Client
```
Initializing dynamic library: koboldcpp_rpc.so
ggml_vulkan: Found 2 Vulkan devices:
[RPC] Connecting to RPC server(s): 192.168.1.101:50054
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Found RPC device 0: RPC0
[RPC] Found RPC device 1: RPC1
[RPC] Enumerating local GPU devices...
[RPC] Found local GPU device: VULKAN0 (registry: Vulkan)
[RPC] Found local GPU device: VULKAN1 (registry: Vulkan)
[RPC] Ordering devices: VULKAN0 RPC0 RPC1 VULKAN1
[RPC] Total devices for offloading: 4 (manually ordered)
llama_model_load: offloading 999 layers to GPU
load_tensors: RPC0[...] model buffer size = XXX MiB
load_tensors: RPC1[...] model buffer size = XXX MiB
load_tensors: VULKAN0 model buffer size = XXX MiB
load_tensors: VULKAN1 model buffer size = XXX MiB
```

---

## Common Scenarios

### Scenario 1: Single Server
```bash
# Server
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0 -c

# Client
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50054 --gpulayers 999
```

### Scenario 2: Multiple Servers
```bash
# Server 1
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0 -c

# Server 2
./rpc-server-vulkan -H 192.168.1.16 --port 50054 --device VULKAN0 -c

# Client
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --gpulayers 999
```

### Scenario 3: Hybrid Mode (Local + RPC)
```bash
# Server (192.168.1.101)
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c

# Client (with local GPUs)
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,RPC1,VULKAN1 \
    --gpulayers 999
```

**Result**: Uses both RPC server GPUs AND local client GPUs automatically!

### Scenario 4: AMD HIPBLAS Setup
```bash
# Server (AMD GPU with ROCm)
./rpc-server-hip -H 192.168.1.101 --port 50054 --device HIP0,HIP1 -c

# Client
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device HIP0,RPC0,RPC1,HIP1 \
    --gpulayers 999
```

### Scenario 5: NVIDIA CUDA Setup
```bash
# Server (NVIDIA GPU with CUDA)
./rpc-server-cuda -H 192.168.1.101 --port 50054 --device CUDA0,CUDA1 -c

# Client
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device CUDA0,RPC0,RPC1,CUDA1 \
    --gpulayers 999
```

### Scenario 6: Localhost Testing
```bash
# Terminal 1
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Terminal 2
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054 --gpulayers 999
```

---

## Troubleshooting Quick Fixes

### "Connection refused"
```bash
# Check server is running
ps aux | grep rpc-server

# Test connection
ping 192.168.1.101
python3 -c "import socket; s=socket.socket(); s.settimeout(5); print('OK' if s.connect_ex(('192.168.1.101', 50054))==0 else 'FAILED')"
```

### "No RPC devices found"
```bash
# Verify server shows devices
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Check Vulkan
vulkaninfo | grep "GPU id"
```

### "undefined symbol: ggml_backend_rpc_add_server"
```bash
# Using wrong library - need RPC-enabled version
ls -lh koboldcpp_*.so

# Rebuild with RPC
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
# or
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc
```

### "Device HIP0 not found"
```bash
# Check which library you're using
python koboldcpp.py ... 2>&1 | grep "Initializing dynamic"

# If using koboldcpp_rpc.so (Vulkan), use VULKAN names
python koboldcpp.py --device VULKAN0,RPC0 --rpc 192.168.1.101:50054

# If using koboldcpp_hipblas_rpc.so, HIP names work
python koboldcpp.py --device HIP0,RPC0 --rpc 192.168.1.101:50054
```

### "HIP assertion failed" (ROCm 7.2+ bug)
```bash
# Known HIP runtime bug when mixing RPC + local HIP
# Workaround 1: Use RPC only
python koboldcpp.py --device RPC0,RPC1,RPC2 --rpc 192.168.1.101:50054

# Workaround 2: Use Vulkan backend instead
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all
python koboldcpp.py --device VULKAN0,RPC0,RPC1,VULKAN1 --rpc 192.168.1.101:50054
```

### "Local GPUs not detected"
```bash
# Verify you're using the right library
ls -lh koboldcpp_rpc.so koboldcpp_hipblas_rpc.so

# Check Vulkan devices
vulkaninfo | grep -A 3 "deviceName"

# Check HIP devices
rocminfo
```

---

## What Works

✅ RPC Server starts and advertises devices  
✅ RPC Client connects to servers  
✅ Multiple RPC servers work together  
✅ GPU offloading to RPC server GPUs  
✅ Model distribution across RPC devices  
✅ **Hybrid mode** (local GPUs + RPC servers)  
✅ **Manual device ordering** (--device argument)  
✅ **Device reordering** (mix RPC and local in any order)  
✅ **Multiple naming conventions** (HIP/CUDA/ROCm all work)  
✅ **Automatic backend detection** (make rpc-full-all)  

---

## Backend Compatibility

### Critical Rule: Cannot Mix Backends

You **cannot** mix Vulkan devices with HIPBLAS/CUDA devices in the same session.

| Library | Can Use These Devices | Cannot Use |
|---------|----------------------|------------|
| **Vulkan** (`koboldcpp_vulkan.so`) | `VULKAN0`, `VULKAN1` | `HIP*`, `CUDA*`, `ROCm*` |
| **Vulkan + RPC** (`koboldcpp_rpc.so`) | `VULKAN*`, `RPC*` | `HIP*`, `CUDA*`, `ROCm*` |
| **HIPBLAS** (`koboldcpp_hipblas.so`) | `HIP*`, `CUDA*`, `ROCm*` | `VULKAN*` |
| **HIPBLAS + RPC** (`koboldcpp_hipblas_rpc.so`) | `HIP*`, `CUDA*`, `ROCm*`, `RPC*` | `VULKAN*` |
| **CUDA** (`koboldcpp_cublas.so`) | `CUDA*`, `HIP*` | `VULKAN*`, `ROCm*` |
| **CUDA + RPC** (`koboldcpp_cublas_rpc.so`) | `CUDA*`, `HIP*`, `RPC*` | `VULKAN*`, `ROCm*` |

**Important**: Each library only supports its own backend type. Use consistent device names matching the selected backend.

### GUI Dropdown Options

After building with `make rpc-full-all`, the koboldcpp.py GUI dropdown will show:

```
Use CPU
Use CUDA
Use CUDA + RPC
Use hipBLAS (ROCm)
Use hipBLAS + RPC
Use Vulkan
Use Vulkan + RPC          ← Previously "Use RPC (Remote)"
Use CPU (Old CPU)
Use Vulkan (Old CPU)
Use Vulkan (Older CPU)
Failsafe Mode (Older CPU)
```

---

## Security Warning

⚠️ **NEVER expose RPC to public internet!**

**Safe**:
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0 -c
```

**Dangerous**:
```bash
./rpc-server-vulkan -H 0.0.0.0 --port 50054 --device VULKAN0 -c
```

**Recommended**: Use SSH tunneling for remote access:
```bash
# Create tunnel
ssh -L 50054:localhost:50054 user@192.168.1.101

# Server binds to localhost
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Client connects to localhost
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054
```

---

## Performance Tips

### Network
- Use wired Ethernet (not WiFi)
- Ensure low latency (< 5ms ping)
- Same subnet is best

### Backend Choice
- **Vulkan**: Most stable with RPC, universal compatibility
- **HIPBLAS**: Best performance for AMD GPUs, but may hit assertion bugs with RPC
- **CUDA**: Best performance for NVIDIA GPUs

### Layer Distribution
- More powerful GPUs get higher tensor_split ratios
- Local GPUs typically faster (no network overhead)
- Example: Local 40%, Remote 60%

---

## More Info

- **Complete Manual**: `RPC_MANUAL.md`
- **Build Guide**: `RPC_BUILD_GUIDE.md`
- **Backend Compatibility**: `RPC_BACKEND_COMPATIBILITY.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`

---

**License**: MIT  
**Version**: 1.111.2
