# KoboldCPP RPC - Complete Manual

**Version**: 1.111.2  
**Last Updated**: 2026-04-17  
**Status**: ✅ Complete - All Features Working

---

## Overview

RPC (Remote Procedure Call) allows distributing model inference across multiple machines with GPUs. KoboldCPP RPC now supports:

**What Works**:
- ✅ RPC Server on GPU machines
- ✅ RPC Client connects to servers
- ✅ Multiple servers simultaneously
- ✅ GPU offloading to RPC servers
- ✅ **Hybrid mode** (local GPUs + RPC servers)
- ✅ **Manual tensor_split** for layer distribution control
- ✅ **Manual device ordering** with --device argument
- ✅ **Multiple naming conventions** (HIP/CUDA/ROCm all work for HIPBLAS)
- ✅ **Automatic backend detection** (make rpc-full-all)
- ✅ **Full GUI dropdown** with all backend options

---

## Build

### Prerequisites

**Base Requirements** (all builds):
```bash
sudo apt-get install build-essential cmake
```

**Vulkan Support** (AMD/Intel/NVIDIA GPUs):
```bash
sudo apt-get install libvulkan-dev vulkan-tools glslc
```

**CUDA Support** (NVIDIA GPUs only):
- NVIDIA GPU with compute capability 6.0+
- CUDA toolkit 11.0+ installed
- NVIDIA driver installed

**ROCm Support** (AMD GPUs only):
- AMD GPU with GCN 8.0+ (RX 400 series or newer)
- ROCm 5.0+ installed
- AMD driver installed

---

### Full Build - All Backends (Recommended)

#### Option A: Automatic Build (Easiest!)

Build all RPC clients and servers with automatic backend detection:

```bash
cd koboldcpp_rpc_attempt
make clean
make rpc-full-all -j8
```

**What This Builds**:

#### Regular Backends (Non-RPC)
- `koboldcpp_vulkan.so` - Vulkan backend (always built)
- `koboldcpp_hipblas.so` - HIPBLAS backend (if hipcc found)
- `koboldcpp_cublas.so` - CUDA backend (if nvcc found, no hipcc)

#### RPC Clients
- `koboldcpp_rpc.so` - Vulkan + RPC (always built)
- `koboldcpp_hipblas_rpc.so` - HIPBLAS + RPC (if hipcc found)
- `koboldcpp_cublas_rpc.so` - CUDA + RPC (if nvcc found, no hipcc)

#### RPC Servers
- `rpc-server-vulkan` - Vulkan backend (always built)
- `rpc-server-hip` - HIPBLAS backend (if hipcc found)
- `rpc-server-cuda` - CUDA backend (if nvcc found, no hipcc)

**What This Does**:
- ✅ Detects if CUDA (`nvcc`) is available
- ✅ Detects if HIPBLAS (`hipcc`) is available
- ✅ Always builds Vulkan (universal fallback)
- ✅ Builds only compatible backends (CUDA and HIPBLAS are mutually exclusive)
- ✅ Builds regular backends (non-RPC)
- ✅ Builds RPC clients (all variants)
- ✅ Builds RPC servers (all variants)

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
  - koboldcpp_cublas.so (CUDA) - not built

Built RPC Clients:
  ✓ koboldcpp_rpc.so (Vulkan + RPC)
  ✓ koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)
  - koboldcpp_cublas_rpc.so (CUDA + RPC) - not built

Built RPC Servers:
  ✓ rpc-server-vulkan (Vulkan backend)
  ✓ rpc-server-hip (HIPBLAS backend)
  - rpc-server-cuda (CUDA backend) - not built

Summary:
  - Regular backends: koboldcpp_vulkan.so, koboldcpp_hipblas.so
  - RPC clients: koboldcpp_rpc.so, koboldcpp_hipblas_rpc.so
  - RPC servers: rpc-server-vulkan, rpc-server-hip
```

#### Option B: Manual Backend Selection

If you want specific backends only:

```bash
cd koboldcpp_rpc_attempt
make clean

# Vulkan + RPC (works on all systems)
make LLAMA_VULKAN=1 rpc-all -j8

# HIPBLAS + RPC (AMD GPUs only, requires ROCm)
make LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8

# CUDA + RPC (NVIDIA GPUs only, requires CUDA toolkit)
make LLAMA_CUBLAS=1 koboldcpp_cublas -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
```

**Important**: CUDA and HIPBLAS cannot be built together. If both `nvcc` and `hipcc` are detected, HIPBLAS takes precedence.

---

### Build Troubleshooting

**"Vulkan build fails silently"** - Check for errors:
```bash
# Install Vulkan dependencies
sudo apt-get install libvulkan-dev vulkan-tools glslc

# Rebuild with verbose output
make clean
make LLAMA_VULKAN=1 koboldcpp_vulkan -j8
```

**"undefined reference to vulkan"** - Install Vulkan development packages:
```bash
sudo apt-get install libvulkan-dev vulkan-tools glslc
```

**"undefined reference to cuda" or "undefined reference to hip"** - Missing CUDA/HIP libraries. Ensure proper ROCm/CUDA installation.

**"undefined symbol: cudaHostRegister"** - HIPBLAS trying to link CUDA symbols without proper ROCm:
```bash
# Verify ROCm installation
rocminfo

# Rebuild HIPBLAS
make clean
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
```

**"undefined symbol: __hipRegisterFunction"** - Trying to build both CUDA and HIPBLAS together:
```bash
# make rpc-full-all now skips CUDA if HIPBLAS detected
make clean
make rpc-full-all
```

**Build succeeds but koboldcpp.py doesn't show backend** - Ensure `.so` files are in the same directory as `koboldcpp.py`:
```bash
ls -lh koboldcpp_*.so rpc-server-*
```

**Verify builds**:
```bash
ls -lh *.so rpc-server-*
```

You should see regular backends, RPC clients, and RPC servers listed.

**Test RPC Server**:
```bash
# Test Vulkan RPC server
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Test HIPBLAS RPC server (AMD only)
./rpc-server-hip -H 127.0.0.1 --port 50054 --device ROCm0 -c

# Test CUDA RPC server (NVIDIA only)
./rpc-server-cuda -H 127.0.0.1 --port 50054 --device CUDA0 -c
```

Each server should show detected GPU devices and start the RPC server. If no devices are found, check that:
- GPU drivers are installed
- Vulkan/CUDA/HIP runtime is working
- Device names are correct (use `vulkaninfo`, `nvidia-smi`, or `rocminfo` to list devices)

---

## Usage

### Start RPC Server

On machine with GPUs:
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c
```

**Options**:
- `-H` - Host IP (use private LAN IP, NOT 0.0.0.0)
- `--port` - Port number (default: 50052)
- `--device` - GPU devices (VULKAN0, VULKAN1, HIP0, CUDA0, ROCm0, etc.)
- `-c` - Use cache directory

### Start RPC Client

On any machine:
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50054 --gpulayers 999
```

**Options**:
- `--rpc` - RPC server endpoints (comma-separated for multiple)
- `--gpulayers` - Layers to offload (use 999 for full offload)
- `--model` - Model file path
- `--tensor_split` - Manual layer distribution (optional)
- `--device` - Manual device ordering (optional)

### Multiple Servers

```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --gpulayers 999
```

### Hybrid Mode (Local + RPC)

```bash
# Server (192.168.1.101 with 2 GPUs)
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c

# Client (with 2 local GPUs)
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999
```

**Result**: Uses all 4 GPUs (2 remote + 2 local) automatically!

### Manual Layer Distribution

```bash
# 2 RPC devices + 2 local GPUs = 4 total
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor_split 10 10 40 40 \
    --gpulayers 999
```

**Distribution**:
- RPC0 (remote GPU 0): 10%
- RPC1 (remote GPU 1): 10%
- Vulkan0 (local GPU 0): 40%
- Vulkan1 (local GPU 1): 40%

### Manual Device Ordering

Override automatic device ordering with `--device`:

```bash
# Local GPUs first, then RPC
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,VULKAN1,RPC0,RPC1 \
    --gpulayers 999

# Interleaved for optimal performance
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,VULKAN1,RPC1 \
    --tensor_split 25 25 25 25 \
    --gpulayers 999
```

---

## Device Naming Conventions

### HIPBLAS Backend (AMD)

When using HIPBLAS libraries (`koboldcpp_hipblas.so`, `koboldcpp_hipblas_rpc.so`), you can use **any** of these names for the same physical device:

| Physical Device | Valid Names |
|----------------|-------------|
| HIP Device 0 | `HIP0`, `CUDA0`, `ROCm0` |
| HIP Device 1 | `HIP1`, `CUDA1`, `ROCm1` |

**All equivalent**:
```bash
python koboldcpp.py --device HIP0,RPC0,HIP1 --rpc 192.168.1.101:50054
python koboldcpp.py --device CUDA0,RPC0,CUDA1 --rpc 192.168.1.101:50054
python koboldcpp.py --device ROCm0,RPC0,ROCm1 --rpc 192.168.1.101:50054
```

**Example Server Command (HIPBLAS)**:
```bash
./rpc-server-hip -H 192.168.1.101 --port 50054 --device ROCm0,ROCm1 -c
```

### Vulkan Backend

When using Vulkan libraries (`koboldcpp_vulkan.so`, `koboldcpp_rpc.so`):

| Physical Device | Valid Names |
|----------------|-------------|
| Vulkan Device 0 | `VULKAN0` |
| Vulkan Device 1 | `VULKAN1` |

**Example Server Command (Vulkan)**:
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c
```

### CUDA Backend (NVIDIA)

When using CUDA libraries (`koboldcpp_cublas.so`, `koboldcpp_cublas_rpc.so`):

| Physical Device | Valid Names |
|----------------|-------------|
| CUDA Device 0 | `CUDA0`, `HIP0` |
| CUDA Device 1 | `CUDA1`, `HIP1` |

**Example Server Command (CUDA)**:
```bash
./rpc-server-cuda -H 192.168.1.101 --port 50054 --device CUDA0,CUDA1 -c
```

### Device Name Summary

| Name Prefix | Backend | Example Libraries |
|------------|---------|-------------------|
| `VULKAN*` | Vulkan | `koboldcpp_vulkan.so`, `koboldcpp_rpc.so` |
| `HIP*` | HIPBLAS | `koboldcpp_hipblas.so`, `koboldcpp_hipblas_rpc.so` |
| `CUDA*` | CUDA | `koboldcpp_cublas.so`, `koboldcpp_cublas_rpc.so` |
| `ROCm*` | HIPBLAS (alias) | `koboldcpp_hipblas.so`, `koboldcpp_hipblas_rpc.so` |
| `RPC*` | RPC servers | All RPC variants |

**Case-Insensitive**: Device names are case-insensitive (`vulkan0`, `VULKAN0`, `Vulkan0` all work).

**Why Use Device Ordering**:
- Put faster local GPUs first for initial layers
- Balance network latency by interleaving devices
- Optimize for specific workload patterns
- Match tensor_split ratios to device capabilities

---

## Backend Compatibility

### Critical Rule: Cannot Mix Backends

You **cannot** mix Vulkan devices with HIPBLAS/CUDA devices in the same session. Each library only supports its own backend type.

| Library | Can Use These Devices | Cannot Use |
|---------|----------------------|------------|
| **Vulkan** (`koboldcpp_vulkan.so`) | `VULKAN0`, `VULKAN1` | `HIP*`, `CUDA*`, `ROCm*` |
| **Vulkan + RPC** (`koboldcpp_rpc.so`) | `VULKAN*`, `RPC*` | `HIP*`, `CUDA*`, `ROCm*` |
| **HIPBLAS** (`koboldcpp_hipblas.so`) | `HIP*`, `CUDA*`, `ROCm*` | `VULKAN*` |
| **HIPBLAS + RPC** (`koboldcpp_hipblas_rpc.so`) | `HIP*`, `CUDA*`, `ROCm*`, `RPC*` | `VULKAN*` |
| **CUDA** (`koboldcpp_cublas.so`) | `CUDA*`, `HIP*` | `VULKAN*`, `ROCm*` |
| **CUDA + RPC** (`koboldcpp_cublas_rpc.so`) | `CUDA*`, `HIP*`, `RPC*` | `VULKAN*`, `ROCm*` |

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

### How Backend Selection Works

When you select an option from the dropdown:
1. **Vulkan options** → Loads `koboldcpp_vulkan.so` or `koboldcpp_rpc.so`
2. **HIPBLAS options** → Loads `koboldcpp_hipblas.so` or `koboldcpp_hipblas_rpc.so`
3. **CUDA options** → Loads `koboldcpp_cublas.so` or `koboldcpp_cublas_rpc.so`

The library selection automatically detects if RPC is being used and selects the appropriate RPC variant.

---

## Examples

### Example 1: Single Server (Vulkan)

**Server** (192.168.1.101):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0 -c
```

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-0.8B-Q8_0.gguf \
    --rpc 192.168.1.101:50054 --gpulayers 999
```

### Example 2: Single Server (HIPBLAS)

**Server** (192.168.1.101):
```bash
./rpc-server-hip -H 192.168.1.101 --port 50054 --device ROCm0 -c
```

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-0.8B-Q8_0.gguf \
    --rpc 192.168.1.101:50054 \
    --device HIP0,RPC0 \
    --gpulayers 999
```

### Example 3: Multiple Servers (Vulkan)

**Server 1** (192.168.1.101):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c
```

**Server 2** (192.168.1.16):
```bash
./rpc-server-vulkan -H 192.168.1.16 --port 50054 --device VULKAN0 -c
```

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-7B-Q4_K_M.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --gpulayers 999
```

### Example 4: Multiple Servers (Mixed Backend)

**Server 1** (AMD GPUs):
```bash
./rpc-server-hip -H 192.168.1.101 --port 50054 --device ROCm0,ROCm1 -c
```

**Server 2** (NVIDIA GPUs):
```bash
./rpc-server-cuda -H 192.168.1.16 --port 50054 --device CUDA0 -c
```

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-70B-Q4_K_M.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --gpulayers 999
```

**Note**: RPC protocol is the same across all backends, so you can connect to any RPC server regardless of backend type.

### Example 5: Hybrid Mode (AMD HIPBLAS)

**Server** (192.168.1.101 with 3 GPUs):
```bash
./rpc-server-hip -H 192.168.1.101 --port 50054 --device ROCm0,ROCm1,ROCm2 -c
```

**Client** (with 2 local GPUs):
```bash
python koboldcpp.py --model Qwen3.5-70B-Q4_K_M.gguf \
    --rpc 192.168.1.101:50054 \
    --device HIP0,HIP1,RPC0,RPC1,RPC2 \
    --gpulayers 999
```

**Result**: 5 GPUs working together (2 local + 3 remote)

### Example 6: With Tensor Split

**Server** (192.168.1.101 with 2 GPUs):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c
```

**Client** (2 local GPUs):
```bash
python koboldcpp.py --model Qwen3.5-397B-A17B-K_G_2.93.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor_split 10 10 10 10 60 \
    --gpulayers 999
```

**Distribution** (5 devices total):
- RPC0: 10%
- RPC1: 10%
- Vulkan0 (local): 10%
- Vulkan1 (local): 10%
- (Additional device): 60%

### Example 7: Localhost Testing

**Terminal 1**:
```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c
```

**Terminal 2**:
```bash
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054 --gpulayers 999
```

---

## Troubleshooting

### "Connection refused"

```bash
# Check server is running
ps aux | grep rpc-server

# Test connection
ping 192.168.1.101

# Test RPC port
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

```
Assertion `err == hipSuccess' failed in hip_code_object.cpp
```

**Known HIP runtime bug when mixing RPC + local HIP devices.**

**Workarounds**:
1. Use RPC only (no local HIP):
   ```bash
   python koboldcpp.py --device RPC0,RPC1,RPC2 --rpc 192.168.1.101:50054
   ```

2. Use local HIP only (no RPC):
   ```bash
   python koboldcpp.py --device HIP0,HIP1 --gpulayers 999
   ```

3. Use Vulkan backend instead (most stable):
   ```bash
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

### "tensor_split not working"

```bash
# Ensure correct number of values
# Count: RPC devices + local GPUs
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor_split 10 10 40 40 \
    --gpulayers 999
```

### "Device ordering not working"

```bash
# Use correct device names: VULKAN0, VULKAN1, RPC0, RPC1, etc.
# Device names are case-insensitive
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,VULKAN1,RPC1 \
    --gpulayers 999
```

### "Vulkan build fails silently"

```bash
# Install Vulkan dependencies
sudo apt-get install libvulkan-dev vulkan-tools glslc

# Use correct target name
make clean
make LLAMA_VULKAN=1 koboldcpp_vulkan -j8
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

### Memory
- Model split across all devices
- Each device needs enough VRAM for its portion
- KV cache also distributed

---

## API Reference

### RPC Server Options

| Option | Description | Example |
|--------|-------------|---------|
| `-H` | Host IP | `192.168.1.101` |
| `--port` | Port number | `50054` |
| `--device` | GPU devices | `VULKAN0,VULKAN1` |
| `-c` | Use cache | Enable cache directory |

### RPC Client Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--rpc` | RPC endpoints | `192.168.1.101:50054` |
| `--model` | Model path | `/path/to/model.gguf` |
| `--gpulayers` | Layers to offload | `999` |
| `--tensor_split` | Layer distribution | `10 10 40 40` |
| `--device` | Device ordering | `VULKAN0,RPC0,VULKAN1` |
| `--port` | API port | `5001` |

---

## More Info

- **Quick Start**: `RPC_QUICKSTART.md`
- **Build Guide**: `RPC_makefile.md`
- **Backend Compatibility**: `RPC_BACKEND_COMPATIBILITY.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`
- **Code Changes**: `RPC_koboldcpp.py_changes.md`

---

**License**: MIT  
**Version**: 1.111.2
