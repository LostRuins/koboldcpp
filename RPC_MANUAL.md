# KoboldCPP RPC - Complete Manual

**Version**: 1.111.2  
**Last Updated**: 2026-04-16  
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
- ✅ **Case-insensitive** device matching (Vulkan/RADV/CUDA/HIP/METAL)

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

**What This Does**:
- ✅ Detects if CUDA (`nvcc`) is available
- ✅ Detects if HIPBLAS (`hipcc`) is available
- ✅ Always builds Vulkan (universal fallback)
- ✅ Builds only compatible backends (CUDA and HIPBLAS are mutually exclusive)

**Expected Output**:
```
=== Detecting available backends ===
Checking for NVIDIA CUDA (nvcc)...
✗ CUDA: Not available
Checking for AMD HIPBLAS (hipcc)...
✓ HIPBLAS: Available (hipcc found)

=== Building Vulkan + RPC (universal fallback) ===
[builds Vulkan...]

=== Building HIPBLAS + RPC (AMD GPUs) ===
HIPBLAS detected, building...
[builds HIPBLAS...]
✓ HIPBLAS build complete

=== Building CUDA + RPC (NVIDIA GPUs) ===
CUDA not available (nvcc not found), skipping...
```

**What Gets Built**:
| File | Purpose | Backend |
|------|---------|---------|
| `koboldcpp_rpc.so` | RPC client | Vulkan + RPC (always) |
| `koboldcpp_hipblas_rpc.so` | RPC client | HIPBLAS + RPC (if hipcc found) |
| `koboldcpp_cublas_rpc.so` | RPC client | CUDA + RPC (if nvcc found, no hipcc) |
| `rpc-server-vulkan` | RPC server | Vulkan (always) |
| `rpc-server-hip` | RPC server | HIPBLAS (if hipcc found) |
| `rpc-server-cuda` | RPC server | CUDA (if nvcc found, no hipcc) |

#### Option B: Manual Backend Selection

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

**Alternative**: Build just RPC components (clients + servers):
```bash
make rpc-all -j8
```

**What This Creates**:

| File | Purpose | Backend | Size |
|------|---------|---------|------|
| `koboldcpp_default.so` | CPU backend | CPU | ~12 MB |
| `koboldcpp.so` | Vulkan GPU backend | Vulkan | ~67 MB |
| `koboldcpp_cublas.so` | CUDA GPU backend | CUDA | ~80 MB |
| `koboldcpp_hipblas.so` | ROCm GPU backend | HIPBLAS | ~80 MB |
| `koboldcpp_rpc.so` | RPC client | Vulkan+RPC | ~67 MB |
| `koboldcpp_cublas_rpc.so` | RPC client | CUDA+RPC | ~80 MB |
| `koboldcpp_hipblas_rpc.so` | RPC client | HIPBLAS+RPC | ~80 MB |
| `rpc-server-vulkan` | RPC server | Vulkan | ~68 MB |
| `rpc-server-cuda` | RPC server | CUDA | ~85 MB |
| `rpc-server-hip` | RPC server | HIPBLAS | ~85 MB |

**Important Note**: Each RPC server uses its respective GPU backend for actual GPU computation:
- `rpc-server-vulkan` - Uses Vulkan backend, requires Vulkan drivers, works with AMD/Intel/NVIDIA GPUs via Vulkan
- `rpc-server-cuda` - Uses CUDA backend, requires NVIDIA GPU + CUDA toolkit, works with NVIDIA GPUs only
- `rpc-server-hip` - Uses HIPBLAS backend, requires AMD GPU + ROCm, works with AMD GPUs only

**Which RPC Server Should You Build?**
- Build `rpc-server-vulkan` if you have AMD, Intel, or NVIDIA GPUs with Vulkan support
- Build `rpc-server-cuda` if you have NVIDIA GPUs and want CUDA acceleration
- Build `rpc-server-hip` if you have AMD GPUs with ROCm installed

**Note**: The RPC protocol is the same across all backends, so RPC clients can connect to any RPC server regardless of backend type. You can mix Vulkan, CUDA, and HIPBLAS RPC servers and connect to them from a single client.

**After Building**:
- Copy all `.so` files to the same directory as `koboldcpp.py`
- RPC servers stay in `koboldcpp_rpc_attempt/` (run from there)

---

### Minimal Builds

If you only need specific backends, use these targeted builds:

#### Normal Use (Single Machine, No RPC)

```bash
cd koboldcpp_rpc_attempt
make clean

# CPU only
make koboldcpp_default -j8

# Vulkan GPU
make LLAMA_VULKAN=1 koboldcpp -j8

# CUDA GPU (NVIDIA)
make LLAMA_CUBLAS=1 koboldcpp_cublas -j8

# ROCm GPU (AMD)
make LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8
```

Copy the resulting `.so` files to your `koboldcpp.py` directory.

#### RPC Client Only

```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8
```

Copy `koboldcpp_rpc.so` to your `koboldcpp.py` directory.

#### RPC Server Only (Vulkan)

**Requirements**: Vulkan drivers installed (`sudo apt-get install libvulkan-dev vulkan-tools glslc`)

```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8
```

Run `./rpc-server-vulkan` from the build directory.

**Compatibility**: Works with AMD, Intel, and NVIDIA GPUs via Vulkan drivers.

#### RPC Server Only (CUDA)

**Requirements**: NVIDIA GPU + CUDA toolkit 11.0+ installed

```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
```

Run `./rpc-server-cuda` from the build directory.

**Compatibility**: NVIDIA GPUs only. Best performance on NVIDIA hardware.

#### RPC Server Only (ROCm/HIPBLAS)

**Requirements**: AMD GPU (GCN 8.0+) + ROCm 5.0+ installed

```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

Run `./rpc-server-hip` from the build directory.

**Compatibility**: AMD GPUs only with ROCm support. Best performance on AMD hardware.

---

### Build Troubleshooting

**"make: *** No rule to make target"** - Typo in target name. Check spelling.

**Build fails for CUDA** - You likely don't have CUDA toolkit installed. Verify with:
```bash
nvcc --version
which nvcc
```
Install CUDA toolkit from NVIDIA, or skip CUDA builds and use Vulkan instead.

**Build fails for ROCm/HIPBLAS** - You likely don't have ROCm installed. Verify with:
```bash
hipcc --version
which hipcc
```
Install ROCm from AMD, or skip HIPBLAS builds and use Vulkan instead.

**"undefined reference to vulkan"** - Install Vulkan development packages:
```bash
sudo apt-get install libvulkan-dev vulkan-tools glslc
```

**"undefined reference to cuda"** or **"undefined reference to hip"** - Missing CUDA/HIP libraries in linker flags. Ensure `$(CUBLASLD_FLAGS)` or `$(HIPLDFLAGS)` are included in the build command.

**Build succeeds but koboldcpp.py doesn't show backend** - Ensure `.so` files are in the same directory as `koboldcpp.py` when you run it.

**Verify builds**:
```bash
ls -lh *.so rpc-server-*
```

You should see the files listed with their sizes matching the table above.

**Test RPC Server**:
```bash
# Test Vulkan RPC server
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Test CUDA RPC server (NVIDIA only)
./rpc-server-cuda -H 127.0.0.1 --port 50054 --device CUDA0 -c

# Test HIPBLAS RPC server (AMD only)
./rpc-server-hip -H 127.0.0.1 --port 50054 --device HIP0 -c
```

Each server should show detected GPU devices and start the RPC server. If no devices are found, check that:
- GPU drivers are installed
- Vulkan/CUDA/HIP runtime is working
- Device names are correct (use `vulkaninfo`, `nvidia-smi`, or `rocm-smi` to list devices)

---

## Usage

### Start RPC Server

On machine with GPUs:
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1 -c
```

**Options**:
- `-H` - Host IP (use private LAN IP, NOT 0.0.0.0)
- `--port` - Port number (default: 50052)
- `--device` - GPU devices (VULKAN0, VULKAN1, CUDA0, HIP0, etc.)
- `-c` - Use cache directory

### Start RPC Client

On any machine:
```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --port 5001
```

**Options**:
- `--rpc` - RPC server endpoints (comma-separated for multiple)
- `--gpulayers` - Layers to offload (use 999 for full offload)
- `--model` - Model file path
- `--tensor_split` - Manual layer distribution (optional)

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

**Device Order**: By default, RPC devices first, then local GPUs in enumeration order.

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

**Device Names**:
- `VULKAN0`, `VULKAN1`, etc. - Local Vulkan GPUs (use with `koboldcpp_rpc.so`)
- `HIP0`, `HIP1`, etc. - Local HIP/HIPBLAS GPUs (use with `koboldcpp_hipblas_rpc.so`)
- `CUDA0`, `CUDA1`, etc. - Local CUDA GPUs (use with `koboldcpp_cublas_rpc.so`)
- `ROCm0`, `ROCm1`, etc. - Local ROCm GPUs (alias for HIP, use with `koboldcpp_hipblas_rpc.so`)
- `RPC0`, `RPC1`, etc. - RPC server GPUs (in connection order)
- `METAL0`, `METAL1`, etc. - Local Metal GPUs (future support)

**Dual Naming Support**:
When using HIPBLAS backend (`koboldcpp_hipblas_rpc.so`), you can use **any** of these names for the same physical device:
- `HIP0` = `CUDA0` = `ROCm0` (all refer to the same HIP device 0)
- `HIP1` = `CUDA1` = `ROCm1` (all refer to the same HIP device 1)

**Example** (all equivalent with HIPBLAS):
```bash
python koboldcpp.py --device HIP0,RPC0,HIP1 --rpc 192.168.1.101:50054
python koboldcpp.py --device CUDA0,RPC0,CUDA1 --rpc 192.168.1.101:50054
python koboldcpp.py --device ROCm0,RPC0,ROCm1 --rpc 192.168.1.101:50054
```

**Case-Insensitive**: Device names are case-insensitive (`vulkan0`, `VULKAN0`, `Vulkan0` all work).

**Backend Compatibility**:
- **Vulkan library** (`koboldcpp_rpc.so`): Use `VULKAN*`, `RPC*` only
- **HIPBLAS library** (`koboldcpp_hipblas_rpc.so`): Use `HIP*`, `CUDA*`, `ROCm*`, `RPC*`
- **CUDA library** (`koboldcpp_cublas_rpc.so`): Use `CUDA*`, `HIP*`, `RPC*`
- **Cannot mix Vulkan devices with HIPBLAS/CUDA devices** in same session

**Why Use Device Ordering**:
- Put faster local GPUs first for initial layers
- Balance network latency by interleaving devices
- Optimize for specific workload patterns
- Match tensor_split ratios to device capabilities

---

## Examples

### Example 1: Single Server

**Server** (192.168.1.101):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0 -c
```

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-0.8B-Q8_0.gguf \
    --rpc 192.168.1.101:50054 --gpulayers 999
```

### Example 2: Multiple Servers

**Server 1** (192.168.1.101):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1 -c
```

**Server 2** (192.168.1.16):
```bash
./rpc-server-vulkan -H 192.168.1.16 --port 50054 \
    --device VULKAN0 -c
```

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-7B-Q4_K_M.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --gpulayers 999
```

### Example 3: Hybrid Mode

**Server** (192.168.1.101 with 3 GPUs):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1,VULKAN2 -c
```

**Client** (with 2 local GPUs):
```bash
python koboldcpp.py --model Qwen3.5-70B-Q4_K_M.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999
```

**Result**: 5 GPUs working together (3 remote + 2 local)

### Example 4: With Tensor Split

**Server** (192.168.1.101 with 2 GPUs):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1 -c
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
- RPC2: 10%
- Vulkan0 (local): 10%
- Vulkan1 (local): 60%

### Example 5: Localhost Testing

**Terminal 1**:
```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50054 \
    --device VULKAN0 -c
```

**Terminal 2**:
```bash
python koboldcpp.py --model model.gguf \
    --rpc 127.0.0.1:50054 --gpulayers 999
```

### Example 6: Manual Device Ordering

**Server** (192.168.1.101 with 2 GPUs):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1 -c
```

**Client** (2 local GPUs, reorder to put local first):
```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,VULKAN1,RPC0,RPC1 \
    --tensor_split 30 30 20 20 \
    --gpulayers 999
```

**Result**: Local GPUs handle 60% of layers (first layers), RPC handles 40% (later layers).

### Example 7: Interleaved Device Ordering

**Server** (192.168.1.101 with 2 GPUs):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1 -c
```

**Client** (interleave for balanced distribution):
```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,VULKAN1,RPC1 \
    --tensor_split 25 25 25 25 \
    --gpulayers 999
```

**Result**: Alternating layers between local and remote GPUs for balanced load.

---

## Troubleshooting

### "Connection refused"

**Cause**: Server not running or wrong IP/port

**Solution**:
```bash
# Check server
ps aux | grep rpc-server

# Test network
ping 192.168.1.101
python3 -c "import socket; s=socket.socket(); s.settimeout(5); print('OK' if s.connect_ex(('192.168.1.101', 50054))==0 else 'FAILED')"
```

### "No RPC devices found"

**Cause**: Server not advertising devices

**Solution**:
1. Verify server shows devices in output
2. Check server `--device` parameter
3. Test with localhost first
4. Ensure Vulkan is working: `vulkaninfo | grep "GPU id"`

### "Segmentation fault"

**Cause**: Improper offload settings

**Solution**:
```bash
# Always use --gpulayers 999 with RPC
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 --gpulayers 999
```

### "undefined symbol" errors

**Cause**: Build issues

**Solution**:
```bash
make clean && make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
```

### Slow Performance

**Cause**: Network latency

**Solution**:
- Use wired Ethernet (not WiFi)
- Ensure low latency (< 5ms ping)
- Reduce context size
- Use same subnet

### "Local GPUs not detected"

**Cause**: RPC client built without Vulkan backend, or full backend libraries missing

**Solution**:
```bash
# Rebuild with Vulkan (full build recommended)
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc

# For full backend support, also build:
make koboldcpp_default
make LLAMA_VULKAN=1 koboldcpp

# Verify size (~67 MB for RPC with Vulkan)
ls -lh koboldcpp_rpc.so

# Verify all backends exist
ls -lh koboldcpp_default.so koboldcpp.so
```

**Note**: If you only ran the RPC-only build commands, you will only have RPC backend available. For CPU and Vulkan backends, you must build them separately using the full build commands.

### "tensor_split not working"

**Cause**: Incorrect number of values

**Solution**:
```bash
# Count total devices: RPC devices + local GPUs
# Example: 2 RPC + 2 local = 4 values
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor_split 10 10 40 40 \
    --gpulayers 999
```

---

## Security

⚠️ **CRITICAL: RPC has NO authentication or encryption!**

### Safe Configurations

**✅ Localhost**:
```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c
```

**✅ Private LAN**:
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0 -c
```

**✅ With Firewall**:
```bash
sudo ufw allow from 192.168.1.0/24 to any port 50054 proto tcp
./rpc-server-vulkan -H 0.0.0.0 --port 50054 --device VULKAN0 -c
```

### Dangerous Configurations

**❌ NEVER USE**:
```bash
./rpc-server-vulkan -H 0.0.0.0 --port 50054 --device VULKAN0 -c
```

This exposes RPC to the entire internet!

### SSH Tunneling (Recommended for Remote)

```bash
# Create tunnel
ssh -L 50054:localhost:50054 user@192.168.1.101

# Server binds to localhost
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Client connects to localhost
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054
```

---

## Performance

### Network Requirements

**Best**: Gigabit Ethernet, < 5ms latency  
**Acceptable**: Fast WiFi, < 20ms latency  
**Not Recommended**: Internet, > 50ms latency

### Speed Expectations

| Configuration | Speed (tokens/sec) |
|---------------|-------------------|
| Local GPU only | 45 t/s |
| RPC (LAN) | 28 t/s |
| Hybrid (LAN + local) | 50-60 t/s |
| RPC (slow network) | 5-15 t/s |

### Memory Requirements

| Model Size | Minimum VRAM |
|------------|--------------|
| 0.5B | 1-2 GB |
| 7B | 6-8 GB |
| 13B | 10-14 GB |
| 70B | 42-48 GB |
| 397B | 200+ GB |

**Note**: With RPC, memory is distributed across all devices (remote + local).

### Tensor Split Guidelines

- **Equal distribution**: `25 25 25 25` (for 4 devices)
- **Power-weighted**: More to powerful GPUs
- **Memory-weighted**: More to GPUs with more VRAM
- **Network-aware**: Less to remote GPUs on slow networks

Example for hybrid mode:
```bash
# 2 remote (slower) + 2 local (faster)
--tensor_split 10 10 40 40
```

---

## FAQ

**Q: Can I use local GPUs with RPC?**  
A: Yes! Hybrid mode automatically detects and uses local GPUs alongside RPC servers.

**Q: How many RPC servers can I connect to?**  
A: Unlimited, practical limit is ~10 servers.

**Q: Does RPC work over internet?**  
A: Technically yes, but NOT recommended (security + performance).

**Q: What GPU backends does RPC support?**  
A: Currently Vulkan. Device names: VULKAN0, VULKAN1, etc.

**Q: Can I mix different GPU models?**  
A: Yes, RPC works with different GPU models and vendors.

**Q: How do I control layer distribution?**  
A: Use `--tensor_split` with values for each device (RPC + local).

**Q: What's the device order for tensor_split?**  
A: By default: RPC devices first (in server order), then local GPUs (in enumeration order). Override with `--device`.

**Q: Can I reorder devices?**  
A: Yes! Use `--device VULKAN0,RPC0,VULKAN1,RPC1` to specify custom device order.

**Q: What device names can I use?**  
A: `VULKAN0`, `VULKAN1`, etc. (local), `RPC0`, `RPC1`, etc. (remote), `CUDA0`, `HIP0`, `METAL0` (future).

**Q: Are device names case-sensitive?**  
A: No, device names are case-insensitive (`vulkan0`, `VULKAN0`, `Vulkan0` all work).

**Q: Why is tensor_split truncated?**  
A: It shouldn't be anymore. Ensure you have the latest version with the fix.

---

## Advanced Features

### Default Device Enumeration Order

By default, devices are ordered as:
1. RPC Server 1, GPU 0 (RPC0)
2. RPC Server 1, GPU 1 (RPC1)
3. RPC Server 2, GPU 0 (RPC2)
4. RPC Server 2, GPU 1 (RPC3)
5. Local GPU 0 (VULKAN0)
6. Local GPU 1 (VULKAN1)
...

### Manual Device Ordering

Override default ordering with `--device` argument:

```bash
# Custom order: local first, then RPC
--device VULKAN0,VULKAN1,RPC0,RPC1

# Interleaved order
--device VULKAN0,RPC0,VULKAN1,RPC1

# RPC first, then local (default behavior)
--device RPC0,RPC1,VULKAN0,VULKAN1
```

**Benefits**:
- Optimize for network latency (put local GPUs first)
- Balance layer distribution across device types
- Match tensor_split ratios to device capabilities
- Prioritize faster devices for critical layers

**Device Naming Convention**:
- Local devices: `VULKAN0`, `VULKAN1`, `CUDA0`, `HIP0`, `METAL0` (enumeration order)
- RPC devices: `RPC0`, `RPC1`, `RPC2`, ... (connection order)
- Case-insensitive: `vulkan0`, `VULKAN0`, `Vulkan0` all work

### Tensor Split Calculation

Values are ratios, not absolute layers. They are normalized to sum to 1.0.

Example: `10 10 40 40` becomes:
- Device 0: 10/110 = 9.1%
- Device 1: 10/110 = 9.1%
- Device 2: 40/110 = 36.4%
- Device 3: 40/110 = 36.4%

### Backend Support

**Currently Supported**:
- Vulkan (RADV, AMD, NVIDIA, Intel)
- RPC (remote Vulkan devices)

**Future**:
- CUDA (NVIDIA)
- HIP (AMD ROCm)
- METAL (Apple)

---

## More Information

- **Quick Start**: `RPC_QUICKSTART.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`

---

## Backend Libraries Explained

KoboldCPP uses separate backend libraries for different acceleration methods. Each library is built independently:

### Client Libraries (for koboldcpp.py)

| Library File | Size | Build Command | Purpose |
|--------------|------|---------------|---------|
| `koboldcpp_default.so` | ~12 MB | `make koboldcpp_default` | CPU-only backend (no GPU) |
| `koboldcpp.so` | ~67 MB | `make LLAMA_VULKAN=1 koboldcpp` | Vulkan GPU backend (AMD/Intel/NVIDIA) |
| `koboldcpp_cublas.so` | ~80 MB | `make LLAMA_CUBLAS=1 koboldcpp_cublas` | CUDA GPU backend (NVIDIA only) |
| `koboldcpp_hipblas.so` | ~80 MB | `make LLAMA_HIPBLAS=1 koboldcpp_hipblas` | ROCm GPU backend (AMD only) |
| `koboldcpp_rpc.so` | ~67 MB | `make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc` | RPC client for remote GPU acceleration |
| `koboldcpp_vulkan.so` | ~67 MB | `make LLAMA_VULKAN=1 koboldcpp_vulkan` | Vulkan backend (alternative name) |
| `koboldcpp_noavx2.so` | ~10 MB | `make koboldcpp_noavx2` | CPU backend for old CPUs (no AVX2) |
| `koboldcpp_vulkan_noavx2.so` | ~60 MB | `make LLAMA_VULKAN=1 koboldcpp_vulkan_noavx2` | Vulkan backend for old CPUs |
| `koboldcpp_vulkan_failsafe.so` | ~50 MB | `make LLAMA_VULKAN=1 koboldcpp_vulkan_failsafe` | Failsafe Vulkan backend |
| `koboldcpp_failsafe.so` | ~10 MB | `make koboldcpp_failsafe` | Failsafe CPU backend |

### Server Binaries (rpc-server-*)

| Binary | Size | Build Command | GPU Backend | Requirements |
|--------|------|---------------|-------------|--------------|
| `rpc-server-vulkan` | ~68 MB | `make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan` | Vulkan | Vulkan drivers |
| `rpc-server-cuda` | ~85 MB | `make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda` | CUDA | NVIDIA GPU + CUDA toolkit |
| `rpc-server-hip` | ~85 MB | `make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip` | HIPBLAS | AMD GPU + ROCm |

**GPU Backend Differences**:
- **Vulkan**: Works on AMD, Intel, and NVIDIA GPUs via Vulkan drivers. Most portable option.
- **CUDA**: NVIDIA-only backend. Best performance on NVIDIA GPUs. Requires CUDA toolkit.
- **HIPBLAS**: AMD-only backend. Best performance on AMD GPUs with ROCm. Requires ROCm.

**RPC Protocol Compatibility**: All RPC servers use the same RPC protocol, so clients can connect to any server type. You can mix different backend types across multiple machines.

**Why Multiple Libraries?**
- Each backend requires different dependencies (Vulkan, CUDA, HIP, etc.)
- Building all backends into one library would make it huge and slow to compile
- Modular design allows picking only the backends you need
- KoboldCPP automatically detects available libraries and offers them in the UI
- RPC servers can run on different machines with different GPU types

**How Backends Work Together**:

1. **Single Machine (Normal Use)**:
   - koboldcpp.py detects all `.so` files in its directory
   - User selects backend in UI (CPU, Vulkan, CUDA, or ROCm)
   - All GPUs in the machine are used

2. **Multiple Machines (RPC Mode)**:
   - RPC servers run on GPU machines (can be Vulkan, CUDA, or ROCm)
   - RPC client connects to servers over network
   - Model layers distributed across all remote GPUs
   - Client can also use local GPUs alongside RPC servers (hybrid mode)

3. **Mixed Backend RPC**:
   - Server 1: Vulkan GPUs (AMD cards)
   - Server 2: CUDA GPUs (NVIDIA cards)
   - Server 3: ROCm GPUs (AMD cards with ROCm)
   - Client: Connects to all three, distributes layers across all

**Example Mixed Setup**:
```bash
# Machine 1 (AMD GPUs - Vulkan)
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c

# Machine 2 (NVIDIA GPUs - CUDA)
./rpc-server-cuda -H 192.168.1.102 --port 50054 --device CUDA0,CUDA1 -c

# Machine 3 (AMD GPUs - ROCm)
./rpc-server-hip -H 192.168.1.103 --port 50054 --device HIP0,HIP1 -c

# Client (connects to all)
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.102:50054,192.168.1.103:50054 \
    --tensor_split 30 30 20 20 15 15 \
    --gpulayers 999
```

**Recommended Minimum Build** (for most users):
```bash
make koboldcpp_default                    # CPU fallback
make LLAMA_VULKAN=1 koboldcpp             # Vulkan GPU
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc  # RPC client
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan  # RPC server
```

This gives you CPU, Vulkan, and RPC options in the UI.

**Full Build** (maximum flexibility):
```bash
make koboldcpp_default -j8                # CPU
make LLAMA_VULKAN=1 koboldcpp -j8         # Vulkan
make LLAMA_CUBLAS=1 koboldcpp_cublas -j8  # CUDA
make LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8  # ROCm
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8  # RPC client
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8  # RPC server Vulkan
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8  # RPC server CUDA
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8  # RPC server ROCm
```

This gives you all backend options and all RPC server types.

---

**License**: MIT  
**Version**: 1.111.2
