# KoboldCPP RPC - Quick Start Guide

**Version**: 1.111.2  
**Last Updated**: 2026-04-23  
**Status**: ✅ Complete - All Features Working with Full UI (v1.111.2 with RPC Server tab, endpoint fields, config save/load, and backend dropdown)

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
./rpc-server-hip -H 192.168.1.101 --port 50054 --device ROCm0,ROCm1 -c

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

## Start RPC Server via GUI

1. Open `python koboldcpp.py`
2. Go to **RPC Server** tab (between "Loaded Files" and "Network")
3. Check **"Start RPC Server Mode"**
4. Configure:
   - **RPC Server Backend**: `Auto-detect` (recommended), `Vulkan`, `hipBLAS (ROCm)`, or `CUDA`
   - **Listening IP Address**: `0.0.0.0` (all interfaces) or `127.0.0.1` (localhost only)
   - **Listening Port**: `50053` (default)
   - **RPC Devices**: `VULKAN0,VULKAN1` or `ROCm0,ROCm1` (leave empty for auto-detect)
5. Check **"Allow Launch Without Models"** if no model file needed
6. Launch

**Backend Selection**:
- **Auto-detect**: Tries HIPBLAS first, then Vulkan, then CUDA
- **Vulkan**: Uses `VULKAN0`, `VULKAN1` device names
- **hipBLAS (ROCm)**: Uses `ROCm0`, `ROCm1` device names
- **CUDA**: Uses `CUDA0`, `CUDA1` device names

**Device Name Conversion**: If you enter `VULKAN0,VULKAN1` but select "hipBLAS (ROCm)" backend, the system automatically converts to `ROCm0,ROCm1`.

---

## RPC Server via CLI

```bash
python koboldcpp.py \
    --start-rpc-server \
    --rpc-host 0.0.0.0 \
    --rpc-port 50053 \
    --rpc-devices VULKAN0,VULKAN1 \
    --rpc-server-backend Vulkan
```

**Options**:
- `--start-rpc-server` - Launch RPC server mode
- `--rpc-host` - Host IP (use private LAN IP, NOT 0.0.0.0 for production)
- `--rpc-port` - Port number (default: 50053)
- `--rpc-devices` - GPU devices (VULKAN0, VULKAN1, ROCm0, CUDA0, etc.)
- `--rpc-server-backend` - Backend selection (Auto-detect, Vulkan, hipBLAS (ROCm), CUDA)

---

## Start RPC Client via GUI

### Method 1: From Quick Launch Tab

1. Select backend from dropdown (Use Vulkan + RPC, Use hipBLAS + RPC, etc.)
2. Enter RPC server endpoint in **"RPC Server Endpoint"** field (appears below Backend dropdown)
3. The field is shared between Quick Launch and Hardware tabs

### Method 2: From Hardware Tab

1. Select backend from dropdown
2. Enter RPC server endpoint in **"RPC Server Endpoint"** field
3. Both tabs use the same input field - entering in one updates the other

### Method 3: From Command Line

```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --port 5001
```

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

### Client (with Hybrid Mode)
```
Initializing dynamic library: koboldcpp_hipblas_rpc.so
[RPC] Connecting to RPC server(s): 192.168.1.101:50054
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Found RPC device 0: RPC0
[RPC] Found RPC device 1: RPC1
[RPC] Enumerating local GPU devices...
[RPC] Found local GPU device: ROCm0 (registry: ROCm)
[RPC] DEBUG: HIP/ROCM device 0 registered as both HIP0 and ROCm0
[RPC] Found local GPU device: ROCm1 (registry: ROCm)
[RPC] DEBUG: HIP/ROCM device 1 registered as both HIP1 and ROCm1
[RPC] Found local GPU device: ROCm2 (registry: ROCm)
[RPC] DEBUG: HIP/ROCM device 2 registered as both HIP2 and ROCm2
[RPC] Ordering devices: RPC0 RPC1 HIP2
[RPC] Total devices for offloading: 3 (manually ordered)
[RPC] Using 3 device(s) for model offloading
llama_model_load: offloading 999 layers to GPU
load_tensors: RPC0[...] model buffer size = XXX MiB
load_tensors: RPC1[...] model buffer size = XXX MiB
load_tensors: HIP2 model buffer size = XXX MiB
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
./rpc-server-hip -H 192.168.1.101 --port 50054 --device ROCm0,ROCm1 -c

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

### Scenario 7: RPC Server Without Model (GUI)
1. Open `python koboldcpp.py`
2. Go to **RPC Server** tab
3. Check "Start RPC Server Mode"
4. Check **"Allow Launch Without Models"**
5. Launch

---

## GUI Dropdown Options

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

## RPC Endpoint Field (Quick Launch & Hardware Tabs)

When selecting an RPC variant backend (Use Vulkan + RPC, Use hipBLAS + RPC, Use CUDA + RPC), a **"RPC Server Endpoint"** field appears below the Backend dropdown in both tabs.

**Key Features**:
- **Shared Input**: Both tabs use the same `rpc_endpoint_var` StringVar
- **Auto-sync**: Entering in one tab automatically updates the other
- **Auto-show/hide**: Field only appears when RPC variant backend is selected
- **Placeholder Text**: Shows "e.g. 192.168.1.101:50053"

**Implementation**:
- Quick Launch: `rpc_endpoint_label` + `rpc_endpoint_entry` at row=2
- Hardware: `rpc_endpoint_label_hw` + `rpc_endpoint_entry_hw` at row=2
- Visibility controlled in `changerunmode()` function
- Config saved as `rpc_endpoint` in `.kcpps` files

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

# Ensure RPC variant is selected (GUI or CLI)
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

### "Local GPUs not detected with RPC" (FIXED)
This was caused by missing "ROCM" in the device registry check. Now fixed in `gpttype_adapter.cpp`.

**Code fix**: Added "ROCM" to device enum check:
```cpp
if(reg_name_upper.find("VULKAN") != std::string::npos || 
   reg_name_upper.find("RADV") != std::string::npos ||
   reg_name_upper.find("CUDA") != std::string::npos ||
   reg_name_upper.find("HIP") != std::string::npos ||
   reg_name_upper.find("ROCM") != std::string::npos ||  // ← Added
   reg_name_upper.find("METAL") != std::string::npos) {
```

**Rebuild**:
```bash
make clean
make rpc-full-all -j8
```

### "NameError: has_rpc_in_device is not defined" (FIXED)
This occurred when loading config files. Now fixed - `import_vars` initializes `has_rpc_in_device` and `has_rocm_in_device` variables.

**Code fix**: Added to `import_vars()` function:
```python
global has_rpc_in_device, has_rocm_in_device
has_rpc_in_device = False
if args.device and "RPC" in args.device:
    has_rpc_in_device = True
has_rocm_in_device = False
if args.device and ("ROCm" in args.device or "HIP" in args.device):
    has_rocm_in_device = True
```

**Rebuild**:
```bash
make clean
make rpc-full-all -j8
```

### "RPC Server binary not found"
```bash
# Build RPC servers
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan
# or
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip
```

### "ERROR: RPC server failed with exit code 1"
```bash
# Check device names match backend
# Wrong: Using VULKAN0 with rpc-server-hip
# Correct: Using ROCm0 with rpc-server-hip

# Use --rpc-server-backend CLI argument or RPC Server Backend dropdown
```

---

## What Works

✅ RPC Server starts and advertises devices  
✅ RPC Client connects to servers  
✅ Multiple RPC servers work together  
✅ GPU offloading to RPC server GPUs  
✅ Model distribution across RPC devices  
✅ **Hybrid mode** (local GPUs + RPC servers) - local ROCm devices now detected  
✅ **Manual device ordering** (--device argument)  
✅ **Device reordering** (mix RPC and local in any order)  
✅ **Triple naming support** (HIP0 = CUDA0 = ROCm0 for HIPBLAS)  
✅ **Multiple naming conventions** (HIP/CUDA/ROCm all work)  
✅ **Automatic backend detection** (make rpc-full-all)  
✅ **RPC Server mode** (launch RPC server from GUI or CLI)  
✅ **RPC Server tab** in GUI with full configuration  
✅ **RPC Server Backend dropdown** (Auto-detect, Vulkan, hipBLAS, CUDA)  
✅ **Device name conversion** (VULKAN0 -> ROCm0, etc.)  
✅ **RPC endpoint field** shared between Quick Launch and Hardware tabs (auto-sync)  
✅ **Config save/load** for all RPC settings (.kcpps files)  
✅ **Allow Launch Without Models** for RPC Server mode  
✅ **ROCm device registry fix** (added "ROCM" to enum check)  
✅ **has_rpc_in_device initialization** in import_vars() - fixes NameError on config load  

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

**Library Selection Priority**: When RPC is used, the selected backend variant from dropdown is preferred (HIPBLAS + RPC > CUDA + RPC > Vulkan + RPC).

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
