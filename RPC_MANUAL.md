# KoboldCPP RPC - Complete Manual

**Version**: 1.111.2  
**Last Updated**: 2026-04-18  
**Status**: ✅ Complete - All Features Working (v1.111.2 with RPC endpoint fields and config save/load)

---

## Overview

RPC (Remote Procedure Call) allows distributing model inference across multiple machines with GPUs. KoboldCPP RPC now supports:

**What Works**:
- ✅ RPC Server on GPU machines
- ✅ RPC Client connects to servers
- ✅ Multiple servers simultaneously
- ✅ GPU offloading to RPC servers
- ✅ **Hybrid mode** (local GPUs + RPC servers) - local ROCm devices now detected correctly
- ✅ **Manual tensor_split** for layer distribution control
- ✅ **Manual device ordering** with --device argument
- ✅ **Device reordering** (mix RPC and local in any order)
- ✅ **Triple naming support** (HIP0 = CUDA0 = ROCm0 for HIPBLAS)
- ✅ **Automatic backend detection** (make rpc-full-all)
- ✅ **Full GUI dropdown** with all backend options
- ✅ **RPC Server mode** (launch RPC server from GUI or CLI)
- ✅ **RPC Server tab** in GUI with full configuration
- ✅ **RPC Server Backend dropdown** (Auto-detect, Vulkan, hipBLAS, CUDA)
- ✅ **Device name conversion** (VULKAN0 -> ROCm0, etc.)
- ✅ **Allow Launch Without Models** for RPC Server mode
- ✅ **RPC endpoint field** in both Quick Launch and Hardware tabs (shared input)
- ✅ **Config save/load** for all RPC settings (.kcpps files)
- ✅ **ROCm device registry check** fix - added "ROCM" to device enum check
- ✅ **has_rpc_in_device initialization** in import_vars() - fixes NameError on config load

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

**What This Does**:
- ✅ Detects if CUDA (`nvcc`) is available
- ✅ Detects if HIPBLAS (`hipcc`) is available
- ✅ Always builds Vulkan (universal fallback)
- ✅ Builds only compatible backends (CUDA and HIPBLAS are mutually exclusive)
- ✅ Builds regular backends (non-RPC)
- ✅ Builds RPC clients (all variants)
- ✅ Builds RPC servers (all variants)

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

---

### Build Troubleshooting

**"Vulkan build fails silently"** - Check for errors:
```bash
# Install Vulkan dependencies
sudo apt-get install libvulkan-dev vulkan-tools glslc

# Rebuild with verbose output
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8
```

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

**"undefined symbol: ggml_backend_rpc_add_server"** - Loading non-RPC library when RPC is needed:
```bash
# Ensure RPC variant is built and selected
make clean
make rpc-full-all -j8

# For CLI, ensure --rpc is used
python koboldcpp.py --rpc 192.168.1.101:50054 --rpc-server-backend Vulkan
```

**"Local GPUs not detected with RPC"** (ROCm registry fix):
```bash
# This is now fixed in gpttype_adapter.cpp - ROCm devices are properly detected
# Make sure you rebuilt with the latest code
make clean
make rpc-full-all -j8
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

---

## Usage

### RPC Server Mode

#### Method 1: From GUI (Recommended)

1. Open `python koboldcpp.py`
2. Go to **RPC Server** tab (located between "Loaded Files" and "Network")
3. Check **"Start RPC Server Mode"**
4. Configure:
   - **RPC Server Backend**: `Auto-detect` (recommended), `Vulkan`, `hipBLAS (ROCm)`, or `CUDA`
   - **Listening IP Address**: `0.0.0.0` (all interfaces) or `127.0.0.1` (localhost only)
   - **Listening Port**: `50053` (default)
   - **RPC Devices**: `VULKAN0,VULKAN1` or `ROCm0,ROCm1` (leave empty for auto-detect)
5. Check **"Allow Launch Without Models"** if no model file needed
6. Launch

**Backend Selection Guide**:
- **Auto-detect**: Automatically selects the best available backend (tries HIPBLAS first, then Vulkan, then CUDA)
- **Vulkan**: Uses `rpc-server-vulkan` with `VULKAN0`, `VULKAN1` device names
- **hipBLAS (ROCm)**: Uses `rpc-server-hip` with `ROCm0`, `ROCm1` device names
- **CUDA**: Uses `rpc-server-cuda` with `CUDA0`, `CUDA1` device names

**Device Name Conversion**: If you specify `VULKAN0,VULKAN1` but select "hipBLAS (ROCm)" backend, the system automatically converts to `ROCm0,ROCm1`.

#### Method 2: From Command Line

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
- `-H` or `--rpc-host` - Host IP (use private LAN IP, NOT 0.0.0.0 for production)
- `--rpc-port` - Port number (default: 50053)
- `--rpc-devices` - GPU devices (VULKAN0, VULKAN1, HIP0, CUDA0, ROCm0, etc.)
- `--rpc-server-backend` - Backend selection (Auto-detect, Vulkan, hipBLAS (ROCm), CUDA)

### Start RPC Client

#### Method 1: From GUI

1. Select backend from dropdown (Use Vulkan + RPC, Use hipBLAS + RPC, etc.)
2. Enter RPC server endpoint in **"RPC Server Endpoint"** field (appears below Backend dropdown)
3. The field is shared between Quick Launch and Hardware tabs - entering in one updates both

#### Method 2: From Command Line

```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --port 5001
```

**Options**:
- `--rpc` or `--userpc` - RPC server endpoints (comma-separated for multiple)
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
    --device VULKAN0,RPC0,RPC1,VULKAN1 \
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

### HIPBLAS Backend (Triple Naming Support)

When using HIPBLAS libraries (`koboldcpp_hipblas.so`, `koboldcpp_hipblas_rpc.so`), you can use **any** of these names for the same physical device:

| Physical Device | Valid Names |
|----------------|-------------|
| HIP Device 0 | `HIP0`, `CUDA0`, `ROCm0` |
| HIP Device 1 | `HIP1`, `CUDA1`, `ROCm1` |

**Triple naming**: `HIP0` = `CUDA0` = `ROCm0` (all refer to the same HIP device 0)

**All equivalent**:
```bash
python koboldcpp.py --device HIP0,RPC0,HIP1 --rpc 192.168.1.101:50054
python koboldcpp.py --device CUDA0,RPC0,CUDA1 --rpc 192.168.1.101:50054
python koboldcpp.py --device ROCm0,RPC0,ROCm1 --rpc 192.168.1.101:50054
```

### Vulkan Backend

When using Vulkan libraries (`koboldcpp_vulkan.so`, `koboldcpp_rpc.so`):

| Physical Device | Valid Names |
|----------------|-------------|
| Vulkan Device 0 | `VULKAN0` |
| Vulkan Device 1 | `VULKAN1` |

### CUDA Backend (NVIDIA)

When using CUDA libraries (`koboldcpp_cublas.so`, `koboldcpp_cublas_rpc.so`):

| Physical Device | Valid Names |
|----------------|-------------|
| CUDA Device 0 | `CUDA0`, `HIP0` |
| CUDA Device 1 | `CUDA1`, `HIP1` |

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

### How Backend Selection Works

When you select an option from the dropdown:
1. **Vulkan options** → Loads `koboldcpp_vulkan.so` or `koboldcpp_rpc.so`
2. **HIPBLAS options** → Loads `koboldcpp_hipblas.so` or `koboldcpp_hipblas_rpc.so`
3. **CUDA options** → Loads `koboldcpp_cublas.so` or `koboldcpp_cublas_rpc.so`

**Library Selection Priority**: When RPC is used, the selected backend variant from dropdown is preferred (HIPBLAS + RPC > CUDA + RPC > Vulkan + RPC).

---

## Examples

### Example 1: RPC Server with GUI (Vulkan)

**Server** (192.168.1.101):
1. Open `python koboldcpp.py`
2. Go to **RPC Server** tab
3. Check "Start RPC Server Mode"
4. Set **RPC Server Backend**: `Vulkan`
5. Set Listening IP Address: `0.0.0.0`
6. Set Listening Port: `50053`
7. Set RPC Devices: `VULKAN0,VULKAN1`
8. Launch

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-0.8B-Q8_0.gguf \
    --rpc 192.168.1.101:50053 --gpulayers 999
```

### Example 2: RPC Server with CLI (HIPBLAS)

**Server** (192.168.1.101):
```bash
python koboldcpp.py --start-rpc-server \
    --rpc-host 0.0.0.0 \
    --rpc-port 50053 \
    --rpc-devices ROCm0,ROCm1 \
    --rpc-server-backend "hipBLAS (ROCm)"
```

**Client**:
```bash
python koboldcpp.py --model Qwen3.5-0.8B-Q8_0.gguf \
    --rpc 192.168.1.101:50053 \
    --device HIP0,RPC0,HIP1 \
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
    --device ROCm0,ROCm1,RPC0,RPC1,RPC2 \
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

### Example 8: RPC Server Without Model (GUI)

**Server** (GUI):
1. Open `python koboldcpp.py`
2. Go to **RPC Server** tab
3. Check "Start RPC Server Mode"
4. Check **"Allow Launch Without Models"**
5. Launch

### Example 9: Device Name Conversion (GUI)

**Server** (GUI):
1. Open `python koboldcpp.py`
2. Go to **RPC Server** tab
3. Check "Start RPC Server Mode"
4. Set **RPC Server Backend**: `Vulkan`
5. Set RPC Devices: `VULKAN0,VULKAN1`
6. Launch

**Client**:
```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50053 \
    --device VULKAN0,RPC0,VULKAN1 \
    --gpulayers 999
```

**Output shows automatic conversion if backend mismatch**:
```
Converting VULKAN0 -> ROCm0
Converting VULKAN1 -> ROCm1
Devices to expose: ROCm0, ROCm1
```

---

## RPC Server Tab (GUI)

The RPC Server tab is located between "Loaded Files" and "Network" tabs in the GUI.

### Fields

**Start RPC Server Mode** (Checkbox)
- Starts KoboldCPP in RPC server mode
- Exposes local GPUs for remote clients to use

**RPC Server Backend** (Dropdown)
- **Auto-detect**: Automatically selects best backend (recommended)
- **Vulkan**: Uses `rpc-server-vulkan` with `VULKAN0`, `VULKAN1` devices
- **hipBLAS (ROCm)**: Uses `rpc-server-hip` with `ROCm0`, `ROCm1` devices
- **CUDA**: Uses `rpc-server-cuda` with `CUDA0`, `CUDA1` devices

**Listening IP Address** (Text Input)
- Default: `0.0.0.0`
- IP address for RPC server to listen on
- Use `0.0.0.0` for all interfaces
- Use `127.0.0.1` for localhost only
- ⚠️ Using `0.0.0.0` accepts connections on all interfaces!

**Listening Port** (Text Input)
- Default: `50053`
- Port number for RPC server connections

**RPC Devices** (Text Input)
- Comma-separated list of GPUs to expose via RPC
- Example: `VULKAN0,VULKAN1` or `ROCm0,ROCm1`
- Leave empty to auto-detect all available devices
- Device names are automatically converted to match selected backend

**Allow Launch Without Models** (Checkbox)
- Allows starting RPC server without loading a model file
- Useful for dedicated RPC server machines

**Warning Message**
- Red 14pt bold text warning that RPC Server mode replaces the WebUI API
- Clients must connect via `--rpc`

---

## RPC Client Endpoint Field (Quick Launch & Hardware Tabs)

When selecting an RPC variant backend (Use Vulkan + RPC, Use hipBLAS + RPC, Use CUDA + RPC), a **"RPC Server Endpoint"** field appears below the Backend dropdown in both Quick Launch and Hardware tabs.

**Key Features**:
- **Shared Input**: Both tabs use the same input field (`rpc_endpoint_var` StringVar)
- **Auto-sync**: Entering an endpoint in one tab automatically updates the other
- **Auto-show/hide**: Field only appears when RPC variant backend is selected
- **Placeholder Text**: Shows "e.g. 192.168.1.101:50053"

**Implementation Details**:
- Quick Launch tab: `rpc_endpoint_label` and `rpc_endpoint_entry` at row=2
- Hardware tab: `rpc_endpoint_label_hw` and `rpc_endpoint_entry_hw` at row=2
- Both use `textvariable=rpc_endpoint_var` for shared state
- Visibility controlled in `changerunmode()` function based on selected backend

**Usage**:
1. Select an RPC variant backend from dropdown
2. Enter your RPC server endpoint (e.g., `192.168.1.101:50053`)
3. The field will persist in both tabs

---

## Config Save/Load (RPC Settings)

RPC settings are now saved and loaded in `.kcpps` config files.

**Saved Settings**:
- `rpc_server_mode` - Whether RPC Server mode was enabled
- `rpc_server_backend` - Selected RPC Server backend
- `rpc_host` - Listening IP address
- `rpc_port` - Listening port
- `rpc_devices` - RPC devices list
- `rpc_endpoint` - RPC server endpoint for client mode
- `userpc` - Legacy RPC endpoint storage (backward compatibility)

**Saving**: In `convert_args_to_template()`, RPC endpoint is saved as:
```python
savdict["rpc_endpoint"] = rpc_endpoint_var.get()
```

**Loading**: In `import_vars()`, RPC settings are loaded and applied:
```python
# Load from userpc (legacy) or rpc_endpoint
if "userpc" in dict:
    rpc_endpoint_var.set(dict["userpc"][0] if isinstance(dict["userpc"], list) else str(dict["userpc"]))
if "rpc_endpoint" in dict:
    rpc_endpoint_var.set(str(dict["rpc_endpoint"]))
```

**Variable Initialization**: `import_vars()` now initializes `has_rpc_in_device` and `has_rocm_in_device` to prevent NameError when loading configs:
```python
has_rpc_in_device = False
if args.device and "RPC" in args.device:
    has_rpc_in_device = True

has_rocm_in_device = False
if args.device and ("ROCm" in args.device or "HIP" in args.device):
    has_rocm_in_device = True
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
   python koboldcpp.py --device VULKAN0,RPC0,RPC1,VULKAN1 --rpc 192.168.1.101:50054
   ```

### "Local GPUs not detected with RPC" (FIXED)

This was caused by missing "ROCM" in the device registry check. Now fixed in `gpttype_adapter.cpp`.

**Rebuild**:
```bash
make clean
make rpc-full-all -j8
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

### "RPC Server binary not found"

```bash
# Build RPC servers
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan
# or
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip
```

### "ERROR: RPC server failed with exit code 1"

```bash
# Check if device names match backend
# Wrong: Using VULKAN0 with rpc-server-hip
# Correct: Using ROCm0 with rpc-server-hip

# Use RPC Server Backend dropdown to match backend
# Or use --rpc-server-backend CLI argument
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

### RPC Server CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `--start-rpc-server` | Launch RPC server mode | `--start-rpc-server` |
| `-H` or `--rpc-host` | Host IP address | `192.168.1.101` |
| `--rpc-port` | Port number | `50053` |
| `--rpc-devices` | GPUs to expose | `VULKAN0,VULKAN1` |
| `--rpc-server-backend` | Backend selection | `Vulkan`, `hipBLAS (ROCm)`, `CUDA`, `Auto-detect` |

### RPC Client Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--rpc` or `--userpc` | RPC endpoints | `192.168.1.101:50053` |
| `--model` | Model path | `/path/to/model.gguf` |
| `--gpulayers` | Layers to offload | `999` |
| `--tensor_split` | Layer distribution | `10 10 40 40` |
| `--device` | Device ordering | `VULKAN0,RPC0,VULKAN1` |
| `--port` | API port | `5001` |

---

## More Info

- **Quick Start**: `RPC_QUICKSTART.md`
- **Build Guide**: `RPC_BUILD_GUIDE.md`
- **Backend Compatibility**: `RPC_BACKEND_COMPATIBILITY.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`
- **Code Changes**: `RPC_koboldcpp.py_changes.md`

---

**License**: MIT  
**Version**: 1.111.2
