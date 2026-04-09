# KoboldCPP RPC - Complete Manual

**Version**: koboldcpp_rpc_attempt (1.111.2)  
**RPC Protocol**: 3.6.1  
**Last Updated**: 2026-04-09  
**Status**: ✅ Working - Hybrid Mode Supported

---

## Table of Contents

1. [Overview](#overview)
2. [What's New in v1.111.2](#whats-new-in-v1112)
3. [Architecture](#architecture)
4. [Prerequisites](#prerequisites)
5. [Building RPC Components](#building-rpc-components)
6. [Quick Start](#quick-start)
7. [Advanced Usage](#advanced-usage)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Security](#security)

---

## Overview

KoboldCPP includes **complete RPC (Remote Procedure Call) functionality** integrated from llama.cpp. This allows you to:

- **Distribute model inference** across multiple machines
- **Offload computation** to remote GPU servers
- **Use hybrid mode**: Local GPUs + Remote RPC servers simultaneously
- **Connect multiple clients** to a single RPC server
- **Use multiple RPC servers** from a single client

### Key Features

✅ **Standalone RPC Server** - Built directly into koboldcpp  
✅ **Vulkan GPU Support** - AMD, Intel, NVIDIA via Vulkan  
✅ **Hybrid Mode** - Automatic local GPU + RPC server combination  
✅ **Python Integration** - `--rpc` flag in koboldcpp.py  
✅ **Multi-Server** - Connect to multiple RPC servers  
✅ **Auto-Detection** - Local GPUs automatically detected and used  
✅ **Distributed Memory** - Split large models across multiple GPUs  

---

## What's New in v1.111.2

### Critical Fixes (2026-04-09)

#### 1. RPC Device Enumeration Fixed
**Before**: Devices not properly enumerated from RPC servers  
**After**: Devices correctly retrieved from server registry  
**Impact**: RPC servers now work reliably

#### 2. Hybrid Mode Added
**Before**: RPC OR local GPUs (mutually exclusive)  
**After**: RPC AND local GPUs used together automatically  
**Impact**: Maximum performance with all available GPUs

#### 3. Null Terminator Fix
**Before**: Segmentation faults during model load  
**After**: Proper null-terminated device arrays  
**Impact**: No more crashes, stable operation

#### 4. Auto-Offload Fixed
**Before**: Only triggered with `--gpulayers 0`  
**After**: Triggers with `--gpulayers 0` or `-1` (default)  
**Impact**: RPC works without manual configuration

### Performance Improvements

- **Automatic device detection** - No manual `--device` specification needed
- **Optimal device ordering** - RPC servers prioritized, local GPUs added
- **Memory distribution** - Models automatically split across all devices
- **Network optimization** - RPC devices listed first to minimize transfers

---

## Architecture

### Components

```
┌─────────────────┐         ┌─────────────────┐
│  RPC Client     │         │  RPC Client     │
│  (koboldcpp.py) │         │  (koboldcpp.py) │
│  + Local GPUs   │         │  (CPU only)     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │      ┌─────────────┐      │
         └──────┤   Network   │──────┘
                │  (TCP/IP)   │
                └─────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
┌────────▼────────┐         ┌────────▼────────┐
│  RPC Server     │         │  RPC Server     │
│  (GPU Machine)  │         │  (GPU Machine)  │
│  Vulkan/CUDA    │         │  Vulkan/CUDA    │
│  2-4 GPUs       │         │  1-2 GPUs       │
└─────────────────┘         └─────────────────┘
```

### Data Flow (Hybrid Mode)

1. **Client starts** with `--rpc` parameter
2. **Client connects** to RPC server(s)
3. **Servers advertise** available GPU devices
4. **Client enumerates** RPC devices from registry
5. **Client adds** local GPU devices automatically
6. **Model loads** across ALL devices (RPC + local)
7. **Inference runs** with distributed computation
8. **Results return** to client

### Device Naming

- **RPC devices**: `RPC0`, `RPC1`, `RPC2`, etc. (from servers)
- **Local Vulkan**: `VULKAN0`, `VULKAN1`, etc. (client GPUs)
- **Local CUDA**: `CUDA0`, `CUDA1`, etc. (client NVIDIA GPUs)
- **Local HIP**: `HIP0`, `HIP1`, etc. (client AMD ROCm)

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu/Debian recommended)
- **Architecture**: x86_64, ARM64
- **RAM**: 8GB minimum, 32GB+ recommended
- **Network**: 1 Gbps minimum, 10 Gbps recommended
- **Latency**: < 10ms for good performance

### Backend Dependencies

#### For Vulkan Support (Required for RPC)
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libvulkan-dev \
    vulkan-tools \
    glslc
```

#### Verify Vulkan Installation
```bash
vulkaninfo | grep "GPU id"
# Should list your Vulkan-capable GPUs
```

---

## Building RPC Components

### Quick Build (Vulkan)

```bash
cd koboldcpp_rpc_attempt

# Clean previous builds
make clean

# Build RPC client library (with Vulkan support)
make LLAMA_RPC=1 LLAMA_VULKAN=1 koboldcpp_rpc

# Build RPC server (with Vulkan support)
make LLAMA_RPC=1 LLAMA_VULKAN=1 rpc-server-vulkan
```

### Build Output

After successful build:
- `koboldcpp_rpc.so` - RPC client library (~10-15 MB)
- `rpc-server-vulkan` - RPC server binary (~50-70 MB)

### Verification

```bash
# Check RPC client library
ls -lh koboldcpp_rpc.so

# Check RPC server
ls -lh rpc-server-vulkan

# Test Vulkan installation
vulkaninfo | grep "GPU id"
```

---

## Quick Start

### Step 1: Start RPC Server

On the machine with GPUs (server):

```bash
./rpc-server-vulkan \
    -H 192.168.1.101 \
    --port 50054 \
    --device VULKAN0,VULKAN1 \
    -c
```

**Expected Output:**
```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT ...
ggml_vulkan: 1 = AMD Radeon RX 9060 XT ...
Starting RPC server v3.6.1
  endpoint       : 192.168.1.101:50054
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
```

**Server is now waiting for client connections.**

### Step 2: Start RPC Client

On the client machine (can be same or different machine):

```bash
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --port 5001
```

**Expected Output (Hybrid Mode):**
```
Loading Text Model: /path/to/model.gguf
[RPC] Connecting to RPC server(s): 192.168.1.101:50054
[RPC] Adding RPC server: 192.168.1.101:50054
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Found RPC device 0: RPC0
[RPC] Found RPC device 1: RPC1
[RPC] Using 2 RPC device(s) for offloading
[RPC] Enumerating local GPU devices to use alongside RPC...
[RPC] Found local GPU device: VULKAN0 (registry: Vulkan)
[RPC] Found local GPU device: VULKAN1 (registry: Vulkan)
[RPC] Total devices for offloading: 4 (RPC + local GPUs)
[RPC] Using 4 device(s) for model offloading
llama_model_load: offloading 999 layers to GPU
Starting KoboldCpp on http://localhost:5001
```

### Step 3: Connect to Web Interface

Open browser: `http://localhost:5001`

**You're now running inference with distributed GPUs!**

---

## Advanced Usage

### Multiple RPC Servers

Connect to multiple servers simultaneously:

```bash
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.102:50054 \
    --gpulayers 999
```

### Explicit Device Selection

Override automatic device selection:

```bash
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,RPC1 \
    --gpulayers 999
```

### Tensor Splitting

Control how model layers are distributed:

```bash
# Split across 3 devices: 40% to local, 30% to RPC0, 30% to RPC1
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --tensor-split 40 30 30
```

### Context Size and Memory

Adjust for available memory:

```bash
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --contextsize 8192 \
    --batchsize 512
```

### Low VRAM Mode

For servers with limited GPU memory:

```bash
./rpc-server-vulkan \
    -H 192.168.1.101 \
    --port 50054 \
    --device VULKAN0 \
    -c \
    --lowvram
```

---

## Performance Tuning

### Optimal Network Configuration

**Best Performance**:
- Gigabit Ethernet (1 Gbps) or faster
- Low latency (< 5ms ping)
- Same subnet (no router hops)

**Acceptable**:
- Fast WiFi (802.11ac/ax)
- Latency < 20ms

**Not Recommended**:
- Internet connections (high latency)
- WiFi with > 50ms latency

### Memory Requirements

**Model Size Guidelines**:

| Model Size | Quantization | Minimum VRAM | Recommended VRAM |
|------------|--------------|--------------|------------------|
| 0.5B       | Q4_K_M       | 1 GB         | 2 GB             |
| 0.5B       | Q8_0         | 1 GB         | 2 GB             |
| 1B         | Q4_K_M       | 2 GB         | 4 GB             |
| 7B         | Q4_K_M       | 6 GB         | 8 GB             |
| 13B        | Q4_K_M       | 10 GB        | 14 GB            |
| 70B        | Q4_K_M       | 42 GB        | 48 GB            |

**Note**: With RPC, memory can be distributed across multiple GPUs.

### Device Distribution Strategies

**Strategy 1: Balanced Distribution**
```bash
# Equal split across 3 devices
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor-split 33 33 34
```

**Strategy 2: Memory-Optimized**
```bash
# More layers on GPUs with more VRAM
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor-split 20 40 40
```

**Strategy 3: Latency-Optimized**
```bash
# More layers on local GPUs (lower latency)
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor-split 60 20 20
```

### Batch Size Tuning

**Higher Batch Size**:
- Better throughput
- More VRAM usage
- Good for multiple concurrent users

**Lower Batch Size**:
- Lower latency
- Less VRAM usage
- Good for single user

```bash
# High throughput (multiple users)
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --batchsize 2048 \
    --ubatchsize 512

# Low latency (single user)
python koboldcpp.py \
    --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --batchsize 512 \
    --ubatchsize 256
```

---

## Troubleshooting

### Server Won't Start

**Symptom**: `rpc-server-vulkan` fails to start

**Solutions**:
1. Check Vulkan installation:
   ```bash
   vulkaninfo | grep "GPU id"
   ```
2. Verify port is available:
   ```bash
   ss -tlnp | grep 50054
   ```
3. Check firewall:
   ```bash
   sudo ufw allow 50054/tcp
   ```

### Client Can't Connect

**Symptom**: `[RPC] WARNING: Failed to connect to RPC server`

**Solutions**:
1. Verify server is running:
   ```bash
   ps aux | grep rpc-server
   ```
2. Test network connectivity:
   ```bash
   ping 192.168.1.101
   telnet 192.168.1.101 50054
   ```
3. Check server logs for errors

### No RPC Devices Found

**Symptom**: `[RPC] WARNING: No RPC devices found after connecting to server`

**Solutions**:
1. Verify server shows devices in output
2. Check server `--device` parameter includes valid GPUs
3. Try localhost test:
   ```bash
   # Server
   ./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c
   
   # Client
   python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054 --gpulayers 999
   ```

### Segmentation Fault

**Symptom**: Client crashes with segfault during model load

**Solutions**:
1. Ensure you have v1.111.2 with null terminator fix
2. Check available GPU memory:
   ```bash
   vulkaninfo | grep -A 5 "memoryHeaps"
   ```
3. Try smaller model or reduce context size
4. Verify all devices have sufficient memory

### Only CPU Used

**Symptom**: Model loads but only uses CPU, not GPUs

**Solutions**:
1. Check logs for "offloading X layers to GPU"
2. Verify `--gpulayers 999` is used
3. Check RPC devices are enumerated in logs
4. Ensure local GPUs are detected:
   ```bash
   rocminfo  # For AMD GPUs
   nvidia-smi  # For NVIDIA GPUs
   ```

### Slow Performance

**Symptom**: Generation speed is very slow (< 2 tokens/sec)

**Solutions**:
1. Check network latency:
   ```bash
   ping 192.168.1.101
   # Should be < 5ms on LAN
   ```
2. Use wired connection instead of WiFi
3. Reduce context size
4. Add more GPU devices (local or RPC)
5. Check GPU utilization:
   ```bash
   radeontop  # AMD
   nvtop  # NVIDIA
   ```

### Memory Errors

**Symptom**: "out of memory" or model fails to load

**Solutions**:
1. Reduce context size: `--contextsize 2048`
2. Enable low VRAM mode: `--lowvram`
3. Use smaller quantization (Q4_K_M instead of Q8_0)
4. Distribute across more devices
5. Check total free memory across all devices

---

## Security

### ⚠️ CRITICAL SECURITY WARNINGS

**NEVER expose RPC server to the public internet!**

RPC has **no authentication** and **no encryption**. Anyone who can connect can:
- Use your GPU resources
- Potentially execute code
- Access your network

### Safe Configurations

**✅ Safest - Localhost Only**:
```bash
./rpc-server-vulkan \
    -H 127.0.0.1 \
    --port 50054 \
    --device VULKAN0 \
    -c
```

**✅ Safe - Private LAN Only**:
```bash
./rpc-server-vulkan \
    -H 192.168.1.101 \
    --port 50054 \
    --device VULKAN0 \
    -c
```

**✅ Safer - Specific Subnet** (with firewall rules):
```bash
# Allow only from 192.168.1.0/24 subnet
sudo ufw allow from 192.168.1.0/24 to any port 50054 proto tcp

./rpc-server-vulkan \
    -H 0.0.0.0 \
    --port 50054 \
    --device VULKAN0 \
    -c
```

### ❌ DANGEROUS Configurations

**NEVER USE**:
```bash
./rpc-server-vulkan \
    -H 0.0.0.0 \
    --port 50054 \
    --device VULKAN0 \
    -c
```

This binds to ALL interfaces and is accessible from anywhere!

### Firewall Configuration

**Ubuntu/Debian (UFW)**:
```bash
# Allow only from specific IP
sudo ufw allow from 192.168.1.50 to any port 50054 proto tcp

# Allow from subnet
sudo ufw allow from 192.168.1.0/24 to any port 50054 proto tcp

# Deny all other access to port 50054
sudo ufw deny 50054/tcp
```

### Network Isolation

For production use:
1. Use a **separate VLAN** for RPC traffic
2. Implement **network segmentation**
3. Use **VPN** for cross-network RPC
4. Consider **SSH tunneling**:
   ```bash
   # Create SSH tunnel
   ssh -L 50054:localhost:50054 user@192.168.1.101
   
   # Connect client to localhost
   python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054
   ```

---

## Command Reference

### RPC Server Options

| Option | Description | Example |
|--------|-------------|---------|
| `-H` | Host IP address | `-H 192.168.1.101` |
| `--port` | Port number | `--port 50054` |
| `--device` | GPU devices to expose | `--device VULKAN0,VULKAN1` |
| `-c` | Use cache directory | `-c` |
| `--cache-dir` | Custom cache path | `--cache-dir /tmp/rpc-cache` |
| `--lowvram` | Low VRAM mode | `--lowvram` |
| `--verbose` | Verbose logging | `--verbose` |

### RPC Client Options

| Option | Description | Example |
|--------|-------------|---------|
| `--rpc` | RPC server endpoints | `--rpc 192.168.1.101:50054` |
| `--gpulayers` | Layers to offload | `--gpulayers 999` |
| `--device` | Specific devices | `--device VULKAN0,RPC0,RPC1` |
| `--tensor-split` | Layer distribution | `--tensor-split 40 30 30` |
| `--contextsize` | Context length | `--contextsize 8192` |
| `--batchsize` | Batch size | `--batchsize 512` |
| `--lowvram` | Low VRAM mode | `--lowvram` |

---

## Changelog

### 2026-04-09 (1.111.2) - Critical Fixes
- ✅ Fixed RPC device enumeration from registry
- ✅ Added automatic local GPU detection alongside RPC
- ✅ Fixed null terminator for device array
- ✅ Fixed auto-offload trigger for gpulayers=-1
- ✅ Added comprehensive logging
- ✅ Working hybrid mode (local + remote GPUs)

### 2026-04-08 (1.111.1) - Initial Integration
- Initial RPC integration from llama.cpp
- Basic RPC client/server functionality
- Manual device configuration required

---

## Support

**Documentation Issues**: Report at GitHub  
**Bugs**: Create issue with logs and reproduction steps  
**Feature Requests**: Discuss in GitHub Discussions  

**Logs Location**: Check client output for `[RPC]` prefixed messages  
**Server Logs**: Server outputs to stdout/stderr  

---

**License**: MIT  
**Version**: 1.111.2  
**Last Updated**: 2026-04-09
