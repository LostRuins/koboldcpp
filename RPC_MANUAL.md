# KoboldCPP RPC - Complete Manual

**Version**: 1.111.1  
**RPC Protocol**: 3.6.1  
**Last Updated**: 2026-04-07

## Table of Contents

1. [Overview](#overview)
2. [What's Included](#whats-included)
3. [Prerequisites](#prerequisites)
4. [Building RPC Components](#building-rpc-components)
5. [Usage Guide](#usage-guide)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Security](#security)

---

## Overview

KoboldCPP now includes **complete RPC (Remote Procedure Call) functionality** integrated from llama.cpp. This allows you to:

- **Distribute model inference** across multiple machines
- **Offload computation** to remote GPU servers
- **Connect multiple clients** to a single RPC server
- **Use multiple RPC servers** from a single client

All components are **standalone and self-contained** - no dependency on llama.cpp builds.

### Key Features

✅ **Standalone RPC Server** - Built directly into koboldcpp  
✅ **Vulkan GPU Support** - AMD, Intel, NVIDIA via Vulkan  
✅ **HIPBLAS Support** - AMD ROCm backend  
✅ **CUDA Support** - NVIDIA GPU backend  
✅ **Python Integration** - `--rpc` flag in koboldcpp.py  
✅ **Multi-Server** - Connect to multiple RPC servers  
✅ **Hybrid Mode** - Local GPU + Remote RPC  

---

## What's Included

### 1. RPC Client Library
- **File**: `koboldcpp_rpc.so`
- **Purpose**: Connect to remote RPC servers
- **Integration**: Built into koboldcpp Python wrapper

### 2. RPC Server Binaries

#### CPU-Only Server
- **File**: `rpc-server`
- **Use Case**: Testing, CPU-only systems
- **Limitation**: No GPU acceleration

#### Vulkan Server
- **File**: `rpc-server-vulkan`
- **Use Case**: AMD, Intel, NVIDIA GPUs via Vulkan
- **Requirement**: Vulkan SDK, glslc

#### HIPBLAS Server (ROCm)
- **File**: `rpc-server-hipblas`
- **Use Case**: AMD GPUs via ROCm
- **Requirement**: ROCm, hipblas

#### CUDA Server
- **File**: `rpc-server-cuda`
- **Use Case**: NVIDIA GPUs via CUDA
- **Requirement**: CUDA toolkit

### 3. Python Wrapper Support
- **Argument**: `--rpc` or `--userpc`
- **Format**: `--rpc <ip>:<port>`
- **Multiple**: `--rpc <ip1>:<port1> --rpc <ip2>:<port2>`

---

## Prerequisites

### System Requirements

- **OS**: Linux (Windows support via WSL)
- **Architecture**: x86_64, ARM64
- **RAM**: 8GB minimum, 32GB+ recommended for large models
- **Network**: 1 Gbps minimum, 10 Gbps recommended

### Backend Dependencies

#### For Vulkan Support
```bash
sudo apt-get update
sudo apt-get install glslc vulkan-tools libvulkan-dev mesa-vulkan-drivers
```

#### For HIPBLAS/ROCm Support
```bash
# Install ROCm from AMD
# https://rocm.docs.amd.com/en/latest/
sudo apt-get install rocblas hipblas-dev
```

#### For CUDA Support
```bash
# Install CUDA from NVIDIA
# https://developer.nvidia.com/cuda-downloads
sudo apt-get install nvidia-cuda-toolkit nvidia-cudnn
```

### Verify Installation

```bash
# Check Vulkan
vulkaninfo | head -20

# Check ROCm
rocminfo

# Check CUDA
nvidia-smi
```

---

## Building RPC Components

### Quick Build (Vulkan)

```bash
cd koboldcpp-1.111.1

# 1. Generate Vulkan shaders
make vulkan-shaders-gen

# 2. Build RPC client
make LLAMA_RPC=1 koboldcpp_rpc

# 3. Build RPC server with Vulkan
make LLAMA_VULKAN=1 rpc-server-vulkan
```

### Full Build (All Backends)

```bash
cd koboldcpp-1.111.1

# Build RPC client with all backends
make LLAMA_RPC=1 LLAMA_VULKAN=1 LLAMA_HIPBLAS=1 LLAMA_CUDA=1 koboldcpp_rpc

# Build all RPC servers
make LLAMA_VULKAN=1 rpc-server-vulkan
make LLAMA_HIPBLAS=1 rpc-server-hipblas
make LLAMA_CUDA=1 rpc-server-cuda
```

### Using Build Script

```bash
chmod +x build_rpc_full.sh
./build_rpc_full.sh
```

### Build Targets

| Target | Command | Output |
|--------|---------|--------|
| RPC Client | `make LLAMA_RPC=1 koboldcpp_rpc` | `koboldcpp_rpc.so` |
| RPC Server (CPU) | `make rpc-server` | `rpc-server` |
| RPC Server (Vulkan) | `make LLAMA_VULKAN=1 rpc-server-vulkan` | `rpc-server-vulkan` |
| RPC Server (HIPBLAS) | `make LLAMA_HIPBLAS=1 rpc-server-hipblas` | `rpc-server-hipblas` |
| RPC Server (CUDA) | `make LLAMA_CUDA=1 rpc-server-cuda` | `rpc-server-cuda` |

### Verify Build

```bash
# Check client library
ls -lh koboldcpp_rpc.so

# Check server binaries
ls -lh rpc-server*

# Test library loads
python3 -c "import ctypes; ctypes.CDLL('./koboldcpp_rpc.so'); print('OK')"
```

---

## Usage Guide

### Starting RPC Server

#### Vulkan Server Example

```bash
# Single GPU
./rpc-server-vulkan -H 0.0.0.0 --port 50052 --device VULKAN0 -c

# Multiple GPUs
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0,VULKAN1,VULKAN2 -c

# Localhost only (safer)
./rpc-server-vulkan -H 127.0.0.1 --port 50052 --device VULKAN0 -c
```

#### HIPBLAS Server Example

```bash
./rpc-server-hipblas -H 0.0.0.0 --port 50052 --device 0,1 -c
```

#### CUDA Server Example

```bash
./rpc-server-cuda -H 0.0.0.0 --port 50052 --device 0 -c
```

### Starting RPC Client

#### Basic Usage

```bash
# Single RPC server
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --gpulayers 999

# Multiple RPC servers
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.16:50052 \
    --rpc 192.168.1.17:50052 \
    --gpulayers 999
```

#### Hybrid Configuration (Local + Remote)

```bash
# Use local Vulkan GPU + remote RPC
python koboldcpp.py --model model.gguf \
    --usevulkan 0 \
    --gpulayers 20 \
    --rpc 192.168.1.16:50052
```

#### Command Line Options

```bash
# Full example with all options
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.16:50052 \
    --gpulayers 999 \
    --contextsize 8192 \
    --threads 8 \
    --host 0.0.0.0 \
    --port 5001
```

### Complete Example Session

**Terminal 1 - RPC Server (GPU Machine):**
```bash
cd koboldcpp-1.111.1
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0,VULKAN1 -c
```

**Output:**
```
Starting RPC server v3.6.1
  endpoint       : 192.168.1.16:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
```

**Terminal 2 - RPC Client (User Machine):**
```bash
cd koboldcpp-1.111.1
python koboldcpp.py --model /models/Qwen3.5-397B-A17B-K_G_2.93.gguf \
    --rpc 192.168.1.16:50052 \
    --gpulayers 999 \
    --port 5001
```

**Terminal 3 - Web Browser:**
```
http://localhost:5001
```

### Testing RPC Server

```bash
# Start server
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c

# Expected output:
Starting RPC server v3.6.1
  endpoint       : 127.0.0.1:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
```

---

## Advanced Configuration

### Multiple RPC Servers

Distribute load across multiple machines:

```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.16:50052 \
    --rpc 192.168.1.17:50052 \
    --rpc 192.168.1.18:50052 \
    --gpulayers 999
```

### Network Optimization

#### Jumbo Frames

Enable on both client and server:

```bash
# Set MTU to 9000
sudo ip link set dev eth0 mtu 9000
```

#### Firewall Rules

**Server Machine (Linux):**
```bash
# Allow RPC port
sudo ufw allow from 192.168.1.0/24 to any port 50052 proto tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 50052 -s 192.168.1.0/24 -j ACCEPT
```

#### Quality of Service

Prioritize RPC traffic:

```bash
sudo tc qdisc add dev eth0 root handle 1: prio
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
    match ip dport 50052 0xffff flowid 1:1
```

### Performance Tuning

#### Optimal Layer Distribution

```bash
# For 70B model with 2 RPC servers:
python koboldcpp.py --model 70B.gguf \
    --rpc server1:50052 \
    --rpc server2:50052 \
    --gpulayers 999 \
    --tensor_split 50,50
```

#### Batch Size Optimization

```bash
# Larger batches for better throughput
python koboldcpp.py --model model.gguf \
    --rpc server:50052 \
    --batchsize 2048 \
    --ubatchsize 512
```

### Custom RPC Endpoints

#### Non-Standard Ports

```bash
# Server on custom port
./rpc-server-vulkan -H 0.0.0.0 --port 9999 --device VULKAN0 -c

# Client connects to custom port
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:9999
```

---

## Troubleshooting

### Common Issues

#### "Failed to find RPC backend"

**Problem**: RPC server can't find backend (OLD ISSUE - FIXED)

**Solution**: This was fixed in v1.111.1 by calling `ggml_backend_rpc_start_server()` directly. Rebuild with latest code.

#### "No devices found"

**Problem**: RPC server can't find GPU devices

**Solution**:
```bash
# Check available devices
vulkaninfo | grep -A5 "GPU id"

# Use correct device names
./rpc-server-vulkan --device VULKAN0  # Not "Vulkan" or "GPU0"
```

#### "Connection refused"

**Problem**: Client can't connect to server

**Solutions**:
1. **Check server is running**: `ps aux | grep rpc-server`
2. **Verify IP/port**: `netstat -tlnp | grep 50052`
3. **Test connectivity**: `telnet 192.168.1.16 50052`
4. **Check firewall**: `sudo ufw status`

#### "Segmentation fault"

**Problem**: Client crashes on startup

**Solutions**:
1. **Start server BEFORE client** (critical!)
2. **Use `--gpulayers 999`** for full offload
3. **Check library compatibility**: `ldd koboldcpp_rpc.so`

#### "Vulkan shaders not found"

**Problem**: Build fails missing shader files

**Solution**:
```bash
# Generate shaders
make vulkan-shaders-gen
```

#### Slow Performance

**Problem**: Inference is slower than expected

**Solutions**:
1. **Check network speed**: `iperf3 -c server_ip`
2. **Use wired Ethernet** (not WiFi)
3. **Enable jumbo frames**: `sudo ip link set dev eth0 mtu 9000`
4. **Reduce network hops** (same switch/segment)
5. **Increase batch size**: `--batchsize 2048`

### Debug Mode

Enable verbose logging:

```bash
# Server debug
./rpc-server-vulkan -H 0.0.0.0 --port 50052 --device VULKAN0 -c --verbose

# Client debug
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --debugmode 2
```

### Log Files

```bash
# Server logs
tail -f /var/log/syslog | grep rpc-server

# Client logs
tail -f koboldcpp.log
```

### Testing Server

```bash
# Start server in background
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c &

# Check if listening
ss -tlnp | grep 50052

# Test connection
telnet 127.0.0.1 50052
```

---

## Security

### ⚠️ CRITICAL SECURITY WARNINGS

The RPC protocol is **NOT SECURE** and should **NEVER** be exposed to:

- ❌ Public internet
- ❌ Untrusted networks
- ❌ Open networks (coffee shops, airports)
- ❌ Networks with untrusted users

### Safe Configurations

#### ✅ Localhost Only (Safest)

```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50052 --device VULKAN0 -c
```

#### ✅ Private LAN (Acceptable)

```bash
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0 -c
```

With firewall:
```bash
sudo ufw allow from 192.168.1.0/24 to any port 50052 proto tcp
```

#### ❌ Public Network (DANGEROUS)

```bash
# DO NOT USE - UNSAFE!
./rpc-server-vulkan -H 0.0.0.0 --port 50052 --device VULKAN0 -c
```

### Security Best Practices

1. **Use localhost** whenever possible
2. **Firewall restrictions** - Only allow trusted IPs
3. **VPN tunnel** - Use WireGuard/OpenVPN for remote access
4. **SSH tunnel** - Forward RPC port through SSH
5. **Network isolation** - Use separate VLAN for RPC traffic

### SSH Tunnel Example

```bash
# Create secure tunnel
ssh -L 50052:localhost:50052 user@192.168.1.16

# Server binds to localhost
./rpc-server-vulkan -H 127.0.0.1 --port 50052 --device VULKAN0 -c

# Client connects to localhost
python koboldcpp.py --model model.gguf --rpc localhost:50052
```

---

## Appendix

### File Locations

```
koboldcpp-1.111.1/
├── koboldcpp_rpc.so           # RPC client library
├── rpc-server                 # CPU-only RPC server
├── rpc-server-vulkan          # Vulkan RPC server
├── rpc-server-hipblas         # HIPBLAS RPC server
├── rpc-server-cuda            # CUDA RPC server
├── tools/rpc-server.cpp       # Server source code
├── ggml/
│   ├── include/ggml-rpc.h     # RPC header
│   └── src/ggml-rpc/
│       └── ggml-rpc.cpp       # RPC implementation
└── koboldcpp.py               # Python wrapper
```

### Protocol Specification

- **Version**: 3.6.1
- **Transport**: TCP
- **Default Port**: 50052
- **Serialization**: Custom binary (packed structures)
- **Endianness**: Little-endian

### Environment Variables

```bash
# Override GPU device selection
export GGML_VULKAN_VISIBLE_DEVICES=0,1

# Override RPC timeout
export GGML_RPC_TIMEOUT=30000

# Enable debug logging
export GGML_RPC_DEBUG=1
```

### Build Fix (v1.111.1)

The RPC backend registration issue was fixed by calling the start server function directly:

```cpp
// OLD (didn't work for static backends):
ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
auto fn = (decltype(ggml_backend_rpc_start_server)*) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
fn(endpoint.c_str(), cache_dir, threads, devices.size(), devices.data());

// NEW (works for static backends):
ggml_backend_rpc_start_server(endpoint.c_str(), cache_dir, threads, devices.size(), devices.data());
```

### Related Documentation

- [RPC Integration Summary](../RPC_INTEGRATION_SUMMARY.md)
- [RPC Build Guide](RPC_BUILD_GUIDE.md)
- [RPC Server Guide](RPC_SERVER_GUIDE.md)
- [Quick Reference](../QUICK_REFERENCE.md)

### Support Resources

- **GitHub**: https://github.com/LostRuins/koboldcpp
- **Wiki**: https://github.com/LostRuins/koboldcpp/wiki
- **Discord**: KoboldAI Discord server
- **Issues**: https://github.com/LostRuins/koboldcpp/issues

### Version History

- **v1.111.1** (2026-04-07): Fixed RPC backend registration for static builds
- **v1.111.0** (2026-04-06): Initial RPC integration from llama.cpp

---

**License**: MIT  
**Authors**: KoboldCPP Team, based on llama.cpp RPC implementation  
**Version**: 1.111.1 (2026-04-07)
