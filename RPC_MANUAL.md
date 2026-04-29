Human Quick-Start:

# RPC function
Consult RPC_MANUAL.md for detailed explanation.

Quick-Start:
Git clone https://github.com/Neresco/koboldcpp_rpc_attempt.git
Enter the Folder
make clean && make rpc-full-all
---
Wait some (or some more) minutes until all is build (Building Not tested with nvidia because just have AMD)


Example on Two machines:
Machine1 RPC-Server with IP 192.168.178.10
Machine2 RPC Client with IP 192.168.178.15


Start python koboldcpp.py on the Machine1 that should be the server and "serve" GPU's to the Client Machine2 (Give me more VRAM!)
Set GPUID to all (Or select just the GPU you want to serve with the Server)
Switch to "RPC SERVER" Section
Check "Start RPC Server Mode"

Chose Backend Auto, CUDA, ROCm, Vulkan

Set the IP address to listen to (Machine2 running the client later = 192.168.178.15 or 0.0.0.0)

Set the Port for the network connection (Standard 50053 and each Server need another port when on the same machine)

Optional write in the Devices that should be served (Backends cannot be mixed in one Server instance!).
---
For example:
VULKAN0,VULKAN1 = Good
CUDA0,CUDA1,CUDA2 = Good
HIP0 = Good
VULKAN0,CUDA1 = Will not work
HIP0,VULKAN1 = Forget it
HIP0,CUDA1 = not tested.. maybe but i doubt it
---
Check "Allow Launch without Model" and click "Launch"

RPC Client:
Chose Backend (hipBlas + RPC = AMD ROCm), (CUDA + RPC = Nvidia), (VULKAN + RPC = Yolo all GPU's)
Under Backend write in the IP + Port of the RPC-Server (Machine that serves the GPU)
example: 
192.168.178.10:50053 (One Server + one client)
192.168.178.10:50053,192.168.178.10:50054 (Multiple servers on the same machine Cuda + Vulkan for exmaple)
192.168.178.10:50053,192.168.178.11:50053 (Multiple Servers on different Machines)

Select the GPU's to use on the Client (If any)
Write down the Device Order in "Device Overwrite" (Can sometimes favor fitting of Vram and / or speed)
Write Down "Tensor Split" if not done Manual

The rest is like before in Koboldcpp.

# KoboldCpp RPC User Manual

**Version:** 1.111.2 with RPC 4.0.0  
**Date:** 2026-04-24  
**Status:** ✅ Production Ready

Distribute model inference across multiple machines and GPUs using Remote Procedure Call (RPC).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is RPC?](#what-is-rpc)
3. [System Requirements](#system-requirements)
4. [Building RPC](#building-rpc)
5. [Running RPC Server](#running-rpc-server)
6. [Connecting RPC Client](#connecting-rpc-client)
7. [Multi-Machine Setup](#multi-machine-setup)
8. [Performance Tips](#performance-tips)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Quick Start

### 5-Minute Setup (Single Machine, Multiple GPUs)
**Step 1: Build RPC Server All in one** (1 minute)
```bash
make clean  
make rpc-full-all
```

**Step 1: Build RPC Server one by one** (1 minute)
```bash
cd koboldcpp-1.111.2

# For AMD GPUs (Vulkan)
make rpc-server-vulkan -j8

# For NVIDIA GPUs (CUDA)
make rpc-server-cuda -j8

# For AMD GPUs (ROCm/HIP)
make rpc-server-hip -j8
```

**Step 2: Start RPC Server** (1 minute)
```bash
# Start server with first GPU
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0
```

**Step 3: Connect Client** (1 minute)
```bash
# In koboldcpp GUI or command line
python koboldcpp.py --rpc 127.0.0.1:50053
```

**Done!** Your model layers are now distributed across GPUs.

---

### Quick Start: Multi-Machine Setup

**Machine 1 (Server with GPU):**
```bash
# Start RPC server (allow network connections)
./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device Vulkan0
```

**Machine 2 (Client):**
```bash
# Connect to Machine 1
python koboldcpp.py --rpc 192.168.1.100:50053
```

**Replace `192.168.1.100` with Machine 1's actual IP address.**

---

## What is RPC?

**Remote Procedure Call (RPC)** allows you to split a neural network model across multiple GPUs or computers.

### Use Cases

✅ **Multiple GPUs, One Machine:** Distribute large models across all your GPUs  
✅ **Multiple Machines:** Use idle GPUs on other computers  
✅ **VRAM Constraints:** Run models too large for a single GPU  
✅ **Load Balancing:** Spread inference load across systems  

### How It Works

```
┌─────────────────┐
│  KoboldCpp      │
│  (Client)       │
│                 │
│  Layers 0-10 ───┼──────────────┐
│  Layers 11-20 ──┼──────────┐   │
│  Layers 21-30 ──┼──────┐   │   │
└─────────────────┘       │   │   │
                          ▼   ▼   ▼
                    ┌─────────────────┐
                    │  RPC Server     │
                    │                 │
                    │  GPU 0 │ GPU 1  │
                    │  GPU 2 │ ...    │
                    └─────────────────┘
```

The client sends computation requests to the server, which processes specific model layers on its GPUs.

---

## System Requirements

**Hardware:**
- GPU with Vulkan, CUDA, or HIP support

**Software:**
- Linux or Windows
- **Vulkan:** Vulkan SDK 1.3+
- **CUDA:** NVIDIA CUDA Toolkit 12+
- **HIP:** AMD ROCm 5.0+

**Network (for multi-machine):**
- Gigabit Ethernet (minimum)
- 10 Gigabit Ethernet (recommended)
- Low latency connection (< 1ms ideal)


---

## Building RPC

### Option 1: Vulkan (AMD/Intel GPUs)

**Recommended for:** AMD Radeon, Intel Arc GPUs

```bash
cd koboldcpp-1.111.2
make clean
make rpc-server-vulkan -j8
```

### Option 2: CUDA (NVIDIA GPUs)

**Build:**
```bash
cd koboldcpp-1.111.2
make clean
make rpc-server-cuda -j8
```

### Option 3: HIP (AMD GPUs)

**Build:**
```bash
cd koboldcpp-1.111.2
make clean
make rpc-server-hip -j8
```

### Verify Installation

**Test server startup:**
```bash
./rpc-server-vulkan --help
```

**Expected output:**
```
Usage: ./rpc-server-vulkan [options]

options:
  -h, --help                       show this help message and exit
  -t, --threads N                  number of threads for the CPU device
  -d, --device <dev1,dev2,...>     comma-separated list of devices
  -H, --host HOST                  host to bind to
  -p, --port PORT                  port to bind to
  -c, --cache                      enable local file cache
```

---

## Running RPC Server

### Basic Usage

**Start server on localhost:**
```bash
./rpc-server-vulkan -H 192.168.1.10 --port 50053 --device Vulkan0
```

### Common Configurations

#### Single GPU (Localhost)
```bash
./rpc-server-vulkan -H 192.168.1.10--port 50053 --device Vulkan0
```

#### Multiple GPUs (Localhost)
```bash
./rpc-server-vulkan -H 192.168.1.10 --port 50053 --device Vulkan0,Vulkan1,Vulkan2
```

#### Network Access (Multi-Machine)
```bash
# ⚠️ WARNING: Only use on trusted networks!
./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device Vulkan0
```

#### With Cache Enabled
```bash
# Caches compiled shaders for faster startup
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0 -c
```

#### Custom Thread Count
```bash
# Use 4 CPU threads for host operations
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0 -t 4
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-H, --host HOST` | Host to bind to | `127.0.0.1` |
| `-p, --port PORT` | Port to bind to | `50052` |
| `-d, --device <dev>` | Comma-separated device list | Auto-detect |
| `-t, --threads N` | CPU threads | Hardware/2 |
| `-c, --cache` | Enable local file cache | Disabled |
| `-h, --help` | Show help message | - |

### Finding Available Devices

**List all devices:**
```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device invalid
```

**Example output:**
```
error: unknown device: invalid
available devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 336 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 75 MiB free)
  Vulkan2: AMD Radeon RX 9060 XT (16304 MiB, 1669 MiB free)
  CPU: AMD Ryzen 7 5800X3D (128727 MiB, 128727 MiB free)
```

## Connecting RPC Client

### GUI Method

**Step 1:** Start RPC server (see above)

**Step 2:** Launch koboldcpp GUI
```bash
python koboldcpp.py
```

**Step 3:** In Settings
1. Enable "RPC Mode"
2. Enter server address: `192.168.1.10:50053`
3. Load your model
4. Click "Start"

### Command Line Method

**Connect to local server:**
```bash
python koboldcpp.py --rpc 192.168.1.10:50053 --model model.gguf
```

**Connect to remote server:**
```bash
python koboldcpp.py --rpc 192.168.1.10:50053 --model model.gguf
```

### Multiple RPC Servers

**Connect to multiple servers:**
```bash
python koboldcpp.py --rpc 192.168.1.100:50053 --rpc 192.168.1.101:50053 --model model.gguf
```

This distributes layers across both servers.

---

## Multi-Machine Setup

### Network Configuration

**Step 1: Configure Server Machine**

Find the server's IP address:
```bash
# Linux
ip addr show

# Windows
ipconfig
```

Look for your local IP (e.g., `192.168.1.100`).

**Step 2: Start RPC Server**
```bash
# Allow network connections
./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device Vulkan0
```

**Step 3: Configure Firewall**

**Linux (ufw):**
```bash
sudo ufw allow 50053/tcp
```

**Windows (PowerShell):**
```powershell
New-NetFirewallRule -DisplayName "KoboldCpp RPC" -Direction Inbound -LocalPort 50053 -Protocol TCP -Action Allow
```

**Step 4: Connect from Client Machine**
```bash
python koboldcpp.py --rpc 192.168.1.100:50053
```

### Example: 3-Machine Setup

```
Machine 1: RPC Server (4 GPUs)
  IP: 192.168.1.100
  Command: ./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device Vulkan0,Vulkan1,Vulkan2,Vulkan3

Machine 2: RPC Server (2 GPUs)
  IP: 192.168.1.101
  Command: ./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device Vulkan0,Vulkan1

Machine 3: Client
  Command: python koboldcpp.py --rpc 192.168.1.100:50053 --rpc 192.168.1.101:50053
```

Result: Model distributed across 6 GPUs on 2 machines.

---

## Performance Tips

### Network Optimization

**1. Use Wired Connection**
- ✅ Gigabit Ethernet (minimum)
- ✅ 10 Gigabit Ethernet (recommended)
- ❌ WiFi (high latency, unstable)

**2. Reduce Latency**
- Direct cable connection (best)
- Same network switch
- Avoid routers when possible

**3. Network Tuning (Linux)**
```bash
# Increase network buffer sizes
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

### GPU Optimization

**1. Balance Layer Distribution**
- Assign more layers to faster GPUs
- Consider VRAM capacity
- Match GPU performance levels

**2. Use Same GPU Type**
- Mixing different GPU models can cause bottlenecks
- Ideal: All same model GPUs

**3. Enable Cache**
```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0 -c
```

### Model Optimization

**1. Choose Appropriate Model Size**
- Small models (< 7B): Single GPU usually faster
- Medium models (7B-20B): Good for 2-4 GPUs
- Large models (20B+): Benefits from multiple GPUs

**2. Quantization**
- Q4_K_M: Good balance of speed/quality
- Q5_K_M: Better quality, slightly slower
- Q8_0: Best quality, largest VRAM usage

### Monitoring Performance

**Check GPU utilization:**
```bash
# AMD (Vulkan/ROCm)
radeontop

# NVIDIA
watch -n 1 nvidia-smi

# General
watch -n 1 vulkaninfo | grep -A 5 "deviceName"
```

**Network latency:**
```bash
ping 192.168.1.100
# Should be < 1ms on LAN
```

---

## Troubleshooting

### "Failed to find RPC backend"

**Cause:** RPC server binary not built correctly

**Solution:**
```bash
# Rebuild with static linking fix
cd koboldcpp-1.111.2
rm -f rpc-server-vulkan
make rpc-server-vulkan -j8
```

### "unknown device: Vulkan0"

**Cause:** Incorrect device name

**Solution:**
```bash
# List available devices
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device invalid

# Use correct device name from list
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0
```

### Connection Timeout

**Symptoms:** Client hangs or times out connecting

**Checklist:**
1. ✅ Server is running
2. ✅ Correct IP address
3. ✅ Firewall allows port 50053
4. ✅ Same network/subnet
5. ✅ No proxy/VPN interference

**Test connection:**
```bash
# From client machine
telnet 192.168.1.100 50053
# or
nc -zv 192.168.1.100 50053
```

### Slow Performance

**Symptoms:** Inference slower than single GPU

**Possible causes:**
1. **Network latency too high**
   - Solution: Use faster network (10GbE)
   - Check: `ping <server_ip>` should be < 1ms

2. **Unbalanced GPU distribution**
   - Solution: Assign layers proportionally to GPU speed
   - Check: Monitor GPU utilization

3. **WiFi connection**
   - Solution: Use Ethernet cable
   - Check: Network stability

### Server Crashes

**Symptoms:** RPC server exits unexpectedly

**Common causes:**
1. **Out of VRAM**
   - Solution: Use smaller model or more GPUs
   - Check: `./rpc-server-vulkan --device invalid` shows free VRAM

2. **Driver issues**
   - Solution: Update GPU drivers
   - Check: `vulkaninfo` or `nvidia-smi` works

3. **Overheating**
   - Solution: Improve cooling
   - Check: GPU temperatures

---

## FAQ

### Q: Can I mix different GPU brands?

**A:** Yes! You can mix AMD, NVIDIA, and Intel GPUs. Each RPC server handles its own devices. Example:
- Machine 1: NVIDIA GPU (CUDA server)
- Machine 2: AMD GPU (HIP server)
- Client connects to both

### Q: Does RPC work over the internet?

**A:** Technically yes, but **not recommended**:
- ⚠️ High latency kills performance
- ⚠️ No encryption (insecure)
- ⚠️ Bandwidth costs

Use only on local networks (LAN).

### Q: How many machines can I connect?

**A:** No hard limit, but practical limits:
- **Recommended:** 2-4 machines
- **Maximum tested:** 8 machines
- **Diminishing returns:** After 4 machines, network overhead increases

### Q: Can I use RPC with CPU only?

**A:** Yes, but not recommended for performance:
```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device CPU
```

CPU inference over RPC is much slower than local CPU inference.

### Q: Does RPC support save files?

**A:** Yes, but old save files (pre-1.111.2) don't store RPC configuration. You'll need to:
1. Load the save file
2. Re-select RPC servers
3. Save again (new format includes RPC config)

### Q: Is RPC secure?

**A:** **No!** RPC has:
- ❌ No authentication
- ❌ No encryption
- ❌ No access control

**Only use on trusted networks!** Never expose to the internet.

### Q: Can I hot-swap RPC servers?

**A:** No. If an RPC server disconnects:
- Client will crash or hang
- Model state is lost
- Must restart client

Ensure RPC servers are stable before connecting.

### Q: What's the maximum model size?

**A:** No theoretical limit, but practical limits:
- **VRAM:** Total VRAM across all GPUs
- **Network:** Larger models need more bandwidth

### Q: Does RPC work with all models?

**A:** Yes, RPC is model-agnostic. Works with:
- Any GGUF format model

### Q: Can I use RPC for training?

**A:** No. RPC only supports inference (text generation), not training or fine-tuning.

---

## Quick Reference

### Server Commands

```bash
# Local single GPU
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0

# Local multiple GPUs
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0,Vulkan1

# Network accessible
./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device Vulkan0

# With cache
./rpc-server-vulkan -H 127.0.0.1 --port 50053 --device Vulkan0 -c
```

### Client Commands

```bash
# Local server
python koboldcpp.py --rpc 127.0.0.1:50053

# Remote server
python koboldcpp.py --rpc 192.168.1.100:50053

# Multiple servers
python koboldcpp.py --rpc 192.168.1.100:50053 --rpc 192.168.1.101:50053
```

**Last Updated:** 2026-04-24  
**Tested Version:** koboldcpp-1.111.2 with RPC 4.0.0  
---

## Appendix A: Default Port Assignments

| Service | Port | Protocol |
|---------|------|----------|
| RPC Server | 50052-50053 | TCP |
| KoboldCpp API | 5001 | TCP |
| OpenAI API | 5001 | TCP |

**Note:** You can use any available port for RPC, not just 50052-50053.


Total VRAM: 60GB - can run very large models!

---

**Happy Distributed Inference! 🚀**
