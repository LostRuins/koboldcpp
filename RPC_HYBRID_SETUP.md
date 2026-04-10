# RPC Hybrid Mode Setup Guide

## Overview

Hybrid mode allows you to use **both local GPUs AND remote RPC servers** simultaneously. When you start the client with RPC endpoints, it automatically:
1. Connects to RPC servers
2. Detects local GPUs
3. Combines all devices for layer distribution

## Your Scenario

```
Machine 1 (RPC Server): 2 GPUs
Machine 2 (Client): Local GPUs + connects to Machine 1
```

**Result**: Model layers distributed across ALL GPUs (local + remote)

---

## Setup Instructions

### Step 1: Start RPC Server (Machine 1)

On the machine with 2 GPUs:

```bash
# For Vulkan
./rpc-server-vulkan --endpoint 0.0.0.0:50054

# For CUDA
./rpc-server-cuda --endpoint 0.0.0.0:50054
```

**Important**: 
- Bind to `0.0.0.0` to allow remote connections
- Default port is `50054`
- Server automatically detects and exposes all GPUs

### Step 2: Start Client (Machine 2)

On the client machine that will load the GGUF file:

```bash
python koboldcpp.py \
  --model /path/to/your-model.gguf \
  --rpc 192.168.1.101:50054 \
  --gpulayers 999 \
  --contextsize 8192 \
  --port 5002
```

**Replace**:
- `/path/to/your-model.gguf` with your actual model path
- `192.168.1.101` with the RPC server's IP address

### What Happens Automatically

The client will:
1. Connect to RPC server at `192.168.1.101:50054`
2. Detect RPC server has 2 GPUs
3. Enumerate local GPUs on client machine
4. Combine all devices (e.g., 2 local + 2 remote = 4 total)
5. Distribute model layers across all 4 GPUs

---

## Console Output Example

When hybrid mode activates, you should see:

```
[RPC] Connecting to RPC server(s): 192.168.1.101:50054
[RPC] Adding RPC server: 192.168.1.101:50054
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Found RPC device 0: RPC@192.168.1.101:50054[0]
[RPC] Found RPC device 1: RPC@192.168.1.101:50054[1]
[RPC] Using 2 RPC device(s) for offloading
[RPC] Enumerating local GPU devices to use alongside RPC...
[RPC] Found local GPU device: VULKAN0 (registry: VULKAN)
[RPC] Found local GPU device: VULKAN1 (registry: VULKAN)
[RPC] Total devices for offloading: 4 (RPC + local GPUs)
[RPC] Using 4 device(s) for model offloading
```

---

## Advanced: Manual Layer Distribution

### Equal Distribution (All GPUs)

If you have 2 local GPUs and 2 remote GPUs (4 total):

```bash
python koboldcpp.py \
  --model model.gguf \
  --rpc 192.168.1.101:50054 \
  --tensor_split 25 25 25 25 \
  --gpulayers 999
```

### Power-Weighted Distribution

If remote GPUs are more powerful:

```bash
python koboldcpp.py \
  --model model.gguf \
  --rpc 192.168.1.101:50054 \
  --tensor_split 10 10 40 40 \
  --gpulayers 999
```

Distribution:
- Local GPU 0: 10%
- Local GPU 1: 10%
- Remote GPU 0: 40%
- Remote GPU 1: 40%

### Memory-Weighted Distribution

If local GPUs have more VRAM:

```bash
python koboldcpp.py \
  --model model.gguf \
  --rpc 192.168.1.101:50054 \
  --tensor_split 35 35 15 15 \
  --gpulayers 999
```

---

## Multiple RPC Servers

You can connect to multiple RPC servers:

```bash
python koboldcpp.py \
  --model model.gguf \
  --rpc 192.168.1.101:50054,192.168.1.102:50054 \
  --gpulayers 999
```

**Example**: 
- Client has 2 local GPUs
- Server 1 has 2 GPUs
- Server 2 has 2 GPUs
- **Total**: 6 GPUs working together

---

## Troubleshooting

### Hybrid Mode Not Activating

**Check**:
1. Ensure you're using `--gpulayers 999` or `--gpulayers -1` or `--gpulayers 0`
2. Verify RPC server is running and accessible
3. Check firewall allows port 50054

**Test RPC connection**:
```bash
telnet 192.168.1.101 50054
# or
nc -zv 192.168.1.101 50054
```

### Local GPUs Not Detected

**Check**:
1. Vulkan/CUDA backend is loaded
2. GPUs are visible to system:
   ```bash
   vulkaninfo  # For Vulkan
   nvidia-smi  # For CUDA
   ```

### RPC Server Not Detected

**On server machine**:
```bash
# Check server is listening
netstat -tlnp | grep 50054
# or
ss -tlnp | grep 50054
```

**Firewall rules**:
```bash
# Allow RPC port
sudo ufw allow 50054/tcp
# or
sudo iptables -A INPUT -p tcp --dport 50054 -j ACCEPT
```

### Performance Issues

**Check network latency**:
```bash
ping 192.168.1.101
# Should be < 5ms on LAN
```

**Try equal distribution first**:
```bash
--tensor_split 25 25 25 25
```

Then adjust based on each GPU's capabilities.

---

## Security

### ⚠️ CRITICAL

**NEVER expose RPC to public internet!**

### Safe Configurations

**Option 1: LAN Only (Recommended)**
```bash
# Server binds to specific LAN IP
./rpc-server-vulkan --endpoint 192.168.1.101:50054
```

**Option 2: Firewall Protection**
```bash
# Allow only from specific IP
sudo ufw allow from 192.168.1.0/24 to any port 50054 proto tcp
./rpc-server-vulkan --endpoint 0.0.0.0:50054
```

**Option 3: SSH Tunneling (Most Secure)**
```bash
# Create SSH tunnel from client to server
ssh -L 50054:localhost:50054 user@192.168.1.101

# Server binds to localhost only
./rpc-server-vulkan --endpoint 127.0.0.1:50054

# Client connects to localhost
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054
```

---

## Complete Example

### Network Setup
```
Machine 1 (Server): 192.168.1.101
  - GPU 0: RTX 3090 (24 GB)
  - GPU 1: RTX 3090 (24 GB)

Machine 2 (Client): 192.168.1.102
  - GPU 0: RTX 4090 (24 GB)
  - GPU 1: RTX 4090 (24 GB)
  
Model: Qwen3.5-397B-A17B-K_G_2.93.gguf
```

### Commands

**Machine 1 (RPC Server)**:
```bash
./rpc-server-vulkan --endpoint 0.0.0.0:50054
```

**Machine 2 (Client)**:
```bash
python koboldcpp.py \
  --model /models/Qwen3.5-397B-A17B-K_G_2.93.gguf \
  --rpc 192.168.1.101:50054 \
  --tensor_split 30 30 20 20 \
  --gpulayers 999 \
  --contextsize 262144 \
  --port 5002 \
  --quiet
```

**Distribution**:
- Client GPU 0 (RTX 4090): 30% - More powerful, gets more layers
- Client GPU 1 (RTX 4090): 30% - More powerful, gets more layers
- Server GPU 0 (RTX 3090): 20%
- Server GPU 1 (RTX 3090): 20%

---

## Verification

### Check Hybrid Mode is Active

Look for these console messages:
```
[RPC] Enumerating local GPU devices to use alongside RPC...
[RPC] Total devices for offloading: X (RPC + local GPUs)
```

### Check Layer Distribution

With `--tensor_split`, you should see:
```
[RPC] Manual layer distribution configured across X GPU devices
[RPC] Tensor split ratios: 30.0 30.0 20.0 20.0
```

### Monitor GPU Usage

**On client machine**:
```bash
nvtop  # For CUDA
radeontop  # For AMD
```

**On server machine**:
Same commands - you should see GPU activity during inference.

---

## Performance Expectations

### Hybrid Mode vs RPC Only

| Configuration | Speed (tokens/sec) | Memory Usage |
|---------------|-------------------|--------------|
| RPC only (2 GPUs) | ~28 t/s | Distributed on server |
| Hybrid (2 local + 2 remote) | ~45-55 t/s | Distributed on all |
| Local only (2 GPUs) | ~45 t/s | Distributed locally |

**Hybrid mode provides best performance** by utilizing all available resources.

---

## Quick Reference

### Minimal Command
```bash
python koboldcpp.py --model model.gguf --rpc SERVER_IP:50054 --gpulayers 999
```

### With Layer Distribution
```bash
python koboldcpp.py --model model.gguf --rpc SERVER_IP:50054 --tensor_split RATIO1 RATIO2 ... --gpulayers 999
```

### Multiple Servers
```bash
python koboldcpp.py --model model.gguf --rpc SERVER1:50054,SERVER2:50054 --gpulayers 999
```

---

## Additional Resources

- `RPC_MANUAL_LAYERS.md` - Per-server layer distribution
- `RPC_PER_GPU_DISTRIBUTION.md` - Per-GPU distribution
- `RPC_FINAL_SUMMARY.md` - Complete RPC documentation
- `RPC_QUICKSTART.md` - Quick start guide

---

**Last Updated**: 2026-04-10  
**Version**: 1.111.2  
**Status**: ✅ Hybrid Mode Working
