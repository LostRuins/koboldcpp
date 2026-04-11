# KoboldCPP RPC - Quick Start Guide

**Version**: 1.111.2  
**Last Updated**: 2026-04-11  
**Status**: ✅ Complete - All Features Working

---

## 1-Minute Quick Start

### Step 1: Build

```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan
```

### Step 2: Start Server (on GPU machine)

```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c
```

### Step 3: Start Client (on any machine)

```bash
python koboldcpp.py --model /path/to/model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --port 5001
```

### Step 4: Open Browser

```
http://localhost:5001
```

**Done!**

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
[RPC] Enumerating local GPU devices to use alongside RPC...
[RPC] Found local GPU device: Vulkan0 (registry: Vulkan)
[RPC] Found local GPU device: Vulkan1 (registry: Vulkan)
[RPC] Total devices for offloading: 4 (RPC + local GPUs)
llama_model_load: offloading 999 layers to GPU
load_tensors: RPC0[...] model buffer size = XXX MiB
load_tensors: RPC1[...] model buffer size = XXX MiB
load_tensors: Vulkan0 model buffer size = XXX MiB
load_tensors: Vulkan1 model buffer size = XXX MiB
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
    --gpulayers 999
```

**Result**: Uses both RPC server GPUs AND local client GPUs automatically!

### Scenario 4: With Tensor Split
```bash
# Server (192.168.1.101 with 2 GPUs)
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c

# Client (2 local GPUs + 2 remote = 4 total)
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor_split 10 10 40 40 \
    --gpulayers 999
```

**Distribution**:
- RPC0 (remote): 10%
- RPC1 (remote): 10%
- Vulkan0 (local): 40%
- Vulkan1 (local): 40%

### Scenario 5: Manual Device Ordering
```bash
# Server (192.168.1.101 with 2 GPUs)
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c

# Client: Reorder devices (local first, then RPC)
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,VULKAN1,RPC0,RPC1 \
    --gpulayers 999
```

**Result**: Local GPUs handle first layers, RPC handles later layers.

### Scenario 5: Localhost Testing
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

### "Segmentation fault"
```bash
# Ensure --gpulayers 999
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50054 --gpulayers 999
```

### "undefined symbol" errors
```bash
# Rebuild with Vulkan
make clean && make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
```

### "Local GPUs not detected"
```bash
# Verify hybrid build
ls -lh koboldcpp_rpc.so
# Should be ~67 MB (not ~12 MB RPC-only)

# Check Vulkan devices
vulkaninfo | grep -A 3 "deviceName"
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

---

## What Works

✅ RPC Server starts and advertises devices  
✅ RPC Client connects to servers  
✅ Multiple RPC servers work together  
✅ GPU offloading to RPC server GPUs  
✅ Model distribution across RPC devices  
✅ **Hybrid mode** (local GPUs + RPC servers)  
✅ **Manual tensor_split** for layer distribution  
✅ **Case-insensitive** device matching (Vulkan/RADV/CUDA/HIP)  
✅ **Manual device ordering** (--device argument)  
✅ **Device reordering** (mix RPC and local in any order)  

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

### Layer Distribution
- More powerful GPUs get higher tensor_split ratios
- Local GPUs typically faster (no network overhead)
- Example: Local 40%, Remote 60%

### Memory
- Model split across all devices
- Each device needs enough VRAM for its portion
- KV cache also distributed

---

## More Info

- **Complete Manual**: `RPC_MANUAL.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`

---

**License**: MIT  
**Version**: 1.111.2
