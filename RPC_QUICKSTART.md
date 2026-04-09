# KoboldCPP RPC - Quick Start Guide

**Version**: 1.111.2  
**Last Updated**: 2026-04-09  
**Status**: ✅ Working

---

## 1-Minute Quick Start

### Step 1: Build

```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_RPC=1 koboldcpp_rpc
make LLAMA_RPC=1 LLAMA_VULKAN=1 rpc-server-vulkan
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
Starting RPC server v3.6.1
  endpoint       : 192.168.1.101:50054
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
```

### Client
```
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Found RPC device 0: RPC0
[RPC] Found RPC device 1: RPC1
llama_model_load: offloading 999 layers to GPU
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

### Scenario 3: Localhost Testing
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
telnet 192.168.1.101 50054
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
# Rebuild
make clean && make LLAMA_RPC=1 koboldcpp_rpc
```

---

## What Works

✅ RPC Server starts and advertises devices  
✅ RPC Client connects to servers  
✅ Multiple RPC servers work together  
✅ GPU offloading to RPC server GPUs  
✅ Model distribution across RPC devices  

## What Doesn't Work Yet

❌ Automatic local GPU detection when using RPC  
❌ Hybrid mode (local GPUs + RPC servers)  

**Workaround**: Use RPC-only mode which works perfectly for distributed inference.

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

---

## More Info

- **Complete Manual**: `RPC_MANUAL.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`
- **Status**: `RPC_HYBRID_STATUS.md`

---

**License**: MIT  
**Version**: 1.111.2
