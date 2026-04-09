# KoboldCPP RPC - Quick Start Guide

**Version**: 1.111.2  
**Last Updated**: 2026-04-09  
**Status**: ✅ Working

---

## 1-Minute Quick Start

### Step 1: Build (if not already built)

```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_RPC=1 LLAMA_VULKAN=1 koboldcpp_rpc
make LLAMA_RPC=1 LLAMA_VULKAN=1 rpc-server-vulkan
```

### Step 2: Start Server (on GPU machine)

```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0,VULKAN1 -c
```

### Step 3: Start Client (on any machine)

```bash
python koboldcpp.py --model /path/to/model.gguf --rpc 192.168.1.101:50054 --gpulayers 999 --port 5001
```

### Step 4: Open Browser

```
http://localhost:5001
```

**Done!** You're now using distributed GPUs!

---

## Expected Output

### Server Output
```
Starting RPC server v3.6.1
  endpoint       : 192.168.1.101:50054
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
```

### Client Output
```
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Found RPC device 0: RPC0
[RPC] Found RPC device 1: RPC1
[RPC] Enumerating local GPU devices...
[RPC] Found local GPU device: VULKAN0
[RPC] Total devices: 3 (RPC + local)
llama_model_load: offloading 999 layers to GPU
```

---

## Common Scenarios

### Scenario 1: Localhost Testing

**Terminal 1**:
```bash
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c
```

**Terminal 2**:
```bash
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054 --gpulayers 999
```

### Scenario 2: Multiple Servers

**Server 1**:
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0 -c
```

**Server 2**:
```bash
./rpc-server-vulkan -H 192.168.1.102 --port 50054 --device VULKAN0 -c
```

**Client**:
```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.102:50054 \
    --gpulayers 999
```

### Scenario 3: Hybrid Mode (Automatic)

**Server**:
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 --device VULKAN0 -c
```

**Client** (has local GPUs + uses RPC):
```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999
```

**Note**: Local GPUs are automatically detected and used alongside RPC!

---

## Troubleshooting Quick Fixes

### "Failed to connect"
```bash
# Check server is running
ps aux | grep rpc-server

# Test connection
ping 192.168.1.101
telnet 192.168.1.101 50054
```

### "No devices found"
```bash
# Check Vulkan
vulkaninfo | grep "GPU id"

# Use correct device names
./rpc-server-vulkan --device VULKAN0
```

### "Segmentation fault"
```bash
# Ensure latest build with fixes
make clean && make LLAMA_RPC=1 LLAMA_VULKAN=1 koboldcpp_rpc

# Use --gpulayers 999
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50054 --gpulayers 999
```

---

## Next Steps

- Read [RPC_MANUAL.md](RPC_MANUAL.md) for complete documentation
- Check [RPC_PORTING_GUIDE.md](RPC_PORTING_GUIDE.md) for technical details
- Join Discord for support

---

**License**: MIT  
**Version**: 1.111.2
