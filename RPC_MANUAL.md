# KoboldCPP RPC - Manual

**Version**: 1.111.2  
**Last Updated**: 2026-04-09  
**Status**: ✅ Working

---

## Overview

RPC (Remote Procedure Call) allows distributing model inference across multiple machines with GPUs.

**What Works**:
- ✅ RPC Server on GPU machines
- ✅ RPC Client connects to servers
- ✅ Multiple servers simultaneously
- ✅ GPU offloading to RPC servers

**What Doesn't Work**:
- ❌ Automatic local GPU detection with RPC
- ❌ Hybrid mode (local + RPC)

---

## Build

### Prerequisites
```bash
sudo apt-get install build-essential cmake libvulkan-dev vulkan-tools glslc
```

### Build Commands
```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_RPC=1 koboldcpp_rpc
make LLAMA_RPC=1 LLAMA_VULKAN=1 rpc-server-vulkan
```

**Output**:
- `koboldcpp_rpc.so` - Client library
- `rpc-server-vulkan` - Server binary

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
- `--device` - GPU devices (VULKAN0, VULKAN1, etc.)
- `-c` - Use cache

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

### Multiple Servers

```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --gpulayers 999
```

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

### Example 3: Localhost Testing

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
telnet 192.168.1.101 50054
```

### "No RPC devices found"

**Cause**: Server not advertising devices

**Solution**:
1. Verify server shows devices in output
2. Check server `--device` parameter
3. Test with localhost first

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
make clean && make LLAMA_RPC=1 koboldcpp_rpc
```

### Slow Performance

**Cause**: Network latency

**Solution**:
- Use wired Ethernet (not WiFi)
- Ensure low latency (< 5ms ping)
- Reduce context size
- Use same subnet

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
| RPC (slow network) | 5-15 t/s |

### Memory Requirements

| Model Size | Minimum VRAM |
|------------|--------------|
| 0.5B | 1-2 GB |
| 7B | 6-8 GB |
| 13B | 10-14 GB |
| 70B | 42-48 GB |

**Note**: With RPC, memory is distributed across servers.

---

## FAQ

**Q: Can I use local GPUs with RPC?**  
A: Not automatically yet. Use RPC-only mode which works perfectly.

**Q: How many RPC servers can I connect to?**  
A: Unlimited, practical limit is ~10 servers.

**Q: Does RPC work over internet?**  
A: Technically yes, but NOT recommended (security + performance).

**Q: What GPU backends does RPC support?**  
A: Currently Vulkan. CUDA/HIP planned.

**Q: Can I mix different GPU models?**  
A: Yes, RPC works with different GPU models.

---

## More Information

- **Quick Start**: `RPC_QUICKSTART.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`
- **Hybrid Status**: `RPC_HYBRID_STATUS.md`

---

**License**: MIT  
**Version**: 1.111.2
