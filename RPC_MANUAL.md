# KoboldCPP RPC - Complete Manual

**Version**: 1.111.2  
**Last Updated**: 2026-04-11  
**Status**: ✅ Complete - All Features Working

---

## Overview

RPC (Remote Procedure Call) allows distributing model inference across multiple machines with GPUs. KoboldCPP RPC now supports:

**What Works**:
- ✅ RPC Server on GPU machines
- ✅ RPC Client connects to servers
- ✅ Multiple servers simultaneously
- ✅ GPU offloading to RPC servers
- ✅ **Hybrid mode** (local GPUs + RPC servers)
- ✅ **Manual tensor_split** for layer distribution control
- ✅ **Case-insensitive** device matching (Vulkan/RADV/CUDA/HIP/METAL)

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
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan
```

**Output**:
- `koboldcpp_rpc.so` (~67 MB) - Client library with Vulkan support
- `rpc-server-vulkan` (~68 MB) - Server binary

**Important**: Building with `LLAMA_VULKAN=1` is required for hybrid mode support.

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
- `--device` - GPU devices (VULKAN0, VULKAN1, CUDA0, HIP0, etc.)
- `-c` - Use cache directory

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
- `--tensor_split` - Manual layer distribution (optional)

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

**Device Order**: RPC devices first, then local GPUs in enumeration order.

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

### Example 3: Hybrid Mode

**Server** (192.168.1.101 with 3 GPUs):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1,VULKAN2 -c
```

**Client** (with 2 local GPUs):
```bash
python koboldcpp.py --model Qwen3.5-70B-Q4_K_M.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999
```

**Result**: 5 GPUs working together (3 remote + 2 local)

### Example 4: With Tensor Split

**Server** (192.168.1.101 with 2 GPUs):
```bash
./rpc-server-vulkan -H 192.168.1.101 --port 50054 \
    --device VULKAN0,VULKAN1 -c
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
- RPC2: 10%
- Vulkan0 (local): 10%
- Vulkan1 (local): 60%

### Example 5: Localhost Testing

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
python3 -c "import socket; s=socket.socket(); s.settimeout(5); print('OK' if s.connect_ex(('192.168.1.101', 50054))==0 else 'FAILED')"
```

### "No RPC devices found"

**Cause**: Server not advertising devices

**Solution**:
1. Verify server shows devices in output
2. Check server `--device` parameter
3. Test with localhost first
4. Ensure Vulkan is working: `vulkaninfo | grep "GPU id"`

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
make clean && make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
```

### Slow Performance

**Cause**: Network latency

**Solution**:
- Use wired Ethernet (not WiFi)
- Ensure low latency (< 5ms ping)
- Reduce context size
- Use same subnet

### "Local GPUs not detected"

**Cause**: RPC client built without Vulkan backend

**Solution**:
```bash
# Rebuild with Vulkan
make clean && make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc

# Verify size (~67 MB)
ls -lh koboldcpp_rpc.so
```

### "tensor_split not working"

**Cause**: Incorrect number of values

**Solution**:
```bash
# Count total devices: RPC devices + local GPUs
# Example: 2 RPC + 2 local = 4 values
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor_split 10 10 40 40 \
    --gpulayers 999
```

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
| Hybrid (LAN + local) | 50-60 t/s |
| RPC (slow network) | 5-15 t/s |

### Memory Requirements

| Model Size | Minimum VRAM |
|------------|--------------|
| 0.5B | 1-2 GB |
| 7B | 6-8 GB |
| 13B | 10-14 GB |
| 70B | 42-48 GB |
| 397B | 200+ GB |

**Note**: With RPC, memory is distributed across all devices (remote + local).

### Tensor Split Guidelines

- **Equal distribution**: `25 25 25 25` (for 4 devices)
- **Power-weighted**: More to powerful GPUs
- **Memory-weighted**: More to GPUs with more VRAM
- **Network-aware**: Less to remote GPUs on slow networks

Example for hybrid mode:
```bash
# 2 remote (slower) + 2 local (faster)
--tensor_split 10 10 40 40
```

---

## FAQ

**Q: Can I use local GPUs with RPC?**  
A: Yes! Hybrid mode automatically detects and uses local GPUs alongside RPC servers.

**Q: How many RPC servers can I connect to?**  
A: Unlimited, practical limit is ~10 servers.

**Q: Does RPC work over internet?**  
A: Technically yes, but NOT recommended (security + performance).

**Q: What GPU backends does RPC support?**  
A: Currently Vulkan. Device names: VULKAN0, VULKAN1, etc.

**Q: Can I mix different GPU models?**  
A: Yes, RPC works with different GPU models and vendors.

**Q: How do I control layer distribution?**  
A: Use `--tensor_split` with values for each device (RPC + local).

**Q: What's the device order for tensor_split?**  
A: RPC devices first (in server order), then local GPUs (in enumeration order).

**Q: Why is tensor_split truncated?**  
A: It shouldn't be anymore. Ensure you have the latest version with the fix.

---

## Advanced Features

### Device Enumeration Order

Devices are ordered as:
1. RPC Server 1, GPU 0
2. RPC Server 1, GPU 1
3. RPC Server 2, GPU 0
4. RPC Server 2, GPU 1
5. Local GPU 0 (Vulkan0)
6. Local GPU 1 (Vulkan1)
...

### Tensor Split Calculation

Values are ratios, not absolute layers. They are normalized to sum to 1.0.

Example: `10 10 40 40` becomes:
- Device 0: 10/110 = 9.1%
- Device 1: 10/110 = 9.1%
- Device 2: 40/110 = 36.4%
- Device 3: 40/110 = 36.4%

### Backend Support

**Currently Supported**:
- Vulkan (RADV, AMD, NVIDIA, Intel)
- RPC (remote Vulkan devices)

**Future**:
- CUDA (NVIDIA)
- HIP (AMD ROCm)
- METAL (Apple)

---

## More Information

- **Quick Start**: `RPC_QUICKSTART.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`

---

**License**: MIT  
**Version**: 1.111.2
