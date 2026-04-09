# RPC GPU Offload Fix - Testing Instructions

## Summary

Three critical issues have been fixed that prevented RPC from properly utilizing GPU resources:

1. **Device Override Overwrite** - RPC devices were being overwritten by local device overrides
2. **Incorrect RPC Device Detection** - Was checking buffer type name instead of backend registry name  
3. **RPC Auto-Offload Not Triggering** - Only triggered when gpulayers=0, but default is -1

## Files Modified

- `gpttype_adapter.cpp` - Fixed RPC device handling and detection
- `koboldcpp.py` - Fixed RPC auto-offload trigger condition

## Build Commands

### Option 1: Build RPC Client Library
```bash
cd koboldcpp_rpc_attempt
make LLAMA_RPC=1 koboldcpp_rpc
```

### Option 2: Build RPC Server with Vulkan
```bash
cd koboldcpp_rpc_attempt
make LLAMA_RPC=1 LLAMA_VULKAN=1 rpc-server-vulkan
```

### Option 3: Full Rebuild (Recommended)
```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_RPC=1 LLAMA_VULKAN=1 koboldcpp_rpc
make LLAMA_RPC=1 LLAMA_VULKAN=1 rpc-server-vulkan
```

## Testing Procedure

### Step 1: Start RPC Server (on GPU machine)
```bash
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0,VULKAN1 -c
```

**Expected Output:**
```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT ...
ggml_vulkan: 1 = AMD Radeon RX 9060 XT ...
Starting RPC server v3.6.1
  endpoint       : 192.168.1.16:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
```

### Step 2: Start RPC Client (on client machine)
```bash
python koboldcpp.py --model /path/to/model.gguf --rpc 192.168.1.16:50052 --gpulayers 999
```

**Expected Output:**
```
Initializing dynamic library: koboldcpp_rpc.so
Loading Text Model: /path/to/model.gguf
[RPC] Connecting to RPC server(s): 192.168.1.16:50052
[RPC] Adding RPC server: 192.168.1.16:50052
[RPC] Enumerating RPC devices...
[RPC] Found RPC device 0: RPC[192.168.1.16:50052] (registry: RPC)
[RPC] Using 1 RPC device(s) for offloading
...
llama_model_load: using RPC devices for offloading
llama_model_load: offloading 999 layers to GPU
Starting KoboldCpp on http://localhost:5001
```

### Step 3: Verify GPU Usage

#### On Server Side:
Monitor Vulkan GPU memory:
```bash
watch -n 1 'vulkaninfo | grep -A 5 "memoryHeaps"'
```

Or use `radeontop` for AMD GPUs:
```bash
sudo radeontop
```

**Expected:** GPU memory usage should increase during model loading.

#### On Client Side:
Check that local CPU usage is low during inference:
```bash
top -p $(pgrep koboldcpp)
```

**Expected:** CPU usage should be relatively low (< 20%) during generation.

### Step 4: Test Inference

Open browser to `http://localhost:5001` and send a prompt.

**Expected:**
- Generation speed should be reasonable (2-10 tokens/sec depending on model size and network)
- Server logs should show "Processing graph compute request"
- GPU memory should remain allocated during inference

## Verification Checklist

- [ ] Server starts and shows Vulkan devices
- [ ] Client connects to RPC server
- [ ] Client output shows "[RPC] Found RPC device" with registry name
- [ ] Client output shows "offloading X layers to GPU" (X > 0)
- [ ] Server GPU memory increases during model load
- [ ] Inference works and generates tokens
- [ ] Server logs show graph compute requests

## Troubleshooting

### Issue: "No RPC devices found"
**Solution:**
1. Check server is running: `ps aux | grep rpc-server`
2. Verify network connectivity: `ping 192.168.1.16`
3. Check port is open: `ss -tlnp | grep 50052`
4. Try localhost test: `--rpc 127.0.0.1:50052`

### Issue: "Segmentation fault" on client
**Solution:**
1. Ensure server starts BEFORE client
2. Wait for "Starting RPC server" message before starting client
3. Use `--gpulayers 999` explicitly
4. Check model file path is correct

### Issue: Only CPU used (no GPU offload)
**Solution:**
1. Verify RPC devices are enumerated (check client output)
2. Ensure `--gpulayers 999` is used
3. Check model is loading with RPC (look for "using RPC devices")
4. Try rebuilding with `make clean && make LLAMA_RPC=1 LLAMA_VULKAN=1 koboldcpp_rpc`

### Issue: Slow generation speed
**Solution:**
1. Check network latency: `ping 192.168.1.16`
2. Reduce model size or context length
3. Use faster network connection (wired vs wireless)
4. Consider using compression or quantization

## Hybrid Mode (Client + Server GPUs)

To use both client and server GPUs:

```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.16:50052 \
    --device VULKAN0 \
    --gpulayers 999
```

**Expected:** RPC devices listed first, then local devices added.

## Performance Expectations

### Local GPU (no RPC)
- Qwen3.5-0.8B-Q8_0: 20-50 tokens/sec
- Larger models: proportionally slower

### RPC (network dependent)
- Same model on LAN: 5-20 tokens/sec
- Depends on:
  - Network latency (ping time)
  - Network bandwidth (100Mbps vs 1Gbps)
  - Model size (more params = more data transfer)

## Security Notes

⚠️ **NEVER expose RPC server to public internet!**

Safe configurations:
```bash
# Localhost only (safest)
./rpc-server-vulkan -H 127.0.0.1 --port 50052 --device VULKAN0 -c

# Private LAN only
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0 -c
```

Dangerous (DO NOT USE):
```bash
./rpc-server-vulkan -H 0.0.0.0 --port 50052 --device VULKAN0 -c
```

## Additional Resources

- `RPC_QUICKSTART.md` - Quick start guide
- `RPC_MANUAL.md` - Full manual
- `RPC_PORTING_GUIDE.md` - Porting guide
- `RPC_FIX_2026-04-08.md` - Detailed fix documentation

---

**Status**: ✅ Fixes Applied, Ready for Testing  
**Date**: 2026-04-08  
**Version**: koboldcpp_rpc_attempt (1.111.2)
