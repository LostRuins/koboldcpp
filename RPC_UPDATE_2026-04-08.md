# RPC Integration Updates - 2026-04-08

**Version**: koboldcpp_rpc_attempt (1.111.2)

## Latest Changes

### Fixed: RPC Device Enumeration

The RPC integration now properly connects to RPC servers and enumerates devices for offloading.

### Key Changes

1. **gpttype_adapter.cpp** - Added RPC connection and device enumeration:
   - Calls `ggml_backend_rpc_add_server()` for each endpoint
   - Enumerates RPC devices using `ggml_backend_dev_count()` and `ggml_backend_dev_get()`
   - Adds RPC devices to override list
   - Sets `model_params.n_gpu_layers = 999` for full offload

2. **expose.cpp** - Added RPC endpoint handling:
   - Sets `GGML_RPC_ENDPOINTS` environment variable
   - Prints debug message when RPC is used

3. **koboldcpp.py** - Python wrapper support:
   - `--rpc` / `--userpc` argument
   - `rpc_endpoints` field in structure
   - Auto-offload for RPC (gpulayers=999)

### Expected Output

**Client:**
```
[RPC] Connecting to RPC server(s): 192.168.1.16:50052
[RPC] Adding RPC server: 192.168.1.16:50052
[RPC] Enumerating RPC devices...
[RPC] Found RPC device 0: RPC[192.168.1.16:50052]
[RPC] Using 1 RPC device(s) for offloading
...
llama_context: backend_ptrs.size() = 2
```

**Server:**
```
Starting RPC server v3.6.1
  endpoint       : 192.168.1.16:50052
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)

Accepted client connection
```

## Testing Commands

**Server:**
```bash
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0 -c
```

**Client:**
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --gpulayers 999
```

## Status

✅ RPC server builds and starts  
✅ RPC client builds  
✅ Client connects to server  
✅ RPC devices enumerated  
✅ Devices added to override list  
🔄 Model loading with RPC (ready for test)

---

**Date**: 2026-04-08  
**Version**: 1.111.2
