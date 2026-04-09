# RPC Segfault Fix - 2026-04-09

## Issue
Segmentation fault during model loading after RPC devices are enumerated.

## Root Causes

### 1. Local GPU Detection Not Triggering
**Problem**: Code checked `inputs.devices_override == nullptr` but it was an empty string `""` instead.

**Fix**: Check for both nullptr and empty string:
```cpp
std::string dev_override_str = inputs.devices_override ? inputs.devices_override : "";
if(dev_override_str == "") {
    // Enumerate local GPUs
}
```

### 2. Missing Null Terminator
**Problem**: llama.cpp expects a **null-terminated** device array, but we weren't adding the null terminator.

**Fix**: Add nullptr to end of devices_override vector:
```cpp
devices_override.push_back(nullptr);
model_params.devices = devices_override.data();
```

## Changes Made

### File: gpttype_adapter.cpp

```cpp
// If using RPC but no local devices specified, also add local GPU devices
if(use_rpc && devices_override.size() > 0) {
    std::string dev_override_str = inputs.devices_override ? inputs.devices_override : "";
    if(dev_override_str == "") {
        // Enumerate local GPU devices and add them
        printf("[RPC] Enumerating local GPU devices to use alongside RPC...\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* dev = ggml_backend_dev_get(i);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            std::string reg_name = reg ? ggml_backend_reg_name(reg) : "";
            std::string dev_name = ggml_backend_dev_name(dev);
            // Add local GPU devices (not RPC, not CPU)
            if(reg_name.find("RPC") == std::string::npos && 
               reg_name.find("CPU") == std::string::npos &&
               (reg_name.find("VULKAN") != std::string::npos || 
                reg_name.find("CUDA") != std::string::npos ||
                reg_name.find("HIP") != std::string::npos)) {
                printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
                devices_override.push_back(dev);
            }
        }
        printf("[RPC] Total devices for offloading: %zu (RPC + local GPUs)\n", devices_override.size());
    }
}

// Apply device overrides if we have any devices
if(devices_override.size() > 0) {
    // Add null terminator for llama.cpp (expects null-terminated array)
    devices_override.push_back(nullptr);
    model_params.devices = devices_override.data();
    // Force GPU usage by setting n_gpu_layers to max when using RPC or custom devices
    if(use_rpc) {
        model_params.n_gpu_layers = 999;
    }
    printf("[RPC] Using %zu device(s) for model offloading\n", devices_override.size() - 1);
}
```

## Expected Output After Fix

```
Loading Text Model: /home/lunarbuntu/Downloads/Qwen3.5-0.8B-Q8_0.gguf
[RPC] Connecting to RPC server(s): 192.168.1.101:50054
[RPC] Adding RPC server: 192.168.1.101:50054
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Found RPC device 0: RPC0
[RPC] Found RPC device 1: RPC1
[RPC] Using 2 RPC device(s) for offloading
[RPC] Enumerating local GPU devices to use alongside RPC...
[RPC] Found local GPU device: VULKAN0 (registry: Vulkan)
[RPC] Found local GPU device: VULKAN1 (registry: Vulkan)
[RPC] Total devices for offloading: 4 (RPC + local GPUs)
[RPC] Using 4 device(s) for model offloading
llama_model_load_from_file_impl: using device RPC0 (192.168.1.101:50054) - 16234 MiB free
llama_model_load_from_file_impl: using device RPC1 (192.168.1.101:50054) - 16242 MiB free
llama_model_load_from_file_impl: using device VULKAN0 (AMD Radeon RX 9060 XT) - 15232 MiB free
llama_model_load_from_file_impl: using device VULKAN1 (Radeon 8060S Graphics) - 130048 MiB free
...
llama_model_load: offloading 999 layers to GPU
```

## Testing Commands

### Rebuild
```bash
cd koboldcpp_rpc_attempt
make clean
make LLAMA_RPC=1 LLAMA_VULKAN=1 koboldcpp_rpc
```

### Test
```bash
python koboldcpp.py --model /home/lunarbuntu/Downloads/Qwen3.5-0.8B-Q8_0.gguf \
    --rpc 192.168.1.101:50054 \
    --gpulayers 999 \
    --port 5002
```

## Verification Checklist

- [ ] "[RPC] Enumerating local GPU devices" appears in log
- [ ] Local GPUs are listed (VULKAN0, VULKAN1, etc.)
- [ ] "[RPC] Total devices for offloading: X" shows correct count
- [ ] No segmentation fault
- [ ] Model loads successfully
- [ ] "offloading 999 layers to GPU" appears
- [ ] Inference works

## Memory Requirements

For Qwen3.5-0.8B-Q8_0 (764 MB model):
- **Minimum**: ~1 GB free GPU memory total
- **Recommended**: ~2 GB free GPU memory total
- **With RPC**: Network latency affects speed, not memory

Your setup:
- RPC Server: 2x AMD Radeon RX 9060 XT (16 GB each)
- Client: 1x AMD Radeon RX 9060 XT (16 GB) + 1x Radeon 8060S Graphics (128 GB)
- **Total available**: ~160 GB GPU memory ✅

## Troubleshooting

### Still getting segfault?
1. Check total free GPU memory across all devices
2. Try with smaller model first
3. Reduce context size: `--contextsize 2048`
4. Check if model file is corrupted

### Local GPUs not showing?
1. Verify Vulkan is installed: `vulkaninfo | grep GPU`
2. Check GGML_USE_VULKAN is defined in build
3. Rebuild with: `make LLAMA_VULKAN=1 clean koboldcpp_rpc`

### RPC devices not showing?
1. Check server is running
2. Verify network connectivity
3. Check firewall rules
4. Try localhost test first

---

**Status**: ✅ Segfault Fix Applied  
**Date**: 2026-04-09  
**Version**: koboldcpp_rpc_attempt (1.111.2)
