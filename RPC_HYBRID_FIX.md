# Hybrid RPC Client Build Guide

## Problem

The current `koboldcpp_rpc.so` is built **without Vulkan/CUDA backends**, so it cannot detect local GPUs. This breaks hybrid mode.

## Solution: Create Hybrid Build

You need to rebuild with **both RPC and Vulkan** backends included.

### Step 1: Modify Makefile

Add a new hybrid build target. Edit `Makefile` and add this after line 934:

```makefile
# Hybrid RPC + Vulkan build target
ifdef VULKAN_BUILD
ifdef RPC_BUILD
koboldcpp_rpc_hybrid: ggml_v4_vulkan.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter_vulkan.o ggml-rpc.o ggml-vulkan.o ggml-vulkan-shaders.o sdcpp_vulkan.o whispercpp_vulkan.o tts_default.o music_default.o embeddings_default.o llavaclip_vulkan.o llava.o ggml-backend_vulkan.o ggml-backend-reg_vulkan.o ggml-backend_default.o ggml-backend-reg_default.o ggml-repack.o $(OBJS_FULL) $(OBJS)
	$(VULKAN_BUILD)
else
koboldcpp_rpc_hybrid:
	$(DONOTHING)
endif
else
koboldcpp_rpc_hybrid:
	$(DONOTHING)
endif
```

### Step 2: Build Hybrid Library

```bash
# Clean previous builds
make clean

# Build with both Vulkan and RPC
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc_hybrid
```

This creates `koboldcpp_rpc_hybrid.so` with both backends.

### Step 3: Update koboldcpp.py

Edit `koboldcpp.py` line ~1700 to use the hybrid library:

```python
# Find this section and modify:
if args.userpc:
    libname = "koboldcpp_rpc_hybrid.so"  # Changed from koboldcpp_rpc.so
else:
    # ... rest of logic
```

Or add command-line detection:

```python
# Auto-select hybrid library when using RPC
if args.userpc:
    # Try hybrid first, fallback to regular RPC
    if os.path.exists("koboldcpp_rpc_hybrid.so"):
        libname = "koboldcpp_rpc_hybrid.so"
        print("[RPC] Using hybrid RPC library with Vulkan support")
    else:
        libname = "koboldcpp_rpc.so"
        print("[RPC] WARNING: Hybrid library not found, local GPUs may not be detected")
```

### Step 4: Test Hybrid Mode

```bash
python koboldcpp.py \
  --model /home/lunarbuntu/Downloads/Qwen3.5-0.8B-Q8_0.gguf \
  --rpc 192.168.1.101:50054 \
  --tensor_split 25 25 25 25 \
  --gpulayers 999
```

**Expected output:**
```
[RPC] Using hybrid RPC library with Vulkan support
[RPC] Connecting to RPC server(s): 192.168.1.101:50054
[RPC] Server 192.168.1.101:50054 has 2 devices
[RPC] Enumerating local GPU devices to use alongside RPC...
[RPC] Found local GPU device: VULKAN0 (registry: VULKAN)
[RPC] Found local GPU device: VULKAN1 (registry: VULKAN)
[RPC] Total devices for offloading: 4 (RPC + local GPUs)
```

---

## Alternative: Quick Fix (Modify Existing Target)

If you don't want to create a separate hybrid target, modify the existing `koboldcpp_rpc` target to include Vulkan:

### Edit Makefile Line 929:

**Before:**
```makefile
koboldcpp_rpc: ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter.o ggml-rpc.o sdcpp_default.o whispercpp_default.o tts_default.o music_default.o embeddings_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o ggml-repack.o $(OBJS_FULL) $(OBJS)
```

**After:**
```makefile
koboldcpp_rpc: ggml_v4_vulkan.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter_vulkan.o ggml-rpc.o ggml-vulkan.o ggml-vulkan-shaders.o sdcpp_vulkan.o whispercpp_vulkan.o tts_default.o music_default.o embeddings_default.o llavaclip_vulkan.o llava.o ggml-backend_vulkan.o ggml-backend-reg_vulkan.o ggml-backend_default.o ggml-backend-reg_default.o ggml-repack.o $(OBJS_FULL) $(OBJS)
```

Then rebuild:
```bash
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
```

**⚠️ Warning**: This makes the regular RPC build require Vulkan. If Vulkan isn't available, the build will fail.

---

## Why This Happens

The RPC client library needs to:
1. **Load Vulkan backend** - To detect local GPUs
2. **Load RPC backend** - To connect to remote servers
3. **Combine both** - For hybrid mode

Current `koboldcpp_rpc.so` only has #2, missing #1.

---

## Verification Checklist

After rebuilding, check:

- [ ] `koboldcpp_rpc_hybrid.so` exists (or modified `koboldcpp_rpc.so`)
- [ ] Console shows: `[RPC] Using hybrid RPC library`
- [ ] Console shows: `Found local GPU device: VULKAN0`
- [ ] Device count increases: `Total devices for offloading: 4`
- [ ] Model loads across all devices (local + remote)

---

## Build Commands Summary

### Full Rebuild (Recommended)
```bash
cd /home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp_rpc_attempt
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc_hybrid
```

### Quick Rebuild (Modify Existing)
```bash
cd /home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp_rpc_attempt
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
```

### Test Command
```bash
python koboldcpp.py \
  --model /home/lunarbuntu/Downloads/Qwen3.5-0.8B-Q8_0.gguf \
  --rpc 192.168.1.101:50054 \
  --tensor_split 25 25 25 25 \
  --gpulayers 999
```

---

## Technical Details

### Current Build (Broken)
```
koboldcpp_rpc.so includes:
  ✅ RPC backend
  ✅ CPU backend
  ❌ Vulkan backend (MISSING)
  ❌ CUDA backend (MISSING)
```

### Hybrid Build (Fixed)
```
koboldcpp_rpc_hybrid.so includes:
  ✅ RPC backend
  ✅ CPU backend
  ✅ Vulkan backend
  ✅ All shader dependencies
```

### Why Vulkan Backend is Complex

Vulkan backend requires:
1. `ggml-backend_vulkan.o` - Backend implementation
2. `ggml-vulkan.o` - Vulkan API wrappers
3. `ggml-vulkan-shaders.o` - Precompiled shader binaries
4. Shader compilation data (`ggml-vulkan-shaders.cpp`)

These are large and take time to compile, which is why they're excluded from the basic RPC build.

---

## Next Steps

1. **Rebuild** with hybrid configuration
2. **Test** with your setup
3. **Verify** local GPUs are detected
4. **Document** any issues in `RPC_HYBRID_TROUBLESHOOTING.md`

---

**Status**: 🔧 Fix Required  
**Last Updated**: 2026-04-10
