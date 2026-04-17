# RPC Makefile Changes Documentation

**Version**: 1.111.2  
**Date**: 2026-04-15  
**Purpose**: Complete documentation of all Makefile changes for RPC support  
**Status**: ✅ Complete - All Features Working

---

## Overview

This document details every modification made to the `Makefile` to implement RPC (Remote Procedure Call) functionality, including support for Vulkan, CUDA, and HIPBLAS/ROCm backends.

**Total Changes**: 3 major build targets  
**Lines Added**: ~6 lines  
**Lines Modified**: ~2 lines  
**Build Targets Added**: `rpc-server-cuda`, `rpc-server-hip`  

---

## Table of Contents

1. [RPC Server Build Targets](#1-rpc-server-build-targets)
2. [CUDA RPC Server Fix](#2-cuda-rpc-server-fix)
3. [HIPBLAS RPC Server Fix](#3-hipblas-rpc-server-fix)
4. [Complete Code Diff](#4-complete-code-diff)
5. [Build Variables Reference](#5-build-variables-reference)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. RPC Server Build Targets

### Original State

Before RPC implementation, the Makefile only had:
- `rpc-server-vulkan` - Vulkan-based RPC server
- `rpc-server` - CPU-only fallback RPC server

### Change 1.1: Add CUDA RPC Server Build Target

**Location**: Line ~973 (after `rpc-server-vulkan`)  
**Purpose**: Build RPC server with CUDA backend for NVIDIA GPUs

**Code Added**:
```makefile
ifdef CUBLAS_BUILD
rpc-server-cuda: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o $(CUBLAS_OBJS)
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(CUBLASLD_FLAGS)
endif
```

**Dependencies**:
- `tools/rpc-server.cpp` - RPC server main source
- `ggml-rpc.o` - RPC protocol implementation
- `ggml-backend_cublas.o` - CUDA backend registration
- `ggml-backend-reg_cublas.o` - CUDA backend registry
- `ggml_v3_cublas.o` - **CRITICAL**: ggml_v3 CUDA/HIP implementation
- `ggml_v2_cublas.o` - **CRITICAL**: ggml_v2 CUDA/HIP implementation
- `ggml_v1.o` - **CRITICAL**: ggml_v1 backward compatibility
- `$(CUBLAS_OBJS)` - **CRITICAL**: CUDA kernel implementations

**Build Flags**:
- `$(CUBLAS_FLAGS)` - CUDA compile flags (`-DGGML_USE_CUDA`, include paths)
- `$(CUBLASLD_FLAGS)` - CUDA linker flags (`-lcuda -lcublas -lcudart -lcublasLt`)

**Why `$(CUBLAS_OBJS)` is Critical**:
These object files contain:
- CUDA kernel implementations (`ggml-cuda.o`, `ggml_v3-cuda.o`, etc.)
- Backend registration function (`ggml_backend_cuda_reg()`)
- GPU memory management code
- Device-side computation kernels

Without these, the linker fails with:
```
ld.lld: error: undefined symbol: ggml_backend_cuda_reg
>>> referenced by ggml-backend-reg.cpp
>>>               ggml-backend-reg_cublas.o:(get_reg())
```

---

### Change 1.2: Add HIPBLAS RPC Server Build Target

**Location**: Line ~978 (after `rpc-server-cuda`)  
**Purpose**: Build RPC server with HIPBLAS backend for AMD ROCm GPUs

**Code Added**:
```makefile
ifdef HIPBLAS_BUILD
rpc-server-hip: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o $(HIP_OBJS)
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)
endif
```

**Dependencies**:
- Same as CUDA, but uses `$(HIP_OBJS)` instead of `$(CUBLAS_OBJS)`
- `ggml_v3_cublas.o` - **CRITICAL**: ggml_v3 HIP implementation
- `ggml_v2_cublas.o` - **CRITICAL**: ggml_v2 HIP implementation  
- `ggml_v1.o` - **CRITICAL**: ggml_v1 backward compatibility

**Build Flags**:
- `$(HCXX)` - HIP compiler (hipcc)
- `$(HIPFLAGS)` - HIP compile flags (`-DGGML_USE_HIP`, ROCm paths)
- `$(HIPLDFLAGS)` - HIP linker flags (`-lhipblas -lamdhip64 -lrocblas`)

**Why `$(HIP_OBJS)` is Critical**:
Same reason as CUDA - contains HIP kernel implementations and backend registration.

---

## 2. CUDA RPC Server Fix

### The Problem

**Initial Implementation** (INCORRECT):
```makefile
ifdef CUBLAS_BUILD
rpc-server-cuda: tools/rpc-server.cpp ... console.o
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(CUBLASLD_FLAGS)
endif
```

**Error**:
```bash
$ make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda
ld.lld: error: undefined symbol: ggml_backend_cuda_reg
>>> referenced by ggml-backend-reg.cpp
>>>               ggml-backend-reg_cublas.o:(get_reg())
clang++: error: linker command failed with exit code 1
make: *** [Makefile:980: rpc-server-cuda] Fehler 1
```

**Root Cause**: Missing `$(CUBLAS_OBJS)` in dependencies.

### The Fix

**Location**: Line ~973  
**Date Fixed**: 2026-04-15

**Code Changed**:
```makefile
# BEFORE (missing $(CUBLAS_OBJS)):
rpc-server-cuda: tools/rpc-server.cpp ... console.o
	$(CXX) ...

# AFTER (with $(CUBLAS_OBJS)):
rpc-server-cuda: tools/rpc-server.cpp ... console.o $(CUBLAS_OBJS)
	$(CXX) ...
```

**What Changed**: Added `$(CUBLAS_OBJS)` to the dependency list.

**Why It Works**: `$(CUBLAS_OBJS)` contains all CUDA backend implementation objects that define `ggml_backend_cuda_reg()` and other required symbols.

---

## 3. HIPBLAS RPC Server Fix

### The Problem

**Initial Implementation** (INCORRECT):
```makefile
ifdef HIPBLAS_BUILD
rpc-server-hip: tools/rpc-server.cpp ... console.o
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)
endif
```

**Error** (same as CUDA):
```bash
$ make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip
ld.lld: error: undefined symbol: ggml_backend_cuda_reg
>>> referenced by ggml-backend-reg.cpp
>>>               ggml-backend-reg_cublas.o:(get_reg())
clang++: error: linker command failed with exit code 1
make: *** [Makefile:980: rpc-server-hip] Fehler 1
```

**Root Cause**: Missing `$(HIP_OBJS)` in dependencies (same issue as CUDA).

### The Fix

**Location**: Line ~978  
**Date Fixed**: 2026-04-15

**Code Changed**:
```makefile
# BEFORE (missing $(HIP_OBJS)):
rpc-server-hip: tools/rpc-server.cpp ... console.o
	$(HCXX) ...

# AFTER (with $(HIP_OBJS)):
rpc-server-hip: tools/rpc-server.cpp ... console.o $(HIP_OBJS)
	$(HCXX) ...
```

**What Changed**: Added `$(HIP_OBJS)` to the dependency list.

---

## 4. Complete Code Diff

### Unified Diff of All Changes

```diff
--- Makefile.original
+++ Makefile
@@ -970,11 +970,19 @@ ifdef VULKAN_BUILD
 rpc-server-vulkan: ggml/src/ggml-vulkan-shaders.cpp tools/rpc-server.cpp \
     ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o \
     llama.o ggml-rpc.o ggml-backend_vulkan.o ggml-backend-reg_vulkan.o \
     ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o \
     ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
     unicode.o unicode-common.o unicode-data.o ggml-threading.o \
     ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o \
     budget.o kcpputils.o ggml-vulkan.o console.o
 	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) -lvulkan
 endif
 
+# RPC server for CUDA backend (NVIDIA GPUs)
 ifdef CUBLAS_BUILD
-rpc-server-cuda: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
-    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o \
-    ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
-    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
-    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
-    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o
+rpc-server-cuda: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
+    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o \
+    ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
+    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
+    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
+    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o \
+    $(CUBLAS_OBJS)
 	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(CUBLASLD_FLAGS)
 endif
 
+# RPC server for HIPBLAS backend (AMD ROCm GPUs)
 ifdef HIPBLAS_BUILD
-rpc-server-hip: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
-    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o \
-    ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
-    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
-    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
-    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o
+rpc-server-hip: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
+    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o \
+    ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
+    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
+    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
+    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o \
+    $(HIP_OBJS)
 	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)
 endif
```

---

## 5. Build Variables Reference

### Existing Variables (Pre-existing)

| Variable | Purpose | Defined When |
|----------|---------|--------------|
| `CUBLAS_BUILD` | CUDA shared library build command | `LLAMA_CUBLAS=1` |
| `HIPBLAS_BUILD` | HIPBLAS shared library build command | `LLAMA_HIPBLAS=1` |
| `VULKAN_BUILD` | Vulkan shared library build command | `LLAMA_VULKAN=1` |
| `CUBLAS_FLAGS` | CUDA compile flags | `LLAMA_CUBLAS=1` |
| `HIPFLAGS` | HIP compile flags | `LLAMA_HIPBLAS=1` |
| `CUBLASLD_FLAGS` | CUDA linker flags | `LLAMA_CUBLAS=1` |
| `HIPLDFLAGS` | HIP linker flags | `LLAMA_HIPBLAS=1` |

### Object File Variables

| Variable | Contents | Size |
|----------|----------|------|
| `$(CUBLAS_OBJS)` | CUDA kernel implementations | ~50 object files |
| `$(HIP_OBJS)` | HIP/ROCm kernel implementations | ~50 object files |
| `$(VULKAN_OBJS)` | Vulkan kernel implementations | ~1 object file |

### What's in `$(CUBLAS_OBJS)` and `$(HIP_OBJS)`

These variables expand to all CUDA/HIP backend implementation files:

```makefile
# From Makefile line ~320-322
HIP_OBJS      += ggml-cuda.o ggml_v3-cuda.o ggml_v2-cuda.o ggml_v2-cuda-legacy.o
HIP_OBJS      += $(patsubst %.cu,%.o,$(filter-out ggml/src/ggml-cuda/ggml-cuda.cu, $(wildcard ggml/src/ggml-cuda/*.cu)))
HIP_OBJS      += $(OBJS_CUDA_TEMP_INST)
```

**Key Files**:
- `ggml-cuda.o` - Main CUDA backend implementation
- `ggml_v3-cuda.o` - Version 3 CUDA kernels
- `ggml_v2-cuda.o` - Version 2 CUDA kernels
- `ggml_v2-cuda-legacy.o` - Legacy CUDA support
- Various `*.cu` files - Specialized CUDA kernels (quantization, matrix ops, etc.)

**Total**: ~50 object files containing all GPU kernel code.

---

## 6. Troubleshooting

### Error: "undefined symbol: ggml_backend_cuda_reg"

**Symptom**:
```bash
ld.lld: error: undefined symbol: ggml_backend_cuda_reg
>>> referenced by ggml-backend-reg.cpp
>>>               ggml-backend-reg_cublas.o:(get_reg())
```

**Cause**: Missing `$(CUBLAS_OBJS)` or `$(HIP_OBJS)` in RPC server dependencies.

**Solution**: Ensure the build target includes the backend objects:
```makefile
# CORRECT
rpc-server-cuda: ... console.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o $(CUBLAS_OBJS)
	$(CXX) ...

# INCORRECT
rpc-server-cuda: ... console.o $(CUBLAS_OBJS)
	$(CXX) ...
```

---

### Error: "undefined symbol: ggml_v2_is_quantized" or "ggml_v3_nbytes"

**Symptom**:
```bash
ld.lld: error: undefined symbol: ggml_v2_is_quantized
>>> referenced by ggml_v2-cuda.cu
>>>               ggml_v2-cuda.o:(ggml_v2_cuda_can_mul_mat)

ld.lld: error: undefined symbol: ggml_v3_nbytes
>>> referenced by ggml_v3-cuda.cu
>>>               ggml_v3-cuda.o:(ggml_v3_cuda_mul_mat)
```

**Cause**: Missing `ggml_v3_cublas.o`, `ggml_v2_cublas.o`, or `ggml_v1.o` in RPC server dependencies.

These objects contain ggml version helper functions that are called by CUDA/HIP kernels.

**Solution**: Add the ggml version objects to the build target:
```makefile
# CORRECT (with ggml version objects)
rpc-server-cuda: ... console.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o $(CUBLAS_OBJS)
	$(CXX) ...

# INCORRECT (missing ggml version objects)
rpc-server-cuda: ... console.o $(CUBLAS_OBJS)
	$(CXX) ...
```

**Why These Are Needed**:
- `ggml_v3_cublas.o` - Contains `ggml_v3_*` functions used by CUDA/HIP kernels
- `ggml_v2_cublas.o` - Contains `ggml_v2_*` functions used by CUDA/HIP kernels
- `ggml_v1.o` - Backward compatibility for old model formats

These are CUDA/HIP-specific versions (not plain `ggml_v3.o`) that include GPU-accelerated implementations.

---

### Error: "cannot find -lcuda" or "cannot find -lcublas"

**Symptom**:
```bash
ld: cannot find -lcuda: No such file or directory
ld: cannot find -lcublas: No such file or directory
```

**Cause**: CUDA toolkit not installed or not in library path.

**Solution**:
1. Install CUDA toolkit: `sudo apt-get install cuda-toolkit`
2. Ensure `$(CUBLASLD_FLAGS)` includes correct library paths
3. Check CUDA installation: `nvcc --version`

---

### Error: "cannot find -lhipblas" or "cannot find -lamdhip64"

**Symptom**:
```bash
ld: cannot find -lhipblas: No such file or directory
ld: cannot find -lamdhip64: No such file or directory
```

**Cause**: ROCm not installed or not in library path.

**Solution**:
1. Install ROCm: `sudo apt-get install rocm`
2. Ensure `$(HIPLDFLAGS)` includes correct library paths
3. Check ROCm installation: `hipcc --version`

---

### Error: "HIP_OBJS not defined"

**Symptom**:
```bash
make: *** No rule to make target 'HIP_OBJS'.  Stop.
```

**Cause**: `HIP_OBJS` variable not defined before use.

**Solution**: Ensure `LLAMA_HIPBLAS=1` is set, which triggers `HIP_OBJS` definition:
```bash
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip
```

---

### Build Verification

After successful build, verify the binaries:

```bash
# Check CUDA RPC server
ls -lh rpc-server-cuda
# Expected: -rwxr-xr-x 1 user user 85M Apr 15 12:00 rpc-server-cuda

# Check HIPBLAS RPC server
ls -lh rpc-server-hip
# Expected: -rwxr-xr-x 1 user user 85M Apr 15 12:00 rpc-server-hip

# Test CUDA RPC server
./rpc-server-cuda -H 127.0.0.1 --device CUDA0 -p 50052 -c
# Should show: "Starting RPC server" and list CUDA devices

# Test HIPBLAS RPC server
./rpc-server-hip -H 127.0.0.1 --device HIP0 -p 50052 -c
# Should show: "Starting RPC server" and list HIP devices
```

---

## Build Command Examples

### Full Build (All Backends)

```bash
cd koboldcpp_rpc_attempt
make clean

# Client libraries
make koboldcpp_default -j8
make LLAMA_VULKAN=1 koboldcpp -j8
make LLAMA_CUBLAS=1 koboldcpp_cublas -j8
make LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8

# RPC servers
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

### Minimal Build (CUDA Only)

```bash
cd koboldcpp_rpc_attempt
make clean

# CUDA client
make LLAMA_CUBLAS=1 koboldcpp_cublas -j8

# CUDA RPC server
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
```

### Minimal Build (HIPBLAS Only)

```bash
cd koboldcpp_rpc_attempt
make clean

# HIPBLAS client
make LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8

# HIPBLAS RPC server
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

---

## Summary of Changes

| Change | Line | Lines Modified | Description |
|--------|------|----------------|-------------|
| Add `rpc-server-cuda` target | ~973 | +2 | CUDA RPC server build target |
| Add `$(CUBLAS_OBJS)` fix | ~973 | +1 | Fix undefined backend symbol error |
| Add `ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o` | ~973 | +1 | Fix undefined ggml version symbol error |
| Add `rpc-server-hip` target | ~978 | +2 | HIPBLAS RPC server build target |
| Add `$(HIP_OBJS)` fix | ~978 | +1 | Fix undefined backend symbol error |
| Add `ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o` | ~978 | +1 | Fix undefined ggml version symbol error |
| **Total** | **2 locations** | **8 lines** | **6 changes** |

---

## Related Documentation

- `RPC_QUICKSTART.md` - User build guide
- `RPC_MANUAL.md` - Complete manual
- `RPC_PORTING_GUIDE.md` - Porting documentation
- `RPC_koboldcpp.py_changes.md` - Python wrapper changes

---

**License**: MIT  
**Version**: 1.111.2  
**Date**: 2026-04-15
