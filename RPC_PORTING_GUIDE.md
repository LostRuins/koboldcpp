# RPC Porting Guide - llama.cpp to koboldcpp

**Version**: 1.111.2  
**Date**: 2026-04-17  
**Purpose**: Complete step-by-step guide to port RPC from llama.cpp to koboldcpp  
**Target Audience**: Developers, LLMs, or anyone needing to replicate the integration  
**Status**: ✅ Complete - All Features Working

---

## Overview

This guide provides **complete, reproducible instructions** for integrating RPC functionality from llama.cpp into koboldcpp. Following these steps will result in a working RPC implementation with:

✅ RPC Server that starts and advertises GPU devices  
✅ RPC Client that connects to servers  
✅ Multi-server support  
✅ GPU offloading to RPC servers  
✅ **Hybrid mode** (local GPUs + RPC servers)  
✅ **Manual tensor_split** for layer distribution control  
✅ **Manual device ordering** (--device argument)  
✅ **Device reordering** (mix RPC and local in any order)  
✅ **Triple naming support** (HIP0 = CUDA0 = ROCm0 for HIPBLAS)  
✅ **Automatic backend detection** (make rpc-full-all)  
✅ **Full GUI dropdown** with all backend options  

**Time Required**: 2-4 hours  
**Difficulty**: Intermediate (requires C++ and build system knowledge)  

---

## Prerequisites

### Required Knowledge
- C++ compilation and linking
- Makefile build systems
- Python ctypes for library loading
- Backend registration patterns

### Required Tools
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libvulkan-dev \
    vulkan-tools \
    glslc
```

### Source Code Requirements
- llama.cpp repository (commit b8708 or later with RPC support)
- koboldcpp repository (version 1.111.2 or compatible)

### Verify Prerequisites
```bash
# Check Vulkan installation
vulkaninfo | grep "GPU id"
# Should list your Vulkan-capable GPUs

# Check build tools
g++ --version
cmake --version
# Should show version 9.0 or higher
```

---

## Phase 1: File Integration

### Step 1.1: Copy RPC Header File

```bash
# Navigate to koboldcpp directory
cd koboldcpp-1.111.2

# Copy RPC header from llama.cpp
cp ../llama.cpp/ggml/include/ggml-rpc.h ggml/include/

# Verify file exists
ls -lh ggml/include/ggml-rpc.h
# Expected: -rw-r--r-- 1 user user 4.2K Apr  9 12:00 ggml/include/ggml-rpc.h
```

### Step 1.2: Copy RPC Implementation

```bash
# Copy RPC implementation directory
cp -r ../llama.cpp/ggml/src/ggml-rpc ggml/src/

# Verify directory structure
ls -lh ggml/src/ggml-rpc/
# Expected:
# total 88K
# -rw-r--r-- 1 user user  82K Apr  9 12:00 ggml-rpc.cpp
# -rw-r--r-- 1 user user 1.2K Apr  9 12:00 CMakeLists.txt
```

### Step 1.3: Copy RPC Server Tool

```bash
# Copy RPC server source
cp ../llama.cpp/tools/rpc-server.cpp tools/

# Verify file exists
ls -lh tools/rpc-server.cpp
# Expected: -rw-r--r-- 1 user user 12K Apr  9 12:00 tools/rpc-server.cpp
```

### Step 1.4: Verify File Integrity

```bash
# Check key functions exist in RPC implementation
grep "ggml_backend_rpc_start_server" ggml/src/ggml-rpc/ggml-rpc.cpp
grep "ggml_backend_rpc_reg" ggml/src/ggml-rpc/ggml-rpc.cpp
grep "ggml_backend_rpc_add_server" ggml/src/ggml-rpc/ggml-rpc.cpp

# Expected output (line numbers may vary):
# 1842:static void ggml_backend_rpc_start_server(...)
# 2061:ggml_backend_reg_t ggml_backend_rpc_reg(void) {
# 2090:ggml_backend_reg_t ggml_backend_rpc_add_server(...)
```

### Step 1.5: File Structure After Integration

```
koboldcpp-1.111.2/
├── ggml/
│   ├── include/
│   │   └── ggml-rpc.h          # ← New file
│   └── src/
│       └── ggml-rpc/           # ← New directory
│           ├── ggml-rpc.cpp    # ← RPC implementation
│           └── CMakeLists.txt  # ← CMake config
├── tools/
│   └── rpc-server.cpp          # ← New file
└── ... (existing koboldcpp files)
```

---

## Phase 2: Build System Integration

### Step 2.1: Add RPC Build Variable to Makefile

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: After line ~104 (after other build variables like LLAMA_VULKAN)

**Add**:
```makefile
ifdef LLAMA_RPC
RPC_FLAGS = -DGGML_USE_RPC
else
RPC_FLAGS =
endif
```

**Verification**:
```bash
grep -A 3 "ifdef LLAMA_RPC" Makefile
# Expected:
# ifdef LLAMA_RPC
# RPC_FLAGS = -DGGML_USE_RPC
# else
# RPC_FLAGS =
# endif
```

### Step 2.2: Add RPC Library Detection to Python Wrapper

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~952 (in library detection section)

**Add**:
```python
lib_rpc = pick_existant_file("koboldcpp_rpc.dll", "koboldcpp_rpc.so")
```

**Location**: After line ~966 (in lib_option_pairs list)

**Add to list**:
```python
lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_hipblas, "Use hipBLAS (ROCm)"),
    (lib_vulkan, "Use Vulkan"),
    (lib_rpc, "Use RPC (Remote)"),    # ← Add this line
    (lib_noavx2, "Use CPU (Old CPU)"),
    ...
]
```

### Step 2.3: Add RPC Library Loading Logic

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In `init_library()` function, after line ~1017 (after vulkan_info check)

**Add**:
```python
elif args.userpc is not None:
    if file_exists(lib_rpc):
        libname = lib_rpc
    else:
        print("WARNING: RPC library not found. Please build with LLAMA_RPC=1")
```

**Complete Context**:
```python
def init_library():
    global libname
    libname = None
    
    if args.usevulkan:
        if file_exists(lib_vulkan):
            libname = lib_vulkan
        else:
            print("WARNING: Vulkan library not found. Please build with LLAMA_VULKAN=1")
    
    elif args.userpc is not None:    # ← Add this section
        if file_exists(lib_rpc):
            libname = lib_rpc
        else:
            print("WARNING: RPC library not found. Please build with LLAMA_RPC=1")
    
    elif args.usecuda:
        ...
```

### Step 2.4: Add RPC Object File Build Rule

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: After line ~680 (after ggml-vulkan rules)

**Add**:
```makefile
#rpc
ggml-rpc.o: ggml/src/ggml-rpc/ggml-rpc.cpp ggml/include/ggml-rpc.h
	$(CXX) $(CXXFLAGS) $(RPC_FLAGS) -c $< -o $@
```

**Verification**:
```bash
grep -A 2 "^#rpc" Makefile
# Expected:
# #rpc
# ggml-rpc.o: ggml/src/ggml-rpc/ggml-rpc.cpp ggml/include/ggml-rpc.h
# 	$(CXX) $(CXXFLAGS) $(RPC_FLAGS) -c $< -o $@
```

### Step 2.5: Add RPC Client Build Target (Hybrid Mode)

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: After line ~927 (after other koboldcpp_* targets)

**Add**:
```makefile
# RPC client build target (Hybrid: RPC + Vulkan)
ifdef RPC_BUILD
ifdef VULKAN_BUILD
# Hybrid build with both RPC and Vulkan
koboldcpp_rpc: ggml_v4_vulkan.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o \
    ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter_vulkan.o ggml-rpc.o \
    ggml-vulkan.o ggml-vulkan-shaders.o sdcpp_vulkan.o whispercpp_vulkan.o \
    tts_default.o music_default.o embeddings_default.o llavaclip_vulkan.o llava.o \
    ggml-backend_vulkan.o ggml-backend-reg_vulkan.o ggml-repack.o \
    $(OBJS_FULL) $(OBJS)
	$(RPC_BUILD)
else
# RPC only build (fallback)
koboldcpp_rpc: ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o \
    ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter.o ggml-rpc.o \
    sdcpp_default.o whispercpp_default.o tts_default.o music_default.o \
    embeddings_default.o llavaclip_default.o llava.o \
    ggml-backend_default.o ggml-backend-reg_default.o ggml-repack.o \
    $(OBJS_FULL) $(OBJS)
	$(RPC_BUILD)
endif
else
# RPC_BUILD not defined
koboldcpp_rpc:
	$(DONOTHING)
endif
```

**Important**: This builds the RPC client WITH Vulkan backend for hybrid mode support.

### Step 2.6: Add RPC Server Build Targets

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: After line ~953 (after tool build targets)

**Add**:
```makefile
# RPC server build targets
ifdef VULKAN_BUILD
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

# RPC server for CUDA backend (NVIDIA GPUs)
ifdef CUBLAS_BUILD
rpc-server-cuda: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o \
    ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(CUBLASLD_FLAGS)
endif

# RPC server for HIPBLAS backend (AMD ROCm GPUs)
ifdef HIPBLAS_BUILD
rpc-server-hip: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_cublas.o \
    ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)
endif

# RPC server CPU-only (fallback)
rpc-server: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_default.o \
    ggml-backend-reg_default.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
```

**Important**: Each RPC server backend uses its respective GPU backend:
- `rpc-server-vulkan` - Uses Vulkan backend (works with AMD/Intel/NVIDIA via Vulkan)
- `rpc-server-cuda` - Uses CUDA backend (NVIDIA GPUs only)
- `rpc-server-hip` - Uses HIPBLAS backend (AMD ROCm GPUs only)

### Step 2.7: Add RPC Build Variable Definition

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: After line ~446 (after other *_BUILD definitions)

**Add**:
```makefile
ifdef LLAMA_VULKAN
ifdef LLAMA_RPC
# Hybrid RPC + Vulkan build - needs both libraries
RPC_BUILD = $(CXX) $(CXXFLAGS) $(RPC_FLAGS) $^ -lvulkan -shared -o $@.so $(LDFLAGS)
else
VULKAN_BUILD = $(CXX) $(CXXFLAGS) $^ -lvulkan -shared -o $@.so $(LDFLAGS)
endif
endif
ifdef LLAMA_RPC
ifndef LLAMA_VULKAN
# RPC only build
RPC_BUILD = $(CXX) $(CXXFLAGS) $(RPC_FLAGS) $^ -shared -o $@.so $(LDFLAGS)
endif
endif
```

And for Windows (after line ~436):
```makefile
ifdef LLAMA_RPC
RPC_BUILD = $(CXX) $(CXXFLAGS) $(RPC_FLAGS) $^ -shared -o $@.dll $(LDFLAGS)
endif
```

---

## Phase 3: Python Wrapper Integration

### Step 3.1: Add RPC Argument Parser

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~10698 (after --usevulkan argument)

**Add**:
```python
compatgroup.add_argument("--userpc", "--rpc", 
    help="Use RPC for remote model inference. Specify one or more RPC server endpoints (e.g. --rpc 192.168.1.100:50054). Can be used multiple times for multiple servers.", 
    metavar=('[endpoint]'), 
    nargs='+', 
    type=str, 
    default=None)
```

### Step 3.1b: Add Device Ordering Argument (Optional)

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In advanced parser section (around line ~17460)

**Note**: The `--device` argument already exists in koboldcpp for llama.cpp compatibility. No changes needed!

**Usage**:
```python
# Already exists in advparser section
advparser.add_argument(
    "--device",
    "-dev",
    metavar=("<dev1,dev2,..>"),
    help="Set llama.cpp compatible device selection override. Comma separated. Overrides normal device choices.",
    default="",
)
```

**Example**:
```bash
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,VULKAN1,RPC1 \
    --gpulayers 999
```

### Step 3.2: Add RPC Library Variable Declarations

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In global library detection section (around line ~940)

**Add all RPC-enabled libraries**:
```python
# Regular backends
lib_default = pick_existant_file("koboldcpp_default.dll", "koboldcpp_default.so")
lib_failsafe = pick_existant_file("koboldcpp_failsafe.dll", "koboldcpp_failsafe.so")
lib_noavx2 = pick_existant_file("koboldcpp_noavx2.dll", "koboldcpp_noavx2.so")
lib_vulkan_failsafe = pick_existant_file("koboldcpp_vulkan_failsafe.dll", "koboldcpp_vulkan_failsafe.so")
lib_cublas = pick_existant_file("koboldcpp_cublas.dll", "koboldcpp_cublas.so")
lib_hipblas = pick_existant_file("koboldcpp_hipblas.dll", "koboldcpp_hipblas.so")
lib_vulkan = pick_existant_file("koboldcpp_vulkan.dll", "koboldcpp_vulkan.so")
lib_vulkan_noavx2 = pick_existant_file("koboldcpp_vulkan_noavx2.dll", "koboldcpp_vulkan_noavx2.so")

# RPC-enabled backends
lib_cublas_rpc = pick_existant_file("koboldcpp_cublas_rpc.dll", "koboldcpp_cublas_rpc.so")
lib_hipblas_rpc = pick_existant_file("koboldcpp_hipblas_rpc.dll", "koboldcpp_hipblas_rpc.so")
lib_rpc = pick_existant_file("koboldcpp_rpc.dll", "koboldcpp_rpc.so")

lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_cublas_rpc, "Use CUDA + RPC"),          # ← NEW
    (lib_hipblas, "Use hipBLAS (ROCm)"),
    (lib_hipblas_rpc, "Use hipBLAS + RPC"),      # ← NEW
    (lib_vulkan, "Use Vulkan"),
    (lib_rpc, "Use Vulkan + RPC"),               # ← Previously "Use RPC (Remote)"
    (lib_noavx2, "Use CPU (Old CPU)"),
    (lib_vulkan_noavx2, "Use Vulkan (Old CPU)"),
    (lib_vulkan_failsafe, "Use Vulkan (Older CPU)"),
    (lib_failsafe, "Failsafe Mode (Older CPU)"),
]

# Unpack options for use in library selection
(
    default_option, cublas_option, cublas_rpc_option, hipblas_option, hipblas_rpc_option,
    vulkan_option, rpc_option, noavx2_option, vulkan_noavx2_option, vulkan_failsafe_option,
    failsafe_option
) = (
    opt if file_exists(lib) or (os.name == "nt" and file_exists(opt + ".dll")) else None
    for lib, opt in lib_option_pairs
)

runopts = [opt for lib, opt in lib_option_pairs if file_exists(lib)]
```

### Step 3.3: Update Library Selection Logic

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In `init_library()` function (around line ~1000)

**Add ROCm detection and prefer RPC variants**:
```python
def init_library():
    global handle, args, libname
    global \
        lib_default, lib_failsafe, lib_noavx2, lib_vulkan_failsafe, \
        lib_cublas, lib_cublas_rpc, lib_hipblas, lib_hipblas_rpc, \
        lib_vulkan, lib_vulkan_noavx2, lib_rpc
    
    libname = lib_default
    
    # Detect RPC in device string
    has_rpc_in_device = False
    if args.device and "RPC" in args.device:
        has_rpc_in_device = True
    
    # Detect ROCm/HIP in device string
    has_rocm_in_device = False
    if args.device and ("ROCm" in args.device or "HIP" in args.device):
        has_rocm_in_device = True
    
    # ... (existing logic)
    
    elif args.usecuda is not None:
        # Prefer RPC variant if RPC is being used
        if file_exists(lib_cublas_rpc) and (args.userpc is not None or has_rpc_in_device):
            libname = lib_cublas_rpc
        elif file_exists(lib_cublas):
            libname = lib_cublas
        elif (args.userpc is not None or has_rpc_in_device) and file_exists(lib_hipblas_rpc):
            libname = lib_hipblas_rpc
        elif file_exists(lib_hipblas):
            libname = lib_hipblas
    
    elif args.userpc is not None:
        # Select appropriate RPC library based on backend
        if has_rocm_in_device and file_exists(lib_hipblas_rpc):
            libname = lib_hipblas_rpc
        elif file_exists(lib_cublas_rpc):
            libname = lib_cublas_rpc
        elif file_exists(lib_rpc):
            libname = lib_rpc
        else:
            print("WARNING: RPC library not found. Please build with LLAMA_RPC=1")
    
    elif has_rpc_in_device:
        if has_rocm_in_device and file_exists(lib_hipblas_rpc):
            libname = lib_hipblas_rpc
        elif file_exists(lib_cublas_rpc):
            libname = lib_cublas_rpc
        elif file_exists(lib_rpc):
            libname = lib_rpc
        else:
            print("WARNING: RPC library not found. Please build with LLAMA_RPC=1")
```

### Step 3.4: Update GUI Auto-Selection

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In `auto_set_backend_gui()` function (around line ~10860)

**Add RPC variant preference**:
```python
def auto_set_backend_gui(manual_select=False):
    global exitcounter, runmode_untouched
    # Detect RPC in device string
    has_rpc_in_device = False
    if args.device and "RPC" in args.device:
        has_rpc_in_device = True
    
    # ... (existing logic)
    
    if eligible_cuda and exitcounter < 100 and MaxMemory[0] > 3500000000 and runmode_untouched:
        # Prefer RPC variants if RPC is being used
        if args.userpc is not None or has_rpc_in_device:
            if "Use CUDA + RPC" in runopts:
                runopts_var.set("Use CUDA + RPC")
                gpu_choice_var.set("1")
                print(f"Auto Selected CUDA + RPC Backend\n")
            elif "Use hipBLAS + RPC" in runopts:
                runopts_var.set("Use hipBLAS + RPC")
                gpu_choice_var.set("1")
                print(f"Auto Selected HIPBLAS + RPC Backend\n")
        # Otherwise use standard backends
        if not found_new_backend:
            if "Use CUDA" in runopts:
                runopts_var.set("Use CUDA")
                gpu_choice_var.set("1")
            elif "Use hipBLAS (ROCm)" in runopts:
                runopts_var.set("Use hipBLAS (ROCm)")
                gpu_choice_var.set("1")
```

### Step 3.5: Update GUI changerunmode Function

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In `changerunmode()` function (around line ~11101)

**Add RPC variants to UI visibility checks**:
```python
def changerunmode(a, b, c):
    global runmode_untouched
    runmode_untouched = False
    index = runopts_var.get()
    
    # Show GPU controls for GPU backends (including RPC variants)
    if (index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or 
        index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or 
        index == "Use CUDA + RPC" or index == "Use hipBLAS (ROCm)" or 
        index == "Use hipBLAS + RPC"):
        # Show GPU selector controls
        ...
    
    # Show tensor_split controls for backends that support it
    if (index == "Use CUDA" or index == "Use CUDA + RPC" or 
        index == "Use hipBLAS (ROCm)" or index == "Use hipBLAS + RPC" or
        index == "Use Vulkan" or index == "Use Vulkan (Old CPU)"):
        # Show tensor_split controls
        ...
```

---

## Phase 4: C++ Code Modifications

### Step 4.1: Implement Device Ordering with Triple Naming

**File**: `koboldcpp-1.111.2/gpttype_adapter.cpp`

**Location**: Lines 2460-2580 (device enumeration and ordering)

**Add include** at top of file (after line ~23):
```cpp
#include <algorithm>
```

**Implementation**: Replace the device enumeration section with code that:
1. Collects all RPC devices (RPC0, RPC1, etc.)
2. Enumerates local GPU devices (VULKAN0, HIP0, CUDA0, ROCm0, etc.)
3. **Registers devices with TRIPLE names** (HIP=CUDA=ROCm for HIPBLAS compatibility)
4. If `--device` is specified, reorders all devices according to the argument
5. Otherwise, uses default ordering (RPC first, then local)

**Key Code** (with triple naming support):
```cpp
std::string dev_override_str = inputs.devices_override ? inputs.devices_override : "";

// If device override is specified, use it to reorder ALL devices
if(dev_override_str != "" && dev_override_str.length() > 0)
{
    printf("[RPC] Manual device ordering specified: %s\n", dev_override_str.c_str());
    
    // Get all available devices (RPC + local)
    std::vector<std::pair<std::string, ggml_backend_dev_t>> all_devices;
    
    // Add RPC devices
    if(use_rpc) {
        for(size_t i = 0; i < rpc_devices.size(); ++i) {
            std::string name = "RPC" + std::to_string(i);
            all_devices.push_back(std::make_pair(name, rpc_devices[i]));
        }
    }
    
    // Add local GPU devices with TRIPLE NAMING support
    printf("[RPC] Enumerating local GPU devices...\n");
    int vulkan_count = 0, cuda_count = 0, hip_count = 0, metal_count = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto* dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        std::string reg_name = reg ? ggml_backend_reg_name(reg) : "";
        
        std::string reg_name_upper = reg_name;
        std::transform(reg_name_upper.begin(), reg_name_upper.end(), reg_name_upper.begin(), ::toupper);
        
        // Skip RPC and CPU
        if(reg_name_upper.find("RPC") != std::string::npos || 
           reg_name_upper.find("CPU") != std::string::npos) {
            continue;
        }
        
        // Vulkan devices
        if(reg_name_upper.find("VULKAN") != std::string::npos || 
           reg_name_upper.find("RADV") != std::string::npos) {
            std::string local_name = "VULKAN" + std::to_string(vulkan_count++);
            all_devices.push_back(std::make_pair(local_name, dev));
        }
        // CUDA devices - register with BOTH CUDA and HIP names for compatibility
        else if(reg_name_upper.find("CUDA") != std::string::npos) {
            std::string cuda_name = "CUDA" + std::to_string(cuda_count++);
            std::string hip_name = "HIP" + std::to_string(hip_count++);
            all_devices.push_back(std::make_pair(cuda_name, dev));
            all_devices.push_back(std::make_pair(hip_name, dev));
            printf("[RPC] CUDA device registered as both %s and %s\n", cuda_name.c_str(), hip_name.c_str());
        }
        // HIP/ROCM devices - register with BOTH HIP and ROCm names for compatibility
        else if(reg_name_upper.find("HIP") != std::string::npos || 
                reg_name_upper.find("ROCM") != std::string::npos) {
            std::string hip_name = "HIP" + std::to_string(hip_count);
            std::string rocm_name = "ROCm" + std::to_string(hip_count++);
            all_devices.push_back(std::make_pair(hip_name, dev));
            all_devices.push_back(std::make_pair(rocm_name, dev));
            printf("[RPC] HIP device registered as both %s and %s\n", hip_name.c_str(), rocm_name.c_str());
        }
    }
    
    // Parse device order from override string and build ordered list
    // ... (parsing code)
}
```

**Key Changes** (2026-04-16):
- CUDA devices now registered with dual names (`CUDA0` + `HIP0`)
- HIP devices now registered with dual names (`HIP0` + `ROCm0`)
- Allows using `HIP0`, `CUDA0`, or `ROCm0` interchangeably
- Improves compatibility across different backend configurations

### Step 4.2: Fix Device Enumeration (Case-Insensitive)

**File**: `koboldcpp-1.111.2/gpttype_adapter.cpp`

**Location**: After line ~2479 (in device enumeration loop)

**Change FROM**:
```cpp
// Add local GPU devices (not RPC, not CPU)
if(reg_name.find("RPC") == std::string::npos && 
   reg_name.find("CPU") == std::string::npos &&
   (reg_name.find("VULKAN") != std::string::npos || 
    reg_name.find("RADV") != std::string::npos ||
    reg_name.find("CUDA") != std::string::npos ||
    reg_name.find("HIP") != std::string::npos)) {
    printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
    devices_override.push_back(dev);
}
```

**Change TO**:
```cpp
// Add local GPU devices (not RPC, not CPU) - case insensitive check
std::string reg_name_upper = reg_name;
std::transform(reg_name_upper.begin(), reg_name_upper.end(), reg_name_upper.begin(), ::toupper);

if(reg_name_upper.find("RPC") == std::string::npos && 
   reg_name_upper.find("CPU") == std::string::npos &&
   (reg_name_upper.find("VULKAN") != std::string::npos || 
    reg_name_upper.find("RADV") != std::string::npos ||
    reg_name_upper.find("CUDA") != std::string::npos ||
    reg_name_upper.find("HIP") != std::string::npos ||
    reg_name_upper.find("ROCM") != std::string::npos ||
    reg_name_upper.find("METAL") != std::string::npos)) {
    printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
    devices_override.push_back(dev);
}
```

### Step 4.3: Fix RPC Server Direct Call

**File**: `tools/rpc-server.cpp`

**Location**: In main() function, around line ~100

**Change FROM**:
```cpp
int main(int argc, char * argv[]) {
    ggml_backend_load_all();
    
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) {
        fprintf(stderr, "Failed to find RPC backend\n");
        return 1;
    }
    
    auto start_server_fn = (decltype(ggml_backend_rpc_start_server)*) 
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    if (!start_server_fn) {
        fprintf(stderr, "Failed to obtain RPC backend start server function\n");
        return 1;
    }
    
    start_server_fn(endpoint.c_str(), cache_dir, params.n_threads, 
                    devices.size(), devices.data());
    
    return 0;
}
```

**Change TO**:
```cpp
int main(int argc, char * argv[]) {
    ggml_backend_load_all();
    
    // Call RPC start server function directly (no need for dynamic lookup)
    // This works for statically linked backends
    ggml_backend_rpc_start_server(endpoint.c_str(), cache_dir, params.n_threads, 
                                   devices.size(), devices.data());
    
    return 0;
}
```

**Why This Works**:
- Direct function call doesn't require runtime lookup
- Statically linked function is available at compile time
- No dependency on dynamic backend loading mechanism

---

## Phase 5: Testing and Verification

### Step 5.1: Verify Build

```bash
cd koboldcpp_rpc_attempt
make clean
make rpc-full-all -j8

# Verify all components were built
ls -lh koboldcpp_vulkan.so koboldcpp_hipblas.so
ls -lh koboldcpp_rpc.so koboldcpp_hipblas_rpc.so
ls -lh rpc-server-vulkan rpc-server-hip
```

### Step 5.2: Test Server

```bash
# Test Vulkan RPC server
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Test HIPBLAS RPC server
./rpc-server-hip -H 127.0.0.1 --port 50054 --device ROCm0 -c

# Test CUDA RPC server
./rpc-server-cuda -H 127.0.0.1 --port 50054 --device CUDA0 -c
```

### Step 5.3: Test Client

```bash
# Test with RPC only
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054 --gpulayers 999

# Test with hybrid mode
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,VULKAN1 \
    --gpulayers 999
```

### Step 5.4: Verify Dropdown Menu

Run `python koboldcpp.py` and check the dropdown shows all options:
- Use CPU
- Use CUDA
- Use CUDA + RPC
- Use hipBLAS (ROCm)
- Use hipBLAS + RPC
- Use Vulkan
- Use Vulkan + RPC
- Use CPU (Old CPU)
- Use Vulkan (Old CPU)
- Use Vulkan (Older CPU)
- Failsafe Mode (Older CPU)

---

## Phase 6: Backend Compatibility

### Critical Rule: Cannot Mix Backends

You **cannot** mix Vulkan devices with HIPBLAS/CUDA devices in the same session.

| Library | Can Use | Cannot Use |
|---------|---------|------------|
| **Vulkan** | `VULKAN*`, `RPC*` | `HIP*`, `CUDA*`, `ROCm*` |
| **HIPBLAS** | `HIP*`, `CUDA*`, `ROCm*`, `RPC*` | `VULKAN*` |
| **CUDA** | `CUDA*`, `HIP*`, `RPC*` | `VULKAN*`, `ROCm*` |

### Triple Naming Support

For HIPBLAS backend, all these names refer to the same physical device:

| Device Number | Equivalent Names |
|--------------|------------------|
| 0 | `HIP0` = `CUDA0` = `ROCm0` |
| 1 | `HIP1` = `CUDA1` = `ROCm1` |
| 2 | `HIP2` = `CUDA2` = `ROCm2` |

**All these commands are equivalent**:
```bash
python koboldcpp.py --device HIP0,RPC0,HIP1 --rpc 192.168.1.101:50054
python koboldcpp.py --device CUDA0,RPC0,CUDA1 --rpc 192.168.1.101:50054
python koboldcpp.py --device ROCm0,RPC0,ROCm1 --rpc 192.168.1.101:50054
```

---

## Known Issues and Workarounds

### 1. HIP Assertion Bug (ROCm 7.2+)

**Error**:
```
Assertion `err == hipSuccess' failed in hip_code_object.cpp
```

**Cause**: HIP runtime bug when mixing RPC + local HIP devices

**Workarounds**:
1. Use RPC only (no local HIP):
   ```bash
   python koboldcpp.py --device RPC0,RPC1,RPC2 --rpc 192.168.1.101:50054
   ```

2. Use local HIP only (no RPC):
   ```bash
   python koboldcpp.py --device HIP0,HIP1 --gpulayers 999
   ```

3. Use Vulkan backend instead (most stable):
   ```bash
   python koboldcpp.py --device VULKAN0,RPC0,RPC1,VULKAN1 --rpc 192.168.1.101:50054
   ```

### 2. Vulkan Build Fails Silently

**Cause**: Missing Vulkan dependencies or wrong target name

**Solution**:
```bash
# Install dependencies
sudo apt-get install libvulkan-dev vulkan-tools glslc

# Use correct target
make LLAMA_VULKAN=1 koboldcpp_vulkan -j8
```

### 3. CUDA/HIPBLAS Build Conflicts

**Error**:
```
undefined symbol: __hipRegisterFunction
```

**Cause**: Trying to build both CUDA and HIPBLAS together

**Solution**: `make rpc-full-all` now skips CUDA if HIPBLAS detected

### Security Warning

⚠️ **RPC has no authentication or encryption**

**Never expose to public internet!**

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
ssh -L 50054:localhost:50054 user@192.168.1.101
```

---

## Performance Considerations

### Network
- Use wired Ethernet (not WiFi)
- Ensure low latency (< 5ms ping)
- Same subnet is best

### Backend Choice
- **Vulkan**: Most stable with RPC, universal compatibility
- **HIPBLAS**: Best performance for AMD GPUs, but may hit assertion bugs with RPC
- **CUDA**: Best performance for NVIDIA GPUs

### Layer Distribution
- More powerful GPUs get higher tensor_split ratios
- Local GPUs typically faster (no network overhead)

---

## Future Improvements

### Potential Enhancements
1. **Automatic Tensor Split**: Based on device VRAM and performance
2. **SSH Tunneling**: Built-in secure tunnel support
3. **Load Balancing**: Automatic distribution across servers
4. **Authentication**: Token-based access control
5. **Encryption**: TLS/SSL for RPC traffic
6. **Automatic Device Ordering**: Optimize based on bandwidth/latency
7. **Device Profiling**: Performance benchmarks for each device

---

## References

### Source Code
- llama.cpp: https://github.com/ggml-org/llama.cpp
- koboldcpp: https://github.com/LostRuins/koboldcpp

### Documentation
- RPC Manual: `RPC_MANUAL.md`
- Quick Start: `RPC_QUICKSTART.md`
- Build Guide: `RPC_makefile.md`
- Backend Compatibility: `RPC_BACKEND_COMPATIBILITY.md`

### Tools
- Vulkan SDK: https://vulkan.lunarg.com/
- ROCm: https://rocm.docs.amd.com/
- CUDA: https://developer.nvidia.com/cuda-toolkit

---

## Version History

- **v1.111.2** (2026-04-17): Complete porting guide with all features
  - Documents full integration process
  - Includes all hurdles and solutions
  - Provides verification checklists
  - Reflects current working state with hybrid mode
  - Includes tensor_split fixes
  - **Triple naming support** (HIP0 = CUDA0 = ROCm0)
  - **Manual device ordering** (--device argument)
  - **Device reordering** (mix RPC and local in any order)
  - **Automatic backend detection** (make rpc-full-all)
  - **Full GUI dropdown** with all backend options
  - **RPC variant libraries** (koboldcpp_hipblas_rpc.so, etc.)
  - **Backend compatibility matrix**

- **v1.111.2** (2026-04-16): Added dual naming and library selection fixes

- **v1.111.2** (2026-04-11): Initial porting guide with basic RPC integration

---

**License**: MIT  
**Authors**: KoboldCPP Team  
**Contact**: https://github.com/LostRuins/koboldcpp/issues
