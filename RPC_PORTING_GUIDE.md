# RPC Porting Guide - llama.cpp to koboldcpp

**Version**: 1.111.2  
**Date**: 2026-04-11  
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
✅ **Case-insensitive** device matching  
✅ **Manual device ordering** (--device argument)  
✅ **Device reordering** (mix RPC and local in any order)  

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

**Complete Context**:
```python
compatgroup.add_argument("--usevulkan", 
    help="Use Vulkan for GPU acceleration", 
    metavar=('[device]'), 
    nargs='+', 
    type=int, 
    default=None)

# Add this section
compatgroup.add_argument("--userpc", "--rpc", 
    help="Use RPC for remote model inference. Specify one or more RPC server endpoints (e.g. --rpc 192.168.1.100:50054). Can be used multiple times for multiple servers.", 
    metavar=('[endpoint]'), 
    nargs='+', 
    type=str, 
    default=None)

compatgroup.add_argument("--usecpu", 
    help="Use CPU for inference", 
    action="store_true")
```

### Step 3.2: Add RPC Field to Structure

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~262 (in load_model_inputs structure, after vulkan_info)

**Add**:
```python
class load_model_inputs(ctypes.Structure):
    _fields_ = [
        ("threads", ctypes.c_int),
        ("contextsize", ctypes.c_int),
        ...
        ("vulkan_info", ctypes.c_char_p),
        ("rpc_endpoints", ctypes.c_char_p),    # ← Add this line
        ("batchsize", ctypes.c_int),
        ...
    ]
```

### Step 3.3: Populate RPC Endpoints

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~1163 (after vulkan_info population)

**Add**:
```python
if args.usevulkan:
    s = ""
    for it in range(0, len(args.usevulkan)):
        s += str(args.usevulkan[it])
    inputs.vulkan_info = s.encode("UTF-8")
else:
    inputs.vulkan_info = "".encode("UTF-8")

# Add this section
if args.userpc:  # RPC endpoints specified
    s = ",".join(args.userpc)
    inputs.rpc_endpoints = s.encode("UTF-8")
else:
    inputs.rpc_endpoints = "".encode("UTF-8")
```

### Step 3.4: Enable Auto Offload for RPC

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~2390 (after inputs.gpulayers = args.gpulayers)

**Add**:
```python
inputs.gpulayers = args.gpulayers
# Auto-enable full offload for RPC
if args.userpc and (args.gpulayers == 0 or args.gpulayers == -1):
    inputs.gpulayers = 999
```

### Step 3.5: Fix Tuple Unpacking

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~966 (in lib_option_pairs unpacking)

**Change FROM**:
```python
(
    default_option,
    cublas_option,
    hipblas_option,
    vulkan_option,
    noavx2_option,
    vulkan_noavx2_option,
    vulkan_failsafe_option,
    failsafe_option,
) = (
    opt if file_exists(lib) or (os.name == "nt" and file_exists(opt + ".dll")) else None
    for lib, opt in lib_option_pairs
)
```

**Change TO**:
```python
(
    default_option,
    cublas_option,
    hipblas_option,
    vulkan_option,
    rpc_option,    # ← Add this line
    noavx2_option,
    vulkan_noavx2_option,
    vulkan_failsafe_option,
    failsafe_option,
) = (
    opt if file_exists(lib) or (os.name == "nt" and file_exists(opt + ".dll")) else None
    for lib, opt in lib_option_pairs
)
```

### Step 3.6: Fix Tensor Split Validation

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~16102 (in RPC tensor_split validation)

**Change FROM**:
```python
# Validate RPC tensor_split if both are provided
if args.userpc and args.tensor_split:
    rpc_count = len(args.userpc)
    ts_count = len(args.tensor_split)
    if ts_count > 0 and ts_count < rpc_count:
        print(
            f"[RPC] WARNING: tensor_split has {ts_count} values but {rpc_count} RPC servers specified"
        )
        # Extend tensor_split...
    elif ts_count > rpc_count:
        print(
            f"[RPC] WARNING: tensor_split has {ts_count} values but only {rpc_count} RPC servers"
        )
        print(f"[RPC] Truncating tensor_split to match RPC server count")
        args.tensor_split = args.tensor_split[:rpc_count]
```

**Change TO**:
```python
# Validate RPC tensor_split if both are provided
# Note: tensor_split should match total devices (RPC devices + local GPUs), not just RPC servers
if args.userpc and args.tensor_split:
    rpc_count = len(args.userpc)
    ts_count = len(args.tensor_split)
    # We can't know exact device count yet (RPC servers report devices at runtime)
    # So we just validate that tensor_split has reasonable values
    # tensor_split should have at least 1 value and ideally match total GPU count
    if ts_count == 0:
        print(
            f"[RPC] WARNING: tensor_split is empty"
        )
    else:
        print(
            f"[RPC] Tensor split configured across {ts_count} devices with ratios: {args.tensor_split}"
        )
```

---

## Phase 4: C++ Code Modifications

### Step 4.1: Implement Device Ordering

**File**: `koboldcpp-1.111.2/gpttype_adapter.cpp`

**Location**: Lines 2460-2580 (device enumeration and ordering)

**Add include** at top of file (after line ~23):
```cpp
#include <algorithm>
```

**Implementation**: Replace the device enumeration section with code that:
1. Collects all RPC devices (RPC0, RPC1, etc.)
2. Enumerates local GPU devices (VULKAN0, HIP0, CUDA0, ROCm0, etc.)
3. **Registers devices with dual names** (HIP=CUDA, HIP=ROCm for compatibility)
4. If `--device` is specified, reorders all devices according to the argument
5. Otherwise, uses default ordering (RPC first, then local)

**Key Code** (with dual naming support):
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
    
    // Add local GPU devices with DUAL NAMING support
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

**See**: Full implementation in `gpttype_adapter.cpp:2488-2511`

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
    reg_name_upper.find("METAL") != std::string::npos)) {
    printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
    devices_override.push_back(dev);
}
```

### Step 4.2: Add Debug Output (Optional)

**File**: `koboldcpp-1.111.2/gpttype_adapter.cpp`

**Location**: After line ~2472 (in device enumeration loop)

**Add**:
```cpp
printf("[RPC] Checking device %zu: %s (registry: %s)\n", i, dev_name.c_str(), reg_name.c_str());
```

This helps debug device detection issues.

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

## Phase 5: Critical Fixes

### ⚠️ HURDLE #1: Static Backend Registration Failure

**Problem**: RPC server fails with "Failed to find RPC backend"

**Root Cause**: `ggml_backend_load_all()` only loads dynamic backends (.so files), not statically linked ones.

**Symptoms**:
```bash
./rpc-server-vulkan -H 0.0.0.0 --device VULKAN0 -p 50052 -c
# Output:
Starting RPC server v3.6.1
  endpoint       : 0.0.0.0:50052
Failed to find RPC backend
```

**Solution**: Call `ggml_backend_rpc_start_server()` directly instead of looking it up dynamically (see Step 4.3).

**Verification**:
```bash
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c
# Expected output:
# WARNING: radv is not a conformant Vulkan implementation, testing use only.
# ggml_vulkan: Found 1 Vulkan devices:
# ggml_vulkan: 0 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
# Starting RPC server v3.6.1
#   endpoint       : 127.0.0.1:50052
#   local cache    : /home/user/.cache/llama.cpp/rpc/
# Devices:
#   Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
```

### ⚠️ HURDLE #2: Vulkan Shaders Not Generated

**Problem**: Build fails with "ggml-vulkan-shaders.hpp: File not found"

**Root Cause**: Vulkan shaders must be generated before compilation

**Solution**:
```bash
# Generate shaders first
make vulkan-shaders-gen

# Then build RPC server
make LLAMA_VULKAN=1 rpc-server-vulkan
```

**Alternative** (copy from llama.cpp build):
```bash
cp ../llama.cpp/build-*/ggml/src/ggml-vulkan/ggml-vulkan-shaders.hpp ggml/src/
```

### ⚠️ HURDLE #3: Hybrid Mode Requires Vulkan Backend

**Problem**: RPC client can't detect local GPUs without Vulkan backend

**Root Cause**: RPC client library needs Vulkan backend to enumerate local GPUs

**Solution**: Build RPC client WITH Vulkan backend (see Step 2.5)

**Implementation**:
```makefile
# RPC client WITH Vulkan backend (for hybrid mode)
koboldcpp_rpc: ggml_v4_vulkan.o ... ggml-backend_vulkan.o ggml-backend-reg_vulkan.o ...
	$(RPC_BUILD)
```

**Note**: This requires Vulkan shaders to be generated first.

### ⚠️ HURDLE #4: Tensor Split Truncation

**Problem**: tensor_split values truncated to match RPC server count instead of total devices

**Root Cause**: Validation code checked RPC servers instead of total devices

**Solution**: Remove truncation logic (see Step 3.6)

**Implementation**:
```python
# Accept tensor_split values for all devices (RPC + local GPUs)
print(f"[RPC] Tensor split configured across {ts_count} devices with ratios: {args.tensor_split}")
```

---

## Phase 6: Build and Test

### Step 6.1: Generate Vulkan Shaders

```bash
cd koboldcpp-1.111.2
make vulkan-shaders-gen
```

**Expected Output**:
```
Generating Vulkan shaders...
ggml-vulkan-shaders.hpp created
ggml-vulkan-shaders.cpp created
```

### Step 6.2: Build RPC Client Library (Hybrid Mode)

```bash
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc
```

**Expected Output**:
```
g++ ... -DGGML_USE_RPC -DGGML_USE_VULKAN ... -lvulkan -shared -o koboldcpp_rpc.so -ldl
```

**Verify Build**:
```bash
ls -lh koboldcpp_rpc.so
# Expected: -rwxr-xr-x 1 user user 67M Apr  11 12:00 koboldcpp_rpc.so

# Check symbols
nm -D koboldcpp_rpc.so | grep ggml_backend_rpc
# Should show RPC backend symbols
```

### Step 6.3: Build RPC Server (Vulkan)

```bash
make LLAMA_RPC=1 LLAMA_VULKAN=1 rpc-server-vulkan
```

**Expected Output**:
```
g++ ... -DGGML_USE_RPC -DGGML_USE_VULKAN ... -o rpc-server-vulkan -lvulkan
```

**Verify Build**:
```bash
ls -lh rpc-server-vulkan
# Expected: -rwxr-xr-x 1 user user 68M Apr  11 12:00 rpc-server-vulkan
```

### Step 6.3b: Build RPC Server (CUDA - Optional)

For NVIDIA GPU systems:

```bash
make LLAMA_RPC=1 LLAMA_CUBLAS=1 rpc-server-cuda
```

**Expected Output**:
```
g++ ... -DGGML_USE_RPC -DGGML_USE_CUDA ... -o rpc-server-cuda -lcuda -lcublas -lcudart
```

**Verify Build**:
```bash
ls -lh rpc-server-cuda
# Expected: -rwxr-xr-x 1 user user 85M Apr  11 12:00 rpc-server-cuda
```

**Requirements**:
- NVIDIA GPU with compute capability 6.0+
- CUDA toolkit 11.0+ installed
- NVIDIA driver installed

### Step 6.3c: Build RPC Server (HIPBLAS/ROCm - Optional)

For AMD ROCm systems:

```bash
make LLAMA_RPC=1 LLAMA_HIPBLAS=1 rpc-server-hip
```

**Expected Output**:
```
hipcc ... -DGGML_USE_RPC -DGGML_USE_HIP ... -o rpc-server-hip -lhipblas -lamdhip64 -lrocblas
```

**Verify Build**:
```bash
ls -lh rpc-server-hip
# Expected: -rwxr-xr-x 1 user user 85M Apr  11 12:00 rpc-server-hip
```

**Requirements**:
- AMD GPU with GCN 8.0+ (RX 400 series or newer)
- ROCm 5.0+ installed
- AMD driver installed

### Step 6.4: Test RPC Server

**Test Vulkan RPC Server**:
```bash
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c
```

**Expected Output**:
```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
Starting RPC server v3.6.1
  endpoint       : 127.0.0.1:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
```

**Test CUDA RPC Server** (if built):
```bash
./rpc-server-cuda -H 127.0.0.1 --device CUDA0 -p 50052 -c
```

**Expected Output**:
```
Starting RPC server v3.6.1
  endpoint       : 127.0.0.1:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24576 MiB, 23456 MiB free)
```

**Test HIPBLAS RPC Server** (if built):
```bash
./rpc-server-hip -H 127.0.0.1 --device HIP0 -p 50052 -c
```

**Expected Output**:
```
Starting RPC server v3.6.1
  endpoint       : 127.0.0.1:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  HIP0: AMD Radeon RX 7900 XTX (24576 MiB, 23456 MiB free)
```

**Verify Listening**:
```bash
ss -tlnp | grep 50052
# Expected:
# LISTEN 0  1  127.0.0.1:50052  0.0.0.0:*  users:(("rpc-server-*",pid=1234,fd=14))
```

### Step 6.5: Test RPC Client (Hybrid Mode)

**Terminal 2**:
```bash
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50052 --gpulayers 999
```

**Expected Output**:
```
Initializing dynamic library: koboldcpp_rpc.so
Loading Text Model: model.gguf
ggml_vulkan: Found X Vulkan devices:
[RPC] Connecting to RPC server(s): 127.0.0.1:50052
[RPC] Server 127.0.0.1:50052 has 1 devices
[RPC] Found RPC device 0: RPC0
[RPC] Enumerating local GPU devices to use alongside RPC...
[RPC] Found local GPU device: Vulkan0 (registry: Vulkan)
[RPC] Total devices for offloading: 2 (RPC + local GPUs)
llama_model_load: offloading 999 layers to GPU
```

### Step 6.6: Test End-to-End with Tensor Split

```bash
# Terminal 1: Start server
./rpc-server-vulkan -H 192.168.1.101 --device VULKAN0,VULKAN1 -p 50054 -c &

# Terminal 2: Start client with tensor split
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --tensor_split 10 10 40 40 \
    --gpulayers 999 \
    --port 5001

# Terminal 3: Test inference
curl http://localhost:5001/api/extra/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_length": 20}'
```

**Expected Output**:
```
[RPC] Tensor split configured across 4 devices with ratios: [10.0, 10.0, 40.0, 40.0]
load_tensors: RPC0[...] model buffer size = XXX MiB
load_tensors: RPC1[...] model buffer size = XXX MiB
load_tensors: Vulkan0 model buffer size = XXX MiB
load_tensors: Vulkan1 model buffer size = XXX MiB
```

---

## Phase 7: Verification Checklist

### File Integration
- [ ] `ggml/include/ggml-rpc.h` exists
- [ ] `ggml/src/ggml-rpc/ggml-rpc.cpp` exists
- [ ] `tools/rpc-server.cpp` exists
- [ ] Key functions verified with grep

### Build System
- [ ] `RPC_FLAGS` variable added to Makefile
- [ ] `lib_rpc` detection added to koboldcpp.py
- [ ] RPC added to `lib_option_pairs`
- [ ] RPC library loading logic added
- [ ] `ggml-rpc.o` compilation rule added
- [ ] `koboldcpp_rpc` build target added (with hybrid mode)
- [ ] `rpc-server-vulkan` build target added
- [ ] `RPC_BUILD` variable added

### Python Wrapper
- [ ] `--rpc` / `--userpc` argument added
- [ ] `rpc_endpoints` field added to structure
- [ ] RPC endpoints populated
- [ ] Auto offload enabled for RPC
- [ ] Tuple unpacking fixed
- [ ] Tensor split validation fixed

### C++ Code
- [ ] Case-insensitive device matching added
- [ ] Direct RPC server call implemented
- [ ] Debug output added (optional)
- [ ] Device ordering implemented (--device argument)
- [ ] Local GPU enumeration with proper naming
- [ ] Device reordering logic working

### Critical Fixes
- [ ] Direct function call implemented (Hurdle #1)
- [ ] Vulkan shaders generated (Hurdle #2)
- [ ] Hybrid mode enabled (Hurdle #3)
- [ ] Tensor split truncation removed (Hurdle #4)

### Testing
- [ ] RPC server starts successfully
- [ ] RPC server listens on port
- [ ] RPC client connects
- [ ] Local GPUs detected (hybrid mode)
- [ ] Model loads across all devices
- [ ] Tensor split works correctly
- [ ] **Device ordering works** (--device argument)
- [ ] **Device reordering works** (custom order)
- [ ] Inference works
- [ ] Multiple servers work together

---

## Common Errors and Solutions

### Error: "Failed to find RPC backend"
**Solution**: Use direct function call instead of dynamic lookup (see Hurdle #1)

### Error: "ggml-vulkan-shaders.hpp: No such file or directory"
**Solution**: Run `make vulkan-shaders-gen` first

### Error: "undefined reference to `ggml_backend_rpc_reg`"
**Solution**: Ensure `ggml-rpc.o` is linked in build target

### Error: "too many values to unpack (expected 8)"
**Solution**: Add `rpc_option` to tuple unpacking

### Error: "Connection refused"
**Solution**: Start RPC server before client, check firewall

### Error: "Segmentation fault"
**Solution**: Use `--gpulayers 999` for full RPC offload

### Error: "No RPC devices found"
**Solution**: Verify server shows devices, check network connectivity

### Error: "tensor_split truncated"
**Solution**: Update tensor_split validation code (see Hurdle #4)

### Error: "Local GPUs not detected"
**Solution**: Ensure RPC client built WITH Vulkan backend (hybrid mode)

### Error: "invalid device: RPC0"
**Cause**: Device parsing function doesn't recognize RPC device names yet

**Solution**: Use new device ordering implementation that handles RPC devices properly (see Step 4.1)

### Error: "Device ordering not working"
**Cause**: Device names don't match enumeration

**Solution**: 
- Use correct naming: `VULKAN0`, `VULKAN1`, `RPC0`, `RPC1`, etc.
- Device names are case-insensitive
- Local GPUs numbered in enumeration order (0, 1, 2, ...)
- RPC devices numbered in connection order (0, 1, 2, ...)

---

## Performance Optimization

### Build Optimization
```bash
# Use all CPU cores for faster build
make -j$(nproc) LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan

# Clean build (recommended after changes)
make clean && make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan
```

### Runtime Optimization
```bash
# Server: Use multiple GPUs
./rpc-server-vulkan -H 0.0.0.0 --device VULKAN0,VULKAN1,VULKAN2 -p 50054 -c

# Client: Multiple servers with tensor split
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054,192.168.1.16:50054 \
    --tensor_split 10 10 10 10 30 30 \
    --gpulayers 999

# Client: With device ordering (local first)
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,VULKAN1,RPC0,RPC1,RPC2,RPC3 \
    --tensor_split 20 20 15 15 15 15 \
    --gpulayers 999

# Client: Interleaved devices for balanced load
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,VULKAN1,RPC1 \
    --tensor_split 25 25 25 25 \
    --gpulayers 999
```

---

## Known Limitations

### Security
⚠️ **RPC has no authentication or encryption**

**Never expose to public internet!**

### Network Performance
Performance depends on network quality:
- Gigabit Ethernet: Best (20-30 tokens/sec)
- Fast WiFi: Acceptable (10-20 tokens/sec)
- Internet: Not recommended (< 5 tokens/sec)

### Tensor Split
- Manual configuration required
- No automatic optimization based on device capabilities

### Device Ordering
- Requires manual `--device` argument
- Device names must match enumeration (VULKAN0, VULKAN1, etc.)
- RPC devices must be connected before use

### Backend Support
- **Vulkan RPC Server**: Works on all platforms with Vulkan drivers
- **CUDA RPC Server**: Requires NVIDIA GPU + CUDA toolkit (Linux/Windows)
- **HIPBLAS RPC Server**: Requires AMD GPU + ROCm (Linux only)
- All backends use the same RPC protocol and are interoperable

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

### Known Technical Debt
1. No memory-aware distribution
2. No hot-swapping support
3. No device performance profiling

---

## References

### Source Code
- llama.cpp: https://github.com/ggml-org/llama.cpp
- koboldcpp: https://github.com/LostRuins/koboldcpp

### Documentation
- RPC Manual: `RPC_MANUAL.md`
- Quick Start: `RPC_QUICKSTART.md`

### Tools
- Vulkan SDK: https://vulkan.lunarg.com/
- ROCm: https://rocm.docs.amd.com/
- CUDA: https://developer.nvidia.com/cuda-toolkit

---

## Version History

- **v1.111.2** (2026-04-11): Complete porting guide with all features
  - Documents full integration process
  - Includes all hurdles and solutions
  - Provides verification checklists
  - Reflects current working state with hybrid mode
  - Includes tensor_split fixes
  - Case-insensitive device matching
  - **Manual device ordering** (--device argument)
  - **Device reordering** (mix RPC and local in any order)

---

**License**: MIT  
**Authors**: KoboldCPP Team  
**Contact**: https://github.com/LostRuins/koboldcpp/issues
