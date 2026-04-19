# RPC Porting Guide - llama.cpp to koboldcpp

**Version**: 1.111.2  
**Date**: 2026-04-18  
**Purpose**: Complete step-by-step guide to port RPC from llama.cpp to koboldcpp  
**Target Audience**: Developers, LLMs, or anyone needing to replicate the integration  
**Status**: ✅ Complete - All Features Working (v1.111.2 with RPC endpoint fields, config save/load, and independent CUDA+HIPBLAS builds)

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
✅ **RPC Server mode** (launch RPC server from GUI or CLI)  
✅ **RPC Server Backend dropdown** (Auto-detect, Vulkan, hipBLAS, CUDA)  
✅ **Device name conversion** (VULKAN0 -> ROCm0, etc.)  
✅ **Allow Launch Without Models** for RPC Server mode  
✅ **RPC endpoint field** in Quick Launch and Hardware tabs (shared input, auto-sync)  
✅ **Config save/load** for all RPC settings (.kcpps files) including backend selection (runopts)  
✅ **ROCm device registry fix** (added "ROCM" to device enum)  
✅ **has_rpc_in_device initialization** in import_vars() - now also checks saved RPC endpoint  
✅ **Independent CUDA+HIPBLAS builds** - both built when both toolchains present (no mutual exclusion)  
✅ **runopts saved to config** - backend selection persists across reloads  

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

### Step 2.2: Add RPC Object File Build Rule

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

### Step 2.3: Add RPC Build Variable Definitions

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

### Step 2.4: Add RPC Client Build Targets

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: After line ~947 (after other koboldcpp_* targets)

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

# RPC client for HIPBLAS backend (AMD)
ifdef HIPBLAS_BUILD
koboldcpp_hipblas_rpc: ggml_v4_cublas.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o \
    ggml-unops.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o \
    gpttype_adapter_cublas.o ggml-rpc.o sdcpp_cublas.o whispercpp_cublas.o \
    tts_default.o music_default.o embeddings_default.o llavaclip_cublas.o llava.o \
    ggml-backend_cublas.o ggml-backend-reg_cublas.o ggml-repack.o $(HIP_OBJS) \
    $(OBJS_FULL) $(OBJS)
	$(HIPBLAS_BUILD)
else
koboldcpp_hipblas_rpc:
	$(DONOTHING)
endif

# RPC client for CUDA backend (NVIDIA)
ifdef CUBLAS_BUILD
koboldcpp_cublas_rpc: ggml_v4_cublas.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o \
    ggml-unops.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o \
    gpttype_adapter_cublas.o ggml-rpc.o sdcpp_cublas.o whispercpp_cublas.o \
    tts_default.o music_default.o embeddings_default.o llavaclip_cublas.o llava.o \
    ggml-backend_cublas.o ggml-backend-reg_cublas.o ggml-repack.o $(CUBLAS_OBJS) \
    $(OBJS_FULL) $(OBJS)
	$(CUBLAS_BUILD)
else
koboldcpp_cublas_rpc:
	$(DONOTHING)
endif
```

### Step 2.5: Add RPC Server Build Targets

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: After line ~994 (after tool build targets)

**Add**:
```makefile
# RPC server build targets
ifdef VULKAN_BUILD
rpc-server-vulkan: tools/rpc-server.cpp ggml/src/ggml-vulkan-shaders.cpp \
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
rpc-server-cuda: tools/rpc-server.cpp \
    ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o \
    llama.o ggml-rpc.o ggml-backend_cublas.o ggml-backend-reg_cublas.o \
    ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o \
    ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
    unicode.o unicode-common.o unicode-data.o ggml-threading.o \
    ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o \
    budget.o kcpputils.o console.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o $(CUBLAS_OBJS)
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(CUBLASLD_FLAGS)
endif

# RPC server for HIPBLAS backend (AMD ROCm GPUs)
ifdef HIPBLAS_BUILD
rpc-server-hip: tools/rpc-server.cpp \
    ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o \
    llama.o ggml-rpc.o ggml-backend_cublas.o ggml-backend-reg_cublas.o \
    ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o \
    ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
    unicode.o unicode-common.o unicode-data.o ggml-threading.o \
    ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o \
    budget.o kcpputils.o console.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o $(HIP_OBJS)
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)
endif
```

**Important**: Each RPC server backend uses its respective GPU backend:
- `rpc-server-vulkan` - Uses Vulkan backend (works with AMD/Intel/NVIDIA via Vulkan)
- `rpc-server-cuda` - Uses CUDA backend (NVIDIA GPUs only)
- `rpc-server-hip` - Uses HIPBLAS backend (AMD ROCm GPUs only)

### Step 2.6: Add Automatic Backend Detection (rpc-full-all)

**File**: `koboldcpp-1.111.2/Makefile`

**Location**: At end of Makefile

**Add**:
```makefile
# Automatic backend detection and full RPC build
.PHONY: rpc-full-all
rpc-full-all:
	@echo "=== Detecting available backends ==="
	@echo "Checking for NVIDIA CUDA (nvcc)..."
	@if command -v nvcc &> /dev/null; then \
		echo "✓ CUDA: Available (nvcc found)"; \
		HAS_CUDA=1; \
	else \
		echo "✗ CUDA: Not available"; \
		HAS_CUDA=0; \
	fi
	@echo "Checking for AMD HIPBLAS (hipcc)..."
	@if command -v hipcc &> /dev/null; then \
		echo "✓ HIPBLAS: Available (hipcc found)"; \
		HAS_HIPBLAS=1; \
	else \
		echo "✗ HIPBLAS: Not available"; \
		HAS_HIPBLAS=0; \
	fi
	@echo ""
	@echo "=== Building Regular Backends (Non-RPC) ==="
	@echo "Building Vulkan backend..."
	$(MAKE) LLAMA_VULKAN=1 koboldcpp_vulkan -j8
ifeq ($(HAS_HIPBLAS),1)
	@echo "Building HIPBLAS backend..."
	$(MAKE) LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8
else
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "Building CUDA backend..."; \
		$(MAKE) LLAMA_CUBLAS=1 koboldcpp_cublas -j8; \
	fi
endif
	@echo ""
	@echo "=== Building Vulkan + RPC (universal fallback) ==="
	$(MAKE) LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all -j8
ifeq ($(HAS_HIPBLAS),1)
	@echo ""
	@echo "=== Building HIPBLAS + RPC (AMD GPUs) ==="
	@echo "HIPBLAS detected, building..."
	$(MAKE) LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
	$(MAKE) LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
endif
ifeq ($(HAS_CUDA),1)
ifeq ($(HAS_HIPBLAS),0)
	@echo ""
	@echo "=== Building CUDA + RPC (NVIDIA GPUs) ==="
	@echo "CUDA detected, building..."
	$(MAKE) LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8
	$(MAKE) LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
endif
endif
	@echo ""
	@echo "=== All RPC components built successfully! ==="
```

---

## Phase 3: Python Wrapper Integration

### Step 3.1: Add RPC Server Mode to GUI Variables

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~10420 (in GUI variable section)

**Add RPC Server variables**:
```python
# RPC Server variables
rpc_server_mode_var = ctk.IntVar(value=0)
rpc_server_backend_var = ctk.StringVar(value="Auto-detect")
rpc_host_var = ctk.StringVar(value="0.0.0.0")
rpc_port_var = ctk.StringVar(value="50053")
rpc_devices_var = ctk.StringVar(value="")
```

### Step 3.2: Add RPC Server Tab to GUI

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~12670 (in GUI tab creation section)

**Add RPC Server tab**:
```python
# RPC Server Tab Section
ctk.CTkLabel(
    rpc_tab,
    text="RPC Server Configuration",
    fg_color="transparent",
    text_color="#5DA5E5",
    font=("Helvetica", 14, "bold"),
).grid(row=0, column=0, columnspan=2, sticky="w", padx=0, pady=10)

makecheckbox(
    rpc_tab,
    "Start RPC Server Mode",
    rpc_server_mode_var,
    1,
    0,
    tooltiptxt="Start KoboldCPP in RPC server mode. Exposes local GPUs for remote clients to use.",
)

# RPC Server Backend dropdown
makelabel(rpc_tab, "RPC Server Backend:", 2, 0, padx=0)
makelabel(rpc_tab, "GPU backend for RPC server. Must match device names below.", 3, 0, padx=0)
rpc_backend_options = ["Auto-detect", "Vulkan", "hipBLAS (ROCm)", "CUDA"]
rpc_server_backend_dropdown = ctk.CTkComboBox(
    rpc_tab,
    values=rpc_backend_options,
    variable=rpc_server_backend_var,
    width=200,
    state="readonly",
)
rpc_server_backend_dropdown.grid(row=4, column=0, columnspan=2, sticky="w", padx=0, pady=5)
makelabel(
    rpc_tab,
    "Vulkan: Use VULKAN0,VULKAN1...  |  hipBLAS: Use ROCm0,ROCm1...  |  CUDA: Use CUDA0,CUDA1...",
    5, 0, padx=0
)

# Listening IP Address
makelabel(rpc_tab, "Listening IP Address:", 6, 0, padx=0)
makelabel(
    rpc_tab,
    "IP address for RPC server to listen on. Use 0.0.0.0 for all interfaces, 127.0.0.1 for localhost only.",
    7, 0, padx=0
)
makelabelentry(rpc_tab, "", rpc_host_var, 8, 150, singleline=True)

# Listening Port
makelabel(rpc_tab, "Listening Port:", 9, 0, padx=0)
makelabel(
    rpc_tab,
    "Port number for RPC server connections (default: 50053).",
    10, 0, padx=0
)
makelabelentry(rpc_tab, "", rpc_port_var, 11, 100, singleline=True)

# RPC Devices
makelabel(rpc_tab, "RPC Devices:", 12, 0, padx=0)
makelabel(
    rpc_tab,
    "Comma-separated list of GPUs to expose via RPC. Leave empty to auto-detect all devices.",
    13, 0, padx=0
)
makelabelentry(rpc_tab, "", rpc_devices_var, 14, 200, singleline=True)

makecheckbox(
    rpc_tab,
    "Allow Launch Without Models",
    nomodel,
    16, 0,
    tooltiptxt="Allows starting RPC server without loading a model file.",
)

# Warning message (14pt red bold)
ctk.CTkLabel(
    rpc_tab,
    text="WARNING: RPC Server mode replaces the WebUI API. Clients must connect via --rpc.",
    text_color="red",
    font=("Helvetica", 14, "bold"),
).grid(row=18, column=0, columnspan=2, sticky="w", padx=0, pady=10)
```

### Step 3.3: Add RPC Server Arguments to Argument Parser

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~17225 (after --userpc argument)

**Add RPC Server arguments**:
```python
parser.add_argument(
    "--start-rpc-server",
    action="store_true",
    help="Start as RPC server mode instead of client/server. Exposes local GPUs via RPC.",
)
parser.add_argument(
    "--rpc-host",
    default="0.0.0.0",
    help="Host address for RPC server (default: 0.0.0.0). Use 127.0.0.1 for localhost only.",
)
parser.add_argument(
    "--rpc-port",
    type=int,
    default=50053,
    help="Port for RPC server (default: 50053).",
)
parser.add_argument(
    "--rpc-devices",
    default="",
    metavar=("[dev1,dev2,...]"),
    help="Comma-separated list of devices to expose via RPC server (e.g. VULKAN0,VULKAN1 or ROCm0,ROCm1). Leave empty to expose all available devices.",
)
parser.add_argument(
    "--rpc-server-backend",
    default="Auto-detect",
    choices=["Auto-detect", "Vulkan", "hipBLAS (ROCm)", "CUDA"],
    help="GPU backend for RPC server. Vulkan uses VULKAN* devices, hipBLAS uses ROCm* devices, CUDA uses CUDA* devices. Default: Auto-detect.",
)
```

### Step 3.4: Add RPC Server Args Population

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In export_vars function (around line ~13385)

**Add**:
```python
# RPC Server arguments
args.start_rpc_server = rpc_server_mode_var.get() == 1
args.rpc_host = rpc_host_var.get()
args.rpc_port = int(rpc_port_var.get()) if rpc_port_var.get() else 50053
args.rpc_devices = rpc_devices_var.get()
args.rpc_server_backend = rpc_server_backend_var.get()
```

### Step 3.5: Add RPC Server Mode Launcher

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~16965 (before main server logic)

**Add RPC Server mode implementation**:
```python
# RPC Server Mode
if getattr(args, "start_rpc_server", False):
    print("=" * 60)
    print("Starting KoboldCPP in RPC Server Mode")
    print("=" * 60)
    print(f"RPC Server Host: {args.rpc_host}")
    print(f"RPC Server Port: {args.rpc_port}")

    # Determine backend from dropdown or auto-detect
    backend = getattr(args, "rpc_server_backend", "Auto-detect")
    if not backend:
        backend = "Auto-detect"
    
    rpc_server_path = None
    device_prefix = None
    
    if backend == "Vulkan":
        device_prefix = "VULKAN"
        rpc_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpc-server-vulkan")
    elif backend == "CUDA":
        device_prefix = "CUDA"
        rpc_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpc-server-cuda")
    elif backend == "hipBLAS (ROCm)":
        device_prefix = "ROCm"
        rpc_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpc-server-hip")
    else:
        # Auto-detect: try in order of preference
        print("Auto-detecting RPC server backend...")
        for candidate in ["rpc-server-hip", "rpc-server-vulkan", "rpc-server-cuda"]:
            candidate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), candidate)
            if os.path.exists(candidate_path):
                if "hip" in candidate:
                    device_prefix = "ROCm"
                elif "cuda" in candidate:
                    device_prefix = "CUDA"
                else:
                    device_prefix = "VULKAN"
                rpc_server_path = candidate_path
                print(f"Auto-detected: {candidate} (device prefix: {device_prefix})")
                break

    # Build and validate device list
    devices = []
    if args.rpc_devices and args.rpc_devices.strip():
        devices = [d.strip() for d in args.rpc_devices.split(",") if d.strip()]
    
    # Convert device names if prefix is specified and devices are provided
    if device_prefix and devices:
        converted_devices = []
        for dev in devices:
            dev_upper = dev.upper()
            if dev_upper.startswith("VULKAN"):
                if device_prefix != "VULKAN":
                    dev_num = dev_upper.replace("VULKAN", "")
                    converted_devices.append(f"{device_prefix}{dev_num}")
                    print(f"  Converting {dev} -> {converted_devices[-1]}")
                else:
                    converted_devices.append(dev)
            elif dev_upper.startswith("ROCm") or dev_upper.startswith("HIP"):
                if device_prefix != "VULKAN":
                    dev_num = dev_upper.replace("ROCm", "").replace("HIP", "")
                    if device_prefix == "CUDA":
                        converted_devices.append(f"CUDA{dev_num}")
                    else:
                        converted_devices.append(f"{device_prefix}{dev_num}")
                    print(f"  Converting {dev} -> {converted_devices[-1]}")
                else:
                    converted_devices.append(dev)
            elif dev_upper.startswith("CUDA"):
                if device_prefix != "VULKAN":
                    dev_num = dev_upper.replace("CUDA", "")
                    converted_devices.append(f"{device_prefix}{dev_num}")
                    print(f"  Converting {dev} -> {converted_devices[-1]}")
                else:
                    converted_devices.append(dev)
            else:
                converted_devices.append(dev)
        devices = converted_devices

    if devices:
        print(f"Devices to expose: {', '.join(devices)}")

    print("")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if args.rpc_host != "127.0.0.1" and args.rpc_host != "0.0.0.0":
        print("WARNING: RPC server bound to specific non-loopback address!")
    if args.rpc_host == "0.0.0.0":
        print("WARNING: Host is '0.0.0.0' - RPC server accepts connections on all interfaces!")
    print("         NEVER expose the RPC server to an open network!")
    print("         This is an experimental feature and is not secure!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("")

    if rpc_server_path is None:
        print("ERROR: RPC server binary not found!")
        print("       Please build with: make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-full-all")
        sys.exit(1)

    print(f"Launching RPC server: {rpc_server_path}")
    cmd = [rpc_server_path, "-H", args.rpc_host, "--port", str(args.rpc_port)]
    if devices:
        cmd.extend(["--device", ",".join(devices)])
    cmd.append("-c")

    print(" ".join(cmd))
    print("")
    print("RPC server running. Clients can connect via --rpc")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: RPC server failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nRPC server stopped.")
    sys.exit(0)
```

### Step 3.6: Skip Library Loading for RPC Server Mode

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In kcpp_main_process function (around line ~16278)

**Change**:
```python
# OLD CODE:
init_library()  # Note: if blas does not exist and is enabled, program will crash.
print("==========")
time.sleep(1)

# NEW CODE:
# Skip library loading when RPC Server mode is enabled (subprocess-based)
if getattr(args, "start_rpc_server", False):
    print("RPC Server mode enabled, skipping KoboldCpp API library loading")
else:
    init_library()  # Note: if blas does not exist and is enabled, program will crash.
    print("==========")
    time.sleep(1)
```

---

### Step 3.7: Add RPC Endpoint Field (Shared Between Tabs)

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: After line ~10478 (global variable declarations)

**Add RPC endpoint variable**:
```python
# Shared RPC endpoint variable for both Quick Launch and Hardware tabs
rpc_endpoint_var = ctk.StringVar(value="")
```

**Location**: In Quick Launch tab creation section (around row=2)

**Add RPC endpoint field to Quick Launch**:
```python
rpc_endpoint_label = ctk.CTkLabel(
    quick_tab,
    text="RPC Server Endpoint:",
    font=("Helvetica", 10, "bold"),
)
rpc_endpoint_entry = ctk.CTkEntry(
    quick_tab,
    width=300,
    textvariable=rpc_endpoint_var,
    placeholder_text="e.g. 192.168.1.101:50053",
)
rpc_endpoint_label.grid(row=2, column=0, padx=8, pady=1, stick="nw")
rpc_endpoint_entry.grid(row=2, column=1, padx=8, pady=1, stick="nw")
```

**Location**: In Hardware tab creation section (around row=2)

**Add RPC endpoint field to Hardware (shared)**:
```python
rpc_endpoint_label_hw = ctk.CTkLabel(
    hardware_tab,
    text="RPC Server Endpoint:",
    font=("Helvetica", 10, "bold"),
)
rpc_endpoint_entry_hw = ctk.CTkEntry(
    hardware_tab,
    width=300,
    textvariable=rpc_endpoint_var,  # ← Same variable - auto-syncs
)
rpc_endpoint_label_hw.grid(row=2, column=0, padx=160, pady=1, stick="nw")
rpc_endpoint_entry_hw.grid(row=2, column=0, padx=250, pady=1, stick="nw")
```

**Location**: In `changerunmode()` function

**Add show/hide logic**:
```python
# Show RPC endpoint field for RPC variants only
rpc_selected = index in ["Use Vulkan + RPC", "Use hipBLAS + RPC", "Use CUDA + RPC"]
if rpc_selected:
    rpc_endpoint_label.grid(row=2, column=0, padx=8, pady=1, stick="nw")
    rpc_endpoint_entry.grid(row=2, column=1, padx=8, pady=1, stick="nw")
    rpc_endpoint_label_hw.grid(row=2, column=0, padx=160, pady=1, stick="nw")
    rpc_endpoint_entry_hw.grid(row=2, column=0, padx=250, pady=1, stick="nw")
else:
    rpc_endpoint_label.grid_remove()
    rpc_endpoint_entry.grid_remove()
    rpc_endpoint_label_hw.grid_remove()
    rpc_endpoint_entry_hw.grid_remove()
```

**Purpose**: Creates shared RPC endpoint input field in both Quick Launch and Hardware tabs. Both fields use the same `rpc_endpoint_var` StringVar so they auto-sync.

---

### Step 3.8: Add Config Save/Load for RPC Settings

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In `export_vars()` function (around line ~13283)

**Add RPC endpoint to args population**:
```python
# Use RPC endpoint from GUI field if RPC variant is selected
if rpc_selected and rpc_endpoint_var.get() != "":
    args.userpc = [rpc_endpoint_var.get()]
```

**Location**: In `import_vars()` function (around line ~13526)

**Initialize RPC variables to prevent NameError**:
```python
def import_vars():
    global args, ...
    global has_rpc_in_device, has_rocm_in_device  # ← NEW
    
    has_rpc_in_device = False
    if args.device and "RPC" in args.device:
        has_rpc_in_device = True
    
    has_rocm_in_device = False
    if args.device and ("ROCm" in args.device or "HIP" in args.device):
        has_rocm_in_device = True
    
    # ... (existing function continues)
```

**Location**: When loading config (around line ~13669)

**Add RPC endpoint loading**:
```python
# Load RPC endpoint from userpc (legacy) or rpc_endpoint
if "userpc" in dict:
    if isinstance(dict["userpc"], list):
        rpc_endpoint_var.set(dict["userpc"][0])
    else:
        rpc_endpoint_var.set(str(dict["userpc"]))

# Load from rpc_endpoint (new format)
if "rpc_endpoint" in dict:
    rpc_endpoint_var.set(str(dict["rpc_endpoint"]))
```

**Location**: In `convert_args_to_template()` function (around line ~14907)

**Add RPC endpoint saving**:
```python
savdict["rpc_endpoint"] = rpc_endpoint_var.get()
```

**Purpose**: Enables saving and loading RPC settings (including endpoint) in `.kcpps` config files. Also initializes `has_rpc_in_device` and `has_rocm_in_device` to prevent NameError when loading configs.

---

## Phase 4: Library Detection and Selection

### Step 4.1: Add RPC-Enabled Library Variables

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: Lines ~940-960 (library detection section)

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
```

### Step 4.2: Update Library Selection Logic

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In `init_library()` function (around line ~1031)

**Add RPC library preference and ROCm/HIP detection**:
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
        need_rpc = args.userpc is not None or has_rpc_in_device
        if file_exists(lib_cublas_rpc) and need_rpc:
            libname = lib_cublas_rpc
        elif file_exists(lib_hipblas_rpc) and need_rpc:
            libname = lib_hipblas_rpc
        elif file_exists(lib_cublas):
            libname = lib_cublas
        elif file_exists(lib_hipblas):
            libname = lib_hipblas
        elif file_exists(lib_rpc) and need_rpc:
            libname = lib_rpc
        else:
            print("WARNING: No suitable GPU library found")
    
    elif args.usevulkan is not None:
        if file_exists(lib_rpc) and (args.userpc is not None or has_rpc_in_device):
            libname = lib_rpc
        elif file_exists(lib_vulkan):
            libname = lib_vulkan
        elif file_exists(lib_vulkan_noavx2):
            libname = lib_vulkan_noavx2
    
    elif args.userpc is not None:
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
    
    elif has_rocm_in_device:
        if file_exists(lib_hipblas_rpc):
            libname = lib_hipblas_rpc
        elif file_exists(lib_hipblas):
            libname = lib_hipblas
```

### Step 4.3: Update GUI Auto-Selection for RPC Server Mode

**File**: `koboldcpp-1.111.2/koboldcpp.py`

**Location**: In `auto_set_backend_gui()` function (around line ~10867)

**Add RPC Server mode detection at start of function**:
```python
def auto_set_backend_gui(manual_select=False):
    global exitcounter, runmode_untouched
    
    # check for avx2 and avx support
    cpusupport = old_cpu_check()
    eligible_cuda = (cpusupport < 1 and not is_oldpc_ver) or (cpusupport < 2 and is_oldpc_ver)
    
    # Force RPC backend if RPC Server mode is enabled from GUI
    if rpc_server_mode_var.get() == 1:
        runmode_untouched = True
        if "Use hipBLAS + RPC" in runopts and eligible_cuda:
            runopts_var.set("Use hipBLAS + RPC")
            print("Auto Selected HIPBLAS + RPC Backend (RPC Server Mode)\n")
            found_new_backend = True
        elif "Use CUDA + RPC" in runopts and eligible_cuda:
            runopts_var.set("Use CUDA + RPC")
            print("Auto Selected CUDA + RPC Backend (RPC Server Mode)\n")
            found_new_backend = True
        elif "Use Vulkan + RPC" in runopts:
            runopts_var.set("Use Vulkan + RPC")
            print("Auto Selected Vulkan + RPC Backend (RPC Server Mode)\n")
            found_new_backend = True
        else:
            if "Use hipBLAS + RPC" in runopts:
                runopts_var.set("Use hipBLAS + RPC")
            elif "Use CUDA + RPC" in runopts:
                runopts_var.set("Use CUDA + RPC")
            elif "Use Vulkan + RPC" in runopts:
                runopts_var.set("Use Vulkan + RPC")
            found_new_backend = True
    
    # Detect RPC in device string
    has_rpc_in_device = False
    if args.device and "RPC" in args.device:
        has_rpc_in_device = True
    
    # ... (existing logic continues)
```

---

## Phase 5: C++ Code Modifications

### Step 5.1: Implement Device Ordering with Triple Naming

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

**Key Changes**:
- CUDA devices now registered with dual names (`CUDA0` + `HIP0`)
- HIP devices now registered with dual names (`HIP0` + `ROCm0`)
- Allows using `HIP0`, `CUDA0`, or `ROCm0` interchangeably
- Improves compatibility across different backend configurations

### Step 5.2: Fix Device Enumeration (Case-Insensitive)

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

### Step 5.3: Fix RPC Server Direct Call

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

## Phase 6: Testing and Verification

### Step 6.1: Verify Build

```bash
cd koboldcpp_rpc_attempt
make clean
make rpc-full-all -j8

# Verify all components were built
ls -lh koboldcpp_vulkan.so koboldcpp_hipblas.so
ls -lh koboldcpp_rpc.so koboldcpp_hipblas_rpc.so
ls -lh rpc-server-vulkan rpc-server-hip rpc-server-cuda
```

### Step 6.2: Test Server

```bash
# Test Vulkan RPC server
./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c

# Test HIPBLAS RPC server
./rpc-server-hip -H 127.0.0.1 --port 50054 --device ROCm0 -c

# Test CUDA RPC server
./rpc-server-cuda -H 127.0.0.1 --port 50054 --device CUDA0 -c
```

### Step 6.3: Test Client

```bash
# Test with RPC only
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50054 --gpulayers 999

# Test with hybrid mode
python koboldcpp.py --model model.gguf \
    --rpc 192.168.1.101:50054 \
    --device VULKAN0,RPC0,VULKAN1 \
    --gpulayers 999
```

### Step 6.4: Verify Dropdown Menu

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

### Step 6.5: Verify RPC Server Tab

1. Open `python koboldcpp.py`
2. Check for **RPC Server** tab between "Loaded Files" and "Network"
3. Verify fields:
   - Start RPC Server Mode (checkbox)
   - RPC Server Backend (dropdown)
   - Listening IP Address (text input)
   - Listening Port (text input)
   - RPC Devices (text input)
   - Allow Launch Without Models (checkbox)
   - Red warning message

### Step 6.6: Test RPC Server Mode from GUI

1. Go to **RPC Server** tab
2. Check "Start RPC Server Mode"
3. Set **RPC Server Backend**: `Auto-detect`
4. Set Listening IP Address: `127.0.0.1`
5. Set Listening Port: `50053`
6. Leave RPC Devices empty for auto-detect
7. Check "Allow Launch Without Models"
8. Launch
9. Verify output shows RPC server starting

### Step 6.7: Test RPC Server Mode from CLI

```bash
python koboldcpp.py \
    --start-rpc-server \
    --rpc-host 127.0.0.1 \
    --rpc-port 50053 \
    --rpc-devices VULKAN0 \
    --rpc-server-backend Vulkan
```

---

## Phase 7: Backend Compatibility

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
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8
```

### 3. CUDA/HIPBLAS Build Conflicts

**Error**:
```
undefined symbol: __hipRegisterFunction
```

**Cause**: Trying to build both CUDA and HIPBLAS together

**Solution**: `make rpc-full-all` now skips CUDA if HIPBLAS detected

### 4. Loading Non-RPC Library When RPC Needed

**Error**:
```
OSError: koboldcpp_hipblas.so: undefined symbol: ggml_backend_rpc_add_server
```

**Cause**: Non-RPC library loaded when RPC is required

**Solution**:
```bash
# Ensure RPC variant is built
make clean
make rpc-full-all -j8

# RPC Server mode skips library loading automatically
# For RPC client mode, ensure --rpc is used
```

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
- Code Changes: `RPC_koboldcpp.py_changes.md`

### Tools
- Vulkan SDK: https://vulkan.lunarg.com/
- ROCm: https://rocm.docs.amd.com/
- CUDA: https://developer.nvidia.com/cuda-toolkit

---

## Version History

- **v1.111.2** (2026-04-18): Complete porting guide with all latest features
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
  - **RPC Server mode** (launch RPC server from GUI or CLI)
  - **RPC Server Backend dropdown** (Auto-detect, Vulkan, hipBLAS, CUDA)
  - **Device name conversion** (VULKAN0 -> ROCm0, etc.)
  - **RPC variant libraries** (koboldcpp_hipblas_rpc.so, etc.)
  - **Backend compatibility matrix**
  - **Skip library loading** for RPC Server mode
  - **RPC endpoint field** shared between Quick Launch and Hardware tabs (auto-sync)
  - **Config save/load** for all RPC settings (.kcpps files) including backend selection (runopts)
  - **ROCm device registry fix** (added "ROCM" to device enum check)
  - **has_rpc_in_device initialization** in import_vars() - now also checks saved RPC endpoint
  - **Independent CUDA+HIPBLAS builds** - both built when both toolchains present
  - **runopts saved to config** - backend selection persists across reloads
  - **Object file cleanup** in rpc-full-all - prevents "undefined symbol" errors when mixing RPC and non-RPC builds
  - **gpttype_adapter.cpp RPC code guarded** - all RPC code wrapped in #ifdef GGML_USE_RPC (prevents RPC symbols in non-RPC builds)

- **v1.111.2** (2026-04-17): Added GUI dropdown updates and library selection logic

- **v1.111.2** (2026-04-16): Added dual naming and library selection fixes

- **v1.111.2** (2026-04-11): Initial porting guide with basic RPC integration

---

**License**: MIT  
**Authors**: KoboldCPP Team  
**Contact**: https://github.com/LostRuins/koboldcpp/issues
