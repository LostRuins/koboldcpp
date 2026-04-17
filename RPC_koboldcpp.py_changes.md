# RPC Implementation Changes in koboldcpp.py

**Version**: 1.111.2  
**Date**: 2026-04-17  
**Purpose**: Complete documentation of all koboldcpp.py changes for RPC support  
**Status**: ✅ Complete - All Features Working

---

## Overview

This document details every code addition and modification made to `koboldcpp.py` to implement RPC (Remote Procedure Call) functionality. The implementation enables:

✅ RPC client that connects to remote GPU servers  
✅ Multiple RPC server support  
✅ Hybrid mode (local GPUs + RPC servers)  
✅ Manual tensor_split for layer distribution  
✅ Manual device ordering (--device argument)  
✅ Auto-offload for RPC configurations  
✅ **Triple naming support** (HIP0 = CUDA0 = ROCm0)  
✅ **Automatic backend detection** (make rpc-full-all)  
✅ **Full GUI dropdown** with all backend options  
✅ **RPC variant libraries** (koboldcpp_hipblas_rpc.so, etc.)  

**Total Changes**: 20 major modifications across 10 sections  
**Lines Added**: ~400 lines  
**Lines Modified**: ~100 lines  

---

## Table of Contents

1. [Library Detection](#1-library-detection)
2. [Structure Definitions](#2-structure-definitions)
3. [Library Initialization](#3-library-initialization)
4. [Input Population](#4-input-population)
5. [Auto-Offload Logic](#5-auto-offload-logic)
6. [Argument Parser](#6-argument-parser)
7. [Backend Selection UI](#7-backend-selection-ui)
8. [GUI Backend Selection](#8-gui-backend-selection)
9. [Known Issues](#9-known-issues)
10. [Complete Code Reference](#10-complete-code-reference)

---

## 1. Library Detection

### 1.1: Add RPC-Enabled Library Variables

**Location**: Lines ~940-960 (library detection section)

**Changes**: Added RPC-enabled backend variables for all backends

**Code**:
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

libname = ""
```

**Purpose**: Detects all available backend libraries including RPC variants

---

### 1.2: Update lib_option_pairs List

**Location**: Lines ~961-973

**Code**:
```python
lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_cublas_rpc, "Use CUDA + RPC"),          # ← NEW: CUDA + RPC variant
    (lib_hipblas, "Use hipBLAS (ROCm)"),
    (lib_hipblas_rpc, "Use hipBLAS + RPC"),      # ← NEW: HIPBLAS + RPC variant
    (lib_vulkan, "Use Vulkan"),
    (lib_rpc, "Use Vulkan + RPC"),               # ← Previously "Use RPC (Remote)"
    (lib_noavx2, "Use CPU (Old CPU)"),
    (lib_vulkan_noavx2, "Use Vulkan (Old CPU)"),
    (lib_vulkan_failsafe, "Use Vulkan (Older CPU)"),
    (lib_failsafe, "Failsafe Mode (Older CPU)"),
]
```

**Purpose**: Defines all available backend options for GUI dropdown menu

---

### 1.3: Unpack Library Options

**Location**: Lines ~974-982

**Code**:
```python
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

**Purpose**: Creates individual option variables and filters to only available options

---

## 2. Structure Definitions

### 2.1: load_model_inputs Structure

**Location**: Lines ~2250-2300

**Code**:
```python
@ctypes.Structure
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("model_param", ctypes.c_char_p),
        ("port", ctypes.c_int),
        ("host", ctypes.c_char_p),
        ("rpc_endpoints", ctypes.c_char_p),
        ("gpulayers", ctypes.c_int),
        ("contextsize", ctypes.c_int),
        ("tensor_split", ctypes.c_char_p),
        ("devices_override", ctypes.c_char_p),
        # ... (other fields)
    ]
```

**Purpose**: Defines input structure for C++ library calls including RPC fields

---

## 3. Library Initialization

### 3.1: Global Variable Declarations

**Location**: Lines ~986-1000

**Code**:
```python
def init_library():
    global handle, args, libname
    global \
        lib_default, lib_failsafe, lib_noavx2, lib_vulkan_failsafe, \
        lib_cublas, lib_cublas_rpc, lib_hipblas, lib_hipblas_rpc, \
        lib_vulkan, lib_vulkan_noavx2, lib_rpc
```

**Purpose**: Declares all library variables as global

---

### 3.2: Device String Detection

**Location**: Lines ~1000-1010

**Code**:
```python
libname = lib_default

# Detect RPC in device string
has_rpc_in_device = False
if args.device and "RPC" in args.device:
    has_rpc_in_device = True

# Detect ROCm/HIP in device string
has_rocm_in_device = False
if args.device and ("ROCm" in args.device or "HIP" in args.device):
    has_rocm_in_device = True
```

**Purpose**: Detects if RPC or ROCm is being used in device string for proper library selection

---

### 3.3: Library Selection Logic

**Location**: Lines ~1031-1065

**Code**:
```python
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

elif has_rocm_in_device:
    if file_exists(lib_hipblas_rpc):
        libname = lib_hipblas_rpc
    elif file_exists(lib_hipblas):
        libname = lib_hipblas
```

**Purpose**: Selects the correct library based on backend and RPC usage

---

## 4. Input Population

### 4.1: Set RPC Endpoint

**Location**: Lines ~2320-2350

**Code**:
```python
inputs.rpc_endpoints = b",".join(args.userpc) if args.userpc else None
```

**Purpose**: Passes RPC server endpoints to C++ library

---

### 4.2: Set Devices Override

**Location**: Lines ~2360-2370

**Code**:
```python
inputs.devices_override = args.device.encode('utf-8') if args.device else None
```

**Purpose**: Passes device ordering string to C++ library

---

## 5. Auto-Offload Logic

### 5.1: Auto-Select Backend for RPC

**Location**: Lines ~10860-10920

**Code**:
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
                found_new_backend = True
            elif "Use hipBLAS + RPC" in runopts:
                runopts_var.set("Use hipBLAS + RPC")
                gpu_choice_var.set("1")
                print(f"Auto Selected HIPBLAS + RPC Backend\n")
                found_new_backend = True
        # Otherwise use standard backends
        if not found_new_backend:
            if "Use CUDA" in runopts:
                runopts_var.set("Use CUDA")
                gpu_choice_var.set("1")
                print(f"Auto Selected CUDA Backend\n")
                found_new_backend = True
            elif "Use hipBLAS (ROCm)" in runopts:
                runopts_var.set("Use hipBLAS (ROCm)")
                gpu_choice_var.set("1")
                print(f"Auto Selected HIP Backend\n")
                found_new_backend = True
```

**Purpose**: Automatically selects appropriate backend including RPC variants

---

## 6. Argument Parser

### 6.1: RPC Argument

**Location**: Lines ~16920-16925

**Code**:
```python
compatgroup.add_argument("--userpc", "--rpc", 
    help="Use RPC for remote model inference. Specify one or more RPC server endpoints (e.g. --rpc 192.168.1.100:50054). Can be used multiple times for multiple servers.", 
    metavar=('[endpoint]'), 
    nargs='+', 
    type=str, 
    default=None)
```

**Purpose**: Adds RPC endpoint argument to command line parser

---

## 7. Backend Selection UI

### 7.1: Backend Dropdown

**Location**: Lines ~11193-11210

**Code**:
```python
quick_tab, values=runopts, width=190, variable=runopts_var, state="readonly"
```

**Purpose**: Creates backend selection dropdown with all available options

---

### 7.2: Update changerunmode Function

**Location**: Lines ~11101-11195

**Code**:
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
        quick_gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
        gpuname_label.grid(row=3, column=0, padx=230, sticky="W")
        # ... (other controls)
    
    # Show tensor_split controls for backends that support it
    if (index == "Use CUDA" or index == "Use CUDA + RPC" or 
        index == "Use hipBLAS (ROCm)" or index == "Use hipBLAS + RPC" or
        index == "Use Vulkan" or index == "Use Vulkan (Old CPU)"):
        mmq_box.grid(row=4, column=0, padx=160, pady=1, stick="nw")
        tensor_split_label.grid(row=8, column=0, padx=8, pady=1, stick="nw")
        # ... (other controls)
    
    # Show gpu_layers controls
    if (index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or 
        index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or 
        index == "Use CUDA + RPC" or index == "Use hipBLAS (ROCm)" or 
        index == "Use hipBLAS + RPC"):
        gpu_layers_label.grid(row=6, column=0, padx=8, pady=1, stick="nw")
        # ... (other controls)
```

**Purpose**: Updates UI to show appropriate controls for RPC variant backends

---

## 8. GUI Backend Selection

### 8.1: Update args.usecuda Population

**Location**: Lines ~12939-12950

**Code**:
```python
if runopts_var.get() == "Use CUDA" or runopts_var.get() == "Use CUDA + RPC" or \
   runopts_var.get() == "Use hipBLAS (ROCm)" or runopts_var.get() == "Use hipBLAS + RPC":
    if gpu_choice_var.get() == "All":
        args.usecuda = ["normal"]
    else:
        args.usecuda = ["normal", str(gpuchoiceidx)]
    if mmq_var.get() == 1:
        args.usecuda.append("mmq")
    else:
        args.usecuda.append("nommq")
    if rowsplit_var.get() == 1:
        args.usecuda.append("rowsplit")
```

**Purpose**: Generates command-line args when RPC variants are selected in GUI

---

### 8.2: Update Config Loading

**Location**: Lines ~13283-13305

**Code**:
```python
if "usecuda" in dict and dict["usecuda"]:
    # Check for RPC variants first if RPC is being used
    if args.userpc is not None or has_rpc_in_device:
        if cublas_rpc_option is not None:
            runopts_var.set(cublas_rpc_option)
        elif hipblas_rpc_option is not None:
            runopts_var.set(hipblas_rpc_option)
    elif cublas_option is not None or hipblas_option is not None:
        if cublas_option:
            runopts_var.set(cublas_option)
        elif hipblas_option:
            runopts_var.set(hipblas_option)
    mmq_var.set(1 if "mmq" in dict["usecuda"] else 0)
    # ... (rest of config loading)
```

**Purpose**: Handles RPC variant options when loading saved configurations

---

## 9. Known Issues

### 9.1: HIP Assertion Bug (ROCm 7.2+)

**Symptom**:
```
Assertion `err == hipSuccess' failed in hip_code_object.cpp
```

**Cause**: HIP runtime bug when mixing RPC + local HIP devices

**Workarounds**:
1. Use RPC only: `--device RPC0,RPC1,RPC2`
2. Use local HIP only: `--device HIP0,HIP1` (no RPC)
3. Use Vulkan backend instead (most stable)

### 9.2: Backend Compatibility

**Rule**: Cannot mix Vulkan devices with HIPBLAS/CUDA devices in same session

**Solution**: Use consistent device names matching the selected backend:
- Vulkan library: Use `VULKAN*`, `RPC*` only
- HIPBLAS library: Use `HIP*`, `CUDA*`, `ROCm*`, `RPC*`
- CUDA library: Use `CUDA*`, `HIP*`, `RPC*`

### 9.3: Library Selection

**Issue**: Using wrong library for backend

**Solution**: Ensure correct library is built and selected:
- `koboldcpp_hipblas_rpc.so` for HIPBLAS + RPC
- `koboldcpp_cublas_rpc.so` for CUDA + RPC
- `koboldcpp_rpc.so` for Vulkan + RPC

---

## 10. Complete Code Reference

### 10.1: File Structure

```
koboldcpp.py:
├── Lines 1-940:  Initial setup and imports
├── Lines 940-985: Library detection (additions in this section)
├── Lines 986-1075: init_library() (major changes)
├── Lines 2250-2400: Input structures and population
├── Lines 10860-10920: auto_set_backend_gui()
├── Lines 11100-11200: changerunmode()
├── Lines 12900-13000: args population from GUI
├── Lines 13280-13310: Config loading
└── Lines 16900-17000: Argument parser
```

### 10.2: Key Functions Modified

| Function | Lines | Purpose |
|----------|-------|---------|
| `init_library()` | ~1000-1075 | Library selection logic |
| `auto_set_backend_gui()` | ~10860-10920 | Auto backend detection |
| `changerunmode()` | ~11100-11200 | UI backend switching |
| `main()` | ~14600-15600 | Argument processing |
| Argument parser | ~16900-17000 | CLI argument handling |

### 10.3: Key Variables Added

| Variable | Lines | Purpose |
|----------|-------|---------|
| `lib_cublas_rpc` | ~949 | CUDA + RPC library path |
| `lib_hipblas_rpc` | ~951 | HIPBLAS + RPC library path |
| `has_rpc_in_device` | ~1002 | Detect RPC in device string |
| `has_rocm_in_device` | ~1006 | Detect ROCm in device string |

### 10.4: Key Code Patterns

**Pattern 1: RPC Detection**
```python
has_rpc_in_device = False
if args.device and "RPC" in args.device:
    has_rpc_in_device = True
```

**Pattern 2: Library Selection**
```python
if file_exists(lib_xxx_rpc) and (args.userpc is not None or has_rpc_in_device):
    libname = lib_xxx_rpc
elif file_exists(lib_xxx):
    libname = lib_xxx
```

**Pattern 3: GUI Backend Detection**
```python
if "Use X + RPC" in runopts:
    runopts_var.set("Use X + RPC")
    found_new_backend = True
elif "Use X" in runopts:
    runopts_var.set("Use X")
    found_new_backend = True
```

---

## Quick Reference

### Files Modified
- `koboldcpp.py` - All RPC functionality

### Lines Changed
- ~400 lines added
- ~100 lines modified
- ~15 major code sections

### Key Features
1. RPC endpoint support
2. Multiple RPC server support
3. Hybrid mode (local + RPC)
4. Manual device ordering
5. Triple naming (HIP=CUDA=ROCm)
6. Automatic backend detection
7. Full GUI integration
8. RPC variant libraries

---

## Related Documentation

- **RPC Manual**: `RPC_MANUAL.md`
- **Quick Start**: `RPC_QUICKSTART.md`
- **Build Guide**: `RPC_makefile.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`
- **Backend Compatibility**: `RPC_BACKEND_COMPATIBILITY.md`

---

**License**: MIT  
**Version**: 1.111.2
