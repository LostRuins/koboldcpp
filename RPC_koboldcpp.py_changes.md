# RPC Implementation Changes in koboldcpp.py

**Version**: 1.111.2  
**Date**: 2026-04-18  
**Purpose**: Complete documentation of all koboldcpp.py changes for RPC support  
**Status**: ✅ Complete - All Features Working (v1.111.2 with RPC endpoint fields, config save/load, and independent CUDA+HIPBLAS builds)

---

## Overview

This document details every code addition and modification made to `koboldcpp.py` to implement RPC (Remote Procedure Call) functionality. The implementation enables:

✅ RPC client that connects to remote GPU servers  
✅ Multiple RPC server support  
✅ **RPC Server mode** (launch RPC server from GUI or CLI)  
✅ **RPC Server tab** in GUI with full configuration  
✅ **RPC Server Backend dropdown** (Auto-detect, Vulkan, hipBLAS, CUDA)  
✅ **Device name conversion** (VULKAN0 -> ROCm0, etc.)  
✅ Hybrid mode (local GPUs + RPC servers)  
✅ Manual tensor_split for layer distribution  
✅ Manual device ordering (--device argument)  
✅ Auto-offload for RPC configurations  
✅ **Triple naming support** (HIP0 = CUDA0 = ROCm0)  
✅ **Automatic backend detection** (make rpc-full-all)  
✅ **Full GUI dropdown** with all backend options  
✅ **RPC variant libraries** (koboldcpp_hipblas_rpc.so, etc.)  
✅ **Allow Launch Without Models** for RPC Server mode  
✅ **Skip library loading** when RPC Server mode enabled  
✅ **RPC endpoint field** in Quick Launch and Hardware tabs (shared input, auto-sync)  
✅ **Config save/load** for all RPC settings (.kcpps files) including backend selection (runopts)  
✅ **ROCm device registry check** fix (added "ROCM" to device enum)  
✅ **has_rpc_in_device initialization** in import_vars() - now also checks saved RPC endpoint  
✅ **runopts saved to config** - backend selection persists across reloads

**Total Changes**: 38 major modifications across 19 sections  
**Lines Added**: ~950 lines  
**Lines Modified**: ~280 lines

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
9. [RPC Server Mode (GUI Tab)](#9-rpc-server-mode-gui-tab)
10. [RPC Server Mode (Launcher)](#10-rpc-server-mode-launcher)
11. [RPC Endpoint Field](#11-rpc-endpoint-field)
12. [Config Save/Load](#12-config-saveload)
13. [runopts Save/Load](#13-runopts-saveload)
14. [Usage Examples](#14-usage-examples)
15. [Known Issues](#15-known-issues)
16. [Complete Code Reference](#16-complete-code-reference)

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

**Location**: Lines ~260-265

**Code**:
```python
@ctypes.Structure
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("model_param", ctypes.c_char_p),
        ("port", ctypes.c_int),
        ("host", ctypes.c_char_p),
        ("rpc_endpoints", ctypes.c_char_p),      # ← NEW: RPC endpoint field
        ("gpulayers", ctypes.c_int),
        ("contextsize", ctypes.c_int),
        ("tensor_split", ctypes.c_char_p),
        ("devices_override", ctypes.c_char_p),     # ← NEW: Device ordering override
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

**Location**: Lines ~1031-1075

**Code**:
```python
elif args.usecuda is not None:
    # Prefer RPC variant if RPC is being used
    if file_exists(lib_cublas_rpc) and (args.userpc is not None or has_rpc_in_device):
        libname = lib_cublas_rpc
    elif file_exists(lib_hipblas_rpc) and (args.userpc is not None or has_rpc_in_device):
        libname = lib_hipblas_rpc
    elif file_exists(lib_cublas):
        libname = lib_cublas
    elif file_exists(lib_hipblas):
        libname = lib_hipblas
    elif file_exists(lib_rpc) and (args.userpc is not None or has_rpc_in_device):
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

**Purpose**: Selects the correct library based on backend and RPC usage. When RPC is used, the selected backend variant from dropdown is preferred (HIPBLAS + RPC > CUDA + RPC > Vulkan + RPC).

---

## 4. Input Population

### 4.1: Set RPC Endpoint

**Location**: Lines ~1243-1245

**Code**:
```python
# From RPC endpoint field (GUI)
if rpc_selected and rpc_endpoint_var.get() != "":
    inputs.rpc_endpoints = s.encode("UTF-8")
else:
    inputs.rpc_endpoints = "".encode("UTF-8")
```

**Purpose**: Passes RPC server endpoints to C++ library from GUI field

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

### 5.1: Auto-Select Backend for RPC with RPC Server Mode

**Location**: Lines ~10867-10960

**Code**:
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
    
    # ... (existing auto-selection logic continues)
```

**Purpose**: Automatically selects appropriate backend including RPC variants when RPC Server mode is enabled from GUI

---

## 6. Argument Parser

### 6.1: RPC Argument

**Location**: Lines ~17354-17360

**Code**:
```python
compatgroup.add_argument("--userpc", "--rpc", 
    help="Use RPC for remote GPU acceleration. Specify one or more RPC endpoints (e.g. --rpc 192.168.1.101:50054).", 
    metavar=("[RPC endpoints]"), 
    nargs="+", 
)
```

**Purpose**: Adds RPC endpoint argument to command line parser

---

### 6.2: RPC Server Arguments

**Location**: Lines ~17361-17389

**Code**:
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

**Purpose**: Adds command-line arguments for RPC Server mode including backend selection

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

**Location**: Lines ~11101-11290

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
    
    # Show RPC endpoint field for RPC variants
    rpc_selected = index in ["Use Vulkan + RPC", "Use hipBLAS + RPC", "Use CUDA + RPC"]
    if rpc_selected:
        rpc_endpoint_label.grid(...)
        rpc_endpoint_entry.grid(...)
        rpc_endpoint_label_hw.grid(...)
        rpc_endpoint_entry_hw.grid(...)
    else:
        rpc_endpoint_label.grid_remove()
        rpc_endpoint_entry.grid_remove()
        rpc_endpoint_label_hw.grid_remove()
        rpc_endpoint_entry_hw.grid_remove()
```

**Purpose**: Updates UI to show appropriate controls for RPC variant backends and RPC endpoint fields

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

## 9. RPC Server Mode (GUI Tab)

### 9.1: Add New Tab Name

**Location**: Lines ~10196-10207 (tabnames list)

**Changes**: Added "RPC Server" tab between "Loaded Files" and "Network"

**Code**:
```python
tabnames = [
    "Quick Launch",
    "Hardware",
    "Context",
    "Loaded Files",
    "RPC Server",          # ← NEW: RPC Server tab
    "Network",
    "Horde Worker",
    "Image Gen",
    "Audio",
    "Admin",
    "Extra",
]
```

**Purpose**: Creates a new tab for RPC Server configuration in the GUI

---

### 9.2: Add RPC Server GUI Variables

**Location**: Lines ~10420-10430 (global variable declarations)

**Changes**: Added StringVar and IntVar objects for RPC Server tab controls

**Code**:
```python
# RPC Server variables
rpc_server_mode_var = ctk.IntVar(value=0)
rpc_server_backend_var = ctk.StringVar(value="Auto-detect")
rpc_host_var = ctk.StringVar(value="0.0.0.0")
rpc_port_var = ctk.StringVar(value="50053")
rpc_devices_var = ctk.StringVar(value="")
```

**Purpose**: Stores RPC Server configuration values from GUI

---

### 9.3: Create RPC Server Tab UI

**Location**: Lines ~12790-12925 (after music low VRAM checkbox, before kcpp_export_template)

**Changes**: Created complete RPC Server tab with configuration controls including backend dropdown

**Code**:
```python
rpc_tab = tabcontent["RPC Server"]

# Header
ctk.CTkLabel(
    rpc_tab,
    text="RPC Server Configuration",
    fg_color="transparent",
    text_color="#5DA5E5",
    font=("Helvetica", 14, "bold"),
).grid(row=0, column=0, columnspan=2, sticky="w", padx=0, pady=10)

# Start RPC Server Mode checkbox
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
makelabel(rpc_tab, "Vulkan: Use VULKAN0,VULKAN1...  |  hipBLAS: Use ROCm0,ROCm1...  |  CUDA: Use CUDA0,CUDA1...", 5, 0, padx=0)

# Listening IP Address
makelabel(rpc_tab, "Listening IP Address:", 6, 0, padx=0)
makelabel(rpc_tab, "IP address for RPC server to listen on. Use 0.0.0.0 for all interfaces, 127.0.0.1 for localhost only.", 7, 0, padx=0)
makelabelentry(rpc_tab, "", rpc_host_var, 8, 150, singleline=True)

# Listening Port
makelabel(rpc_tab, "Listening Port:", 9, 0, padx=0)
makelabel(rpc_tab, "Port number for RPC server connections (default: 50053).", 10, 0, padx=0)
makelabelentry(rpc_tab, "", rpc_port_var, 11, 100, singleline=True)

# RPC Devices
makelabel(rpc_tab, "RPC Devices:", 12, 0, padx=0)
makelabel(rpc_tab, "Comma-separated list of GPUs to expose via RPC. Leave empty to auto-detect all devices.", 13, 0, padx=0)
makelabelentry(rpc_tab, "", rpc_devices_var, 14, 200, singleline=True)

# Allow Launch Without Models
makecheckbox(
    rpc_tab,
    "Allow Launch Without Models",
    nomodel,
    16,
    0,
    tooltiptxt="Allows starting RPC server without loading a model file.",
)

# Warning message (14pt red bold)
ctk.CTkLabel(
    rpc_tab,
    text="WARNING: RPC Server mode replaces the WebUI API. Clients must connect via --rpc.",
    fg_color="transparent",
    text_color="red",
    font=("Helvetica", 14, "bold"),
).grid(row=18, column=0, columnspan=2, sticky="w", padx=0, pady=10)
```

**Purpose**: Provides complete GUI configuration interface for RPC Server mode with backend selection dropdown

---

### 9.4: Add RPC Server Args Population

**Location**: Lines ~13385-13390 (export_vars function)

**Changes**: Added RPC Server arguments to args population including backend selection

**Code**:
```python
args.adminunloadtimeout = (
    0
    if admin_unload_timeout_var.get() == ""
    else int(admin_unload_timeout_var.get())
)
args.showgui = False  # prevent showgui from leaking into configs, its cli only

# RPC Server arguments
args.start_rpc_server = rpc_server_mode_var.get() == 1
args.rpc_host = rpc_host_var.get()
args.rpc_port = int(rpc_port_var.get()) if rpc_port_var.get() else 50053
args.rpc_devices = rpc_devices_var.get()
args.rpc_server_backend = rpc_server_backend_var.get()
```

**Purpose**: Exports RPC Server GUI values to command-line args

---

## 10. RPC Server Mode (Launcher)

### 10.1: RPC Server Mode Launcher with Backend Selection

**Location**: Lines ~17037-17175 (after benchmark logic, before start_server)

**Changes**: Implemented RPC Server mode launcher with backend dropdown support and device name conversion

**Code**:
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
    elif not devices and device_prefix:
        print(f"WARNING: No devices specified. Device prefix is set to: {device_prefix}")
    else:
        print("WARNING: No devices specified. RPC server will use default device enumeration.")

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

**Purpose**: Launches the RPC server binary with proper device configuration and backend selection. Device name conversion automatically converts VULKAN0 -> ROCm0, etc. when backend differs from device prefix.

---

### 10.2: Skip Library Loading for RPC Server Mode

**Location**: Lines ~16302-16307 (before main server logic)

**Code**:
```python
# Skip library loading when RPC Server mode is enabled (subprocess-based)
if getattr(args, "start_rpc_server", False):
    print("RPC Server mode enabled, skipping KoboldCpp API library loading")
else:
    init_library()  # Note: if blas does not exist and is enabled, program will crash.
    print("==========")
    time.sleep(1)
    # ... (rest of normal startup)
```

**Purpose**: Prevents init_library() from loading GPU libraries when RPC Server mode is enabled, since RPC Server runs as subprocess not KoboldCpp API

---

## 11. RPC Endpoint Field

### 11.1: RPC Endpoint Variable Declaration

**Location**: Line ~10478 (global variable declarations)

**Code**:
```python
rpc_endpoint_var = ctk.StringVar(value="")
```

**Purpose**: Shared StringVar for RPC endpoint input across Quick Launch and Hardware tabs

---

### 11.2: RPC Endpoint Field in Quick Launch Tab

**Location**: Lines ~11329-11342

**Code**:
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

**Purpose**: Creates RPC endpoint input field in Quick Launch tab

---

### 11.3: RPC Endpoint Field in Hardware Tab (Shared)

**Location**: Lines ~11472-11485

**Code**:
```python
rpc_endpoint_label_hw = ctk.CTkLabel(
    hardware_tab,
    text="RPC Server Endpoint:",
    font=("Helvetica", 10, "bold"),
)
rpc_endpoint_entry_hw = ctk.CTkEntry(
    hardware_tab,
    width=300,
    textvariable=rpc_endpoint_var,  # ← Same variable as Quick Launch
)
rpc_endpoint_label_hw.grid(row=2, column=0, padx=160, pady=1, stick="nw")
rpc_endpoint_entry_hw.grid(row=2, column=0, padx=250, pady=1, stick="nw")
```

**Purpose**: Creates second RPC endpoint input field in Hardware tab using same StringVar (auto-syncs with Quick Launch)

---

### 11.4: Show/Hide RPC Endpoint Field Based on Backend

**Location**: Lines ~11270-11290 (in changerunmode function)

**Code**:
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

**Purpose**: Auto-shows/hides RPC endpoint fields when RPC variant backend is selected/deselected

---

### 11.5: Use RPC Endpoint in Args Population

**Location**: Lines ~13283-13284

**Code**:
```python
if rpc_selected and rpc_endpoint_var.get() != "":
    args.userpc = [rpc_endpoint_var.get()]
```

**Purpose**: Populates args.userpc from RPC endpoint field when RPC variant is selected

---

## 12. Config Save/Load

### 12.1: Initialize RPC Variables in import_vars()

**Location**: Lines ~13526-13530 (in import_vars function)

**Code**:
```python
def import_vars():
    global args, ...
    global has_rpc_in_device, has_rocm_in_device  # ← NEW: Initialize RPC variables
    
    has_rpc_in_device = False
    if args.device and "RPC" in args.device:
        has_rpc_in_device = True
    
    has_rocm_in_device = False
    if args.device and ("ROCm" in args.device or "HIP" in args.device):
        has_rocm_in_device = True
    
    # ... (rest of function)
```

**Purpose**: Initializes `has_rpc_in_device` and `has_rocm_in_device` variables in `import_vars()` function to prevent NameError when loading config files

---

### 12.2: Load RPC Endpoint from Config

**Location**: Lines ~13669-13700

**Code**:
```python
# Load RPC endpoint from userpc field
if "userpc" in dict:
    if isinstance(dict["userpc"], list):
        rpc_endpoint_var.set(dict["userpc"][0])
    else:
        rpc_endpoint_var.set(str(dict["userpc"]))

# Load RPC endpoint from rpc_endpoint field
if "rpc_endpoint" in dict:
    rpc_endpoint_var.set(str(dict["rpc_endpoint"]))
```

**Purpose**: Loads RPC endpoint value from saved config into GUI field

---

### 12.3: Save RPC Endpoint to Config

**Location**: Lines ~14907

**Code**:
```python
savdict["rpc_endpoint"] = rpc_endpoint_var.get()
```

**Purpose**: Saves RPC endpoint value to .kcpps config file

---

### 12.4: RPC Server Config Save/Load

**Location**: In convert_args_to_template() and import_vars()

**Saved Settings**:
- `rpc_server_mode` - Whether RPC Server mode was enabled
- `rpc_server_backend` - Selected RPC Server backend
- `rpc_host` - Listening IP address
- `rpc_port` - Listening port
- `rpc_devices` - RPC devices list
- `rpc_endpoint` - RPC server endpoint for client mode

**Loading**: When loading a `.kcpps` config file, all RPC settings are automatically applied to the GUI fields.

---

### 12.5: runopts Save/Load (Backend Selection Persists)

**Location**: In `convert_args_to_template()` function (line ~14903)

**Purpose**: Saves the selected backend (runopts) to config file so it persists across reloads.

**Code**:
```python
savdict["runopts"] = runopts_var.get()
```

---

**Location**: In `import_vars()` function (lines ~13548-13550)

**Purpose**: Loads the saved backend selection BEFORE any backend selection logic runs, ensuring the correct backend is used.

**Code**:
```python
# Load saved backend selection (runopts) FIRST, before any backend logic
if "runopts" in dict and dict["runopts"]:
    runopts_var.set(dict["runopts"])
```

---

**Location**: In `import_vars()` function (lines ~13536-13539)

**Purpose**: Detects RPC in both device string AND saved RPC endpoint from config file. Prevents NameError and ensures correct library selection.

**Code**:
```python
# Detect RPC in device string and RPC endpoint (same as init_library)
has_rpc_in_device = False
device_override_str = str(dict.get("device", ""))
if device_override_str and "RPC" in device_override_str:
    has_rpc_in_device = True
# Also check if RPC endpoint is specified in config
rpc_ep = dict.get("rpc_endpoint", "") or dict.get("userpc", "")
if rpc_ep:
    has_rpc_in_device = True
```

---

**Loading**: When loading a `.kcpps` config file:
1. `runopts` is loaded FIRST to set the backend selection
2. `rpc_endpoint` is loaded to populate the RPC endpoint field
3. `has_rpc_in_device` is set based on both device string AND saved RPC endpoint
4. All RPC Server settings are applied to GUI fields

---

## 13. Usage Examples

### GUI RPC Server Mode
1. Open `python koboldcpp.py`
2. Go to **RPC Server** tab
3. Check "Start RPC Server Mode"
4. Set **RPC Server Backend**: "Auto-detect", "Vulkan", "hipBLAS (ROCm)", or "CUDA"
5. Configure:
   - Listening IP Address: `0.0.0.0` (all) or `127.0.0.1` (localhost)
   - Listening Port: `50053` (default)
   - RPC Devices: `VULKAN0,VULKAN1` or `ROCm0,ROCm1` (or empty for auto-detect)
6. Check "Allow Launch Without Models" if no model file
7. Start - launches RPC server

### Command Line RPC Server Mode
```bash
python koboldcpp.py \
    --start-rpc-server \
    --rpc-host 0.0.0.0 \
    --rpc-port 50053 \
    --rpc-devices VULKAN0,VULKAN1 \
    --rpc-server-backend Vulkan
```

### RPC Client Connection
```bash
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.100:50053 \
    --device VULKAN0,RPC0,VULKAN1 \
    --gpulayers 999 \
    --port 5001
```

---

## 15. Known Issues

### 1. HIP Assertion Bug (ROCm 7.2+)

**Symptom**:
```
Assertion `err == hipSuccess' failed in hip_code_object.cpp
```

**Cause**: HIP runtime bug when mixing RPC + local HIP devices

**Workarounds**:
1. Use RPC only: `--device RPC0,RPC1,RPC2`
2. Use local HIP only: `--device HIP0,HIP1` (no RPC)
3. Use Vulkan backend instead (most stable)

### 2. Backend Compatibility

**Rule**: Cannot mix Vulkan devices with HIPBLAS/CUDA devices in same session

**Solution**: Use consistent device names matching the selected backend:
- Vulkan library: Use `VULKAN*`, `RPC*` only
- HIPBLAS library: Use `HIP*`, `CUDA*`, `ROCm*`, `RPC*`
- CUDA library: Use `CUDA*`, `HIP*`, `RPC*`

### 3. Library Selection

**Issue**: Using wrong library for backend

**Solution**: Ensure correct library is built and selected:
- `koboldcpp_hipblas_rpc.so` for HIPBLAS + RPC
- `koboldcpp_cublas_rpc.so` for CUDA + RPC
- `koboldcpp_rpc.so` for Vulkan + RPC

### 4. NameError: has_rpc_in_device is not defined (FIXED)

This occurred when loading config files. Now fixed - `import_vars` initializes `has_rpc_in_device` and `has_rocm_in_device` variables.

**Code fix**: Added to `import_vars()` function:
```python
global has_rpc_in_device, has_rocm_in_device
has_rpc_in_device = False
if args.device and "RPC" in args.device:
    has_rpc_in_device = True
# Also check saved RPC endpoint in config
rpc_ep = dict.get("rpc_endpoint", "") or dict.get("userpc", "")
if rpc_ep:
    has_rpc_in_device = True
has_rocm_in_device = False
if args.device and ("ROCm" in args.device or "HIP" in args.device):
    has_rocm_in_device = True
```

**Rebuild**:
```bash
make clean
make rpc-full-all -j8
```

### 5. RPC Endpoint Not Loading from Save File (FIXED)

**Symptom**: When loading a `.kcpps` config file, the RPC endpoint field was empty and backend defaulted incorrectly.

**Cause**: `has_rpc_in_device` was only checked from device string, not from saved config. Also `runopts` (backend selection) was not being saved or loaded.

**Code fixes**:
1. Added RPC endpoint detection from config in `import_vars()`:
```python
rpc_ep = dict.get("rpc_endpoint", "") or dict.get("userpc", "")
if rpc_ep:
    has_rpc_in_device = True
```

2. Added `runopts` save in `convert_args_to_template()`:
```python
savdict["runopts"] = runopts_var.get()
```

3. Added `runopts` load at START of `import_vars()`:
```python
# Load saved backend selection (runopts) FIRST, before any backend logic
if "runopts" in dict and dict["runopts"]:
    runopts_var.set(dict["runopts"])
```

4. Removed duplicate RPC Server argument import block.

### 5. "undefined symbol: ggml_backend_rpc_add_server" in koboldcpp_vulkan.so (FIXED)

**Symptom**: When running Vulkan backend (non-RPC), the library fails with:
```
OSError: koboldcpp_vulkan.so: undefined symbol: ggml_backend_rpc_add_server
```

**Root Cause**: `gpttype_adapter.cpp` lines 2391-2607 contained RPC code that was NOT guarded by `#ifdef GGML_USE_RPC`. The code called `ggml_backend_rpc_add_server()` unconditionally, so the symbol appeared in ALL builds, not just RPC builds.

**PRIMARY FIX**: Wrapped entire RPC block in `gpttype_adapter.cpp` with `#ifdef GGML_USE_RPC / #endif`:
```cpp
#ifdef GGML_USE_RPC
        std::string rpc_endpoints_str = inputs.rpc_endpoints;
        bool use_rpc = false;
        std::vector<ggml_backend_dev_t> rpc_devices;
        // ... (all RPC code)
#endif
```

**SECONDARY FIX**: Build order in `rpc-full-all` - RPC builds run first, then objects cleaned, then non-RPC builds run last.

**Workaround (if already broken)**:
```bash
make clean
make LLAMA_VULKAN=1 koboldcpp_vulkan -j8
```
OSError: koboldcpp_vulkan.so: undefined symbol: ggml_backend_rpc_add_server
```

**Cause**: The `rpc-full-all` Makefile target builds non-RPC and RPC libraries sharing the same `.o` files. When object files are compiled without RPC first, then reused in RPC builds with RPC symbols, the non-RPC `.so` files get corrupted references.

**Fix**: Added object file cleanup step in `Makefile` `rpc-full-all` target between non-RPC and RPC builds:
```makefile
@echo "=== Cleaning object files before RPC builds (prevents symbol mismatch) ==="
@rm -vf ggml_v4_vulkan.o gpttype_adapter.o ggml-backend_vulkan.o ...
```

**Workaround (if already broken)**:
```bash
make clean
make LLAMA_VULKAN=1 koboldcpp_vulkan -j8
```

---

## Quick Reference

### Files Modified
- `koboldcpp.py` - All RPC functionality

### Lines Changed
- ~950 lines added
- ~280 lines modified
- ~38 major code sections

### Key Features
1. RPC endpoint support
2. Multiple RPC server support
3. **RPC Server mode** (GUI + CLI)
4. **New RPC Server tab** in GUI
5. **RPC Server Backend dropdown** (Auto-detect, Vulkan, hipBLAS, CUDA)
6. **Device name conversion** (VULKAN0 -> ROCm0, etc.)
7. Hybrid mode (local + RPC)
8. Manual device ordering
9. Triple naming (HIP=CUDA=ROCm)
10. Automatic backend detection
11. Full GUI integration
12. RPC variant libraries
13. Allow Launch Without Models
14. Subprocess-based RPC server launch
15. Skip library loading for RPC Server mode
16. **RPC endpoint field** shared between Quick Launch and Hardware tabs (auto-sync)
17. **Config save/load** for all RPC settings (.kcpps files) including backend selection (runopts)
18. **ROCm device registry fix** - added "ROCM" to device enum check
19. **has_rpc_in_device initialization** in import_vars() - now also checks saved RPC endpoint
20. **runopts saved to config** - backend selection persists across reloads

### C++ Fix: gpttype_adapter.cpp RPC Guards

**File**: `gpttype_adapter.cpp`  
**Lines**: 2391-2607

**Problem**: All RPC code in `gpttype_adapter.cpp` was unguarded, causing `ggml_backend_rpc_add_server` symbol to appear in non-RPC builds.

**Fix**: Wrapped entire RPC block in `#ifdef GGML_USE_RPC / #endif`:
```cpp
#ifdef GGML_USE_RPC
        // Handle RPC endpoints - connect to RPC server(s) FIRST
        std::string rpc_endpoints_str = inputs.rpc_endpoints;
        bool use_rpc = false;
        std::vector<ggml_backend_dev_t> rpc_devices;
        // ... (all RPC connection, device enumeration, and reordering code)
#endif
```

This ensures RPC code is completely excluded from non-RPC library builds.

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
