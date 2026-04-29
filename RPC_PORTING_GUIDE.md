# RPC Porting Guide for koboldcpp

**Version:** 1.0  
**Last Updated:** 2026-04-29  
**Purpose:** Complete guide for adding RPC support to koboldcpp (for humans and LLMs)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Makefile Changes](#step-1-makefile-changes)
4. [Step 2: expose.h Changes](#step-2-exposeh-changes)
5. [Step 3: gpttype_adapter.cpp Changes](#step-3-gpttype_adaptercpp-changes)
6. [Step 4: koboldcpp.py Changes](#step-4-koboldcpppy-changes)
7. [Step 5: Build RPC Library](#step-5-build-rpc-library)
8. [Step 6: Testing](#step-6-testing)
9. [Troubleshooting](#troubleshooting)
10. [Reference Files](#reference-files)

---

## Overview

This guide explains how to add RPC (Remote Procedure Call) support to koboldcpp. RPC allows distributing model inference across multiple machines and GPUs.

**What RPC enables:**
- Connect to remote GPU servers
- Distribute model layers across multiple devices
- Use GPUs on different machines as if they were local

**Key components:**
- `koboldcpp_rpc.so` - RPC-enabled library
- `rpc-server-*` - RPC server executables
- RPC backend code in `ggml/src/ggml-rpc/`

---

## Prerequisites

Before starting, ensure you have:

1. **Source code with RPC support** from llama.cpp/ggml
   - `ggml/src/ggml-rpc/ggml-rpc.cpp`
   - `ggml/src/ggml-rpc/transport.cpp`
   - `ggml/src/ggml-rpc/transport.h`
   - `ggml/include/ggml-rpc.h`
   - `tools/rpc-server.cpp`

2. **Working build environment**
   - GCC/G++ 11+ (tested with GCC 15.2.1)
   - CMake (optional, for RPC server)
   - Vulkan/CUDA/HIP libraries (for GPU backends)

3. **Base koboldcpp installation**
   - Working koboldcpp without RPC
   - All standard dependencies installed

---

## Step 1: Makefile Changes

### Location: `koboldcpp-1.112.2/Makefile`

### 1.1 Add RPC Object Files

**Find this section** (around line 115-120):
```makefile
ifdef LLAMA_VULKAN
OBJS_FULL += ggml-vulkan.o ggml-vulkan-shaders.o
endif
```

**Add after it:**
```makefile
ifdef LLAMA_RPC
OBJS_FULL += ggml-rpc.o transport.o
endif
```

**Why:** This tells the build system to compile RPC object files when `LLAMA_RPC=1` is set.

---

### 1.2 Add RPC Compilation Rules

**Find this section** (around line 460-470):
```makefile
ifdef LLAMA_VULKAN
ggml-vulkan.o: ggml/src/ggml-vulkan/ggml-vulkan.cpp
	$(CXX) $(CXXFLAGS) -DGGML_USE_VULKAN -c $< -o $@
endif
```

**Add after it:**
```makefile
ifdef LLAMA_RPC
ggml-rpc.o: ggml/src/ggml-rpc/ggml-rpc.cpp ggml/include/ggml-rpc.h ggml/src/ggml-rpc/transport.h
	$(CXX) $(CXXFLAGS) -DGGML_USE_RPC -c $< -o $@

transport.o: ggml/src/ggml-rpc/transport.cpp ggml/src/ggml-rpc/transport.h
	$(CXX) $(CXXFLAGS) -DGGML_USE_RPC -c $< -o $@
endif
```

**Why:** Defines how to compile RPC source files with the correct flags.

---

### 1.3 Add RPC Library Target

**Find this section** (around line 935-950):
```makefile
koboldcpp_vulkan: ggml_v4_vulkan.o ... ggml-vulkan.o ...
	$(CXX) ... -lvulkan -shared -o koboldcpp_vulkan.so -ldl
```

**Add new target after it:**
```makefile
koboldcpp_rpc: ggml_v4_vulkan.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter_vulkan.o ggml-rpc.o transport.o ggml-vulkan.o ggml-vulkan-shaders.o sdcpp_vulkan.o whispercpp_vulkan.o tts_default.o music_default.o embeddings_default.o llavaclip_vulkan.o llava.o ggml-backend.o ggml-backend-meta.o ggml-backend-reg_vulkan.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o mtmdaudio.o -lvulkan -shared -o koboldcpp_rpc.so -ldl
```

**Why:** Creates the RPC-enabled shared library.

---

### 1.4 Add RPC Server Targets

**Add after library targets** (around line 955-965):
```makefile
# RPC Server targets
rpc-server-vulkan: tools/rpc-server.cpp ggml/src/ggml-vulkan-shaders.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o ggml-rpc.o transport.o ggml-backend.o ggml-backend-meta.o ggml-backend-reg_vulkan.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o console.o
	$(CXX) $(CXXFLAGS) -DGGML_USE_VULKAN -DGGML_USE_RPC $^ -lvulkan -o $@

rpc-server-cuda: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o ggml-rpc.o transport.o ggml-backend.o ggml-backend-meta.o ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o ggml-cuda.o $(CUBLAS_OBJS)
	$(CXX) $(CXXFLAGS) -DGGML_USE_CUDA -DGGML_USE_RPC $^ -lcuda -lcudart -o $@

rpc-server-hip: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o ggml-rpc.o transport.o ggml-backend.o ggml-backend-meta.o ggml-backend-reg_cublas.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o $(HIP_OBJS)
	$(CXX) $(CXXFLAGS) -DGGML_USE_HIPBLAS -DGGML_USE_RPC $^ -lamdhip64 -o $@
```

**Why:** Creates RPC server executables for different GPU backends.

---

### 1.5 Add Convenience Targets

**Add at the end** (around line 965-990):
```makefile
.PHONY: rpc-full-all
rpc-full-all:
	@echo "Building all RPC variants..."
	$(MAKE) LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8
	$(MAKE) LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8
ifeq ($(shell which hipcc 2>/dev/null),)
	@echo "HIP not available, skipping HIP builds"
else
	$(MAKE) LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
	$(MAKE) LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
endif
ifeq ($(shell which nvcc 2>/dev/null),)
	@echo "CUDA not available, skipping CUDA builds"
else
	$(MAKE) LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8
	$(MAKE) LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
endif
	@echo "RPC build complete!"
```

**Why:** One command to build everything RPC-related.

---

## Step 2: expose.h Changes

### Location: `koboldcpp-1.112.2/expose.h`

### 2.1 Add RPC Endpoints Field

**Find the `load_model_inputs` struct** (around line 50-80):
```cpp
struct load_model_inputs {
    const char * model_filename = nullptr;
    const char * executable_path = nullptr;
    const int kcpp_main_gpu = -1;
    const char * vulkan_info = nullptr;
    // ADD HERE:
    const char * rpc_endpoints = nullptr;
    const int batchsize = 512;
    // ... rest of struct
};
```

**Add this line:**
```cpp
const char * rpc_endpoints = nullptr;
```

**Why:** Stores RPC server endpoint strings (e.g., "192.168.1.101:50053,192.168.1.101:50054")

---

### 2.2 Add to Other Structs

**Repeat for these structs** (if they exist in your version):
- `load_model_inputs_sd` (around line 180)
- `load_model_inputs_tts` (around line 265)
- `load_model_inputs_music` (around line 290)
- `load_model_inputs_whisper` (around line 320)
- `load_model_inputs_embeddings` (around line 350)

**Add to each:**
```cpp
const char * rpc_endpoints = nullptr;
```

**Why:** Ensures RPC support across all model types.

---

## Step 3: gpttype_adapter.cpp Changes

### Location: `koboldcpp-1.112.2/gpttype_adapter.cpp`

### 3.1 Add Required Includes

**Find the includes section** (top of file, around line 1-30):
```cpp
#include <stdio.h>
#include <string.h>
#include <vector>
// ADD HERE:
#include <set>
#include "ggml-rpc.h"
```

**Add these lines:**
```cpp
#include <set>
#include "ggml-rpc.h"
```

**Why:** `<set>` for device name tracking, `"ggml-rpc.h"` for RPC API.

---

### 3.2 Add RPC Initialization Code

**Find the model loading function** `gpttype_load_model` (around line 2400-2450):

**Find this code:**
```cpp
model_params.use_mlock = inputs.use_mlock;
model_params.n_gpu_layers = inputs.gpulayers;

//set device overrides if needed
std::vector<ggml_backend_dev_t> devices_override;
```

**Add BEFORE it:**
```cpp
// Load all backends including RPC
ggml_backend_load_all();
printf("[RPC] Backends loaded, device count: %zu\n", ggml_backend_dev_count());

// Handle RPC endpoints - connect to RPC server(s) FIRST
std::string rpc_endpoints_str = inputs.rpc_endpoints;
bool use_rpc = false;
std::vector<ggml_backend_dev_t> rpc_devices;
if(rpc_endpoints_str != "" && rpc_endpoints_str.length() > 0)
{
    printf("[RPC] Connecting to RPC server(s): %s\n", rpc_endpoints_str.c_str());
    
    // Parse comma-separated endpoints and add each one
    size_t start = 0;
    size_t end = rpc_endpoints_str.find(',');
    while (end != std::string::npos) {
        std::string endpoint = rpc_endpoints_str.substr(start, end - start);
        printf("[RPC] Adding RPC server: %s\n", endpoint.c_str());
        ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint.c_str());
        if(reg != nullptr) {
            size_t dev_count = ggml_backend_reg_dev_count(reg);
            printf("[RPC] Server %s has %zu devices\n", endpoint.c_str(), dev_count);
            for(size_t i = 0; i < dev_count; ++i) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
                printf("[RPC] Found RPC device %zu: %s\n", i, ggml_backend_dev_name(dev));
                rpc_devices.push_back(dev);
            }
        } else {
            printf("[RPC] WARNING: Failed to connect to RPC server %s\n", endpoint.c_str());
        }
        start = end + 1;
        end = rpc_endpoints_str.find(',', start);
    }
    // Add last endpoint
    std::string endpoint = rpc_endpoints_str.substr(start);
    printf("[RPC] Adding RPC server: %s\n", endpoint.c_str());
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint.c_str());
    if(reg != nullptr) {
        size_t dev_count = ggml_backend_reg_dev_count(reg);
        printf("[RPC] Server %s has %zu devices\n", endpoint.c_str(), dev_count);
        for(size_t i = 0; i < dev_count; ++i) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
            printf("[RPC] Found RPC device %zu: %s\n", i, ggml_backend_dev_name(dev));
            rpc_devices.push_back(dev);
        }
    } else {
        printf("[RPC] WARNING: Failed to connect to RPC server %s\n", endpoint.c_str());
    }
    
    if(rpc_devices.size() > 0) {
        printf("[RPC] Using %zu RPC device(s) for offloading\n", rpc_devices.size());
        devices_override.insert(devices_override.begin(), rpc_devices.begin(), rpc_devices.end());
        use_rpc = true;
    } else {
        printf("[RPC] WARNING: No RPC devices found after connecting to server\n");
    }
}
```

**Why:** This code:
1. Loads all backends including RPC
2. Parses comma-separated RPC endpoint strings
3. Connects to each RPC server
4. Collects RPC devices for model offloading

---

### 3.3 Add Device Reordering Code

**Find the device override section** (after RPC initialization, around line 2500-2600):

**Add this code block:**
```cpp
std::string dev_override_str = inputs.devices_override ? inputs.devices_override : "";

// If device override is specified, use it to reorder ALL devices
if(dev_override_str != "" && dev_override_str.length() > 0)
{
    printf("[RPC] Manual device ordering specified: %s\n", dev_override_str.c_str());
    
    // Get all available devices (RPC + local)
    std::vector<std::pair<std::string, ggml_backend_dev_t>> all_devices;
    
    // Add RPC devices FIRST
    if(use_rpc) {
        for(size_t i = 0; i < rpc_devices.size(); ++i) {
            std::string name = "RPC" + std::to_string(i);
            all_devices.push_back(std::make_pair(name, rpc_devices[i]));
        }
        printf("[RPC] Added %zu RPC device(s) to device pool\n", rpc_devices.size());
    }
    
    // Add local GPU devices
    printf("[RPC] Enumerating local GPU devices...\n");
    int vulkan_count = 0, cuda_count = 0, hip_count = 0, metal_count = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto* dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        std::string reg_name = reg ? ggml_backend_reg_name(reg) : "";
        std::string dev_name = ggml_backend_dev_name(dev);
        
        // Skip RPC and CPU devices
        std::string reg_name_upper = reg_name;
        std::transform(reg_name_upper.begin(), reg_name_upper.end(), reg_name_upper.begin(), ::toupper);
        
        if(reg_name_upper.find("RPC") != std::string::npos || 
           reg_name_upper.find("CPU") != std::string::npos) {
            continue;
        }
        
        // Add local GPU devices
        if(reg_name_upper.find("VULKAN") != std::string::npos || 
           reg_name_upper.find("RADV") != std::string::npos) {
            printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
            std::string local_name = "VULKAN" + std::to_string(vulkan_count++);
            all_devices.push_back(std::make_pair(local_name, dev));
        }
        else if(reg_name_upper.find("CUDA") != std::string::npos) {
            printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
            std::string local_name = "CUDA" + std::to_string(cuda_count++);
            all_devices.push_back(std::make_pair(local_name, dev));
        }
        else if(reg_name_upper.find("HIP") != std::string::npos || 
                reg_name_upper.find("ROCM") != std::string::npos) {
            printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
            std::string local_name = "HIP" + std::to_string(hip_count++);
            all_devices.push_back(std::make_pair(local_name, dev));
        }
        else if(reg_name_upper.find("METAL") != std::string::npos) {
            printf("[RPC] Found local GPU device: %s (registry: %s)\n", dev_name.c_str(), reg_name.c_str());
            std::string local_name = "METAL" + std::to_string(metal_count++);
            all_devices.push_back(std::make_pair(local_name, dev));
        }
    }
    
    // Parse device order from override string
    std::vector<std::string> device_order;
    size_t start = 0;
    size_t end = dev_override_str.find(',');
    while (end != std::string::npos) {
        std::string device = dev_override_str.substr(start, end - start);
        device.erase(0, device.find_first_not_of(" \t"));
        device.erase(device.find_last_not_of(" \t") + 1);
        device_order.push_back(device);
        start = end + 1;
        end = dev_override_str.find(',', start);
    }
    std::string last_device = dev_override_str.substr(start);
    last_device.erase(0, last_device.find_first_not_of(" \t"));
    last_device.erase(last_device.find_last_not_of(" \t") + 1);
    device_order.push_back(last_device);
    
    // If RPC devices are connected but NOT in the override string, append them
    if(use_rpc) {
        std::set<std::string> override_names;
        for(const auto& d : device_order) {
            std::string upper = d;
            std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
            override_names.insert(upper);
        }
        bool has_any_rpc = false;
        for(size_t i = 0; i < rpc_devices.size(); ++i) {
            std::string rpc_name = "RPC" + std::to_string(i);
            std::string rpc_upper = rpc_name;
            std::transform(rpc_upper.begin(), rpc_upper.end(), rpc_upper.begin(), ::toupper);
            if(override_names.find(rpc_upper) == override_names.end()) {
                device_order.push_back(rpc_name);
                printf("[RPC] Appending missing RPC device: %s\n", rpc_name.c_str());
                has_any_rpc = true;
            }
        }
        if(has_any_rpc) {
            printf("[RPC] Override string modified to include RPC devices\n");
        }
    }
    
    // Build ordered device list
    devices_override.clear();
    printf("[RPC] Ordering devices: ");
    for(const auto& dev_name : device_order) {
        std::string dev_name_upper = dev_name;
        std::transform(dev_name_upper.begin(), dev_name_upper.end(), dev_name_upper.begin(), ::toupper);
        printf("%s ", dev_name.c_str());
        
        bool found = false;
        for(const auto& pair : all_devices) {
            if(pair.first == dev_name_upper) {
                devices_override.push_back(pair.second);
                found = true;
                break;
            }
        }
        if(!found) {
            printf("\n[RPC] ERROR: Device %s not found\n", dev_name.c_str());
            printf("[RPC] ERROR: This usually means the RPC server is not running or failed to connect.\n");
            printf("[RPC] Aborting due to missing device: %s\n", dev_name.c_str());
            return ModelLoadResult::FAIL;
        }
    }
    printf("\n");
    if(devices_override.size() > 0) {
        printf("[RPC] Total devices for offloading: %zu (manually ordered)\n", devices_override.size());
    } else {
        printf("[RPC] ERROR: No valid devices found after parsing device override string.\n");
        return ModelLoadResult::FAIL;
    }
}
else if(use_rpc && devices_override.size() > 0)
{
    printf("[RPC] Enumerating local GPU devices to use alongside RPC...\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto* dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        std::string reg_name = reg ? ggml_backend_reg_name(reg) : "";
        std::string dev_name = ggml_backend_dev_name(dev);
        printf("[RPC] Checking device %zu: %s (registry: %s)\n", i, dev_name.c_str(), reg_name.c_str());
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
    }
    printf("[RPC] Total devices for offloading: %zu (RPC + local GPUs)\n", devices_override.size());
}

if(devices_override.size() > 0) {
    devices_override.push_back(nullptr);
    model_params.devices = devices_override.data();
    if(use_rpc) {
        model_params.n_gpu_layers = 999;
    }
    printf("[RPC] Using %zu device(s) for model offloading\n", devices_override.size() - 1);
} else {
    printf("[RPC] ERROR: No devices available for model offloading.\n");
    printf("[RPC] Please check that RPC servers are running or local GPUs are available.\n");
    return ModelLoadResult::FAIL;
}
```

**Why:** This code:
1. Handles manual device ordering
2. Combines RPC and local GPU devices
3. Ensures RPC devices are included in device list
4. Validates all requested devices exist

---

## Step 4: koboldcpp.py Changes

### Location: `koboldcpp-1.112.2/koboldcpp.py`

### 4.1 Add RPC Field to ctypes Structure

**Find the `load_model_inputs` definition** (around line 230-250):
```python
class load_model_inputs(ctypes.Structure):
    _fields_ = [
        ("model_filename", ctypes.c_char_p),
        ("executable_path", ctypes.c_char_p),
        ("kcpp_main_gpu", ctypes.c_int),
        ("vulkan_info", ctypes.c_char_p),
        # ADD HERE:
        ("rpc_endpoints", ctypes.c_char_p),
        ("batchsize", ctypes.c_int),
        # ... rest of fields
    ]
```

**Add this line:**
```python
("rpc_endpoints", ctypes.c_char_p),
```

**Why:** Allows Python to pass RPC endpoint strings to C++ code.

---

### 4.2 Add RPC Library Selection

**Find the library selection code** (around line 880-1000):
```python
lib_cublas = pick_existant_file("koboldcpp_cublas.dll","koboldcpp_cublas.so")
lib_vulkan = pick_existant_file("koboldcpp_vulkan.dll","koboldcpp_vulkan.so")
```

**Add after it:**
```python
lib_cublas_rpc = pick_existant_file("koboldcpp_cublas_rpc.dll","koboldcpp_cublas_rpc.so")
lib_hipblas_rpc = pick_existant_file("koboldcpp_hipblas_rpc.dll","koboldcpp_hipblas_rpc.so")
lib_rpc = pick_existant_file("koboldcpp_rpc.dll", "koboldcpp_rpc.so")
```

**Why:** Makes RPC libraries available for selection.

---

### 4.3 Add RPC Command Line Argument

**Find the argument parser section** (around line 17470-17500):
```python
parser.add_argument(
    "--tensor-split",
    help="Split model layers across devices",
    metavar=("[Ratios]"),
    type=float,
    nargs="+",
)
# ADD AFTER:
parser.add_argument(
    "--userpc",
    "--rpc",
    help="Use RPC for remote GPU acceleration. Specify one or more RPC endpoints (e.g. --rpc 192.168.1.101:50054).",
    metavar=("[RPC endpoints]"),
    nargs="+",
)
```

**Add this block:**
```python
parser.add_argument(
    "--userpc",
    "--rpc",
    help="Use RPC for remote GPU acceleration. Specify one or more RPC endpoints (e.g. --rpc 192.168.1.101:50054).",
    metavar=("[RPC endpoints]"),
    nargs="+",
)
```

**Why:** Allows users to specify RPC servers via command line.

---

### 4.4 Set RPC Endpoints in Model Inputs

**Find where model inputs are set** (around line 1185-1195):
```python
if args.vulkan_info:
    s = args.vulkan_info
    inputs.vulkan_info = s.encode("UTF-8")
else:
    inputs.vulkan_info = "".encode("UTF-8")
# ADD AFTER:
if args.userpc: # RPC endpoints specified
    s = ",".join(args.userpc)
    inputs.rpc_endpoints = s.encode("UTF-8")
else:
    inputs.rpc_endpoints = "".encode("UTF-8")
```

**Add this code:**
```python
if args.userpc: # RPC endpoints specified
    s = ",".join(args.userpc)
    inputs.rpc_endpoints = s.encode("UTF-8")
else:
    inputs.rpc_endpoints = "".encode("UTF-8")
```

**Why:** Converts command-line RPC arguments to comma-separated string for C++.

---

### 4.5 Add RPC Library Loading Logic

**Find the library loading section** (around line 940-1050):

**Add this code block:**
```python
# Check if RPC is needed
need_rpc = args.userpc is not None or has_rpc_in_device

if need_rpc:
    # Check which RPC variant was selected from dropdown
    if selected_backend:
        if "Vulkan + RPC" in selected_backend:
            if file_exists(lib_rpc):
                lib = lib_rpc
            else:
                print("WARNING: Vulkan + RPC library not found")
        elif "hipBLAS + RPC" in selected_backend:
            if file_exists(lib_hipblas_rpc):
                lib = lib_hipblas_rpc
            else:
                print("WARNING: hipBLAS + RPC library not found")
        elif "CUDA + RPC" in selected_backend:
            if file_exists(lib_cublas_rpc):
                lib = lib_cublas_rpc
            else:
                print("WARNING: CUDA + RPC library not found")
    elif args.userpc is not None:
        # RPC endpoints specified, try to find RPC library
        if file_exists(lib_rpc):
            lib = lib_rpc
        elif file_exists(lib_hipblas_rpc):
            lib = lib_hipblas_rpc
        elif file_exists(lib_cublas_rpc):
            lib = lib_cublas_rpc
        else:
            print("WARNING: RPC library not found. Please build with LLAMA_RPC=1")
```

**Why:** Selects appropriate RPC library based on user configuration.

---

## Step 5: Build RPC Library

### 5.1 Build RPC Client Library

```bash
cd /home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp-1.112.2

# Clean previous builds
make clean

# Build RPC library with Vulkan support
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8

# OR with CUDA support
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8

# OR with HIP support
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
```

**Expected output:**
```
g++ ... -DGGML_USE_RPC -c ggml/src/ggml-rpc/ggml-rpc.cpp -o ggml-rpc.o
g++ ... -DGGML_USE_RPC -c ggml/src/ggml-rpc/transport.cpp -o transport.o
g++ ... -DGGML_USE_RPC ggml-rpc.o transport.o ... -shared -o koboldcpp_rpc.so
```

---

### 5.2 Build RPC Servers

```bash
# Build Vulkan RPC server
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8

# Build CUDA RPC server
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8

# Build HIP RPC server
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8

# OR build everything at once
make rpc-full-all
```

**Expected output:**
```
g++ ... tools/rpc-server.cpp ggml-rpc.o ... -o rpc-server-vulkan
```

---

### 5.3 Verify Build Artifacts

```bash
ls -lh koboldcpp_rpc.so rpc-server-*
```

**Expected files:**
```
-rwxr-xr-x ... koboldcpp_rpc.so (70-80 MB)
-rwxr-xr-x ... rpc-server-vulkan (60-70 MB)
-rwxr-xr-x ... rpc-server-cuda (70-80 MB)
-rwxr-xr-x ... rpc-server-hip (70-80 MB)
```

---

## Step 6: Testing

### 6.1 Start RPC Server

```bash
# On remote machine with GPU
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0

# OR with CUDA
./rpc-server-cuda -d CUDA0 -p 50053 -H 0.0.0.0

# OR with HIP
./rpc-server-hip -d HIP0 -p 50053 -H 0.0.0.0
```

**Expected output:**
```
Starting RPC server v4.0.0
  endpoint       : 0.0.0.0:50053
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  VULKAN0: AMD Radeon RX 9060 XT (16304 MiB, 16236 MiB free)
  transport      : TCP
```

---

### 6.2 Connect Client

```bash
# On client machine
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.101:50053 \
    --gpulayers 99 \
    --threads 7 \
    --contextsize 8192
```

**Expected output:**
```
[RPC] Backends loaded, device count: 4
[RPC] Connecting to RPC server(s): 192.168.1.101:50053
[RPC] Adding RPC server: 192.168.1.101:50053
[RPC] Server 192.168.1.101:50053 has 1 devices
[RPC] Found RPC device 0: RPC0[192.168.1.101:50053]
[RPC] Using 1 RPC device(s) for offloading
llama_model_load_from_file_impl: using device RPC0 (192.168.1.101:50053)
Model loaded successfully!
```

---

### 6.3 Multiple RPC Servers

```bash
python koboldcpp.py \
    --model /path/to/model.gguf \
    --rpc 192.168.1.101:50053 192.168.1.102:50053 \
    --gpulayers 99
```

**Expected output:**
```
[RPC] Connecting to RPC server(s): 192.168.1.101:50053,192.168.1.102:50053
[RPC] Adding RPC server: 192.168.1.101:50053
[RPC] Server 192.168.1.101:50053 has 1 devices
[RPC] Found RPC device 0: RPC0[192.168.1.101:50053]
[RPC] Adding RPC server: 192.168.1.102:50053
[RPC] Server 192.168.1.102:50053 has 1 devices
[RPC] Found RPC device 0: RPC0[192.168.1.102:50053]
[RPC] Using 2 RPC device(s) for offloading
llama_model_load_from_file_impl: using device RPC0 (192.168.1.101:50053)
llama_model_load_from_file_impl: using device RPC1 (192.168.1.102:50053)
```

---

## Troubleshooting

### Issue: "Failed to connect to RPC server"

**Cause:** RPC server not running or firewall blocking

**Solution:**
```bash
# Check if server is running
netstat -tlnp | grep 50053

# Test connection
nc -zv 192.168.1.101 50053

# Start server if not running
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0
```

---

### Issue: "No RPC devices found"

**Cause:** RPC server started without GPU devices

**Solution:**
```bash
# Check server output for devices
./rpc-server-vulkan -d VULKAN0 -p 50053

# Verify GPU is detected
vulkaninfo | grep "GPU id"
```

---

### Issue: Segmentation fault on exit

**Cause:** Known issue with newly compiled RPC library

**Solution:**
```bash
# Use pre-compiled working library
cp /path/to/working/koboldcpp_rpc.so koboldcpp_rpc.so
```

---

### Issue: "RPC library not found"

**Cause:** Library not built or wrong name

**Solution:**
```bash
# Build RPC library
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8

# Verify it exists
ls -lh koboldcpp_rpc.so
```

---

## Reference Files

### Key Files Modified

| File | Purpose | Lines Changed |
|------|---------|---------------|
| `Makefile` | Build configuration | ~100 lines |
| `expose.h` | C API structure | ~1 line per struct |
| `gpttype_adapter.cpp` | RPC initialization | ~300 lines |
| `koboldcpp.py` | Python integration | ~100 lines |

### RPC Source Files (from llama.cpp)

| File | Purpose |
|------|---------|
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | RPC implementation |
| `ggml/src/ggml-rpc/transport.cpp` | Network transport |
| `ggml/src/ggml-rpc/transport.h` | Transport interface |
| `ggml/include/ggml-rpc.h` | Public API |
| `tools/rpc-server.cpp` | Server executable |

### Build Artifacts

| File | Purpose |
|------|---------|
| `koboldcpp_rpc.so` | RPC client library |
| `rpc-server-vulkan` | Vulkan RPC server |
| `rpc-server-cuda` | CUDA RPC server |
| `rpc-server-hip` | HIP RPC server |

---

## Quick Reference

### Build Commands

```bash
# Clean build
make clean

# Build RPC library
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8

# Build RPC server
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan -j8

# Build everything
make rpc-full-all
```

### Usage Commands

```bash
# Start RPC server
./rpc-server-vulkan -d VULKAN0 -p 50053 -H 0.0.0.0

# Connect client
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50053 --gpulayers 99

# Multiple servers
python koboldcpp.py --model model.gguf --rpc 192.168.1.101:50053,192.168.1.102:50053 --gpulayers 99
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-29 | Initial release |

---

## Credits

- RPC implementation from llama.cpp/ggml
- Porting guide by koboldcpp team
- Testing and validation by community

---

**End of RPC Porting Guide**

