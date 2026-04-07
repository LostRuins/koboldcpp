# RPC Porting Guide - llama.cpp to koboldcpp

**Version**: 1.0  
**Date**: 2026-04-07  
**Purpose**: Step-by-step guide for porting RPC functionality from llama.cpp to koboldcpp

---

## Overview

This guide documents the complete process of integrating RPC (Remote Procedure Call) functionality from llama.cpp into koboldcpp. It includes all steps taken, challenges encountered, and solutions implemented.

**Goal**: Create a standalone RPC implementation in koboldcpp that:
- Works without dependency on llama.cpp builds
- Supports Vulkan, HIPBLAS, and CUDA backends
- Includes both server and client components
- Integrates with koboldcpp Python wrapper

---

## Prerequisites

### Required Knowledge
- C++ compilation and linking
- CMake and Makefile build systems
- Dynamic library loading
- Backend registration patterns

### Required Tools
```bash
sudo apt-get install build-essential cmake git
sudo apt-get install glslc vulkan-tools libvulkan-dev  # For Vulkan
sudo apt-get install rocblas hipblas-dev               # For HIPBLAS
sudo apt-get install nvidia-cuda-toolkit               # For CUDA
```

### Source Code
- llama.cpp repository (source of RPC code)
- koboldcpp repository (target for integration)

---

## Phase 1: Code Integration

### Step 1.1: Copy RPC Source Files

```bash
# From llama.cpp to koboldcpp
cd koboldcpp-1.111.1

# Copy RPC header
cp ../llama.cpp-b8665/ggml/include/ggml-rpc.h ggml/include/

# Copy RPC implementation
cp -r ../llama.cpp-b8665/ggml/src/ggml-rpc ggml/src/

# Copy RPC server tool
cp ../llama.cpp-b8665/tools/rpc/rpc-server.cpp tools/
```

**File Structure After Copy:**
```
koboldcpp-1.111.1/
├── ggml/
│   ├── include/
│   │   └── ggml-rpc.h          # ← New
│   └── src/
│       └── ggml-rpc/           # ← New directory
│           ├── ggml-rpc.cpp
│           └── CMakeLists.txt
└── tools/
    └── rpc-server.cpp          # ← New
```

### Step 1.2: Verify File Integrity

```bash
# Check files exist
ls -lh ggml/include/ggml-rpc.h
ls -lh ggml/src/ggml-rpc/ggml-rpc.cpp
ls -lh tools/rpc-server.cpp

# Check key functions exist
grep "ggml_backend_rpc_start_server" ggml/src/ggml-rpc/ggml-rpc.cpp
grep "ggml_backend_rpc_reg" ggml/src/ggml-rpc/ggml-rpc.cpp
```

**Expected Output:**
```
2132:GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
2061:ggml_backend_reg_t ggml_backend_rpc_reg(void) {
```

---

## Phase 2: Build System Integration

### Step 2.1: Add RPC Library Variable to Makefile

Edit `koboldcpp-1.111.1/Makefile`:

```makefile
# Add RPC build variable (after line ~427)
ifdef LLAMA_RPC
RPC_FLAGS = -DGGML_USE_RPC
else
RPC_FLAGS =
endif
```

### Step 2.2: Add RPC Library Detection

```makefile
# Add RPC library file detection (after line ~952)
lib_rpc = pick_existant_file("koboldcpp_rpc.dll", "koboldcpp_rpc.so")

# Add to lib_option_pairs (add RPC option)
lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_hipblas, "Use hipBLAS (ROCm)"),
    (lib_vulkan, "Use Vulkan"),
    (lib_rpc, "Use RPC (Remote)"),    # ← New
    (lib_noavx2, "Use CPU (Old CPU)"),
    ...
]
```

### Step 2.3: Add RPC Library Loading Logic

```makefile
# Add RPC library selection in init_library() function (after line ~1017)
elif args.userpc is not None:
    if file_exists(lib_rpc):
        libname = lib_rpc
    else:
        print("WARNING: RPC library not found. Please build with LLAMA_RPC=1")
```

### Step 2.4: Add RPC Build Target

```makefile
# Add RPC client build target (after existing koboldcpp_* targets)
ifdef RPC_BUILD
koboldcpp_rpc: ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o \
    ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter.o ggml-rpc.o \
    sdcpp_default.o whispercpp_default.o tts_default.o music_default.o \
    embeddings_default.o llavaclip_default.o llava.o \
    ggml-backend_default.o ggml-backend-reg_default.o ggml-repack.o \
    $(OBJS_FULL) $(OBJS)
	$(RPC_BUILD)
else
koboldcpp_rpc:
	$(DONOTHING)
endif
```

### Step 2.5: Add RPC Object File Build Rule

```makefile
# Add ggml-rpc.o compilation rule (after ggml-vulkan rules)
#rpc
ggml-rpc.o: ggml/src/ggml-rpc/ggml-rpc.cpp ggml/include/ggml-rpc.h
	$(CXX) $(CXXFLAGS) $(RPC_FLAGS) -c $< -o $@
```

### Step 2.6: Add RPC Server Build Target

```makefile
# Add RPC server with Vulkan support
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

# RPC server CPU-only (fallback)
rpc-server: tools/rpc-server.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o \
    ggml-binops.o ggml-unops.o llama.o ggml-rpc.o ggml-backend_default.o \
    ggml-backend-reg_default.o ggml-repack.o ggml-alloc.o ggml-cpu-traits.o \
    ggml-quants.o ggml-cpu-quants.o kcpp-quantmapper.o kcpp-repackmapper.o \
    unicode.o unicode-common.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o \
    gguf.o sgemm.o common.o llama-impl.o sampling.o budget.o kcpputils.o
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
```

### Step 2.7: Add Vulkan Shaders Generation

```makefile
# Ensure Vulkan shaders are generated before rpc-server-vulkan
rpc-server-vulkan: ggml/src/ggml-vulkan-shaders.cpp ...
```

**Build Command:**
```bash
make vulkan-shaders-gen
make LLAMA_VULKAN=1 rpc-server-vulkan
```

---

## Phase 3: Python Wrapper Integration

### Step 3.1: Add RPC Argument Parser

Edit `koboldcpp-1.111.1/koboldcpp.py`:

```python
# Add RPC argument (after --usevulkan, around line 10698)
compatgroup.add_argument("--usecuda", "--usecublas", "--usehipblas", ...)
compatgroup.add_argument("--usevulkan", ...)
compatgroup.add_argument("--userpc", "--rpc", help="Use RPC for remote model inference. Specify one or more RPC server endpoints (e.g. --rpc 192.168.1.100:50052). Can be used multiple times for multiple servers.", metavar=('[endpoint]'), nargs='+', type=str, default=None)
compatgroup.add_argument("--usecpu", ...)
```

### Step 3.2: Add RPC Field to Structure

```python
# Add rpc_endpoints field to load_model_inputs structure (after vulkan_info, around line 262)
class load_model_inputs(ctypes.Structure):
    _fields_ = [
        ("threads", ctypes.c_int),
        ...
        ("vulkan_info", ctypes.c_char_p),
        ("rpc_endpoints", ctypes.c_char_p),    # ← New
        ("batchsize", ctypes.c_int),
        ...
    ]
```

### Step 3.3: Populate RPC Endpoints

```python
# Set RPC endpoints (after vulkan_info, around line 1163)
if args.usevulkan:
    s = ""
    for it in range(0, len(args.usevulkan)):
        s += str(args.usevulkan[it])
    inputs.vulkan_info = s.encode("UTF-8")
else:
    inputs.vulkan_info = "".encode("UTF-8")

if args.userpc:  # RPC endpoints specified
    s = ",".join(args.userpc)
    inputs.rpc_endpoints = s.encode("UTF-8")
else:
    inputs.rpc_endpoints = "".encode("UTF-8")
```

### Step 3.4: Enable Auto Offload for RPC

```python
# Auto-enable full offload for RPC (after inputs.gpulayers = args.gpulayers, around line 2390)
inputs.gpulayers = args.gpulayers
# Auto-enable full offload for RPC
if args.userpc and args.gpulayers == 0:
    inputs.gpulayers = 999
```

### Step 3.5: Fix Tuple Unpacking

```python
# Update tuple unpacking to include rpc_option (around line 966)
(
    default_option,
    cublas_option,
    hipblas_option,
    vulkan_option,
    rpc_option,    # ← New
    noavx2_option,
    vulkan_noavx2_option,
    vulkan_failsafe_option,
    failsafe_option,
) = (
    opt if file_exists(lib) or (os.name == "nt" and file_exists(opt + ".dll")) else None
    for lib, opt in lib_option_pairs
)
```

---

## Phase 4: Critical Fixes

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

**Solution**: Call `ggml_backend_rpc_start_server()` directly instead of looking it up dynamically.

**Implementation**:

Edit `tools/rpc-server.cpp`:

```cpp
// OLD CODE (DOESN'T WORK FOR STATIC BACKENDS):
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

// NEW CODE (WORKS FOR STATIC BACKENDS):
// Call RPC start server function directly (no need for dynamic lookup)
ggml_backend_rpc_start_server(endpoint.c_str(), cache_dir, params.n_threads, 
                               devices.size(), devices.data());
```

**Why This Works**:
- Direct function call doesn't require runtime lookup
- Statically linked function is available at compile time
- No dependency on dynamic backend loading mechanism

**Verification**:
```bash
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c
# Expected output:
Starting RPC server v3.6.1
  endpoint       : 127.0.0.1:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
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

**Alternative**: Copy from llama.cpp build (temporary):
```bash
cp ../llama.cpp-b8665/build-*/ggml/src/ggml-vulkan/ggml-vulkan-shaders.hpp ggml/src/
```

### ⚠️ HURDLE #3: Linker Optimizes Away RPC Backend

**Problem**: RPC backend function not found at runtime

**Root Cause**: Linker removes unused static functions

**Symptoms**: `nm` shows no RPC symbols in binary

**Solution**: Force linker to keep RPC backend:

```cpp
// In rpc-server.cpp main()
int main(int argc, char * argv[]) {
    // Force RPC backend to be linked (prevents linker from optimizing it away)
    volatile ggml_backend_reg_t rpc_reg = ggml_backend_rpc_reg();
    (void)rpc_reg;
    
    ggml_backend_load_all();
    ...
}
```

**Note**: This was superseded by Hurdle #1 solution (direct function call).

### ⚠️ HURDLE #4: Missing Object Files in Build

**Problem**: Build fails with "ggml-vulkan.o: No such file or directory"

**Root Cause**: Object files not built before linking

**Solution**: Ensure proper dependency order in Makefile:

```makefile
rpc-server-vulkan: ggml/src/ggml-vulkan-shaders.cpp tools/rpc-server.cpp \
    ggml.o ggml-cpu.o ... ggml-vulkan.o console.o
	$(CXX) ...
```

**Build Order**:
1. Generate Vulkan shaders
2. Compile all object files
3. Link final binary

---

## Phase 5: Testing

### Step 5.1: Test RPC Server

```bash
# Start RPC server
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c

# Expected output:
WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT (RADV GFX1200) ...
Starting RPC server v3.6.1
  endpoint       : 127.0.0.1:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
```

**Verify Listening**:
```bash
ss -tlnp | grep 50052
# Expected:
LISTEN 0  1  127.0.0.1:50052  0.0.0.0:*  users:(("rpc-server-vulkan",pid=1234,fd=14))
```

### Step 5.2: Test RPC Client

```bash
# Start RPC client
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50052 --gpulayers 999

# Expected output:
Initializing dynamic library: koboldcpp_rpc.so
Loading Text Model: model.gguf
[RPC] Connecting to 127.0.0.1:50052...
[RPC] Connected successfully
```

### Step 5.3: Test End-to-End

```bash
# Terminal 1: Start server
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c &

# Terminal 2: Start client
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50052 --gpulayers 999

# Verify inference works
curl http://localhost:5001/api/extra/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_length": 20}'
```

---

## Phase 6: Documentation

### Step 6.1: Update RPC_MANUAL.md

Include:
- Build commands
- Usage examples
- Troubleshooting section
- Security warnings

### Step 6.2: Create RPC_BUILD_GUIDE.md

Include:
- Prerequisites
- Step-by-step build instructions
- Backend-specific instructions (Vulkan/HIPBLAS/CUDA)

### Step 6.3: Create RPC_PORTING_GUIDE.md (This Document)

Include:
- Complete integration process
- All hurdles and solutions
- Code snippets for each step

---

## Checklist

### Code Integration
- [ ] Copy `ggml-rpc.h` to `ggml/include/`
- [ ] Copy `ggml-rpc/` directory to `ggml/src/`
- [ ] Copy `rpc-server.cpp` to `tools/`
- [ ] Verify all files exist

### Build System
- [ ] Add `RPC_FLAGS` variable to Makefile
- [ ] Add `lib_rpc` detection
- [ ] Add RPC to `lib_option_pairs`
- [ ] Add RPC library loading logic
- [ ] Add `koboldcpp_rpc` build target
- [ ] Add `ggml-rpc.o` compilation rule
- [ ] Add `rpc-server` build target
- [ ] Add `rpc-server-vulkan` build target
- [ ] Add Vulkan shaders dependency

### Python Wrapper
- [ ] Add `--rpc` / `--userpc` argument
- [ ] Add `rpc_endpoints` field to structure
- [ ] Populate RPC endpoints
- [ ] Enable auto offload for RPC
- [ ] Fix tuple unpacking

### Critical Fixes
- [ ] Implement direct function call (Hurdle #1)
- [ ] Generate Vulkan shaders (Hurdle #2)
- [ ] Force linker to keep backend (Hurdle #3)
- [ ] Ensure proper build order (Hurdle #4)

### Testing
- [ ] Test RPC server starts
- [ ] Test RPC server listens on port
- [ ] Test RPC client connects
- [ ] Test end-to-end inference
- [ ] Test with multiple GPUs
- [ ] Test with multiple servers

### Documentation
- [ ] Update RPC_MANUAL.md
- [ ] Create RPC_BUILD_GUIDE.md
- [ ] Create RPC_PORTING_GUIDE.md (this document)
- [ ] Add version history

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

### Error: "argument model_param: not allowed with argument --model/-m"
**Solution**: Use `--rpc` not positional argument for model

### Error: "Connection refused"
**Solution**: Start RPC server before client, check firewall

### Error: "Segmentation fault"
**Solution**: Use `--gpulayers 999` for full RPC offload

---

## Performance Optimization

### Build Optimization
```bash
# Use all CPU cores for faster build
make -j$(nproc) LLAMA_VULKAN=1 rpc-server-vulkan

# Clean build (recommended after changes)
make clean && make LLAMA_VULKAN=1 rpc-server-vulkan
```

### Runtime Optimization
```bash
# Server: Use multiple GPUs
./rpc-server-vulkan -H 0.0.0.0 --device VULKAN0,VULKAN1,VULKAN2 -p 50052 -c

# Client: Enable debug for troubleshooting
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --debugmode 2
```

---

## Future Improvements

### Potential Enhancements
1. **SSH Tunneling**: Built-in secure tunnel support
2. **Load Balancing**: Automatic distribution across servers
3. **Failover**: Automatic server switching on failure
4. **Authentication**: Token-based access control
5. **Encryption**: TLS/SSL for RPC traffic

### Known Limitations
1. **Security**: RPC protocol is not secure (documented warning)
2. **Static Backends**: Requires direct function call workaround
3. **Vulkan Shaders**: Must be generated before build
4. **Network**: Performance depends on network quality

---

## References

### Source Code
- llama.cpp: https://github.com/ggml-org/llama.cpp
- koboldcpp: https://github.com/LostRuins/koboldcpp

### Documentation
- RPC Manual: `RPC_MANUAL.md`
- Build Guide: `RPC_BUILD_GUIDE.md`
- Quick Reference: `QUICK_REFERENCE.md`

### Tools
- Vulkan SDK: https://vulkan.lunarg.com/
- ROCm: https://rocm.docs.amd.com/
- CUDA: https://developer.nvidia.com/cuda-toolkit

---

## Version History

- **v1.0** (2026-04-07): Initial porting guide created
  - Documents complete integration process
  - Includes all hurdles and solutions
  - Provides step-by-step instructions

---

**License**: MIT  
**Authors**: KoboldCPP Team  
**Contact**: https://github.com/LostRuins/koboldcpp/issues
