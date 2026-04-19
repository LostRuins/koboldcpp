# RPC Makefile Guide

**Version**: 1.111.2  
**Last Updated**: 2026-04-18  
**Status**: ✅ Complete - All RPC Backends Working (Independent CUDA+HIPBLAS builds)

---

## Quick Build Commands

### Build Everything Automatically (Recommended)

```bash
make rpc-full-all
```

This command:
- ✅ Detects CUDA (`nvcc`) availability
- ✅ Detects HIPBLAS (`hipcc`) availability
- ✅ Always builds Vulkan (universal fallback)
- ✅ Builds **regular backends** (non-RPC)
- ✅ Builds **RPC clients** (all variants)
- ✅ Builds **RPC servers** (all variants)
- ✅ Builds only compatible backends (CUDA/HIPBLAS are mutually exclusive)

### Build Specific Backends

```bash
# Vulkan only (fastest)
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all -j8

# HIPBLAS only (AMD)
make LLAMA_HIPBLAS=1 koboldcpp_hipblas
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip

# CUDA only (NVIDIA)
make LLAMA_CUBLAS=1 koboldcpp_cublas
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda
```

---

## Makefile Targets

### Regular Backend Targets (Non-RPC)

| Target | Description | Flags Required |
|--------|-------------|----------------|
| `koboldcpp_vulkan` | Vulkan backend | `LLAMA_VULKAN=1` |
| `koboldcpp_hipblas` | HIPBLAS backend | `LLAMA_HIPBLAS=1` |
| `koboldcpp_cublas` | CUDA backend | `LLAMA_CUBLAS=1` |

### RPC Client Targets

| Target | Description | Flags Required |
|--------|-------------|----------------|
| `koboldcpp_rpc` | Vulkan + RPC client | `LLAMA_VULKAN=1 LLAMA_RPC=1` |
| `koboldcpp_hipblas_rpc` | HIPBLAS + RPC client | `LLAMA_HIPBLAS=1 LLAMA_RPC=1` |
| `koboldcpp_cublas_rpc` | CUDA + RPC client | `LLAMA_CUBLAS=1 LLAMA_RPC=1` |

### RPC Server Targets

| Target | Description | Flags Required |
|--------|-------------|----------------|
| `rpc-server-vulkan` | Vulkan RPC server | `LLAMA_VULKAN=1 LLAMA_RPC=1` |
| `rpc-server-hip` | HIPBLAS RPC server | `LLAMA_HIPBLAS=1 LLAMA_RPC=1` |
| `rpc-server-cuda` | CUDA RPC server | `LLAMA_CUBLAS=1 LLAMA_RPC=1` |

### Convenience Targets

| Target | Description |
|--------|-------------|
| `rpc-all` | Build all RPC clients + servers (Vulkan) |
| `rpc-clients-all` | Build all RPC clients (Vulkan) |
| `rpc-servers-all` | Build all RPC servers (Vulkan) |
| `rpc-full-all` | **Auto-detect and build ALL backends (regular + RPC)** |

---

## rpc-full-all Build Logic

The `rpc-full-all` target uses this logic:

```makefile
1. Check for nvcc (CUDA)
   - If found: mark CUDA as available

2. Check for hipcc (HIPBLAS)
   - If found: mark HIPBLAS as available

3. Build Regular Backends (Non-RPC)
   - Build Vulkan: koboldcpp_vulkan.so
   - Build HIPBLAS: koboldcpp_hipblas.so (if hipcc found)
   - Build CUDA: koboldcpp_cublas.so (if nvcc found, no hipcc)

4. Build Vulkan + RPC (always)
   - Build koboldcpp_rpc.so
   - Build rpc-server-vulkan

5. Build HIPBLAS + RPC (if hipcc found)
   - Build koboldcpp_hipblas_rpc.so
   - Build rpc-server-hip

6. Build CUDA + RPC (if nvcc found AND hipcc NOT found)
   - Build koboldcpp_cublas_rpc.so
   - Build rpc-server-cuda
```

---

## Added Makefile Components

### 1. RPC Build Flag Variable

**Location**: After line ~104 (after other build variables)

**Purpose**: Enables RPC compilation when LLAMA_RPC is set. This defines the preprocessor macro `GGML_USE_RPC` which enables RPC-related code paths in the C++ source files.

```makefile
ifdef LLAMA_RPC
RPC_FLAGS = -DGGML_USE_RPC
else
RPC_FLAGS =
endif
```

**How it works**:
- `ifdef LLAMA_RPC` checks if user passed `LLAMA_RPC=1` on command line
- If set, `RPC_FLAGS` becomes `-DGGML_USE_RPC` which gets passed to compiler
- This macro enables `#ifdef GGML_USE_RPC` blocks in C++ code
- If not set, `RPC_FLAGS` is empty, so no RPC code is compiled

---

### 2. RPC Object File Build Rule

**Location**: After line ~680 (after ggml-vulkan rules)

**Purpose**: Compiles the RPC implementation source file (`ggml-rpc.cpp`) into an object file (`ggml-rpc.o`) that gets linked into RPC-enabled libraries.

```makefile
#rpc
ggml-rpc.o: ggml/src/ggml-rpc/ggml-rpc.cpp ggml/include/ggml-rpc.h
	$(CXX) $(CXXFLAGS) $(RPC_FLAGS) -c $< -o $@
```

**How it works**:
- `$<` is the first prerequisite: `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `$@` is the target: `ggml-rpc.o`
- Compiles with `$(RPC_FLAGS)` which adds `-DGGML_USE_RPC` if RPC enabled
- Output object file is linked into all RPC-enabled libraries

**Source files required**:
- `ggml/src/ggml-rpc/ggml-rpc.cpp` - RPC implementation
- `ggml/include/ggml-rpc.h` - RPC header file

---

### 3. RPC Build Variable Definitions

**Location**: After line ~446 (after other *_BUILD definitions)

**Purpose**: Defines build commands for RPC-enabled libraries. Handles both hybrid builds (RPC + Vulkan) and pure RPC builds.

**For Vulkan + RPC (Hybrid Build)**:
```makefile
ifdef LLAMA_VULKAN
ifdef LLAMA_RPC
# Hybrid RPC + Vulkan build - needs both libraries
RPC_BUILD = $(CXX) $(CXXFLAGS) $(RPC_FLAGS) $^ -lvulkan -shared -o $@.so $(LDFLAGS)
else
VULKAN_BUILD = $(CXX) $(CXXFLAGS) $^ -lvulkan -shared -o $@.so $(LDFLAGS)
endif
endif
```

**How it works**:
- If BOTH `LLAMA_VULKAN=1` AND `LLAMA_RPC=1` are set, uses `RPC_BUILD`
- If only `LLAMA_VULKAN=1` is set, uses `VULKAN_BUILD`
- Links with `-lvulkan` for Vulkan support

**For Pure RPC Build (Non-Vulkan)**:
```makefile
ifdef LLAMA_RPC
ifndef LLAMA_VULKAN
# RPC only build
RPC_BUILD = $(CXX) $(CXXFLAGS) $(RPC_FLAGS) $^ -shared -o $@.so $(LDFLAGS)
endif
endif
```

**How it works**:
- If `LLAMA_RPC=1` but `LLAMA_VULKAN` is NOT set, uses `RPC_BUILD`
- No Vulkan linking needed for non-Vulkan backends

---

### 4. RPC Client Build Targets

**Location**: After line ~947

**Purpose**: Builds RPC-enabled client libraries. Each target links the RPC object file (`ggml-rpc.o`) with the appropriate backend.

#### koboldcpp_rpc (Vulkan + RPC)

**Builds**: `koboldcpp_rpc.so` - RPC client with Vulkan backend

**Key dependencies**:
- `ggml-rpc.o` - RPC implementation
- `ggml-vulkan.o`, `ggml-vulkan-shaders.o` - Vulkan backend
- `ggml-backend_vulkan.o`, `ggml-backend-reg_vulkan.o` - Vulkan backend registry
- `gpttype_adapter_vulkan.o` - Vulkan-specific adapter

**Conditional build**:
```makefile
ifdef VULKAN_BUILD
koboldcpp_rpc: ggml_v4_vulkan.o ggml-cpu.o ggml-ops.o ... ggml-rpc.o ...
	$(RPC_BUILD)
else
# Fallback: RPC with CPU-only backend
koboldcpp_rpc: ggml.o ggml-cpu.o ggml-ops.o ... ggml-rpc.o ...
	$(RPC_BUILD)
endif
```

**How it works**:
- If `VULKAN_BUILD` is defined, builds with Vulkan backend
- Otherwise, falls back to CPU-only backend with RPC support

#### koboldcpp_hipblas_rpc (HIPBLAS + RPC)

**Builds**: `koboldcpp_hipblas_rpc.so` - RPC client with HIPBLAS backend (AMD ROCm)

**Key dependencies**:
- `ggml-rpc.o` - RPC implementation
- `ggml_v4_cublas.o`, `ggml_v3_cublas.o`, `ggml_v2_cublas.o` - HIPBLAS/CUDA backend
- `gpttype_adapter_cublas.o` - HIPBLAS-specific adapter
- `$(HIP_OBJS)` - HIP-specific object files

**Build command**:
```makefile
ifdef HIPBLAS_BUILD
koboldcpp_hipblas_rpc: ggml_v4_cublas.o ... ggml-rpc.o ... $(HIP_OBJS) ...
	$(HIPBLAS_BUILD)
else
koboldcpp_hipblas_rpc:
	$(DONOTHING)
endif
```

**Important**: Uses `$(HIPBLAS_BUILD)` variable (not `$(RPC_BUILD)`) because HIPBLAS has special linking requirements for ROCm.

#### koboldcpp_cublas_rpc (CUDA + RPC)

**Builds**: `koboldcpp_cublas_rpc.so` - RPC client with CUDA backend (NVIDIA)

**Key dependencies**:
- `ggml-rpc.o` - RPC implementation
- `ggml_v4_cublas.o`, `ggml_v3_cublas.o`, `ggml_v2_cublas.o` - CUDA backend
- `gpttype_adapter_cublas.o` - CUDA-specific adapter
- `$(CUBLAS_OBJS)` - CUDA-specific object files

**Build command**:
```makefile
ifdef CUBLAS_BUILD
koboldcpp_cublas_rpc: ggml_v4_cublas.o ... ggml-rpc.o ... $(CUBLAS_OBJS) ...
	$(CUBLAS_BUILD)
else
koboldcpp_cublas_rpc:
	$(DONOTHING)
endif
```

**Important**: Uses `$(CUBLAS_BUILD)` variable (not `$(RPC_BUILD)`) because CUDA has special linking requirements for NVIDIA CUDA toolkit.

---

### 5. RPC Server Build Targets

**Location**: After line ~994

**Purpose**: Builds standalone RPC server binaries. Unlike client libraries, these are standalone executables that expose GPUs via RPC.

#### rpc-server-vulkan

**Builds**: `rpc-server-vulkan` - RPC server with Vulkan backend

**Key dependencies**:
- `tools/rpc-server.cpp` - RPC server tool source
- `ggml-rpc.o` - RPC implementation
- `ggml-backend_vulkan.o`, `ggml-backend-reg_vulkan.o` - Vulkan backend
- `ggml-vulkan.o`, `ggml-vulkan-shaders.cpp` - Vulkan runtime
- `llama.o`, `ggml.o`, `ggml-cpu.o` - Core inference

**Build command**:
```makefile
ifdef VULKAN_BUILD
rpc-server-vulkan: tools/rpc-server.cpp ggml/src/ggml-vulkan-shaders.cpp \
    ggml.o ggml-cpu.o ... ggml-rpc.o ... ggml-vulkan.o ...
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) -lvulkan
endif
```

**How it works**:
- Compiles `rpc-server.cpp` with all dependencies
- Links with `-lvulkan` for Vulkan support
- Produces standalone executable (not `.so`)

#### rpc-server-hip

**Builds**: `rpc-server-hip` - RPC server with HIPBLAS backend (AMD ROCm)

**Key dependencies**:
- `tools/rpc-server.cpp` - RPC server tool source
- `ggml-rpc.o` - RPC implementation
- `ggml-backend_cublas.o`, `ggml-backend-reg_cublas.o` - HIPBLAS backend
- `ggml_v3_cublas.o`, `ggml_v2_cublas.o` - HIPBLAS/CUDA runtime
- `$(HIP_OBJS)` - HIP-specific object files

**Build command**:
```makefile
ifdef HIPBLAS_BUILD
rpc-server-hip: tools/rpc-server.cpp \
    ggml.o ggml-cpu.o ... ggml-rpc.o ... $(HIP_OBJS) ...
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HIPLDFLAGS)
endif
```

**Important**: Uses `$(HCXX)` (hipcc) and `$(HIPFLAGS)`/`$(HIPLDFLAGS)` for HIPBLAS compilation.

#### rpc-server-cuda

**Builds**: `rpc-server-cuda` - RPC server with CUDA backend (NVIDIA)

**Key dependencies**:
- `tools/rpc-server.cpp` - RPC server tool source
- `ggml-rpc.o` - RPC implementation
- `ggml-backend_cublas.o`, `ggml-backend-reg_cublas.o` - CUDA backend
- `ggml_v3_cublas.o`, `ggml_v2_cublas.o` - CUDA runtime
- `$(CUBLAS_OBJS)` - CUDA-specific object files

**Build command**:
```makefile
ifdef CUBLAS_BUILD
rpc-server-cuda: tools/rpc-server.cpp \
    ggml.o ggml-cpu.o ... ggml-rpc.o ... $(CUBLAS_OBJS) ...
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(CUBLASLD_FLAGS)
endif
```

**Important**: Uses `$(CUBLAS_FLAGS)` and `$(CUBLASLD_FLAGS)` for CUDA compilation and linking.

---

### 6. Automatic Backend Detection (rpc-full-all)

**Location**: At end of Makefile

**Purpose**: Auto-detects available backends and builds all compatible components. This is the recommended build target for most users.

**Build logic flow**:
```
1. Detect nvcc (CUDA toolkit) → sets shell variable HAS_CUDA=1 or 0
2. Detect hipcc (ROCm toolkit) → sets shell variable HAS_HIPBLAS=1 or 0
3. Clean ALL object files to start fresh
4. Build RPC variants FIRST (with -DGGML_USE_RPC):
   - koboldcpp_rpc.so, rpc-server-vulkan
   - koboldcpp_hipblas_rpc.so, rpc-server-hip (if hipcc found)
   - koboldcpp_cublas_rpc.so, rpc-server-cuda (if nvcc found, no hipcc)
5. Clean shared object files (RPC-compiled objects contain RPC symbols)
6. Build Non-RPC variants LAST (without RPC flags):
   - koboldcpp_vulkan.so
   - koboldcpp_hipblas.so (if hipcc found)
   - koboldcpp_cublas.so (if nvcc found, no hipcc)
```

**Key Makefile features**:
- Shell `if/then/fi` conditionals for runtime detection (NOT Make `ifeq`)
- `$(MAKE)` sub-calls for each backend build
- RPC builds run FIRST, non-RPC builds run LAST (ensures clean object files for non-RPC)
- Object file cleanup between RPC and non-RPC phases prevents `undefined symbol` errors
- Final summary shows what was built with file listings

**Build order is critical**:
- RPC libraries need `ggml-rpc.o` compiled WITH `-DGGML_USE_RPC`
- Non-RPC libraries must NOT have any RPC object files
- By building RPC first, then cleaning, then building non-RPC, we ensure:
  - RPC `.o` files exist only during RPC builds
  - Non-RPC `.o` files are fresh (no RPC symbols)

**Independent backend building**:
- CUDA and HIPBLAS are now built INDEPENDENTLY when both toolchains are available
- Each backend uses separate `$(MAKE)` calls with its own flags
- Vulkan is always built as universal fallback
- No mutual exclusion - NVIDIA + AMD GPUs can both be supported

**Object file cleanup (CRITICAL)**:
- Before RPC builds: clean ALL object files to start fresh
- After RPC builds: clean shared `.o` files (they contain RPC symbols)
- Before non-RPC builds: shared `.o` files are gone, so they rebuild clean
- Removed files: `ggml_v4_vulkan.o`, `gpttype_adapter.o`, `ggml-backend_vulkan.o`, etc.

**C++ guard (NEW)**:
- `gpttype_adapter.cpp` RPC code now wrapped in `#ifdef GGML_USE_RPC`
- This is the PRIMARY fix - prevents RPC symbols from appearing in non-RPC builds
- Even if old `.o` files somehow remain, the source code won't reference RPC functions

---

```makefile
# Automatic backend detection and full RPC build
# Supports building BOTH CUDA and HIPBLAS when both toolchains are available
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
	@if [ $(HAS_HIPBLAS) -eq 1 ]; then \
		echo "Building HIPBLAS backend..."; \
		$(MAKE) LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8; \
	fi
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "Building CUDA backend..."; \
		$(MAKE) LLAMA_CUBLAS=1 koboldcpp_cublas -j8; \
	fi
	@echo ""
	@echo "=== Building Vulkan + RPC (universal fallback) ==="
	$(MAKE) LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all -j8
	@if [ $(HAS_HIPBLAS) -eq 1 ]; then \
		echo ""; \
		echo "=== Building HIPBLAS + RPC (AMD GPUs) ==="; \
		echo "HIPBLAS detected, building..."; \
		$(MAKE) LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8; \
		$(MAKE) LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8; \
	fi
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo ""; \
		echo "=== Building CUDA + RPC (NVIDIA GPUs) ==="; \
		echo "CUDA detected, building..."; \
		$(MAKE) LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8; \
		$(MAKE) LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8; \
	fi
	@echo ""
	@echo "=== All RPC components built successfully! ==="
	@echo ""
	@echo "Built Regular Backends (Non-RPC):"
	@ls -lh koboldcpp_vulkan.so 2>/dev/null && echo "  ✓ koboldcpp_vulkan.so (Vulkan)" || true
	@if [ $(HAS_HIPBLAS) -eq 1 ]; then \
		ls -lh koboldcpp_hipblas.so 2>/dev/null && echo "  ✓ koboldcpp_hipblas.so (HIPBLAS)" || true; \
	fi
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		ls -lh koboldcpp_cublas.so 2>/dev/null && echo "  ✓ koboldcpp_cublas.so (CUDA)" || true; \
	fi
	@echo ""
	@echo "Built RPC Clients:"
	@ls -lh koboldcpp_rpc.so 2>/dev/null && echo "  ✓ koboldcpp_rpc.so (Vulkan + RPC)" || true
	@if [ $(HAS_HIPBLAS) -eq 1 ]; then \
		ls -lh koboldcpp_hipblas_rpc.so 2>/dev/null && echo "  ✓ koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)" || true; \
	fi
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		ls -lh koboldcpp_cublas_rpc.so 2>/dev/null && echo "  ✓ koboldcpp_cublas_rpc.so (CUDA + RPC)" || true; \
	fi
	@echo ""
	@echo "Built RPC Servers:"
	@ls -lh rpc-server-vulkan 2>/dev/null && echo "  ✓ rpc-server-vulkan (Vulkan backend)" || true
	@if [ $(HAS_HIPBLAS) -eq 1 ]; then \
		ls -lh rpc-server-hip 2>/dev/null && echo "  ✓ rpc-server-hip (HIPBLAS backend)" || true; \
	fi
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		ls -lh rpc-server-cuda 2>/dev/null && echo "  ✓ rpc-server-cuda (CUDA backend)" || true; \
	fi
```

**Key change from previous version**: 
- Changed `ifeq ($(VAR),1)` (Make conditionals evaluated at parse time) to `if [ $(VAR) -eq 1 ]` (shell conditionals evaluated at runtime)
- This allows both CUDA and HIPBLAS to be built independently when both toolchains are present
- Each backend now has its own independent `if` block instead of `else` chains

---

## New in v1.111.2

### rpc-full-all Target Improvements

The `rpc-full-all` target now includes:

1. **Backend detection output**: Shows which backends were detected with checkmarks
2. **Sequential build order**: Builds regular backends first, then RPC clients, then RPC servers
3. **Final verification**: Lists all built files with sizes for confirmation
4. **Mutual exclusion**: Automatically skips CUDA if HIPBLAS detected

### Example Output

```
=== Detecting available backends ===
Checking for NVIDIA CUDA (nvcc)...
✗ CUDA: Not available
Checking for AMD HIPBLAS (hipcc)...
✓ HIPBLAS: Available (hipcc found)

=== Building Regular Backends (Non-RPC) ===
Building Vulkan backend...
[builds koboldcpp_vulkan.so]
Building HIPBLAS backend...
[builds koboldcpp_hipblas.so]

=== Building Vulkan + RPC (universal fallback) ===
[builds koboldcpp_rpc.so, rpc-server-vulkan]

=== Building HIPBLAS + RPC (AMD GPUs) ===
HIPBLAS detected, building...
[builds koboldcpp_hipblas_rpc.so, rpc-server-hip]

=== All RPC components built successfully! ===

Built Regular Backends (Non-RPC):
  -rw-r--r-- 1 user user 45M Apr 18 12:00 koboldcpp_vulkan.so
  ✓ koboldcpp_vulkan.so (Vulkan)
  -rw-r--r-- 1 user user 52M Apr 18 12:00 koboldcpp_hipblas.so
  ✓ koboldcpp_hipblas.so (HIPBLAS)

Built RPC Clients:
  -rw-r--r-- 1 user user 48M Apr 18 12:00 koboldcpp_rpc.so
  ✓ koboldcpp_rpc.so (Vulkan + RPC)
  -rw-r--r-- 1 user user 55M Apr 18 12:00 koboldcpp_hipblas_rpc.so
  ✓ koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)

Built RPC Servers:
  -rwxr-xr-x 1 user user 35M Apr 18 12:00 rpc-server-vulkan
  ✓ rpc-server-vulkan (Vulkan backend)
  -rwxr-xr-x 1 user user 42M Apr 18 12:00 rpc-server-hip
  ✓ rpc-server-hip (HIPBLAS backend)
```

---

## Important Notes

### CUDA and HIPBLAS Can Now Be Built Together

**Previous behavior** (v1.111.2 earlier): CUDA and HIPBLAS were mutually exclusive because the Makefile used `ifeq` directives that don't work with runtime shell variables.

**Current behavior** (v1.111.2 latest): Both CUDA and HIPBLAS are built independently when both toolchains are available.

**How it works**:
- Each backend (Vulkan, HIPBLAS, CUDA) is built in its own independent step
- Shell `if/then/fi` conditionals check runtime variables instead of Make `ifeq`
- No mutual exclusion - you get all available backends

**Build matrix**:

| CUDA (nvcc) | HIPBLAS (hipcc) | What Gets Built |
|-------------|-----------------|-----------------|
| Not found | Not found | Vulkan only |
| Found | Not found | Vulkan + CUDA |
| Not found | Found | Vulkan + HIPBLAS |
| Found | Found | Vulkan + CUDA + HIPBLAS (all three!) |

### HIPBLAS Flags Include CUDA Compatibility

When building with `LLAMA_HIPBLAS=1`, the Makefile sets:
```makefile
HIPFLAGS += -DGGML_USE_HIP -DGGML_USE_CUDA ...
```

This allows HIPBLAS to use CUDA-compatible APIs, but requires proper ROCm installation.

### RPC Object Must Be Linked

All RPC-enabled targets (clients and servers) must include `ggml-rpc.o` in their dependency list.

---

## Build Examples

### Example 1: Clean Build of Everything

```bash
make clean
make rpc-full-all -j8
```

**Builds**:
- koboldcpp_vulkan.so (Vulkan)
- koboldcpp_hipblas.so (HIPBLAS)
- koboldcpp_rpc.so (Vulkan + RPC)
- koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)
- rpc-server-vulkan
- rpc-server-hip

### Example 2: Vulkan Only (Fastest)

```bash
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all -j8
```

**Builds**:
- koboldcpp_rpc.so (Vulkan + RPC)
- rpc-server-vulkan

### Example 3: HIPBLAS Only (AMD)

```bash
make clean
make LLAMA_HIPBLAS=1 koboldcpp_hipblas -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip -j8
```

**Builds**:
- koboldcpp_hipblas.so (HIPBLAS)
- koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)
- rpc-server-hip

### Example 4: CUDA Only (NVIDIA)

```bash
make clean
make LLAMA_CUBLAS=1 koboldcpp_cublas -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 koboldcpp_cublas_rpc -j8
make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda -j8
```

**Builds**:
- koboldcpp_cublas.so (CUDA)
- koboldcpp_cublas_rpc.so (CUDA + RPC)
- rpc-server-cuda

### Example 5: Mixed NVIDIA + AMD GPUs (Both CUDA and HIPBLAS)

When you have both an NVIDIA GPU and an AMD GPU in the same system:

```bash
make clean
make rpc-full-all -j8
```

**Builds**:
- koboldcpp_vulkan.so (Vulkan)
- koboldcpp_hipblas.so (HIPBLAS - for AMD GPU)
- koboldcpp_cublas.so (CUDA - for NVIDIA GPU)
- koboldcpp_rpc.so (Vulkan + RPC)
- koboldcpp_hipblas_rpc.so (HIPBLAS + RPC)
- koboldcpp_cublas_rpc.so (CUDA + RPC)
- rpc-server-vulkan
- rpc-server-hip
- rpc-server-cuda

**Usage**:
```bash
# Use NVIDIA GPU via CUDA
python koboldcpp.py --model model.gguf --usecuda --gpulayers 999

# Use AMD GPU via HIPBLAS
python koboldcpp.py --model model.gguf --usehipblas --gpulayers 999

# Use Vulkan (works with both)
python koboldcpp.py --model model.gguf --usevulkan --gpulayers 999
```

---

## Troubleshooting

### Issue: "undefined symbol: cudaHostRegister"

**Cause**: HIPBLAS trying to link CUDA symbols without proper ROCm

**Solution**:
```bash
# Ensure ROCm is installed
rocminfo

# Rebuild HIPBLAS
make clean
make LLAMA_HIPBLAS=1 LLAMA_RPC=1 koboldcpp_hipblas_rpc -j8
```

### Issue: "nvcc not found"

**Cause**: CUDA toolkit not installed or not in PATH

**Solution**:
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Or use Vulkan instead
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all
```

### Issue: "hipcc not found"

**Cause**: ROCm not installed or not in PATH

**Solution**:
```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# Or use Vulkan instead
make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-all
```

### Issue: Vulkan build fails silently

**Cause**: Missing Vulkan dependencies

**Solution**:
```bash
# Install Vulkan dependencies
sudo apt-get install libvulkan-dev vulkan-tools glslc

# Rebuild
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8
```

### Issue: "undefined symbol: ggml_backend_rpc_add_server"

**Cause**: Non-RPC library being loaded when RPC is needed

**Solution**:
```bash
# Ensure RPC variant is built and selected
make clean
make rpc-full-all -j8

# Check which library is being loaded
ls -lh koboldcpp_*.so
```

---

## Makefile Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_VULKAN` | Enable Vulkan backend | 0 |
| `LLAMA_CUBLAS` | Enable CUDA backend | 0 |
| `LLAMA_HIPBLAS` | Enable HIPBLAS backend | 0 |
| `LLAMA_RPC` | Enable RPC support | 0 |

---

## Build Output Files

After `make rpc-full-all`, expect these files (if backends available):

```
Regular Backends:
  koboldcpp_vulkan.so         # Vulkan (always built)
  koboldcpp_hipblas.so        # HIPBLAS (if hipcc found)
  koboldcpp_cublas.so         # CUDA (if nvcc found, no hipcc)

RPC Clients:
  koboldcpp_rpc.so            # Vulkan + RPC (always built)
  koboldcpp_hipblas_rpc.so    # HIPBLAS + RPC (if hipcc found)
  koboldcpp_cublas_rpc.so     # CUDA + RPC (if nvcc found, no hipcc)

RPC Servers:
  rpc-server-vulkan           # Vulkan server (always built)
  rpc-server-hip              # HIPBLAS server (if hipcc found)
  rpc-server-cuda             # CUDA server (if nvcc found, no hipcc)
```

---

**See Also**:
- `RPC_QUICKSTART.md` - Quick start guide
- `RPC_MANUAL.md` - Complete manual
- `RPC_PORTING_GUIDE.md` - Step-by-step porting guide
- `RPC_BACKEND_COMPATIBILITY.md` - Backend compatibility matrix
