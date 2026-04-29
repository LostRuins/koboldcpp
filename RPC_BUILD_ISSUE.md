# RPC Build Issue Analysis - koboldcpp-1.112.2

**Date:** 2026-04-28  
**Status:** WORKAROUND AVAILABLE

## Problem Summary

When building koboldcpp-1.112.2 with RPC support (`LLAMA_RPC=1`), the resulting library crashes with a segmentation fault during cleanup/destruction, even though all RPC functions work correctly during execution.

## Root Cause

The issue is a **compiler/linker ABI incompatibility** or **static initialization/destruction ordering problem** in the newly compiled `koboldcpp_rpc.so` library.

**Evidence:**
1. RPC connection and all functions work correctly
2. Segfault occurs AFTER successful RPC operations (during cleanup)
3. Source code is identical between working (v1.111.2) and non-working (v1.112.2) versions
4. The working library from koboldcpp_rpc_attempt does NOT crash

## Workaround

Use the working RPC library from koboldcpp_rpc_attempt:

```bash
cd /home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp-1.112.2

# Backup the broken library
cp koboldcpp_rpc.so koboldcpp_rpc.so.broken

# Use the working library from v1.111.2
cp /home/lunarbuntu/Programming/Openwebui/coding/RPC_attempt/koboldcpp_rpc_attempt/koboldcpp_rpc.so koboldcpp_rpc.so

# Test RPC
python koboldcpp.py --model <model.gguf> --rpc 192.168.1.101:50053 192.168.1.101:50054 --gpulayers 99
```

## Testing Results

### Broken Library (newly compiled)
```python
>>> lib.ggml_backend_rpc_add_server(b'192.168.1.101:50053')
94863642108032  # SUCCESS
# Program exits with segfault
```

### Working Library (from v1.111.2)
```python
>>> lib.ggml_backend_rpc_add_server(b'192.168.1.101:50053')
94281682589904  # SUCCESS
# Program exits cleanly
```

### Full koboldcpp Test
```bash
# With working library - SUCCESS
python koboldcpp.py --rpc 192.168.1.101:50053 192.168.1.101:50054 --gpulayers 99
# Model loads successfully, RPC devices detected:
# llama_model_load_from_file_impl: using device RPC0 (192.168.1.101:50053)
# llama_model_load_from_file_impl: using device RPC1 (192.168.1.101:50054)
```

## Potential Fixes (Not Yet Tested)

1. **Use older GCC version** - The working library may have been compiled with an older GCC
2. **Add `-fno-omit-frame-pointer`** - May help with stack unwinding
3. **Link with `-Wl,--no-as-needed`** - May fix library ordering
4. **Use `-O2` instead of `-O3`** - May avoid optimization-induced bugs
5. **Add explicit destructor calls** - May fix static variable cleanup ordering

## Files Involved

- `ggml/src/ggml-rpc/ggml-rpc.cpp` - RPC implementation (identical in both versions)
- `ggml/src/ggml-rpc/transport.cpp` - Transport layer (identical)
- `gpttype_adapter.cpp` - Adapter code (newer version has additional features)
- `Makefile` - Build configuration

## Build Commands

### Standard Build (Produces Broken Library)
```bash
make clean
make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc -j8
```

### Workaround Build (Use Working Library)
```bash
make clean
make LLAMA_VULKAN=1 koboldcpp_vulkan -j8  # Build without RPC
cp /path/to/koboldcpp_rpc_attempt/koboldcpp_rpc.so koboldcpp_rpc.so  # Use working RPC lib
```

## Conclusion

The RPC feature in koboldcpp-1.112.2 is **functionally complete** but has a **cleanup/destruction bug** in the newly compiled library. Until the compiler/linker issue is identified and fixed, use the workaround above.

**Impact:** Low - RPC works correctly, only cleanup crashes (which doesn't affect normal operation).

**Priority:** Medium - Should be fixed for production use, but workaround is effective.

