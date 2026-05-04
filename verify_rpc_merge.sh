#!/bin/bash
# RPC Merge Verification Script

echo "=== RPC Merge Verification ==="
echo ""

# Check RPC source files
echo "1. Checking RPC source files..."
if [ -f "ggml/src/ggml-rpc/ggml-rpc.cpp" ]; then
    echo "   ✓ ggml-rpc.cpp exists"
else
    echo "   ✗ ggml-rpc.cpp MISSING"
    exit 1
fi

if [ -f "ggml/src/ggml-rpc/transport.cpp" ]; then
    echo "   ✓ transport.cpp exists"
else
    echo "   ✗ transport.cpp MISSING"
    exit 1
fi

if [ -f "ggml/src/ggml-rpc/transport.h" ]; then
    echo "   ✓ transport.h exists"
else
    echo "   ✗ transport.h MISSING"
    exit 1
fi

if [ -f "ggml/src/ggml-rpc/CMakeLists.txt" ]; then
    echo "   ✓ CMakeLists.txt exists"
else
    echo "   ✗ CMakeLists.txt MISSING"
    exit 1
fi

# Check RPC server
echo ""
echo "2. Checking RPC server tool..."
if [ -f "tools/rpc-server.cpp" ]; then
    echo "   ✓ rpc-server.cpp exists"
else
    echo "   ✗ rpc-server.cpp MISSING"
    exit 1
fi

# Check Makefile RPC support
echo ""
echo "3. Checking Makefile RPC support..."
if grep -q "LLAMA_RPC" Makefile; then
    echo "   ✓ LLAMA_RPC flag defined"
else
    echo "   ✗ LLAMA_RPC flag MISSING"
    exit 1
fi

if grep -q "rpc-server-vulkan" Makefile; then
    echo "   ✓ rpc-server-vulkan target defined"
else
    echo "   ✗ rpc-server-vulkan target MISSING"
    exit 1
fi

if grep -q "rpc-full-all" Makefile; then
    echo "   ✓ rpc-full-all target defined"
else
    echo "   ✗ rpc-full-all target MISSING"
    exit 1
fi

# Check RPC header
echo ""
echo "4. Checking RPC header..."
if [ -f "ggml/include/ggml-rpc.h" ]; then
    if grep -q "ggml_backend_rpc_init" ggml/include/ggml-rpc.h; then
        echo "   ✓ ggml-rpc.h exists with RPC API"
    else
        echo "   ✗ ggml-rpc.h MISSING RPC API"
        exit 1
    fi
else
    echo "   ✗ ggml-rpc.h MISSING"
    exit 1
fi

# Check backend registration
echo ""
echo "5. Checking backend registration..."
if grep -q "ggml_backend_rpc_reg()" ggml/src/ggml-backend-reg.cpp; then
    echo "   ✓ RPC backend registration found"
else
    echo "   ✗ RPC backend registration MISSING"
    exit 1
fi

# Check command-line argument support
echo ""
echo "6. Checking command-line argument support..."
if grep -q "add_rpc_devices" common/arg.cpp; then
    echo "   ✓ RPC command-line argument support found"
else
    echo "   ✗ RPC command-line argument support MISSING"
    exit 1
fi

echo ""
echo "=== All Checks Passed! ==="
echo ""
echo "RPC functionality has been successfully merged."
echo ""
echo "To build RPC components:"
echo "  make rpc-full-all                    # Build all RPC components"
echo "  make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan  # Vulkan RPC server"
echo "  make LLAMA_VULKAN=1 LLAMA_RPC=1 koboldcpp_rpc      # Vulkan RPC client"
echo ""
