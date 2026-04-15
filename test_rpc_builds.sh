#!/bin/bash

echo "=== RPC Server Build Verification ==="
echo ""

# Check Vulkan RPC server
if [ -f "rpc-server-vulkan" ]; then
    echo "✓ rpc-server-vulkan exists"
    ls -lh rpc-server-vulkan | awk '{print "  Size:", $5}'
else
    echo "✗ rpc-server-vulkan NOT found"
    echo "  Build with: make LLAMA_VULKAN=1 LLAMA_RPC=1 rpc-server-vulkan"
fi

# Check CUDA RPC server
if [ -f "rpc-server-cuda" ]; then
    echo "✓ rpc-server-cuda exists"
    ls -lh rpc-server-cuda | awk '{print "  Size:", $5}'
else
    echo "✗ rpc-server-cuda NOT found"
    echo "  Build with: make LLAMA_CUBLAS=1 LLAMA_RPC=1 rpc-server-cuda"
    echo "  Requires: NVIDIA GPU + CUDA toolkit"
fi

# Check HIPBLAS RPC server
if [ -f "rpc-server-hip" ]; then
    echo "✓ rpc-server-hip exists"
    ls -lh rpc-server-hip | awk '{print "  Size:", $5}'
else
    echo "✗ rpc-server-hip NOT found"
    echo "  Build with: make LLAMA_HIPBLAS=1 LLAMA_RPC=1 rpc-server-hip"
    echo "  Requires: AMD GPU + ROCm"
fi

echo ""
echo "=== GPU Backend Detection ==="

# Check Vulkan
if command -v vulkaninfo &> /dev/null; then
    echo "✓ Vulkan is installed"
    vulkaninfo | grep "deviceName" | head -3 | sed 's/^/  /'
else
    echo "✗ Vulkan NOT detected"
    echo "  Install with: sudo apt-get install vulkan-tools"
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA CUDA is available"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -3 | sed 's/^/  /'
else
    echo "ℹ NVIDIA CUDA NOT detected (normal if you have AMD GPU)"
fi

# Check ROCm
if command -v rocm-smi &> /dev/null; then
    echo "✓ AMD ROCm is installed"
    rocm-smi --showid 2>/dev/null | grep -A 1 "Card" | head -4 | sed 's/^/  /'
else
    echo "ℹ AMD ROCm NOT detected (normal if you have NVIDIA GPU)"
fi

echo ""
echo "=== Test RPC Server (Localhost) ==="
echo "To test your RPC server builds, run one of these commands:"
echo ""
echo "  ./rpc-server-vulkan -H 127.0.0.1 --port 50054 --device VULKAN0 -c"
echo "  ./rpc-server-cuda -H 127.0.0.1 --port 50054 --device CUDA0 -c"
echo "  ./rpc-server-hip -H 127.0.0.1 --port 50054 --device HIP0 -c"
echo ""
