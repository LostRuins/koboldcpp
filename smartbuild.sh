#!/bin/bash
# KoboldCpp Smart Build Script
# Auto-detects hardware and builds appropriate backends

set -e

echo "=== KoboldCpp Smart Build ==="
echo ""

# Detect hardware
HAS_AMD=0
HAS_NVIDIA=0
HAS_VULKAN=0
HAS_INTEL_GPU=0

# Check for AMD/ROCm
if lspci -nn 2>/dev/null | grep -qi "1002:"; then
    HAS_AMD=1
    echo "✓ AMD GPU detected"
fi

# Check for NVIDIA
if lspci -nn 2>/dev/null | grep -qi "10de:"; then
    HAS_NVIDIA=1
    echo "✓ NVIDIA GPU detected"
fi

# Check for Intel GPU
if lspci -nn 2>/dev/null | grep -qi "8086:"; then
    HAS_INTEL_GPU=1
    echo "✓ Intel GPU detected"
fi

# Vulkan is generally available on Linux with GPU
if [ $HAS_AMD -eq 1 ] || [ $HAS_NVIDIA -eq 1 ] || [ $HAS_INTEL_GPU -eq 1 ]; then
    HAS_VULKAN=1
    echo "✓ Vulkan support available"
fi

echo ""

# Clean previous builds
echo "Cleaning previous builds..."
make clean >/dev/null 2>&1 || true

# Build standard backends first (always needed for koboldcpp.py)
echo ""
echo "=== Building Standard Backends ==="

# Build CPU default (always)
echo "Building CPU backend (koboldcpp_default.so)..."
make koboldcpp_default -j$(nproc) 2>&1 | tail -5

# Build Vulkan if available
if [ $HAS_VULKAN -eq 1 ]; then
    echo ""
    echo "Building Vulkan backend (koboldcpp_vulkan.so)..."
    make koboldcpp_vulkan -j$(nproc) LLAMA_VULKAN=1 2>&1 | tail -5 || echo "Vulkan build skipped"
fi

# Build HIPBLAS if AMD detected
if [ $HAS_AMD -eq 1 ]; then
    echo ""
    echo "Building HIPBLAS backend (koboldcpp_hipblas.so)..."
    make koboldcpp_hipblas -j$(nproc) LLAMA_HIPBLAS=1 2>&1 | tail -5 || echo "HIPBLAS build skipped (ROCm may not be installed)"
fi

# Build CUDA if NVIDIA detected
if [ $HAS_NVIDIA -eq 1 ]; then
    echo ""
    echo "Building CUDA backend (koboldcpp_cublas.so)..."
    make koboldcpp_cublas -j$(nproc) LLAMA_CUBLAS=1 2>&1 | tail -5 || echo "CUDA build skipped (CUDA may not be installed)"
fi

# Build RPC backends if requested
if [ "$1" == "--rpc" ]; then
    echo ""
    echo "=== Building RPC Backends ==="
    
    # RPC Vulkan
    if [ $HAS_VULKAN -eq 1 ]; then
        echo "Building RPC Vulkan..."
        make rpc-server-vulkan koboldcpp_rpc -j$(nproc) LLAMA_RPC=1 LLAMA_VULKAN=1 2>&1 | tail -5 || echo "RPC Vulkan skipped"
    fi
    
    # RPC HIPBLAS
    if [ $HAS_AMD -eq 1 ]; then
        echo "Building RPC HIPBLAS..."
        make rpc-server-hip koboldcpp_hipblas_rpc -j$(nproc) LLAMA_RPC=1 LLAMA_HIPBLAS=1 2>&1 | tail -5 || echo "RPC HIPBLAS skipped"
    fi
    
    # RPC CUDA
    if [ $HAS_NVIDIA -eq 1 ]; then
        echo "Building RPC CUDA..."
        make rpc-server-cuda koboldcpp_cublas_rpc -j$(nproc) LLAMA_RPC=1 LLAMA_CUBLAS=1 2>&1 | tail -5 || echo "RPC CUDA skipped"
    fi
fi

echo ""
echo "=== Build Summary ==="
echo "Standard backends:"
ls -lh koboldcpp_default.so koboldcpp_vulkan.so koboldcpp_hipblas.so koboldcpp_cublas.so 2>/dev/null || echo "  (checking files...)"

if [ "$1" == "--rpc" ]; then
    echo ""
    echo "RPC backends:"
    ls -lh rpc-server-* koboldcpp_rpc.so 2>/dev/null || echo "  (checking files...)"
fi

echo ""
echo "=== Testing koboldcpp.py ==="
python ./koboldcpp.py --help >/dev/null 2>&1 && echo "✓ koboldcpp.py ready!" || echo "✗ koboldcpp.py has issues"

echo ""
echo "Build complete!"
