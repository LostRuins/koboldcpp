#!/bin/bash
# RPC Server Startup Script

# Kill any existing RPC servers
echo "Stopping existing RPC servers..."
pkill -f rpc-server-vulkan
sleep 2

# Start RPC servers
echo "Starting RPC servers..."
nohup ./rpc-server-vulkan -H 0.0.0.0 --port 50053 --device VULKAN0 > /tmp/rpc1.log 2>&1 &
nohup ./rpc-server-vulkan -H 0.0.0.0 --port 50054 --device VULKAN1 > /tmp/rpc2.log 2>&1 &
sleep 3

# Verify servers are running
ps aux | grep rpc-server-vulkan | grep -v grep

echo ""
echo "RPC servers started. You can now run:"
echo "  python koboldcpp.py --config qwen08rpctest3.kcpps"
