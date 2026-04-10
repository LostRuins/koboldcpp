#!/bin/bash

# Test script for Manual RPC Layer Distribution
# This script demonstrates the new tensor_split feature for RPC servers

echo "=== Manual RPC Layer Distribution Test ==="
echo ""

# Test 1: Check argument parsing
echo "Test 1: Argument Parsing Validation"
echo "-----------------------------------"
python koboldcpp.py --help | grep -A 3 "tensor_split"
echo ""

# Test 2: Example command with 2 RPC servers (commented out - uncomment to run)
echo "Test 2: Example with 2 RPC Servers"
echo "-----------------------------------"
echo "Command (not executed):"
echo "python koboldcpp.py \\"
echo "  --model /path/to/model.gguf \\"
echo "  --rpc 192.168.1.101:50054 192.168.1.16:50054 \\"
echo "  --tensor_split 60 40 \\"
echo "  --gpulayers 999 \\"
echo "  --contextsize 8192 \\"
echo "  --quiet"
echo ""

# Test 3: Example command with 4 RPC servers (commented out - uncomment to run)
echo "Test 3: Example with 4 RPC Servers"
echo "-----------------------------------"
echo "Command (not executed):"
echo "python koboldcpp.py \\"
echo "  --model /path/to/model.gguf \\"
echo "  --rpc server1:50054 server2:50054 server3:50054 server4:50054 \\"
echo "  --tensor_split 20 40 20 20 \\"
echo "  --gpulayers 999 \\"
echo "  --contextsize 8192 \\"
echo "  --quiet"
echo ""

# Test 4: Validation test - mismatched counts
echo "Test 4: Validation - Mismatched Counts"
echo "---------------------------------------"
echo "If you provide 3 tensor_split values for 4 RPC servers:"
echo "  --rpc server1:50054 server2:50054 server3:50054 server4:50054 \\"
echo "  --tensor_split 30 30 40"
echo ""
echo "The system will automatically extend it to: 30 30 40 40"
echo ""

# Test 5: Horde integration example
echo "Test 5: Horde Integration Example"
echo "----------------------------------"
echo "Command (not executed):"
echo "python koboldcpp.py \\"
echo "  --model /path/to/model.gguf \\"
echo "  --rpc 192.168.1.101:50054 192.168.1.16:50054 \\"
echo "  --tensor_split 50 50 \\"
echo "  --gpulayers 999 \\"
echo "  --hordemodelname TestModel \\"
echo "  --hordeworkername TestWorker \\"
echo "  --hordekey test_key \\"
echo "  --hordemaxctx 8192 \\"
echo "  --hordegenlen 256"
echo ""

echo "=== Test Script Complete ==="
echo ""
echo "Note: To actually test with real RPC servers, uncomment the commands above"
echo "and replace the model path and server addresses with your actual values."
