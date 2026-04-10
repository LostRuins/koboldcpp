# Per-GPU Layer Distribution with RPC

## How It Works

The RPC system automatically exposes **each GPU as a separate device**. When you connect to multiple RPC servers, the client sees all GPU devices from all servers and can distribute layers among them individually.

## Your Scenario

```
Server 1 (192.168.1.101:50054): 3 GPUs
Server 2 (192.168.1.16:50054): 2 GPUs
Total: 5 GPU devices
```

### Desired Distribution

- Server 1, GPU 0: 10% of layers
- Server 1, GPU 1: 40% of layers
- Server 1, GPU 2: 10% of layers
- Server 2, GPU 0: 10% of layers
- Server 2, GPU 1: 30% of layers

### Command

```bash
python koboldcpp.py \
  --model /path/to/model.gguf \
  --rpc 192.168.1.101:50054 192.168.1.16:50054 \
  --tensor_split 10 40 10 10 30 \
  --gpulayers 999 \
  --contextsize 262144 \
  --quiet
```

## What Happens Internally

1. **Client connects to Server 1** (192.168.1.101:50054)
   - Server 1 reports: "I have 3 GPU devices"
   - Client adds all 3 devices to its device list

2. **Client connects to Server 2** (192.168.1.16:50054)
   - Server 2 reports: "I have 2 GPU devices"
   - Client adds all 2 devices to its device list

3. **Client has 5 total GPU devices**
   - Applies tensor_split ratios: [10, 40, 10, 10, 30]
   - Device 0 (Server 1, GPU 0): 10% of layers
   - Device 1 (Server 1, GPU 1): 40% of layers
   - Device 2 (Server 1, GPU 2): 10% of layers
   - Device 3 (Server 2, GPU 0): 10% of layers
   - Device 4 (Server 2, GPU 1): 30% of layers

## Console Output

When you run the command, you should see output like:

```
[RPC] Connecting to RPC server(s): 192.168.1.101:50054,192.168.1.16:50054
[RPC] Adding RPC server: 192.168.1.101:50054
[RPC] Server 192.168.1.101:50054 has 3 devices
[RPC] Found RPC device 0: RPC@192.168.1.101:50054[0]
[RPC] Found RPC device 1: RPC@192.168.1.101:50054[1]
[RPC] Found RPC device 2: RPC@192.168.1.101:50054[2]
[RPC] Adding RPC server: 192.168.1.16:50054
[RPC] Server 192.168.1.16:50054 has 2 devices
[RPC] Found RPC device 0: RPC@192.168.1.16:50054[0]
[RPC] Found RPC device 1: RPC@192.168.1.16:50054[1]
[RPC] Using 5 RPC device(s) for offloading
[RPC] Total GPU devices available: 5
[RPC] Layers will be distributed across all 5 GPU devices based on tensor_split ratios
[RPC] Manual layer distribution requested via tensor_split
[RPC] Total GPU devices discovered: 5
[RPC] === Per-GPU Layer Distribution ===
[RPC] Tensor split ratios (per GPU device): 10.0 40.0 10.0 10.0 30.0 (total: 100.0)
[RPC] Device assignment:
[RPC]   Server 0 (192.168.1.101:50054): GPU devices 0-2 with ratios 10.0 40.0 10.0
[RPC]   Server 1 (192.168.1.16:50054): GPU devices 3-4 with ratios 10.0 30.0
[RPC] Manual layer distribution configured across 5 GPU devices on 2 servers
```

## Important Notes

1. **No Special Server Configuration Needed**: Each rpc-server automatically detects and exposes all its GPUs

2. **Order Matters**: GPU devices are enumerated in the order servers are listed, and within each server in GPU index order (0, 1, 2, ...)

3. **Ratios Are Proportional**: Values like `10 40 10 10 30` work the same as `1 4 1 1 3` - they're ratios, not absolute layer counts

4. **Automatic GPU Detection**: Each rpc-server uses CUDA/Vulkan/HIP to detect all available GPUs on that server

## RPC Server Setup

**Server 1 (3 GPUs):**
```bash
# No special configuration needed!
# Just start the RPC server - it will auto-detect all 3 GPUs
./rpc-server --endpoint 0.0.0.0:50054
```

**Server 2 (2 GPUs):**
```bash
# Same here - auto-detects both GPUs
./rpc-server --endpoint 0.0.0.0:50054
```

### Optional: Control Which GPUs Are Exposed

If you want to hide certain GPUs from the RPC server, use `CUDA_VISIBLE_DEVICES`:

**Server 1 - Use only GPUs 0 and 2 (skip GPU 1):**
```bash
CUDA_VISIBLE_DEVICES=0,2 ./rpc-server --endpoint 0.0.0.0:50054
# Client will see only 2 devices from this server
```

**Server 2 - Use all GPUs (default):**
```bash
./rpc-server --endpoint 0.0.0.0:50054
# Client will see all 2 devices from this server
```

## Testing

To verify the distribution is working:

1. Start the RPC servers on each machine
2. Run the client command with `--tensor_split`
3. Check the console output for the "Per-GPU Layer Distribution" section
4. Verify the device assignment matches your expectations

## Troubleshooting

### GPU Count Mismatch

If the client reports a different number of GPUs than expected:

1. Check that all GPUs are visible on each server:
   ```bash
   nvidia-smi  # For CUDA
   vulkaninfo  # For Vulkan
   ```

2. Verify the RPC server is detecting all GPUs:
   ```
   [RPC] Server X has N devices  # Check N matches expected GPU count
   ```

3. If using `CUDA_VISIBLE_DEVICES`, ensure it includes all desired GPUs

### Layers Not Distributing Correctly

1. Ensure you're using `--gpulayers 999`
2. Verify tensor_split has the correct number of values (one per GPU)
3. Check for warning messages about tensor_split adjustment

## Example Commands

### Equal Distribution (5 GPUs)
```bash
python koboldcpp.py --model model.gguf \
  --rpc server1:50054 server2:50054 \
  --tensor_split 20 20 20 20 20 \
  --gpulayers 999
```

### Power-Weighted Distribution
(Server 1 has more powerful GPUs)
```bash
python koboldcpp.py --model model.gguf \
  --rpc server1:50054 server2:50054 \
  --tensor_split 30 30 30 5 5 \
  --gpulayers 999
```

### Memory-Weighted Distribution
(GPUs with more VRAM get more layers)
```bash
python koboldcpp.py --model model.gguf \
  --rpc server1:50054 server2:50054 \
  --tensor_split 15 50 15 10 10 \
  --gpulayers 999
```
