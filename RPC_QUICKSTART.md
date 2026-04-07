# KoboldCPP RPC - Quick Start Guide

**Version**: koboldcpp_rpc_attempt (based on 1.111.1)  
**Date**: 2026-04-07

---

## Quick Build

### Prerequisites
```bash
sudo apt-get install glslc vulkan-tools libvulkan-dev
```

### Build RPC Components
```bash
cd koboldcpp_rpc_attempt

# 1. Generate Vulkan shaders
make vulkan-shaders-gen

# 2. Build RPC client library
make LLAMA_RPC=1 koboldcpp_rpc

# 3. Build RPC server with Vulkan
make LLAMA_VULKAN=1 rpc-server-vulkan
```

---

## Quick Usage

### Step 1: Start RPC Server (on GPU machine)

```bash
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0,VULKAN1,VULKAN2 -c
```

**Expected Output:**
```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 3 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT ...
Starting RPC server v3.6.1
  endpoint       : 192.168.1.16:50052
  local cache    : /home/user/.cache/llama.cpp/rpc/
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
  Vulkan2: AMD Radeon RX 9060 XT (16304 MiB, 16238 MiB free)
```

**Server is now waiting for client connections.**

### Step 2: Start RPC Client (on client machine)

```bash
python koboldcpp.py --model /path/to/model.gguf --rpc 192.168.1.16:50052 --gpulayers 999
```

**Important Notes:**
- **Server MUST start BEFORE client**
- **Use `--gpulayers 999`** for full RPC offload
- **Don't use `--device`** with RPC client (that's for server)

### Correct Command Examples

**✅ CORRECT - Server:**
```bash
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0,VULKAN1 -c
```

**✅ CORRECT - Client:**
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --gpulayers 999
```

**❌ WRONG - Client (don't use --device):**
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --device VULKAN0
```

**❌ WRONG - Client (don't use local GPU with RPC):**
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --usevulkan 0
```

---

## Common Issues

### Issue 1: Segmentation Fault

**Symptom**: Client crashes immediately

**Cause**: Server not started or wrong command

**Solution**:
1. Start server FIRST
2. Wait for "Starting RPC server" message
3. Then start client with `--gpulayers 999`

### Issue 2: "No GPU backend selected"

**Symptom**: Client says "No GPU or CPU backend was selected"

**Cause**: Using RPC without proper offload settings

**Solution**: Always use `--gpulayers 999` with RPC:
```bash
python koboldcpp.py --model model.gguf --rpc 192.168.1.16:50052 --gpulayers 999
```

### Issue 3: "Connection refused"

**Symptom**: Client can't connect to server

**Cause**: Server not running or wrong IP/port

**Solution**:
1. Check server is running: `ps aux | grep rpc-server`
2. Verify IP address is correct
3. Check port is open: `ss -tlnp | grep 50052`

### Issue 4: Server shows "Client connection closed"

**Symptom**: Server logs show connection then immediate closure

**Cause**: Client crashed or command mismatch

**Solution**:
1. Check client command is correct
2. Ensure `--gpulayers 999` is used
3. Don't mix `--rpc` with `--usevulkan` or `--device`

---

## Server Logs Explanation

**Normal Server Behavior:**
```
Starting RPC server v3.6.1
  endpoint       : 192.168.1.16:50052
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)

Accepted client connection    # ← Client connected
Client connection closed      # ← Client disconnected (normal)
Accepted client connection    # ← Client reconnected
```

**This is NORMAL** - the server accepts connections, serves the client, then closes when done.

---

## Testing

### Test Server Only
```bash
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c
# Should show: Starting RPC server v3.6.1
```

### Test Client Connection
```bash
# Terminal 1: Start server
./rpc-server-vulkan -H 127.0.0.1 --device VULKAN0 -p 50052 -c

# Terminal 2: Start client
python koboldcpp.py --model model.gguf --rpc 127.0.0.1:50052 --gpulayers 999
```

### Verify Listening
```bash
ss -tlnp | grep 50052
# Should show: LISTEN 0  1  127.0.0.1:50052  users:(("rpc-server-vulkan",pid=1234,fd=14))
```

---

## Full Example Session

### Terminal 1 - RPC Server (192.168.1.16)
```bash
cd koboldcpp_rpc_attempt
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0,VULKAN1 -c
```

**Wait for output:**
```
Starting RPC server v3.6.1
  endpoint       : 192.168.1.16:50052
Devices:
  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15232 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (16304 MiB, 16246 MiB free)
```

### Terminal 2 - RPC Client
```bash
cd koboldcpp_rpc_attempt
python koboldcpp.py --model /models/Qwen3.5-0.8B-Q8_0.gguf \
    --rpc 192.168.1.16:50052 \
    --gpulayers 999 \
    --contextsize 8192 \
    --port 5001
```

**Expected output:**
```
Initializing dynamic library: koboldcpp_rpc.so
Loading Text Model: /models/Qwen3.5-0.8B-Q8_0.gguf
[RPC] Connecting to 192.168.1.16:50052...
[RPC] Connected successfully
Starting KoboldCpp on http://localhost:5001
```

### Terminal 3 - Web Browser
```
http://localhost:5001
```

---

## Command Reference

### Server Commands

| Option | Description | Example |
|--------|-------------|---------|
| `-H` | Host IP address | `-H 192.168.1.16` |
| `--port` | Port number | `--port 50052` |
| `--device` | GPU devices | `--device VULKAN0,VULKAN1` |
| `-c` | Use cache | `-c` |

### Client Commands

| Option | Description | Example |
|--------|-------------|---------|
| `--rpc` | RPC server endpoint | `--rpc 192.168.1.16:50052` |
| `--gpulayers` | Layers to offload | `--gpulayers 999` |
| `--model` | Model file | `--model model.gguf` |
| `--contextsize` | Context size | `--contextsize 8192` |
| `--port` | Web server port | `--port 5001` |

---

## Security Warning

⚠️ **NEVER expose RPC server to public internet!**

**Safe configurations:**
```bash
# Localhost only (safest)
./rpc-server-vulkan -H 127.0.0.1 --port 50052 --device VULKAN0 -c

# Private LAN only
./rpc-server-vulkan -H 192.168.1.16 --port 50052 --device VULKAN0 -c
```

**Dangerous (DO NOT USE):**
```bash
./rpc-server-vulkan -H 0.0.0.0 --port 50052 --device VULKAN0 -c
```

---

## Troubleshooting Checklist

- [ ] Server started BEFORE client
- [ ] Server shows "Starting RPC server v3.6.1"
- [ ] Client uses `--rpc` (not `--device`)
- [ ] Client uses `--gpulayers 999`
- [ ] No `--usevulkan` or `--device` on client
- [ ] IP address is correct
- [ ] Port 50052 is open
- [ ] Firewall allows connection

---

## Additional Resources

- **Full Manual**: `RPC_MANUAL.md`
- **Porting Guide**: `RPC_PORTING_GUIDE.md`
- **Build Guide**: See `RPC_MANUAL.md` Section 4

---

**License**: MIT  
**Version**: koboldcpp_rpc_attempt  
**Date**: 2026-04-07
