#!/bin/bash

# Exit on any error
set -e

if [ "$(uname -o)" != "Android" ]; then
echo "Error: This script is only intended for Termux on Android!"
exit 1
fi

echo "--------------------------------------------"
echo "KoboldCPP Quick Installer for Termux (Android only!)"
echo "--------------------------------------------"
if [ $# -ge 1 ]; then
    choice="$1"
    echo "Using command-line argument: $choice"
# Check if running interactively (terminal input)
elif [ -t 0 ]; then
    # Running interactively
    echo "[1] - Proceed to install and launch with default model Gemma3-1B"
    echo "[2] - Proceed to install without a model, you can download one later."
    echo "[3] - Download GGUF model from web URL (Requires already installed)"
    echo "[4] - Load existing GGUF model from disk (Requires already installed)"
    echo "[5] - Rebuild existing KoboldCPP installation"
    echo "[6] - Exit script"
    echo "--------------------------------------------"
    read -r -p "Enter your choice [1-6]: " choice
else
    # Non-interactive, default to choice 1
    echo "Defaulting to normal install and model download. Run script interactively for other options. Install will start in 3 seconds."
    choice="1"
    sleep 3
fi

# Determine script directory (works for both curl|sh and ./install.sh)
if [ -f "$0" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"  # Normal execution (./install.sh)
else
    SCRIPT_DIR="$(pwd)"  # Piped execution (curl | sh)
fi

# Locate an existing checkout regardless of the caller's current directory.
find_koboldcpp_dir() {
    if [ -f "$SCRIPT_DIR/koboldcpp.py" ]; then
        KOBOLDCPP_DIR="$SCRIPT_DIR"
    elif [ -f "$SCRIPT_DIR/koboldcpp/koboldcpp.py" ]; then
        KOBOLDCPP_DIR="$SCRIPT_DIR/koboldcpp"
    else
        return 1
    fi
}

require_koboldcpp_dir() {
    if ! find_koboldcpp_dir; then
        echo "Error: No existing KoboldCPP installation found near $SCRIPT_DIR"
        exit 1
    fi
}

launch_model() {
    local model="$1"
    local context_size="${2:-}"

    if [ -z "$context_size" ] && [ -t 0 ]; then
        read -r -p "Enter desired context size [8192]: " context_size
    fi
    context_size="${context_size:-8192}"

    if ! [[ "$context_size" =~ ^[0-9]+$ ]] || [ "$context_size" -lt 256 ] || [ "$context_size" -gt 524288 ]; then
        echo "Error: Context size must be an integer between 256 and 524288."
        exit 1
    fi

    context_size=$((10#$context_size))
    echo "[*] Launching with context size $context_size..."
    python koboldcpp.py --contextsize "$context_size" --model "$model"
}

CONTEXT_SIZE_ARG="${2:-}"
FORCE_REBUILD=false

# handle user choice
if [ "$choice" = "6" ]; then
    echo "Exiting script. Goodbye!"
    exit 0
elif [ "$choice" = "4" ]; then
    require_koboldcpp_dir
    echo "[*] Searching for .gguf model files in $KOBOLDCPP_DIR..."
    mapfile -d '' -t MODEL_FILES < <(find "$KOBOLDCPP_DIR" -maxdepth 1 -type f -iname "*.gguf" -print0 2>/dev/null)
    if [ "${#MODEL_FILES[@]}" -eq 0 ]; then
        echo "No .gguf model files found in $KOBOLDCPP_DIR"
        exit 1
    fi
    echo "Available model files:"
    for i in "${!MODEL_FILES[@]}"; do
        echo "[$((i+1))] ${MODEL_FILES[$i]}"
    done
    read -r -p "Enter the number of the model you want to load: " model_choice
    # Validate input
    if ! [[ "$model_choice" =~ ^[0-9]+$ ]] || [ "$model_choice" -lt 1 ] || [ "$model_choice" -gt "${#MODEL_FILES[@]}" ]; then
        echo "Invalid selection."
        exit 1
    fi
    selected_index=$((10#$model_choice - 1))
    SELECTED_MODEL="${MODEL_FILES[$selected_index]}"
    echo "Now launching with model $SELECTED_MODEL"
    cd "$KOBOLDCPP_DIR"
    launch_model "$SELECTED_MODEL" "$CONTEXT_SIZE_ARG"
    exit 0
elif [ "$choice" = "3" ]; then
    require_koboldcpp_dir
    read -r -p "Please input FULL URL of model you wish to download and run: " SELECTED_MODEL
    echo "Starting download of model $SELECTED_MODEL"
    cd "$KOBOLDCPP_DIR"
    launch_model "$SELECTED_MODEL" "$CONTEXT_SIZE_ARG"
    exit 0
elif [ "$choice" = "5" ]; then
    echo "[*] Rebuild existing KoboldCPP installation..."
    require_koboldcpp_dir
    INSTALL_MODEL=false
    FORCE_REBUILD=true
elif [ "$choice" = "2" ]; then
    echo "[*] Install without model download..."
    INSTALL_MODEL=false
elif [ "$choice" = "1" ]; then
    echo "[*] Install with model download..."
    INSTALL_MODEL=true
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "[*] Checking Dependencies..."
check_wget=$(command -v wget || true)
check_git=$(command -v git || true)
check_python=$(command -v python || true)
if [ -n "$check_wget" ] && [ -n "$check_git" ] && [ -n "$check_python" ]; then
    echo "[*] Dependencies are already installed..."
else
    echo "[*] Setup dependencies..."
    apt update
    DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" openssl
    pkg install -y wget git python
    pkg upgrade -o Dpkg::Options::="--force-confold" -y
fi

# Check if koboldcpp.py already exists nearby
if find_koboldcpp_dir; then
    echo "[*] Detected existing koboldcpp.py in $KOBOLDCPP_DIR"
else
    echo "[*] No existing koboldcpp found. Cloning repository..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/LostRuins/koboldcpp.git
    KOBOLDCPP_DIR="$SCRIPT_DIR/koboldcpp"
fi

# build if needed
cd "$KOBOLDCPP_DIR"
if [ "$FORCE_REBUILD" = true ]; then
    echo "[*] Cleaning the existing build..."
    make clean
    echo "[*] Rebuilding KoboldCPP now..."
    make -j 2
elif [ -f "$KOBOLDCPP_DIR/koboldcpp_default.so" ]; then
    echo "[*] Found koboldcpp_default.so — skipping build step."
else
    echo "[*] Building KoboldCPP now..."
    make -j 2
fi

# grab model if needed
echo "==="
echo "[*] Your KoboldCPP Installation is Complete!"
if [ "$INSTALL_MODEL" = true ]; then
    echo "[*] Downloading Gemma3-1B, a small GGUF model..."
    launch_model "https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf" "$CONTEXT_SIZE_ARG"
else
    echo "To use it, please obtain a GGUF model, then run it with the command 'python koboldcpp.py --model (your_gguf)' and then open a web browser to http://localhost:5001"
fi
