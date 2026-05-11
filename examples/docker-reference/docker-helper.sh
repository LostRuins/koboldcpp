
#!/bin/bash
shopt -s extglob

WORKSPACE=/workspace

if [ -d /runpod-volume ]; then
        WORKSPACE=/runpod-volume
fi

if [ ! -d $WORKSPACE/$KCPP_SUBFOLDER ]; then
        mkdir -p $WORKSPACE
        mkdir -p $WORKSPACE/$KCPP_SUBFOLDER
fi
cd $WORKSPACE/$KCPP_SUBFOLDER

if [[ ! -n "$KCPP_MODEL" ]] && [[ ! -n "$KCPP_IMGMODEL" ]] && [[ ! -n "$KCPP_WHISPERMODEL" ]] && [[ ! -n "$KCPP_TTSMODEL" ]] && [[ ! -n "$KCPP_EMBEDMODEL" ]] && [[ ! -n "$KCPP_TTSMODEL" ]]; then
        if [[ ! -n "$KCPP_ARGS" ]]; then
                if [[ $KCPP_DONT_TUNNEL != "true" ]]; then
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                echo "!! This docker will setup a cloudflare tunnel by default as it was designed for GPU rental services.       !!"
                echo "!! Use the KCPP_DONT_TUNNEL=true environment variable if you do not wish this to happen.                   !!"
                echo "!! For example: docker run --rm -e KCPP_DONT_TUNNEL=true -p 5001:5001 -it koboldai/koboldcpp               !!"
                echo "!! The docker compose example mentioned below has optimized defaults for local usage and does not do this. !!"
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                fi
                echo
                echo "Welcome to the official KoboldCpp Docker!"
                echo
                echo "To use KoboldCpp in a docker you must define environment variables."
                echo "Our built in Model Downloader can be used with KCPP_MODEL, KCPP_IMGMODEL, KCPP_MMPROJ, KCPP_EMBEDMODEL, KCPP_TTSMODEL and KCPP_WHISPERMODEL"
                echo "Additional arguments can be specified with KCPP_ARGS, for example for GPU usage: --usecuda mmq --gpulayers 99 --multiuser 20"
                echo "KoboldCpp runs on port 5001 by default, make sure to port forward in docker if you wish to run on your local network."
                echo "For a full list of arguments use --help as the KCPP_ARGS argument."
                echo "Mounting your own models locally instead? Use the --model arg instead of our KCPP_MODEL environment variable"
                echo "You can also mount a volume to /workspace to persist the KCPP_MODEL across restarts"
                echo
                echo "Need an example for Docker Compose? Run: docker run --rm -it koboldai/koboldcpp compose-example"
                echo "Optionally use the following to extract docker-compose.yml to the current directory : docker run --rm -v .:/workspace -it koboldai/koboldcpp compose-example"
                echo
                echo "Questions? Reach out to us on https://koboldai.org/discord for one on one support"
                echo
                echo "Launching KoboldCpp with a tiny demo model in 1 minute, you can test functionality but expect weak logic."
                sleep 60
                KCPP_MODEL=https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_1.gguf?download=true
        fi
fi

# Setup ssh
if [[ $PUBLIC_KEY ]]; then
    echo "Setting up SSH..."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh

    if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
        ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ''
        echo "RSA key fingerprint:"
        ssh-keygen -lf /etc/ssh/ssh_host_rsa_key.pub
    fi

    if [ ! -f /etc/ssh/ssh_host_dsa_key ]; then
        ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -q -N ''
        echo "DSA key fingerprint:"
        ssh-keygen -lf /etc/ssh/ssh_host_dsa_key.pub
    fi

    if [ ! -f /etc/ssh/ssh_host_ecdsa_key ]; then
        ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -q -N ''
        echo "ECDSA key fingerprint:"
        ssh-keygen -lf /etc/ssh/ssh_host_ecdsa_key.pub
    fi

    if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
        ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''
        echo "ED25519 key fingerprint:"
        ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
    fi

    service ssh start

    echo "SSH host keys:"
    for key in /etc/ssh/*.pub; do
        echo "Key: $key"
        ssh-keygen -lf $key
    done
fi

#cat /proc/cpuinfo

export SSL_CERT_DIR=/etc/ssl/certs

if [[ $DEVMODE == "true" ]]; then
        export GLANCES=true
        curl -fsSL https://code-server.dev/install.sh | sh
        code-server --bind-addr 0.0.0.0 --port 3 &
fi

if [[ $GLANCES == "true" ]]; then
        glances -w --enable-mcp &
fi

if [[ $DEVMODE == "true" ]]; then
        rm /usr/sbin/reboot
        rm /usr/sbin/poweroff
        rm /usr/sbin/halt
        rm /usr/sbin/shutdown
        echo "echo 1 > /rr && echo Attempting Reboot..." > /usr/sbin/reboot
        echo "echo 0 > /rr && echo Attempting Shutdown..." > /usr/sbin/poweroff
        echo "echo 0 > /rr && echo Attempting Shutdown..." > /usr/sbin/halt
        echo "echo Please use the poweroff or reboot" > /usr/sbin/shutdown
        chmod +x /usr/sbin/reboot
        chmod +x /usr/sbin/poweroff
        chmod +x /usr/sbin/halt
        chmod +x /usr/sbin/shutdown


        # Wait for a signal to restart
        while true; do
        if [ -f /rr ]; then
                echo "Received exit request by user"
                rebootflag=$(cat /rr)
                # Remove the restart flag
                rm -f /rr
                exit $rebootflag
        fi
        sleep 10
        done
fi

if [ -n "$KCPP_DONT_REMOVE_MODELS" ]; then
        echo "KCPP_DONT_REMOVE_MODELS has been removed and will be ignored, we only delete all files in the mounted workspace if KCPP_REMOVE_FILES is set to true."
fi

if [[ $KCPP_REMOVE_FILES == "true" ]]; then
        echo "REMOVING EVERYTHING THATS NOT KOBOLDCPP"
        rm !(koboldcpp)
        rm -rf splitmodel/
fi

KCPP_CMDLINE="$KCPP_ARGS"

if [ -f "/opt/koboldcpp/default.args" ]; then
        KCPP_CMDLINE="$KCPP_CMDLINE `cat /opt/koboldcpp/default.args`"
fi

if [[ $KCPP_DONT_TUNNEL != "true" ]]; then
        KCPP_CMDLINE="$KCPP_CMDLINE --remotetunnel"
fi

if [[ -n "$KCPP_GIT" ]]; then
        apt update && apt install git curl bzip2 -y
        git clone --recurse-submodules $KCPP_GIT koboldcpp
        cd koboldcpp
        KCPP_BIN=condascript
else
        if [ -f "./koboldcpp.py" ]; then
                echo "A bundled KoboldCpp was detected, we will be using the bundled copy."
                KCPP_BIN=python
        else
                KCPP_BIN=https://koboldai.org/cpplinuxcu12

                if [ ! -f "/usr/bin/nvidia-smi" ]; then
                        if [[ -e "/dev/dri" && "${KCPP_ARGS:-}" != *"--usevulkan"* ]]; then
                                echo "GPU might be AMD and Vulkan not explicitly requested, enabling ROCm support."
                                KCPP_BIN=https://koboldai.org/cpplinuxrocm
                                if ! grep -q avx2 "/proc/cpuinfo"; then
                                        echo "Ancient CPU detected: $(lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1')"
                                        echo "This CPU does not have AVX1 support, we will be using a slower AVX1 mode with Vulkan, compatibility and speed will be degraded."
                                        echo "If your GPU is not Vulkan compatible do not pass it trough to this docker."
                                        KCPP_CMDLINE="$KCPP_CMDLINE --noavx2"
                                        KCPP_BIN=https://koboldai.org/cpplinuxnocu
                                        sleep 30
                                fi
                        else
                                echo "NVIDIA/AMD have not been detected, generic KoboldCpp will be used."
                                KCPP_BIN=https://koboldai.org/cpplinuxnocu
                                if ! grep -q avx2 "/proc/cpuinfo"; then
                                        echo "Ancient CPU detected: $(lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1')"
                                        echo "This CPU does not have AVX2 support, we will be using a much slower AVX1 mode, expect bad performance".
                                        KCPP_CMDLINE="$KCPP_CMDLINE --noavx2"
                                        sleep 30
                                fi
                        fi
                else
                        if nvidia-smi | grep -q 'CUDA Version: 11'; then
                                echo "WARNING: CUDA 11 detected, we will use a binary with AVX1 and CUDA11 for legacy compatibility. Performance is not representative of KoboldCpp's ability."
                                echo "If your GPU supports a newer CUDA version it is highly recommended to update the drivers, if this is a cloud instance consider switching to an updated instance for maximum performance."
                                KCPP_BIN=https://koboldai.org/cpplinux
                                sleep 15
                        fi
                        if nvidia-smi | grep -q 'CUDA Version: 12.0'; then
                                echo "WARNING: CUDA 12.0 detected, we will use a binary with AVX1 and CUDA11 for legacy compatibility. Performance is not representative of KoboldCpp's ability."
                                echo "If your GPU supports a newer CUDA version it is highly recommended to update the drivers, if this is a cloud instance consider switching to an updated instance for maximum performance."
                                KCPP_BIN=https://koboldai.org/cpplinux
                                sleep 15
                        fi
                        if ! grep -q avx2 "/proc/cpuinfo"; then
                                echo "Ancient CPU detected: $(lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1')"
                                echo "This CPU does not have AVX2 support, we will be using AVX1 and CUDA 11, expect worse performance especially when offloading layers.".
                                echo "If this is a cloud instance its recommended to switch to an instance with a modern CPU".
                                KCPP_BIN=https://koboldai.org/cpplinux
                                sleep 30
                        fi
                fi
        fi
fi

if [[ -n "$KCPP_BIN_OVERRIDE" ]]; then
        echo "KCPP Binary Override applied, you are responsible this binary is compatible with the hardware."
        KCPP_BIN=$KCPP_BIN_OVERRIDE
fi

if [[ -n "$KCPP_CONFIG_GIT" ]]; then
        apt update && apt install git -y
        git clone --recurse-submodules $KCPP_CONFIG_GIT configs
        KCPP_CMDLINE="$KCPP_CMDLINE --admin --admindir configs"
fi

if [[ $KCPP_MODEL =~ "LLaMA2-13B" ]]; then
        echo "LLaMA2 13B detected, loading default Llava model."
        export KCPP_MMPROJ="https://huggingface.co/koboldcpp/mmproj/resolve/main/llama-13b-mmproj-v1.5.Q4_1.gguf?download=true"
fi

if [[ -n "$KCPP_MODEL" ]]; then
        KCPP_MODEL=${KCPP_MODEL/"blob"/"resolve"}
        if [[ $KCPP_MODEL =~ "," ]]; then
                if [[ $KCPP_MODEL =~ "00001-of-" ]]; then
                        SPLIT_FIRST_FILE=${KCPP_MODEL%%,*}
                        SPLIT_FIRST_FILE=${SPLIT_FIRST_FILE##*/}
                        SPLIT_FIRST_FILE=${SPLIT_FIRST_FILE%%[?#]*}
                        echo "$SPLIT_FIRST_FILE is a gguf-split file make sure to append all split files with a comma"
                        for i in ${KCPP_MODEL//,/ }
                                do
                                url_filename=${i##*/}
                                url_filename=${url_filename%%[?#]*}
                                aria2c -x 16 -s 16 -o $url_filename -d splitmodel --summary-interval=5 --download-result=default --continue=true --file-allocation=none ${i/"blob"/"resolve"}
                        done
                        KCPP_CMDLINE="$KCPP_CMDLINE --model splitmodel/$SPLIT_FIRST_FILE"
                else
                        for i in ${KCPP_MODEL//,/ }
                                do
                                aria2c -x 16 -s 16 -o kcpp_append -d /tmp --summary-interval=5 --download-result=default --allow-overwrite=true --file-allocation=none ${i/"blob"/"resolve"}
                                echo Appending split... Please wait.
                                cat /tmp/kcpp_append >> ./model.gguf
                                rm /tmp/kcpp_append
                        done
                        KCPP_CMDLINE="$KCPP_CMDLINE --model ./model.gguf"
                fi
        else
                KCPP_CMDLINE="$KCPP_CMDLINE --model $KCPP_MODEL"
        fi
fi

if [[ -n "$KCPP_IMGMODEL" ]]; then
        if [[ $KCPP_IMGMODEL =~ "," ]]; then
                for i in ${KCPP_IMGMODEL//,/ }
                        do
                        aria2c -x 16 -s 16 -o kcpp_append -d /tmp --summary-interval=5 --download-result=default --allow-overwrite=true --file-allocation=none ${i/"blob"/"resolve"}
                        echo Appending split... Please wait.
                        cat /tmp/kcpp_append >> ./imgmodel.gguf
                        rm /tmp/kcpp_append
                done
                KCPP_CMDLINE="$KCPP_CMDLINE --sdmodel ./imgmodel.gguf"
        else
                KCPP_CMDLINE="$KCPP_CMDLINE --sdmodel $KCPP_IMGMODEL"
        fi

fi

if [[ -n "$KCPP_MMPROJ" ]]; then
        if [[ $KCPP_MMPROJ =~ "," ]]; then
                for i in ${KCPP_MMPROJ//,/ }
                        do
                        aria2c -x 16 -s 16 -o kcpp_append -d /tmp --summary-interval=5 --download-result=default --allow-overwrite=true --file-allocation=none ${i/"blob"/"resolve"}
                        echo Appending split... Please wait.
                        cat /tmp/kcpp_append >> ./mmproj.gguf
                        rm /tmp/kcpp_append
                done
                KCPP_CMDLINE="$KCPP_CMDLINE --mmproj ./mmproj.gguf"
        else
                KCPP_CMDLINE="$KCPP_CMDLINE --mmproj $KCPP_MMPROJ"
        fi
fi

if [[ -n "$KCPP_EMBEDMODEL" ]]; then
        if [[ $KCPP_EMBEDMODEL =~ "," ]]; then
                for i in ${KCPP_EMBEDMODEL//,/ }
                        do
                        aria2c -x 16 -s 16 -o kcpp_append -d /tmp --summary-interval=5 --download-result=default --allow-overwrite=true --file-allocation=none ${i/"blob"/"resolve"}
                        echo Appending split... Please wait.
                        cat /tmp/kcpp_append >> ./embeddings.gguf
                        rm /tmp/kcpp_append
                done
                KCPP_CMDLINE="$KCPP_CMDLINE --embeddingsmodel ./embeddings.gguf"
        else
                KCPP_CMDLINE="$KCPP_CMDLINE --embeddingsmodel $KCPP_EMBEDMODEL"
        fi
fi

if [[ -n "$KCPP_WHISPERMODEL" ]]; then
        if [[ $KCPP_WHISPERMODEL =~ "," ]]; then
                for i in ${KCPP_WHISPERMODEL//,/ }
                        do
                        aria2c -x 16 -s 16 -o kcpp_append -d /tmp --summary-interval=5 --download-result=default --allow-overwrite=true --file-allocation=none ${i/"blob"/"resolve"}
                        echo Appending split... Please wait.
                        cat /tmp/kcpp_append >> ./whisper.gguf
                        rm /tmp/kcpp_append
                done
                KCPP_CMDLINE="$KCPP_CMDLINE --whispermodel ./whisper.gguf"
        else
                KCPP_CMDLINE="$KCPP_CMDLINE --whispermodel $KCPP_WHISPERMODEL"
        fi
fi

if [[ -n "$KCPP_TTSMODEL" ]]; then
        if [[ $KCPP_TTSMODEL =~ "," ]]; then
                for i in ${KCPP_TTSMODEL//,/ }
                        do
                        aria2c -x 16 -s 16 -o kcpp_append -d /tmp --summary-interval=5 --download-result=default --allow-overwrite=true --file-allocation=none ${i/"blob"/"resolve"}
                        echo Appending split... Please wait.
                        cat /tmp/kcpp_append >> ./ttsmodel.gguf
                        rm /tmp/kcpp_append
                done
        else
                aria2c -x 16 -s 16 -o ttsmodel.gguf --summary-interval=5 --download-result=default --continue=true --file-allocation=none ${KCPP_TTSMODEL/"blob"/"resolve"}
        fi
        aria2c -x 16 -s 16 -o wavmodel.gguf --summary-interval=5 --download-result=default --continue=true --file-allocation=none https://huggingface.co/koboldcpp/tts/resolve/main/WavTokenizer-Large-75-Q4_0.gguf
        KCPP_CMDLINE="$KCPP_CMDLINE --ttsmodel ./ttsmodel.gguf --ttswavtokenizer ./wavmodel.gguf"
fi

if [[ $KCPP_BIN == "python" ]]; then
        bash -c "python3 koboldcpp.py --quiet $KCPP_CMDLINE"
elif [[ $KCPP_BIN == "condascript" ]]; then
        bash -c "./koboldcpp.sh --quiet $KCPP_CMDLINE"
else
        if [[ $KCPP_DONT_UPDATE == "true" ]] && [[ -f "./koboldcpp" ]]; then
                echo Update check skipped
        else
                aria2c -x 16 -s 16 -o koboldcpp --summary-interval=5 --download-result=default --allow-overwrite=true --file-allocation=none $KCPP_BIN && chmod +x ./koboldcpp
        fi
        bash -c "./koboldcpp --quiet $KCPP_CMDLINE" # Dumb double bash workaround because otherwise user defined quotes don't work for some reason - Henk
fi
echo "Something possibly went wrong, stalling for 3 minutes before exiting so you can check for errors. (No error? You may have run out of memory. Try deleting the image generation model if you don't need it or use a larger GPU.)"
echo "Need some help? https://koboldai.org/discord for one on one support"
