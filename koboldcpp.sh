#!/bin/bash
ARCH=$(uname -m)
KCPP_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
KCPP_CONDA_CONFIG_DIR="$KCPP_REPO_DIR/kcpp_src/packaging/environments"
KCPP_MICROMAMBA="$KCPP_REPO_DIR/bin/micromamba"
# micromamba runs pip from a temporary working directory, so these prefixes
# must be absolute rather than relative to the launcher process.
KCPP_MAMBA_ROOT_PREFIX="$KCPP_REPO_DIR/conda"
KCPP_MAMBA_ENV_PREFIX="$KCPP_MAMBA_ROOT_PREFIX/envs/linux"

kcpp_mamba() {
	local command=$1
	shift
	"$KCPP_MICROMAMBA" "$command" -r "$KCPP_MAMBA_ROOT_PREFIX" -p "$KCPP_MAMBA_ENV_PREFIX" "$@"
}

KCPP_ENVIRONMENT_TMP=""
KCPP_RESTORE_OCL_ICD=0
kcpp_cleanup() {
	local status=$?
	local restore_status
	if [ "$KCPP_RESTORE_OCL_ICD" = 1 ]; then
		kcpp_mamba install --no-rc ocl-icd -c conda-forge -y
		restore_status=$?
		if [ "$restore_status" -ne 0 ]; then
			echo "Error: failed to restore ocl-icd." >&2
			if [ "$status" -eq 0 ]; then
				status=$restore_status
			fi
		fi
	fi
	if [ -n "$KCPP_ENVIRONMENT_TMP" ]; then
		rm -f -- "$KCPP_ENVIRONMENT_TMP"
	fi
	exit "$status"
}
trap kcpp_cleanup EXIT

if [ "$ARCH" = "x86_64" ]; then
	ARCH=x64
fi

if [ ! -f "$KCPP_MICROMAMBA" ]; then
	if [ "$ARCH" = "x64" ]; then
		 curl -Ls https://anaconda.org/conda-forge/micromamba/1.5.3/download/linux-64/micromamba-1.5.3-0.tar.bz2 | tar -C "$KCPP_REPO_DIR" -xvj bin/micromamba
	elif [ "$ARCH" = "aarch64" ]; then
		 curl -Ls https://anaconda.org/conda-forge/micromamba/1.5.3/download/linux-aarch64/micromamba-1.5.3-0.tar.bz2 | tar -C "$KCPP_REPO_DIR" -xvj bin/micromamba
	else
		 echo "CPU Architecture $ARCH is not supported by this script, please try compiling manually."
		 exit 1
	fi
fi

NVIDIA_GPU=0
NVIDIA_CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -qE '^GPU [0-9]+:'; then
	NVIDIA_GPU=1
	NVIDIA_CUDA_VERSION=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)
fi

KCPP_INSTALLED_CUDA=""
if [ -f "$KCPP_MAMBA_ENV_PREFIX/cudaver" ]; then
	KCPP_INSTALLED_CUDA=$(<"$KCPP_MAMBA_ENV_PREFIX/cudaver")
fi
KCPP_CREATE_ENV=0
if [[ ! -f "$KCPP_MAMBA_ENV_PREFIX/bin/python" || -z "$KCPP_INSTALLED_CUDA" || $1 == "rebuild" ]] ||
   [[ -n "$KCPP_CUDA" && "$KCPP_CUDA" != "$KCPP_INSTALLED_CUDA" ]]; then
	KCPP_CREATE_ENV=1
fi

if [ "$KCPP_CREATE_ENV" = 1 ] && [ -z "$KCPP_CUDA" ]; then
	if [ "$NVIDIA_GPU" = 1 ]; then
		NVIDIA_CUDA_MAJOR=${NVIDIA_CUDA_VERSION%%.*}
		NVIDIA_CUDA_MINOR=${NVIDIA_CUDA_VERSION#*.}
		NVIDIA_CUDA_MINOR=${NVIDIA_CUDA_MINOR%%.*}
		if [[ "$NVIDIA_CUDA_MAJOR" =~ ^[0-9]+$ && "$NVIDIA_CUDA_MINOR" =~ ^[0-9]+$ ]] &&
		   (( NVIDIA_CUDA_MAJOR > 12 || NVIDIA_CUDA_MAJOR == 12 && NVIDIA_CUDA_MINOR >= 8 )); then
			# CUDA 12.8 retains support for Maxwell/Pascal/Volta and adds
			# native Blackwell SM 100/120 support.
			KCPP_CUDA=12.8.0
		elif [[ "$NVIDIA_CUDA_VERSION" =~ ^(11\.|12\.0) ]]; then
			KCPP_CUDA=11.4.0
		else
			KCPP_CUDA=12.1.0
		fi
	elif command -v rocmsmi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
		KCPP_CUDA=rocm
	else
		KCPP_CUDA=12.1.0
	fi
fi

if [ "$KCPP_CREATE_ENV" = 1 ]; then
	KCPP_ENVIRONMENT_FILE="$KCPP_CONDA_CONFIG_DIR/environment-nocuda.yaml"
	if [ "$KCPP_CUDA" != "rocm" ]; then
		KCPP_ENVIRONMENT_TMP=$(mktemp "${TMPDIR:-/tmp}/koboldcpp-environment.XXXXXX.yaml") || exit 1
		cp "$KCPP_CONDA_CONFIG_DIR/environment.yaml" "$KCPP_ENVIRONMENT_TMP" || exit 1
		sed -i -e "s/nvidia\/label\/cuda-12.1.0/nvidia\/label\/cuda-$KCPP_CUDA/g" "$KCPP_ENVIRONMENT_TMP" || exit 1
		KCPP_ENVIRONMENT_FILE="$KCPP_ENVIRONMENT_TMP"
	fi
	# A failed update can leave a partially changed environment. Only mark it
	# usable after both environment creation and removal of old build objects succeed.
	rm -f "$KCPP_MAMBA_ENV_PREFIX/cudaver" || exit 1
	kcpp_mamba create --no-rc --no-shortcuts -f "$KCPP_ENVIRONMENT_FILE" -y || exit 1
	kcpp_mamba run make -C "$KCPP_REPO_DIR" clean || exit 1
	printf '%s\n' "$KCPP_CUDA" > "$KCPP_MAMBA_ENV_PREFIX/cudaver" || exit 1
else
	KCPP_CUDA="$KCPP_INSTALLED_CUDA"
fi

if [ -z "$KCPP_CUDA" ]; then
	echo "Error: environment toolkit version is missing. Run ./koboldcpp.sh rebuild."
	exit 1
fi
KCPP_CUDAAPPEND=-cuda${KCPP_CUDA//.}$KCPP_APPEND

if [[ "$KCPP_CUDA" == 11.* && -z "$ARCHES_CU11$ARCHES_CU12$ARCHES_CU13" ]]; then
	ARCHES_CU11=1
fi

LLAMA_NOAVX1_FLAG=""
LLAMA_NOAVX2_FLAG=""
ARCHES_FLAG=""
if [ -n "$NOAVX2" ]; then
	LLAMA_NOAVX2_FLAG="LLAMA_NOAVX2=1"
fi
if [ -n "$NOAVX1" ]; then
	LLAMA_NOAVX1_FLAG="LLAMA_NOAVX1=1"
fi
if [ -n "$ARCHES_CU11" ]; then
	ARCHES_FLAG="LLAMA_ARCHES_CU11=1"
fi
if [ -n "$ARCHES_CU12" ]; then
	ARCHES_FLAG="LLAMA_ARCHES_CU12=1"
fi
if [ -n "$ARCHES_CU13" ]; then
	ARCHES_FLAG="LLAMA_ARCHES_CU13=1"
fi

# CUDA's native architecture flag needs a visible GPU; old-CPU builds need the fallback libraries.
if [[ "$KCPP_CUDA" != "rocm" && "$NVIDIA_GPU" = 0 ]] || [[ "$ARCH" = x64 && -n "$NOAVX1$NOAVX2" ]]; then
	KCPP_PORTABLE=1
fi

if [ -n "$KCPP_PORTABLE" ]; then
	LLAMA_PORTABLE_FLAG="LLAMA_PORTABLE=1"
	# The Makefile generates these fallback libraries only on x86.
	if [ "$ARCH" = x64 ]; then
		PORTABLE_SO="--add-data ./koboldcpp_failsafe.so:. --add-data ./koboldcpp_noavx2.so:. --add-data ./koboldcpp_vulkan_noavx2.so:."
		VULKAN_FAILSAFE_SO="--add-data ./koboldcpp_vulkan_failsafe.so:."
	fi
fi

if [ "$KCPP_CUDA" = "rocm" ]; then
	kcpp_mamba run make -C "$KCPP_REPO_DIR" -j$(nproc) LLAMA_VULKAN=1 LLAMA_HIPBLAS=1 LLAMA_USE_BUNDLED_GLSLC=1 LLAMA_ADD_CONDA_PATHS=1 $LLAMA_PORTABLE_FLAG $LLAMA_NOAVX1_FLAG $LLAMA_NOAVX2_FLAG $ARCHES_FLAG
else
	kcpp_mamba run make -C "$KCPP_REPO_DIR" -j$(nproc) LLAMA_VULKAN=1 LLAMA_CUBLAS=1 LLAMA_USE_BUNDLED_GLSLC=1 LLAMA_ADD_CONDA_PATHS=1 $LLAMA_PORTABLE_FLAG $LLAMA_NOAVX1_FLAG $LLAMA_NOAVX2_FLAG $ARCHES_FLAG
fi

if [ $? -ne 0 ]; then
    echo "Error: make failed."
    exit 1
fi

if [[ $1 == "rebuild" ]]; then
	echo Rebuild complete, you can now try to launch Koboldcpp.
elif [[ $1 == "dist" ]]; then
	# Packaging inputs and outputs are relative to the repository. Keep the
	# caller's working directory for normal launches and relative model paths.
	cd -- "$KCPP_REPO_DIR" || exit 1
	if [ ! -n "$KCPP_PORTABLE" ]; then
		echo "WARNING: KCPP_PORTABLE NOT SPECIFIED, THIS BINARY WILL ONLY RUN ON YOUR SYSTEM!!"
		sleep 5
	fi
	KCPP_RESTORE_OCL_ICD=1
	kcpp_mamba remove --no-rc --force ocl-icd -y || exit $?
	kcpp_mamba run pyinstaller --noconfirm --onedir --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --clean --console koboldcpp.py -n "koboldcpp-launcher" || exit $?
	if [ "$KCPP_CUDA" = "rocm" ]; then
		if [ ! -n "$ROCM_PATH" ]; then
			ROCM_PATH=/opt/rocm
		fi
		if [[ "$ARCH" = x64 && -n "$NOAVX1" ]]; then
			kcpp_mamba run pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_hipblas.so:.' $PORTABLE_SO $VULKAN_FAILSAFE_SO --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './embd_res:./embd_res' --add-data "$ROCM_PATH/lib/rocblas:." --add-data "$ROCM_PATH/lib/libamd_comgr.so:." --clean --console koboldcpp.py -n "koboldcpp-linux-$ARCH-rocm" || exit $?
		elif [[ "$ARCH" = x64 && -n "$NOAVX2" ]]; then
			kcpp_mamba run pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_hipblas.so:.' $PORTABLE_SO $VULKAN_FAILSAFE_SO --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './embd_res:./embd_res' --add-data "$ROCM_PATH/lib/rocblas:." --add-data "$ROCM_PATH/lib/libamd_comgr.so:." --clean --console koboldcpp.py -n "koboldcpp-linux-$ARCH-rocm" || exit $?
		else
			kcpp_mamba run pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_default.so:.' --add-data './koboldcpp_hipblas.so:.' --add-data './koboldcpp_vulkan.so:.' $PORTABLE_SO $VULKAN_FAILSAFE_SO --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './embd_res:./embd_res' --add-data "$ROCM_PATH/lib/rocblas:." --add-data "$ROCM_PATH/lib/libamd_comgr.so:." --clean --console koboldcpp.py -n "koboldcpp-linux-$ARCH-rocm" || exit $?
		fi
	else
		if [[ "$ARCH" = x64 && -n "$NOAVX1" ]]; then
			kcpp_mamba run pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_cublas.so:.' $PORTABLE_SO $VULKAN_FAILSAFE_SO --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './embd_res:./embd_res' --clean --console koboldcpp.py -n "koboldcpp-linux-$ARCH$KCPP_CUDAAPPEND" || exit $?
		elif [[ "$ARCH" = x64 && -n "$NOAVX2" ]]; then
			kcpp_mamba run pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_cublas.so:.' $PORTABLE_SO $VULKAN_FAILSAFE_SO --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './embd_res:./embd_res' --clean --console koboldcpp.py -n "koboldcpp-linux-$ARCH$KCPP_CUDAAPPEND" || exit $?
		else
			kcpp_mamba run pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_default.so:.' --add-data './koboldcpp_cublas.so:.' --add-data './koboldcpp_vulkan.so:.' $PORTABLE_SO --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './embd_res:./embd_res' --clean --console koboldcpp.py -n "koboldcpp-linux-$ARCH$KCPP_CUDAAPPEND" || exit $?
			kcpp_mamba run pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all jinja2 --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_default.so:.' --add-data './koboldcpp_vulkan.so:.' $PORTABLE_SO --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './kcpp_agent.py:.' --add-data './kcpp_src/json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './embd_res:./embd_res' --clean --console koboldcpp.py -n "koboldcpp-linux-$ARCH-nocuda$KCPP_APPEND" || exit $?
		fi
	fi
	# The EXIT trap restores ocl-icd on both success and failure.
else
	kcpp_mamba run python "$KCPP_REPO_DIR/koboldcpp.py" "$@"
fi
