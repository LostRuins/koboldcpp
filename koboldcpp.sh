#!/bin/bash

if [ ! -f "bin/micromamba" ]; then
	curl -Ls https://anaconda.org/conda-forge/micromamba/1.5.3/download/linux-64/micromamba-1.5.3-0.tar.bz2 | tar -xvj bin/micromamba
fi

if [[ ! -f "conda/envs/linux/bin/python" && $KCPP_CUDA != "rocm" || $1 == "rebuild" && $KCPP_CUDA != "rocm" ]]; then
	cp environment.yaml environment.tmp.yaml
	if [ -n "$KCPP_CUDA" ]; then
		sed -i -e "s/nvidia\/label\/cuda-12.1.0/nvidia\/label\/cuda-$KCPP_CUDA/g" environment.tmp.yaml
	else
		KCPP_CUDA=12.1.0
	fi
	bin/micromamba create --no-rc --no-shortcuts -r conda -p conda/envs/linux -f environment.tmp.yaml -y
	bin/micromamba create --no-rc --no-shortcuts -r conda -p conda/envs/linux -f environment.tmp.yaml -y
	bin/micromamba run -r conda -p conda/envs/linux make clean
	echo $KCPP_CUDA > conda/envs/linux/cudaver
	echo rm environment.tmp.yaml
fi

if [[ ! -f "conda/envs/linux/bin/python" && $KCPP_CUDA == "rocm" || $1 == "rebuild" && $KCPP_CUDA == "rocm" ]]; then
	bin/micromamba create --no-rc --no-shortcuts -r conda -p conda/envs/linux -f environment-nocuda.yaml -y
	bin/micromamba run -r conda -p conda/envs/linux make clean
	echo "rocm" > conda/envs/linux/cudaver
fi

KCPP_CUDA=$(<conda/envs/linux/cudaver)
KCPP_CUDAAPPEND=-cuda${KCPP_CUDA//.}$KCPP_APPEND

LLAMA_NOAVX2_FLAG=""
ARCHES_FLAG=""
NO_WMMA_FLAG=""
if [ -n "$NOAVX2" ]; then
	LLAMA_NOAVX2_FLAG="LLAMA_NOAVX2=1"
fi
if [ -n "$ARCHES_CU11" ]; then
	ARCHES_FLAG="LLAMA_ARCHES_CU11=1"
fi
if [ -n "$ARCHES_CU12" ]; then
	ARCHES_FLAG="LLAMA_ARCHES_CU12=1"
fi
if [ -n "$NO_WMMA" ]; then
	NO_WMMA_FLAG="LLAMA_NO_WMMA=1"
fi

if [ "$KCPP_CUDA" = "rocm" ]; then
	bin/micromamba run -r conda -p conda/envs/linux make -j$(nproc) LLAMA_VULKAN=1 LLAMA_CLBLAST=1 LLAMA_HIPBLAS=1 LLAMA_PORTABLE=1 LLAMA_USE_BUNDLED_GLSLC=1 LLAMA_ADD_CONDA_PATHS=1 $LLAMA_NOAVX2_FLAG $ARCHES_FLAG $NO_WMMA_FLAG
else
	bin/micromamba run -r conda -p conda/envs/linux make -j$(nproc) LLAMA_VULKAN=1 LLAMA_CLBLAST=1 LLAMA_CUBLAS=1 LLAMA_PORTABLE=1 LLAMA_USE_BUNDLED_GLSLC=1 LLAMA_ADD_CONDA_PATHS=1 $LLAMA_NOAVX2_FLAG $ARCHES_FLAG $NO_WMMA_FLAG
fi

if [ $? -ne 0 ]; then
    echo "Error: make failed."
    exit 1
fi
bin/micromamba run -r conda -p conda/envs/linux chmod +x "./create_ver_file.sh"
bin/micromamba run -r conda -p conda/envs/linux ./create_ver_file.sh

if [[ $1 == "rebuild" ]]; then
	echo Rebuild complete, you can now try to launch Koboldcpp.
elif [[ $1 == "dist" ]]; then
	bin/micromamba remove --no-rc -r conda -p conda/envs/linux --force ocl-icd -y
	bin/micromamba run -r conda -p conda/envs/linux pyinstaller --noconfirm --onedir --collect-all customtkinter --collect-all psutil --add-data './koboldcpp.py:.' --add-data './json_to_gbnf.py:.' --version-file './version.txt' --clean --console koboldcpp.py -n "koboldcpp-launcher"
	if [ "$KCPP_CUDA" = "rocm" ]; then
		if [ ! -n "$ROCM_PATH" ]; then
			ROCM_PATH=/opt/rocm
		fi
		if [ -n "$NOAVX2" ]; then
			bin/micromamba run -r conda -p conda/envs/linux pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_hipblas.so:.' --add-data './koboldcpp_failsafe.so:.' --add-data './koboldcpp_noavx2.so:.' --add-data './koboldcpp_clblast_noavx2.so:.' --add-data './koboldcpp_clblast_failsafe.so:.' --add-data './koboldcpp_vulkan_noavx2.so:.' --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './klite.embd:.' --add-data './kcpp_docs.embd:.' --add-data './kcpp_sdui.embd:.' --add-data './taesd.embd:.' --add-data './taesd_xl.embd:.' --add-data './taesd_f.embd:.' --add-data './taesd_3.embd:.' --add-data './rwkv_vocab.embd:.' --add-data './rwkv_world_vocab.embd:.' --add-data "$ROCM_PATH/lib/rocblas:." --add-data "$ROCM_PATH/lib/libamd_comgr.so:." --version-file './version.txt' --clean --console koboldcpp.py -n "koboldcpp-linux-x64-rocm"
		else
			bin/micromamba run -r conda -p conda/envs/linux pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_default.so:.' --add-data './koboldcpp_hipblas.so:.' --add-data './koboldcpp_vulkan.so:.' --add-data './koboldcpp_clblast.so:.' --add-data './koboldcpp_failsafe.so:.' --add-data './koboldcpp_noavx2.so:.' --add-data './koboldcpp_clblast_noavx2.so:.' --add-data './koboldcpp_clblast_failsafe.so:.' --add-data './koboldcpp_vulkan_noavx2.so:.' --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './klite.embd:.' --add-data './kcpp_docs.embd:.' --add-data './kcpp_sdui.embd:.' --add-data './taesd.embd:.' --add-data './taesd_xl.embd:.' --add-data './taesd_f.embd:.' --add-data './taesd_3.embd:.' --add-data './rwkv_vocab.embd:.' --add-data './rwkv_world_vocab.embd:.' --add-data "$ROCM_PATH/lib/rocblas:." --add-data "$ROCM_PATH/lib/libamd_comgr.so:." --version-file './version.txt' --clean --console koboldcpp.py -n "koboldcpp-linux-x64-rocm"
		fi
	else
		bin/micromamba run -r conda -p conda/envs/linux pyinstaller --noconfirm --onedir --collect-all customtkinter --collect-all psutil --add-data './koboldcpp.py:.' --add-data './json_to_gbnf.py:.' --version-file './version.txt' --clean --console koboldcpp.py -n "koboldcpp-launcher"
		if [ -n "$NOAVX2" ]; then
			bin/micromamba run -r conda -p conda/envs/linux pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_cublas.so:.' --add-data './koboldcpp_failsafe.so:.' --add-data './koboldcpp_noavx2.so:.' --add-data './koboldcpp_clblast_noavx2.so:.' --add-data './koboldcpp_clblast_failsafe.so:.' --add-data './koboldcpp_vulkan_noavx2.so:.' --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './klite.embd:.' --add-data './kcpp_docs.embd:.' --add-data './kcpp_sdui.embd:.' --add-data './taesd.embd:.' --add-data './taesd_xl.embd:.' --add-data './taesd_f.embd:.' --add-data './taesd_3.embd:.' --add-data './rwkv_vocab.embd:.' --add-data './rwkv_world_vocab.embd:.' --version-file './version.txt' --clean --console koboldcpp.py -n "koboldcpp-linux-x64$KCPP_CUDAAPPEND"
		else
			bin/micromamba run -r conda -p conda/envs/linux pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_default.so:.' --add-data './koboldcpp_cublas.so:.' --add-data './koboldcpp_vulkan.so:.' --add-data './koboldcpp_clblast.so:.' --add-data './koboldcpp_failsafe.so:.' --add-data './koboldcpp_noavx2.so:.' --add-data './koboldcpp_clblast_noavx2.so:.' --add-data './koboldcpp_clblast_failsafe.so:.' --add-data './koboldcpp_vulkan_noavx2.so:.' --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './klite.embd:.' --add-data './kcpp_docs.embd:.' --add-data './kcpp_sdui.embd:.' --add-data './taesd.embd:.' --add-data './taesd_xl.embd:.' --add-data './taesd_f.embd:.' --add-data './taesd_3.embd:.' --add-data './rwkv_vocab.embd:.' --add-data './rwkv_world_vocab.embd:.' --version-file './version.txt' --clean --console koboldcpp.py -n "koboldcpp-linux-x64$KCPP_CUDAAPPEND"
			bin/micromamba run -r conda -p conda/envs/linux pyinstaller --noconfirm --onefile --collect-all customtkinter --collect-all psutil --add-data './dist/koboldcpp-launcher/koboldcpp-launcher:.' --add-data './koboldcpp_default.so:.' --add-data './koboldcpp_vulkan.so:.' --add-data './koboldcpp_clblast.so:.' --add-data './koboldcpp_failsafe.so:.' --add-data './koboldcpp_noavx2.so:.' --add-data './koboldcpp_clblast_noavx2.so:.' --add-data './koboldcpp_clblast_failsafe.so:.' --add-data './koboldcpp_vulkan_noavx2.so:.' --add-data './kcpp_adapters:./kcpp_adapters' --add-data './koboldcpp.py:.' --add-data './json_to_gbnf.py:.' --add-data './LICENSE.md:.' --add-data './MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:.' --add-data './klite.embd:.' --add-data './kcpp_docs.embd:.' --add-data './kcpp_sdui.embd:.' --add-data './taesd.embd:.' --add-data './taesd_xl.embd:.' --add-data './taesd_f.embd:.' --add-data './taesd_3.embd:.' --add-data './rwkv_vocab.embd:.' --add-data './rwkv_world_vocab.embd:.' --version-file './version.txt' --clean --console koboldcpp.py -n "koboldcpp-linux-x64-nocuda$KCPP_APPEND"
		fi
	fi
	bin/micromamba install --no-rc -r conda -p conda/envs/linux ocl-icd -c conda-forge -y
else
	bin/micromamba run -r conda -p conda/envs/linux python koboldcpp.py $*
fi
