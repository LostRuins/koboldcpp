# Add custom options to Makefile.local rather than editing this file.
-include $(abspath $(lastword ${MAKEFILE_LIST})).local

.PHONY: finishedmsg

default: koboldcpp_default koboldcpp_failsafe koboldcpp_noavx2 koboldcpp_clblast koboldcpp_clblast_noavx2 koboldcpp_clblast_failsafe koboldcpp_cublas koboldcpp_hipblas koboldcpp_vulkan koboldcpp_vulkan_noavx2 finishedmsg
tools: quantize_gpt2 quantize_gptj quantize_gguf quantize_neox quantize_mpt quantize_clip ttsmain whispermain sdmain gguf-split

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

ifndef UNAME_O
UNAME_O := $(shell uname -o)
endif

ifneq ($(shell grep -e "Arch Linux" -e "ID_LIKE=arch" /etc/os-release 2>/dev/null),)
ARCH_ADD = -lcblas
endif


# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
ifneq ($(UNAME_P),arm)
	SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
ifeq ($(SYSCTL_M),1)
	# UNAME_P := arm
	# UNAME_M := arm64
	warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
endif
endif
endif

#
# Compile flags
#

# keep standard at C11 and C++17
CFLAGS =
CXXFLAGS =
ifdef KCPP_DEBUG
	CFLAGS = -g -O0
	CXXFLAGS = -g -O0
endif
CFLAGS   += -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -Iinclude -Isrc -I./common -I./vendor -I./vendor/stb -I./include -I./include/CL -I./otherarch -I./otherarch/tools -I./otherarch/sdcpp -I./otherarch/sdcpp/thirdparty -I./include/vulkan -O3 -fno-finite-math-only -std=c11 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_CPU -DGGML_USE_CPU_AARCH64
CXXFLAGS += -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -Iinclude -Isrc -I./common -I./vendor -I./vendor/stb -I./include -I./include/CL -I./otherarch -I./otherarch/tools -I./otherarch/sdcpp -I./otherarch/sdcpp/thirdparty -I./include/vulkan -O3 -fno-finite-math-only -std=c++17 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_CPU -DGGML_USE_CPU_AARCH64
ifndef KCPP_DEBUG
	CFLAGS += -DNDEBUG -s
	CXXFLAGS += -DNDEBUG -s
endif
ifdef LLAMA_NO_LLAMAFILE
GGML_NO_LLAMAFILE := 1
endif
ifndef GGML_NO_LLAMAFILE
	CFLAGS += -DGGML_USE_LLAMAFILE
	CXXFLAGS += -DGGML_USE_LLAMAFILE
endif

#lets try enabling everything
CFLAGS   += -pthread -Wno-deprecated -Wno-deprecated-declarations -Wno-unused-variable
CXXFLAGS += -pthread -Wno-multichar -Wno-write-strings -Wno-deprecated -Wno-deprecated-declarations -Wno-unused-variable

LDFLAGS  =
FASTCFLAGS = $(subst -O3,-Ofast,$(CFLAGS))
FASTCXXFLAGS = $(subst -O3,-Ofast,$(CXXFLAGS))

# these are used on windows, to build some libraries with extra old device compatibility
SIMPLECFLAGS =
SIMPLERCFLAGS =
FULLCFLAGS =
NONECFLAGS =

CLBLAST_FLAGS = -DGGML_USE_CLBLAST
FAILSAFE_FLAGS = -DUSE_FAILSAFE
VULKAN_FLAGS = -DGGML_USE_VULKAN -DSD_USE_VULKAN
ifdef LLAMA_CUBLAS
	CUBLAS_FLAGS = -DGGML_USE_CUDA -DSD_USE_CUDA
else
	CUBLAS_FLAGS =
endif
CUBLASLD_FLAGS =
CUBLAS_OBJS =

OBJS_FULL += ggml-alloc.o ggml-cpu-traits.o ggml-quants.o ggml-cpu-quants.o ggml-cpu-aarch64.o unicode.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm.o common.o sampling.o kcpputils.o
OBJS_SIMPLE += ggml-alloc.o ggml-cpu-traits.o ggml-quants_noavx2.o ggml-cpu-quants_noavx2.o ggml-cpu-aarch64_noavx2.o unicode.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm_noavx2.o common.o sampling.o kcpputils.o
OBJS_SIMPLER += ggml-alloc.o ggml-cpu-traits.o ggml-quants_noavx1.o ggml-cpu-quants_noavx1.o ggml-cpu-aarch64_noavx1.o unicode.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm_noavx1.o common.o sampling.o kcpputils.o
OBJS_FAILSAFE += ggml-alloc.o ggml-cpu-traits.o ggml-quants_failsafe.o ggml-cpu-quants_failsafe.o ggml-cpu-aarch64_failsafe.o unicode.o unicode-data.o ggml-threading.o ggml-cpu-cpp.o gguf.o sgemm_failsafe.o common.o sampling.o kcpputils.o

# OS specific
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
	LDFLAGS += -ldl
endif

ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
	CLANG_VER = $(shell clang -v 2>&1 | head -n 1 | awk 'BEGIN {FS="[. ]"};{print $$1 $$2 $$4}')
ifeq ($(CLANG_VER),Appleclang15)
	LDFLAGS += -ld_classic
endif
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),OpenBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifdef LLAMA_PERF
	CFLAGS   += -DGGML_PERF
	CXXFLAGS += -DGGML_PERF
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Architecture specific
# For x86 based architectures
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
ifdef LLAMA_PORTABLE
	SIMPLECFLAGS += -mavx -msse3 -mssse3
	SIMPLERCFLAGS += -msse3 -mssse3
ifdef LLAMA_NOAVX2
	FULLCFLAGS += -msse3 -mssse3 -mavx
else
	FULLCFLAGS += -mavx2 -msse3 -mssse3 -mfma -mf16c -mavx
endif # LLAMA_NOAVX2
else
	CFLAGS += -march=native -mtune=native
	SIMPLECFLAGS += -march=native -mtune=native
	SIMPLERCFLAGS += -march=native -mtune=native
	FULLCFLAGS += -march=native -mtune=native
endif # LLAMA_PORTABLE
endif # if x86

ifndef LLAMA_NO_ACCELERATE
# Mac M1 - include Accelerate framework.
# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
ifeq ($(UNAME_S),Darwin)
	CFLAGS  += -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DGGML_BLAS_USE_ACCELERATE
	CXXFLAGS  += -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DGGML_BLAS_USE_ACCELERATE
	LDFLAGS += -framework Accelerate
	OBJS += ggml-blas.o
endif
endif

# it is recommended to use the CMAKE file to build for cublas if you can - will likely work better
OBJS_CUDA_TEMP_INST = $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-mma*.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/mmq*.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*f16-f16.cu))

ifdef LLAMA_CUBLAS
	CUBLAS_FLAGS = -DGGML_USE_CUDA -DSD_USE_CUDA -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	CUBLASLD_FLAGS = -lcuda -lcublas -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib -L$(CUDA_PATH)/lib64/stubs -L/usr/local/cuda/targets/aarch64-linux/lib -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/lib/wsl/lib
	CUBLAS_OBJS = ggml-cuda.o ggml_v3-cuda.o ggml_v2-cuda.o ggml_v2-cuda-legacy.o
	CUBLAS_OBJS += $(patsubst %.cu,%.o,$(filter-out ggml/src/ggml-cuda/ggml-cuda.cu, $(wildcard ggml/src/ggml-cuda/*.cu)))
	CUBLAS_OBJS += $(OBJS_CUDA_TEMP_INST)
	NVCC      = nvcc
	NVCCFLAGS = --forward-unknown-to-host-compiler -use_fast_math -extended-lambda

ifdef LLAMA_ADD_CONDA_PATHS
	CUBLASLD_FLAGS += -Lconda/envs/linux/lib -Lconda/envs/linux/lib/stubs
endif


ifdef LLAMA_PORTABLE

ifdef LLAMA_ARCHES_CU11
	NVCCFLAGS += -Wno-deprecated-gpu-targets \
	             -gencode arch=compute_35,code=compute_35 \
	             -gencode arch=compute_50,code=compute_50 \
	             -gencode arch=compute_61,code=compute_61 \
	             -gencode arch=compute_70,code=compute_70 \
	             -gencode arch=compute_75,code=compute_75 \
	             -DKCPP_LIMIT_CUDA_MAX_ARCH=750

else ifdef LLAMA_ARCHES_CU12
	NVCCFLAGS += -Wno-deprecated-gpu-targets \
	             -gencode arch=compute_50,code=compute_50 \
	             -gencode arch=compute_61,code=compute_61 \
	             -gencode arch=compute_70,code=compute_70 \
	             -gencode arch=compute_75,code=compute_75 \
	             -gencode arch=compute_80,code=compute_80

else
	NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=all
endif

else
	NVCCFLAGS += -arch=native
endif # LLAMA_PORTABLE

ifdef LLAMA_CUDA_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_F16
ifdef LLAMA_CUDA_DMMV_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_DMMV_F16

ifdef LLAMA_CUDA_CCBIN
	NVCCFLAGS += -ccbin $(LLAMA_CUDA_CCBIN)
endif

ggml/src/ggml-cuda/%.o: ggml/src/ggml-cuda/%.cu ggml/include/ggml.h ggml/src/ggml-common.h ggml/src/ggml-cuda/common.cuh
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(HIPFLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml-cuda.o: ggml/src/ggml-cuda/ggml-cuda.cu ggml/include/ggml-cuda.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/src/ggml-backend-impl.h ggml/src/ggml-common.h $(wildcard ggml/src/ggml-cuda/*.cuh)
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(HIPFLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v2-cuda.o: otherarch/ggml_v2-cuda.cu otherarch/ggml_v2-cuda.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(HIPFLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v2-cuda-legacy.o: otherarch/ggml_v2-cuda-legacy.cu otherarch/ggml_v2-cuda-legacy.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(HIPFLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v3-cuda.o: otherarch/ggml_v3-cuda.cu otherarch/ggml_v3-cuda.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(HIPFLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
endif # LLAMA_CUBLAS

ifdef LLAMA_HIPBLAS
ifeq ($(wildcard /opt/rocm),)
	ROCM_PATH   ?= /usr
ifdef LLAMA_PORTABLE
	GPU_TARGETS ?= gfx803 gfx900 gfx906 gfx908 gfx90a gfx942 gfx1010 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201 $(shell $(shell which amdgpu-arch))
else
	GPU_TARGETS ?= $(shell $(shell which amdgpu-arch))
endif
	HCC         := $(ROCM_PATH)/bin/hipcc
	HCXX        := $(ROCM_PATH)/bin/hipcc
else
	ROCM_PATH   ?= /opt/rocm
ifdef LLAMA_PORTABLE
	GPU_TARGETS ?= gfx803 gfx900 gfx906 gfx908 gfx90a gfx942 gfx1010 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201 $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
else
	GPU_TARGETS ?= $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
endif
	HCC         := $(ROCM_PATH)/llvm/bin/clang
	HCXX        := $(ROCM_PATH)/llvm/bin/clang++
endif
ifdef LLAMA_ALLOW_WMMA
	DETECT_ROCWMMA := $(shell find -L /opt/rocm/include /usr/include -type f -name rocwmma.hpp 2>/dev/null | head -n 1)
ifdef DETECT_ROCWMMA
	HIPFLAGS   += -DGGML_HIP_ROCWMMA_FATTN -I$(dir $(DETECT_ROCWMMA))
else
	HIPFLAGS   += -DGGML_HIP_NO_ROCWMMA_FATTN
endif
else
	HIPFLAGS   += -DGGML_HIP_NO_ROCWMMA_FATTN
endif

	HIPFLAGS   += -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DGGML_USE_CUDA -DSD_USE_CUDA $(shell $(ROCM_PATH)/bin/hipconfig -C)
	HIPLDFLAGS    += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib
	HIPLDFLAGS    += -L$(ROCM_PATH)/lib64 -Wl,-rpath=$(ROCM_PATH)/lib64
	HIPLDFLAGS    += -lhipblas -lamdhip64 -lrocblas
	HIP_OBJS      += ggml-cuda.o ggml_v3-cuda.o ggml_v2-cuda.o ggml_v2-cuda-legacy.o
	HIP_OBJS      += $(patsubst %.cu,%.o,$(filter-out ggml/src/ggml-cuda/ggml-cuda.cu, $(wildcard ggml/src/ggml-cuda/*.cu)))
	HIP_OBJS      += $(OBJS_CUDA_TEMP_INST)

	HIPFLAGS2    += $(addprefix --offload-arch=,$(GPU_TARGETS))

ggml/src/ggml-cuda/%.o: ggml/src/ggml-cuda/%.cu ggml/include/ggml.h ggml/src/ggml-common.h ggml/src/ggml-cuda/common.cuh
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml-cuda.o: ggml/src/ggml-cuda/ggml-cuda.cu ggml/include/ggml-cuda.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/src/ggml-backend-impl.h ggml/src/ggml-common.h $(wildcard ggml/src/ggml-cuda/*.cuh)
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml_v2-cuda.o: otherarch/ggml_v2-cuda.cu otherarch/ggml_v2-cuda.h
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml_v2-cuda-legacy.o: otherarch/ggml_v2-cuda-legacy.cu otherarch/ggml_v2-cuda-legacy.h
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml_v3-cuda.o: otherarch/ggml_v3-cuda.cu otherarch/ggml_v3-cuda.h
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
endif # LLAMA_HIPBLAS


ifdef LLAMA_METAL
	CFLAGS   += -DGGML_USE_METAL -DGGML_METAL_NDEBUG -DSD_USE_METAL
	CXXFLAGS += -DGGML_USE_METAL -DSD_USE_METAL
	LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
	OBJS     += ggml-metal.o

ggml-metal.o: ggml/src/ggml-metal/ggml-metal.m ggml/src/ggml-metal/ggml-metal-impl.h ggml/include/ggml-metal.h
	@echo "== Preparing merged Metal file =="
	@sed -e '/#include "ggml-common.h"/r ggml/src/ggml-common.h' -e '/#include "ggml-common.h"/d' < ggml/src/ggml-metal/ggml-metal.metal > ggml/src/ggml-metal/ggml-metal-embed.metal.tmp
	@sed -e '/#include "ggml-metal-impl.h"/r ggml/src/ggml-metal/ggml-metal-impl.h' -e '/#include "ggml-metal-impl.h"/d' < ggml/src/ggml-metal/ggml-metal-embed.metal.tmp > ggml/src/ggml-metal/ggml-metal-merged.metal
	@cp ggml/src/ggml-metal/ggml-metal-merged.metal ./ggml-metal-merged.metal
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_METAL

ifneq ($(filter aarch64%,$(UNAME_M)),)
# Apple M1, M2, etc.
# Raspberry Pi 3, 4, Zero 2 (64-bit)
ifdef LLAMA_PORTABLE
	CFLAGS +=
	CXXFLAGS +=
else
# sve is cooked on termux so we are disabling it
ifeq ($(UNAME_O), Android)
ifneq ($(findstring clang, $(CCV)), )
	CFLAGS += -mcpu=native+nosve
	CXXFLAGS += -mcpu=native+nosve
else
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif
else
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif
endif
endif

ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, Zero
	CFLAGS 	 += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
	CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 2
	CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
	CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 3, 4, Zero 2 (32-bit)
	CFLAGS   += -mfp16-format=ieee -mno-unaligned-access
	CXXFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
ifneq (,$(findstring POWER9,$(POWER9_M)))
	CFLAGS   += -mcpu=power9
	CXXFLAGS += -mcpu=power9
endif
endif


DEFAULT_BUILD =
FAILSAFE_BUILD =
NOAVX2_BUILD =
CLBLAST_BUILD =
CUBLAS_BUILD =
HIPBLAS_BUILD =
VULKAN_BUILD =
NOTIFY_MSG =

ifeq ($(OS),Windows_NT)
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.dll $(LDFLAGS)
ifdef LLAMA_PORTABLE
	FAILSAFE_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.dll $(LDFLAGS)
	NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.dll $(LDFLAGS)
endif

ifdef LLAMA_CLBLAST
	CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ lib/OpenCL.lib lib/clblast.lib -shared -o $@.dll $(LDFLAGS)
endif
ifdef LLAMA_VULKAN
	VULKAN_BUILD = $(CXX) $(CXXFLAGS) $^ lib/vulkan-1.lib -shared -o $@.dll $(LDFLAGS)
endif

ifdef LLAMA_CUBLAS
	CUBLAS_BUILD = $(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $^ -shared -o $@.dll $(CUBLASLD_FLAGS) $(LDFLAGS)
endif
ifdef LLAMA_HIPBLAS
	HIPBLAS_BUILD = $(HCXX) $(CXXFLAGS) $(HIPFLAGS) $^ -shared -o $@.dll $(HIPLDFLAGS) $(LDFLAGS)
endif
else
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
ifdef LLAMA_PORTABLE
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	FAILSAFE_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
	NOAVX2_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
endif
endif

ifdef LLAMA_CLBLAST
ifeq ($(UNAME_S),Darwin)
	CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ -lclblast -framework OpenCL $(ARCH_ADD) -shared -o $@.so $(LDFLAGS)
else
	CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ -lclblast -lOpenCL $(ARCH_ADD) -shared -o $@.so $(LDFLAGS)
endif
endif
ifdef LLAMA_CUBLAS
	CUBLAS_BUILD = $(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $^ -shared -o $@.so $(CUBLASLD_FLAGS) $(LDFLAGS)
endif
ifdef LLAMA_HIPBLAS
	HIPBLAS_BUILD = $(HCXX) $(CXXFLAGS) $(HIPFLAGS) $^ -shared -o $@.so $(HIPLDFLAGS) $(LDFLAGS)
endif
ifdef LLAMA_VULKAN
	VULKAN_BUILD = $(CXX) $(CXXFLAGS) $^ -lvulkan -shared -o $@.so $(LDFLAGS)
endif
endif

ifndef LLAMA_CLBLAST
ifndef LLAMA_CUBLAS
ifndef LLAMA_HIPBLAS
ifndef LLAMA_VULKAN
ifndef LLAMA_METAL
NOTIFY_MSG = @echo -e '\n***\nYou did a basic CPU build. For faster speeds, consider installing and linking a GPU BLAS library. For example, set LLAMA_CLBLAST=1 LLAMA_VULKAN=1 to compile with Vulkan and CLBlast support. Add LLAMA_PORTABLE=1 to make a sharable build that other devices can use. Read the KoboldCpp Wiki for more information. This is just a reminder, not an error.\n***\n'
endif
endif
endif
endif
endif

ifdef NO_VULKAN_EXTENSIONS
	VKGEN_NOEXT_ADD = -DNO_VULKAN_EXTENSIONS
endif
VKGEN_NOEXT_FORCE = -DNO_VULKAN_EXTENSIONS

#
# Print build information
#

$(info I koboldcpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I UNAME_O:  $(UNAME_O))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

#
# Build library
#

ggml.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v4_failsafe.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(NONECFLAGS) -c $< -o $@
ggml_v4_noavx2.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml_v4_clblast.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v4_cublas.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
ggml_v4_clblast_noavx2.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v4_clblast_failsafe.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(SIMPLERCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v4_vulkan.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(VULKAN_FLAGS) -c $< -o $@
ggml_v4_vulkan_noavx2.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) $(VULKAN_FLAGS) -c $< -o $@

# cpu and clblast separated
ggml-cpu.o: ggml/src/ggml-cpu/ggml-cpu.c ggml/include/ggml-cpu.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml-cpu_v4_failsafe.o: ggml/src/ggml-cpu/ggml-cpu.c ggml/include/ggml-cpu.h
	$(CC)  $(FASTCFLAGS) $(NONECFLAGS) -c $< -o $@
ggml-cpu_v4_noavx2.o: ggml/src/ggml-cpu/ggml-cpu.c ggml/include/ggml-cpu.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml-cpu_v4_clblast.o: ggml/src/ggml-cpu/ggml-cpu.c ggml/include/ggml-cpu.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml-cpu_v4_clblast_noavx2.o: ggml/src/ggml-cpu/ggml-cpu.c ggml/include/ggml-cpu.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml-cpu_v4_clblast_failsafe.o: ggml/src/ggml-cpu/ggml-cpu.c ggml/include/ggml-cpu.h
	$(CC)  $(FASTCFLAGS) $(SIMPLERCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
# addon cpu files
ggml-binops.o: ggml/src/ggml-cpu/binary-ops.cpp ggml/src/ggml-cpu/binary-ops.h ggml/src/ggml-cpu/common.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
ggml-unops.o: ggml/src/ggml-cpu/unary-ops.cpp ggml/src/ggml-cpu/unary-ops.h ggml/src/ggml-cpu/common.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
ggml-ops.o: ggml/src/ggml-cpu/ops.cpp ggml/src/ggml-cpu/ops.h
	$(CXX) $(FASTCXXFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml-ops-noavx2.o: ggml/src/ggml-cpu/ops.cpp ggml/src/ggml-cpu/ops.h
	$(CXX) $(FASTCXXFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml-ops-failsafe.o: ggml/src/ggml-cpu/ops.cpp ggml/src/ggml-cpu/ops.h
	$(CXX) $(FASTCXXFLAGS) $(NONECFLAGS) -c $< -o $@
ggml-vec.o: ggml/src/ggml-cpu/vec.cpp ggml/src/ggml-cpu/vec.h
	$(CXX) $(FASTCXXFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml-vec-noavx2.o: ggml/src/ggml-cpu/vec.cpp ggml/src/ggml-cpu/vec.h
	$(CXX) $(FASTCXXFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml-vec-failsafe.o: ggml/src/ggml-cpu/vec.cpp ggml/src/ggml-cpu/vec.h
	$(CXX) $(FASTCXXFLAGS) $(NONECFLAGS) -c $< -o $@

#quants
ggml-quants.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml-quants_noavx2.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml-quants_noavx1.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(SIMPLERCFLAGS) -c $< -o $@
ggml-quants_failsafe.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(NONECFLAGS) -c $< -o $@
ggml-cpu-quants.o: ggml/src/ggml-cpu/ggml-cpu-quants.c ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml-cpu-quants_noavx2.o: ggml/src/ggml-cpu/ggml-cpu-quants.c ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml-cpu-quants_noavx1.o: ggml/src/ggml-cpu/ggml-cpu-quants.c ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(SIMPLERCFLAGS) -c $< -o $@
ggml-cpu-quants_failsafe.o: ggml/src/ggml-cpu/ggml-cpu-quants.c ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(NONECFLAGS) -c $< -o $@

#aarch64
ggml-cpu-aarch64.o: ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-aarch64.h
	$(CXX) $(CXXFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml-cpu-aarch64_noavx2.o: ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-aarch64.h
	$(CXX) $(CXXFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml-cpu-aarch64_noavx1.o: ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-aarch64.h
	$(CXX) $(CXXFLAGS) $(SIMPLERCFLAGS) -c $< -o $@
ggml-cpu-aarch64_failsafe.o: ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ggml/include/ggml.h ggml/src/ggml-cpu/ggml-cpu-aarch64.h
	$(CXX) $(CXXFLAGS) $(NONECFLAGS) -c $< -o $@

#sgemm
sgemm.o: ggml/src/ggml-cpu/llamafile/sgemm.cpp ggml/src/ggml-cpu/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(FULLCFLAGS) -c $< -o $@
sgemm_noavx2.o: ggml/src/ggml-cpu/llamafile/sgemm.cpp ggml/src/ggml-cpu/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(SIMPLECFLAGS) -c $< -o $@
sgemm_noavx1.o: ggml/src/ggml-cpu/llamafile/sgemm.cpp ggml/src/ggml-cpu/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(SIMPLERCFLAGS) -c $< -o $@
sgemm_failsafe.o: ggml/src/ggml-cpu/llamafile/sgemm.cpp ggml/src/ggml-cpu/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(NONECFLAGS) -c $< -o $@

#there's no intrinsics or special gpu ops used here, so we can have a universal object
ggml-alloc.o: ggml/src/ggml-alloc.c ggml/include/ggml.h ggml/include/ggml-alloc.h
	$(CC)  $(CFLAGS) -c $< -o $@
llava.o: tools/mtmd/llava.cpp tools/mtmd/llava.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
unicode.o: src/unicode.cpp src/unicode.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
unicode-data.o: src/unicode-data.cpp src/unicode-data.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
ggml-cpu-traits.o: ggml/src/ggml-cpu/ggml-cpu-traits.cpp ggml/src/ggml-cpu/ggml-cpu-traits.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
ggml-threading.o: ggml/src/ggml-threading.cpp ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
ggml-cpu-cpp.o: ggml/src/ggml-cpu/ggml-cpu.cpp ggml/include/ggml.h ggml/src/ggml-common.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
gguf.o: ggml/src/gguf.cpp ggml/include/gguf.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
kcpputils.o: otherarch/utils.cpp otherarch/utils.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

#these have special gpu defines
ggml-backend_default.o: ggml/src/ggml-backend.cpp ggml/src/ggml-backend-impl.h ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CXX)  $(CXXFLAGS) -c $< -o $@
ggml-backend_vulkan.o: ggml/src/ggml-backend.cpp ggml/src/ggml-backend-impl.h ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CXX)  $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@
ggml-backend_cublas.o: ggml/src/ggml-backend.cpp ggml/src/ggml-backend-impl.h ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CXX)  $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
ggml-backend-reg_default.o: ggml/src/ggml-backend-reg.cpp ggml/src/ggml-backend-impl.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/include/ggml-cpu.h
	$(CXX)  $(CXXFLAGS) -c $< -o $@
ggml-backend-reg_vulkan.o: ggml/src/ggml-backend-reg.cpp ggml/src/ggml-backend-impl.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/include/ggml-cpu.h
	$(CXX)  $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@
ggml-backend-reg_cublas.o: ggml/src/ggml-backend-reg.cpp ggml/src/ggml-backend-impl.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/include/ggml-cpu.h
	$(CXX)  $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
llavaclip_default.o: tools/mtmd/clip.cpp tools/mtmd/clip.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
llavaclip_cublas.o: tools/mtmd/clip.cpp tools/mtmd/clip.h
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
llavaclip_vulkan.o: tools/mtmd/clip.cpp tools/mtmd/clip.h
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@

#this is only used for accelerate
ggml-blas.o: ggml/src/ggml-blas/ggml-blas.cpp ggml/include/ggml-blas.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

#version 3 libs
ggml_v3.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v3_failsafe.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(NONECFLAGS) -c $< -o $@
ggml_v3_noavx2.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml_v3_clblast.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v3_cublas.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
ggml_v3_clblast_noavx2.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v3_clblast_failsafe.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(SIMPLERCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@

#version 2 libs
ggml_v2.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v2_failsafe.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(NONECFLAGS) -c $< -o $@
ggml_v2_noavx2.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml_v2_clblast.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2_cublas.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
ggml_v2_clblast_noavx2.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2_clblast_failsafe.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(SIMPLERCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@

#extreme old version compat
ggml_v1.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v1_failsafe.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(FASTCFLAGS) $(NONECFLAGS) -c $< -o $@

#opencl
ggml-opencl.o: otherarch/ggml_v3b-opencl.cpp otherarch/ggml_v3b-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2-opencl.o: otherarch/ggml_v2-opencl.cpp otherarch/ggml_v2-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2-opencl-legacy.o: otherarch/ggml_v2-opencl-legacy.c otherarch/ggml_v2-opencl-legacy.h
	$(CC) $(CFLAGS) -c $< -o $@
ggml_v3-opencl.o: otherarch/ggml_v3-opencl.cpp otherarch/ggml_v3-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@

#vulkan
ggml-vulkan.o: ggml/src/ggml-vulkan/ggml-vulkan.cpp ggml/include/ggml-vulkan.h ggml/src/ggml-vulkan-shaders.cpp
	$(CXX) $(CXXFLAGS) $(VKGEN_NOEXT_ADD) $(VULKAN_FLAGS) -c $< -o $@
ggml-vulkan-noext.o: ggml/src/ggml-vulkan/ggml-vulkan.cpp ggml/include/ggml-vulkan.h ggml/src/ggml-vulkan-shaders-noext.cpp
	$(CXX) $(CXXFLAGS) $(VKGEN_NOEXT_FORCE) $(VULKAN_FLAGS) -c $< -o $@

# intermediate objects
llama.o: src/llama.cpp ggml/include/ggml.h ggml/include/ggml-alloc.h ggml/include/ggml-backend.h ggml/include/ggml-cuda.h ggml/include/ggml-metal.h include/llama.h otherarch/llama-util.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
common.o: common/common.cpp common/common.h common/log.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
sampling.o: common/sampling.cpp common/common.h common/sampling.h common/log.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
console.o: common/console.cpp common/console.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
expose.o: expose.cpp expose.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# sd.cpp objects
sdcpp_default.o: otherarch/sdcpp/sdtype_adapter.cpp otherarch/sdcpp/stable-diffusion.h otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/util.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c
	$(CXX) $(CXXFLAGS) -c $< -o $@
sdcpp_cublas.o: otherarch/sdcpp/sdtype_adapter.cpp otherarch/sdcpp/stable-diffusion.h otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/util.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
sdcpp_vulkan.o: otherarch/sdcpp/sdtype_adapter.cpp otherarch/sdcpp/stable-diffusion.h otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/util.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@


#whisper objects
whispercpp_default.o: otherarch/whispercpp/whisper_adapter.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
whispercpp_cublas.o: otherarch/whispercpp/whisper_adapter.cpp
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@

#tts objects
tts_default.o: otherarch/tts_adapter.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

embeddings_default.o: otherarch/embeddings_adapter.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# idiotic "for easier compilation"
GPTTYPE_ADAPTER = gpttype_adapter.cpp otherarch/llama_v2.cpp otherarch/llama_v3.cpp src/llama.cpp src/llama-impl.cpp src/llama-chat.cpp src/llama-mmap.cpp src/llama-context.cpp src/llama-adapter.cpp src/llama-arch.cpp src/llama-batch.cpp src/llama-vocab.cpp src/llama-grammar.cpp src/llama-sampling.cpp src/llama-kv-cache-unified.cpp src/llama-kv-cache-unified-iswa.cpp src/llama-kv-cache-recurrent.cpp src/llama-model-loader.cpp src/llama-model.cpp src/llama-quant.cpp src/llama-hparams.cpp otherarch/gptj_v1.cpp otherarch/gptj_v2.cpp otherarch/gptj_v3.cpp otherarch/gpt2_v1.cpp otherarch/gpt2_v2.cpp otherarch/gpt2_v3.cpp otherarch/rwkv_v2.cpp otherarch/rwkv_v3.cpp otherarch/neox_v2.cpp otherarch/neox_v3.cpp otherarch/mpt_v3.cpp ggml/include/ggml.h ggml/include/ggml-cpu.h ggml/include/ggml-cuda.h include/llama.h otherarch/llama-util.h
gpttype_adapter_failsafe.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(FAILSAFE_FLAGS) -c $< -o $@
gpttype_adapter.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) -c $< -o $@
gpttype_adapter_clblast.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
gpttype_adapter_cublas.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $(HIPFLAGS) -c $< -o $@
gpttype_adapter_clblast_noavx2.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(FAILSAFE_FLAGS) $(CLBLAST_FLAGS) -c $< -o $@
gpttype_adapter_vulkan.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@
gpttype_adapter_vulkan_noavx2.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(FAILSAFE_FLAGS) $(VULKAN_FLAGS) -c $< -o $@

clean:
	rm -vf *.o main ttsmain sdmain whispermain quantize_gguf quantize_clip quantize_gpt2 quantize_gptj quantize_neox quantize_mpt vulkan-shaders-gen vulkan-shaders-gen-noext gguf-split mtmd-cli mainvk mainvk.exe mtmd-cli.exe gguf-split.exe vulkan-shaders-gen.exe vulkan-shaders-gen-noext.exe main.exe ttsmain.exe sdmain.exe whispermain.exe quantize_clip.exe quantize_gguf.exe quantize_gptj.exe quantize_gpt2.exe quantize_neox.exe quantize_mpt.exe koboldcpp_default.dll koboldcpp_failsafe.dll koboldcpp_noavx2.dll koboldcpp_clblast.dll koboldcpp_clblast_noavx2.dll koboldcpp_clblast_failsafe.dll koboldcpp_cublas.dll koboldcpp_hipblas.dll koboldcpp_vulkan.dll koboldcpp_vulkan_noavx2.dll koboldcpp_default.so koboldcpp_failsafe.so koboldcpp_noavx2.so koboldcpp_clblast.so koboldcpp_clblast_noavx2.so koboldcpp_clblast_failsafe.so koboldcpp_cublas.so koboldcpp_hipblas.so koboldcpp_vulkan.so koboldcpp_vulkan_noavx2.so ggml/src/ggml-vulkan-shaders.cpp ggml/src/ggml-vulkan-shaders.hpp ggml/src/ggml-vulkan-shaders-noext.cpp ggml/src/ggml-vulkan-shaders-noext.hpp
	rm -vrf ggml/src/ggml-cuda/*.o
	rm -vrf ggml/src/ggml-cuda/template-instances/*.o

# useful tools
main: tools/main/main.cpp common/arg.cpp build-info.h ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
sdmain: otherarch/sdcpp/util.cpp otherarch/sdcpp/main.cpp otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c build-info.h ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o console.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
whispermain: otherarch/whispercpp/main.cpp otherarch/whispercpp/whisper.cpp build-info.h ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o console.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
ttsmain: tools/tts/tts.cpp common/arg.cpp build-info.h ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
gguf-split: tools/gguf-split/gguf-split.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o build-info.h llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
mtmd-cli: tools/mtmd/mtmd-cli.cpp tools/mtmd/mtmd.cpp common/arg.cpp build-info.h ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
mainvk: tools/main/main.cpp common/arg.cpp build-info.h ggml_v4_vulkan.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o console.o llavaclip_vulkan.o llava.o ggml-backend_vulkan.o ggml-backend-reg_vulkan.o ggml-vulkan.o $(OBJS_FULL) $(OBJS) lib/vulkan-1.lib
	$(CXX) $(CXXFLAGS) -DGGML_USE_VULKAN -DSD_USE_VULKAN $(filter-out %.h,$^) -o $@ $(LDFLAGS)
embedding: examples/embedding/embedding.cpp common/arg.cpp build-info.h ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

ggml/src/ggml-vulkan-shaders.cpp:
ifdef VULKAN_BUILD
	@$(MAKE) vulkan-shaders-gen
endif
ggml/src/ggml-vulkan-shaders-noext.cpp:
ifdef VULKAN_BUILD
	@$(MAKE) vulkan-shaders-gen-noext
endif

vulkan-shaders-gen: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
	@echo 'Vulkan shaders need to be regenerated. This can only be done on Windows or Linux. Please stand by...'
	$(CXX) $(CXXFLAGS) $(VKGEN_NOEXT_ADD) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
ifeq ($(OS),Windows_NT)
	@echo 'Now rebuilding vulkan shaders for Windows...'
	$(shell) vulkan-shaders-gen --glslc glslc --input-dir ggml/src/ggml-vulkan/vulkan-shaders --target-hpp ggml/src/ggml-vulkan-shaders.hpp --target-cpp ggml/src/ggml-vulkan-shaders.cpp --output-dir vulkan-spv-tmp
	@echo 'Vulkan Shaders Rebuilt for Windows...'
else
	@echo 'Now rebuilding vulkan shaders for Linux...'
	@chmod +x vulkan-shaders-gen
	@echo 'Checking if system glslc-linux binary is usable...'
	@GLSLC_BIN=$$( \
		if [ -n "$$LLAMA_USE_BUNDLED_GLSLC" ]; then \
			chmod +x ./glslc-linux; \
			if [ -x ./glslc-linux ] && ./glslc-linux --version 2>/dev/null | grep -q "glslang"; then \
				echo "./glslc-linux"; \
			elif command -v glslc >/dev/null 2>&1; then \
				echo "glslc"; \
			else \
				echo ""; \
			fi; \
		else \
			if command -v glslc >/dev/null 2>&1 && glslc --version 2>/dev/null | grep -q "glslang"; then \
				echo "glslc"; \
			elif [ -x ./glslc-linux ]; then \
				chmod +x ./glslc-linux; \
				if ./glslc-linux --version 2>/dev/null | grep -q "glslang"; then \
					echo "./glslc-linux"; \
				else \
					echo ""; \
				fi; \
			else \
				echo ""; \
			fi; \
		fi); \
	if [ -z "$$GLSLC_BIN" ]; then \
		echo "Error: No usable glslc found. Vulkan shaders cannot be compiled!"; \
	else \
		echo "Using GLSLC: $$GLSLC_BIN"; \
		./vulkan-shaders-gen --glslc "$$GLSLC_BIN" --input-dir ggml/src/ggml-vulkan/vulkan-shaders --target-hpp ggml/src/ggml-vulkan-shaders.hpp --target-cpp ggml/src/ggml-vulkan-shaders.cpp --output-dir vulkan-spv-tmp; \
	fi
	@echo 'Vulkan Shaders Rebuilt for Linux...'
endif

vulkan-shaders-gen-noext: ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
	@echo 'Vulkan shaders need to be regenerated (no extensions). This can only be done on Windows or Linux. Please stand by...'
	$(CXX) $(CXXFLAGS) $(VKGEN_NOEXT_FORCE) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
ifeq ($(OS),Windows_NT)
	@echo 'Now rebuilding vulkan shaders (no extensions) for Windows...'
	$(shell) vulkan-shaders-gen-noext --glslc glslc --input-dir ggml/src/ggml-vulkan/vulkan-shaders --target-hpp ggml/src/ggml-vulkan-shaders-noext.hpp --target-cpp ggml/src/ggml-vulkan-shaders-noext.cpp --output-dir vulkan-spv-noext-tmp
	@echo 'Vulkan Shaders (no extensions) Rebuilt for Windows...'
else
	@echo 'Now rebuilding vulkan shaders (no extensions) for Linux...'
	@chmod +x vulkan-shaders-gen-noext
	@echo 'Checking if system glslc-linux binary is usable...'
	@GLSLC_BIN=$$( \
		if [ -n "$$LLAMA_USE_BUNDLED_GLSLC" ]; then \
			chmod +x ./glslc-linux; \
			if [ -x ./glslc-linux ] && ./glslc-linux --version 2>/dev/null | grep -q "glslang"; then \
				echo "./glslc-linux"; \
			elif command -v glslc >/dev/null 2>&1; then \
				echo "glslc"; \
			else \
				echo ""; \
			fi; \
		else \
			if command -v glslc >/dev/null 2>&1 && glslc --version 2>/dev/null | grep -q "glslang"; then \
				echo "glslc"; \
			elif [ -x ./glslc-linux ]; then \
				chmod +x ./glslc-linux; \
				if ./glslc-linux --version 2>/dev/null | grep -q "glslang"; then \
					echo "./glslc-linux"; \
				else \
					echo ""; \
				fi; \
			else \
				echo ""; \
			fi; \
		fi); \
	if [ -z "$$GLSLC_BIN" ]; then \
		echo "Error: No usable glslc found. Vulkan shaders (no extensions) cannot be compiled!"; \
	else \
		echo "Using GLSLC: $$GLSLC_BIN"; \
		./vulkan-shaders-gen-noext --glslc "$$GLSLC_BIN" --input-dir ggml/src/ggml-vulkan/vulkan-shaders --target-hpp ggml/src/ggml-vulkan-shaders-noext.hpp --target-cpp ggml/src/ggml-vulkan-shaders-noext.cpp --output-dir vulkan-spv-noext-tmp; \
	fi
	@echo 'Vulkan Shaders (no extensions) Rebuilt for Linux...'
endif

#generated libraries
koboldcpp_default: ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter.o sdcpp_default.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(DEFAULT_BUILD)

ifdef FAILSAFE_BUILD
koboldcpp_failsafe: ggml_v4_failsafe.o ggml-cpu_v4_failsafe.o ggml-ops-failsafe.o ggml-vec-failsafe.o ggml-binops.o ggml-unops.o ggml_v3_failsafe.o ggml_v2_failsafe.o ggml_v1_failsafe.o expose.o gpttype_adapter_failsafe.o sdcpp_default.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FAILSAFE) $(OBJS)
	$(FAILSAFE_BUILD)
else
koboldcpp_failsafe:
	$(DONOTHING)
endif

ifdef NOAVX2_BUILD
koboldcpp_noavx2: ggml_v4_noavx2.o ggml-cpu_v4_noavx2.o ggml-ops-noavx2.o ggml-vec-noavx2.o ggml-binops.o ggml-unops.o ggml_v3_noavx2.o ggml_v2_noavx2.o ggml_v1_failsafe.o expose.o gpttype_adapter_failsafe.o sdcpp_default.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_SIMPLE) $(OBJS)
	$(NOAVX2_BUILD)
else
koboldcpp_noavx2:
	$(DONOTHING)
endif

ifdef CLBLAST_BUILD
koboldcpp_clblast: ggml_v4_clblast.o ggml-cpu_v4_clblast.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3_clblast.o ggml_v2_clblast.o ggml_v1.o expose.o gpttype_adapter_clblast.o ggml-opencl.o ggml_v3-opencl.o ggml_v2-opencl.o ggml_v2-opencl-legacy.o sdcpp_default.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL) $(OBJS)
	$(CLBLAST_BUILD)
ifdef NOAVX2_BUILD
koboldcpp_clblast_noavx2: ggml_v4_clblast_noavx2.o ggml-cpu_v4_clblast_noavx2.o ggml-ops-noavx2.o ggml-vec-noavx2.o ggml-binops.o ggml-unops.o ggml_v3_clblast_noavx2.o ggml_v2_clblast_noavx2.o ggml_v1_failsafe.o expose.o gpttype_adapter_clblast_noavx2.o ggml-opencl.o ggml_v3-opencl.o ggml_v2-opencl.o ggml_v2-opencl-legacy.o sdcpp_default.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_SIMPLE) $(OBJS)
	$(CLBLAST_BUILD)
koboldcpp_clblast_failsafe: ggml_v4_clblast_failsafe.o ggml-cpu_v4_clblast_failsafe.o ggml-ops-failsafe.o ggml-vec-failsafe.o ggml-binops.o ggml-unops.o ggml_v3_clblast_failsafe.o ggml_v2_clblast_failsafe.o ggml_v1_failsafe.o expose.o gpttype_adapter_clblast_noavx2.o ggml-opencl.o ggml_v3-opencl.o ggml_v2-opencl.o ggml_v2-opencl-legacy.o sdcpp_default.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_SIMPLER) $(OBJS)
	$(CLBLAST_BUILD)
else
koboldcpp_clblast_noavx2:
	$(DONOTHING)
koboldcpp_clblast_failsafe:
	$(DONOTHING)
endif
else
koboldcpp_clblast:
	$(DONOTHING)
koboldcpp_clblast_noavx2:
	$(DONOTHING)
koboldcpp_clblast_failsafe:
	$(DONOTHING)
endif

ifdef CUBLAS_BUILD
koboldcpp_cublas: ggml_v4_cublas.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o gpttype_adapter_cublas.o sdcpp_cublas.o whispercpp_cublas.o tts_default.o embeddings_default.o llavaclip_cublas.o llava.o ggml-backend_cublas.o ggml-backend-reg_cublas.o $(CUBLAS_OBJS) $(OBJS_FULL) $(OBJS)
	$(CUBLAS_BUILD)
else
koboldcpp_cublas:
	$(DONOTHING)
endif

ifdef HIPBLAS_BUILD
koboldcpp_hipblas: ggml_v4_cublas.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o gpttype_adapter_cublas.o sdcpp_cublas.o whispercpp_cublas.o tts_default.o embeddings_default.o llavaclip_cublas.o llava.o ggml-backend_cublas.o ggml-backend-reg_cublas.o $(HIP_OBJS) $(OBJS_FULL) $(OBJS)
	$(HIPBLAS_BUILD)
else
koboldcpp_hipblas:
	$(DONOTHING)
endif

ifdef VULKAN_BUILD
koboldcpp_vulkan: ggml_v4_vulkan.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter_vulkan.o ggml-vulkan.o sdcpp_vulkan.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_vulkan.o llava.o ggml-backend_vulkan.o ggml-backend-reg_vulkan.o $(OBJS_FULL) $(OBJS)
	$(VULKAN_BUILD)
ifdef NOAVX2_BUILD
koboldcpp_vulkan_noavx2: ggml_v4_vulkan_noavx2.o ggml-cpu_v4_noavx2.o ggml-ops-noavx2.o ggml-vec-noavx2.o ggml-binops.o ggml-unops.o ggml_v3_noavx2.o ggml_v2_noavx2.o ggml_v1_failsafe.o expose.o gpttype_adapter_vulkan_noavx2.o ggml-vulkan-noext.o sdcpp_vulkan.o whispercpp_default.o tts_default.o embeddings_default.o llavaclip_vulkan.o llava.o ggml-backend_vulkan.o ggml-backend-reg_vulkan.o $(OBJS_SIMPLE) $(OBJS)
	$(VULKAN_BUILD)
else
koboldcpp_vulkan_noavx2:
	$(DONOTHING)
endif
else
koboldcpp_vulkan:
	$(DONOTHING)
koboldcpp_vulkan_noavx2:
	$(DONOTHING)
endif

# tools
quantize_gguf: tools/quantize/quantize.cpp ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gptj: otherarch/tools/gptj_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gpt2: otherarch/tools/gpt2_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_neox: otherarch/tools/neox_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_mpt: otherarch/tools/mpt_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o llavaclip_default.o llava.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_clip: tools/mtmd/clip.cpp tools/mtmd/clip.h tools/quantclip.cpp ggml_v3.o ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o llama.o ggml-backend_default.o ggml-backend-reg_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

#window simple clinfo
simpleclinfo: simpleclinfo.cpp
	$(CXX) $(CXXFLAGS) $^ lib/OpenCL.lib lib/clblast.lib -o $@ $(LDFLAGS)

build-info.h:
	$(DONOTHING)

#phony for printing messages
finishedmsg:
	$(NOTIFY_MSG)
	$(DONOTHING)
