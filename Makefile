# Add custom options to Makefile.local rather than editing this file.
-include $(abspath $(lastword ${MAKEFILE_LIST})).local

default: koboldcpp_default koboldcpp_failsafe koboldcpp_openblas koboldcpp_noavx2 koboldcpp_clblast koboldcpp_clblast_noavx2 koboldcpp_cublas koboldcpp_hipblas koboldcpp_vulkan koboldcpp_vulkan_noavx2
tools: quantize_gpt2 quantize_gptj quantize_gguf quantize_neox quantize_mpt quantize_clip whispermain sdmain gguf-split
dev: koboldcpp_openblas
dev2: koboldcpp_clblast
dev3: koboldcpp_vulkan

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
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

# keep standard at C11 and C++11
CFLAGS =
CXXFLAGS =
ifdef KCPP_DEBUG
	CFLAGS = -g -O0
	CXXFLAGS = -g -O0
endif
CFLAGS   += -I. -Iggml/include -Iggml/src -Iinclude -Isrc -I./include -I./include/CL -I./otherarch -I./otherarch/tools -I./otherarch/sdcpp -I./otherarch/sdcpp/thirdparty -I./include/vulkan -O3 -fno-finite-math-only -std=c11 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_LLAMAFILE
CXXFLAGS += -I. -Iggml/include -Iggml/src -Iinclude -Isrc -I./common -I./include -I./include/CL -I./otherarch -I./otherarch/tools -I./otherarch/sdcpp -I./otherarch/sdcpp/thirdparty -I./include/vulkan -O3 -fno-finite-math-only -std=c++11 -fPIC -DLOG_DISABLE_LOGS -D_GNU_SOURCE -DGGML_USE_LLAMAFILE
ifndef KCPP_DEBUG
	CFLAGS += -DNDEBUG -s
	CXXFLAGS += -DNDEBUG -s
endif

#lets try enabling everything
CFLAGS   += -pthread -Wno-deprecated -Wno-deprecated-declarations -Wno-unused-variable
CXXFLAGS += -pthread -Wno-multichar -Wno-write-strings -Wno-deprecated -Wno-deprecated-declarations -Wno-unused-variable

LDFLAGS  =
FASTCFLAGS = $(subst -O3,-Ofast,$(CFLAGS))
FASTCXXFLAGS = $(subst -O3,-Ofast,$(CXXFLAGS))

# these are used on windows, to build some libraries with extra old device compatibility
SIMPLECFLAGS =
FULLCFLAGS =
NONECFLAGS =

OPENBLAS_FLAGS = -DGGML_USE_OPENBLAS -DGGML_USE_BLAS -I/usr/local/include/openblas
CLBLAST_FLAGS = -DGGML_USE_CLBLAST
FAILSAFE_FLAGS = -DUSE_FAILSAFE
VULKAN_FLAGS = -DGGML_USE_VULKAN -DSD_USE_VULKAN
ifdef LLAMA_CUBLAS
	CUBLAS_FLAGS = -DGGML_USE_CUDA -DSD_USE_CUBLAS
else
	CUBLAS_FLAGS =
endif
CUBLASLD_FLAGS =
CUBLAS_OBJS =

OBJS_FULL += ggml-alloc.o ggml-aarch64.o ggml-quants.o unicode.o unicode-data.o sgemm.o common.o sampling.o grammar-parser.o
OBJS_SIMPLE += ggml-alloc.o ggml-aarch64.o ggml-quants_noavx2.o unicode.o unicode-data.o sgemm_noavx2.o common.o sampling.o grammar-parser.o
OBJS_FAILSAFE += ggml-alloc.o ggml-aarch64.o ggml-quants_failsafe.o unicode.o unicode-data.o sgemm_failsafe.o common.o sampling.o grammar-parser.o

# OS specific
# TODO: support Windows
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
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

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
# feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	# Use all CPU extensions that are available:
# old library NEEDS mf16c to work. so we must build with it. new one doesnt
	ifeq ($(OS),Windows_NT)
		CFLAGS +=
		NONECFLAGS +=
		SIMPLECFLAGS += -mavx -msse3
		ifdef LLAMA_NOAVX2
			FULLCFLAGS += -msse3 -mavx
		else
			FULLCFLAGS += -mavx2 -msse3 -mfma -mf16c -mavx
		endif
	else
# if not on windows, they are clearly building it themselves, so lets just use whatever is supported
		ifdef LLAMA_PORTABLE
		CFLAGS +=
		NONECFLAGS +=
		SIMPLECFLAGS += -mavx -msse3
		ifdef LLAMA_NOAVX2
			FULLCFLAGS += -msse3 -mavx
		else
			FULLCFLAGS += -mavx2 -msse3 -mfma -mf16c -mavx
		endif
		else
		CFLAGS += -march=native -mtune=native
		endif
	endif
endif

ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE -DGGML_USE_BLAS
		CXXFLAGS  += -DGGML_USE_ACCELERATE -DGGML_USE_BLAS
		LDFLAGS += -framework Accelerate
		OBJS += ggml-blas.o
	endif
endif

# it is recommended to use the CMAKE file to build for cublas if you can - will likely work better
OBJS_CUDA_TEMP_INST = $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-wmma*.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/mmq*.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu))
OBJS_CUDA_TEMP_INST += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/template-instances/fattn-vec*f16-f16.cu))

ifdef LLAMA_CUBLAS
	CUBLAS_FLAGS = -DGGML_USE_CUDA -DSD_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	CUBLASLD_FLAGS = -lcuda -lcublas -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib -L$(CUDA_PATH)/lib64/stubs -L/usr/local/cuda/targets/aarch64-linux/lib -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/lib/wsl/lib
	CUBLAS_OBJS = ggml-cuda.o ggml_v3-cuda.o ggml_v2-cuda.o ggml_v2-cuda-legacy.o
	CUBLAS_OBJS += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/*.cu))
	CUBLAS_OBJS += $(OBJS_CUDA_TEMP_INST)
	NVCC      = nvcc
	NVCCFLAGS = --forward-unknown-to-host-compiler -use_fast_math

ifdef LLAMA_ADD_CONDA_PATHS
	CUBLASLD_FLAGS += -Lconda/envs/linux/lib -Lconda/envs/linux/lib/stubs
endif

ifdef CUDA_DOCKER_ARCH
	NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=$(CUDA_DOCKER_ARCH)
else
ifdef LLAMA_PORTABLE
ifdef LLAMA_COLAB #colab does not need all targets, all-major doesnt work correctly with pascal
	NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=all-major
else
	NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=all
endif #LLAMA_COLAB
else
	NVCCFLAGS += -arch=native
endif #LLAMA_PORTABLE
endif # CUDA_DOCKER_ARCH

ifdef LLAMA_CUDA_FORCE_DMMV
	NVCCFLAGS += -DGGML_CUDA_FORCE_DMMV
endif # LLAMA_CUDA_FORCE_DMMV
ifdef LLAMA_CUDA_DMMV_X
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
else
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=32
endif # LLAMA_CUDA_DMMV_X
ifdef LLAMA_CUDA_MMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
else ifdef LLAMA_CUDA_DMMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_DMMV_Y) # for backwards compatibility
else
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=1
endif # LLAMA_CUDA_MMV_Y
ifdef LLAMA_CUDA_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_F16
ifdef LLAMA_CUDA_DMMV_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_DMMV_F16
ifdef LLAMA_CUDA_KQUANTS_ITER
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)
else
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=2
endif

ifdef LLAMA_CUDA_CCBIN
	NVCCFLAGS += -ccbin $(LLAMA_CUDA_CCBIN)
endif

ggml/src/ggml-cuda/%.o: ggml/src/ggml-cuda/%.cu ggml/include/ggml.h ggml/src/ggml-common.h ggml/src/ggml-cuda/common.cuh
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml-cuda.o: ggml/src/ggml-cuda.cu ggml/include/ggml-cuda.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/src/ggml-backend-impl.h ggml/src/ggml-common.h $(wildcard ggml/src/ggml-cuda/*.cuh)
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v2-cuda.o: otherarch/ggml_v2-cuda.cu otherarch/ggml_v2-cuda.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v2-cuda-legacy.o: otherarch/ggml_v2-cuda-legacy.cu otherarch/ggml_v2-cuda-legacy.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v3-cuda.o: otherarch/ggml_v3-cuda.cu otherarch/ggml_v3-cuda.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
endif # LLAMA_CUBLAS

ifdef LLAMA_HIPBLAS
	ifeq ($(wildcard /opt/rocm),)
		ROCM_PATH	?= /usr
		GPU_TARGETS ?= $(shell $(shell which amdgpu-arch))
		HCC         := $(ROCM_PATH)/bin/hipcc
		HCXX        := $(ROCM_PATH)/bin/hipcc
	else
		ROCM_PATH	?= /opt/rocm
		GPU_TARGETS ?= gfx803 gfx900 gfx906 gfx908 gfx90a gfx1030 gfx1100 $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
		HCC         := $(ROCM_PATH)/llvm/bin/clang
		HCXX        := $(ROCM_PATH)/llvm/bin/clang++
	endif
	LLAMA_CUDA_DMMV_X ?= 32
	LLAMA_CUDA_MMV_Y ?= 1
	LLAMA_CUDA_KQUANTS_ITER ?= 2
	HIPFLAGS   += -DGGML_USE_HIPBLAS -DGGML_USE_CUDA -DSD_USE_CUBLAS $(shell $(ROCM_PATH)/bin/hipconfig -C)
	HIPLDFLAGS    += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib -lhipblas -lamdhip64 -lrocblas
	HIP_OBJS      += ggml-cuda.o ggml_v3-cuda.o ggml_v2-cuda.o ggml_v2-cuda-legacy.o
	HIP_OBJS      += $(patsubst %.cu,%.o,$(wildcard ggml/src/ggml-cuda/*.cu))
	HIP_OBJS      += $(OBJS_CUDA_TEMP_INST)

	HIPFLAGS2    += $(addprefix --offload-arch=,$(GPU_TARGETS))
	HIPFLAGS2    += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
	HIPFLAGS2    += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
	HIPFLAGS2    += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)

ggml/src/ggml-cuda/%.o: ggml/src/ggml-cuda/%.cu ggml/include/ggml.h ggml/src/ggml-common.h ggml/src/ggml-cuda/common.cuh
	$(HCXX) $(CXXFLAGS) $(HIPFLAGS) $(HIPFLAGS2) -x hip -c -o $@ $<
ggml-cuda.o: ggml/src/ggml-cuda.cu ggml/include/ggml-cuda.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/src/ggml-backend-impl.h ggml/src/ggml-common.h $(wildcard ggml/src/ggml-cuda/*.cuh)
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

ggml-metal.o: ggml/src/ggml-metal.m ggml/include/ggml-metal.h
	@echo "== Preparing merged Metal file =="
	@sed -e '/#include "ggml-common.h"/r ggml/src/ggml-common.h' -e '/#include "ggml-common.h"/d' < ggml/src/ggml-metal.metal > ggml/src/ggml-metal-merged.metal
	@cp ggml/src/ggml-metal-merged.metal ./ggml-metal-merged.metal
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_METAL

ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	CFLAGS 	 +=
	CXXFLAGS +=
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
OPENBLAS_BUILD =
NOAVX2_BUILD =
CLBLAST_BUILD =
CUBLAS_BUILD =
HIPBLAS_BUILD =
VULKAN_BUILD =

ifeq ($(OS),Windows_NT)
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.dll $(LDFLAGS)
	FAILSAFE_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.dll $(LDFLAGS)
	OPENBLAS_BUILD = $(CXX) $(CXXFLAGS) $^ lib/libopenblas.lib -shared -o $@.dll $(LDFLAGS)
	NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.dll $(LDFLAGS)
	CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ lib/OpenCL.lib lib/clblast.lib -shared -o $@.dll $(LDFLAGS)
	VULKAN_BUILD = $(CXX) $(CXXFLAGS) $^ lib/vulkan-1.lib -shared -o $@.dll $(LDFLAGS)

	ifdef LLAMA_CUBLAS
		CUBLAS_BUILD = $(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $^ -shared -o $@.dll $(CUBLASLD_FLAGS) $(LDFLAGS)
	endif
	ifdef LLAMA_HIPBLAS
		HIPBLAS_BUILD = $(HCXX) $(CXXFLAGS) $(HIPFLAGS) $^ -shared -o $@.dll $(HIPLDFLAGS) $(LDFLAGS)
	endif
else
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
	ifdef LLAMA_PORTABLE
	FAILSAFE_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
	NOAVX2_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
	endif

	ifdef LLAMA_OPENBLAS
	OPENBLAS_BUILD = $(CXX) $(CXXFLAGS) $^ $(ARCH_ADD) -lopenblasp -shared -o $@.so $(LDFLAGS)
	endif
	ifdef LLAMA_CLBLAST
		ifeq ($(UNAME_S),Darwin)
			CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ -lclblast -framework OpenCL $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
		else
			CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ -lclblast -lOpenCL $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
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

	ifndef LLAMA_OPENBLAS
	ifndef LLAMA_CLBLAST
	ifndef LLAMA_CUBLAS
	ifndef LLAMA_HIPBLAS
	ifndef LLAMA_VULKAN
	OPENBLAS_BUILD = @echo 'Your OS $(OS) does not appear to be Windows. For faster speeds, install and link a BLAS library. Set LLAMA_OPENBLAS=1 to compile with OpenBLAS support or LLAMA_CLBLAST=1 to compile with ClBlast support. This is just a reminder, not an error.'
	endif
	endif
	endif
	endif
	endif
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
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
ggml_v4_openblas.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
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
ggml_v4_vulkan.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(VULKAN_FLAGS) -c $< -o $@
ggml_v4_vulkan_noavx2.o: ggml/src/ggml.c ggml/include/ggml.h
	$(CC)  $(FASTCFLAGS) $(SIMPLECFLAGS) $(VULKAN_FLAGS) -c $< -o $@

#quants
ggml-quants.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml-quants_noavx2.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(SIMPLECFLAGS) -c $< -o $@
ggml-quants_failsafe.o: ggml/src/ggml-quants.c ggml/include/ggml.h ggml/src/ggml-quants.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) $(NONECFLAGS) -c $< -o $@

#sgemm
sgemm.o: ggml/src/llamafile/sgemm.cpp ggml/src/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(FULLCFLAGS) -c $< -o $@
sgemm_noavx2.o: ggml/src/llamafile/sgemm.cpp ggml/src/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(SIMPLECFLAGS) -c $< -o $@
sgemm_failsafe.o: ggml/src/llamafile/sgemm.cpp ggml/src/llamafile/sgemm.h ggml/include/ggml.h
	$(CXX) $(CXXFLAGS) $(NONECFLAGS) -c $< -o $@

#there's no intrinsics or special gpu ops used here, so we can have a universal object
ggml-alloc.o: ggml/src/ggml-alloc.c ggml/include/ggml.h ggml/include/ggml-alloc.h
	$(CC)  $(CFLAGS) -c $< -o $@
llava.o: examples/llava/llava.cpp examples/llava/llava.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
unicode.o: src/unicode.cpp src/unicode.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
unicode-data.o: src/unicode-data.cpp src/unicode-data.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
ggml-aarch64.o: ggml/src/ggml-aarch64.c ggml/include/ggml.h ggml/src/ggml-aarch64.h ggml/src/ggml-common.h
	$(CC)  $(CFLAGS) -c $< -o $@

#these have special gpu defines
ggml-backend_default.o: ggml/src/ggml-backend.c ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CC)  $(CFLAGS) -c $< -o $@
ggml-backend_vulkan.o: ggml/src/ggml-backend.c ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CC)  $(CFLAGS) $(VULKAN_FLAGS) -c $< -o $@
ggml-backend_cublas.o: ggml/src/ggml-backend.c ggml/include/ggml.h ggml/include/ggml-backend.h
	$(CC)  $(CFLAGS) $(CUBLAS_FLAGS) -c $< -o $@
llavaclip_default.o: examples/llava/clip.cpp examples/llava/clip.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
llavaclip_cublas.o: examples/llava/clip.cpp examples/llava/clip.h
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) -c $< -o $@
llavaclip_vulkan.o: examples/llava/clip.cpp examples/llava/clip.h
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@

#this is only used for openblas and accelerate
ggml-blas.o: ggml/src/ggml-blas.cpp ggml/include/ggml-blas.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

#version 3 libs
ggml_v3.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v3_openblas.o: otherarch/ggml_v3.c otherarch/ggml_v3.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
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

#version 2 libs
ggml_v2.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v2_openblas.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
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

#extreme old version compat
ggml_v1.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(FASTCFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v1_failsafe.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(FASTCFLAGS) $(NONECFLAGS) -c $< -o $@

#opencl
ggml-opencl.o: ggml-opencl.cpp ggml-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2-opencl.o: otherarch/ggml_v2-opencl.cpp otherarch/ggml_v2-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2-opencl-legacy.o: otherarch/ggml_v2-opencl-legacy.c otherarch/ggml_v2-opencl-legacy.h
	$(CC) $(CFLAGS) -c $< -o $@
ggml_v3-opencl.o: otherarch/ggml_v3-opencl.cpp otherarch/ggml_v3-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@

#vulkan
ggml-vulkan.o: ggml/src/ggml-vulkan.cpp ggml/include/ggml-vulkan.h ggml/src/ggml-vulkan-shaders.hpp ggml/src/ggml-vulkan-shaders.cpp
	$(CXX) $(CXXFLAGS) $(VULKAN_FLAGS) -c $< -o $@

# intermediate objects
llama.o: src/llama.cpp ggml/include/ggml.h ggml/include/ggml-alloc.h ggml/include/ggml-backend.h ggml/include/ggml-cuda.h ggml/include/ggml-metal.h include/llama.h otherarch/llama-util.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
common.o: common/common.cpp common/common.h common/log.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
sampling.o: common/sampling.cpp common/common.h common/sampling.h common/log.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
console.o: common/console.cpp common/console.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
grammar-parser.o: common/grammar-parser.cpp common/grammar-parser.h
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

# idiotic "for easier compilation"
GPTTYPE_ADAPTER = gpttype_adapter.cpp otherarch/llama_v2.cpp otherarch/llama_v3.cpp src/llama.cpp src/llama-grammar.cpp src/llama-sampling.cpp src/llama-vocab.cpp otherarch/utils.cpp otherarch/gptj_v1.cpp otherarch/gptj_v2.cpp otherarch/gptj_v3.cpp otherarch/gpt2_v1.cpp otherarch/gpt2_v2.cpp otherarch/gpt2_v3.cpp otherarch/rwkv_v2.cpp otherarch/rwkv_v3.cpp otherarch/neox_v2.cpp otherarch/neox_v3.cpp otherarch/mpt_v3.cpp ggml/include/ggml.h ggml/include/ggml-cuda.h include/llama.h otherarch/llama-util.h
gpttype_adapter_failsafe.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(FAILSAFE_FLAGS) -c $< -o $@
gpttype_adapter.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) -c $< -o $@
gpttype_adapter_openblas.o: $(GPTTYPE_ADAPTER)
	$(CXX) $(CXXFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
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
	rm -vf *.o main sdmain whispermain quantize_gguf quantize_clip quantize_gpt2 quantize_gptj quantize_neox quantize_mpt quantize-stats perplexity embedding benchmark-matmult save-load-state gguf imatrix vulkan-shaders-gen gguf-split gguf-split.exe vulkan-shaders-gen.exe imatrix.exe gguf.exe main.exe sdmain.exe whispermain.exe quantize_clip.exe quantize_gguf.exe quantize_gptj.exe quantize_gpt2.exe quantize_neox.exe quantize_mpt.exe koboldcpp_default.dll koboldcpp_openblas.dll koboldcpp_failsafe.dll koboldcpp_noavx2.dll koboldcpp_clblast.dll koboldcpp_clblast_noavx2.dll koboldcpp_cublas.dll koboldcpp_hipblas.dll koboldcpp_vulkan.dll koboldcpp_vulkan_noavx2.dll koboldcpp_default.so koboldcpp_openblas.so koboldcpp_failsafe.so koboldcpp_noavx2.so koboldcpp_clblast.so koboldcpp_clblast_noavx2.so koboldcpp_cublas.so koboldcpp_hipblas.so koboldcpp_vulkan.so koboldcpp_vulkan_noavx2.so
	rm -vrf ggml/src/ggml-cuda/*.o
	rm -vrf ggml/src/ggml-cuda/template-instances/*.o

# useful tools
main: examples/main/main.cpp build-info.h ggml.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	@echo '====  Run ./main -h for help.  ===='
sdmain: otherarch/sdcpp/util.cpp otherarch/sdcpp/main.cpp otherarch/sdcpp/stable-diffusion.cpp otherarch/sdcpp/upscaler.cpp otherarch/sdcpp/model.cpp otherarch/sdcpp/thirdparty/zip.c build-info.h ggml.o llama.o console.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
whispermain: otherarch/whispercpp/main.cpp otherarch/whispercpp/whisper.cpp build-info.h ggml.o llama.o console.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
imatrix: examples/imatrix/imatrix.cpp build-info.h ggml.o llama.o console.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
gguf: examples/gguf/gguf.cpp build-info.h ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
gguf-split: examples/gguf-split/gguf-split.cpp ggml.o llama.o build-info.h llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

vulkan-shaders-gen: ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp
	@echo 'This command can be MANUALLY run to regenerate vulkan shaders. Normally concedo will do it, so you do not have to.'
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	@echo 'Now rebuilding vulkan shaders...'
	$(shell) vulkan-shaders-gen --glslc glslc --input-dir ggml/src/vulkan-shaders --target-hpp ggml/src/ggml-vulkan-shaders.hpp --target-cpp ggml/src/ggml-vulkan-shaders.cpp

#generated libraries
koboldcpp_default: ggml.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(DEFAULT_BUILD)

ifdef OPENBLAS_BUILD
koboldcpp_openblas: ggml_v4_openblas.o ggml_v3_openblas.o ggml_v2_openblas.o ggml_v1.o expose.o gpttype_adapter_openblas.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_default.o ggml-blas.o $(OBJS_FULL) $(OBJS)
	$(OPENBLAS_BUILD)
else
koboldcpp_openblas:
	$(DONOTHING)
endif

ifdef FAILSAFE_BUILD
koboldcpp_failsafe: ggml_v4_failsafe.o ggml_v3_failsafe.o ggml_v2_failsafe.o ggml_v1_failsafe.o expose.o gpttype_adapter_failsafe.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FAILSAFE) $(OBJS)
	$(FAILSAFE_BUILD)
else
koboldcpp_failsafe:
	$(DONOTHING)
endif

ifdef NOAVX2_BUILD
koboldcpp_noavx2: ggml_v4_noavx2.o ggml_v3_noavx2.o ggml_v2_noavx2.o ggml_v1_failsafe.o expose.o gpttype_adapter_failsafe.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_SIMPLE) $(OBJS)
	$(NOAVX2_BUILD)
else
koboldcpp_noavx2:
	$(DONOTHING)
endif

ifdef CLBLAST_BUILD
koboldcpp_clblast: ggml_v4_clblast.o ggml_v3_clblast.o ggml_v2_clblast.o ggml_v1.o expose.o gpttype_adapter_clblast.o ggml-opencl.o ggml_v3-opencl.o ggml_v2-opencl.o ggml_v2-opencl-legacy.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL) $(OBJS)
	$(CLBLAST_BUILD)
ifdef NOAVX2_BUILD
koboldcpp_clblast_noavx2: ggml_v4_clblast_noavx2.o ggml_v3_clblast_noavx2.o ggml_v2_clblast_noavx2.o ggml_v1_failsafe.o expose.o gpttype_adapter_clblast_noavx2.o ggml-opencl.o ggml_v3-opencl.o ggml_v2-opencl.o ggml_v2-opencl-legacy.o sdcpp_default.o whispercpp_default.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_SIMPLE) $(OBJS)
	$(CLBLAST_BUILD)
else
koboldcpp_clblast_noavx2:
	$(DONOTHING)
endif
else
koboldcpp_clblast:
	$(DONOTHING)
koboldcpp_clblast_noavx2:
	$(DONOTHING)
endif

ifdef CUBLAS_BUILD
koboldcpp_cublas: ggml_v4_cublas.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o gpttype_adapter_cublas.o sdcpp_cublas.o whispercpp_cublas.o llavaclip_cublas.o llava.o ggml-backend_cublas.o $(CUBLAS_OBJS) $(OBJS_FULL) $(OBJS)
	$(CUBLAS_BUILD)
else
koboldcpp_cublas:
	$(DONOTHING)
endif

ifdef HIPBLAS_BUILD
koboldcpp_hipblas: ggml_v4_cublas.o ggml_v3_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o gpttype_adapter_cublas.o sdcpp_cublas.o whispercpp_cublas.o llavaclip_cublas.o llava.o ggml-backend_cublas.o $(HIP_OBJS) $(OBJS_FULL) $(OBJS)
	$(HIPBLAS_BUILD)
else
koboldcpp_hipblas:
	$(DONOTHING)
endif

ifdef VULKAN_BUILD
koboldcpp_vulkan: ggml_v4_vulkan.o ggml_v3.o ggml_v2.o ggml_v1.o expose.o gpttype_adapter_vulkan.o ggml-vulkan.o sdcpp_vulkan.o whispercpp_default.o llavaclip_vulkan.o llava.o ggml-backend_vulkan.o $(OBJS_FULL) $(OBJS)
	$(VULKAN_BUILD)
ifdef NOAVX2_BUILD
koboldcpp_vulkan_noavx2: ggml_v4_vulkan_noavx2.o ggml_v3_noavx2.o ggml_v2_noavx2.o ggml_v1_failsafe.o expose.o gpttype_adapter_vulkan_noavx2.o ggml-vulkan.o sdcpp_vulkan.o whispercpp_default.o llavaclip_vulkan.o llava.o ggml-backend_vulkan.o $(OBJS_SIMPLE) $(OBJS)
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
quantize_gguf: examples/quantize/quantize.cpp ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gptj: otherarch/tools/gptj_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gpt2: otherarch/tools/gpt2_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_neox: otherarch/tools/neox_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_mpt: otherarch/tools/mpt_quantize.cpp otherarch/tools/common-ggml.cpp ggml_v3.o ggml.o llama.o llavaclip_default.o llava.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_clip: examples/llava/clip.cpp examples/llava/clip.h examples/llava/quantclip.cpp ggml_v3.o ggml.o llama.o ggml-backend_default.o $(OBJS_FULL)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

#window simple clinfo
simpleclinfo: simpleclinfo.cpp
	$(CXX) $(CXXFLAGS) $^ lib/OpenCL.lib lib/clblast.lib -o $@ $(LDFLAGS)

build-info.h:
	$(DONOTHING)
