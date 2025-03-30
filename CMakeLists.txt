# THIS FILE IS ONLY INTENDED CUBLAS BUILD PURPOSES ON WINDOWS VISUAL STUDIO.
# YOU'RE NOT RECOMMENDED TO USE IT

message(STATUS "============== ============== ==============")
message(STATUS "WARNING! Recommend NOT to use this file. It is UNSUPPORTED for normal users. Use MAKE instead.")
message(STATUS "It is ONLY for CUBLAS builds on windows visual studio. IT WILL OVERWRITE YOUR EXISTING MAKEFILE !!!")
message(STATUS "IF YOU ARE SEEING THIS, you MUST ONLY be building CUBLAS BUILDS! NOTHING ELSE WILL BE SUPPORTED !!!")
message(STATUS "============== ============== ==============")

cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("llama.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLAMA_STANDALONE ON)
set(BUILD_SHARED_LIBS_DEFAULT ON)
set(LLAMA_STATIC OFF)
set(LLAMA_NATIVE OFF)
set(LLAMA_LTO OFF)
set(LLAMA_ALL_WARNINGS OFF)
set(LLAMA_ALL_WARNINGS_3RD_PARTY OFF)
set(LLAMA_GPROF OFF)
set(LLAMA_SANITIZE_THREAD OFF)
set(LLAMA_SANITIZE_ADDRESS OFF)
set(LLAMA_SANITIZE_UNDEFINED OFF)


# instruction set specific
option(LLAMA_AVX                    "llama: enable AVX"                                     ON)
option(LLAMA_AVX2                   "llama: enable AVX2"                                    ON)
option(LLAMA_AVX512                 "llama: enable AVX512"                                  OFF)
option(LLAMA_AVX512_VBMI            "llama: enable AVX512-VBMI"                             OFF)
option(LLAMA_AVX512_VNNI            "llama: enable AVX512-VNNI"                             OFF)
option(LLAMA_FMA                    "llama: enable FMA"                                     ON)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(LLAMA_F16C               "llama: enable F16C"                                    ON)
endif()

# 3rd party libs
option(LLAMA_CUBLAS                          "llama: use CUDA"                                ON)
option(LLAMA_CUDA_F16                        "llama: use 16 bit floats for dmmv CUDA kernels"   OFF)
set(LLAMA_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                             "llama: max. batch size for using peer access")

option(LLAMA_HIPBLAS                         "llama: use hipBLAS"                               OFF)

# Other
option(LLAMA_OPENMP                          "llama: use OpenMP"                                OFF)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_compile_definitions(LOG_DISABLE_LOGS)
add_compile_definitions(GGML_USE_CPU)
add_compile_definitions(GGML_USE_CPU_AARCH64)

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

file(GLOB GGML_SOURCES_CUDA "ggml/src/ggml-cuda/*.cu")
list(APPEND GGML_SOURCES_CUDA "ggml/src/ggml-cuda/ggml-cuda.cu")
file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
set(GGML_V3_CUDA_SOURCES otherarch/ggml_v3-cuda.cu otherarch/ggml_v3-cuda.h)
set(GGML_V2_CUDA_SOURCES otherarch/ggml_v2-cuda.cu otherarch/ggml_v2-cuda.h)
set(GGML_V2_LEGACY_CUDA_SOURCES otherarch/ggml_v2-cuda-legacy.cu otherarch/ggml_v2-cuda-legacy.h)


if (LLAMA_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        add_compile_definitions(GGML_USE_LLAMAFILE)
        add_compile_definitions(GGML_USE_CUDA)
        add_compile_definitions(SD_USE_CUBLAS)

        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            add_compile_definitions(GGML_CUDA_F16)
        endif()
        add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${LLAMA_CUDA_PEER_MAX_BATCH_SIZE})

        # only build minimal quants required for fattn quant kv
        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})

        if (LLAMA_STATIC)
            if (WIN32)
                # As of 12.3.1 CUDA Tookit for Windows does not offer a static cublas library
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
            else ()
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            endif()
        else()
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cuda_driver)

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 50 == lowest CUDA 12 standard
        # 60 == f16 CUDA intrinsics
        # 61 == integer CUDA intrinsics
        # 70 == (assumed) compute capability at which unrolling a loop in mul_mat_q kernels is faster
        # 75 == int8 tensor cores
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75") # needed for f16 CUDA intrinsics
        else()
            message("CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
            if(CUDAToolkit_VERSION VERSION_GREATER 12)
                 add_compile_definitions(GGML_CUDA_USE_GRAPHS) #try enable cuda graphs on cu12 build
                 set(CMAKE_CUDA_ARCHITECTURES "50;61;70;75") # lowest CUDA 12 standard + lowest for integer intrinsics
            else()
                 set(CMAKE_CUDA_ARCHITECTURES "37;50;61;70;75") # lowest CUDA 12 standard + lowest for integer intrinsics
            endif()
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (LLAMA_HIPBLAS)
    if (MSVC)
        list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/AMD/ROCm/5.5")
    else()
        list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    endif()


    if (NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CC=/opt/rocm/llvm/bin/clang")
    endif()
    if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
    endif()

    find_package(hip)
    find_package(hipblas)
    find_package(rocblas)

    if (${hipblas_FOUND} AND ${hip_FOUND})
        message(STATUS "HIP and hipBLAS found")
        file(GLOB GGML_SOURCES_ROCM "ggml/src/ggml-cuda/*.cu")
        list(APPEND GGML_SOURCES_ROCM "ggml/src/ggml-cuda/ggml-cuda.cu")
        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-mma*.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/mmq*.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        add_compile_definitions(GGML_USE_HIP GGML_USE_CUDA SD_USE_CUBLAS)
        add_library(ggml-rocm ${GGML_SOURCES_CUDA})

        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        file(GLOB SRCS "ggml/src/ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})

        # only build minimal quants required for fattn quant kv
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
        target_link_libraries(ggml-rocm PUBLIC hip::device hip::host roc::rocblas roc::hipblas)

        add_library(ggml-v2-rocm ${GGML_V2_CUDA_SOURCES})
        set_source_files_properties(otherarch/ggml_v2-cuda.cu PROPERTIES LANGUAGE CXX)
        target_link_libraries(ggml-v2-rocm PUBLIC hip::device hip::host roc::rocblas roc::hipblas)

        add_library(ggml-v3-rocm ${GGML_V3_CUDA_SOURCES})
        set_source_files_properties(otherarch/ggml_v3-cuda.cu PROPERTIES LANGUAGE CXX)
        target_link_libraries(ggml-v3-rocm PUBLIC hip::device hip::host roc::rocblas roc::hipblas)

        add_library(ggml-v2-legacy-rocm ${GGML_V2_LEGACY_CUDA_SOURCES})
        set_source_files_properties(otherarch/ggml_v2-cuda-legacy.cu PROPERTIES LANGUAGE CXX)
        target_link_libraries(ggml-v2-legacy-rocm PUBLIC hip::device hip::host roc::rocblas roc::hipblas)

        if (LLAMA_STATIC)
            message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
        endif()
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} ggml-rocm ggml-v2-rocm ggml-v2-legacy-rocm)
    else()
        message(WARNING "hipBLAS or HIP not found. Try setting CMAKE_PREFIX_PATH=/opt/rocm")
    endif()
endif()

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        set(warning_flags -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        set(c_flags -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int
            -Werror=implicit-function-declaration)
        set(cxx_flags -Wmissing-declarations -Wmissing-noreturn)

        if (CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(warning_flags ${warning_flags} -Wunreachable-code-break -Wunreachable-code-return)
            set(cxx_flags ${cxx_flags} -Wmissing-prototypes -Wextra-semi)

            if (
                (CMAKE_C_COMPILER_ID STREQUAL "Clang"      AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 3.8.0) OR
                (CMAKE_C_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3.0)
            )
                set(c_flags ${c_flags} -Wdouble-promotion)
            endif()
        elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
            set(c_flags ${c_flags} -Wdouble-promotion)
            set(cxx_flags ${cxx_flags} -Wno-array-bounds)

            if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.1.0)
                set(cxx_flags ${cxx_flags} -Wno-format-truncation)
            endif()
            if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.1.0)
                set(cxx_flags ${cxx_flags} -Wextra-semi)
            endif()
        endif()
    else()
        # todo : msvc
    endif()

    add_compile_options(
            ${warning_flags}
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (LLAMA_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (LLAMA_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        message(STATUS "OpenMP found")
        add_compile_definitions(GGML_USE_OPENMP)
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found")
    endif()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
)
if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (LLAMA_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (LLAMA_GPROF)
        add_compile_options(-pg)
    endif()
    if (LLAMA_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if ((${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm") OR (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64"))
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            # Raspberry Pi 2
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (LLAMA_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (LLAMA_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (LLAMA_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (LLAMA_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (LLAMA_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (LLAMA_F16C)
            add_compile_options(-mf16c)
        endif()
        if (LLAMA_FMA)
            add_compile_options(-mfma)
        endif()
        if (LLAMA_AVX)
            add_compile_options(-mavx)
        endif()
        if (LLAMA_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (LLAMA_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
        if (LLAMA_AVX512_VBMI)
            add_compile_options(-mavx512vbmi)
        endif()
        if (LLAMA_AVX512_VNNI)
            add_compile_options(-mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        add_compile_options(-mcpu=powerpc64le)
    else()
        add_compile_options(-mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
else()
    message(STATUS "Unknown architecture")
endif()

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=0x602)
endif()

#
# Build libraries
#

add_library(ggml
            ggml/src/ggml.c
            ggml/include/ggml.h
            ggml/src/ggml-cpu/ggml-cpu.c
            ggml/include/ggml-cpu.h
            ggml/src/ggml-alloc.c
            ggml/include/ggml-alloc.h
            ggml/src/ggml-backend.cpp
            ggml/src/ggml-backend-impl.h
            ggml/include/ggml-backend.h
            ggml/include/ggml-cpp.h
            ggml/src/ggml-quants.c
            ggml/src/ggml-quants.h
            ggml/src/ggml-cpu/llamafile/sgemm.cpp
            ggml/src/ggml-cpu/llamafile/sgemm.h
            ggml/src/ggml-cpu/ggml-cpu-traits.cpp
            ggml/src/ggml-cpu/ggml-cpu-traits.h
            ggml/src/ggml-threading.cpp
            ggml/src/ggml-cpu/ggml-cpu.cpp
            ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp
            ggml/src/ggml-cpu/ggml-cpu-aarch64.h
            ggml/src/ggml-cpu/ggml-cpu-quants.c
            ggml/src/ggml-cpu/ggml-cpu-quants.h
            ggml/src/ggml-cpu/common.h
            ggml/src/ggml-cpu/binary-ops.h
            ggml/src/ggml-cpu/binary-ops.cpp
            ggml/src/ggml-cpu/unary-ops.h
            ggml/src/ggml-cpu/unary-ops.cpp
            ggml/src/ggml-backend-reg.cpp
            ggml/include/gguf.h
            ggml/src/gguf.cpp
            ${GGML_SOURCES_CUDA})
target_include_directories(ggml PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools)
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ggml_v1
            otherarch/ggml_v1.c
            otherarch/ggml_v1.h)
target_include_directories(ggml_v1 PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools)
target_compile_features(ggml_v1 PUBLIC c_std_11) # don't bump
target_link_libraries(ggml_v1 PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
set_target_properties(ggml_v1 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ggml_v2
            otherarch/ggml_v2.c
            otherarch/ggml_v2.h
            ${GGML_V2_CUDA_SOURCES}
            ${GGML_V2_LEGACY_CUDA_SOURCES})
target_include_directories(ggml_v2 PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools)
target_compile_features(ggml_v2 PUBLIC c_std_11) # don't bump
target_link_libraries(ggml_v2 PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
set_target_properties(ggml_v2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ggml_v3
            otherarch/ggml_v3.c
            otherarch/ggml_v3.h
            ${GGML_V3_CUDA_SOURCES})
target_include_directories(ggml_v3 PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools)
target_compile_features(ggml_v3 PUBLIC c_std_11) # don't bump
target_link_libraries(ggml_v3 PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
set_target_properties(ggml_v3 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(common2
            common/common.cpp
            common/common.h
            common/sampling.cpp
            common/sampling.h
            examples/llava/llava.cpp
            examples/llava/llava.h
            examples/llava/clip.cpp
            examples/llava/clip.h
            src/unicode.h
            src/unicode.cpp
            src/unicode-data.cpp
            otherarch/utils.cpp
            otherarch/utils.h)
target_include_directories(common2 PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./otherarch/sdcpp ./otherarch/sdcpp/thirdparty ./examples ./common)
target_compile_features(common2 PUBLIC cxx_std_17) # don't bump
target_link_libraries(common2 PRIVATE ggml ${LLAMA_EXTRA_LIBS})
set_target_properties(common2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(sdtype_adapter
            otherarch/sdcpp/sdtype_adapter.cpp)
target_include_directories(sdtype_adapter PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./otherarch/sdcpp ./otherarch/sdcpp/thirdparty ./examples ./common)
target_compile_features(sdtype_adapter PUBLIC cxx_std_17) # don't bump
target_link_libraries(sdtype_adapter PRIVATE common2 ggml ${LLAMA_EXTRA_LIBS})
set_target_properties(sdtype_adapter PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(whisper_adapter
            otherarch/whispercpp/whisper_adapter.cpp)
target_include_directories(whisper_adapter PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./otherarch/whispercpp ./examples ./common)
target_compile_features(whisper_adapter PUBLIC cxx_std_17) # don't bump
target_link_libraries(whisper_adapter PRIVATE common2 ggml ${LLAMA_EXTRA_LIBS})
set_target_properties(whisper_adapter PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tts_adapter
            otherarch/tts_adapter.cpp)
target_include_directories(tts_adapter PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./examples ./common)
target_compile_features(tts_adapter PUBLIC cxx_std_17) # don't bump
target_link_libraries(tts_adapter PRIVATE common2 ggml ${LLAMA_EXTRA_LIBS})
set_target_properties(tts_adapter PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(embeddings_adapter
            otherarch/embeddings_adapter.cpp)
target_include_directories(embeddings_adapter PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./examples ./common)
target_compile_features(embeddings_adapter PUBLIC cxx_std_17) # don't bump
target_link_libraries(embeddings_adapter PRIVATE common2 ggml ${LLAMA_EXTRA_LIBS})
set_target_properties(embeddings_adapter PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gpttype_adapter
            gpttype_adapter.cpp)
target_include_directories(gpttype_adapter PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./otherarch/sdcpp ./otherarch/sdcpp/thirdparty ./examples ./common)
target_compile_features(gpttype_adapter PUBLIC cxx_std_17) # don't bump
target_link_libraries(gpttype_adapter PRIVATE common2 ggml ggml_v1 ggml_v2 ggml_v3 ${LLAMA_EXTRA_LIBS})
set_target_properties(gpttype_adapter PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (LLAMA_CUBLAS)
    set(TARGET koboldcpp_cublas)
    add_library(${TARGET} SHARED expose.cpp expose.h)
    target_include_directories(${TARGET} PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./otherarch/sdcpp ./otherarch/sdcpp/thirdparty ./examples ./common)
    target_compile_features(${TARGET} PUBLIC cxx_std_17) # don't bump
    set_target_properties(${TARGET} PROPERTIES PREFIX "")
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "koboldcpp_cublas")
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${TARGET} PUBLIC Threads::Threads ggml ggml_v1 ggml_v2 ggml_v3 common2 gpttype_adapter whisper_adapter tts_adapter embeddings_adapter sdtype_adapter ${LLAMA_EXTRA_LIBS})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    add_custom_command(
    TARGET koboldcpp_cublas POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:koboldcpp_cublas>     # The generated DLL
        ${CMAKE_SOURCE_DIR}/                # Destination directory
    COMMENT "Copying DLL to parent directory"
    )
endif()

if (LLAMA_HIPBLAS)
    set(TARGET koboldcpp_hipblas)
    add_library(${TARGET} SHARED expose.cpp expose.h)
    target_include_directories(${TARGET} PUBLIC . ./ggml/include ./ggml/src ./ggml/src/ggml-cpu ./include ./otherarch ./otherarch/tools ./otherarch/sdcpp ./otherarch/sdcpp/thirdparty ./examples ./common)
    target_compile_features(${TARGET} PUBLIC cxx_std_17) # don't bump
    set_target_properties(${TARGET} PROPERTIES PREFIX "")
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "koboldcpp_hipblas")
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${TARGET} PUBLIC Threads::Threads ggml ggml_v1 ggml_v2 ggml_v3 common2 gpttype_adapter whisper_adapter tts_adapter embeddings_adapter sdtype_adapter ${LLAMA_EXTRA_LIBS})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    add_custom_command(
    TARGET koboldcpp_hipblas POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:koboldcpp_hipblas>     # The generated DLL
        ${CMAKE_SOURCE_DIR}/                # Destination directory
    COMMENT "Copying DLL to parent directory"
    )
endif()

