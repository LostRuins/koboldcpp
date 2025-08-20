# CMake build configuration for MLX-Inspired Multi-LoRA System
# This builds the advanced multi-LoRA processing system for KoboldCPP

cmake_minimum_required(VERSION 3.18)
project(MultiLoRAMLX VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    if(APPLE)
        # Apple Silicon specific optimizations
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
    endif()
endif()

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Optional: Find BLAS for optimized matrix operations
find_package(BLAS)
if(BLAS_FOUND)
    add_definitions(-DUSE_BLAS)
endif()

# Platform-specific settings
if(APPLE)
    # Apple Silicon optimizations
    add_definitions(-DAPPLE_SILICON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Accelerate")
    
    # Enable Metal integration if available
    find_library(METAL_FRAMEWORK Metal)
    if(METAL_FRAMEWORK)
        add_definitions(-DUSE_METAL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Metal -framework MetalKit")
    endif()
    
elseif(UNIX AND NOT APPLE)
    # Linux optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    
elseif(WIN32)
    # Windows optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/otherarch/sdcpp
)

# Source files for the MLX-inspired multi-LoRA system
set(MLX_LORA_SOURCES
    multi_lora_mlx_fusion.cpp
    advanced_quantization_mlx.cpp
    memory_optimization_mlx.cpp  # We'll create this
)

set(MLX_LORA_HEADERS
    multi_lora_mlx_fusion.hpp
    advanced_quantization_mlx.hpp
    memory_optimization_mlx.hpp
)

# Create the main library
add_library(multilora_mlx SHARED ${MLX_LORA_SOURCES})

# Link libraries
target_link_libraries(multilora_mlx
    OpenMP::OpenMP_CXX
    Threads::Threads
)

if(BLAS_FOUND)
    target_link_libraries(multilora_mlx ${BLAS_LIBRARIES})
endif()

if(APPLE)
    target_link_libraries(multilora_mlx "-framework Accelerate")
    if(METAL_FRAMEWORK)
        target_link_libraries(multilora_mlx ${METAL_FRAMEWORK})
    endif()
endif()

# Create Python bindings library
add_library(multilora_mlx_python SHARED
    ${MLX_LORA_SOURCES}
    python_bindings_c_interface.cpp  # We'll create this
)

target_link_libraries(multilora_mlx_python
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Set proper extension for Python module
if(APPLE)
    set_target_properties(multilora_mlx_python PROPERTIES SUFFIX ".dylib")
elseif(WIN32)
    set_target_properties(multilora_mlx_python PROPERTIES SUFFIX ".dll")
else()
    set_target_properties(multilora_mlx_python PROPERTIES SUFFIX ".so")
endif()

# Integration with existing KoboldCPP
add_library(koboldcpp_mlx_integration STATIC
    ${MLX_LORA_SOURCES}
    koboldcpp_integration.cpp  # Integration layer
)

target_compile_definitions(koboldcpp_mlx_integration PRIVATE -DKOBOLDCPP_INTEGRATION)

# Create example/test executables
add_executable(test_multilora_mlx
    test_multilora_performance.cpp
    ${MLX_LORA_SOURCES}
)

target_link_libraries(test_multilora_mlx
    OpenMP::OpenMP_CXX
    Threads::Threads
)

add_executable(benchmark_multilora_mlx
    benchmark_multilora.cpp
    ${MLX_LORA_SOURCES}
)

target_link_libraries(benchmark_multilora_mlx
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Installation
install(TARGETS multilora_mlx multilora_mlx_python
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES ${MLX_LORA_HEADERS}
        DESTINATION include/multilora_mlx)

# Create pkg-config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/multilora_mlx.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/multilora_mlx.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/multilora_mlx.pc"
        DESTINATION lib/pkgconfig)

# Testing
enable_testing()
add_test(NAME test_basic_functionality COMMAND test_multilora_mlx)
add_test(NAME benchmark_performance COMMAND benchmark_multilora_mlx --quick)

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_PROJECT_NAME "MultiLoRA MLX")
    set(DOXYGEN_PROJECT_BRIEF "MLX-Inspired Multi-LoRA Processing System")
    
    doxygen_add_docs(docs
        ${MLX_LORA_HEADERS}
        ${MLX_LORA_SOURCES}
        COMMENT "Generate documentation"
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "MultiLoRA MLX Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP: ${OpenMP_FOUND}")
message(STATUS "  BLAS: ${BLAS_FOUND}")
if(APPLE)
    message(STATUS "  Apple Silicon: ON")
    message(STATUS "  Metal Support: ${METAL_FRAMEWORK}")
endif()
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

# Performance optimization hints
message(STATUS "Performance Optimization Hints:")
message(STATUS "  For maximum performance, build with:")
message(STATUS "    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=\"-O3 -march=native\"")
if(APPLE)
    message(STATUS "    On Apple Silicon, consider: -mcpu=apple-m1")
endif()
message(STATUS "")

# Create build targets for different optimization levels
add_custom_target(build_optimized
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config Release --target multilora_mlx
    COMMENT "Building with maximum optimizations"
)

add_custom_target(build_debug
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config Debug --target multilora_mlx
    COMMENT "Building with debug information"
)

add_custom_target(build_profile
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=RelWithDebInfo --build ${CMAKE_BINARY_DIR} --target multilora_mlx
    COMMENT "Building with profiling information"
)

# Performance testing targets
add_custom_target(run_benchmarks
    COMMAND benchmark_multilora_mlx --full
    DEPENDS benchmark_multilora_mlx
    COMMENT "Running comprehensive performance benchmarks"
)

add_custom_target(run_quick_test
    COMMAND test_multilora_mlx --quick
    DEPENDS test_multilora_mlx
    COMMENT "Running quick functionality tests"
)

# Integration testing
add_custom_target(test_integration
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/python_bindings_mlx_lora.py
    DEPENDS multilora_mlx_python
    COMMENT "Testing Python integration"
)

# Memory profiling (if valgrind is available)
find_program(VALGRIND_PATH valgrind)
if(VALGRIND_PATH)
    add_custom_target(memory_profile
        COMMAND ${VALGRIND_PATH} --tool=massif --massif-out-file=massif.out ./test_multilora_mlx
        DEPENDS test_multilora_mlx
        COMMENT "Profiling memory usage with Valgrind"
    )
endif()

# Code formatting (if clang-format is available)
find_program(CLANG_FORMAT_PATH clang-format)
if(CLANG_FORMAT_PATH)
    add_custom_target(format_code
        COMMAND ${CLANG_FORMAT_PATH} -i ${MLX_LORA_SOURCES} ${MLX_LORA_HEADERS}
        COMMENT "Formatting code with clang-format"
    )
endif()