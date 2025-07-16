//LCPP had to go and split up our nice all in one cpu quant handling. It's always something eh?
//Now, we need to determine at compile time which subfile to load for kcpp.
//priority goes X86_64 > ARM/AARCH > everything else. may god help us all.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(__amd64__) || defined(__AMD64__)
#pragma message("KoboldCpp Compiling Quants for x86/x64")
#include "arch/x86/quants.c"
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM) || defined(__arm64__) || defined(__ARM64__)
#pragma message("KoboldCpp Compiling Quants for ARM")
#include "arch/arm/quants.c"
#elif defined(__powerpc64__) || defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__ppc64le__)
#pragma message("KoboldCpp Compiling Quants for PowerPC")
#include "arch/powerpc/quants.c"
#elif defined(__loongarch__) || defined(__loongarch64)
#pragma message("KoboldCpp Compiling Quants for LoongArch")
#include "arch/loongarch/quants.c"
#elif defined(__riscv) && (__riscv_xlen == 64)
#pragma message("KoboldCpp Compiling Quants for RISCV")
#include "arch/riscv/quants.c"
#elif defined(__s390x__)
#pragma message("KoboldCpp Compiling Quants for S390X")
#include "arch/s390/quants.c"
#else
#pragma message("KoboldCpp Cannot Compile Quants! Unknown Architecture!")
#error "Compilation halted due to unknown architecture."
#endif