#pragma once

// glibc 2.41 added C23 math declarations such as sinpi/cospi, followed by
// rsqrt/rsqrtf in 2.42, with a noexcept specification. CUDA toolkits older
// than 13.2 declare overlapping functions without it, which makes NVCC reject
// otherwise valid sources.
// Include glibc's feature selection first, then hide only that declaration
// group for affected NVCC versions. CUDA 13.2 and newer contain NVIDIA's fix.
#if defined(__linux__) && defined(__CUDACC__)
#include <features.h>

#ifdef __GLIBC_PREREQ
#if __GLIBC_PREREQ(2, 41) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    (__CUDACC_VER_MAJOR__ < 13 || \
     (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 2))
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#endif
#endif
#endif
