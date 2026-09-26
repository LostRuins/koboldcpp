/*
 * oniguruma/config.h — static replacement for the file that CMake used to
 * generate from config.h.cmake.in.
 *
 * All platform facts are derived at preprocessing time, so one file works
 * on Linux, the BSDs, macOS and Windows (MSVC and MinGW), 32- and 64-bit.
 *
 * Requirements: __has_include (GCC >= 5, Clang, MSVC >= 2017 15.3) and
 * C99 <limits.h>/<stdint.h> (MSVC >= 2013). A best-effort fallback exists
 * for older compilers.
 */

#ifndef ONIGURUMA_CONFIG_H
#define ONIGURUMA_CONFIG_H

/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those
   systems.  (No platform targeted here is a Cray.) */
/* #undef CRAY_STACKSEG_END */

/* Define to 1 if using `alloca.c'.  (Not needed: every platform targeted
   here has a working alloca.) */
/* #undef C_ALLOCA */

/* --------------------------------------------------------------------
 * Header availability — replaces check_include_files().
 * __has_include is evaluated by the actual compiler, so the answers are
 * right for whatever toolchain is building: plain MSVC gets none of the
 * POSIX headers, MinGW/BSD/macOS/Linux get all of them.
 * ------------------------------------------------------------------ */
#if defined(__has_include)

#  if __has_include(<alloca.h>)
#    define HAVE_ALLOCA_H 1
#  endif

#  if __has_include(<stdint.h>)
#    define HAVE_STDINT_H 1
#  endif

#  if __has_include(<inttypes.h>)
#    define HAVE_INTTYPES_H 1
#  endif

#  if __has_include(<sys/types.h>)
#    define HAVE_SYS_TYPES_H 1
#  endif

#  if __has_include(<sys/time.h>)
#    define HAVE_SYS_TIME_H 1
#  endif

#  if __has_include(<sys/times.h>)
#    define HAVE_SYS_TIMES_H 1
#  endif

#  if __has_include(<unistd.h>)
#    define HAVE_UNISTD_H 1
#  endif

#else /* no __has_include: ancient compiler, guess by platform */

#  if defined(_MSC_VER) && !defined(__MINGW32__) && !defined(__MINGW64__)
     /* Plain old MSVC: only C runtime headers. */
#    define HAVE_STDINT_H 1
#  else
     /* Old gcc/clang on a POSIX-ish system (adjust if yours differs). */
#    define HAVE_ALLOCA_H 1
#    define HAVE_STDINT_H 1
#    define HAVE_INTTYPES_H 1
#    define HAVE_SYS_TYPES_H 1
#    define HAVE_SYS_TIME_H 1
#    define HAVE_SYS_TIMES_H 1
#    define HAVE_UNISTD_H 1
#  endif

#endif /* __has_include */

/* Define to 1 if you have `alloca', as a function or macro.
   True for every supported toolchain:
     - gcc/clang: __builtin_alloca / <alloca.h>
     - MinGW:     <alloca.h> (and <malloc.h>)
     - MSVC:      <malloc.h> defines alloca as _alloca
   This matches what check_symbol_exists() found in the CMake build on
   all of them; the library includes the right header itself. */
#define HAVE_ALLOCA 1

/* --------------------------------------------------------------------
 * Type sizes — replaces check_type_size().
 * The preprocessor can't apply sizeof, but the ranges published by
 * <limits.h> (plus UINTPTR_MAX from <stdint.h>) pin the sizes down on
 * any machine with 8-bit bytes, including the LP64 vs LLP64 split:
 *
 *                int   long   long long   void*
 *   Linux64 LP64   4     8        8         8
 *   Win64   LLP64  4     4        8         8
 *   32-bit         4     4        8         4
 * ------------------------------------------------------------------ */

#include <limits.h>
#if defined(HAVE_STDINT_H)
#  include <stdint.h>
#endif

#if CHAR_BIT != 8
#  error "config.h assumes CHAR_BIT == 8; set the SIZEOF_* macros by hand for this target."
#endif

/* The size of `int', as computed by sizeof. */
#if UINT_MAX > 0xFFFFFFFF
#  define SIZEOF_INT 8
#elif UINT_MAX > 0xFFFF
#  define SIZEOF_INT 4
#elif UINT_MAX > 0xFF
#  define SIZEOF_INT 2
#else
#  define SIZEOF_INT 1
#endif

/* The size of `long', as computed by sizeof.
   8 on LP64 Unix (64-bit Linux/macOS/BSD), 4 on Windows (LLP64) and
   on 32-bit systems. */
#if ULONG_MAX > 0xFFFFFFFF
#  define SIZEOF_LONG 8
#else
#  define SIZEOF_LONG 4
#endif

/* The size of `long long', as computed by sizeof. */
#if defined(ULLONG_MAX)
#  if ULLONG_MAX > 0xFFFFFFFF
#    define SIZEOF_LONG_LONG 8
#  else
#    define SIZEOF_LONG_LONG 4
#  endif
#else
#  define SIZEOF_LONG_LONG 8   /* no ULLONG_MAX: assume 64-bit */
#endif

/* The size of `void*', as computed by sizeof. */
#if defined(UINTPTR_MAX)
#  if UINTPTR_MAX > 0xFFFFFFFF
#    define SIZEOF_VOIDP 8
#  else
#    define SIZEOF_VOIDP 4
#  endif
#elif defined(_WIN64) || defined(__LP64__)   /* no <stdint.h>: best guess */
#  define SIZEOF_VOIDP 8
#else
#  define SIZEOF_VOIDP 4
#endif

/* --------------------------------------------------------------------
 * Package identity — the literals from CMakeLists.txt:
 *   set(PACKAGE onig)  set(PACKAGE_VERSION 6.9.10)  set(VERSION ...)
 * Autotools boilerplate the library never actually uses; quoted the way
 * autoconf would (configure_file() emitted the unquoted tokens
 * `onig' / `6.9.10', which is only usable if nothing expands them).
 * ------------------------------------------------------------------ */

/* Name of package */
#define PACKAGE         "onig"

/* Define to the version of this package. */
#define PACKAGE_VERSION "6.9.10"

/* Version number of package */
#define VERSION         "6.9.10"

/* Define if enable CR+NL as line terminator.
   The CMake build set this to 0, i.e. left it undefined — same here.
   Uncomment to enable. */
/* #undef USE_CRNL_AS_LINE_TERMINATOR */

#endif /* ONIGURUMA_CONFIG_H */
