#ifndef BUILD_INFO_H
#define BUILD_INFO_H

#define LLAMA_BUILD_NUMBER 999
#define LLAMA_COMMIT "KOBOLDCPP"
#define LLAMA_COMPILER "KCPP"
#define LLAMA_TARGET "KCPP"
#define LLAMA_BUILD_COMMIT "KOBOLDCPP"
#define LLAMA_BUILD_COMPILER "KCPP"
#define LLAMA_BUILD_TARGET "KCPP"

#include <string>

static inline int llama_build_number(void) {
    return LLAMA_BUILD_NUMBER;
}

static inline const char * llama_commit(void) {
    return LLAMA_COMMIT;
}

static inline const char * llama_compiler(void) {
    return LLAMA_COMPILER;
}

static inline const char * llama_build_target(void) {
    return LLAMA_BUILD_TARGET;
}

static inline const char * llama_build_info(void) {
    static std::string s = "b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT;
    return s.c_str();
}

static inline void llama_print_build_info(void) {
    fprintf(stderr, "%s: build = %d (%s)\n",      __func__, llama_build_number(), llama_commit());
    fprintf(stderr, "%s: built with %s for %s\n", __func__, llama_compiler(), llama_build_target());
}

#endif // BUILD_INFO_H
