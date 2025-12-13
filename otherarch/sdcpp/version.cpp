#include "stable-diffusion.h"

#ifndef SDCPP_BUILD_COMMIT
#define SDCPP_BUILD_COMMIT unknown
#endif

#ifndef SDCPP_BUILD_VERSION
#define SDCPP_BUILD_VERSION unknown
#endif

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

const char* sd_commit(void) {
    return STRINGIZE(SDCPP_BUILD_COMMIT);
}

const char* sd_version(void) {
    return STRINGIZE(SDCPP_BUILD_VERSION);
}
